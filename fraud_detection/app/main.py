import os
import pickle
import mlflow
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import LoanApplication

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "loan_xgb_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

# Container for artifacts to avoid global variable issues
artifacts = {
    "model": None,
    "scaler": None,
    "encoder": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifacts
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    
    try:
        # 1. Load the model
        artifacts["model"] = mlflow.pyfunc.load_model(model_uri)
        
        # 2. Get the specific Run ID that created this model version
        run_id = artifacts["model"].metadata.run_id
        print(f"📦 Fetching preprocessors from Run ID: {run_id}")

        # 3. Download the specific pickle files directly from the artifacts folder
        # MLflow returns the local path to each downloaded file
        scaler_local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="artifacts/scaler.pkl")
        encoder_local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="artifacts/encoder.pkl")

        # 4. Load them into memory
        with open(scaler_local, "rb") as f:
            artifacts["scaler"] = pickle.load(f)
        with open(encoder_local, "rb") as f:
            artifacts["encoder"] = pickle.load(f)
            
        print("✅ Successfully loaded Model, Scaler, and Encoder!")
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        artifacts["model"] = None 

    yield
    artifacts.clear()

app = FastAPI(title="Loan Status Predictor", lifespan=lifespan)

@app.post("/predict")
async def predict(data: LoanApplication):
    if not artifacts["model"]:
        raise HTTPException(status_code=503, detail="Model assets not ready")

    try:
        # Convert Pydantic request to DataFrame
        df = pd.DataFrame([data.model_dump()])

        # 1. Encoding
        cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        encoded_array = artifacts["encoder"].transform(df[cat_cols])
        encoded_cols = artifacts["encoder"].get_feature_names_out(cat_cols)
        
        df_encoded = df.drop(columns=cat_cols).reset_index(drop=True).join(
            pd.DataFrame(encoded_array, columns=encoded_cols)
        )

        # 2. Scaling
        stand_cols = [
            "person_age", "person_income", "person_emp_length", 
            "loan_amnt", "loan_int_rate", "loan_percent_income", 
            "cb_person_cred_hist_length"
        ]
        df_encoded[stand_cols] = artifacts["scaler"].transform(df_encoded[stand_cols])

        # 3. Column Alignment
        # The exact order the model was trained on
        expected_order = [
            'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 
            'person_home_ownership_OWN', 'person_home_ownership_RENT', 
            'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
            'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 
            'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_A', 
            'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E', 
            'loan_grade_F', 'loan_grade_G', 'cb_person_default_on_file_N', 
            'cb_person_default_on_file_Y', 'person_age', 'person_income', 
            'person_emp_length', 'loan_amnt', 'loan_int_rate', 
            'loan_percent_income', 'cb_person_cred_hist_length'
        ]
        
        df_final = df_encoded[expected_order]

        # 4. Inference
        prediction = artifacts["model"].predict(df_final)
        pred_val = int(prediction[0])

        return {
            "prediction": pred_val,
            "status": "Approved" if pred_val == 0 else "Rejected"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health():
    is_ready = all([artifacts["model"], artifacts["scaler"], artifacts["encoder"]])
    return {
        "status": "healthy" if is_ready else "loading/error",
        "model_ready": artifacts["model"] is not None,
        "scaler_ready": artifacts["scaler"] is not None,
        "encoder_ready": artifacts["encoder"] is not None
    }