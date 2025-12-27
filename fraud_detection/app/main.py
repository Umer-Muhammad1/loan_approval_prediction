import os
import pickle
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import LoanApplication

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MODEL_NAME = "loan_xgb_model"
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables for artifacts
model = None
scaler = None
encoder = None

# Exact order expected by the XGBoost model
EXPECTED_FEATURES = [
    'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 'person_home_ownership_OWN', 
    'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
    'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 
    'loan_intent_VENTURE', 'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 
    'loan_grade_E', 'loan_grade_F', 'loan_grade_G', 'cb_person_default_on_file_N', 
    'cb_person_default_on_file_Y', 'person_age', 'person_income', 'person_emp_length', 
    'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'
]

def load_production_artifacts():
    """Syncs the model and preprocessors from the MLflow Registry."""
    global model, scaler, encoder
    try:
        # 1. Load Model
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        print(f"📡 Syncing artifacts from MLflow: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 2. Extract Run ID for preprocessing artifacts
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run_id = model_version.run_id
        
        # 3. Download Scaler and Encoder
        scaler_path = client.download_artifacts(run_id, "preprocessing/scaler.pkl")
        encoder_path = client.download_artifacts(run_id, "preprocessing/encoder.pkl")
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
            
        print("✅ All artifacts successfully synced from MLflow.")
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        # In production K8s, raising an error here ensures the container 
        # doesn't pass the readiness probe.
        raise RuntimeError(f"Could not load model artifacts: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI Lifespan management (Startup/Shutdown)."""
    # Logic executed on startup
    load_production_artifacts()
    yield
    # Logic executed on shutdown
    print("Shutting down API...")

app = FastAPI(title="Loan Status Predictor", lifespan=lifespan)

@app.post("/predict")
def predict(payload: LoanApplication):
    if not all([model, scaler, encoder]):
        raise HTTPException(status_code=503, detail="Model assets not ready")

    try:
        # Pydantic V2: Using model_dump() instead of .dict()
        data_dict = payload.model_dump()
        df = pd.DataFrame([data_dict])

        # --- Preprocessing ---
        cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        stand_cols = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]

        # 1. Encoding
        encoded_array = encoder.transform(df[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_encoded = df.drop(columns=cat_cols).reset_index(drop=True).join(
            pd.DataFrame(encoded_array, columns=encoded_cols)
        )

        # 2. Scaling
        df_encoded[stand_cols] = scaler.transform(df_encoded[stand_cols])

        # 3. Alignment
        df_final = df_encoded[EXPECTED_FEATURES]

        # --- Inference ---
        prediction = model.predict(df_final)
        probability = model.predict_proba(df_final).max() if hasattr(model, "predict_proba") else None

        return {
            "prediction": int(prediction[0]),
            "status": "Approved" if int(prediction[0]) == 0 else "Rejected",
            "confidence": float(probability) if probability else None,
            "model_alias": MODEL_ALIAS
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health():
    is_ready = all([model is not None, scaler is not None, encoder is not None])
    return {
        "status": "healthy" if is_ready else "initializing",
        "model_ready": model is not None,
        "preprocessing_ready": all([scaler is not None, encoder is not None])
    }

@app.get("/")
def home():
    return {"service": "Loan Approval API", "version": "1.0.0"}