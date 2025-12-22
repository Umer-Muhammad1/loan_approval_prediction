import os
import mlflow
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from app.schemas import LoanApplication

app = FastAPI(title="Loan Status Predictor")

# Config from Environment Variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "loan_xgb_model"
MODEL_ALIAS = "champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables for model and preprocessing artifacts
model = None
scaler = None
encoder = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, encoder
    
    # 1. Load the MLflow Model
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"📡 Loading model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

    # 2. Load the Scaler and Encoder from your Kedro data folder
    try:
        with open("data/06_models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("data/06_models/encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        print("✅ Scaler and Encoder loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load preprocessing artifacts: {e}")

@app.post("/predict")
def predict(data: LoanApplication):
    try:
        # ... (Existing conversion to DF)
        df = pd.DataFrame([data.dict()])

        # 1. Encoding
        cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        encoded_array = encoder.transform(df[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_encoded = df.drop(columns=cat_cols).reset_index(drop=True).join(
            pd.DataFrame(encoded_array, columns=encoded_cols)
        )

        # 2. Scaling
        stand_cols = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
        df_encoded[stand_cols] = scaler.transform(df_encoded[stand_cols])

        # --- THE CRITICAL FIX: REORDER COLUMNS ---
        # Copy the EXACT order from your error message "expected" list
        expected_order = [
            'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 'person_home_ownership_OWN', 
            'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
            'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 
            'loan_intent_VENTURE', 'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 
            'loan_grade_E', 'loan_grade_F', 'loan_grade_G', 'cb_person_default_on_file_N', 
            'cb_person_default_on_file_Y', 'person_age', 'person_income', 'person_emp_length', 
            'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'
        ]
        
        # Rearrange the columns to match the expected order
        df_final = df_encoded[expected_order]

        # 3. Predict using the reordered DataFrame
        prediction = model.predict(df_final)

        return {
            "prediction": int(prediction[0]),
            "status": "Approved" if int(prediction[0]) == 1 else "Rejected"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "artifacts_loaded": all([scaler is not None, encoder is not None])
    }

@app.get("/")
def home():
    return {"message": "Loan Status Predictor API is up and running!"}

#uvicorn app.main:app --reload  (execution)