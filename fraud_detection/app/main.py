import os
import pickle
import mlflow
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import LoanApplication

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "loan_xgb_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables for artifacts (initialized as None)
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
    """
    Syncs the model and preprocessors from the MLflow Registry.
    This runs in a background thread to prevent blocking the event loop.
    """
    global model, scaler, encoder
    try:
        # 1. Load Model
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        print(f"📡 Syncing artifacts from MLflow: {model_uri}")
        # Note: mlflow.pyfunc.load_model is heavy and blocks
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 2. Extract Run ID for preprocessing artifacts
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run_id = model_version.run_id
        
        # 3. Download Scaler and Encoder
        print(f"📦 Run ID detected: {run_id}. Downloading preprocessors...")
        scaler_path = "data\06_models\scaler.pkl"
        encoder_path =  "data\06_models\encoder.pkl"
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
            
        print("✅ All artifacts successfully synced and loaded into memory.")
    except Exception as e:
        print(f"❌ Critical Error during artifact sync: {e}")
        # We don't raise here to keep the process alive for debugging, 
        # but the /health endpoint will reflect the failure.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown. 
    Offloads heavy I/O (MLflow) to a thread pool so the API can start immediately.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        # Schedule the loading in the background
        loop.run_in_executor(executor, load_production_artifacts)
    
    yield
    print("👋 Shutting down API...")

app = FastAPI(title="Loan Status Predictor", lifespan=lifespan)

@app.post("/predict")
def predict(payload: LoanApplication):
    # Check if artifacts are ready
    if model is None or scaler is None or encoder is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is still loading or failed to initialize. Please try again in a few seconds."
        )

    try:
        # Pydantic V2 processing
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
            "status": "Approved" if int(prediction[0]) == 1 else "Rejected",
            "confidence": float(probability) if probability else None,
            "model_info": {"name": MODEL_NAME, "alias": MODEL_ALIAS}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health():
    """
    Kubernetes Readiness Probe hits this.
    Returns 200 OK even if loading, but details specify readiness.
    """
    ready = all([model is not None, scaler is not None, encoder is not None])
    return {
        "status": "ready" if ready else "initializing",
        "model_loaded": model is not None,
        "preprocessors_loaded": all([scaler is not None, encoder is not None])
    }

@app.get("/")
def home():
    return {"service": "Loan Approval API", "status": "online"}