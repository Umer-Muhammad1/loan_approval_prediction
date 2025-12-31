import os
import pickle
import mlflow
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import LoanApplication

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "loan_xgb_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables
model = None
scaler = None
encoder = None
DECISION_THRESHOLD = 0.5

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
    Downloads and loads the model, scaler, and encoder into memory.
    """
    global model, scaler, encoder
    client = mlflow.tracking.MlflowClient()

    try:
        # 1. Load Model (Try Alias -> Fallback to Latest)
        try:
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            logger.info(f"📡 Attempting to load primary model: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            logger.info(f"✅ Loaded model using alias: {MODEL_ALIAS}")
        except Exception as alias_err:
            logger.warning(f"⚠️ Alias '{MODEL_ALIAS}' not found: {alias_err}. Falling back to latest...")
            model_uri = f"models:/{MODEL_NAME}/latest"
            model = mlflow.pyfunc.load_model(model_uri)
            versions = client.get_latest_versions(MODEL_NAME)
            if not versions:
                raise RuntimeError(f"No versions found for model name '{MODEL_NAME}'")
            model_version = versions[0]
            logger.info(f"✅ Loaded latest version (Version {model_version.version})")

        run_id = model_version.run_id
        logger.info(f"📦 Run ID detected: {run_id}. Downloading preprocessors...")

        # 2. Download and Unpickle Scaler and Encoder
        scaler_path = client.download_artifacts(run_id, "scaler.pkl")
        encoder_path = client.download_artifacts(run_id, "encoder.pkl")
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
            
        logger.info("✅ All artifacts successfully synced and loaded into memory.")

    except Exception as e:
        logger.error(f"❌ Critical Error during artifact sync: {str(e)}")
        # We don't raise here so the container doesn't crash loop, 
        # allowing you to exec in and debug or check /health.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup logic. Note: In Production, you might want this to block 
    startup so K8s doesn't send traffic until the model is ready.
    """
    # Simply call the loader. If you want the API to wait for the model 
    # before starting, don't use a background thread.
    load_production_artifacts()
    yield
    logger.info("👋 Shutting down API...")

app = FastAPI(title="Loan Status Predictor", lifespan=lifespan)

@app.post("/predict")
async def predict(payload: LoanApplication):
    if model is None or scaler is None or encoder is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is still loading or failed to initialize. Check logs."
        )

    try:
        # Prepare Data
        data_dict = payload.model_dump()
        df = pd.DataFrame([data_dict])

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

        # 3. Alignment (Ensures XGBoost gets columns in the right order)
        df_final = df_encoded[EXPECTED_FEATURES]

        # 4. Inference
        # Handling different model flavors (some pyfunc models return ndarray, others have predict_proba)
        # Assuming XGBoost with probability output enabled
        proba_approve = model.predict(df_final)
        
        # If model.predict returns the probability directly:
        val = float(proba_approve[0])
        decision = int(val >= DECISION_THRESHOLD)

        return {
            "prediction": decision,
            "status": "Approved" if decision == 1 else "Rejected",
            "confidence": val,
            "threshold": DECISION_THRESHOLD,
            "model_info": {
                "name": MODEL_NAME,
                "version_run_id": getattr(model, "metadata", {}).get("run_id", "unknown")
            }
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health():
    ready = all([model is not None, scaler is not None, encoder is not None])
    return {
        "status": "ready" if ready else "initializing",
        "model_loaded": model is not None,
        "preprocessors_loaded": all([scaler is not None, encoder is not None]),
        "mlflow_uri": MLFLOW_TRACKING_URI
    }

@app.get("/")
def home():
    return {"service": "Loan Approval API", "status": "online"}