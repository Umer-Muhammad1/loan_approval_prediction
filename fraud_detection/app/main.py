import os
import pickle
import mlflow
import pandas as pd
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import LoanApplication

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "loan_xgb_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Global variables for artifacts (initialized as None)
model = None
scaler = None
encoder = None
DECISION_THRESHOLD = 0.5

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
    Loads the model from MLflow but loads scaler/encoder from fixed local paths.
    """
    global model, scaler, encoder
    client = mlflow.tracking.MlflowClient()

    try:
        # 1. Load Model from MLflow
        try:
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            logger.info(f"📡 Attempting to load model: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception as alias_err:
            logger.warning(f"⚠️ Alias not found, falling back to latest...")
            model_uri = f"models:/{MODEL_NAME}/latest"
            model = mlflow.pyfunc.load_model(model_uri)

        # 2. Load Scaler and Encoder from manual local paths
        scaler_path = "data/06_models/scaler.pkl"
        encoder_path = "data/06_models/encoder.pkl"

        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            with open(encoder_path, "rb") as f:
                encoder = pickle.load(f)
            logger.info(f"✅ Preprocessors loaded locally from data/06_models/")
        except FileNotFoundError as e:
            logger.error(f"❌ Local artifact file not found: {e}")
            raise

        logger.info("✅ All artifacts successfully loaded into memory.")

    except Exception as e:
        logger.error(f"❌ Critical Error during artifact sync: {str(e)}")

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
        #prediction = model.predict(df_final)
        positive_class_index = list(model.classes_).index(1)
        proba_approve = model.predict_proba(df_final)[0, positive_class_index]
        # Threshold-based decision
        decision = int(proba_approve >= DECISION_THRESHOLD)

        return {
            "prediction": decision,  # 0 = Rejected, 1 = Approved
            "status": "Approved" if decision == 1 else "Rejected",
            "confidence": float(proba_approve),
            "threshold": DECISION_THRESHOLD,
            "model_info": {
                "name": MODEL_NAME,
                "alias": MODEL_ALIAS,
                "positive_class": 1,
                "business_meaning": "1=Approved, 0=Rejected"
            }
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