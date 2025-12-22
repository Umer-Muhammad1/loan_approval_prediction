import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import LoanApplication

app = FastAPI(title="Loan Status Predictor")

# Config from Environment Variables (Essential for Docker)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
#MODEL_NAME = os.getenv("MODEL_NAME", "MODEL_NAME = "loan_xgb_modelv21"")
# Use 'Production' stage or a specific alias like 'champion'
#MODEL_VERSION_OR_STAGE = os.getenv("MODEL_VERSION_OR_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = None

# Updated Configuration
MODEL_NAME = "loan_xgb_model"
MODEL_ALIAS = "champion"  # Matches your '@champion' in UI

@app.on_event("startup")
def load_model():
    global model
    # The '@' syntax is the key for modern MLflow aliases
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    
    print(f"📡 Loading model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Champion Model (Version 21) loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")



def apply_inference_schema(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Categorical casting
    cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2. Downcasting Integers (Skip 'loan_status' and 'id' as they aren't in API input)
    int_map = {
        "person_age": "int8",
        "person_income": "int32",
        "cb_person_cred_hist_length": "int8"
    }
    for col, dtype in int_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # 3. Downcasting Floats
    float_cols = ["person_emp_length", "loan_int_rate", "loan_percent_income"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df

@app.post("/predict")
def predict(data: LoanApplication):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])

        # Apply the optimized schema
        df = apply_inference_schema(df)

        # Run prediction
        prediction = model.predict(df)

        return {
            "prediction": int(prediction[0]),
            "status": "Approved" if int(prediction[0]) == 1 else "Rejected"
           # "model_version": MODEL_VERSION_OR_STAGE
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/")
def home():
    return {"message": "Loan Status Predictor API is up and running!"}