import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import LoanApplication

app = FastAPI(title="Loan Status Predictor")

# Config from Environment Variables (Essential for Docker)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "loan_status_predictor")
# Use 'Production' stage or a specific alias like 'champion'
MODEL_VERSION_OR_STAGE = os.getenv("MODEL_VERSION_OR_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Pulling by Stage (Production) is more dynamic than ID 21
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION_OR_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"🚀 Model '{MODEL_NAME}' ({MODEL_VERSION_OR_STAGE}) loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # In production, you might want the app to fail if the model can't load
        model = None

@app.post("/predict")
def predict(data: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
        
    try:
        # Convert Pydantic model to Dict, then DataFrame
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])
        
        # Expert Tip: Ensure column order matches training!
        # Your 'cast_loan_data_types' logic from Kedro nodes should be mirrored here
        cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        
        prediction = model.predict(df)
        
        return {
            "prediction": int(prediction[0]),
            "status": "Approved" if int(prediction[0]) == 1 else "Rejected",
            "model_version": MODEL_VERSION_OR_STAGE
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}