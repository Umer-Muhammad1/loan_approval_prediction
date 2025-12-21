import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import LoanApplication

app = FastAPI(title="Loan Status Predictor")

# Config
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "loan_status_predictor"
MODEL_ALIAS = "21" 

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Pushing the model URI
        model_uri = f"models:/{MODEL_NAME}/{MODEL_ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"🚀 Model '{MODEL_NAME}' loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.post("/predict")
def predict(data: LoanApplication):
    try:
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])
        
        # Identify your string columns
        cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        
        # Convert objects to 'category' type
        for col in cat_cols:
            df[col] = df[col].astype("category")
        
        # Make prediction
        prediction = model.predict(df)
        
        return {
            "prediction": int(prediction[0]),
            "status": "Approved" if int(prediction[0]) == 1 else "Rejected"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running. Go to /docs for Swagger UI."}