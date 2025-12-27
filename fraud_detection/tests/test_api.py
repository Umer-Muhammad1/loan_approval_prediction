import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_artifacts():
    """Mocks the global variables in main.py"""
    with patch("app.main.model") as mock_model, \
         patch("app.main.scaler") as mock_scaler, \
         patch("app.main.encoder") as mock_encoder:
        
        # Mock Encoder behavior
        mock_encoder.transform.return_value = MagicMock() # Simulates transformation
        mock_encoder.get_feature_names_out.return_value = [
            'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 
            'person_home_ownership_OWN', 'person_home_ownership_RENT',
            'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
            'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 
            'loan_intent_PERSONAL', 'loan_intent_VENTURE', 
            'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 
            'loan_grade_E', 'loan_grade_F', 'loan_grade_G', 
            'cb_person_default_on_file_N', 'cb_person_default_on_file_Y'
        ]

        # Mock Scaler behavior (7 columns as defined in main.py)
        mock_scaler.transform.return_value = [[0] * 7]

        # Mock Model prediction (0 = Approved)
        mock_model.predict.return_value = [0]
        
        yield mock_model, mock_scaler, mock_encoder

def test_predict_success(mock_artifacts):
    payload = {
        "person_age": 25,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 2.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "A",
        "loan_amnt": 10000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.2,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "Approved"
    assert "model_version" in response.json()

def test_health_check_ready(mock_artifacts):
    # This tests if the health endpoint correctly reports 'healthy'
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"