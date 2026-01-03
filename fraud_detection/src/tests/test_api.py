import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_artifacts():
    """Mocks the global variables in main.py with correct Scikit-Learn behavior"""
    with patch("app.main.model") as mock_model, \
         patch("app.main.scaler") as mock_scaler, \
         patch("app.main.encoder") as mock_encoder, \
         patch("app.main.EXPECTED_FEATURES") as mock_features:
        
        # 1. Mock Encoder: Must return a 2D array matching the length of feature_names
        feature_names = [
            'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 
            'person_home_ownership_OWN', 'person_home_ownership_RENT',
            'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
            'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 
            'loan_intent_PERSONAL', 'loan_intent_VENTURE', 
            'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 
            'loan_grade_E', 'loan_grade_F', 'loan_grade_G', 
            'cb_person_default_on_file_N', 'cb_person_default_on_file_Y'
        ]
        mock_encoder.get_feature_names_out.return_value = np.array(feature_names)
        # Return a dummy row of 1s and 0s for the encoded categories
        mock_encoder.transform.return_value = np.zeros((1, len(feature_names)))

        # 2. Mock Scaler: 7 columns as defined in stand_cols
        mock_scaler.transform.return_value = np.zeros((1, 7))

        # 3. Mock Model: Crucial fix for main.py logic
        mock_model.classes_ = [0, 1]
        # Return 90% probability for class 1 (Approved)
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]]) 
        
        # 4. Mock Feature Alignment: Ensure the list exists
        # This list should contain all names from encoded_cols + any remaining numerical cols
        mock_features.__getitem__.side_effect = lambda x: x 
        # Alternatively, set it to a fixed list if your main.py uses it strictly
        mock_features.__iter__.return_value = ['person_age'] # etc.

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
    
    # Debugging print: If this still fails, this will show why
    if response.status_code != 200:
        print(f"Error Detail: {response.json()}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Approved"
    assert data["prediction"] == 1
    assert "model_info" in data

def test_health_check_ready(mock_artifacts):
    # This tests if the health endpoint correctly reports 'ready'
    response = client.get("/health")
    assert response.status_code == 200
    
    # Updated: Checking for 'ready' instead of 'healthy'
    assert response.json()["status"] == "ready"

#python -m pytest --cov=src/fd --cov-report=term-missing src/tests/