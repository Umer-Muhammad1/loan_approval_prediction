import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app

# Create a TestClient instance
client = TestClient(app)

@pytest.fixture
def valid_payload():
    return {
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

# We mock the model so we don't need a real MLflow server running during tests


@patch("app.main.model")
@patch("app.main.scaler")
@patch("app.main.encoder")
def test_predict_success(mock_encoder, mock_scaler, mock_model, valid_payload):
    # 1. Setup Mock Encoder (must return an array with correct shape)
    mock_encoder.transform.return_value = MagicMock() # Or a dummy numpy array
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

    # 2. Setup Mock Scaler
    mock_scaler.transform.return_value = [[0] * 7] # Dummy scaled values

    # 3. Setup Mock Model
    mock_model.predict.return_value = [1]

    # Execute
    response = client.post("/predict", json=valid_payload)

    # Assert
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["status"] == "Approved"

def test_predict_invalid_data(valid_payload):
    # Modify payload to fail Pydantic validation (age < 18)
    valid_payload["person_age"] = 10 
    
    response = client.post("/predict", json=valid_payload)
    
    # FastAPI returns 422 Unprocessable Entity for schema validation errors
    assert response.status_code == 422
    assert "person_age" in response.text