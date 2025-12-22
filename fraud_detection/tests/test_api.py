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
def test_predict_success(mock_model, valid_payload):
    # Setup the mock to return a prediction of [1]
    mock_model.predict.return_value = [1]
    
    response = client.post("/predict", json=valid_payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "Approved"
    assert response.json()["prediction"] == 1

def test_predict_invalid_data(valid_payload):
    # Modify payload to fail Pydantic validation (age < 18)
    valid_payload["person_age"] = 10 
    
    response = client.post("/predict", json=valid_payload)
    
    # FastAPI returns 422 Unprocessable Entity for schema validation errors
    assert response.status_code == 422
    assert "person_age" in response.text