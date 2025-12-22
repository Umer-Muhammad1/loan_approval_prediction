import pytest
import pandas as pd

@pytest.fixture
def sample_loan_data():
    return pd.DataFrame({
        "person_age": [25, 30],
        "person_income": [50000, 60000],
        "loan_status": [0, 1],
        "person_home_ownership": ["RENT", "OWN"],
        "loan_intent": ["EDUCATION", "MEDICAL"],
        "loan_grade": ["A", "B"],
        "cb_person_default_on_file": ["N", "N"],
        "loan_amnt": [5000, 10000],
        "loan_int_rate": [10.5, 12.0],
        "loan_percent_income": [0.1, 0.2],
        "id": [1, 2]
    })