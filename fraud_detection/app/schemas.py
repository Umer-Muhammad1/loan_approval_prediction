from pydantic import BaseModel, Field, ConfigDict

class LoanApplication(BaseModel):
    person_age: int = Field(..., gt=18, lt=100)
    person_income: int = Field(..., ge=0)
    person_home_ownership: str 
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

    # Updated for Pydantic V2
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "person_age": 25,
                "person_income": 55000,
                "person_home_ownership": "RENT",
                "person_emp_length": 4.0,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 5000,
                "loan_int_rate": 11.5,
                "loan_percent_income": 0.1,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 3
            }
        }
    )