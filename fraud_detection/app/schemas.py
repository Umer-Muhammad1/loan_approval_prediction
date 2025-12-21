from pydantic import BaseModel, Field
from typing import Optional

class LoanApplication(BaseModel):
    person_age: int = Field(..., gt=0)
    person_income: float = Field(..., ge=0)
    person_home_ownership: str # e.g., RENT, OWN, MORTGAGE
    person_emp_length: float = Field(..., ge=0)
    loan_intent: str # e.g., EDUCATION, MEDICAL, PERSONAL
    loan_grade: str # e.g., A, B, C
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0)
    loan_percent_income: float = Field(..., ge=0, le=1.0)
    cb_person_default_on_file: str # Y or N
    cb_person_cred_hist_length: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "person_age": 37,
                "person_income": 35000,
                "person_home_ownership": "RENT",
                "person_emp_length": 0.0,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 6000,
                "loan_int_rate": 11.49,
                "loan_percent_income": 0.17,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 14
            }
        }