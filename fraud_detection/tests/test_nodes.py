from src.fd.pipelines.data_engineering.de_nodes import validate_loan_data
import pytest
import pandas as pd


def test_validate_loan_data_returns_dataframe(sample_loan_data):
    """
    Ensures that when data is valid, the node returns a 
    DataFrame and not None (preventing pipeline breaks).
    """
    result = validate_loan_data(sample_loan_data)
    
    assert result is not None, "Node returned None; check return statement."
    assert isinstance(result, pd.DataFrame), "Output must be a Pandas DataFrame."
    assert result.shape == sample_loan_data.shape, "Node should not drop rows/cols unless intended."

def test_validation_logic_blocks_bad_data(sample_loan_data):
    """
    Verify that an age > 123 actually raises the ValueError 
    you defined in your node.
    """
    bad_data = sample_loan_data.copy()
    bad_data.loc[0, "person_age"] = 150  # Violates your GX rule
    
    with pytest.raises(ValueError) as excinfo:
        validate_loan_data(bad_data)
    
    assert "Pipeline halted" in str(excinfo.value)