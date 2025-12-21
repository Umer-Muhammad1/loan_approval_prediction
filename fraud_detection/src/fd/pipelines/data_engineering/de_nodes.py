
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import logging


#from ...data_validation import SimpleLoanValidator
import pandas as pd
import json


import great_expectations as gx
import pandas as pd

# NO NEED for ExpectationConfiguration import - we'll use the simpler approach

import pandas as pd
import great_expectations as gx
from great_expectations.datasource.fluent import PandasDatasource

import pandas as pd
import great_expectations as gx

import time

#from great_expectations.datasource.pandas_datasource import PandasDatasource
import pandas as pd
import great_expectations as gx

import pandas as pd
import great_expectations as gx

def validate_loan_data(df: pd.DataFrame):
    # 1. Initialize context
    context = gx.get_context()
    
    datasource_name = "loan_datasource"
    asset_name = "loan_asset"
    suite_name = "loan_suite"
    
    # 2. Get or add the pandas datasource
    try:
        ds = context.data_sources.get(datasource_name)
    except Exception:
        ds = context.data_sources.add_pandas(name=datasource_name)

    # 3. Add the Asset (without data initially)
    try:
        asset = ds.get_asset(asset_name)
    except Exception:
        asset = ds.add_dataframe_asset(name=asset_name)

    # 4. BUILD BATCH REQUEST (The Fix)
    # The error message says it needs 1 key called 'dataframe'
    batch_request = asset.build_batch_request(options={"dataframe": df})

    # 5. Handle the Suite
    try:
        suite = context.suites.get(name=suite_name)
    except Exception:
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # 6. Get Validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    print("--- Running Great Expectations Validation ---")

    # 7. Add your Rules
    validator.expect_column_values_to_be_between("person_age", min_value=18, max_value=123)
    validator.expect_column_values_to_be_between("person_income", min_value=0)
    validator.expect_column_values_to_not_be_null("loan_status")
# --- Categorical Rules (Allowed Values) ---
    validator.expect_column_values_to_be_in_set(
        "person_home_ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )
    validator.expect_column_values_to_be_in_set(
        "loan_intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    )
    validator.expect_column_values_to_be_in_set(
        "loan_grade", ["A", "B", "C", "D", "E", "F", "G"]
    )
    validator.expect_column_values_to_be_in_set(
        "cb_person_default_on_file", ["Y", "N"]
    )

    # --- Numerical Rules (Ranges) ---
    # Employment length shouldn't exceed age! But let's start with a basic range
    #validator.expect_column_values_to_be_between("person_emp_length", min_value=0, max_value=60)
    
    # Loan amount should be positive
    validator.expect_column_values_to_be_between("loan_amnt", min_value=500, max_value=50000)
    
    # Interest rate usually falls between 5% and 25%
    validator.expect_column_values_to_be_between("loan_int_rate", min_value=0, max_value=30)
    
    # Credit history length
    #validator.expect_column_values_to_be_between("cb_person_cred_hist_length", min_value=0, max_value=50)

    # --- Business Logic Rules ---
    # Loan percent of income should be a fraction between 0 and 1
    validator.expect_column_values_to_be_between("loan_percent_income", min_value=0, max_value=1.0)

    # 8. Run and Return
    results = validator.validate()
    # Visualization TODO
    #context.build_data_docs()
    #context.open_data_docs()
    
    if results["success"]:
            print("✅ Data Quality Validated!")
            return df
    else:
        print("❌ Data Quality Failed!")
        # ADD THIS LOOP TO SEE THE DETAILS:
        for result in results["results"]:
            if not result["success"]:
                # Use .expectation_config.expectation_type (as a property/attribute)
                # or result.expectation_config.type depending on your exact build
                rule_name = getattr(result.expectation_config, "expectation_type", "Unknown Rule")
                print(f"\nFAILED RULE: {rule_name}")
                
                # result.result is a dictionary containing the "partial_unexpected_list"
                unexpected = result.result.get("partial_unexpected_list", [])
                print(f"OFFENDING VALUES: {unexpected}")
        
        raise ValueError("Pipeline halted due to Data Quality issues.")

def quick_validate_csv(data: pd.DataFrame) -> pd.DataFrame:
    """Quick validation for a CSV file"""

    #df = pd.read_csv("data")
    return validate_loan_data(data)




def cast_loan_data_types(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Categorical casting (The big memory winners)
    cat_cols = [
        "person_home_ownership", "loan_intent", 
        "loan_grade", "cb_person_default_on_file"
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # 2. Downcasting Integers (The precision fix)
    df["person_age"] = df["person_age"].astype("int8")
    df["loan_status"] = df["loan_status"].astype("int8")
    df["id"] = df["id"].astype("int32")
    df["person_income"] = df["person_income"].astype("int32")
    df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].astype("int8")

    # 3. Downcasting Floats
    float_cols = ["person_emp_length", "loan_int_rate", "loan_percent_income"]
    for col in float_cols:
        df[col] = df[col].astype("float32")

    print("✅ Schema optimized for Docker/API deployment.")
    return df

#def validate_loan_data(data):
#    """Use in your Kedro pipeline"""
#    validator = SimpleLoanValidator()
#    results = validator.validate_data(data)
#    
#    # Optional: Fail pipeline if validation fails
#    if not results.success:
#        print("Warning: Data validation failed")
#        # Don't raise error, just log warning
#    
#    return data  # Pass the data through to the next node if valid


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    df= data.drop_duplicates()
   
    return df


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    
    data= data.drop(['id'], axis=1)
    data_1 = data[data['person_age'] < 90].reset_index(drop=True)
    data_2 = data_1[data_1['person_income'] < 1e6].reset_index(drop=True)
    df2 = data_2[data_2['person_emp_length'] < 60].reset_index(drop=True)

    return df2


def feature_target_split(data: pd.DataFrame):
    
    
    features = data.drop(columns='loan_status', axis =1)
    target = data['loan_status']
    return features, target
    
    

    
    
def train_test_df_split(features: pd.DataFrame, target: pd.Series, test_size: float , random_state: float):
    
    loan_amounts = features['loan_amnt'].copy()
 
    X_train, X_test, y_train, y_test , loan_amt_train, loan_amt_test= train_test_split(features, target,loan_amounts, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test , loan_amt_train, loan_amt_test


def one_hot_encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    
    # Concatenate X_train and X_test
    combined_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    cat_cols= combined_data.select_dtypes(['object']).columns
    # One-hot encode the categorical columns
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_data = onehot_encoder.fit_transform(combined_data[cat_cols])
    
    # Convert encoded data to DataFrame
    encoded_columns = onehot_encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

    # Drop the original categorical columns and add the encoded ones
    combined_data_encoded = combined_data.drop(columns=cat_cols).join(encoded_df)
    
    # Split the data back into X_train and X_test
    X_train_encoded = combined_data_encoded.iloc[:len(X_train), :]
    X_test_encoded = combined_data_encoded.iloc[len(X_train):, :]

    return X_train_encoded, X_test_encoded




def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, stand_col: list):
    
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train[stand_col])
    X_test_scaled = scaler.transform(X_test[stand_col])

    # Convert back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=stand_col, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=stand_col, index=X_test.index)

    # Replace original columns with scaled columns
    X_train_final = X_train.drop(columns=stand_col).join(X_train_scaled_df)
    X_test_final = X_test.drop(columns=stand_col).join(X_test_scaled_df)

    return X_train_final, X_test_final