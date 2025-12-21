
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import logging





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