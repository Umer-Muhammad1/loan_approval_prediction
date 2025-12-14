import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

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




import xgboost as xgb


import mlflow
import mlflow.xgboost
from datetime import datetime

def train_evaluate_xgb(X_train, y_train, X_test, y_test, xgb_params):
    """
    Train and evaluate XGBoost model with MLflow logging
    Fixed version that cleans ALL problematic parameters
    """
    
    print(f"DEBUG: Raw parameters received: {xgb_params}")
    
    # FIX 1: Clean up any active MLflow runs
    try:
        if mlflow.active_run():
            print(f"Ending active run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()
    except Exception as e:
        print(f"Warning: Could not end active run: {e}")
    
    # FIX 2: Clean ALL problematic parameters
    # These are parameters that should NOT be passed to XGBClassifier constructor
    problematic_params = [
        'use_label_encoder',  # Deprecated in newer XGBoost
        'eval_metric',        # Should be passed to fit(), not constructor
        'early_stopping_rounds',  # Should be passed to fit()
        'callbacks',          # Should be passed to fit()
        'xgb_model',          # For continued training
        'verbosity',          # Can cause conflicts
        'n_jobs'              # Can cause conflicts with cross_val_score
    ]
    
    xgb_params_clean = {}
    
    # Copy only valid constructor parameters
    for key, value in xgb_params.items():
        if key not in problematic_params:
            xgb_params_clean[key] = value
        else:
            print(f"DEBUG: Filtering out parameter: {key} = {value}")
    
    # Log what parameters we're using
    print(f"DEBUG: Cleaned constructor parameters: {xgb_params_clean}")
    
    # Separate fit parameters (for use in model.fit())
    fit_params = {}
    if 'eval_metric' in xgb_params:
        print(f"DEBUG: Adding eval_metric to fit parameters: {xgb_params['eval_metric']}")
        fit_params['eval_metric'] = xgb_params['eval_metric']
    #if 'early_stopping_rounds' in xgb_params:
       # print(f"DEBUG: Adding early_stopping_rounds to fit parameters: {xgb_params['early_stopping_rounds']}")
        #fit_params['early_stopping_rounds'] = xgb_params['early_stopping_rounds']
    
    try:
        # Start MLflow run
        run_name = f"xgb_loan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Starting MLflow run: {run_name}")
        
        #mlflow.set_experiment("loan_prediction")
        
        #with mlflow.start_run(run_name=run_name):
        print("MLflow run started")
        
        # Log ALL parameters (both constructor and fit params)
        mlflow.log_params(xgb_params)  # Log original parameters
        
        # Log dataset statistics
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("positive_class_ratio", y_train.mean())
        
        # Initialize model with CLEANED constructor parameters
        print(f"Initializing XGBClassifier with: {xgb_params_clean}")
        xgb_model = xgb.XGBClassifier(
            **xgb_params_clean,
            use_label_encoder=False  # Always False for newer versions
        )
        
        # Cross-validation
        print("Starting cross-validation...")
        # For cross_val_score, we need to handle n_jobs carefully
        cv_model = xgb.XGBClassifier(
            **xgb_params_clean,
            use_label_encoder=False
        )
        
        cv_scores = cross_val_score(
            cv_model, 
            X_train, 
            y_train, 
            cv=10, 
            scoring='roc_auc', 
            n_jobs=1  # Avoid nested parallelism
        )
        
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_mean:.4f}")
        print(f"Std CV AUC: {cv_std:.4f}")
        
        mlflow.log_metric("mean_cv_auc", cv_mean)
        mlflow.log_metric("std_cv_auc", cv_std)
        
        # Train final model
        print("Training final model...")
        
        # Prepare validation set for early stopping if specified
        if 'early_stopping_rounds' in fit_params:
            print("Using early stopping with validation set")
            # Split training data further
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42
            )
            
            # Train with early stopping
            xgb_model.fit(
                X_train_fit, 
                y_train_fit,
                eval_set=[(X_val, y_val)],
                **fit_params,
                verbose=False
            )
        else:
            # Train without early stopping
            xgb_model.fit(X_train, y_train)
        
        # Predictions
        print("Making predictions...")
        y_pred = xgb_model.predict(X_test)
        #y_pred_csv = y_pred.to_csv(index=False)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_auc", test_auc)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log top 10 features
        top_features = feature_importance.head(10)
        for idx, row in top_features.iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        # Log model
        print("Logging model to MLflow...")
        mlflow.xgboost.log_model(
            xgb_model,
            artifact_path="model",
            registered_model_name="loan_status_predictor"
        )
        
        # Add tags
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("framework", "Kedro")
        mlflow.set_tag("status", "completed")
        
        # Log run info
        run_id = mlflow.active_run().info.run_id
        print(f"✅ MLflow Run ID: {run_id}")
        
        return  xgb_model, test_accuracy, test_auc, cv_scores, feature_importance,y_pred, y_pred_proba
            
            
    except Exception as e:
        print(f"❌ Error in train_evaluate_xgb: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'model': None,
            'test_accuracy': 0,
            'test_auc': 0,
            'cv_scores': [],
            'feature_importance': pd.DataFrame(),
            'run_id': None,
            'error': str(e)
        }
        
        
def plot_roc_curve(y_test, y_pred_proba):
    """
    Generate ROC-AUC curve plot
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities for positive class
    
    Returns:
        matplotlib.figure.Figure: ROC curve figure
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5)')
    
    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with additional info
    textstr = f'AUC Score: {roc_auc:.4f}\nSamples: {len(y_test)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.1, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Log to MLflow if active run exists
    #try:
    #    if mlflow.active_run():
    #        mlflow.log_figure(fig, "roc_curve.png")
    #        mlflow.log_metric("roc_auc_from_curve", roc_auc)
    #        print(f"✅ ROC curve logged to MLflow (AUC: {roc_auc:.4f})")
    #except Exception as e:
    #    print(f"Warning: Could not log to MLflow: {e}")
    
    return fig