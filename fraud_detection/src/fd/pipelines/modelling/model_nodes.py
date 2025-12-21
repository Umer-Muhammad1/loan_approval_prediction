"""
Refactored Kedro nodes for multi-model training pipeline
Split into: Training → Model Selection → Cross-Validation → Testing
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix , roc_curve , auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# ==============================================================================
# STEP 1: TRAIN MULTIPLE MODELS
# ==============================================================================

def ensure_no_active_run():
    """Force end any active MLflow runs"""
    while mlflow.active_run():

        mlflow.end_run()



def train_logistic_regression(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    lr_params: Dict[str, Any]
) -> Tuple[Any, str]:
    """
    Train Logistic Regression model
    
    Returns:
        model: Trained model
        run_id: MLflow run ID
    """
    # Clean up any active runs
    ensure_no_active_run()
    
    run_name = f"lr_loan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"Training Logistic Regression: {run_name}")
    print(f"{'='*60}")
    print(f"Parameters: {lr_params}")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params(lr_params)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("positive_class_ratio", y_train.mean())
        
        # Train model
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(**lr_params)
        lr_model.fit(X_train, y_train)
        
        # Quick validation score on training data
        train_pred = lr_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_proba = lr_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_auc", train_auc)
        
        # Log model
        mlflow.sklearn.log_model(
            lr_model,
            artifact_path="model",
            registered_model_name="loan_lr_model"
        )
        
        # Add tags
        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("stage", "training")
        
        run_id = run.info.run_id
        print(f"✅ Logistic Regression trained | Run ID: {run_id}")
        print(f"   Train Accuracy: {train_accuracy:.4f} | Train AUC: {train_auc:.4f}\n")
        
        return lr_model, run_id


def train_random_forest(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    rf_params: Dict[str, Any]
) -> Tuple[Any, str]:
    """
    Train Random Forest model
    
    Returns:
        model: Trained model
        run_id: MLflow run ID
    """
    # Clean up any active runs
    ensure_no_active_run()
    
    run_name = f"rf_loan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"Training Random Forest: {run_name}")
    print(f"{'='*60}")
    print(f"Parameters: {rf_params}")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params(rf_params)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("positive_class_ratio", y_train.mean())
        
        # Train model
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        
        # Quick validation score
        train_pred = rf_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_proba = rf_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_auc", train_auc)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log top 10 features
        for idx, row in feature_importance.head(10).iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        # Log model
        mlflow.sklearn.log_model(
            rf_model,
            artifact_path="model",
            registered_model_name="loan_rf_model"
        )
        
        # Add tags
        mlflow.set_tag("model_type", "Random Forest")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("stage", "training")
        
        run_id = run.info.run_id
        print(f"✅ Random Forest trained | Run ID: {run_id}")
        print(f"   Train Accuracy: {train_accuracy:.4f} | Train AUC: {train_auc:.4f}\n")
        
        return rf_model, run_id


def train_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    xgb_params: Dict[str, Any]
) -> Tuple[Any, str]:
    """
    Train XGBoost model (cleaned version of your original)
    
    Returns:
        model: Trained model
        run_id: MLflow run ID
    """
    # Clean up any active runs
    ensure_no_active_run()
    
    run_name = f"xgb_loan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"Training XGBoost: {run_name}")
    print(f"{'='*60}")
    print(f"Raw parameters: {xgb_params}")
    
    # Clean problematic parameters
    problematic_params = [
        'use_label_encoder', 'eval_metric', 'early_stopping_rounds',
        'callbacks', 'xgb_model', 'verbosity', 'n_jobs'
    ]
    
    xgb_params_clean = {k: v for k, v in xgb_params.items() 
                        if k not in problematic_params}
    
    #fit_params = {}
    #if 'eval_metric' in xgb_params:
    #    fit_params['eval_metric'] = xgb_params['eval_metric']
    
    print(f"Cleaned parameters: {xgb_params_clean}")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log ALL parameters
        mlflow.log_params(xgb_params)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("positive_class_ratio", y_train.mean())
        
        # Train model
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            **xgb_params_clean,
            use_label_encoder=False
        )
        
        # Handle early stopping if needed
        if 'early_stopping_rounds' in xgb_params:
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42
            )
            xgb_model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            xgb_model.fit(X_train, y_train)
        
        # Quick validation score
        train_pred = xgb_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_proba = xgb_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_auc", train_auc)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log top 10 features
        for idx, row in feature_importance.head(10).iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        # Log model
        mlflow.xgboost.log_model(
            xgb_model,
            artifact_path="model",
            registered_model_name="loan_xgb_model"
        )
        
        # Add tags
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("stage", "training")
        
        run_id = run.info.run_id
        print(f"✅ XGBoost trained | Run ID: {run_id}")
        print(f"   Train Accuracy: {train_accuracy:.4f} | Train AUC: {train_auc:.4f}\n")
        
        return xgb_model, run_id


# ==============================================================================
# STEP 2: SELECT BEST MODEL
# ==============================================================================

def select_best_model(
    lr_model: Any,
    lr_run_id: str,
    rf_model: Any,
    rf_run_id: str,
    xgb_model: Any,
    xgb_run_id: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selection_metric: str = "roc_auc"
) -> Tuple[Any, str, str, Dict[str, float]]:
    """
    Select the best model based on quick validation
    
    Args:
        lr_model, rf_model, xgb_model: Trained models
        lr_run_id, rf_run_id, xgb_run_id: MLflow run IDs
        X_train, y_train: Training data
        selection_metric: Metric to use for selection ('accuracy' or 'roc_auc')
    
    Returns:
        best_model: The selected model
        best_model_name: Name of the best model
        best_run_id: MLflow run ID of best model
        all_scores: Dictionary of all model scores
    """
    print(f"\n{'='*60}")
    print("MODEL SELECTION")
    print(f"{'='*60}")
    print(f"Selection metric: {selection_metric}\n")
    
    models = {
        'Logistic Regression': (lr_model, lr_run_id),
        'Random Forest': (rf_model, rf_run_id),
        'XGBoost': (xgb_model, xgb_run_id)
    }
    
    scores = {}
    
    # Evaluate each model
    for name, (model, run_id) in models.items():
        pred = model.predict(X_train)
        proba = model.predict_proba(X_train)[:, 1]
        
        accuracy = accuracy_score(y_train, pred)
        auc = roc_auc_score(y_train, proba)
        
        scores[name] = {
            'accuracy': accuracy,
            'roc_auc': auc,
            'run_id': run_id
        }
        
        print(f"{name:25s} | Accuracy: {accuracy:.4f} | AUC: {auc:.4f}")
    
    # Select best model
    if selection_metric == 'accuracy':
        best_name = max(scores, key=lambda x: scores[x]['accuracy'])
    else:  # roc_auc
        best_name = max(scores, key=lambda x: scores[x]['roc_auc'])
    
    best_model = models[best_name][0]
    best_run_id = scores[best_name]['run_id']
    best_score = scores[best_name][selection_metric]
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"{'='*60}")
    print(f"   {selection_metric}: {best_score:.4f}")
    print(f"   MLflow Run ID: {best_run_id}\n")
    
    # Log comparison to MLflow
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except:
        pass
    
    with mlflow.start_run(run_name=f"model_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("selection_metric", selection_metric)
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("best_run_id", best_run_id)
        
        for name, score_dict in scores.items():
            mlflow.log_metric(f"{name}_accuracy", score_dict['accuracy'])
            mlflow.log_metric(f"{name}_auc", score_dict['roc_auc'])
        
        mlflow.set_tag("stage", "model_selection")
        mlflow.set_tag("best_model", best_name)
    
    # Prepare output scores
    all_scores = {name: score_dict[selection_metric] for name, score_dict in scores.items()}
    
    return best_model, best_name, best_run_id, all_scores


# ==============================================================================
# STEP 3: CROSS-VALIDATION ON BEST MODEL
# ==============================================================================

def cross_validate_model(
    best_model: Any,
    best_model_name: str,
    best_run_id: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 10
) -> Tuple[np.ndarray, float, float]:
    """
    Perform cross-validation on the selected best model
    
    Args:
        best_model: The selected model
        best_model_name: Name of the model
        best_run_id: MLflow run ID
        X_train, y_train: Training data
        cv_folds: Number of CV folds
    
    Returns:
        cv_scores: Array of CV scores
        cv_mean: Mean CV score
        cv_std: Standard deviation of CV scores
    """
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION: {best_model_name}")
    print(f"{'='*60}")
    print(f"Folds: {cv_folds}\n")
    
    # Perform cross-validation
    print("Running cross-validation...")
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=1  # Avoid nested parallelism
    )
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_mean:.4f} (± {cv_std:.4f})")
    
    # Log to MLflow
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except:
        pass
    
    with mlflow.start_run(run_name=f"cv_{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("model_name", best_model_name)
        mlflow.log_param("parent_run_id", best_run_id)
        mlflow.log_param("cv_folds", cv_folds)
        
        mlflow.log_metric("mean_cv_auc", cv_mean)
        mlflow.log_metric("std_cv_auc", cv_std)
        mlflow.log_metric("min_cv_auc", cv_scores.min())
        mlflow.log_metric("max_cv_auc", cv_scores.max())
        
        # Log individual fold scores
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_fold_{i+1}_auc", score)
        
        mlflow.set_tag("stage", "cross_validation")
        mlflow.set_tag("model_type", best_model_name)
    
    print(f"✅ Cross-validation completed\n")
    
    return cv_scores, cv_mean, cv_std


# ==============================================================================
# STEP 4: TEST FINAL MODEL
# ==============================================================================

def test_final_model(
    best_model: Any,
    best_model_name: str,
    best_run_id: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame  # For feature importance
) -> Dict[str, Any]:
    """
    Test the final model on test set and log comprehensive metrics
    
    Args:
        best_model: The selected model
        best_model_name: Name of the model
        best_run_id: MLflow run ID
        X_test, y_test: Test data
        X_train: Training data (for feature names)
    
    Returns:
        results: Dictionary containing all test metrics
    """
    print(f"\n{'='*60}")
    print(f"FINAL TESTING: {best_model_name}")
    print(f"{'='*60}\n")
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance (if available)
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    # Log to MLflow
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except:
        pass
    
    with mlflow.start_run(run_name=f"test_{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("model_name", best_model_name)
        mlflow.log_param("parent_run_id", best_run_id)
        mlflow.log_metric("test_samples", len(X_test))
        
        # Log test metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_auc", test_auc)
        
        # Log feature importance if available
        if feature_importance is not None:
            for idx, row in feature_importance.head(10).iterrows():
                mlflow.log_metric(f"test_feature_importance_{row['feature']}", row['importance'])
        
        mlflow.set_tag("stage", "testing")
        mlflow.set_tag("model_type", best_model_name)
        mlflow.set_tag("status", "completed")
    
    print(f"\n✅ Final testing completed\n")
    
    # Prepare results dictionary
    results = {
        'model': best_model,
        'model_name': best_model_name,
        'run_id': best_run_id,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba,
        'feature_importance': feature_importance,
        'y_true': y_test
    }
    
    return results , y_pred


# ==============================================================================
# HELPER: Combined Training Function (if you want single node)
# ==============================================================================

#def train_all_models(
#    X_train: pd.DataFrame,
#    y_train: pd.Series,
#    lr_params: Dict[str, Any],
#    rf_params: Dict[str, Any],
#    xgb_params: Dict[str, Any]
#) -> Tuple[Any, str, Any, str, Any, str]:
#    """
#    Train all three models in one node (alternative approach)
#    
#    Returns:
#        lr_model, lr_run_id, rf_model, rf_run_id, xgb_model, xgb_run_id
#    """
#    print(f"\n{'='*60}")
#    print("TRAINING ALL MODELS")
#    print(f"{'='*60}\n")
#    
#    # Train each model
#    lr_model, lr_run_id = train_logistic_regression(X_train, y_train, lr_params)
#    rf_model, rf_run_id = train_random_forest(X_train, y_train, rf_params)
#    xgb_model, xgb_run_id = train_xgboost(X_train, y_train, xgb_params)
#    
#    print(f"\n{'='*60}")
#    print("ALL MODELS TRAINED SUCCESSFULLY")
#    print(f"{'='*60}\n")
#    
#    return lr_model, lr_run_id, rf_model, rf_run_id, xgb_model, xgb_run_id
        
        
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