from kedro.pipeline import Pipeline, node

from .model_nodes import ( train_logistic_regression,
    train_random_forest,
    train_xgboost,
    select_best_model,
    cross_validate_model,
    test_final_model)
def create_pipeline(**kwargs):
    data_science_pipeline= Pipeline(
        [
            node(
                func=train_logistic_regression,
                inputs=["X_train_scaled", "y_train", "params:lr_params"],
                outputs="lr_model",
                name="train_logistic_regression_node",
                tags=["model_training", "lr"]
            ),

            # Train Random Forest
            node(
                func=train_random_forest,
                inputs=["X_train_scaled", "y_train", "params:rf_params"],
                outputs="rf_model",
                name="train_random_forest_node",
                tags=["model_training", "rf"]
            ),

            # Train XGBoost
            node(
                func=train_xgboost,
                inputs=["X_train_scaled", "y_train", "params:xgb_params"],
                outputs="xgb_model",
                name="train_xgboost_node",
                tags=["model_training", "xgb"]
            ),

            # ==================== MODEL SELECTION STAGE ====================
            node(
                func=select_best_model,
                inputs=[
                    "lr_model", 
                    "rf_model",
                    "xgb_model", 
                    "X_train_scaled", "y_train",
                    "params:selection_metric"
                ],
                outputs=["best_model", "best_model_name", "all_model_scores"],
                name="select_best_model_node",
                tags=["model_training" , "model_selection"]
            ),

            # ==================== CROSS-VALIDATION STAGE ====================
            node(
                func=cross_validate_model,
                inputs=[
                    "best_model",
                    "best_model_name",
                    #"best_run_id",
                    "X_train_scaled", "y_train",
                    "params:cv_folds"
                ],
                outputs=["cv_scores", "cv_mean", "cv_std"],
                name="cross_validate_best_model_node",
                tags=["model_training" , "cross_validation"]
            ),

            # ==================== TESTING STAGE ====================
            node(
                func=test_final_model,
                inputs=[
                    "best_model",
                    "best_model_name",
                   # "best_run_id",
                    "X_test_scaled",
                    "y_test",
                    "X_train_scaled"  # For feature names
                ],
                outputs=["final_model_results","y_pred"],
                name="test_final_model_node",
                tags=["model_training" , "testing"]
            ),
            #node(
            #    func=train_evaluate_xgb,
            #    inputs=["X_train_scaled", "y_train", "X_test_scaled", "y_test", "params:xgb_params"],
            #    outputs=["xgb_results","test_accuracy", "test_auc", "cv_scores", "feature_importance","y_pred", "y_pred_proba"],
            #    name="train_evaluate_xgb_node",
            #),
            #node(
            #    func=plot_roc_curve,
            #    inputs=["y_test",  "y_pred_proba"],
            #    outputs="roc_curve_plot",
            #    name="roc_curve_node",
            #),
            
        ]
    )
    return data_science_pipeline