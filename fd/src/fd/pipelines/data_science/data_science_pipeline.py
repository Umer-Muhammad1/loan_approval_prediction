from kedro.pipeline import Pipeline, node

from .data_science_node import (filter_data , feature_target_split, train_test_df_split, scale_features,
                                one_hot_encode, train_evaluate_xgb , plot_roc_curve)
def create_pipeline(**kwargs):
    data_science_pipeline= Pipeline(
        [
            node(
                func=filter_data, 
                inputs=["data_without_duplicates"],
                outputs="filtered_data", 
                name="filtering_data"
                ),
            
            
            node(
                func=feature_target_split,
                inputs="filtered_data",
                outputs=["features", "target"],
                name="splitting_feature_target",
            ),
            node(
                func=train_test_df_split,
                inputs=["features", "target", "params:test_size", "params:random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test" , "loan_amt_train", "loan_amt_test"],
                name="split_data_node",
            ),
            node(
                func=one_hot_encode,
                inputs=["X_train","X_test"],
                outputs=["X_train_encoded","X_test_encoded"],
                name="encoding_node",
            ),
            node(
                func=scale_features,
                inputs=["X_train_encoded", "X_test_encoded", "params:stand_col"],
                outputs=["X_train_scaled", "X_test_scaled"],
                name="scale_features_node",
            ),
            node(
                func=train_evaluate_xgb,
                inputs=["X_train_scaled", "y_train", "X_test_scaled", "y_test", "params:xgb_params"],
                outputs=["xgb_results","test_accuracy", "test_auc", "cv_scores", "feature_importance","y_pred", "y_pred_proba"],
                name="train_evaluate_xgb_node",
            ),
            node(
                func=plot_roc_curve,
                inputs=["y_test",  "y_pred_proba"],
                outputs="roc_curve_plot",
                name="roc_curve_node",
            ),
            
        ]
    )
    return data_science_pipeline