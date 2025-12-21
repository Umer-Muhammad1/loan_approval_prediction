from kedro.pipeline import Pipeline, node

from .de_nodes import ( quick_validate_csv, remove_duplicates ,filter_data , feature_target_split, train_test_df_split, scale_features,
                                one_hot_encode)

def create_pipeline(**kwargs):
    data_engineering_pipeline= Pipeline(
        [
            node(
                func=quick_validate_csv,
                inputs="raw_data",
                outputs="validated_loan_data",
                name="data_validation_gatekeeper",
                tags="data_engineering",
            ),
            node(
                func=remove_duplicates,
                inputs=["validated_loan_data"],
                outputs="data_without_duplicates",
                name="removing_duplicates",
                tags="data_engineering",
            
            ),
            node(
                func=filter_data, 
                inputs=["data_without_duplicates"],
                outputs="filtered_data", 
                name="filtering_data",
                tags="data_engineering",
                ),
            
            
            node(
                func=feature_target_split,
                inputs="filtered_data",
                outputs=["features", "target"],
                name="splitting_feature_target",
                tags="data_engineering",
            ),
            node(
                func=train_test_df_split,
                inputs=["features", "target", "params:test_size", "params:random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test" , "loan_amt_train", "loan_amt_test"],
                name="split_data_node",
                tags="data_engineering",
            ),
            node(
                func=one_hot_encode,
                inputs=["X_train","X_test"],
                outputs=["X_train_encoded","X_test_encoded"],
                name="encoding_node",
                tags="data_engineering",
            ),
            node(
                func=scale_features,
                inputs=["X_train_encoded", "X_test_encoded", "params:stand_col"],
                outputs=["X_train_scaled", "X_test_scaled"],
                name="scale_features_node",
                tags="data_engineering",
            ),


        ]
    )
    return data_engineering_pipeline