from kedro.pipeline import Pipeline, node

from .de_nodes import ( quick_validate_csv, cast_loan_data_types,remove_duplicates ,filter_data , feature_target_split, train_test_df_split, scale_features,
                                one_hot_encode, generate_correlation_heatmap_by_order, plot_outliers_all_columns, plot_correlation_matrix,
plot_loan_acceptance_by_categorical_features, plot_distributions , plot_categorical_distributions,
plot_categorical_relations, plot_categorical_relations_grade, plot_histograms_kde)

def create_pipeline(**kwargs):
    data_engineering_pipeline= Pipeline(

        [
           # node(
           #     func=generate_correlation_heatmap_by_order, 
           #     inputs="datatypes_casted_data",
           #     outputs="correlation_matrix_by_order", 
           #     name="correlation",
           #     tags=["visualisations", "data_exploration"],
           #     )
           # ,
            node(
                func=plot_outliers_all_columns,
                inputs=["datatypes_casted_data"],
                outputs="columns_outliers_plot",
                name="plot_outliers_all_columns_node",
                tags=["visualisations", "data_exploration"],
            ),
            node(
                func= plot_correlation_matrix,
                inputs=["datatypes_casted_data"],
                outputs= "correlation_matrix",
                name= "correlation_matrix",
                tags=["visualisations", "data_exploration"],
                
            ),
           # node(
           #     func= plot_loan_acceptance_by_categorical_features,
           #     inputs=["datatypes_casted_data", 'params:categorical_features'],
           #     outputs= None,
           #     name= "categorical_features_by_loan_status",
           #     tags=["visualisations", "data_exploration"],
                
            #),
            node(
                func=plot_distributions,
                inputs=["datatypes_casted_data"],
                outputs="distribution_plot",
                name="plot_distributions_node",
                tags=["visualisations", "data_exploration"],
            ),
            
            node(
                func=plot_categorical_distributions,
                inputs="data_without_duplicates",  # Input dataset (e.g., train DataFrame)
                outputs="categorical_features_distribution_plot",  # No outputs as we are showing plots
                name="plot_categorical_distributions_node",
                tags=["visualisations", "data_exploration"],
            ),
           # node(
           #     func=plot_categorical_relations,
           #     inputs=["data_without_duplicates", "params:categorical_features", "params:hue_feature"],  # Input dataset (e.g., train DataFrame)
           #     outputs="categorical_relation_plot",  # No outputs as we are showing plots
           #     name="plot_categorical_relations",
           #     tags=["visualisations", "data_exploration"],
           # ),
           # node(
           #     func=plot_categorical_relations_grade,
           #     inputs=["data_without_duplicates", "params:categorical_features", "params:hue_feature"],  # Input dataset (e.g., train DataFrame)
           #     outputs="categorical_relation_plot_grade",  # No outputs as we are showing plots
           #     name="plot_categorical_relations_grade",
           #     tags=["visualisations", "data_exploration"],
           # ),
            
            
            node(
                func=plot_histograms_kde,
                inputs=["data_without_duplicates", "params:hist_columns", "params:hue_column"],
                outputs="kde_histogram",
                name="plot_histograms_node",
                tags=["visualisations", "data_exploration"],
                ),
            node(
                func=quick_validate_csv,
                inputs="raw_data",
                outputs="validated_loan_data",
                name="data_validation_gatekeeper",
                tags="data_engineering",
            ),
            node(
                func=cast_loan_data_types,
                inputs=["validated_loan_data"],
                outputs="datatypes_casted_data",
                name="data_types_casted",
                tags="data_engineering",
            
            ),
            node(
                func=remove_duplicates,
                inputs=["datatypes_casted_data"],
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
                inputs=["features", "target"],
                outputs=["X_train", "X_test", "y_train", "y_test" ,  "loan_amt_test"],
                name="split_data_node",
                tags="data_engineering",
            ),
            node(
                func=one_hot_encode,
                inputs=["X_train","X_test"],
                outputs=["X_train_encoded","X_test_encoded" , "fitted_encoder"],
                name="encoding_node",
                tags="data_engineering",
            ),
            node(
                func=scale_features,
                inputs=["X_train_encoded", "X_test_encoded", "params:stand_col"],
                outputs=["X_train_scaled", "X_test_scaled", "fitted_scaler"],
                name="scale_features_node",
                tags="data_engineering",
            ),


        ]
    )
    return data_engineering_pipeline