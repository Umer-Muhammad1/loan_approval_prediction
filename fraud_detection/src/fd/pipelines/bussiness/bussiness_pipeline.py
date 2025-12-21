"""
This is a boilerplate pipeline 'bussiness_metrics'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline , node

from .bussiness_nodes import (calculate_business_impact ,visualize_business_metrics , 
                              plot_feature_importance ,generate_feature_importance, generate_roc_auc_plot)

def create_pipeline(**kwargs):
    bussiness_pipeline= Pipeline(
        [
            node(
                func=generate_feature_importance,
                inputs=[
                    "final_model_results",
                    "X_train_scaled",
                ],
                outputs="feature_importance",
                name="feature_importance_node",
                tags=["evaluation"],
            ),
            node(
                func=plot_feature_importance,
                inputs="feature_importance",
                outputs="feature_importance_plot",
                name="feature_importance_plot_node",
                tags=["evaluation","visualisations"],
            ),

            node(
                func=generate_roc_auc_plot,
                inputs="final_model_results",
                outputs="roc_auc_plot",
                name="roc_auc_plot_node",
                tags=["evaluation", "visualisations"],
            ),
            node(
                func=calculate_business_impact, 
                inputs=["y_test", "y_pred", "loan_amt_test"],
                outputs="bussiness_metrics", 
                name="business_metrics_node",
                tags=["bussiness"]
                ),
            node(
                func=visualize_business_metrics, 
                inputs="bussiness_metrics",
                outputs="plot_bussiness_metrics", 
                name="visualize_business_metrics_node",
                tags=["bussiness","visualisations"]
                )
            
            ])
    
    
    
    return bussiness_pipeline