"""
This is a boilerplate pipeline 'bussiness_metrics'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline , node

from .bussiness_nodes import calculate_business_impact ,visualize_business_metrics

def create_pipeline(**kwargs):
    bussiness_pipeline= Pipeline(
        [
            node(
                func=calculate_business_impact, 
                inputs=["y_test", "y_pred", "y_pred_proba", "loan_amt_test"],
                outputs="bussiness_metrics", 
                name="business_metrics_node"
                ),
            node(
                func=visualize_business_metrics, 
                inputs="bussiness_metrics",
                outputs="plot_bussiness_metrics", 
                name="visualize_business_metrics_node"
                )
            
            ])
    
    
    
    return bussiness_pipeline