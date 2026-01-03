"""
This is a boilerplate pipeline 'bussiness_metrics'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline , node

from .bussiness_nodes import ( segment_portfolio , 
                              plot_feature_importance ,generate_feature_importance, generate_roc_auc_plot , calculate_risk_metrics,
                              loan_economics_parameters , calculate_model_portfolio_financials, calculate_baseline_approve_all
                              , plot_business_summary , log_business_metrics , create_business_dashboard
                              )

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
                func=segment_portfolio, 
                inputs=["y_test", "y_pred", "loan_amt_test"],
                outputs="portfolio_segments",
                name="segment_portfolio_node",
                tags=["bussiness"]
                ),

           # node(
           #     func=loan_economics_parameters,
           #     inputs="params:loan_economics",
           #     outputs="loan_economics",
           #     name="loan_economics_node",
           #     tags=["bussiness"]
           # ),

            # 3️⃣ Model portfolio financials
            node(
                func=calculate_model_portfolio_financials,
                inputs=[
                    "portfolio_segments",
                    "params:loan_economics",
                ],
                outputs="model_financials",
                name="model_portfolio_financials_node",
                tags=["bussiness"]
            ),

            # 4️⃣ Baseline (approve all)
            node(
                func=calculate_baseline_approve_all,
                inputs=[
                    "y_test",
                    "loan_amt_test",
                    "params:loan_economics",
                ],
                outputs="baseline_financials",
                name="baseline_financials_node",
                tags=["bussiness"]
            ),

            # 5️⃣ Risk metrics
            node(
                func=calculate_risk_metrics,
                inputs="portfolio_segments",
                outputs="risk_metrics",
                name="risk_metrics_node",
                tags=["bussiness"]
            ),

            # 6️⃣ Visualization (optional, non-blocking)
           # node(
           #     func=plot_business_summary,
           #     inputs=[
           #         "model_financials",
           #         "baseline_financials",
           #     ],
           #     outputs="plot_summary",
           #     name="business_dashboard_node",
           #     tags=["bussiness"]
           # ),

            # 7️⃣ Centralized MLflow logging
            node(
                func=log_business_metrics,
                inputs=[
                    "model_financials",
                    "baseline_financials",
                    "risk_metrics",
                ],
                outputs=None,
                name="mlflow_business_metrics_logging_node",
                tags=["bussiness"]
            ),
            node(
                func=create_business_dashboard,
                inputs=["model_financials", "baseline_financials", "risk_metrics"],
                outputs="business_dashboard_plot",
                name="create_business_dashboard_node",
                tags=["bussiness"]
            )






        #    node(
        #        func=calculate_business_impact, 
        #        inputs=["y_test", "y_pred", "loan_amt_test"],
        #        outputs="bussiness_metrics", 
        #        name="business_metrics_node",
        #        tags=["bussiness"]
        #        ),
        #    node(
        #        func=visualize_business_metrics, 
        #        inputs="bussiness_metrics",
        #        outputs="plot_bussiness_metrics", 
        #        name="visualize_business_metrics_node",
        #        tags=["bussiness","visualisations"]
        #        )
        #    
           ])
    
    
    
    return bussiness_pipeline