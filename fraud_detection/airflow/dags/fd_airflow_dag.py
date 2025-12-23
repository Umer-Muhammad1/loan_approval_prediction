from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project


class KedroOperator(BaseOperator):
    @apply_defaults
    def __init__(
        self,
        package_name: str,
        pipeline_name: str,
        node_name: str | list[str],
        project_path: str | Path,
        env: str,
        conf_source: str,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.package_name = package_name
        self.pipeline_name = pipeline_name
        self.node_name = node_name
        self.project_path = project_path
        self.env = env
        self.conf_source = conf_source

    def execute(self, context):
        configure_project(self.package_name)
        with KedroSession.create(self.project_path, env=self.env, conf_source=self.conf_source) as session:
            if isinstance(self.node_name, str):
                self.node_name = [self.node_name]
            session.run(self.pipeline_name, node_names=self.node_name)

# Kedro settings required to run your pipeline
env = "airflow"
pipeline_name = "__default__"
project_path = Path.cwd()
package_name = "fd"
conf_source = "" or Path.cwd() / "conf"


# Using a DAG context manager, you don't have to specify the dag property of each task
with DAG(
    dag_id="fd",
    start_date=datetime(2023,1,1),
    max_active_runs=3,
    # https://airflow.apache.org/docs/stable/scheduler.html#dag-runs
    schedule_interval="@once",
    catchup=False,
    # Default settings applied to all tasks
    default_args=dict(
        owner="airflow",
        depends_on_past=False,
        email_on_failure=False,
        email_on_retry=False,
        retries=1,
        retry_delay=timedelta(minutes=5)
    )
) as dag:
    tasks = {
        "data-validation-gatekeeper": KedroOperator(
            task_id="data-validation-gatekeeper",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="data_validation_gatekeeper",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "data-types-casted": KedroOperator(
            task_id="data-types-casted",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="data_types_casted",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "categorical-features-by-loan-status": KedroOperator(
            task_id="categorical-features-by-loan-status",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="categorical_features_by_loan_status",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "correlation": KedroOperator(
            task_id="correlation",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="correlation",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "correlation-matrix": KedroOperator(
            task_id="correlation-matrix",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="correlation_matrix",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "plot-distributions-node": KedroOperator(
            task_id="plot-distributions-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="plot_distributions_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "plot-outliers-all-columns-node": KedroOperator(
            task_id="plot-outliers-all-columns-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="plot_outliers_all_columns_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "removing-duplicates": KedroOperator(
            task_id="removing-duplicates",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="removing_duplicates",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "filtering-data": KedroOperator(
            task_id="filtering-data",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="filtering_data",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "plot-categorical-distributions-node": KedroOperator(
            task_id="plot-categorical-distributions-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="plot_categorical_distributions_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "plot-categorical-relations": KedroOperator(
            task_id="plot-categorical-relations",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="plot_categorical_relations",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "plot-categorical-relations-grade": KedroOperator(
            task_id="plot-categorical-relations-grade",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="plot_categorical_relations_grade",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "plot-histograms-node": KedroOperator(
            task_id="plot-histograms-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="plot_histograms_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "splitting-feature-target": KedroOperator(
            task_id="splitting-feature-target",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="splitting_feature_target",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "split-data-node": KedroOperator(
            task_id="split-data-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="split_data_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "encoding-node": KedroOperator(
            task_id="encoding-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="encoding_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "scale-features-node": KedroOperator(
            task_id="scale-features-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="scale_features_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-logistic-regression-node": KedroOperator(
            task_id="train-logistic-regression-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_logistic_regression_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-random-forest-node": KedroOperator(
            task_id="train-random-forest-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_random_forest_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-xgboost-node": KedroOperator(
            task_id="train-xgboost-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_xgboost_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "select-best-model-node": KedroOperator(
            task_id="select-best-model-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="select_best_model_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "cross-validate-best-model-node": KedroOperator(
            task_id="cross-validate-best-model-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="cross_validate_best_model_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "test-final-model-node": KedroOperator(
            task_id="test-final-model-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="test_final_model_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "business-metrics-node": KedroOperator(
            task_id="business-metrics-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="business_metrics_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "feature-importance-node": KedroOperator(
            task_id="feature-importance-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="feature_importance_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "roc-auc-plot-node": KedroOperator(
            task_id="roc-auc-plot-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="roc_auc_plot_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "feature-importance-plot-node": KedroOperator(
            task_id="feature-importance-plot-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="feature_importance_plot_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "visualize-business-metrics-node": KedroOperator(
            task_id="visualize-business-metrics-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="visualize_business_metrics_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        )
    }
    tasks["data-validation-gatekeeper"] >> tasks["data-types-casted"]
    tasks["data-types-casted"] >> tasks["categorical-features-by-loan-status"]
    tasks["data-types-casted"] >> tasks["correlation"]
    tasks["data-types-casted"] >> tasks["correlation-matrix"]
    tasks["data-types-casted"] >> tasks["plot-distributions-node"]
    tasks["data-types-casted"] >> tasks["plot-outliers-all-columns-node"]
    tasks["data-types-casted"] >> tasks["removing-duplicates"]
    tasks["removing-duplicates"] >> tasks["filtering-data"]
    tasks["removing-duplicates"] >> tasks["plot-categorical-distributions-node"]
    tasks["removing-duplicates"] >> tasks["plot-categorical-relations"]
    tasks["removing-duplicates"] >> tasks["plot-categorical-relations-grade"]
    tasks["removing-duplicates"] >> tasks["plot-histograms-node"]
    tasks["filtering-data"] >> tasks["splitting-feature-target"]
    tasks["splitting-feature-target"] >> tasks["split-data-node"]
    tasks["split-data-node"] >> tasks["encoding-node"]
    tasks["encoding-node"] >> tasks["scale-features-node"]
    tasks["scale-features-node"] >> tasks["train-logistic-regression-node"]
    tasks["split-data-node"] >> tasks["train-logistic-regression-node"]
    tasks["scale-features-node"] >> tasks["train-random-forest-node"]
    tasks["split-data-node"] >> tasks["train-random-forest-node"]
    tasks["scale-features-node"] >> tasks["train-xgboost-node"]
    tasks["split-data-node"] >> tasks["train-xgboost-node"]
    tasks["train-xgboost-node"] >> tasks["select-best-model-node"]
    tasks["scale-features-node"] >> tasks["select-best-model-node"]
    tasks["train-logistic-regression-node"] >> tasks["select-best-model-node"]
    tasks["train-random-forest-node"] >> tasks["select-best-model-node"]
    tasks["split-data-node"] >> tasks["select-best-model-node"]
    tasks["scale-features-node"] >> tasks["cross-validate-best-model-node"]
    tasks["split-data-node"] >> tasks["cross-validate-best-model-node"]
    tasks["select-best-model-node"] >> tasks["cross-validate-best-model-node"]
    tasks["scale-features-node"] >> tasks["test-final-model-node"]
    tasks["split-data-node"] >> tasks["test-final-model-node"]
    tasks["select-best-model-node"] >> tasks["test-final-model-node"]
    tasks["split-data-node"] >> tasks["business-metrics-node"]
    tasks["test-final-model-node"] >> tasks["business-metrics-node"]
    tasks["scale-features-node"] >> tasks["feature-importance-node"]
    tasks["test-final-model-node"] >> tasks["feature-importance-node"]
    tasks["test-final-model-node"] >> tasks["roc-auc-plot-node"]
    tasks["feature-importance-node"] >> tasks["feature-importance-plot-node"]
    tasks["business-metrics-node"] >> tasks["visualize-business-metrics-node"]