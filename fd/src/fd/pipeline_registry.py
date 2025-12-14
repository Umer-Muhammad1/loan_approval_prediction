"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
#
from fd.pipelines.eda import eda_pipeline
from fd.pipelines.data_science import data_science_pipeline
from fd.pipelines.bussiness import bussiness_pipeline

def register_pipelines() -> dict:
    """Register the project's pipelines."""
    
    # Register the preprocessing pipeline
    exploratory_data_analysis_pipeline = eda_pipeline.create_pipeline()
    data_science_pipeline_part = data_science_pipeline.create_pipeline()
    bussiness_pipeline_part = bussiness_pipeline.create_pipeline()

    # Placeholder for model training pipeline (currently empty)
    model_training_pipeline = Pipeline([])  # Empty pipeline for now

    # Combine pipelines if needed for default pipeline
    __default__ = exploratory_data_analysis_pipeline+ model_training_pipeline + data_science_pipeline_part + bussiness_pipeline_part
    
    return {
        "__default__": __default__,
        "exploratory_data_analysis_pipeline": exploratory_data_analysis_pipeline,
        "data_science_pipeline": data_science_pipeline_part,
        "model_training": model_training_pipeline,
        "bussiness_pipeline": bussiness_pipeline_part,
    }