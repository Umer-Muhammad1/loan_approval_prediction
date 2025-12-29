"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
#
#from fd.pipelines.data_processing import dp_pipeline as dp
from fd.pipelines.data_engineering import de_pipeline as de
from fd.pipelines.modelling import model_pipeline as mp
from fd.pipelines.bussiness import bussiness_pipeline as bp

def register_pipelines() -> dict:
    """Register the project's pipelines."""
    
    # Register the preprocessing pipeline
    #exploratory_data_analysis_pipeline = dp.create_pipeline()
    model_pipeline_part = mp.create_pipeline()
    bussiness_pipeline_part = bp.create_pipeline()
    data_engineering_pipeline_part = de.create_pipeline()

    # Placeholder for model training pipeline (currently empty)
    model_training_pipeline = Pipeline([])  # Empty pipeline for now

    # Combine pipelines if needed for default pipeline
    __default__ =  model_pipeline_part + bussiness_pipeline_part+data_engineering_pipeline_part
    
    return {
        "__default__": __default__,
       # "data_processing": exploratory_data_analysis_pipeline,
        "data_engineering": data_engineering_pipeline_part,
        "modelling_pipeline": model_pipeline_part,
        
        "bussiness_pipeline": bussiness_pipeline_part,
    }