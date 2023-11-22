import numpy as np
import pandas as pd
import logging
import os
import json
from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from zenml.steps import BaseParameters, Output
from zenml.integrations.constants import MLFLOW
import warnings

from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
from steps.splitter import sklearn_split_data, drift_splitting
from steps.evaluate import evaluate_model
from steps.model import test_model
from steps.monitoring import model_monitoring
from steps.model_deployer import deploy_model_step
from pipelines.load_test_data import load_test_data_model

warnings.filterwarnings("ignore")
# Define Docker settings with MLflow integration
docker_settings = DockerSettings(required_integrations = {MLFLOW})


class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger config."""
    min_accuracy:float = 0.90
    

#Define step which trigger the deployment only the accuracy score met the  threshold score.
@step(enable_cache=False)
def deployment_trigger(
    accuracy:float,
    config: DeploymentTriggerConfig,
):
    """
    It trigger the deployment only if accuracy is greater than min accuracy.
    Args:
        accuracy: accuracy of the model.
        config: Minimum accuracy thereshold.
    """
    try:
        return accuracy >= config.min_accuracy
    except Exception as e:
        logging.error("Error in deployment trigger",e)
        raise e
    
    
#Define a continuous deployment pipeline
@pipeline(enable_cache=False,settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    min_accuracy:float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    try:
        raw_data = ingest_df()
        df = cleaning_data(df=raw_data)
        X_train, X_test, y_train, y_test = sklearn_split_data(df)
        random_forest = test_model(model_name="Random-forest-model",stage="Production")
        predict_random_forest = evaluate_model(model=random_forest, X_test=X_test,y_test=y_test, model_name="Random Forest")  
        deploy_model_step(model_name="Random-forest-model",version=7) 
    except Exception as e:
        logging.error("Error in deployment trigger: %s", e)
        raise e
    
    
