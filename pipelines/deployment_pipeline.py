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
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
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
        accuracy_score_rf = evaluate_model(model=random_forest, X_test=X_test,y_test=y_test, model_name="Random Forest")

        # ada_boost = test_model("ada-boost-model",stage="Production")
        # accuracy_score_ad = evaluate_model(model=ada_boost,X_test=X_test, y_test=y_test, model_name='ada boost')

        # gradient_boost = test_model(model_name="gradient-boost-model",stage="Production")
        # accuracy_score_gd = evaluate_model(model=gradient_boost,X_test=X_test, y_test=y_test,model_name="gradient boost")

        model_deploy1 = deploy_model_step(model_name="Random-forest-model")
        
    except Exception as e:
        logging.error("Error in deployment trigger: %s", e)
        raise e
    
    
