import numpy as np
import pandas as pd
import logging
import os
import json
from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.client import Client
from zenml.steps import BaseParameters, Output
from zenml.integrations.constants import MLFLOW,SKLEARN
import warnings


from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
from steps.splitter import sklearn_split_data, drift_splitting
from steps.evaluate import evaluate_model
from steps.model import test_model, train_and_save_model
from steps.model_deployer import deploy_model_step
from steps.load_test_data import load_test_data_model
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor

from steps.monitoring import model_monitoring


from mlflow.tracking import MlflowClient
from typing_extensions import Tuple,Annotated


warnings.filterwarnings("ignore")
# Define Docker settings with MLflow integration
docker_settings = DockerSettings(required_integrations = [MLFLOW,SKLEARN])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def deploy_and_predict() -> None:
    prediction_service_loader.after(deploy_model_step)
    deploy_model_step()
    data = ingest_df()
    df = cleaning_data(df=data)
    _, inference_data, _, _ = sklearn_split_data(df)
    model_deployment_service = prediction_service_loader('train_pipeline')
    reference_data, current_data = drift_splitting(df)
    model_monitoring(reference_data=reference_data, current_data=current_data, model=model_deployment_service)
    predictor(service=model_deployment_service, data=inference_data)

