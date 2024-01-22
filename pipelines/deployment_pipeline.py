import numpy as np
import pandas as pd
import warnings
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN
from zenml.client import Client
from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
from steps.splitter import sklearn_split_data, drift_splitting
from steps.model_deployer import deploy_model_step
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor


# Configure warnings
warnings.filterwarnings("ignore")

# Define Docker settings with MLflow and SKlearn integration
docker_settings = DockerSettings(required_integrations=[MLFLOW, SKLEARN])

# Define the deployment and prediction pipeline
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def deploy_and_predict() -> None:
    # Define execution order for pipeline steps
    prediction_service_loader.after(deploy_model_step)
    deploy_model_step()

    # Ingest and clean data
    data = ingest_df()
    df = cleaning_data(df=data)

    # Split data for inference and drift monitoring
    _, inference_data, _, _ = sklearn_split_data(df)
    reference_data, current_data = drift_splitting(df)

    model_deployment_service = prediction_service_loader('train_pipeline')

    # Make predictions using the deployed model
    predictor(service=model_deployment_service, data=inference_data)
