from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from zenml.client import Client
from zenml import step
from zenml.services import BaseService


@step(enable_cache=False)
def prediction_service_loader(pipeline_name: str) -> BaseService:
    """
    Load the prediction service associated with the specified pipeline name.

    Args:
        pipeline_name (str): Name of the pipeline for which the prediction service is loaded.

    Returns:
        BaseService: Loaded prediction service associated with the specified pipeline.
    """
    # Get the active ZenML client
    client = Client()

    # Get the active model deployer from the client's active stack
    model_deployer = client.active_stack.model_deployer

    # Find the model server (prediction service) for the specified pipeline name
    # Note: It returns a list of services, and we assume the first one is the desired service
    service = model_deployer.find_model_server(pipeline_name=pipeline_name)

    # Return the loaded prediction service
    return service[0]
