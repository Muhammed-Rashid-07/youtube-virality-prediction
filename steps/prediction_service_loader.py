from zenml import step
from zenml.client import Client
from zenml.services import BaseService

step(enable_cache=False)
def prediction_service_loader() -> BaseService:
    """Load the model of our training_pipeline

    Returns:
        BaseService: _description_
    """
    client = Client()
    model_deployer = client.activate_stack.model_deployer
    services = model_deployer.find_model_server(
        pipeline_name="training_pipeline"
    )
    return services[0]