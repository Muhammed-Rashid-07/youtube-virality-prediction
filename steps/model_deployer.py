from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_registry_deployer_step
from zenml import step
from steps.register_model import model_name

@step(enable_cache=True)
def deploy_model_step(version:int = 1):
    # Call the mlflow_model_registry_deployer_step to execute it
    mlflow_model_registry_deployer_step(
            registry_model_name=model_name,  # Use the provided model_name
            registry_model_version=version,
            timeout=300,
        )
    
