from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_registry_deployer_step
from zenml import step

@step(enable_cache=False)
def deploy_model_step(model_name:str):
    # Call the mlflow_model_registry_deployer_step to execute it
    mlflow_model_registry_deployer_step(
        registry_model_name=model_name,  # Use the provided model_name
        registry_model_version=6,
        timeout=300,
    )
