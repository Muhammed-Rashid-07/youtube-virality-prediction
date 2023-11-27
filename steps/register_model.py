from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)

# Set the name for the model to be registered
model_name = "Best Model"

# Define options for the MLFlow registry step
register_model = mlflow_register_model_step.with_options(
    parameters=dict(
        name=model_name,
        description=f"The best model from the training pipeline: {model_name}."
    )
)