
from pipelines.training_pipeline import train_pipeline

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)
from zenml.model_registries.base_model_registry import (
    ModelRegistryModelMetadata,
)

docker_settings = DockerSettings(required_integrations=[MLFLOW, SKLEARN])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def mlflow_registry_training_pipeline(
    epochs: int = 1,
    lr: float = 0.001,
    num_run: int = 1,
):
    random_forest, ada_boost, gradient_boost = train_pipeline(epochs=epochs, lr=lr)
    mlflow_register_model_step(
        model=random_forest,
        name="Random-forest-model",
        metadata=ModelRegistryModelMetadata(
            lr=lr, epochs=epochs, optimizer="Rashid"
        ),
        description=(
            f"Run #{num_run} of the mlflow_registry_training_pipeline."
        ),
    )