from zenml import step
from typing_extensions import Tuple, Annotated
import joblib
import mlflow
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)
import logging
from sklearn.base import ClassifierMixin

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


@step(experiment_tracker = experiment_tracker.name, enable_cache=False)
def train_models() -> Tuple[
    Annotated[ClassifierMixin,"random_forest"],   
    Annotated[ClassifierMixin,"ada_boost",],   
    Annotated[ClassifierMixin,"gradient_boost",],   
]:  
    logging.info("experiment tracker: ",experiment_tracker.name)
    random_forest = joblib.load('./models/random_forest_model.pkl')
    mlflow.sklearn.autolog()
    ada_boost = joblib.load('./models/ada_boost_model.pkl')
    mlflow.sklearn.autolog()
    gradient_boost = joblib.load('./models/gradient_boost_model.pkl')
    mlflow.sklearn.autolog()
    return random_forest, ada_boost, gradient_boost


