import pandas as pd

from zenml import step
import joblib
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def evaluate_model(model:ClassifierMixin, X_test:pd.DataFrame, y_test:pd.Series,model_name:str) -> float:
    y_pred = model.predict(X_test)
    accuracy_scr = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision_scr = precision_score(y_true=y_test, y_pred=y_pred)
    recall_scr = recall_score(y_true=y_test, y_pred=y_pred)
    f1_scr = f1_score(y_pred=y_pred, y_true=y_test)
    logging.info(f"model: {model_name},\naccuracy score: {accuracy_scr}, \nprecision score: {precision_scr}, \nrecall score: {recall_scr}, \nf1_score: {f1_scr}")
    logging.info("experiment tracker: ",experiment_tracker.name)
    return accuracy_scr



