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
import mlflow
from mlflow.pyfunc import PyFuncModel
from typing_extensions import Union


# Get the active MLFlow experiment tracker from the ZenML client
experiment_tracker = Client().active_stack.experiment_tracker


# Ensure that the active stack contains an MLFlow experiment tracker
if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


# Define a ZenML step for model evaluation using MLFlow
@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def evaluate_model(
    model: Union[PyFuncModel, ClassifierMixin],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> float:
    """
    Evaluate a machine learning model and log metrics to MLFlow.

    Args:
        model (PyFuncModel | ClassifierMixin): The trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model_name (str): Name of the model for identification.

    Returns:
        float: Accuracy score of the model on the test set.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and log evaluation metrics using MLFlow
    accuracy_scr = accuracy_score(y_true=y_test, y_pred=y_pred)
    mlflow.log_metric("accuracy", accuracy_scr)
    precision_scr = precision_score(y_true=y_test, y_pred=y_pred)
    mlflow.log_metric("precision", precision_scr)
    recall_scr = recall_score(y_true=y_test, y_pred=y_pred)
    mlflow.log_metric("recall", recall_scr)
    f1_scr = f1_score(y_pred=y_pred, y_true=y_test)
    mlflow.log_metric("f1_score", f1_scr)

    # Log evaluation results and experiment tracker name
    logging.info(
        f"Model: {model_name},\nAccuracy Score: {accuracy_scr}, "
        f"\nPrecision Score: {precision_scr}, \nRecall Score: {recall_scr}, "
        f"\nF1 Score: {f1_scr}"
    )
    logging.info("Experiment Tracker: ", experiment_tracker.name)

    # Return the accuracy score
    return accuracy_scr
