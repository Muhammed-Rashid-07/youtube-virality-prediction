import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
import time
from zenml import step
from zenml.client import Client
import mlflow
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)
from mlflow import pyfunc


def get_model(model: str) -> ClassifierMixin:
    """
    Create and return an instance of a scikit-learn classifier.

    Args:
        model (str): The name of the model to create.

    Returns:
        ClassifierMixin: An instance of the specified classifier.
    """
    if model == "gradient_boost":
        mlflow.sklearn.autolog()
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    if model == 'ada_boost':
        mlflow.sklearn.autolog()
        weak_classifier = DecisionTreeClassifier(max_depth=1)
        return AdaBoostClassifier(base_estimator=weak_classifier, n_estimators=100)
    if model == 'random_forest':
        mlflow.sklearn.autolog()
        return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)


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

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)  
def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """
    Train a scikit-learn model and save it using MLflow.

    Args:
        model_name (str): The name of the model to train.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.

    Returns:
        ClassifierMixin: The trained model.
    """
    # Create an instance of the specified model
    model = get_model(model_name)

    # Train the model
    print(f"Training the {model_name} model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    mlflow.sklearn.autolog()
    print("Model training completed...")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time taken to complete: {elapsed_time:.2f} seconds")
    return model
    
