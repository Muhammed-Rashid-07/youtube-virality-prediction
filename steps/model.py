import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
import joblib
import time
from zenml import step
from typing_extensions import Tuple, Annotated
from zenml.client import Client

from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)

def get_model(model:str) -> ClassifierMixin:
    if model == "gradient_boost":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    if model == 'ada_boost':
        weak_classifier = DecisionTreeClassifier(max_depth=1)
        return AdaBoostClassifier(base_estimator=weak_classifier, n_estimators=100)
    if model == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


@step(experiment_tracker = experiment_tracker.name, enable_cache=False)  
def train_and_save_model(model_name:str, X_train:pd.DataFrame, y_train:pd.Series) -> ClassifierMixin:
    model = get_model(model_name)

    print(f"Training the {model_name} model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print("Model training completed...")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time taken to complete: {elapsed_time:.2f} seconds")
    return model
    # Save the trained model for later use
    #joblib.dump(model, f"../models/{model_name}_model.pkl")
    #print(f"{model_name} model saved as {model_name}_model.joblib")


