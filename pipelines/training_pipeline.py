from zenml import pipeline
import pandas as pd
from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
from steps.splitter import sklearn_split_data, drift_splitting
from steps.evaluate import evaluate_model
from steps.model import train_and_save_model,test_model
import joblib
from typing_extensions import Tuple,Annotated
from sklearn.base import ClassifierMixin
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,   
)
from zenml.model_registries.base_model_registry import (
    ModelRegistryModelMetadata,
)
from steps.monitoring import model_monitoring
import mlflow



docker_settings = DockerSettings(required_integrations = {MLFLOW, SKLEARN})

@pipeline(enable_cache = False, settings = {"docker":docker_settings})
def train_pipeline(  epochs: int = 2, lr: float = 0.001, num_run:int = 1):
    raw_data = ingest_df()
    df = cleaning_data(df=raw_data)
    X_train, X_test, y_train, y_test = sklearn_split_data(df)
    #random_forest, ada_boost, gradient_boost = train_models()
    
    
    random_forest = train_and_save_model("random_forest",X_train=X_train, y_train=y_train)
    predict_random_forest = evaluate_model(model=random_forest, X_test=X_test,y_test=y_test, model_name="Random Forest")
    mlflow_register_model_step.with_options(
        parameters=dict(
            name="Random-forest-model",
            metadata=ModelRegistryModelMetadata(
                lr=lr, epochs=epochs, optimizer=None
            ),
            description=f"Run #{num_run} of the mlflow_registry_training_pipeline.")
        )(random_forest)
    
    
    ada_boost = train_and_save_model("ada_boost",X_train=X_train, y_train=y_train)
    # ada_boost = test_model("ada-boost-model",stage="Production")
    predict_ada_boost = evaluate_model(model=ada_boost,X_test=X_test, y_test=y_test, model_name='ada boost')
    mlflow_register_model_step.with_options(
        parameters=dict(
            name="ada-boost-model",
            metadata=ModelRegistryModelMetadata(
                lr=lr, epochs=epochs, optimizer=None
            ),
            description=f"Run #{num_run} of the mlflow_registry_training_pipeline.")
        )(ada_boost)
    
    
    # gradient_boost = test_model(model_name="gradient-boost-model",stage="Production")
    gradient_boost = train_and_save_model("gradient_boost",X_train=X_train, y_train=y_train)
    predict_gradient_boost = evaluate_model(model=gradient_boost,X_test=X_test, y_test=y_test,model_name="gradient boost")  
    mlflow_register_model_step.with_options(
        parameters=dict(
            name="gradient-boost-model",
            metadata=ModelRegistryModelMetadata(
                lr=lr, epochs=epochs, optimizer=None
            ),
            description=f"Run #{num_run} of the mlflow_registry_training_pipeline.")
        )(gradient_boost)  
    
    reference_data, current_data = drift_splitting(df)
    
    monitoring_first_model = model_monitoring(reference_data, current_data, model=random_forest)
    monitoring_second_model = model_monitoring(reference_data, current_data, model=ada_boost)
    monitoring_third_model = model_monitoring(reference_data, current_data, model=gradient_boost)
    
  
    
