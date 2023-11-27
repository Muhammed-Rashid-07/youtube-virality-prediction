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
from steps.best_model_select import best_model_selector
from steps.register_model import register_model
import mlflow




docker_settings = DockerSettings(required_integrations = {MLFLOW, SKLEARN})

@pipeline(enable_cache = False, settings = {"docker":docker_settings})
def train_pipeline(  epochs: int = 2, lr: float = 0.001, num_run:int = 1):
    raw_data = ingest_df()
    df = cleaning_data(df=raw_data)
    X_train, X_test, y_train, y_test = sklearn_split_data(df)
    reference, current = drift_splitting(df)
   
    random_forest = train_and_save_model("random_forest",X_train=X_train, y_train=y_train)
    predict_random_forest = evaluate_model(model=random_forest, X_test=X_test,y_test=y_test, model_name="Random Forest")
    
    
    ada_boost = train_and_save_model("ada_boost",X_train=X_train, y_train=y_train)
    predict_ada_boost = evaluate_model(model=ada_boost,X_test=X_test, y_test=y_test, model_name='ada boost')
    
    

    gradient_boost = train_and_save_model("gradient_boost",X_train=X_train, y_train=y_train)
    predict_gradient_boost = evaluate_model(model=gradient_boost,X_test=X_test, y_test=y_test,model_name="gradient boost")  
    
    
    
    best_model, best_model_test_acc = best_model_selector(X_test=X_test, y_test=y_test, model1=random_forest, model2=ada_boost, model3=gradient_boost)
    
    register_model(best_model)
    model_monitoring(reference_data=reference, current_data=current, model=best_model)
    
    
    
