from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
from steps.splitter import sklearn_split_data, drift_splitting
from steps.evaluate import evaluate_model
from steps.model import train_and_save_model, test_model
from steps.monitoring import model_monitoring
from steps.best_model_select import best_model_selector
from steps.register_model import register_model
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from typing_extensions import Tuple, Annotated
from sklearn.base import ClassifierMixin

# Configure Docker settings for the pipeline
docker_settings = DockerSettings(required_integrations={MLFLOW, SKLEARN})

# Define a ZenML pipeline for training and evaluating models
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline(epochs: int = 2, lr: float = 0.001, num_run: int = 1):
    # Ingest raw data
    raw_data = ingest_df()
    
    # Clean the data
    df = cleaning_data(df=raw_data)
    
    # Split the data for training and evaluation
    X_train, X_test, y_train, y_test = sklearn_split_data(df)
    
    # Split the data for drift monitoring
    reference, current = drift_splitting(df)
   
    # Train and save the Random Forest model
    random_forest = train_and_save_model("random_forest", X_train=X_train, y_train=y_train)
    
    # Evaluate the Random Forest model
    predict_random_forest = evaluate_model(model=random_forest, X_test=X_test, y_test=y_test, model_name="Random Forest")
    
    # Train and save the AdaBoost model
    ada_boost = train_and_save_model("ada_boost", X_train=X_train, y_train=y_train)
    
    # Evaluate the AdaBoost model
    predict_ada_boost = evaluate_model(model=ada_boost, X_test=X_test, y_test=y_test, model_name='ada boost')
    
    # Train and save the Gradient Boosting model
    gradient_boost = train_and_save_model("gradient_boost", X_train=X_train, y_train=y_train)
    
    # Evaluate the Gradient Boosting model
    predict_gradient_boost = evaluate_model(model=gradient_boost, X_test=X_test, y_test=y_test, model_name="gradient boost")  
    
    # Select the best model based on test accuracy
    best_model, best_model_test_acc = best_model_selector(X_test=X_test, y_test=y_test, model1=random_forest, model2=ada_boost, model3=gradient_boost)
    
    # Register the best model
    register_model(best_model)
    
    # Monitor the best model using drift reports
    model_monitoring(reference_data=reference, current_data=current, model=best_model)
