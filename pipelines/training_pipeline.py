from zenml import pipeline
import pandas as pd
from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
from steps.splitter import sklearn_split_data, drift_splitting
from steps.evaluate import evaluate_model
from steps.train_model import train_models
from steps.model import train_and_save_model
import joblib
from typing_extensions import Tuple,Annotated
from sklearn.base import ClassifierMixin
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN

docker_settings = DockerSettings(required_integrations = {MLFLOW, SKLEARN})

@pipeline(enable_cache = False, settings = {"docker":docker_settings})
def train_pipeline(  epochs: int = 2, lr: float = 0.001) -> Tuple[
    Annotated[ClassifierMixin,"random_forest"],
    Annotated[ClassifierMixin, "ada_boost"],
    Annotated[ClassifierMixin, "gradient_boost"],
]:
    raw_data = ingest_df()
    df = cleaning_data(df=raw_data)
    X_train, X_test, y_train, y_test = sklearn_split_data(df)
    #random_forest, ada_boost, gradient_boost = train_models()
    random_forest = train_and_save_model("random_forest",X_train=X_train,y_train=y_train)
    ada_boost = train_and_save_model("ada_boost",X_train=X_train,y_train=y_train)
    gradient_boost = train_and_save_model("gradient_boost",X_train=X_train,y_train=y_train)
    
    predict_random_forest = evaluate_model(model=random_forest, X_test=X_test,y_test=y_test, model_name="Random Forest")
    predict_ada_boost = evaluate_model(model=ada_boost,X_test=X_test, y_test=y_test, model_name='ada boost')
    predict_gradient_boost = evaluate_model(model=gradient_boost,X_test=X_test, y_test=y_test,model_name="gradient boost")
    referene, current = drift_splitting(df)
    return random_forest, ada_boost, gradient_boost
  
    