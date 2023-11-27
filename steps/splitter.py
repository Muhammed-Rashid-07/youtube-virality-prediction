# Import necessary libraries and modules
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from zenml import step
from typing_extensions import Tuple, Annotated

# Define a class for handling data splitting
class SplitData:
    def sklearn_split_train(df: pd.DataFrame):
        try:
            # Split the DataFrame into features (X) and target (y)
            X = df.drop(['is_viral'], axis=1)
            y = df['is_viral']
            
            # Use sklearn's train_test_split to split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            # Log and raise an error if an exception occurs
            logging.error(e)
            raise e
    
    def reference_current_split(df: pd.DataFrame):
        try:
            # Split the DataFrame into reference and current sets
            reference, current = train_test_split(df, test_size=0.2, random_state=23)
            return reference, current
        except Exception as e:
            # Log and raise an error if an exception occurs
            logging.error(e)
            raise e

# Define a ZenML step for sklearn data splitting
@step
def sklearn_split_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],   
    Annotated[pd.DataFrame, "X_test"],   
    Annotated[pd.Series, "y_train"],   
    Annotated[pd.Series, "y_test"]
    ]:
    try:
        # Split the DataFrame into features (X) and target (y) using sklearn
        X = df.drop(['is_viral'], axis=1)
        y = df['is_viral']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        # Log and raise an error if an exception occurs during splitting
        logging.error("Error in splitting")
        raise e

# Define a ZenML step for drift report data splitting
@step
def drift_splitting(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "reference"],
    Annotated[pd.DataFrame, "current"]
]:
    try:
        # Split the DataFrame into reference and current sets for drift report
        reference, current = train_test_split(df, test_size=0.2, random_state=23)
        logging.info(reference.columns)
        return reference, current
    except Exception as e:
        # Log and raise an error if an exception occurs during splitting
        logging.error("Error in drift report splitting. {}", format(e))
        raise e
