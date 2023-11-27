import pandas as pd
from typing_extensions import Annotated

from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import json
import numpy as np
import logging
from zenml.services import BaseService


@step(enable_cache=True)
def predictor(
    service: BaseService,
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Run an inference request against a prediction service.

    Args:
        service (BaseService): The prediction service to use.
        data (pd.DataFrame): The input data for inference.

    Returns:
        np.ndarray: Array containing the predictions.
    """
    # Start the prediction service
    service.start(timeout=300)

    # Log the columns of the input data
    logging.info(data.columns)

    # Display the input data for the first individual
    print(f"Running predictions on data (single individual): {data.to_numpy()[0]}")

    # Convert the input data to JSON format
    df = json.loads(json.dumps(list(data.T.to_dict().values())))
    json_data = np.array(df)

    # Make predictions using the prediction service
    prediction = service.predict(json_data)

    # Log the predictions
    logging.info(prediction)

    # Display the prediction for the first example slice
    print(f"Prediction (for single example slice) is: {bool(prediction.tolist()[0])}")

    # Return the predictions
    return prediction
