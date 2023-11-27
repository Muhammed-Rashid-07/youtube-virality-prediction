import pandas as pd
from typing_extensions import Annotated

from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import json
import numpy as np
import logging
from zenml.services import BaseService


# @step
# def predictor(
#     service: MLFlowDeploymentService,
#     data: str
# ) -> np.ndarray:
#     """Run a inference request against a prediction service."""
#     service.start(timeout=10)  # should be a NOP if already started
#     data = json.loads(data)
#     data.pop("columns")
#     data.pop("index")
#     columns_for_df =['likes','dislikes','comment_count',
#                      'comments_disabled','ratings_disabled','video_error_or_removed',
#                      'time_since_publish','tag_count',
#                      'like_dislike_ratio','comment_view_ratio','title_words_count','description_words_count'
#                      ]
#     df = pd.DataFrame(data["data"], columns=columns_for_df)
#     json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
#     data = np.array(json_list)
#     prediction = service.predict(data)
#     logging.info(prediction)
#     return prediction

@step(enable_cache=True)
def predictor(
    service: BaseService,
    data: pd.DataFrame,
) -> np.ndarray:
    """Run a inference request against a prediction service."""
    service.start(timeout=300)
    logging.info(data.columns)
    print(
        f"Running predictions on data (single individual): {data.to_numpy()[0]}"
    )
    df = json.loads(json.dumps(list(data.T.to_dict().values())))
    json_data = np.array(df)
    prediction = service.predict(json_data)
    logging.info(prediction)
    print(
        f"Prediction (for single example slice) is: {bool(prediction.tolist()[0])}"
    )
    return prediction