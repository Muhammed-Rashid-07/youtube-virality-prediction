import logging

import pandas as pd
from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data   


def load_test_data_model():
    """
    Load test data from Azure blob storage.
    """
    try:
        raw_data = pd.read_csv('./data/processed_data.csv')
        df = raw_data.sample(n=1000,random_state=42)
        data = df.drop(['is_viral'],axis=1)
        result = data.to_json(orient="split")
        logging.info("datatype: %s",type(result))
        return result
    except Exception as e:
        logging.error("Error in loading test data",e)
        raise e