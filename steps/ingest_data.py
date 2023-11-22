import os
import pandas as pd
import logging
from zenml import step

from azure.storage.blob import BlobSasPermissions, generate_blob_sas, BlobServiceClient
from datetime import datetime, timedelta
from dotenv import load_dotenv


load_dotenv()
class IngestData:
    def __init__(self):
        """
        Args:
            data_path (str): path of the datafile
        """
        
        storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')
        storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
        connection_string = os.getenv('CONNECTION_STRING')
        container_name = os.getenv('CONTAINER_NAME')
    
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
    
        #get a list of all blob files in the container
        blob_list = []

        for blob in container_client.list_blobs():
            blob_list.append(blob.name)

        for blob in blob_list:
            sas = generate_blob_sas(
                account_name=storage_account_name,
                container_name= container_name,
                blob_name=blob,
                account_key=storage_account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=1)
                )
        sas_url = 'https://'+storage_account_name+'.blob.core.windows.net/'+container_name+'/'+blob+'?'+sas
        self.path = sas_url
        
    
    def get_data(self):
        df = pd.read_csv(self.path)
        logging.info("Reading csv file successfully completed.")
        return df
    

@step(enable_cache = True)
def ingest_df() -> pd.DataFrame:
    """
    
    ZenML step for ingesting data from a CSV file.
    
    Args:
        data_path (str): Path of the datafile to be ingested.
    
    Returns:
        df (pd.DataFrame): DataFrame containing the ingested data.
        
    """
    try:
        #Creating an instance of IngestData class and ingest the data
        ingest_data = IngestData()
        df = ingest_data.get_data()
        logging.info("Ingesting data completed")
        return df
    except Exception as e:
        #Log an error message if data ingestion fails and raise the exception
        logging.error("Error while ingesting data")
        raise e
    
    
