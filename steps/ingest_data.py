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
        
        storage_account_key = "Tbj9y/FEdT6Vel4FMMTj5cunzyJt4iEZpEMiw6Xb0uztrVuljofVIcwCQYSsRXECl9JcANDIkwnT+AStIrqu/A=="
        storage_account_name = "mlprojectdemo7552488106"
        connection_string = "DefaultEndpointsProtocol=https;AccountName=mlprojectdemo7552488106;AccountKey=Tbj9y/FEdT6Vel4FMMTj5cunzyJt4iEZpEMiw6Xb0uztrVuljofVIcwCQYSsRXECl9JcANDIkwnT+AStIrqu/A==;EndpointSuffix=core.windows.net"
        container_name = "youtube-data"
        
    
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
        
    
    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        logging.info("Reading csv file successfully completed.")
        return df
    

@step(enable_cache = True)
def ingest_df(sample:bool=False) -> pd.DataFrame:
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
        if sample:
            return df.sample(n=1000, random_state=1)
        else:
            return df
    except Exception as e:
        #Log an error message if data ingestion fails and raise the exception
        logging.error("Error while ingesting data")
        raise e
    
    
