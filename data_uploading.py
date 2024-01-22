from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

load_dotenv()

def uploadToBlobStorage(file_path, file_name):
    try:
        storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')
        storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
        connection_string = os.getenv('CONNECTION_STRING')
        container_name = os.getenv('CONTAINER_NAME')
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container = container_name, blob = file_name)

        with open(file_path_up,"rb") as data:
            blob_client.upload_blob(data)
        print("Upload " + file_name + " from local to container " + container_name)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

uploadToBlobStorage('file_path','file_name')


