from azure.storage.blob import BlobServiceClient
from config.constant import storage_account_key, storage_account_name, connection_string, container_name, file_path_up, file_name


def uploadToBlobStorage():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container = container_name, blob = file_name)

        with open(file_path_up,"rb") as data:
            blob_client.upload_blob(data)
        print("Upload " + file_name + " from local to container " + container_name)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

uploadToBlobStorage()


