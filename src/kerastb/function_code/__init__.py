import logging
import azure.functions as func
from azure.storage.blob import BlobClient
from model_architecture import load_model_weights_from_blob
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    connection_string = os.environ["AzureWebJobsStorage"]  # Retrieve connection string from application settings

    blob_url = "https://csb10032002a3ba9f46.blob.core.windows.net/staticwebapp/kerastb01_model_weights.h5"
    # You can alternatively construct blob_url dynamically using container_name and blob_name

    # Call the function to load model weights
    loaded_model = load_model_weights_from_blob(blob_url, connection_string)

    if loaded_model is not None:
        return func.HttpResponse(f"Model loaded successfully from Blob Storage: {blob_url}")
    else:
        return func.HttpResponse("Failed to load model from Blob Storage. Check logs for details.", status_code=500)
