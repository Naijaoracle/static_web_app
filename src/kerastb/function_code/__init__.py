import logging
import azure.functions as func
from azure.storage.blob import BlobClient
from model_architecture import load_model_weights_from_blob
import os
import io
import requests

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    blob_function_url = "https://daviddasa.azurewebsites.net/api/httptrigger1"

    try:
        # Check if the request contains a file
        if 'imageFile' in req.files:
            image_file = req.files['imageFile'][0]

            # Upload the image to the blob function
            response = requests.post(blob_function_url, files={'image': (image_file.filename, io.BytesIO(image_file.read()))})

            if response.status_code == 200:
                return func.HttpResponse(response.text)
            else:
                return func.HttpResponse("Failed to process the image.", status_code=500)
        else:
            return func.HttpResponse("No image file found in the request.", status_code=400)

    except Exception as e:
        logging.error(f"Error during image processing: {str(e)}")
        return func.HttpResponse("An error occurred during image processing.", status_code=500)