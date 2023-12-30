import logging
import os
import numpy as np
from PIL import Image
import azure.functions as func

def load_model_weights_from_blob(container_name, blob_name):
    connection_string = os.environ["AzureWebJobsStorage"]
    # Add your code to load model weights from Azure Blob Storage

def load_and_preprocess_image(uploaded_image, target_size=(224, 224)):
    try:
        # Open the image using PIL
        image = Image.open(uploaded_image)

        # Resize the image to the target size while preserving aspect ratio
        image = image.resize(target_size, Image.ANTIALIAS)

        # Convert the image to a NumPy array
        image_array = np.asarray(image)

        # Normalize the pixel values to the range [0, 1]
        image_array = image_array / 255.0

        # Expand the dimensions to match expected model input shape
        # (Assuming a model expecting a single-channel image)
        preprocessed_image = np.expand_dims(image_array, axis=0)

        return preprocessed_image

    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        return None

def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")

    try:
        # Preprocess the uploaded image
        preprocessed_image = load_and_preprocess_image(myblob)

        if preprocessed_image is not None:
            # Load model weights from Azure Blob Storage
            model = load_model_weights_from_blob(
                container_name="csb10032002a3ba9f46",
                blob_name="kerastb01_model_weights.h5"
            )

            # Perform inference
            if model is not None:
                prediction = model.predict(preprocessed_image)
                prediction_class = 'TB' if prediction > 0.5 else 'No TB'
                return func.HttpResponse(f"Prediction: {prediction_class}")
            else:
                return func.HttpResponse(
                    "Failed to load model from Blob Storage.",
                    status_code=500
                )
        else:
            return func.HttpResponse(
                "An error occurred during image preprocessing.",
                status_code=500
            )

    except Exception as e:
        logging.error(f"Error during image classification: {e}")
        return func.HttpResponse(
            "An error occurred during image processing.",
            status_code=500
        )
