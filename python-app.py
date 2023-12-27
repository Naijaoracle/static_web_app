import os
import azure.functions as func
from PIL import Image
import io
import torchvision.transforms as transforms
import torch
from src.py_ipynb.tuberculosis import SimpleCNN
from azure.storage.blob import BlobServiceClient
import apt-utils

def load_model_from_blob(connection_string, container_name, model_weights_blob_name):
    try:
        # Download the model weights from Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_weights_blob_name)
        model_weights_stream = blob_client.download_blob()
        model_weights_bytes = model_weights_stream.readall()

        # Load your model using the downloaded model weights
        model = SimpleCNN()  # Assuming your model class is defined in model_file.py
        model.load_state_dict(torch.load(io.BytesIO(model_weights_bytes)))
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

# Retrieve the Azure Blob Storage connection string from environment variables
connection_string = os.environ.get('AZURE_BLOB_CONNECTION_STRING')
container_name = "csb10032002a3ba9f46"
model_weights_blob_name = "model_weights.pth"

model = load_model_from_blob(connection_string, container_name, model_weights_blob_name)

# Define the transformation to be applied to the uploaded image
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        file = req.files['file']

        # Read the uploaded image file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Apply transformations to the image
        img = data_transform(img)
        img = img.unsqueeze(0)  # Add a batch dimension

        # Perform prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = "TB positive" if predicted.item() == 1 else "TB negative"

        return func.HttpResponse(f"Prediction: {prediction}", status_code=200)

    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
