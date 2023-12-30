import os
from PIL import Image
import io
import torchvision.transforms as transforms
import torch
from src.py_ipynb.tuberculosis import SimpleCNN
from azure.storage.blob import BlobServiceClient

def load_model_from_blob(connection_string, container_name, model_weights_blob_name):
    try:
        # To download the model weights from Azure
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_weights_blob_name)
        model_weights_stream = blob_client.download_blob()
        model_weights_bytes = model_weights_stream.readall()

        # To load the model using the model weights
        model = SimpleCNN() 
        model.load_state_dict(torch.load(io.BytesIO(model_weights_bytes)))
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Sorry. Failed to load model: {str(e)}")

# Retrieve the Azure Blob Storage connection string from environment variables
def load_model():
    connection_string = os.environ.get('AZURE_BLOB_CONNECTION_STRING')
    container_name = "csb10032002a3ba9f46"
    model_weights_blob_name = "model_weights.pth"

    return load_model_from_blob(connection_string, container_name, model_weights_blob_name)

# Defining the transformation to be applied to the uploaded image
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])