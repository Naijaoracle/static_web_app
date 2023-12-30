import azure.functions as func
from PIL import Image
import io
import torchvision.transforms as transforms
import torch
from model_file import SimpleCNN

# Loading trained model
model = SimpleCNN()
# ... [Loading model weights and transformations]

# Define data transformation function
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        file = req.files['file']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = data_transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = "TB positive" if predicted.item() == 1 else "TB negative"

        return func.HttpResponse(f"Prediction: {prediction}", status_code=200)
    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)