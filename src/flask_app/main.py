from flask import Flask, request
from model_file import load_model, data_transform
import torch

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file found", 400

        file = request.files['file']

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = data_transform(img)
        img = img.unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = "TB positive" if predicted.item() == 1 else "TB negative"

        return prediction, 200
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)