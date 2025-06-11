from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
from utils import load_model, predict

app = Flask(__name__)

# Load the model
model = load_model("model.pth")
class_names = ['fake', 'real']  # Ajustez si nécessaire

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')

    # Appliquer les transformations
    image_tensor = transform(image).unsqueeze(0)

    # Prédire
    label = predict(model, image_tensor, class_names)
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
