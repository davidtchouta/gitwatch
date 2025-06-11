import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Définir l'appareil (device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Supposant une classification binaire
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict(model, image_tensor, class_names):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()]
