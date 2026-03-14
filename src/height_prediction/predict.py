import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -------- SETTINGS --------
STAGE_MODEL_PATH = "models/stage_model.pth"
HEIGHT_MODEL_PATH = "models/height_model.pth"

IMAGE_SIZE = 224
STAGE_CLASSES = ["seed", "germinating", "sprout", "young_plant"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -------- MODEL DEFINITIONS --------
class StageCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # features.0
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # features.3
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # features.6
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),   # classifier.1
            nn.ReLU(),
            nn.Linear(128, num_classes)     # classifier.3
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class HeightCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# -------- LOAD MODELS --------
def load_stage_model():
    model = StageCNN(num_classes=len(STAGE_CLASSES))
    model.load_state_dict(torch.load(STAGE_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_height_model():
    model = HeightCNN()
    model.load_state_dict(torch.load(HEIGHT_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# -------- PREDICT FUNCTIONS --------
def predict_stage(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_idx].item()

    return STAGE_CLASSES[predicted_idx], confidence


def predict_height(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        height = output.item()

    return height

# -------- MAIN --------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(STAGE_MODEL_PATH):
        print(f"Stage model not found: {STAGE_MODEL_PATH}")
        sys.exit(1)

    stage_model = load_stage_model()
    stage, confidence = predict_stage(stage_model, image_path)

    print(f"Image: {image_path}")
    print(f"Predicted stage: {stage}")
    print(f"Confidence: {confidence:.4f}")

    if os.path.exists(HEIGHT_MODEL_PATH):
        try:
            height_model = load_height_model()
            height = predict_height(height_model, image_path)
            print(f"Predicted height: {height:.2f} cm")
        except Exception as e:
            print("Could not load height model.")
            print(f"Reason: {e}")
    else:
        print("Height model not found, skipping height prediction.")


if __name__ == "__main__":
    main()