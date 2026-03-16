import os
import torch
import cv2
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import glob

MODEL_PATH = "models/seed_detector_fasterrcnn.pth"
IMAGE_PATH = glob.glob("data/detection/images/val/**/*.jpg", recursive=True)[0]
OUTPUT_PATH = "output/detection_images/prediction.jpg"

NUM_CLASSES = 3  # background + seed + germination
THRESHOLD = 0.3

CLASS_NAMES = {
    1: "seed",
    2: "germination"
}


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    os.makedirs("output/detection_images", exist_ok=True)

    device = torch.device("cpu")
    print("Using device:", device)

    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    image_cv = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    drawn = 0

    for box, label, score in zip(boxes, labels, scores):
        if score < THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES.get(int(label), f"class_{label}")
        text = f"{class_name}: {score:.2f}"

        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_cv,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        drawn += 1

    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, image_cv)

    print("Detections drawn:", drawn)
    print("Saved output to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()