import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detection_dataset import SeedDetectionDataset, collate_fn


TRAIN_IMAGES = "data/detection/images/train"
TRAIN_LABELS = "data/detection/labels/train"
VAL_IMAGES = "data/detection/images/val"
VAL_LABELS = "data/detection/labels/val"

MODEL_SAVE_PATH = "models/seed_detector_fasterrcnn.pth"

NUM_CLASSES = 3
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    valid_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses).item():
            print(f"Skipping batch {batch_idx + 1} due to invalid loss: {loss_dict}")
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += losses.item()
        valid_batches += 1

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Batch {batch_idx + 1}/{len(dataloader)} | "
            f"Loss: {losses.item():.4f}"
        )

    if valid_batches == 0:
        print(f"Epoch {epoch + 1} Training Loss: no valid batches")
        return

    avg_loss = running_loss / valid_batches
    print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")


def validate_one_epoch(model, dataloader, device, epoch):
    model.train()  # detection losses are computed in train mode
    running_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses).item():
                print(f"Skipping val batch {batch_idx + 1} due to invalid loss")
                continue

            running_loss += losses.item()
            valid_batches += 1

    if valid_batches == 0:
        print(f"Epoch {epoch + 1} Validation Loss: no valid batches")
        return

    avg_loss = running_loss / valid_batches
    print(f"Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}")


def main():
    os.makedirs("models", exist_ok=True)

    # Force CPU to avoid Apple MPS Faster R-CNN crashes
    device = torch.device("cpu")
    print("Using device:", device)

    train_dataset = SeedDetectionDataset(TRAIN_IMAGES, TRAIN_LABELS)
    val_dataset = SeedDetectionDataset(VAL_IMAGES, VAL_LABELS)

    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    if len(train_dataset) == 0:
        print("Training dataset is empty.")
        return

    if len(val_dataset) == 0:
        print("Validation dataset is empty.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = get_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        train_one_epoch(model, train_loader, optimizer, device, epoch)
        validate_one_epoch(model, val_loader, device, epoch)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("\nModel saved to:", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()