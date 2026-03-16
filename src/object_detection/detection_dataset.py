import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class SeedDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = []

        for root, _, files in os.walk(images_dir):
            for file_name in files:
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(full_path, images_dir)
                    self.image_files.append(rel_path)

        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_rel_path = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_rel_path)

        label_rel_path = os.path.splitext(image_rel_path)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_rel_path)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_center, y_center, box_w, box_h = map(float, parts)

                    x_center *= width
                    y_center *= height
                    box_w *= width
                    box_h *= height

                    xmin = x_center - box_w / 2
                    ymin = y_center - box_h / 2
                    xmax = x_center + box_w / 2
                    ymax = y_center + box_h / 2

                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(width, xmax)
                    ymax = min(height, ymax)

                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(int(class_id) + 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        image = F.to_tensor(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))