import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PlantDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        self.stage_map = {
            "seed":0,
            "sprout":1,
            "seedling":2,
            "plant":3
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = row["image_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        day = row["day"]

        height = row["height_cm"]
        if pd.isna(height):
            height = -1

        if "day_01" in image_path or "day_02" in image_path:
            stage = self.stage_map["seed"]
        else:
            stage = -1


        return image, day, float(height), int(stage)