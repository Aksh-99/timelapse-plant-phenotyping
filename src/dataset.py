import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PlantDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        """
        csv_file: path to dataset csv
        transform: torchvision transforms
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        day = row["day"]

        height = row["height_cm"]
        stage = row["stage"]

        # convert to tensor (optional if empty)
        height = torch.tensor(height) if not pd.isna(height) else torch.tensor(-1)
        stage = torch.tensor(stage) if not pd.isna(stage) else torch.tensor(-1)

        return image, day, height, stage