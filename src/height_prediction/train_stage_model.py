import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PlantDataset

# Simple CNN for 4-class classification
class StageCNN(nn.Module):
    def __init__(self):
        super(StageCNN, self).__init__()

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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 4)   # 4 classes: seed, sprout, seedling, plant
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    dataset = PlantDataset("data/labels/dataset.csv")
    
    # keep only rows that actually have a valid stage label
    valid_samples = []
    for i in range(len(dataset)):
        image, day, height, stage = dataset[i]
        if stage != -1:
            valid_samples.append((image, stage))

    if len(valid_samples) == 0:
        print("No labeled stage data found in dataset.csv")
        return

    class StageOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    stage_dataset = StageOnlyDataset(valid_samples)
    dataloader = DataLoader(stage_dataset, batch_size=4, shuffle=True)

    model = StageCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), "models/stage_model.pth")
    print("Stage model saved to models/stage_model.pth")


if __name__ == "__main__":
    main()