from dataset import PlantDataset

dataset = PlantDataset("data/labels/dataset.csv")

print("Dataset size:", len(dataset))

img, day, height, stage = dataset[0]

print("Image shape:", img.shape)
print("Day:", day)
print("Height:", height)
print("Stage:", stage)