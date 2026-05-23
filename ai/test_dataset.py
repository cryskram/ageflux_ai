from torch.utils.data import DataLoader
from dataset import UTKFaceDataset

dataset = UTKFaceDataset("data/UTKFace")

print("Dataset size:", len(dataset))

loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, labels = next(iter(loader))

print("Image batch shape:", images.shape)
print("Labels:", labels)
