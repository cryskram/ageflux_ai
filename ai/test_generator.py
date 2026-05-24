import torch
from models.generator import Generator

model = Generator()

x = torch.randn(4, 3, 128, 128)
labels = torch.tensor([0, 1, 3, 4])
out = model(x, labels)

print("Output shape:", out.shape)
