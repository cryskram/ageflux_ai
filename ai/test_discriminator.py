import torch
from models.discriminator import Discriminator

model = Discriminator()

x = torch.randn(4, 3, 128, 128)
labels = torch.tensor([0, 1, 0, 1])

out = model(x, labels)

print("Output shape:", out.shape)
print("Scores:", out)
