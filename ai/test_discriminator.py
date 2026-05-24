import torch
from models.discriminator import Discriminator

model = Discriminator()

x = torch.randn(4, 3, 128, 128)

validity, age_logits = model(x)

print("Validity shape:", validity.shape)
print("Age logits shape:", age_logits.shape)
