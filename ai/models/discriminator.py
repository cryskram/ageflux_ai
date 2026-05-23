import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, age_classes=2):
        super().__init__()

        self.label_emb = nn.Embedding(age_classes, 128 * 128)

        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 8),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        batch_size = x.size(0)

        label_map = self.label_emb(labels)
        label_map = label_map.view(batch_size, 1, 128, 128)

        d_input = torch.cat([x, label_map], dim=1)

        out = self.model(d_input)

        return out.view(-1)
