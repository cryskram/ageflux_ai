import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, age_classes=2):
        super().__init__()

        self.label_emb = nn.Embedding(age_classes, 128)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
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
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(640, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x, labels):
        batch_size = x.size(0)

        features = self.encoder(x)

        label_embedding = self.label_emb(labels)
        label_embedding = label_embedding.view(batch_size, 128, 1, 1)
        label_embedding = label_embedding.expand(-1, -1, 8, 8)

        combined = torch.cat([features, label_embedding], dim=1)

        combined = self.bottleneck(combined)
        out = self.decoder(combined)

        return out
