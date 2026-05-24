import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, age_classes=2):
        super().__init__()

        self.label_emb = nn.Embedding(age_classes, 128)

        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2))

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(640, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dec4 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x, labels):
        batch_size = x.size(0)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        label_embedding = self.label_emb(labels)
        label_embedding = label_embedding.view(batch_size, 128, 1, 1)
        label_embedding = label_embedding.expand(-1, -1, 8, 8)

        bottleneck_input = torch.cat([e4, label_embedding], dim=1)
        b = self.bottleneck(bottleneck_input)

        d1 = self.dec1(b)

        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.dec2(d1)

        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.dec3(d2)

        d3 = torch.cat([d3, e1], dim=1)
        out = self.dec4(d3)

        return out
