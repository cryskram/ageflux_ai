import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, age_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.LeakyReLU(0.01),
        )

        self.adv_head = nn.Conv2d(1024, 1, 3, 1, 1)

        self.cls_head = nn.Conv2d(1024, age_classes, 4)

    def forward(self, x):
        features = self.features(x)

        validity = self.adv_head(features)
        validity = validity.view(x.size(0), -1)

        age_logits = self.cls_head(features)
        age_logits = age_logits.view(x.size(0), -1)

        return validity, age_logits
