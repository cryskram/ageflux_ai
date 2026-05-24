import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, age_classes=2):
        super().__init__()

        self.label_emb = nn.Embedding(age_classes, 128 * 128)

        self.features = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),  # 128 -> 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        # Real/Fake head
        self.adv_head = nn.Sequential(nn.Conv2d(512, 1, 8), nn.Sigmoid())

        # Age classifier head
        self.cls_head = nn.Sequential(nn.Conv2d(512, age_classes, 8))

    def forward(self, x, labels):
        batch_size = x.size(0)

        # Label map
        label_map = self.label_emb(labels)
        label_map = label_map.view(batch_size, 1, 128, 128)

        d_input = torch.cat([x, label_map], dim=1)

        features = self.features(d_input)

        validity = self.adv_head(features).view(-1)

        age_logits = self.cls_head(features)
        age_logits = age_logits.view(batch_size, -1)

        return validity, age_logits
