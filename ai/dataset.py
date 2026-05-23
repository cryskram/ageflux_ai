import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        self.image_files = []

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        for file in os.listdir(root_dir):
            if not file.endswith(".jpg"):
                continue

            try:
                age = int(file.split("_")[0])

                if age <= 25:
                    label = 0

                elif age >= 50:
                    label = 1

                else:
                    continue

                self.image_files.append((file, label))

            except:
                continue

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename, label = self.image_files[idx]

        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
