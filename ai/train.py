import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from dataset import UTKFaceDataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.losses import adversarial_loss, classification_loss, reconstruction_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


BATCH_SIZE = 32
LR = 0.0001
EPOCHS = 20


DATASET_DIR = "data/UTKFace"
CHECKPOINT_DIR = "../checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


dataset = UTKFaceDataset(DATASET_DIR)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))

d_optimizer = Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))


for epoch in range(EPOCHS):

    g_epoch_loss = 0
    d_epoch_loss = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        target_labels = torch.randint(0, 5, (batch_size,), device=device)

        real = torch.ones(batch_size, 16, device=device)
        fake = torch.zeros(batch_size, 16, device=device)

        g_optimizer.zero_grad()

        fake_images = generator(images, target_labels)

        validity, fake_cls_logits = discriminator(fake_images)

        reconstructed = generator(fake_images, labels)

        g_adv = adversarial_loss(validity, real)
        g_cls = classification_loss(fake_cls_logits, target_labels)
        g_rec = reconstruction_loss(reconstructed, images)

        g_loss = g_adv + g_cls + 10 * g_rec

        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()

        real_validity, real_cls_logits = discriminator(images)
        fake_validity, _ = discriminator(fake_images.detach())

        d_real_loss = adversarial_loss(real_validity, real)
        d_fake_loss = adversarial_loss(fake_validity, fake)

        d_cls_loss = classification_loss(real_cls_logits, labels)

        d_loss = d_real_loss + d_fake_loss + d_cls_loss

        d_loss.backward()
        d_optimizer.step()

        g_epoch_loss += g_loss.item()
        d_epoch_loss += d_loss.item()

        progress_bar.set_postfix(
            {"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"}
        )

    avg_g_loss = g_epoch_loss / len(loader)
    avg_d_loss = d_epoch_loss / len(loader)

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}] "
        f"| Avg G Loss: {avg_g_loss:.4f} "
        f"| Avg D Loss: {avg_d_loss:.4f}"
    )

    torch.save(
        generator.state_dict(),
        os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch+1}.pth"),
    )

    torch.save(
        discriminator.state_dict(),
        os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch+1}.pth"),
    )

    print(f"Checkpoint saved for epoch {epoch+1}\n")


print("Training complete!")
