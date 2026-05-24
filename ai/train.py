import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from dataset import UTKFaceDataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.losses import adversarial_loss, reconstruction_loss, classification_loss

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------
# Hyperparameters
# -----------------------
BATCH_SIZE = 32
LR = 0.0002
EPOCHS = 20


# -----------------------
# Paths
# -----------------------
CHECKPOINT_DIR = "../checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------
# Dataset
# -----------------------
dataset = UTKFaceDataset("data/UTKFace")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# -----------------------
# Models
# -----------------------
generator = Generator().to(device)
discriminator = Discriminator().to(device)


# -----------------------
# Optimizers
# -----------------------
g_optimizer = Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))

d_optimizer = Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))


# -----------------------
# Training Loop
# -----------------------
for epoch in range(EPOCHS):

    g_epoch_loss = 0
    d_epoch_loss = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Flip age class
        target_labels = 1 - labels

        # Real / fake targets
        real = torch.ones(batch_size, device=device)
        fake = torch.zeros(batch_size, device=device)

        # =====================
        # Train Generator
        # =====================
        g_optimizer.zero_grad()

        fake_images = generator(images, target_labels)

        validity, age_logits = discriminator(fake_images, target_labels)

        # Generator losses
        g_adv = adversarial_loss(validity, real)
        g_recon = reconstruction_loss(fake_images, images)
        g_cls = classification_loss(age_logits, target_labels)

        g_loss = g_adv + 5 * g_recon + 2 * g_cls

        g_loss.backward()
        g_optimizer.step()

        # =====================
        # Train Discriminator
        # =====================
        d_optimizer.zero_grad()

        real_validity, real_age_logits = discriminator(images, labels)

        fake_validity, fake_age_logits = discriminator(
            fake_images.detach(), target_labels
        )

        # Real/Fake losses
        d_real_loss = adversarial_loss(real_validity, real)
        d_fake_loss = adversarial_loss(fake_validity, fake)

        # Age classification losses
        d_real_cls = classification_loss(real_age_logits, labels)
        d_fake_cls = classification_loss(fake_age_logits, target_labels)

        d_loss = (d_real_loss + d_fake_loss + d_real_cls + d_fake_cls) / 4

        d_loss.backward()
        d_optimizer.step()

        # Track losses
        g_epoch_loss += g_loss.item()
        d_epoch_loss += d_loss.item()

        progress_bar.set_postfix(
            {"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"}
        )

    # Average epoch loss
    avg_g_loss = g_epoch_loss / len(loader)
    avg_d_loss = d_epoch_loss / len(loader)

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}] "
        f"| Avg G Loss: {avg_g_loss:.4f} "
        f"| Avg D Loss: {avg_d_loss:.4f}"
    )

    # =====================
    # Save checkpoints
    # =====================
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
