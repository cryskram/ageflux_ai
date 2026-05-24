import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from models.generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

to_pil = transforms.ToPILImage()


def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5
    return tensor.clamp(0, 1)


def generate_face(image_path, checkpoint_path, target_label, output_path):
    model = Generator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    label = torch.tensor([target_label], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor, label)

    output = output.squeeze(0).cpu()
    output = denormalize(output)

    output_image = to_pil(output)

    output_image.save(output_path)

    print(f"Saved generated image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image", type=str, required=True, help="Path to input face image"
    )

    parser.add_argument(
        "--target",
        type=str,
        choices=["young", "old"],
        required=True,
        help="Target age class",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/generator_epoch_5.pth",
        help="Path to trained generator checkpoint",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../samples/output.jpg",
        help="Path to save generated image",
    )

    args = parser.parse_args()

    os.makedirs("../samples", exist_ok=True)

    target_label = 0 if args.target == "young" else 1

    generate_face(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        target_label=target_label,
        output_path=args.output,
    )
