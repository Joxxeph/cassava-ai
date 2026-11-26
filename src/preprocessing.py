import json

from PIL import Image, ImageOps

from torchvision import transforms
import json

from PIL import Image, ImageOps

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def load_image(filepath):

    with Image.open(filepath) as img:

        img = ImageOps.exif_transpose(img)

        img = img.convert("RGB")

    return img


def build_transforms(config_path):

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    image_size = tuple(config["image_size"])

    mean = config["normalization"]["mean"]
    std = config["normalization"]["std"]

    aug = config["augmentation"]

    interpolation = InterpolationMode(
        config["resize"]["interpolation"]
    )

    train_transform = transforms.Compose([

        transforms.Resize(
            image_size,
            interpolation=interpolation,
            antialias=config["resize"]["antialias"]
        ),

        transforms.RandomHorizontalFlip(
            p=aug["horizontal_flip"]
        ),

        transforms.RandomVerticalFlip(
            p=aug["vertical_flip"]
        ),

        transforms.RandomRotation(
            degrees=aug["rotation_degrees"],
            interpolation=interpolation,
            fill=tuple(aug["rotation_fill"])
        ),

        transforms.ColorJitter(
            brightness=aug["brightness"],
            contrast=aug["contrast"],
            saturation=aug["saturation"]
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        )

    ])

    eval_transform = transforms.Compose([

        transforms.Resize(
            image_size,
            interpolation=interpolation,
            antialias=config["resize"]["antialias"]
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        )

    ])

    return train_transform, eval_transform