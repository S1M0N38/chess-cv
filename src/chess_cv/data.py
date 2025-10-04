"""Data loading utilities for chess piece images."""

# Class names in alphabetical order
CLASS_NAMES = [
    "black_bishop",
    "black_king",
    "black_knight",
    "black_pawn",
    "black_queen",
    "black_rook",
    "free",
    "white_bishop",
    "white_king",
    "white_knight",
    "white_pawn",
    "white_queen",
    "white_rook",
]


import glob
import os
from typing import Callable, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# Custom transform for Gaussian Noise
class AddGaussianNoise:
    """Adds Gaussian noise to a PIL image."""

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.std = std
        self.mean = mean

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to be augmented.

        Returns:
            PIL Image: Augmented image.
        """
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(self.mean, self.std * 255, np_img.shape)
        np_img = np_img + noise
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def get_image_files(data_dir: str) -> List[str]:
    """Get all image files from a directory."""
    return glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)


def get_label_from_path(image_path: str) -> str:
    """Get the label from the image path."""
    return image_path.split(os.sep)[-2]


def get_all_labels(image_files: List[str]) -> List[str]:
    """Get all labels from a list of image files."""
    return [get_label_from_path(image_file) for image_file in image_files]


def get_label_map(labels: List[str]) -> dict:
    """Get a map from labels to integers."""
    unique_labels = sorted(list(set(labels)))
    return {label: i for i, label in enumerate(unique_labels)}


class ChessPiecesDataset(Dataset):
    """A PyTorch Dataset for loading chess piece images."""

    def __init__(
        self,
        image_files: List[str],
        label_map: dict,
        transform: Optional[Callable] = None,
    ):
        self.image_files = image_files
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_files[idx]

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # On error, return the next item
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        # Normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        label_name = get_label_from_path(image_path)
        label = self.label_map[label_name]

        return img_array, label


def collate_fn(batch: list) -> tuple[mx.array, mx.array]:
    """
    Custom collate function to convert a batch of numpy arrays from the dataset
    into a single MLX array for images and an MLX array for labels.
    """
    images, labels = zip(*batch)
    images = np.stack(images)
    labels = np.array(labels)
    return mx.array(images), mx.array(labels)
