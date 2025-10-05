"""Data loading utilities for chess piece images."""

import glob
import os
from typing import Callable

import mlx.core as mx
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    "CLASS_NAMES",
    "AddGaussianNoise",
    "ChessPiecesDataset",
    "HuggingFaceChessPiecesDataset",
    "collate_fn",
    "get_all_labels",
    "get_image_files",
    "get_label_from_path",
    "get_label_map",
]

# Class names in alphabetical order (used across multiple modules)
CLASS_NAMES = [
    "bB",  # black bishop
    "bK",  # black king
    "bN",  # black knight
    "bP",  # black pawn
    "bQ",  # black queen
    "bR",  # black rook
    "wB",  # white bishop
    "wK",  # white king
    "wN",  # white knight
    "wP",  # white pawn
    "wQ",  # white queen
    "wR",  # white rook
    "xx",  # empty square
]


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


def get_image_files(data_dir: str) -> list[str]:
    """Get all image files from a directory."""
    return glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)


def get_label_from_path(image_path: str) -> str:
    """Get the label from the image path."""
    return image_path.split(os.sep)[-2]


def get_all_labels(image_files: list[str]) -> list[str]:
    """Get all labels from a list of image files."""
    return [get_label_from_path(image_file) for image_file in image_files]


def get_label_map(labels: list[str]) -> dict[str, int]:
    """Get a map from labels to integers."""
    unique_labels = sorted(list(set(labels)))
    return {label: i for i, label in enumerate(unique_labels)}


class ChessPiecesDataset(Dataset):
    """A PyTorch Dataset for loading chess piece images."""

    def __init__(
        self,
        image_files: list[str],
        label_map: dict[str, int],
        transform: Callable | None = None,
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


class HuggingFaceChessPiecesDataset(Dataset):
    """A PyTorch Dataset for loading chess piece images from HuggingFace datasets."""

    def __init__(
        self,
        dataset_id: str,
        label_map: dict[str, int],
        split: str = "train",
        transform: Callable | None = None,
    ):
        """
        Args:
            dataset_id: HuggingFace dataset ID (e.g., "S1M0N38/chess-cv-openboard")
            label_map: Dictionary mapping label names to integers
            split: Dataset split to use (default: "train")
            transform: Optional transform to apply to images
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            msg = "datasets library is required for HuggingFace dataset loading. Install it with: pip install datasets"
            raise ImportError(msg) from e

        self.dataset = load_dataset(dataset_id, split=split)
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple:
        item = self.dataset[idx]  # type: ignore[index]

        try:
            # HuggingFace datasets typically store images in an 'image' column
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")  # type: ignore[arg-type]
            else:
                img = img.convert("RGB")
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # On error, return the next item
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        # Normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Get label from the dataset
        # HuggingFace imagefolder datasets store labels in a 'label' column (as integers)
        # but we need to map them correctly
        label_idx = item["label"]

        return img_array, label_idx
