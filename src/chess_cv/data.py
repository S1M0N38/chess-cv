"""Data loading utilities for chess piece images."""

import glob
import os
import random
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

__all__ = [
    "CLASS_NAMES",
    "RandomArrowOverlay",
    "RandomHighlightOverlay",
    "RandomMouseOverlay",
    "RandomMoveOverlay",
    "ChessPiecesDataset",
    "HuggingFaceChessPiecesDataset",
    "ConcatenatedHuggingFaceDataset",
    "collate_fn",
    "get_all_labels",
    "get_image_files",
    "get_label_from_path",
    "get_label_map",
    "get_label_map_from_class_names",
]

# Class names for pieces model (alphabetically ordered)
# Note: This is specific to the 'pieces' model. Other models may have different class names.
# Use get_model_config(model_id)["class_names"] from constants.py for model-specific classes.
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


class RandomArrowOverlay:
    """Randomly overlays arrow images on chess piece images."""

    def __init__(self, arrow_dir: Path | str, probability: float = 0.3):
        """
        Args:
            arrow_dir: Directory containing arrow component subdirectories
                      (head-*, tail-*, middle-*, corner-*)
            probability: Probability of applying arrow overlay (0.0 to 1.0)
        """
        self.probability = probability
        self.arrow_images: list[Image.Image] = []

        # Load all arrow images from all subdirectories
        arrow_dir = Path(arrow_dir)
        arrow_paths = glob.glob(str(arrow_dir / "*" / "*.png"))

        if not arrow_paths:
            raise FileNotFoundError(
                f"No arrow images found in {arrow_dir} subdirectories"
            )

        # Load and cache all arrow images
        for arrow_path in arrow_paths:
            try:
                arrow_img = Image.open(arrow_path).convert("RGBA")
                self.arrow_images.append(arrow_img)
            except Exception as e:
                print(f"Warning: Could not load arrow image {arrow_path}: {e}")

        if not self.arrow_images:
            raise ValueError("No valid arrow images could be loaded")

        print(f"Loaded {len(self.arrow_images)} arrow images for augmentation")

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to be augmented.

        Returns:
            PIL Image: Augmented image (with or without arrow overlay).
        """
        # Randomly decide whether to apply arrow overlay
        if random.random() > self.probability:
            return img

        # Convert to RGBA to support alpha compositing
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Randomly select an arrow (no rotation needed, all rotations pre-generated)
        arrow = random.choice(self.arrow_images).copy()

        # Composite arrow onto the image using alpha composition
        img = Image.alpha_composite(img, arrow)

        # Convert back to RGB
        return img.convert("RGB")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_arrows={len(self.arrow_images)}, "
            f"probability={self.probability})"
        )


class RandomHighlightOverlay:
    """Randomly overlays highlight images on chess piece images."""

    def __init__(self, highlight_dir: Path | str, probability: float = 0.3):
        """
        Args:
            highlight_dir: Directory containing highlight subdirectories (circle, sqaure)
            probability: Probability of applying highlight overlay (0.0 to 1.0)
        """
        self.probability = probability
        self.highlight_images: list[Image.Image] = []

        # Load all highlight images from circle subdirectory only
        highlight_dir = Path(highlight_dir)
        highlight_paths = []
        # NOTE: we are only using circle highlights. Square highlights (sqaure/)
        # are just different background colors for the square, which we already
        # handle with different board backgrounds
        for subdir in ["circle"]:
            subdir_path = highlight_dir / subdir
            if subdir_path.exists():
                highlight_paths.extend(glob.glob(str(subdir_path / "*.png")))

        if not highlight_paths:
            raise FileNotFoundError(
                f"No highlight images found in {highlight_dir}/circle"
            )

        # Load and cache all highlight images
        for highlight_path in highlight_paths:
            try:
                highlight_img = Image.open(highlight_path).convert("RGBA")
                self.highlight_images.append(highlight_img)
            except Exception as e:
                print(f"Warning: Could not load highlight image {highlight_path}: {e}")

        if not self.highlight_images:
            raise ValueError("No valid highlight images could be loaded")

        print(f"Loaded {len(self.highlight_images)} highlight images for augmentation")

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to be augmented.

        Returns:
            PIL Image: Augmented image (with or without highlight overlay).
        """
        # Randomly decide whether to apply highlight overlay
        if random.random() > self.probability:
            return img

        # Convert to RGBA to support alpha compositing
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Randomly select a highlight (no rotation needed, all rotations pre-generated)
        highlight = random.choice(self.highlight_images).copy()

        # Composite highlight onto the image using alpha composition
        img = Image.alpha_composite(img, highlight)

        # Convert back to RGB
        return img.convert("RGB")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_highlights={len(self.highlight_images)}, "
            f"probability={self.probability})"
        )


class RandomMoveOverlay:
    """Randomly overlays move indicator images on chess piece images."""

    def __init__(self, move_dir: Path | str, probability: float = 0.3):
        """
        Args:
            move_dir: Directory containing move indicator subdirectories (dot, ring)
            probability: Probability of applying move overlay (0.0 to 1.0)
        """
        self.probability = probability
        self.move_images: list[Image.Image] = []

        # Load all move images from all subdirectories
        move_dir = Path(move_dir)
        move_paths = glob.glob(str(move_dir / "*" / "*.png"))

        if not move_paths:
            raise FileNotFoundError(
                f"No move images found in {move_dir} subdirectories"
            )

        # Load and cache all move images
        for move_path in move_paths:
            try:
                move_img = Image.open(move_path).convert("RGBA")
                self.move_images.append(move_img)
            except Exception as e:
                print(f"Warning: Could not load move image {move_path}: {e}")

        if not self.move_images:
            raise ValueError("No valid move images could be loaded")

        print(f"Loaded {len(self.move_images)} move images for augmentation")

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to be augmented.

        Returns:
            PIL Image: Augmented image (with or without move overlay).
        """
        # Randomly decide whether to apply move overlay
        if random.random() > self.probability:
            return img

        # Convert to RGBA to support alpha compositing
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Randomly select a move indicator (no rotation needed, all rotations pre-generated)
        move = random.choice(self.move_images).copy()

        # Composite move onto the image using alpha composition
        img = Image.alpha_composite(img, move)

        # Convert back to RGB
        return img.convert("RGB")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_moves={len(self.move_images)}, "
            f"probability={self.probability})"
        )


class RandomMouseOverlay:
    """Randomly overlays mouse cursor images on chess piece images with geometric transformations."""

    def __init__(
        self,
        mouse_dir: Path | str,
        probability: float = 0.3,
        aug_config: dict | None = None,
    ):
        """
        Args:
            mouse_dir: Directory containing mouse cursor PNG images
            probability: Probability of applying mouse overlay (0.0 to 1.0)
            aug_config: Augmentation configuration for mouse transformations
        """
        self.probability = probability
        self.mouse_images: list[Image.Image] = []

        # Load all mouse images from the directory
        mouse_dir = Path(mouse_dir)
        mouse_paths = glob.glob(str(mouse_dir / "*.png"))

        if not mouse_paths:
            raise FileNotFoundError(f"No mouse images found in {mouse_dir}")

        # Load and cache all mouse images
        for mouse_path in mouse_paths:
            try:
                mouse_img = Image.open(mouse_path).convert("RGBA")
                self.mouse_images.append(mouse_img)
            except Exception as e:
                print(f"Warning: Could not load mouse image {mouse_path}: {e}")

        if not self.mouse_images:
            raise ValueError("No valid mouse images could be loaded")

        print(f"Loaded {len(self.mouse_images)} mouse images for augmentation")

        # Set up mouse transformation pipeline
        if aug_config is None:
            # Default configuration for mouse transformations
            aug_config = {
                "mouse_padding": 16,
                "mouse_rotation_degrees": 5,
                "mouse_center_crop_size": 40,
                "mouse_final_size": 32,
                "mouse_scale_range": (0.10, 0.30),
                "mouse_ratio_range": (0.8, 1.2),
            }

        # Create mouse transformation pipeline
        self.mouse_transform = v2.Compose(
            [
                # Step 1: Pad to create rotation space (32x32 → 64x64)
                v2.Pad(
                    padding=aug_config["mouse_padding"], padding_mode="constant", fill=0
                ),
                # Step 2: Random rotation with small degrees
                v2.RandomRotation(degrees=aug_config["mouse_rotation_degrees"], fill=0),
                # Step 3: Remove black corners from rotation
                v2.CenterCrop(size=aug_config["mouse_center_crop_size"]),
                # Step 4: Random crop + scale variation + resize back to 32×32
                # This makes cursor smaller and positions it randomly
                v2.RandomResizedCrop(
                    size=aug_config["mouse_final_size"],
                    scale=aug_config["mouse_scale_range"],
                    ratio=aug_config["mouse_ratio_range"],
                    antialias=True,
                ),
            ]
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to be augmented.

        Returns:
            PIL Image: Augmented image (with or without mouse overlay).
        """
        # Randomly decide whether to apply mouse overlay
        if random.random() > self.probability:
            return img

        # Convert to RGBA to support alpha compositing
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Randomly select a mouse cursor
        mouse = random.choice(self.mouse_images).copy()

        # Apply geometric transformations to the mouse cursor
        # This makes the cursor smaller and positions it randomly
        mouse = self.mouse_transform(mouse)

        # Composite transformed mouse onto the image using alpha composition
        img = Image.alpha_composite(img, mouse)

        # Convert back to RGB
        return img.convert("RGB")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_mice={len(self.mouse_images)}, "
            f"probability={self.probability})"
        )


def get_image_files(data_dir: str) -> list[str]:
    """Get all image files from a directory."""
    return glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)


def get_label_from_path(image_path: str) -> str:
    """Get the label from the image path.

    Images are stored in a standard ImageFolder format:
    e.g., data/splits/pieces/train/bB/image.png

    This extracts the parent directory name, which is the class label.
    """
    return Path(image_path).parent.name


def get_all_labels(image_files: list[str]) -> list[str]:
    """Get all labels from a list of image files."""
    return [get_label_from_path(image_file) for image_file in image_files]


def get_label_map(labels: list[str]) -> dict[str, int]:
    """Get a map from labels to integers.

    Args:
        labels: List of label names

    Returns:
        Dictionary mapping label names to integer indices (alphabetically sorted)
    """
    unique_labels = sorted(list(set(labels)))
    return {label: i for i, label in enumerate(unique_labels)}


def get_label_map_from_class_names(class_names: list[str]) -> dict[str, int]:
    """Create a label map directly from class names.

    This is the preferred method as it ensures consistency across all splits
    and doesn't depend on scanning actual data directories.

    Args:
        class_names: List of class names from model configuration

    Returns:
        Dictionary mapping class names to integer indices
    """
    return {label: i for i, label in enumerate(class_names)}


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


class ConcatenatedHuggingFaceDataset(Dataset):
    """A PyTorch Dataset for loading chess piece images from multiple HuggingFace dataset splits."""

    def __init__(
        self,
        dataset_id: str,
        label_map: dict[str, int],
        splits: list[str] | None = None,
        transform: Callable | None = None,
    ):
        """
        Args:
            dataset_id: HuggingFace dataset ID (e.g., "S1M0N38/chess-cv-chessvision")
            label_map: Dictionary mapping label names to integers
            splits: List of dataset splits to concatenate (default: None, will use all available splits)
            transform: Optional transform to apply to images
        """
        try:
            from datasets import DatasetDict, concatenate_datasets, load_dataset
        except ImportError as e:
            msg = "datasets library is required for HuggingFace dataset loading. Install it with: pip install datasets"
            raise ImportError(msg) from e

        # Load all splits or specified splits
        if splits is None:
            # Load the full dataset to get all available splits
            full_dataset: DatasetDict = load_dataset(dataset_id)  # type: ignore[assignment]
            splits = [str(key) for key in full_dataset.keys()]

        print(f"Loading and concatenating splits: {splits}")
        datasets_to_concat = []
        for split in splits:
            ds = load_dataset(dataset_id, split=split)
            datasets_to_concat.append(ds)

        # Concatenate all datasets
        self.dataset = concatenate_datasets(datasets_to_concat)
        self.label_map = label_map
        self.transform = transform
        print(f"Total samples after concatenation: {len(self.dataset)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        item = self.dataset[idx]

        try:
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")  # type: ignore[arg-type]
            else:
                img = img.convert("RGB")
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        img_array = np.array(img, dtype=np.float32) / 255.0

        # Get label from the dataset
        # HuggingFace imagefolder datasets store labels in a 'label' column (as integers)
        label_idx = item["label"]

        return img_array, label_idx
