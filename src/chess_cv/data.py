"""Data loading utilities for chess piece images."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image


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


def load_image(image_path: Path) -> np.ndarray:
    """Load and preprocess a single image.

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image as numpy array (H, W, C) with values in [0, 1]
    """
    # Load image with PIL
    img = Image.open(image_path)

    # Convert RGBA to RGB
    if img.mode == "RGBA":
        # Create white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    return img_array


def load_dataset(data_dir: Path) -> tuple[mx.array, mx.array, list[str]]:
    """Load all images from a directory.

    Args:
        data_dir: Directory containing class subdirectories

    Returns:
        Tuple of (images, labels, file_paths)
        - images: mx.array of shape (N, H, W, C)
        - labels: mx.array of shape (N,) with class indices
        - file_paths: List of file paths for debugging
    """
    images = []
    labels = []
    file_paths = []

    # Create class to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}

    # Iterate through class directories
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        # Load all images in this class
        for img_path in sorted(class_dir.glob("*.png")):
            img = load_image(img_path)
            images.append(img)
            labels.append(class_to_idx[class_name])
            file_paths.append(str(img_path))

    # Convert to MLX arrays
    images_array = mx.array(np.array(images))
    labels_array = mx.array(np.array(labels, dtype=np.int32))

    return images_array, labels_array, file_paths


def batch_iterate(
    images: mx.array,
    labels: mx.array,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Iterate over dataset in batches.

    Args:
        images: Images array of shape (N, H, W, C)
        labels: Labels array of shape (N,)
        batch_size: Batch size
        shuffle: Whether to shuffle data (default: True)
        seed: Random seed for shuffling (default: None)

    Yields:
        Tuples of (batch_images, batch_labels)
    """
    n_samples = images.shape[0]

    # Create indices
    if shuffle:
        if seed is not None:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_samples)
        else:
            perm = np.random.permutation(n_samples)
        indices = mx.array(perm)
    else:
        indices = mx.arange(n_samples)

    # Iterate in batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_images = images[batch_indices]
        batch_labels = labels[batch_indices]

        yield batch_images, batch_labels


class DataLoader:
    """Data loader for chess piece images."""

    def __init__(
        self,
        data_dir: Path | str,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        """Initialize data loader.

        Args:
            data_dir: Directory containing class subdirectories
            batch_size: Batch size (default: 32)
            shuffle: Whether to shuffle data (default: True)
            seed: Random seed for shuffling (default: None)
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Load dataset
        print(f"Loading data from {self.data_dir}...")
        self.images, self.labels, self.file_paths = load_dataset(self.data_dir)
        print(f"Loaded {len(self.images)} images")

    def __iter__(self):
        """Return iterator over batches."""
        return batch_iterate(
            self.images,
            self.labels,
            self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed,
        )

    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.images) + self.batch_size - 1) // self.batch_size

    @property
    def num_samples(self) -> int:
        """Return total number of samples."""
        return len(self.images)
