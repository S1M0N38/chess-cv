"""Data preprocessing: split data into train/validate/test sets."""

import shutil
from pathlib import Path

import numpy as np


def split_data(
    source_dir: Path = Path("data/all"),
    train_dir: Path = Path("data/train"),
    val_dir: Path = Path("data/validate"),
    test_dir: Path = Path("data/test"),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Split data into train/validate/test sets with stratification.

    Args:
        source_dir: Source directory containing class subdirectories
        train_dir: Destination directory for training data
        val_dir: Destination directory for validation data
        test_dir: Destination directory for test data
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    # Set random seed
    rng = np.random.default_rng(seed)

    # Get all class directories
    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    class_dirs = [d for d in class_dirs if not d.name.startswith(".")]

    print(f"Found {len(class_dirs)} classes:")
    for class_dir in class_dirs:
        print(f"  - {class_dir.name}")

    # Create destination directories
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_dir in class_dirs:
            (split_dir / class_dir.name).mkdir(exist_ok=True)

    # Split each class
    total_train = 0
    total_val = 0
    total_test = 0

    print(f"\nSplitting data (train/val/test = {train_ratio}/{val_ratio}/{test_ratio}):")

    for class_dir in class_dirs:
        # Get all image files
        image_files = sorted(list(class_dir.glob("*.png")))

        # Shuffle files
        indices = rng.permutation(len(image_files))

        # Calculate split points
        n_train = int(len(image_files) * train_ratio)
        n_val = int(len(image_files) * val_ratio)

        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        # Copy files to respective directories
        for idx in train_indices:
            shutil.copy2(
                image_files[idx],
                train_dir / class_dir.name / image_files[idx].name,
            )
        for idx in val_indices:
            shutil.copy2(
                image_files[idx],
                val_dir / class_dir.name / image_files[idx].name,
            )
        for idx in test_indices:
            shutil.copy2(
                image_files[idx],
                test_dir / class_dir.name / image_files[idx].name,
            )

        total_train += len(train_indices)
        total_val += len(val_indices)
        total_test += len(test_indices)

        print(
            f"  {class_dir.name:20s}: {len(train_indices):4d} train, "
            f"{len(val_indices):4d} val, {len(test_indices):4d} test"
        )

    print(f"\nTotal:")
    print(f"  Train:      {total_train:5d} images")
    print(f"  Validation: {total_val:5d} images")
    print(f"  Test:       {total_test:5d} images")
    print(f"  Total:      {total_train + total_val + total_test:5d} images")


def main() -> None:
    """Run data preprocessing."""
    split_data()
    print("\n✓ Data preprocessing complete!")


if __name__ == "__main__":
    main()
