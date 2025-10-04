"""Data preprocessing: split data into train/validate/test sets."""

import argparse
import shutil
from pathlib import Path

__all__ = ["split_data", "main"]

import numpy as np

from .constants import (
    DEFAULT_ALL_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_DIR,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_DIR,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_DIR,
    DEFAULT_VAL_RATIO,
)


def split_data(
    source_dir: Path = DEFAULT_ALL_DIR,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    val_dir: Path = DEFAULT_VAL_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
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

    print(
        f"\nSplitting data (train/val/test = {train_ratio}/{val_ratio}/{test_ratio}):"
    )

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

    print("\nTotal:")
    print(f"  Train:      {total_train:5d} images")
    print(f"  Validation: {total_val:5d} images")
    print(f"  Test:       {total_test:5d} images")
    print(f"  Total:      {total_train + total_val + total_test:5d} images")


def main() -> None:
    """Run data preprocessing with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Split chess piece data into train/validate/test sets"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_ALL_DIR,
        help=f"Source directory containing class subdirectories (default: {DEFAULT_ALL_DIR})",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help=f"Training data output directory (default: {DEFAULT_TRAIN_DIR})",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help=f"Validation data output directory (default: {DEFAULT_VAL_DIR})",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help=f"Test data output directory (default: {DEFAULT_TEST_DIR})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Training data ratio (default: {DEFAULT_TRAIN_RATIO})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Validation data ratio (default: {DEFAULT_VAL_RATIO})",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Test data ratio (default: {DEFAULT_TEST_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})",
    )

    args = parser.parse_args()

    split_data(
        source_dir=args.source_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print("\n✓ Data preprocessing complete!")


if __name__ == "__main__":
    main()
