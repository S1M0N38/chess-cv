"""Data preprocessing: generate train/validate/test sets from board-piece combinations.

This module provides functionality to generate synthetic training data by combining:
- Board images (256×256px) from data/boards/
- Piece images (32×32px) from data/pieces/
- Arrow overlays (32×32px) from data/arrows/

The generated data is split into train/validate/test sets according to configured ratios
and saved to data/splits/{model_id}/{train,validate,test}/.
"""

import os
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    get_model_config,
    get_test_dir,
    get_train_dir,
    get_val_dir,
)

__all__ = ["generate_split_data"]

# Constants
BOARD_SIZE = 256  # Full board in pixels
SQUARE_SIZE = 32  # Each square in pixels (256 / 8)
BOARDS_DIR = DEFAULT_DATA_DIR / "boards"
PIECES_DIR = DEFAULT_DATA_DIR / "pieces"
ARROWS_DIR = DEFAULT_DATA_DIR / "arrows"

LIGHT_SQUARE_COORDS = (  # e4 square
    SQUARE_SIZE * 4,  # e file
    SQUARE_SIZE * 4,  # 4 rank
    SQUARE_SIZE * 5,  # f file
    SQUARE_SIZE * 5,  # 3 rank
)
DARK_SQUARE_COORDS = (  # d4 square
    SQUARE_SIZE * 3,  # d file
    SQUARE_SIZE * 4,  # 4 rank
    SQUARE_SIZE * 4,  # e file
    SQUARE_SIZE * 5,  # 3 rank
)

# Configuration
# Split ratios and random seed are configured via constants.py
# Future enhancement: Add ratio/seed parameters to generate_split_data()
NUM_SNAP_VARIATIONS = 4  # Number of random translations to generate per image

# Global caches for loaded images (populated by worker initializers)
# These are shared across worker processes for efficiency
SQUARES: dict[str, Image.Image] = {}
PIECES: dict[str, dict[str, Image.Image]] = defaultdict(dict)
ARROWS: dict[str, dict[str, Image.Image]] = defaultdict(dict)

np.random.seed(DEFAULT_RANDOM_SEED)
random.seed(DEFAULT_RANDOM_SEED)

################################################################################
# Validation
################################################################################


def _validate_data_sources(model_id: str) -> None:
    """Validate that required data sources exist.

    Args:
        model_id: Model identifier

    Raises:
        FileNotFoundError: If required directories or files are missing
    """
    if not BOARDS_DIR.exists():
        raise FileNotFoundError(f"Boards directory not found: {BOARDS_DIR}")

    boards = list(BOARDS_DIR.glob("*.png"))
    if not boards:
        raise FileNotFoundError(f"No board images found in {BOARDS_DIR}")

    if not PIECES_DIR.exists():
        raise FileNotFoundError(f"Pieces directory not found: {PIECES_DIR}")

    piece_sets = [d for d in PIECES_DIR.iterdir() if d.is_dir()]
    if not piece_sets:
        raise FileNotFoundError(f"No piece sets found in {PIECES_DIR}")

    if model_id == "arrows":
        if not ARROWS_DIR.exists():
            raise FileNotFoundError(f"Arrows directory not found: {ARROWS_DIR}")
        arrow_types = [d for d in ARROWS_DIR.iterdir() if d.is_dir()]
        if not arrow_types:
            raise FileNotFoundError(f"No arrow types found in {ARROWS_DIR}")

    print("✓ Validated data sources:")
    print(f"  - Boards: {len(boards)}")
    print(f"  - Piece sets: {len(piece_sets)}")
    if model_id == "arrows":
        print(f"  - Arrow types: {len(arrow_types)}")


################################################################################
# Utils
################################################################################


def stats_splits(model_id: str) -> None:
    """Print comprehensive statistics about generated splits.

    Args:
        model_id: Model identifier for directory lookup
    """
    train_dir = get_train_dir(model_id)
    val_dir = get_val_dir(model_id)
    test_dir = get_test_dir(model_id)

    # Count images in each split
    train_count = sum(1 for _ in train_dir.rglob("*.png"))
    val_count = sum(1 for _ in val_dir.rglob("*.png"))
    test_count = sum(1 for _ in test_dir.rglob("*.png"))
    total_count = train_count + val_count + test_count

    # Count classes (from training directory)
    train_classes = len([d for d in train_dir.iterdir() if d.is_dir()])

    # Print statistics
    print("\n" + "=" * 60)
    print(f"SPLIT STATISTICS: {model_id}")
    print("=" * 60)
    print(f"Classes:           {train_classes}")
    print(
        f"Training images:   {train_count:,} ({train_count / total_count * 100:.1f}%)"
    )
    print(f"Validation images: {val_count:,} ({val_count / total_count * 100:.1f}%)")
    print(f"Test images:       {test_count:,} ({test_count / total_count * 100:.1f}%)")
    print(f"Total images:      {total_count:,}")
    print("=" * 60 + "\n")


def assign_split(train_dir: Path, val_dir: Path, test_dir: Path) -> Path:
    """Randomly assign an image to train/validate/test split.

    Args:
        train_dir: Training directory
        val_dir: Validation directory
        test_dir: Test directory

    Returns:
        Directory path for the assigned split
    """
    rand = np.random.random()
    if rand < DEFAULT_TRAIN_RATIO:
        return train_dir
    elif rand < DEFAULT_TRAIN_RATIO + DEFAULT_VAL_RATIO:
        return val_dir
    else:
        return test_dir


################################################################################
# Load functions
################################################################################


def load_squares() -> None:
    """Load light and dark squares from boards directory."""
    for board_name in BOARDS_DIR.glob("*.png"):
        board_img = Image.open(board_name).convert("RGBA")
        SQUARES[f"{board_name.stem}_light"] = board_img.crop(LIGHT_SQUARE_COORDS)
        SQUARES[f"{board_name.stem}_dark"] = board_img.crop(DARK_SQUARE_COORDS)


def load_pieces() -> None:
    """Load pieces from pieces directory + transparent image for empty squares."""
    for piece_set in PIECES_DIR.iterdir():
        if piece_set.is_dir():
            for piece_class in piece_set.glob("*.png"):
                piece_img = Image.open(piece_class).convert("RGBA")
                PIECES[piece_class.stem][piece_set.stem] = piece_img
            xx_image = Image.new("RGBA", (SQUARE_SIZE, SQUARE_SIZE), (0, 0, 0, 0))
            PIECES["xx"][piece_set.stem] = xx_image


def load_arrows() -> None:
    """Load arrows from arrows directory."""
    for arrow_class in ARROWS_DIR.iterdir():
        if arrow_class.is_dir():
            for arrow_type in arrow_class.glob("*.png"):
                arrow_img = Image.open(arrow_type).convert("RGBA")
                ARROWS[arrow_class.stem][arrow_type.stem] = arrow_img
    xx_image = Image.new("RGBA", (SQUARE_SIZE, SQUARE_SIZE), (0, 0, 0, 0))
    ARROWS["xx"]["no-arrows"] = xx_image


################################################################################
# Pieces model
################################################################################


def _init_pieces_dirs() -> tuple[Path, Path, Path]:
    """Create directories for train/validate/test splits for pieces model.

    Returns:
        Tuple of (train_dir, val_dir, test_dir)
    """
    model_id = "pieces"

    train_dir = get_train_dir(model_id)
    val_dir = get_val_dir(model_id)
    test_dir = get_test_dir(model_id)

    for split_dir in [train_dir, val_dir, test_dir]:
        for piece_class in PIECES.keys():
            (split_dir / piece_class).mkdir(exist_ok=True, parents=True)

    return train_dir, val_dir, test_dir


def _process_piece_class(piece_class: str) -> int:
    """Worker function to process a single piece class.

    Args:
        piece_class: The piece class to process (e.g., "wP", "bK", "xx")

    Returns:
        Number of images generated for this class
    """
    train_dir, val_dir, test_dir = _init_pieces_dirs()
    piece_set = PIECES[piece_class]
    count = 0

    for piece_name, piece in piece_set.items():
        for square_name, square in SQUARES.items():
            image = Image.alpha_composite(square, piece).convert("RGB")
            split_dir = assign_split(train_dir, val_dir, test_dir)
            image.save(split_dir / piece_class / f"{square_name}_{piece_name}.png")
            count += 1

    return count


def _init_worker_pieces() -> None:
    """Initialize worker process by loading squares and pieces."""
    load_squares()
    load_pieces()


def _save_splits_pieces() -> None:
    """Generate and save piece images to train/validate/test splits (parallelized)."""
    # Initialize directories in main process
    _init_pieces_dirs()

    # Load data in main process to get class list
    load_squares()
    load_pieces()

    piece_classes = list(PIECES.keys())
    num_workers = os.cpu_count() or 1

    # Calculate total images upfront
    num_pieces = sum(len(PIECES[piece_class]) for piece_class in piece_classes)
    total_images = len(SQUARES) * num_pieces

    print(
        f"Processing {len(piece_classes)} piece classes with {num_workers} workers..."
    )

    with Pool(processes=num_workers, initializer=_init_worker_pieces) as pool:
        with tqdm(total=total_images, desc="Generating images") as pbar:
            for count in pool.imap(_process_piece_class, piece_classes):
                pbar.update(count)


################################################################################
# Arrows model
################################################################################


def _init_arrows_dirs() -> tuple[Path, Path, Path]:
    """Create directories for train/validate/test splits for arrows model.

    Returns:
        Tuple of (train_dir, val_dir, test_dir)
    """
    model_id = "arrows"

    train_dir = get_train_dir(model_id)
    val_dir = get_val_dir(model_id)
    test_dir = get_test_dir(model_id)

    for split_dir in [train_dir, val_dir, test_dir]:
        for arrow_class in ARROWS.keys():
            (split_dir / arrow_class).mkdir(exist_ok=True, parents=True)

    return train_dir, val_dir, test_dir


def _process_arrow_class(arrow_class: str) -> int:
    """Worker function to process a single arrow class.

    Args:
        arrow_class: The arrow class to process (e.g., "head-N", "tail-E")

    Returns:
        Number of images generated for this class
    """
    train_dir, val_dir, test_dir = _init_arrows_dirs()
    arrow_set = ARROWS[arrow_class]
    count = 0

    for piece_class, piece_set in PIECES.items():
        for piece_name, piece in piece_set.items():
            for square_name, square in SQUARES.items():
                arrow_name, arrow_img = random.choice(list(arrow_set.items()))
                square_img = Image.alpha_composite(square, piece)
                image = Image.alpha_composite(square_img, arrow_img)
                split_dir = assign_split(train_dir, val_dir, test_dir)
                image.save(
                    split_dir
                    / arrow_class
                    / f"{square_name}_{piece_class}_{piece_name}_{arrow_name}.png"
                )
                count += 1

    return count


def _init_worker_arrows() -> None:
    """Initialize worker process by loading squares, pieces, and arrows."""
    load_squares()
    load_pieces()
    load_arrows()


def _save_splits_arrows() -> None:
    """Generate and save arrow images to train/validate/test splits (parallelized)."""
    # Initialize directories in main process
    _init_arrows_dirs()

    # Load data in main process to get class list
    load_squares()
    load_pieces()
    load_arrows()

    arrow_classes = list(ARROWS.keys())
    num_workers = os.cpu_count() or 1

    # Calculate total images upfront
    num_pieces = sum(len(PIECES[piece_class]) for piece_class in PIECES.keys())
    num_arrows = len(ARROWS)  # one random arrow from the set of arrows for each class
    total_images = len(SQUARES) * num_pieces * num_arrows

    print(
        f"Processing {len(arrow_classes)} arrow classes with {num_workers} workers..."
    )

    with Pool(processes=num_workers, initializer=_init_worker_arrows) as pool:
        with tqdm(total=total_images, desc="Generating images") as pbar:
            for count in pool.imap(_process_arrow_class, arrow_classes):
                pbar.update(count)


################################################################################
# Snap model
################################################################################


def _apply_snap_transform(image: Image.Image, snap_class: str) -> Image.Image:
    """Apply centering transformation to simulate piece positioning.

    Args:
        image: RGBA image to transform
        snap_class: Either "ok" (centered/slightly off-centered) or "bad" (off-centered)

    Returns:
        Transformed RGBA image
    """
    if snap_class == "ok":
        # For "ok" class: minimal shifting (0-3 pixels) to simulate slight misalignment
        min_shift = 0
        max_shift = 3
    else:  # snap_class == "bad"
        # For "bad" class: significant shifting (3-14 pixels)
        min_shift = 3
        max_shift = 14

    # Convert PIL to numpy array for OpenCV processing
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Generate random shift values
    if snap_class == "ok":
        shift_x = random.randint(min_shift, max_shift) * random.choice([-1, 1])
        shift_y = random.randint(min_shift, max_shift) * random.choice([-1, 1])
    else:  # snap_class == "bad"
        shift_x = random.randint(min_shift, max_shift) * random.choice([-1, 1])
        shift_y = random.randint(min_shift, max_shift) * random.choice([-1, 1])

    # Create transformation matrix for translation
    M: np.ndarray = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)

    # Apply the transformation
    shifted = cv2.warpAffine(
        img_array,
        M,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )  # Transparent border

    # Convert back to PIL Image
    return Image.fromarray(shifted, "RGBA")


def _init_snap_dirs() -> tuple[Path, Path, Path]:
    """Create directories for train/validate/test splits for snap model.

    Returns:
        Tuple of (train_dir, val_dir, test_dir)
    """
    model_id = "snap"

    train_dir = get_train_dir(model_id)
    val_dir = get_val_dir(model_id)
    test_dir = get_test_dir(model_id)

    for split_dir in [train_dir, val_dir, test_dir]:
        for snap_class in ["ok", "bad"]:
            (split_dir / snap_class).mkdir(exist_ok=True, parents=True)

    return train_dir, val_dir, test_dir


def _process_snap_piece_class(piece_class: str) -> int:
    """Worker function to process a single piece class for snap model.

    Args:
        piece_class: The piece class to process (e.g., "wP", "bK", "xx")

    Returns:
        Number of images generated for this piece class
    """
    train_dir, val_dir, test_dir = _init_snap_dirs()
    piece_set = PIECES[piece_class]
    count = 0

    for piece_name, piece in piece_set.items():
        for square_name, square in SQUARES.items():
            if piece_class == "xx":
                # Empty square - only generate "ok" variations (no transformation)
                for variation in range(NUM_SNAP_VARIATIONS):
                    image = square.convert("RGB")
                    split_dir = assign_split(train_dir, val_dir, test_dir)
                    image.save(
                        split_dir
                        / "ok"
                        / f"{square_name}_{piece_name}_var{variation}.png"
                    )
                    count += 1
            else:
                # Non-empty piece - generate both "ok" and "bad" variations
                for variation in range(NUM_SNAP_VARIATIONS):
                    # Generate "ok" variation
                    transformed_piece_ok = _apply_snap_transform(piece, "ok")
                    square_img_ok = Image.alpha_composite(square, transformed_piece_ok)
                    image_ok = square_img_ok.convert("RGB")
                    split_dir_ok = assign_split(train_dir, val_dir, test_dir)
                    image_ok.save(
                        split_dir_ok
                        / "ok"
                        / f"{square_name}_{piece_class}_{piece_name}_var{variation}.png"
                    )
                    count += 1

                    # Generate "bad" variation
                    transformed_piece_bad = _apply_snap_transform(piece, "bad")
                    square_img_bad = Image.alpha_composite(
                        square, transformed_piece_bad
                    )
                    image_bad = square_img_bad.convert("RGB")
                    split_dir_bad = assign_split(train_dir, val_dir, test_dir)
                    image_bad.save(
                        split_dir_bad
                        / "bad"
                        / f"{square_name}_{piece_class}_{piece_name}_var{variation}.png"
                    )
                    count += 1

    return count


def _init_worker_snap() -> None:
    """Initialize worker process by loading squares and pieces."""
    load_squares()
    load_pieces()


def _save_splits_snap() -> None:
    """Generate and save snap images to train/validate/test splits (parallelized)."""
    # Initialize directories in main process
    _init_snap_dirs()

    # Load data in main process to get class list
    load_squares()
    load_pieces()

    piece_classes = list(PIECES.keys())
    num_workers = os.cpu_count() or 1

    # Calculate total images upfront
    # For empty squares: 4 "ok" variations per combination
    # For non-empty pieces: 4 "ok" + 4 "bad" = 8 variations per combination
    num_empty = len(PIECES["xx"])
    num_non_empty = sum(len(PIECES[pc]) for pc in PIECES.keys()) - num_empty
    total_empty_images = len(SQUARES) * num_empty * NUM_SNAP_VARIATIONS
    total_non_empty_images = len(SQUARES) * num_non_empty * NUM_SNAP_VARIATIONS * 2
    total_images = total_empty_images + total_non_empty_images

    print(
        f"Processing {len(piece_classes)} piece classes with {num_workers} workers..."
    )

    with Pool(processes=num_workers, initializer=_init_worker_snap) as pool:
        with tqdm(total=total_images, desc="Generating images") as pbar:
            for count in pool.imap(_process_snap_piece_class, piece_classes):
                pbar.update(count)


################################################################################
# Public API
################################################################################


def generate_split_data(
    model_id: str,
    train_dir: Path | None = None,
    val_dir: Path | None = None,
    test_dir: Path | None = None,
) -> None:
    """Generate train/validate/test sets from board-piece combinations.

    Main entry point called by CLI. Routes to appropriate model-specific
    generator based on model_id. This function wraps the model-specific
    generation logic and provides a unified interface.

    The function generates synthetic training data by:
    1. Loading board images and extracting light/dark squares
    2. Loading piece/arrow images
    3. Compositing pieces/arrows onto squares
    4. Randomly splitting into train/validate/test sets
    5. Saving to model-specific directories

    Args:
        model_id: Model identifier ('pieces' or 'arrows')
        train_dir: Training directory (default: data/splits/{model_id}/train)
        val_dir: Validation directory (default: data/splits/{model_id}/validate)
        test_dir: Test directory (default: data/splits/{model_id}/test)

    Raises:
        ValueError: If model_id is not supported
        FileNotFoundError: If required data sources are missing

    Examples:
        >>> # Generate data for pieces model
        >>> generate_split_data("pieces")

        >>> # Generate data for arrows model
        >>> generate_split_data("arrows")
    """
    # Validate model_id
    model_config = get_model_config(model_id)

    # Set default directories if not provided
    if train_dir is None:
        train_dir = get_train_dir(model_id)
    if val_dir is None:
        val_dir = get_val_dir(model_id)
    if test_dir is None:
        test_dir = get_test_dir(model_id)

    # Validate data sources exist
    _validate_data_sources(model_id)

    # Route to appropriate generator
    print(f"\nGenerating data for model: {model_id}")
    print(f"Description: {model_config['description']}")
    print(f"Classes: {model_config['num_classes']}\n")

    if model_id == "pieces":
        _save_splits_pieces()
    elif model_id == "arrows":
        _save_splits_arrows()
    elif model_id == "snap":
        _save_splits_snap()
    else:
        raise ValueError(f"No generator implemented for model_id: {model_id}")

    # Print statistics
    stats_splits(model_id)
