"""Data preprocessing: generate train/validate/test sets from board-piece combinations."""

import multiprocessing
from pathlib import Path

__all__ = ["generate_split_data"]

import numpy as np
from PIL import Image
from tqdm import tqdm

from .constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
)

# Image generation constants
BOARD_SIZE = 256  # Full board in pixels
SQUARE_SIZE = 32  # Each square in pixels (256 / 8)
BOARDS_DIR = DEFAULT_DATA_DIR / "boards"
PIECES_DIR = DEFAULT_DATA_DIR / "pieces"

# Square coordinates for rendering on dark (a1) and light (a2) squares
DARK_SQUARE = {"file": 0, "rank": 7, "name": "dark"}
LIGHT_SQUARE = {"file": 0, "rank": 6, "name": "light"}


def get_boards() -> list[str]:
    """Get all board names from boards directory.

    Returns:
        List of board names (without .png extension)
    """
    return sorted([f.stem for f in BOARDS_DIR.glob("*.png")])


def get_piece_sets() -> list[str]:
    """Get all piece set names from pieces directory.

    Returns:
        List of piece set directory names
    """
    return sorted([d.name for d in PIECES_DIR.iterdir() if d.is_dir()])


def render_square_with_piece(
    board_name: str, piece_set: str, piece_class: str, square_info: dict
) -> Image.Image:
    """Render a single square with a piece on it.

    Args:
        board_name: Name of the board (without .png)
        piece_set: Name of the piece set directory
        piece_class: Piece class name (e.g., 'bB', 'wP')
        square_info: Dict with 'file', 'rank', 'name' keys

    Returns:
        PIL Image of the 32x32 square
    """
    # Load board
    board_path = BOARDS_DIR / f"{board_name}.png"
    board_img = Image.open(board_path)

    if board_img.mode != "RGBA":
        board_img = board_img.convert("RGBA")

    # Load piece
    piece_path = PIECES_DIR / piece_set / f"{piece_class}.png"
    piece_img = Image.open(piece_path)

    if piece_img.mode != "RGBA":
        piece_img = piece_img.convert("RGBA")

    # Calculate pixel position
    x = square_info["file"] * SQUARE_SIZE
    y = square_info["rank"] * SQUARE_SIZE

    # Paste piece on board
    board_img.paste(piece_img, (x, y), piece_img)

    # Crop the square
    crop_box = (x, y, x + SQUARE_SIZE, y + SQUARE_SIZE)
    return board_img.crop(crop_box)


def render_empty_square(board_name: str, square_info: dict) -> Image.Image:
    """Render a single empty square.

    Args:
        board_name: Name of the board (without .png)
        square_info: Dict with 'file', 'rank', 'name' keys

    Returns:
        PIL Image of the 32x32 square
    """
    # Load board
    board_path = BOARDS_DIR / f"{board_name}.png"
    board_img = Image.open(board_path)

    if board_img.mode != "RGBA":
        board_img = board_img.convert("RGBA")

    # Calculate pixel position
    x = square_info["file"] * SQUARE_SIZE
    y = square_info["rank"] * SQUARE_SIZE

    # Crop the square
    crop_box = (x, y, x + SQUARE_SIZE, y + SQUARE_SIZE)
    return board_img.crop(crop_box)


def _process_combination(args: tuple) -> dict:
    """Worker function to process one board-piece set combination.

    Generates 26 images: 12 pieces × 2 squares + 2 empty squares.

    Args:
        args: Tuple of (board_name, piece_set, split_name, split_dir, piece_classes)

    Returns:
        Dict with split name and count of images generated
    """
    board, piece_set, split_name, split_dir, piece_classes = args

    # Generate images for each piece on dark and light squares
    for piece_class in piece_classes:
        if piece_class == "xx":
            continue  # Handle empty squares separately

        # Dark square
        img_dark = render_square_with_piece(board, piece_set, piece_class, DARK_SQUARE)
        output_path = split_dir / piece_class / f"{board}_{piece_set}_dark.png"
        img_dark.save(output_path)

        # Light square
        img_light = render_square_with_piece(
            board, piece_set, piece_class, LIGHT_SQUARE
        )
        output_path = split_dir / piece_class / f"{board}_{piece_set}_light.png"
        img_light.save(output_path)

    # Generate empty square images
    # Dark square
    img_dark = render_empty_square(board, DARK_SQUARE)
    output_path = split_dir / "xx" / f"{board}_{piece_set}_dark.png"
    img_dark.save(output_path)

    # Light square
    img_light = render_empty_square(board, LIGHT_SQUARE)
    output_path = split_dir / "xx" / f"{board}_{piece_set}_light.png"
    img_light.save(output_path)

    return {"split": split_name, "count": len(piece_classes) * 2}


def generate_split_data(
    model_id: str,
    train_dir: Path | None = None,
    val_dir: Path | None = None,
    test_dir: Path | None = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """Generate train/validate/test sets from board-piece combinations.

    Args:
        model_id: Model identifier (e.g., 'pieces')
        train_dir: Destination directory for training data
        val_dir: Destination directory for validation data
        test_dir: Destination directory for test data
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    """
    from .constants import get_model_config, get_test_dir, get_train_dir, get_val_dir

    # Get model configuration
    model_config = get_model_config(model_id)
    piece_classes = model_config["class_names"]

    # Set default directories if not provided
    if train_dir is None:
        train_dir = get_train_dir(model_id)
    if val_dir is None:
        val_dir = get_val_dir(model_id)
    if test_dir is None:
        test_dir = get_test_dir(model_id)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    # Set random seed
    rng = np.random.default_rng(seed)

    # Get all boards and piece sets
    boards = get_boards()
    piece_sets = get_piece_sets()

    print(f"Found {len(boards)} boards and {len(piece_sets)} piece sets")

    # Create output directory structure
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for piece_class in piece_classes:
            (split_dir / piece_class).mkdir(exist_ok=True)

    # Create all combinations and assign to splits
    combinations = [(board, piece_set) for board in boards for piece_set in piece_sets]
    total_combinations = len(combinations)

    print(f"Total combinations: {total_combinations}")

    # Randomly assign each combination to a split
    split_assignments = rng.choice(
        ["train", "val", "test"],
        size=total_combinations,
        p=[train_ratio, val_ratio, test_ratio],
    )

    # Prepare tasks for multiprocessing
    tasks = []
    for (board, piece_set), split_name in zip(combinations, split_assignments):
        if split_name == "train":
            split_dir = train_dir
        elif split_name == "val":
            split_dir = val_dir
        else:
            split_dir = test_dir
        tasks.append((board, piece_set, split_name, split_dir, piece_classes))

    # Use all CPU cores
    num_processes = multiprocessing.cpu_count()

    print(
        f"\nGenerating images using {num_processes} processes "
        f"(train/val/test = {train_ratio}/{val_ratio}/{test_ratio})..."
    )

    # Process combinations in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_process_combination, tasks),
                total=total_combinations,
                desc="Generating images",
            )
        )

    # Count images per split
    train_count = sum(r["count"] for r in results if r["split"] == "train")
    val_count = sum(r["count"] for r in results if r["split"] == "val")
    test_count = sum(r["count"] for r in results if r["split"] == "test")

    print("\nGeneration complete!")
    print(f"  Train:      {train_count:6d} images")
    print(f"  Validation: {val_count:6d} images")
    print(f"  Test:       {test_count:6d} images")
    print(f"  Total:      {train_count + val_count + test_count:6d} images")
