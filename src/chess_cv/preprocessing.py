"""Data preprocessing: generate train/validate/test sets from board-piece combinations."""

import multiprocessing
import random
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


def _index_to_subdir(index: int) -> str:
    """Convert index (0-675) to subdir name (aa-zz).

    Args:
        index: Integer from 0 to 675

    Returns:
        Two-letter subdir name (e.g., 'aa', 'ab', 'zz')

    Example:
        0 → 'aa', 1 → 'ab', 25 → 'az', 26 → 'ba', 675 → 'zz'
    """
    first = index // 26
    second = index % 26
    return chr(ord("a") + first) + chr(ord("a") + second)


def _get_subdir_for_counter(counter: int, max_files_per_dir: int = 8192) -> str:
    """Get subdir name based on file counter.

    Args:
        counter: Number of files already saved in this class
        max_files_per_dir: Maximum files per subdirectory

    Returns:
        Subdir name (e.g., 'aa', 'ab', etc.)
    """
    index = counter // max_files_per_dir
    return _index_to_subdir(index)


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


def get_arrow_types() -> list[str]:
    """Get all arrow type names from arrows directory.

    Returns:
        List of arrow type directory names (sorted)
    """
    arrows_dir = DEFAULT_DATA_DIR / "arrows"
    if not arrows_dir.exists():
        return []
    return sorted([d.name for d in arrows_dir.iterdir() if d.is_dir()])


def overlay_arrow(base_image: Image.Image, arrow_type: str) -> tuple[Image.Image, str]:
    """Overlay a random arrow image of the specified type onto a base image.

    Args:
        base_image: Base PIL Image (32x32 square)
        arrow_type: Arrow type directory name (e.g., 'head-N', 'corner-E-S')

    Returns:
        Tuple of (PIL Image with arrow overlayed, overlay_name)
        overlay_name is the filename stem (e.g., 'chess-com-blue', 'lichess-red')
    """
    arrows_dir = DEFAULT_DATA_DIR / "arrows" / arrow_type

    # Get all arrow images for this type
    arrow_files = list(arrows_dir.glob("*.png"))
    if not arrow_files:
        raise FileNotFoundError(f"No arrow images found in {arrows_dir}")

    # Pick a random arrow image from this type
    arrow_path = random.choice(arrow_files)
    overlay_name = arrow_path.stem  # e.g., 'chess-com-blue'

    arrow_img = Image.open(arrow_path)

    if arrow_img.mode != "RGBA":
        arrow_img = arrow_img.convert("RGBA")

    # Convert base image to RGBA for compositing
    if base_image.mode != "RGBA":
        base_image = base_image.convert("RGBA")

    # Composite arrow on top of base using alpha channel
    result = Image.alpha_composite(base_image, arrow_img)

    return result, overlay_name


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
        args: Tuple of (board_name, piece_set, split_name, split_dir, piece_classes, counters)

    Returns:
        Dict with split name and count of images generated
    """
    board, piece_set, split_name, split_dir, piece_classes, counters = args

    # Generate images for each piece on dark and light squares
    for piece_class in piece_classes:
        if piece_class == "xx":
            continue  # Handle empty squares separately

        # Dark square
        img_dark = render_square_with_piece(board, piece_set, piece_class, DARK_SQUARE)
        counter = counters[piece_class]
        subdir = _get_subdir_for_counter(counter)
        output_dir = split_dir / piece_class / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{board}_{piece_set}_dark.png"
        img_dark.save(output_path)
        counters[piece_class] = counter + 1

        # Light square
        img_light = render_square_with_piece(
            board, piece_set, piece_class, LIGHT_SQUARE
        )
        counter = counters[piece_class]
        subdir = _get_subdir_for_counter(counter)
        output_dir = split_dir / piece_class / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{board}_{piece_set}_light.png"
        img_light.save(output_path)
        counters[piece_class] = counter + 1

    # Generate empty square images
    # Dark square
    img_dark = render_empty_square(board, DARK_SQUARE)
    counter = counters["xx"]
    subdir = _get_subdir_for_counter(counter)
    output_dir = split_dir / "xx" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{board}_{piece_set}_dark.png"
    img_dark.save(output_path)
    counters["xx"] = counter + 1

    # Light square
    img_light = render_empty_square(board, LIGHT_SQUARE)
    counter = counters["xx"]
    subdir = _get_subdir_for_counter(counter)
    output_dir = split_dir / "xx" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{board}_{piece_set}_light.png"
    img_light.save(output_path)
    counters["xx"] = counter + 1

    return {"split": split_name, "count": len(piece_classes) * 2}


def _process_arrows_combination(args: tuple) -> dict:
    """Worker function to process one board-piece set combination for arrows model.

    Step 1: Generate all base images (pieces + empty squares) and save to xx category (no arrows).
    Step 2: For each arrow category, duplicate all base images with arrow overlays applied.

    Args:
        args: Tuple of (board_name, piece_set, split_name, split_dir, arrow_types, counters)

    Returns:
        Dict with split name and count of images generated
    """
    board, piece_set, split_name, split_dir, arrow_types, counters = args
    piece_classes = [
        "bB",
        "bK",
        "bN",
        "bP",
        "bQ",
        "bR",
        "wB",
        "wK",
        "wN",
        "wP",
        "wQ",
        "wR",
        "xx",
    ]

    # Step 1: Generate and save base images to xx category (no arrows)
    # Store filenames to reuse for arrow overlay generation
    base_image_files = []

    # Generate images for each piece on dark and light squares
    for piece_class in piece_classes:
        if piece_class == "xx":
            continue

        # Dark square with piece
        img_dark = render_square_with_piece(board, piece_set, piece_class, DARK_SQUARE)
        filename_dark = f"{board}_{piece_set}_{piece_class}_dark.png"
        counter = counters["xx"]
        subdir = _get_subdir_for_counter(counter)
        output_dir = split_dir / "xx" / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path_dark = output_dir / filename_dark
        img_dark.save(output_path_dark)
        counters["xx"] = counter + 1
        base_image_files.append((piece_class, "dark", output_path_dark))

        # Light square with piece
        img_light = render_square_with_piece(
            board, piece_set, piece_class, LIGHT_SQUARE
        )
        filename_light = f"{board}_{piece_set}_{piece_class}_light.png"
        counter = counters["xx"]
        subdir = _get_subdir_for_counter(counter)
        output_dir = split_dir / "xx" / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path_light = output_dir / filename_light
        img_light.save(output_path_light)
        counters["xx"] = counter + 1
        base_image_files.append((piece_class, "light", output_path_light))

    # Generate empty square images
    # Dark empty square
    img_dark = render_empty_square(board, DARK_SQUARE)
    filename_dark = f"{board}_{piece_set}_xx_dark.png"
    counter = counters["xx"]
    subdir = _get_subdir_for_counter(counter)
    output_dir = split_dir / "xx" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_dark = output_dir / filename_dark
    img_dark.save(output_path_dark)
    counters["xx"] = counter + 1
    base_image_files.append(("xx", "dark", output_path_dark))

    # Light empty square
    img_light = render_empty_square(board, LIGHT_SQUARE)
    filename_light = f"{board}_{piece_set}_xx_light.png"
    counter = counters["xx"]
    subdir = _get_subdir_for_counter(counter)
    output_dir = split_dir / "xx" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_light = output_dir / filename_light
    img_light.save(output_path_light)
    counters["xx"] = counter + 1
    base_image_files.append(("xx", "light", output_path_light))

    # Step 2: For each arrow category, apply arrow overlays to all base images
    for arrow_type in arrow_types:
        if arrow_type == "xx":
            continue  # Already generated in Step 1

        # Apply arrow overlay to each base image
        for piece_class, square_type, base_image_path in base_image_files:
            # Load the base image
            base_img = Image.open(base_image_path)

            # Apply arrow overlay and get the overlay name
            arrow_img, overlay_name = overlay_arrow(base_img, arrow_type)

            # Save with filename including piece class and overlay name
            # Format: {board}_{piece_set}_{piece_class}_{square_type}-{overlay_name}.png
            filename = (
                f"{board}_{piece_set}_{piece_class}_{square_type}-{overlay_name}.png"
            )
            counter = counters[arrow_type]
            subdir = _get_subdir_for_counter(counter)
            output_dir = split_dir / arrow_type / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
            arrow_img.save(output_path)
            counters[arrow_type] = counter + 1

    # Total images = 26 for xx + (26 * 48 arrow types) = 26 * 49 = 1274 images
    total_images = len(base_image_files) * len(arrow_types)
    return {"split": split_name, "count": total_images}


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
    class_names = model_config["class_names"]

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

    # Initialize counters for each split using multiprocessing Manager
    manager = multiprocessing.Manager()
    train_counters = manager.dict({class_name: 0 for class_name in class_names})
    val_counters = manager.dict({class_name: 0 for class_name in class_names})
    test_counters = manager.dict({class_name: 0 for class_name in class_names})

    # Prepare tasks for multiprocessing
    tasks = []
    for (board, piece_set), split_name in zip(combinations, split_assignments):
        if split_name == "train":
            split_dir = train_dir
            counters = train_counters
        elif split_name == "val":
            split_dir = val_dir
            counters = val_counters
        else:
            split_dir = test_dir
            counters = test_counters

        # Ensure base split directory exists
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create base class directories
        for class_name in class_names:
            (split_dir / class_name).mkdir(exist_ok=True)

        # Add counter to task arguments
        tasks.append((board, piece_set, split_name, split_dir, class_names, counters))

    # Use all CPU cores
    num_processes = multiprocessing.cpu_count()

    print(
        f"\nGenerating images using {num_processes} processes "
        f"(train/val/test = {train_ratio}/{val_ratio}/{test_ratio})..."
    )

    # Choose the appropriate processing function based on model_id
    if model_id == "arrows":
        process_func = _process_arrows_combination
    else:  # pieces or other models
        process_func = _process_combination

    # Process combinations in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_func, tasks),
                total=total_combinations,
                desc="Generating images",
                smoothing=0.9,
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
