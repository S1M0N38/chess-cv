"""Constants and default configuration values for chess-cv."""

from pathlib import Path

# Data paths
DEFAULT_DATA_DIR = Path("data")
DEFAULT_ALL_DIR = DEFAULT_DATA_DIR / "all"


def get_splits_dir(model_id: str) -> Path:
    """Get splits directory for a specific model."""
    return DEFAULT_DATA_DIR / "splits" / model_id


def get_train_dir(model_id: str) -> Path:
    """Get training directory for a specific model."""
    return get_splits_dir(model_id) / "train"


def get_val_dir(model_id: str) -> Path:
    """Get validation directory for a specific model."""
    return get_splits_dir(model_id) / "validate"


def get_test_dir(model_id: str) -> Path:
    """Get test directory for a specific model."""
    return get_splits_dir(model_id) / "test"


def get_checkpoint_dir(model_id: str) -> Path:
    """Get checkpoint directory for a specific model.

    This is primarily used for training/development checkpoints.
    For loading bundled pre-trained weights, use get_bundled_weight_path()
    from the utils module instead.
    """
    return Path("checkpoints") / model_id


def get_output_dir(model_id: str) -> Path:
    """Get output directory for a specific model."""
    return Path("outputs") / model_id


# Model parameters
DEFAULT_NUM_CLASSES = 13
DEFAULT_IMAGE_SIZE = 32
DEFAULT_DROPOUT = 0.5

# Training hyperparameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0003  # Used only when scheduler is disabled
DEFAULT_WEIGHT_DECAY = 0.001
DEFAULT_NUM_EPOCHS = 200
DEFAULT_PATIENCE = 999999  # Effectively disabled

# Learning rate scheduler parameters
DEFAULT_BASE_LR = 0.001  # Peak LR after warmup
DEFAULT_MIN_LR = 1e-5
DEFAULT_WARMUP_RATIO = 0.03  # Warmup for 3% of total training steps

# Data loading
DEFAULT_NUM_WORKERS = 8

# Logging configuration
LOG_TRAIN_EVERY_N_STEPS = 200  # Log training metrics every N batches
LOG_VALIDATE_EVERY_N_STEPS = 1000  # Run full validation every N batches

# Data splitting ratios
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42

# Augmentation resource directories
DEFAULT_ARROW_DIR = DEFAULT_DATA_DIR / "arrows"
DEFAULT_HIGHLIGHT_DIR = DEFAULT_DATA_DIR / "highlights"

# Model-specific augmentation configurations
AUGMENTATION_CONFIGS = {
    "pieces": {
        # Step 1: Pad to create rotation space (32 + 16*2 = 64x64)
        "padding": 16,
        "padding_mode": "edge",
        # Step 2: Random rotation (±10 degrees)
        "rotation_degrees": 10,
        # Step 3: Center crop to remove black bands from rotation
        # Formula: 64 - (ceil(tan(10°) * 64) * 2) = 64 - 24 = 40
        "center_crop_size": 40,
        # Step 4: Random crop with zoom variation, then resize to final size
        # Base scale: (32/40)² = 0.64 (area ratio for translation without zoom)
        # With variation: (0.54, 0.74) adds ±16% zoom range
        "final_size": 32,
        "resized_crop_scale": (0.54, 0.74),  # (32²/40²) ± 0.1 = 0.64 ± 0.1
        "resized_crop_ratio": (0.9, 1.1),  # (1.0, 1.0) ± 0.1 for slight stretch
        # Step 5-9: Overlay, flip, color, and noise augmentations
        "arrow_probability": 0.80,
        "highlight_probability": 0.25,
        "horizontal_flip": True,
        "horizontal_flip_prob": 0.5,
        "brightness": 0.15,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.2,
        "noise_mean": 0.0,
        "noise_sigma": 0.05,
    },
    "arrows": {
        "arrow_probability": 0.0,
        "highlight_probability": 0.25,
        "scale_min": 0.75,
        "scale_max": 1.0,
        "horizontal_flip": False,
        "brightness": 0.20,
        "contrast": 0.20,
        "saturation": 0.20,
        "hue": 0.2,
        "rotation_degrees": 2,
        "noise_mean": 0.0,
        "noise_sigma": 0.10,
    },
}

# File patterns
IMAGE_PATTERN = "**/*.png"

# Checkpoint filenames
OPTIMIZER_FILENAME = "optimizer.safetensors"


def get_model_filename(model_id: str) -> str:
    """Get model checkpoint filename for a specific model.

    Args:
        model_id: Model identifier (e.g., 'pieces', 'arrows')

    Returns:
        Model filename (e.g., 'pieces.safetensors', 'arrows.safetensors')
    """
    return f"{model_id}.safetensors"


# Output filenames
TRAINING_CURVES_FILENAME = "training_curves.png"
AUGMENTATION_EXAMPLE_FILENAME = "augmentation_example.png"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
PER_CLASS_ACCURACY_FILENAME = "per_class_accuracy.png"
TEST_CONFUSION_MATRIX_FILENAME = "test_confusion_matrix.png"
TEST_PER_CLASS_ACCURACY_FILENAME = "test_per_class_accuracy.png"
TEST_SUMMARY_FILENAME = "test_summary.json"
MISCLASSIFIED_DIR = "misclassified_images"
MAX_MISCLASSIFIED_IMAGES = 512

# Model configurations
MODEL_CONFIGS = {
    "pieces": {
        "num_classes": 13,
        "class_names": [
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
        ],
        "description": "Chess piece classifier (12 pieces + empty square)",
    },
    "arrows": {
        "num_classes": 49,
        "class_names": [
            "corner-E-S",
            "corner-N-E",
            "corner-S-W",
            "corner-W-N",
            "head-E",
            "head-ENE",
            "head-ESE",
            "head-N",
            "head-NE",
            "head-NNE",
            "head-NNW",
            "head-NW",
            "head-S",
            "head-SE",
            "head-SSE",
            "head-SSW",
            "head-SW",
            "head-W",
            "head-WNW",
            "head-WSW",
            "middle-E-NNE",
            "middle-E-SSE",
            "middle-E-W",
            "middle-N-ENE",
            "middle-N-S",
            "middle-N-WNW",
            "middle-S-ESE",
            "middle-S-WSW",
            "middle-SE-NW",
            "middle-SW-NE",
            "middle-W-NNW",
            "middle-W-SSW",
            "tail-E",
            "tail-ENE",
            "tail-ESE",
            "tail-N",
            "tail-NE",
            "tail-NNE",
            "tail-NNW",
            "tail-NW",
            "tail-S",
            "tail-SE",
            "tail-SSE",
            "tail-SSW",
            "tail-SW",
            "tail-W",
            "tail-WNW",
            "tail-WSW",
            "xx",
        ],
        "description": "Chess square arrow overlay classifier (48 arrow types + empty)",
    },
    # Future models can be added here:
    # "board": {
    #     "num_classes": 64,
    #     "class_names": [...],
    #     "description": "Full board state classifier",
    # },
}


def get_model_config(model_id: str) -> dict:
    """Get configuration for a specific model.

    Args:
        model_id: Model identifier (e.g., 'pieces')

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If model_id is not found
    """
    if model_id not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        msg = f"Unknown model_id: {model_id}. Available: {available}"
        raise ValueError(msg)
    return MODEL_CONFIGS[model_id]
