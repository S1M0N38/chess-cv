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
    return Path("src/chess_cv/weights")


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
LOG_VALIDATE_EVERY_N_STEPS = 2000  # Run full validation every N batches

# Data splitting ratios
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42

# Augmentation resource directories
DEFAULT_ARROW_DIR = DEFAULT_DATA_DIR / "arrows"
DEFAULT_HIGHLIGHT_DIR = DEFAULT_DATA_DIR / "highlights"
DEFAULT_MOUSE_DIR = DEFAULT_DATA_DIR / "mouse"
DEFAULT_MOVE_DIR = DEFAULT_DATA_DIR / "moves"

# Model-specific augmentation configurations
AUGMENTATION_CONFIGS = {
    "pieces": {
        # ===========================================
        # GEOMETRIC TRANSFORMATIONS (Steps 1-4)
        # ===========================================
        # Step 1: Padding - Create rotation space by expanding the canvas
        # Original 32x32 → 64x64 (16 pixels padding on each side)
        "padding": 16,
        "padding_mode": "edge",
        # Step 2: Rotation - Apply random rotation to the padded image
        # ±10 degrees provides sufficient variation while preserving piece recognizability
        "rotation_degrees": 10,
        # Step 3: Center Crop - Remove black bands created by rotation
        # Formula: 64 - (ceil(tan(10°) * 64) * 2) = 64 - 24 = 40
        # This ensures no black borders remain after rotation
        "center_crop_size": 40,
        # Step 4: Random Resized Crop - Simulate different distances and positions
        # Final resize to target model input size (32x32)
        "final_size": 32,
        # Scale range controls zoom: (0.54, 0.74) = 0.64 ± 0.1 (±16% zoom variation)
        # Base scale 0.64 = (32/40)² = area ratio for translation without zoom
        "resized_crop_scale": (0.54, 0.74),
        # Ratio range controls aspect ratio changes: (0.9, 1.1) = ±10% stretch
        "resized_crop_ratio": (0.9, 1.1),
        # ===========================================
        # OVERLAY AUGMENTATIONS (Step 5)
        # ===========================================
        # Arrow overlay - Simulate arrow graphics on pieces
        "arrow_probability": 0.80,
        # Highlight overlay - Simulate square highlighting effects
        "highlight_probability": 0.25,
        # Move indicator overlay - Simulate move indicators on pieces
        "move_probability": 0.5,
        # Mouse cursor overlay - Simulate cursor interaction with pieces
        "mouse_probability": 0.90,
        # ===========================================
        # MOUSE CURSOR SPECIFIC PARAMETERS
        # ===========================================
        # Mouse cursor geometric transformations (separate from piece transforms)
        "mouse_padding": 134,  # Pad to create rotation space: 32 → 300 (x = 134)
        "mouse_rotation_degrees": 5,  # Smaller rotation for cursor stability
        # Center crop calculation: x - (ceil(tan(5°) * x) * 2) = 256 - 10 = 246
        "mouse_center_crop_size": 246,  # Remove rotation artifacts
        "mouse_final_size": 32,  # Resize to match piece size
        # Scale range makes cursor appear smaller: 20-30% of canvas area
        "mouse_scale_range": (0.20, 0.30),
        # Ratio range allows cursor shape distortion: ±20% stretch
        "mouse_ratio_range": (0.8, 1.2),
        # ===========================================
        # SPATIAL TRANSFORMATIONS (Step 6)
        # ===========================================
        # Horizontal flip - Mirror pieces horizontally (valid for chess pieces)
        "horizontal_flip": True,
        "horizontal_flip_prob": 0.5,  # 50% chance of horizontal flip
        # ===========================================
        # COLOR AUGMENTATIONS (Steps 7-8)
        # ===========================================
        # Brightness variation - Simulate different lighting conditions
        "brightness": 0.15,  # ±15% brightness variation
        # Contrast variation - Simulate different display contrast settings
        "contrast": 0.2,  # ±20% contrast variation
        # Saturation variation - Simulate different color saturation levels
        "saturation": 0.2,  # ±20% saturation variation
        # Hue variation - Simulate slight color temperature changes
        "hue": 0.2,  # ±20% hue rotation (affects white/black pieces)
        # ===========================================
        # NOISE AUGMENTATION (Step 9)
        # ===========================================
        # Gaussian noise - Simulate sensor noise and compression artifacts
        "noise_mean": 0.0,  # Center noise distribution at 0 (no bias)
        "noise_sigma": 0.05,  # Low noise level (5% of pixel range)
    },
    "arrows": {
        # ===========================================
        # OVERLAY AUGMENTATIONS
        # ===========================================
        # Arrow overlay - Disabled for arrow model (arrows are the target, not augmentation)
        "arrow_probability": 0.0,
        # Highlight overlay - Simulate square highlighting effects
        "highlight_probability": 0.25,
        # Move indicator overlay - Simulate move indicators on arrows
        "move_probability": 0.5,
        # ===========================================
        # GEOMETRIC TRANSFORMATIONS
        # ===========================================
        # Scale range - Control zoom level for arrow detection
        # Arrows can appear at different sizes depending on viewing distance
        "scale_min": 0.75,  # Minimum scale: 75% of original size
        "scale_max": 1.0,  # Maximum scale: 100% of original size (no enlargement)
        # Horizontal flip - Disabled for directional arrows
        # Arrows have specific directions that would be invalidated by flipping
        "horizontal_flip": False,
        # Rotation - Small rotations to account for slight camera angles
        # Limited to ±2 degrees to preserve arrow direction semantics
        "rotation_degrees": 2,
        # ===========================================
        # COLOR AUGMENTATIONS
        # ===========================================
        # Brightness variation - Simulate different lighting conditions
        # Slightly higher than pieces model as arrows may be viewed in varied lighting
        "brightness": 0.20,  # ±20% brightness variation
        # Contrast variation - Simulate different display contrast settings
        "contrast": 0.20,  # ±20% contrast variation
        # Saturation variation - Simulate different color saturation levels
        "saturation": 0.20,  # ±20% saturation variation
        # Hue variation - Simulate slight color temperature changes
        "hue": 0.2,  # ±20% hue rotation
        # ===========================================
        # NOISE AUGMENTATION
        # ===========================================
        # Gaussian noise - Simulate sensor noise and compression artifacts
        # Higher noise level than pieces model as arrows may be captured in varied conditions
        "noise_mean": 0.0,  # Center noise distribution at 0 (no bias)
        "noise_sigma": 0.10,  # Higher noise level (10% of pixel range)
    },
    "snap": {
        # ===========================================
        # OVERLAY AUGMENTATIONS
        # ===========================================
        # Arrow overlay - Simulate arrow graphics on pieces
        "arrow_probability": 0.50,
        # Highlight overlay - Simulate square highlighting effects
        "highlight_probability": 0.20,
        # Move indicator overlay - Simulate move indicators on pieces
        "move_probability": 0.50,
        # Mouse cursor overlay - Simulate cursor interaction with pieces
        "mouse_probability": 0.80,
        # ===========================================
        # MOUSE CURSOR SPECIFIC PARAMETERS
        # ===========================================
        # Mouse cursor geometric transformations (same as pieces model)
        "mouse_padding": 134,  # Pad to create rotation space: 32 → 300 (x = 134)
        "mouse_rotation_degrees": 5,  # Smaller rotation for cursor stability
        # Center crop calculation: x - (ceil(tan(5°) * x) * 2) = 256 - 10 = 246
        "mouse_center_crop_size": 246,  # Remove rotation artifacts
        "mouse_final_size": 32,  # Resize to match piece size
        # Scale range makes cursor appear smaller: 20-30% of canvas area
        "mouse_scale_range": (0.20, 0.30),
        # Ratio range allows cursor shape distortion: ±20% stretch
        "mouse_ratio_range": (0.8, 1.2),
        # ===========================================
        # SPATIAL TRANSFORMATIONS
        # ===========================================
        # Horizontal flip - Mirror pieces horizontally (valid for chess pieces)
        "horizontal_flip": True,
        "horizontal_flip_prob": 0.5,  # 50% chance of horizontal flip
        # ===========================================
        # COLOR AUGMENTATIONS (same as pieces model)
        # ===========================================
        # Brightness variation - Simulate different lighting conditions
        "brightness": 0.15,  # ±15% brightness variation
        # Contrast variation - Simulate different display contrast settings
        "contrast": 0.2,  # ±20% contrast variation
        # Saturation variation - Simulate different color saturation levels
        "saturation": 0.2,  # ±20% saturation variation
        # Hue variation - Simulate slight color temperature changes
        "hue": 0.2,  # ±20% hue rotation (affects white/black pieces)
        # ===========================================
        # NOISE AUGMENTATION (same as pieces model)
        # ===========================================
        # Gaussian noise - Simulate sensor noise and compression artifacts
        "noise_mean": 0.0,  # Center noise distribution at 0 (no bias)
        "noise_sigma": 0.05,  # Low noise level (5% of pixel range)
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
    "snap": {
        "num_classes": 2,
        "class_names": [
            "ok",  # Centered or slightly off-centered pieces, or empty squares
            "bad",  # Significantly off-centered pieces
        ],
        "description": "Chess piece centering classifier (centered vs off-centered)",
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
