"""Constants and default configuration values for chess-cv."""

from pathlib import Path

# Data paths
DEFAULT_DATA_DIR = Path("data")
DEFAULT_TRAIN_DIR = DEFAULT_DATA_DIR / "train"
DEFAULT_VAL_DIR = DEFAULT_DATA_DIR / "validate"
DEFAULT_TEST_DIR = DEFAULT_DATA_DIR / "test"
DEFAULT_ALL_DIR = DEFAULT_DATA_DIR / "all"

# Output paths
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("outputs")

# Model parameters
DEFAULT_NUM_CLASSES = 13
DEFAULT_IMAGE_SIZE = 32
DEFAULT_DROPOUT = 0.5

# Training hyperparameters
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_NUM_EPOCHS = 200
DEFAULT_PATIENCE = 999999  # Effectively disabled

# Data loading
DEFAULT_NUM_WORKERS = 8

# Data splitting ratios
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42

# Augmentation parameters
AUGMENTATION_SCALE_MIN = 0.8
AUGMENTATION_SCALE_MAX = 1.0
AUGMENTATION_BRIGHTNESS = 0.2
AUGMENTATION_CONTRAST = 0.2
AUGMENTATION_SATURATION = 0.2
AUGMENTATION_ROTATION_DEGREES = 5
AUGMENTATION_NOISE_MEAN = 0.0
AUGMENTATION_NOISE_STD = 0.05

# File patterns
IMAGE_PATTERN = "**/*.png"

# Checkpoint filenames
BEST_MODEL_FILENAME = "best_model.safetensors"
OPTIMIZER_FILENAME = "optimizer.safetensors"

# Output filenames
TRAINING_CURVES_FILENAME = "training_curves.png"
AUGMENTATION_EXAMPLE_FILENAME = "augmentation_example.png"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
PER_CLASS_ACCURACY_FILENAME = "per_class_accuracy.png"
TEST_CONFUSION_MATRIX_FILENAME = "test_confusion_matrix.png"
TEST_PER_CLASS_ACCURACY_FILENAME = "test_per_class_accuracy.png"
MISCLASSIFIED_DIR = "misclassified_images"
