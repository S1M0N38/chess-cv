"""Pre-trained model weights bundled with the package."""

__all__ = ["BUNDLED_WEIGHTS"]

from pathlib import Path

# Directory containing bundled model weights
WEIGHTS_DIR = Path(__file__).parent

# Available bundled models
BUNDLED_WEIGHTS = {
    "pieces": WEIGHTS_DIR / "pieces.safetensors",
    "arrows": WEIGHTS_DIR / "arrows.safetensors",
    "snap": WEIGHTS_DIR / "snap.safetensors",
}
