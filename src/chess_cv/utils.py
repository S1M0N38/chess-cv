"""Utility functions for loading models and weights."""

from pathlib import Path

import mlx.core as mx

from .constants import get_model_config
from .model import SimpleCNN
from .weights import BUNDLED_WEIGHTS

__all__ = ["get_bundled_weight_path", "load_bundled_model"]


def get_bundled_weight_path(model_id: str) -> Path:
    """Get the path to bundled model weights.

    Args:
        model_id: Model identifier (e.g., 'pieces', 'arrows')

    Returns:
        Path to the bundled .safetensors file

    Raises:
        ValueError: If model_id is not available in bundled weights
    """
    if model_id not in BUNDLED_WEIGHTS:
        available = list(BUNDLED_WEIGHTS.keys())
        msg = f"No bundled weights for model_id: {model_id}. Available: {available}"
        raise ValueError(msg)

    weight_path = BUNDLED_WEIGHTS[model_id]
    if not weight_path.exists():
        msg = f"Bundled weights not found at {weight_path}"
        raise FileNotFoundError(msg)

    return weight_path


def load_bundled_model(model_id: str) -> SimpleCNN:
    """Load a pre-trained model with bundled weights.

    This is a convenience function that creates a model and loads the
    bundled weights in one step.

    Args:
        model_id: Model identifier (e.g., 'pieces', 'arrows')

    Returns:
        SimpleCNN model with pre-trained weights loaded

    Raises:
        ValueError: If model_id is not available

    Example:
        >>> from chess_cv.utils import load_bundled_model
        >>> model = load_bundled_model('pieces')
        >>> predictions = model(image_tensor)
    """
    # Get model configuration
    config = get_model_config(model_id)
    num_classes = config["num_classes"]

    # Create model
    model = SimpleCNN(num_classes=num_classes)

    # Load bundled weights
    weight_path = get_bundled_weight_path(model_id)
    weights = mx.load(str(weight_path))
    model.load_weights(list(weights.items()))  # type: ignore[attr-defined]

    return model
