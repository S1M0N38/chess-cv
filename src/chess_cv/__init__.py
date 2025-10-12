"""CNN-based chess piece classifier using MLX for Apple Silicon."""

__version__ = "0.3.0"

__all__ = ["__version__", "main", "load_bundled_model", "get_bundled_weight_path"]


def main() -> None:
    """Main entry point for chess-cv CLI."""
    from .cli import cli

    cli()


# Convenience imports for users
from .utils import get_bundled_weight_path, load_bundled_model
