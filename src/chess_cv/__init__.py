"""CNN-based chess piece classifier using MLX for Apple Silicon."""

from .utils import get_bundled_weight_path, load_bundled_model

__version__ = "0.7.2"

__all__ = ["__version__", "main", "load_bundled_model", "get_bundled_weight_path"]


def main() -> None:
    """Main entry point for chess-cv CLI."""
    from .cli import cli

    cli()
