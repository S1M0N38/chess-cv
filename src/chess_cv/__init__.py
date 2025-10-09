"""CNN-based chess piece classifier using MLX for Apple Silicon."""

__version__ = "0.3.0"

__all__ = ["__version__", "main"]


def main() -> None:
    """Main entry point for chess-cv CLI."""
    from .cli import cli

    cli()
