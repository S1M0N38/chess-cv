"""CNN-based chess piece classifier using MLX for Apple Silicon."""

__version__ = "0.2.1"

__all__ = ["__version__"]


def main() -> None:
    """Main entry point for chess-cv CLI."""
    print(f"Chess CV v{__version__}")
    print("Available commands:")
    print("  python -m chess_cv.preprocessing  # Generate training data")
    print("  python -m chess_cv.train          # Train the model")
    print("  python -m chess_cv.test           # Evaluate the model")
    print("  python -m chess_cv.upload         # Upload model to Hugging Face Hub")
