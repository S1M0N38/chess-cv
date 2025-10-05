"""Upload trained models to Hugging Face Hub."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from huggingface_hub import HfApi, create_repo

from chess_cv.constants import BEST_MODEL_FILENAME, DEFAULT_CHECKPOINT_DIR

__all__ = ["upload_to_hub", "main"]


def upload_to_hub(
    repo_id: str,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    readme_path: Optional[Path] = None,
    commit_message: str = "Upload trained model",
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload trained model and artifacts to Hugging Face Hub.

    Args:
        repo_id: Repository ID on Hugging Face Hub (format: "username/repo-name")
        checkpoint_dir: Directory containing model checkpoints
        readme_path: Path to model card README (defaults to docs/README_hf.md)
        commit_message: Commit message for the upload
        private: Whether to create a private repository
        token: Hugging Face API token (if not provided, uses cached token)

    Returns:
        URL of the uploaded repository

    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If repo_id format is invalid
    """
    # Validate repo_id format
    if "/" not in repo_id:
        msg = f"Invalid repo_id format: {repo_id}. Expected 'username/repo-name'"
        raise ValueError(msg)

    # Validate required files exist
    checkpoint_dir = Path(checkpoint_dir)
    model_file = checkpoint_dir / BEST_MODEL_FILENAME

    if not model_file.exists():
        msg = f"Model file not found: {model_file}"
        raise FileNotFoundError(msg)

    # Set default README path
    if readme_path is None:
        readme_path = Path(__file__).parent.parent.parent / "docs" / "README_hf.md"
    else:
        readme_path = Path(readme_path)

    if not readme_path.exists():
        msg = f"README file not found: {readme_path}"
        raise FileNotFoundError(msg)

    # Initialize Hugging Face API
    api = HfApi(token=token)

    # Create repository if it doesn't exist
    print(f"Creating repository: {repo_id}")
    repo_url = create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True,
        repo_type="model",
    )
    print(f"Repository URL: {repo_url}")

    # Prepare files for upload in a temporary directory
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("Preparing files for upload...")

        # Copy model weights
        print(f"  - Copying model: {model_file.name}")
        shutil.copy2(model_file, tmpdir / BEST_MODEL_FILENAME)

        # Copy README (model card)
        print(f"  - Copying README: {readme_path.name}")
        shutil.copy2(readme_path, tmpdir / "README.md")

        # Create a model config file with metadata
        config_content = """{
  "architecture": "SimpleCNN",
  "num_classes": 13,
  "input_size": [32, 32, 3],
  "num_parameters": 156000,
  "framework": "mlx",
  "task": "image-classification",
  "classes": [
    "bB", "bK", "bN", "bP", "bQ", "bR",
    "wB", "wK", "wN", "wP", "wQ", "wR",
    "xx"
  ]
}
"""
        print("  - Creating config.json")
        (tmpdir / "config.json").write_text(config_content)

        # Upload all files to the Hub
        print(f"\nUploading to {repo_id}...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(tmpdir),
            commit_message=commit_message,
            repo_type="model",
        )

    print(f"\n✅ Successfully uploaded model to: {repo_url}")
    return repo_url


def main() -> None:
    """Main entry point for the upload script."""
    parser = argparse.ArgumentParser(
        description="Upload trained chess-cv model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with default settings
  python -m chess_cv.upload --repo-id username/chess-cv

  # Upload with custom commit message
  python -m chess_cv.upload --repo-id username/chess-cv --message "feat: improved model v2"

  # Upload to private repository
  python -m chess_cv.upload --repo-id username/chess-cv --private

  # Specify custom paths
  python -m chess_cv.upload --repo-id username/chess-cv \\
    --checkpoint-dir ./my-checkpoints \\
    --readme docs/custom_README.md
        """,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face Hub (format: 'username/repo-name')",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory containing model checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )

    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Path to model card README (default: docs/README_hf.md)",
    )

    parser.add_argument(
        "--message",
        type=str,
        default="feat: upload new model version",
        help="Commit message for the upload (default: 'feat: upload new model version')",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (if not provided, uses cached token from 'hf login')",
    )

    args = parser.parse_args()

    try:
        upload_to_hub(
            repo_id=args.repo_id,
            checkpoint_dir=args.checkpoint_dir,
            readme_path=args.readme,
            commit_message=args.message,
            private=args.private,
            token=args.token,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
