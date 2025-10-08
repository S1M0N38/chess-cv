"""Upload trained models to Hugging Face Hub."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from huggingface_hub import HfApi, create_repo

from chess_cv.constants import get_model_filename

__all__ = ["upload_to_hub"]


def upload_to_hub(
    model_id: str,
    repo_id: str,
    checkpoint_dir: Path | None = None,
    readme_path: Optional[Path] = None,
    commit_message: str = "Upload trained model",
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload trained model and artifacts to Hugging Face Hub.

    Args:
        model_id: Model identifier (e.g., 'pieces')
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
    from chess_cv.constants import get_checkpoint_dir, get_model_config

    # Get model configuration
    model_config = get_model_config(model_id)

    # Set default checkpoint_dir if not provided
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir(model_id)

    # Validate repo_id format
    if "/" not in repo_id:
        msg = f"Invalid repo_id format: {repo_id}. Expected 'username/repo-name'"
        raise ValueError(msg)

    # Validate required files exist
    checkpoint_dir = Path(checkpoint_dir)
    model_file = checkpoint_dir / get_model_filename(model_id)

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
        shutil.copy2(model_file, tmpdir / get_model_filename(model_id))

        # Copy README (model card)
        print(f"  - Copying README: {readme_path.name}")
        shutil.copy2(readme_path, tmpdir / "README.md")

        # Create a model config file with metadata
        config_data = {
            "model_id": model_id,
            "architecture": "SimpleCNN",
            "num_classes": model_config["num_classes"],
            "input_size": [32, 32, 3],
            "num_parameters": 156000,
            "framework": "mlx",
            "task": "image-classification",
            "classes": model_config["class_names"],
        }
        print("  - Creating config.json")
        (tmpdir / "config.json").write_text(json.dumps(config_data, indent=2))

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
