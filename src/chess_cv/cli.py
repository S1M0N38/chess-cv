"""Command-line interface for chess-cv."""

import click
from pathlib import Path

from .constants import (
    BEST_MODEL_FILENAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_DIR,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_DIR,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_DIR,
    DEFAULT_VAL_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
)


@click.group()
@click.version_option()
def cli():
    """Chess-CV: CNN-based chess piece classifier using MLX."""
    pass


@cli.command()
@click.option(
    "--train-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_TRAIN_DIR,
    help=f"Training data output directory (default: {DEFAULT_TRAIN_DIR})",
)
@click.option(
    "--val-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_VAL_DIR,
    help=f"Validation data output directory (default: {DEFAULT_VAL_DIR})",
)
@click.option(
    "--test-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_TEST_DIR,
    help=f"Test data output directory (default: {DEFAULT_TEST_DIR})",
)
@click.option(
    "--train-ratio",
    type=float,
    default=DEFAULT_TRAIN_RATIO,
    help=f"Training data ratio (default: {DEFAULT_TRAIN_RATIO})",
)
@click.option(
    "--val-ratio",
    type=float,
    default=DEFAULT_VAL_RATIO,
    help=f"Validation data ratio (default: {DEFAULT_VAL_RATIO})",
)
@click.option(
    "--test-ratio",
    type=float,
    default=DEFAULT_TEST_RATIO,
    help=f"Test data ratio (default: {DEFAULT_TEST_RATIO})",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_RANDOM_SEED,
    help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})",
)
def preprocessing(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    """Generate train/validate/test sets from board-piece combinations."""
    from .preprocessing import generate_split_data

    generate_split_data(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    click.echo("\n✓ Data generation complete!")


@cli.command()
@click.option(
    "--train-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_TRAIN_DIR,
    help=f"Training data directory (default: {DEFAULT_TRAIN_DIR})",
)
@click.option(
    "--val-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_VAL_DIR,
    help=f"Validation data directory (default: {DEFAULT_VAL_DIR})",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_CHECKPOINT_DIR,
    help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
)
@click.option(
    "--learning-rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
    help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
)
@click.option(
    "--weight-decay",
    type=float,
    default=DEFAULT_WEIGHT_DECAY,
    help=f"Weight decay (default: {DEFAULT_WEIGHT_DECAY})",
)
@click.option(
    "--num-epochs",
    type=int,
    default=DEFAULT_NUM_EPOCHS,
    help=f"Number of epochs (default: {DEFAULT_NUM_EPOCHS})",
)
@click.option(
    "--patience",
    type=int,
    default=DEFAULT_PATIENCE,
    help=f"Early stopping patience (default: {DEFAULT_PATIENCE})",
)
@click.option(
    "--image-size",
    type=int,
    default=DEFAULT_IMAGE_SIZE,
    help=f"Image size (default: {DEFAULT_IMAGE_SIZE})",
)
@click.option(
    "--num-workers",
    type=int,
    default=DEFAULT_NUM_WORKERS,
    help=f"Number of data loading workers (default: {DEFAULT_NUM_WORKERS})",
)
@click.option(
    "--wandb",
    is_flag=True,
    help="Enable Weights & Biases logging (disables matplotlib visualization)",
)
def train(
    train_dir: Path,
    val_dir: Path,
    checkpoint_dir: Path,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    wandb: bool,
):
    """Train chess piece classification model."""
    from .train import train as train_model

    train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        use_wandb=wandb,
    )


@cli.command()
@click.option(
    "--test-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_TEST_DIR,
    help=f"Test data directory (default: {DEFAULT_TEST_DIR})",
)
@click.option(
    "--train-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_TRAIN_DIR,
    help=f"Training data directory for label map (default: {DEFAULT_TRAIN_DIR})",
)
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    default=DEFAULT_CHECKPOINT_DIR / BEST_MODEL_FILENAME,
    help=f"Model checkpoint path (default: {DEFAULT_CHECKPOINT_DIR / BEST_MODEL_FILENAME})",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
)
@click.option(
    "--image-size",
    type=int,
    default=DEFAULT_IMAGE_SIZE,
    help=f"Image size (default: {DEFAULT_IMAGE_SIZE})",
)
@click.option(
    "--num-workers",
    type=int,
    default=DEFAULT_NUM_WORKERS,
    help=f"Number of data loading workers (default: {DEFAULT_NUM_WORKERS})",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
)
@click.option(
    "--wandb",
    is_flag=True,
    help="Enable Weights & Biases logging (disables matplotlib visualization)",
)
@click.option(
    "--hf-test-dir",
    type=str,
    default=None,
    help="HuggingFace dataset ID (e.g., 'S1M0N38/chess-cv-openboard'). If provided, --test-dir is ignored.",
)
def test(
    test_dir: Path,
    train_dir: Path,
    checkpoint: Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
    output_dir: Path,
    wandb: bool,
    hf_test_dir: str | None,
):
    """Test and evaluate trained chess piece classification model."""
    from .test import test as test_model

    test_model(
        test_dir=test_dir,
        train_dir=train_dir,
        checkpoint_path=checkpoint,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        output_dir=output_dir,
        use_wandb=wandb,
        hf_test_dir=hf_test_dir,
    )


@cli.command()
@click.option(
    "--repo-id",
    type=str,
    required=True,
    help="Repository ID on Hugging Face Hub (format: 'username/repo-name')",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_CHECKPOINT_DIR,
    help=f"Directory containing model checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
)
@click.option(
    "--readme",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to model card README (default: docs/README_hf.md)",
)
@click.option(
    "--message",
    type=str,
    default="feat: upload new model version",
    help="Commit message for the upload (default: 'feat: upload new model version')",
)
@click.option(
    "--private",
    is_flag=True,
    help="Create a private repository",
)
@click.option(
    "--token",
    type=str,
    default=None,
    help="Hugging Face API token (if not provided, uses cached token from 'hf login')",
)
def upload(
    repo_id: str,
    checkpoint_dir: Path,
    readme: Path | None,
    message: str,
    private: bool,
    token: str | None,
):
    """Upload trained chess-cv model to Hugging Face Hub.
    
    Examples:
    
      # Upload with default settings
      chess-cv upload --repo-id username/chess-cv
    
      # Upload with custom commit message
      chess-cv upload --repo-id username/chess-cv --message "feat: improved model v2"
    
      # Upload to private repository
      chess-cv upload --repo-id username/chess-cv --private
    
      # Specify custom paths
      chess-cv upload --repo-id username/chess-cv \\
        --checkpoint-dir ./my-checkpoints \\
        --readme docs/custom_README.md
    """
    from .upload import upload_to_hub

    try:
        upload_to_hub(
            repo_id=repo_id,
            checkpoint_dir=checkpoint_dir,
            readme_path=readme,
            commit_message=message,
            private=private,
            token=token,
        )
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        raise click.Abort() from e
