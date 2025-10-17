"""Command-line interface for chess-cv."""

from pathlib import Path

import click

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WEIGHT_DECAY,
    get_checkpoint_dir,
    get_model_config,
    get_model_filename,
    get_output_dir,
    get_test_dir,
    get_train_dir,
    get_val_dir,
)


@click.group()
@click.version_option()
def cli():
    """Chess-CV: CNN-based chess piece classifier using MLX."""
    pass


@cli.command()
@click.argument("model-id", type=str)
@click.option(
    "--train-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Training data output directory (default: data/splits/{model-id}/train)",
)
@click.option(
    "--val-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Validation data output directory (default: data/splits/{model-id}/validate)",
)
@click.option(
    "--test-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Test data output directory (default: data/splits/{model-id}/test)",
)
def preprocessing(
    model_id: str,
    train_dir: Path | None,
    val_dir: Path | None,
    test_dir: Path | None,
):
    """Generate train/validate/test sets from board-piece combinations.

    MODEL_ID: Model identifier (e.g., 'pieces')
    """
    from .preprocessing import generate_split_data

    # Validate model_id
    _ = get_model_config(model_id)

    # Set defaults based on model_id if not provided
    if train_dir is None:
        train_dir = get_train_dir(model_id)
    if val_dir is None:
        val_dir = get_val_dir(model_id)
    if test_dir is None:
        test_dir = get_test_dir(model_id)

    generate_split_data(
        model_id=model_id,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
    )
    click.echo("\n✓ Data generation complete!")


@cli.command()
@click.argument("model-id", type=str)
@click.option(
    "--train-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Training data directory (default: data/splits/{model-id}/train)",
)
@click.option(
    "--val-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Validation data directory (default: data/splits/{model-id}/validate)",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Checkpoint directory (default: src/chess_cv/weights)",
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
@click.option(
    "--sweep",
    is_flag=True,
    help="Run hyperparameter sweep with W&B (requires --wandb flag)",
)
def train(
    model_id: str,
    train_dir: Path | None,
    val_dir: Path | None,
    checkpoint_dir: Path | None,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    num_workers: int,
    wandb: bool,
    sweep: bool,
):
    """Train chess piece classification model.

    MODEL_ID: Model identifier (e.g., 'pieces')
    """
    # Validate model_id
    _ = get_model_config(model_id)

    # Handle sweep mode
    if sweep:
        if not wandb:
            raise click.UsageError(
                "--sweep requires --wandb flag. "
                "Use: chess-cv train MODEL_ID --sweep --wandb"
            )
        from .sweep import run_sweep

        run_sweep(model_id=model_id)
        return  # Exit after sweep completes

    # Normal training mode
    from .train import train as train_model

    # Set defaults based on model_id if not provided
    if train_dir is None:
        train_dir = get_train_dir(model_id)
    if val_dir is None:
        val_dir = get_val_dir(model_id)
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir(model_id)

    train_model(
        model_id=model_id,
        train_dir=train_dir,
        val_dir=val_dir,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        num_workers=num_workers,
        use_wandb=wandb,
    )


@cli.command()
@click.argument("model-id", type=str)
@click.option(
    "--test-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Test data directory (default: data/splits/{model-id}/test)",
)
@click.option(
    "--train-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Training data directory for label map (default: data/splits/{model-id}/train)",
)
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    default=None,
    help="Model checkpoint path (default: src/chess_cv/weights/{model-id}.safetensors)",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
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
    default=None,
    help="Output directory for results (default: outputs/{model-id})",
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
@click.option(
    "--concat-splits",
    is_flag=True,
    help="Concatenate all splits from HuggingFace dataset (only applicable with --hf-test-dir)",
)
def test(
    model_id: str,
    test_dir: Path | None,
    train_dir: Path | None,
    checkpoint: Path | None,
    batch_size: int,
    num_workers: int,
    output_dir: Path | None,
    wandb: bool,
    hf_test_dir: str | None,
    concat_splits: bool,
):
    """Test and evaluate trained chess piece classification model.

    MODEL_ID: Model identifier (e.g., 'pieces')
    """
    from .test import test as test_model

    # Validate model_id
    _ = get_model_config(model_id)

    # Set defaults based on model_id if not provided
    if test_dir is None:
        test_dir = get_test_dir(model_id)
    if train_dir is None:
        train_dir = get_train_dir(model_id)
    if checkpoint is None:
        checkpoint = get_checkpoint_dir(model_id) / get_model_filename(model_id)
    if output_dir is None:
        output_dir = get_output_dir(model_id)

    test_model(
        model_id=model_id,
        test_dir=test_dir,
        train_dir=train_dir,
        checkpoint_path=checkpoint,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
        use_wandb=wandb,
        hf_test_dir=hf_test_dir,
        concat_splits=concat_splits,
    )


@cli.command()
@click.argument("model-id", type=str)
@click.option(
    "--repo-id",
    type=str,
    default="S1M0N38/chess-cv",
    help="Repository ID on Hugging Face Hub (format: 'username/repo-name')",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing model checkpoints (default: src/chess_cv/weights)",
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
    model_id: str,
    repo_id: str,
    checkpoint_dir: Path | None,
    readme: Path | None,
    message: str,
    private: bool,
    token: str | None,
):
    """Upload trained chess-cv model to Hugging Face Hub.

    MODEL_ID: Model identifier (e.g., 'pieces')

    Examples:

      # Upload with default settings (to S1M0N38/chess-cv)
      chess-cv upload pieces

      # Upload to custom repository
      chess-cv upload pieces --repo-id username/chess-cv

      # Upload with custom commit message
      chess-cv upload pieces --repo-id username/chess-cv --message "feat: improved model v2"

      # Upload to private repository
      chess-cv upload pieces --repo-id username/chess-cv --private

      # Specify custom paths
      chess-cv upload pieces --repo-id username/chess-cv \\
        --checkpoint-dir ./my-checkpoints \\
        --readme docs/custom_README.md
    """
    from .upload import upload_to_hub

    # Validate model_id
    _ = get_model_config(model_id)

    # Set defaults based on model_id if not provided
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir(model_id)

    try:
        upload_to_hub(
            model_id=model_id,
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
