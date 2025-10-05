"""Training script for chess piece classification."""

import argparse
import os
from pathlib import Path

__all__ = ["train", "main"]

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .constants import (
    AUGMENTATION_BRIGHTNESS,
    AUGMENTATION_CONTRAST,
    AUGMENTATION_NOISE_MEAN,
    AUGMENTATION_NOISE_STD,
    AUGMENTATION_ROTATION_DEGREES,
    AUGMENTATION_SATURATION,
    AUGMENTATION_SCALE_MAX,
    AUGMENTATION_SCALE_MIN,
    BEST_MODEL_FILENAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PATIENCE,
    DEFAULT_TRAIN_DIR,
    DEFAULT_VAL_DIR,
    DEFAULT_WEIGHT_DECAY,
    OPTIMIZER_FILENAME,
    TRAINING_CURVES_FILENAME,
)
from .data import (
    AddGaussianNoise,
    ChessPiecesDataset,
    collate_fn,
    get_all_labels,
    get_image_files,
    get_label_map,
)
from .model import create_model
from .visualize import TrainingVisualizer
from .wandb_utils import WandbLogger


def loss_fn(model: nn.Module, images: mx.array, labels: mx.array) -> mx.array:
    """Compute cross-entropy loss."""
    logits = model(images)
    loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
    return loss


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    loss_and_grad_fn,
) -> tuple[float, float]:
    """Train for one epoch."""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_images, batch_labels in pbar:
        loss, grads = loss_and_grad_fn(model, batch_images, batch_labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        logits = model(batch_images)
        predictions = mx.argmax(logits, axis=1)
        correct = mx.sum(predictions == batch_labels)

        batch_size = len(batch_labels)
        total_loss += loss.item() * batch_size
        total_correct += correct.item()
        total_samples += batch_size

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct.item() / batch_size:.4f}",
            }
        )

    return total_loss / total_samples, total_correct / total_samples


def validate_epoch(model: nn.Module, val_loader: DataLoader) -> tuple[float, float]:
    """Validate for one epoch."""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for batch_images, batch_labels in pbar:
        loss = loss_fn(model, batch_images, batch_labels)
        logits = model(batch_images)
        predictions = mx.argmax(logits, axis=1)
        correct = mx.sum(predictions == batch_labels)

        batch_size = len(batch_labels)
        total_loss += loss.item() * batch_size
        total_correct += correct.item()
        total_samples += batch_size

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct.item() / batch_size:.4f}",
            }
        )

    return total_loss / total_samples, total_correct / total_samples


def train(
    train_dir: Path | str = DEFAULT_TRAIN_DIR,
    val_dir: Path | str = DEFAULT_VAL_DIR,
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    use_wandb: bool = False,
) -> None:
    """Train the model."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb logger
    wandb_logger = WandbLogger(enabled=use_wandb)
    if use_wandb:
        wandb_logger.init(
            project="chess-cv-training",
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "patience": patience,
                "image_size": image_size,
                "num_workers": num_workers,
                "train_dir": str(train_dir),
                "val_dir": str(val_dir),
            },
        )

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Get image files and create label map from training data
    train_files = get_image_files(str(train_dir))
    val_files = get_image_files(str(val_dir))
    all_labels = get_all_labels(train_files)
    label_map = get_label_map(all_labels)
    num_classes = len(label_map)

    # Define augmentations
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(AUGMENTATION_SCALE_MIN, AUGMENTATION_SCALE_MAX),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=AUGMENTATION_BRIGHTNESS,
                contrast=AUGMENTATION_CONTRAST,
                saturation=AUGMENTATION_SATURATION,
            ),
            transforms.RandomRotation(degrees=AUGMENTATION_ROTATION_DEGREES),
            AddGaussianNoise(mean=AUGMENTATION_NOISE_MEAN, std=AUGMENTATION_NOISE_STD),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
        ]
    )

    # Create datasets
    train_dataset = ChessPiecesDataset(
        train_files, label_map, transform=train_transforms
    )
    val_dataset = ChessPiecesDataset(val_files, label_map, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    print(f"\nTraining samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes:  {num_classes}")
    print(f"Batch size:         {batch_size}")
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Visualization of augmentation
    if train_files and len(train_files) >= 8:
        num_examples = 8
        import numpy as np

        if use_wandb:
            # Log augmentation examples to wandb
            for i in range(num_examples):
                original_image = Image.open(train_files[i]).convert("RGB")
                resized_original = val_transforms(original_image)
                augmented_image = train_transforms(original_image)

                # Log original
                wandb_logger.log_image(
                    f"augmentation/original_{i + 1}",
                    np.array(resized_original),
                    caption=f"Original {i + 1}",
                    step=0,
                    commit=False,
                )

                # Log augmented version
                is_last = i == num_examples - 1
                wandb_logger.log_image(
                    f"augmentation/augmented_{i + 1}",
                    np.array(augmented_image),
                    caption=f"Augmented {i + 1}",
                    step=0,
                    commit=is_last,
                )
            print(
                f"\nLogged augmentation examples to wandb ({num_examples} original + {num_examples} augmented)"
            )
        else:
            # Save to file when not using wandb
            # Create 8x2 grid: 8 rows, each with original and augmented
            fig, axes = plt.subplots(8, 2, figsize=(8, 24))

            for i in range(num_examples):
                original_image = Image.open(train_files[i]).convert("RGB")
                resized_original = val_transforms(original_image)
                augmented_image = train_transforms(original_image)

                # Show original
                axes[i, 0].imshow(resized_original)
                axes[i, 0].set_title(f"Original {i + 1}", fontsize=10)
                axes[i, 0].axis("off")

                # Show augmented
                axes[i, 1].imshow(augmented_image)
                axes[i, 1].set_title(f"Augmented {i + 1}", fontsize=10)
                axes[i, 1].axis("off")

            plt.tight_layout()
            output_dir = DEFAULT_OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            from .constants import AUGMENTATION_EXAMPLE_FILENAME

            output_path = os.path.join(output_dir, AUGMENTATION_EXAMPLE_FILENAME)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(
                f"\nSaved augmentation examples to {output_path} ({num_examples} pairs)"
            )
            plt.close(fig)

    # Create model
    print("\n" + "=" * 60)
    print("MODEL")
    print("=" * 60)
    model = create_model(num_classes=num_classes)
    print(model)

    # Calculate total parameters
    param_list = tree_flatten(model.parameters())
    num_params = sum(v.size for _, v in param_list)  # type: ignore[attr-defined]
    print(f"\nTotal parameters: {num_params:,}")

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    print("\nOptimizer: AdamW")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay:  {weight_decay}")

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Only use TrainingVisualizer when not using wandb
    if not use_wandb:
        visualizer = TrainingVisualizer()

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", leave=True)
    for epoch in epoch_pbar:
        train_loss, train_acc = train_epoch(
            model, optimizer, train_loader, loss_and_grad_fn
        )
        val_loss, val_acc = validate_epoch(model, val_loader)

        # Update visualizer or log to wandb
        if use_wandb:
            wandb_logger.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "best_val_accuracy": best_val_acc,
                },
                step=epoch + 1,
            )
        else:
            visualizer.update(epoch + 1, train_loss, train_acc, val_loss, val_acc)

        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            model_path = checkpoint_dir / BEST_MODEL_FILENAME
            mx.save_safetensors(str(model_path), dict(tree_flatten(model.parameters())))
            optimizer_path = checkpoint_dir / OPTIMIZER_FILENAME
            mx.save_safetensors(
                str(optimizer_path), dict(tree_flatten(optimizer.state))
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Save visualizations or log model to wandb
    if use_wandb:
        # Log the best model as an artifact
        model_path = checkpoint_dir / BEST_MODEL_FILENAME
        if model_path.exists():
            wandb_logger.log_model(model_path, name="chess-cv-model", aliases=["best"])
            print("\nLogged best model to wandb")
    else:
        visualizer.save(TRAINING_CURVES_FILENAME)
        visualizer.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {checkpoint_dir / BEST_MODEL_FILENAME}")

    # Finish wandb run
    wandb_logger.finish()


def main() -> None:
    """Run training script with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train chess piece classification model"
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help=f"Training data directory (default: {DEFAULT_TRAIN_DIR})",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help=f"Validation data directory (default: {DEFAULT_VAL_DIR})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_NUM_EPOCHS})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience (default: {DEFAULT_PATIENCE})",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Image size (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of data loading workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (disables matplotlib visualization)",
    )

    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience=args.patience,
        image_size=args.image_size,
        num_workers=args.num_workers,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
