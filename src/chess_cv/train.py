"""Training script for chess piece classification."""

from pathlib import Path

__all__ = ["train"]

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import torch
from mlx.utils import tree_flatten
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from .constants import (
    AUGMENTATION_CONFIGS,
    DEFAULT_ARROW_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIGHLIGHT_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    LOG_TRAIN_EVERY_N_STEPS,
    LOG_VALIDATE_EVERY_N_STEPS,
    OPTIMIZER_FILENAME,
    TRAINING_CURVES_FILENAME,
    get_model_filename,
    get_output_dir,
)
from .data import (
    ChessPiecesDataset,
    RandomArrowOverlay,
    RandomHighlightOverlay,
    collate_fn,
    get_image_files,
    get_label_map_from_class_names,
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
    val_loader: DataLoader,
    loss_and_grad_fn,
    wandb_logger: WandbLogger,
    visualizer,
    epoch: int,
    global_step: int,
    log_every_n_steps: int,
    validate_every_n_steps: int,
    checkpoint_dir: Path,
    model_id: str,
    best_val_acc: float,
    best_val_loss: float,
    epochs_without_improvement: int,
) -> tuple[float, float, int, float, float, int]:
    """Train for one epoch with mid-epoch validation.

    Args:
        model: Model to train
        optimizer: Optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_and_grad_fn: Combined loss and gradient function
        wandb_logger: WandB logger instance
        visualizer: TrainingVisualizer instance or None
        epoch: Current epoch number (1-indexed)
        global_step: Global step counter across all epochs
        log_every_n_steps: Log training metrics every N steps
        validate_every_n_steps: Run validation every N steps
        checkpoint_dir: Directory to save checkpoints
        model_id: Model identifier for checkpoint naming
        best_val_acc: Best validation accuracy so far
        best_val_loss: Best validation loss so far
        epochs_without_improvement: Counter for early stopping

    Returns:
        Tuple of (average_loss, average_accuracy, updated_global_step,
                  updated_best_val_acc, updated_best_val_loss,
                  updated_epochs_without_improvement)
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (batch_images, batch_labels) in enumerate(pbar):
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

        # Increment global step
        global_step += 1

        # Log mid-epoch training metrics to wandb
        if (
            wandb_logger is not None
            and wandb_logger.enabled
            and global_step % log_every_n_steps == 0
        ):
            batch_acc = correct.item() / batch_size
            wandb_logger.log(
                {
                    "global_step": global_step,
                    "loss/train_step": loss.item(),
                    "accuracy/train_step": batch_acc,
                    "epoch": epoch,
                },
                step=global_step,
            )

        # Mid-epoch validation
        if global_step % validate_every_n_steps == 0:
            # Run validation on full validation set
            val_loss, val_acc = validate_epoch(model, val_loader, leave=False)

            # Log to wandb
            if wandb_logger is not None and wandb_logger.enabled:
                wandb_logger.log(
                    {
                        "global_step": global_step,
                        "loss/val_step": val_loss,
                        "accuracy/val_step": val_acc,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            # Log to local visualizer (if not using wandb)
            if visualizer is not None:
                # For mid-epoch, use fractional epoch number
                fractional_epoch = epoch + (batch_idx + 1) / len(train_loader)
                # Get current training metrics (running average)
                current_train_loss = (
                    total_loss / total_samples if total_samples > 0 else 0.0
                )
                current_train_acc = (
                    total_correct / total_samples if total_samples > 0 else 0.0
                )
                visualizer.update(
                    fractional_epoch,
                    current_train_loss,
                    current_train_acc,
                    val_loss,
                    val_acc,
                )

            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Check if this is the best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0

                # Save model checkpoint
                from mlx.utils import tree_flatten

                model_path = checkpoint_dir / get_model_filename(model_id)
                mx.save_safetensors(
                    str(model_path), dict(tree_flatten(model.parameters()))
                )
                optimizer_path = checkpoint_dir / OPTIMIZER_FILENAME
                mx.save_safetensors(
                    str(optimizer_path), dict(tree_flatten(optimizer.state))
                )
            else:
                epochs_without_improvement += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct.item() / batch_size:.4f}",
            }
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        global_step,
        best_val_acc,
        best_val_loss,
        epochs_without_improvement,
    )


def validate_epoch(
    model: nn.Module, val_loader: DataLoader, leave: bool = False
) -> tuple[float, float]:
    """Validate for one epoch.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        leave: Whether to leave the progress bar visible after completion

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(val_loader, desc="Validation", leave=leave)
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
    model_id: str,
    train_dir: Path | str | None = None,
    val_dir: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    use_wandb: bool = False,
) -> None:
    """Train the model.

    Args:
        model_id: Model identifier (e.g., 'pieces')
        train_dir: Training data directory
        val_dir: Validation data directory
        checkpoint_dir: Checkpoint directory for saving models
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        image_size: Image size for resizing
        num_workers: Number of data loading workers
        use_wandb: Enable Weights & Biases logging
    """
    from .constants import (
        get_checkpoint_dir,
        get_model_config,
        get_train_dir,
        get_val_dir,
    )

    # Only support pieces and arrows models
    if model_id not in ["pieces", "arrows"]:
        msg = f"Model '{model_id}' is not supported. Only 'pieces' and 'arrows' models are supported."
        raise ValueError(msg)

    # Get model configuration
    model_config = get_model_config(model_id)
    num_classes = model_config["num_classes"]
    class_names = model_config["class_names"]

    # Set default directories if not provided
    if train_dir is None:
        train_dir = get_train_dir(model_id)
    if val_dir is None:
        val_dir = get_val_dir(model_id)
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir(model_id)

    # Convert to Path objects
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    checkpoint_dir = Path(checkpoint_dir)

    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb logger
    wandb_logger = WandbLogger(enabled=use_wandb)
    if use_wandb:
        config = {
            "model_id": model_id,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }

        wandb_logger.init(
            project=f"chess-cv-{model_id}",
            config=config,
        )

        # Define custom metrics with step as x-axis
        wandb_logger.define_metrics()

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Create label map from model configuration (not from scanning directories)
    # This ensures consistency across all splits and avoids directory parsing issues
    label_map = get_label_map_from_class_names(class_names)

    # Get image files
    train_files = get_image_files(str(train_dir))
    val_files = get_image_files(str(val_dir))

    # Define augmentations using model-specific configuration
    aug_config = AUGMENTATION_CONFIGS[model_id]

    train_transform_list = []

    if model_id == "pieces":
        # Step 1: Expand canvas for rotation space
        train_transform_list.append(
            v2.Pad(
                padding=aug_config["padding"], padding_mode=aug_config["padding_mode"]
            )
        )

        # Step 2: Rotate with black fill (will be cropped out)
        train_transform_list.append(
            v2.RandomRotation(degrees=aug_config["rotation_degrees"], fill=0)
        )

        # Step 3: Remove black corners from rotation
        train_transform_list.append(v2.CenterCrop(size=aug_config["center_crop_size"]))

        # Step 4: Random crop + scale variation + resize back to 32×32
        train_transform_list.append(
            v2.RandomResizedCrop(
                size=aug_config["final_size"],
                scale=aug_config["resized_crop_scale"],
                ratio=aug_config["resized_crop_ratio"],
                antialias=True,
            )
        )

        # Step 5: Arrow overlay (after geometric transforms for crisp graphics)
        if aug_config["arrow_probability"] > 0:
            train_transform_list.append(
                RandomArrowOverlay(
                    arrow_dir=DEFAULT_ARROW_DIR,
                    probability=aug_config["arrow_probability"],
                )
            )

        # Step 6: Highlight overlay
        if aug_config["highlight_probability"] > 0:
            train_transform_list.append(
                RandomHighlightOverlay(
                    highlight_dir=DEFAULT_HIGHLIGHT_DIR,
                    probability=aug_config["highlight_probability"],
                )
            )

        # Step 7: Horizontal flip
        if aug_config["horizontal_flip"]:
            train_transform_list.append(
                v2.RandomHorizontalFlip(p=aug_config["horizontal_flip_prob"])
            )

        # Step 8: Color jitter
        train_transform_list.append(
            v2.ColorJitter(
                brightness=aug_config["brightness"],
                contrast=aug_config["contrast"],
                saturation=aug_config["saturation"],
                hue=aug_config["hue"],
            )
        )

        # Step 9: Convert to tensor, apply Gaussian noise
        # GaussianNoise requires tensor input, not PIL
        train_transform_list.append(v2.ToImage())
        train_transform_list.append(v2.ToDtype(dtype=torch.float32, scale=True))
        train_transform_list.append(
            v2.GaussianNoise(
                mean=aug_config["noise_mean"],
                sigma=aug_config["noise_sigma"],
            )
        )
        train_transform_list.append(v2.ToPILImage())

    elif model_id == "arrows":
        # Step 1: Highlight overlay (applied early, before geometric transforms)
        if aug_config["highlight_probability"] > 0:
            train_transform_list.append(
                RandomHighlightOverlay(
                    highlight_dir=DEFAULT_HIGHLIGHT_DIR,
                    probability=aug_config["highlight_probability"],
                )
            )

        # Step 2: Color jitter
        train_transform_list.append(
            v2.ColorJitter(
                brightness=aug_config["brightness"],
                contrast=aug_config["contrast"],
                saturation=aug_config["saturation"],
                hue=aug_config["hue"],
            )
        )

        # Step 3: Small rotation (±2 degrees)
        train_transform_list.append(
            v2.RandomRotation(degrees=aug_config["rotation_degrees"])
        )

        # Step 4: Convert to tensor, apply Gaussian noise
        # GaussianNoise requires tensor input, not PIL
        train_transform_list.append(v2.ToImage())
        train_transform_list.append(v2.ToDtype(dtype=torch.float32, scale=True))
        train_transform_list.append(
            v2.GaussianNoise(
                mean=aug_config["noise_mean"],
                sigma=aug_config["noise_sigma"],
            )
        )
        train_transform_list.append(v2.ToPILImage())

    train_transforms = v2.Compose(train_transform_list)
    val_transforms = v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
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
    best_val_loss = float("inf")
    best_train_acc = 0.0
    best_train_loss = float("inf")
    epochs_without_improvement = 0

    # Only use TrainingVisualizer when not using wandb
    if not use_wandb:
        output_dir = get_output_dir(model_id)
        visualizer = TrainingVisualizer(output_dir=output_dir)

    # Initialize global step counter for mid-epoch logging
    global_step = 0

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", leave=True)
    for epoch in epoch_pbar:
        (
            train_loss,
            train_acc,
            global_step,
            best_val_acc,
            best_val_loss,
            epochs_without_improvement,
        ) = train_epoch(
            model,
            optimizer,
            train_loader,
            val_loader,
            loss_and_grad_fn,
            wandb_logger,
            visualizer if not use_wandb else None,
            epoch + 1,
            global_step,
            LOG_TRAIN_EVERY_N_STEPS,
            LOG_VALIDATE_EVERY_N_STEPS,
            checkpoint_dir,
            model_id,
            best_val_acc,
            best_val_loss,
            epochs_without_improvement,
        )
        val_loss, val_acc = validate_epoch(model, val_loader, leave=False)

        # Update visualizer or log to wandb
        if use_wandb:
            # Log epoch-level metrics
            # Don't pass step parameter - let define_metric handle x-axis assignment
            wandb_logger.log(
                {
                    "epoch": epoch + 1,
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "accuracy/train": train_acc,
                    "accuracy/val": val_acc,
                }
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

        # Track best metrics (training only - validation is tracked in train_epoch)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if train_loss < best_train_loss:
            best_train_loss = train_loss

        # Check end-of-epoch validation results
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            model_path = checkpoint_dir / get_model_filename(model_id)
            mx.save_safetensors(str(model_path), dict(tree_flatten(model.parameters())))
            optimizer_path = checkpoint_dir / OPTIMIZER_FILENAME
            mx.save_safetensors(
                str(optimizer_path), dict(tree_flatten(optimizer.state))
            )
        else:
            epochs_without_improvement += 1

        # Check early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Save visualizations or log summary to wandb
    if use_wandb:
        wandb_logger.log_summary(
            {
                "best_val_accuracy": best_val_acc,
                "best_val_loss": best_val_loss,
            }
        )
    else:
        visualizer.save(TRAINING_CURVES_FILENAME)
        visualizer.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss:     {best_val_loss:.4f}")
    print(f"Best training accuracy:   {best_train_acc:.4f}")
    print(f"Best training loss:       {best_train_loss:.4f}")
    print(f"Total epochs:             {epoch + 1}")
    print(f"Model saved to: {checkpoint_dir / get_model_filename(model_id)}")

    # Finish wandb run
    wandb_logger.finish()
