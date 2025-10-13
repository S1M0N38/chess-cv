"""Training script for chess piece classification."""

from pathlib import Path

__all__ = ["train"]

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from .constants import (
    AUGMENTATION_CONFIGS,
    DEFAULT_ARROW_DIR,
    DEFAULT_BASE_LR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIGHLIGHT_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_LR,
    DEFAULT_MOUSE_DIR,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    DEFAULT_WARMUP_RATIO,
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
    RandomMouseOverlay,
    collate_fn,
    get_image_files,
    get_label_map_from_class_names,
)
from .model import create_model
from .visualize import TrainingVisualizer
from .wandb_utils import WandbLogger


def loss_fn(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
) -> torch.Tensor:
    """Compute cross-entropy loss."""
    logits = model(images)
    loss = criterion(logits, labels)
    return loss


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
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
        criterion: Loss function
        device: Device to run training on
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
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (batch_images, batch_labels) in enumerate(pbar):
        # Move data to device
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        # Zero gradients, forward pass, backward pass, optimize
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predictions = torch.max(outputs, 1)
        correct = (predictions == batch_labels).sum().item()

        batch_size = batch_labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

        # Increment global step
        global_step += 1

        # Log mid-epoch training metrics to wandb
        if (
            wandb_logger is not None
            and wandb_logger.enabled
            and global_step % log_every_n_steps == 0
        ):
            batch_acc = correct / batch_size
            current_lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "global_step": global_step,
                "loss/train_step": loss.item(),
                "accuracy/train_step": batch_acc,
                "learning_rate": current_lr,
                "epoch": epoch,
            }
            wandb_logger.log(log_dict, step=global_step)

        # Mid-epoch validation
        if global_step % validate_every_n_steps == 0:
            # Run validation on full validation set
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device, leave=False
            )

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
                model_path = checkpoint_dir / get_model_filename(model_id)
                torch.save(model.state_dict(), model_path)
                optimizer_path = checkpoint_dir / OPTIMIZER_FILENAME
                torch.save(optimizer.state_dict(), optimizer_path)
            else:
                epochs_without_improvement += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / batch_size:.4f}",
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
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    leave: bool = False,
) -> tuple[float, float]:
    """Validate for one epoch.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
        leave: Whether to leave the progress bar visible after completion

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(val_loader, desc="Validation", leave=leave)
    with torch.no_grad():
        for batch_images, batch_labels in pbar:
            # Move data to device
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            _, predictions = torch.max(outputs, 1)
            correct = (predictions == batch_labels).sum().item()

            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct / batch_size:.4f}",
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
    use_scheduler: bool = True,
    base_lr: float = DEFAULT_BASE_LR,
    min_lr: float = DEFAULT_MIN_LR,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
) -> None:
    """Train the model.

    Args:
        model_id: Model identifier (e.g., 'pieces')
        train_dir: Training data directory
        val_dir: Validation data directory
        checkpoint_dir: Checkpoint directory for saving models
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer (used only if use_scheduler=False)
        weight_decay: Weight decay for optimizer
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        image_size: Image size for resizing
        num_workers: Number of data loading workers
        use_wandb: Enable Weights & Biases logging
        use_scheduler: Use learning rate scheduler (warmup + cosine decay)
        base_lr: Peak learning rate after warmup (used if use_scheduler=True)
        min_lr: Minimum learning rate at end of decay (used if use_scheduler=True)
        warmup_ratio: Fraction of training for warmup phase (used if use_scheduler=True)
    """
    from .constants import (
        get_checkpoint_dir,
        get_model_config,
        get_train_dir,
        get_val_dir,
    )

    # Only support pieces, arrows, and snap models
    if model_id not in ["pieces", "arrows", "snap"]:
        msg = f"Model '{model_id}' is not supported. Only 'pieces', 'arrows', and 'snap' models are supported."
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
            "use_scheduler": use_scheduler,
        }
        # Add scheduler-specific config if enabled
        if use_scheduler:
            config.update(
                {
                    "base_lr": base_lr,
                    "min_lr": min_lr,
                    "warmup_ratio": warmup_ratio,
                }
            )

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

        # Step 7: Mouse overlay
        if aug_config["mouse_probability"] > 0:
            train_transform_list.append(
                RandomMouseOverlay(
                    mouse_dir=DEFAULT_MOUSE_DIR,
                    probability=aug_config["mouse_probability"],
                    aug_config=aug_config,
                )
            )

        # Step 8: Horizontal flip
        if aug_config["horizontal_flip"]:
            train_transform_list.append(
                v2.RandomHorizontalFlip(p=aug_config["horizontal_flip_prob"])
            )

        # Step 9: Color jitter
        train_transform_list.append(
            v2.ColorJitter(
                brightness=aug_config["brightness"],
                contrast=aug_config["contrast"],
                saturation=aug_config["saturation"],
                hue=aug_config["hue"],
            )
        )

        # Step 10: Convert to tensor, apply Gaussian noise
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

    elif model_id == "snap":
        # Step 1: Arrow overlay (applied early, before geometric transforms)
        if aug_config["arrow_probability"] > 0:
            train_transform_list.append(
                RandomArrowOverlay(
                    arrow_dir=DEFAULT_ARROW_DIR,
                    probability=aug_config["arrow_probability"],
                )
            )

        # Step 2: Highlight overlay
        if aug_config["highlight_probability"] > 0:
            train_transform_list.append(
                RandomHighlightOverlay(
                    highlight_dir=DEFAULT_HIGHLIGHT_DIR,
                    probability=aug_config["highlight_probability"],
                )
            )

        # Step 3: Mouse overlay
        if aug_config["mouse_probability"] > 0:
            train_transform_list.append(
                RandomMouseOverlay(
                    mouse_dir=DEFAULT_MOUSE_DIR,
                    probability=aug_config["mouse_probability"],
                    aug_config=aug_config,
                )
            )

        # Step 4: Horizontal flip
        if aug_config["horizontal_flip"]:
            train_transform_list.append(
                v2.RandomHorizontalFlip(p=aug_config["horizontal_flip_prob"])
            )

        # Step 5: Color jitter
        train_transform_list.append(
            v2.ColorJitter(
                brightness=aug_config["brightness"],
                contrast=aug_config["contrast"],
                saturation=aug_config["saturation"],
                hue=aug_config["hue"],
            )
        )

        # Step 6: Convert to tensor, apply Gaussian noise
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
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    # Create optimizer with optional learning rate scheduler
    print("\n" + "=" * 60)
    print("OPTIMIZER")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if use_scheduler:
        # Calculate total training steps
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(warmup_ratio * total_steps)

        # Create optimizer with base learning rate
        optimizer = optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )

        # Create learning rate scheduler: warmup + cosine decay using PyTorch's SequentialLR (MLX-compatible)
        # Warmup scheduler - starts from exactly zero LR and linearly increases to base_lr (MLX behavior)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.0,  # Start at exactly zero LR (MLX behavior)
            total_iters=warmup_steps,
        )

        # Cosine decay scheduler - starts from base_lr and decays to min_lr
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,  # Decay phase duration
            eta_min=min_lr,  # Minimum learning rate
        )

        # Chain schedulers: warmup first, then cosine decay
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[
                warmup_steps
            ],  # Switch from warmup to cosine decay at this step
        )

        print("Optimizer: AdamW with SequentialLR scheduler (MLX-compatible)")
        print(f"  Base LR:       {base_lr}")
        print(f"  Min LR:        {min_lr}")
        print(f"  Start factor:  0.0 (exactly zero LR, matching MLX)")
        print(f"  Schedule:      0→{base_lr}→{min_lr} (warmup + cosine decay)")
        print(
            f"  Warmup steps:  {warmup_steps} ({warmup_ratio * 100:.1f}% of {total_steps} total steps)"
        )
        print(f"  Total steps:   {total_steps}")
        print(f"  Weight decay:  {weight_decay}")
        print(
            f"  Scheduler:     LinearLR(0→{base_lr}) → CosineAnnealingLR({base_lr}→{min_lr})"
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = None
        print("Optimizer: AdamW (no scheduler)")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay:  {weight_decay}")

    print(f"Device: {device}")

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
            criterion,
            device,
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
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, leave=False
        )

        # Update scheduler if using one
        if scheduler is not None:
            scheduler.step()

        # Update visualizer or log to wandb
        if use_wandb:
            # Log epoch-level metrics
            # Don't pass step parameter - let define_metric handle x-axis assignment
            epoch_log_dict = {
                "epoch": epoch + 1,
                "loss/train": train_loss,
                "loss/val": val_loss,
                "accuracy/train": train_acc,
                "accuracy/val": val_acc,
            }
            # Add learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_log_dict["learning_rate"] = current_lr
            wandb_logger.log(epoch_log_dict)
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
            torch.save(model.state_dict(), model_path)
            optimizer_path = checkpoint_dir / OPTIMIZER_FILENAME
            torch.save(optimizer.state_dict(), optimizer_path)
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
