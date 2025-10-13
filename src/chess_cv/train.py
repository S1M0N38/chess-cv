"""Training script for chess piece classification."""

from pathlib import Path

__all__ = ["train"]

import torch
import torch.nn as nn
import torch.optim as optim
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
    DEFAULT_MOUSE_DIR,
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
    RandomMouseOverlay,
    collate_fn,
    get_image_files,
    get_label_map_from_class_names,
)
from .model import create_model
from .visualize import TrainingVisualizer
from .wandb_utils import WandbLogger


def safe_move_to_device(
    tensor: torch.Tensor, device: torch.device, non_blocking: bool = True
) -> torch.Tensor:
    """Safely move tensor to device with error handling for MPS.

    Args:
        tensor: Tensor to move
        device: Target device
        non_blocking: Whether to use non-blocking transfer

    Returns:
        Tensor on target device, or original tensor if move fails
    """
    try:
        return tensor.to(device, non_blocking=non_blocking)
    except RuntimeError as e:
        if device.type == "mps":
            print(f"Warning: MPS transfer failed, falling back to CPU. Error: {e}")
            return tensor.to("cpu", non_blocking=False)
        raise


def get_device_specific_config(device: torch.device) -> dict:
    """Get device-specific configuration for training.

    Args:
        device: PyTorch device

    Returns:
        Dictionary of device-specific configurations
    """
    config = {
        "supports_channels_last": device.type == "cuda",
        "supports_pin_memory": device.type == "cuda",
        "supports_mixed_precision": device.type in ["cuda", "mps"],
        "autocast_device_type": device.type
        if device.type in ["cuda", "mps"]
        else "cpu",
    }

    # Device-specific warnings
    if device.type == "mps":
        config.update(
            {
                "memory_warning": "MPS uses unified memory - monitor GPU memory usage",
                "num_workers_warning": "Consider reducing num_workers if you encounter MPS issues",
            }
        )

    return config


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    checkpoint_dir: Path,
    model_id: str,
    scaler: torch.amp.GradScaler | None = None,  # type: ignore
    wandb_logger: WandbLogger | None = None,
    visualizer=None,
    epoch: int = 0,
    global_step: int = 0,
    log_every_n_steps: int = 50,
    validate_every_n_steps: int = 1000,
    best_val_acc: float = 0.0,
    best_val_loss: float = float("inf"),
    epochs_without_improvement: int = 0,
) -> tuple[float, float, int, float, float, int]:
    """Train for one epoch with mid-epoch validation.

    Args:
        model: Model to train
        optimizer: Optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        model_id: Model identifier for checkpoint naming
        scaler: Gradient scaler for mixed precision (None if not using AMP)
        wandb_logger: WandB logger instance (optional)
        visualizer: TrainingVisualizer instance or None
        epoch: Current epoch number (1-indexed)
        global_step: Global step counter across all epochs
        log_every_n_steps: Log training metrics every N steps
        validate_every_n_steps: Run validation every N steps
        best_val_acc: Best validation accuracy so far
        best_val_loss: Best validation loss so far
        epochs_without_improvement: Counter for early stopping

    Returns:
        Tuple of (average_loss, accuracy, updated_global_step,
                  updated_best_val_acc, updated_best_val_loss,
                  updated_epochs_without_improvement)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (batch_images, batch_labels) in enumerate(pbar):
        # Safely move to device with error handling
        try:
            batch_images = safe_move_to_device(batch_images, device, non_blocking=True)
            batch_labels = safe_move_to_device(batch_labels, device, non_blocking=True)
        except Exception as e:
            print(f"Error moving batch to device {device}: {e}")
            continue

        # Apply device-specific memory format optimizations
        if device.type == "cuda":
            try:
                batch_images = batch_images.to(memory_format=torch.channels_last)
            except Exception as e:
                print(f"Warning: Failed to apply channels_last format: {e}")
        # MPS and CPU use standard memory format (channels_last not supported on MPS)

        optimizer.zero_grad()

        # Mixed precision training with device-specific autocast
        if scaler is not None:
            device_type = device.type  # cuda, mps, or cpu
            try:
                with torch.amp.autocast(device_type):  # type: ignore
                    logits = model(batch_images)
                    loss = criterion(logits, batch_labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                if device.type == "mps":
                    print(
                        f"Warning: MPS autocast failed, falling back to float32. Error: {e}"
                    )
                    # Fallback to float32 without autocast
                    logits = model(batch_images)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                else:
                    raise
        else:
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            correct = torch.sum(predictions == batch_labels)

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
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device, leave=True
            )

            # Return to training mode
            model.train()

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
                from .constants import OPTIMIZER_FILENAME, get_model_filename

                model_path = checkpoint_dir / get_model_filename(model_id)
                try:
                    from safetensors.torch import save_model

                    save_model(model, str(model_path))
                except ImportError:
                    # Fallback to regular torch.save if safetensors not available
                    torch.save(model.state_dict(), str(model_path))

                # Save optimizer state
                optimizer_path = checkpoint_dir / OPTIMIZER_FILENAME
                torch.save(optimizer.state_dict(), str(optimizer_path))

                print(
                    f"\n[Step {global_step}] New best validation accuracy: {val_acc:.4f} - Model saved"
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
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    leave: bool = False,
) -> tuple[float, float]:
    """Validate for one epoch.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        leave: Whether to leave the progress bar visible after completion

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(val_loader, desc="Validation", leave=leave)
    with torch.inference_mode():
        for batch_images, batch_labels in pbar:
            # Safely move to device with error handling
            try:
                batch_images = safe_move_to_device(
                    batch_images, device, non_blocking=True
                )
                batch_labels = safe_move_to_device(
                    batch_labels, device, non_blocking=True
                )
            except Exception as e:
                print(f"Error moving validation batch to device {device}: {e}")
                continue

            # Apply device-specific memory format optimizations
            if device.type == "cuda":
                try:
                    batch_images = batch_images.to(memory_format=torch.channels_last)
                except Exception as e:
                    print(
                        f"Warning: Failed to apply channels_last format in validation: {e}"
                    )
            # MPS and CPU use standard memory format (channels_last not supported on MPS)

            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            predictions = torch.argmax(logits, dim=1)
            correct = torch.sum(predictions == batch_labels)

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

    # Setup device with priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
    else:
        device = torch.device("cpu")
        device_type = "cpu"

    print(f"Using device: {device}")

    # Apply device-specific optimizations
    if device.type == "cuda":
        # Enable cuDNN benchmark for optimal conv performance
        torch.backends.cudnn.benchmark = True
        print("CUDA optimizations enabled:")
        print("  - cuDNN benchmark: True")
        print("  - Channels last memory format: True")
        print("  - Mixed precision training (AMP): True")
    elif device.type == "mps":
        print("MPS optimizations enabled:")
        print("  - Metal Performance Shaders: Active")
        print("  - Mixed precision training (AMP): True")
        print("  - Standard memory format (channels_last not supported)")
    else:
        print("CPU mode (no GPU acceleration available)")

    # Initialize wandb logger
    wandb_logger = WandbLogger(enabled=use_wandb)
    if use_wandb:
        config = {
            "model_id": model_id,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(device),
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

    # Advanced augmentation pipeline for models that support it (pieces, snap)
    if "padding" in aug_config:
        # Multi-step geometric transformations (from snap model)

        # Step 1: Padding - Create rotation space
        train_transform_list.append(
            v2.Pad(
                padding=aug_config["padding"],
                padding_mode=aug_config.get("padding_mode", "edge"),
            )
        )

        # Step 2: Rotation
        if aug_config["rotation_degrees"] > 0:
            train_transform_list.append(
                v2.RandomRotation(degrees=aug_config["rotation_degrees"])
            )

        # Step 3: Center Crop - Remove black bands from rotation
        if "center_crop_size" in aug_config:
            train_transform_list.append(
                v2.CenterCrop(size=aug_config["center_crop_size"])
            )

        # Step 4: Random Resized Crop - Simulate distance/position variation
        if "resized_crop_scale" in aug_config and "resized_crop_ratio" in aug_config:
            train_transform_list.append(
                v2.RandomResizedCrop(
                    size=aug_config["final_size"],
                    scale=aug_config["resized_crop_scale"],
                    ratio=aug_config["resized_crop_ratio"],
                    antialias=True,
                )
            )
        else:
            # Fallback to simple resize for compatibility
            train_transform_list.append(
                v2.Resize((image_size, image_size), antialias=True)
            )
    else:
        # Simple augmentation pipeline for backward compatibility (arrows model)
        # Random resized crop
        train_transform_list.append(
            v2.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(aug_config["scale_min"], aug_config["scale_max"]),
                antialias=True,
            )
        )

    # Arrow overlay (after geometric transforms, same as snap-model branch)
    if aug_config["arrow_probability"] > 0:
        train_transform_list.append(
            RandomArrowOverlay(
                arrow_dir=DEFAULT_ARROW_DIR,
                probability=aug_config["arrow_probability"],
            )
        )

    # Highlight overlay
    if aug_config["highlight_probability"] > 0:
        train_transform_list.append(
            RandomHighlightOverlay(
                highlight_dir=DEFAULT_HIGHLIGHT_DIR,
                probability=aug_config["highlight_probability"],
            )
        )

    # Mouse cursor overlay (new feature from snap model)
    if aug_config.get("mouse_probability", 0) > 0:
        train_transform_list.append(
            RandomMouseOverlay(
                mouse_dir=DEFAULT_MOUSE_DIR,
                probability=aug_config["mouse_probability"],
                aug_config=aug_config,
            )
        )

    # Horizontal flip
    if aug_config["horizontal_flip"]:
        train_transform_list.append(v2.RandomHorizontalFlip(p=aug_config["horizontal_flip_prob"]))

    # Color jitter (with hue support for models that have it)
    color_jitter_kwargs = {
        "brightness": aug_config["brightness"],
        "contrast": aug_config["contrast"],
        "saturation": aug_config["saturation"],
    }
    # Add hue parameter if available (new feature from snap model)
    if "hue" in aug_config:
        color_jitter_kwargs["hue"] = aug_config["hue"]

    train_transform_list.append(v2.ColorJitter(**color_jitter_kwargs))

    # Small rotation for arrows model (applied after color jitter, same as snap-model branch)
    # Only apply if we're in the simple pipeline (arrows model) and rotation_degrees > 0
    if "padding" not in aug_config and aug_config.get("rotation_degrees", 0) > 0:
        train_transform_list.append(v2.RandomRotation(degrees=aug_config["rotation_degrees"]))

    # Gaussian noise using v2 transforms (same as snap-model branch)
    # Convert to tensor, apply Gaussian noise, convert back to PIL
    train_transform_list.append(v2.ToImage())
    train_transform_list.append(v2.ToDtype(dtype=torch.float32, scale=True))
    train_transform_list.append(v2.GaussianNoise(mean=aug_config["noise_mean"], sigma=aug_config["noise_sigma"]))
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

    # Create DataLoaders with device-specific optimizations
    # Pin memory configuration: CUDA=True, MPS=False, CPU=False
    pin_memory_enabled = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory_enabled,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory_enabled,
        persistent_workers=(num_workers > 0),
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
    model = create_model(num_classes=num_classes, device=str(device))
    print(model)

    # Calculate total parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    # Create optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print("\nOptimizer: AdamW")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay:  {weight_decay}")
    print("\nLoss function: CrossEntropyLoss")

    # Setup mixed precision training for CUDA and MPS
    scaler = None
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")  # type: ignore
        print("Mixed precision: Enabled (CUDA)")
    elif device.type == "mps":
        scaler = torch.amp.GradScaler("mps")  # type: ignore
        print("Mixed precision: Enabled (MPS)")
    else:
        print("Mixed precision: Disabled (CPU)")

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
            checkpoint_dir,
            model_id,
            scaler,
            wandb_logger,
            visualizer if not use_wandb else None,
            epoch + 1,
            global_step,
            LOG_TRAIN_EVERY_N_STEPS,
            LOG_VALIDATE_EVERY_N_STEPS,
            best_val_acc,
            best_val_loss,
            epochs_without_improvement,
        )
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, leave=False
        )

        # Update visualizer or log to wandb
        if use_wandb:
            # Log metrics with epoch as both a metric and the step parameter
            # This ensures proper x-axis alignment
            wandb_logger.log(
                {
                    "epoch": epoch + 1,
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "accuracy/train": train_acc,
                    "accuracy/val": val_acc,
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

            # Save model using safetensors
            model_path = checkpoint_dir / get_model_filename(model_id)
            try:
                from safetensors.torch import save_model

                save_model(model, str(model_path))
            except ImportError:
                # Fallback to regular torch.save if safetensors not available
                torch.save(model.state_dict(), str(model_path))

            # Save optimizer state
            optimizer_path = checkpoint_dir / OPTIMIZER_FILENAME
            torch.save(optimizer.state_dict(), str(optimizer_path))
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
