"""Training script for chess piece classification."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from tqdm import tqdm

from .data import CLASS_NAMES, DataLoader
from .evaluate import compute_per_class_accuracy, evaluate_model
from .model import create_model
from .visualize import TrainingVisualizer


def loss_fn(model: nn.Module, images: mx.array, labels: mx.array) -> mx.array:
    """Compute cross-entropy loss.

    Args:
        model: Neural network model
        images: Batch of images
        labels: Batch of labels

    Returns:
        Loss value
    """
    logits = model(images)
    loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
    return loss


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    loss_and_grad_fn,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Neural network model
        optimizer: Optimizer
        train_loader: Training data loader
        loss_and_grad_fn: Function that computes loss and gradients

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Progress bar for batches
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_images, batch_labels in pbar:
        # Compute loss and gradients
        loss, grads = loss_and_grad_fn(model, batch_images, batch_labels)

        # Update model parameters
        optimizer.update(model, grads)

        # Evaluate the updated parameters and optimizer state
        mx.eval(model.parameters(), optimizer.state, loss)

        # Compute accuracy for this batch
        logits = model(batch_images)
        predictions = mx.argmax(logits, axis=1)
        correct = mx.sum(predictions == batch_labels)

        # Update statistics
        batch_size = len(batch_labels)
        total_loss += loss.item() * batch_size
        total_correct += correct.item()
        total_samples += batch_size

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct.item() / batch_size:.4f}",
            }
        )

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def validate_epoch(
    model: nn.Module, val_loader: DataLoader
) -> tuple[float, float]:
    """Validate for one epoch.

    Args:
        model: Neural network model
        val_loader: Validation data loader

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Progress bar for batches
    pbar = tqdm(val_loader, desc="Validation", leave=False)

    for batch_images, batch_labels in pbar:
        # Compute loss
        loss = loss_fn(model, batch_images, batch_labels)

        # Compute predictions
        logits = model(batch_images)
        predictions = mx.argmax(logits, axis=1)
        correct = mx.sum(predictions == batch_labels)

        # Update statistics
        batch_size = len(batch_labels)
        total_loss += loss.item() * batch_size
        total_correct += correct.item()
        total_samples += batch_size

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct.item() / batch_size:.4f}",
            }
        )

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def train(
    train_dir: Path | str = "data/train",
    val_dir: Path | str = "data/validate",
    checkpoint_dir: Path | str = "checkpoints",
    batch_size: int = 128,
    learning_rate: float = 2e-4,
    weight_decay: float = 5e-4,
    num_epochs: int = 100,
    patience: int = 10,
    num_classes: int = 13,
) -> None:
    """Train the model.

    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        checkpoint_dir: Directory to save checkpoints
        batch_size: Batch size for training (default: 128)
        learning_rate: Learning rate for AdamW (default: 2e-4)
        weight_decay: Weight decay for AdamW (default: 5e-4)
        num_epochs: Maximum number of epochs (default: 100)
        patience: Early stopping patience (default: 10)
        num_classes: Number of output classes (default: 13)
    """
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    train_loader = DataLoader(train_dir, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dir, batch_size=batch_size, shuffle=False)

    print(f"\nTraining samples:   {train_loader.num_samples}")
    print(f"Validation samples: {val_loader.num_samples}")
    print(f"Batch size:         {batch_size}")
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    print("\n" + "=" * 60)
    print("MODEL")
    print("=" * 60)
    model = create_model(num_classes=num_classes)
    print(model)

    # Count parameters
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"\nTotal parameters: {num_params:,}")

    # Create optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay:  {weight_decay}")

    # Create loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training setup
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Visualization
    visualizer = TrainingVisualizer()

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, optimizer, train_loader, loss_and_grad_fn
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader)

        # Update visualization
        visualizer.update(epoch + 1, train_loss, train_acc, val_loss, val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            # Save model
            model_path = checkpoint_dir / "best_model.safetensors"
            model_params = dict(tree_flatten(model.parameters()))
            mx.save_safetensors(str(model_path), model_params)

            # Save optimizer state
            optimizer_path = checkpoint_dir / "optimizer.safetensors"
            optimizer_state = dict(tree_flatten(optimizer.state))
            mx.save_safetensors(str(optimizer_path), optimizer_state)
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Save final visualization
    visualizer.save()
    visualizer.close()

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.safetensors'}")


def main() -> None:
    """Run training script."""
    train()


if __name__ == "__main__":
    main()
