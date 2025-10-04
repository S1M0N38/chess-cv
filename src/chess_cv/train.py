"""Training script for chess piece classification."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

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
    train_dir: Path | str = "data/train",
    val_dir: Path | str = "data/validate",
    checkpoint_dir: Path | str = "checkpoints",
    batch_size: int = 128,
    learning_rate: float = 2e-4,
    weight_decay: float = 5e-4,
    num_epochs: int = 100,
    patience: int = 15,
    image_size: int = 32,
    num_workers: int = 4,
) -> None:
    """Train the model."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
                size=(image_size, image_size), scale=(0.8, 1.0), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=5),
            AddGaussianNoise(mean=0.0, std=0.05),
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
    if train_files:
        original_image = Image.open(train_files[0]).convert("RGB")
        resized_original = val_transforms(original_image)
        augmented_image = train_transforms(original_image)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(resized_original)
        ax[0].set_title("Original (Resized)")
        ax[0].axis("off")
        ax[1].imshow(augmented_image)
        ax[1].set_title("Augmented")
        ax[1].axis("off")

        plt.tight_layout()
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "augmentation_example.png")
        plt.savefig(output_path)
        print(f"\nSaved augmentation example to {output_path}")
        plt.close(fig)

    # Create model
    print("\n" + "=" * 60)
    print("MODEL")
    print("=" * 60)
    model = create_model(num_classes=num_classes)
    print(model)

    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"\nTotal parameters: {num_params:,}")

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    print("\nOptimizer: AdamW")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay:  {weight_decay}")

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    best_val_acc = 0.0
    epochs_without_improvement = 0
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
            model_path = checkpoint_dir / "best_model.safetensors"
            mx.save_safetensors(str(model_path), dict(tree_flatten(model.parameters())))
            optimizer_path = checkpoint_dir / "optimizer.safetensors"
            mx.save_safetensors(
                str(optimizer_path), dict(tree_flatten(optimizer.state))
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    visualizer.save()
    visualizer.close()

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
