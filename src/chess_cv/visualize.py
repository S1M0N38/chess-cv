"""Visualization utilities for training and evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "TrainingVisualizer",
    "plot_confusion_matrix",
    "plot_per_class_accuracy",
]


class TrainingVisualizer:
    """Real-time training visualization."""

    def __init__(self, output_dir: Path | str = "outputs"):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.epochs = []

        # Set up the plot
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.suptitle("Training Progress")

    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """Update plots with new metrics.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        # Update history
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Plot loss
        self.ax1.plot(self.epochs, self.train_losses, "b-", label="Train Loss")
        self.ax1.plot(self.epochs, self.val_losses, "r-", label="Val Loss")
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss")
        self.ax1.set_title("Training and Validation Loss")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(top=0.6)

        # Plot accuracy
        self.ax2.plot(self.epochs, self.train_accs, "b-", label="Train Acc")
        self.ax2.plot(self.epochs, self.val_accs, "r-", label="Val Acc")
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("Accuracy")
        self.ax2.set_title("Training and Validation Accuracy")
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(bottom=0.8)

        # Refresh the plot
        plt.tight_layout()
        plt.pause(0.01)

    def save(self, filename: str = "training_curves.png") -> None:
        """Save the current plot to file.

        Args:
            filename: Output filename
        """
        save_path = self.output_dir / filename
        self.fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    def close(self) -> None:
        """Close the interactive plot."""
        plt.ioff()
        plt.close(self.fig)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    output_dir: Path | str = "outputs",
    filename: str = "confusion_matrix.png",
) -> None:
    """Plot confusion matrix as a heatmap.

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)
        output_dir: Directory to save the plot
        filename: Output filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(confusion_matrix, cmap="Blues", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                str(confusion_matrix[i, j]),
                ha="center",
                va="center",
                color="white"
                if confusion_matrix[i, j] > confusion_matrix.max() / 2
                else "black",
            )

    # Labels and title
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to {save_path}")


def plot_per_class_accuracy(
    per_class_acc: dict[str, float],
    output_dir: Path | str = "outputs",
    filename: str = "per_class_accuracy.png",
) -> None:
    """Plot per-class accuracy as a bar chart.

    Args:
        per_class_acc: Dictionary mapping class names to accuracy values
        output_dir: Directory to save the plot
        filename: Output filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract data
    classes = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())

    # Create bar chart
    bars = ax.bar(range(len(classes)), accuracies, color="steelblue", alpha=0.7)

    # Customize plot
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved per-class accuracy to {save_path}")
