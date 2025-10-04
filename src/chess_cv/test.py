"""Test script for evaluating trained model."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from .data import DataLoader
from .evaluate import compute_confusion_matrix, evaluate_model, print_evaluation_results
from .model import create_model
from .visualize import plot_confusion_matrix, plot_per_class_accuracy


def test(
    test_dir: Path | str = "data/test",
    checkpoint_path: Path | str = "checkpoints/best_model.safetensors",
    batch_size: int = 256,
    num_classes: int = 13,
    output_dir: Path | str = "outputs",
) -> None:
    """Test the trained model.

    Args:
        test_dir: Test data directory
        checkpoint_path: Path to saved model checkpoint
        batch_size: Batch size for evaluation (default: 256)
        num_classes: Number of output classes (default: 13)
        output_dir: Directory to save plots (default: outputs)
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python -m chess_cv.train")
        return

    # Load test data
    print("=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)
    test_loader = DataLoader(test_dir, batch_size=batch_size, shuffle=False)
    print(f"Test samples: {test_loader.num_samples}")
    print(f"Batch size:   {batch_size}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    model = create_model(num_classes=num_classes)

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = mx.load(str(checkpoint_path))
    model.update(tree_unflatten(list(checkpoint.items())))
    mx.eval(model.parameters())
    print("✓ Model loaded successfully")

    # Evaluate model
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    results = evaluate_model(model, test_loader, batch_size=batch_size)

    # Print results
    print_evaluation_results(results)

    # Compute confusion matrix
    print("Computing confusion matrix...")
    confusion_matrix = compute_confusion_matrix(
        model, test_loader.images, test_loader.labels, num_classes=num_classes
    )

    # Save visualizations
    print("\n" + "=" * 60)
    print("SAVING VISUALIZATIONS")
    print("=" * 60)

    plot_confusion_matrix(
        confusion_matrix, output_dir=output_dir, filename="test_confusion_matrix.png"
    )

    plot_per_class_accuracy(
        results["per_class_accuracy"],
        output_dir=output_dir,
        filename="test_per_class_accuracy.png",
    )

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


def main() -> None:
    """Run test script."""
    test()


if __name__ == "__main__":
    main()
