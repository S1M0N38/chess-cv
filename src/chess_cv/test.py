"""Test script for evaluating trained model."""

import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .data import (
    CLASS_NAMES,
    ChessPiecesDataset,
    collate_fn,
    get_all_labels,
    get_image_files,
    get_label_map,
)
from .evaluate import (
    compute_confusion_matrix,
    evaluate_model,
    print_evaluation_results,
)
from .model import create_model
from .visualize import plot_confusion_matrix, plot_per_class_accuracy


def test(
    test_dir: Path | str = "data/test",
    train_dir: Path | str = "data/train",  # For label map
    checkpoint_path: Path | str = "checkpoints/best_model.safetensors",
    batch_size: int = 256,
    image_size: int = 32,
    num_workers: int = 4,
    output_dir: Path | str = "outputs",
) -> None:
    """Test the trained model."""
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python -m chess_cv.train")
        return

    print("=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)

    # Use training data to create a consistent label map
    train_files = get_image_files(str(train_dir))
    all_train_labels = get_all_labels(train_files)
    label_map = get_label_map(all_train_labels)
    num_classes = len(label_map)

    test_transforms = transforms.Compose(
        [transforms.Resize((image_size, image_size), antialias=True)]
    )

    test_files = get_image_files(str(test_dir))
    test_dataset = ChessPiecesDataset(test_files, label_map, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size:   {batch_size}")
    print(f"Test batches: {len(test_loader)}")

    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    model = create_model(num_classes=num_classes)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = mx.load(str(checkpoint_path))
    model.update(tree_unflatten(list(checkpoint.items())))
    mx.eval(model.parameters())
    print("✓ Model loaded successfully")

    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    results = evaluate_model(model, test_loader, batch_size=batch_size)
    print_evaluation_results(results)

    # Gather all data for confusion matrix
    print("\nGathering all test data for confusion matrix...")
    all_images = []
    all_labels = []
    for images, labels in tqdm(test_loader, desc="Gathering Data"):
        all_images.append(images)
        all_labels.append(labels)
    images_array = mx.concatenate(all_images, axis=0)
    labels_array = mx.concatenate(all_labels, axis=0)

    print("Computing confusion matrix...")
    confusion_matrix = compute_confusion_matrix(
        model, images_array, labels_array, num_classes=num_classes
    )

    # Save misclassified images
    print("Saving misclassified images...")
    misclassified_dir = output_dir / "misclassified_images"
    if misclassified_dir.exists():
        shutil.rmtree(misclassified_dir)
    misclassified_dir.mkdir(parents=True)

    predictions = mx.argmax(model(images_array), axis=1)
    misclassified_indices = np.nonzero((predictions != labels_array).tolist())[0]

    for i in misclassified_indices.tolist():
        image_path = Path(test_dataset.image_files[i])
        true_label = CLASS_NAMES[labels_array[i].item()]
        predicted_label = CLASS_NAMES[predictions[i].item()]

        # Open the original image (not the transformed one)
        img = Image.open(image_path)

        # Save the image with a descriptive name
        new_filename = f"true_{true_label}_pred_{predicted_label}_{image_path.name}"
        img.save(misclassified_dir / new_filename)

    print("\n" + "=" * 60)
    print("SAVING VISUALIZATIONS")
    print("=" * 60)

    plot_confusion_matrix(
        confusion_matrix,
        class_names=CLASS_NAMES,
        output_dir=output_dir,
        filename="test_confusion_matrix.png",
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
