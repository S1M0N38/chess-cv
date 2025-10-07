"""Test script for evaluating trained model."""

import json
import shutil
from pathlib import Path

__all__ = ["test"]

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .constants import (
    BEST_MODEL_FILENAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_WORKERS,
    MISCLASSIFIED_DIR,
    TEST_CONFUSION_MATRIX_FILENAME,
    TEST_PER_CLASS_ACCURACY_FILENAME,
    TEST_SUMMARY_FILENAME,
)
from .data import (
    ChessPiecesDataset,
    HuggingFaceChessPiecesDataset,
    collate_fn,
    get_all_labels,
    get_image_files,
    get_label_map,
)
from .evaluate import (
    compute_confusion_matrix,
    compute_f1_score,
    evaluate_model,
    print_evaluation_results,
)
from .model import create_model
from .visualize import plot_confusion_matrix, plot_per_class_accuracy
from .wandb_utils import WandbLogger


def test(
    model_id: str,
    test_dir: Path | str | None = None,
    train_dir: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    output_dir: Path | str | None = None,
    use_wandb: bool = False,
    hf_test_dir: str | None = None,
) -> None:
    """Test the trained model.

    Args:
        model_id: Model identifier (e.g., 'pieces')
        test_dir: Local test data directory
        train_dir: Training data directory for label map
        checkpoint_path: Path to model checkpoint
        batch_size: Batch size for testing
        image_size: Image size for resizing
        num_workers: Number of data loading workers
        output_dir: Directory for saving results
        use_wandb: Enable Weights & Biases logging
        hf_test_dir: HuggingFace dataset ID (e.g., "S1M0N38/chess-cv-openboard").
                     If provided, test_dir is ignored.
    """
    from .constants import (
        get_checkpoint_dir,
        get_model_config,
        get_output_dir,
        get_test_dir,
        get_train_dir,
    )

    # Get model configuration
    model_config = get_model_config(model_id)
    num_classes = model_config["num_classes"]
    class_names = model_config["class_names"]

    # Set default directories if not provided
    if test_dir is None:
        test_dir = get_test_dir(model_id)
    if train_dir is None:
        train_dir = get_train_dir(model_id)
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_dir(model_id) / BEST_MODEL_FILENAME
    if output_dir is None:
        output_dir = get_output_dir(model_id)

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb logger
    wandb_logger = WandbLogger(enabled=use_wandb)
    if use_wandb:
        wandb_logger.init(
            project=f"chess-cv-{model_id}-evaluation",
            config={
                "model_id": model_id,
                "num_classes": num_classes,
                "test_dir": str(test_dir) if hf_test_dir is None else hf_test_dir,
                "checkpoint_path": str(checkpoint_path),
                "batch_size": batch_size,
                "image_size": image_size,
                "num_workers": num_workers,
                "hf_test_dir": hf_test_dir,
            },
        )

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

    test_transforms = transforms.Compose(
        [transforms.Resize((image_size, image_size), antialias=True)]
    )

    # Load test dataset from HuggingFace or local directory
    if hf_test_dir is not None:
        print(f"Loading test data from HuggingFace dataset: {hf_test_dir}")
        test_dataset = HuggingFaceChessPiecesDataset(
            hf_test_dir, label_map, split="train", transform=test_transforms
        )
    else:
        print(f"Loading test data from local directory: {test_dir}")
        test_files = get_image_files(str(test_dir))
        test_dataset = ChessPiecesDataset(
            test_files, label_map, transform=test_transforms
        )

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
    # MLX load returns dict-like structure
    checkpoint_items = list(checkpoint.items())  # type: ignore[attr-defined]
    model.update(tree_unflatten(checkpoint_items))
    mx.eval(model.parameters())
    print("✓ Model loaded successfully")

    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    results = evaluate_model(
        model, test_loader, class_names=class_names, batch_size=batch_size
    )
    print_evaluation_results(results, class_names=class_names)

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
    results["f1_score_macro"] = compute_f1_score(confusion_matrix)

    # Save summary to JSON
    print(f"Saving test summary to: {output_dir / TEST_SUMMARY_FILENAME}")
    summary = {
        "overall_accuracy": results["overall_accuracy"],
        "f1_score_macro": results["f1_score_macro"],
        "per_class_accuracy": results["per_class_accuracy"],
        "checkpoint_path": str(checkpoint_path),
        "test_dir": str(test_dir) if hf_test_dir is None else hf_test_dir,
        "num_test_samples": len(test_dataset),
    }
    with open(output_dir / TEST_SUMMARY_FILENAME, "w") as f:
        json.dump(summary, f, indent=2)

    # Log test results to wandb
    if use_wandb:
        wandb_logger.log(
            {
                "test/accuracy": results["overall_accuracy"],
                "test/f1_score_macro": results["f1_score_macro"],
            }
        )
        # Log per-class accuracy
        per_class_acc = results["per_class_accuracy"]
        if isinstance(per_class_acc, dict):
            for class_name, acc in per_class_acc.items():
                wandb_logger.log({f"test/class_accuracy/{class_name}": acc})

    # Save misclassified images
    print("Saving misclassified images...")
    misclassified_dir = output_dir / MISCLASSIFIED_DIR
    if misclassified_dir.exists():
        shutil.rmtree(misclassified_dir)
    misclassified_dir.mkdir(parents=True)

    predictions = mx.argmax(model(images_array), axis=1)
    misclassified_mask = predictions != labels_array  # type: ignore[assignment]
    misclassified_array = np.array(misclassified_mask)  # type: ignore[arg-type]
    misclassified_indices = np.nonzero(misclassified_array)[0]

    for idx in misclassified_indices.tolist():
        true_label_idx = int(labels_array[idx].item())  # type: ignore[union-attr]
        pred_label_idx = int(predictions[idx].item())  # type: ignore[union-attr]
        true_label = class_names[true_label_idx]
        predicted_label = class_names[pred_label_idx]

        # Get the original image based on dataset type
        if isinstance(test_dataset, HuggingFaceChessPiecesDataset):
            # For HuggingFace datasets, get image from the dataset directly
            item = test_dataset.dataset[idx]  # type: ignore[index]
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")  # type: ignore[arg-type]
            else:
                img = img.convert("RGB")
            image_name = f"{idx}.png"
        else:
            # For local datasets, open from file path
            image_path = Path(test_dataset.image_files[idx])
            img = Image.open(image_path)
            image_name = image_path.name

        # Save the image with a descriptive name
        new_filename = f"true_{true_label}_pred_{predicted_label}_{image_name}"
        img.save(misclassified_dir / new_filename)

    print("\n" + "=" * 60)
    print("SAVING VISUALIZATIONS")
    print("=" * 60)

    # Save matplotlib plots to files
    plot_confusion_matrix(
        confusion_matrix,
        class_names=class_names,
        output_dir=output_dir,
        filename=TEST_CONFUSION_MATRIX_FILENAME,
    )
    per_class_acc = results["per_class_accuracy"]
    assert isinstance(per_class_acc, dict)
    plot_per_class_accuracy(
        per_class_acc,
        output_dir=output_dir,
        filename=TEST_PER_CLASS_ACCURACY_FILENAME,
    )

    if use_wandb:
        # Log sample misclassified images to wandb
        print("Logging sample misclassified images to wandb...")
        max_samples = min(20, len(misclassified_indices))  # Log up to 20 samples
        for i, idx in enumerate(misclassified_indices[:max_samples].tolist()):
            true_label_idx = int(labels_array[idx].item())  # type: ignore[union-attr]
            pred_label_idx = int(predictions[idx].item())  # type: ignore[union-attr]
            true_label = class_names[true_label_idx]
            predicted_label = class_names[pred_label_idx]

            # Get image path based on dataset type
            if isinstance(test_dataset, HuggingFaceChessPiecesDataset):
                # For HuggingFace datasets, use the saved misclassified image
                image_name = f"{idx}.png"
                new_filename = f"true_{true_label}_pred_{predicted_label}_{image_name}"
                image_path = misclassified_dir / new_filename
            else:
                # For local datasets, use the original file path
                image_path = Path(test_dataset.image_files[idx])

            wandb_logger.log_image(
                f"misclassified/{i}",
                image_path,
                caption=f"True: {true_label}, Predicted: {predicted_label}",
            )

        print(f"Logged {max_samples} misclassified images to wandb")

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    if use_wandb:
        print("Results logged to wandb")
    else:
        print(f"Results saved to: {output_dir}")

    # Finish wandb run
    wandb_logger.finish()
