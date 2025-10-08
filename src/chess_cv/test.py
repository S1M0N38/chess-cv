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

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_WORKERS,
    MAX_MISCLASSIFIED_IMAGES,
    MISCLASSIFIED_DIR,
    TEST_CONFUSION_MATRIX_FILENAME,
    TEST_PER_CLASS_ACCURACY_FILENAME,
    TEST_SUMMARY_FILENAME,
    get_model_filename,
)
from .data import (
    ChessPiecesDataset,
    ConcatenatedHuggingFaceDataset,
    HuggingFaceChessPiecesDataset,
    collate_fn,
    get_all_labels,
    get_image_files,
    get_label_map,
)
from .evaluate import (
    benchmark_inference_speed,
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
    concat_splits: bool = False,
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
        concat_splits: If True, concatenate all splits from HuggingFace dataset.
                       Only applicable when hf_test_dir is provided.
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
        checkpoint_path = get_checkpoint_dir(model_id) / get_model_filename(model_id)
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
                "concat_splits": concat_splits,
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
        if concat_splits:
            print(
                f"Loading test data from HuggingFace dataset (all splits): {hf_test_dir}"
            )
            test_dataset = ConcatenatedHuggingFaceDataset(
                hf_test_dir, label_map, splits=None, transform=test_transforms
            )
        else:
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

    # Extract predictions and labels from results
    predictions = results["predictions"]
    labels = results["labels"]
    assert isinstance(predictions, list)
    assert isinstance(labels, list)

    # Print evaluation results (without predictions/labels)
    overall_acc = results["overall_accuracy"]
    per_class = results["per_class_accuracy"]
    assert isinstance(overall_acc, float)
    assert isinstance(per_class, dict)
    eval_results: dict[str, float | dict[str, float]] = {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class,
    }
    print_evaluation_results(eval_results, class_names=class_names)

    # Compute confusion matrix
    print("\nComputing confusion matrix and F1 score...")
    confusion_matrix = compute_confusion_matrix(
        model, test_loader, num_classes=num_classes
    )
    results["f1_score_macro"] = compute_f1_score(labels, predictions)

    # Benchmark inference speed
    print("\n" + "=" * 60)
    print("BENCHMARKING INFERENCE SPEED")
    print("=" * 60)
    print("Testing inference speed for batch sizes: 1, 64, 512, 1024")
    print("Running warmup and measurement iterations...")

    benchmark_results = benchmark_inference_speed(
        model=model,
        image_size=image_size,
        batch_sizes=[1, 64, 512, 1024],
        num_warmup=10,
        num_iterations=50,
    )

    # Print benchmark results
    print("\nBenchmark Results:")
    print("-" * 60)
    print(f"{'Batch Size':<12} {'Images/sec':<15} {'ms/batch':<15} {'ms/image':<15}")
    print("-" * 60)
    for batch_size_str, metrics in benchmark_results.items():
        print(
            f"{batch_size_str:<12} "
            f"{metrics['images_per_second']:<15.2f} "
            f"{metrics['ms_per_batch']:<15.4f} "
            f"{metrics['ms_per_image']:<15.4f}"
        )
    print("-" * 60)

    # Save summary to JSON
    print(f"\nSaving test summary to: {output_dir / TEST_SUMMARY_FILENAME}")
    summary = {
        "overall_accuracy": results["overall_accuracy"],
        "f1_score_macro": results["f1_score_macro"],
        "per_class_accuracy": results["per_class_accuracy"],
        "checkpoint_path": str(checkpoint_path),
        "test_dir": str(test_dir) if hf_test_dir is None else hf_test_dir,
        "num_test_samples": len(test_dataset),
        "inference_benchmark": benchmark_results,
    }
    with open(output_dir / TEST_SUMMARY_FILENAME, "w") as f:
        json.dump(summary, f, indent=2)

    # Log test results to wandb
    if use_wandb:
        # Log overall metrics
        wandb_logger.log(
            {
                "test/accuracy": results["overall_accuracy"],
                "test/f1_score_macro": results["f1_score_macro"],
            }
        )

        # Log benchmark results to wandb
        for batch_size_str, metrics in benchmark_results.items():
            wandb_logger.log(
                {
                    f"benchmark/batch_size_{batch_size_str}/images_per_second": metrics[
                        "images_per_second"
                    ],
                    f"benchmark/batch_size_{batch_size_str}/ms_per_batch": metrics[
                        "ms_per_batch"
                    ],
                    f"benchmark/batch_size_{batch_size_str}/ms_per_image": metrics[
                        "ms_per_image"
                    ],
                }
            )

        # Log summary metrics
        wandb_logger.log_summary(
            {
                "test_accuracy": results["overall_accuracy"],
                "test_f1_score_macro": results["f1_score_macro"],
                "benchmark_images_per_second_batch_64": benchmark_results["64"][
                    "images_per_second"
                ],
            }
        )

        # Log per-class accuracy as a table for better organization
        per_class_acc = results["per_class_accuracy"]
        if isinstance(per_class_acc, dict):
            table_data = [
                [class_name, acc] for class_name, acc in per_class_acc.items()
            ]
            wandb_logger.log_table(
                key="test/per_class_accuracy",
                columns=["Class", "Accuracy"],
                data=table_data,
            )

            # Also log individual metrics for filtering
            for class_name, acc in per_class_acc.items():
                wandb_logger.log({f"test/class_accuracy/{class_name}": acc})

    # Save misclassified images
    print("\nSaving misclassified images...")
    misclassified_dir = output_dir / MISCLASSIFIED_DIR
    if misclassified_dir.exists():
        shutil.rmtree(misclassified_dir)
    misclassified_dir.mkdir(parents=True)

    # Find misclassified samples
    predictions_array = np.array(predictions)
    labels_array = np.array(labels)
    misclassified_indices = np.where(predictions_array != labels_array)[0]

    # Limit the number of misclassified images to save
    num_to_save = min(len(misclassified_indices), MAX_MISCLASSIFIED_IMAGES)
    print(
        f"Saving {num_to_save} of {len(misclassified_indices)} misclassified images..."
    )

    for idx in misclassified_indices[:num_to_save]:
        true_label_idx = int(labels_array[idx])
        pred_label_idx = int(predictions_array[idx])
        true_label = class_names[true_label_idx]
        predicted_label = class_names[pred_label_idx]

        # Get the original image based on dataset type
        if isinstance(
            test_dataset,
            (HuggingFaceChessPiecesDataset, ConcatenatedHuggingFaceDataset),
        ):
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
        for i, idx in enumerate(misclassified_indices[:max_samples]):
            true_label_idx = int(labels_array[idx])
            pred_label_idx = int(predictions_array[idx])
            true_label = class_names[true_label_idx]
            predicted_label = class_names[pred_label_idx]

            # Get image path based on dataset type
            if isinstance(
                test_dataset,
                (HuggingFaceChessPiecesDataset, ConcatenatedHuggingFaceDataset),
            ):
                # For HuggingFace datasets, use the saved misclassified image
                image_name = f"{idx}.png"
                new_filename = f"true_{true_label}_pred_{predicted_label}_{image_name}"
                image_path = misclassified_dir / new_filename
            else:
                # For local datasets, use the original file path
                image_path = Path(test_dataset.image_files[idx])

            # Add to table data (collect first, then log as table)
            if i == 0:
                # Initialize table data list on first iteration
                wandb_table_data = []

            if image_path.exists():
                wandb_table_data.append(
                    [
                        i,
                        wandb_logger.wandb.Image(str(image_path))
                        if wandb_logger.wandb
                        else str(image_path),
                        true_label,
                        predicted_label,
                    ]
                )

        # Log as a table for better organization
        if "wandb_table_data" in locals() and wandb_table_data and wandb_logger.wandb:
            wandb_logger.log_table(
                key="test/misclassified_samples",
                columns=["Index", "Image", "True Label", "Predicted Label"],
                data=wandb_table_data,
            )
            print(f"Logged {len(wandb_table_data)} misclassified images to wandb table")

        # Log evaluation artifacts
        overall_acc = results["overall_accuracy"]
        f1_macro = results["f1_score_macro"]
        assert isinstance(overall_acc, float)
        assert isinstance(f1_macro, float)

        wandb_logger.log_artifact(
            artifact_path=output_dir,
            artifact_type="evaluation",
            name=f"chess-cv-{model_id}-evaluation",
            aliases=["latest", "test-results"],
            metadata={
                "test_accuracy": overall_acc,
                "test_f1_score_macro": f1_macro,
                "num_test_samples": len(test_dataset),
                "num_misclassified": len(misclassified_indices),
            },
        )

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    if use_wandb:
        print("Results logged to wandb")
    else:
        print(f"Results saved to: {output_dir}")

    # Finish wandb run
    wandb_logger.finish()
