"""Test script for evaluating trained model."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = ["test"]

import mlx.core as mx
import mlx.nn as nn
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
    get_image_files,
    get_label_map_from_class_names,
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


@dataclass
class TestConfig:
    """Configuration for model testing."""

    model_id: str
    num_classes: int
    class_names: list[str]
    test_dir: Path
    checkpoint_path: Path
    output_dir: Path
    batch_size: int
    image_size: int
    num_workers: int
    hf_test_dir: str | None
    concat_splits: bool


def _load_test_dataset(
    config: TestConfig,
    label_map: dict[str, int],
    transform: transforms.Compose,
) -> tuple[
    ChessPiecesDataset | HuggingFaceChessPiecesDataset | ConcatenatedHuggingFaceDataset,
    str,
]:
    """Load test dataset from HuggingFace or local directory.

    Args:
        config: Test configuration
        label_map: Dictionary mapping label names to integers
        transform: Torchvision transforms to apply

    Returns:
        Tuple of (dataset, description_string)
    """
    if config.hf_test_dir is not None:
        if config.concat_splits:
            print(
                f"Loading test data from HuggingFace dataset (all splits): {config.hf_test_dir}"
            )
            dataset = ConcatenatedHuggingFaceDataset(
                config.hf_test_dir, label_map, splits=None, transform=transform
            )
            description = f"HuggingFace (all splits): {config.hf_test_dir}"
        else:
            print(f"Loading test data from HuggingFace dataset: {config.hf_test_dir}")
            dataset = HuggingFaceChessPiecesDataset(
                config.hf_test_dir, label_map, split="train", transform=transform
            )
            description = f"HuggingFace: {config.hf_test_dir}"
    else:
        print(f"Loading test data from local directory: {config.test_dir}")
        test_files = get_image_files(str(config.test_dir))
        dataset = ChessPiecesDataset(test_files, label_map, transform=transform)
        description = f"Local: {config.test_dir}"

    return dataset, description


def _load_model(config: TestConfig) -> nn.Module:
    """Load trained model from checkpoint.

    Args:
        config: Test configuration

    Returns:
        Loaded MLX model

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {config.checkpoint_path}\n"
            "Please train the model first using: chess-cv train pieces"
        )

    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = create_model(num_classes=config.num_classes)
    print(f"Loading checkpoint from: {config.checkpoint_path}")

    checkpoint = mx.load(str(config.checkpoint_path))
    checkpoint_items = list(checkpoint.items())  # type: ignore[attr-defined]
    model.update(tree_unflatten(checkpoint_items))
    mx.eval(model.parameters())

    print("✓ Model loaded successfully")
    return model


def _get_image_from_dataset(
    dataset: ChessPiecesDataset
    | HuggingFaceChessPiecesDataset
    | ConcatenatedHuggingFaceDataset,
    idx: int,
) -> tuple[Image.Image, str]:
    """Get original image from dataset by index.

    Args:
        dataset: Test dataset (local or HuggingFace)
        idx: Sample index

    Returns:
        Tuple of (PIL Image, image_name)
    """
    if isinstance(
        dataset, (HuggingFaceChessPiecesDataset, ConcatenatedHuggingFaceDataset)
    ):
        # For HuggingFace datasets
        item = dataset.dataset[idx]  # type: ignore[index]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")  # type: ignore[arg-type]
        else:
            img = img.convert("RGB")
        image_name = f"{idx}.png"
    else:
        # For local datasets
        image_path = Path(dataset.image_files[idx])  # type: ignore[attr-defined]
        img = Image.open(image_path).convert("RGB")
        image_name = image_path.name

    return img, image_name


def _save_misclassified_images(
    dataset: ChessPiecesDataset
    | HuggingFaceChessPiecesDataset
    | ConcatenatedHuggingFaceDataset,
    predictions: list[int],
    labels: list[int],
    class_names: list[str],
    output_dir: Path,
) -> tuple[Path, int]:
    """Save misclassified images to disk.

    Args:
        dataset: Test dataset
        predictions: Model predictions
        labels: True labels
        class_names: List of class names
        output_dir: Output directory

    Returns:
        Tuple of (misclassified_dir, num_saved)
    """
    misclassified_dir = output_dir / MISCLASSIFIED_DIR
    if misclassified_dir.exists():
        shutil.rmtree(misclassified_dir)
    misclassified_dir.mkdir(parents=True)

    # Find misclassified samples
    predictions_array = np.array(predictions)
    labels_array = np.array(labels)
    misclassified_indices = np.where(predictions_array != labels_array)[0]

    # Limit the number to save
    num_to_save = min(len(misclassified_indices), MAX_MISCLASSIFIED_IMAGES)
    print(
        f"Saving {num_to_save} of {len(misclassified_indices)} misclassified images..."
    )

    for idx in misclassified_indices[:num_to_save]:
        true_label = class_names[int(labels_array[idx])]
        predicted_label = class_names[int(predictions_array[idx])]

        img, image_name = _get_image_from_dataset(dataset, idx)

        # Save with descriptive name
        new_filename = f"true_{true_label}_pred_{predicted_label}_{image_name}"
        img.save(misclassified_dir / new_filename)

    return misclassified_dir, num_to_save


def _save_results(
    config: TestConfig,
    results: dict,
    confusion_matrix: np.ndarray,
    benchmark_results: dict,
    dataset_size: int,
    dataset_description: str,
) -> None:
    """Save all evaluation results (JSON, plots).

    Args:
        config: Test configuration
        results: Evaluation results dictionary
        confusion_matrix: Confusion matrix
        benchmark_results: Inference benchmark results
        dataset_size: Number of test samples
        dataset_description: Description of test dataset
    """
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save JSON summary
    print(f"Saving test summary to: {config.output_dir / TEST_SUMMARY_FILENAME}")
    summary = {
        "overall_accuracy": results["overall_accuracy"],
        "f1_score_macro": results["f1_score_macro"],
        "per_class_accuracy": results["per_class_accuracy"],
        "checkpoint_path": str(config.checkpoint_path),
        "test_dir": dataset_description,
        "num_test_samples": dataset_size,
        "inference_benchmark": benchmark_results,
    }
    with open(config.output_dir / TEST_SUMMARY_FILENAME, "w") as f:
        json.dump(summary, f, indent=2)

    # Save plots
    plot_confusion_matrix(
        confusion_matrix,
        class_names=config.class_names,
        output_dir=config.output_dir,
        filename=TEST_CONFUSION_MATRIX_FILENAME,
    )
    plot_per_class_accuracy(
        results["per_class_accuracy"],  # type: ignore[arg-type]
        output_dir=config.output_dir,
        filename=TEST_PER_CLASS_ACCURACY_FILENAME,
    )


def _print_benchmark_results(benchmark_results: dict) -> None:
    """Print formatted benchmark results.

    Args:
        benchmark_results: Dictionary of benchmark results by batch size
    """
    print("\n" + "=" * 60)
    print("BENCHMARKING INFERENCE SPEED")
    print("=" * 60)
    print("Testing inference speed for batch sizes: 1, 64, 512, 1024")
    print("Running warmup and measurement iterations...")
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


def _log_misclassified_images_to_wandb(
    wandb_logger: WandbLogger,
    dataset: ChessPiecesDataset
    | HuggingFaceChessPiecesDataset
    | ConcatenatedHuggingFaceDataset,
    predictions: list[int],
    labels: list[int],
    class_names: list[str],
    misclassified_dir: Path,
    max_samples: int = 20,
) -> None:
    """Log sample misclassified images to wandb as a table.

    Args:
        wandb_logger: WandbLogger instance
        dataset: Test dataset
        predictions: Model predictions
        labels: True labels
        class_names: List of class names
        misclassified_dir: Directory with misclassified images
        max_samples: Maximum number of samples to log
    """
    if not wandb_logger.wandb:
        return

    print("Logging sample misclassified images to wandb...")

    predictions_array = np.array(predictions)
    labels_array = np.array(labels)
    misclassified_indices = np.where(predictions_array != labels_array)[0]

    max_samples = min(max_samples, len(misclassified_indices))
    table_data = []

    for i, idx in enumerate(misclassified_indices[:max_samples]):
        true_label = class_names[int(labels_array[idx])]
        predicted_label = class_names[int(predictions_array[idx])]

        # Get image path based on dataset type
        if isinstance(
            dataset, (HuggingFaceChessPiecesDataset, ConcatenatedHuggingFaceDataset)
        ):
            image_name = f"{idx}.png"
            image_path = (
                misclassified_dir
                / f"true_{true_label}_pred_{predicted_label}_{image_name}"
            )
        else:
            image_path = Path(dataset.image_files[idx])  # type: ignore[attr-defined]

        if image_path.exists():
            table_data.append(
                [
                    i,
                    wandb_logger.wandb.Image(str(image_path)),
                    true_label,
                    predicted_label,
                ]
            )

    if table_data:
        wandb_logger.log_table(
            key="test/misclassified_samples",
            columns=["Index", "Image", "True Label", "Predicted Label"],
            data=table_data,
        )
        print(f"Logged {len(table_data)} misclassified images to wandb table")


def _log_to_wandb(
    wandb_logger: WandbLogger,
    config: TestConfig,
    results: dict,
    benchmark_results: dict,
    dataset: ChessPiecesDataset
    | HuggingFaceChessPiecesDataset
    | ConcatenatedHuggingFaceDataset,
    dataset_description: str,
    misclassified_dir: Path,
    num_misclassified: int,
) -> None:
    """Log all results to Weights & Biases.

    Args:
        wandb_logger: WandbLogger instance
        config: Test configuration
        results: Evaluation results
        benchmark_results: Inference benchmark results
        dataset: Test dataset
        dataset_description: Description of test dataset
        misclassified_dir: Directory containing misclassified images
        num_misclassified: Total number of misclassified samples
    """
    if not wandb_logger.enabled:
        return

    predictions = results["predictions"]
    labels = results["labels"]

    # Log overall metrics
    wandb_logger.log(
        {
            "test/accuracy": results["overall_accuracy"],
            "test/f1_score_macro": results["f1_score_macro"],
        }
    )

    # Log benchmark results
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

    # Log per-class accuracy as table
    per_class_acc = results["per_class_accuracy"]
    if isinstance(per_class_acc, dict):
        table_data = [[class_name, acc] for class_name, acc in per_class_acc.items()]
        wandb_logger.log_table(
            key="test/per_class_accuracy",
            columns=["Class", "Accuracy"],
            data=table_data,
        )

        # Also log individual metrics for filtering
        for class_name, acc in per_class_acc.items():
            wandb_logger.log({f"test/class_accuracy/{class_name}": acc})

    # Log sample misclassified images
    _log_misclassified_images_to_wandb(
        wandb_logger,
        dataset,
        predictions,
        labels,
        config.class_names,
        misclassified_dir,
    )

    # Log evaluation artifacts
    wandb_logger.log_artifact(
        artifact_path=config.output_dir,
        artifact_type="evaluation",
        name=f"chess-cv-{config.model_id}-evaluation",
        aliases=["latest", "test-results"],
        metadata={
            "test_accuracy": results["overall_accuracy"],
            "test_f1_score_macro": results["f1_score_macro"],
            "num_test_samples": len(dataset),
            "num_misclassified": num_misclassified,
        },
    )


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
    )

    # Build configuration
    model_config = get_model_config(model_id)
    config = TestConfig(
        model_id=model_id,
        num_classes=model_config["num_classes"],
        class_names=model_config["class_names"],
        test_dir=Path(test_dir) if test_dir else get_test_dir(model_id),
        checkpoint_path=Path(checkpoint_path)
        if checkpoint_path
        else get_checkpoint_dir(model_id) / get_model_filename(model_id),
        output_dir=Path(output_dir) if output_dir else get_output_dir(model_id),
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        hf_test_dir=hf_test_dir,
        concat_splits=concat_splits,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_logger = WandbLogger(enabled=use_wandb)
    if use_wandb:
        wandb_logger.init(
            project=f"chess-cv-{model_id}-evaluation",
            config={
                "model_id": model_id,
                "num_classes": config.num_classes,
                "test_dir": str(config.test_dir)
                if hf_test_dir is None
                else hf_test_dir,
                "checkpoint_path": str(config.checkpoint_path),
                "batch_size": batch_size,
                "image_size": image_size,
                "num_workers": num_workers,
                "hf_test_dir": hf_test_dir,
                "concat_splits": concat_splits,
            },
        )

    # Load model
    try:
        model = _load_model(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load test dataset
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)

    label_map = get_label_map_from_class_names(config.class_names)
    test_transforms = transforms.Compose(
        [transforms.Resize((image_size, image_size), antialias=True)]
    )

    test_dataset, dataset_description = _load_test_dataset(
        config, label_map, test_transforms
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

    # Evaluate model
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    results = evaluate_model(
        model, test_loader, class_names=config.class_names, batch_size=batch_size
    )

    # Compute additional metrics
    print("\nComputing confusion matrix and F1 score...")
    confusion_matrix = compute_confusion_matrix(
        model, test_loader, num_classes=config.num_classes
    )
    predictions = results["predictions"]
    labels = results["labels"]
    assert isinstance(predictions, list)
    assert isinstance(labels, list)
    results["f1_score_macro"] = compute_f1_score(labels, predictions)

    # Print results
    eval_results = {
        "overall_accuracy": results["overall_accuracy"],
        "per_class_accuracy": results["per_class_accuracy"],
        "f1_score_macro": results["f1_score_macro"],
    }
    print_evaluation_results(eval_results, class_names=config.class_names)

    # Benchmark inference speed
    benchmark_results = benchmark_inference_speed(
        model=model,
        image_size=image_size,
        batch_sizes=[1, 64, 512, 1024],
        num_warmup=10,
        num_iterations=50,
    )
    _print_benchmark_results(benchmark_results)

    # Save misclassified images
    print("\nSaving misclassified images...")
    predictions_list = results["predictions"]
    labels_list = results["labels"]
    assert isinstance(predictions_list, list)
    assert isinstance(labels_list, list)
    misclassified_dir, _ = _save_misclassified_images(
        test_dataset,
        predictions_list,
        labels_list,
        config.class_names,
        config.output_dir,
    )

    num_misclassified = len(
        np.where(np.array(results["predictions"]) != np.array(results["labels"]))[0]
    )

    # Save results
    _save_results(
        config,
        results,
        confusion_matrix,
        benchmark_results,
        len(test_dataset),
        dataset_description,
    )

    # Log to wandb
    _log_to_wandb(
        wandb_logger,
        config,
        results,
        benchmark_results,
        test_dataset,
        dataset_description,
        misclassified_dir,
        num_misclassified,
    )

    # Finish
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    if use_wandb:
        print("Results logged to wandb")
    else:
        print(f"Results saved to: {config.output_dir}")

    wandb_logger.finish()
