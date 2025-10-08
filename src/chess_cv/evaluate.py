"""Evaluation utilities for model performance."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

__all__ = [
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_per_class_accuracy",
    "evaluate_model",
    "print_evaluation_results",
    "compute_f1_score",
    "benchmark_inference_speed",
]


def compute_accuracy(model: nn.Module, images: mx.array, labels: mx.array) -> float:
    """Compute accuracy on a dataset.

    Args:
        model: Trained model
        images: Images array of shape (N, H, W, C)
        labels: Labels array of shape (N,)

    Returns:
        Accuracy as a float between 0 and 1
    """
    logits = model(images)
    predictions = mx.argmax(logits, axis=1)
    correct = mx.sum(predictions == labels)  # type: ignore[arg-type]
    accuracy = correct / len(labels)
    return accuracy.item()


def compute_per_class_accuracy(
    model: nn.Module,
    images: mx.array,
    labels: mx.array,
    class_names: list[str],
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute per-class accuracy.

    Args:
        model: Trained model
        images: Images array of shape (N, H, W, C)
        labels: Labels array of shape (N,)
        class_names: List of class names
        num_classes: Number of classes (if None, inferred from class_names)

    Returns:
        Dictionary mapping class names to accuracy values
    """
    if num_classes is None:
        num_classes = len(class_names)

    logits = model(images)
    predictions = mx.argmax(logits, axis=1)

    per_class_acc = {}
    for class_idx in range(num_classes):
        # Find samples belonging to this class
        class_mask = labels == class_idx  # type: ignore[assignment]
        class_samples = mx.sum(class_mask)  # type: ignore[arg-type]

        if class_samples > 0:
            # Compute accuracy for this class
            class_correct = mx.sum((predictions == labels) & class_mask)  # type: ignore[arg-type,operator]
            class_accuracy = class_correct / class_samples
            per_class_acc[class_names[class_idx]] = class_accuracy.item()
        else:
            per_class_acc[class_names[class_idx]] = 0.0

    return per_class_acc


def compute_confusion_matrix(
    model: nn.Module, data_loader: DataLoader, num_classes: int = 13
) -> np.ndarray:
    """Compute confusion matrix using scikit-learn.

    Args:
        model: Trained model
        data_loader: DataLoader for test data
        num_classes: Number of classes

    Returns:
        Confusion matrix as numpy array of shape (num_classes, num_classes)
    """
    from tqdm import tqdm

    all_predictions = []
    all_labels = []

    for batch_images, batch_labels in tqdm(data_loader, desc="Computing predictions"):
        logits = model(batch_images)
        predictions = mx.argmax(logits, axis=1)

        all_predictions.extend(np.array(predictions).tolist())
        all_labels.extend(np.array(batch_labels).tolist())

    return confusion_matrix(
        all_labels, all_predictions, labels=list(range(num_classes))
    )


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: list[str],
    batch_size: int = 256,
) -> dict[str, float | dict[str, float] | list[int]]:
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        class_names: List of class names
        batch_size: Batch size for evaluation (unused, kept for compatibility)

    Returns:
        Dictionary containing overall accuracy, per-class accuracies, predictions, and labels
    """
    all_predictions = []
    all_labels = []

    for batch_images, batch_labels in data_loader:
        logits = model(batch_images)
        predictions = mx.argmax(logits, axis=1)

        all_predictions.extend(np.array(predictions).tolist())
        all_labels.extend(np.array(batch_labels).tolist())

    # Overall accuracy
    correct = sum(pred == label for pred, label in zip(all_predictions, all_labels))
    overall_accuracy = correct / len(all_labels)

    # Per-class accuracy using scikit-learn
    per_class_acc = {}
    for class_idx, class_name in enumerate(class_names):
        class_mask = [label == class_idx for label in all_labels]
        if sum(class_mask) > 0:
            class_correct = sum(
                pred == label
                for pred, label, mask in zip(all_predictions, all_labels, class_mask)
                if mask
            )
            per_class_acc[class_name] = class_correct / sum(class_mask)
        else:
            per_class_acc[class_name] = 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_acc,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def print_evaluation_results(
    results: dict[str, float | dict[str, float]], class_names: list[str]
) -> None:
    """Pretty print evaluation results.

    Args:
        results: Dictionary from evaluate_model
        class_names: List of class names
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    overall_acc = results["overall_accuracy"]
    assert isinstance(overall_acc, float)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")

    f1_score = results.get("f1_score_macro")
    if f1_score:
        assert isinstance(f1_score, float)
        print(f"Macro F1-Score:   {f1_score:.4f}")

    print("\nPer-Class Accuracy:")
    print("-" * 60)
    per_class = results["per_class_accuracy"]
    assert isinstance(per_class, dict)
    for class_name in class_names:
        acc = per_class[class_name]
        print(f"  {class_name:20s}: {acc:.4f}")

    print("=" * 60 + "\n")


def compute_f1_score(y_true: list[int], y_pred: list[int]) -> float:
    """Compute macro F1-score using scikit-learn.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Macro F1-score
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division="warn"))


def benchmark_inference_speed(
    model: nn.Module,
    image_size: int = 32,
    batch_sizes: list[int] | None = None,
    num_warmup: int = 10,
    num_iterations: int = 50,
) -> dict[str, dict[str, float]]:
    """Benchmark model inference speed for various batch sizes.

    Args:
        model: Trained model to benchmark
        image_size: Size of input images (default: 32)
        batch_sizes: List of batch sizes to test (default: [1, 64, 512, 1024])
        num_warmup: Number of warmup iterations (default: 10)
        num_iterations: Number of iterations for measurement (default: 50)

    Returns:
        Dictionary mapping batch size to performance metrics:
        {
            "1": {
                "images_per_second": 1234.56,
                "ms_per_batch": 0.81,
                "ms_per_image": 0.81
            },
            ...
        }
    """
    import time

    if batch_sizes is None:
        batch_sizes = [1, 64, 512, 1024]

    results = {}

    for batch_size in batch_sizes:
        # Create dummy input data (batch_size, height, width, channels)
        # MLX uses NHWC format
        dummy_input = mx.random.uniform(
            shape=(batch_size, image_size, image_size, 3), dtype=mx.float32
        )

        # Warmup phase - ensure model is compiled and caches are warm
        for _ in range(num_warmup):
            _ = model(dummy_input)
            mx.eval(dummy_input)  # Force evaluation

        # Measurement phase
        times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            logits = model(dummy_input)
            mx.eval(logits)  # Force evaluation to ensure computation is complete
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        # Calculate statistics
        avg_time = sum(times) / len(times)
        ms_per_batch = avg_time * 1000
        ms_per_image = ms_per_batch / batch_size
        images_per_second = batch_size / avg_time

        results[str(batch_size)] = {
            "images_per_second": round(images_per_second, 2),
            "ms_per_batch": round(ms_per_batch, 4),
            "ms_per_image": round(ms_per_image, 4),
        }

    return results
