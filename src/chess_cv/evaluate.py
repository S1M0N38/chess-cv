"""Evaluation utilities for model performance."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from .data import CLASS_NAMES

__all__ = [
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_per_class_accuracy",
    "evaluate_model",
    "print_evaluation_results",
    "compute_f1_score",
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
    model: nn.Module, images: mx.array, labels: mx.array, num_classes: int = 13
) -> dict[str, float]:
    """Compute per-class accuracy.

    Args:
        model: Trained model
        images: Images array of shape (N, H, W, C)
        labels: Labels array of shape (N,)
        num_classes: Number of classes

    Returns:
        Dictionary mapping class names to accuracy values
    """
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
            per_class_acc[CLASS_NAMES[class_idx]] = class_accuracy.item()
        else:
            per_class_acc[CLASS_NAMES[class_idx]] = 0.0

    return per_class_acc


def compute_confusion_matrix(
    model: nn.Module, images: mx.array, labels: mx.array, num_classes: int = 13
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        model: Trained model
        images: Images array of shape (N, H, W, C)
        labels: Labels array of shape (N,)
        num_classes: Number of classes

    Returns:
        Confusion matrix as numpy array of shape (num_classes, num_classes)
    """
    logits = model(images)
    predictions = mx.argmax(logits, axis=1)

    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Populate confusion matrix
    pred_np = np.array(predictions)
    labels_np = np.array(labels)

    for true_label, pred_label in zip(labels_np, pred_np):
        confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, batch_size: int = 256
) -> dict[str, float | dict[str, float]]:
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        batch_size: Batch size for evaluation

    Returns:
        Dictionary containing overall accuracy and per-class accuracies
    """
    all_predictions = []
    all_labels = []

    # Collect predictions in batches
    for batch_images, batch_labels in data_loader:
        logits = model(batch_images)
        predictions = mx.argmax(logits, axis=1)

        all_predictions.append(predictions)
        all_labels.append(batch_labels)

    # Concatenate all predictions and labels
    all_predictions = mx.concatenate(all_predictions)
    all_labels = mx.concatenate(all_labels)

    # Compute overall accuracy
    correct = mx.sum(all_predictions == all_labels)  # type: ignore[arg-type]
    overall_accuracy = (correct / len(all_labels)).item()

    # Compute per-class accuracy
    per_class_acc = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Find samples belonging to this class
        class_mask = all_labels == class_idx  # type: ignore[assignment]
        class_samples = mx.sum(class_mask)  # type: ignore[arg-type]

        if class_samples > 0:
            # Compute accuracy for this class
            class_correct = mx.sum((all_predictions == all_labels) & class_mask)  # type: ignore[arg-type,operator]
            class_accuracy = (class_correct / class_samples).item()
            per_class_acc[class_name] = class_accuracy
        else:
            per_class_acc[class_name] = 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_acc,
    }


def print_evaluation_results(results: dict[str, float | dict[str, float]]) -> None:
    """Pretty print evaluation results.

    Args:
        results: Dictionary from evaluate_model
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
    for class_name in CLASS_NAMES:
        acc = per_class[class_name]
        print(f"  {class_name:20s}: {acc:.4f}")

    print("=" * 60 + "\n")


def compute_f1_score(confusion_matrix: np.ndarray) -> float:
    """Compute macro F1-score from a confusion matrix.

    Args:
        confusion_matrix: Confusion matrix as numpy array

    Returns:
        Macro F1-score
    """
    num_classes = confusion_matrix.shape[0]
    f1_scores = []

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
        else:
            f1_scores.append(0)

    macro_f1 = np.mean(f1_scores)
    return float(macro_f1)
