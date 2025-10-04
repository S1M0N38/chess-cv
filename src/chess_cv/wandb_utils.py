"""Utilities for optional Weights & Biases integration."""

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["WandbLogger"]


class WandbLogger:
    """Wrapper for optional W&B logging.

    Provides a consistent API whether wandb is enabled or not.
    When disabled, all methods become no-ops.
    """

    def __init__(self, enabled: bool = False):
        """Initialize the wandb logger.

        Args:
            enabled: Whether to enable wandb logging
        """
        self.enabled = enabled
        self.run = None

        if self.enabled:
            try:
                import wandb

                self.wandb = wandb
            except ImportError:
                print(
                    "Warning: wandb is not installed. "
                    "Install it with: uv pip install wandb"
                )
                self.enabled = False
                self.wandb = None

    def init(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize a wandb run.

        Args:
            project: Project name
            config: Configuration dictionary (hyperparameters)
            name: Run name (optional)
            tags: List of tags (optional)
        """
        if not self.enabled:
            return

        self.run = self.wandb.init(
            project=project,
            config=config or {},
            name=name,
            tags=tags,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (epoch, iteration, etc.)
        """
        if not self.enabled or self.run is None:
            return

        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics)

    def log_image(
        self,
        key: str,
        image: np.ndarray | Path | str,
        caption: str | None = None,
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log an image to wandb.

        Args:
            key: Image name/key
            image: Image as numpy array or path to image file
            caption: Optional caption
            step: Optional step number
            commit: Whether to commit the log (increment step counter)
        """
        if not self.enabled or self.run is None:
            return

        log_dict = {
            key: self.wandb.Image(
                str(image) if isinstance(image, (Path, str)) else image, caption=caption
            )
        }

        if step is not None:
            self.wandb.log(log_dict, step=step, commit=commit)
        else:
            self.wandb.log(log_dict, commit=commit)

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str],
        title: str = "Confusion Matrix",
    ) -> None:
        """Log a confusion matrix to wandb.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Title for the confusion matrix
        """
        if not self.enabled or self.run is None:
            return

        self.wandb.log(
            {
                title: self.wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names,
                )
            }
        )

    def log_bar_chart(
        self,
        data: dict[str, float],
        title: str,
        x_label: str = "Class",
        y_label: str = "Value",
    ) -> None:
        """Log a bar chart to wandb.

        Args:
            data: Dictionary mapping labels to values
            title: Title for the chart
            x_label: Label for x-axis
            y_label: Label for y-axis
        """
        if not self.enabled or self.run is None:
            return

        # Create a wandb Table for bar chart
        table = self.wandb.Table(
            data=[[k, v] for k, v in data.items()], columns=[x_label, y_label]
        )
        self.wandb.log(
            {title: self.wandb.plot.bar(table, x_label, y_label, title=title)}
        )

    def log_model(
        self,
        model_path: Path | str,
        name: str = "best_model",
        aliases: list[str] | None = None,
    ) -> None:
        """Log a model artifact to wandb.

        Args:
            model_path: Path to the model file
            name: Name for the model artifact
            aliases: List of aliases (e.g., ["best", "latest"])
        """
        if not self.enabled or self.run is None:
            return

        artifact = self.wandb.Artifact(name, type="model")
        artifact.add_file(str(model_path))
        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self) -> None:
        """Finish the wandb run."""
        if not self.enabled or self.run is None:
            return

        self.wandb.finish()
