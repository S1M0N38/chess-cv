"""Utilities for optional Weights & Biases integration."""

import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

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
        self.wandb: types.ModuleType | None

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
        else:
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
        if not self.enabled or self.wandb is None:
            return

        self.run = self.wandb.init(
            project=project,
            config=config or {},
            name=name,
            tags=tags,
        )

    def define_metrics(self) -> None:
        """Define custom metrics and their summary statistics.

        This defines summary statistics for key metrics. WandB automatically
        groups metrics with the same prefix (e.g., loss/* and accuracy/*) on the same graph.
        """
        if not self.enabled or self.run is None or self.wandb is None:
            return

        # Define global_step as the primary x-axis for step-based metrics
        self.run.define_metric("global_step")

        # Define epoch as the x-axis for epoch-based metrics
        self.run.define_metric("epoch")

        # Step-based training metrics (logged during epoch)
        self.run.define_metric("loss/train_step", step_metric="global_step")
        self.run.define_metric("accuracy/train_step", step_metric="global_step")

        # Epoch-based metrics (logged at end of epoch)
        self.run.define_metric("loss/train", step_metric="epoch", summary="min")
        self.run.define_metric("loss/val", step_metric="epoch", summary="min")
        self.run.define_metric(
            "accuracy/train", step_metric="epoch", summary="max,last"
        )
        self.run.define_metric("accuracy/val", step_metric="epoch", summary="max,last")

    def update_config(self, config: dict[str, Any]) -> None:
        """Update the run configuration.

        Args:
            config: Dictionary of configuration values to update
        """
        if not self.enabled or self.run is None or self.wandb is None:
            return

        self.run.config.update(config)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (epoch, iteration, etc.)
        """
        if not self.enabled or self.run is None or self.wandb is None:
            return

        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)

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
        if not self.enabled or self.run is None or self.wandb is None:
            return

        log_dict = {
            key: self.wandb.Image(
                str(image) if isinstance(image, (Path, str)) else image, caption=caption
            )
        }

        if step is not None:
            self.run.log(log_dict, step=step, commit=commit)
        else:
            self.run.log(log_dict, commit=commit)

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
        if not self.enabled or self.run is None or self.wandb is None:
            return

        self.run.log(
            {
                title: self.wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true.tolist(),
                    preds=y_pred.tolist(),
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
        if not self.enabled or self.run is None or self.wandb is None:
            return

        # Create a wandb Table for bar chart
        table = self.wandb.Table(
            data=[[k, v] for k, v in data.items()], columns=[x_label, y_label]
        )
        self.run.log({title: self.wandb.plot.bar(table, x_label, y_label, title=title)})

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
        if not self.enabled or self.run is None or self.wandb is None:
            return

        artifact = self.wandb.Artifact(name, type="model")
        artifact.add_file(str(model_path))
        self.run.log_artifact(artifact, aliases=aliases)

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """Log a table to wandb.

        Args:
            key: Table name/key
            columns: List of column names
            data: List of rows, where each row is a list of values
        """
        if not self.enabled or self.run is None or self.wandb is None:
            return

        table = self.wandb.Table(columns=columns, data=data)
        self.run.log({key: table})

    def log_artifact(
        self,
        artifact_path: Path | str,
        artifact_type: str = "model",
        name: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an artifact to wandb with metadata.

        Args:
            artifact_path: Path to the artifact file or directory
            artifact_type: Type of artifact (e.g., "model", "dataset", "evaluation")
            name: Name for the artifact (defaults to filename)
            aliases: List of aliases (e.g., ["best", "latest", "production"])
            metadata: Dictionary of metadata to attach to the artifact
        """
        if not self.enabled or self.run is None or self.wandb is None:
            return

        if name is None:
            name = Path(artifact_path).stem

        artifact = self.wandb.Artifact(
            name, type=artifact_type, metadata=metadata or {}
        )

        # Add file or directory
        if Path(artifact_path).is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path))

        self.run.log_artifact(artifact, aliases=aliases or ["latest"])

    def create_alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
        wait_duration: int = 0,
    ) -> None:
        """Create an alert in wandb.

        Args:
            title: Alert title
            text: Alert message
            level: Alert level ("INFO", "WARN", or "ERROR")
            wait_duration: Minimum seconds to wait between alerts (default: 0)
        """
        if not self.enabled or self.run is None or self.wandb is None:
            return

        self.wandb.alert(
            title=title,
            text=text,
            level=getattr(self.wandb.AlertLevel, level, self.wandb.AlertLevel.INFO),
            wait_duration=wait_duration,
        )

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log summary metrics that appear at the top of the run page.

        Args:
            summary: Dictionary of summary metrics
        """
        if not self.enabled or self.run is None or self.wandb is None:
            return

        for key, value in summary.items():
            self.run.summary[key] = value

    def finish(self) -> None:
        """Finish the wandb run."""
        if not self.enabled or self.run is None or self.wandb is None:
            return

        self.wandb.finish()
