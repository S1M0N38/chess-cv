"""Programmatic W&B sweep runner for hyperparameter optimization."""

from pathlib import Path

__all__ = ["run_sweep"]


def run_sweep(model_id: str, sweep_config_path: Path | str | None = None) -> None:
    """Run W&B sweep programmatically.

    This function:
    1. Loads the sweep configuration from YAML
    2. Sets up the program and command for the sweep
    3. Initializes the sweep on W&B
    4. Runs the sweep agent (blocking until stopped)

    Args:
        model_id: Model identifier (e.g., 'pieces', 'arrows')
        sweep_config_path: Path to sweep config YAML file.
                          Defaults to sweeps/{model_id}.yaml

    Raises:
        FileNotFoundError: If sweep config file doesn't exist
        ImportError: If wandb or yaml packages are not installed
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for sweep functionality. "
            "Install it with: pip install pyyaml"
        )

    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is required for sweep functionality. "
            "Install it with: pip install wandb"
        )

    # Determine sweep config path
    if sweep_config_path is None:
        sweep_config_path = Path(f"sweeps/{model_id}.yaml")
    else:
        sweep_config_path = Path(sweep_config_path)

    # Check if config file exists
    if not sweep_config_path.exists():
        raise FileNotFoundError(
            f"Sweep configuration not found: {sweep_config_path}\n"
            f"Available configs should be in sweeps/ directory."
        )

    # Load sweep configuration
    print(f"Loading sweep configuration from {sweep_config_path}...")
    with open(sweep_config_path) as f:
        sweep_config = yaml.safe_load(f)

    # Determine num_epochs based on model_id (not swept, kept constant)
    model_epochs = {
        "pieces": 200,
        "arrows": 5,
    }
    num_epochs = model_epochs.get(model_id, 200)  # Default to 200 if unknown model

    # Set command to use modern CLI approach
    # The sweep will call: chess-cv train {model_id} {sweep_params} --num-epochs {num_epochs} --wandb
    sweep_config["command"] = [
        "chess-cv",
        "train",
        model_id,
        "${args}",  # W&B will insert sweep parameters here
        f"--num-epochs={num_epochs}",
        "--wandb",
    ]

    # Initialize sweep
    print(f"\nInitializing sweep for {model_id} model...")
    print(f"Project: chess-cv-{model_id}")
    print(f"Method: {sweep_config.get('method', 'grid')}")
    print(f"Metric: {sweep_config.get('metric', {}).get('name', 'unknown')}")
    print(f"Run cap: {sweep_config.get('run_cap', 'unlimited')}")

    sweep_id = wandb.sweep(sweep_config, project=f"chess-cv-{model_id}")

    print(f"\n{'=' * 60}")
    print("SWEEP INITIALIZED")
    print(f"{'=' * 60}")
    print(f"Sweep ID: {sweep_id}")
    print(
        f"URL: https://wandb.ai/{wandb.Api().default_entity}/chess-cv-{model_id}/sweeps/{sweep_id}"
    )
    print("\nStarting sweep agent...")
    print(f"{'=' * 60}\n")

    # Run agent (blocking - will continue until stopped or run_cap reached)
    try:
        wandb.agent(sweep_id, project=f"chess-cv-{model_id}")
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
        print(f"Sweep ID: {sweep_id}")
        print("You can resume this sweep later with:")
        print(
            f"  wandb agent {wandb.Api().default_entity}/chess-cv-{model_id}/{sweep_id}"
        )
