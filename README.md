<div align="center">

# Chess CV

<img src="docs/assets/model.svg" alt="Model Architecture" width="800">
<br>

*CNN-based chess piece classifier*

</div>

---

## Overview

A machine learning project that trains a lightweight CNN (156k parameters) from scratch to classify chess pieces from synthetically generated chessboard square images. The project combines 55 board styles (256×256px) with 64 piece sets (32×32px) from chess.com and lichess to generate a diverse training dataset of ~93,000 images. By rendering pieces onto different board backgrounds and extracting individual squares, the model learns robust piece recognition across various visual styles.

## Features

**Model Architecture**

- Lightweight CNN for 32x32px chess square images
- 13-class classification (6 white pieces, 6 black pieces, 1 empty)
- MLX framework for Apple Silicon optimization
- Data augmentation pipeline (rotation, flip, color jitter, noise)

**Development Tools**

- **MLX** – Apple's ML framework for efficient training on Mac
- **PyTorch/torchvision** – Data loading and augmentation
- **NumPy** – Numerical computing for data processing
- **Matplotlib** – Training curve visualization and analysis
- **Weights & Biases** – Experiment tracking and visualization (optional)
- **Ruff** – Fast Python linter and formatter
- **Basedpyright** – Static type checking
- **pytest** – Testing framework

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- MLX support (optimized for Apple Silicon Macs)

### Setup

```bash
# Clone the repository
git clone https://github.com/S1M0N38/chess-cv.git
cd chess-cv

# Install dependencies
uv sync
```

## Usage

### 1. Data Generation

Generate synthetic chess piece images by combining board styles with piece sets:

**How it works:**

The preprocessing script renders chess pieces onto board backgrounds to create training data:
- **Boards**: 55 board style images (256×256px) from chess.com and lichess
- **Pieces**: 64 piece set directories with individual piece images (32×32px)
- **Generation**: For each board-piece combination, pieces are composited onto the board and 32×32 squares are extracted
- **Output**: 26 images per combination (12 pieces × 2 square colors + 2 empty squares)
- **Total dataset**: ~93,000 images (65k train / 14k validation / 14k test)

```bash
# Using default settings (70% train, 15% val, 15% test)
python -m chess_cv.preprocessing

# Custom split ratios
python -m chess_cv.preprocessing \
  --train-dir data/train \
  --val-dir data/validate \
  --test-dir data/test \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

**Expected directory structure:**

```
data/
├── boards/           # Source: 55 board images (256x256px)
│   ├── chess-com-*.png
│   └── lichess-*.png
├── pieces/           # Source: 64 piece set directories
│   ├── chess-com-*/  # Each contains bB.png, bK.png, ..., wR.png
│   └── lichess-*/
├── train/            # Generated: ~65k images
│   ├── bB/
│   ├── bK/
│   ├── ...
│   └── xx/
├── validate/         # Generated: ~14k images
└── test/             # Generated: ~14k images
```

### 2. Training

Train the model with default settings:

```bash
python -m chess_cv.train
```

**Training options:**
```bash
python -m chess_cv.train \
  --train-dir data/train \
  --val-dir data/validate \
  --checkpoint-dir checkpoints \
  --batch-size 128 \
  --learning-rate 0.0002 \
  --weight-decay 0.0005 \
  --num-epochs 100 \
  --patience 15 \
  --image-size 32 \
  --num-workers 4
```

**Training features:**

- Early stopping with patience
- AdamW optimizer with weight decay
- Data augmentation (RandomResizedCrop, ColorJitter, Rotation, Gaussian Noise)
- Real-time training visualization
- Automatic checkpoint saving

**Output:**

- `checkpoints/best_model.safetensors` – Best model weights
- `checkpoints/optimizer.safetensors` – Optimizer state
- `outputs/training_curves.png` – Loss and accuracy plots
- `outputs/augmentation_example.png` – Example of data augmentation

### 3. Experiment Tracking with Weights & Biases (Optional)

For advanced experiment tracking and visualization, you can use Weights & Biases (wandb). When enabled with the `--wandb` flag, metrics are logged to the W&B dashboard instead of using matplotlib.

**Setup:**

```bash
# wandb is already installed as a dependency
# Login to your W&B account (first time only)
wandb login
```

**Training with W&B:**

```bash
# Train with wandb logging (disables matplotlib)
python -m chess_cv.train --wandb

# You can combine with other arguments
python -m chess_cv.train \
  --wandb \
  --num-epochs 150 \
  --learning-rate 0.0001 \
  --batch-size 64
```

**What gets logged to W&B:**

- Hyperparameters (batch size, learning rate, epochs, etc.)
- Training metrics (loss, accuracy) per epoch
- Validation metrics (loss, accuracy) per epoch
- Data augmentation examples
- Best model artifact
- Real-time metric visualization in W&B dashboard

**Testing with W&B:**

```bash
# Evaluate with wandb logging
python -m chess_cv.test --wandb
```

**What gets logged during testing:**

- Test accuracy and loss
- Per-class accuracy metrics
- Confusion matrix visualization
- Sample misclassified images (up to 20)

**Note:** When using `--wandb`, matplotlib visualization is automatically disabled to avoid redundancy.

#### Hyperparameter Optimization with W&B Sweeps

For automated hyperparameter tuning, use W&B Sweeps to run multiple training experiments with different configurations:

**What are Sweeps?**

W&B Sweeps automate hyperparameter search using Bayesian optimization, grid search, or random search. The included `sweep.yaml` configuration uses Bayesian optimization to tune:
- Batch size (32, 64, 128, 256)
- Learning rate (log-uniform: 1e-5 to 1e-3)
- Weight decay (log-uniform: 1e-5 to 1e-3)

**Running a Sweep:**

```bash
# 1. Initialize the sweep (creates a sweep ID)
wandb sweep sweep.yaml

# 2. Start one or more sweep agents (copy the sweep ID from step 1)
wandb agent <your-username>/<project-name>/<sweep-id>

# 3. (Optional) Run multiple agents in parallel for faster exploration
# In separate terminals or machines:
wandb agent <your-username>/<project-name>/<sweep-id>
wandb agent <your-username>/<project-name>/<sweep-id>
```

**What gets logged:**
- All hyperparameter combinations tested
- Training and validation metrics for each run
- Early termination of poor-performing runs (via Hyperband)
- Visualization of parameter importance and parallel coordinates plots

**Customizing the sweep:**

Edit `sweep.yaml` to modify search parameters, add new hyperparameters, or change the optimization method. See [W&B Sweeps documentation](https://docs.wandb.ai/guides/sweeps) for advanced configuration options.

### 4. Evaluation

Evaluate the trained model on test data:

```bash
python -m chess_cv.test
```

**Evaluation options:**

```bash
python -m chess_cv.test \
  --test-dir data/test \
  --train-dir data/train \
  --checkpoint checkpoints/best_model.safetensors \
  --batch-size 256 \
  --image-size 32 \
  --num-workers 4 \
  --output-dir outputs
```

**Output:**

- `outputs/test_confusion_matrix.png` – Confusion matrix heatmap
- `outputs/test_per_class_accuracy.png` – Per-class accuracy bar chart
- `outputs/misclassified_images/` – Misclassified examples for analysis
- Console output with overall and per-class accuracy

## Project Structure

```
chess-cv/
├── src/chess_cv/
│   ├── __init__.py           # Package initialization
│   ├── constants.py          # Configuration constants
│   ├── data.py               # Data loading utilities
│   ├── evaluate.py           # Evaluation metrics
│   ├── model.py              # CNN architecture
│   ├── preprocessing.py      # Data splitting
│   ├── test.py               # Testing script
│   ├── train.py              # Training script
│   └── visualize.py          # Visualization utilities
├── data/                     # Dataset directory
├── checkpoints/              # Model checkpoints
├── outputs/                  # Training/evaluation outputs
├── tests/                    # Test suite
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Model Details

### Architecture

- **Input**: 32×32 RGB images
- **Layer 1**: Conv2d(3→16) + ReLU + MaxPool
- **Layer 2**: Conv2d(16→32) + ReLU + MaxPool
- **Layer 3**: Conv2d(32→64) + ReLU + MaxPool
- **FC1**: Linear(1024→128) + ReLU + Dropout(0.5)
- **FC2**: Linear(128→13) - Output logits

### Training Configuration

- **Optimizer**: AdamW (lr=2e-4, weight_decay=5e-4)
- **Loss**: Cross-entropy
- **Batch Size**: 128
- **Image Size**: 32×32
- **Classes**: 13 (bB, bK, bN, bP, bQ, bR, wB, wK, wN, wP, wQ, wR, xx)

### Data Augmentation

- Random resized crop (scale: 0.8-1.0)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation: ±0.2)
- Random rotation (±5°)
- Gaussian noise (std: 0.05)

## Development

### Code Quality

```bash
# Format code
ruff format src/

# Lint code
ruff check src/

# Type check
basedpyright src/

# Run tests
pytest
```

### Configuration

All default values are defined in `src/chess_cv/constants.py` and can be overridden via command-line arguments.

## Troubleshooting

**Issue: MLX not working**

- Ensure MLX is properly installed: `uv pip install --force-reinstall mlx`
- MLX is optimized for Apple Silicon but may work on other platforms with limited functionality

**Issue: Out of memory during training**

- Reduce `--batch-size` (try 64 or 32)
- Reduce `--num-workers` (try 2 or 1)

**Issue: Poor model performance**

- Increase `--num-epochs` (try 150-200)
- Adjust `--learning-rate` (try 1e-4 or 5e-4)
- Use W&B Sweeps to automatically find optimal hyperparameters
- Check data quality and class balance
- Review misclassified images in `outputs/misclassified_images/`

**Issue: Data generation fails**

- Verify `data/boards/` contains board images (256×256px PNG files)
- Verify `data/pieces/` contains piece set directories with piece images (32×32px PNG files)
- Ensure all images are valid PNG files
- Check that split ratios sum to 1.0

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

<div align="center">

[Get Started](#quick-start) • [Contribute](CONTRIBUTING.md) • [Report Issues](https://github.com/S1M0N38/chess-cv/issues)

</div>
