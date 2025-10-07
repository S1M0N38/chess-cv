# AGENTS.md

This file provides guidance to AI tools (e.g. Claude Code, Codex, Gemini CLI, ...) when working with code in this repository.

## Project Overview

Chess-CV is a CNN-based chess piece classifier that uses MLX (Apple's ML framework) to train a lightweight 156k parameter model. The model classifies 32×32px square images into 13 classes (6 white pieces, 6 black pieces, 1 empty square) with ~99.85% accuracy.

**Key Technologies:**

- MLX for model training (Apple Silicon optimized)
- PyTorch DataLoader for data loading
- NumPy/PIL for image processing
- Weights & Biases (optional) for experiment tracking
- Hugging Face Hub for model/dataset distribution

## Development Commands

### Setup

```bash
# Install dependencies (uses uv package manager)
make install
# or: uv sync --all-extras

# Set up environment variables
cp .envrc.example .envrc
# Edit .envrc with your settings, then:
source .envrc
```

### Code Quality

```bash
# Run all quality checks (lint, typecheck, format)
make quality

# Individual checks
make lint        # ruff linter with auto-fix
make format      # ruff + mdformat formatters
make typecheck   # basedpyright type checker
```

### Testing

```bash
make test        # pytest
make all         # lint, format, typecheck, test
```

### Documentation

```bash
make docs        # Serve at http://127.0.0.1:8000
```

### Model Pipeline

```bash
# 1. Generate synthetic training data (boards + pieces → square images in data/splits/pieces/)
python -m chess_cv.preprocessing

# 2. Train model
python -m chess_cv.train
python -m chess_cv.train --wandb  # with W&B logging

# 3. Evaluate model
python -m chess_cv.test
make eval  # Run full evaluation suite

# 4. Upload to Hugging Face Hub
python -m chess_cv.upload
```

### Run Single Test

```bash
pytest tests/test_example.py::test_specific_function -v
```

## Architecture

### Data Pipeline (src/chess_cv/)

**preprocessing.py** - Generates synthetic training data:

- Combines 55 board images (256×256px) with 64 piece sets (32×32px)
- Renders pieces on dark/light squares, extracts 32×32px crops
- Splits into train/val/test (70/15/15 by default)
- Uses multiprocessing for parallel generation

**data.py** - Data loading:

- `ChessPiecesDataset`: PyTorch Dataset for local image files
- `HuggingFaceChessPiecesDataset`: Dataset for HF datasets
- `collate_fn`: Converts batches to MLX arrays
- `CLASS_NAMES`: 13 classes in alphabetical order (bB, bK, bN, bP, bQ, bR, wB, wK, wN, wP, wQ, wR, xx)

### Model (src/chess_cv/model.py)

**SimpleCNN Architecture:**

```
Conv2d(3→16) → ReLU → MaxPool2d → [32×32 → 16×16]
Conv2d(16→32) → ReLU → MaxPool2d → [16×16 → 8×8]
Conv2d(32→64) → ReLU → MaxPool2d → [8×8 → 4×4]
Flatten → Linear(1024→128) → ReLU → Dropout(0.5)
Linear(128→num_classes)
```

**Important Notes:**

- MLX uses NHWC format (batch, height, width, channels)
- Model weights use safetensors format
- Create with `create_model()` to properly initialize parameters

### Training (src/chess_cv/train.py)

**Key Features:**

- AdamW optimizer with configurable learning rate/weight decay
- Aggressive data augmentation: random crop, flip, color jitter, rotation, Gaussian noise
- Early stopping based on validation accuracy
- Saves best model to `checkpoints/best_model.safetensors`
- Optional W&B integration (pass `--wandb` flag)

**Default Hyperparameters:**

- Batch size: 64
- Learning rate: 0.0003
- Weight decay: 0.0003
- Epochs: 200
- Image size: 32×32

### Evaluation (src/chess_cv/)

**evaluate.py** - Core metrics:

- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Macro F1-score

**test.py** - Full evaluation script:

- Supports local directories or HuggingFace datasets
- Generates confusion matrix and per-class accuracy plots
- Saves misclassified images for analysis
- Outputs JSON summary

### Constants (src/chess_cv/constants.py)

Central location for all default values:

- Data paths (DEFAULT_TRAIN_DIR, DEFAULT_VAL_DIR, etc.)
- Model parameters (DEFAULT_NUM_CLASSES, DEFAULT_DROPOUT)
- Training hyperparameters
- Augmentation parameters
- Output filenames

## Project Structure

```
chess-cv/
├── src/chess_cv/          # Main package
│   ├── model.py           # SimpleCNN architecture
│   ├── train.py           # Training script
│   ├── test.py            # Evaluation script
│   ├── preprocessing.py   # Data generation
│   ├── data.py            # Dataset classes
│   ├── evaluate.py        # Metrics computation
│   ├── visualize.py       # Training visualization
│   ├── wandb_utils.py     # W&B integration
│   └── constants.py       # Configuration
├── data/                  # Data directory
│   ├── boards/            # Board images (256×256px)
│   ├── pieces/            # Piece sets (32×32px)
│   └── splits/            # Generated training/validation/test splits
│       └── pieces/        # Piece classification data
│           ├── train/     # Training data
│           ├── validate/  # Validation data
│           └── test/      # Test data
├── checkpoints/           # Model checkpoints
├── outputs/               # Training outputs
├── evals/                 # Evaluation results
├── docs/                  # MkDocs documentation
└── tests/                 # Pytest tests
```

## Important Conventions

### Release Process

- Uses **Conventional Commits** for all commits
- **Never manually modify**: `uv.lock`, `CHANGELOG.md`, version numbers in `pyproject.toml` or `__init__.py`
- Release Please PR automatically updates versions and changelog
- Merging Release Please PR triggers PyPI publication

### MLX vs PyTorch

- **Data loading**: PyTorch DataLoader (for efficient I/O)
- **Training/inference**: MLX arrays and models
- **Conversion**: `collate_fn` converts PyTorch batches → MLX arrays

### Image Format

- Input: 32×32px RGB images
- Normalization: Divide by 255.0 to [0, 1] range
- MLX format: NHWC (batch, height, width, channels)

### Label Mapping

- Labels are **alphabetically sorted** class names
- Empty square is "xx"
- Use `get_label_map()` to create label→index mapping
- CLASS_NAMES in data.py maintains canonical order
