# AGENTS.md

This file provides guidance to AI tools (e.g. Claude Code, Codex, Gemini CLI, ...) when working with code in this repository.

## Project Overview

Chess-CV is a CNN-based chess piece classifier that uses PyTorch with CUDA support to train a lightweight 156k parameter model. The model classifies 32×32px square images into 13 classes (6 white pieces, 6 black pieces, 1 empty square) with ~99.85% accuracy.

**Key Technologies:**

- PyTorch for model training (with CUDA GPU acceleration)
- Mixed precision training (AMP) for faster training on CUDA
- Channels-last memory format for optimal GPU performance
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
chess-cv preprocessing pieces

# 2. Train model
chess-cv train pieces
chess-cv train pieces --wandb  # with W&B logging

# 3. Evaluate model
chess-cv test pieces
make eval  # Run full evaluation suite

# 4. Upload to Hugging Face Hub
chess-cv upload pieces --repo-id username/chess-cv
```

### CLI Help

```bash
# See all available commands
chess-cv --help

# See help for specific command
chess-cv preprocessing --help
chess-cv train --help
chess-cv test --help
chess-cv upload --help

# Example usage with model-id
chess-cv preprocessing pieces --help
chess-cv train pieces --help
chess-cv test pieces --help
chess-cv upload pieces --repo-id username/chess-cv --help
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
- `collate_fn`: Converts batches to PyTorch tensors (HWC → NCHW format)
- Supports `pin_memory=True` for faster GPU transfers
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

- PyTorch uses NCHW format (batch, channels, height, width)
- Model uses channels_last memory format for optimal CUDA performance
- Model weights use safetensors format
- Create with `create_model()` to properly initialize and move to device
- Automatically detects and uses CUDA if available

### Training (src/chess_cv/train.py)

**Key Features:**

- AdamW optimizer with configurable learning rate/weight decay
- Mixed precision training (AMP) for faster CUDA training
- Channels-last memory format for optimal GPU performance
- Pin memory and persistent workers for efficient data loading
- cuDNN benchmark mode for optimized convolutions
- Aggressive data augmentation: random crop, flip, color jitter, rotation, Gaussian noise
- Early stopping based on validation accuracy
- Saves best model to `checkpoints/{model-id}/{model-id}.safetensors`
- Optional W&B integration (pass `--wandb` flag)

**CUDA Optimizations:**

- Mixed precision (FP16/FP32) with gradient scaling
- Channels-last memory format
- Pin memory for faster host-to-GPU transfer
- Non-blocking GPU transfers
- cuDNN benchmark for optimal conv algorithms

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
- Inference speed benchmarking with CUDA event timing
- Uses `torch.inference_mode()` for optimal performance

**test.py** - Full evaluation script:

- Supports local directories or HuggingFace datasets
- Automatically uses CUDA if available
- Generates confusion matrix and per-class accuracy plots
- Saves misclassified images for analysis
- Outputs JSON summary with performance metrics

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
│       └── {model-id}/    # Model-specific data (e.g., pieces/)
│           ├── train/     # Training data
│           ├── validate/  # Validation data
│           └── test/      # Test data
├── checkpoints/           # Model checkpoints
│   └── {model-id}/        # Model-specific checkpoints (e.g., pieces/)
│       ├── {model-id}.safetensors
│       └── optimizer.safetensors
├── outputs/               # Training outputs
│   └── {model-id}/        # Model-specific outputs (e.g., pieces/)
├── evals/                 # Evaluation results
│   └── {model-id}/        # Model-specific evals (e.g., pieces/)
├── docs/                  # MkDocs documentation
└── tests/                 # Pytest tests
```

## Important Conventions

### Model Configuration

- **Model IDs**: Used to organize data, checkpoints, and outputs
- **Current model**: `pieces` (chess piece classifier with 13 classes)
- **Adding models**: Add to `MODEL_CONFIGS` in `constants.py`
- **Model-specific settings**: Each model has `num_classes`, `class_names`, `description`
- **Architecture**: SimpleCNN architecture is shared across all models
- **Input size**: 32×32×3 RGB images for all models

### Release Process

- Uses **Conventional Commits** for all commits
- **Never manually modify**: `uv.lock`, `CHANGELOG.md`, version numbers in `pyproject.toml` or `__init__.py`
- Release Please PR automatically updates versions and changelog
- Merging Release Please PR triggers PyPI publication

### PyTorch with CUDA

- **Framework**: Pure PyTorch (CPU/CUDA support)
- **Data loading**: PyTorch DataLoader with pin_memory for GPU
- **Training/inference**: PyTorch tensors and models
- **Optimization**: Mixed precision, channels_last, cuDNN benchmark

### Image Format

- Input: 32×32px RGB images
- Normalization: Divide by 255.0 to [0, 1] range
- PyTorch format: NCHW (batch, channels, height, width)
- Memory format: channels_last for CUDA optimization

### Label Mapping

- Labels are **alphabetically sorted** class names
- Empty square is "xx" (for pieces model)
- Use `get_label_map()` to create label→index mapping
- Use `get_model_config(model_id)["class_names"]` for model-specific class names
