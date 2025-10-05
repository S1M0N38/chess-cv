# Setup

This guide will help you install and configure Chess CV for training chess piece classifiers.

## Prerequisites

- **Python 3.13+**: Chess CV requires Python 3.13 or later
- **uv**: Fast Python package manager ([installation guide](https://docs.astral.sh/uv/))
- **MLX**: Apple's machine learning framework ([installation guide](https://ml-explore.github.io/mlx/))

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/S1M0N38/chess-cv.git
cd chess-cv
```

### 2. Install Dependencies

For model usage only:

```bash
pip install chess-cv
# or with uv
uv add chess-cv
```

For training your own models:

```bash
# Copy environment template
cp .envrc.example .envrc

# Install all dependencies
uv sync --all-extras
```

### 3. Verify Installation

```bash
# Check that chess-cv is installed
python -c "import chess_cv; print(chess_cv.__version__)"
```

## Data Preparation

### Directory Structure

The project expects the following data structure:

```
data/
├── boards/           # Source: 55 board images (256×256px)
│   ├── chess-com-*.png
│   └── lichess-*.png
├── pieces/           # Source: 64 piece set directories
│   ├── chess-com-*/  # Each contains bB.png, bK.png, ..., wR.png
│   └── lichess-*/
├── train/            # Generated: ~65k images
├── validate/         # Generated: ~14k images
└── test/             # Generated: ~14k images
```

### Board Images

Board images should be:

- **Format**: PNG
- **Size**: 256×256 pixels
- **Naming**: Any descriptive name (e.g., `chess-com-board-1.png`)
- **Location**: `data/boards/`

### Piece Sets

Each piece set should be a directory containing:

- **Files**: `bB.png`, `bK.png`, `bN.png`, `bP.png`, `bQ.png`, `bR.png` (black pieces)
- **Files**: `wB.png`, `wK.png`, `wN.png`, `wP.png`, `wQ.png`, `wR.png` (white pieces)
- **Format**: PNG with transparency
- **Size**: 32×32 pixels
- **Location**: `data/pieces/piece-set-name/`

### Generate Training Data

Run the preprocessing script to generate training, validation, and test datasets:

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

**How it works:**

The preprocessing script renders chess pieces onto board backgrounds to create training data:

- **Generation**: For each board-piece combination, pieces are composited onto the board and 32×32 squares are extracted
- **Output**: 26 images per combination (12 pieces × 2 square colors + 2 empty squares)
- **Total dataset**: ~93,000 images (65k train / 14k validation / 14k test)

## Development Setup

For contributing to the project:

```bash
# Install with development dependencies
uv sync --all-extras --group dev

# Verify development tools
ruff --version
basedpyright --version
pytest --version
```

## Troubleshooting

**MLX Not Working**: Ensure MLX is properly installed with `uv pip install --force-reinstall mlx` and check the [MLX documentation](https://ml-explore.github.io/mlx/) for platform-specific guides.

**Data Generation Fails**: Verify that `data/boards/` contains board images (256×256px PNG files) and `data/pieces/` contains piece set directories with piece images (32×32px PNG files). Ensure all images are valid PNG files and split ratios sum to 1.0.

**Import Errors**: Reinstall in editable mode with `uv pip install -e .` or `pip install -e .`

**Out of Memory**: If you encounter memory issues during data generation, process board-piece combinations in smaller batches and check available disk space.
