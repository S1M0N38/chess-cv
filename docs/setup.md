# Setup

This guide will help you install and configure Chess CV for training chess piece classifiers.

## Prerequisites

- **uv**: Fast Python package manager ([installation guide](https://docs.astral.sh/uv/))
- **Python 3.13+**: Chess CV requires Python 3.13 or later (can be installed with `uv`)
- **MLX**: Apple's machine learning framework (installed with `chess-cv`)

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

## Next Steps

- Use pre-trained models for inference with [Model Usage](inference.md)
- Train custom models with [Train and Evaluate](train-and-eval.md)
- Contribute to the project with [CONTRIBUTING.md](../CONTRIBUTING.md)
