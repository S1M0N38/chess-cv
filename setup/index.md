# Setup Guide

This guide will help you install and configure Chess CV for inference and training chess board classification models.

## Prerequisites

- **Python 3.13+**: Chess CV requires Python 3.13 or later
- **uv**: Fast Python package manager (required for training, [installation guide](https://docs.astral.sh/uv/))
- **MLX**: Apple's machine learning framework (installed with `chess-cv`)

## Installation

### For Inference Only

Install Chess CV directly from PyPI:

```bash
pip install chess-cv
# or with uv
uv add chess-cv
```

### For Training and Evaluation

Clone the repository and install all dependencies:

```bash
# Clone the repository
git clone https://github.com/S1M0N38/chess-cv.git
cd chess-cv

# Copy environment template
cp .envrc.example .envrc

# Install all dependencies
uv sync --all-extras

# For contributors: add development tools
uv sync --all-extras --group dev
```

## Next Steps

- Use pre-trained models for inference with [Model Usage](../inference/)
- Train custom models with [Train and Evaluate](../train-and-eval/)
- Contribute to the project with [CONTRIBUTING.md](../CONTRIBUTING.md)
