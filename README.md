<div align="center">

# Chess CV

*CNN-based chess piece classifier using MLX for Apple Silicon*

</div>

---

## Overview

A machine learning project that uses Convolutional Neural Networks (CNNs) to classify chess pieces from chessboard square images. Built with MLX for efficient training and inference on Apple Silicon, this project processes 32x32 pixel images of individual chess squares to identify pieces and reconstruct FEN (Forsyth-Edwards Notation) representations.

## Features

**Model Architecture**
- CNN-based classifier for 32x32px chess square images
- Batch processing of 64 squares (full chessboard)
- 13-class classification (6 white pieces, 6 black pieces, 1 empty)
- MLX framework for Apple Silicon optimization

**Development Tools**
- **MLX** – Apple's ML framework for efficient training on Mac
- **NumPy** – Numerical computing for data processing
- **Matplotlib** – Training curve visualization and debugging
- **Ruff** – Fast Python linter and formatter
- **Pyright** – Static type checking
- **pytest** – Testing framework

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Apple Silicon Mac (for MLX acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/S1M0N38/chess-cv.git
cd chess-cv

# Install dependencies
uv sync --all-extras
```

### Development Commands

```bash
# Run tests
pytest

# Lint and format code
ruff check
ruff format

# Type check
basedpyright

# Run the application
python -m chess_cv
```

## Project Structure

```
chess-cv/
├── src/chess_cv/          # Main package source
│   ├── model.py          # CNN model definition
│   ├── train.py          # Training loop
│   └── visualize.py      # Training curve visualization
├── tests/                 # Test suite
├── data/                  # Training/validation data
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Model Details

- **Input**: 32x32 pixel grayscale/RGB images of chess squares
- **Output**: 13 classes (K, Q, R, B, N, P for both colors + empty)
- **Framework**: MLX for Apple Silicon optimization
- **Visualization**: Matplotlib for loss/accuracy curves

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Train chess piece classifiers on Apple Silicon**

[Get Started](#quick-start) • [Contribute](CONTRIBUTING.md) • [Report Issues](https://github.com/S1M0N38/chess-cv/issues)

</div>
