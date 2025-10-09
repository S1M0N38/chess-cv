# Architecture

Detailed information about the Chess CV model architecture, training strategy, and performance characteristics.

## Model Architecture

### Overview

<figure markdown>
  ![Model Architecture](assets/model.svg)
  <figcaption>CNN architecture for chess piece classification</figcaption>
</figure>

Chess CV uses a lightweight Convolutional Neural Network (CNN) designed for efficient inference while maintaining high accuracy on 32×32 pixel chess square images.

### Network Design

```
Input: 32×32×3 RGB image

Conv Layer 1:
├── Conv2d(3 → 16 channels, 3×3 kernel)
├── ReLU activation
└── MaxPool2d(2×2) → 16×16×16

Conv Layer 2:
├── Conv2d(16 → 32 channels, 3×3 kernel)
├── ReLU activation
└── MaxPool2d(2×2) → 8×8×32

Conv Layer 3:
├── Conv2d(32 → 64 channels, 3×3 kernel)
├── ReLU activation
└── MaxPool2d(2×2) → 4×4×64

Flatten → 1024 features

Fully Connected 1:
├── Linear(1024 → 128)
├── ReLU activation
└── Dropout(0.5)

Fully Connected 2:
└── Linear(128 → 13) → Output logits

Softmax → 13-class probabilities
```

### Model Statistics

- **Total Parameters**: 156,077
- **Trainable Parameters**: 156,077
- **Model Size**: ~600 KB (safetensors format)
- **Input Size**: 32×32×3 (RGB)
- **Output Classes**: 13

### Class Labels

The model classifies chess squares into 13 categories:

**Black Pieces (6):**

- `bB` – Black Bishop
- `bK` – Black King
- `bN` – Black Knight
- `bP` – Black Pawn
- `bQ` – Black Queen
- `bR` – Black Rook

**White Pieces (6):**

- `wB` – White Bishop
- `wK` – White King
- `wN` – White Knight
- `wP` – White Pawn
- `wQ` – White Queen
- `wR` – White Rook

**Empty (1):**

- `xx` – Empty square

## Performance Characteristics

### Expected Results

With the default configuration:

- **Test Accuracy**: ~99.94%
- **F1 Score (Macro)**: ~99.94%
- **Training Time**: ~90 minutes (varies by hardware)
- **Inference Speed**: 0.05 ms per image (batch size 8192, varying by hardware)

### Inference Benchmarks

Inference performance on **Apple M4** (MacBook Air, 2025):

**Hardware Specifications:**

- **Chip**: Apple M4
- **CPU**: 10 cores (4 performance + 6 efficiency)
- **GPU**: 10 cores with Metal 4 support
- **Memory**: 16 GB unified memory
- **macOS**: Version 26.0.1

**Benchmark Results:**

| Batch Size | Images/sec | ms/batch | ms/image |
| ---------- | ---------- | -------- | -------- |
| 1          | TBD        | TBD      | TBD      |
| 64         | TBD        | TBD      | TBD      |
| 512        | TBD        | TBD      | TBD      |
| 1024       | TBD        | TBD      | TBD      |

!!! tip "Running Benchmarks"

    To benchmark inference speed on your machine, run:

    ```bash
    chess-cv test pieces
    ```

    The benchmark results will be included in the test summary at `outputs/pieces/test_summary.json`.

### Per-Class Performance

Actual accuracy by piece type (Test Dataset):

| Class | Accuracy | Class | Accuracy |
| ----- | -------- | ----- | -------- |
| bB    | 99.90%   | wB    | 99.90%   |
| bK    | 100.00%  | wK    | 99.81%   |
| bN    | 100.00%  | wN    | 100.00%  |
| bP    | 99.91%   | wP    | 99.90%   |
| bQ    | 99.90%   | wQ    | 100.00%  |
| bR    | 100.00%  | wR    | 100.00%  |
| xx    | 99.91%   |       |          |

### Evaluation on External Datasets

The model has been evaluated on external datasets to assess generalization:

#### OpenBoard

- **Dataset**: [S1M0N38/chess-cv-openboard](https://huggingface.co/datasets/S1M0N38/chess-cv-openboard)
- **Number of samples**: 6,016
- **Overall Accuracy**: 99.30%
- **F1 Score (Macro)**: 98.26%

Per-class performance on OpenBoard:

| Class | Accuracy | Class | Accuracy |
| ----- | -------- | ----- | -------- |
| bB    | 100.00%  | wB    | 100.00%  |
| bK    | 100.00%  | wK    | 100.00%  |
| bN    | 98.91%   | wN    | 97.94%   |
| bP    | 99.81%   | wP    | 99.61%   |
| bQ    | 97.10%   | wQ    | 98.48%   |
| bR    | 99.32%   | wR    | 98.68%   |
| xx    | 99.24%   |       |          |

#### ChessVision

- **Dataset**: [S1M0N38/chess-cv-chessvision](https://huggingface.co/datasets/S1M0N38/chess-cv-chessvision)
- **Number of samples**: 3,186
- **Overall Accuracy**: 86.38%
- **F1 Score (Macro)**: 83.47%

Per-class performance on ChessVision:

| Class | Accuracy | Class | Accuracy |
| ----- | -------- | ----- | -------- |
| bB    | 90.00%   | wB    | 95.04%   |
| bK    | 84.43%   | wK    | 91.82%   |
| bN    | 100.00%  | wN    | 98.18%   |
| bP    | 83.83%   | wP    | 80.09%   |
| bQ    | 95.70%   | wQ    | 89.66%   |
| bR    | 86.56%   | wR    | 85.08%   |
| xx    | 86.50%   |       |          |

!!! note "Multi-Split Dataset"

    The ChessVision dataset contains multiple splits. All splits are concatenated during evaluation to produce a single comprehensive score.

!!! note "Out of Sample Performance"

    The lower performance on OpenBoard (99.30% accuracy, 98.26% F1) and ChessVision (86.38% accuracy, 83.47% F1) compared to the test set (99.94% accuracy, 99.94% F1) indicates some domain gap between the synthetic training data and these external datasets. ChessVision shows significantly lower performance, particularly on specific piece types like black kings (84.43%) and pawns (80-84%).

## Dataset Characteristics

### Synthetic Data Generation

The training data is synthetically generated:

**Source Materials:**

- 55 board styles (256×256px)
- 64 piece sets (32×32px)
- Multiple visual styles from chess.com and lichess

**Generation Process:**

1. Render each piece onto each board style
2. Extract 32×32 squares at piece locations
3. Extract empty squares from light and dark squares
4. Split combinations across train/val/test sets

**Data Statistics:**

- **Total Combinations**: ~3,520 (55 boards × 64 piece sets)
- **Images per Combination**: 26 (12 pieces × 2 colors + 2 empty)
- **Total Images**: ~91,500
- **Train Set**: ~64,000 (70%)
- **Validation Set**: ~13,500 (15%)
- **Test Set**: ~13,500 (15%)

### Class Balance

The dataset is perfectly balanced:

- Each class has equal representation
- Each board-piece combination contributes equally
- Train/val/test splits maintain class balance

### Diversity

The synthetic approach provides diversity:

- **Visual Styles**: 55 different board appearances
- **Piece Designs**: 64 different piece set styles
- **Square Colors**: Both light and dark squares
- **Augmentation**: Geometric and color transforms
