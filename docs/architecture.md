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

- **Test Accuracy**: ~99.85%
- **F1 Score (Macro)**: ~99.89%
- **Training Time**: ~90 minutes (varies by hardware)

### Per-Class Performance

Actual accuracy by piece type (Test Dataset):

| Class | Accuracy | Class | Accuracy |
| ----- | -------- | ----- | -------- |
| bB    | 99.82%   | wB    | 99.91%   |
| bK    | 99.82%   | wK    | 99.54%   |
| bN    | 99.73%   | wN    | 99.91%   |
| bP    | 99.82%   | wP    | 100%     |
| bQ    | 100%     | wQ    | 99.82%   |
| bR    | 99.64%   | wR    | 100%     |
| xx    | 100%     |       |          |

### Evaluation on External Datasets

The model has been evaluated on external datasets to assess generalization:

#### OpenBoard

- **Dataset**: [S1M0N38/chess-cv-openboard](https://huggingface.co/datasets/S1M0N38/chess-cv-openboard)
- **Number of samples**: 6,016
- **Overall Accuracy**: 97.57%
- **F1 Score (Macro)**: 95.78%

Per-class performance on OpenBoard:

| Class | Accuracy | Class | Accuracy |
| ----- | -------- | ----- | -------- |
| bB    | 100%     | wB    | 100%     |
| bK    | 98.94%   | wK    | 100%     |
| bN    | 100%     | wN    | 98.97%   |
| bP    | 99.62%   | wP    | 98.82%   |
| bQ    | 97.10%   | wQ    | 100%     |
| bR    | 99.32%   | wR    | 97.37%   |
| xx    | 96.72%   |       |          |

!!! note "Out of Sample Performance"

    The lower performance on OpenBoard compared to the test set (97.57% vs 99.85%) indicates some domain gap between the synthetic training data and this external dataset.

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
