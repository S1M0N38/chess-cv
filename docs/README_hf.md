---
license: mit
library_name: mlx
tags:
  - computer-vision
  - image-classification
  - chess
  - cnn
  - lightweight
datasets:
  - synthetic-chess-squares
model-index:
  - name: chess-cv
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: Chess CV Test Dataset
          type: chess-cv-test
        metrics:
          - type: accuracy
            value: 0.9986
            name: Accuracy
            verified: false
          - type: f1
            value: 0.9984
            name: F1 Score (Macro)
            verified: false
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: Chess CV OpenBoard Dataset
          type: chess-cv-openboard
        metrics:
          - type: accuracy
            value: 0.9889
            name: Accuracy
            verified: false
          - type: f1
            value: 0.9738
            name: F1 Score (Macro)
            verified: false
pipeline_tag: image-classification
---

# Chess CV

<div align="center">
  <img src="https://raw.githubusercontent.com/S1M0N38/chess-cv/main/docs/assets/model.svg" alt="Model Architecture" width="600">
</div>

Lightweight CNN (156k parameters) that classifies chess pieces from 32×32 pixel square images into 13 classes (6 white pieces, 6 black pieces, empty square). Trained on synthetic data from chess.com/lichess boards and piece sets.

| Dataset                                                                                     | Accuracy | F1-Score (Macro) |
| ------------------------------------------------------------------------------------------- | :------: | :--------------: |
| Test Data                                                                                   |  99.86%  |      99.84%      |
| [S1M0N38/chess-cv-openboard](https://huggingface.co/datasets/S1M0N38/chess-cv-openboard) \* |    -     |      97.38%      |

\* *Dataset with unbalanced class distribution (e.g. many more samples for empty square class), so accuracy is not representative.*

## Quick Start

```bash
pip install chess-cv
```

```python
import mlx.core as mx
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from chess_cv.model import SimpleCNN

# Load model
model_path = hf_hub_download(repo_id="S1M0N38/chess-cv", filename="pieces.safetensors")
model = SimpleCNN(num_classes=13)
model.load_weights(model_path)
model.eval()

# Predict
img = Image.open("square.png").convert('RGB').resize((32, 32))
img_array = mx.array(np.array(img, dtype=np.float32)[None, ...] / 255.0)
pred_idx = mx.argmax(model(img_array), axis=-1).item()

classes = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR', 'xx']
print(f"Predicted: {classes[pred_idx]}")
```

## Models

This repository contains two specialized models for chess board analysis:

### Pieces Model (`pieces.safetensors`)

**Overview:**

The pieces model classifies chess square images into 13 classes: 6 white pieces (wP, wN, wB, wR, wQ, wK), 6 black pieces (bP, bN, bB, bR, bQ, bK), and empty squares (xx). This model is designed for board state recognition and FEN generation from chess board images.

**Training:**

- **Architecture**: SimpleCNN (156k parameters)
- **Input**: 32×32px RGB square images
- **Data**: ~93,000 synthetic images from 55 board styles × 64 piece sets
- **Augmentation**: Aggressive augmentation with arrow overlays (80%), highlight overlays (25%), random crops, horizontal flips, color jitter, rotation (±5°), and Gaussian noise
- **Optimizer**: AdamW (lr=0.0003, weight_decay=0.0003)
- **Training**: 200 epochs, batch size 64

**Performance:**

| Dataset | Accuracy | F1-Score (Macro) |
|---------|----------|------------------|
| Test Data (synthetic) | 99.86% | 99.84% |
| [chess-cv-openboard](https://huggingface.co/datasets/S1M0N38/chess-cv-openboard) | - | 97.38% |

The model achieves near-perfect accuracy on synthetic data and maintains strong performance on real board images from OpenBoard dataset.

### Arrows Model (`arrows.safetensors`)

**Overview:**

The arrows model classifies chess square images into 49 classes representing different arrow overlay patterns: 20 arrow heads (N, NE, E, SE, S, SW, W, NW, and intermediate directions), 12 arrow tails, 8 middle segments, 4 corner pieces, and empty squares (xx). This model enables detection and reconstruction of arrow annotations commonly used in chess analysis interfaces.

**Training:**

- **Architecture**: SimpleCNN (156k parameters, same as pieces model)
- **Input**: 32×32px RGB square images
- **Data**: ~93,000 synthetic images from 55 board styles × arrow overlays
- **Augmentation**: Conservative augmentation with highlight overlays (25%), random crops, and minimal color jitter/noise. No horizontal flips or rotation to preserve arrow directionality
- **Optimizer**: AdamW (lr=0.0003, weight_decay=0.0003)
- **Training**: 200 epochs, batch size 64

**Performance:**

| Dataset | Accuracy | F1-Score (Macro) |
|---------|----------|------------------|
| Test Data (synthetic) | ~99% | ~99% |

The arrows model is optimized for detecting directional annotations while maintaining spatial consistency across the board.

## Training Your Own Model

To train or evaluate the model yourself:

```bash
git clone https://github.com/S1M0N38/chess-cv.git
cd chess-cv
uv sync --all-extras

# Generate training data
chess-cv preprocessing

# Train model
chess-cv train

# Evaluate model
chess-cv test
```

See the [Setup Guide](https://s1m0n38.github.io/chess-cv/setup/) and [Usage Guide](https://s1m0n38.github.io/chess-cv/usage/) for detailed instructions on data generation, training configuration, and evaluation.

## Limitations

- Requires precisely cropped 32×32 pixel square images (no board detection)
- Trained on synthetic data; may not generalize to real-world photos
- Not suitable for non-standard piece designs or chess game logic
- Optimized for Apple Silicon (slower on CPU)

For detailed documentation, architecture details, and advanced usage, see the [full documentation](https://s1m0n38.github.io/chess-cv/).

## Citation

```bibtex
@software{bertolotto2025chesscv,
  author = {Bertolotto, Simone},
  title = {{Chess CV: CNN-based Chess Piece Classifier}},
  url = {https://github.com/S1M0N38/chess-cv},
  year = {2025}
}
```

**Repository:** [github.com/S1M0N38/chess-cv](https://github.com/S1M0N38/chess-cv) • **PyPI:** [pypi.org/project/chess-cv](https://pypi.org/project/chess-cv/)
