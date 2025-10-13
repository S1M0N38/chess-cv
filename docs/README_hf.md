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
            value: 0.9995
            name: Accuracy
            verified: false
          - type: f1
            value: 0.9995
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
            value: 0.9955
            name: Accuracy
            verified: false
          - type: f1
            value: 0.9891
            name: F1 Score (Macro)
            verified: false
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: Chess CV ChessVision Dataset
          type: chess-cv-chessvision
        metrics:
          - type: accuracy
            value: 0.9470
            name: Accuracy
            verified: false
          - type: f1
            value: 0.9405
            name: F1 Score (Macro)
            verified: false
  - name: chess-cv-arrows
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: Chess CV Arrows Test Dataset
          type: chess-cv-arrows-test
        metrics:
          - type: accuracy
            value: 0.9997
            name: Accuracy
            verified: false
          - type: f1
            value: 0.9997
            name: F1 Score (Macro)
            verified: false
  - name: chess-cv-snap
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: Chess CV Snap Test Dataset
          type: chess-cv-snap-test
        metrics:
          - type: accuracy
            value: TBD
            name: Accuracy
            verified: false
          - type: f1
            value: TBD
            name: F1 Score (Macro)
            verified: false
pipeline_tag: image-classification
---

<div align="center">

# Chess CV

<img src="https://raw.githubusercontent.com/S1M0N38/chess-cv/main/docs/assets/model.svg" alt="Model Architecture" width="600">

</div>

Lightweight CNN (156k parameters) that classifies chess pieces, arrows, and piece centering from 32×32 pixel square images. Trained on synthetic data from chess.com/lichess boards, piece sets, arrow overlays, and centering variations.

## Quick Start

```bash
pip install chess-cv
```

```python
import mlx.core as mx
import numpy as np
from PIL import Image
from chess_cv import load_bundled_model
from chess_cv.constants import get_model_config

# Load model with bundled weights (included in package)
model = load_bundled_model('pieces')
model.eval()

# Predict
img = Image.open("square.png").convert('RGB').resize((32, 32))
img_array = mx.array(np.array(img, dtype=np.float32)[None, ...] / 255.0)
pred_idx = mx.argmax(model(img_array), axis=-1).item()

# Get class names for the model
classes = get_model_config('pieces')['class_names']
print(f"Predicted: {classes[pred_idx]}")
```

**Alternative: Load latest version from Hugging Face Hub**

```python
from huggingface_hub import hf_hub_download
from chess_cv.model import SimpleCNN
import mlx.core as mx

# Download latest weights from Hugging Face
model_path = hf_hub_download(repo_id="S1M0N38/chess-cv", filename="pieces.safetensors")
model = SimpleCNN(num_classes=13)
weights = mx.load(str(model_path))
model.load_weights(list(weights.items()))
model.eval()
```

## Models

This repository contains three specialized models for chess board analysis:

### ♟️ Pieces Model (`pieces.safetensors`)

**Overview:**

The pieces model classifies chess square images into 13 classes: 6 white pieces (wP, wN, wB, wR, wQ, wK), 6 black pieces (bP, bN, bB, bR, bQ, bK), and empty squares (xx). This model is designed for board state recognition and FEN generation from chess board images.

**Training:**

- **Architecture**: SimpleCNN (156k parameters)
- **Input**: 32×32px RGB square images
- **Data**: ~93,000 synthetic images from 55 board styles × 64 piece sets
- **Augmentation**: Aggressive augmentation with arrow overlays (80%), highlight overlays (25%), random crops, horizontal flips, color jitter, rotation (±10°), and Gaussian noise
- **Optimizer**: AdamW (weight_decay=0.001) with LR scheduler (warmup + cosine decay: 0→0.001→1e-5)
- **Training**: 1000 epochs, batch size 64

**Performance:**

| Dataset                                                                                         | Accuracy | F1-Score (Macro) |
| ----------------------------------------------------------------------------------------------- | :------: | :--------------: |
| Test Data                                                                                       |  99.95%  |      99.95%      |
| [S1M0N38/chess-cv-openboard](https://huggingface.co/datasets/S1M0N38/chess-cv-openboard) \*     |    -     |      98.91%      |
| [S1M0N38/chess-cv-chessvision](https://huggingface.co/datasets/S1M0N38/chess-cv-chessvision) \* |    -     |      94.05%      |

\* *Dataset with unbalanced class distribution (e.g. many more samples for empty square class), so accuracy is not representative.*

### ↗ Arrows Model (`arrows.safetensors`)

**Overview:**

The arrows model classifies chess square images into 49 classes representing different arrow overlay patterns: 20 arrow heads, 12 arrow tails, 8 middle segments (for straight and diagonal arrows), 4 corner pieces (for knight-move arrows), and empty squares (xx). This model enables detection and reconstruction of arrow annotations commonly used in chess analysis interfaces. The NSEW naming convention (North/South/East/West) indicates arrow orientation and direction.

**Training:**

- **Architecture**: SimpleCNN (156k parameters, same as pieces model)
- **Input**: 32×32px RGB square images
- **Data**: ~4.5M synthetic images from 55 board styles × arrow overlays (~3.14M train, ~672K val, ~672K test)
- **Augmentation**: Conservative augmentation with highlight overlays (25%), random crops, and minimal color jitter/noise. No horizontal flips to preserve arrow directionality
- **Optimizer**: AdamW (lr=0.0005, weight_decay=0.00005)
- **Training**: 20 epochs, batch size 128

**Performance:**

| Dataset               | Accuracy | F1-Score (Macro) |
| --------------------- | -------- | ---------------- |
| Test Data (synthetic) | 99.97%   | 99.97%           |

The arrows model is optimized for detecting directional annotations while maintaining spatial consistency across the board.

**Limitation:** Classification accuracy degrades when multiple arrow components overlap in a single square.

### 📐 Snap Model (`snap.safetensors`)

**Overview:**

The snap model classifies chess square images into 2 classes: centered ("ok") and off-centered ("bad") pieces. This model is designed for automated board analysis and piece positioning validation, helping ensure proper piece placement in digital chess interfaces and automated analysis systems.

**Training:**

- **Architecture**: SimpleCNN (156k parameters)
- **Input**: 32×32px RGB square images
- **Data**: ~176,000 synthetic images from centered and off-centered piece positions (~123,300 train, ~26,400 val, ~26,300 test)
- **Augmentation**: Conservative augmentation with arrow overlays (50%), highlight overlays (20%), mouse overlays (80%), horizontal flips (50%), color jitter, and Gaussian noise. No rotation or geometric transformations to preserve centering semantics
- **Optimizer**: AdamW (weight_decay=0.001) with LR scheduler (warmup + cosine decay: 0→0.001→1e-5)
- **Training**: 200 epochs, batch size 64

**Performance:**

| Dataset               | Accuracy | F1-Score (Macro) |
| --------------------- | :------: | :--------------: |
| Test Data (synthetic) |   TBD    |       TBD        |

*Training in progress*

The snap model is optimized for detecting piece centering issues while maintaining robustness to various board styles and visual conditions.

**Use Cases:**

- Automated board state validation
- Piece positioning quality control
- Chess interface usability testing
- Digital chess board quality assurance

## Training Your Own Model

To train or evaluate a model yourself:

```bash
git clone https://github.com/S1M0N38/chess-cv.git
cd chess-cv
uv sync --all-extras

# Generate training data for a specific model
chess-cv preprocessing pieces  # or 'arrows' or 'snap'

# Train model
chess-cv train pieces  # or 'arrows' or 'snap'

# Evaluate model
chess-cv test pieces  # or 'arrows' or 'snap'
```

See the [Setup Guide](https://s1m0n38.github.io/chess-cv/setup/) and [Train and Evaluate](https://s1m0n38.github.io/chess-cv/train-and-eval/) for detailed instructions on data generation, training configuration, and evaluation.

## Limitations

- Requires precisely cropped 32×32 pixel square images (no board detection)
- Trained on synthetic data; may not generalize to real-world photos
- Not suitable for non-standard piece designs
- Optimized for Apple Silicon (slower on CPU)

For detailed documentation, architecture details, and advanced usage, see the [full documentation](https://s1m0n38.github.io/chess-cv/).

## Citation

```bibtex
@software{bertolotto2025chesscv,
  author = {Bertolotto, Simone},
  title = {{Chess CV}},
  url = {https://github.com/S1M0N38/chess-cv},
  year = {2025}
}
```

<div align="center">

**Repo:** [github.com/S1M0N38/chess-cv](https://github.com/S1M0N38/chess-cv) • **PyPI:** [pypi.org/project/chess-cv](https://pypi.org/project/chess-cv/)

</div>
