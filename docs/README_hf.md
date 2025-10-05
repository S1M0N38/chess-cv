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
            value: 0.9985
            name: Accuracy
            verified: false
          - type: f1
            value: 0.9989
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
            value: 0.9757
            name: Accuracy
            verified: false
          - type: f1
            value: 0.9578
            name: F1 Score (Macro)
            verified: false
pipeline_tag: image-classification
---

# Chess CV

<div align="center">
  <img src="https://raw.githubusercontent.com/S1M0N38/chess-cv/main/docs/assets/model.svg" alt="Model Architecture" width="600">
</div>

Lightweight CNN (156k parameters) that classifies chess pieces from 32×32 pixel square images into 13 classes (6 white pieces, 6 black pieces, empty square). Trained on synthetic data from chess.com/lichess boards and piece sets.

| Dataset                                                                                  | Accuracy | F1-Score (Macro) |
| ---------------------------------------------------------------------------------------- | -------- | ---------------- |
| Test Data                                                                                | 99.85%   | 99.89%           |
| [S1M0N38/chess-cv-openboard](https://huggingface.co/datasets/S1M0N38/chess-cv-openboard) | \*       | 95.78%           |

\* OpenBoard has an unbalanced class distribution (many more samples for empty square class), so accuracy is not representative.

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
model_path = hf_hub_download(repo_id="S1M0N38/chess-cv", filename="best_model.safetensors")
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

## Training Your Own Model

To train or evaluate the model yourself:

```bash
git clone https://github.com/S1M0N38/chess-cv.git
cd chess-cv
uv sync --all-extras

# Generate training data
python -m chess_cv.preprocessing

# Train model
python -m chess_cv.train

# Evaluate model
python -m chess_cv.test
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
@software{bertolotto2024chesscv,
  author = {Bertolotto, Simone},
  title = {Chess CV: Lightweight CNN for Chess Piece Recognition},
  year = {2025},
  url = {https://github.com/S1M0N38/chess-cv}
}
```

**Repository:** [github.com/S1M0N38/chess-cv](https://github.com/S1M0N38/chess-cv) • **PyPI:** [pypi.org/project/chess-cv](https://pypi.org/project/chess-cv/)
