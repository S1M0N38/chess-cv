# Model Usage

Learn how to use Chess CV as a library and load pre-trained models from Hugging Face Hub.

## Installation

### For Model Usage Only

If you only want to use pre-trained models:

```bash
pip install chess-cv
```

Or with uv:

```bash
uv add chess-cv
# or
uv pip install chess-cv
```

### For Training Your Own Models

If you plan to train your own models, clone the repository:

```bash
git clone https://github.com/S1M0N38/chess-cv.git
cd chess-cv

# Copy environment template
cp .envrc.example .envrc

# Install all dependencies
uv sync --all-extras
```

## Using Pre-trained Models

### Loading from Hugging Face Hub

Load a pre-trained model directly from Hugging Face:

```python
import mlx.core as mx
from huggingface_hub import hf_hub_download
from chess_cv.model import SimpleCNN

# Download model weights from Hugging Face
model_path = hf_hub_download(
    repo_id="S1M0N38/chess-cv",
    filename="best_model.safetensors"
)

# Create model and load weights
model = SimpleCNN(num_classes=13)
model.load_weights(model_path)
model.eval()

print("Model loaded successfully!")
```

### Making Predictions

Classify a chess square image:

```python
import mlx.core as mx
import numpy as np
from PIL import Image

# Load and preprocess image
def preprocess_image(image_path: str) -> mx.array:
    """Load and preprocess a chess square image.
    
    Args:
        image_path: Path to 32×32 RGB image
        
    Returns:
        Preprocessed image tensor ready for model
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension and convert to MLX array
    # MLX uses NHWC format: (batch, height, width, channels)
    img_tensor = mx.array(img_array[None, ...])
    
    return img_tensor

# Class labels
CLASSES = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 
           'wB', 'wK', 'wN', 'wP', 'wQ', 'wR', 'xx']

# Make prediction
image_tensor = preprocess_image("square.png")
logits = model(image_tensor)
probabilities = mx.softmax(logits, axis=-1)
predicted_class = mx.argmax(probabilities, axis=-1).item()

print(f"Predicted class: {CLASSES[predicted_class]}")
print(f"Confidence: {probabilities[0, predicted_class].item():.2%}")
```

### Batch Predictions

Process multiple images efficiently:

```python
import mlx.core as mx
from pathlib import Path

def predict_batch(model: SimpleCNN, image_paths: list[str]) -> list[dict]:
    """Predict classes for multiple images.
    
    Args:
        model: Trained SimpleCNN model
        image_paths: List of paths to chess square images
        
    Returns:
        List of prediction dictionaries with class and confidence
    """
    # Preprocess all images
    images = [preprocess_image(path) for path in image_paths]
    batch = mx.concatenate(images, axis=0)
    
    # Make predictions
    logits = model(batch)
    probabilities = mx.softmax(logits, axis=-1)
    predicted_classes = mx.argmax(probabilities, axis=-1)
    
    # Format results
    results = []
    for i, path in enumerate(image_paths):
        pred_idx = predicted_classes[i].item()
        confidence = probabilities[i, pred_idx].item()
        results.append({
            'path': path,
            'class': CLASSES[pred_idx],
            'confidence': confidence
        })
    
    return results

# Example usage
image_paths = ["square1.png", "square2.png", "square3.png"]
predictions = predict_batch(model, image_paths)

for pred in predictions:
    print(f"{pred['path']}: {pred['class']} ({pred['confidence']:.2%})")
```

## Using the Chess CV Library

### Training a Custom Model

Train your own model with custom data:

```python
from chess_cv.model import create_model
from chess_cv.train import train_model
from chess_cv.data import create_dataloaders

# Create data loaders
train_loader, val_loader = create_dataloaders(
    train_dir="data/train",
    val_dir="data/validate",
    batch_size=128,
    image_size=32,
    num_workers=4
)

# Create model
model = create_model(num_classes=13)

# Train
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=0.0002,
    weight_decay=0.0005,
    patience=15,
    checkpoint_dir="checkpoints"
)
```

### Evaluating Model Performance

Evaluate a trained model:

```python
from chess_cv.test import evaluate_model
from chess_cv.data import create_test_dataloader

# Load model
model = SimpleCNN(num_classes=13)
model.load_weights("checkpoints/best_model.safetensors")
model.eval()

# Create test loader
test_loader = create_test_dataloader(
    test_dir="data/test",
    batch_size=256,
    image_size=32
)

# Evaluate
results = evaluate_model(
    model=model,
    test_loader=test_loader,
    output_dir="outputs"
)

print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"Per-class accuracy: {results['per_class_accuracy']}")
```

## Real-time Inference

Optimize for real-time classification:

```python
import mlx.core as mx
from chess_cv.model import SimpleCNN
import time

class ChessSquareClassifier:
    """Real-time chess square classifier."""
    
    def __init__(self, model_path: str):
        """Initialize classifier with pre-trained model."""
        self.model = SimpleCNN(num_classes=13)
        self.model.load_weights(model_path)
        self.model.eval()
        
        self.classes = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR',
                       'wB', 'wK', 'wN', 'wP', 'wQ', 'wR', 'xx']
    
    def classify(self, image: mx.array) -> tuple[str, float]:
        """Classify a single chess square.
        
        Args:
            image: Preprocessed image tensor (1, 32, 32, 3)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        logits = self.model(image)
        probabilities = mx.softmax(logits, axis=-1)
        pred_idx = mx.argmax(probabilities, axis=-1).item()
        confidence = probabilities[0, pred_idx].item()
        
        return self.classes[pred_idx], confidence
    
    def classify_board(self, board_squares: list[mx.array]) -> list[str]:
        """Classify all 64 squares of a chessboard.
        
        Args:
            board_squares: List of 64 preprocessed square images
            
        Returns:
            List of 64 piece labels
        """
        # Batch process for efficiency
        batch = mx.concatenate(board_squares, axis=0)
        logits = self.model(batch)
        predictions = mx.argmax(logits, axis=-1)
        
        return [self.classes[pred.item()] for pred in predictions]

# Example usage
classifier = ChessSquareClassifier("checkpoints/best_model.safetensors")

# Classify single square
piece, conf = classifier.classify(square_image)
print(f"Detected: {piece} ({conf:.2%} confidence)")

# Classify entire board
board_state = classifier.classify_board(all_64_squares)
print(f"Board FEN: {convert_to_fen(board_state)}")
```

## Troubleshooting

**Model Loading Issues**: If model loading fails, verify the file path exists and that the model architecture matches the weights (use `SimpleCNN(num_classes=13)`).

**Memory Issues**: Process images in smaller batches if you encounter memory problems during batch prediction.

**Performance**: Warm up the model with a dummy input before timing inference for accurate performance measurements.

## Next Steps

- Explore the [Architecture](architecture.md) documentation for model details
- Check out [Usage](usage.md) for training custom models
- See [Setup](setup.md) for development environment configuration
