# Inference

Learn how to use Chess CV as a library and load pre-trained models.

## Using Pre-trained Models

### Using Bundled Models (Recommended)

The chess-cv package includes pre-trained weights for all three models. This is the simplest way to get started:

```python
from chess_cv import load_bundled_model

# Load models with bundled weights (included in package)
pieces_model = load_bundled_model('pieces')
arrows_model = load_bundled_model('arrows')
snap_model = load_bundled_model('snap')

# Set to evaluation mode
pieces_model.eval()
arrows_model.eval()
snap_model.eval()

print("Models loaded successfully!")
```

**Get model configuration:**

```python
from chess_cv.constants import get_model_config

# Get class names and other config for each model
pieces_config = get_model_config('pieces')
print(f"Pieces classes: {pieces_config['class_names']}")  # ['bB', 'bK', ..., 'xx']
print(f"Number of classes: {pieces_config['num_classes']}")  # 13

arrows_config = get_model_config('arrows')
print(f"Number of arrow classes: {arrows_config['num_classes']}")  # 49

snap_config = get_model_config('snap')
print(f"Snap classes: {snap_config['class_names']}")  # ['ok', 'bad']
print(f"Number of classes: {snap_config['num_classes']}")  # 2
```

**Advanced: Get bundled weight paths:**

```python
from chess_cv import get_bundled_weight_path
from chess_cv.model import SimpleCNN
import mlx.core as mx

# Get path to bundled weights
weight_path = get_bundled_weight_path('pieces')
print(f"Bundled weights location: {weight_path}")

# Load manually if needed
model = SimpleCNN(num_classes=13)
weights = mx.load(str(weight_path))
model.load_weights(list(weights.items()))
model.eval()
```

### Loading Latest Version from Hugging Face Hub

To get the latest version of the models (if updated after package release):

```python
import mlx.core as mx
from huggingface_hub import hf_hub_download
from chess_cv.model import SimpleCNN

# Download latest model weights from Hugging Face
model_path = hf_hub_download(
    repo_id="S1M0N38/chess-cv",
    filename="pieces.safetensors"
)

# Create model and load weights
model = SimpleCNN(num_classes=13)
weights = mx.load(str(model_path))
model.load_weights(list(weights.items()))
model.eval()

print("Model loaded successfully!")
```

### Making Predictions

#### Pieces Model - Classify Chess Pieces

Classify a chess square image to identify pieces:

```python
import mlx.core as mx
import numpy as np
from PIL import Image
from chess_cv import load_bundled_model
from chess_cv.constants import get_model_config

# Load model
model = load_bundled_model('pieces')
model.eval()

# Get class names
classes = get_model_config('pieces')['class_names']

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

# Make prediction
image_tensor = preprocess_image("square.png")
logits = model(image_tensor)
probabilities = mx.softmax(logits, axis=-1)
predicted_class = mx.argmax(probabilities, axis=-1).item()

print(f"Predicted class: {classes[predicted_class]}")
print(f"Confidence: {probabilities[0, predicted_class].item():.2%}")
```

#### Arrows Model - Classify Arrow Components

Classify arrow overlay components on chess board squares:

```python
import mlx.core as mx
import numpy as np
from PIL import Image
from chess_cv import load_bundled_model
from chess_cv.constants import get_model_config

# Load arrows model
model = load_bundled_model('arrows')
model.eval()

# Get class names (49 arrow components + empty)
classes = get_model_config('arrows')['class_names']

# Preprocess and predict (using same preprocess_image function as above)
image_tensor = preprocess_image("square_with_arrow.png")
logits = model(image_tensor)
probabilities = mx.softmax(logits, axis=-1)
predicted_class = mx.argmax(probabilities, axis=-1).item()

print(f"Predicted arrow component: {classes[predicted_class]}")
print(f"Confidence: {probabilities[0, predicted_class].item():.2%}")
```

**Arrow Classes**: The arrows model classifies 49 components including arrow heads (e.g., `head-N`, `head-SE`), tails (e.g., `tail-W`), middle segments (e.g., `middle-N-S`), corners (e.g., `corner-N-E`), and empty squares (`xx`).

#### Snap Model - Classify Piece Centering

Detect whether chess pieces are properly centered in squares:

```python
import mlx.core as mx
import numpy as np
from PIL import Image
from chess_cv import load_bundled_model
from chess_cv.constants import get_model_config

# Load snap model
model = load_bundled_model('snap')
model.eval()

# Get class names ('ok' or 'bad')
classes = get_model_config('snap')['class_names']

# Preprocess and predict (using same preprocess_image function as above)
image_tensor = preprocess_image("square_to_check.png")
logits = model(image_tensor)
probabilities = mx.softmax(logits, axis=-1)
predicted_class = mx.argmax(probabilities, axis=-1).item()

print(f"Piece centering: {classes[predicted_class]}")
print(f"Confidence: {probabilities[0, predicted_class].item():.2%}")

if classes[predicted_class] == 'bad':
    print("⚠️ Piece is off-centered - needs adjustment")
else:
    print("✓ Piece is properly centered or square is empty")
```

**Use Cases**: The snap model is useful for automated board state validation, piece positioning quality control, and chess interface usability testing.

### Batch Predictions

Process multiple images efficiently:

```python
import mlx.core as mx
from pathlib import Path
from chess_cv import load_bundled_model
from chess_cv.constants import get_model_config
from chess_cv.model import SimpleCNN

# Load model and get class names
model = load_bundled_model('pieces')
model.eval()
classes = get_model_config('pieces')['class_names']

def predict_batch(model: SimpleCNN, image_paths: list[str], classes: list[str]) -> list[dict]:
    """Predict classes for multiple images.

    Args:
        model: Trained SimpleCNN model
        image_paths: List of paths to chess square images
        classes: List of class names

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
            'class': classes[pred_idx],
            'confidence': confidence
        })

    return results

# Example usage
image_paths = ["square1.png", "square2.png", "square3.png"]
predictions = predict_batch(model, image_paths, classes)

for pred in predictions:
    print(f"{pred['path']}: {pred['class']} ({pred['confidence']:.2%})")
```

## Troubleshooting

**Model Loading Issues**: If model loading fails, verify the file path exists and that the model architecture matches the weights. Use `SimpleCNN(num_classes=13)` for pieces, `SimpleCNN(num_classes=49)` for arrows, or `SimpleCNN(num_classes=2)` for snap.

**Memory Issues**: Process images in smaller batches if you encounter memory problems during batch prediction.

**Wrong Predictions**: Ensure input images are properly preprocessed (32×32px, RGB, normalized to [0,1]).

## Next Steps

- Explore the [Architecture](../architecture/) documentation for model details
- Check out [Train and Evaluate](../train-and-eval/) for training custom models
