# Inference

Learn how to use Chess CV as a library and load pre-trained models from Hugging Face Hub.

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

## Troubleshooting

**Model Loading Issues**: If model loading fails, verify the file path exists and that the model architecture matches the weights (use `SimpleCNN(num_classes=13)`).

**Memory Issues**: Process images in smaller batches if you encounter memory problems during batch prediction.

## Next Steps

- Explore the [Architecture](architecture.md) documentation for model details
- Check out [Train and Evaluate](train-and-eval.md) for training custom models
