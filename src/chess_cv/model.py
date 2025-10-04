"""CNN model architecture for chess piece classification."""

import mlx.core as mx
import mlx.nn as nn

__all__ = ["SimpleCNN", "create_model"]


class SimpleCNN(nn.Module):
    """Simple CNN for 32x32 RGB image classification.

    Architecture:
        Conv2d(3→16) → ReLU → MaxPool2d(2) → [32x32 → 16x16]
        Conv2d(16→32) → ReLU → MaxPool2d(2) → [16x16 → 8x8]
        Conv2d(32→64) → ReLU → MaxPool2d(2) → [8x8 → 4x4]
        Flatten → Linear(1024→128) → ReLU → Dropout(0.5)
        Linear(128→num_classes)
    """

    def __init__(self, num_classes: int = 13, dropout: float = 0.5):
        """Initialize the CNN model.

        Args:
            num_classes: Number of output classes (default: 13 for chess pieces)
            dropout: Dropout probability (default: 0.5)
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        # Pooling layer (reused)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 3 pooling layers: 32 -> 16 -> 8 -> 4
        # Feature map size: 4x4x64 = 1024
        self.fc1 = nn.Linear(input_dims=64 * 4 * 4, output_dims=128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(input_dims=128, output_dims=num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, height, width, channels)
               MLX uses NHWC format by default

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Block 1: Conv → ReLU → Pool
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.pool(x)

        # Block 2: Conv → ReLU → Pool
        x = self.conv2(x)
        x = nn.relu(x)
        x = self.pool(x)

        # Block 3: Conv → ReLU → Pool
        x = self.conv3(x)
        x = nn.relu(x)
        x = self.pool(x)

        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Fully connected layers
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def create_model(num_classes: int = 13) -> SimpleCNN:
    """Create and initialize a SimpleCNN model.

    Args:
        num_classes: Number of output classes

    Returns:
        Initialized SimpleCNN model
    """
    model = SimpleCNN(num_classes=num_classes)
    # Initialize parameters by evaluating them
    mx.eval(model.parameters())
    return model
