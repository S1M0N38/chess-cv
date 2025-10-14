# Train and Evaluate

Learn how to generate data, train models, and evaluate performance with Chess CV.

## Data Generation

### Basic Usage

Generate synthetic training data for a specific model:

```bash
# Generate data for pieces model (default: 70% train, 15% val, 15% test)
chess-cv preprocessing pieces

# Generate data for arrows model
chess-cv preprocessing arrows

# Generate data for snap model
chess-cv preprocessing snap
```

### Custom Output Directories

If you need to specify custom output directories:

```bash
chess-cv preprocessing pieces \
  --train-dir custom/pieces/train \
  --val-dir custom/pieces/validate \
  --test-dir custom/pieces/test
```

Note: The default 70/15/15 train/val/test split with seed 42 is used. These values are defined in `src/chess_cv/constants.py` and provide consistent, reproducible splits.

### Understanding Data Generation

The preprocessing script generates model-specific training data:

**Pieces Model:**

1. Reads board images from `data/boards/` and piece sets from `data/pieces/`
2. For each board-piece combination:
    - Renders pieces onto the board
    - Extracts 32×32 pixel squares
    - Saves images to train/validate/test directories
3. Generates ~93,000 images split into train (70%), val (15%), test (15%)
4. Balanced across 13 classes (12 pieces + empty square)

**Arrows Model:**

1. Reads board images and arrow overlays from `data/arrows/`
2. For each board-arrow combination:
    - Renders arrow components onto boards
    - Extracts 32×32 pixel squares
3. Generates ~4.5M images for 49 arrow component classes

**Snap Model:**

1. Reads board images and piece sets with positioning variations
2. Generates centered and off-centered piece examples
3. Generates ~1.4M synthetic images for 2 centering classes (ok/bad)

All models use the 70/15/15 train/val/test split with seed 42 for reproducibility.

## Model Training

### Basic Training

Train a specific model with default settings:

```bash
# Train pieces model
chess-cv train pieces

# Train arrows model
chess-cv train arrows

# Train snap model
chess-cv train snap
```

### Custom Training Configuration

```bash
chess-cv train pieces \
  --train-dir data/splits/pieces/train \
  --val-dir data/splits/pieces/validate \
  --checkpoint-dir checkpoints/pieces \
  --batch-size 64 \
  --weight-decay 0.001 \
  --num-epochs 1000 \
  --num-workers 8
```

**Model-Specific Defaults:**

Each model type uses optimized hyperparameters defined in `src/chess_cv/constants.py`. The arrows model, for example, uses larger batch sizes (128) and fewer epochs (20) as it converges faster.

Note: Image size is fixed at 32×32 pixels (model architecture requirement).

### Training Parameters

**Optimizer Settings:**

- `--weight-decay`: Weight decay for regularization (default: 0.001)

**Learning Rate Scheduler (enabled by default):**

- Base LR: 0.001 (peak after warmup)
- Min LR: 1e-5 (end of cosine decay)
- Warmup: 3% of total steps (~30 epochs for 1000-epoch training)

**Training Control:**

- `--num-epochs`: Maximum number of epochs (default: 200, recent models use 1000)
- `--batch-size`: Batch size for training (default: 64)

**Data Settings:**

- `--num-workers`: Number of data loading workers (default: 8)

**Directories:**

- `--train-dir`: Training data directory (default: data/splits/pieces/train)
- `--val-dir`: Validation data directory (default: data/splits/pieces/validate)
- `--checkpoint-dir`: Where to save model checkpoints (default: checkpoints)

### Training Features

**Learning Rate Schedule:**

- Warmup phase: linear increase from 0 to 0.001 over first 3% of steps
- Cosine decay: gradual decrease from 0.001 to 1e-5 over remaining steps

**Data Augmentation:**

- Random resized crop (scale: 0.54-0.74)
- Random horizontal flip
- Color jitter (brightness: ±0.15, contrast/saturation/hue: ±0.2)
- Random rotation (±10°)
- Gaussian noise (std: 0.05)
- Arrow overlay (80% probability)
- Highlight overlay (25% probability)

**Early Stopping:**

Early stopping is disabled by default (patience set to 999999), allowing the full training schedule to run. This default is set in `src/chess_cv/constants.py` and ensures consistent training across runs.

**Automatic Checkpointing:**

- Best model weights saved to `checkpoints/{model-id}/{model-id}.safetensors`
- Optimizer state saved to `checkpoints/optimizer.safetensors`

### Training Output

**Files Generated:**

- `checkpoints/{model-id}/{model-id}.safetensors` – Best model weights
- `checkpoints/optimizer.safetensors` – Optimizer state
- `outputs/training_curves.png` – Loss and accuracy plots
- `outputs/augmentation_example.png` – Example of data augmentation

## Experiment Tracking

### Weights & Biases Integration

Track experiments with the W&B dashboard by adding the `--wandb` flag:

```bash
# First time setup
wandb login

# Train with wandb logging
chess-cv train pieces --wandb
```

**Features**: Real-time metric logging, hyperparameter tracking, model comparison, and experiment organization.

### Hyperparameter Sweeps

Optimize hyperparameters with W&B sweeps using the integrated `--sweep` flag:

```bash
# First time setup
wandb login

# Run hyperparameter sweep for a model (requires --wandb)
chess-cv train pieces --sweep --wandb

# The sweep will use the configuration defined in src/chess_cv/sweep.py
```

**Important**: The `--sweep` flag requires `--wandb` to be enabled. The sweep configuration is defined in `src/chess_cv/sweep.py` and includes parameters like learning rate, batch size, and weight decay optimized for each model type.

## Model Evaluation

### Basic Evaluation

Evaluate a trained model on its test set:

```bash
# Evaluate pieces model
chess-cv test pieces

# Evaluate arrows model
chess-cv test arrows

# Evaluate snap model
chess-cv test snap
```

### Custom Evaluation

```bash
chess-cv test pieces \
  --test-dir data/splits/pieces/test \
  --train-dir data/splits/pieces/train \
  --checkpoint checkpoints/pieces/pieces.safetensors \
  --batch-size 64 \
  --num-workers 8 \
  --output-dir outputs/pieces
```

### Evaluating on External Datasets

Test model performance on HuggingFace datasets:

```bash
# Evaluate on a specific dataset
chess-cv test pieces \
  --hf-test-dir S1M0N38/chess-cv-openboard

# Concatenate all splits from the dataset
chess-cv test pieces \
  --hf-test-dir S1M0N38/chess-cv-chessvision \
  --concat-splits
```

### Evaluation Output

**Files Generated:**

- `outputs/test_confusion_matrix.png` – Confusion matrix heatmap
- `outputs/test_per_class_accuracy.png` – Per-class accuracy bar chart
- `outputs/misclassified_images/` – Misclassified examples for analysis

### Analyzing Results

**Confusion Matrix:**

Shows where the model makes mistakes. Look for:

- High off-diagonal values (common misclassifications)
- Patterns in similar piece types (e.g., knights vs bishops)

**Misclassified Images:**

Review examples in `outputs/misclassified_images/` to understand:

- Which board/piece combinations are challenging
- Whether augmentation needs adjustment
- If more training data would help

## Model Deployment

### Upload to Hugging Face Hub

Share your trained models on Hugging Face Hub:

```bash
# First time setup
hf login

# Upload a specific model
chess-cv upload pieces --repo-id username/chess-cv
```

**Examples:**

```bash
# Upload pieces model with default settings
chess-cv upload pieces --repo-id username/chess-cv

# Upload arrows model with custom message
chess-cv upload arrows --repo-id username/chess-cv \
  --message "feat: improved arrows model v2"

# Upload to private repository
chess-cv upload snap --repo-id username/chess-cv --private

# Specify custom paths
chess-cv upload pieces --repo-id username/chess-cv \
  --checkpoint-dir ./my-checkpoints/pieces \
  --readme docs/custom_README.md
```

**What gets uploaded**: Model weights (`{model-id}.safetensors`), model card with metadata, and model configuration.

## Troubleshooting

**Out of Memory During Training**: Reduce batch size with `--batch-size 64` or reduce number of workers with `--num-workers 2`.

**Poor Model Performance**: Try adjusting hyperparameters with W&B sweeps for optimization, or review misclassified images to verify data quality. To enable early stopping for faster experimentation, modify `DEFAULT_PATIENCE` in `src/chess_cv/constants.py`.

**Training Too Slow**: Increase batch size if memory allows (`--batch-size 128`). For faster experimentation, modify `DEFAULT_PATIENCE` in `src/chess_cv/constants.py` to enable early stopping.

**Evaluation Issues**: Ensure the checkpoint exists, verify the test data directory is populated, and run with appropriate batch size.

## Next Steps

- Use your trained model for inference with [Model Usage](inference.md)
- Explore model internals with [Architecture](architecture.md)
- Share your model on Hugging Face Hub using the upload command above
