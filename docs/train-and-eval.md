# Train and Evaluate

Learn how to generate data, train models, and evaluate performance with Chess CV.

## Data Generation

### Basic Usage

Generate synthetic chess piece images:

```bash
# Using default settings (70% train, 15% val, 15% test)
chess-cv preprocessing
```

### Custom Configuration

```bash
chess-cv preprocessing \
  --train-dir data/splits/pieces/train \
  --val-dir data/splits/pieces/validate \
  --test-dir data/splits/pieces/test \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

### Understanding Data Generation

The preprocessing script:

1. Reads board images from `data/boards/`
2. Reads piece sets from `data/pieces/`
3. For each board-piece combination:
    - Renders pieces onto the board
    - Extracts 32×32 pixel squares
    - Saves images to train/validate/test directories
4. Splits data according to specified ratios

**Output:**

- **Train set**: ~65,000 images (70%)
- **Validation set**: ~14,000 images (15%)
- **Test set**: ~14,000 images (15%)

Each set contains balanced examples of all 13 classes.

## Model Training

### Basic Training

Train with default settings:

```bash
chess-cv train
```

### Custom Training Configuration

```bash
chess-cv train \
  --train-dir data/splits/pieces/train \
  --val-dir data/splits/pieces/validate \
  --checkpoint-dir checkpoints \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --weight-decay 0.0003 \
  --num-epochs 200 \
  --patience 999999 \
  --image-size 32 \
  --num-workers 8
```

### Training Parameters

**Optimizer Settings:**

- `--learning-rate`: Learning rate for AdamW optimizer (default: 0.0003)
- `--weight-decay`: Weight decay for regularization (default: 0.0003)

**Training Control:**

- `--num-epochs`: Maximum number of epochs (default: 200)
- `--patience`: Early stopping patience (default: 999999, effectively disabled)
- `--batch-size`: Batch size for training (default: 64)

**Data Settings:**

- `--image-size`: Input image size (default: 32)
- `--num-workers`: Number of data loading workers (default: 8)

**Directories:**

- `--train-dir`: Training data directory (default: data/splits/pieces/train)
- `--val-dir`: Validation data directory (default: data/splits/pieces/validate)
- `--checkpoint-dir`: Where to save model checkpoints (default: checkpoints)

### Training Features

**Data Augmentation:**

- Random resized crop (scale: 0.8-1.0)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation: ±0.2)
- Random rotation (±5°)
- Gaussian noise (std: 0.05)

**Early Stopping:**

Training can automatically stop if validation accuracy doesn't improve for `--patience` epochs. By default, patience is set very high (999999) to effectively disable early stopping, allowing the full training schedule to run.

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
chess-cv train --wandb
```

**Features**: Real-time metric logging, hyperparameter tracking, model comparison, and experiment organization.

### Hyperparameter Sweeps

Optimize hyperparameters with W&B sweeps:

```bash
# Initialize sweep with configuration
wandb sweep sweep.yaml

# Run sweep agent
wandb agent your-entity/chess-cv/sweep-id
```

Example `sweep.yaml`:

```yaml
program: chess-cv
command:
  - train
  - --wandb
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform
    min: 0.0001
    max: 0.001
  batch_size:
    values: [32, 64, 128]
  weight_decay:
    distribution: log_uniform
    min: 0.0001
    max: 0.001
```

## Model Evaluation

### Basic Evaluation

Evaluate trained model on test set:

```bash
chess-cv test
```

### Custom Evaluation

```bash
chess-cv test \
  --test-dir data/splits/pieces/test \
  --train-dir data/splits/pieces/train \
  --checkpoint checkpoints/pieces/pieces.safetensors \
  --batch-size 64 \
  --image-size 32 \
  --num-workers 8 \
  --output-dir outputs
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

Share your trained model on Hugging Face Hub:

```bash
# First time setup
hf login

# Upload model
chess-cv upload --repo-id username/chess-cv
```

**Options:**

```bash
chess-cv upload \
  --repo-id username/chess-cv \
  --checkpoint-dir checkpoints \
  --message "feat: initial model release" \
  --private  # Optional: create private repository
```

**What gets uploaded**: Model weights, model card with metadata, training visualizations, and model configuration.

## Troubleshooting

**Out of Memory During Training**: Reduce batch size with `--batch-size 64` or reduce number of workers with `--num-workers 2`.

**Poor Model Performance**: Try adjusting hyperparameters, use W&B sweeps for optimization, or review misclassified images to verify data quality. Enable early stopping with lower patience (`--patience 15`) to save time during hyperparameter tuning.

**Training Too Slow**: Increase batch size if memory allows (`--batch-size 128`), or enable early stopping (`--patience 15`) for faster iteration during experimentation.

**Evaluation Issues**: Ensure the checkpoint exists, verify the test data directory is populated, and run with appropriate batch size.

## Next Steps

- Use your trained model for inference with [Model Usage](inference.md)
- Explore model internals with [Architecture](architecture.md)
- Share your model on Hugging Face Hub using the upload command above
