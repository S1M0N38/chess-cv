#!/usr/bin/env python3
"""Generate augmentation examples for documentation.

This script generates side-by-side original/augmented image pairs for both
pieces and arrows models, saving them to docs/assets/{pieces,arrows}/ directories.
"""

# Add parent directory to path for imports
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chess_cv.constants import (
    AUGMENTATION_CONFIGS,
    DEFAULT_ARROW_DIR,
    DEFAULT_HIGHLIGHT_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MOUSE_DIR,
)
from src.chess_cv.data import (
    RandomArrowOverlay,
    RandomHighlightOverlay,
    RandomMouseOverlay,
)

# Hard-coded image paths
PIECES_IMAGES = [
    "data/splits/pieces/train/wP/chess-com-graffiti_dark_lichess-california.png",
    "data/splits/pieces/train/wP/chess-com-metal_dark_chess-com-space.png",
    "data/splits/pieces/train/bQ/chess-com-graffiti_dark_lichess-california.png",
    "data/splits/pieces/train/bQ/chess-com-metal_dark_chess-com-space.png",
]

ARROWS_IMAGES = [
    "data/splits/arrows/train/head-N/chess-com-stone_light_wK_chess-com-neo_chess-com-yellow.png",
    "data/splits/arrows/train/head-N/lichess-ncf-board_light_wQ_lichess-firi_lichess-yellow.png",
    "data/splits/arrows/train/head-SE/chess-com-graffiti_light_bN_chess-com-neo_wood_lichess-blue.png",
    "data/splits/arrows/train/head-SE/chess-com-orange_light_bR_lichess-caliente_chess-com-blue.png",
]

SNAP_IMAGES = [
    "data/splits/snap/train/bad/chess-com-8_bit_dark_bB_chess-com-alpha_var0.png",
    "data/splits/snap/train/bad/lichess-canvas2_dark_bN_lichess-pirouetti_var3.png",
    "data/splits/snap/train/ok/chess-com-8_bit_dark_bB_chess-com-8_bit_var2.png",
    "data/splits/snap/train/ok/lichess-canvas2_light_wR_lichess-kosal_var0.png",
]


def build_augmentation_pipeline(model_id: str):
    """Build augmentation pipeline for the given model."""
    aug_config = AUGMENTATION_CONFIGS[model_id]
    train_transform_list = []

    # New pipeline for pieces model using v2 transforms
    if model_id == "pieces":
        # Step 1: Expand canvas for rotation space
        train_transform_list.append(
            v2.Pad(
                padding=aug_config["padding"], padding_mode=aug_config["padding_mode"]
            )
        )

        # Step 2: Rotate with black fill (will be cropped out)
        train_transform_list.append(
            v2.RandomRotation(degrees=aug_config["rotation_degrees"], fill=0)
        )

        # Step 3: Remove black corners from rotation
        train_transform_list.append(v2.CenterCrop(size=aug_config["center_crop_size"]))

        # Step 4: Random crop + scale variation + resize back to 32×32
        train_transform_list.append(
            v2.RandomResizedCrop(
                size=aug_config["final_size"],
                scale=aug_config["resized_crop_scale"],
                ratio=aug_config["resized_crop_ratio"],
                antialias=True,
            )
        )

        # Step 5: Arrow overlay (after geometric transforms for crisp graphics)
        if aug_config["arrow_probability"] > 0:
            train_transform_list.append(
                RandomArrowOverlay(
                    arrow_dir=DEFAULT_ARROW_DIR,
                    probability=aug_config["arrow_probability"],
                )
            )

        # Step 6: Highlight overlay
        if aug_config["highlight_probability"] > 0:
            train_transform_list.append(
                RandomHighlightOverlay(
                    highlight_dir=DEFAULT_HIGHLIGHT_DIR,
                    probability=aug_config["highlight_probability"],
                )
            )

        # Step 7: Mouse overlay
        if aug_config["mouse_probability"] > 0:
            train_transform_list.append(
                RandomMouseOverlay(
                    mouse_dir=DEFAULT_MOUSE_DIR,
                    probability=aug_config["mouse_probability"],
                    aug_config=aug_config,
                )
            )

        # Step 8: Horizontal flip
        if aug_config["horizontal_flip"]:
            train_transform_list.append(
                v2.RandomHorizontalFlip(p=aug_config["horizontal_flip_prob"])
            )

        # Step 9: Color jitter
        train_transform_list.append(
            v2.ColorJitter(
                brightness=aug_config["brightness"],
                contrast=aug_config["contrast"],
                saturation=aug_config["saturation"],
                hue=aug_config["hue"],
            )
        )

        # Step 10: Convert to tensor, apply Gaussian noise
        # GaussianNoise requires tensor input, not PIL
        train_transform_list.append(v2.ToImage())
        train_transform_list.append(v2.ToDtype(dtype=torch.float32, scale=True))
        train_transform_list.append(
            v2.GaussianNoise(
                mean=aug_config["noise_mean"],
                sigma=aug_config["noise_sigma"],
            )
        )
        train_transform_list.append(v2.ToPILImage())

    elif model_id == "arrows":
        # New pipeline for arrows model using v2 transforms
        # Step 1: Highlight overlay (applied early, before geometric transforms)
        if aug_config["highlight_probability"] > 0:
            train_transform_list.append(
                RandomHighlightOverlay(
                    highlight_dir=DEFAULT_HIGHLIGHT_DIR,
                    probability=aug_config["highlight_probability"],
                )
            )

        # Step 2: Color jitter (with hue now included)
        train_transform_list.append(
            v2.ColorJitter(
                brightness=aug_config["brightness"],
                contrast=aug_config["contrast"],
                saturation=aug_config["saturation"],
                hue=aug_config["hue"],
            )
        )

        # Step 3: Small rotation (±2 degrees)
        train_transform_list.append(
            v2.RandomRotation(degrees=aug_config["rotation_degrees"])
        )

        # Step 4: Convert to tensor, apply Gaussian noise
        # GaussianNoise requires tensor input, not PIL
        train_transform_list.append(v2.ToImage())
        train_transform_list.append(v2.ToDtype(dtype=torch.float32, scale=True))
        train_transform_list.append(
            v2.GaussianNoise(
                mean=aug_config["noise_mean"],
                sigma=aug_config["noise_sigma"],
            )
        )
        train_transform_list.append(v2.ToPILImage())

    elif model_id == "snap":
        # Pipeline for snap model using v2 transforms
        # Step 1: Arrow overlay (simulated arrow graphics on pieces)
        if aug_config["arrow_probability"] > 0:
            train_transform_list.append(
                RandomArrowOverlay(
                    arrow_dir=DEFAULT_ARROW_DIR,
                    probability=aug_config["arrow_probability"],
                )
            )

        # Step 2: Highlight overlay
        if aug_config["highlight_probability"] > 0:
            train_transform_list.append(
                RandomHighlightOverlay(
                    highlight_dir=DEFAULT_HIGHLIGHT_DIR,
                    probability=aug_config["highlight_probability"],
                )
            )

        # Step 3: Mouse overlay
        if aug_config["mouse_probability"] > 0:
            train_transform_list.append(
                RandomMouseOverlay(
                    mouse_dir=DEFAULT_MOUSE_DIR,
                    probability=aug_config["mouse_probability"],
                    aug_config=aug_config,
                )
            )

        # Step 4: Horizontal flip
        if aug_config["horizontal_flip"]:
            train_transform_list.append(
                v2.RandomHorizontalFlip(p=aug_config["horizontal_flip_prob"])
            )

        # Step 5: Color jitter
        train_transform_list.append(
            v2.ColorJitter(
                brightness=aug_config["brightness"],
                contrast=aug_config["contrast"],
                saturation=aug_config["saturation"],
                hue=aug_config["hue"],
            )
        )

        # Step 6: Convert to tensor, apply Gaussian noise
        train_transform_list.append(v2.ToImage())
        train_transform_list.append(v2.ToDtype(dtype=torch.float32, scale=True))
        train_transform_list.append(
            v2.GaussianNoise(
                mean=aug_config["noise_mean"],
                sigma=aug_config["noise_sigma"],
            )
        )
        train_transform_list.append(v2.ToPILImage())

    else:
        raise ValueError(f"Unknown model ID: {model_id}")

    return v2.Compose(train_transform_list)


def generate_examples(model_id: str, image_paths: list[str], output_dir: Path):
    """Generate original and augmented image pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = DEFAULT_IMAGE_SIZE
    train_transforms = build_augmentation_pipeline(model_id)

    for idx, img_path in enumerate(image_paths, start=1):
        if not Path(img_path).exists():
            print(f"⚠️  Image not found: {img_path}")
            continue

        # Load original image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

        # Save original
        original_output = output_dir / f"{idx:02d}-original.png"
        img.save(original_output)
        print(f"✅ Saved: {original_output}")

        # Apply augmentation and save
        aug_img = train_transforms(img)
        augmented_output = output_dir / f"{idx:02d}-augmented.png"
        aug_img.save(augmented_output)
        print(f"✅ Saved: {augmented_output}")


def main():
    """Generate augmentation examples for all models."""
    script_dir = Path(__file__).parent

    # Generate pieces examples
    print("\n🎨 Generating pieces augmentation examples...")
    pieces_dir = script_dir / "pieces"
    generate_examples("pieces", PIECES_IMAGES, pieces_dir)

    # Generate arrows examples
    print("\n🎯 Generating arrows augmentation examples...")
    arrows_dir = script_dir / "arrows"
    generate_examples("arrows", ARROWS_IMAGES, arrows_dir)

    # Generate snap examples
    print("\n📐 Generating snap augmentation examples...")
    snap_dir = script_dir / "snap"
    generate_examples("snap", SNAP_IMAGES, snap_dir)

    print("\n✨ Done! Generated augmentation examples in docs/assets/")


if __name__ == "__main__":
    main()
