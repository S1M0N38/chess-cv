"""Convert and optimize board texture images to 256x256 PNG format."""

import argparse
from pathlib import Path

from PIL import Image


def process_board_images(
    source_dir: Path = Path(__file__).parent,
    target_size: tuple[int, int] = (256, 256),
    dry_run: bool = False,
) -> None:
    """Convert and resize board images to optimized PNG format.

    Args:
        source_dir: Directory containing board images (default: script directory)
        target_size: Target dimensions for resizing (default: (256, 256))
        dry_run: If True, show what would be done without making changes
    """
    # Find all image files (JPEG and PNG)
    image_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(source_dir.glob(pattern))

    image_files = sorted(image_files)

    if not image_files:
        print("No image files found in the directory.")
        return

    # Filter out images that are already 256x256
    files_to_process = []
    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                if img.size != target_size:
                    files_to_process.append(img_file)
        except Exception:
            files_to_process.append(img_file)  # Include if we can't check size

    if not files_to_process:
        print(f"All images are already {target_size[0]}x{target_size[1]}.")
        return

    print(f"Found {len(files_to_process)} image(s) to process:")
    for img_file in files_to_process:
        print(f"  - {img_file.name}")

    if dry_run:
        print("\n[DRY RUN MODE] No changes will be made.")
        print(f"\nWould resize {len(files_to_process)} image(s) to {target_size[0]}x{target_size[1]} PNG:")
        for img_file in files_to_process:
            print(f"  {img_file.name}")
        return

    print(f"\nProcessing images (resizing to {target_size[0]}x{target_size[1]})...")

    total_original_size = 0
    total_optimized_size = 0
    processed = 0

    for img_file in files_to_process:
        try:
            # Track original file size
            original_size = img_file.stat().st_size
            total_original_size += original_size

            # Load image
            img = Image.open(img_file)

            # Resize with high-quality resampling
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to RGB if necessary (remove alpha channel)
            if img_resized.mode in ("RGBA", "LA", "P"):
                img_rgb = Image.new("RGB", img_resized.size, (255, 255, 255))
                if img_resized.mode == "P":
                    img_resized = img_resized.convert("RGB")
                else:
                    img_rgb.paste(img_resized, mask=img_resized.split()[-1] if "A" in img_resized.mode else None)
                    img_resized = img_rgb

            # Save as optimized PNG (always with .png extension)
            png_path = img_file.with_suffix(".png")
            img_resized.save(
                png_path,
                "PNG",
                optimize=True,
                compress_level=9,
            )

            # Track optimized file size
            optimized_size = png_path.stat().st_size
            total_optimized_size += optimized_size

            # Calculate size reduction
            size_diff = original_size - optimized_size
            size_ratio = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0

            # Delete original if it was JPEG or different filename
            if img_file != png_path:
                img_file.unlink()

            processed += 1
            status = f"  ✓ {img_file.name:25s}"
            if img_file != png_path:
                status += f" -> {png_path.name:25s}"
            else:
                status += " (resized)".ljust(28)
            status += f" ({original_size/1024:>6.1f}KB -> {optimized_size/1024:>6.1f}KB, {size_ratio:+.1f}%)"
            print(status)

        except Exception as e:
            print(f"  ✗ {img_file.name}: Error - {e}")

    # Print summary
    if processed > 0:
        total_diff = total_original_size - total_optimized_size
        total_ratio = (1 - total_optimized_size / total_original_size) * 100 if total_original_size > 0 else 0

        print("\nSummary:")
        print(f"  Processed:     {processed:3d} image(s)")
        print(f"  Original size: {total_original_size/1024:>8.1f} KB")
        print(f"  Final size:    {total_optimized_size/1024:>8.1f} KB")
        print(f"  Size change:   {total_diff/1024:>+8.1f} KB ({total_ratio:+.1f}%)")
        print("\n✓ Board image processing complete!")


def main() -> None:
    """Run board image processing with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Resize and optimize board images to 256x256 PNG format"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing board images (default: script directory)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("WIDTH", "HEIGHT"),
        help="Target image size (default: 256 256)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    process_board_images(
        source_dir=args.source_dir,
        target_size=tuple(args.size),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
