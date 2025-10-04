"""Convert and optimize chess piece images to 32x32 PNG format with standardized naming."""

import argparse
from pathlib import Path

from PIL import Image

try:
    import cairosvg

    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False


def get_standard_name(filename: str) -> str:
    """Convert filename to standard naming convention: <lowercase_color><UPPERCASE_piece>.png

    Args:
        filename: Original filename (e.g., 'bb.png', 'bK.svg')

    Returns:
        Standardized filename (e.g., 'bB.png', 'bK.png')
    """
    stem = Path(filename).stem.lower()

    # Map piece codes to uppercase
    piece_map = {"k": "K", "q": "Q", "r": "R", "b": "B", "n": "N", "p": "P"}

    if len(stem) == 2:
        color = stem[0]  # b or w
        piece = stem[1]  # k, q, r, b, n, p

        if color in ("b", "w") and piece in piece_map:
            return f"{color}{piece_map[piece]}.png"

    # If already in correct format or unknown format, return with .png extension
    return Path(filename).stem + ".png"


def process_png_file(
    png_file: Path,
    target_size: tuple[int, int],
) -> tuple[str, int, int]:
    """Process a PNG file: resize and rename.

    Args:
        png_file: Path to PNG file
        target_size: Target dimensions (width, height)

    Returns:
        Tuple of (output_filename, original_size, new_size)
    """
    # Load image
    img = Image.open(png_file)
    original_size = png_file.stat().st_size

    # Resize with high-quality resampling
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to RGBA to preserve transparency
    if img_resized.mode != "RGBA":
        img_resized = img_resized.convert("RGBA")

    # Get standardized filename
    new_name = get_standard_name(png_file.name)
    output_path = png_file.parent / new_name

    # Handle case-insensitive filesystems (macOS, Windows)
    # If names differ only in case, use temporary file to avoid overwriting
    if output_path != png_file and output_path.name.lower() == png_file.name.lower():
        temp_path = png_file.parent / f".temp_{new_name}"
        img_resized.save(temp_path, "PNG", optimize=True, compress_level=9)
        png_file.unlink()
        temp_path.rename(output_path)
    else:
        # Save as optimized PNG
        img_resized.save(output_path, "PNG", optimize=True, compress_level=9)
        # Delete original if renamed
        if output_path != png_file:
            png_file.unlink()

    new_size = output_path.stat().st_size

    return new_name, original_size, new_size


def process_svg_file(
    svg_file: Path,
    target_size: tuple[int, int],
) -> tuple[str, int, int]:
    """Process an SVG file: convert to PNG.

    Args:
        svg_file: Path to SVG file
        target_size: Target dimensions (width, height)

    Returns:
        Tuple of (output_filename, original_size, new_size)
    """
    if not HAS_CAIROSVG:
        raise ImportError("cairosvg is required for SVG conversion")

    original_size = svg_file.stat().st_size

    # Get standardized filename (SVGs from lichess already use correct naming)
    new_name = get_standard_name(svg_file.name)
    output_path = svg_file.parent / new_name

    # Convert SVG to PNG at target size
    cairosvg.svg2png(
        url=str(svg_file),
        write_to=str(output_path),
        output_width=target_size[0],
        output_height=target_size[1],
    )

    # Re-optimize the PNG
    img = Image.open(output_path)
    img.save(
        output_path,
        "PNG",
        optimize=True,
        compress_level=9,
    )

    new_size = output_path.stat().st_size

    # Delete original SVG
    svg_file.unlink()

    return new_name, original_size, new_size


def process_piece_images(
    source_dir: Path = Path(__file__).parent,
    target_size: tuple[int, int] = (32, 32),
    skip_svg: bool = False,
    dry_run: bool = False,
) -> None:
    """Convert and resize chess piece images to optimized PNG format.

    Args:
        source_dir: Root directory containing piece set subdirectories
        target_size: Target dimensions for resizing (default: (32, 32))
        skip_svg: Skip SVG files (default: False)
        dry_run: If True, show what would be done without making changes
    """
    # Check for cairosvg if not skipping SVG
    if not skip_svg and not HAS_CAIROSVG:
        print("⚠ cairosvg not installed. SVG files will be skipped.")
        print("  Install with: pip install cairosvg")
        print("  Or run with --skip-svg to process PNG files only\n")
        skip_svg = True

    # Find all piece set directories
    piece_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    piece_dirs = [d for d in piece_dirs if not d.name.startswith(".")]

    if not piece_dirs:
        print("No piece set directories found.")
        return

    print(f"Found {len(piece_dirs)} piece set(s) to process")

    # Collect all files to process
    all_files = []
    for piece_dir in piece_dirs:
        png_files = list(piece_dir.glob("*.png")) + list(piece_dir.glob("*.PNG"))
        svg_files = [] if skip_svg else list(piece_dir.glob("*.svg")) + list(piece_dir.glob("*.SVG"))
        all_files.extend([(piece_dir, f, "png") for f in png_files])
        all_files.extend([(piece_dir, f, "svg") for f in svg_files])

    if not all_files:
        print("No image files found to process.")
        return

    print(f"Found {len(all_files)} total file(s) to process\n")

    if dry_run:
        print("[DRY RUN MODE] No changes will be made.\n")
        for piece_dir in piece_dirs:
            png_files = list(piece_dir.glob("*.png")) + list(piece_dir.glob("*.PNG"))
            svg_files = [] if skip_svg else list(piece_dir.glob("*.svg")) + list(piece_dir.glob("*.SVG"))

            if png_files or svg_files:
                print(f"{piece_dir.name}:")
                for f in png_files:
                    new_name = get_standard_name(f.name)
                    action = "resize + rename" if new_name != f.name else "resize"
                    print(f"  {f.name:15s} → {new_name:15s} ({action})")
                for f in svg_files:
                    new_name = get_standard_name(f.name)
                    print(f"  {f.name:15s} → {new_name:15s} (convert SVG)")
        return

    print(f"Processing images (resizing to {target_size[0]}x{target_size[1]})...\n")

    # Process each directory
    total_original_size = 0
    total_new_size = 0
    total_processed = 0
    dirs_processed = 0

    for piece_dir in piece_dirs:
        png_files = list(piece_dir.glob("*.png")) + list(piece_dir.glob("*.PNG"))
        svg_files = [] if skip_svg else list(piece_dir.glob("*.svg")) + list(piece_dir.glob("*.SVG"))

        if not png_files and not svg_files:
            continue

        dir_original_size = 0
        dir_new_size = 0
        dir_processed = 0

        print(f"{piece_dir.name}:")

        # Process PNG files
        for png_file in png_files:
            try:
                new_name, orig_size, new_size = process_png_file(png_file, target_size)
                dir_original_size += orig_size
                dir_new_size += new_size
                dir_processed += 1
                total_processed += 1

                if new_name != png_file.name:
                    print(f"  ✓ {png_file.name:15s} → {new_name:15s} (resized + renamed)")
                else:
                    print(f"  ✓ {png_file.name:15s} (resized)")

            except Exception as e:
                print(f"  ✗ {png_file.name}: Error - {e}")

        # Process SVG files
        for svg_file in svg_files:
            try:
                new_name, orig_size, new_size = process_svg_file(svg_file, target_size)
                dir_original_size += orig_size
                dir_new_size += new_size
                dir_processed += 1
                total_processed += 1

                print(f"  ✓ {svg_file.name:15s} → {new_name:15s} (converted from SVG)")

            except Exception as e:
                print(f"  ✗ {svg_file.name}: Error - {e}")

        if dir_processed > 0:
            size_diff = dir_original_size - dir_new_size
            size_ratio = (
                (1 - dir_new_size / dir_original_size) * 100 if dir_original_size > 0 else 0
            )
            print(
                f"  → {dir_processed} files: {dir_original_size / 1024:.1f}KB → "
                f"{dir_new_size / 1024:.1f}KB ({size_ratio:+.1f}%)\n"
            )
            total_original_size += dir_original_size
            total_new_size += dir_new_size
            dirs_processed += 1

    # Print summary
    if total_processed > 0:
        total_diff = total_original_size - total_new_size
        total_ratio = (
            (1 - total_new_size / total_original_size) * 100 if total_original_size > 0 else 0
        )

        print("=" * 60)
        print("Summary:")
        print(f"  Directories:   {dirs_processed:3d}")
        print(f"  Files:         {total_processed:3d}")
        print(f"  Original size: {total_original_size / 1024:>8.1f} KB")
        print(f"  Final size:    {total_new_size / 1024:>8.1f} KB")
        print(f"  Size change:   {total_diff / 1024:>+8.1f} KB ({total_ratio:+.1f}%)")
        print("\n✓ Piece image processing complete!")


def main() -> None:
    """Run piece image processing with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Resize and optimize chess piece images to 32x32 PNG format"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Root directory containing piece set subdirectories (default: script directory)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[32, 32],
        metavar=("WIDTH", "HEIGHT"),
        help="Target image size (default: 32 32)",
    )
    parser.add_argument(
        "--skip-svg",
        action="store_true",
        help="Skip SVG files (only process PNG)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    process_piece_images(
        source_dir=args.source_dir,
        target_size=tuple(args.size),
        skip_svg=args.skip_svg,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
