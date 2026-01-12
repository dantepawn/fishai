"""
YOLO Keypoint Generation Script

Processes images from an input folder using a YOLO pose estimation model,
generating annotated images with keypoints/bounding boxes and YOLO-format label files.

Usage:
    python keypoint_generation.py --input_folder <path> --output_folder <path> --model_path <path>
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from utils.yolo_utils import generate_labels


def get_image_files(input_folder: Path, filter_lenses: bool = False) -> List[Path]:
    """
    Collect image files from the input folder.

    Args:
        input_folder (Path): Path to input directory.
        filter_lenses (bool): If True, only process l0 and l1 lens images.

    Returns:
        List[Path]: List of image file paths.
    """
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_folder.glob(ext))
    
    if filter_lenses:
        # Only process lens 0 and 1 (segmentation handles lens 2 and 3)
        image_files = [x for x in image_files if "l0" in x.stem or "l1" in x.stem]
    
    return sorted(image_files)


def process_images(
    model: YOLO,
    image_files: List[Path],
    output_folder: Path,
    batch_size: int = 8,
    confidence: float = 0.2,
    line_width: int = 8,
    kpt_radius: int = 10
) -> None:
    """
    Process images in batches and generate labels.

    Args:
        model (YOLO): Loaded YOLO model.
        image_files (List[Path]): List of image file paths to process.
        output_folder (Path): Base output directory.
        batch_size (int): Number of images to process per batch.
        confidence (float): Confidence threshold for detections.
        line_width (int): Line width for bounding box visualization.
        kpt_radius (int): Keypoint radius for visualization.
    """
    # Create output subdirectories
    images_folder = output_folder / "images"
    labels_folder = output_folder / "labels"
    
    images_folder.mkdir(parents=True, exist_ok=True)
    labels_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(image_files)} images in batches of {batch_size}...")
    print(f"Output images: {images_folder}")
    print(f"Output labels: {labels_folder}\n")
    
    # Process images in batches
    results = []
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch = image_files[i:i + batch_size]
        batch_results = model.predict(batch, verbose=False)
        results.extend(batch_results)
    
    # Generate labels and save annotated images
    print("\nGenerating labels and saving annotated images...")
    generate_labels(
        results,
        target_folder=images_folder,
        labels_folder=labels_folder,
        confidence=confidence,
        line_width=line_width,
        kpt_radius=kpt_radius
    )
    
    print(f"\n✓ Processing complete!")
    print(f"  Annotated images saved to: {images_folder}")
    print(f"  Label files saved to: {labels_folder}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate YOLO keypoint annotations for images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python keypoint_generation.py --input_folder ./images --output_folder ./output --model_path ./best.pt
  python keypoint_generation.py -i ./images -o ./output -m ./best.pt --confidence 0.3 --batch_size 16
  python keypoint_generation.py -i ./images -o ./output -m ./best.pt --filter_lenses
        """
    )
    
    parser.add_argument(
        "--input_folder", "-i",
        type=str,
        required=True,
        help="Path to folder containing input images"
    )
    
    parser.add_argument(
        "--output_folder", "-o",
        type=str,
        required=True,
        help="Path to output folder (will create -images and -labels subdirectories)"
    )
    
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        required=True,
        help="Path to YOLO model weights file (.pt)"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.2,
        help="Confidence threshold for detections (default: 0.2)"
    )
    
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="Number of images to process per batch (default: 32)"
    )
    
    parser.add_argument(
        "--line_width",
        type=int,
        default=8,
        help="Line width for bounding box visualization (default: 8)"
    )
    
    parser.add_argument(
        "--kpt_radius",
        type=int,
        default=10,
        help="Keypoint radius for visualization (default: 10)"
    )
    
    parser.add_argument(
        "--filter_lenses",
        action="store_true",
        help="Only process images with 'l0' or 'l1' in filename (lens filtering)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file does not exist: {model_path}")
        sys.exit(1)
    
    output_folder = Path(args.output_folder)
    
    # Load model
    print(f"Loading YOLO model from: {model_path}")
    try:
        model = YOLO(str(model_path))
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Get image files
    image_files = get_image_files(input_folder, filter_lenses=args.filter_lenses)
    
    if not image_files:
        print(f"Error: No images found in {input_folder}")
        if args.filter_lenses:
            print("Note: Lens filtering is enabled (only l0 and l1 images)")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    if args.filter_lenses:
        print("(Lens filtering enabled: only l0 and l1 images)")
    
    # Process images
    process_images(
        model=model,
        image_files=image_files,
        output_folder=output_folder,
        batch_size=args.batch_size,
        confidence=args.confidence,
        line_width=args.line_width,
        kpt_radius=args.kpt_radius
    )


if __name__ == "__main__":
    main()
