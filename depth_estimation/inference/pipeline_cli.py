"""Command-line pipeline runner for fish stereo processing.

This script replicates the workflow that was previously executed inside
`Fish_Area_Pipeline.ipynb`.  It can be executed on a cloud server or any
Python environment where the dependencies are installed.

High level steps:
    1. (Optional) split composite images into per-lens crops.
    2. Run YOLO pose model to obtain keypoints/bounding boxes.
    3. Generate SAM2 segmentations for each detection.
    4. Match stereo masks and compute distance / area statistics.

Example usage:
    python -m main.pipeline_cli \
        --source-images /data/raw_images \
        --output-root /data/workdir \
        --yolo-weights /models/yolo_best.pt \
        --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
        --sam-checkpoint /models/sam2_checkpoint.pt \
        --calibration-dir-12 /data/calibration_results_12 \
        --calibration-dir-03 /data/calibration_results_03 \
        --split-images \
        --rotate right \
        --disparity mean \
        --max-images 40

The script expects that the `fishai` package is available on PYTHONPATH and
that all helper functions live in `fishai.fishai_utils`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from fishai import fishai_utils as fu
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

LOGGER = logging.getLogger("fishai.pipeline")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fish stereo pipeline")
    parser.add_argument("--source-images", required=True, type=Path,
                        help="Folder containing the raw composite (or already split) images")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Root directory where intermediate and final outputs will be written")
    parser.add_argument("--yolo-weights", required=True, type=Path,
                        help="Path to YOLO pose weights (.pt)")
    parser.add_argument("--sam-config", required=True, type=Path,
                        help="Path to SAM2 config YAML")
    parser.add_argument("--sam-checkpoint", required=True, type=Path,
                        help="Path to SAM2 checkpoint (.pt)")
    parser.add_argument("--calibration-dir-12", required=True, type=Path,
                        help="Directory containing calibration artifacts for lenses 1↔2")
    parser.add_argument("--calibration-dir-03", required=True, type=Path,
                        help="Directory containing calibration artifacts for lenses 0↔3")
    parser.add_argument("--split-images", action="store_true",
                        help="If set, split composite images into lens quadrants before processing")
    parser.add_argument("--split-rotate", choices=["none", "left", "right"], default="none",
                        help="Rotation to apply before splitting")
    parser.add_argument("--split-output", type=Path,
                        help="Optional override for split image destination (defaults to output_root/split)")
    parser.add_argument("--keypoint-lenses", default="l0,l1",
                        help="Comma-separated lens suffixes to include when running YOLO (e.g. l0,l1)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="YOLO inference batch size")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="YOLO input image size")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Confidence threshold for saving detections")
    parser.add_argument("--max-images", type=int,
                        help="Optional limit on number of images to segment")
    parser.add_argument("--ratio-min", type=float,
                        help="Minimum head/tail ratio filter")
    parser.add_argument("--ratio-max", type=float,
                        help="Maximum head/tail ratio filter")
    parser.add_argument("--rectify", action="store_true", default=False,
                        help="Apply stereo rectification when computing disparity/area")
    parser.add_argument("--disparity", choices=["centroid", "shift", "mean"], default="mean",
                        help="Disparity strategy for stereo matching")
    parser.add_argument("--dx-agree-px", type=float, default=50.0,
                        help="Tolerance when averaging centroid/shift disparities")
    parser.add_argument("--mismatch-threshold", type=float, default=100.0,
                        help="Reject matches whose disparities disagree beyond this (px)")
    parser.add_argument("--skip-on-mismatch", action="store_true",
                        help="Drop matches instead of keeping them when mismatch detected")
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        help="Force inference device (defaults to auto)")
    parser.add_argument("--output-csv", type=Path,
                        help="Optional CSV path for the stereo measurement DataFrame")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ...)")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    output_root = fu.ensure_dir(args.output_root)

    # ------------------------------------------------------------------
    # Step 0: Calibrations / rectification maps
    # ------------------------------------------------------------------
    bundle_12 = fu.load_calibration_bundle(args.calibration_dir_12)
    bundle_03 = fu.load_calibration_bundle(args.calibration_dir_03)
    fu.configure_rectification_maps(bundle_12, bundle_03)

    calib_results_12 = bundle_12["results"]
    calib_results_03 = bundle_03["results"]

    focal_length_12 = float(calib_results_12["left_P"][0][0])
    baseline_12 = -float(calib_results_12["right_P"][0][3]) / focal_length_12

    focal_length_03 = float(calib_results_03["left_P"][0][0])
    baseline_03 = -float(calib_results_03["right_P"][0][3]) / focal_length_03

    LOGGER.info("Calibration: lenses12 f=%.2fpx baseline=%.4fm | lenses03 f=%.2fpx baseline=%.4fm",
                focal_length_12, baseline_12, focal_length_03, baseline_03)

    # ------------------------------------------------------------------
    # Step 1: Prepare images (split if requested)
    # ------------------------------------------------------------------
    processed_images_dir: Path
    if args.split_images:
        split_dir = fu.ensure_dir(args.split_output or (output_root / "split"))
        rotate = None if args.split_rotate == "none" else args.split_rotate
        LOGGER.info("Splitting %s -> %s (rotate=%s)", args.source_images, split_dir, rotate)
        fu.split_and_rotate_images(str(args.source_images), str(split_dir), rotate=rotate)
        processed_images_dir = split_dir
    else:
        processed_images_dir = args.source_images

    LOGGER.info("Using %s as per-lens image directory", processed_images_dir)

    # ------------------------------------------------------------------
    # Step 2: YOLO inference and label generation
    # ------------------------------------------------------------------
    yolo_model = YOLO(str(args.yolo_weights))
    device = fu.resolve_device(args.device)
    LOGGER.info("Running YOLO on device=%s", device)

    allowed_suffixes = [token.strip() for token in args.keypoint_lenses.split(",") if token.strip()]
    image_paths = fu.filtered_image_paths(processed_images_dir, allowed_suffixes)
    if not image_paths:
        raise RuntimeError(f"No images found in {processed_images_dir} matching {allowed_suffixes}")

    LOGGER.info("Collected %d images for keypoint inference", len(image_paths))

    yolo_results = fu.run_yolo_inference(
        yolo_model,
        image_paths,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        conf=args.confidence,
        device=device,
    )

    keypoints_img_dir = fu.ensure_dir(output_root / "keypoints_images")
    keypoints_lbl_dir = fu.ensure_dir(output_root / "keypoints")
    LOGGER.info("Writing keypoint annotations to %s", keypoints_lbl_dir)

    fu.generate_labels(
        yolo_results,
        target_folder=str(keypoints_img_dir),
        labels_folder=str(keypoints_lbl_dir),
        confidence=args.confidence,
    )

    # ------------------------------------------------------------------
    # Step 3: SAM2 segmentation
    # ------------------------------------------------------------------
    predictor_device = device
    LOGGER.info("Loading SAM2 (device=%s)", predictor_device)
    sam2 = build_sam2(str(args.sam_config), str(args.sam_checkpoint), device=predictor_device)
    predictor = SAM2ImagePredictor(sam2)

    # Filter images to segment
    images_to_segment = fu.gather_image_list(keypoints_lbl_dir, processed_images_dir, args.max_images)
    LOGGER.info("Segmenting %d images", len(images_to_segment))

    ratio_filter = None
    if args.ratio_min is not None and args.ratio_max is not None:
        ratio_filter = [args.ratio_min, args.ratio_max]

    segmentation_root = fu.ensure_dir(output_root / "segmentation")
    fu.generate_segmentation(
        predictor,
        images_to_segment,
        labels_folder=str(keypoints_lbl_dir),
        save_folder=str(segmentation_root),
        calibration_results_03=calib_results_03,
        calibration_results_12=calib_results_12,
        ratio_filter=ratio_filter,
        save_origin=True,
    )

    # ------------------------------------------------------------------
    # Step 4: Stereo matching & metrics
    # ------------------------------------------------------------------
    LOGGER.info("Running stereo measurement")
    df, distances, areas, lens_counter, dx_pairs, area_pairs = fu.stereo_measure_from_boxes(
        predictor=predictor,
        boxes_folder=str(segmentation_root / "boxes"),
        images_folder=str(processed_images_dir),
        logits_folder=str(segmentation_root / "logits"),
        masks_folder=str(segmentation_root / "masks"),
        focal_length_12=focal_length_12,
        focal_length_03=focal_length_03,
        baseline_12=baseline_12,
        baseline_03=baseline_03,
        rectify=args.rectify,
        disparity=args.disparity,
        dx_agree_px=args.dx_agree_px,
        mismatch_px_threshold=args.mismatch_threshold,
        skip_on_mismatch=args.skip_on_mismatch,
        verbose=True,
    )

    LOGGER.info("Stereo matches found: %d", len(df))

    clean_dist = np.array([d for d in distances if d is not None], dtype=float)
    clean_area = np.array([a for a in areas if a is not None], dtype=float)

    if clean_dist.size:
        LOGGER.info("Distance stats -> mean: %.3f m | std: %.3f m", np.nanmean(clean_dist), np.nanstd(clean_dist))
    else:
        LOGGER.info("Distance stats -> no valid distances")

    if clean_area.size:
        LOGGER.info("Area stats    -> mean: %.3f cm^2 | std: %.3f cm^2", np.nanmean(clean_area), np.nanstd(clean_area))
    else:
        LOGGER.info("Area stats    -> no valid areas")

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        LOGGER.info("Wrote stereo summary CSV to %s", args.output_csv)


if __name__ == "__main__":
    main()
