"""
YOLO Utility Functions

Contains reusable functions for YOLO keypoint detection and label generation.
"""

from pathlib import Path
import cv2


def generate_labels(
    results,
    target_folder: Path,
    labels_folder: Path,
    confidence: float = 0.2,
    line_width: int = 8,
    kpt_radius: int = 10
) -> None:
    """
    Generate YOLO-style label files with bounding boxes and keypoints from detection results.

    Each detection result is saved as:
      - An annotated image in `target_folder`.
      - A `.txt` file in `labels_folder` containing YOLO-format annotations:
        class_id center_x center_y width height kpt1_x kpt1_y ... kptN_x kptN_y

    Bounding boxes and keypoints are normalized to [0, 1] relative to image size.

    Args:
        results (list): List of detection results from YOLO model.
        target_folder (Path): Directory where annotated images will be saved.
        labels_folder (Path): Directory where YOLO-format label files will be saved.
        confidence (float, optional): Minimum confidence threshold. Defaults to 0.2.
        line_width (int, optional): Line thickness for bounding boxes. Defaults to 8.
        kpt_radius (int, optional): Radius for keypoint drawing. Defaults to 10.

    Returns:
        None
    """
    target_folder.mkdir(parents=True, exist_ok=True)
    labels_folder.mkdir(parents=True, exist_ok=True)

    for result in results:
        image_file = target_folder / f"{Path(result.path).stem}.jpg"
        labels_file = labels_folder / f"{Path(result.path).stem}.txt"

        # Save annotated image
        result.save(str(image_file), line_width=line_width, kpt_radius=kpt_radius)

        # Load image size
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"[WARN] Could not read {image_file}, skipping.")
            continue
        height, width, _ = img.shape

        with open(labels_file, "w") as f:
            for box, keypoint, conf in zip(result.boxes.xyxy, result.keypoints.xy, result.boxes.conf):
                if conf < confidence:
                    continue

                # Bounding box in YOLO format (normalized)
                box_width = (box[2] - box[0]) / width
                box_height = (box[3] - box[1]) / height
                box_center_x = (box[0] + (box[2] - box[0]) / 2) / width
                box_center_y = (box[1] + (box[3] - box[1]) / 2) / height

                # Format bounding box
                box_str = f"0 {box_center_x:.6f} {box_center_y:.6f} {box_width:.6f} {box_height:.6f}"

                # Format keypoints (normalized)
                keypoints_str = " ".join([f"{x / width:.6f} {y / height:.6f}" for x, y in keypoint])

                f.write(f"{box_str} {keypoints_str}\n")
