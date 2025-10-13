
#from fishai.utilities import *
#from fishai.depth_utilities import *

import os
import random
from PIL import Image
import numpy as np
import cv2
import shutil
from tqdm import tqdm
from pathlib import Path
import re
import json
import pandas as pd
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import supervision as sv
from supervision.draw.color import Color

box_annotator = sv.BoxAnnotator(
    color=Color.GREEN,
    thickness=8,
)
mask_annotator = sv.MaskAnnotator(
    color=Color.RED,
    opacity=0.5,
    color_lookup=sv.ColorLookup.INDEX
)

# google colab utilities
from google.colab.patches import cv2_imshow
from google.colab import sheets
# Keypoints detection
from random import sample
from ultralytics import YOLO

# Segmentation
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def split_and_rotate_images(
    input_folder: str,
    output_folder: str,
    rotate: str = None  # None, "left", or "right"
):
    """
    Splits each image in input_folder into four quadrants and saves them to output_folder.
    Optionally rotates each image 90° left or right before splitting.

    Args:
        input_folder (str): Path to folder with input images.
        output_folder (str): Path to folder to save split images.
        rotate (str): "left" for 90° CCW, "right" for 90° CW, or None for no rotation.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = list(Path(input_folder).glob("*.*"))

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Optional rotation
        if rotate == "left":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == "right":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        height, width = img.shape[:2]
        h_half, w_half = height // 2, width // 2

        # Split into quadrants
        l0 = img[:h_half, :w_half]
        l1 = img[h_half:, :w_half]
        l2 = img[h_half:, w_half:]
        l3 = img[:h_half, w_half:]

        # Save quadrants
        stem = Path(img_path).stem
        cv2.imwrite(f"{output_folder}/{stem}_l0.jpg", l0)
        cv2.imwrite(f"{output_folder}/{stem}_l1.jpg", l1)
        cv2.imwrite(f"{output_folder}/{stem}_l2.jpg", l2)
        cv2.imwrite(f"{output_folder}/{stem}_l3.jpg", l3)


def remove_fins_morphological(mask, kernel_size=5, iterations=3):

    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask = (mask * 255).astype(np.uint8)

    # Create elliptical structuring element (better for biological shapes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Erode to disconnect fins from body
    eroded = cv2.erode(mask, kernel, iterations=iterations)

    # Dilate to restore body size while keeping fins removed
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)

    # Find contours to isolate main body component
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only largest contour (main fish body)
    largest_contour = max(contours, key=cv2.contourArea)
    cleaned_mask = np.zeros_like(mask)
    #cv2.drawContours(cleaned_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return cleaned_mask
def generate_segmentation(predictor, img_list, labels_folder, save_folder,
                          calibration_results_03, calibration_results_12,
                          ratio_filter=None, save_origin=True):
    """
    For each image/box pair generate the mask and save it.
    Also computes fish ratios (head-tail / top-bottom) from keypoints.

    ratio_filter: list [min, max] -> only boxes with ratio in this range are kept
    Instead of saving box images, saves filtered YOLO-style labels in a single .txt file per image.
    """
    fish_counter = 0
    ratios = []

    for img_path in tqdm(img_list):
        label_file_path = os.path.join(labels_folder, img_path.stem + ".txt")
        if not os.path.exists(label_file_path):
            print(f"Label file not found : {label_file_path}")
            continue

        try:
            image = Image.open(img_path)
            image = np.array(image.convert("RGB"))
            image_height, image_width = image.shape[:2]
            boxes = []
            filtered_labels = []  # store YOLO-format strings

            with open(label_file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    points = [float(x) for x in line.split()]

                    if len(points) == 13:  # valid with keypoints
                        fish_counter += 1
                        keypoints = points[5:]  # last 8 floats = keypoints

                        # select calibration
                        if img_path.stem[-1] == '0':
                            cr = calibration_results_03
                        elif   img_path.stem[-1] == '3':  
                            cr = calibration_results_03
                        elif   img_path.stem[-1] == '1':
                            cr = calibration_results_12
                        elif img_path.stem[-1] == '2':  
                            cr = calibration_results_12 

                        ratio = compute_ratio(keypoints, cr)
                        ratios.append(ratio)

                        # apply ratio filter before keeping the box
                        if ratio_filter is not None:
                            if not (ratio_filter[0] < ratio < ratio_filter[1]):
                                continue  # skip this box

                        # build the box if ratio passes
                        x_center = points[1] * image_width
                        y_center = points[2] * image_height
                        width = points[3] * image_width
                        height = points[4] * image_height
                        x_min = x_center - width / 2
                        x_max = x_center + width / 2
                        y_min = y_center - height / 2
                        y_max = y_center + height / 2
                        boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])

                        # save in YOLO format (class=0, normalized)
                        filtered_labels.append(
                            f"0 {points[1]} {points[2]} {points[3]} {points[4]}"
                        )
                    else:
                        print(f"Not all keypoints detected in {label_file_path}")

            if not boxes:
                print(f"No valid boxes found in {label_file_path}")
                continue

            # ------------------
            # Save filtered labels instead of box images
            # ------------------
            output_labels_dir = os.path.join(save_folder, "boxes")
            os.makedirs(output_labels_dir, exist_ok=True)
            label_out_path = os.path.join(output_labels_dir, img_path.stem + ".txt")

            with open(label_out_path, "w") as out_f:
                for lbl in filtered_labels:
                    out_f.write(lbl + "\n")

            # ------------------
            # SAM segmentation for filtered boxes
            # ------------------
            predictor.set_image(image)
            for i, box in enumerate(boxes):
                masks, scores, logits = predictor.predict(
                    box=np.array(box)[None],
                    multimask_output=True,
                )

                if masks is None or masks.shape[0] == 0:
                    print(f"No masks predicted for box {i} in {img_path.stem}")
                    continue
                best_mask , best_score , best_logit = masks[np.argmax(scores)], scores[np.argmax(scores)], logits[np.argmax(scores)]

                # refine mask for better segmentation
                masks, scores, logits = predictor.predict(
                    box=np.array(box)[None],
                    multimask_output=True,
                    mask_input = np.array([best_logit])
                    )

                best_mask , best_score , best_logit = masks[np.argmax(scores)], scores[np.argmax(scores)], logits[np.argmax(scores)]
                if best_score < .70: # confidence threshold for accepting the mask
                    print(f"Confidence score for fish {i} in {img_path.stem} too low: {best_score:.3f}")
                    continue
                detections = sv.Detections(
                    xyxy=np.array([box]),
                    mask=best_mask.astype(bool)[np.newaxis, :, :],
                    class_id=np.array([1]),
                    confidence=np.array([best_score])
                )

                annotated_image = image.copy()
                annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

                output_mask_dir = os.path.join(save_folder, "annotations")
                os.makedirs(output_mask_dir, exist_ok=True)
                cv2.imwrite(f"{output_mask_dir}/{img_path.stem}_{i}.png",
                            cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                # save mask/logits
                output_mask_dir = os.path.join(save_folder, "masks")
                os.makedirs(output_mask_dir, exist_ok=True)
                np.save(f"{output_mask_dir}/{img_path.stem}_{i}.npy", best_mask)

                output_logits_dir = os.path.join(save_folder, "logits")
                os.makedirs(output_logits_dir, exist_ok=True)
                np.save(f"{output_logits_dir}/{img_path.stem}_{i}.npy", best_logit)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Processed {fish_counter} fish")
    return ratios

def load_bounding_boxes(txt_file , image_height , image_width):
    boxes = []
    with open(txt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            points = [float(x) for x in line.split()]
            if len(points) >= 5:
                x_center = points[1]*image_width
                y_center = points[2]*image_height
                width = points[3]*image_width
                height = points[4]*image_height
                x_min = x_center - width/2
                x_max = x_center + width/2
                y_min = y_center - height/2
                y_max = y_center + height/2
                boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])

    return boxes
def sliding_boxes(candidate_box , up = True):
    """ returns 15 possible candidate boxes for detecting the fish mask"""
    stride = np.linspace(20,abs(candidate_box[3]-candidate_box[1]) , 15 , dtype = int )
    if up:
        candidate_boxes = [[candidate_box[0] , candidate_box[1]-x , candidate_box[2] , candidate_box[3]-x] for x in stride]
    else:
        candidate_boxes = [[candidate_box[0] , candidate_box[1]+x , candidate_box[2] , candidate_box[3]+x] for x in stride]
    return candidate_boxes
def mask_area(mask_path):
    mask = np.load(mask_path)
    return np.sum(mask)
def shift_matrix(matrix, delta, fill_value=-32):
    """
    Shifts a 256x256 (or 1x256x256) numpy matrix up or down by a specified delta.
    Returns shape (1,256,256).
    """
    m = np.asarray(matrix)

    # Normalize shape
    if m.shape == (256, 256):
        m = m[np.newaxis, :, :]
    elif m.shape == (1, 256, 256):
        pass  # already fine
    else:
        raise ValueError(f"Unexpected shape {m.shape}, expected (256,256) or (1,256,256)")

    shifted = np.full_like(m, fill_value)

    if delta > 0:  # shift down
        shifted[:, delta:, :] = m[:, :-delta, :]
    elif delta < 0:  # shift up
        shifted[:, :delta, :] = m[:, -delta:, :]
    else:  # no shift
        shifted = m.copy()

    return shifted
def slide_masks(mask_logits , direction):
    """ returns 15 possible candidate masks for detecting the fish mask"""
    stride = np.linspace(10, 50, 15, dtype = int )
    candidate_masks = []
    if direction == False:
        coeff = 1
    else :
        coeff = -1
    for x in stride :
        candidate_masks.append(shift_matrix(mask_logits , coeff*int(x) , fill_value=-32))
    return candidate_masks
def get_mask_centroid_difference(mask_upper, mask_lower):
    """
    Calculates the difference between the centroids of two binary masks.

    This function finds the center of mass for each mask and returns the
    pixel shift between them in the x and y directions.

    Args:
        mask_upper (np.ndarray): The binary mask from the upper camera view.
        mask_lower (np.ndarray): The binary mask from the lower camera view.

    Returns:
        tuple: A tuple `(dx, dy)` representing the horizontal and vertical
               pixel shift between the centroids. Returns (None, None) if
               either mask is empty and a centroid cannot be found.
    """
    # Calculate moments for the upper mask
    M_upper = cv2.moments(mask_upper)

    # Calculate moments for the lower mask
    M_lower = cv2.moments(mask_lower)

    # Ensure both masks have non-zero area to avoid division by zero
    if M_upper["m00"] == 0 or M_lower["m00"] == 0:
        print("Warning: At least one mask is empty. Cannot calculate centroids.")
        return (None, None)

    # Calculate the centroid for the upper mask
    cX_upper = int(M_upper["m10"] / M_upper["m00"])
    cY_upper = int(M_upper["m01"] / M_upper["m00"])

    # Calculate the centroid for the lower mask
    cX_lower = int(M_lower["m10"] / M_lower["m00"])
    cY_lower = int(M_lower["m01"] / M_lower["m00"])

    # Calculate the difference in the centroids' coordinates
    dx = cX_upper - cX_lower
    dy = cY_upper - cY_lower

    return (dx, dy)
def produce_coupled_images(file_names):
    """
    Produces coupled images by swapping the labels in the file names.
    """
    substitution_dict = {
        "l0": "l3",
        "l1": "l2",
        "l2": "l1",
        "l3": "l0"
    }
    coupled_images = []
    for f in file_names:
        for k in substitution_dict.keys():
            if k in f:
                new_name = f.replace(k, substitution_dict[k])
                # extract the number from the label (like 0,1,2,3)
                num1 = int(re.search(r"l(\d)", f).group(1))
                num2 = int(re.search(r"l(\d)", new_name).group(1))
                coupled_images.append((f, new_name, num1 < num2, abs(num1 - num2)))
                break
    return coupled_images

def plot_masks(source_image, target_image, source_mask, source_box,
               best_mask, best_box, best_score, distance, area,
               index, show=True, save_dir="./images_distance", filename=None):
    """
    Save a side-by-side visualization of source/target masks.
    If show=False, the figure won't be displayed, only saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build detections for target (best) mask
    scores_array = np.array([best_score]) if best_score is not None else np.array([0.0])
    binary_mask = best_mask.astype(bool)[np.newaxis, :, :]

    detections = sv.Detections(
        xyxy=np.array([best_box]),
        mask=binary_mask,
        class_id=np.array([1]),
        confidence=scores_array
    )

    annotated_image = target_image.copy()
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

    # Build detections for source mask/box
    detections_source = sv.Detections(
        xyxy=np.array([[source_box[0], source_box[1], source_box[2], source_box[3]]]),
        mask=source_mask[np.newaxis, :, :].astype(bool),
        class_id=np.array([1]),
        confidence=np.array([1.0])
    )
    annotated_image_source = source_image.copy()
    annotated_image_source = mask_annotator.annotate(scene=annotated_image_source, detections=detections_source)

    # Make smaller and add text on the target image
    annotated_image_source = cv2.resize(annotated_image_source, dsize=(0, 0), fx=0.3, fy=0.3)
    annotated_image = cv2.resize(annotated_image, dsize=(0, 0), fx=0.3, fy=0.3)

    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_source_rgb = cv2.cvtColor(annotated_image_source, cv2.COLOR_BGR2RGB)

    cv2.putText(annotated_image_rgb, f"Distance: {distance:.2f} m", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_image_rgb, f"Area: {area:.2f} cm2", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Compose figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].set_title("Detected Fish")
    axes[0].imshow(annotated_source_rgb)
    axes[0].axis("off")

    axes[1].set_title(f"Confidence Score: {best_score:.3f}" if best_score is not None else "Confidence Score: N/A")
    axes[1].imshow(annotated_image_rgb)
    axes[1].axis("off")

    plt.tight_layout()

    # Save
    name = filename if filename else f"result_{index:04d}"
    save_path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show or close
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path
def detect_similar_mask(
    box,
    predictor,
    direction,
    source_logits,
    stride_logits,
    stride_boxes,
    source_mask,
    sim_tol=0.1,
    sim_weight=0.5
):
    """
    Find the best candidate mask in the target image by shifting source logits
    and comparing similarity with the source mask.

    Args:
        box: bounding box [x0, y0, x1, y1]
        predictor: segmentation predictor (with .predict)
        direction: stereo direction (False=down, True=up)
        source_logits: logits from source image
        stride_logits: array of logit shifts (px)
        stride_boxes: array of bbox shifts (px)
        source_mask: binary mask from source image
        sim_tol: tolerance for binary similarity check
        sim_weight: weight [0-1] blending predictor score with mask similarity

    Returns:
        best_mask, best_box, best_score
    """

    coeff = 1 if direction is False else -1

    # Ensure stride_logits is 1D
    stride_logits = np.array(stride_logits).ravel()

    candidate_logits = [
        shift_matrix(source_logits, coeff * int(sm), fill_value=-32)
        for sm in stride_logits
    ]
    candidate_boxes = [
        [box[0], box[1] + coeff * int(sb), box[2], box[3] + coeff * int(sb)]
        for sb in stride_boxes
    ]


    best_score = np.inf
    best_box, best_mask, best_logit = None, None, None

    for cb, cl in zip(candidate_boxes, candidate_logits):
        masks, scores, logits = predictor.predict(
            box=np.array(cb)[None],
            multimask_output=True,
            mask_input=cl,
        )
        sorted_ind = np.argsort(scores)[::-1]
        # sort from max to min score 
        masks, scores, logits = (
            masks[sorted_ind], scores[sorted_ind], logits[sorted_ind]
        )

        for cm, cs, cl_2 in zip(masks, scores, logits):
            same_shape, sim_difference = are_masks_same_translation_invariant(
                cm, source_mask, method="match", tol=sim_tol
            )

            if not same_shape:
                continue

            if sim_difference <= best_score:
                best_score = sim_difference
                best_box, best_mask, best_logit = cb, cm, cl_2

    # no valid candidate found → bail out
    if best_logit is None:
        return None, None, None

    # refine the best candidate
    best_logit = np.array(best_logit)[np.newaxis, :, :]
    best_mask, scores, best_logit = predictor.predict(
        box=np.array(best_box)[None],
        multimask_output=False,
        mask_input=best_logit,
    )

    return best_mask[0], best_box, best_score
def _to_binary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim > 2:
        m = m.squeeze()
    return (m > 0).astype(np.uint8)
def _largest_contour(mask_bin: np.ndarray):
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)
def _centroid(mask_bin: np.ndarray):
    M = cv2.moments(mask_bin)
    if M["m00"] == 0:
        return None
    return (M["m10"]/M["m00"], M["m01"]/M["m00"])
def _translate(mask_bin: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = mask_bin.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(mask_bin, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
def mask_shape_distance_translation_invariant(mask1: np.ndarray,
                                              mask2: np.ndarray,
                                              method: str = "match") -> float:
    """
    Distance is 0.0 when shapes are the same regardless of position.

    method:
      - "match": cv2.matchShapes on largest contours (translation/scale/rotation invariant)
      - "aligned_xor": translate by centroid alignment, then normalized XOR over union
    """
    m1 = _to_binary(mask1)
    m2 = _to_binary(mask2)

    if m1.shape != m2.shape:
        raise ValueError(f"Mask shapes differ: {m1.shape} vs {m2.shape}")

    if method == "match":
        c1 = _largest_contour(m1)
        c2 = _largest_contour(m2)
        if c1 is None and c2 is None:
            return 0.0
        if c1 is None or c2 is None:
            return 1.0
        return float(cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I3, 0.0))

    if method == "aligned_xor":
        c1 = _centroid(m1)
        c2 = _centroid(m2)
        if c1 is None and c2 is None:
            return 0.0
        if c1 is None or c2 is None:
            return 1.0
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        m1_aligned = _translate(m1, dx, dy)
        union = np.logical_or(m1_aligned, m2).sum()
        if union == 0:
            return 0.0
        diff = np.logical_xor(m1_aligned, m2).sum()
        return float(diff) / float(union)

    raise ValueError(f"Unknown method: {method}")
def are_masks_same_translation_invariant(mask1: np.ndarray,
                                         mask2: np.ndarray,
                                         method: str = "match",
                                         tol: float = 1e-4) -> tuple[bool, float]:
    try :
        d = mask_shape_distance_translation_invariant(mask1, mask2, method=method)
    except :
        return False , 100                                         
    return (d <= tol), d
def _mask_similarity_score(candidate_mask: np.ndarray,
                           reference_mask: np.ndarray,
                           sam_score: float) -> tuple[float, dict]:
    """
    Returns (final_score, debug): higher is better.
    Combines translation-invariant shape similarity and SAM confidence.
    """
    # Ensure binary and same shape for comparison
    ref = (reference_mask > 0).astype(np.uint8)
    cand = (candidate_mask > 0).astype(np.uint8)
    if cand.shape != ref.shape:
        cand = cv2.resize(cand, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Distances in [0, +inf), lower is better
    d_match = mask_shape_distance_translation_invariant(cand, ref, method="match")
    d_xor   = mask_shape_distance_translation_invariant(cand, ref, method="aligned_xor")

    # Convert to similarities in [0,1]
    s_match = 1.0 - float(min(d_match, 1.0))
    s_xor   = 1.0 - float(min(d_xor, 1.0))

    # Weighted combination of shape similarity and SAM score
    shape_sim = 0.6 * s_match + 0.4 * s_xor
    final = 0.7 * shape_sim + 0.3 * float(sam_score)

    dbg = {"d_match": float(d_match), "d_xor": float(d_xor),
           "s_match": float(s_match), "s_xor": float(s_xor),
           "shape_sim": float(shape_sim), "sam_score": float(sam_score),
           "final": float(final)}
    return float(final), dbg
def _binarize(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    return m
def _crop_union_roi(a: np.ndarray, b: np.ndarray, pad: int = 20):
    def bbox(m):
        pts = cv2.findNonZero(m)
        if pts is None:
            return None
        x, y, w, h = cv2.boundingRect(pts)
        return x, y, x+w, y+h

    ha, wa = a.shape[:2]
    bb_a = bbox(a)
    bb_b = bbox(b)
    if bb_a is None or bb_b is None:
        return (a, b, (0, 0))  # nothing to crop; will handle as-is

    x1a, y1a, x2a, y2a = bb_a
    x1b, y1b, x2b, y2b = bb_b
    x1 = max(0, min(x1a, x1b) - pad)
    y1 = max(0, min(y1a, y1b) - pad)
    x2 = min(wa, max(x2a, x2b) + pad)
    y2 = min(ha, max(y2a, y2b) + pad)
    return a[y1:y2, x1:x2], b[y1:y2, x1:x2], (x1, y1)
def _signed_distance(mask_bin: np.ndarray) -> np.ndarray:
    # cv2.distanceTransform expects 8-bit, non-zero as foreground
    fg = mask_bin.astype(np.uint8)
    bg = (1 - mask_bin).astype(np.uint8)
    dt_f = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    dt_b = cv2.distanceTransform(bg, cv2.DIST_L2, 3)
    sdt = dt_f - dt_b
    # normalize to [-1, 1] for stability
    maxv = np.max(np.abs(sdt)) + 1e-6
    return (sdt / maxv).astype(np.float32)
def _hann2d(h, w):
    wx = np.hanning(w)
    wy = np.hanning(h)
    return (wy[:, None] * wx[None, :]).astype(np.float32)
def _phase_corr(a: np.ndarray, b: np.ndarray):
    # returns (dx, dy, response)
    h, w = a.shape
    win = _hann2d(h, w)
    shift, response = cv2.phaseCorrelate(a * win, b * win)
    dx, dy = shift  # OpenCV returns (dx, dy)
    return float(dx), float(dy), float(response)
def _shift_mask_binary(mask: np.ndarray, dy: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    if dy > 0:
        out[dy:, :] = mask[:h-dy, :]
    elif dy < 0:
        out[:h+dy, :] = mask[-dy:, :]
    else:
        out[:] = mask
    return out
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)
def estimate_mask_shift(mask_a: np.ndarray,
                        mask_b: np.ndarray,
                        refine_window: int = 6,
                        ecc_iters: int = 1000,
                        ecc_eps: float = 1e-6):
    """
    Returns: dy, dx, score, method
    dy > 0 means mask_a is below mask_b (shift a down aligns to b).
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError("Masks must have the same shape")

    # Binarize and crop to ROI
    a_bin = _binarize(mask_a)
    b_bin = _binarize(mask_b)
    a_roi, b_roi, (x0, y0) = _crop_union_roi(a_bin, b_bin, pad=20)

    # Handle empty masks
    if a_roi.sum() == 0 or b_roi.sum() == 0:
        return None, None, 0.0, "invalid"

    # Build SDTs for robustness to shape differences
    a_sdt = _signed_distance(a_roi)
    b_sdt = _signed_distance(b_roi)

    # Try ECC (translation model)
    method = "ecc_sdt"
    dy = dx = 0.0
    score = 0.0
    try:
        warp_mode = cv2.MOTION_TRANSLATION
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, ecc_eps)
        cc, warp = cv2.findTransformECC(templateImage=b_sdt, inputImage=a_sdt,
                                        warpMatrix=warp, motionType=warp_mode,
                                        criteria=criteria, inputMask=None, gaussFiltSize=5)
        dx = float(warp[0, 2])
        dy = float(warp[1, 2])
        score = float(cc)
    except cv2.error:
        # Fallback to phase correlation
        method = "phasecorr_sdt"
        dx, dy, resp = _phase_corr(a_sdt, b_sdt)
        score = resp

    # Optional small-window IoU refine on vertical shift only
    # Round to nearest int and sweep ±refine_window
    base_dy = int(round(dy))
    best_iou = -1.0
    best_dy = base_dy
    for ddy in range(base_dy - refine_window, base_dy + refine_window + 1):
        a_shift = _shift_mask_binary(a_roi, ddy)
        iou = _iou(a_shift.astype(bool), b_roi.astype(bool))
        if iou > best_iou:
            best_iou = iou
            best_dy = ddy

    # Replace dy with refined value; keep dx from ECC/phase if needed
    dy_refined = float(best_dy)
    return dy_refined, dx, max(score, best_iou), method
def pixel_area_to_cm2(area_px: float, distance_m: float, focal_length_px: float) -> float:
    """
    Convert object area from pixels to cm^2 using the pinhole camera model.

    Notes:
    - area_px must be the COUNT of foreground pixels (from a 0/1 mask).
    - If a 0/255 mask sum is passed by mistake, we auto-normalize by ~255.
    """
    import numpy as np

    apx = float(area_px)
    if apx <= 0 or distance_m <= 0 or focal_length_px <= 0:
        return 0.0

    # Heuristic: if it looks like a 0/255 sum, normalize
    if apx >= 1e6 and np.isclose(apx / 255.0, round(apx / 255.0), atol=1e-2):
        apx = apx / 255.0

    # A = area_px * (Z / f_px)^2 converted to cm^2
    return 1e4 * (distance_m ** 2 / float(focal_length_px) ** 2) * apx
def rotate90_anticlock(x, y, width, height):
    """Rotate a pixel coordinate 90° anticlockwise"""
    return y, width - 1 - x
def compute_ratio(keypoints, calib_json):
    """
    keypoints: list of 8 floats (normalized) [hx, hy, tx, ty, topx, topy, bottomx, bottomy]
    calib_json: dict loaded from calibration file
    """

    # Image size
    width = calib_json["width"]
    height = calib_json["height"]

    # Camera intrinsics
    cam_matrix = np.array(calib_json["left_camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib_json["left_dist"], dtype=np.float32).reshape(-1, 1)

    # Prepare dict of keypoints
    kd = {
        "head":   (keypoints[0], keypoints[1]),
        "tail":   (keypoints[2], keypoints[3]),
        "top":    (keypoints[4], keypoints[5]),
        "bottom": (keypoints[6], keypoints[7]),
    }

    pts = {}
    for k, (xn, yn) in kd.items():
        # 1. normalized → pixel coords
        x = xn * width
        y = yn * height

        # 2. rotate 90° anticlockwise
        x_rot, y_rot = rotate90_anticlock(x, y, width, height)

        # 3. undistort single point
        pt = np.array([[[x_rot, y_rot]]], dtype=np.float32)  # shape (1,1,2)
        undist = cv2.undistortPoints(pt, cam_matrix, dist_coeffs, P=cam_matrix)
        x_u, y_u = undist[0, 0]
        pts[k] = (x_u, y_u)

    # 4. compute distances
    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    d_ht = dist(pts["head"], pts["tail"])
    d_tb = dist(pts["top"], pts["bottom"])

    return d_ht / d_tb if d_tb != 0 else float("inf")
def calculate_rectified_distance_and_area(
    left_mask,
    right_mask,
    baseline,
    focal_length,
    lenses,
    rect_maps, # rectification dictionary
    rectify: bool = True,  # True: use stereo rectification maps; False: skip remap
    disparity: str = "centroid",           # "centroid" | "shift" | "mean"
    dx_agree_px: float = 50.0          # mismatch threshold when using "mean"
):
    """
    Compute distance (m) and area (cm^2) from stereo masks with configurable options.

    Args:
      left_mask, right_mask: binary masks (any dtype/shape; will be resized/rotated)
      baseline: meters
      focal_length: pixels
      lenses: "12" or "03"
      rectify: if True, apply rectification maps; if False, only resize/rotate
      disparity:
        - "centroid": dx from centroid difference after (optional) rectification
        - "shift":    dx from estimate_mask_shift (ECC/phasecorr) after (optional) rectification
        - "mean":     mean of available dx estimates (flags mismatch if they differ > dx_agree_px)
      dx_agree_px: mismatch threshold for "mean"
      rect_maps: rectification dictionary for each couple of lenses

    Returns:
      distance_m, area_cm2, (dx_centroid, dx_est), (area_px_left, area_px_right)
    """
    H, W = 2312, 1736

    # Ensure 2D binary masks
    left = np.asarray(left_mask)
    right = np.asarray(right_mask)
    if left.ndim > 2:
        left = left.squeeze()
    if right.ndim > 2:
        right = right.squeeze()
    left = (left > 0).astype(np.uint8)
    right = (right > 0).astype(np.uint8)

    # Resize to expected sensor shape
    if left.shape != (H, W):
        left = cv2.resize(left, (W, H), interpolation=cv2.INTER_NEAREST)
    if right.shape != (H, W):
        right = cv2.resize(right, (W, H), interpolation=cv2.INTER_NEAREST)

    # Rotate to match calibration orientation
    left = cv2.rotate(left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    right = cv2.rotate(right, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Pick rectification maps
    if lenses == "12":
        lmxr, lmyr = rect_maps["left_map_x_rectify_12"], rect_maps["left_map_y_rectify_12"]
        rmxr, rmyr = rect_maps["right_map_x_rectify_12"], rect_maps["right_map_y_rectify_12"]
    elif lenses == "03":
        lmxr, lmyr = rect_maps["left_map_x_rectify_03"], rect_maps["left_map_y_rectify_03"]
        rmxr, rmyr = rect_maps["right_map_x_rectify_03"], rect_maps["right_map_y_rectify_03"]
    else:
        raise ValueError("Lenses must be '12' or '03'")

    # Rectify (optional)
    if rectify:
        left_rect = cv2.remap(left, lmxr, lmyr, interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        right_rect = cv2.remap(right, rmxr, rmyr, interpolation=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        left_rect, right_rect = left, right

    # Guard empty masks
    if left_rect.sum() == 0 or right_rect.sum() == 0:
        return None, None, (None, None), (None, None)

    # Disparity estimates
    dx_centroid, _ = get_mask_centroid_difference(left_rect, right_rect)  # returns (dx, dy)
    _, dx_est, _, _ = estimate_mask_shift(left_rect, right_rect)          # returns (dy, dx, score, method)

    # Select disparity
    dx = None
    disp = disparity.lower()
    if disp in ("centroid", "centroid_only"):
        dx = dx_centroid
    elif disp in ("shift", "est", "ecc", "phasecorr"):
        dx = dx_est
    elif disp in ("mean", "avg", "auto"):
        vals = [v for v in (dx_centroid, dx_est) if v is not None and np.isfinite(v)]
        if not vals:
            dx = None
        elif len(vals) == 1:
            dx = float(vals[0])
        else:
            # Optionally flag mismatch; we still return mean
            if abs(float(vals[0]) - float(vals[1])) > float(dx_agree_px):
                pass  # caller can check difference in the returned tuple
            dx = float(np.mean(vals))
    else:
        raise ValueError("disparity must be one of ['centroid','shift','mean']")

    if dx is None or abs(dx) < 1e-6:
        return None, None, (dx_centroid, dx_est), (int(left_rect.sum()), int(right_rect.sum()))

    # Distance (baseline in meters, focal length in pixels)
    distance_m = (baseline * float(focal_length)) / abs(dx)

    # Area: mean of foreground pixels (rectified or not), converted to cm^2
    area_px = float(left_rect.sum() + right_rect.sum()) / 2.0
    area_cm2 = pixel_area_to_cm2(area_px, distance_m, float(focal_length))

    return distance_m, area_cm2, (dx_centroid, dx_est), (int(left_rect.sum()), int(right_rect.sum()))

def stereo_measure_from_boxes(
    predictor,
    boxes_folder: str,
    images_folder: str,
    logits_folder: str,
    masks_folder: str,
    focal_length_12: float,
    focal_length_03: float,
    rect_maps, # rectification dictionary
    baseline_12: float = 0.06,
    baseline_03: float = 0.18,
    stride_logit_min: int = 5,
    stride_logit_max: int = 80,
    stride_logit_n: int = 20,
    logit_grid_div: int = 250,
    save_folder: str = "./twin_fish",
    # Flags passed to calculate_rectified_distance_and_area
    rectify: bool = True,
    disparity: str = "mean",         # "centroid" | "shift" | "mean"
    dx_agree_px: float = 50.0,
    # Post-checks
    mismatch_px_threshold: float | None = None,
    skip_on_mismatch: bool = False,
    verbose: bool = False
):
    """
    Run stereo matching for all boxes and return a DataFrame and raw lists.

    Returns:
      df, distances, areas, lens_counter, dx_pairs, area_pairs
    """
    boxes_folder = Path(boxes_folder)
    images_folder = Path(images_folder)
    logits_folder = Path(logits_folder)
    masks_folder = Path(masks_folder)

    box_files = list(boxes_folder.glob("*.txt"))
    stems = [x.stem for x in box_files]
    coupled_images = produce_coupled_images(stems)
    pair_map = {s: (t, direction, baseline_code) for (s, t, direction, baseline_code) in coupled_images}

    distances, areas, lens_counter = [], [], []
    dx_pairs, area_pairs, rows = [], [], []

    for bf in box_files:
        source_img_name = bf.stem
        if source_img_name not in pair_map:
            if verbose:
                print(f"[WARN] No pair for {source_img_name}")
            continue

        target_img_name, direction, baseline_code = pair_map[source_img_name]
        label_file_path = boxes_folder / f"{source_img_name}.txt"

        if baseline_code == 1:
            lenses = "12"; focal_length = float(focal_length_12); baseline = float(baseline_12)
        elif baseline_code == 3:
            lenses = "03"; focal_length = float(focal_length_03); baseline = float(baseline_03)
        else:
            if verbose:
                print(f"[WARN] Unknown baseline_code {baseline_code} for {source_img_name}")
            continue

        source_img = cv2.imread(str(images_folder / f"{source_img_name}.jpg"))
        target_image = cv2.imread(str(images_folder / f"{target_img_name}.jpg"))
        if target_image is None:
            if verbose:
                print(f"[WARN] Missing target image {target_img_name}.jpg")
            continue

        logit_to_image_coeff = int(target_image.shape[0] // logit_grid_div)
        predictor.set_image(target_image)

        boxes = load_bounding_boxes(
            str(label_file_path),
            image_height=target_image.shape[0],
            image_width=target_image.shape[1]
        )
        stride_logits = np.linspace(stride_logit_min, stride_logit_max, stride_logit_n, dtype=int)
        stride_boxes = stride_logits * logit_to_image_coeff

        for i, box in enumerate(boxes):
            logits_path = logits_folder / f"{source_img_name}_{i}.npy"
            mask_path = masks_folder / f"{source_img_name}_{i}.npy"
            if not logits_path.exists() or not mask_path.exists():
                if verbose:
                    print(f"[WARN] Missing npy for {source_img_name}_{i}")
                continue

            source_logits = np.load(str(logits_path))
            source_mask = np.load(str(mask_path)).astype(np.float32).squeeze()

            best_mask, best_box, best_score = detect_similar_mask(
                box,
                predictor,
                direction,
                source_logits,
                stride_logits,
                stride_boxes,
                source_mask,
                sim_tol=0.1
            )

            if best_mask is None:
                if verbose:
                    print(f"Twin fish not detected in {target_img_name} (box {i})")
                continue

            try:
                distance, area, dx_pair, area_pair = calculate_rectified_distance_and_area(
                    source_mask, best_mask, baseline, focal_length, lenses,
                    rectify=rectify ,rect_maps = rect_maps , disparity=disparity, dx_agree_px=dx_agree_px
                )


                distances.append(distance)
                areas.append(area)
                lens_counter.append(lenses)
                dx_pairs.append(dx_pair)
                area_pairs.append(area_pair)

                rows.append({
                    "source": source_img_name,
                    "target": target_img_name,
                    "box_index": i,
                    "direction_up": bool(direction),
                    "baseline_code": int(baseline_code),
                    "lenses": lenses,
                    "baseline_m": baseline,
                    "focal_px": focal_length,
                    "distance_m": None if distance is None else float(distance),
                    "area_cm2": None if area is None else float(area),
                    "dx_centroid": None if dx_pair[0] is None else float(dx_pair[0]),
                    "dx_est": None if dx_pair[1] is None else float(dx_pair[1]),
                    "area_px_left": None if area_pair[0] is None else int(area_pair[0]),
                    "area_px_right": None if area_pair[1] is None else int(area_pair[1]),
                    "best_score": None if best_score is None else float(best_score),
                    "rectify": bool(rectify),
                    "disparity_mode": disparity,
                })
                # save image of both masks 
                saved_file = plot_masks(
                    source_image=source_img,
                    target_image=target_image,
                    source_mask=source_mask,
                    source_box=box,
                    best_mask=best_mask,
                    best_box=best_box,
                    best_score=best_score,
                    distance=distance if distance is not None else 0.0,
                    area=area if area is not None else 0.0,
                    index=i,
                    show=False,
                    save_dir=save_folder,
                    filename=f"{source_img_name}_{i}"
                )

            except Exception as e:
                if verbose:
                    print(f"Error on {source_img_name}_{i}: {e}")

    df = pd.DataFrame(rows)
    return df, distances, areas, lens_counter, dx_pairs, area_pairs


def remove_outliers(df, cols, method="iqr", factor=1.5, z_thresh=3.0, pct=0.01):
    """
    method:
      - "iqr": Tukey fences (factor=1.5 or 3.0 for aggressive)
      - "zscore": mean/std
      - "mad": robust z using Median Absolute Deviation
      - "trim": percentile trimming (pct each side)
    Returns filtered_df, mask_kept, mask_removed
    """
    mask = np.ones(len(df), dtype=bool)

    for c in cols:
        x = df[c].values

        if method == "iqr":
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            lo = q1 - factor * iqr
            hi = q3 + factor * iqr
            mask &= (x >= lo) & (x <= hi)

        elif method == "zscore":
            m = np.nanmean(x)
            s = np.nanstd(x)
            if s == 0:
                continue
            z = (x - m)/s
            mask &= np.abs(z) <= z_thresh

        elif method == "mad":
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            if mad == 0:
                continue
            robust_z = 0.6745 * (x - med) / mad
            mask &= np.abs(robust_z) <= z_thresh

        elif method == "trim":
            lo = np.percentile(x, 100*pct)
            hi = np.percentile(x, 100*(1-pct))
            mask &= (x >= lo) & (x <= hi)

        else:
            raise ValueError("Unknown method")

    filtered = df[mask].reset_index(drop=True)
    return filtered, mask, ~mask
def rel_diff(tup):
    a, b = map(int, tup)   # convert np.uint64 to plain int
    return abs(a - b) / ((a + b) / 2)
def calculate_rectified_distance(left_mask, right_mask, baseline , focal_length ):

    height , width = 2312 , 1736

    left_mask = cv2.resize(left_mask , dsize = (width , height))
    left_mask = cv2.rotate(left_mask , cv2.ROTATE_90_COUNTERCLOCKWISE)
    temp_undistorted_left_mask = cv2.remap(
        left_mask,
        left_map_x_rectify,
        left_map_y_rectify,
        interpolation=cv2.INTER_NEAREST,  # nearest is best for binary masks
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    right_mask = cv2.resize(right_mask , dsize = (width , height))
    right_mask = cv2.rotate(right_mask , cv2.ROTATE_90_COUNTERCLOCKWISE)
    temp_undistorted_right_mask = cv2.remap(
        right_mask,
        right_map_x_rectify,
        right_map_y_rectify,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    horizontal_shift = np.mean((abs(get_mask_centroid_difference(temp_undistorted_left_mask, temp_undistorted_right_mask )[0]), abs(estimate_mask_shift(temp_undistorted_left_mask , temp_undistorted_right_mask)[1])))


    distance = baseline * focal_length / abs(horizontal_shift)
    return distance
def calculate_rectified_area(left_mask, right_mask, distance ,focal_length  ):

    height , width = 2312 , 1736


    left_mask = cv2.resize(left_mask , dsize = (width , height))
    left_mask = cv2.rotate(left_mask , cv2.ROTATE_90_COUNTERCLOCKWISE)
    temp_undistorted_left_mask = cv2.remap(
        left_mask,
        left_map_x_rectify,
        left_map_y_rectify,
        interpolation=cv2.INTER_NEAREST,  # nearest is best for binary masks
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    right_mask = cv2.resize(right_mask , dsize = (width , height))
    right_mask = cv2.rotate(right_mask , cv2.ROTATE_90_COUNTERCLOCKWISE)
    temp_undistorted_right_mask = cv2.remap(
        right_mask,
        right_map_x_rectify,
        right_map_y_rectify,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    area = pixel_area_to_cm2(np.mean((int(temp_undistorted_left_mask.sum()) , int(temp_undistorted_right_mask.sum()))) , distance , focal_length )


    return area
def visualize_mask_shift(original_mask, shifted_mask):
    """
    Visualize the original mask and a shifted version with estimated shift values.

    Args:
        original_mask: The original binary mask
        shifted_mask: The shifted version of the mask
    """
    # Ensure masks are binary
    original_bin = _binarize(original_mask)
    shifted_bin = _binarize(shifted_mask)

    # Calculate shift
    dy, dx, score, method = estimate_mask_shift(original_bin, shifted_bin)

    # Create visualization image
    original_color = np.zeros((original_bin.shape[0], original_bin.shape[1], 3), dtype=np.uint8)
    shifted_color = np.zeros_like(original_color)

    # Original mask in red
    original_color[original_bin > 0] = [0, 0, 255]  # Red in BGR

    # Shifted mask in green
    shifted_color[shifted_bin > 0] = [0, 255, 0]    # Green in BGR

    # Combine with overlay
    combined = cv2.addWeighted(original_color, 0.7, shifted_color, 0.7, 0)

    # Add text for shift values
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"dy={dy:.1f}, dx={dx:.1f}, method={method}"
    cv2.putText(combined, text, (10, 50), font, .7, (255, 255, 255), 20)

    # Display with matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Mask Shift Visualization - Score={score:.3f}")
    plt.show()

    return dy, dx, score, method
def generate_keypoints(results, target_folder, labels_folder, confidence: float = 0.2, line_width: int = 8, kpt_radius: int = 10) -> None:
    """
    Generate YOLO-style label files with bounding boxes and keypoints from detection results.

    Each detection result is saved as:
      - An annotated image in `target_folder`.
      - A `.txt` file in `labels_folder` containing YOLO-format annotations:
        class_id center_x center_y width height kpt1_x kpt1_y ... kptN_x kptN_y

    Bounding boxes and keypoints are normalized to [0, 1] relative to image size.

    Args:
        results (list): List of detection results, each containing `.path`, `.boxes.xyxy`, `.boxes.conf`, and `.keypoints.xy`.
        target_folder (str or Path): Directory where annotated images will be saved.
        labels_folder (str or Path): Directory where YOLO-format label files will be saved.
        confidence (float, optional): Minimum confidence threshold for saving a detection. Defaults to 0.2.
        line_width (int, optional): Line thickness for saved bounding box annotations. Defaults to 8.
        kpt_radius (int, optional): Radius for keypoint drawing. Defaults to 10.

    Returns:
        None
    """
    target_folder = Path(target_folder)
    labels_folder = Path(labels_folder)
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
