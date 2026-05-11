#!/usr/bin/env python3
"""
Prepare a COCO-style dataset for fine-tuning Segment Anything 2.1.

Inputs:
- An images directory containing all training images (jpg/png)
- A single merged annotations JSON file (e.g., segmantation_training_data.json)

Outputs (under --output-dir):
- annotations/instances_train.json
- annotations/instances_val.json

This script is flexible and accepts:
1) Already-COCO JSON (keys: images, annotations, categories) -> validates and rewrites paths
2) A list of records (e.g., Roboflow-style) with per-image annotations. It tries keys:
   - image name: ["image", "file_name", "filename", "img", "img_name", "name"]
   - per-object: ["annotations", "objects", "labels"] items with polygon under ["points", "segmentation"],
     or bbox under ["bbox", (x, y, width, height)]
   - class under ["class", "label", "category", "category_name"]

If polygon points are normalized (0..1), they are scaled to pixel coords using image width/height.

Example usage:
  python prepare_sam2_dataset.py \
    --images-dir "./Fish-SAM-Segmentation--2/train" \
    --annotations-json "./segmantation_training_data.json" \
    --output-dir "./out_sam2_dataset" \
    --val-split 0.1

"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def try_import_pil():
    try:
        from PIL import Image  # type: ignore
        return Image
    except Exception:
        return None


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Return (width, height) for an image. Requires Pillow if size not provided."""
    Image = try_import_pil()
    if Image is None:
        raise RuntimeError(
            f"Pillow is required to read image sizes for {image_path}. Install with: pip install Pillow"
        )
    with Image.open(str(image_path)) as im:
        return im.size  # (width, height)


def find_images(images_dir: Path) -> Dict[str, Path]:
    """Map base filename -> absolute Path for all images in images_dir (recursive)."""
    mapping: Dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            mapping[p.name] = p
    return mapping


def is_coco(data: Any) -> bool:
    return isinstance(data, dict) and "images" in data and "annotations" in data


def polygon_area(xy: List[float]) -> float:
    """Shoelace formula. xy is [x1,y1,x2,y2,...]. Returns absolute area."""
    if len(xy) < 6:
        return 0.0
    it = iter(xy)
    pts = list(zip(it, it))
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def bbox_from_polygon(xy: List[float]) -> List[float]:
    xs = xy[0::2]
    ys = xy[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def close_polygon(xy: List[float]) -> List[float]:
    # COCO doesn't require repeating the first point; ensure at least 3 points
    if len(xy) >= 6:
        return xy
    return xy


def scale_points_if_normalized(points: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[float, float]]:
    # Heuristic: if all values are in [0, 1.5], assume normalized
    if points and all(0.0 <= x <= 1.5 and 0.0 <= y <= 1.5 for x, y in points):
        return [(x * w, y * h) for x, y in points]
    return points


def flatten_points(points: List[Tuple[float, float]]) -> List[float]:
    xy: List[float] = []
    for x, y in points:
        xy.append(float(x))
        xy.append(float(y))
    return xy


def to_points_list(obj: Any) -> Optional[List[Tuple[float, float]]]:
    """Support formats like [{"x":..,"y":..}], [[x,y],[x,y]], or flat [x1,y1,...]."""
    if not obj:
        return None
    pts: List[Tuple[float, float]] = []
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        # List of dicts
        if isinstance(obj[0], dict) and "x" in obj[0] and "y" in obj[0]:
            for d in obj:
                pts.append((float(d["x"]), float(d["y"])))
            return pts
        # List of list/tuples
        if isinstance(obj[0], (list, tuple)) and len(obj[0]) >= 2:
            for xy in obj:
                pts.append((float(xy[0]), float(xy[1])))
            return pts
        # Flat list
        if len(obj) % 2 == 0 and all(isinstance(v, (int, float)) for v in obj):
            it = iter(obj)
            return [(float(x), float(y)) for x, y in zip(it, it)]
    return None


def detect_image_name(rec: Dict[str, Any]) -> Optional[str]:
    for k in ["image", "file_name", "filename", "img", "img_name", "name", "path"]:
        if k in rec and isinstance(rec[k], str) and rec[k]:
            return os.path.basename(rec[k])
    return None


def iter_objects(rec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # Common fields to hold object annotations
    for k in ["annotations", "objects", "labels"]:
        if k in rec and isinstance(rec[k], list):
            for obj in rec[k]:
                if isinstance(obj, dict):
                    yield obj
    # Some datasets store a single object at top-level
    if "segmentation" in rec or "points" in rec or "bbox" in rec:
        yield rec


def detect_category_name(obj: Dict[str, Any]) -> str:
    for k in ["class", "label", "category", "category_name", "name"]:
        v = obj.get(k)
        if isinstance(v, str) and v:
            return v
    return "object"


def extract_polygon(obj: Dict[str, Any], w: int, h: int) -> Optional[List[float]]:
    # Prefer explicit segmentation list
    seg = obj.get("segmentation")
    pts = to_points_list(seg) if seg is not None else None
    if pts is None:
        # Try points
        pts = to_points_list(obj.get("points"))
    if pts is None and "bbox" in obj:
        # Derive polygon from bbox
        bbox = obj["bbox"]
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            pts = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
        elif isinstance(bbox, dict):
            x = float(bbox.get("x", 0))
            y = float(bbox.get("y", 0))
            bw = float(bbox.get("width", bbox.get("w", 0)))
            bh = float(bbox.get("height", bbox.get("h", 0)))
            pts = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
    if pts is None:
        return None
    pts = scale_points_if_normalized(pts, w, h)
    xy = flatten_points(pts)
    return close_polygon(xy)


def build_coco_from_list(records: List[Dict[str, Any]], images_dir: Path) -> Dict[str, Any]:
    images_index: Dict[str, int] = {}
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    categories_map: Dict[str, int] = {}
    categories: List[Dict[str, Any]] = []

    image_paths = find_images(images_dir)
    if not image_paths:
        eprint(f"No images found under {images_dir}")

    ann_id = 1
    img_id = 1

    for rec in records:
        if not isinstance(rec, dict):
            continue
        img_name = detect_image_name(rec)
        if not img_name:
            continue
        base = os.path.basename(img_name)
        img_path = image_paths.get(base)
        if img_path is None:
            # try case-insensitive match
            base_lower = base.lower()
            for k, v in image_paths.items():
                if k.lower() == base_lower:
                    img_path = v
                    break
        if img_path is None:
            eprint(f"Warning: image not found for record: {base}")
            continue

        # Determine size
        width = None
        height = None
        for k in ["width", "image_width", "w"]:
            if k in rec and isinstance(rec[k], (int, float)):
                width = int(rec[k])
                break
        for k in ["height", "image_height", "h"]:
            if k in rec and isinstance(rec[k], (int, float)):
                height = int(rec[k])
                break
        if width is None or height is None:
            try:
                width, height = get_image_size(img_path)
            except Exception as ex:
                eprint(f"Error getting size for {img_path}: {ex}")
                continue

        # Add image if not already added
        if base not in images_index:
            images.append({
                "id": img_id,
                "file_name": base,  # keep relative file name; training code should point to images_dir
                "width": int(width),
                "height": int(height),
            })
            images_index[base] = img_id
            cur_img_id = img_id
            img_id += 1
        else:
            cur_img_id = images_index[base]

        # Objects
        for obj in iter_objects(rec):
            cat_name = detect_category_name(obj)
            if cat_name not in categories_map:
                cid = len(categories_map) + 1
                categories_map[cat_name] = cid
                categories.append({"id": cid, "name": cat_name, "supercategory": "object"})

            xy = extract_polygon(obj, width, height)
            if xy is None or len(xy) < 6:
                # skip degenerate
                continue
            bbox = bbox_from_polygon(xy)
            area = polygon_area(xy)
            annotations.append({
                "id": ann_id,
                "image_id": cur_img_id,
                "category_id": categories_map[cat_name],
                "segmentation": [xy],
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0,
            })
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def rewrite_coco_paths(data: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure file_name is a basename only; do not modify image ids
    for img in data.get("images", []):
        if isinstance(img, dict) and "file_name" in img and isinstance(img["file_name"], str):
            img["file_name"] = os.path.basename(img["file_name"]) or img["file_name"]
    return data


def train_val_split(coco: Dict[str, Any], val_split: float, seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    images = coco["images"]
    anns = coco["annotations"]
    cats = coco.get("categories", [])

    # Group annotations by image_id
    by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for a in anns:
        by_img[int(a["image_id"])].append(a)

    img_ids = [int(img["id"]) for img in images]
    random.Random(seed).shuffle(img_ids)
    n_val = int(len(img_ids) * val_split)
    val_ids = set(img_ids[:n_val])
    train_ids = set(img_ids[n_val:])

    def subset(ids: set[int]) -> Dict[str, Any]:
        imgs = [img for img in images if int(img["id"]) in ids]
        ann_list = [a for a in anns if int(a["image_id"]) in ids]
        return {"images": imgs, "annotations": ann_list, "categories": cats}

    return subset(train_ids), subset(val_ids)


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO dataset for SAM 2.1 fine-tuning")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--annotations-json", type=str, required=True, help="Path to merged annotations JSON")
    parser.add_argument("--output-dir", type=str, default="./sam2_dataset_out", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction (0-1)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    ann_path = Path(args.annotations_json).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not images_dir.exists():
        eprint(f"Images dir not found: {images_dir}")
        sys.exit(1)
    if not ann_path.exists():
        # Common typos fallback
        fallback1 = ann_path.with_name("segmantation_training_data.json")
        fallback2 = ann_path.with_name("segmantation_trainaing_data.json")
        if fallback1.exists():
            ann_path = fallback1
        elif fallback2.exists():
            ann_path = fallback2
        else:
            eprint(f"Annotations JSON not found: {ann_path}")
            sys.exit(1)

    data = load_json(ann_path)
    if is_coco(data):
        eprint("Detected COCO-style annotations. Rewriting image file names to basenames.")
        coco = rewrite_coco_paths(data)
    elif isinstance(data, list):
        eprint("Detected list-based annotations. Converting to COCO format...")
        coco = build_coco_from_list(data, images_dir)
    else:
        eprint("Unknown annotations format. Expecting COCO dict or list of records.")
        sys.exit(1)

    # Basic validation
    if not coco.get("images"):
        eprint("No images found after processing.")
        sys.exit(1)
    if not coco.get("annotations"):
        eprint("Warning: No annotations found. Proceeding to write empty annotations.")

    train, val = train_val_split(coco, args.val_split)

    ann_out_dir = out_dir / "annotations"
    save_json(train, ann_out_dir / "instances_train.json")
    save_json(val, ann_out_dir / "instances_val.json")

    # Write an images manifest for convenience (list of found image file names)
    img_names = [img["file_name"] for img in coco["images"]]
    save_json(img_names, out_dir / "images_manifest.json")

    print("Done.")
    print(f"Images dir: {images_dir}")
    print(f"Wrote: {ann_out_dir / 'instances_train.json'}")
    print(f"Wrote: {ann_out_dir / 'instances_val.json'}")
    print(f"Images manifest: {out_dir / 'images_manifest.json'}")
    print(f"Counts -> images: {len(coco['images'])}, annotations: {len(coco['annotations'])}, categories: {len(coco.get('categories', []))}")


if __name__ == "__main__":
    main()
