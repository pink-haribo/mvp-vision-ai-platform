#!/usr/bin/env python3
"""
DICE Format Converter

Converts existing datasets (ImageFolder, YOLO) to DICE Format v1.0.

Usage:
    python convert_to_dice_format.py

This script converts all datasets in c:/datasets/ to DICE Format
and outputs them to c:/datasets/dice_format/.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image

# COCO class mapping (80 classes, using subset present in coco8/coco128)
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# Imagenette class mapping
IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute"
}


def calculate_content_hash(data: dict) -> str:
    """Calculate SHA256 hash of annotations data."""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def get_image_info(image_path: Path) -> Tuple[int, int, int]:
    """Get image width, height, depth."""
    with Image.open(image_path) as img:
        width, height = img.size
        depth = len(img.getbands())  # RGB=3, RGBA=4, L=1
        return width, height, depth


def convert_classification_dataset(
    source_dir: Path,
    output_dir: Path,
    dataset_id: str,
    dataset_name: str
) -> None:
    """
    Convert ImageFolder format to DICE Format (Classification).

    Expected structure:
        source_dir/
        ├── train/
        │   ├── class_0/
        │   │   ├── img001.jpg
        │   │   └── img002.jpg
        │   └── class_1/
        └── val/
            ├── class_0/
            └── class_1/
    """
    print(f"\n{'='*60}")
    print(f"Converting Classification Dataset: {dataset_name}")
    print(f"{'='*60}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Scan class directories
    train_dir = source_dir / "train"
    val_dir = source_dir / "val"

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    # Build class mapping
    classes = []
    class_name_to_id = {}

    for idx, class_dir in enumerate(class_dirs):
        class_code = class_dir.name
        class_name = IMAGENETTE_CLASSES.get(class_code, class_code)

        classes.append({
            "id": idx,
            "name": class_name,
            "code": class_code,
            "color": f"#{(idx * 40 + 100) % 256:02X}{(idx * 80 + 150) % 256:02X}{(idx * 120 + 200) % 256:02X}"
        })
        class_name_to_id[class_code] = idx

    print(f"Found {len(classes)} classes: {[c['name'] for c in classes]}")

    # Process images
    images = []
    image_id = 1
    class_distribution = {c["name"]: 0 for c in classes}

    for split in ["train", "val"]:
        split_dir = source_dir / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_code = class_dir.name
            class_id = class_name_to_id.get(class_code)
            if class_id is None:
                continue

            class_name = classes[class_id]["name"]

            for img_file in class_dir.glob("*.JPEG"):
                if not img_file.is_file():
                    continue

                # Copy image to output
                rel_path = f"{split}/{class_code}/{img_file.name}"
                dest_path = images_dir / split / class_code
                dest_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_file, dest_path / img_file.name)

                # Get image dimensions
                width, height, depth = get_image_info(img_file)

                # Add image entry
                images.append({
                    "id": image_id,
                    "file_name": rel_path,
                    "width": width,
                    "height": height,
                    "depth": depth,
                    "split": split,
                    "annotation": {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": 1.0
                    },
                    "metadata": {
                        "labeled_by": "original_dataset",
                        "labeled_at": datetime.now().isoformat(),
                        "source": "imagefolder_conversion"
                    }
                })

                class_distribution[class_name] += 1
                image_id += 1

    # Count splits
    splits = {"train": 0, "val": 0}
    for img in images:
        splits[img["split"]] += 1

    # Create annotations.json
    annotations = {
        "format_version": "1.0",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task_type": "image_classification",
        "created_at": datetime.now().isoformat(),
        "last_modified_at": datetime.now().isoformat(),
        "version": 1,
        "content_hash": "",  # Will be filled after
        "classes": classes,
        "splits": splits,
        "images": images,
        "statistics": {
            "total_images": len(images),
            "total_annotations": len(images),
            "labeled_images": len(images),
            "unlabeled_images": 0,
            "avg_annotations_per_image": 1.0,
            "class_distribution": class_distribution,
            "labeling_progress": 1.0
        }
    }

    # Calculate content hash
    annotations["content_hash"] = f"sha256:{calculate_content_hash(annotations)}"

    # Write annotations.json
    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Create meta.json
    meta = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "version": 1,
        "content_hash": annotations["content_hash"],
        "last_modified_at": annotations["last_modified_at"],
        "total_images": len(images),
        "task_type": "image_classification",
        "num_classes": len(classes)
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Converted {len(images)} images")
    print(f"   - Train: {splits['train']}, Val: {splits['val']}")
    print(f"   - Classes: {len(classes)}")
    print(f"   - Output: {output_dir}")


def convert_yolo_dataset(
    source_dir: Path,
    output_dir: Path,
    dataset_id: str,
    dataset_name: str,
    task_type: str  # "object_detection" or "instance_segmentation"
) -> None:
    """
    Convert YOLO format to DICE Format (Detection or Segmentation).

    Expected structure:
        source_dir/
        ├── images/
        │   └── train/
        │       ├── img001.jpg
        │       └── img002.jpg
        └── labels/
            └── train/
                ├── img001.txt
                └── img002.txt
    """
    print(f"\n{'='*60}")
    print(f"Converting YOLO Dataset: {dataset_name}")
    print(f"Task: {task_type}")
    print(f"{'='*60}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Find labels directory (try different split names)
    labels_base = source_dir / "labels"
    labels_dir = None

    for split_name in ["train", "train2017", "val", "val2017"]:
        candidate = labels_base / split_name
        if candidate.exists() and list(candidate.glob("*.txt")):
            labels_dir = candidate
            split_name_used = split_name
            break

    if labels_dir is None:
        print(f"[ERROR] No labels found in {labels_base}")
        return

    print(f"Using split: {split_name_used}")

    # Scan for class IDs used in labels
    class_ids_used = set()

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids_used.add(int(parts[0]))

    class_ids_used = sorted(class_ids_used)
    print(f"Found class IDs: {class_ids_used}")

    # Build class mapping (using COCO classes)
    classes = []
    coco_id_to_dice_id = {}

    for idx, coco_id in enumerate(class_ids_used):
        class_name = COCO_CLASSES.get(coco_id, f"class_{coco_id}")
        classes.append({
            "id": idx,
            "name": class_name,
            "coco_id": coco_id,
            "color": f"#{(idx * 40 + 100) % 256:02X}{(idx * 80 + 150) % 256:02X}{(idx * 120 + 200) % 256:02X}"
        })
        coco_id_to_dice_id[coco_id] = idx

    print(f"Classes: {[c['name'] for c in classes]}")

    # Process images and labels (use same split as labels)
    images_source_dir = source_dir / "images" / split_name_used
    if not images_source_dir.exists():
        print(f"[ERROR] Images directory not found: {images_source_dir}")
        return

    images_list = []
    image_id = 1

    for img_file in images_source_dir.glob("*.jpg"):
        label_file = labels_dir / f"{img_file.stem}.txt"

        if not label_file.exists():
            print(f"[WARN] No label file for {img_file.name}")
            continue

        # Copy image
        rel_path = f"{split_name_used}/{img_file.name}"
        dest_path = images_dir / split_name_used
        dest_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_file, dest_path / img_file.name)

        # Get image dimensions
        width, height, depth = get_image_info(img_file)

        # Parse labels
        annotations_list = []
        annotation_id = image_id * 1000

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                coco_class_id = int(parts[0])
                dice_class_id = coco_id_to_dice_id.get(coco_class_id)
                if dice_class_id is None:
                    continue

                class_name = classes[dice_class_id]["name"]

                if task_type == "object_detection":
                    # YOLO format: class_id x_center y_center width height (normalized)
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    # Convert to absolute coordinates (xywh format)
                    x = int((x_center_norm - w_norm / 2) * width)
                    y = int((y_center_norm - h_norm / 2) * height)
                    w = int(w_norm * width)
                    h = int(h_norm * height)

                    annotations_list.append({
                        "id": annotation_id,
                        "class_id": dice_class_id,
                        "class_name": class_name,
                        "bbox": [x, y, w, h],
                        "bbox_format": "xywh",
                        "area": w * h,
                        "iscrowd": 0
                    })

                elif task_type == "instance_segmentation":
                    # Segmentation: class_id x1 y1 x2 y2 x3 y3 ... (normalized polygon)
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    # Convert bbox
                    x = int((x_center_norm - w_norm / 2) * width)
                    y = int((y_center_norm - h_norm / 2) * height)
                    w = int(w_norm * width)
                    h = int(h_norm * height)

                    # For polygon (if more than 5 values, it's polygon points)
                    if len(parts) > 5:
                        # Parse polygon points (normalized)
                        polygon_points = []
                        for i in range(1, len(parts), 2):
                            if i + 1 < len(parts):
                                px = float(parts[i]) * width
                                py = float(parts[i + 1]) * height
                                polygon_points.append([px, py])

                        segmentation = [[coord for point in polygon_points for coord in point]]
                    else:
                        # No polygon, use bbox as approximation
                        segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]

                    annotations_list.append({
                        "id": annotation_id,
                        "class_id": dice_class_id,
                        "class_name": class_name,
                        "bbox": [x, y, w, h],
                        "segmentation": segmentation,
                        "area": w * h,
                        "iscrowd": 0
                    })

                annotation_id += 1

        # Add image entry
        if task_type == "object_detection":
            annotation_field = "annotations"
        else:
            annotation_field = "annotations"

        # Normalize split name (train2017 → train)
        split_normalized = "train" if "train" in split_name_used else "val"

        images_list.append({
            "id": image_id,
            "file_name": rel_path,
            "width": width,
            "height": height,
            "depth": depth,
            "split": split_normalized,
            annotation_field: annotations_list,
            "metadata": {
                "labeled_by": "original_dataset",
                "labeled_at": datetime.now().isoformat(),
                "source": "yolo_conversion",
                "num_annotations": len(annotations_list)
            }
        })

        image_id += 1

    # Calculate statistics
    total_annotations = sum(len(img.get("annotations", [])) for img in images_list)
    class_distribution = {c["name"]: 0 for c in classes}

    for img in images_list:
        for ann in img.get("annotations", []):
            class_distribution[ann["class_name"]] += 1

    # Create annotations.json
    # Get split name from first image if exists
    split_key = images_list[0]["split"] if images_list else "train"

    annotations = {
        "format_version": "1.0",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "created_at": datetime.now().isoformat(),
        "last_modified_at": datetime.now().isoformat(),
        "version": 1,
        "content_hash": "",  # Will be filled
        "classes": classes,
        "splits": {split_key: len(images_list)},
        "images": images_list,
        "statistics": {
            "total_images": len(images_list),
            "total_annotations": total_annotations,
            "labeled_images": len(images_list),
            "unlabeled_images": 0,
            "avg_annotations_per_image": total_annotations / len(images_list) if images_list else 0,
            "class_distribution": class_distribution,
            "labeling_progress": 1.0
        }
    }

    # Calculate content hash
    annotations["content_hash"] = f"sha256:{calculate_content_hash(annotations)}"

    # Write annotations.json
    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Create meta.json
    meta = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "version": 1,
        "content_hash": annotations["content_hash"],
        "last_modified_at": annotations["last_modified_at"],
        "total_images": len(images_list),
        "task_type": task_type,
        "num_classes": len(classes)
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if len(images_list) == 0:
        print(f"[WARN] No images converted (dataset may be empty or incorrectly formatted)")
        return

    print(f"[OK] Converted {len(images_list)} images")
    print(f"   - Total annotations: {total_annotations}")
    print(f"   - Avg per image: {total_annotations / len(images_list):.1f}")
    print(f"   - Output: {output_dir}")


def main():
    """Main conversion function."""
    source_root = Path("c:/datasets")
    output_root = Path("c:/datasets/dice_format")

    print("\n" + "=" * 60)
    print("DICE Format Converter v1.0")
    print("=" * 60)

    datasets_to_convert = [
        # Classification datasets
        {
            "source": source_root / "cls-imagenet-10",
            "output": output_root / "cls-imagenet-10",
            "id": "imagenet-10",
            "name": "ImageNet-10 Subset",
            "type": "classification"
        },
        {
            "source": source_root / "cls-imagenette2-160",
            "output": output_root / "cls-imagenette2-160",
            "id": "imagenette2-160",
            "name": "Imagenette2-160",
            "type": "classification"
        },
        # Detection datasets
        {
            "source": source_root / "det-coco8",
            "output": output_root / "det-coco8",
            "id": "coco8-detection",
            "name": "COCO8 Object Detection",
            "type": "detection"
        },
        {
            "source": source_root / "det-coco128",
            "output": output_root / "det-coco128",
            "id": "coco128-detection",
            "name": "COCO128 Object Detection",
            "type": "detection"
        },
        # Segmentation datasets
        {
            "source": source_root / "seg-coco8",
            "output": output_root / "seg-coco8",
            "id": "coco8-segmentation",
            "name": "COCO8 Instance Segmentation",
            "type": "segmentation"
        },
        {
            "source": source_root / "seg-coco128",
            "output": output_root / "seg-coco128",
            "id": "coco128-segmentation",
            "name": "COCO128 Instance Segmentation",
            "type": "segmentation"
        }
    ]

    for dataset in datasets_to_convert:
        if not dataset["source"].exists():
            print(f"\n[SKIP] {dataset['name']} - source not found: {dataset['source']}")
            continue

        if dataset["type"] == "classification":
            convert_classification_dataset(
                dataset["source"],
                dataset["output"],
                dataset["id"],
                dataset["name"]
            )
        elif dataset["type"] == "detection":
            convert_yolo_dataset(
                dataset["source"],
                dataset["output"],
                dataset["id"],
                dataset["name"],
                "object_detection"
            )
        elif dataset["type"] == "segmentation":
            convert_yolo_dataset(
                dataset["source"],
                dataset["output"],
                dataset["id"],
                dataset["name"],
                "instance_segmentation"
            )

    print("\n" + "=" * 60)
    print("[SUCCESS] All conversions complete!")
    print(f"Output directory: {output_root}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
