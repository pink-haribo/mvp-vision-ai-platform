#!/usr/bin/env python3
"""
DICE Format Converter v2.0

Improvements:
- Flat image structure (images/000001.jpg)
- UUID-based dataset IDs
- Full ImageNet class mapping
- Sequential file naming
- Standardized split names (train/val/test)

Usage:
    python convert_to_dice_format_v2.py
"""

import os
import json
import shutil
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image

# COCO class mapping (80 classes)
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

# ImageNet class mapping (extended for datasets we have)
IMAGENET_CLASSES = {
    # Imagenette2-160 classes
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",

    # ImageNet-10 classes (fish and birds)
    "n01443537": "goldfish",
    "n01484850": "great white shark",
    "n01491361": "tiger shark",
    "n01494475": "hammerhead shark",
    "n01496331": "electric ray",
    "n01498041": "stingray",
    "n01514668": "rooster",
    "n01514859": "hen",
    "n01518878": "ostrich",
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


def generate_dataset_id(dataset_name: str, user_id: str = "platform") -> str:
    """Generate UUID-based dataset ID."""
    # Slugify dataset name
    slug = dataset_name.lower().replace(" ", "-").replace("_", "-")
    # Remove special characters
    slug = "".join(c for c in slug if c.isalnum() or c == "-")

    # Generate short UUID
    short_uuid = uuid.uuid4().hex[:8]

    return f"{user_id}-{slug}-{short_uuid}"


def convert_classification_dataset(
    source_dir: Path,
    output_dir: Path,
    dataset_name: str,
    user_id: str = "platform"
) -> None:
    """
    Convert ImageFolder format to DICE Format (Classification) with flat structure.
    """
    print(f"\n{'='*60}")
    print(f"Converting Classification Dataset: {dataset_name}")
    print(f"{'='*60}")

    # Generate dataset ID
    dataset_id = generate_dataset_id(dataset_name, user_id)
    print(f"Dataset ID: {dataset_id}")

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
        class_name = IMAGENET_CLASSES.get(class_code, class_code)

        classes.append({
            "id": idx,
            "name": class_name,
            "code": class_code,
            "color": f"#{(idx * 40 + 100) % 256:02X}{(idx * 80 + 150) % 256:02X}{(idx * 120 + 200) % 256:02X}"
        })
        class_name_to_id[class_code] = idx

    print(f"Found {len(classes)} classes")

    # Process images with flat structure
    images = []
    image_counter = 1
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

                # Generate sequential filename
                file_ext = img_file.suffix
                new_filename = f"{image_counter:06d}{file_ext}"
                rel_path = f"images/{new_filename}"

                # Copy image to flat images/ directory
                dest_file = images_dir / new_filename
                shutil.copy2(img_file, dest_file)

                # Get image dimensions
                width, height, depth = get_image_info(img_file)

                # Add image entry
                images.append({
                    "id": image_counter,
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
                        "source": "imagefolder_conversion",
                        "original_path": str(img_file.relative_to(source_dir))
                    }
                })

                class_distribution[class_name] += 1
                image_counter += 1

    # Count splits
    splits = {"train": 0, "val": 0, "test": 0}
    for img in images:
        splits[img["split"]] += 1

    # Remove empty splits
    splits = {k: v for k, v in splits.items() if v > 0}

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
    print(f"   - Splits: {splits}")
    print(f"   - Classes: {len(classes)}")
    print(f"   - Structure: Flat (images/000001.jpg)")
    print(f"   - Output: {output_dir}")


def convert_yolo_dataset(
    source_dir: Path,
    output_dir: Path,
    dataset_name: str,
    task_type: str,  # "object_detection" or "instance_segmentation"
    user_id: str = "platform"
) -> None:
    """
    Convert YOLO format to DICE Format with flat structure.
    """
    print(f"\n{'='*60}")
    print(f"Converting YOLO Dataset: {dataset_name}")
    print(f"Task: {task_type}")
    print(f"{'='*60}")

    # Generate dataset ID
    dataset_id = generate_dataset_id(dataset_name, user_id)
    print(f"Dataset ID: {dataset_id}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Find labels directory
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

    # Normalize split name
    split_normalized = "train" if "train" in split_name_used else "val"
    print(f"Using split: {split_name_used} -> {split_normalized}")

    # Scan for class IDs
    class_ids_used = set()
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids_used.add(int(parts[0]))

    class_ids_used = sorted(class_ids_used)
    print(f"Found {len(class_ids_used)} classes")

    # Build class mapping
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

    # Process images with flat structure
    images_source_dir = source_dir / "images" / split_name_used
    if not images_source_dir.exists():
        print(f"[ERROR] Images directory not found: {images_source_dir}")
        return

    images_list = []
    image_counter = 1

    for img_file in sorted(images_source_dir.glob("*.jpg")):
        label_file = labels_dir / f"{img_file.stem}.txt"

        if not label_file.exists():
            print(f"[WARN] No label file for {img_file.name}")
            continue

        # Generate sequential filename
        new_filename = f"{image_counter:06d}.jpg"
        rel_path = f"images/{new_filename}"

        # Copy image to flat images/ directory
        dest_file = images_dir / new_filename
        shutil.copy2(img_file, dest_file)

        # Get image dimensions
        width, height, depth = get_image_info(img_file)

        # Parse labels
        annotations_list = []
        annotation_id = image_counter * 1000

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
                    # YOLO bbox format
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    # Convert to absolute xywh
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
                    # Segmentation: parse bbox
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    x = int((x_center_norm - w_norm / 2) * width)
                    y = int((y_center_norm - h_norm / 2) * height)
                    w = int(w_norm * width)
                    h = int(h_norm * height)

                    # Parse polygon if available
                    if len(parts) > 5:
                        polygon_points = []
                        for i in range(1, len(parts), 2):
                            if i + 1 < len(parts):
                                px = float(parts[i]) * width
                                py = float(parts[i + 1]) * height
                                polygon_points.append([px, py])
                        segmentation = [[coord for point in polygon_points for coord in point]]
                    else:
                        # No polygon, use bbox
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
        images_list.append({
            "id": image_counter,
            "file_name": rel_path,
            "width": width,
            "height": height,
            "depth": depth,
            "split": split_normalized,
            "annotations": annotations_list,
            "metadata": {
                "labeled_by": "original_dataset",
                "labeled_at": datetime.now().isoformat(),
                "source": "yolo_conversion",
                "num_annotations": len(annotations_list),
                "original_filename": img_file.name
            }
        })

        image_counter += 1

    if len(images_list) == 0:
        print(f"[WARN] No images converted")
        return

    # Calculate statistics
    total_annotations = sum(len(img.get("annotations", [])) for img in images_list)
    class_distribution = {c["name"]: 0 for c in classes}

    for img in images_list:
        for ann in img.get("annotations", []):
            class_distribution[ann["class_name"]] += 1

    # Create annotations.json
    annotations = {
        "format_version": "1.0",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "created_at": datetime.now().isoformat(),
        "last_modified_at": datetime.now().isoformat(),
        "version": 1,
        "content_hash": "",
        "classes": classes,
        "splits": {split_normalized: len(images_list)},
        "images": images_list,
        "statistics": {
            "total_images": len(images_list),
            "total_annotations": total_annotations,
            "labeled_images": len(images_list),
            "unlabeled_images": 0,
            "avg_annotations_per_image": total_annotations / len(images_list),
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

    print(f"[OK] Converted {len(images_list)} images")
    print(f"   - Total annotations: {total_annotations}")
    print(f"   - Avg per image: {total_annotations / len(images_list):.1f}")
    print(f"   - Structure: Flat (images/000001.jpg)")
    print(f"   - Output: {output_dir}")


def main():
    """Main conversion function."""
    source_root = Path("c:/datasets")
    output_root = Path("c:/datasets/dice_format")

    # Clean existing output
    if output_root.exists():
        print(f"Removing existing output: {output_root}")
        shutil.rmtree(output_root)

    print("\n" + "=" * 60)
    print("DICE Format Converter v2.0 (Improved)")
    print("=" * 60)
    print("Improvements:")
    print("- Flat image structure (images/000001.jpg)")
    print("- UUID-based dataset IDs")
    print("- Full ImageNet class mapping")
    print("- Sequential file naming")
    print("=" * 60)

    datasets_to_convert = [
        # Classification datasets
        {
            "source": source_root / "cls-imagenet-10",
            "output": output_root / "cls-imagenet-10",
            "name": "ImageNet-10 Subset",
            "type": "classification"
        },
        {
            "source": source_root / "cls-imagenette2-160",
            "output": output_root / "cls-imagenette2-160",
            "name": "Imagenette2-160",
            "type": "classification"
        },
        # Detection datasets
        {
            "source": source_root / "det-coco8",
            "output": output_root / "det-coco8",
            "name": "COCO8 Object Detection",
            "type": "detection"
        },
        {
            "source": source_root / "det-coco128",
            "output": output_root / "det-coco128",
            "name": "COCO128 Object Detection",
            "type": "detection"
        },
        # Segmentation datasets
        {
            "source": source_root / "seg-coco8",
            "output": output_root / "seg-coco8",
            "name": "COCO8 Instance Segmentation",
            "type": "segmentation"
        },
        {
            "source": source_root / "seg-coco128",
            "output": output_root / "seg-coco128",
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
                dataset["name"]
            )
        elif dataset["type"] == "detection":
            convert_yolo_dataset(
                dataset["source"],
                dataset["output"],
                dataset["name"],
                "object_detection"
            )
        elif dataset["type"] == "segmentation":
            convert_yolo_dataset(
                dataset["source"],
                dataset["output"],
                dataset["name"],
                "instance_segmentation"
            )

    print("\n" + "=" * 60)
    print("[SUCCESS] All conversions complete!")
    print(f"Output directory: {output_root}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
