"""YOLO Format to DICE Format Converter.

Converts YOLO detection format to DICE (COCO-style) format with proper filename matching.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image


# COCO 80 classes (standard object detection classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def convert_yolo_bbox_to_coco(
    yolo_bbox: List[float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert YOLO bbox format to COCO format.

    Args:
        yolo_bbox: [x_center, y_center, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple[x, y, w, h] in COCO format (absolute pixels, top-left corner)
    """
    x_center, y_center, width, height = yolo_bbox

    # Convert from normalized to absolute
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    # Convert from center to top-left corner
    x = x_center_abs - (width_abs / 2)
    y = y_center_abs - (height_abs / 2)

    return (x, y, width_abs, height_abs)


def read_yolo_label(label_file: Path) -> List[Tuple[int, List[float]]]:
    """
    Read YOLO label file.

    Args:
        label_file: Path to YOLO .txt label file

    Returns:
        List of (class_id, [x_center, y_center, width, height])
    """
    annotations = []

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            annotations.append((class_id, [x_center, y_center, width, height]))

    return annotations


def convert_yolo_to_dice(
    yolo_dataset_dir: str,
    output_dir: str,
    class_names: List[str] = None
) -> Dict:
    """
    Convert YOLO format dataset to DICE (COCO-style) format.

    YOLO Format:
    - images/train2017/*.jpg
    - labels/train2017/*.txt (one per image)

    DICE Format:
    - images/*.jpg (all images in one folder)
    - annotations.json (COCO-style)

    Args:
        yolo_dataset_dir: Path to YOLO dataset root
        output_dir: Path to output DICE dataset
        class_names: List of class names (defaults to COCO 80 classes)

    Returns:
        Dict with conversion statistics
    """
    yolo_path = Path(yolo_dataset_dir)
    output_path = Path(output_dir)

    # Use COCO classes by default
    if class_names is None:
        class_names = COCO_CLASSES

    print(f"[YOLO→DICE] Converting YOLO dataset to DICE format")
    print(f"[YOLO→DICE] Source: {yolo_path}")
    print(f"[YOLO→DICE] Output: {output_path}")

    # Find all image and label directories
    images_dirs = list(yolo_path.glob("images/*"))
    labels_dirs = list(yolo_path.glob("labels/*"))

    print(f"[YOLO→DICE] Found {len(images_dirs)} image directories")
    print(f"[YOLO→DICE] Found {len(labels_dirs)} label directories")

    # Create output directories
    output_images_dir = output_path / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DICE structures
    dice_images = []
    dice_annotations = []
    dice_categories = []

    # Create categories (use all detected class IDs)
    detected_class_ids = set()

    # First pass: collect all class IDs from labels
    for labels_dir in labels_dirs:
        for label_file in labels_dir.glob("*.txt"):
            annotations = read_yolo_label(label_file)
            for class_id, _ in annotations:
                detected_class_ids.add(class_id)

    # Create categories for detected classes
    for class_id in sorted(detected_class_ids):
        if class_id < len(class_names):
            category_name = class_names[class_id]
        else:
            category_name = f"class_{class_id}"

        dice_categories.append({
            "id": class_id,
            "name": category_name,
            "supercategory": "object"
        })

    print(f"[YOLO→DICE] Detected {len(detected_class_ids)} classes: {sorted(detected_class_ids)}")

    # Process each split (train2017, val2017, etc.)
    image_id = 1
    annotation_id = 1
    total_images = 0
    total_annotations = 0

    for images_dir in images_dirs:
        split_name = images_dir.name  # e.g., "train2017"
        labels_dir = yolo_path / "labels" / split_name

        if not labels_dir.exists():
            print(f"[YOLO→DICE] Warning: No labels directory for {split_name}")
            continue

        print(f"[YOLO→DICE] Processing split: {split_name}")

        # Process each image
        image_files = sorted(images_dir.glob("*.jpg"))
        print(f"[YOLO→DICE] Found {len(image_files)} images in {split_name}")

        for img_file in image_files:
            # Get corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"

            # Copy image to output
            output_img_path = output_images_dir / img_file.name
            shutil.copy2(img_file, output_img_path)

            # Get image dimensions
            with Image.open(img_file) as img:
                img_width, img_height = img.size

            # Create image entry with EXACT filename
            dice_image = {
                "id": image_id,
                "file_name": img_file.name,  # ← CRITICAL: Use actual filename
                "width": img_width,
                "height": img_height
            }
            dice_images.append(dice_image)

            # Process annotations if label file exists
            if label_file.exists():
                yolo_annotations = read_yolo_label(label_file)

                for class_id, yolo_bbox in yolo_annotations:
                    # Convert YOLO bbox to COCO format
                    x, y, w, h = convert_yolo_bbox_to_coco(yolo_bbox, img_width, img_height)

                    # Create annotation entry
                    dice_annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    }
                    dice_annotations.append(dice_annotation)
                    annotation_id += 1
                    total_annotations += 1

            image_id += 1
            total_images += 1

    # Create annotations.json
    annotations_data = {
        "version": "1.0",
        "dataset_info": {
            "description": f"Converted from YOLO format: {yolo_path.name}",
            "source": "YOLO",
            "date_created": "2025-11-05"
        },
        "categories": dice_categories,
        "images": dice_images,
        "annotations": dice_annotations
    }

    annotations_file = output_path / "annotations.json"
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations_data, f, indent=2, ensure_ascii=False)

    print(f"\n[YOLO→DICE] Conversion complete!")
    print(f"[YOLO→DICE] Total images: {total_images}")
    print(f"[YOLO→DICE] Total annotations: {total_annotations}")
    print(f"[YOLO→DICE] Categories: {len(dice_categories)}")
    print(f"[YOLO→DICE] Output: {output_path}")
    print(f"[YOLO→DICE] annotations.json: {annotations_file}")

    # Verify filename matching
    print(f"\n[YOLO→DICE] Verifying filename matching...")
    mismatched = []
    for dice_img in dice_images[:5]:  # Check first 5
        img_path = output_images_dir / dice_img['file_name']
        if not img_path.exists():
            mismatched.append(dice_img['file_name'])

    if mismatched:
        print(f"[YOLO→DICE] WARNING: {len(mismatched)} files not found: {mismatched}")
    else:
        print(f"[YOLO→DICE] SUCCESS: All filenames match correctly!")

    return {
        'output_dir': str(output_path),
        'annotations_file': str(annotations_file),
        'num_images': total_images,
        'num_annotations': total_annotations,
        'num_categories': len(dice_categories),
        'categories': [cat['name'] for cat in dice_categories]
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python yolo_to_dice.py <yolo_dataset_dir> <output_dir>")
        print("Example: python yolo_to_dice.py C:/datasets/det-coco32 C:/datasets_dice/det-coco32")
        sys.exit(1)

    result = convert_yolo_to_dice(sys.argv[1], sys.argv[2])
    print(f"\n{json.dumps(result, indent=2)}")
