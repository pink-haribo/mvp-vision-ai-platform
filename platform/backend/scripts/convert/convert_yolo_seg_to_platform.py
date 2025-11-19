"""
Convert YOLO segmentation format to Platform v1.0 format.

Usage:
    python convert_yolo_seg_to_platform.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import hashlib

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def parse_yolo_segmentation(label_path):
    """
    Parse YOLO segmentation format.

    Format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)

    Returns:
        List of annotations
    """
    annotations = []

    if not os.path.exists(label_path):
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                continue

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # Convert normalized coords to list of [x, y] pairs
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    points.append([coords[i], coords[i + 1]])

            annotations.append({
                'class_id': class_id,
                'points': points  # Normalized coordinates (0-1)
            })

    return annotations


def denormalize_segmentation(points, img_width, img_height):
    """
    Convert normalized YOLO coordinates (0-1) to absolute pixel coordinates.

    Args:
        points: List of [x, y] normalized coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Flattened list of absolute coordinates [x1, y1, x2, y2, ...]
    """
    absolute_coords = []
    for x_norm, y_norm in points:
        x_abs = x_norm * img_width
        y_abs = y_norm * img_height
        absolute_coords.extend([x_abs, y_abs])

    return absolute_coords


def calculate_bbox_from_polygon(flat_coords):
    """
    Calculate bounding box from polygon coordinates.

    Args:
        flat_coords: [x1, y1, x2, y2, x3, y3, ...]

    Returns:
        [x_min, y_min, width, height]
    """
    xs = [flat_coords[i] for i in range(0, len(flat_coords), 2)]
    ys = [flat_coords[i] for i in range(1, len(flat_coords), 2)]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def calculate_polygon_area(flat_coords):
    """
    Calculate polygon area using shoelace formula.

    Args:
        flat_coords: [x1, y1, x2, y2, x3, y3, ...]

    Returns:
        Area in square pixels
    """
    xs = [flat_coords[i] for i in range(0, len(flat_coords), 2)]
    ys = [flat_coords[i] for i in range(1, len(flat_coords), 2)]

    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]

    return abs(area) / 2.0


def calculate_content_hash(annotations_dict):
    """Calculate SHA256 hash of annotations content."""
    content_str = json.dumps(annotations_dict, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()


def convert_yolo_seg_to_platform(
    source_dir,
    output_dir,
    dataset_name="COCO Segmentation Dataset",
    dataset_id="coco-seg-32"
):
    """
    Convert YOLO segmentation dataset to Platform v1.0 format.

    Args:
        source_dir: Path to YOLO dataset (with images/ and labels/)
        output_dir: Path to output directory
        dataset_name: Name of dataset
        dataset_id: Dataset ID
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Paths
    images_dir = source_path / 'images' / 'train2017'
    labels_dir = source_path / 'labels' / 'train2017'

    output_images_dir = output_path / 'images'
    output_images_dir.mkdir(exist_ok=True)

    print(f"Converting YOLO segmentation dataset to Platform v1.0 format...")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()

    # Collect all images
    image_files = sorted([f for f in images_dir.glob('*.jpg')])
    print(f"Found {len(image_files)} images")

    # Build annotations
    images_data = []
    annotation_id_counter = 1000
    total_annotations = 0
    class_distribution = {name: 0 for name in COCO_CLASSES}
    used_classes = set()

    for idx, img_path in enumerate(image_files, start=1):
        img_filename = img_path.name
        label_path = labels_dir / (img_path.stem + '.txt')

        # Copy image to output
        output_img_path = output_images_dir / img_filename
        if not output_img_path.exists():
            import shutil
            shutil.copy2(img_path, output_img_path)

        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                img_depth = len(img.getbands())
        except Exception as e:
            print(f"Warning: Failed to read image {img_filename}: {e}")
            continue

        # Parse YOLO labels
        yolo_annotations = parse_yolo_segmentation(label_path)

        if not yolo_annotations:
            continue  # Skip images without annotations

        # Convert to Platform v1.0 format
        platform_annotations = []

        for yolo_ann in yolo_annotations:
            class_id = yolo_ann['class_id']
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

            # Denormalize coordinates
            flat_coords = denormalize_segmentation(yolo_ann['points'], img_width, img_height)

            # Calculate bbox and area
            bbox = calculate_bbox_from_polygon(flat_coords)
            area = calculate_polygon_area(flat_coords)

            platform_annotations.append({
                "id": annotation_id_counter,
                "class_id": class_id,
                "class_name": class_name,
                "bbox": bbox,
                "bbox_format": "xywh",
                "segmentation": [flat_coords],
                "area": area,
                "iscrowd": 0
            })

            annotation_id_counter += 1
            total_annotations += 1
            class_distribution[class_name] += 1
            used_classes.add(class_id)

        # Add image entry
        images_data.append({
            "id": idx,
            "file_name": img_filename,
            "width": img_width,
            "height": img_height,
            "depth": img_depth,
            "split": "train",
            "annotations": platform_annotations,
            "metadata": {
                "labeled_by": "coco_dataset",
                "labeled_at": datetime.utcnow().isoformat() + "Z",
                "source": "converted_from_yolo_segmentation"
            }
        })

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(image_files)} images...")

    print(f"\nTotal images with annotations: {len(images_data)}")
    print(f"Total annotations: {total_annotations}")
    print(f"Classes used: {len(used_classes)}")

    # Build classes list (only used classes)
    classes_data = []
    color_palette = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B88B", "#FAD7A0",
        "#E8DAEF", "#D5F4E6", "#FADBD8", "#D6EAF8", "#FCF3CF"
    ]

    for class_id in sorted(used_classes):
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        classes_data.append({
            "id": class_id,
            "name": class_name,
            "color": color_palette[class_id % len(color_palette)],
            "supercategory": "object"
        })

    # Calculate statistics
    class_dist_filtered = {k: v for k, v in class_distribution.items() if v > 0}
    avg_annotations = total_annotations / len(images_data) if images_data else 0

    # Build final annotations.json
    annotations_json = {
        "format_version": "1.0",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task_type": "instance_segmentation",

        "created_at": datetime.utcnow().isoformat() + "Z",
        "last_modified_at": datetime.utcnow().isoformat() + "Z",
        "version": 1,

        "migration_info": {
            "migrated_from": "yolo_segmentation",
            "migration_date": datetime.utcnow().isoformat() + "Z",
            "original_format": "YOLO segmentation (normalized polygons)"
        },

        "classes": classes_data,

        "splits": {
            "train": len(images_data)
        },

        "images": images_data,

        "statistics": {
            "total_images": len(images_data),
            "total_annotations": total_annotations,
            "avg_annotations_per_image": round(avg_annotations, 2),
            "num_classes": len(used_classes),
            "class_distribution": class_dist_filtered
        }
    }

    # Calculate content hash
    annotations_json["content_hash"] = "sha256:" + calculate_content_hash(annotations_json)

    # Write annotations.json
    annotations_path = output_path / 'annotations.json'
    with open(annotations_path, 'w', encoding='utf-8') as f:
        json.dump(annotations_json, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Conversion complete!")
    print(f"   Output: {annotations_path}")
    print(f"   Images: {output_images_dir}")
    print(f"\nDataset Summary:")
    print(f"   Format: Platform v1.0 (instance_segmentation)")
    print(f"   Images: {len(images_data)}")
    print(f"   Annotations: {total_annotations}")
    print(f"   Classes: {len(used_classes)}")
    print(f"\nTop 5 classes:")
    for class_name, count in sorted(class_dist_filtered.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   - {class_name}: {count}")


if __name__ == "__main__":
    # Configuration
    SOURCE_DIR = r"C:\datasets\seg-coco32"
    OUTPUT_DIR = r"C:\datasets\dice_format\seg-coco32"
    DATASET_NAME = "COCO Segmentation 32 Images"
    DATASET_ID = "coco-seg-32"

    convert_yolo_seg_to_platform(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        dataset_name=DATASET_NAME,
        dataset_id=DATASET_ID
    )
