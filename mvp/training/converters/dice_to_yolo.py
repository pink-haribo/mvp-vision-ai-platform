"""DICE Format to YOLO Format Converter."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import shutil


def convert_dice_to_yolo(dice_dataset_dir: str, output_dir: str) -> Dict[str, any]:
    """
    Convert DICE format dataset to YOLO format.

    DICE Format:
    - annotations.json with COCO-style structure
    - images/ directory

    YOLO Format:
    - images/ directory (train/, val/)
    - labels/ directory (train/, val/) with .txt files
    - data.yaml with class names and paths

    Args:
        dice_dataset_dir: Path to DICE dataset directory
        output_dir: Path to output YOLO dataset directory

    Returns:
        Dict with conversion statistics and dataset info
    """
    dice_path = Path(dice_dataset_dir)
    output_path = Path(output_dir)

    # Load annotations.json
    annotations_file = dice_path / "annotations.json"
    if not annotations_file.exists():
        raise FileNotFoundError(f"annotations.json not found in {dice_path}")

    with open(annotations_file, 'r', encoding='utf-8') as f:
        dice_data = json.load(f)

    # Extract info
    categories = dice_data.get('categories', [])
    images = dice_data.get('images', [])
    annotations = dice_data.get('annotations', [])

    print(f"[DICE→YOLO] Categories: {len(categories)}, Images: {len(images)}, Annotations: {len(annotations)}")

    # Create category ID to index mapping (YOLO uses 0-indexed class IDs)
    category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    category_names = [cat['name'] for cat in categories]

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Create output directories
    images_train_dir = output_path / "images" / "train"
    labels_train_dir = output_path / "labels" / "train"
    images_train_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)

    # Convert annotations
    converted_images = 0
    converted_annotations = 0

    # If images list is empty, reconstruct from annotations
    if not images:
        print("[DICE→YOLO] No images in annotations.json, reconstructing from annotations")
        images_dir = dice_path / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"images/ directory not found in {dice_path}")

        # Get all image files
        image_files = sorted(images_dir.glob("*.jpg"))
        print(f"[DICE→YOLO] Found {len(image_files)} image files")

        # Get unique image_ids from annotations
        unique_image_ids = sorted(set(ann['image_id'] for ann in annotations if 'image_id' in ann))
        print(f"[DICE→YOLO] Found {len(unique_image_ids)} unique image_ids in annotations")

        # Strategy 1: Try exact filename match
        image_id_to_file = {}
        remaining_files = list(image_files)

        for image_id in unique_image_ids:
            # Try matching with or without "image_" prefix
            possible_names = [
                f"{image_id}.jpg",
                f"{image_id.replace('image_', '')}.jpg",
                f"{image_id.replace('_', '')}.jpg"
            ]

            for img_file in remaining_files:
                if img_file.name in possible_names:
                    image_id_to_file[image_id] = img_file
                    remaining_files.remove(img_file)
                    break

        # Strategy 2: Sequential matching if Strategy 1 didn't work
        if len(image_id_to_file) < len(unique_image_ids) and len(image_files) == len(unique_image_ids):
            print(f"[DICE→YOLO] Using sequential matching strategy")
            image_id_to_file = dict(zip(unique_image_ids, image_files))

        # Create images list
        for image_id, img_file in image_id_to_file.items():
            # Try to get actual image dimensions
            width, height = 640, 480  # defaults
            try:
                from PIL import Image
                with Image.open(img_file) as img:
                    width, height = img.size
            except Exception:
                pass

            images.append({
                'id': image_id,
                'file_name': img_file.name,
                'width': width,
                'height': height
            })

        print(f"[DICE→YOLO] Reconstructed {len(images)} image entries")

    for image in images:
        image_id = image.get('id')
        file_name = image.get('file_name')
        width = image.get('width', 640)
        height = image.get('height', 480)

        if not file_name or not image_id:
            continue

        # Copy image file
        src_image = dice_path / "images" / file_name
        if not src_image.exists():
            # Try with just the filename
            possible_files = list((dice_path / "images").glob(f"*{Path(file_name).stem}*"))
            if possible_files:
                src_image = possible_files[0]
            else:
                print(f"[DICE→YOLO] Warning: Image not found: {src_image}")
                continue

        dst_image = images_train_dir / file_name
        shutil.copy2(src_image, dst_image)

        # Convert annotations to YOLO format
        image_annotations = annotations_by_image.get(image_id, [])

        # Create label file
        label_file = labels_train_dir / f"{Path(file_name).stem}.txt"
        yolo_lines = []

        for ann in image_annotations:
            bbox = ann.get('bbox')
            category_id = ann.get('category_id')

            if not bbox or category_id is None:
                continue

            # DICE/COCO bbox format: [x, y, width, height] (top-left corner)
            x, y, w, h = bbox

            # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            # Get class index
            class_idx = category_id_to_idx.get(category_id, 0)

            yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            converted_annotations += 1

        # Write label file
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))

        converted_images += 1

    # Create data.yaml
    data_yaml_content = f"""# YOLO Dataset Configuration
# Converted from DICE format

# Paths (relative to this file)
path: {output_path.absolute()}  # dataset root dir
train: images/train  # train images
val: images/train  # using train for both (split later if needed)

# Classes
nc: {len(category_names)}  # number of classes
names: {category_names}  # class names
"""

    data_yaml_path = output_path / "data.yaml"
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)

    print(f"[DICE→YOLO] Converted {converted_images} images with {converted_annotations} annotations")
    print(f"[DICE→YOLO] Output: {output_path}")
    print(f"[DICE→YOLO] Classes: {category_names}")

    return {
        'output_dir': str(output_path),
        'data_yaml': str(data_yaml_path),
        'num_classes': len(category_names),
        'class_names': category_names,
        'num_images': converted_images,
        'num_annotations': converted_annotations
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python dice_to_yolo.py <dice_dataset_dir> <output_dir>")
        sys.exit(1)

    result = convert_dice_to_yolo(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
