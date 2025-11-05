"""
DICE format to split files generator (NO file copy).

Generates train.txt and val.txt files with image paths and class labels,
similar to YOLO's text-file-based split strategy.

DICE format (classification with bounding boxes):
    dataset/
    ├── images/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── annotations/
        └── instances_default.json

Generated output (in dataset root):
    dataset/
    ├── images/              # Original images (not copied)
    ├── annotations/
    └── splits/
        ├── train.txt        # image_path class_id
        ├── val.txt          # image_path class_id
        └── classes.txt      # class names (one per line)

Note: This approach saves disk space and is identical to YOLO's split strategy.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict


class DiceSplitGenerator:
    """
    Generate train/val split files from DICE format annotations.

    Output format (similar to YOLO):
        train.txt:
            images/img1.jpg 0
            images/img3.jpg 1
            images/img5.jpg 0

        val.txt:
            images/img2.jpg 0
            images/img4.jpg 1

        classes.txt:
            cat
            dog
    """

    def __init__(
        self,
        dice_root: str,
        split_ratio: float = 0.8,
        split_strategy: str = "stratified",
        seed: int = 42
    ):
        """
        Args:
            dice_root: Path to DICE format dataset root
            split_ratio: Train/val split ratio (default: 0.8)
            split_strategy: 'stratified', 'random', 'sequential'
            seed: Random seed for reproducibility
        """
        self.dice_root = Path(dice_root)
        self.split_ratio = split_ratio
        self.split_strategy = split_strategy
        self.seed = seed

        # Set random seed
        random.seed(seed)

        # Paths
        self.images_dir = self.dice_root / "images"

        # Support multiple annotation file locations
        # 1. annotations/instances_default.json (standard DICE format)
        # 2. annotations.json (root level, common from R2 downloads)
        annotation_candidates = [
            self.dice_root / "annotations" / "instances_default.json",
            self.dice_root / "annotations.json",
        ]

        self.annotations_file = None
        for candidate in annotation_candidates:
            if candidate.exists():
                self.annotations_file = candidate
                break

        self.splits_dir = self.dice_root / "splits"

        # Stats
        self.class_id_to_name = {}  # {category_id: category_name}
        self.class_name_to_id = {}  # {category_name: class_id (0-indexed)}
        self.image_to_class = {}    # {image_filename: class_name}
        self.class_distribution = Counter()

    def load_annotations(self) -> Dict:
        """Load COCO format annotations."""
        if self.annotations_file is None or not self.annotations_file.exists():
            raise FileNotFoundError(
                f"Annotations file not found in dataset: {self.dice_root}\n"
                f"Expected one of:\n"
                f"  - annotations/instances_default.json (standard DICE format)\n"
                f"  - annotations.json (root level)"
            )

        print(f"[DiceSplit] Loading annotations from: {self.annotations_file}")
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_image_classes(self, annotations: Dict) -> None:
        """
        Extract primary class for each image.

        Strategy: If image has multiple objects, use most common class.
        """
        # Build category mapping (COCO category_id -> class_name)
        for category in annotations['categories']:
            self.class_id_to_name[category['id']] = category['name']

        # Build 0-indexed class mapping (class_name -> class_id)
        sorted_class_names = sorted(self.class_id_to_name.values())
        for idx, class_name in enumerate(sorted_class_names):
            self.class_name_to_id[class_name] = idx

        # Build image ID to filename mapping
        image_id_to_filename = {}
        for img in annotations['images']:
            image_id_to_filename[img['id']] = img['file_name']

        # Count objects per class for each image
        image_class_counts = defaultdict(Counter)
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            class_name = self.class_id_to_name[category_id]
            image_class_counts[image_id][class_name] += 1

        # Assign primary class to each image
        for image_id, class_counter in image_class_counts.items():
            most_common_class = class_counter.most_common(1)[0][0]
            filename = image_id_to_filename.get(image_id)
            if filename:
                self.image_to_class[filename] = most_common_class
                self.class_distribution[most_common_class] += 1

        print(f"\n[DiceSplit] Extracted classes for {len(self.image_to_class)} images")
        print(f"[DiceSplit] Found {len(self.class_name_to_id)} classes:")
        for class_name, class_id in sorted(self.class_name_to_id.items(), key=lambda x: x[1]):
            count = self.class_distribution[class_name]
            percentage = (count / len(self.image_to_class)) * 100
            print(f"  [{class_id}] {class_name}: {count} images ({percentage:.1f}%)")

    def split_dataset(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Split images into train and val sets.

        Returns:
            (train_list, val_list) where each list contains (filename, class_name) tuples
        """
        # Group images by class
        class_to_images = defaultdict(list)
        for filename, class_name in self.image_to_class.items():
            class_to_images[class_name].append(filename)

        train_list = []
        val_list = []

        if self.split_strategy == "stratified":
            # Stratified split: maintain class distribution
            print(f"\n[DiceSplit] Stratified split per class:")
            for class_name, filenames in sorted(class_to_images.items()):
                shuffled = filenames.copy()
                random.shuffle(shuffled)

                split_idx = int(len(shuffled) * self.split_ratio)
                train_files = shuffled[:split_idx]
                val_files = shuffled[split_idx:]

                train_list.extend([(f, class_name) for f in train_files])
                val_list.extend([(f, class_name) for f in val_files])

                print(f"  [{class_name}] train: {len(train_files)}, val: {len(val_files)}")

        elif self.split_strategy == "random":
            # Random split
            all_items = list(self.image_to_class.items())
            random.shuffle(all_items)

            split_idx = int(len(all_items) * self.split_ratio)
            train_list = all_items[:split_idx]
            val_list = all_items[split_idx:]

        elif self.split_strategy == "sequential":
            # Sequential split
            all_items = list(self.image_to_class.items())
            split_idx = int(len(all_items) * self.split_ratio)
            train_list = all_items[:split_idx]
            val_list = all_items[split_idx:]

        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

        print(f"\n[DiceSplit] Split strategy: {self.split_strategy}")
        print(f"[DiceSplit] Train: {len(train_list)} images")
        print(f"[DiceSplit] Val: {len(val_list)} images")
        print(f"[DiceSplit] Ratio: {len(train_list)/(len(train_list)+len(val_list)):.3f}")

        return train_list, val_list

    def write_split_files(
        self,
        train_list: List[Tuple[str, str]],
        val_list: List[Tuple[str, str]]
    ) -> None:
        """
        Write train.txt, val.txt, and classes.txt files.

        Format:
            train.txt: image_path class_id (relative to dataset root)
            classes.txt: class_name (one per line, sorted by class_id)
        """
        # Create splits directory
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # Write train.txt
        train_txt = self.splits_dir / "train.txt"
        with open(train_txt, 'w', encoding='utf-8') as f:
            for filename, class_name in train_list:
                class_id = self.class_name_to_id[class_name]
                # Use relative path from dataset root
                rel_path = f"images/{filename}"
                f.write(f"{rel_path} {class_id}\n")

        print(f"[DiceSplit] ✓ Created: {train_txt} ({len(train_list)} images)")

        # Write val.txt
        val_txt = self.splits_dir / "val.txt"
        with open(val_txt, 'w', encoding='utf-8') as f:
            for filename, class_name in val_list:
                class_id = self.class_name_to_id[class_name]
                rel_path = f"images/{filename}"
                f.write(f"{rel_path} {class_id}\n")

        print(f"[DiceSplit] ✓ Created: {val_txt} ({len(val_list)} images)")

        # Write classes.txt (sorted by class_id)
        classes_txt = self.splits_dir / "classes.txt"
        sorted_classes = sorted(self.class_name_to_id.items(), key=lambda x: x[1])
        with open(classes_txt, 'w', encoding='utf-8') as f:
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")

        print(f"[DiceSplit] ✓ Created: {classes_txt} ({len(sorted_classes)} classes)")

    def save_metadata(self, train_list: List[Tuple[str, str]], val_list: List[Tuple[str, str]]) -> None:
        """Save split metadata."""
        metadata = {
            "dice_root": str(self.dice_root),
            "split_ratio": self.split_ratio,
            "split_strategy": self.split_strategy,
            "seed": self.seed,
            "num_classes": len(self.class_name_to_id),
            "class_names": [name for name, _ in sorted(self.class_name_to_id.items(), key=lambda x: x[1])],
            "class_distribution": dict(self.class_distribution),
            "train_size": len(train_list),
            "val_size": len(val_list),
        }

        metadata_path = self.splits_dir / "split_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[DiceSplit] ✓ Metadata: {metadata_path}")

    def generate(self) -> Path:
        """
        Generate split files (NO file copy).

        Returns:
            Path to splits directory
        """
        print(f"\n{'='*80}")
        print(f"DICE Split Generator (No File Copy)")
        print(f"{'='*80}")
        print(f"Dataset: {self.dice_root}")
        print(f"Output:  {self.splits_dir}")
        print(f"{'='*80}\n")

        # Load and parse annotations
        print("[1/4] Loading DICE annotations...")
        annotations = self.load_annotations()
        self.extract_image_classes(annotations)

        # Split dataset
        print(f"\n[2/4] Splitting dataset...")
        train_list, val_list = self.split_dataset()

        # Write split files
        print(f"\n[3/4] Writing split files...")
        self.write_split_files(train_list, val_list)

        # Save metadata
        print(f"\n[4/4] Saving metadata...")
        self.save_metadata(train_list, val_list)

        print(f"\n{'='*80}")
        print(f"✓ Split generation complete!")
        print(f"{'='*80}")
        print(f"Train: {len(train_list)} images")
        print(f"Val:   {len(val_list)} images")
        print(f"Files: {self.splits_dir}/{{train,val,classes}}.txt")
        print(f"{'='*80}\n")

        return self.splits_dir


def generate_dice_split(
    dice_root: str,
    split_ratio: float = 0.8,
    split_strategy: str = "stratified",
    seed: int = 42
) -> str:
    """
    Generate train/val split files from DICE dataset (no file copy).

    Args:
        dice_root: Path to DICE format dataset
        split_ratio: Train/val split ratio
        split_strategy: 'stratified', 'random', 'sequential'
        seed: Random seed

    Returns:
        Path to splits directory

    Example:
        >>> splits_dir = generate_dice_split(
        ...     dice_root="/path/to/dice/dataset",
        ...     split_ratio=0.8,
        ...     split_strategy="stratified"
        ... )
        >>> print(f"Splits created at: {splits_dir}")
    """
    generator = DiceSplitGenerator(
        dice_root=dice_root,
        split_ratio=split_ratio,
        split_strategy=split_strategy,
        seed=seed
    )

    splits_dir = generator.generate()
    return str(splits_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate train/val split files from DICE dataset (no file copy)"
    )
    parser.add_argument(
        "--dice-root",
        required=True,
        help="Path to DICE format dataset root"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--split-strategy",
        choices=["stratified", "random", "sequential"],
        default="stratified",
        help="Split strategy (default: stratified)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    generate_dice_split(
        dice_root=args.dice_root,
        split_ratio=args.split_ratio,
        split_strategy=args.split_strategy,
        seed=args.seed
    )
