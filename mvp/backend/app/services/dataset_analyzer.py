"""Dataset analyzer service for automatic dataset inspection."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


class DatasetAnalyzer:
    """Analyze dataset structure and extract metadata automatically."""

    def __init__(self, base_path: str = "./mvp"):
        """Initialize dataset analyzer.

        Args:
            base_path: Base path for resolving relative dataset paths
        """
        self.base_path = Path(base_path).resolve()

    def analyze(self, dataset_path: str) -> Dict:
        """
        Analyze dataset and return comprehensive metadata.

        Args:
            dataset_path: Path to dataset (can be relative or absolute)

        Returns:
            Dictionary containing:
            - format: Dataset format (ImageFolder, COCO, YOLO, Unlabeled, Unknown)
            - num_classes: Number of classes detected
            - class_names: List of class names
            - is_labeled: Whether dataset has labels
            - total_images: Total number of images
            - has_train_val_split: Whether train/val split exists
            - image_counts: Dict of image counts per split/class
            - warnings: List of potential issues
        """
        try:
            # Resolve path (handle relative paths)
            if not Path(dataset_path).is_absolute():
                full_path = self.base_path / dataset_path
            else:
                full_path = Path(dataset_path)

            full_path = full_path.resolve()

            logger.info(f"Analyzing dataset at: {full_path}")

            if not full_path.exists():
                return {
                    "format": "Unknown",
                    "error": f"Dataset path does not exist: {full_path}",
                    "exists": False,
                }

            if not full_path.is_dir():
                return {
                    "format": "Unknown",
                    "error": "Dataset path is not a directory",
                    "exists": True,
                }

            # Try to detect format
            result = self._detect_format(full_path)
            result["exists"] = True
            result["resolved_path"] = str(full_path)

            logger.info(f"Dataset analysis complete: {result['format']} format detected")
            return result

        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}", exc_info=True)
            return {
                "format": "Unknown",
                "error": f"Analysis failed: {str(e)}",
                "exists": False,
            }

    def _detect_format(self, path: Path) -> Dict:
        """Detect dataset format and extract metadata."""

        # Check for ImageFolder format (most common for classification)
        if self._is_imagefolder(path):
            return self._analyze_imagefolder(path)

        # Check for COCO format
        if self._is_coco(path):
            return self._analyze_coco(path)

        # Check for YOLO format
        if self._is_yolo(path):
            return self._analyze_yolo(path)

        # Check if it's just unlabeled images
        if self._has_images(path):
            return self._analyze_unlabeled(path)

        # Unknown format
        return {
            "format": "Unknown",
            "num_classes": 0,
            "class_names": [],
            "is_labeled": False,
            "total_images": 0,
            "has_train_val_split": False,
            "warnings": ["Could not detect dataset format"],
        }

    def _is_imagefolder(self, path: Path) -> bool:
        """Check if dataset follows ImageFolder structure."""
        # ImageFolder: dataset/train/class1/, dataset/train/class2/, ...
        #              dataset/val/class1/, dataset/val/class2/, ...
        # OR:          dataset/class1/, dataset/class2/, ...

        # Check for train/val split
        train_dir = path / "train"
        if train_dir.exists() and train_dir.is_dir():
            subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
            if subdirs and any(self._has_images(d) for d in subdirs):
                return True

        # Check for direct class folders
        subdirs = [d for d in path.iterdir() if d.is_dir() and d.name not in {"train", "val", "test"}]
        if subdirs and any(self._has_images(d) for d in subdirs):
            return True

        return False

    def _is_coco(self, path: Path) -> bool:
        """Check if dataset follows COCO format."""
        # COCO: annotations/instances_train.json, annotations/instances_val.json
        annotations_dir = path / "annotations"
        if annotations_dir.exists():
            json_files = list(annotations_dir.glob("*.json"))
            return len(json_files) > 0
        return False

    def _is_yolo(self, path: Path) -> bool:
        """Check if dataset follows YOLO format."""
        # YOLO: data.yaml file is the most reliable indicator
        data_yaml = path / "data.yaml"
        if data_yaml.exists():
            return True

        # Backup check: images/ and labels/ directories with .txt files
        images_dir = path / "images"
        labels_dir = path / "labels"

        if images_dir.exists() and labels_dir.exists():
            has_txt_labels = any(labels_dir.glob("*.txt"))
            return has_txt_labels
        return False

    def _has_images(self, path: Path) -> bool:
        """Check if directory contains any images."""
        if not path.is_dir():
            return False

        for file in path.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                return True
        return False

    def _count_images(self, path: Path) -> int:
        """Count total images in a directory (recursively)."""
        count = 0
        for file in path.rglob("*"):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                count += 1
        return count

    def _analyze_imagefolder(self, path: Path) -> Dict:
        """Analyze ImageFolder format dataset."""
        warnings = []
        has_split = False
        image_counts = {}
        class_names = set()

        # Check for train/val split
        train_dir = path / "train"
        val_dir = path / "val"
        test_dir = path / "test"

        if train_dir.exists() and train_dir.is_dir():
            has_split = True

            # Extract classes from train directory
            for class_dir in train_dir.iterdir():
                if class_dir.is_dir() and self._has_images(class_dir):
                    class_names.add(class_dir.name)
                    count = self._count_images(class_dir)
                    image_counts[f"train/{class_dir.name}"] = count

            # Check val directory
            if val_dir.exists() and val_dir.is_dir():
                for class_dir in val_dir.iterdir():
                    if class_dir.is_dir() and self._has_images(class_dir):
                        if class_dir.name not in class_names:
                            warnings.append(
                                f"Class '{class_dir.name}' exists in val but not in train"
                            )
                        count = self._count_images(class_dir)
                        image_counts[f"val/{class_dir.name}"] = count
            else:
                warnings.append("No validation split found - recommend creating train/val split")

            # Check test directory
            if test_dir.exists() and test_dir.is_dir():
                for class_dir in test_dir.iterdir():
                    if class_dir.is_dir() and self._has_images(class_dir):
                        count = self._count_images(class_dir)
                        image_counts[f"test/{class_dir.name}"] = count
        else:
            # Direct class folders (no train/val split)
            for class_dir in path.iterdir():
                if class_dir.is_dir() and self._has_images(class_dir):
                    if class_dir.name not in {"train", "val", "test"}:
                        class_names.add(class_dir.name)
                        count = self._count_images(class_dir)
                        image_counts[class_dir.name] = count

            warnings.append("No train/val split detected - recommend splitting dataset")

        class_names_list = sorted(list(class_names))
        total_images = sum(image_counts.values())

        # Additional validation
        if len(class_names_list) < 2:
            warnings.append(
                f"Only {len(class_names_list)} class detected - need at least 2 for classification"
            )

        # Check for imbalanced dataset
        if image_counts:
            counts = list(image_counts.values())
            max_count = max(counts)
            min_count = min(counts)
            if max_count > min_count * 3:
                warnings.append(
                    f"Dataset is imbalanced - largest class has {max_count} images, "
                    f"smallest has {min_count} images"
                )

        return {
            "format": "imagefolder",  # lowercase for consistency
            "num_classes": len(class_names_list),
            "class_names": class_names_list,
            "is_labeled": True,
            "total_images": total_images,
            "has_train_val_split": has_split,
            "image_counts": image_counts,
            "warnings": warnings,
        }

    def _analyze_coco(self, path: Path) -> Dict:
        """Analyze COCO format dataset."""
        warnings = ["COCO format detected but full analysis not yet implemented"]

        annotations_dir = path / "annotations"
        json_files = list(annotations_dir.glob("*.json"))

        # Try to extract basic info from first annotation file
        class_names = []
        total_images = 0

        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    data = json.load(f)
                    if "categories" in data:
                        class_names = [cat["name"] for cat in data["categories"]]
                    if "images" in data:
                        total_images = len(data["images"])
            except Exception as e:
                warnings.append(f"Could not parse COCO annotation file: {str(e)}")

        return {
            "format": "coco",  # lowercase for consistency
            "num_classes": len(class_names),
            "class_names": class_names,
            "is_labeled": True,
            "total_images": total_images,
            "has_train_val_split": len(json_files) > 1,
            "warnings": warnings,
        }

    def _analyze_yolo(self, path: Path) -> Dict:
        """Analyze YOLO format dataset."""
        warnings = []
        class_names = []
        has_split = False
        total_images = 0
        image_counts = {}

        # YOLO format requires data.yaml file for class names
        data_yaml = path / "data.yaml"

        if data_yaml.exists():
            try:
                import yaml
                with open(data_yaml, "r") as f:
                    data = yaml.safe_load(f)

                    # Extract class names
                    if "names" in data:
                        if isinstance(data["names"], dict):
                            # Format: names: {0: 'class1', 1: 'class2'}
                            class_names = [data["names"][i] for i in sorted(data["names"].keys())]
                        elif isinstance(data["names"], list):
                            # Format: names: ['class1', 'class2']
                            class_names = data["names"]

                    # Check for train/val paths
                    if "train" in data or "val" in data:
                        has_split = True

            except Exception as e:
                warnings.append(f"Could not parse data.yaml: {str(e)}")
        else:
            warnings.append("No data.yaml found - YOLO format requires this file")

        # Count images in different splits
        images_dir = path / "images"
        if images_dir.exists():
            # Check for train/val subdirectories
            train_images_dir = images_dir / "train"
            val_images_dir = images_dir / "val"
            test_images_dir = images_dir / "test"

            if train_images_dir.exists():
                has_split = True
                train_count = self._count_images(train_images_dir)
                image_counts["train"] = train_count
                total_images += train_count

            if val_images_dir.exists():
                val_count = self._count_images(val_images_dir)
                image_counts["val"] = val_count
                total_images += val_count

            if test_images_dir.exists():
                test_count = self._count_images(test_images_dir)
                image_counts["test"] = test_count
                total_images += test_count

            # If no split subdirectories, count all images
            if not has_split:
                total_images = self._count_images(images_dir)
        else:
            warnings.append("No images/ directory found")

        if not class_names:
            warnings.append("Could not determine class names - check data.yaml format")

        return {
            "format": "yolo",  # lowercase for consistency
            "num_classes": len(class_names),
            "class_names": class_names,
            "is_labeled": True,
            "total_images": total_images,
            "has_train_val_split": has_split,
            "image_counts": image_counts,
            "warnings": warnings,
        }

    def _analyze_unlabeled(self, path: Path) -> Dict:
        """Analyze unlabeled image directory."""
        total_images = self._count_images(path)

        return {
            "format": "Unlabeled",
            "num_classes": 0,
            "class_names": [],
            "is_labeled": False,
            "total_images": total_images,
            "has_train_val_split": False,
            "warnings": [
                "Dataset appears to be unlabeled - cannot use for supervised learning",
                "Consider organizing images into class folders for classification",
            ],
        }


# Global instance
dataset_analyzer = DatasetAnalyzer()
