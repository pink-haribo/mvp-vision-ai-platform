"""
Dataset analyzer utility for automatic format detection and statistics collection.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging
from collections import Counter
import base64

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyzes dataset structure, format, and statistics"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    def detect_format(self, hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect dataset format automatically or validate hint.

        Returns:
            dict with 'format' and 'confidence'
        """
        if hint:
            # Validate hint
            if hint == 'imagefolder' and self._is_imagefolder():
                return {'format': 'imagefolder', 'confidence': 1.0}
            elif hint == 'yolo' and self._is_yolo():
                return {'format': 'yolo', 'confidence': 1.0}
            elif hint == 'coco' and self._is_coco():
                return {'format': 'coco', 'confidence': 1.0}
            else:
                logger.warning(f"Hint '{hint}' doesn't match structure, auto-detecting")

        # Auto-detect
        if self._is_imagefolder():
            return {'format': 'imagefolder', 'confidence': 0.95, 'task_type': 'image_classification'}
        elif self._is_yolo():
            return {'format': 'yolo', 'confidence': 0.90, 'task_type': 'object_detection'}
        elif self._is_coco():
            return {'format': 'coco', 'confidence': 0.90, 'task_type': 'object_detection'}
        else:
            return {'format': 'unknown', 'confidence': 0.0, 'task_type': None}

    def _is_imagefolder(self) -> bool:
        """
        Check if dataset follows ImageFolder structure.
        Structure: dataset/class1/img1.jpg or dataset/train/class1/img1.jpg
        """
        try:
            # Check for subdirectories
            subdirs = [d for d in self.path.iterdir() if d.is_dir()]

            # Need at least 2 subdirectories
            if len(subdirs) < 2:
                return False

            # Check if this is a split structure (train/val/test)
            split_names = {'train', 'val', 'test', 'training', 'validation', 'testing'}
            has_splits = any(d.name.lower() in split_names for d in subdirs)

            if has_splits:
                # For split structures, check inside split directories
                logger.info(f"[SPLIT DETECTION] Found split structure in {self.path}")
                found_images = False
                for subdir in subdirs[:3]:  # Sample first 3
                    if not subdir.is_dir():
                        continue

                    # Check for class directories inside splits
                    class_dirs = [d for d in subdir.iterdir() if d.is_dir()]
                    logger.info(f"[SPLIT DETECTION] {subdir.name}: {len(class_dirs)} class dirs")
                    if len(class_dirs) < 1:
                        continue  # This split might be empty, check others

                    # Check if class directories contain images
                    for class_dir in class_dirs[:2]:  # Sample 2 classes
                        images = self._find_images(class_dir, recursive=False)
                        logger.info(f"[SPLIT DETECTION] {class_dir.name}: {len(images)} images")
                        if len(images) > 0:
                            found_images = True
                            break

                    if found_images:
                        break

                logger.info(f"[SPLIT DETECTION] Found images: {found_images}")
                return found_images  # Return True if we found images in any split
            else:
                # For flat structures, check directly in subdirectories
                for subdir in subdirs[:3]:  # Sample first 3
                    images = self._find_images(subdir, recursive=False)
                    if len(images) == 0:
                        return False

                return True

        except Exception as e:
            logger.error(f"Error checking ImageFolder format: {e}")
            return False

    def _is_yolo(self) -> bool:
        """
        Check if dataset follows YOLO structure.
        Structure: images/*.jpg + labels/*.txt (supports subdirectories like train/val)
        """
        try:
            images_dir = self.path / 'images'
            labels_dir = self.path / 'labels'

            if not (images_dir.exists() and labels_dir.exists()):
                return False

            # Check for matching image and label files (recursive to support train/val structure)
            images = self._find_images(images_dir)
            labels = list(labels_dir.rglob('*.txt'))  # Changed to rglob for recursive search

            if len(images) == 0 or len(labels) == 0:
                return False

            # Validate YOLO format (sample one file)
            if labels:
                try:
                    with open(labels[0], 'r') as f:
                        line = f.readline().strip()
                        if line:
                            parts = line.split()
                            # YOLO format: <class> <x> <y> <w> <h>
                            if len(parts) >= 5 and all(self._is_number(p) for p in parts):
                                return True
                except:
                    pass

            return False

        except Exception as e:
            logger.error(f"Error checking YOLO format: {e}")
            return False

    def _is_coco(self) -> bool:
        """
        Check if dataset follows COCO structure.
        Structure: annotations/*.json + images/
        """
        try:
            anno_dir = self.path / 'annotations'
            img_dir = self.path / 'images'

            if not (anno_dir.exists() and img_dir.exists()):
                return False

            # Check for JSON annotation files
            json_files = list(anno_dir.glob('*.json'))
            if len(json_files) == 0:
                return False

            # Validate COCO JSON structure
            try:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    required_keys = ['images', 'annotations', 'categories']
                    if all(key in data for key in required_keys):
                        return True
            except:
                pass

            return False

        except Exception as e:
            logger.error(f"Error checking COCO format: {e}")
            return False


    def _encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Encode image file to base64 string for preview"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                # Limit size to 100KB for preview
                if len(image_data) > 100 * 1024:
                    return None
                encoded = base64.b64encode(image_data).decode('utf-8')
                # Detect image type from extension
                ext = image_path.suffix.lower()
                mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else f'image/{ext[1:]}'
                return f'data:{mime_type};base64,{encoded}'
        except Exception as e:
            logger.error(f'Error encoding image {image_path}: {e}')
            return None

    def collect_statistics(self, format_type: str) -> Dict[str, Any]:
        """
        Collect dataset statistics based on detected format.

        Returns comprehensive statistics including:
        - structure info (classes, samples)
        - file statistics (sizes, formats)
        - image statistics (resolutions)
        """
        if format_type == 'imagefolder':
            return self._collect_imagefolder_stats()
        elif format_type == 'yolo':
            return self._collect_yolo_stats()
        elif format_type == 'coco':
            return self._collect_coco_stats()
        else:
            return {}

    def _collect_imagefolder_stats(self) -> Dict[str, Any]:
        """Collect statistics for ImageFolder format"""
        try:
            subdirs = [d for d in self.path.iterdir() if d.is_dir()]

            # Detect train/val/test splits
            split_names = {'train', 'val', 'test', 'training', 'validation', 'testing'}
            has_splits = any(d.name.lower() in split_names for d in subdirs)

            samples_per_class = {}
            total_samples = 0
            all_images = []
            class_names = []

            if has_splits:
                # For split structures, collect from class directories inside splits
                # Use train split to get class names (or first available split)
                train_dir = None
                for d in subdirs:
                    if d.name.lower() in {'train', 'training'}:
                        train_dir = d
                        break

                # If no train, use first split directory
                if train_dir is None:
                    train_dir = subdirs[0]

                # Get class names from train directory
                class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
                class_names = [d.name for d in class_dirs]

                # Collect samples from each class across all splits
                for class_dir in class_dirs:
                    class_name = class_dir.name
                    class_total = 0

                    # Count samples across all splits for this class
                    for split_dir in subdirs:
                        if not split_dir.is_dir():
                            continue
                        class_in_split = split_dir / class_name
                        if class_in_split.exists() and class_in_split.is_dir():
                            images = self._find_images(class_in_split, recursive=False)
                            class_total += len(images)
                            all_images.extend(images[:2])  # Sample 2 per class

                    samples_per_class[class_name] = class_total
                    total_samples += class_total
            else:
                # For flat structures, collect directly
                class_names = [d.name for d in subdirs]

                for subdir in subdirs:
                    images = self._find_images(subdir, recursive=False)
                    samples_per_class[subdir.name] = len(images)
                    total_samples += len(images)
                    all_images.extend(images[:2])  # Sample 2 per class for preview

            # Calculate file statistics
            total_size_bytes = sum(img.stat().st_size for img in all_images if img.exists())
            total_size_mb = total_size_bytes / (1024 * 1024)

            # Image format distribution
            format_counter = Counter(img.suffix.lower() for img in all_images)

            # Detect train/val/test splits (for output)
            has_train = any('train' in d.name.lower() for d in self.path.iterdir() if d.is_dir())
            has_val = any('val' in d.name.lower() for d in self.path.iterdir() if d.is_dir())
            has_test = any('test' in d.name.lower() for d in self.path.iterdir() if d.is_dir())

            return {
                'structure': {
                    'num_classes': len(class_names),
                    'class_names': class_names[:20],  # Limit to 20 for display
                    'num_samples': total_samples,
                    'has_train_split': has_train,
                    'has_val_split': has_val,
                    'has_test_split': has_test
                },
                'statistics': {
                    'total_size_mb': round(total_size_mb, 2),
                    'avg_file_size_kb': round((total_size_bytes / total_samples / 1024), 2) if total_samples > 0 else 0,
                    'image_formats': dict(format_counter)
                },
                'samples_per_class': samples_per_class,
                'preview_images': [
                    {
                        'class': img.parent.name,
                        'path': str(img.relative_to(self.path)),
                        'thumbnail': self._encode_image_to_base64(self.path / img.relative_to(self.path))
                    }
                    for img in all_images[:5]  # Limit to 5 for performance
                ]
            }

        except Exception as e:
            logger.error(f"Error collecting ImageFolder stats: {e}")
            return {}

    def _collect_yolo_stats(self) -> Dict[str, Any]:
        """Collect statistics for YOLO format"""
        try:
            images_dir = self.path / 'images'
            labels_dir = self.path / 'labels'

            images = self._find_images(images_dir)
            labels = list(labels_dir.rglob('*.txt'))  # Changed to rglob for recursive search

            # Parse class IDs from labels
            class_ids = set()
            for label_file in labels[:100]:  # Sample 100 files
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_ids.add(int(parts[0]))
                except:
                    continue

            # Check for train/val splits in images and labels directories
            has_train_split = (images_dir / 'train').exists() or (labels_dir / 'train').exists()
            has_val_split = (images_dir / 'val').exists() or (labels_dir / 'val').exists()
            has_test_split = (images_dir / 'test').exists() or (labels_dir / 'test').exists()

            total_size_bytes = sum(img.stat().st_size for img in images[:100] if img.exists())
            total_size_mb = total_size_bytes / (1024 * 1024)

            return {
                'structure': {
                    'num_classes': len(class_ids),
                    'class_names': [f"class_{i}" for i in sorted(class_ids)],
                    'num_samples': len(images),
                    'has_train_split': has_train_split,
                    'has_val_split': has_val_split,
                    'has_test_split': has_test_split
                },
                'statistics': {
                    'total_size_mb': round(total_size_mb, 2),
                    'avg_file_size_kb': round((total_size_bytes / len(images[:100]) / 1024), 2) if len(images) > 0 else 0,
                    'image_formats': {'jpg': len(images)}  # Simplified
                },
                'samples_per_class': {},  # Would need full parsing
                'preview_images': []
            }

        except Exception as e:
            logger.error(f"Error collecting YOLO stats: {e}")
            return {}

    def _collect_coco_stats(self) -> Dict[str, Any]:
        """Collect statistics for COCO format"""
        try:
            anno_dir = self.path / 'annotations'
            img_dir = self.path / 'images'

            json_files = list(anno_dir.glob('*.json'))
            if not json_files:
                return {}

            with open(json_files[0], 'r') as f:
                data = json.load(f)

            categories = data.get('categories', [])
            images = data.get('images', [])

            return {
                'structure': {
                    'num_classes': len(categories),
                    'class_names': [cat['name'] for cat in categories[:20]],
                    'num_samples': len(images),
                    'has_train_split': 'train' in json_files[0].name.lower(),
                    'has_val_split': any('val' in f.name.lower() for f in json_files),
                    'has_test_split': any('test' in f.name.lower() for f in json_files)
                },
                'statistics': {
                    'total_size_mb': 0,  # Would need to scan images
                    'avg_file_size_kb': 0,
                    'image_formats': {'jpg': len(images)}
                },
                'samples_per_class': {},
                'preview_images': []
            }

        except Exception as e:
            logger.error(f"Error collecting COCO stats: {e}")
            return {}

    def check_quality(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform quality checks on the dataset.

        Returns:
            dict with 'corrupted_files', 'warnings', etc.
        """
        warnings = []
        structure = stats.get('structure', {})
        samples_per_class = stats.get('samples_per_class', {})

        # Check class imbalance
        if samples_per_class:
            counts = list(samples_per_class.values())
            if counts:
                max_count = max(counts)
                min_count = min(counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

                if imbalance_ratio > 2.0:
                    warnings.append(f"클래스 불균형 감지 (비율: {imbalance_ratio:.2f})")

                # Find specific imbalanced classes
                avg_count = sum(counts) / len(counts)
                for class_name, count in samples_per_class.items():
                    if count < avg_count * 0.7:
                        warnings.append(f"'{class_name}' 클래스가 평균보다 30% 적습니다 ({count}개)")

        # Check minimum samples
        num_samples = structure.get('num_samples', 0)
        if num_samples < 100:
            warnings.append(f"샘플 수가 너무 적습니다 ({num_samples}개). 최소 100개 이상 권장")

        # Check train/val split
        if not structure.get('has_train_split') and not structure.get('has_val_split'):
            warnings.append("학습/검증 데이터 분할이 없습니다. 수동으로 분할해야 할 수 있습니다")

        return {
            'corrupted_files': [],  # Would need full file scan
            'duplicate_images': [],  # Would need hash comparison
            'class_imbalance_ratio': imbalance_ratio if samples_per_class else 1.0,
            'warnings': warnings
        }

    def _find_images(self, directory: Path, recursive: bool = True) -> List[Path]:
        """Find all image files in a directory"""
        images = []
        pattern = '**/*' if recursive else '*'

        for ext in self.image_extensions:
            images.extend(directory.glob(f"{pattern}{ext}"))

        return images

    @staticmethod
    def _is_number(s: str) -> bool:
        """Check if string is a number"""
        try:
            float(s)
            return True
        except ValueError:
            return False


# Convenience function for tool_registry
def analyze_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Analyze a dataset and return comprehensive statistics

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        dict: Analysis results including format, classes, statistics, and suggestions
    """
    analyzer = DatasetAnalyzer(dataset_path)

    # Detect format
    format_info = analyzer.detect_format()
    format_type = format_info.get("format", "unknown")

    # Collect statistics
    stats = analyzer.collect_statistics(format_type)

    # Check quality
    quality = analyzer.check_quality(stats)

    # Combine all results
    # Extract from nested structure
    structure = stats.get("structure", {})

    return {
        "format": format_type,
        "classes": structure.get("class_names", []),
        "total_samples": structure.get("num_samples", 0),
        "num_classes": structure.get("num_classes", 0),
        "class_distribution": stats.get("samples_per_class", {}),
        "dataset_info": format_info,
        "statistics": stats.get("statistics", {}),
        "quality": quality,
        "suggestions": quality.get("suggestions", []),
        "has_train_split": structure.get("has_train_split", False),
        "has_val_split": structure.get("has_val_split", False),
        "has_test_split": structure.get("has_test_split", False),
    }
