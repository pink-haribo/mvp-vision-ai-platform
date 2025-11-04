"""Ultralytics YOLO adapter for object detection, segmentation, and pose estimation."""

import os
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from platform_sdk import TrainingAdapter, MetricsResult, TaskType, DatasetFormat, ConfigSchema, ConfigField


class UltralyticsAdapter(TrainingAdapter):
    """
    Adapter for Ultralytics YOLO models.

    Supported tasks:
    - Object Detection (yolov8n.pt, yolov9e.pt)
    - Instance Segmentation (yolov8n-seg.pt)
    - Pose Estimation (yolov8n-pose.pt)
    - Classification (yolov8n-cls.pt)
    - OBB (yolov8n-obb.pt)
    - Zero-shot Detection (YOLO-World with custom text prompts)
    """

    TASK_SUFFIX_MAP = {
        TaskType.OBJECT_DETECTION: "",
        TaskType.INSTANCE_SEGMENTATION: "-seg",
        TaskType.POSE_ESTIMATION: "-pose",
        TaskType.IMAGE_CLASSIFICATION: "-cls",
        TaskType.ZERO_SHOT_DETECTION: "",  # YOLO-World uses base model
    }

    @classmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Return configuration schema for YOLO models."""
        from training.config_schemas import get_ultralytics_schema
        return get_ultralytics_schema()

    @classmethod
    def _get_config_schema_inline(cls) -> ConfigSchema:
        """Return configuration schema for YOLO models (object detection, segmentation, pose)."""
        fields = [
            # ========== Optimizer Settings ==========
            ConfigField(
                name="optimizer_type",
                type="select",
                default="Adam",
                options=["Adam", "AdamW", "SGD", "RMSprop"],
                description="Optimizer algorithm",
                group="optimizer",
                required=False
            ),
            ConfigField(
                name="weight_decay",
                type="float",
                default=0.0005,
                min=0.0,
                max=0.01,
                step=0.0001,
                description="Weight decay (L2 regularization)",
                group="optimizer",
                advanced=True
            ),
            ConfigField(
                name="momentum",
                type="float",
                default=0.937,
                min=0.0,
                max=1.0,
                step=0.001,
                description="Momentum for SGD",
                group="optimizer",
                advanced=True
            ),

            # ========== Scheduler Settings ==========
            ConfigField(
                name="cos_lr",
                type="bool",
                default=True,
                description="Use cosine learning rate scheduler",
                group="scheduler",
                required=False
            ),
            ConfigField(
                name="lrf",
                type="float",
                default=0.01,
                min=0.0,
                max=1.0,
                step=0.01,
                description="Final learning rate (lr0 * lrf)",
                group="scheduler",
                advanced=False
            ),
            ConfigField(
                name="warmup_epochs",
                type="int",
                default=3,
                min=0,
                max=20,
                step=1,
                description="Number of warmup epochs",
                group="scheduler",
                advanced=False
            ),
            ConfigField(
                name="warmup_momentum",
                type="float",
                default=0.8,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Warmup initial momentum",
                group="scheduler",
                advanced=True
            ),
            ConfigField(
                name="warmup_bias_lr",
                type="float",
                default=0.1,
                min=0.0,
                max=1.0,
                step=0.01,
                description="Warmup initial bias learning rate",
                group="scheduler",
                advanced=True
            ),

            # ========== Augmentation Settings (YOLO-specific) ==========
            ConfigField(
                name="aug_enabled",
                type="bool",
                default=True,
                description="Enable data augmentation",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="mosaic",
                type="float",
                default=1.0,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Mosaic augmentation probability (4-image blend)",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="mixup",
                type="float",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.1,
                description="MixUp augmentation probability",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="copy_paste",
                type="float",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Copy-Paste augmentation probability",
                group="augmentation",
                advanced=True
            ),
            ConfigField(
                name="fliplr",
                type="float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Horizontal flip probability",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="flipud",
                type="float",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Vertical flip probability",
                group="augmentation",
                advanced=True
            ),
            ConfigField(
                name="degrees",
                type="float",
                default=0.0,
                min=0.0,
                max=180.0,
                step=5.0,
                description="Rotation range in degrees",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="translate",
                type="float",
                default=0.1,
                min=0.0,
                max=0.9,
                step=0.05,
                description="Translation fraction",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="scale",
                type="float",
                default=0.5,
                min=0.0,
                max=0.9,
                step=0.1,
                description="Scaling gain",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="shear",
                type="float",
                default=0.0,
                min=0.0,
                max=10.0,
                step=0.5,
                description="Shear angle in degrees",
                group="augmentation",
                advanced=True
            ),
            ConfigField(
                name="perspective",
                type="float",
                default=0.0,
                min=0.0,
                max=0.001,
                step=0.0001,
                description="Perspective transformation",
                group="augmentation",
                advanced=True
            ),

            # HSV Augmentation (Color space)
            ConfigField(
                name="hsv_h",
                type="float",
                default=0.015,
                min=0.0,
                max=1.0,
                step=0.005,
                description="HSV-Hue augmentation",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="hsv_s",
                type="float",
                default=0.7,
                min=0.0,
                max=1.0,
                step=0.1,
                description="HSV-Saturation augmentation",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="hsv_v",
                type="float",
                default=0.4,
                min=0.0,
                max=1.0,
                step=0.1,
                description="HSV-Value (brightness) augmentation",
                group="augmentation",
                advanced=False
            ),

            # ========== Training Optimization ==========
            ConfigField(
                name="amp",
                type="bool",
                default=True,
                description="Automatic Mixed Precision training",
                group="optimization",
                required=False
            ),
            ConfigField(
                name="close_mosaic",
                type="int",
                default=10,
                min=0,
                max=50,
                step=1,
                description="Disable mosaic in last N epochs",
                group="optimization",
                advanced=True
            ),

            # ========== Validation Settings ==========
            ConfigField(
                name="val_interval",
                type="int",
                default=1,
                min=1,
                max=10,
                step=1,
                description="Validate every N epochs",
                group="validation",
                required=False
            ),
        ]

        presets = {
            "easy": {
                "optimizer_type": "Adam",
                "cos_lr": True,
                "aug_enabled": True,
                "mosaic": 1.0,
                "fliplr": 0.5,
                "amp": True,
            },
            "medium": {
                "optimizer_type": "AdamW",
                "weight_decay": 0.0005,
                "cos_lr": True,
                "warmup_epochs": 3,
                "aug_enabled": True,
                "mosaic": 1.0,
                "mixup": 0.1,
                "fliplr": 0.5,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "amp": True,
            },
            "advanced": {
                "optimizer_type": "AdamW",
                "weight_decay": 0.001,
                "momentum": 0.937,
                "cos_lr": True,
                "warmup_epochs": 5,
                "lrf": 0.01,
                "aug_enabled": True,
                "mosaic": 1.0,
                "mixup": 0.15,
                "copy_paste": 0.1,
                "fliplr": 0.5,
                "flipud": 0.0,
                "degrees": 10.0,
                "translate": 0.2,
                "scale": 0.9,
                "shear": 2.0,
                "perspective": 0.0001,
                "hsv_h": 0.02,
                "hsv_s": 0.8,
                "hsv_v": 0.5,
                "amp": True,
                "close_mosaic": 10,
            }
        }

        return ConfigSchema(fields=fields, presets=presets)

    def prepare_model(self):
        """Initialize YOLO model (standard YOLO or YOLO-World for zero-shot detection)."""
        print("[prepare_model] Step 1: Importing ultralytics...")
        sys.stdout.flush()

        # Check if this is YOLO-World (zero-shot detection)
        is_yolo_world = self.task_type == TaskType.ZERO_SHOT_DETECTION

        try:
            if is_yolo_world:
                from ultralytics import YOLOWorld
                print("[prepare_model] YOLOWorld imported successfully")
                sys.stdout.flush()
            else:
                from ultralytics import YOLO
                print("[prepare_model] ultralytics YOLO imported successfully")
                sys.stdout.flush()
        except ImportError as e:
            print(f"[prepare_model] ERROR: Failed to import ultralytics: {e}")
            sys.stdout.flush()
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        except Exception as e:
            print(f"[prepare_model] ERROR during ultralytics import: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            raise

        # Determine model name with suffix based on task
        suffix = self.TASK_SUFFIX_MAP.get(self.task_type, "")

        # Check if model_name already has the suffix to avoid duplication
        # e.g., "yolo11n-seg" + "-seg" = "yolo11n-seg-seg" (WRONG)
        if suffix and self.model_config.model_name.endswith(suffix):
            # Model name already has suffix, don't add it again
            full_model_name = self.model_config.model_name
            print(f"[prepare_model] Model name already contains suffix '{suffix}', not adding again")
        else:
            # Add suffix to model name
            full_model_name = f"{self.model_config.model_name}{suffix}"

        print(f"[prepare_model] Step 2: Getting model weights: {full_model_name}.pt")
        print(f"[prepare_model] Task type: {self.task_type}")
        print(f"[prepare_model] Model type: {'YOLO-World' if is_yolo_world else 'Standard YOLO'}")
        print(f"[prepare_model] Suffix: '{suffix}'")
        sys.stdout.flush()

        # Get model weights with auto-caching
        from platform_sdk import get_model_weights

        def download_yolo():
            """Download YOLO model from original source."""
            print(f"[prepare_model] Downloading from Ultralytics...")
            sys.stdout.flush()

            # YOLO constructor auto-downloads
            if is_yolo_world:
                temp_model = YOLOWorld(f"{full_model_name}.pt")
            else:
                temp_model = YOLO(f"{full_model_name}.pt")

            # Get actual downloaded path from model
            import os
            model_path = os.path.abspath(temp_model.ckpt_path)

            if os.path.exists(model_path):
                print(f"[prepare_model] YOLO downloaded to: {model_path}")
                sys.stdout.flush()
                return model_path
            else:
                raise FileNotFoundError(f"YOLO model not found at: {model_path}")

        # Get weights (local cache → R2 → original source)
        model_path = get_model_weights(
            model_name=full_model_name,
            framework="ultralytics",
            download_fn=download_yolo,
            file_extension="pt"
        )

        print(f"[prepare_model] Using model weights: {model_path}")
        sys.stdout.flush()

        try:
            if is_yolo_world:
                print(f"[prepare_model] Loading YOLOWorld from: {model_path}")
                sys.stdout.flush()

                self.model = YOLOWorld(model_path)

                # Set custom classes if provided
                if hasattr(self.model_config, 'custom_prompts') and self.model_config.custom_prompts:
                    print(f"[prepare_model] Setting custom classes: {self.model_config.custom_prompts}")
                    sys.stdout.flush()
                    self.model.set_classes(self.model_config.custom_prompts)
                    # Store class names for inference
                    self.class_names = self.model_config.custom_prompts
                    print(f"[prepare_model] Custom classes set successfully")
                    sys.stdout.flush()
                else:
                    print("[prepare_model] WARNING: No custom_prompts provided for YOLO-World")
                    print("[prepare_model] YOLO-World requires custom text prompts to function properly")
                    sys.stdout.flush()
            else:
                print(f"[prepare_model] Loading YOLO from: {model_path}")
                sys.stdout.flush()

                self.model = YOLO(model_path)

            print(f"[prepare_model] Model object created")
            sys.stdout.flush()
            print(f"[prepare_model] Model loaded successfully")
            sys.stdout.flush()
        except Exception as e:
            print(f"[prepare_model] ERROR loading model: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise

    def _resolve_dataset_path(self) -> str:
        """
        Resolve dataset path with R2 lazy download support.

        Handles 3 cases:
        1. Built-in Ultralytics datasets (e.g., "coco8.yaml") → return as-is
        2. Simple dataset names (e.g., "det-coco8") → download from R2
        3. Absolute paths → return as-is

        Returns:
            Resolved dataset path
        """
        dataset_path_str = self.dataset_config.dataset_path

        # Case 1: Built-in Ultralytics datasets (.yaml)
        if dataset_path_str.endswith('.yaml'):
            print(f"[_resolve_dataset_path] Using Ultralytics built-in: {dataset_path_str}")
            sys.stdout.flush()
            return dataset_path_str

        # Case 2: Absolute path (custom dataset)
        if os.path.isabs(dataset_path_str) and os.path.exists(dataset_path_str):
            print(f"[_resolve_dataset_path] Using existing path: {dataset_path_str}")
            sys.stdout.flush()
            return dataset_path_str

        # Case 3: Simple name → try R2 lazy download
        # Simple name = no path separators
        if '/' not in dataset_path_str and '\\' not in dataset_path_str:
            print(f"[_resolve_dataset_path] Simple name detected: {dataset_path_str}")
            print(f"[_resolve_dataset_path] Attempting R2 lazy download...")
            sys.stdout.flush()

            try:
                from platform_sdk import get_dataset

                # Download from R2 (or use local cache)
                downloaded_path = get_dataset(
                    dataset_id=dataset_path_str,
                    download_fn=None  # No fallback download function
                )

                print(f"[_resolve_dataset_path] Dataset resolved: {downloaded_path}")
                sys.stdout.flush()
                return downloaded_path

            except Exception as e:
                print(f"[_resolve_dataset_path] WARNING: R2 download failed: {e}")
                print(f"[_resolve_dataset_path] Falling back to original path (may fail)")
                sys.stdout.flush()
                return dataset_path_str

        # Case 4: Path doesn't exist → return as-is (will likely fail later)
        print(f"[_resolve_dataset_path] Using original path: {dataset_path_str}")
        sys.stdout.flush()
        return dataset_path_str

    def prepare_dataset(self):
        """Prepare dataset in YOLO format."""
        # Check if dataset needs to be downloaded from R2
        dataset_path = self._resolve_dataset_path()

        # Update dataset_config with resolved path
        if dataset_path != self.dataset_config.dataset_path:
            print(f"[prepare_dataset] Resolved dataset path: {dataset_path}")
            sys.stdout.flush()
            self.dataset_config.dataset_path = dataset_path

        # Convert DICE format to YOLO format if needed
        if self.dataset_config.format == DatasetFormat.DICE:
            print(f"[prepare_dataset] DICE format detected, converting to YOLO...")
            sys.stdout.flush()

            dataset_path = self._convert_dice_to_yolo(dataset_path)
            self.dataset_config.dataset_path = dataset_path

            print(f"[prepare_dataset] DICE → YOLO conversion completed: {dataset_path}")
            sys.stdout.flush()

        # Clear any existing YOLO cache files to avoid stale data issues
        self._clear_yolo_cache()

        # YOLO requires data.yaml file
        self.data_yaml = self._create_data_yaml()
        print(f"Dataset config created: {self.data_yaml}")

        # Load class names from data.yaml for callbacks
        try:
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
                if 'names' in data_config:
                    self.class_names = data_config['names']
                    print(f"[prepare_dataset] Loaded {len(self.class_names)} class names from data.yaml")
                    sys.stdout.flush()
        except Exception as e:
            print(f"[prepare_dataset] WARNING: Failed to load class names from data.yaml: {e}")
            sys.stdout.flush()

    def _convert_dice_to_yolo(self, dice_dataset_path: str) -> str:
        """
        Convert DICE format dataset to YOLO format.

        Args:
            dice_dataset_path: Path to DICE format dataset directory

        Returns:
            Path to converted YOLO format dataset directory
        """
        from pathlib import Path
        import sys
        import os

        # Add converters to path
        converters_path = Path(__file__).parent.parent / "converters"
        if str(converters_path) not in sys.path:
            sys.path.insert(0, str(converters_path))

        from dice_to_yolo import convert_dice_to_yolo

        # Create output directory for converted dataset
        dice_path = Path(dice_dataset_path)
        output_dir = dice_path.parent / f"{dice_path.name}_yolo"

        print(f"[_convert_dice_to_yolo] Converting DICE → YOLO")
        print(f"[_convert_dice_to_yolo] Source: {dice_dataset_path}")
        print(f"[_convert_dice_to_yolo] Output: {output_dir}")
        sys.stdout.flush()

        # Perform conversion
        result = convert_dice_to_yolo(str(dice_dataset_path), str(output_dir))

        print(f"[_convert_dice_to_yolo] Conversion result:")
        print(f"  - Images: {result['num_images']}")
        print(f"  - Annotations: {result['num_annotations']}")
        print(f"  - Classes: {result['num_classes']}")
        print(f"  - Class names: {result['class_names']}")
        print(f"  - data.yaml: {result['data_yaml']}")
        sys.stdout.flush()

        return str(output_dir)

    def _clear_yolo_cache(self):
        """Clear YOLO cache files in the dataset directory."""
        from pathlib import Path
        import glob

        dataset_path = Path(self.dataset_config.dataset_path)
        labels_dir = dataset_path / "labels"

        if not labels_dir.exists():
            return

        # Find and remove all .cache files
        cache_files = list(labels_dir.rglob("*.cache"))

        for cache_file in cache_files:
            try:
                cache_file.unlink()
                print(f"[_clear_yolo_cache] Removed cache file: {cache_file}")
                sys.stdout.flush()
            except Exception as e:
                print(f"[_clear_yolo_cache] Warning: Failed to remove {cache_file}: {e}")
                sys.stdout.flush()

    def _create_data_yaml(self) -> str:
        """
        Create YOLO format data.yaml configuration file.

        Expected directory structure:
        dataset_path/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
        """
        dataset_path_str = self.dataset_config.dataset_path

        # Check if dataset_path is a Ultralytics built-in dataset
        # Built-in datasets: coco8, coco128, coco, etc.
        # Can be specified as "coco8" or "coco8.yaml"

        # Case 1: Already ends with .yaml - likely a built-in dataset
        if dataset_path_str.endswith('.yaml'):
            print(f"[_create_data_yaml] Using Ultralytics built-in dataset: {dataset_path_str}")
            sys.stdout.flush()
            return dataset_path_str

        # Case 2: Simple name without path separators - convert to built-in
        if '/' not in dataset_path_str and '\\' not in dataset_path_str and not os.path.isabs(dataset_path_str):
            yaml_name = f"{dataset_path_str}.yaml"
            print(f"[_create_data_yaml] Converting simple name to built-in dataset: {yaml_name}")
            print(f"[_create_data_yaml] Ultralytics will auto-download to ~/.cache/ultralytics/datasets/")
            sys.stdout.flush()
            return yaml_name

        # Case 3: Custom dataset path - continue with existing logic
        # Check if data.yaml already exists in dataset
        existing_yaml = os.path.join(dataset_path_str, "data.yaml")
        if os.path.exists(existing_yaml):
            print(f"[_create_data_yaml] Using existing data.yaml: {existing_yaml}")
            sys.stdout.flush()
            return os.path.abspath(existing_yaml)

        # Ensure output_dir is absolute path
        output_dir = os.path.abspath(self.output_dir)
        print(f"[_create_data_yaml] Output directory (absolute): {output_dir}")
        sys.stdout.flush()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"[_create_data_yaml] Output directory created/verified")
        sys.stdout.flush()

        # Create new data.yaml
        if self.dataset_config.format == DatasetFormat.YOLO:
            # Extract actual class IDs from label files to determine correct nc
            actual_class_ids = self._extract_class_ids_from_labels()
            max_class_id = max(actual_class_ids) if actual_class_ids else self.model_config.num_classes - 1

            # nc must be max_class_id + 1 (e.g., if labels use ID 75, nc must be >= 76)
            nc = max(max_class_id + 1, self.model_config.num_classes)

            print(f"[_create_data_yaml] Found {len(actual_class_ids)} unique class IDs: {sorted(actual_class_ids)}")
            print(f"[_create_data_yaml] Max class ID: {max_class_id}, setting nc={nc}")
            sys.stdout.flush()

            # Detect actual train/val folder names
            train_path, val_path = self._detect_yolo_folders()
            print(f"[_create_data_yaml] Detected train path: {train_path}")
            print(f"[_create_data_yaml] Detected val path: {val_path}")
            sys.stdout.flush()

            # YOLO format - create data.yaml
            # Check if train_path is a .txt file (auto-split dataset)
            if train_path.endswith('.txt'):
                # txt files contain absolute paths, so path should point to output_dir
                data = {
                    'path': output_dir,
                    'train': train_path,
                    'val': val_path,
                    'nc': nc,
                    'names': [f'class_{i}' for i in range(nc)]
                }
                print(f"[_create_data_yaml] Using txt file format (auto-split dataset)")
                sys.stdout.flush()
            else:
                # Regular folder structure
                data = {
                    'path': os.path.abspath(self.dataset_config.dataset_path),
                    'train': train_path,
                    'val': val_path,
                    'nc': nc,
                    'names': [f'class_{i}' for i in range(nc)]
                }
        elif self.dataset_config.format == DatasetFormat.COCO:
            # COCO format - convert to YOLO format reference
            data = {
                'path': os.path.abspath(self.dataset_config.dataset_path),
                'train': self.dataset_config.train_split,
                'val': self.dataset_config.val_split,
                'nc': self.model_config.num_classes,
                'names': [f'class_{i}' for i in range(self.model_config.num_classes)]
            }
        else:
            raise ValueError(f"Unsupported dataset format for YOLO: {self.dataset_config.format}")

        # Save data.yaml with absolute path
        yaml_path = os.path.join(output_dir, "data.yaml")
        print(f"[_create_data_yaml] Writing data.yaml to: {yaml_path}")
        sys.stdout.flush()

        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"[_create_data_yaml] File written successfully")
            sys.stdout.flush()
        except Exception as e:
            print(f"[_create_data_yaml] ERROR writing file: {e}")
            sys.stdout.flush()
            raise

        # Verify file was created
        if os.path.exists(yaml_path):
            file_size = os.path.getsize(yaml_path)
            print(f"Created YOLO data.yaml with {data['nc']} classes (size: {file_size} bytes)")
            sys.stdout.flush()
        else:
            print(f"[_create_data_yaml] ERROR: File does not exist after writing!")
            sys.stdout.flush()
            raise IOError(f"Failed to create data.yaml at {yaml_path}")

        return yaml_path

    def _extract_class_ids_from_labels(self) -> set:
        """
        Extract unique class IDs from YOLO label files.

        Returns:
            set: Set of unique class IDs found in label files
        """
        from pathlib import Path

        class_ids = set()
        dataset_path = Path(self.dataset_config.dataset_path)
        labels_dir = dataset_path / "labels"

        print(f"[_extract_class_ids] Searching for labels in: {labels_dir}")
        sys.stdout.flush()

        if not labels_dir.exists():
            print(f"[_extract_class_ids] Labels directory not found: {labels_dir}")
            sys.stdout.flush()
            return class_ids

        # Find all .txt label files recursively (train/, val/, etc.)
        label_files = list(labels_dir.rglob("*.txt"))
        print(f"[_extract_class_ids] Found {len(label_files)} label files")
        sys.stdout.flush()

        for label_file in label_files:
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            # First column is class ID in YOLO format
                            class_id = int(parts[0])
                            class_ids.add(class_id)
            except (ValueError, IOError) as e:
                print(f"[_extract_class_ids] Warning: Error reading {label_file}: {e}")
                sys.stdout.flush()
                continue

        print(f"[_extract_class_ids] Extracted {len(class_ids)} unique class IDs")
        sys.stdout.flush()
        return class_ids

    def _detect_yolo_folders(self) -> Tuple[str, str]:
        """
        Detect actual train/val folder names in YOLO dataset.

        If val folder doesn't exist, automatically creates train/val split
        from all available data.

        YOLO datasets can have various folder structures:
        - images/train, images/val (standard)
        - images/train2017, images/val2017 (COCO format)
        - train, val (flat structure)
        - images/ only (will auto-split)

        Returns:
            tuple: (train_path, val_path) relative to dataset root or output dir
        """
        from pathlib import Path
        import random
        import shutil

        dataset_path = Path(self.dataset_config.dataset_path)
        print(f"[_detect_yolo_folders] Scanning dataset: {dataset_path}")
        sys.stdout.flush()

        # Check for images/ subdirectory
        images_dir = dataset_path / "images"
        if images_dir.exists() and images_dir.is_dir():
            # List all subdirectories under images/
            subdirs = [d.name for d in images_dir.iterdir() if d.is_dir()]
            print(f"[_detect_yolo_folders] Found subdirs in images/: {subdirs}")
            sys.stdout.flush()

            # Try to find train and val directories
            train_candidates = [d for d in subdirs if 'train' in d.lower()]
            val_candidates = [d for d in subdirs if 'val' in d.lower() or 'valid' in d.lower()]

            # Prefer exact matches, then take first match
            train_name = None
            if 'train' in subdirs:
                train_name = 'train'
            elif train_candidates:
                train_name = train_candidates[0]

            val_name = None
            if 'val' in subdirs:
                val_name = 'val'
            elif val_candidates:
                val_name = val_candidates[0]

            # If both train and val exist, use them
            if train_name and val_name:
                train_path = f'images/{train_name}'
                val_path = f'images/{val_name}'
                print(f"[_detect_yolo_folders] Using train={train_path}, val={val_path}")
                sys.stdout.flush()
                return train_path, val_path

            # If only train exists (no val), auto-split
            if train_name and not val_name:
                print(f"[_detect_yolo_folders] Val folder not found, auto-splitting train data...")
                sys.stdout.flush()
                return self._auto_split_dataset(
                    images_base=images_dir,
                    train_folder=train_name,
                    split_ratio=0.8
                )

        # Fallback: Check for flat structure (train/, val/ directly under dataset root)
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"

        if train_dir.exists() and train_dir.is_dir():
            if val_dir.exists():
                train_path = 'train'
                val_path = 'val'
                print(f"[_detect_yolo_folders] Using flat structure: train={train_path}, val={val_path}")
                sys.stdout.flush()
                return train_path, val_path
            else:
                print(f"[_detect_yolo_folders] Val folder not found, auto-splitting train data...")
                sys.stdout.flush()
                return self._auto_split_dataset(
                    images_base=dataset_path,
                    train_folder='train',
                    split_ratio=0.8
                )

        # No train/val structure found, treat entire images/ folder as data
        print(f"[_detect_yolo_folders] No train/val structure detected")
        print(f"[_detect_yolo_folders] Auto-splitting all data in images/ folder...")
        sys.stdout.flush()

        if images_dir.exists():
            return self._auto_split_dataset(
                images_base=dataset_path,
                train_folder='images',
                split_ratio=0.8
            )

        # Last resort: use default (will likely fail in YOLO)
        print(f"[_detect_yolo_folders] Warning: Could not detect folder structure, using defaults")
        sys.stdout.flush()
        return 'images/train', 'images/val'

    def _auto_split_dataset(self, images_base: Path, train_folder: str, split_ratio: float = 0.8) -> Tuple[str, str]:
        """
        Automatically split dataset into train/val sets.

        Creates .txt files with image paths for YOLO to use.
        This avoids copying/moving actual image files.

        Args:
            images_base: Base directory containing images
            train_folder: Name of the folder with all images
            split_ratio: Train/val split ratio (default 0.8 = 80% train)

        Returns:
            tuple: (train_txt_path, val_txt_path) relative paths for data.yaml
        """
        from pathlib import Path
        import random

        print(f"[_auto_split_dataset] Splitting dataset with ratio {split_ratio}")
        sys.stdout.flush()

        # Find all images in the source folder
        source_dir = images_base / train_folder
        if not source_dir.exists():
            source_dir = images_base  # Fallback to base directory

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        all_images = []

        for ext in image_extensions:
            all_images.extend(source_dir.glob(f'*{ext}'))
            all_images.extend(source_dir.glob(f'*{ext.upper()}'))

        if not all_images:
            print(f"[_auto_split_dataset] ERROR: No images found in {source_dir}")
            sys.stdout.flush()
            return 'images/train', 'images/val'  # Fallback

        # Shuffle and split
        random.seed(42)  # For reproducibility
        random.shuffle(all_images)

        split_idx = int(len(all_images) * split_ratio)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        print(f"[_auto_split_dataset] Total images: {len(all_images)}")
        print(f"[_auto_split_dataset] Train: {len(train_images)}, Val: {len(val_images)}")
        sys.stdout.flush()

        # Create .txt files in output directory
        output_dir = Path(self.output_dir)
        train_txt = output_dir / 'train.txt'
        val_txt = output_dir / 'val.txt'

        # Write absolute paths to txt files
        with open(train_txt, 'w') as f:
            for img_path in train_images:
                # YOLO expects absolute paths in .txt files
                f.write(f"{img_path.absolute()}\n")

        with open(val_txt, 'w') as f:
            for img_path in val_images:
                f.write(f"{img_path.absolute()}\n")

        print(f"[_auto_split_dataset] Created train.txt: {train_txt}")
        print(f"[_auto_split_dataset] Created val.txt: {val_txt}")
        sys.stdout.flush()

        # Return relative paths for data.yaml
        # YOLO will resolve these relative to data.yaml location
        return 'train.txt', 'val.txt'

    def train(self, start_epoch: int = 0, checkpoint_path: str = None, resume_training: bool = False) -> List[MetricsResult]:
        """
        Train using YOLO's built-in training API.

        YOLO handles the full training loop internally,
        so we override the base train() method.

        Args:
            start_epoch: Starting epoch (for resume training)
            checkpoint_path: Path to checkpoint file (for resume training)
            resume_training: Whether this is a resumed training session
        """
        # Clear previous validation results if this is a new training (not resume)
        if not resume_training:
            print("[YOLO] Clearing previous validation results (new training)...")
            self._clear_validation_results()
            print("[YOLO] Previous validation results cleared")
            sys.stdout.flush()

        try:
            print("\n[YOLO] Preparing model...")
            self.prepare_model()
            print("[YOLO] Model prepared successfully")
        except Exception as e:
            print(f"[YOLO] ERROR preparing model: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            print("[YOLO] Preparing dataset...")
            self.prepare_dataset()
            print(f"[YOLO] Dataset prepared successfully")
            print(f"[YOLO] data.yaml path: {self.data_yaml}")
        except Exception as e:
            print(f"[YOLO] ERROR preparing dataset: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"\nStarting YOLO training...")
        print(f"  Model: {self.model_config.model_name}")
        print(f"  Task: {self.task_type.value}")
        print(f"  Epochs: {self.training_config.epochs}")
        print(f"  Batch size: {self.training_config.batch_size}")
        print(f"  Image size: {self.model_config.image_size}")
        print(f"  Device: {self.training_config.device}")

        if resume_training and checkpoint_path:
            print(f"  Resume from: {checkpoint_path}")
            print(f"  Start epoch: {start_epoch}")

        # Build YOLO training arguments from advanced config
        try:
            train_args = self._build_yolo_train_args()
            print(f"[YOLO] Training configuration ready ({len(train_args)} parameters)")
            sys.stdout.flush()
        except Exception as e:
            print(f"[YOLO] ERROR building training args: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            raise

        # Add resume training support
        if resume_training and checkpoint_path:
            train_args['resume'] = checkpoint_path

        # Initialize TrainingCallbacks
        from .base import TrainingCallbacks
        callbacks = TrainingCallbacks(
            job_id=self.job_id,
            model_config=self.model_config,
            training_config=self.training_config,
            db_session=None  # No DB session in subprocess
        )

        # Get primary metric configuration from database
        primary_metric = None
        primary_metric_mode = None

        try:
            import sqlite3
            from pathlib import Path

            # Get database path
            training_dir = Path(__file__).parent.parent
            mvp_dir = training_dir.parent
            db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()

                # Query job's primary metric configuration
                cursor.execute(
                    "SELECT primary_metric, primary_metric_mode FROM training_jobs WHERE id = ?",
                    (self.job_id,)
                )
                result = cursor.fetchone()
                conn.close()

                if result:
                    primary_metric, primary_metric_mode = result
                    print(f"[YOLO INFO] Primary metric for best checkpoint: {primary_metric} ({primary_metric_mode})")
                    print(f"[YOLO INFO] YOLO will save best.pt based on its internal fitness calculation")
                else:
                    print(f"[YOLO WARNING] Job {self.job_id} not found in database")
            else:
                print(f"[YOLO WARNING] Database not found at {db_path}")
        except Exception as e:
            print(f"[YOLO WARNING] Failed to load primary metric config: {e}")

        # Start training (creates MLflow run)
        callbacks.on_train_begin()

        # Track recorded epochs to prevent duplicates
        recorded_epochs = set()

        # Define real-time callback for YOLO training
        def on_yolo_epoch_end(trainer):
            """
            YOLO callback for real-time metric collection.

            Called at the end of each training epoch by Ultralytics.
            Extracts metrics from trainer and reports to TrainingCallbacks.
            """
            try:
                epoch_num = trainer.epoch + 1  # Convert to 1-indexed

                # Prevent duplicate recording of the same epoch
                if epoch_num in recorded_epochs:
                    print(f"[YOLO Callback] Skipping duplicate epoch {epoch_num}")
                    sys.stdout.flush()
                    return

                recorded_epochs.add(epoch_num)

                # Extract loss components (trainer.label_loss_items returns dict-like object)
                loss_items = trainer.label_loss_items(trainer.tloss, prefix="train")

                # Get validation metrics if available
                metrics_dict = {}

                # Training losses
                if isinstance(loss_items, dict):
                    metrics_dict.update(loss_items)

                # Validation metrics from trainer.metrics
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    # trainer.metrics is a dict with keys like:
                    # 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50(M)', etc.
                    val_metrics = trainer.metrics

                    # Extract validation losses and metrics
                    for key, value in val_metrics.items():
                        if isinstance(value, (int, float)):
                            # Clean up metric names but preserve (B) and (M) suffixes
                            clean_key = key.replace('metrics/', '')
                            metrics_dict[clean_key] = value

                            # Also add without suffix for backward compatibility (use Box metrics)
                            if '(B)' in clean_key:
                                base_key = clean_key.replace('(B)', '')
                                if base_key not in metrics_dict:  # Don't overwrite
                                    metrics_dict[base_key] = value

                # Calculate total losses
                train_loss = sum(v for k, v in metrics_dict.items() if k.startswith('train/') and 'loss' in k)
                val_loss = sum(v for k, v in metrics_dict.items() if k.startswith('val/') and 'loss' in k)

                # Add summary metrics
                metrics_dict['train_loss'] = train_loss if train_loss > 0 else metrics_dict.get('train/box_loss', 0)
                metrics_dict['val_loss'] = val_loss if val_loss > 0 else metrics_dict.get('val/box_loss', 0)

                # Get checkpoint path if this is a checkpoint epoch
                # YOLO saves checkpoints automatically (best.pt and last.pt)
                checkpoint_path = None
                if hasattr(trainer, 'save_dir') and hasattr(trainer, 'best_fitness'):
                    # Check if we have weights saved
                    import os
                    last_pt = os.path.join(trainer.save_dir, 'weights', 'last.pt')
                    if os.path.exists(last_pt):
                        checkpoint_path = last_pt

                # Report to callbacks for unified metric collection (MLflow + Database)
                print(f"[YOLO Callback] Epoch {epoch_num} completed, reporting metrics to callbacks...")
                sys.stdout.flush()
                callbacks.on_epoch_end(epoch_num, metrics_dict, checkpoint_path)

                # Save validation results to database (use metrics_dict which has all metrics)
                # Note: trainer.metrics may be None when val=False, but metrics_dict has training metrics
                if True:  # Always try to save, use metrics_dict instead of trainer.metrics
                    try:
                        # Add parent directory to sys.path for imports
                        from pathlib import Path
                        training_dir = Path(__file__).parent.parent
                        if str(training_dir) not in sys.path:
                            sys.path.insert(0, str(training_dir))

                        from validators.metrics import ValidationMetricsCalculator, TaskType

                        # Extract validation metrics
                        map_50 = metrics_dict.get('mAP50', 0.0)
                        map_50_95 = metrics_dict.get('mAP50-95', 0.0)
                        precision = metrics_dict.get('precision', 0.0)
                        recall = metrics_dict.get('recall', 0.0)

                        # Extract per-class metrics from YOLO trainer
                        per_class_metrics = None
                        class_names = None
                        pr_curve_path = None

                        if hasattr(trainer, 'validator') and trainer.validator is not None:
                            validator = trainer.validator

                            # Get class names from validator
                            if hasattr(validator, 'names') and validator.names is not None:
                                class_names = list(validator.names.values())
                                print(f"[YOLO Callback] Found {len(class_names)} classes: {class_names}")
                                sys.stdout.flush()
                            else:
                                # Validation disabled, use class_names from adapter
                                if hasattr(self, 'class_names') and self.class_names:
                                    class_names = self.class_names
                                    print(f"[YOLO Callback] Using adapter class_names: {len(class_names)} classes")
                                    sys.stdout.flush()
                                else:
                                    print(f"[YOLO Callback] No class_names available (validation disabled)")
                                    sys.stdout.flush()

                            # Get per-class AP from validator results
                            if hasattr(validator, 'metrics') and validator.metrics is not None:
                                val_metrics = validator.metrics

                                # YOLO metrics structure: metrics.box.maps (per-class mAP50-95 array)
                                if hasattr(val_metrics, 'box') and val_metrics.box is not None:
                                    box_metrics = val_metrics.box

                                    # maps: array of per-class mAP@0.5:0.95
                                    per_class_ap50_95 = None
                                    if hasattr(box_metrics, 'maps') and box_metrics.maps is not None:
                                        import numpy as np
                                        if isinstance(box_metrics.maps, np.ndarray):
                                            per_class_ap50_95 = box_metrics.maps.tolist()
                                        elif isinstance(box_metrics.maps, list):
                                            per_class_ap50_95 = box_metrics.maps

                                    # map50: array of per-class mAP@0.5
                                    per_class_ap50 = None
                                    if hasattr(box_metrics, 'ap50') and box_metrics.ap50 is not None:
                                        import numpy as np
                                        if isinstance(box_metrics.ap50, np.ndarray):
                                            per_class_ap50 = box_metrics.ap50.tolist()
                                        elif isinstance(box_metrics.ap50, list):
                                            per_class_ap50 = box_metrics.ap50

                                    # Get per-class precision and recall
                                    per_class_precision = None
                                    per_class_recall = None
                                    if hasattr(box_metrics, 'p') and box_metrics.p is not None:
                                        import numpy as np
                                        if isinstance(box_metrics.p, np.ndarray):
                                            per_class_precision = box_metrics.p.tolist()
                                        elif isinstance(box_metrics.p, list):
                                            per_class_precision = box_metrics.p

                                    if hasattr(box_metrics, 'r') and box_metrics.r is not None:
                                        import numpy as np
                                        if isinstance(box_metrics.r, np.ndarray):
                                            per_class_recall = box_metrics.r.tolist()
                                        elif isinstance(box_metrics.r, list):
                                            per_class_recall = box_metrics.r

                                    # Get per-class object count (support)
                                    per_class_count = {}

                                    # Try multiple sources for ground truth counts
                                    # 1. Try nt_per_class from box_metrics
                                    if hasattr(box_metrics, 'nt_per_class') and box_metrics.nt_per_class is not None:
                                        import numpy as np
                                        nt_per_class = box_metrics.nt_per_class
                                        print(f"[YOLO Callback DEBUG] nt_per_class type: {type(nt_per_class)}, value: {nt_per_class}")
                                        sys.stdout.flush()
                                        if isinstance(nt_per_class, np.ndarray) and len(nt_per_class) > 0:
                                            for i, count in enumerate(nt_per_class):
                                                if count > 0:
                                                    per_class_count[i] = int(count)
                                        elif isinstance(nt_per_class, dict):
                                            per_class_count = {int(k): int(v) for k, v in nt_per_class.items()}

                                    # 2. Try stats from validator
                                    if not per_class_count and hasattr(validator, 'stats') and validator.stats is not None:
                                        try:
                                            import numpy as np
                                            stats = validator.stats
                                            print(f"[YOLO Callback DEBUG] validator.stats available, len: {len(stats) if hasattr(stats, '__len__') else 'N/A'}")
                                            sys.stdout.flush()
                                            # stats is usually a list of arrays, each containing detection info per image
                                            # We need to count how many ground truth objects per class
                                            if isinstance(stats, list) and len(stats) > 0:
                                                class_gt_counts = {}
                                                for stat in stats:
                                                    if isinstance(stat, tuple) and len(stat) > 1:
                                                        # stat typically contains (tp, conf, pred_cls, target_cls)
                                                        target_cls = stat[3] if len(stat) > 3 else stat[-1]
                                                        if isinstance(target_cls, np.ndarray):
                                                            for cls_id in target_cls:
                                                                cls_id = int(cls_id)
                                                                class_gt_counts[cls_id] = class_gt_counts.get(cls_id, 0) + 1
                                                per_class_count = class_gt_counts
                                                print(f"[YOLO Callback] Extracted counts from validator.stats: {per_class_count}")
                                                sys.stdout.flush()
                                        except Exception as e:
                                            print(f"[YOLO Callback] Failed to extract from stats: {e}")
                                            import traceback
                                            traceback.print_exc()
                                            sys.stdout.flush()

                                    # 3. Try confusion matrix (row sums = ground truth counts)
                                    if not per_class_count and hasattr(validator, 'confusion_matrix') and validator.confusion_matrix is not None:
                                        try:
                                            import numpy as np
                                            cm = validator.confusion_matrix
                                            print(f"[YOLO Callback DEBUG] confusion_matrix available: {hasattr(cm, 'matrix')}")
                                            sys.stdout.flush()
                                            if hasattr(cm, 'matrix') and cm.matrix is not None:
                                                matrix = cm.matrix
                                                print(f"[YOLO Callback DEBUG] matrix shape: {matrix.shape}")
                                                sys.stdout.flush()
                                                if isinstance(matrix, np.ndarray) and matrix.shape[0] > 0:
                                                    # Sum each row to get ground truth count per class
                                                    # Row i = ground truth class i
                                                    for i in range(min(matrix.shape[0] - 1, len(class_names) if class_names else matrix.shape[0])):
                                                        # Exclude last column if it's background
                                                        row_sum = int(matrix[i, :-1].sum()) if matrix.shape[1] > len(class_names) else int(matrix[i, :].sum())
                                                        if row_sum > 0:
                                                            per_class_count[i] = row_sum
                                                    print(f"[YOLO Callback] Extracted counts from confusion matrix: {per_class_count}")
                                                    sys.stdout.flush()
                                        except Exception as e:
                                            print(f"[YOLO Callback] Failed to extract from confusion matrix: {e}")
                                            import traceback
                                            traceback.print_exc()
                                            sys.stdout.flush()

                                    print(f"[YOLO Callback] Final per-class counts: {per_class_count}")
                                    sys.stdout.flush()

                                    # Build per-class metrics dict
                                    if per_class_ap50_95 is not None and class_names is not None:
                                        per_class_metrics = {}
                                        for i, class_name in enumerate(class_names):
                                            if i < len(per_class_ap50_95):
                                                per_class_metrics[class_name] = {
                                                    'ap_50_95': float(per_class_ap50_95[i]),
                                                    'ap_50': float(per_class_ap50[i]) if per_class_ap50 and i < len(per_class_ap50) else 0.0,
                                                    'precision': float(per_class_precision[i]) if per_class_precision and i < len(per_class_precision) else 0.0,
                                                    'recall': float(per_class_recall[i]) if per_class_recall and i < len(per_class_recall) else 0.0,
                                                    'support': per_class_count.get(i, 0),
                                                }

                                        print(f"[YOLO Callback] Extracted per-class metrics for {len(per_class_metrics)} classes")
                                        print(f"[YOLO Callback] Per-class counts: {per_class_count}")
                                        if per_class_precision:
                                            print(f"[YOLO Callback] Per-class precision: {per_class_precision[:3]}...")
                                        if per_class_recall:
                                            print(f"[YOLO Callback] Per-class recall: {per_class_recall[:3]}...")
                                        sys.stdout.flush()

                            # Get PR curve path (YOLO saves it automatically)
                            if hasattr(trainer, 'save_dir'):
                                import os
                                pr_curve_file = os.path.join(trainer.save_dir, 'PR_curve.png')
                                if os.path.exists(pr_curve_file):
                                    pr_curve_path = pr_curve_file
                                    print(f"[YOLO Callback] Found PR curve: {pr_curve_path}")
                                    sys.stdout.flush()

                        # Create ValidationMetrics using calculator
                        validation_metrics = ValidationMetricsCalculator.compute_metrics(
                            task_type=self.task_type,
                            predictions=None,  # YOLO pre-computes metrics
                            labels=None,
                            class_names=class_names,
                            loss=val_loss,
                            map_50=map_50,
                            map_50_95=map_50_95,
                            precision=precision,
                            recall=recall
                        )

                        # Add Box/Mask metrics as extra_metrics for segmentation tasks
                        # Use string comparison to avoid enum identity issues
                        task_type_value = self.task_type.value if hasattr(self.task_type, 'value') else str(self.task_type)
                        print(f"[YOLO Callback DEBUG] task_type_value = '{task_type_value}', checking for instance_segmentation")
                        sys.stdout.flush()

                        if task_type_value == "instance_segmentation":
                            print(f"[YOLO Callback DEBUG] Instance segmentation detected, extracting Box/Mask metrics")
                            sys.stdout.flush()
                            extra_metrics = {}
                            # Use metrics_dict that was already extracted from trainer.metrics
                            for key, value in metrics_dict.items():
                                # Include all metrics with (B) or (M) suffix
                                if '(B)' in key or '(M)' in key:
                                    extra_metrics[key] = float(value) if isinstance(value, (int, float)) else 0.0

                            if extra_metrics:
                                validation_metrics._extra_metrics = extra_metrics
                                print(f"[YOLO Callback] ✓ Added {len(extra_metrics)} Box/Mask metrics to validation result")
                                print(f"[YOLO Callback] Box/Mask keys: {list(extra_metrics.keys())}")
                                sys.stdout.flush()
                            else:
                                print(f"[YOLO Callback WARNING] No Box/Mask metrics found in metrics_dict")
                                print(f"[YOLO Callback] metrics_dict keys: {list(metrics_dict.keys())}")
                                sys.stdout.flush()
                        else:
                            print(f"[YOLO Callback DEBUG] Not instance_segmentation (task_type='{task_type_value}'), skipping Box/Mask extraction")
                            sys.stdout.flush()

                        # Override per_class_metrics if we extracted them
                        if per_class_metrics:
                            # Manually set per_class_metrics in validation_metrics
                            if hasattr(validation_metrics, 'detection') and validation_metrics.detection:
                                validation_metrics.detection.per_class_ap = per_class_metrics

                        # Store class_names in validation_metrics for detection
                        if class_names:
                            validation_metrics._class_names = class_names

                        # Store PR curve path
                        if pr_curve_path:
                            if hasattr(validation_metrics, 'detection') and validation_metrics.detection:
                                validation_metrics.detection.pr_curves = {'image_path': pr_curve_path}

                        # Get checkpoint path for this epoch
                        checkpoint_path = None
                        best_weights = os.path.join(self.output_dir, f'job_{self.job_id}', 'weights', 'best.pt')
                        last_weights = os.path.join(self.output_dir, f'job_{self.job_id}', 'weights', 'last.pt')

                        if os.path.exists(best_weights):
                            checkpoint_path = best_weights
                        elif os.path.exists(last_weights):
                            checkpoint_path = last_weights

                        # Save to database
                        validation_result_id = self._save_validation_result(epoch_num, validation_metrics, checkpoint_path)
                        print(f"[YOLO Callback] Saved validation result ID: {validation_result_id}, checkpoint: {checkpoint_path}")
                        sys.stdout.flush()

                        # Save per-image detection results with bbox info
                        print(f"[YOLO Callback] Checking image results save: validation_result_id={validation_result_id}, has_validator={hasattr(trainer, 'validator')}, validator_not_none={trainer.validator is not None if hasattr(trainer, 'validator') else False}")
                        sys.stdout.flush()

                        if validation_result_id and hasattr(trainer, 'validator') and trainer.validator is not None:
                            try:
                                print(f"[YOLO Callback] Attempting to save image results...")
                                sys.stdout.flush()
                                self._save_yolo_image_results(validation_result_id, epoch_num, trainer.validator, class_names)
                                print(f"[YOLO Callback] Successfully saved image results")
                                sys.stdout.flush()
                            except Exception as e:
                                print(f"[YOLO Callback WARNING] Failed to save image results: {e}")
                                import traceback
                                traceback.print_exc()
                                sys.stdout.flush()
                        else:
                            print(f"[YOLO Callback] Skipping image results save (validation_result_id: {validation_result_id}, validator: {trainer.validator is not None if hasattr(trainer, 'validator') else 'no attr'})")
                            sys.stdout.flush()
                    except Exception as e:
                        print(f"[YOLO Callback WARNING] Failed to save validation result: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()

            except Exception as e:
                print(f"[YOLO Callback ERROR] Failed to collect metrics at epoch {trainer.epoch}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()

        # Register callback with YOLO model
        self.model.add_callback("on_fit_epoch_end", on_yolo_epoch_end)
        print("[YOLO] Registered real-time metric collection callback")
        sys.stdout.flush()

        # Disable YOLO's built-in MLflow integration to avoid parameter conflicts
        # We use our own Callbacks for MLflow logging
        try:
            from ultralytics import settings
            settings.update({'mlflow': False})
            print("[YOLO] Disabled YOLO's built-in MLflow (using custom Callbacks)")
        except Exception as e:
            print(f"[YOLO WARNING] Could not disable MLflow: {e}")
        sys.stdout.flush()

        # YOLO training
        try:
            print("[YOLO] Starting YOLO training loop...")
            print(f"[YOLO] Calling self.model.train() with {len(train_args)} arguments")
            print("[YOLO] Note: YOLO may take a few moments to download the model on first run")
            print("[YOLO] Training output will appear below (this may take a while)...")
            print("="*80)
            sys.stdout.flush()

            # Check CUDA availability and fall back to CPU if needed
            import torch
            print(f"[YOLO] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[YOLO] CUDA device count: {torch.cuda.device_count()}")
                print(f"[YOLO] CUDA device name: {torch.cuda.get_device_name(0)}")
            elif train_args.get('device') == 'cuda':
                print("[YOLO] WARNING: CUDA requested but not available, falling back to CPU")
                train_args['device'] = 'cpu'
            sys.stdout.flush()

            # Call YOLO train - this will block until training is complete
            print(f"[YOLO] Using device: {train_args['device']}")
            print(f"[YOLO] MLflow tracking URI: {os.environ.get('MLFLOW_TRACKING_URI', 'Not set')}")
            print(f"[YOLO] Callbacks MLflow Run ID: {callbacks.mlflow_run_id}")
            print("[YOLO] About to call model.train()...")
            sys.stdout.flush()

            # Force YOLO to use our stdout/stderr
            import logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )

            # Ultralytics uses its own logger, force it to use stdout
            from ultralytics.utils import LOGGER
            LOGGER.setLevel(logging.INFO)
            for handler in LOGGER.handlers[:]:
                LOGGER.removeHandler(handler)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            LOGGER.addHandler(stream_handler)

            print("[YOLO] Logger configured, starting training...")
            sys.stdout.flush()

            results = self.model.train(**train_args)

            print("[YOLO] model.train() returned!")
            sys.stdout.flush()

            print("="*80)
            print("[YOLO] Training loop completed!")
            print(f"[YOLO] Results type: {type(results)}")
            sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n[YOLO] Training interrupted by user")
            sys.stdout.flush()
            callbacks.on_train_end()  # Close MLflow run
            raise
        except Exception as e:
            print(f"\n[YOLO] ERROR during training: {e}")
            print(f"[YOLO] Error type: {type(e).__name__}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            callbacks.on_train_end()  # Close MLflow run even on error
            raise

        print(f"\nTraining completed!")
        print(f"Results saved to: {self.output_dir}/job_{self.job_id}")

        # Metrics are collected in real-time via on_yolo_epoch_end callback
        # No need to parse results.csv after training completes
        print("[YOLO] Metrics collected in real-time via callbacks")

        # End training
        callbacks.on_train_end()

        # For compatibility, still return empty list
        # (metrics were already reported via callbacks during training)
        return []

    def _build_yolo_train_args(self) -> Dict[str, Any]:
        """
        Build YOLO training arguments from advanced config.

        Maps our advanced config schema to YOLO's training parameters.
        """
        args = {
            # Basic parameters
            'data': self.data_yaml,
            'epochs': self.training_config.epochs,
            'imgsz': self.model_config.image_size,
            'batch': self.training_config.batch_size,
            'lr0': self.training_config.learning_rate,
            'device': self.training_config.device,
            'project': os.path.abspath(self.output_dir),  # Use absolute path
            'name': f'job_{self.job_id}',
            'exist_ok': True,
            'verbose': False,  # Reduce YOLO verbosity
            'workers': 0,  # Disable multiprocessing to avoid output issues
            'plots': False,  # Disable plots to skip final validation
            'save': True,  # Save checkpoints
            'save_period': -1,  # Only save last and best
            'val': False,  # Disable validation during training
        }

        adv_config = self.training_config.advanced_config
        if not adv_config:
            return args

        # Optimizer (flat structure)
        if 'optimizer_type' in adv_config:
            optimizer_map = {
                'adam': 'Adam',
                'adamw': 'AdamW',
                'sgd': 'SGD',
                'rmsprop': 'RMSprop',
                'Adam': 'Adam',
                'AdamW': 'AdamW',
                'SGD': 'SGD',
                'RMSprop': 'RMSprop',
            }
            args['optimizer'] = optimizer_map.get(adv_config['optimizer_type'], 'Adam')

        if 'weight_decay' in adv_config:
            args['weight_decay'] = adv_config['weight_decay']

        if 'momentum' in adv_config:
            args['momentum'] = adv_config['momentum']

        # Scheduler (flat structure)
        if 'cos_lr' in adv_config:
            args['cos_lr'] = adv_config['cos_lr']

        if 'lrf' in adv_config:
            args['lrf'] = adv_config['lrf']

        if 'warmup_epochs' in adv_config:
            args['warmup_epochs'] = adv_config['warmup_epochs']

        if 'warmup_momentum' in adv_config:
            args['warmup_momentum'] = adv_config['warmup_momentum']

        if 'warmup_bias_lr' in adv_config:
            args['warmup_bias_lr'] = adv_config['warmup_bias_lr']

        # Augmentation (flat structure - YOLO specific)
        if 'fliplr' in adv_config:
            args['fliplr'] = adv_config['fliplr']

        if 'flipud' in adv_config:
            args['flipud'] = adv_config['flipud']

        if 'mosaic' in adv_config:
            args['mosaic'] = adv_config['mosaic']

        if 'mixup' in adv_config:
            args['mixup'] = adv_config['mixup']

        if 'copy_paste' in adv_config:
            args['copy_paste'] = adv_config['copy_paste']

        if 'degrees' in adv_config:
            args['degrees'] = adv_config['degrees']

        if 'translate' in adv_config:
            args['translate'] = adv_config['translate']

        if 'scale' in adv_config:
            args['scale'] = adv_config['scale']

        if 'shear' in adv_config:
            args['shear'] = adv_config['shear']

        if 'perspective' in adv_config:
            args['perspective'] = adv_config['perspective']

        if 'hsv_h' in adv_config:
            args['hsv_h'] = adv_config['hsv_h']

        if 'hsv_s' in adv_config:
            args['hsv_s'] = adv_config['hsv_s']

        if 'hsv_v' in adv_config:
            args['hsv_v'] = adv_config['hsv_v']

        # Optimization
        if 'amp' in adv_config:
            args['amp'] = adv_config['amp']

        if 'close_mosaic' in adv_config:
            args['close_mosaic'] = adv_config['close_mosaic']

        # Mixed precision fallback
        if 'mixed_precision' in adv_config and 'amp' not in args:
            args['amp'] = adv_config['mixed_precision']

        return args

    def _convert_yolo_results(self, results, callbacks=None) -> List[MetricsResult]:
        """
        Convert YOLO training results to standardized MetricsResult.

        Parses results.csv and reports to callbacks for unified metric logging.

        YOLO results contain metrics like:
        - train/box_loss, train/cls_loss, train/dfl_loss
        - val/box_loss, val/cls_loss, val/dfl_loss
        - metrics/precision, metrics/recall
        - metrics/mAP50, metrics/mAP50-95
        """
        import csv

        metrics_list = []

        # Path to YOLO results.csv
        results_csv = os.path.join(self.output_dir, f'job_{self.job_id}', 'results.csv')

        if not os.path.exists(results_csv):
            print(f"[_convert_yolo_results] Warning: results.csv not found at {results_csv}")
            sys.stdout.flush()
            return metrics_list

        print(f"[_convert_yolo_results] Parsing {results_csv}")
        sys.stdout.flush()

        # Parse results.csv
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]  # Remove whitespace

            for row in reader:
                # Clean up row values (remove whitespace)
                row = {k.strip(): v.strip() for k, v in row.items()}

                epoch = int(row['epoch'])

                # Extract metrics
                train_box_loss = float(row.get('train/box_loss', 0))
                train_seg_loss = float(row.get('train/seg_loss', 0))
                train_cls_loss = float(row.get('train/cls_loss', 0))
                train_dfl_loss = float(row.get('train/dfl_loss', 0))
                val_box_loss = float(row.get('val/box_loss', 0))
                val_seg_loss = float(row.get('val/seg_loss', 0))
                val_cls_loss = float(row.get('val/cls_loss', 0))
                val_dfl_loss = float(row.get('val/dfl_loss', 0))

                # Extract Box metrics
                precision_box = float(row.get('metrics/precision(B)', 0))
                recall_box = float(row.get('metrics/recall(B)', 0))
                mAP50_box = float(row.get('metrics/mAP50(B)', 0))
                mAP50_95_box = float(row.get('metrics/mAP50-95(B)', 0))

                # Extract Mask metrics (for segmentation tasks)
                precision_mask = float(row.get('metrics/precision(M)', 0))
                recall_mask = float(row.get('metrics/recall(M)', 0))
                mAP50_mask = float(row.get('metrics/mAP50(M)', 0))
                mAP50_95_mask = float(row.get('metrics/mAP50-95(M)', 0))

                # Calculate total losses (include seg_loss if present)
                train_loss = train_box_loss + train_cls_loss + train_dfl_loss + train_seg_loss
                val_loss = val_box_loss + val_cls_loss + val_dfl_loss + val_seg_loss

                # Create metrics dict for callbacks
                metrics_dict = {
                    'train_loss': train_loss,
                    'train_box_loss': train_box_loss,
                    'train_cls_loss': train_cls_loss,
                    'train_dfl_loss': train_dfl_loss,
                    'val_loss': val_loss,
                    'val_box_loss': val_box_loss,
                    'val_cls_loss': val_cls_loss,
                    'val_dfl_loss': val_dfl_loss,
                    # Box metrics (use generic names for backward compatibility)
                    'precision': precision_box,
                    'recall': recall_box,
                    'mAP50': mAP50_box,
                    'mAP50-95': mAP50_95_box,
                    # Box metrics (explicit names for segmentation UI)
                    'precision(B)': precision_box,
                    'recall(B)': recall_box,
                    'mAP50(B)': mAP50_box,
                    'mAP50-95(B)': mAP50_95_box,
                    # Mask metrics (for segmentation)
                    'precision(M)': precision_mask,
                    'recall(M)': recall_mask,
                    'mAP50(M)': mAP50_mask,
                    'mAP50-95(M)': mAP50_95_mask,
                }

                # Add segmentation loss if present
                if train_seg_loss > 0 or val_seg_loss > 0:
                    metrics_dict['train_seg_loss'] = train_seg_loss
                    metrics_dict['val_seg_loss'] = val_seg_loss

                # Report to callbacks for unified metric collection
                # Callbacks will handle both MLflow and database logging
                if callbacks:
                    callbacks.on_epoch_end(epoch - 1, metrics_dict)  # epoch is 1-indexed in CSV

                # Save validation results to database
                # Create ValidationMetrics using ValidationMetricsCalculator
                try:
                    from ..validators.metrics import ValidationMetricsCalculator, TaskType

                    # Use self.task_type directly (it's already a TaskType enum)
                    # self.task_type is set in base adapter from TrainingJob.task_type
                    task_type = self.task_type

                    # For detection/segmentation tasks, pass pre-computed YOLO metrics
                    # Note: We use Box metrics for the main metrics (for backward compatibility)
                    # Mask metrics will be added separately to the metrics dict
                    validation_metrics = ValidationMetricsCalculator.compute_metrics(
                        task_type=task_type,
                        predictions=None,  # Not needed, metrics pre-computed
                        labels=None,  # Not needed, metrics pre-computed
                        class_names=self.class_names if hasattr(self, 'class_names') else None,
                        loss=val_loss,
                        map_50=mAP50_box,
                        map_50_95=mAP50_95_box,
                        precision=precision_box,
                        recall=recall_box,
                    )

                    # Add Box and Mask metrics to the detection metrics dict
                    # This allows the UI to display both Box and Mask metrics separately
                    if hasattr(validation_metrics, 'detection') and validation_metrics.detection:
                        # Update the detection metrics dict with explicit Box/Mask metrics
                        detection_dict = validation_metrics.detection.to_dict()
                        # Add Box metrics (explicit names)
                        detection_dict['precision(B)'] = precision_box
                        detection_dict['recall(B)'] = recall_box
                        detection_dict['mAP50(B)'] = mAP50_box
                        detection_dict['mAP50-95(B)'] = mAP50_95_box
                        # Add Mask metrics (for segmentation)
                        detection_dict['precision(M)'] = precision_mask
                        detection_dict['recall(M)'] = recall_mask
                        detection_dict['mAP50(M)'] = mAP50_mask
                        detection_dict['mAP50-95(M)'] = mAP50_95_mask
                        # Update the detection metrics object
                        # Store in a way that will be serialized to JSON
                        validation_metrics._extra_metrics = detection_dict

                    # Save to validation_results table
                    # Note: checkpoint_path would be set if we had checkpoint saving per epoch
                    val_result_id = self._save_validation_result(
                        epoch=epoch,  # Use 1-indexed epoch for database
                        validation_metrics=validation_metrics,
                        checkpoint_path=None  # YOLO saves checkpoints internally
                    )

                    if val_result_id:
                        print(f"[YOLO] Saved validation result {val_result_id} for epoch {epoch}")
                        sys.stdout.flush()

                except Exception as e:
                    print(f"[YOLO] Warning: Failed to save validation result for epoch {epoch}: {e}")
                    sys.stdout.flush()

                # Create MetricsResult for our system
                metrics = MetricsResult(
                    epoch=epoch - 1,  # Convert to 0-indexed
                    step=epoch - 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    metrics=metrics_dict
                )
                metrics_list.append(metrics)

        print(f"[_convert_yolo_results] Parsed {len(metrics_list)} epochs from results.csv")
        sys.stdout.flush()

        return metrics_list

    # These methods are not used since YOLO handles training internally
    # but must be implemented for the interface

    def train_epoch(self, epoch: int) -> MetricsResult:
        """Not used - YOLO handles training internally."""
        pass

    def validate(self, epoch: int) -> MetricsResult:
        """
        Not used - YOLO handles validation internally.

        YOLO automatically computes detection metrics (mAP, precision, recall)
        and saves them to results.csv. The _convert_yolo_results() method
        parses these results and saves them to the validation_results table
        using ValidationMetricsCalculator.

        Validation results are:
        - Logged to TrainingMetric table via callbacks
        - Saved to validation_results table via _save_validation_result()
        - Reported to MLflow via callbacks
        """
        pass

    def _save_yolo_image_results(self, validation_result_id: int, epoch: int, validator, class_names):
        """
        Save per-image detection results with bbox information.

        Parses YOLO predictions.json and ground truth labels to extract bbox info
        in a framework-independent format.

        Args:
            validation_result_id: Database ID of validation result
            epoch: Current epoch
            validator: YOLO validator object with results
            class_names: List of class names
        """
        print(f"[YOLO] Collecting per-image detection results for epoch {epoch}...")
        sys.stdout.flush()

        try:
            import json
            from pathlib import Path
            from collections import defaultdict
            import torch

            # Try to extract predictions directly from validator
            if not hasattr(validator, 'pred') or validator.pred is None:
                print(f"[YOLO] No predictions available in validator (validator.pred is None)")
                sys.stdout.flush()
                return

            print(f"[YOLO] Extracting predictions directly from validator...")
            sys.stdout.flush()

            # Get predictions from validator
            # validator.pred is a list of tensors, one per batch
            # Each tensor has shape [num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class_id]
            all_preds = []
            if isinstance(validator.pred, list):
                for batch_pred in validator.pred:
                    if isinstance(batch_pred, torch.Tensor):
                        all_preds.append(batch_pred.cpu().numpy())
                    elif isinstance(batch_pred, list):
                        for pred in batch_pred:
                            if isinstance(pred, torch.Tensor):
                                all_preds.append(pred.cpu().numpy())

            if not all_preds:
                print(f"[YOLO] Could not extract predictions from validator")
                sys.stdout.flush()
                return

            print(f"[YOLO] Extracted {len(all_preds)} prediction batches")
            sys.stdout.flush()

            # Group predictions by image index
            pred_by_image = defaultdict(list)
            for img_idx, pred_array in enumerate(all_preds):
                if pred_array is not None and len(pred_array) > 0:
                    for detection in pred_array:
                        # YOLO format: [x1, y1, x2, y2, confidence, class_id]
                        if len(detection) >= 6:
                            x1, y1, x2, y2, conf, cls_id = detection[:6]
                            # Convert to [x, y, w, h] format
                            pred_by_image[img_idx].append({
                                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                'class_id': int(cls_id),
                                'confidence': float(conf)
                            })

            # Get image paths from validator's dataloader
            if not hasattr(validator, 'dataloader') or not validator.dataloader:
                print("[YOLO] No dataloader in validator")
                sys.stdout.flush()
                return

            dataset = validator.dataloader.dataset
            image_paths = []

            # Extract image paths
            if hasattr(dataset, 'im_files'):
                image_paths = dataset.im_files
            elif hasattr(dataset, 'img_files'):
                image_paths = dataset.img_files

            if not image_paths:
                print("[YOLO] Could not extract image paths from dataset")
                sys.stdout.flush()
                return

            print(f"[YOLO] Found {len(image_paths)} images in validation set")
            sys.stdout.flush()

            # Collect per-image results
            image_results = []
            dataset_path = Path(self.dataset_config.dataset_path)

            for img_idx, img_path in enumerate(image_paths):
                img_path = Path(img_path)

                # Get ground truth boxes from label file
                # YOLO label format: class_id x_center y_center width height (normalized)
                label_path = self._get_yolo_label_path(img_path, dataset_path)
                true_boxes = []

                if label_path and label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                # Convert normalized coordinates to absolute (will be done in frontend)
                                # Store normalized for now
                                true_boxes.append({
                                    'class_id': class_id,
                                    'bbox': [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                                    'format': 'yolo'  # x_center, y_center, width, height (normalized)
                                })

                # Get predicted boxes
                pred_boxes = pred_by_image.get(img_idx, [])

                # Only save if there are boxes (true or predicted)
                if true_boxes or pred_boxes:
                    image_results.append({
                        'image_index': img_idx,
                        'image_path': str(img_path),
                        'image_name': img_path.name,
                        'true_boxes': true_boxes,
                        'predicted_boxes': pred_boxes,
                    })

            print(f"[YOLO] Collected {len(image_results)} images with detection results")
            sys.stdout.flush()

            # Save to database using base adapter method
            if image_results:
                self._save_detection_image_results(
                    validation_result_id,
                    epoch,
                    image_results,
                    class_names
                )

        except Exception as e:
            print(f"[YOLO] Error collecting image results: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    def _get_yolo_label_path(self, image_path: Path, dataset_path: Path) -> Path:
        """
        Convert image path to corresponding label path.

        YOLO structure: images/val/xxx.jpg -> labels/val/xxx.txt
        """
        try:
            # Get relative path from dataset root
            rel_path = image_path.relative_to(dataset_path)

            # Replace 'images' with 'labels' and change extension
            label_rel = str(rel_path).replace('images', 'labels').replace(image_path.suffix, '.txt')
            label_path = dataset_path / label_rel

            return label_path
        except Exception as e:
            print(f"[YOLO] Error getting label path for {image_path}: {e}")
            return None

    def _save_detection_image_results(self, validation_result_id: int, epoch: int,
                                     image_results: list, class_names: list):
        """
        Save detection image results to database in framework-independent format.

        Args:
            validation_result_id: Validation result ID
            epoch: Epoch number
            image_results: List of dicts with image_path, true_boxes, predicted_boxes
            class_names: List of class names
        """
        import sqlite3
        import json
        from pathlib import Path
        from datetime import datetime

        try:
            # Get database path
            training_dir = Path(__file__).parent.parent
            mvp_dir = training_dir.parent
            db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

            if not db_path.exists():
                print(f"[WARNING] Database not found at {db_path}")
                return

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            records = []
            for img_result in image_results:
                # Determine primary class from boxes
                true_label_id = None
                predicted_label_id = None
                confidence = None
                is_correct = False

                if img_result['true_boxes']:
                    true_label_id = img_result['true_boxes'][0]['class_id']

                if img_result['predicted_boxes']:
                    # Get highest confidence prediction
                    best_pred = max(img_result['predicted_boxes'], key=lambda x: x['confidence'])
                    predicted_label_id = best_pred['class_id']
                    confidence = best_pred['confidence']

                    # Simple correctness: if any true box matches any predicted class
                    true_classes = {box['class_id'] for box in img_result['true_boxes']}
                    pred_classes = {box['class_id'] for box in img_result['predicted_boxes']}
                    is_correct = bool(true_classes & pred_classes)

                true_label = class_names[true_label_id] if true_label_id is not None and class_names and true_label_id < len(class_names) else None
                predicted_label = class_names[predicted_label_id] if predicted_label_id is not None and class_names and predicted_label_id < len(class_names) else None

                records.append((
                    validation_result_id,
                    self.job_id,
                    epoch,
                    img_result['image_path'],
                    img_result['image_name'],
                    img_result['image_index'],
                    true_label,
                    true_label_id,
                    predicted_label,
                    predicted_label_id,
                    confidence,
                    None,  # top5_predictions (not applicable for detection)
                    json.dumps(img_result['true_boxes']),
                    json.dumps(img_result['predicted_boxes']),
                    None,  # true_mask_path
                    None,  # predicted_mask_path
                    None,  # true_keypoints
                    None,  # predicted_keypoints
                    is_correct,
                    None,  # iou (will calculate if needed)
                    None,  # oks
                    None,  # extra_data
                    datetime.utcnow().isoformat()
                ))

            # Batch insert
            cursor.executemany(
                """
                INSERT INTO validation_image_results
                (validation_result_id, job_id, epoch, image_path, image_name, image_index,
                 true_label, true_label_id, predicted_label, predicted_label_id, confidence,
                 top5_predictions, true_boxes, predicted_boxes, true_mask_path, predicted_mask_path,
                 true_keypoints, predicted_keypoints, is_correct, iou, oks, extra_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )

            conn.commit()
            conn.close()

            print(f"[YOLO] Saved {len(records)} detection image results to database (epoch {epoch})")
            sys.stdout.flush()

        except Exception as e:
            print(f"[WARNING] Failed to save detection image results: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    def save_checkpoint(self, epoch: int, metrics: MetricsResult) -> str:
        """
        YOLO automatically saves checkpoints.

        Returns path to best checkpoint.
        """
        best_weights = os.path.join(self.output_dir, f'job_{self.job_id}', 'weights', 'best.pt')
        last_weights = os.path.join(self.output_dir, f'job_{self.job_id}', 'weights', 'last.pt')

        if os.path.exists(best_weights):
            return best_weights
        elif os.path.exists(last_weights):
            return last_weights
        else:
            return f"{self.output_dir}/job_{self.job_id}/weights/"

    # ========== Inference Methods ==========

    def load_checkpoint(
        self,
        checkpoint_path: str,
        inference_mode: bool = True,
        device: Optional[str] = None
    ) -> None:
        """
        Load checkpoint for inference or training resume.

        For YOLO, this creates a new model instance with the checkpoint weights.
        For YOLO-World, also sets custom classes if provided.

        Args:
            checkpoint_path: Path to checkpoint file (.pt)
            inference_mode: If True, load for inference (eval mode)
                           If False, load for training resume
            device: Device to load model on ('cuda', 'cpu'), auto-detect if None
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Check if this is YOLO-World
        is_yolo_world = self.task_type == TaskType.ZERO_SHOT_DETECTION

        print(f"\n{'='*80}")
        print(f"LOADING CHECKPOINT ({'YOLO-World' if is_yolo_world else 'YOLO'})")
        print(f"{'='*80}")
        print(f"[CHECKPOINT] Path: {checkpoint_path}")
        print(f"[CHECKPOINT] Mode: {'Inference' if inference_mode else 'Training Resume'}")

        # Import appropriate model class
        if is_yolo_world:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(checkpoint_path)

            # Set custom classes for YOLO-World
            if hasattr(self.model_config, 'custom_prompts') and self.model_config.custom_prompts:
                print(f"[CHECKPOINT] Setting custom classes: {self.model_config.custom_prompts}")
                self.model.set_classes(self.model_config.custom_prompts)
                # Store class names for inference
                self.class_names = self.model_config.custom_prompts
                print(f"[CHECKPOINT] Custom classes set")
            else:
                print("[CHECKPOINT] WARNING: No custom_prompts for YOLO-World inference")
        else:
            from ultralytics import YOLO
            self.model = YOLO(checkpoint_path)

        # Set device if specified
        if device:
            self.model.to(device)
            print(f"[CHECKPOINT] Device: {device}")
        else:
            print(f"[CHECKPOINT] Device: {self.model.device}")

        print(f"[CHECKPOINT] Model loaded successfully")
        print(f"{'='*80}\n")

    def preprocess_image(self, image_path: str):
        """
        Preprocess not needed for YOLO - handles internally.

        Args:
            image_path: Path to image file

        Returns:
            Image path as-is (YOLO processes internally)
        """
        return image_path

    def infer_single(self, image_path: str) -> 'InferenceResult':
        """
        Run inference on single image with YOLO.

        Args:
            image_path: Path to image file

        Returns:
            InferenceResult with task-specific predictions
        """
        import time
        from pathlib import Path
        from .base import InferenceResult, TaskType

        # YOLO inference
        start_time = time.time()
        print(f"[DEBUG] Running YOLO inference on {image_path}", file=sys.stderr)
        print(f"[DEBUG] Model: {self.model}", file=sys.stderr)
        print(f"[DEBUG] Task type: {self.task_type}", file=sys.stderr)
        print(f"[DEBUG] Image path exists: {Path(image_path).exists()}", file=sys.stderr)

        try:
            print(f"[DEBUG] Calling self.model('{image_path}', verbose=False)...", file=sys.stderr)
            results = self.model(image_path, verbose=False)
            print(f"[DEBUG] YOLO call completed", file=sys.stderr)
            print(f"[DEBUG] Results type: {type(results)}, length: {len(results) if results else 0}", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] YOLO inference failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

        inference_time = (time.time() - start_time) * 1000

        if not results or len(results) == 0:
            raise RuntimeError(f"YOLO returned empty results for {image_path}")

        result = results[0]
        print(f"[DEBUG] Result object: {result}", file=sys.stderr)
        print(f"[DEBUG] Result boxes: {result.boxes if hasattr(result, 'boxes') else 'N/A'}", file=sys.stderr)

        # Extract predictions based on task
        print(f"[DEBUG] About to extract predictions for task: {self.task_type}", file=sys.stderr)
        print(f"[DEBUG] self.task_type type: {type(self.task_type)}", file=sys.stderr)

        # Convert task_type to string for comparison (handles both enum and string cases)
        task_type_str = self.task_type.value if hasattr(self.task_type, 'value') else str(self.task_type)
        print(f"[DEBUG] task_type_str: {task_type_str}", file=sys.stderr)

        if task_type_str == 'object_detection':
            print(f"[DEBUG] Calling _extract_detection_result", file=sys.stderr)
            return self._extract_detection_result(result, image_path, inference_time)
        elif task_type_str == 'instance_segmentation':
            print(f"[DEBUG] Calling _extract_segmentation_result", file=sys.stderr)
            return self._extract_segmentation_result(result, image_path, inference_time)
        elif task_type_str == 'pose_estimation':
            print(f"[DEBUG] Calling _extract_pose_result", file=sys.stderr)
            return self._extract_pose_result(result, image_path, inference_time)
        elif task_type_str == 'image_classification':
            print(f"[DEBUG] Calling _extract_classification_result", file=sys.stderr)
            return self._extract_classification_result(result, image_path, inference_time)
        else:
            print(f"[ERROR] Unsupported task type: {task_type_str}", file=sys.stderr)
            raise ValueError(f"Unsupported task type: {task_type_str}")

    def _extract_classification_result(
        self,
        result,
        image_path: str,
        inference_time: float
    ) -> 'InferenceResult':
        """Extract classification predictions from YOLO result."""
        from pathlib import Path
        from .base import InferenceResult, TaskType

        # Get probabilities
        probs = result.probs

        if probs is None:
            raise ValueError("No classification predictions found")

        # Top-1 prediction
        top1_id = int(probs.top1)
        confidence = float(probs.top1conf)

        # Get class name
        if hasattr(self, 'class_names') and self.class_names:
            predicted_label = self.class_names[top1_id]
        else:
            predicted_label = str(top1_id)

        # Top-5 predictions
        top5_predictions = []
        if hasattr(probs, 'top5'):
            top5_ids = probs.top5
            top5_conf = probs.top5conf

            for i, (class_id, conf) in enumerate(zip(top5_ids, top5_conf)):
                class_id = int(class_id)
                if hasattr(self, 'class_names') and self.class_names:
                    label = self.class_names[class_id]
                else:
                    label = str(class_id)

                top5_predictions.append({
                    'label_id': class_id,
                    'label': label,
                    'confidence': float(conf)
                })

        return InferenceResult(
            image_path=image_path,
            image_name=Path(image_path).name,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            predicted_label=predicted_label,
            predicted_label_id=top1_id,
            confidence=confidence,
            top5_predictions=top5_predictions,
            inference_time_ms=inference_time,
            preprocessing_time_ms=0.0,  # YOLO handles internally
            postprocessing_time_ms=0.0
        )

    def _extract_detection_result(
        self,
        result,
        image_path: str,
        inference_time: float
    ) -> 'InferenceResult':
        """Extract detection predictions from YOLO result."""
        from pathlib import Path
        from .base import InferenceResult, TaskType

        print(f"[DEBUG] _extract_detection_result called", file=sys.stderr)

        # Extract boxes
        boxes = result.boxes
        predicted_boxes = []

        print(f"[DEBUG] boxes: {boxes}", file=sys.stderr)
        print(f"[DEBUG] boxes is not None: {boxes is not None}", file=sys.stderr)
        print(f"[DEBUG] len(boxes): {len(boxes) if boxes is not None else 'N/A'}", file=sys.stderr)

        if boxes is not None and len(boxes) > 0:
            print(f"[DEBUG] Processing {len(boxes)} boxes", file=sys.stderr)
            for i in range(len(boxes)):
                box = boxes[i]
                class_id = int(box.cls.item())

                # Get class name
                if hasattr(self, 'class_names') and self.class_names and class_id < len(self.class_names):
                    label = self.class_names[class_id]
                else:
                    label = str(class_id)

                # Convert YOLO format (x_center, y_center, w, h) to xyxy format
                xywh = box.xywh[0].cpu().tolist()
                xyxy = box.xyxy[0].cpu().tolist()  # [x1, y1, x2, y2]

                predicted_boxes.append({
                    'class_id': class_id,
                    'label': label,
                    'bbox': xywh,  # Keep original YOLO format
                    'x1': xyxy[0],
                    'y1': xyxy[1],
                    'x2': xyxy[2],
                    'y2': xyxy[3],
                    'confidence': float(box.conf.item()),
                    'format': 'yolo'
                })
        else:
            print(f"[DEBUG] No boxes detected (boxes is None or empty)", file=sys.stderr)

        print(f"[DEBUG] predicted_boxes count: {len(predicted_boxes)}", file=sys.stderr)
        print(f"[DEBUG] Creating InferenceResult...", file=sys.stderr)

        inf_result = InferenceResult(
            image_path=image_path,
            image_name=Path(image_path).name,
            task_type=TaskType.OBJECT_DETECTION,
            predicted_boxes=predicted_boxes,
            inference_time_ms=inference_time,
            preprocessing_time_ms=0.0,
            postprocessing_time_ms=0.0
        )

        print(f"[DEBUG] InferenceResult created: {inf_result}", file=sys.stderr)
        print(f"[DEBUG] Returning from _extract_detection_result", file=sys.stderr)

        return inf_result

    def _extract_segmentation_result(
        self,
        result,
        image_path: str,
        inference_time: float
    ) -> 'InferenceResult':
        """Extract segmentation predictions from YOLO result."""
        from pathlib import Path
        from .base import InferenceResult, TaskType

        # Extract boxes and masks
        boxes = result.boxes
        masks = result.masks

        predicted_boxes = []
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                class_id = int(box.cls.item())

                # Get class name
                if hasattr(self, 'class_names') and self.class_names and class_id < len(self.class_names):
                    label = self.class_names[class_id]
                else:
                    label = str(class_id)

                xywh = box.xywh[0].cpu().tolist()
                xyxy = box.xyxy[0].cpu().tolist()

                predicted_boxes.append({
                    'class_id': class_id,
                    'label': label,
                    'bbox': xywh,
                    'x1': xyxy[0],
                    'y1': xyxy[1],
                    'x2': xyxy[2],
                    'y2': xyxy[3],
                    'confidence': float(box.conf.item()),
                    'format': 'yolo'
                })

        # Extract mask polygons if available
        predicted_masks = []
        print(f"[DEBUG SEG] masks is not None: {masks is not None}", file=sys.stderr)
        if masks is not None:
            print(f"[DEBUG SEG] len(masks): {len(masks)}", file=sys.stderr)

        if masks is not None and len(masks) > 0:
            print(f"[DEBUG SEG] masks has xy: {hasattr(masks, 'xy')}", file=sys.stderr)
            print(f"[DEBUG SEG] masks.xy is not None: {masks.xy is not None}", file=sys.stderr)

            # Convert masks to polygon format
            for i in range(len(masks)):
                if hasattr(masks, 'xy') and masks.xy is not None:
                    # Get polygon coordinates (list of [x, y] points)
                    polygon = masks.xy[i]
                    print(f"[DEBUG SEG] Mask {i}: polygon points = {len(polygon) if polygon is not None else 0}", file=sys.stderr)

                    if polygon is not None and len(polygon) > 0:
                        predicted_masks.append({
                            'instance_id': i,
                            'class_id': predicted_boxes[i]['class_id'] if i < len(predicted_boxes) else 0,
                            'polygon': polygon.tolist(),  # Convert numpy array to list
                            'confidence': predicted_boxes[i]['confidence'] if i < len(predicted_boxes) else 0.0
                        })

        print(f"[DEBUG SEG] Total predicted_masks: {len(predicted_masks)}", file=sys.stderr)

        return InferenceResult(
            image_path=image_path,
            image_name=Path(image_path).name,
            task_type=TaskType.INSTANCE_SEGMENTATION,
            predicted_boxes=predicted_boxes,
            predicted_masks=predicted_masks if predicted_masks else None,
            inference_time_ms=inference_time,
            preprocessing_time_ms=0.0,
            postprocessing_time_ms=0.0
        )

    def _extract_pose_result(
        self,
        result,
        image_path: str,
        inference_time: float
    ) -> 'InferenceResult':
        """Extract pose estimation predictions from YOLO result."""
        from pathlib import Path
        from .base import InferenceResult, TaskType

        # Extract boxes and keypoints
        boxes = result.boxes
        keypoints = result.keypoints

        predicted_boxes = []
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                predicted_boxes.append({
                    'class_id': int(box.cls.item()),
                    'bbox': box.xywh[0].cpu().tolist(),
                    'confidence': float(box.conf.item()),
                    'format': 'yolo'
                })

        # Extract keypoints
        predicted_keypoints = []
        if keypoints is not None and len(keypoints) > 0:
            # keypoints.xy is (N, num_keypoints, 2)
            # keypoints.conf is (N, num_keypoints)
            for i in range(len(keypoints)):
                kpts_xy = keypoints.xy[i].cpu().numpy()  # (num_kpts, 2)
                kpts_conf = keypoints.conf[i].cpu().numpy()  # (num_kpts,)

                person_keypoints = []
                for j, (xy, conf) in enumerate(zip(kpts_xy, kpts_conf)):
                    person_keypoints.append({
                        'keypoint_id': j,
                        'x': float(xy[0]),
                        'y': float(xy[1]),
                        'confidence': float(conf)
                    })

                predicted_keypoints.append({
                    'person_id': i,
                    'keypoints': person_keypoints
                })

        return InferenceResult(
            image_path=image_path,
            image_name=Path(image_path).name,
            task_type=TaskType.POSE_ESTIMATION,
            predicted_boxes=predicted_boxes,
            predicted_keypoints=predicted_keypoints,
            inference_time_ms=inference_time,
            preprocessing_time_ms=0.0,
            postprocessing_time_ms=0.0
        )
