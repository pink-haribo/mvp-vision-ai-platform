"""Ultralytics YOLO adapter for object detection, segmentation, and pose estimation."""

import os
import sys
import yaml
from typing import List, Dict, Any
from .base import TrainingAdapter, MetricsResult, TaskType, DatasetFormat, ConfigSchema, ConfigField


class UltralyticsAdapter(TrainingAdapter):
    """
    Adapter for Ultralytics YOLO models.

    Supported tasks:
    - Object Detection (yolov8n.pt, yolov9e.pt)
    - Instance Segmentation (yolov8n-seg.pt)
    - Pose Estimation (yolov8n-pose.pt)
    - Classification (yolov8n-cls.pt)
    - OBB (yolov8n-obb.pt)
    """

    TASK_SUFFIX_MAP = {
        TaskType.OBJECT_DETECTION: "",
        TaskType.INSTANCE_SEGMENTATION: "-seg",
        TaskType.POSE_ESTIMATION: "-pose",
        TaskType.IMAGE_CLASSIFICATION: "-cls",
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
        """Initialize YOLO model."""
        print("[prepare_model] Step 1: Importing ultralytics...")
        sys.stdout.flush()

        try:
            from ultralytics import YOLO
            print("[prepare_model] ultralytics imported successfully")
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

        # Determine model path based on task
        suffix = self.TASK_SUFFIX_MAP.get(self.task_type, "")
        model_path = f"{self.model_config.model_name}{suffix}.pt"

        print(f"[prepare_model] Step 2: Loading YOLO model: {model_path}")
        print(f"[prepare_model] Task type: {self.task_type}")
        print(f"[prepare_model] Suffix: '{suffix}'")
        sys.stdout.flush()

        try:
            print(f"[prepare_model] About to call YOLO('{model_path}')...")
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

    def prepare_dataset(self):
        """Prepare dataset in YOLO format."""
        # Clear any existing YOLO cache files to avoid stale data issues
        self._clear_yolo_cache()

        # YOLO requires data.yaml file
        self.data_yaml = self._create_data_yaml()
        print(f"Dataset config created: {self.data_yaml}")

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
        # Check if data.yaml already exists in dataset
        existing_yaml = os.path.join(self.dataset_config.dataset_path, "data.yaml")
        if os.path.exists(existing_yaml):
            print(f"Using existing data.yaml: {existing_yaml}")
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

            # YOLO format - create data.yaml
            data = {
                'path': os.path.abspath(self.dataset_config.dataset_path),
                'train': 'images/train',
                'val': 'images/val',
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
            print("[YOLO] Building training arguments...")
            train_args = self._build_yolo_train_args()
            print(f"[YOLO] Training args built: {list(train_args.keys())}")

            # Print actual values for debugging
            print("[YOLO] Training arguments values:")
            for key, value in train_args.items():
                print(f"  {key}: {value}")
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

        # Start training (creates MLflow run)
        callbacks.on_train_begin()

        # Define real-time callback for YOLO training
        def on_yolo_epoch_end(trainer):
            """
            YOLO callback for real-time metric collection.

            Called at the end of each training epoch by Ultralytics.
            Extracts metrics from trainer and reports to TrainingCallbacks.
            """
            try:
                epoch = trainer.epoch

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
                    # 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', etc.
                    val_metrics = trainer.metrics

                    # Extract validation losses and metrics
                    for key, value in val_metrics.items():
                        if isinstance(value, (int, float)):
                            # Clean up metric names
                            clean_key = key.replace('metrics/', '').replace('(B)', '')
                            metrics_dict[clean_key] = value

                # Calculate total losses
                train_loss = sum(v for k, v in metrics_dict.items() if k.startswith('train/') and 'loss' in k)
                val_loss = sum(v for k, v in metrics_dict.items() if k.startswith('val/') and 'loss' in k)

                # Add summary metrics
                metrics_dict['train_loss'] = train_loss if train_loss > 0 else metrics_dict.get('train/box_loss', 0)
                metrics_dict['val_loss'] = val_loss if val_loss > 0 else metrics_dict.get('val/box_loss', 0)

                # Report to callbacks for unified metric collection (MLflow + Database)
                print(f"[YOLO Callback] Epoch {epoch} completed, reporting metrics to callbacks...")
                sys.stdout.flush()
                callbacks.on_epoch_end(epoch, metrics_dict)

            except Exception as e:
                print(f"[YOLO Callback ERROR] Failed to collect metrics at epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()

        # Register callback with YOLO model
        self.model.add_callback("on_fit_epoch_end", on_yolo_epoch_end)
        print("[YOLO] Registered real-time metric collection callback")
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
            'verbose': True,
            'workers': 0,  # Disable multiprocessing to avoid output issues
            'plots': False,  # Disable plotting
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
                train_cls_loss = float(row.get('train/cls_loss', 0))
                train_dfl_loss = float(row.get('train/dfl_loss', 0))
                val_box_loss = float(row.get('val/box_loss', 0))
                val_cls_loss = float(row.get('val/cls_loss', 0))
                val_dfl_loss = float(row.get('val/dfl_loss', 0))
                precision = float(row.get('metrics/precision(B)', 0))
                recall = float(row.get('metrics/recall(B)', 0))
                mAP50 = float(row.get('metrics/mAP50(B)', 0))
                mAP50_95 = float(row.get('metrics/mAP50-95(B)', 0))

                # Calculate total losses
                train_loss = train_box_loss + train_cls_loss + train_dfl_loss
                val_loss = val_box_loss + val_cls_loss + val_dfl_loss

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
                    'precision': precision,
                    'recall': recall,
                    'mAP50': mAP50,
                    'mAP50-95': mAP50_95,
                }

                # Report to callbacks for unified metric collection
                # Callbacks will handle both MLflow and database logging
                if callbacks:
                    callbacks.on_epoch_end(epoch - 1, metrics_dict)  # epoch is 1-indexed in CSV

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
        """Not used - YOLO handles validation internally."""
        pass

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
