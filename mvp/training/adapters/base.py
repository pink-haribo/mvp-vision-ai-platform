"""Base adapter interface for training frameworks."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class TaskType(Enum):
    """Supported task types."""
    # Vision
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"

    # Vision-Language
    IMAGE_CAPTIONING = "image_captioning"
    VISUAL_QA = "visual_qa"
    OCR = "ocr"
    DOCUMENT_UNDERSTANDING = "document_understanding"

    # Zero-Shot
    ZERO_SHOT_CLASSIFICATION = "zero_shot_classification"
    ZERO_SHOT_DETECTION = "zero_shot_detection"

    # Video
    VIDEO_CLASSIFICATION = "video_classification"
    VIDEO_DETECTION = "video_detection"


# Metric definitions for each task type
@dataclass
class MetricDefinition:
    """Definition of a metric for display purposes."""
    label: str  # Display label
    format: str  # 'percent', 'float', 'int'
    higher_is_better: bool  # True if higher values are better
    description: str = ""  # Optional description


# Primary metric for each task (used for best model selection)
TASK_PRIMARY_METRICS = {
    TaskType.IMAGE_CLASSIFICATION: 'accuracy',
    TaskType.OBJECT_DETECTION: 'mAP50',
    TaskType.INSTANCE_SEGMENTATION: 'mAP50',
    TaskType.SEMANTIC_SEGMENTATION: 'miou',
    TaskType.POSE_ESTIMATION: 'pck',
}


# Standard metrics for each task (displayed prominently in UI)
TASK_STANDARD_METRICS = {
    TaskType.IMAGE_CLASSIFICATION: {
        'accuracy': MetricDefinition(
            label='Accuracy',
            format='percent',
            higher_is_better=True,
            description='Top-1 accuracy on validation set'
        ),
        'top5_accuracy': MetricDefinition(
            label='Top-5 Accuracy',
            format='percent',
            higher_is_better=True,
            description='Top-5 accuracy on validation set'
        ),
        'train_loss': MetricDefinition(
            label='Train Loss',
            format='float',
            higher_is_better=False,
            description='Training loss'
        ),
        'val_loss': MetricDefinition(
            label='Validation Loss',
            format='float',
            higher_is_better=False,
            description='Validation loss'
        ),
    },

    TaskType.OBJECT_DETECTION: {
        'mAP50': MetricDefinition(
            label='mAP@0.5',
            format='percent',
            higher_is_better=True,
            description='Mean Average Precision at IoU threshold 0.5'
        ),
        'mAP50-95': MetricDefinition(
            label='mAP@[0.5:0.95]',
            format='percent',
            higher_is_better=True,
            description='Mean Average Precision averaged over IoU thresholds 0.5 to 0.95'
        ),
        'precision': MetricDefinition(
            label='Precision',
            format='percent',
            higher_is_better=True,
            description='Detection precision'
        ),
        'recall': MetricDefinition(
            label='Recall',
            format='percent',
            higher_is_better=True,
            description='Detection recall'
        ),
        'train_box_loss': MetricDefinition(
            label='Train Box Loss',
            format='float',
            higher_is_better=False,
            description='Training bounding box loss'
        ),
        'train_cls_loss': MetricDefinition(
            label='Train Class Loss',
            format='float',
            higher_is_better=False,
            description='Training classification loss'
        ),
        'val_box_loss': MetricDefinition(
            label='Val Box Loss',
            format='float',
            higher_is_better=False,
            description='Validation bounding box loss'
        ),
        'val_cls_loss': MetricDefinition(
            label='Val Class Loss',
            format='float',
            higher_is_better=False,
            description='Validation classification loss'
        ),
    },

    TaskType.SEMANTIC_SEGMENTATION: {
        'miou': MetricDefinition(
            label='Mean IoU',
            format='percent',
            higher_is_better=True,
            description='Mean Intersection over Union'
        ),
        'pixel_accuracy': MetricDefinition(
            label='Pixel Accuracy',
            format='percent',
            higher_is_better=True,
            description='Per-pixel classification accuracy'
        ),
        'dice': MetricDefinition(
            label='Dice Coefficient',
            format='percent',
            higher_is_better=True,
            description='Dice similarity coefficient'
        ),
        'train_loss': MetricDefinition(
            label='Train Loss',
            format='float',
            higher_is_better=False,
            description='Training loss'
        ),
        'val_loss': MetricDefinition(
            label='Validation Loss',
            format='float',
            higher_is_better=False,
            description='Validation loss'
        ),
    },
}


def get_task_primary_metric(task_type: TaskType) -> str:
    """Get primary metric name for a task type."""
    return TASK_PRIMARY_METRICS.get(task_type, 'accuracy')


def get_task_standard_metrics(task_type: TaskType) -> Dict[str, MetricDefinition]:
    """Get standard metrics definitions for a task type."""
    return TASK_STANDARD_METRICS.get(task_type, {})


class DatasetFormat(Enum):
    """Dataset format types."""
    IMAGE_FOLDER = "imagefolder"  # Classification: folder per class
    COCO = "coco"  # Detection, Segmentation: JSON annotations
    YOLO = "yolo"  # YOLO format: txt annotations
    PASCAL_VOC = "voc"  # Pascal VOC XML format
    CUSTOM = "custom"  # Custom format


@dataclass
class ConfigField:
    """Describes a single configuration field for dynamic UI generation."""
    name: str
    type: str  # 'int', 'float', 'str', 'bool', 'select', 'multiselect'
    default: Any
    description: str
    required: bool = False

    # For select/multiselect
    options: Optional[List[str]] = None

    # For numeric types
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    # UI organization
    group: Optional[str] = None  # 'optimizer', 'scheduler', 'augmentation', etc.
    advanced: bool = False  # Show in advanced settings only

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'type': self.type,
            'default': self.default,
            'description': self.description,
            'required': self.required,
            'options': self.options,
            'min': self.min,
            'max': self.max,
            'step': self.step,
            'group': self.group,
            'advanced': self.advanced
        }


@dataclass
class ConfigSchema:
    """Model's complete configuration schema."""
    fields: List[ConfigField] = field(default_factory=list)
    presets: Optional[Dict[str, Dict[str, Any]]] = None  # 'easy', 'medium', 'advanced'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'fields': [f.to_dict() for f in self.fields],
            'presets': self.presets
        }


@dataclass
class ModelConfig:
    """Model configuration."""
    framework: str  # 'timm', 'ultralytics', 'transformers'
    task_type: TaskType
    model_name: str
    pretrained: bool = True
    num_classes: Optional[int] = None
    image_size: Union[int, tuple] = 224
    custom_config: Optional[Dict[str, Any]] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    dataset_path: str
    format: DatasetFormat
    train_split: str = "train"
    val_split: str = "val"
    test_split: Optional[str] = None
    augmentation: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    device: str = "cuda"
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    early_stopping: Optional[Dict] = None

    # Advanced configuration (optional)
    advanced_config: Optional[Dict[str, Any]] = None  # Contains OptimizerConfig, SchedulerConfig, etc.


@dataclass
class MetricsResult:
    """
    Training metrics result (standardized format).

    Task-specific metrics examples:
    - Classification: accuracy, top5_accuracy, f1_score
    - Detection: mAP, mAP50, mAP75, precision, recall
    - Segmentation: mIoU, pixel_accuracy
    - OCR: CER, WER, BLEU
    """
    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class TrainingAdapter(ABC):
    """
    Base adapter for training frameworks.

    Implementations:
    - TimmAdapter: timm (Image Classification)
    - UltralyticsAdapter: YOLOv8/v9 (Detection, Segmentation, Pose)
    - TransformersAdapter: HuggingFace (All Vision-Language tasks)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        output_dir: str,
        job_id: int
    ):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.output_dir = output_dir
        self.job_id = job_id

        self.model = None
        self.train_loader = None
        self.val_loader = None

    # ========== Configuration Schema ==========

    @classmethod
    def get_config_schema(cls) -> ConfigSchema:
        """
        Return configuration schema for this adapter.

        Override in subclasses to provide framework-specific configuration options.

        Returns:
            ConfigSchema with fields for dynamic UI generation
        """
        # Default implementation returns empty schema
        # Subclasses should override this method
        return ConfigSchema(fields=[], presets=None)

    # ========== Required Methods ==========

    @abstractmethod
    def prepare_model(self) -> None:
        """
        Initialize model.

        Examples:
        - timm: timm.create_model()
        - ultralytics: YOLO()
        - transformers: AutoModel.from_pretrained()
        """
        pass

    @abstractmethod
    def prepare_dataset(self) -> None:
        """
        Load and preprocess dataset.

        - Load data
        - Apply augmentation
        - Create DataLoader
        """
        pass

    @abstractmethod
    def train_epoch(self, epoch: int) -> MetricsResult:
        """
        Train one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            MetricsResult: Training metrics
        """
        pass

    @abstractmethod
    def validate(self, epoch: int) -> MetricsResult:
        """
        Run validation.

        Args:
            epoch: Current epoch number

        Returns:
            MetricsResult: Validation metrics
        """
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int, metrics: MetricsResult) -> str:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Current metrics

        Returns:
            str: Path to saved checkpoint
        """
        pass

    # ========== Config Builder Methods ==========

    def build_optimizer(self, model_parameters):
        """
        Build optimizer from advanced config or fallback to basic config.

        Args:
            model_parameters: Model parameters to optimize

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        import torch.optim as optim

        # Get optimizer config from advanced_config or use defaults
        adv_config = self.training_config.advanced_config
        if adv_config and 'optimizer' in adv_config:
            opt_config = adv_config['optimizer']
            opt_type = opt_config.get('type', 'adam')
            lr = opt_config.get('learning_rate', self.training_config.learning_rate)
            weight_decay = opt_config.get('weight_decay', 0.0)

            if opt_type == 'adam':
                return optim.Adam(
                    model_parameters,
                    lr=lr,
                    betas=tuple(opt_config.get('betas', (0.9, 0.999))),
                    eps=opt_config.get('eps', 1e-8),
                    weight_decay=weight_decay,
                    amsgrad=opt_config.get('amsgrad', False)
                )
            elif opt_type == 'adamw':
                return optim.AdamW(
                    model_parameters,
                    lr=lr,
                    betas=tuple(opt_config.get('betas', (0.9, 0.999))),
                    eps=opt_config.get('eps', 1e-8),
                    weight_decay=weight_decay,
                    amsgrad=opt_config.get('amsgrad', False)
                )
            elif opt_type == 'sgd':
                return optim.SGD(
                    model_parameters,
                    lr=lr,
                    momentum=opt_config.get('momentum', 0.9),
                    weight_decay=weight_decay,
                    nesterov=opt_config.get('nesterov', False)
                )
            elif opt_type == 'rmsprop':
                return optim.RMSprop(
                    model_parameters,
                    lr=lr,
                    alpha=opt_config.get('alpha', 0.99),
                    eps=opt_config.get('eps', 1e-8),
                    weight_decay=weight_decay,
                    momentum=opt_config.get('momentum', 0.0)
                )
            elif opt_type == 'adagrad':
                return optim.Adagrad(
                    model_parameters,
                    lr=lr,
                    eps=opt_config.get('eps', 1e-10),
                    weight_decay=weight_decay
                )
            else:
                # Fallback to Adam
                return optim.Adam(model_parameters, lr=lr)
        else:
            # Use basic config
            return optim.Adam(model_parameters, lr=self.training_config.learning_rate)

    def build_scheduler(self, optimizer, num_training_steps: Optional[int] = None):
        """
        Build learning rate scheduler from advanced config.

        Args:
            optimizer: Optimizer instance
            num_training_steps: Total number of training steps (for OneCycleLR)

        Returns:
            torch.optim.lr_scheduler: Configured scheduler or None
        """
        import torch.optim.lr_scheduler as lr_scheduler

        adv_config = self.training_config.advanced_config
        if not adv_config or 'scheduler' not in adv_config:
            return None

        sched_config = adv_config['scheduler']
        sched_type = sched_config.get('type', 'none')

        if sched_type == 'none':
            return None
        elif sched_type == 'step':
            return lr_scheduler.StepLR(
                optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'multistep':
            return lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sched_config.get('milestones', [30, 60, 90]),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'exponential':
            return lr_scheduler.ExponentialLR(
                optimizer,
                gamma=sched_config.get('gamma', 0.95)
            )
        elif sched_type == 'cosine':
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_config.get('T_max', self.training_config.epochs),
                eta_min=sched_config.get('eta_min', 0.0)
            )
        elif sched_type == 'cosine_warm_restarts':
            return lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=sched_config.get('T_0', 10),
                T_mult=sched_config.get('T_mult', 2),
                eta_min=sched_config.get('eta_min', 0.0)
            )
        elif sched_type == 'reduce_on_plateau':
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sched_config.get('mode', 'min'),
                factor=sched_config.get('factor', 0.1),
                patience=sched_config.get('patience', 10),
                threshold=sched_config.get('threshold', 1e-4),
                cooldown=sched_config.get('cooldown', 0),
                min_lr=sched_config.get('min_lr', 0.0)
            )
        elif sched_type == 'one_cycle':
            if num_training_steps is None:
                num_training_steps = self.training_config.epochs
            return lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=sched_config.get('max_lr', 0.1),
                total_steps=num_training_steps,
                pct_start=sched_config.get('pct_start', 0.3),
                anneal_strategy=sched_config.get('anneal_strategy', 'cos')
            )
        else:
            return None

    def build_train_transforms(self):
        """
        Build training data augmentation transforms from advanced config.

        Returns:
            Compose: Composition of torchvision transforms
        """
        from torchvision import transforms

        adv_config = self.training_config.advanced_config
        transform_list = []

        # Get preprocessing config
        if adv_config and 'preprocessing' in adv_config:
            preprocess_config = adv_config['preprocessing']
            image_size = preprocess_config.get('image_size', 224)
            resize_mode = preprocess_config.get('resize_mode', 'resize')

            if resize_mode == 'resize':
                transform_list.append(transforms.Resize((image_size, image_size)))
            elif resize_mode == 'resize_crop':
                transform_list.append(transforms.Resize(int(image_size * 1.14)))
                transform_list.append(transforms.CenterCrop(image_size))
            elif resize_mode == 'pad':
                transform_list.append(transforms.Resize(image_size))
                transform_list.append(transforms.Pad(
                    padding=0,
                    fill=preprocess_config.get('pad_value', 0)
                ))
        else:
            # Default resize
            transform_list.append(transforms.Resize((224, 224)))

        # Get augmentation config
        if adv_config and 'augmentation' in adv_config:
            aug_config = adv_config['augmentation']

            if aug_config.get('enabled', True):
                # Random horizontal flip
                if aug_config.get('random_flip', False):
                    transform_list.append(transforms.RandomHorizontalFlip(
                        p=aug_config.get('random_flip_prob', 0.5)
                    ))

                # Random rotation
                if aug_config.get('random_rotation', False):
                    transform_list.append(transforms.RandomRotation(
                        degrees=aug_config.get('rotation_degrees', 15)
                    ))

                # Random crop
                if aug_config.get('random_crop', False):
                    image_size = adv_config.get('preprocessing', {}).get('image_size', 224)
                    transform_list.append(transforms.RandomResizedCrop(
                        size=image_size,
                        scale=tuple(aug_config.get('crop_scale', (0.8, 1.0))),
                        ratio=tuple(aug_config.get('crop_ratio', (0.75, 1.333)))
                    ))

                # Color jitter
                if aug_config.get('color_jitter', False):
                    transform_list.append(transforms.ColorJitter(
                        brightness=aug_config.get('brightness', 0.2),
                        contrast=aug_config.get('contrast', 0.2),
                        saturation=aug_config.get('saturation', 0.2),
                        hue=aug_config.get('hue', 0.1)
                    ))

                # Random grayscale
                if aug_config.get('random_grayscale', False):
                    transform_list.append(transforms.RandomGrayscale(
                        p=aug_config.get('grayscale_prob', 0.1)
                    ))

                # Gaussian blur
                if aug_config.get('gaussian_blur', False):
                    transform_list.append(transforms.GaussianBlur(
                        kernel_size=aug_config.get('blur_kernel_size', 5),
                        sigma=tuple(aug_config.get('blur_sigma', (0.1, 2.0)))
                    ))

                # AutoAugment
                if aug_config.get('autoaugment', False):
                    policy = aug_config.get('autoaugment_policy', 'imagenet')
                    if policy == 'imagenet':
                        transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
                    elif policy == 'cifar10':
                        transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
                    elif policy == 'svhn':
                        transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN))

        # Always convert to tensor and normalize
        transform_list.append(transforms.ToTensor())

        # Normalization
        if adv_config and 'preprocessing' in adv_config:
            preprocess_config = adv_config['preprocessing']
            if preprocess_config.get('normalize', True):
                transform_list.append(transforms.Normalize(
                    mean=list(preprocess_config.get('mean', (0.485, 0.456, 0.406))),
                    std=list(preprocess_config.get('std', (0.229, 0.224, 0.225)))
                ))
        else:
            # Default ImageNet normalization
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))

        # Random erasing (applied after ToTensor)
        if adv_config and 'augmentation' in adv_config:
            aug_config = adv_config['augmentation']
            if aug_config.get('random_erasing', False):
                transform_list.append(transforms.RandomErasing(
                    p=aug_config.get('erasing_prob', 0.5)
                ))

        return transforms.Compose(transform_list)

    def build_val_transforms(self):
        """
        Build validation data transforms (no augmentation).

        Returns:
            Compose: Composition of torchvision transforms
        """
        from torchvision import transforms

        adv_config = self.training_config.advanced_config
        transform_list = []

        # Get preprocessing config
        if adv_config and 'preprocessing' in adv_config:
            preprocess_config = adv_config['preprocessing']
            image_size = preprocess_config.get('image_size', 224)
            transform_list.append(transforms.Resize((image_size, image_size)))
        else:
            transform_list.append(transforms.Resize((224, 224)))

        # Always convert to tensor and normalize
        transform_list.append(transforms.ToTensor())

        # Normalization
        if adv_config and 'preprocessing' in adv_config:
            preprocess_config = adv_config['preprocessing']
            if preprocess_config.get('normalize', True):
                transform_list.append(transforms.Normalize(
                    mean=list(preprocess_config.get('mean', (0.485, 0.456, 0.406))),
                    std=list(preprocess_config.get('std', (0.229, 0.224, 0.225)))
                ))
        else:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))

        return transforms.Compose(transform_list)

    # ========== Common Methods (can be overridden) ==========

    def train(self, start_epoch: int = 0, checkpoint_path: str = None, resume_training: bool = False) -> List[MetricsResult]:
        """
        Full training process.

        1. Prepare model
        2. Prepare dataset
        3. Load checkpoint (if provided)
        4. For each epoch:
           - Train
           - Validate
           - Log metrics (MLflow via Callbacks)
           - Save checkpoint

        Args:
            start_epoch: Starting epoch number (for resuming training)
            checkpoint_path: Path to checkpoint file to load
            resume_training: If True, restore optimizer/scheduler state

        Returns:
            List[MetricsResult]: All metrics history
        """
        # 1. Setup
        self.prepare_model()
        self.prepare_dataset()

        # 2. Load checkpoint if provided
        if checkpoint_path:
            try:
                checkpoint_epoch = self.load_checkpoint(checkpoint_path, resume_training=resume_training)
                if resume_training:
                    start_epoch = checkpoint_epoch + 1
                    print(f"[INFO] Resuming training from epoch {start_epoch + 1}")
                else:
                    print(f"[INFO] Loaded model weights only, starting from epoch 1")
            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint: {e}")
                import traceback
                traceback.print_exc()
                raise

        all_metrics = []

        # 3. Initialize TrainingCallbacks
        callbacks = TrainingCallbacks(
            job_id=self.job_id,
            model_config=self.model_config,
            training_config=self.training_config,
            db_session=None  # No DB session in subprocess
        )

        # 4. Start training
        callbacks.on_train_begin()

        try:
            # 5. Training loop
            for epoch in range(start_epoch, self.training_config.epochs):
                print(f"\n[Epoch {epoch + 1}/{self.training_config.epochs}]")

                # Train
                train_metrics = self.train_epoch(epoch)

                # Validate
                val_metrics = self.validate(epoch)

                # Combine metrics
                combined_metrics = MetricsResult(
                    epoch=epoch,
                    step=train_metrics.step,
                    train_loss=train_metrics.train_loss,
                    val_loss=val_metrics.train_loss if val_metrics else None,
                    accuracy=val_metrics.metrics.get('val_accuracy') if val_metrics else train_metrics.metrics.get('train_accuracy'),
                    metrics={
                        'train_loss': train_metrics.train_loss,
                        **train_metrics.metrics,
                        **({'val_loss': val_metrics.train_loss} if val_metrics else {}),
                        **({f"val_{k}": v for k, v in (val_metrics.metrics if val_metrics else {}).items()})
                    }
                )

                all_metrics.append(combined_metrics)

                # Report metrics to callbacks (handles MLflow logging)
                callbacks.on_epoch_end(epoch, combined_metrics.metrics)

                # Save checkpoint periodically
                checkpoint_path = None
                if epoch % 5 == 0 or epoch == self.training_config.epochs - 1:
                    checkpoint_path = self.save_checkpoint(epoch, combined_metrics)
                    print(f"Checkpoint saved: {checkpoint_path}")

                    # Log checkpoint to MLflow
                    if checkpoint_path:
                        callbacks.log_artifact(checkpoint_path, "checkpoints")

                # Output metrics for backend parsing
                metrics_json = {
                    "epoch": epoch + 1,
                    "train_loss": combined_metrics.train_loss,
                    "train_accuracy": combined_metrics.metrics.get('train_accuracy', 0) / 100.0,  # Convert to 0-1 range
                    "val_loss": combined_metrics.val_loss,
                    "val_accuracy": combined_metrics.metrics.get('val_accuracy', 0) / 100.0 if 'val_accuracy' in combined_metrics.metrics else None,
                    "learning_rate": self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else None,
                    "checkpoint_path": checkpoint_path,
                }
                print(f"[METRICS] {json.dumps(metrics_json)}", flush=True)

            # End training
            final_metrics = all_metrics[-1].metrics if all_metrics else {}
            callbacks.on_train_end(final_metrics)

        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            callbacks.on_train_end()  # Close MLflow run even on error
            raise

        return all_metrics

    def log_metrics_to_mlflow(self, metrics: MetricsResult) -> None:
        """Log metrics to MLflow."""
        import mlflow

        log_dict = {
            "train_loss": metrics.train_loss,
        }

        if metrics.val_loss is not None:
            log_dict["val_loss"] = metrics.val_loss

        if metrics.metrics:
            log_dict.update(metrics.metrics)

        mlflow.log_metrics(log_dict, step=metrics.epoch)

        # Print summary
        print(f"  Train Loss: {metrics.train_loss:.4f}", end="")
        if metrics.val_loss:
            print(f" | Val Loss: {metrics.val_loss:.4f}", end="")

        # Print task-specific metrics
        for key, value in metrics.metrics.items():
            if not key.startswith('val_'):
                print(f" | {key}: {value:.4f}", end="")
        print()

    @property
    def task_type(self) -> TaskType:
        """Get task type."""
        return self.model_config.task_type


class TrainingCallbacks:
    """
    Standardized callback interface for metric collection.

    Provides unified way for all adapters to report metrics to:
    - MLflow (experiment tracking)
    - Database (job status and metrics)
    - WebSocket (real-time frontend updates)

    Usage in adapters:
        callbacks = TrainingCallbacks(job_id=123, db_session=db)
        callbacks.on_train_begin(config)

        for epoch in range(epochs):
            # ... training ...
            callbacks.on_epoch_end(epoch, {
                'train_loss': 0.234,
                'val_loss': 0.245,
                'accuracy': 0.95
            })

        callbacks.on_train_end({'final_accuracy': 0.96})
    """

    def __init__(self, job_id: int, model_config: 'ModelConfig',
                 training_config: 'TrainingConfig', db_session=None):
        """
        Initialize callbacks.

        Args:
            job_id: Training job ID
            model_config: Model configuration
            training_config: Training configuration
            db_session: Database session (optional, for DB updates)
        """
        self.job_id = job_id
        self.model_config = model_config
        self.training_config = training_config
        self.db_session = db_session
        self.mlflow_run = None
        self.mlflow_run_id = None
        self.mlflow_experiment_id = None

    def on_train_begin(self, config: Dict[str, Any] = None):
        """
        Called when training begins.

        Creates MLflow run and stores IDs in database.

        Args:
            config: Additional configuration to log
        """
        import mlflow
        import os

        # Create experiment name from job_id
        experiment_name = f"job_{self.job_id}"

        # Set or create experiment
        experiment = mlflow.set_experiment(experiment_name)
        self.mlflow_experiment_id = experiment.experiment_id

        # Start MLflow run
        run_name = f"{self.model_config.model_name}_training"
        self.mlflow_run = mlflow.start_run(run_name=run_name)
        self.mlflow_run_id = self.mlflow_run.info.run_id

        print(f"[Callbacks] MLflow run started:")
        print(f"  Experiment ID: {self.mlflow_experiment_id}")
        print(f"  Run ID: {self.mlflow_run_id}")

        # Log parameters to MLflow
        mlflow.log_param("model", self.model_config.model_name)
        mlflow.log_param("framework", self.model_config.framework)
        mlflow.log_param("task_type", self.model_config.task_type.value)
        mlflow.log_param("epochs", self.training_config.epochs)
        mlflow.log_param("batch_size", self.training_config.batch_size)
        mlflow.log_param("learning_rate", self.training_config.learning_rate)
        mlflow.log_param("image_size", self.model_config.image_size)
        mlflow.log_param("device", self.training_config.device or 'auto')

        if self.model_config.num_classes:
            mlflow.log_param("num_classes", self.model_config.num_classes)

        # Log additional config if provided
        if config:
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)

        # Update database with MLflow IDs
        if self.db_session:
            from app.db import models
            job = self.db_session.query(models.TrainingJob).filter(
                models.TrainingJob.id == self.job_id
            ).first()

            if job:
                job.mlflow_experiment_id = self.mlflow_experiment_id
                job.mlflow_run_id = self.mlflow_run_id
                self.db_session.commit()
                print(f"[Callbacks] Updated DB with MLflow IDs")

    def on_epoch_begin(self, epoch: int):
        """
        Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number
        """
        pass

    def on_batch_end(self, batch: int, metrics: Dict[str, float]):
        """
        Called after each batch (for real-time updates).

        Args:
            batch: Current batch number
            metrics: Batch metrics
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """
        Called at the end of each epoch.

        Automatically logs metrics to:
        - MLflow (with epoch as step)
        - Database (TrainingMetric table)
        - Console (formatted output)

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics, e.g.:
                {
                    'train_loss': 0.234,
                    'val_loss': 0.245,
                    'accuracy': 0.95,
                    'mAP50': 0.88,
                    ... any custom metrics
                }
        """
        import mlflow

        if not self.mlflow_run:
            print(f"[Callbacks] Warning: MLflow run not started, call on_train_begin() first")
            return

        # Log to MLflow
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=epoch)

        # Log to database
        if self.db_session:
            from app.db import models

            # Extract common metrics
            train_loss = metrics.get('train_loss') or metrics.get('loss')
            val_loss = metrics.get('val_loss')
            accuracy = metrics.get('accuracy') or metrics.get('mAP50')  # mAP50 for detection
            lr = metrics.get('learning_rate') or metrics.get('lr')

            # Store in database
            metric_record = models.TrainingMetric(
                job_id=self.job_id,
                epoch=epoch,
                step=epoch,
                loss=train_loss,
                accuracy=accuracy,
                learning_rate=lr,
                extra_metrics=metrics  # Store all metrics as JSON
            )
            self.db_session.add(metric_record)
            self.db_session.commit()

        # Console output
        print(f"[Callbacks] Epoch {epoch}: ", end="")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}={value:.4f} ", end="")
        print()

    def on_train_end(self, final_metrics: Dict[str, float] = None):
        """
        Called when training ends.

        Logs final metrics and closes MLflow run.

        Args:
            final_metrics: Final training metrics (optional)
        """
        import mlflow

        if not self.mlflow_run:
            print(f"[Callbacks] Warning: MLflow run not active")
            return

        # Log final metrics if provided
        if final_metrics:
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"final_{key}", value)

        # Update database with final accuracy
        if self.db_session and final_metrics:
            from app.db import models
            job = self.db_session.query(models.TrainingJob).filter(
                models.TrainingJob.id == self.job_id
            ).first()

            if job:
                final_acc = final_metrics.get('accuracy') or final_metrics.get('mAP50')
                if final_acc:
                    job.final_accuracy = final_acc
                self.db_session.commit()

        # End MLflow run
        mlflow.end_run()
        print(f"[Callbacks] MLflow run ended: {self.mlflow_run_id}")
        self.mlflow_run = None

    def log_artifact(self, file_path: str, artifact_path: str = None):
        """
        Log artifact (file) to MLflow.

        Args:
            file_path: Path to file to log
            artifact_path: Path within MLflow artifacts directory
        """
        import mlflow

        if self.mlflow_run:
            mlflow.log_artifact(file_path, artifact_path)

    def log_artifacts(self, dir_path: str, artifact_path: str = None):
        """
        Log directory of artifacts to MLflow.

        Args:
            dir_path: Path to directory to log
            artifact_path: Path within MLflow artifacts directory
        """
        import mlflow

        if self.mlflow_run:
            mlflow.log_artifacts(dir_path, artifact_path)
