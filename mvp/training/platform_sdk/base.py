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


@dataclass
class InferenceResult:
    """
    Single image inference result (task-agnostic).

    Supports all task types with optional fields.
    Adapters populate only relevant fields for their task.
    """
    image_path: str
    image_name: str
    task_type: TaskType

    # Classification fields
    predicted_label: Optional[str] = None
    predicted_label_id: Optional[int] = None
    confidence: Optional[float] = None
    top5_predictions: Optional[List[Dict[str, Any]]] = None

    # Detection fields
    predicted_boxes: Optional[List[Dict[str, Any]]] = None

    # Segmentation fields
    predicted_mask: Optional[Any] = None  # np.ndarray or torch.Tensor
    predicted_mask_path: Optional[str] = None

    # Pose estimation fields
    predicted_keypoints: Optional[List[Dict[str, Any]]] = None

    # Performance metrics
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0

    # Extra data for task-specific information
    extra_data: Optional[Dict[str, Any]] = None


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

    def _save_validation_result(
        self,
        epoch: int,
        validation_metrics: 'ValidationMetrics',
        checkpoint_path: Optional[str] = None
    ) -> Optional[int]:
        """
        Save validation result to database.

        This is a common method used by all adapters to store detailed validation
        metrics in the validation_results table. Adapters should call this method
        from their validate() implementation.

        Args:
            epoch: Current epoch number (1-indexed)
            validation_metrics: ValidationMetrics object from ValidationMetricsCalculator
            checkpoint_path: Optional path to the checkpoint file for this epoch

        Returns:
            Optional[int]: Validation result ID if saved successfully, None otherwise

        Example:
            # In adapter's validate() method:
            val_metrics = ValidationMetricsCalculator.compute_metrics(
                task_type=TaskType.CLASSIFICATION,
                predictions=all_preds,
                labels=all_labels,
                class_names=self.class_names
            )
            checkpoint_path = self.save_checkpoint(epoch, metrics)
            val_result_id = self._save_validation_result(epoch, val_metrics, checkpoint_path)
        """
        try:
            import sqlite3
            import json
            from pathlib import Path
            from datetime import datetime

            # Get database path
            training_dir = Path(__file__).parent.parent
            mvp_dir = training_dir.parent
            db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

            if not db_path.exists():
                print(f"[WARNING] Database not found at {db_path}, skipping validation result save")
                return

            # Get task-specific metrics
            task_metrics = validation_metrics.get_task_metrics()

            # Prepare common fields
            task_type = validation_metrics.task_type.value
            primary_metric_name = validation_metrics.primary_metric_name
            primary_metric_value = validation_metrics.primary_metric_value
            overall_loss = validation_metrics.overall_loss

            # Prepare task-specific fields
            metrics_json = json.dumps(task_metrics.to_dict())
            per_class_metrics_json = None
            confusion_matrix_json = None
            pr_curves_json = None
            class_names_json = None

            # Classification-specific
            if validation_metrics.classification:
                clf = validation_metrics.classification
                per_class_metrics_json = json.dumps(clf.per_class_to_dict())
                confusion_matrix_json = json.dumps(clf.confusion_matrix_to_list())
                if clf.class_names:
                    class_names_json = json.dumps(clf.class_names)

            # Detection/Segmentation-specific
            elif validation_metrics.detection:
                det = validation_metrics.detection
                if det.per_class_ap:
                    per_class_metrics_json = json.dumps(det.per_class_ap)
                if det.pr_curves:
                    pr_curves_json = json.dumps(det.pr_curves)
                # Store class_names from validation_metrics (set by adapter)
                if hasattr(validation_metrics, '_class_names') and validation_metrics._class_names:
                    class_names_json = json.dumps(validation_metrics._class_names)

            elif validation_metrics.segmentation:
                seg = validation_metrics.segmentation
                if seg.per_class_iou:
                    per_class_metrics_json = json.dumps(seg.per_class_iou)

            elif validation_metrics.pose:
                pose = validation_metrics.pose
                if pose.per_keypoint_pck:
                    per_class_metrics_json = json.dumps(pose.per_keypoint_pck)

            # Insert into database
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO validation_results
                (job_id, epoch, task_type, primary_metric_value, primary_metric_name,
                 overall_loss, metrics, per_class_metrics, confusion_matrix, pr_curves,
                 class_names, checkpoint_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.job_id,
                    epoch,
                    task_type,
                    primary_metric_value,
                    primary_metric_name,
                    overall_loss,
                    metrics_json,
                    per_class_metrics_json,
                    confusion_matrix_json,
                    pr_curves_json,
                    class_names_json,
                    checkpoint_path,
                    datetime.utcnow().isoformat()
                )
            )

            validation_result_id = cursor.lastrowid
            conn.commit()
            conn.close()

            print(f"[Validation] Saved validation result to database (epoch {epoch}, id={validation_result_id})")
            print(f"  Task: {task_type}")
            print(f"  Primary metric: {primary_metric_name}={primary_metric_value:.4f}")

            return validation_result_id

        except Exception as e:
            print(f"[WARNING] Failed to save validation result to database: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_validation_image_results(
        self,
        validation_result_id: int,
        epoch: int,
        image_results: List[Dict],
        class_names: Optional[List[str]] = None
    ) -> None:
        """
        Save per-image validation results to database.

        Args:
            validation_result_id: ID of the validation result
            epoch: Current epoch number
            image_results: List of per-image result dictionaries
            class_names: List of class names for label mapping
        """
        try:
            import sqlite3
            import json
            from pathlib import Path
            from datetime import datetime

            # Get database path
            training_dir = Path(__file__).parent.parent
            mvp_dir = training_dir.parent
            db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

            if not db_path.exists():
                print(f"[WARNING] Database not found at {db_path}, skipping image results save")
                return

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Prepare batch insert
            records = []
            for img_result in image_results:
                true_label = class_names[img_result['true_label_id']] if class_names else str(img_result['true_label_id'])
                predicted_label = class_names[img_result['predicted_label_id']] if class_names else str(img_result['predicted_label_id'])

                # Use actual image_path if available, otherwise None (nullable)
                image_path = img_result.get('image_path')
                image_name = img_result.get('image_name', f"image_{img_result['image_index']}")

                records.append((
                    validation_result_id,
                    self.job_id,
                    epoch,
                    image_path,
                    image_name,
                    img_result['image_index'],
                    true_label,
                    img_result['true_label_id'],
                    predicted_label,
                    img_result['predicted_label_id'],
                    img_result['confidence'],
                    json.dumps(img_result.get('top5_predictions')),
                    None,  # true_boxes
                    None,  # predicted_boxes
                    None,  # true_mask_path
                    None,  # predicted_mask_path
                    None,  # true_keypoints
                    None,  # predicted_keypoints
                    img_result['is_correct'],
                    None,  # iou
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

            # Count how many have actual paths vs placeholders
            with_paths = sum(1 for r in records if r[3] is not None)  # r[3] is image_path
            print(f"[Validation] Saved {len(records)} image results to database (epoch {epoch})")
            print(f"             - {with_paths} with actual image paths")
            print(f"             - {len(records) - with_paths} with placeholder names only")

        except Exception as e:
            print(f"[WARNING] Failed to save image results to database: {e}")
            import traceback
            traceback.print_exc()

    def _clear_validation_results(self) -> None:
        """
        Clear all validation results for the current job_id.

        This should be called when starting a new training (resume=False)
        to remove old validation data from previous training runs.
        """
        try:
            import sqlite3
            from pathlib import Path

            # Get database path
            training_dir = Path(__file__).parent.parent
            mvp_dir = training_dir.parent
            db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

            if not db_path.exists():
                print(f"[WARNING] Database not found at {db_path}, skipping validation clear")
                return

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Delete image results first (foreign key constraint)
            cursor.execute(
                "DELETE FROM validation_image_results WHERE job_id = ?",
                (self.job_id,)
            )
            deleted_images = cursor.rowcount

            # Delete validation results
            cursor.execute(
                "DELETE FROM validation_results WHERE job_id = ?",
                (self.job_id,)
            )
            deleted_results = cursor.rowcount

            conn.commit()
            conn.close()

            if deleted_results > 0 or deleted_images > 0:
                print(f"[INFO] Cleared previous validation data for job {self.job_id}:")
                print(f"       - {deleted_results} validation results")
                print(f"       - {deleted_images} image results")

        except Exception as e:
            print(f"[WARNING] Failed to clear validation results: {e}")
            import traceback
            traceback.print_exc()

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

    # ========== Inference Methods ==========

    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_path: str,
        inference_mode: bool = True,
        device: Optional[str] = None
    ) -> None:
        """
        Load checkpoint for inference or training resume.

        This method loads model weights and optionally optimizer/scheduler state.
        For inference, only model weights are loaded and model is set to eval mode.

        Args:
            checkpoint_path: Path to checkpoint file (.pth, .pt, etc.)
            inference_mode: If True, load for inference (eval mode, no optimizer)
                           If False, load for training resume (restore full state)
            device: Device to load model on ('cuda', 'cpu'), auto-detect if None

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint format is incompatible

        Example:
            # For inference
            adapter.load_checkpoint('best.pth', inference_mode=True)

            # For training resume
            adapter.load_checkpoint('epoch_10.pth', inference_mode=False)
        """
        pass

    @abstractmethod
    def preprocess_image(self, image_path: str) -> Any:
        """
        Preprocess single image for inference.

        Applies same preprocessing as validation (resize, normalize, etc.)
        but for a single image without labels.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor/array ready for model input
            Format depends on framework (torch.Tensor, np.ndarray, etc.)

        Example:
            tensor = adapter.preprocess_image('image.jpg')
            # Returns: torch.Tensor with shape (1, 3, H, W)
        """
        pass

    @abstractmethod
    def infer_single(self, image_path: str) -> InferenceResult:
        """
        Run inference on a single image.

        Core inference method - label-agnostic.
        Used by both TestRunner (with labels) and InferenceRunner (without labels).

        Args:
            image_path: Path to image file

        Returns:
            InferenceResult with task-specific predictions

        Example:
            # Classification
            result = adapter.infer_single('cat.jpg')
            # result.predicted_label = 'cat'
            # result.confidence = 0.95

            # Detection
            result = adapter.infer_single('street.jpg')
            # result.predicted_boxes = [
            #     {'class_id': 0, 'bbox': [x, y, w, h], 'confidence': 0.9},
            #     ...
            # ]
        """
        pass

    def infer_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> List[InferenceResult]:
        """
        Run batch inference on multiple images.

        Default implementation uses infer_single() in batches.
        Subclasses can override for optimized batching (e.g., DataLoader).

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for inference (for progress display)

        Returns:
            List of InferenceResult, one per image

        Example:
            results = adapter.infer_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
            for result in results:
                print(f"{result.image_name}: {result.predicted_label}")
        """
        results = []
        total = len(image_paths)

        for i in range(0, total, batch_size):
            batch = image_paths[i:i+batch_size]
            batch_end = min(i + batch_size, total)

            print(f"[Inference] Processing images {i+1}-{batch_end}/{total}")

            for img_path in batch:
                try:
                    result = self.infer_single(img_path)
                    results.append(result)
                except Exception as e:
                    print(f"[WARNING] Failed to process {img_path}: {e}")
                    # Continue with next image

        return results

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

        # 1.5. Clear previous validation data if not resuming
        if not resume_training:
            print("[INFO] Starting new training - clearing previous validation data")
            self._clear_validation_results()
        else:
            print("[INFO] Resuming training - keeping previous validation data")

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

        # 3.5. Get primary metric configuration from database
        primary_metric = None
        primary_metric_mode = None
        best_metric_value = None

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
                    # Initialize best_metric_value based on mode
                    if primary_metric_mode == 'max':
                        best_metric_value = float('-inf')
                    else:  # 'min'
                        best_metric_value = float('inf')

                    print(f"[INFO] Best checkpoint selection: {primary_metric} ({primary_metric_mode})")
                else:
                    print(f"[WARNING] Job {self.job_id} not found in database")
            else:
                print(f"[WARNING] Database not found at {db_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load primary metric config: {e}")

        # 4. Start training
        callbacks.on_train_begin()

        try:
            # 5. Training loop
            for epoch in range(start_epoch, self.training_config.epochs):
                epoch_num = epoch + 1  # 1-indexed for display and storage
                print(f"\n[Epoch {epoch_num}/{self.training_config.epochs}]")

                # Train
                train_metrics = self.train_epoch(epoch)

                # Validate
                val_metrics = self.validate(epoch)

                # Combine metrics
                # Note: Avoid duplicate 'val_' prefix
                val_metrics_dict = {}
                if val_metrics:
                    val_metrics_dict['val_loss'] = val_metrics.train_loss
                    # Add validation metrics, checking if they already have 'val_' prefix
                    for k, v in val_metrics.metrics.items():
                        if k.startswith('val_'):
                            # Already has val_ prefix, use as-is
                            val_metrics_dict[k] = v
                        else:
                            # Add val_ prefix
                            val_metrics_dict[f'val_{k}'] = v

                combined_metrics = MetricsResult(
                    epoch=epoch_num,  # Store as 1-indexed
                    step=train_metrics.step,
                    train_loss=train_metrics.train_loss,
                    val_loss=val_metrics.train_loss if val_metrics else None,
                    metrics={
                        'train_loss': train_metrics.train_loss,
                        **train_metrics.metrics,
                        **val_metrics_dict
                    }
                )

                all_metrics.append(combined_metrics)

                # Save checkpoint based on primary metric improvement (before reporting to callbacks)
                checkpoint_path = None
                should_save_checkpoint = False

                # Check if this is the best checkpoint so far
                if primary_metric and best_metric_value is not None:
                    # Get current metric value with flexible key matching
                    current_value = None
                    actual_metric_key = None

                    # Try exact match first
                    if primary_metric in combined_metrics.metrics:
                        current_value = combined_metrics.metrics[primary_metric]
                        actual_metric_key = primary_metric
                    # Try with 'val_' prefix (validation metrics are usually more important)
                    elif f'val_{primary_metric}' in combined_metrics.metrics:
                        current_value = combined_metrics.metrics[f'val_{primary_metric}']
                        actual_metric_key = f'val_{primary_metric}'
                    # Try with 'train_' prefix as fallback
                    elif f'train_{primary_metric}' in combined_metrics.metrics:
                        current_value = combined_metrics.metrics[f'train_{primary_metric}']
                        actual_metric_key = f'train_{primary_metric}'

                    if current_value is not None:
                        # Check if metric improved
                        if primary_metric_mode == 'max':
                            is_better = current_value > best_metric_value
                        else:  # 'min'
                            is_better = current_value < best_metric_value

                        if is_better:
                            best_metric_value = current_value
                            should_save_checkpoint = True
                            print(f"[INFO] New best {actual_metric_key}: {current_value:.4f}")
                    else:
                        print(f"[WARNING] Primary metric '{primary_metric}' not found in metrics")
                        print(f"[WARNING] Available metrics: {list(combined_metrics.metrics.keys())}")
                        # Fallback to periodic saving
                        should_save_checkpoint = (epoch_num % 5 == 0)
                else:
                    # No primary metric configured, use periodic saving
                    should_save_checkpoint = (epoch_num % 5 == 0)

                # Always save last epoch
                if epoch_num == self.training_config.epochs:
                    should_save_checkpoint = True

                if should_save_checkpoint:
                    checkpoint_path = self.save_checkpoint(epoch_num, combined_metrics)
                    print(f"Checkpoint saved: {checkpoint_path}")

                    # Log checkpoint to MLflow
                    if checkpoint_path:
                        callbacks.log_artifact(checkpoint_path, "checkpoints")

                # Report metrics to callbacks with checkpoint path
                # (handles both MLflow and database logging)
                callbacks.on_epoch_end(epoch_num, combined_metrics.metrics, checkpoint_path)

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
        else:
            # If no db_session provided, use direct SQLite connection
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

                    # Update MLflow IDs using direct SQL
                    cursor.execute(
                        "UPDATE training_jobs SET mlflow_experiment_id = ?, mlflow_run_id = ? WHERE id = ?",
                        (self.mlflow_experiment_id, self.mlflow_run_id, self.job_id)
                    )
                    conn.commit()
                    conn.close()

                    print(f"[Callbacks] Updated DB with MLflow IDs")
                    print(f"  Job ID: {self.job_id}")
                    print(f"  Experiment ID: {self.mlflow_experiment_id}")
                    print(f"  Run ID: {self.mlflow_run_id}")
            except Exception as e:
                print(f"[Callbacks WARNING] Failed to update DB with MLflow IDs: {e}")

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

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], checkpoint_path: str = None):
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
            checkpoint_path: Optional path to saved checkpoint
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
        # Extract common metrics
        train_loss = metrics.get('train_loss') or metrics.get('loss')
        val_loss = metrics.get('val_loss')
        accuracy = metrics.get('accuracy') or metrics.get('mAP50')  # mAP50 for detection
        lr = metrics.get('learning_rate') or metrics.get('lr')

        if self.db_session:
            from app.db import models

            # Store in database using SQLAlchemy
            metric_record = models.TrainingMetric(
                job_id=self.job_id,
                epoch=epoch,
                step=epoch,
                loss=train_loss,
                accuracy=accuracy,
                learning_rate=lr,
                checkpoint_path=checkpoint_path,
                extra_metrics=metrics  # Store all metrics as JSON
            )
            self.db_session.add(metric_record)
            self.db_session.commit()
        else:
            # Fallback: Use direct SQLite connection when no db_session available
            try:
                import sqlite3
                import json
                from pathlib import Path

                # Get database path
                training_dir = Path(__file__).parent.parent
                mvp_dir = training_dir.parent
                db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()

                    # Insert metric using direct SQL
                    from datetime import datetime
                    cursor.execute(
                        """
                        INSERT INTO training_metrics
                        (job_id, epoch, step, loss, accuracy, learning_rate, checkpoint_path, extra_metrics, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.job_id,
                            epoch,
                            epoch,
                            train_loss,
                            accuracy,
                            lr,
                            checkpoint_path,
                            json.dumps(metrics),
                            datetime.utcnow().isoformat()
                        )
                    )
                    conn.commit()
                    conn.close()

                    print(f"[Callbacks] Saved metric to database (epoch {epoch})")
            except Exception as e:
                print(f"[Callbacks WARNING] Failed to save metric to database: {e}")

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
