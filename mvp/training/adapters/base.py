"""Base adapter interface for training frameworks."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


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


class DatasetFormat(Enum):
    """Dataset format types."""
    IMAGE_FOLDER = "imagefolder"  # Classification: folder per class
    COCO = "coco"  # Detection, Segmentation: JSON annotations
    YOLO = "yolo"  # YOLO format: txt annotations
    PASCAL_VOC = "voc"  # Pascal VOC XML format
    CUSTOM = "custom"  # Custom format


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

    # ========== Common Methods (can be overridden) ==========

    def train(self) -> List[MetricsResult]:
        """
        Full training process.

        1. Prepare model
        2. Prepare dataset
        3. For each epoch:
           - Train
           - Validate
           - Log metrics (MLflow)
           - Save checkpoint

        Returns:
            List[MetricsResult]: All metrics history
        """
        import mlflow

        # 1. Setup
        self.prepare_model()
        self.prepare_dataset()

        all_metrics = []

        # 2. Start MLflow experiment
        mlflow.set_experiment("vision-ai-training")
        with mlflow.start_run(run_name=f"job_{self.job_id}"):
            # Log config
            mlflow.log_params({
                "framework": self.model_config.framework,
                "model_name": self.model_config.model_name,
                "task_type": self.model_config.task_type.value,
                "epochs": self.training_config.epochs,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
            })

            # 3. Training loop
            for epoch in range(self.training_config.epochs):
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
                    metrics={
                        **train_metrics.metrics,
                        **{f"val_{k}": v for k, v in (val_metrics.metrics if val_metrics else {}).items()}
                    }
                )

                all_metrics.append(combined_metrics)

                # Log to MLflow
                self.log_metrics_to_mlflow(combined_metrics)

                # Save checkpoint periodically
                if epoch % 5 == 0 or epoch == self.training_config.epochs - 1:
                    checkpoint_path = self.save_checkpoint(epoch, combined_metrics)
                    print(f"Checkpoint saved: {checkpoint_path}")

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
