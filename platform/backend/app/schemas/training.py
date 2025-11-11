"""Training-related Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List
from .configs import (
    OptimizerConfig,
    SchedulerConfig,
    AugmentationConfig,
    PreprocessConfig,
    ValidationConfig,
    TrainingConfigAdvanced,
)


class TrainingConfig(BaseModel):
    """Training configuration schema."""

    framework: str = Field("timm", description="Framework (timm, ultralytics, transformers)")
    model_name: str = Field(..., description="Model name (e.g., resnet50, yolov8n)")
    task_type: str = Field(..., description="Task type (e.g., image_classification, object_detection)")
    num_classes: Optional[int] = Field(None, ge=2, description="Number of classes (required for classification)")

    # Dataset specification (provide either dataset_id OR dataset_path)
    dataset_id: Optional[str] = Field(None, description="Dataset ID from database (preferred)")
    dataset_path: Optional[str] = Field(None, description="Direct path to dataset (legacy)")
    dataset_format: str = Field("imagefolder", description="Dataset format (imagefolder, coco, yolo, etc.)")

    # Basic training parameters (backward compatible)
    epochs: int = Field(50, ge=1, le=1000, description="Number of epochs")
    batch_size: int = Field(32, ge=1, le=512, description="Batch size")
    learning_rate: float = Field(0.001, gt=0, lt=1, description="Learning rate")

    # Advanced configurations (optional)
    advanced_config: Optional[TrainingConfigAdvanced] = Field(
        None,
        description="Advanced training configuration (optimizer, scheduler, augmentation, etc.)"
    )

    # Primary metric configuration
    primary_metric: Optional[str] = Field(
        None,
        description="Primary metric to optimize (e.g., 'accuracy', 'mAP50', 'f1_score'). If None, uses framework default."
    )
    primary_metric_mode: Optional[str] = Field(
        "max",
        description="Optimization mode: 'max' to maximize metric, 'min' to minimize"
    )

    # Open-vocabulary / Zero-shot configuration (for YOLO-World, etc.)
    custom_prompts: Optional[List[str]] = Field(
        None,
        description="Custom text prompts for open-vocabulary/zero-shot detection (e.g., YOLO-World). Example: ['red apple', 'damaged box']"
    )
    prompt_mode: Optional[str] = Field(
        "offline",
        description="Prompt mode for open-vocabulary models: 'offline' (pre-computed embeddings) or 'dynamic' (runtime encoding)"
    )

    class Config:
        protected_namespaces = ()  # Allow model_name field


class TrainingJobCreate(BaseModel):
    """Schema for creating a training job."""

    session_id: Optional[int] = Field(None, description="Chat session ID (optional)")
    config: TrainingConfig
    project_id: Optional[int] = Field(None, description="Project ID to associate with")
    experiment_name: Optional[str] = Field(None, max_length=200, description="Experiment name")
    tags: Optional[list[str]] = Field(None, description="Tags for categorization")
    notes: Optional[str] = Field(None, description="User notes about this experiment")


class TrainingJobResponse(BaseModel):
    """Schema for training job response."""

    id: int
    session_id: Optional[int] = Field(None, description="Chat session ID (optional)")
    project_id: Optional[int] = None
    project_name: Optional[str] = None  # Project name for breadcrumb navigation

    # Experiment metadata
    experiment_name: Optional[str] = None
    tags: Optional[list[str]] = None
    notes: Optional[str] = None
    mlflow_run_id: Optional[str] = None

    framework: str
    model_name: str
    task_type: str
    num_classes: Optional[int] = None
    dataset_path: str
    dataset_format: str
    output_dir: str

    # Basic training parameters
    epochs: int
    batch_size: int
    learning_rate: float

    # Advanced configurations (optional)
    advanced_config: Optional[TrainingConfigAdvanced] = None

    status: str
    error_message: Optional[str] = None
    process_id: Optional[int] = None

    final_accuracy: Optional[float] = None
    best_checkpoint_path: Optional[str] = None

    # Primary metric configuration
    primary_metric: Optional[str] = None
    primary_metric_mode: Optional[str] = None

    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        protected_namespaces = ()  # Allow model_name field


class TrainingMetricResponse(BaseModel):
    """Schema for training metric response."""

    id: int
    job_id: int
    epoch: int
    step: Optional[int] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    extra_metrics: Optional[dict] = None
    checkpoint_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingStatusResponse(BaseModel):
    """Schema for training status response."""

    job: TrainingJobResponse
    latest_metrics: list[TrainingMetricResponse]


class TrainingLogResponse(BaseModel):
    """Schema for training log response."""

    id: int
    job_id: int
    log_type: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True
