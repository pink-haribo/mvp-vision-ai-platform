"""Training-related Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class TrainingConfig(BaseModel):
    """Training configuration schema."""

    model_name: str = Field(..., description="Model name (e.g., resnet50)")
    task_type: str = Field(..., description="Task type (e.g., classification)")
    num_classes: int = Field(..., ge=2, description="Number of classes")
    dataset_path: str = Field(..., description="Path to dataset")

    epochs: int = Field(50, ge=1, le=1000, description="Number of epochs")
    batch_size: int = Field(32, ge=1, le=512, description="Batch size")
    learning_rate: float = Field(0.001, gt=0, lt=1, description="Learning rate")


class TrainingJobCreate(BaseModel):
    """Schema for creating a training job."""

    session_id: int
    config: TrainingConfig


class TrainingJobResponse(BaseModel):
    """Schema for training job response."""

    id: int
    session_id: int

    model_name: str
    task_type: str
    num_classes: int
    dataset_path: str
    output_dir: str

    epochs: int
    batch_size: int
    learning_rate: float

    status: str
    error_message: Optional[str] = None
    process_id: Optional[int] = None

    final_accuracy: Optional[float] = None
    best_checkpoint_path: Optional[str] = None

    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


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
