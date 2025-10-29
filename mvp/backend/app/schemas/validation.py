"""
Validation API schemas.

Pydantic models for validation results API responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime


# ========== Classification Metrics ==========

class ClassificationMetricsResponse(BaseModel):
    """Classification task metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    top5_accuracy: Optional[float] = None

    class Config:
        from_attributes = True


class PerClassMetricsResponse(BaseModel):
    """Per-class metrics for classification."""
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int


# ========== Detection Metrics ==========

class DetectionMetricsResponse(BaseModel):
    """Detection task metrics."""
    map_50: float
    map_50_95: float
    precision: float
    recall: float

    class Config:
        from_attributes = True


# ========== Segmentation Metrics ==========

class SegmentationMetricsResponse(BaseModel):
    """Segmentation task metrics."""
    mean_iou: float
    pixel_accuracy: float
    mean_precision: float
    mean_recall: float

    class Config:
        from_attributes = True


# ========== Pose Metrics ==========

class PoseMetricsResponse(BaseModel):
    """Pose estimation metrics."""
    oks: float
    pck: float
    mean_precision: float
    mean_recall: float

    class Config:
        from_attributes = True


# ========== Validation Result ==========

class ValidationResultResponse(BaseModel):
    """
    Validation result for a specific epoch.

    Contains task-agnostic validation metrics with task-specific details.
    """
    id: int
    job_id: int
    epoch: int
    task_type: str

    # Primary metric
    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[float] = None
    overall_loss: Optional[float] = None

    # Task-specific metrics (as dict for flexibility)
    metrics: Optional[Dict[str, Any]] = None
    per_class_metrics: Optional[Dict[str, Any]] = None

    # Visualization data
    confusion_matrix: Optional[List[List[int]]] = None
    pr_curves: Optional[Dict[str, Any]] = None
    class_names: Optional[List[str]] = None
    visualization_data: Optional[Dict[str, Any]] = None

    # Sample images
    sample_correct_images: Optional[List[str]] = None
    sample_incorrect_images: Optional[List[str]] = None

    created_at: datetime

    class Config:
        from_attributes = True


class ValidationResultListResponse(BaseModel):
    """List of validation results for a job."""
    job_id: int
    total_count: int
    results: List[ValidationResultResponse]


# ========== Image-level Results ==========

class ValidationImageResultResponse(BaseModel):
    """
    Image-level validation result.

    Contains per-image predictions and metrics for any task type.
    """
    id: int
    validation_result_id: int
    job_id: int
    epoch: int

    # Image info
    image_path: Optional[str] = None  # May be None if path not available during training
    image_name: str
    image_index: Optional[int] = None

    # Classification fields
    true_label: Optional[str] = None
    true_label_id: Optional[int] = None
    predicted_label: Optional[str] = None
    predicted_label_id: Optional[int] = None
    confidence: Optional[float] = None
    top5_predictions: Optional[List[Dict[str, Any]]] = None

    # Detection fields
    true_boxes: Optional[List[Dict[str, Any]]] = None
    predicted_boxes: Optional[List[Dict[str, Any]]] = None

    # Segmentation fields
    true_mask_path: Optional[str] = None
    predicted_mask_path: Optional[str] = None

    # Pose fields
    true_keypoints: Optional[List[Dict[str, Any]]] = None
    predicted_keypoints: Optional[List[Dict[str, Any]]] = None

    # Common metrics
    is_correct: bool
    iou: Optional[float] = None
    oks: Optional[float] = None

    # Extra data
    extra_data: Optional[Dict[str, Any]] = None

    created_at: datetime

    class Config:
        from_attributes = True


class ValidationImageResultListResponse(BaseModel):
    """List of image-level validation results."""
    validation_result_id: int
    job_id: int
    epoch: int
    total_count: int
    correct_count: int
    incorrect_count: int
    class_names: Optional[List[str]] = None
    images: List[ValidationImageResultResponse]


# ========== Validation Summary ==========

class ValidationSummaryResponse(BaseModel):
    """
    Summary of validation results across all epochs for a job.

    Useful for displaying trends and best performance.
    """
    job_id: int
    task_type: str
    total_epochs: int

    best_epoch: Optional[int] = None
    best_metric_value: Optional[float] = None
    best_metric_name: Optional[str] = None

    # Epoch-wise metrics for charting
    epoch_metrics: List[Dict[str, Any]]  # [{epoch: 1, accuracy: 0.85, loss: 0.23}, ...]

    class Config:
        from_attributes = True
