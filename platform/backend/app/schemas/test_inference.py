"""
Test and Inference API schemas.

Pydantic models for:
- Test: Running tests on labeled datasets (with metrics)
- Inference: Running predictions on unlabeled data (predictions only)
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ========== Test Request/Response ==========

class TestRunRequest(BaseModel):
    """Request to run test on trained model."""
    training_job_id: int = Field(..., description="Training job ID to test")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    dataset_path: str = Field(..., description="Path to test dataset")
    dataset_split: str = Field(default="test", description="Dataset split to use (test, val)")

    class Config:
        json_schema_extra = {
            "example": {
                "training_job_id": 123,
                "checkpoint_path": "/workspace/checkpoints/best_model.pth",
                "dataset_path": "/workspace/datasets/cifar10",
                "dataset_split": "test"
            }
        }


class TestRunResponse(BaseModel):
    """
    Test run result on labeled dataset.

    Contains metrics and performance measurements.
    """
    id: int
    training_job_id: int
    checkpoint_path: str
    dataset_path: str
    dataset_split: str

    # Status
    status: str  # pending, running, completed, failed
    error_message: Optional[str] = None

    # Task info
    task_type: str
    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[float] = None

    # Metrics (task-agnostic)
    overall_loss: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    per_class_metrics: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[List[List[int]]] = None

    # Metadata
    class_names: Optional[List[str]] = None
    total_images: int
    inference_time_ms: Optional[float] = None

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TestRunListResponse(BaseModel):
    """List of test runs for a training job."""
    training_job_id: int
    total_count: int
    test_runs: List[TestRunResponse]


# ========== Test Image Results ==========

class TestImageResultResponse(BaseModel):
    """
    Image-level test result with ground truth.

    Contains predictions and ground truth for accuracy measurement.
    """
    id: int
    test_run_id: int
    training_job_id: int

    # Image info
    image_path: str
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

    # Performance
    inference_time_ms: Optional[float] = None
    preprocessing_time_ms: Optional[float] = None
    postprocessing_time_ms: Optional[float] = None

    # Extra data
    extra_data: Optional[Dict[str, Any]] = None

    created_at: datetime

    class Config:
        from_attributes = True


class TestImageResultListResponse(BaseModel):
    """List of image-level test results."""
    test_run_id: int
    training_job_id: int
    total_count: int
    correct_count: int
    incorrect_count: int
    class_names: Optional[List[str]] = None
    images: List[TestImageResultResponse]


# ========== Inference Request/Response ==========

class InferenceRequest(BaseModel):
    """Request to run inference on trained model."""
    training_job_id: int = Field(..., description="Training job ID")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    inference_type: str = Field(..., description="Type: single, batch, dataset")
    input_data: Dict[str, Any] = Field(..., description="Input data based on type")

    class Config:
        json_schema_extra = {
            "example": {
                "training_job_id": 123,
                "checkpoint_path": "/workspace/checkpoints/best_model.pth",
                "inference_type": "batch",
                "input_data": {
                    "image_paths": [
                        "/data/images/img1.jpg",
                        "/data/images/img2.jpg"
                    ]
                }
            }
        }


class InferenceJobResponse(BaseModel):
    """
    Inference job metadata.

    Used for production inference on unlabeled data.
    """
    id: int
    training_job_id: int
    checkpoint_path: str

    # Input
    inference_type: str  # single, batch, dataset
    input_data: Optional[Dict[str, Any]] = None

    # Status
    status: str  # pending, running, completed, failed
    error_message: Optional[str] = None

    # Task info
    task_type: str

    # Performance metrics
    total_images: int
    total_inference_time_ms: Optional[float] = None
    avg_inference_time_ms: Optional[float] = None

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class InferenceJobListResponse(BaseModel):
    """List of inference jobs for a training job."""
    training_job_id: int
    total_count: int
    inference_jobs: List[InferenceJobResponse]


# ========== Inference Results (Predictions Only) ==========

class InferenceResultResponse(BaseModel):
    """
    Image-level inference result (predictions only, no ground truth).

    Used for production inference where labels are not available.
    """
    id: int
    inference_job_id: int
    training_job_id: int

    # Image info
    image_path: str
    image_name: str
    image_index: Optional[int] = None

    # Classification fields (predictions only)
    predicted_label: Optional[str] = None
    predicted_label_id: Optional[int] = None
    confidence: Optional[float] = None
    top5_predictions: Optional[List[Dict[str, Any]]] = None

    # Detection fields (predictions only)
    predicted_boxes: Optional[List[Dict[str, Any]]] = None

    # Segmentation fields (predictions only)
    predicted_mask_path: Optional[str] = None

    # Pose fields (predictions only)
    predicted_keypoints: Optional[List[Dict[str, Any]]] = None

    # Performance
    inference_time_ms: Optional[float] = None
    preprocessing_time_ms: Optional[float] = None
    postprocessing_time_ms: Optional[float] = None

    # Extra data
    extra_data: Optional[Dict[str, Any]] = None

    created_at: datetime

    class Config:
        from_attributes = True


class InferenceResultListResponse(BaseModel):
    """List of inference results for an inference job."""
    inference_job_id: int
    training_job_id: int
    total_count: int
    avg_inference_time_ms: Optional[float] = None
    results: List[InferenceResultResponse]


# ========== Summary Responses ==========

class TestSummaryResponse(BaseModel):
    """
    Summary of test performance for a trained model.

    Useful for quick overview of model accuracy.
    """
    training_job_id: int
    task_type: str
    total_test_runs: int

    best_test_run_id: Optional[int] = None
    best_metric_value: Optional[float] = None
    best_metric_name: Optional[str] = None

    # Test runs list
    test_runs: List[TestRunResponse]

    class Config:
        from_attributes = True


class InferenceSummaryResponse(BaseModel):
    """
    Summary of inference jobs for a trained model.

    Useful for tracking production usage.
    """
    training_job_id: int
    task_type: str
    total_inference_jobs: int
    total_images_processed: int
    avg_inference_time_ms: Optional[float] = None

    # Recent inference jobs
    recent_jobs: List[InferenceJobResponse]

    class Config:
        from_attributes = True


# ========== Callback Schemas (for Training CLIs) ==========

class TestResultsCallback(BaseModel):
    """
    Callback payload from evaluate.py (Training CLI).

    Sent when test/evaluation completes.
    """
    status: str  # completed, failed
    task_type: str

    # Metrics
    metrics: Optional[Dict[str, Any]] = None
    per_class_metrics: Optional[Dict[str, Any]] = None

    # Metadata
    class_names: Optional[List[str]] = None
    num_images: Optional[int] = None

    # Visualization
    visualization_urls: Optional[Dict[str, str]] = None
    predictions_json_uri: Optional[str] = None

    # Config used
    config: Optional[Dict[str, Any]] = None

    # Error info (if failed)
    error_message: Optional[str] = None
    traceback: Optional[str] = None


class ImagePredictionResult(BaseModel):
    """
    Single image prediction result from predict.py.

    Used in InferenceResultsCallback.
    """
    image_path: str
    image_name: Optional[str] = None
    predictions: List[Dict[str, Any]] = Field(default_factory=list)
    inference_time_ms: Optional[float] = None


class InferenceResultsCallback(BaseModel):
    """
    Callback payload from predict.py (Training CLI).

    Sent when inference job completes.
    """
    status: str  # completed, failed

    # Summary metrics
    total_images: Optional[int] = None
    total_inference_time_ms: Optional[float] = None
    avg_inference_time_ms: Optional[float] = None

    # Per-image results
    results: Optional[List[ImagePredictionResult]] = None

    # Error info (if failed)
    error_message: Optional[str] = None
    traceback: Optional[str] = None
