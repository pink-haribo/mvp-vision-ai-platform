"""
Internal API endpoints for Training Services.

These endpoints are called by Training Services to report logs, metrics,
and status updates. They are protected by X-Internal-Auth header.

DO NOT expose these endpoints publicly!
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.db.database import get_db
from app.db.models import TrainingJob, TrainingLog, TrainingMetric, ValidationResult, ValidationImageResult


router = APIRouter(prefix="/internal/training", tags=["internal"])


# Internal authentication
def verify_internal_auth(x_internal_auth: str = Header(...)):
    """
    Verify internal authentication token.

    Training Services must include X-Internal-Auth header with valid token.
    """
    expected_token = os.environ.get("INTERNAL_AUTH_TOKEN")

    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="INTERNAL_AUTH_TOKEN not configured on Backend"
        )

    if x_internal_auth != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal auth token"
        )


# Request schemas
class MetricRequest(BaseModel):
    """Training metric from Training Service."""
    name: str = Field(..., description="Metric name (e.g., train_loss, val_accuracy)")
    value: float = Field(..., description="Metric value")
    epoch: Optional[int] = Field(None, description="Training epoch")
    step: Optional[int] = Field(None, description="Training step")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")


class LogRequest(BaseModel):
    """Training log message from Training Service."""
    message: str = Field(..., description="Log message")
    level: str = Field("INFO", description="Log level (INFO, WARNING, ERROR, DEBUG)")
    epoch: Optional[int] = Field(None, description="Training epoch")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")


class StatusRequest(BaseModel):
    """Training job status update from Training Service."""
    status: str = Field(..., description="Job status (running, completed, failed)")
    error: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[float] = Field(None, description="Progress 0-100")
    updated_at: Optional[str] = Field(None, description="ISO timestamp")
    mlflow_experiment_id: Optional[str] = Field(None, description="MLflow experiment ID")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")


class ValidationResultRequest(BaseModel):
    """Validation result from Training Service."""
    epoch: int = Field(..., description="Training epoch (1-indexed)")
    task_type: str = Field(..., description="Task type (e.g., image_classification)")
    primary_metric_value: float = Field(..., description="Primary metric value (e.g., accuracy, mAP50)")
    primary_metric_name: str = Field(..., description="Primary metric name (e.g., accuracy, mAP50)")
    overall_loss: Optional[float] = Field(None, description="Overall validation loss")
    metrics: Dict[str, Any] = Field(..., description="All metrics as JSON")
    per_class_metrics: Optional[Dict[str, Any]] = Field(None, description="Per-class metrics (optional)")
    confusion_matrix: Optional[Any] = Field(None, description="Confusion matrix as list of lists (optional)")
    pr_curves: Optional[Dict[str, Any]] = Field(None, description="Precision-recall curves (optional)")
    class_names: Optional[list] = Field(None, description="List of class names (optional)")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint for this epoch")


class ValidationImageResultRequest(BaseModel):
    """Per-image validation result from Training Service."""
    validation_result_id: int = Field(..., description="ID of parent ValidationResult")
    epoch: int = Field(..., description="Training epoch")
    image_path: Optional[str] = Field(None, description="Path to image file")
    image_name: str = Field(..., description="Image filename")
    image_index: int = Field(..., description="Index in validation set")
    true_label: Optional[str] = Field(None, description="True label (classification)")
    true_label_id: Optional[int] = Field(None, description="True label ID")
    predicted_label: Optional[str] = Field(None, description="Predicted label (classification)")
    predicted_label_id: Optional[int] = Field(None, description="Predicted label ID")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    top5_predictions: Optional[Any] = Field(None, description="Top-5 predictions (classification)")
    true_boxes: Optional[Any] = Field(None, description="True bounding boxes (detection)")
    predicted_boxes: Optional[Any] = Field(None, description="Predicted bounding boxes (detection)")
    true_mask_path: Optional[str] = Field(None, description="True mask path (segmentation)")
    predicted_mask_path: Optional[str] = Field(None, description="Predicted mask path (segmentation)")
    true_keypoints: Optional[Any] = Field(None, description="True keypoints (pose)")
    predicted_keypoints: Optional[Any] = Field(None, description="Predicted keypoints (pose)")
    is_correct: Optional[bool] = Field(None, description="Whether prediction is correct")
    iou: Optional[float] = Field(None, description="IoU score (detection/segmentation)")
    oks: Optional[float] = Field(None, description="OKS score (pose)")
    extra_data: Optional[Dict[str, Any]] = Field(None, description="Extra task-specific data")


class TrainingMetricBatchRequest(BaseModel):
    """Batch of training metrics (for epoch end)."""
    epoch: int = Field(..., description="Training epoch (1-indexed)")
    step: int = Field(..., description="Training step")
    loss: Optional[float] = Field(None, description="Training loss")
    accuracy: Optional[float] = Field(None, description="Accuracy or primary metric")
    learning_rate: Optional[float] = Field(None, description="Current learning rate")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint")
    extra_metrics: Dict[str, Any] = Field(default_factory=dict, description="All other metrics")


# Endpoints
@router.post("/{job_id}/metrics")
async def create_metric(
    job_id: int,
    metric: MetricRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_auth)
):
    """
    Receive training metric from Training Service.

    Called by TrainingLogger.log_metric()
    """
    # Verify job exists
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )

    # Map common metric names to TrainingMetric columns
    metric_obj = TrainingMetric(
        job_id=job_id,
        epoch=metric.epoch or 0,
        step=metric.step,
        extra_metrics={}
    )

    # Map well-known metrics to dedicated columns
    if metric.name == "train_loss" or metric.name == "loss":
        metric_obj.loss = metric.value
    elif metric.name == "val_accuracy" or metric.name == "accuracy":
        metric_obj.accuracy = metric.value
    elif metric.name == "learning_rate" or metric.name == "lr":
        metric_obj.learning_rate = metric.value
    else:
        # Store custom metrics in extra_metrics JSON
        metric_obj.extra_metrics = {metric.name: metric.value}

    # Add metadata
    if metric.metadata:
        metric_obj.extra_metrics.update(metric.metadata)

    db.add(metric_obj)
    db.commit()
    db.refresh(metric_obj)

    return {
        "status": "ok",
        "metric_id": metric_obj.id,
        "job_id": job_id
    }


@router.post("/{job_id}/logs")
async def create_log(
    job_id: int,
    log: LogRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_auth)
):
    """
    Receive training log message from Training Service.

    Called by TrainingLogger.log_message()
    """
    # Verify job exists
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )

    # Create log entry
    log_obj = TrainingLog(
        job_id=job_id,
        log_type=log.level.upper()[:10],  # Max 10 chars
        content=log.message
    )

    db.add(log_obj)
    db.commit()
    db.refresh(log_obj)

    return {
        "status": "ok",
        "log_id": log_obj.id,
        "job_id": job_id
    }


@router.patch("/{job_id}/status")
async def update_status(
    job_id: int,
    status_update: StatusRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_auth)
):
    """
    Update training job status from Training Service.

    Called by TrainingLogger.update_status()
    """
    # Get job
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )

    # Update status
    job.status = status_update.status

    if status_update.error:
        job.error_message = status_update.error

    # Update MLflow IDs if provided
    if status_update.mlflow_experiment_id:
        job.mlflow_experiment_id = status_update.mlflow_experiment_id
    if status_update.mlflow_run_id:
        job.mlflow_run_id = status_update.mlflow_run_id

    # Update timestamps
    if status_update.status == "running" and not job.started_at:
        job.started_at = datetime.utcnow()
    elif status_update.status in ["completed", "failed"]:
        job.completed_at = datetime.utcnow()

    db.commit()
    db.refresh(job)

    return {
        "status": "ok",
        "job_id": job_id,
        "new_status": job.status
    }


@router.get("/{job_id}/health")
async def health_check(
    job_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_auth)
):
    """
    Health check for specific job.

    Training Services can use this to verify Backend connectivity.
    """
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )

    return {
        "status": "healthy",
        "job_id": job_id,
        "job_status": job.status
    }


@router.post("/{job_id}/validation-results")
async def create_validation_result(
    job_id: int,
    result: ValidationResultRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_auth)
):
    """
    Receive validation result from Training Service.

    Called by TrainingAdapter._save_validation_result() via callback URL.
    """
    import json

    # Verify job exists
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )

    # Create validation result
    validation_result = ValidationResult(
        job_id=job_id,
        epoch=result.epoch,
        task_type=result.task_type,
        primary_metric_value=result.primary_metric_value,
        primary_metric_name=result.primary_metric_name,
        overall_loss=result.overall_loss,
        metrics=json.dumps(result.metrics) if result.metrics else None,
        per_class_metrics=json.dumps(result.per_class_metrics) if result.per_class_metrics else None,
        confusion_matrix=json.dumps(result.confusion_matrix) if result.confusion_matrix else None,
        pr_curves=json.dumps(result.pr_curves) if result.pr_curves else None,
        class_names=json.dumps(result.class_names) if result.class_names else None,
        checkpoint_path=result.checkpoint_path
    )

    db.add(validation_result)
    db.commit()
    db.refresh(validation_result)

    return {
        "status": "ok",
        "validation_result_id": validation_result.id,
        "job_id": job_id,
        "epoch": result.epoch
    }


@router.post("/{job_id}/validation-image-results")
async def create_validation_image_results(
    job_id: int,
    results: list[ValidationImageResultRequest],
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_auth)
):
    """
    Receive per-image validation results from Training Service (batch).

    Called by TrainingAdapter._save_validation_image_results() via callback URL.
    """
    import json

    # Verify job exists
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )

    # Create image results
    image_results = []
    for result in results:
        image_result = ValidationImageResult(
            validation_result_id=result.validation_result_id,
            job_id=job_id,
            epoch=result.epoch,
            image_path=result.image_path,
            image_name=result.image_name,
            image_index=result.image_index,
            true_label=result.true_label,
            true_label_id=result.true_label_id,
            predicted_label=result.predicted_label,
            predicted_label_id=result.predicted_label_id,
            confidence=result.confidence,
            top5_predictions=json.dumps(result.top5_predictions) if result.top5_predictions else None,
            true_boxes=json.dumps(result.true_boxes) if result.true_boxes else None,
            predicted_boxes=json.dumps(result.predicted_boxes) if result.predicted_boxes else None,
            true_mask_path=result.true_mask_path,
            predicted_mask_path=result.predicted_mask_path,
            true_keypoints=json.dumps(result.true_keypoints) if result.true_keypoints else None,
            predicted_keypoints=json.dumps(result.predicted_keypoints) if result.predicted_keypoints else None,
            is_correct=result.is_correct,
            iou=result.iou,
            oks=result.oks,
            extra_data=json.dumps(result.extra_data) if result.extra_data else None
        )
        image_results.append(image_result)

    db.bulk_save_objects(image_results)
    db.commit()

    return {
        "status": "ok",
        "count": len(image_results),
        "job_id": job_id
    }


@router.post("/{job_id}/training-metrics")
async def create_training_metric_batch(
    job_id: int,
    metric: TrainingMetricBatchRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_auth)
):
    """
    Receive training metrics from Training Service (epoch end).

    Called by TrainingCallbacks.on_epoch_end() via callback URL.
    """
    import json

    # Verify job exists
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )

    # Create training metric
    training_metric = TrainingMetric(
        job_id=job_id,
        epoch=metric.epoch,
        step=metric.step,
        loss=metric.loss,
        accuracy=metric.accuracy,
        learning_rate=metric.learning_rate,
        checkpoint_path=metric.checkpoint_path,
        extra_metrics=metric.extra_metrics
    )

    db.add(training_metric)
    db.commit()
    db.refresh(training_metric)

    return {
        "status": "ok",
        "metric_id": training_metric.id,
        "job_id": job_id,
        "epoch": metric.epoch
    }
