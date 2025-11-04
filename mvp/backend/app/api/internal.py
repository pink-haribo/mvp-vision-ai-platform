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
from app.db.models import TrainingJob, TrainingLog, TrainingMetric


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
