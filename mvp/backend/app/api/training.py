"""Training API endpoints."""

import os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models
from app.schemas import training
from app.core.config import settings
from app.utils.training_manager import TrainingManager
from app.utils.mlflow_client import get_mlflow_client

router = APIRouter()

# Global training manager instance
training_manager = None


@router.post("/jobs", response_model=training.TrainingJobResponse)
async def create_training_job(
    job_request: training.TrainingJobCreate,
    db: Session = Depends(get_db),
):
    """
    Create a new training job.

    This endpoint creates a training job but does not start it immediately.
    Use the /jobs/{job_id}/start endpoint to start training.
    """
    # Verify session exists
    session = db.query(models.Session).filter(models.Session.id == job_request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Create output directory
    job_output_dir = os.path.join(
        settings.OUTPUT_DIR,
        f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(job_output_dir, exist_ok=True)

    # Create training job
    job = models.TrainingJob(
        session_id=job_request.session_id,
        model_name=job_request.config.model_name,
        task_type=job_request.config.task_type,
        num_classes=job_request.config.num_classes,
        dataset_path=job_request.config.dataset_path,
        output_dir=job_output_dir,
        epochs=job_request.config.epochs,
        batch_size=job_request.config.batch_size,
        learning_rate=job_request.config.learning_rate,
        status="pending",
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    return job


@router.get("/jobs/{job_id}", response_model=training.TrainingJobResponse)
async def get_training_job(job_id: int, db: Session = Depends(get_db)):
    """Get a training job by ID."""
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@router.get("/jobs/{job_id}/status", response_model=training.TrainingStatusResponse)
async def get_training_status(job_id: int, db: Session = Depends(get_db)):
    """Get training job status with latest metrics."""
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Get latest metrics (last 10)
    latest_metrics = (
        db.query(models.TrainingMetric)
        .filter(models.TrainingMetric.job_id == job_id)
        .order_by(models.TrainingMetric.created_at.desc())
        .limit(10)
        .all()
    )

    return training.TrainingStatusResponse(
        job=job,
        latest_metrics=list(reversed(latest_metrics)),
    )


@router.get("/jobs/{job_id}/metrics", response_model=list[training.TrainingMetricResponse])
async def get_training_metrics(
    job_id: int,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Get training metrics for a job."""
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    metrics = (
        db.query(models.TrainingMetric)
        .filter(models.TrainingMetric.job_id == job_id)
        .order_by(models.TrainingMetric.created_at)
        .limit(limit)
        .all()
    )

    return metrics


@router.post("/jobs/{job_id}/start", response_model=training.TrainingJobResponse)
async def start_training_job(job_id: int, db: Session = Depends(get_db)):
    """
    Start a training job.

    This endpoint starts the actual training process using subprocess.
    """
    global training_manager

    # Initialize training manager if not already done
    if training_manager is None:
        training_manager = TrainingManager(db)

    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start job with status '{job.status}'",
        )

    # Start training subprocess
    success = training_manager.start_training(job_id)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to start training process"
        )

    db.refresh(job)
    return job


@router.post("/jobs/{job_id}/cancel", response_model=training.TrainingJobResponse)
async def cancel_training_job(job_id: int, db: Session = Depends(get_db)):
    """
    Cancel a running training job.
    """
    global training_manager

    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{job.status}'",
        )

    # Stop training subprocess
    if training_manager and training_manager.stop_training(job_id):
        db.refresh(job)
        return job
    else:
        # Fallback if process not found
        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(job)
        return job


@router.get("/jobs/{job_id}/logs", response_model=list[training.TrainingLogResponse])
async def get_training_logs(
    job_id: int,
    limit: int = 100,
    log_type: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Get training logs for a job.

    Args:
        job_id: Training job ID
        limit: Maximum number of log entries to return (default: 100)
        log_type: Filter by log type ('stdout' or 'stderr'), optional
    """
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Build query
    query = db.query(models.TrainingLog).filter(models.TrainingLog.job_id == job_id)

    # Apply log type filter if specified
    if log_type in ["stdout", "stderr"]:
        query = query.filter(models.TrainingLog.log_type == log_type)

    # Get logs ordered by creation time
    logs = query.order_by(models.TrainingLog.created_at).limit(limit).all()

    return logs


@router.get("/jobs/{job_id}/mlflow/metrics")
async def get_mlflow_metrics(job_id: int, db: Session = Depends(get_db)):
    """
    Get MLflow metrics for a training job.

    Returns all metrics with their history from MLflow tracking server.
    """
    # Verify job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    try:
        client = get_mlflow_client()
        metrics_data = client.get_run_metrics(job_id)
        return metrics_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch MLflow metrics: {str(e)}"
        )


@router.get("/jobs/{job_id}/mlflow/summary")
async def get_mlflow_summary(job_id: int, db: Session = Depends(get_db)):
    """
    Get MLflow run summary for a training job.

    Returns summary information including best metrics, parameters, and run status.
    """
    # Verify job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    try:
        client = get_mlflow_client()
        summary = client.get_run_summary(job_id)
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch MLflow summary: {str(e)}"
        )
