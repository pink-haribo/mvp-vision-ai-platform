"""Training API endpoints."""

import os
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models
from app.schemas import training
from app.core.config import settings
from app.utils.training_manager import TrainingManager
from app.utils.mlflow_client import get_mlflow_client

logger = logging.getLogger(__name__)

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
    # DEBUG: Log what we received
    logger.info(f"[DEBUG] Received training job request:")
    logger.info(f"[DEBUG]   framework: {job_request.config.framework}")
    logger.info(f"[DEBUG]   model_name: {job_request.config.model_name}")
    logger.info(f"[DEBUG]   task_type: {job_request.config.task_type}")

    # Validate required fields
    config = job_request.config

    if not config.dataset_path or config.dataset_path == "None":
        raise HTTPException(
            status_code=400,
            detail="dataset_path is required and cannot be empty"
        )

    if not config.model_name:
        raise HTTPException(
            status_code=400,
            detail="model_name is required"
        )

    if not config.task_type:
        raise HTTPException(
            status_code=400,
            detail="task_type is required"
        )

    # For classification tasks, num_classes is required
    if config.task_type == "image_classification" and not config.num_classes:
        raise HTTPException(
            status_code=400,
            detail="num_classes is required for image classification tasks"
        )

    # Verify session exists (if provided)
    if job_request.session_id:
        session = db.query(models.Session).filter(models.Session.id == job_request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

    # Verify project exists if provided
    if job_request.project_id:
        project = db.query(models.Project).filter(models.Project.id == job_request.project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    # Create output directory
    job_output_dir = os.path.join(
        settings.OUTPUT_DIR,
        f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(job_output_dir, exist_ok=True)

    # Determine primary metric based on framework/task if not provided
    primary_metric = config.primary_metric
    primary_metric_mode = config.primary_metric_mode or "max"

    if not primary_metric:
        # Framework-specific defaults
        if config.task_type == "image_classification":
            primary_metric = "accuracy"
            primary_metric_mode = "max"
        elif config.task_type in ["object_detection", "instance_segmentation"]:
            primary_metric = "mAP50"
            primary_metric_mode = "max"
        elif config.task_type == "pose_estimation":
            primary_metric = "mAP50"
            primary_metric_mode = "max"
        else:
            # Fallback to loss
            primary_metric = "loss"
            primary_metric_mode = "min"

    # Create training job
    job = models.TrainingJob(
        session_id=job_request.session_id,
        project_id=job_request.project_id,
        experiment_name=job_request.experiment_name,
        tags=job_request.tags,  # Will be stored as JSON
        notes=job_request.notes,
        framework=job_request.config.framework,
        model_name=job_request.config.model_name,
        task_type=job_request.config.task_type,
        num_classes=job_request.config.num_classes,
        dataset_path=job_request.config.dataset_path,
        dataset_format=job_request.config.dataset_format,
        output_dir=job_output_dir,
        epochs=job_request.config.epochs,
        batch_size=job_request.config.batch_size,
        learning_rate=job_request.config.learning_rate,
        advanced_config=job_request.config.advanced_config.model_dump() if job_request.config.advanced_config else None,
        primary_metric=primary_metric,
        primary_metric_mode=primary_metric_mode,
        status="pending",
    )

    db.add(job)
    db.commit()
    db.refresh(job)
    # Add project_name for breadcrumb navigation
    if job.project_id and job.project:
        job.project_name = job.project.name
    else:
        job.project_name = None


    return job


@router.get("/jobs/{job_id}", response_model=training.TrainingJobResponse)
async def get_training_job(job_id: int, db: Session = Depends(get_db)):
    """Get a training job by ID."""
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Auto-link MLflow run_id if not already linked and job is running/completed
    if not job.mlflow_run_id and job.status in ["running", "completed"]:
        try:
            mlflow_client = get_mlflow_client()
            mlflow_run = mlflow_client.get_run_by_job_id(job_id)
            if mlflow_run:
                job.mlflow_run_id = mlflow_run.info.run_id
                db.commit()
                db.refresh(job)
                print(f"[INFO] Linked MLflow run_id {job.mlflow_run_id} to job {job_id}")
        except Exception as e:
            # Don't fail the request if MLflow linking fails
            print(f"[WARNING] Failed to link MLflow run for job {job_id}: {e}")
    # Add project_name for breadcrumb navigation
    if job.project_id and job.project:
        job.project_name = job.project.name
    else:
        job.project_name = None


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


@router.get("/jobs/{job_id}/metric-schema")
async def get_metric_schema(job_id: int, db: Session = Depends(get_db)):
    """
    Get available metric columns for a training job.

    Returns a schema describing available metrics, their types,
    and the primary metric configuration.
    """
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Get a sample metric to extract available columns
    sample_metric = (
        db.query(models.TrainingMetric)
        .filter(models.TrainingMetric.job_id == job_id)
        .order_by(models.TrainingMetric.created_at.desc())
        .first()
    )

    available_metrics = []

    if sample_metric and sample_metric.extra_metrics:
        # Extract all metric names from extra_metrics
        for key in sample_metric.extra_metrics.keys():
            # Skip internal/debugging metrics
            if key not in ['batch', 'total_batches', 'epoch_time']:
                available_metrics.append(key)

    # Add standard columns if they have values
    if sample_metric:
        if sample_metric.loss is not None:
            available_metrics.insert(0, 'loss')
        if sample_metric.accuracy is not None:
            available_metrics.insert(1, 'accuracy')
        if sample_metric.learning_rate is not None:
            available_metrics.append('learning_rate')

    return {
        "job_id": job_id,
        "framework": job.framework,
        "task_type": job.task_type,
        "primary_metric": job.primary_metric or "loss",
        "primary_metric_mode": job.primary_metric_mode or "min",
        "available_metrics": available_metrics,
        "metric_count": len(available_metrics)
    }


@router.post("/jobs/{job_id}/start", response_model=training.TrainingJobResponse)
async def start_training_job(
    job_id: int,
    checkpoint_path: str = None,
    resume: bool = False,
    db: Session = Depends(get_db)
):
    """
    Start a training job.

    This endpoint starts the actual training process using subprocess.

    Args:
        job_id: Training job ID
        checkpoint_path: Optional path to checkpoint file to load
        resume: If True, resume training from checkpoint (restore optimizer/scheduler state).
                If False, only load model weights.
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

    # Validate checkpoint path if provided
    if checkpoint_path and not os.path.exists(checkpoint_path):
        raise HTTPException(
            status_code=400,
            detail=f"Checkpoint file not found: {checkpoint_path}"
        )

    # Start training subprocess
    success = training_manager.start_training(job_id, checkpoint_path=checkpoint_path, resume=resume)

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


@router.post("/jobs/{job_id}/restart", response_model=training.TrainingJobResponse)
async def restart_training_job(job_id: int, db: Session = Depends(get_db)):
    """
    Restart a completed or cancelled training job.

    Resets the job status to 'pending' and clears previous training data.
    The job configuration remains the same.
    """
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status not in ["completed", "cancelled", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot restart job with status '{job.status}'. Only completed, cancelled, or failed jobs can be restarted.",
        )

    # Reset job status
    job.status = "pending"
    job.started_at = None
    job.completed_at = None
    job.error_message = None
    job.final_accuracy = None
    job.process_id = None
    job.mlflow_run_id = None

    # Clear previous metrics
    db.query(models.TrainingMetric).filter(models.TrainingMetric.job_id == job_id).delete()

    # Clear previous logs
    db.query(models.TrainingLog).filter(models.TrainingLog.job_id == job_id).delete()

    db.commit()
    db.refresh(job)

    # Add project_name for breadcrumb navigation
    if job.project_id and job.project:
        job.project_name = job.project.name
    else:
        job.project_name = None

    return job


@router.get("/jobs/{job_id}/logs", response_model=list[training.TrainingLogResponse])
async def get_training_logs(
    job_id: int,
    limit: int = 500,
    log_type: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Get training logs for a job.

    Returns the most recent logs up to the limit, ordered chronologically.

    Args:
        job_id: Training job ID
        limit: Maximum number of log entries to return (default: 500)
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

    # Get most recent logs (newest first), then reverse to show chronologically
    logs = query.order_by(models.TrainingLog.created_at.desc()).limit(limit).all()

    # Reverse to show oldest first (chronological order)
    return list(reversed(logs))


@router.get("/jobs/{job_id}/mlflow/metrics")
async def get_mlflow_metrics(job_id: int, db: Session = Depends(get_db)):
    """
    Get MLflow metrics for a training job.

    Returns all metrics with their history from MLflow tracking server,
    plus primary metric information.
    """
    # Verify job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    try:
        client = get_mlflow_client()
        metrics_data = client.get_run_metrics(job_id)

        # Add primary metric information
        metrics_data['primary_metric'] = job.primary_metric or 'loss'
        metrics_data['primary_metric_mode'] = job.primary_metric_mode or 'min'
        metrics_data['task_type'] = job.task_type
        metrics_data['framework'] = job.framework

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


# ==================== Advanced Configuration Endpoints ====================


@router.get("/config/schema")
async def get_config_schema():
    """
    Get the JSON schema for advanced training configuration.

    Returns the complete schema including:
    - Optimizer options and parameters
    - Scheduler options and parameters
    - Augmentation options and parameters
    - Preprocessing options and parameters
    - Validation options and parameters

    This schema can be used by the frontend to dynamically generate configuration forms.
    """
    from app.schemas.configs import (
        OptimizerConfig,
        SchedulerConfig,
        AugmentationConfig,
        PreprocessConfig,
        ValidationConfig,
        TrainingConfigAdvanced,
    )

    return {
        "optimizer": OptimizerConfig.model_json_schema(),
        "scheduler": SchedulerConfig.model_json_schema(),
        "augmentation": AugmentationConfig.model_json_schema(),
        "preprocessing": PreprocessConfig.model_json_schema(),
        "validation": ValidationConfig.model_json_schema(),
        "complete": TrainingConfigAdvanced.model_json_schema(),
    }


@router.get("/config/defaults")
async def get_config_defaults():
    """
    Get default values for advanced training configuration.

    Returns sensible default values for all configuration options.
    """
    from app.schemas.configs import (
        OptimizerConfig,
        SchedulerConfig,
        AugmentationConfig,
        PreprocessConfig,
        ValidationConfig,
        TrainingConfigAdvanced,
    )

    return {
        "optimizer": OptimizerConfig().model_dump(),
        "scheduler": SchedulerConfig().model_dump(),
        "augmentation": AugmentationConfig().model_dump(),
        "preprocessing": PreprocessConfig().model_dump(),
        "validation": ValidationConfig().model_dump(),
        "complete": TrainingConfigAdvanced().model_dump(),
    }


@router.get("/config/presets")
async def get_config_presets():
    """
    Get pre-defined configuration presets.

    Returns preset configurations for common use cases:
    - basic: Simple training with minimal augmentation
    - standard: Balanced configuration for general use
    - aggressive: Heavy augmentation for small datasets
    - fine_tuning: Optimized for fine-tuning pre-trained models
    """
    from app.schemas.configs import (
        OptimizerConfig,
        SchedulerConfig,
        AugmentationConfig,
        PreprocessConfig,
        ValidationConfig,
        TrainingConfigAdvanced,
    )

    presets = {
        "basic": TrainingConfigAdvanced(
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=1e-3,
                weight_decay=0.0
            ),
            scheduler=SchedulerConfig(type="none"),
            augmentation=AugmentationConfig(
                enabled=True,
                random_flip=True,
                random_flip_prob=0.5
            ),
            preprocessing=PreprocessConfig(
                image_size=224,
                resize_mode="resize"
            ),
            validation=ValidationConfig(
                enabled=True,
                val_interval=1,
                save_best=True,
                metrics=["accuracy"]
            )
        ),

        "standard": TrainingConfigAdvanced(
            optimizer=OptimizerConfig(
                type="adamw",
                learning_rate=3e-4,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            ),
            scheduler=SchedulerConfig(
                type="cosine",
                T_max=100,
                eta_min=1e-6,
                warmup_epochs=5,
                warmup_lr=1e-6
            ),
            augmentation=AugmentationConfig(
                enabled=True,
                random_flip=True,
                random_flip_prob=0.5,
                color_jitter=True,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            preprocessing=PreprocessConfig(
                image_size=224,
                resize_mode="resize_crop"
            ),
            validation=ValidationConfig(
                enabled=True,
                val_interval=1,
                save_best=True,
                metrics=["accuracy", "precision", "recall", "f1"],
                early_stopping=True,
                early_stopping_patience=10
            ),
            mixed_precision=True
        ),

        "aggressive": TrainingConfigAdvanced(
            optimizer=OptimizerConfig(
                type="adamw",
                learning_rate=1e-3,
                weight_decay=0.05
            ),
            scheduler=SchedulerConfig(
                type="cosine_warm_restarts",
                T_0=10,
                T_mult=2,
                warmup_epochs=3
            ),
            augmentation=AugmentationConfig(
                enabled=True,
                random_flip=True,
                random_rotation=True,
                rotation_degrees=15,
                random_crop=True,
                crop_scale=(0.7, 1.0),
                color_jitter=True,
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15,
                random_erasing=True,
                erasing_prob=0.5,
                mixup=True,
                mixup_alpha=0.2,
                cutmix=True,
                cutmix_alpha=1.0
            ),
            preprocessing=PreprocessConfig(
                image_size=224,
                resize_mode="resize_crop"
            ),
            validation=ValidationConfig(
                enabled=True,
                val_interval=1,
                save_best=True,
                metrics=["accuracy", "precision", "recall", "f1"],
                early_stopping=True,
                early_stopping_patience=15
            ),
            mixed_precision=True,
            gradient_clip_value=1.0
        ),

        "fine_tuning": TrainingConfigAdvanced(
            optimizer=OptimizerConfig(
                type="sgd",
                learning_rate=1e-4,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True
            ),
            scheduler=SchedulerConfig(
                type="step",
                step_size=10,
                gamma=0.1
            ),
            augmentation=AugmentationConfig(
                enabled=True,
                random_flip=True,
                random_flip_prob=0.5,
                color_jitter=True,
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            preprocessing=PreprocessConfig(
                image_size=224,
                resize_mode="resize"
            ),
            validation=ValidationConfig(
                enabled=True,
                val_interval=1,
                save_best=True,
                save_best_metric="accuracy",
                metrics=["accuracy"],
                early_stopping=True,
                early_stopping_patience=5,
                early_stopping_min_delta=0.001
            )
        )
    }

    return {
        name: preset.model_dump()
        for name, preset in presets.items()
    }


@router.post("/config/validate")
async def validate_config(config: training.TrainingConfigAdvanced):
    """
    Validate an advanced training configuration.

    Validates the provided configuration against the schema and returns
    validation results and any warnings.

    Args:
        config: Advanced training configuration to validate

    Returns:
        Dictionary with validation status and any warnings/suggestions
    """
    from pydantic import ValidationError

    warnings = []
    suggestions = []

    # Check for common issues
    if config.optimizer.learning_rate > 0.1:
        warnings.append("Learning rate is very high (>0.1). Consider using a lower value.")

    if config.optimizer.learning_rate < 1e-6:
        warnings.append("Learning rate is very low (<1e-6). Training may be very slow.")

    if config.scheduler.type != "none" and config.scheduler.warmup_epochs > 0:
        if config.scheduler.warmup_lr >= config.optimizer.learning_rate:
            warnings.append("Warmup LR is higher than or equal to initial LR. Warmup may not work as expected.")

    if config.augmentation.enabled:
        aug_count = sum([
            config.augmentation.random_flip,
            config.augmentation.random_rotation,
            config.augmentation.random_crop,
            config.augmentation.color_jitter,
            config.augmentation.random_erasing,
            config.augmentation.mixup,
            config.augmentation.cutmix,
            config.augmentation.autoaugment
        ])

        if aug_count == 0:
            warnings.append("Augmentation is enabled but no augmentation techniques are selected.")
        elif aug_count > 6:
            suggestions.append("Many augmentation techniques are enabled. This may slow down training significantly.")

    if config.validation.early_stopping and config.validation.early_stopping_patience < 3:
        suggestions.append("Early stopping patience is very low (<3). Training may stop prematurely.")

    if config.mixed_precision and config.gradient_clip_value and config.gradient_clip_value > 10:
        suggestions.append("Gradient clipping value is high (>10) with mixed precision. Consider using a lower value.")

    return {
        "valid": True,
        "warnings": warnings,
        "suggestions": suggestions,
        "config": config.model_dump()
    }


@router.get("/config-schema")
async def get_config_schema(framework: str, task_type: str = None):
    """
    Get configuration schema for a specific framework and task type.

    This endpoint returns the configuration schema that can be used to dynamically
    generate UI forms for advanced training configuration.

    Args:
        framework: Framework name ('timm', 'ultralytics', etc.)
        task_type: Optional task type for framework-specific schemas

    Returns:
        Configuration schema with fields, types, defaults, and presets
    """
    try:
        logger.info(f"[config-schema] Requested framework={framework}, task_type={task_type}")

        # Use Training Service API to get config schema
        # This maintains dependency isolation between Backend and Training code
        from app.utils.training_client import TrainingServiceClient
        import requests

        # Initialize Training Service client for the requested framework
        client = TrainingServiceClient(framework=framework)

        logger.info(f"[config-schema] Fetching schema from Training Service: {client.base_url}")

        # Call Training Service /config-schema endpoint
        response = requests.get(
            f"{client.base_url}/config-schema",
            params={"task_type": task_type or ""},
            timeout=10
        )

        # Handle errors
        if response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Framework '{framework}' not supported or config schema not available"
            )
        elif response.status_code == 503:
            raise HTTPException(
                status_code=503,
                detail=f"Training Service for framework '{framework}' is not available"
            )
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Training Service error: {response.text}"
            )

        # Parse response
        schema_data = response.json()

        logger.info(f"[config-schema] Schema retrieved with {len(schema_data.get('schema', {}).get('fields', []))} fields")

        return schema_data

    except HTTPException:
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[config-schema] Connection error to Training Service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Training Service for framework '{framework}' is not reachable. Please check deployment."
        )
    except requests.exceptions.Timeout as e:
        logger.error(f"[config-schema] Timeout connecting to Training Service: {str(e)}")
        raise HTTPException(
            status_code=504,
            detail=f"Training Service timeout. Please try again later."
        )
    except Exception as e:
        import traceback
        logger.error(f"[config-schema] Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get configuration schema: {str(e)}"
        )


@router.get("/datasets/list")
async def list_available_datasets():
    """
    List available datasets in the default datasets directory.

    Scans the mvp/data/datasets folder and returns a list of available
    datasets with their paths and detected formats.

    Returns:
        List of dataset info objects with:
        - name: Dataset folder name
        - path: Absolute path
        - format: Detected format (imagefolder, yolo, coco, or unknown)
        - size: Number of files (approximate)
    """
    try:
        import os
        from pathlib import Path

        # Get datasets directory (mvp/data/datasets)
        backend_dir = Path(__file__).parent.parent.parent
        mvp_dir = backend_dir.parent
        datasets_dir = mvp_dir / 'data' / 'datasets'

        logger.info(f"[list-datasets] Scanning directory: {datasets_dir}")

        datasets = []

        # Create datasets directory if it doesn't exist
        if not datasets_dir.exists():
            datasets_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[list-datasets] Created datasets directory: {datasets_dir}")
            return {"datasets": []}

        # Scan for dataset folders
        for item in datasets_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                dataset_info = {
                    "name": item.name,
                    "path": str(item.absolute()),
                    "format": "unknown",
                    "size": 0
                }

                # Detect format by folder structure
                if (item / 'train').exists() and (item / 'val').exists():
                    # Could be ImageFolder or YOLO
                    if (item / 'labels').exists():
                        dataset_info["format"] = "yolo"
                    else:
                        # Check if train has class subfolders (ImageFolder)
                        train_dir = item / 'train'
                        subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
                        if subdirs:
                            dataset_info["format"] = "imagefolder"
                        else:
                            dataset_info["format"] = "yolo"

                elif (item / 'annotations').exists():
                    # Likely COCO format
                    dataset_info["format"] = "coco"

                # Count total files (approximate)
                try:
                    file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                    dataset_info["size"] = file_count
                except Exception as e:
                    logger.warning(f"Failed to count files in {item}: {e}")

                datasets.append(dataset_info)

        logger.info(f"[list-datasets] Found {len(datasets)} datasets")

        return {
            "datasets": datasets,
            "datasets_dir": str(datasets_dir.absolute())
        }

    except Exception as e:
        import traceback
        logger.error(f"[list-datasets] Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list datasets: {str(e)}"
        )


@router.get("/jobs/{job_id}/checkpoints")
async def get_job_checkpoints(
    job_id: int,
    db: Session = Depends(get_db),
):
    """
    Get available checkpoints for a training job.

    Returns a list of checkpoints with epoch numbers and file paths.
    """
    try:
        # Get training job
        job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        checkpoints = []

        # Check if output directory exists
        if not job.output_dir:
            return {"checkpoints": []}

        from pathlib import Path
        output_path = Path(job.output_dir)

        if not output_path.exists():
            return {"checkpoints": []}

        # Scan for checkpoint files
        # Common patterns: best.pt, last.pt, epoch_*.pt, checkpoint_*.pth
        checkpoint_files = []

        # Look for .pt files (PyTorch/YOLO) - direct children
        checkpoint_files.extend(list(output_path.glob("*.pt")))
        # Look for .pth files (PyTorch) - direct children
        checkpoint_files.extend(list(output_path.glob("*.pth")))

        # Look in weights subdirectory (YOLO format)
        weights_dir = output_path / "weights"
        if weights_dir.exists():
            checkpoint_files.extend(list(weights_dir.glob("*.pt")))

        # Recursively search for checkpoints in subdirectories (YOLO job_X/weights structure)
        # This handles cases like: output_dir/job_3/weights/best.pt
        for weights_subdir in output_path.glob("*/weights"):
            if weights_subdir.is_dir():
                checkpoint_files.extend(list(weights_subdir.glob("*.pt")))
                checkpoint_files.extend(list(weights_subdir.glob("*.pth")))

        # Parse checkpoint information
        for ckpt_file in checkpoint_files:
            checkpoint_info = {
                "path": str(ckpt_file.absolute()),
                "filename": ckpt_file.name,
                "is_best": False,
                "epoch": None
            }

            # Determine if it's the best checkpoint
            if ckpt_file.name in ["best.pt", "best.pth", "model_best.pth"]:
                checkpoint_info["is_best"] = True
                # Try to get epoch from metrics
                metrics = db.query(models.TrainingMetric).filter(
                    models.TrainingMetric.job_id == job_id
                ).order_by(models.TrainingMetric.accuracy.desc()).first()
                if metrics:
                    checkpoint_info["epoch"] = metrics.epoch

            # Try to extract epoch from filename
            elif "epoch" in ckpt_file.name.lower():
                import re
                match = re.search(r'epoch[_-]?(\d+)', ckpt_file.name, re.IGNORECASE)
                if match:
                    checkpoint_info["epoch"] = int(match.group(1))

            elif ckpt_file.name == "last.pt":
                # Get last epoch from metrics
                metrics = db.query(models.TrainingMetric).filter(
                    models.TrainingMetric.job_id == job_id
                ).order_by(models.TrainingMetric.epoch.desc()).first()
                if metrics:
                    checkpoint_info["epoch"] = metrics.epoch

            checkpoints.append(checkpoint_info)

        # Sort by epoch (None values at end)
        checkpoints.sort(key=lambda x: (x["epoch"] is None, x["epoch"] or 0))

        logger.info(f"[get-checkpoints] Found {len(checkpoints)} checkpoints for job {job_id}")

        return {"checkpoints": checkpoints}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"[get-checkpoints] Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get checkpoints: {str(e)}"
        )
