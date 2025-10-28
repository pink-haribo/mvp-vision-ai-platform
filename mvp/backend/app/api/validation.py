"""
Validation API endpoints.

Provides access to validation results, metrics, and image-level results.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import json

from app.db.database import get_db
from app.db import models
from app.schemas import validation as validation_schemas


router = APIRouter(prefix="/validation", tags=["validation"])


@router.get("/jobs/{job_id}/results", response_model=validation_schemas.ValidationResultListResponse)
async def get_validation_results(
    job_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get all validation results for a training job.

    Returns validation metrics for all epochs, ordered by epoch.

    Args:
        job_id: Training job ID
        skip: Number of results to skip (pagination)
        limit: Maximum number of results to return
        db: Database session

    Returns:
        ValidationResultListResponse with all validation results
    """
    # Check if job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # Query validation results
    query = db.query(models.ValidationResult).filter(
        models.ValidationResult.job_id == job_id
    ).order_by(models.ValidationResult.epoch)

    total_count = query.count()
    results = query.offset(skip).limit(limit).all()

    # SQLAlchemy JSON type already deserializes, no need for json.loads()
    results_data = []
    for result in results:
        result_dict = {
            "id": result.id,
            "job_id": result.job_id,
            "epoch": result.epoch,
            "task_type": result.task_type,
            "primary_metric_name": result.primary_metric_name,
            "primary_metric_value": result.primary_metric_value,
            "overall_loss": result.overall_loss,
            "metrics": result.metrics,
            "per_class_metrics": result.per_class_metrics,
            "confusion_matrix": result.confusion_matrix,
            "pr_curves": result.pr_curves,
            "class_names": result.class_names,
            "visualization_data": result.visualization_data,
            "sample_correct_images": result.sample_correct_images,
            "sample_incorrect_images": result.sample_incorrect_images,
            "created_at": result.created_at,
        }
        results_data.append(validation_schemas.ValidationResultResponse(**result_dict))

    return validation_schemas.ValidationResultListResponse(
        job_id=job_id,
        total_count=total_count,
        results=results_data
    )


@router.get("/jobs/{job_id}/results/{epoch}", response_model=validation_schemas.ValidationResultResponse)
async def get_validation_result_by_epoch(
    job_id: int,
    epoch: int,
    db: Session = Depends(get_db)
):
    """
    Get validation result for a specific epoch.

    Returns detailed validation metrics including confusion matrix,
    per-class metrics, and visualization data.

    Args:
        job_id: Training job ID
        epoch: Epoch number
        db: Database session

    Returns:
        ValidationResultResponse with detailed metrics
    """
    # Query validation result
    result = db.query(models.ValidationResult).filter(
        models.ValidationResult.job_id == job_id,
        models.ValidationResult.epoch == epoch
    ).first()

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Validation result not found for job {job_id}, epoch {epoch}"
        )

    # SQLAlchemy JSON type already deserializes, no need for json.loads()
    result_dict = {
        "id": result.id,
        "job_id": result.job_id,
        "epoch": result.epoch,
        "task_type": result.task_type,
        "primary_metric_name": result.primary_metric_name,
        "primary_metric_value": result.primary_metric_value,
        "overall_loss": result.overall_loss,
        "metrics": result.metrics,
        "per_class_metrics": result.per_class_metrics,
        "confusion_matrix": result.confusion_matrix,
        "pr_curves": result.pr_curves,
        "class_names": result.class_names,
        "visualization_data": result.visualization_data,
        "sample_correct_images": result.sample_correct_images,
        "sample_incorrect_images": result.sample_incorrect_images,
        "created_at": result.created_at,
    }

    return validation_schemas.ValidationResultResponse(**result_dict)


@router.get("/jobs/{job_id}/results/{epoch}/images", response_model=validation_schemas.ValidationImageResultListResponse)
async def get_validation_images(
    job_id: int,
    epoch: int,
    correct_only: Optional[bool] = Query(None, description="Filter by correctness: true=correct, false=incorrect, null=all"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get image-level validation results for a specific epoch.

    Returns per-image predictions, labels, and metrics.
    Useful for error analysis and visualization.

    Args:
        job_id: Training job ID
        epoch: Epoch number
        correct_only: Filter by correctness (true/false/null for all)
        skip: Number of results to skip (pagination)
        limit: Maximum number of results to return
        db: Database session

    Returns:
        ValidationImageResultListResponse with image-level results
    """
    # Get validation result ID
    validation_result = db.query(models.ValidationResult).filter(
        models.ValidationResult.job_id == job_id,
        models.ValidationResult.epoch == epoch
    ).first()

    if not validation_result:
        raise HTTPException(
            status_code=404,
            detail=f"Validation result not found for job {job_id}, epoch {epoch}"
        )

    # Query image results
    query = db.query(models.ValidationImageResult).filter(
        models.ValidationImageResult.validation_result_id == validation_result.id
    )

    # Apply correctness filter if specified
    if correct_only is not None:
        query = query.filter(models.ValidationImageResult.is_correct == correct_only)

    # Count results
    total_count = query.count()
    correct_count = db.query(models.ValidationImageResult).filter(
        models.ValidationImageResult.validation_result_id == validation_result.id,
        models.ValidationImageResult.is_correct == True
    ).count()
    incorrect_count = total_count - correct_count

    # Get paginated results
    image_results = query.order_by(models.ValidationImageResult.image_index).offset(skip).limit(limit).all()

    # SQLAlchemy JSON type already deserializes, no need for json.loads()
    images_data = []
    for img_result in image_results:
        img_dict = {
            "id": img_result.id,
            "validation_result_id": img_result.validation_result_id,
            "job_id": img_result.job_id,
            "epoch": img_result.epoch,
            "image_path": img_result.image_path,
            "image_name": img_result.image_name,
            "image_index": img_result.image_index,
            "true_label": img_result.true_label,
            "true_label_id": img_result.true_label_id,
            "predicted_label": img_result.predicted_label,
            "predicted_label_id": img_result.predicted_label_id,
            "confidence": img_result.confidence,
            "top5_predictions": img_result.top5_predictions,
            "true_boxes": img_result.true_boxes,
            "predicted_boxes": img_result.predicted_boxes,
            "true_mask_path": img_result.true_mask_path,
            "predicted_mask_path": img_result.predicted_mask_path,
            "true_keypoints": img_result.true_keypoints,
            "predicted_keypoints": img_result.predicted_keypoints,
            "is_correct": img_result.is_correct,
            "iou": img_result.iou,
            "oks": img_result.oks,
            "extra_data": img_result.extra_data,
            "created_at": img_result.created_at,
        }
        images_data.append(validation_schemas.ValidationImageResultResponse(**img_dict))

    return validation_schemas.ValidationImageResultListResponse(
        validation_result_id=validation_result.id,
        job_id=job_id,
        epoch=epoch,
        total_count=total_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        images=images_data
    )


@router.get("/jobs/{job_id}/summary", response_model=validation_schemas.ValidationSummaryResponse)
async def get_validation_summary(
    job_id: int,
    db: Session = Depends(get_db)
):
    """
    Get validation summary for a training job.

    Returns best epoch, metrics trends, and overall performance summary.

    Args:
        job_id: Training job ID
        db: Database session

    Returns:
        ValidationSummaryResponse with summary statistics
    """
    # Check if job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # Query all validation results
    results = db.query(models.ValidationResult).filter(
        models.ValidationResult.job_id == job_id
    ).order_by(models.ValidationResult.epoch).all()

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No validation results found for job {job_id}"
        )

    # Get task type
    task_type = results[0].task_type

    # Find best epoch (highest primary metric value)
    best_result = max(results, key=lambda r: r.primary_metric_value if r.primary_metric_value else 0)

    # Build epoch-wise metrics for charting
    epoch_metrics = []
    for result in results:
        metrics_dict = {
            "epoch": result.epoch,
            "primary_metric": result.primary_metric_value,
            "loss": result.overall_loss,
        }
        # Add task-specific metrics (already deserialized by SQLAlchemy)
        if result.metrics:
            metrics_dict.update(result.metrics)

        epoch_metrics.append(metrics_dict)

    return validation_schemas.ValidationSummaryResponse(
        job_id=job_id,
        task_type=task_type,
        total_epochs=len(results),
        best_epoch=best_result.epoch,
        best_metric_value=best_result.primary_metric_value,
        best_metric_name=best_result.primary_metric_name,
        epoch_metrics=epoch_metrics
    )
