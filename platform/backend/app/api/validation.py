"""
Validation API endpoints.

Provides access to validation results, metrics, and image-level results.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from pathlib import Path

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
    true_label_id: Optional[int] = Query(None, description="Filter by true label ID"),
    predicted_label_id: Optional[int] = Query(None, description="Filter by predicted label ID"),
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
        true_label_id: Filter by true label ID (null for all)
        predicted_label_id: Filter by predicted label ID (null for all)
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

    # Apply label filters if specified
    if true_label_id is not None:
        query = query.filter(models.ValidationImageResult.true_label_id == true_label_id)

    if predicted_label_id is not None:
        query = query.filter(models.ValidationImageResult.predicted_label_id == predicted_label_id)

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
        # Handle legacy top5_predictions format (list of ints -> list of dicts)
        top5_predictions = img_result.top5_predictions
        if top5_predictions and len(top5_predictions) > 0:
            if isinstance(top5_predictions[0], int):
                # Legacy format: [0, 4, 3, 9, 1] -> convert to new format
                top5_predictions = [{"label_id": idx, "confidence": None} for idx in top5_predictions]

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
            "top5_predictions": top5_predictions,
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

    # Get class names from validation result
    class_names = None
    if validation_result.class_names:
        if isinstance(validation_result.class_names, str):
            import json
            class_names = json.loads(validation_result.class_names)
        elif isinstance(validation_result.class_names, list):
            class_names = validation_result.class_names

    return validation_schemas.ValidationImageResultListResponse(
        validation_result_id=validation_result.id,
        job_id=job_id,
        epoch=epoch,
        total_count=total_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        class_names=class_names,
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

    # Find last epoch
    last_result = max(results, key=lambda r: r.epoch)

    # Build epoch-wise metrics for charting
    epoch_metrics = []
    for result in results:
        # Determine checkpoint_path from TrainingJob (Option A)
        # - best epoch → job.best_checkpoint_path
        # - last epoch → job.last_checkpoint_path
        # - other epochs → None (checkpoint not saved for intermediate epochs)
        checkpoint_path = None
        if result.epoch == best_result.epoch and job.best_checkpoint_path:
            checkpoint_path = job.best_checkpoint_path
        elif result.epoch == last_result.epoch and job.last_checkpoint_path:
            checkpoint_path = job.last_checkpoint_path

        metrics_dict = {
            "epoch": result.epoch,
            "primary_metric": result.primary_metric_value,
            "loss": result.overall_loss,
            "checkpoint_path": checkpoint_path,  # From TrainingJob best/last checkpoint
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


@router.get("/images/{image_result_id}")
async def get_validation_image(
    image_result_id: int,
    db: Session = Depends(get_db)
):
    """
    Serve validation image file.

    Returns the actual image file for a validation result.

    Args:
        image_result_id: ValidationImageResult ID
        db: Database session

    Returns:
        FileResponse with the image file
    """
    # Get image result
    image_result = db.query(models.ValidationImageResult).filter(
        models.ValidationImageResult.id == image_result_id
    ).first()

    if not image_result:
        raise HTTPException(status_code=404, detail=f"Image result {image_result_id} not found")

    if not image_result.image_path:
        raise HTTPException(status_code=404, detail=f"Image path not available for result {image_result_id}")

    # Get training job to get dataset_path
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == image_result.job_id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {image_result.job_id} not found")

    # Convert Docker container path to host path if needed
    image_path_str = image_result.image_path
    if image_path_str.startswith('/workspace/dataset/'):
        # Replace container path with host path
        relative_path = image_path_str.replace('/workspace/dataset/', '')
        image_path = Path(job.dataset_path) / relative_path
        print(f"[INFO] Converted image path: {image_path_str} -> {image_path}")
    else:
        image_path = Path(image_path_str)

    # Check if file exists
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found: {image_path}"
        )

    # Determine media type based on extension
    extension = image_path.suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_type_map.get(extension, 'application/octet-stream')

    return FileResponse(
        path=str(image_path),
        media_type=media_type,
        filename=image_result.image_name
    )


@router.get("/jobs/{job_id}/results/{epoch}/pr_curve")
async def get_pr_curve_image(
    job_id: int,
    epoch: int,
    db: Session = Depends(get_db)
):
    """
    Serve PR curve image for a specific validation epoch.

    Returns the PR (Precision-Recall) curve visualization generated by the training framework.

    Args:
        job_id: Training job ID
        epoch: Epoch number
        db: Database session

    Returns:
        FileResponse with the PR curve image
    """
    # Get validation result
    result = db.query(models.ValidationResult).filter(
        models.ValidationResult.job_id == job_id,
        models.ValidationResult.epoch == epoch
    ).first()

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Validation result not found for job {job_id}, epoch {epoch}"
        )

    # Check if PR curve data exists
    if not result.pr_curves:
        raise HTTPException(
            status_code=404,
            detail=f"PR curve not available for this validation result"
        )

    # Extract image path from pr_curves JSON
    pr_curves = result.pr_curves if isinstance(result.pr_curves, dict) else json.loads(result.pr_curves)
    image_path_str = pr_curves.get('image_path')

    if not image_path_str:
        raise HTTPException(
            status_code=404,
            detail=f"PR curve image path not found in validation result"
        )

    # Check if file exists
    image_path = Path(image_path_str)
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PR curve image file not found: {image_path_str}"
        )

    return FileResponse(
        path=str(image_path),
        media_type='image/png',
        filename=f'pr_curve_job{job_id}_epoch{epoch}.png'
    )


# ========== POST Endpoints (Callbacks from Trainer) ==========

@router.post("/jobs/{job_id}/results")
async def create_validation_result(
    job_id: int,
    callback: validation_schemas.ValidationCallbackRequest,
    db: Session = Depends(get_db)
):
    """
    Validation callback endpoint (Trainer -> Backend).

    Called by training runner after each validation run.
    Creates ValidationResult and optionally ValidationImageResult records.

    Args:
        job_id: Training job ID
        callback: Validation callback data from trainer
        db: Database session

    Returns:
        Confirmation with validation_result_id
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"[VALIDATION CALLBACK] Received for job {job_id}, epoch {callback.epoch}")

    # Verify job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # Check if validation result already exists for this epoch
    existing = db.query(models.ValidationResult).filter(
        models.ValidationResult.job_id == job_id,
        models.ValidationResult.epoch == callback.epoch
    ).first()

    if existing:
        logger.warning(f"[VALIDATION CALLBACK] Result already exists for job {job_id}, epoch {callback.epoch}. Updating.")
        # Update existing record
        existing.task_type = callback.task_type
        existing.primary_metric_name = callback.primary_metric_name
        existing.primary_metric_value = callback.primary_metric_value
        existing.overall_loss = callback.overall_loss
        existing.metrics = callback.metrics
        existing.per_class_metrics = callback.per_class_metrics
        existing.confusion_matrix = callback.confusion_matrix
        existing.pr_curves = callback.pr_curves
        existing.class_names = callback.class_names
        existing.sample_correct_images = callback.sample_correct_images
        existing.sample_incorrect_images = callback.sample_incorrect_images

        # Store visualization_urls in visualization_data
        if callback.visualization_urls:
            existing.visualization_data = existing.visualization_data or {}
            existing.visualization_data['urls'] = callback.visualization_urls

        validation_result = existing
    else:
        # Create new validation result
        validation_result = models.ValidationResult(
            job_id=job_id,
            epoch=callback.epoch,
            task_type=callback.task_type,
            primary_metric_name=callback.primary_metric_name,
            primary_metric_value=callback.primary_metric_value,
            overall_loss=callback.overall_loss,
            metrics=callback.metrics,
            per_class_metrics=callback.per_class_metrics,
            confusion_matrix=callback.confusion_matrix,
            pr_curves=callback.pr_curves,
            class_names=callback.class_names,
            sample_correct_images=callback.sample_correct_images,
            sample_incorrect_images=callback.sample_incorrect_images,
            visualization_data={'urls': callback.visualization_urls} if callback.visualization_urls else None
        )
        db.add(validation_result)

    db.commit()
    db.refresh(validation_result)

    logger.info(f"[VALIDATION CALLBACK] Created ValidationResult id={validation_result.id}")

    # Process image-level results if provided
    if callback.image_results:
        logger.info(f"[VALIDATION CALLBACK] Processing {len(callback.image_results)} image results")

        # Delete existing image results for this epoch (if updating)
        db.query(models.ValidationImageResult).filter(
            models.ValidationImageResult.job_id == job_id,
            models.ValidationImageResult.epoch == callback.epoch
        ).delete()

        # Create new image results
        for img_data in callback.image_results:
            image_result = models.ValidationImageResult(
                validation_result_id=validation_result.id,
                job_id=job_id,
                epoch=callback.epoch,
                image_name=img_data.image_name,
                image_index=img_data.image_index,
                true_label=img_data.true_label,
                true_label_id=img_data.true_label_id,
                predicted_label=img_data.predicted_label,
                predicted_label_id=img_data.predicted_label_id,
                confidence=img_data.confidence,
                top5_predictions=img_data.top5_predictions,
                true_boxes=img_data.true_boxes,
                predicted_boxes=img_data.predicted_boxes,
                is_correct=img_data.is_correct,
                iou=img_data.iou,
                extra_data=img_data.extra_data
            )
            db.add(image_result)

        db.commit()
        logger.info(f"[VALIDATION CALLBACK] Created {len(callback.image_results)} ValidationImageResult records")

    # TODO: Broadcast validation result via WebSocket
    # ws_manager.broadcast_to_job(job_id, {
    #     'type': 'validation_result',
    #     'epoch': callback.epoch,
    #     'primary_metric': callback.primary_metric_value
    # })

    return {
        "status": "success",
        "validation_result_id": validation_result.id,
        "job_id": job_id,
        "epoch": callback.epoch
    }
