"""
Test and Inference API endpoints.

Provides endpoints for:
- Test: Running tests on labeled datasets with metrics
- Inference: Running predictions on unlabeled data
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime

from app.db.database import get_db
from app.db import models
from app.schemas import test_inference as ti_schemas
from app.utils.test_inference_runner import TestRunner, InferenceRunner


router = APIRouter(prefix="/test_inference", tags=["test_inference"])


# Background task functions
def run_test_task(test_run_id: int):
    """Background task to run test."""
    from app.db.database import SessionLocal
    db = SessionLocal()
    try:
        runner = TestRunner(db)
        runner.run_test(test_run_id)
    finally:
        db.close()


def run_inference_task(inference_job_id: int):
    """Background task to run inference."""
    from app.db.database import SessionLocal
    db = SessionLocal()
    try:
        runner = InferenceRunner(db)
        runner.run_inference(inference_job_id)
    finally:
        db.close()


# ========== Test Endpoints ==========

@router.post("/test/runs", response_model=ti_schemas.TestRunResponse, status_code=201)
async def create_test_run(
    request: ti_schemas.TestRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new test run for a trained model.

    Runs inference on a labeled test dataset and computes metrics.

    Args:
        request: Test run configuration
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        TestRunResponse with test run metadata
    """
    # Check if training job exists
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == request.training_job_id
    ).first()

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {request.training_job_id} not found"
        )

    # Check if checkpoint exists
    checkpoint_path = Path(request.checkpoint_path)
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {request.checkpoint_path}"
        )

    # Check if dataset exists
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {request.dataset_path}"
        )

    # Create test run record
    test_run = models.TestRun(
        training_job_id=request.training_job_id,
        checkpoint_path=request.checkpoint_path,
        dataset_path=request.dataset_path,
        dataset_split=request.dataset_split,
        status="pending",
        task_type=job.task_type,
        created_at=datetime.utcnow()
    )

    db.add(test_run)
    db.commit()
    db.refresh(test_run)

    # Launch background task to run test
    background_tasks.add_task(run_test_task, test_run.id)

    return ti_schemas.TestRunResponse(**test_run.__dict__)


@router.get("/test/jobs/{job_id}/runs", response_model=ti_schemas.TestRunListResponse)
async def get_test_runs(
    job_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get all test runs for a training job.

    Returns test results ordered by creation time.

    Args:
        job_id: Training job ID
        skip: Number of results to skip (pagination)
        limit: Maximum number of results to return
        db: Database session

    Returns:
        TestRunListResponse with all test runs
    """
    # Check if job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # Query test runs
    query = db.query(models.TestRun).filter(
        models.TestRun.training_job_id == job_id
    ).order_by(models.TestRun.created_at.desc())

    total_count = query.count()
    test_runs = query.offset(skip).limit(limit).all()

    test_runs_data = [ti_schemas.TestRunResponse(**run.__dict__) for run in test_runs]

    return ti_schemas.TestRunListResponse(
        training_job_id=job_id,
        total_count=total_count,
        test_runs=test_runs_data
    )


@router.get("/test/runs/{test_run_id}", response_model=ti_schemas.TestRunResponse)
async def get_test_run(
    test_run_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific test run.

    Returns detailed test metrics including confusion matrix and per-class metrics.

    Args:
        test_run_id: Test run ID
        db: Database session

    Returns:
        TestRunResponse with detailed metrics
    """
    test_run = db.query(models.TestRun).filter(
        models.TestRun.id == test_run_id
    ).first()

    if not test_run:
        raise HTTPException(
            status_code=404,
            detail=f"Test run {test_run_id} not found"
        )

    return ti_schemas.TestRunResponse(**test_run.__dict__)


@router.get("/test/runs/{test_run_id}/images", response_model=ti_schemas.TestImageResultListResponse)
async def get_test_images(
    test_run_id: int,
    correct_only: Optional[bool] = Query(None, description="Filter by correctness"),
    true_label_id: Optional[int] = Query(None, description="Filter by true label ID"),
    predicted_label_id: Optional[int] = Query(None, description="Filter by predicted label ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get image-level test results for a test run.

    Returns per-image predictions, labels, and metrics.
    Useful for error analysis.

    Args:
        test_run_id: Test run ID
        correct_only: Filter by correctness (true/false/null for all)
        true_label_id: Filter by true label ID (null for all)
        predicted_label_id: Filter by predicted label ID (null for all)
        skip: Number of results to skip (pagination)
        limit: Maximum number of results to return
        db: Database session

    Returns:
        TestImageResultListResponse with image-level results
    """
    # Get test run
    test_run = db.query(models.TestRun).filter(
        models.TestRun.id == test_run_id
    ).first()

    if not test_run:
        raise HTTPException(
            status_code=404,
            detail=f"Test run {test_run_id} not found"
        )

    # Query image results
    query = db.query(models.TestImageResult).filter(
        models.TestImageResult.test_run_id == test_run_id
    )

    # Apply filters
    if correct_only is not None:
        query = query.filter(models.TestImageResult.is_correct == correct_only)

    if true_label_id is not None:
        query = query.filter(models.TestImageResult.true_label_id == true_label_id)

    if predicted_label_id is not None:
        query = query.filter(models.TestImageResult.predicted_label_id == predicted_label_id)

    # Count results
    total_count = query.count()
    correct_count = db.query(models.TestImageResult).filter(
        models.TestImageResult.test_run_id == test_run_id,
        models.TestImageResult.is_correct == True
    ).count()
    incorrect_count = total_count - correct_count

    # Get paginated results
    image_results = query.order_by(models.TestImageResult.image_index).offset(skip).limit(limit).all()

    images_data = [ti_schemas.TestImageResultResponse(**img.__dict__) for img in image_results]

    return ti_schemas.TestImageResultListResponse(
        test_run_id=test_run_id,
        training_job_id=test_run.training_job_id,
        total_count=total_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        class_names=test_run.class_names,
        images=images_data
    )


@router.get("/test/jobs/{job_id}/summary", response_model=ti_schemas.TestSummaryResponse)
async def get_test_summary(
    job_id: int,
    db: Session = Depends(get_db)
):
    """
    Get test summary for a training job.

    Returns best test run and overall test performance.

    Args:
        job_id: Training job ID
        db: Database session

    Returns:
        TestSummaryResponse with summary statistics
    """
    # Check if job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # Query all test runs
    test_runs = db.query(models.TestRun).filter(
        models.TestRun.training_job_id == job_id,
        models.TestRun.status == "completed"
    ).order_by(models.TestRun.created_at.desc()).all()

    if not test_runs:
        raise HTTPException(
            status_code=404,
            detail=f"No completed test runs found for job {job_id}"
        )

    # Find best test run
    best_run = max(test_runs, key=lambda r: r.primary_metric_value if r.primary_metric_value else 0)

    test_runs_data = [ti_schemas.TestRunResponse(**run.__dict__) for run in test_runs]

    return ti_schemas.TestSummaryResponse(
        training_job_id=job_id,
        task_type=job.task_type,
        total_test_runs=len(test_runs),
        best_test_run_id=best_run.id if best_run else None,
        best_metric_value=best_run.primary_metric_value if best_run else None,
        best_metric_name=best_run.primary_metric_name if best_run else None,
        test_runs=test_runs_data
    )


# ========== Inference Endpoints ==========

@router.post("/inference/jobs", response_model=ti_schemas.InferenceJobResponse, status_code=201)
async def create_inference_job(
    request: ti_schemas.InferenceRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new inference job for a trained model.

    Runs predictions on unlabeled data without computing metrics.

    Args:
        request: Inference job configuration
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        InferenceJobResponse with job metadata
    """
    # Check if training job exists
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == request.training_job_id
    ).first()

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {request.training_job_id} not found"
        )

    # Check if checkpoint exists
    checkpoint_path = Path(request.checkpoint_path)
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {request.checkpoint_path}"
        )

    # Validate inference type
    if request.inference_type not in ["single", "batch", "dataset"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid inference type: {request.inference_type}. Must be 'single', 'batch', or 'dataset'"
        )

    # Create inference job record
    inference_job = models.InferenceJob(
        training_job_id=request.training_job_id,
        checkpoint_path=request.checkpoint_path,
        inference_type=request.inference_type,
        input_data=json.dumps(request.input_data) if request.input_data else None,
        status="pending",
        task_type=job.task_type,
        created_at=datetime.utcnow()
    )

    db.add(inference_job)
    db.commit()
    db.refresh(inference_job)

    # Launch background task to run inference
    background_tasks.add_task(run_inference_task, inference_job.id)

    # Parse input_data back to dict for response
    response_dict = inference_job.__dict__.copy()
    if isinstance(response_dict.get('input_data'), str):
        response_dict['input_data'] = json.loads(response_dict['input_data'])

    return ti_schemas.InferenceJobResponse(**response_dict)


@router.get("/inference/jobs/{job_id}", response_model=ti_schemas.InferenceJobListResponse)
async def get_inference_jobs(
    job_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get all inference jobs for a training job.

    Returns inference jobs ordered by creation time.

    Args:
        job_id: Training job ID
        skip: Number of results to skip (pagination)
        limit: Maximum number of results to return
        db: Database session

    Returns:
        InferenceJobListResponse with all inference jobs
    """
    # Check if job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # Query inference jobs
    query = db.query(models.InferenceJob).filter(
        models.InferenceJob.training_job_id == job_id
    ).order_by(models.InferenceJob.created_at.desc())

    total_count = query.count()
    inference_jobs = query.offset(skip).limit(limit).all()

    # Parse input_data JSON strings back to dicts
    jobs_data = []
    for inf_job in inference_jobs:
        job_dict = inf_job.__dict__.copy()
        if isinstance(job_dict.get('input_data'), str):
            job_dict['input_data'] = json.loads(job_dict['input_data'])
        jobs_data.append(ti_schemas.InferenceJobResponse(**job_dict))

    return ti_schemas.InferenceJobListResponse(
        training_job_id=job_id,
        total_count=total_count,
        inference_jobs=jobs_data
    )


@router.get("/inference/jobs/{inference_job_id}", response_model=ti_schemas.InferenceJobResponse)
async def get_inference_job(
    inference_job_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific inference job.

    Returns inference job metadata and performance statistics.

    Args:
        inference_job_id: Inference job ID
        db: Database session

    Returns:
        InferenceJobResponse with job details
    """
    inference_job = db.query(models.InferenceJob).filter(
        models.InferenceJob.id == inference_job_id
    ).first()

    if not inference_job:
        raise HTTPException(
            status_code=404,
            detail=f"Inference job {inference_job_id} not found"
        )

    # Parse input_data JSON string back to dict
    job_dict = inference_job.__dict__.copy()
    if isinstance(job_dict.get('input_data'), str):
        job_dict['input_data'] = json.loads(job_dict['input_data'])

    return ti_schemas.InferenceJobResponse(**job_dict)


@router.get("/inference/jobs/{inference_job_id}/results", response_model=ti_schemas.InferenceResultListResponse)
async def get_inference_results(
    inference_job_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get inference results for an inference job.

    Returns per-image predictions without ground truth.

    Args:
        inference_job_id: Inference job ID
        skip: Number of results to skip (pagination)
        limit: Maximum number of results to return
        db: Database session

    Returns:
        InferenceResultListResponse with predictions
    """
    # Get inference job
    inference_job = db.query(models.InferenceJob).filter(
        models.InferenceJob.id == inference_job_id
    ).first()

    if not inference_job:
        raise HTTPException(
            status_code=404,
            detail=f"Inference job {inference_job_id} not found"
        )

    # Query inference results
    query = db.query(models.InferenceResult).filter(
        models.InferenceResult.inference_job_id == inference_job_id
    )

    total_count = query.count()
    results = query.order_by(models.InferenceResult.image_index).offset(skip).limit(limit).all()

    results_data = [ti_schemas.InferenceResultResponse(**res.__dict__) for res in results]

    return ti_schemas.InferenceResultListResponse(
        inference_job_id=inference_job_id,
        training_job_id=inference_job.training_job_id,
        total_count=total_count,
        avg_inference_time_ms=inference_job.avg_inference_time_ms,
        results=results_data
    )


@router.get("/inference/jobs/training/{job_id}/summary", response_model=ti_schemas.InferenceSummaryResponse)
async def get_inference_summary(
    job_id: int,
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get inference summary for a training job.

    Returns overall inference usage statistics and recent jobs.

    Args:
        job_id: Training job ID
        limit: Maximum number of recent jobs to return
        db: Database session

    Returns:
        InferenceSummaryResponse with usage statistics
    """
    # Check if job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # Query all completed inference jobs
    inference_jobs = db.query(models.InferenceJob).filter(
        models.InferenceJob.training_job_id == job_id,
        models.InferenceJob.status == "completed"
    ).order_by(models.InferenceJob.created_at.desc()).all()

    # Calculate total images processed
    total_images = sum(inf_job.total_images for inf_job in inference_jobs)

    # Calculate average inference time across all jobs
    total_inference_times = [
        inf_job.avg_inference_time_ms
        for inf_job in inference_jobs
        if inf_job.avg_inference_time_ms is not None
    ]
    avg_inference_time = sum(total_inference_times) / len(total_inference_times) if total_inference_times else None

    # Get recent jobs
    recent_jobs = inference_jobs[:limit]

    # Parse input_data JSON strings
    recent_jobs_data = []
    for inf_job in recent_jobs:
        job_dict = inf_job.__dict__.copy()
        if isinstance(job_dict.get('input_data'), str):
            job_dict['input_data'] = json.loads(job_dict['input_data'])
        recent_jobs_data.append(ti_schemas.InferenceJobResponse(**job_dict))

    return ti_schemas.InferenceSummaryResponse(
        training_job_id=job_id,
        task_type=job.task_type,
        total_inference_jobs=len(inference_jobs),
        total_images_processed=total_images,
        avg_inference_time_ms=avg_inference_time,
        recent_jobs=recent_jobs_data
    )


# ========== Image Serving Endpoints ==========

@router.get("/test/images/{image_result_id}")
async def get_test_image(
    image_result_id: int,
    db: Session = Depends(get_db)
):
    """
    Serve test image file.

    Returns the actual image file for a test result.

    Args:
        image_result_id: TestImageResult ID
        db: Database session

    Returns:
        FileResponse with the image file
    """
    image_result = db.query(models.TestImageResult).filter(
        models.TestImageResult.id == image_result_id
    ).first()

    if not image_result:
        raise HTTPException(status_code=404, detail=f"Image result {image_result_id} not found")

    if not image_result.image_path:
        raise HTTPException(status_code=404, detail=f"Image path not available")

    image_path = Path(image_result.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_result.image_path}")

    # Determine media type
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


@router.get("/inference/images/{result_id}")
async def get_inference_image(
    result_id: int,
    db: Session = Depends(get_db)
):
    """
    Serve inference result image file.

    Returns the actual image file for an inference result.

    Args:
        result_id: InferenceResult ID
        db: Database session

    Returns:
        FileResponse with the image file
    """
    result = db.query(models.InferenceResult).filter(
        models.InferenceResult.id == result_id
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail=f"Inference result {result_id} not found")

    if not result.image_path:
        raise HTTPException(status_code=404, detail=f"Image path not available")

    image_path = Path(result.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {result.image_path}")

    # Determine media type
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
        filename=result.image_name
    )


# ========== Quick Inference Endpoints (for UI) ==========

@router.post("/inference/upload-image")
async def upload_inference_image(
    file: UploadFile = File(...),
):
    """
    Upload an image for inference.

    Saves the image to a temporary location and returns the server path.

    Args:
        file: Image file to upload

    Returns:
        Dict with server_path
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Create temp directory for inference images
        from app.core.config import settings
        temp_dir = Path(settings.UPLOAD_DIR) / "inference_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        file_extension = Path(file.filename or "image.jpg").suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = temp_dir / unique_filename

        # Save file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "server_path": str(file_path.absolute()),
            "filename": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Image upload failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image: {str(e)}"
        )


@router.post("/inference/quick")
async def quick_inference(
    training_job_id: int,
    checkpoint_path: str,
    image_path: str,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    top_k: int = 5,
    db: Session = Depends(get_db)
):
    """
    Run quick inference on a single image.

    This endpoint is optimized for UI interactions - runs inference immediately
    and returns results without creating database records.

    Args:
        training_job_id: Training job ID
        checkpoint_path: Path to checkpoint file
        image_path: Path to image file
        confidence_threshold: Confidence threshold for detection/segmentation
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections
        top_k: Top-K predictions for classification
        db: Database session

    Returns:
        Inference result with predictions
    """
    # Get training job
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == training_job_id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {training_job_id} not found")

    # Lazy import to avoid torch dependency at startup
    from training.adapters.base import TaskType

    try:
        # Create adapter
        runner = InferenceRunner(db)
        adapter = runner._create_adapter(job, job.dataset_path)

        if not adapter:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported framework: {job.framework}"
            )

        # Load checkpoint
        adapter.load_checkpoint(
            checkpoint_path=checkpoint_path,
            inference_mode=True
        )

        # Run inference on single image
        results = adapter.infer_batch([image_path])

        if not results or len(results) == 0:
            raise HTTPException(status_code=500, detail="Inference failed")

        result = results[0]

        # Convert result to dict
        result_dict = {
            "image_path": result.image_path,
            "image_name": result.image_name,
            "task_type": result.task_type.value if hasattr(result.task_type, 'value') else str(result.task_type),
            "inference_time_ms": result.inference_time_ms,
            "preprocessing_time_ms": result.preprocessing_time_ms,
            "postprocessing_time_ms": result.postprocessing_time_ms,
        }

        # Add task-specific results
        if result.task_type == TaskType.IMAGE_CLASSIFICATION:
            result_dict["predicted_label"] = result.predicted_label
            result_dict["predicted_label_id"] = result.predicted_label_id
            result_dict["confidence"] = result.confidence
            result_dict["top5_predictions"] = result.top5_predictions or []

        elif result.task_type == TaskType.OBJECT_DETECTION:
            result_dict["predicted_boxes"] = result.predicted_boxes or []
            result_dict["num_detections"] = len(result.predicted_boxes) if result.predicted_boxes else 0

        elif result.task_type in [TaskType.INSTANCE_SEGMENTATION, TaskType.SEMANTIC_SEGMENTATION]:
            result_dict["predicted_boxes"] = result.predicted_boxes or []
            result_dict["predicted_mask_path"] = result.predicted_mask_path
            result_dict["num_instances"] = len(result.predicted_boxes) if result.predicted_boxes else 0

        elif result.task_type == TaskType.POSE_ESTIMATION:
            result_dict["predicted_keypoints"] = result.predicted_keypoints or []
            result_dict["num_persons"] = len(result.predicted_keypoints) if result.predicted_keypoints else 0

        return result_dict

    except Exception as e:
        import traceback
        print(f"[ERROR] Quick inference failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
