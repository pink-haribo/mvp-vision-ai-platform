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
import os
import subprocess
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
    session_id: Optional[str] = Query(default=None)
):
    """
    Upload an image for inference.

    Saves the image to a session-specific temporary location.

    Args:
        file: Image file to upload
        session_id: Optional session ID for grouping images. If not provided, a new one is generated.

    Returns:
        Dict with server_path and session_id
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Create session-specific temp directory
        from app.core.config import settings
        session_dir = Path(settings.UPLOAD_DIR) / "inference_temp" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        file_extension = Path(file.filename or "image.jpg").suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = session_dir / unique_filename

        # Save file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "server_path": str(file_path.absolute()),
            "filename": file.filename,
            "session_id": session_id
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
    image_path: str,
    checkpoint_path: Optional[str] = None,
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

    Uses subprocess to run inference in the training venv (which has torch).

    Args:
        training_job_id: Training job ID
        image_path: Path to image file
        checkpoint_path: Optional path to checkpoint file. If None, uses pretrained weights.
        confidence_threshold: Confidence threshold for detection/segmentation
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections
        top_k: Top-K predictions for classification
        db: Database session

    Returns:
        Inference result with predictions
    """
    import subprocess
    import sys

    # Get training job
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == training_job_id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {training_job_id} not found")

    # Convert image_path to Path object for cleanup
    image_path_obj = Path(image_path)

    try:
        # Path to training venv python
        backend_dir = Path(__file__).parent.parent.parent
        project_root = backend_dir.parent
        training_venv_python = project_root / "training" / "venv" / "Scripts" / "python.exe"
        inference_script = project_root / "training" / "run_quick_inference.py"

        if not training_venv_python.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Training venv not found at {training_venv_python}"
            )

        if not inference_script.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Inference script not found at {inference_script}"
            )

        # Build command
        cmd = [
            str(training_venv_python),
            str(inference_script),
            "--training_job_id", str(training_job_id),
            "--image_path", image_path,
            "--framework", job.framework,
            "--model_name", job.model_name,
            "--task_type", job.task_type,
            "--num_classes", str(job.num_classes or 0),
            "--dataset_path", job.dataset_path or "",
            "--output_dir", job.output_dir or "",
            "--confidence_threshold", str(confidence_threshold),
            "--iou_threshold", str(iou_threshold),
            "--max_detections", str(max_detections),
            "--top_k", str(top_k)
        ]

        # Add checkpoint_path only if provided (otherwise use pretrained)
        if checkpoint_path:
            # Convert Docker container path to host path if needed
            # Docker uses /workspace/output, but we need the actual host path
            if checkpoint_path.startswith('/workspace/output/'):
                # Replace container path with host path
                checkpoint_filename = checkpoint_path.replace('/workspace/output/', '')
                host_checkpoint_path = os.path.join(job.output_dir, checkpoint_filename)
                print(f"[INFO] Converted checkpoint path: {checkpoint_path} -> {host_checkpoint_path}")
                cmd.extend(["--checkpoint_path", host_checkpoint_path])
            else:
                # Use path as-is (already host path)
                cmd.extend(["--checkpoint_path", checkpoint_path])
        else:
            # Use pretrained weights - script will detect missing checkpoint_path
            print(f"[INFO] Using pretrained weights for inference (no checkpoint provided)")
            cmd.append("--use_pretrained")

        # Run inference subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 120 second timeout (allow model download time)
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Inference subprocess failed"
            print(f"[ERROR] Inference subprocess failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {error_msg}"
            )

        # Parse JSON output (last line contains the JSON result)
        try:
            # Get last non-empty line (which should be the JSON output)
            output_lines = result.stdout.strip().split('\n')
            json_line = output_lines[-1]
            result_dict = json.loads(json_line)

            print(f"[DEBUG] Inference result keys: {result_dict.keys()}")
            print(f"[DEBUG] Task type: {result_dict.get('task_type')}")
            print(f"[DEBUG] Has upscaled_image_path: {result_dict.get('upscaled_image_path')}")

            # Convert upscaled_image_path to URL if present (for super-resolution)
            if result_dict.get('upscaled_image_path'):
                upscaled_path = Path(result_dict['upscaled_image_path'])
                filename = upscaled_path.name
                # Convert to API URL (without /api/v1 prefix - frontend will add it)
                result_dict['upscaled_image_url'] = f"/test_inference/result_image/{training_job_id}/{filename}"
                print(f"[DEBUG] Generated upscaled_image_url: {result_dict['upscaled_image_url']}")
            else:
                print(f"[DEBUG] No upscaled_image_path found in result")

            print(f"[DEBUG] Final result_dict keys: {result_dict.keys()}")
            return result_dict
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse inference output: {result.stdout}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse inference output: {str(e)}"
            )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Inference timeout (120s)"
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Quick inference failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
    finally:
        # No longer delete immediately - session-based cleanup handles this
        pass


@router.delete("/inference/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up all images in a session.

    Called when user leaves the inference panel or explicitly cleans up.

    Args:
        session_id: Session ID to clean up

    Returns:
        Status message
    """
    try:
        from app.core.config import settings
        session_dir = Path(settings.UPLOAD_DIR) / "inference_temp" / session_id

        if session_dir.exists() and session_dir.is_dir():
            # Safety check: ensure it's within inference_temp
            if "inference_temp" in str(session_dir):
                shutil.rmtree(session_dir)
                print(f"[INFO] Cleaned up session: {session_id}")
                return {"status": "cleaned", "session_id": session_id}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid session directory"
                )
        else:
            return {"status": "not_found", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to clean up session {session_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clean up session: {str(e)}"
        )


@router.get("/result_image/{training_job_id}/{filename}")
async def get_result_image(
    training_job_id: int,
    filename: str,
    db: Session = Depends(get_db)
):
    """
    Serve inference result images (e.g., upscaled images from super-resolution).

    Args:
        training_job_id: Training job ID
        filename: Image filename
        db: Database session

    Returns:
        FileResponse with the image
    """
    from fastapi.responses import FileResponse

    # Get training job
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == training_job_id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {training_job_id} not found")

    # Construct image path
    inference_results_dir = Path(job.output_dir) / "inference_results"
    image_path = inference_results_dir / filename

    # Security check: ensure the file is within inference_results directory
    try:
        image_path = image_path.resolve()
        if not str(image_path).startswith(str(inference_results_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid file path")

    # Check if file exists
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=filename
    )
