"""
Test and Inference API endpoints.

Provides endpoints for:
- Test: Running tests on labeled datasets with metrics
- Inference: Running predictions on unlabeled data

Uses TrainingManager abstraction for both subprocess (Tier 0) and Kubernetes (Tier 1+) modes.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
import json
import logging
import shutil
import uuid
import os
from pathlib import Path
from datetime import datetime

from app.db.database import get_db
from app.db import models
from app.schemas import test_inference as ti_schemas
from app.core.training_manager import get_training_manager

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/test_inference", tags=["test_inference"])


# Background task functions
def run_test_task(test_run_id: int):
    """
    Background task to run test/evaluation using TrainingManager.

    Uses TrainingManager.start_evaluation() which works for both:
    - subprocess mode (Tier 0): Runs evaluate.py directly
    - kubernetes mode (Tier 1+): Creates K8s Job for evaluation
    """
    # Run async code in sync context
    asyncio.run(_run_test_task_async(test_run_id))


async def _run_test_task_async(test_run_id: int):
    """Async implementation of test task."""
    from app.db.database import SessionLocal
    from app.core.config import settings

    db = SessionLocal()
    try:
        # Get test run
        test_run = db.query(models.TestRun).filter(
            models.TestRun.id == test_run_id
        ).first()

        if not test_run:
            logger.error(f"[TEST] Test run {test_run_id} not found")
            return

        # Get training job for framework info
        training_job = db.query(models.TrainingJob).filter(
            models.TrainingJob.id == test_run.training_job_id
        ).first()

        if not training_job:
            logger.error(f"[TEST] Training job {test_run.training_job_id} not found")
            test_run.status = "failed"
            test_run.error_message = "Training job not found"
            db.commit()
            return

        # Update status to running
        test_run.status = "running"
        test_run.started_at = datetime.utcnow()
        db.commit()

        logger.info(f"[TEST] Starting test run {test_run_id}")
        logger.info(f"[TEST] Framework: {training_job.framework}, Checkpoint: {test_run.checkpoint_path}")

        # Get TrainingManager (works for both subprocess and kubernetes)
        manager = get_training_manager()

        # Build evaluation config
        eval_config = {
            "task_type": test_run.task_type or training_job.task_type,
            "dataset_split": test_run.dataset_split or "test",
            "batch_size": 16,
            "device": "cpu",
        }

        # Build callback URL
        callback_url = f"{settings.API_BASE_URL}/api/v1"

        # Start evaluation via TrainingManager
        await manager.start_evaluation(
            test_run_id=test_run_id,
            training_job_id=test_run.training_job_id,
            framework=training_job.framework,
            checkpoint_s3_uri=test_run.checkpoint_path,
            dataset_s3_uri=test_run.dataset_path,
            callback_url=callback_url,
            config=eval_config
        )

        logger.info(f"[TEST] Evaluation started for test run {test_run_id}")
        # Status will be updated by callback from evaluate.py

    except Exception as e:
        logger.error(f"[TEST] Test run {test_run_id} error: {e}")
        import traceback
        traceback.print_exc()

        try:
            test_run.status = "failed"
            test_run.error_message = str(e)
            test_run.completed_at = datetime.utcnow()
            db.commit()
        except:
            pass
    finally:
        db.close()


def run_inference_task(inference_job_id: int):
    """
    Background task to run inference using TrainingManager.

    Uses TrainingManager.start_inference() which works for both:
    - subprocess mode (Tier 0): Runs predict.py directly
    - kubernetes mode (Tier 1+): Creates K8s Job for inference
    """
    # Run async code in sync context
    asyncio.run(_run_inference_task_async(inference_job_id))


async def _run_inference_task_async(inference_job_id: int):
    """Async implementation of inference task."""
    from app.db.database import SessionLocal
    from app.core.config import settings

    db = SessionLocal()
    try:
        # Get inference job
        inference_job = db.query(models.InferenceJob).filter(
            models.InferenceJob.id == inference_job_id
        ).first()

        if not inference_job:
            logger.error(f"[INFERENCE] Job {inference_job_id} not found")
            return

        # Get training job for framework info
        training_job = db.query(models.TrainingJob).filter(
            models.TrainingJob.id == inference_job.training_job_id
        ).first()

        if not training_job:
            logger.error(f"[INFERENCE] Training job {inference_job.training_job_id} not found")
            inference_job.status = "failed"
            inference_job.error_message = "Training job not found"
            db.commit()
            return

        # Update status to running
        inference_job.status = "running"
        inference_job.started_at = datetime.utcnow()
        db.commit()

        logger.info(f"[INFERENCE] Starting job {inference_job_id}")
        logger.info(f"[INFERENCE] Framework: {training_job.framework}, Model: {training_job.model_name}")

        # Get TrainingManager (works for both subprocess and kubernetes)
        manager = get_training_manager()

        # Parse input_data
        input_data = json.loads(inference_job.input_data) if isinstance(inference_job.input_data, str) else inference_job.input_data or {}

        # Build inference config
        inference_config = {
            "task_type": inference_job.task_type or training_job.task_type,
            "model_name": training_job.model_name,
            "conf": input_data.get('confidence_threshold', 0.25),
            "iou": input_data.get('iou_threshold', 0.45),
            "max_det": input_data.get('max_detections', 100),
            "imgsz": input_data.get('image_size', 640),
            "device": input_data.get('device', 'cpu'),
            "save_txt": input_data.get('save_txt', True),
            "save_conf": input_data.get('save_conf', True),
            "save_crop": input_data.get('save_crop', False),
        }

        # Build callback URL
        callback_url = f"{settings.API_BASE_URL}/api/v1"

        # Get images S3 URI from input_data
        images_s3_uri = input_data.get('image_paths_s3', '')

        # Start inference via TrainingManager
        await manager.start_inference(
            inference_job_id=inference_job_id,
            training_job_id=inference_job.training_job_id,
            framework=training_job.framework,
            checkpoint_s3_uri=inference_job.checkpoint_path,
            images_s3_uri=images_s3_uri,
            callback_url=callback_url,
            config=inference_config
        )

        logger.info(f"[INFERENCE] Inference started for job {inference_job_id}")
        # Status will be updated by callback from predict.py

    except Exception as e:
        logger.error(f"[INFERENCE] Job {inference_job_id} error: {e}")
        import traceback
        traceback.print_exc()

        try:
            inference_job.status = "failed"
            inference_job.error_message = str(e)
            inference_job.completed_at = datetime.utcnow()
            db.commit()
        except:
            pass
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

    # Validate checkpoint path (accept both S3 URI and local path)
    checkpoint_path_str = request.checkpoint_path

    # If it's an S3 URI, accept it (evaluate.py will download it)
    if not checkpoint_path_str.startswith('s3://'):
        # Local path - check existence
        checkpoint_path = Path(checkpoint_path_str)
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

    # Validate checkpoint path (accept S3 URI, local path, or 'pretrained')
    checkpoint_path_str = request.checkpoint_path

    # Special case: 'pretrained' means use pretrained weights (predict.py will auto-download)
    if checkpoint_path_str == 'pretrained':
        pass  # Accept 'pretrained' as-is, predict.py will handle it
    # If it's an S3 URI, accept it (predict.py will download it)
    elif checkpoint_path_str.startswith('s3://'):
        pass  # Accept S3 URIs
    else:
        # Local path - check existence
        checkpoint_path = Path(checkpoint_path_str)
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


@router.get("/inference/jobs/detail/{inference_job_id}", response_model=ti_schemas.InferenceJobResponse)
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


# ========== Callback Endpoints (for Training CLIs) ==========

@router.post("/test/{test_run_id}/results")
async def receive_test_results(
    test_run_id: int,
    request: ti_schemas.TestResultsCallback,
    db: Session = Depends(get_db)
):
    """
    Callback endpoint for test evaluation results from Training CLI (evaluate.py).

    This endpoint is called by the trainer CLI after running evaluation.

    Args:
        test_run_id: Test run ID
        request: Test results data from CLI
        db: Database session

    Returns:
        Success message
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

    logger.info(f"[TEST CALLBACK] Received results for test run {test_run_id}")

    # Update test run with results
    test_run.status = request.status
    test_run.task_type = request.task_type

    # Update metrics
    if request.metrics:
        test_run.metrics = request.metrics
        # Extract primary metric
        if 'mAP50-95' in request.metrics:
            test_run.primary_metric_name = 'mAP50-95'
            test_run.primary_metric_value = request.metrics['mAP50-95']
        elif 'mAP50' in request.metrics:
            test_run.primary_metric_name = 'mAP50'
            test_run.primary_metric_value = request.metrics['mAP50']

    # Update per-class metrics
    if request.per_class_metrics:
        test_run.per_class_metrics = request.per_class_metrics

    # Update metadata
    if request.class_names:
        test_run.class_names = request.class_names

    if request.num_images:
        test_run.total_images = request.num_images

    # Update visualization URLs
    if request.visualization_urls:
        test_run.visualization_urls = request.visualization_urls

    if request.predictions_json_uri:
        test_run.predictions_json_uri = request.predictions_json_uri

    # Update config
    if request.config:
        test_run.config = request.config

    # Handle error case
    if request.status == 'failed':
        test_run.error_message = request.error_message
        test_run.traceback = request.traceback

    test_run.completed_at = datetime.utcnow()

    db.commit()
    db.refresh(test_run)

    logger.info(f"[TEST CALLBACK] Updated test run {test_run_id} - status: {test_run.status}")

    return {"status": "success", "test_run_id": test_run_id}


@router.post("/inference/{inference_job_id}/results")
async def receive_inference_results(
    inference_job_id: int,
    request: ti_schemas.InferenceResultsCallback,
    db: Session = Depends(get_db)
):
    """
    Callback endpoint for inference results from predict.py (Training CLI).

    Follows the unified pattern: Train/Validate/Infer use same callback structure.
    Creates individual InferenceResult records for each image.

    Args:
        inference_job_id: Inference job ID
        request: Inference results data from predict.py
        db: Database session

    Returns:
        Success message
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

    logger.info(f"[INFERENCE CALLBACK] Received results for inference job {inference_job_id}")
    logger.info(f"[INFERENCE CALLBACK] Status: {request.status}, Total images: {request.total_images}")

    # Update inference job summary
    inference_job.status = request.status
    inference_job.completed_at = datetime.utcnow()

    # Update performance metrics
    if request.total_images is not None:
        inference_job.total_images = request.total_images

    if request.total_inference_time_ms is not None:
        inference_job.total_inference_time_ms = request.total_inference_time_ms

    if request.avg_inference_time_ms is not None:
        inference_job.avg_inference_time_ms = request.avg_inference_time_ms

    # Handle error case
    if request.status == 'failed':
        inference_job.error_message = request.error_message
        if request.traceback:
            logger.error(f"[INFERENCE CALLBACK] Error traceback:\n{request.traceback}")

    # Handle success case - create InferenceResult records
    elif request.status == 'completed' and request.results:
        logger.info(f"[INFERENCE CALLBACK] Creating {len(request.results)} InferenceResult records")

        for idx, image_result in enumerate(request.results):
            # Extract image name from path if not provided
            image_name = image_result.image_name
            if not image_name:
                image_name = Path(image_result.image_path).name

            # Get training job to determine task_type
            training_job = db.query(models.TrainingJob).filter(
                models.TrainingJob.id == inference_job.training_job_id
            ).first()

            # Create InferenceResult record
            result_record = models.InferenceResult(
                inference_job_id=inference_job_id,
                training_job_id=inference_job.training_job_id,
                image_path=image_result.image_path,
                image_name=image_name,
                image_index=idx,
                predictions=image_result.predictions,  # Store raw predictions JSON
                inference_time_ms=image_result.inference_time_ms,
                created_at=datetime.utcnow()
            )

            # Add task-specific fields based on predictions
            if training_job and training_job.task_type:
                if 'detection' in training_job.task_type or 'segment' in training_job.task_type:
                    # For detection/segmentation: extract predicted_boxes
                    result_record.predicted_boxes = [
                        {
                            'label': pred.get('class_name'),
                            'confidence': pred.get('confidence'),
                            'x1': pred['bbox'][0] if 'bbox' in pred else None,
                            'y1': pred['bbox'][1] if 'bbox' in pred else None,
                            'x2': pred['bbox'][2] if 'bbox' in pred else None,
                            'y2': pred['bbox'][3] if 'bbox' in pred else None,
                        }
                        for pred in image_result.predictions
                        if 'bbox' in pred
                    ]
                elif 'classification' in training_job.task_type:
                    # For classification: extract top predictions
                    if len(image_result.predictions) > 0:
                        top_pred = image_result.predictions[0]
                        result_record.predicted_label = top_pred.get('class_name')
                        result_record.predicted_label_id = top_pred.get('class_id')
                        result_record.confidence = top_pred.get('confidence')

                    result_record.top5_predictions = [
                        {
                            'label': pred.get('class_name'),
                            'confidence': pred.get('confidence'),
                            'class_id': pred.get('class_id')
                        }
                        for pred in image_result.predictions[:5]
                    ]

            db.add(result_record)

        logger.info(f"[INFERENCE CALLBACK] Created {len(request.results)} InferenceResult records")

    db.commit()
    db.refresh(inference_job)

    logger.info(f"[INFERENCE CALLBACK] Updated inference job {inference_job_id} - status: {inference_job.status}")

    return {"status": "success", "inference_job_id": inference_job_id}


@router.post("/inference/jobs/{inference_job_id}/callback/started")
async def inference_started_callback(
    inference_job_id: int,
    request: dict,
    db: Session = Depends(get_db)
):
    """
    Callback endpoint for inference job started notification from predict.py.

    Args:
        inference_job_id: Inference job ID
        request: Started notification data
        db: Database session

    Returns:
        Success message
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

    logger.info(f"[INFERENCE CALLBACK] Started notification for job {inference_job_id}")

    # Update status to running if still pending
    if inference_job.status == "pending":
        inference_job.status = "running"
        inference_job.started_at = datetime.utcnow()
        db.commit()

    return {"status": "success", "message": f"Inference job {inference_job_id} started"}


@router.post("/inference/jobs/{inference_job_id}/callback/completion")
async def inference_completion_callback(
    inference_job_id: int,
    request: ti_schemas.InferenceResultsCallback,
    db: Session = Depends(get_db)
):
    """
    Callback endpoint for inference job completion from predict.py.

    This endpoint follows the unified callback pattern used by the TrainerSDK.

    Args:
        inference_job_id: Inference job ID
        request: Completion data with results
        db: Database session

    Returns:
        Success message
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

    logger.info(f"[INFERENCE CALLBACK] Completion for job {inference_job_id}")
    logger.info(f"[INFERENCE CALLBACK] Status: {request.status}, Total images: {request.total_images}")

    # Update inference job summary
    inference_job.status = request.status
    inference_job.completed_at = datetime.utcnow()

    # Update performance metrics
    if request.total_images is not None:
        inference_job.total_images = request.total_images

    if request.total_inference_time_ms is not None:
        inference_job.total_inference_time_ms = request.total_inference_time_ms

    if request.avg_inference_time_ms is not None:
        inference_job.avg_inference_time_ms = request.avg_inference_time_ms

    # Handle error case
    if request.status == 'failed':
        inference_job.error_message = request.error_message
        if request.traceback:
            logger.error(f"[INFERENCE CALLBACK] Error traceback:\n{request.traceback}")

    # Handle success case - create InferenceResult records
    elif request.status == 'completed' and request.results:
        logger.info(f"[INFERENCE CALLBACK] Creating {len(request.results)} InferenceResult records")

        for idx, image_result in enumerate(request.results):
            # Extract image name from path if not provided
            image_name = image_result.image_name
            if not image_name:
                image_name = Path(image_result.image_path).name

            # Get training job to determine task_type
            training_job = db.query(models.TrainingJob).filter(
                models.TrainingJob.id == inference_job.training_job_id
            ).first()

            # Create InferenceResult record
            result_record = models.InferenceResult(
                inference_job_id=inference_job_id,
                training_job_id=inference_job.training_job_id,
                image_path=image_result.image_path,
                image_name=image_name,
                image_index=idx,
                predictions=image_result.predictions,
                inference_time_ms=image_result.inference_time_ms,
                created_at=datetime.utcnow()
            )

            # Add task-specific fields based on predictions
            if training_job and training_job.task_type:
                if 'detection' in training_job.task_type or 'segment' in training_job.task_type:
                    result_record.predicted_boxes = [
                        {
                            'label': pred.get('class_name'),
                            'confidence': pred.get('confidence'),
                            'x1': pred['bbox'][0] if 'bbox' in pred else None,
                            'y1': pred['bbox'][1] if 'bbox' in pred else None,
                            'x2': pred['bbox'][2] if 'bbox' in pred else None,
                            'y2': pred['bbox'][3] if 'bbox' in pred else None,
                        }
                        for pred in image_result.predictions
                        if 'bbox' in pred
                    ]
                elif 'classification' in training_job.task_type:
                    if len(image_result.predictions) > 0:
                        top_pred = image_result.predictions[0]
                        result_record.predicted_label = top_pred.get('class_name')
                        result_record.predicted_label_id = top_pred.get('class_id')
                        result_record.confidence = top_pred.get('confidence')

                    result_record.top5_predictions = [
                        {
                            'label': pred.get('class_name'),
                            'confidence': pred.get('confidence'),
                            'class_id': pred.get('class_id')
                        }
                        for pred in image_result.predictions[:5]
                    ]

            db.add(result_record)

        logger.info(f"[INFERENCE CALLBACK] Created {len(request.results)} InferenceResult records")

    db.commit()
    db.refresh(inference_job)

    logger.info(f"[INFERENCE CALLBACK] Updated inference job {inference_job_id} - status: {inference_job.status}")

    return {"status": "success", "inference_job_id": inference_job_id}


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

@router.post("/inference/upload-images")
async def upload_inference_images(
    training_job_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload images for inference to S3 Internal Storage.

    Follows unified pattern: Images stored in S3 training-checkpoints bucket
    at s3://training-checkpoints/inference/{job_id}/

    Args:
        training_job_id: Training job ID
        files: List of image files to upload

    Returns:
        Dict with S3 URIs for uploaded images
    """
    try:
        # Validate training job exists
        training_job = db.query(models.TrainingJob).filter(
            models.TrainingJob.id == training_job_id
        ).first()

        if not training_job:
            raise HTTPException(
                status_code=404,
                detail=f"Training job {training_job_id} not found"
            )

        # Validate all files are images
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not an image"
                )

        # Generate inference session ID
        inference_session_id = str(uuid.uuid4())
        s3_prefix = f"inference/{inference_session_id}"

        # Initialize S3 client (Internal Storage for inference images)
        from app.utils.dual_storage import DualStorageClient
        storage = DualStorageClient()
        s3_client = storage.internal_client
        bucket_name = 'inference-data'  # Dedicated bucket for inference uploads

        uploaded_files = []

        # Upload each file to S3
        for file in files:
            # Generate unique filename preserving extension
            file_extension = Path(file.filename or "image.jpg").suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            s3_key = f"{s3_prefix}/{unique_filename}"

            # Upload to S3
            file.file.seek(0)  # Reset file pointer
            s3_client.upload_fileobj(
                file.file,
                bucket_name,
                s3_key,
                ExtraArgs={'ContentType': file.content_type or 'image/jpeg'}
            )

            s3_uri = f"s3://{bucket_name}/{s3_key}"
            uploaded_files.append({
                'original_filename': file.filename,
                'unique_filename': unique_filename,
                's3_uri': s3_uri,
                's3_key': s3_key
            })

            logger.info(f"Uploaded {file.filename} → {s3_uri}")

        return {
            'status': 'success',
            'inference_session_id': inference_session_id,
            's3_prefix': f"s3://{bucket_name}/{s3_prefix}/",
            'uploaded_files': uploaded_files,
            'total_files': len(uploaded_files)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Image upload to S3 failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload images to S3: {str(e)}"
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
