"""
Training API Endpoints

Receives training requests from Backend and executes them in background.
"""

import logging
from fastapi import APIRouter, BackgroundTasks

from app.schemas import TrainingRequest, TrainingStartResponse
from app.trainer.train import train_model

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a training job in the background.

    This endpoint:
    1. Validates the request
    2. Extracts model_name from config
    3. Starts training in a background task
    4. Returns immediately with "started" status
    5. Training task will send callbacks to Backend with progress updates

    K8s Job Compatible: Works in both Service and Job modes.
    """
    logger.info(f"Received training request for job {request.job_id}")

    # Extract model_name from config
    model_name = request.config.get("model", "yolo11n")
    logger.info(f"Model: {model_name}, Dataset: {request.dataset_s3_uri}")

    # Add training task to background (in Service mode) or runs directly (in Job mode)
    background_tasks.add_task(
        train_model,
        job_id=request.job_id,
        model_name=model_name,
        dataset_s3_uri=request.dataset_s3_uri,
        callback_url=request.callback_url,
        config=request.config,
    )

    return TrainingStartResponse(
        job_id=request.job_id,
        message=f"Training job {request.job_id} started in background",
    )
