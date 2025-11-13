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
    2. Starts training in a background task
    3. Returns immediately with "started" status
    4. Training task will send callbacks to Backend with progress updates
    """
    logger.info(f"Received training request for job {request.job_id}")

    # Add training task to background
    background_tasks.add_task(
        train_model,
        job_id=request.job_id,
        config=request.config,
        dataset_s3_uri=request.dataset_s3_uri,
        callback_url=request.callback_url,
    )

    return TrainingStartResponse(
        job_id=request.job_id,
        message=f"Training job {request.job_id} started in background",
    )
