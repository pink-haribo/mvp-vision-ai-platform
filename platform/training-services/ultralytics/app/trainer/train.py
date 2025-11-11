"""
Training Logic

Downloads dataset from S3, trains YOLO model, uploads checkpoints to S3,
and sends callbacks to Backend.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from ultralytics import YOLO

from app.config import settings
from app.storage.s3 import S3Client

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(settings.CALLBACK_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=settings.CALLBACK_RETRY_DELAY, max=10),
)
async def send_callback(url: str, data: Dict[str, Any]) -> None:
    """Send callback to Backend with retry logic."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        logger.info(f"Callback sent successfully: {data.get('status')}")


async def train_model(
    job_id: str,
    model_name: str,
    dataset_s3_uri: str,
    callback_url: str,
    config: Dict[str, Any],
) -> None:
    """
    Train a YOLO model.

    Flow:
    1. Download dataset from S3
    2. Train model with callbacks
    3. Upload checkpoint to S3
    4. Send completion callback
    """
    s3_client = S3Client(
        endpoint=settings.S3_ENDPOINT,
        access_key=settings.AWS_ACCESS_KEY_ID,
        secret_key=settings.AWS_SECRET_ACCESS_KEY,
        bucket=settings.BUCKET_NAME,
    )

    workspace = Path(settings.WORKSPACE_DIR) / job_id
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        # Send start callback
        await send_callback(
            callback_url,
            {
                "job_id": job_id,
                "status": "running",
                "progress": 0.0,
                "message": "Starting training...",
            },
        )

        # Parse S3 URI
        if not dataset_s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {dataset_s3_uri}")

        s3_path = dataset_s3_uri.replace(f"s3://{settings.BUCKET_NAME}/", "")

        # Download dataset from S3
        logger.info(f"Downloading dataset from {dataset_s3_uri}")
        dataset_dir = workspace / "dataset"
        dataset_dir.mkdir(exist_ok=True)

        await send_callback(
            callback_url,
            {
                "job_id": job_id,
                "status": "running",
                "progress": 0.05,
                "message": "Downloading dataset from S3...",
            },
        )

        # Download all files under s3_path to dataset_dir
        await s3_client.download_directory(s3_path, dataset_dir)

        logger.info(f"Dataset downloaded to {dataset_dir}")

        # Training config
        epochs = config.get("epochs", 50)
        batch_size = config.get("batch", settings.DEFAULT_BATCH_SIZE)
        image_size = config.get("imgsz", settings.DEFAULT_IMAGE_SIZE)
        device = config.get("device", "cpu")  # cpu or 0,1,2... for GPU

        # Find data.yaml
        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

        # Load YOLO model
        logger.info(f"Loading model: {model_name}")
        model = YOLO(f"{model_name}.pt")

        await send_callback(
            callback_url,
            {
                "job_id": job_id,
                "status": "running",
                "progress": 0.1,
                "message": f"Model loaded, starting training for {epochs} epochs...",
            },
        )

        # Setup project directory for outputs
        project_dir = workspace / "runs"
        project_dir.mkdir(exist_ok=True)

        # Train model
        logger.info(f"Training {model_name} for {epochs} epochs")

        # Run training in executor to not block event loop
        def train_sync():
            return model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                device=device,
                project=str(project_dir),
                name="train",
                exist_ok=True,
                verbose=True,
            )

        # Run in thread pool to not block async
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, train_sync)

        logger.info("Training completed")

        # Find best checkpoint
        train_dir = project_dir / "train"
        weights_dir = train_dir / "weights"
        best_checkpoint = weights_dir / "best.pt"

        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {best_checkpoint}")

        # Upload checkpoint to S3
        logger.info("Uploading checkpoint to S3...")
        checkpoint_s3_key = f"checkpoints/{job_id}/best.pt"
        await s3_client.upload_file(best_checkpoint, checkpoint_s3_key)
        checkpoint_s3_uri = s3_client.get_s3_uri(checkpoint_s3_key)

        logger.info(f"Checkpoint uploaded to {checkpoint_s3_uri}")

        # Get final metrics from results
        final_metrics = {}
        if hasattr(results, "results_dict"):
            final_metrics = results.results_dict

        # Send completion callback
        await send_callback(
            callback_url,
            {
                "job_id": job_id,
                "status": "completed",
                "progress": 1.0,
                "message": "Training completed successfully",
                "checkpoint_s3_uri": checkpoint_s3_uri,
                "metrics": final_metrics,
            },
        )

        logger.info(f"Training job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Training failed for job {job_id}")

        # Send error callback
        try:
            await send_callback(
                callback_url,
                {
                    "job_id": job_id,
                    "status": "failed",
                    "progress": 0.0,
                    "error": str(e),
                    "message": f"Training failed: {str(e)}",
                },
            )
        except Exception as callback_error:
            logger.error(f"Failed to send error callback: {callback_error}")

    finally:
        # Cleanup workspace (optional - comment out for debugging)
        # import shutil
        # if workspace.exists():
        #     shutil.rmtree(workspace)
        logger.info(f"Training job {job_id} finished")
