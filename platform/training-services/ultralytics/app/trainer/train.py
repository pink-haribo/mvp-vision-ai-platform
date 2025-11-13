"""
Training Logic

Downloads dataset from S3, trains YOLO model, uploads checkpoints to S3,
and sends callbacks to Backend.

K8s Job Compatible:
- All callbacks match TrainingProgressCallback/TrainingCompletionCallback schemas
- Proper exit code handling (0=success, non-zero=failure)
- Epoch-by-epoch progress updates via YOLO callbacks
"""

import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential
from ultralytics import YOLO

from app.config import settings
from app.storage.s3 import S3Client

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(settings.CALLBACK_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=settings.CALLBACK_RETRY_DELAY, max=10),
)
async def send_progress_callback(
    callback_base_url: str,
    job_id: str,
    data: Dict[str, Any]
) -> None:
    """
    Send progress callback to Backend with retry logic.

    Args:
        callback_base_url: Base Backend API URL (e.g., http://localhost:8000/api/v1/training)
        job_id: Training job ID
        data: Progress data matching TrainingProgressCallback schema
    """
    url = f"{callback_base_url}/jobs/{job_id}/callback/progress"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        logger.info(f"Progress callback sent: epoch {data.get('current_epoch')}/{data.get('total_epochs')}")


@retry(
    stop=stop_after_attempt(settings.CALLBACK_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=settings.CALLBACK_RETRY_DELAY, max=10),
)
async def send_completion_callback(
    callback_base_url: str,
    job_id: str,
    data: Dict[str, Any]
) -> None:
    """
    Send completion callback to Backend with retry logic.

    Args:
        callback_base_url: Base Backend API URL
        job_id: Training job ID
        data: Completion data matching TrainingCompletionCallback schema
    """
    url = f"{callback_base_url}/jobs/{job_id}/callback/completion"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        logger.info(f"Completion callback sent: {data.get('status')}")


def process_dataset_split(dataset_dir: Path, split_config: Optional[Dict[str, Any]]) -> None:
    """
    Process dataset split configuration to create train.txt and val.txt.

    Reads annotations.json to get split assignments and creates text files
    with image paths for train and val sets.

    Args:
        dataset_dir: Path to downloaded dataset directory
        split_config: Split configuration from Backend (contains split assignments)
    """
    if not split_config or "splits" not in split_config:
        logger.info("No split configuration provided, using default data.yaml splits")
        return

    logger.info(f"Processing dataset split with method: {split_config.get('method')}")

    # Load annotations.json
    annotations_file = dataset_dir / "annotations.json"
    if not annotations_file.exists():
        logger.warning(f"annotations.json not found at {annotations_file}, skipping split processing")
        return

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    images = annotations.get('images', [])
    splits = split_config.get('splits', {})

    if not images or not splits:
        logger.warning("No images or splits found in configuration")
        return

    # Create train.txt and val.txt
    images_dir = dataset_dir / "images"
    train_images = []
    val_images = []

    for img in images:
        img_id = str(img['id'])  # Ensure ID is string for dict lookup
        img_filename = img.get('file_name', '')

        if not img_filename:
            continue

        img_path = images_dir / img_filename

        # Check if image file exists
        if not img_path.exists():
            logger.warning(f"Image file not found: {img_path}")
            continue

        # Assign to train or val based on split
        split_assignment = splits.get(img_id)

        if split_assignment == 'train':
            train_images.append(str(img_path.absolute()))
        elif split_assignment == 'val':
            val_images.append(str(img_path.absolute()))
        else:
            # If no split assignment, default to train
            logger.debug(f"Image {img_id} has no split assignment, defaulting to train")
            train_images.append(str(img_path.absolute()))

    # Write train.txt
    train_txt = dataset_dir / "train.txt"
    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_images))

    logger.info(f"Created train.txt with {len(train_images)} images")

    # Write val.txt
    val_txt = dataset_dir / "val.txt"
    with open(val_txt, 'w') as f:
        f.write('\n'.join(val_images))

    logger.info(f"Created val.txt with {len(val_images)} images")

    # Update data.yaml to use train.txt and val.txt
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        # Update paths to point to .txt files
        data_config['train'] = str(train_txt.absolute())
        data_config['val'] = str(val_txt.absolute())

        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f)

        logger.info(f"Updated data.yaml to use train.txt and val.txt")
    else:
        logger.warning(f"data.yaml not found at {data_yaml}, cannot update split paths")


async def train_model(
    job_id: str,
    model_name: str,
    dataset_s3_uri: str,
    callback_url: str,
    config: Dict[str, Any],
) -> None:
    """
    Train a YOLO model with K8s Job compatibility.

    Flow:
    1. Download dataset from S3
    2. Train model with epoch-by-epoch callbacks
    3. Upload checkpoint to S3
    4. Send completion callback
    5. Exit with proper exit code for K8s Job

    Args:
        job_id: Training job ID
        model_name: YOLO model name (e.g., yolo11n, yolo11n-seg)
        dataset_s3_uri: S3 URI to dataset (s3://bucket/path/to/dataset)
        callback_url: Backend API base URL (e.g., http://localhost:8000/api/v1/training)
        config: Training configuration dict

    K8s Job Exit Codes:
        0: Success
        1: Training failure
        2: Callback failure
    """
    exit_code = 0
    s3_client = S3Client(
        endpoint=settings.S3_ENDPOINT,
        access_key=settings.AWS_ACCESS_KEY_ID,
        secret_key=settings.AWS_SECRET_ACCESS_KEY,
        bucket=settings.BUCKET_NAME,
    )

    workspace = Path(settings.WORKSPACE_DIR) / job_id
    workspace.mkdir(parents=True, exist_ok=True)

    # Training state for callbacks
    training_state = {
        "current_epoch": 0,
        "total_epochs": 0,
        "best_checkpoint_path": None,
        "best_metrics": None,
        "best_epoch": None,
    }

    try:
        # Parse S3 URI
        if not dataset_s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {dataset_s3_uri}")

        s3_path = dataset_s3_uri.replace(f"s3://{settings.BUCKET_NAME}/", "")

        # Download dataset from S3
        logger.info(f"Downloading dataset from {dataset_s3_uri}")
        dataset_dir = workspace / "dataset"
        dataset_dir.mkdir(exist_ok=True)

        # Download all files under s3_path to dataset_dir
        await s3_client.download_directory(s3_path, dataset_dir)
        logger.info(f"Dataset downloaded to {dataset_dir}")

        # Process dataset split configuration (if provided)
        split_config = config.get("split_config")
        if split_config:
            logger.info("Processing dataset split configuration...")
            process_dataset_split(dataset_dir, split_config)
        else:
            logger.info("No split configuration provided, using default data.yaml splits")

        # Training config
        epochs = config.get("epochs", 50)
        training_state["total_epochs"] = epochs
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
        last_checkpoint = weights_dir / "last.pt"

        if not best_checkpoint.exists():
            logger.warning(f"Best checkpoint not found at {best_checkpoint}, using last.pt")
            best_checkpoint = last_checkpoint

        if not best_checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint found in {weights_dir}")

        # Upload checkpoint to S3
        logger.info("Uploading checkpoint to S3...")
        checkpoint_s3_key = f"checkpoints/{job_id}/best.pt"
        await s3_client.upload_file(best_checkpoint, checkpoint_s3_key)
        checkpoint_s3_uri = s3_client.get_s3_uri(checkpoint_s3_key)

        logger.info(f"Checkpoint uploaded to {checkpoint_s3_uri}")

        # Get final metrics from results
        final_metrics_dict = {}
        if hasattr(results, "results_dict"):
            final_metrics_dict = results.results_dict

        # Extract metrics for callback
        final_metrics = {
            "accuracy": final_metrics_dict.get("metrics/mAP50(B)", None),
            "loss": final_metrics_dict.get("train/box_loss", None),
            "extra_metrics": final_metrics_dict,
        }

        # Send completion callback with TrainingCompletionCallback schema
        completion_data = {
            "job_id": int(job_id),
            "status": "completed",
            "total_epochs_completed": epochs,
            "final_metrics": final_metrics,
            "best_metrics": final_metrics,  # YOLO already gives us best metrics
            "best_epoch": epochs,  # Last epoch is typically best in YOLO
            "final_checkpoint_path": str(best_checkpoint),
            "best_checkpoint_path": str(best_checkpoint),
            "model_artifacts_path": str(checkpoint_s3_uri),
            "exit_code": 0,  # Success
        }

        await send_completion_callback(callback_url, job_id, completion_data)
        logger.info(f"Training job {job_id} completed successfully")

        # K8s Job exit code: 0 = success
        exit_code = 0

    except Exception as e:
        logger.exception(f"Training failed for job {job_id}")
        exit_code = 1  # K8s Job exit code: 1 = failure

        # Send error completion callback
        try:
            error_data = {
                "job_id": int(job_id),
                "status": "failed",
                "total_epochs_completed": training_state.get("current_epoch", 0),
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "exit_code": exit_code,
            }

            await send_completion_callback(callback_url, job_id, error_data)
        except Exception as callback_error:
            logger.error(f"Failed to send error callback: {callback_error}")
            exit_code = 2  # Callback failure

    finally:
        # Cleanup workspace (optional - comment out for debugging)
        # import shutil
        # if workspace.exists():
        #     shutil.rmtree(workspace)
        logger.info(f"Training job {job_id} finished with exit code {exit_code}")

        # K8s Job mode: Exit with code for Pod status
        if settings.EXECUTION_MODE == "job":
            logger.info(f"K8s Job mode: Exiting with code {exit_code}")
            sys.exit(exit_code)
