#!/usr/bin/env python3
"""
Ultralytics YOLO Trainer

Simple CLI script for training YOLO models with S3 integration and Backend callbacks.

Usage:
    python train.py \\
        --job-id 123 \\
        --model-name yolo11n \\
        --dataset-s3-uri s3://bucket/datasets/abc-123/ \\
        --callback-url http://localhost:8000/api/v1/training \\
        --config '{"epochs": 50, "batch": 16, "imgsz": 640}'

Environment Variables (alternative to CLI args):
    JOB_ID, MODEL_NAME, DATASET_S3_URI, CALLBACK_URL, CONFIG

Exit Codes:
    0 = Success
    1 = Training failure
    2 = Callback failure
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
from ultralytics import YOLO

from utils import DualStorageClient, CallbackClient, convert_diceformat_to_yolo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Ultralytics YOLO Trainer')

    parser.add_argument('--job-id', type=str, help='Training job ID')
    parser.add_argument('--model-name', type=str, help='YOLO model name (e.g., yolo11n)')
    parser.add_argument('--dataset-s3-uri', type=str, help='S3 URI to dataset')
    parser.add_argument('--callback-url', type=str, help='Backend API base URL')
    parser.add_argument('--config', type=str, help='Training config JSON string')
    parser.add_argument('--config-file', type=str, help='Path to training config JSON file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from args or environment"""
    # Priority: CLI args > env vars
    job_id = args.job_id or os.getenv('JOB_ID')
    model_name = args.model_name or os.getenv('MODEL_NAME')
    dataset_s3_uri = args.dataset_s3_uri or os.getenv('DATASET_S3_URI')
    callback_url = args.callback_url or os.getenv('CALLBACK_URL')

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    elif args.config:
        config = json.loads(args.config)
    elif os.getenv('CONFIG'):
        config = json.loads(os.getenv('CONFIG'))
    else:
        config = {}

    # Validate required fields
    if not all([job_id, model_name, dataset_s3_uri, callback_url]):
        raise ValueError("Missing required arguments: job_id, model_name, dataset_s3_uri, callback_url")

    return {
        'job_id': job_id,
        'model_name': model_name,
        'dataset_s3_uri': dataset_s3_uri,
        'callback_url': callback_url,
        'config': config
    }


def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"MLflow tracking URI: {mlflow_uri}")


def sanitize_metric_name(name: str) -> str:
    """
    Sanitize metric name for MLflow.

    MLflow only allows: alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), slashes (/)

    Args:
        name: Original metric name

    Returns:
        Sanitized metric name
    """
    # Replace parentheses with underscores
    name = name.replace('(', '_').replace(')', '_')
    # Replace other special characters with underscores
    import re
    name = re.sub(r'[^a-zA-Z0-9_\-.\s/]', '_', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove trailing/leading underscores
    name = name.strip('_')
    return name


async def train_model(
    job_id: str,
    model_name: str,
    dataset_s3_uri: str,
    callback_url: str,
    config: Dict[str, Any]
) -> int:
    """
    Main training function

    Returns:
        Exit code (0 = success, 1 = training failure, 2 = callback failure)
    """
    # Training state tracking (needed in both try and except blocks)
    training_state = {
        'current_epoch': 0,
        'best_metric': 0.0
    }

    try:
        logger.info("=" * 80)
        logger.info(f"Ultralytics Training Service - Job {job_id}")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Dataset: {dataset_s3_uri}")
        logger.info(f"Callback URL: {callback_url}")
        logger.info(f"Training config: {json.dumps(config, indent=2)}")
        logger.info("=" * 80)

        # Initialize clients
        storage = DualStorageClient()  # Automatically handles External/Internal storage routing
        callback_client = CallbackClient(callback_url)

        # Extract dataset ID from S3 URI
        # s3://bucket/datasets/abc-123/ -> abc-123
        dataset_id = dataset_s3_uri.rstrip('/').split('/')[-1]

        # Download dataset (automatically uses External Storage - MinIO-Datasets)
        dataset_dir = Path(f"/tmp/training/{job_id}/dataset")
        logger.info(f"Downloading dataset from {dataset_s3_uri}")
        storage.download_dataset(dataset_id, dataset_dir)
        logger.info(f"Dataset downloaded to {dataset_dir}")

        # Convert DICEFormat to YOLO if needed
        split_config = config.get('split_config')
        convert_diceformat_to_yolo(dataset_dir, split_config)

        # Extract training parameters
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch', 16)
        image_size = config.get('imgsz', 640)
        device = config.get('device', 'cpu')
        primary_metric = config.get('primary_metric', 'mAP50-95')

        # Setup MLflow
        setup_mlflow()
        mlflow.set_experiment("vision-training")

        with mlflow.start_run(run_name=f"job-{job_id}") as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")

            # Log parameters
            mlflow.log_params({
                'job_id': job_id,
                'model_name': model_name,
                'framework': 'ultralytics',
                'task_type': config.get('task', 'detection'),
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'device': device,
            })
            logger.info("Logged parameters to MLflow")

            # Load model
            logger.info(f"Loading model: {model_name}")
            model = YOLO(f"{model_name}.pt")

            # Epoch callback for progress updates
            def on_train_epoch_end(trainer):
                """Called at end of each training epoch"""
                try:
                    nonlocal training_state

                    epoch = trainer.epoch + 1
                    training_state['current_epoch'] = epoch

                    # Extract metrics
                    metrics = {}
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        metrics = {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in trainer.metrics.items()}

                    # Send progress callback every N epochs
                    callback_interval = int(os.getenv('CALLBACK_INTERVAL', '1'))
                    if epoch % callback_interval == 0 or epoch == epochs:
                        progress_data = {
                            'job_id': int(job_id),
                            'status': 'running',
                            'current_epoch': epoch,
                            'total_epochs': epochs,
                            'progress_percent': (epoch / epochs) * 100,
                            'metrics': {
                                'extra_metrics': metrics
                            }
                        }

                        # Send callback using synchronous version (Ultralytics callback is synchronous)
                        try:
                            callback_client.send_progress_sync(job_id, progress_data)
                        except Exception as e:
                            logger.warning(f"Failed to send progress callback: {e}")

                except Exception as e:
                    logger.error(f"Error in epoch callback: {e}")

            # Add callback
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            # Run training
            logger.info(f"Training {model_name} for {epochs} epochs")
            data_yaml = dataset_dir / "data.yaml"
            project_dir = Path(f"/tmp/training/{job_id}/runs")

            results = model.train(
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

            logger.info("Training completed")

            # Upload checkpoint (automatically uses Internal Storage - MinIO-Results)
            best_pt = project_dir / "train" / "weights" / "best.pt"
            if best_pt.exists():
                logger.info("Uploading checkpoint to S3...")
                checkpoint_uri = storage.upload_checkpoint(best_pt, job_id, "best.pt")
            else:
                logger.warning("best.pt not found, skipping checkpoint upload")
                checkpoint_uri = None

            # Extract final metrics
            final_metrics = {}
            if hasattr(results, 'results_dict'):
                final_metrics = results.results_dict
            elif hasattr(results, 'box'):
                final_metrics = {
                    'mAP50': float(results.box.map50),
                    'mAP50-95': float(results.box.map),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr),
                }

            # Log final metrics to MLflow (with sanitized names)
            if final_metrics:
                sanitized_metrics = {sanitize_metric_name(k): v for k, v in final_metrics.items()}
                mlflow.log_metrics(sanitized_metrics)
                logger.info(f"Logged {len(sanitized_metrics)} metrics to MLflow")

            # Send completion callback (K8s Job compatible schema)
            completion_data = {
                'job_id': int(job_id),
                'status': 'completed',
                'total_epochs_completed': epochs,  # Required field
                'final_metrics': {
                    'extra_metrics': final_metrics
                } if final_metrics else None,
                'best_checkpoint_path': checkpoint_uri,  # Changed from checkpoint_path
                'mlflow_run_id': run.info.run_id,
                'exit_code': 0,  # K8s Job compatibility
            }

            try:
                await callback_client.send_completion(job_id, completion_data)
                logger.info("✓ Training completed successfully")
                return 0  # Success

            except Exception as e:
                logger.error(f"Failed to send completion callback: {e}")
                logger.error(traceback.format_exc())
                return 2  # Callback failure

    except Exception as e:
        logger.error(f"Training failed for job {job_id}")
        logger.error(traceback.format_exc())

        # Try to send error callback (K8s Job compatible schema)
        try:
            error_data = {
                'job_id': int(job_id),
                'status': 'failed',
                'total_epochs_completed': training_state.get('current_epoch', 0),  # Epochs completed before failure
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'exit_code': 1,  # K8s Job compatibility: non-zero = failure
            }
            await callback_client.send_completion(job_id, error_data)
        except Exception as cb_error:
            logger.error(f"Failed to send error callback: {cb_error}")

        return 1  # Training failure


def main():
    """Main entry point"""
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        # Load configuration
        cfg = load_config(args)

        # Run training
        exit_code = asyncio.run(train_model(
            job_id=cfg['job_id'],
            model_name=cfg['model_name'],
            dataset_s3_uri=cfg['dataset_s3_uri'],
            callback_url=cfg['callback_url'],
            config=cfg['config']
        ))

        logger.info(f"Training job {cfg['job_id']} finished with exit code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
