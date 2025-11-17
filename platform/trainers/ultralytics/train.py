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

        # Extract advanced config parameters (from config_schema.py)
        # These will be passed directly to model.train() if present
        advanced_params = {}

        # Optimizer parameters
        if 'optimizer' in config:
            advanced_params['optimizer'] = config['optimizer']
        if 'weight_decay' in config:
            advanced_params['weight_decay'] = config['weight_decay']
        if 'momentum' in config:
            advanced_params['momentum'] = config['momentum']

        # Scheduler parameters
        if 'cos_lr' in config:
            advanced_params['cos_lr'] = config['cos_lr']
        if 'lrf' in config:
            advanced_params['lrf'] = config['lrf']
        if 'warmup_epochs' in config:
            advanced_params['warmup_epochs'] = config['warmup_epochs']
        if 'warmup_momentum' in config:
            advanced_params['warmup_momentum'] = config['warmup_momentum']
        if 'warmup_bias_lr' in config:
            advanced_params['warmup_bias_lr'] = config['warmup_bias_lr']

        # Augmentation parameters
        if 'mosaic' in config:
            advanced_params['mosaic'] = config['mosaic']
        if 'mixup' in config:
            advanced_params['mixup'] = config['mixup']
        if 'copy_paste' in config:
            advanced_params['copy_paste'] = config['copy_paste']
        if 'degrees' in config:
            advanced_params['degrees'] = config['degrees']
        if 'translate' in config:
            advanced_params['translate'] = config['translate']
        if 'scale' in config:
            advanced_params['scale'] = config['scale']
        if 'shear' in config:
            advanced_params['shear'] = config['shear']
        if 'perspective' in config:
            advanced_params['perspective'] = config['perspective']
        if 'flipud' in config:
            advanced_params['flipud'] = config['flipud']
        if 'fliplr' in config:
            advanced_params['fliplr'] = config['fliplr']
        if 'hsv_h' in config:
            advanced_params['hsv_h'] = config['hsv_h']
        if 'hsv_s' in config:
            advanced_params['hsv_s'] = config['hsv_s']
        if 'hsv_v' in config:
            advanced_params['hsv_v'] = config['hsv_v']

        # Optimization parameters
        if 'amp' in config:
            advanced_params['amp'] = config['amp']
        if 'close_mosaic' in config:
            advanced_params['close_mosaic'] = config['close_mosaic']

        # Validation parameters
        if 'val' in config:
            advanced_params['val'] = config['val']

        logger.info(f"Advanced config parameters: {list(advanced_params.keys())}")

        # Setup MLflow
        setup_mlflow()
        mlflow.set_experiment("vision-training")

        with mlflow.start_run(run_name=f"job-{job_id}") as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")

            # Log parameters (including advanced params)
            mlflow_params = {
                'job_id': job_id,
                'model_name': model_name,
                'framework': 'ultralytics',
                'task_type': config.get('task', 'detection'),
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'device': device,
            }
            # Add advanced params to MLflow logging
            mlflow_params.update(advanced_params)
            mlflow.log_params(mlflow_params)
            logger.info("Logged parameters to MLflow")

            # Load model
            logger.info(f"Loading model: {model_name}")
            model = YOLO(f"{model_name}.pt")

            # Epoch callback for progress updates
            def on_fit_epoch_end(trainer):
                """Called at end of each fit epoch (after validation)"""
                try:
                    nonlocal training_state

                    epoch = trainer.epoch + 1
                    training_state['current_epoch'] = epoch

                    # Extract metrics
                    metrics = {}
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        # DEBUG: Log raw trainer.metrics before processing
                        logger.info(f"[DEBUG] Epoch {epoch} - Raw trainer.metrics: {trainer.metrics}")
                        logger.info(f"[DEBUG] Epoch {epoch} - trainer.metrics type: {type(trainer.metrics)}")

                        metrics = {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in trainer.metrics.items()}

                        # DEBUG: Log processed metrics
                        logger.info(f"[DEBUG] Epoch {epoch} - Processed metrics: {metrics}")

                    # Log metrics to MLflow (with sanitized names and epoch step)
                    if metrics:
                        sanitized_metrics = {sanitize_metric_name(k): v for k, v in metrics.items()}

                        # DEBUG: Log sanitized metrics before MLflow
                        logger.info(f"[DEBUG] Epoch {epoch} - Sanitized metrics: {sanitized_metrics}")

                        mlflow.log_metrics(sanitized_metrics, step=epoch)
                        logger.debug(f"Logged {len(sanitized_metrics)} metrics to MLflow for epoch {epoch}")
                    else:
                        logger.warning(f"[DEBUG] Epoch {epoch} - No metrics to log!")

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

            # Add callback (after validation, so metrics include validation results)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

            # Run training
            logger.info(f"Training {model_name} for {epochs} epochs")

            # Allow local dataset override for testing (DEBUG)
            local_dataset_override = os.getenv('LOCAL_DATASET_PATH')
            if local_dataset_override:
                dataset_dir = Path(local_dataset_override)
                logger.warning(f"[DEBUG] Using LOCAL_DATASET_PATH override: {dataset_dir}")

            data_yaml = dataset_dir / "data.yaml"
            project_dir = Path(f"/tmp/training/{job_id}/runs")

            # Build training arguments
            train_args = {
                'data': str(data_yaml),
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': image_size,
                'device': device,
                'project': str(project_dir),
                'name': 'train',
                'exist_ok': True,
                'verbose': True,
            }

            # Add advanced config parameters
            train_args.update(advanced_params)

            logger.info(f"Training with {len(train_args)} parameters")
            results = model.train(**train_args)

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

            # ========================================================================
            # Process validation results and send validation callback
            # ========================================================================
            try:
                logger.info("Processing validation results...")

                # Find validation plots
                plots_dir = project_dir / "train"
                visualization_urls = {}

                plot_files = {
                    'confusion_matrix': 'confusion_matrix.png',
                    'confusion_matrix_normalized': 'confusion_matrix_normalized.png',
                    'f1_curve': 'F1_curve.png',
                    'pr_curve': 'PR_curve.png',
                    'p_curve': 'P_curve.png',
                    'r_curve': 'R_curve.png',
                }

                # Upload validation plots to MinIO Internal Storage
                for plot_name, plot_file in plot_files.items():
                    plot_path = plots_dir / plot_file
                    if plot_path.exists():
                        try:
                            # Upload to MinIO Internal Storage: s3://training-checkpoints/job{id}/validation/{plot_file}
                            s3_key = f"job{job_id}/validation/{plot_file}"
                            storage.internal_client.client.upload_file(
                                str(plot_path),
                                storage.internal_client.bucket,
                                s3_key,
                                ExtraArgs={'ContentType': 'image/png'}
                            )
                            plot_uri = f"s3://{storage.internal_client.bucket}/{s3_key}"
                            visualization_urls[plot_name] = plot_uri
                            logger.info(f"Uploaded {plot_file} → {plot_uri}")
                        except Exception as e:
                            logger.warning(f"Failed to upload {plot_file}: {e}")

                # Extract class names from data.yaml
                import yaml
                class_names = None
                try:
                    with open(data_yaml, 'r') as f:
                        data_config = yaml.safe_load(f)
                        class_names = data_config.get('names', [])
                except Exception as e:
                    logger.warning(f"Failed to extract class names: {e}")

                # Determine task type from model name
                task_type = 'detection'  # Default
                if 'seg' in model_name.lower():
                    task_type = 'segmentation'
                elif 'pose' in model_name.lower():
                    task_type = 'pose'
                elif 'cls' in model_name.lower() or 'classify' in model_name.lower():
                    task_type = 'classification'

                # Prepare validation callback data
                validation_data = {
                    'job_id': int(job_id),
                    'epoch': epochs,  # Final epoch
                    'task_type': task_type,
                    'primary_metric_name': 'mAP50-95',
                    'primary_metric_value': final_metrics.get('mAP50-95') if final_metrics else None,
                    'overall_loss': None,  # Ultralytics doesn't provide val loss separately
                    'metrics': final_metrics,
                    'per_class_metrics': None,  # TODO: Extract per-class metrics if needed
                    'confusion_matrix': None,  # Could extract from confusion_matrix.png if needed
                    'pr_curves': None,
                    'class_names': class_names,
                    'visualization_urls': visualization_urls if visualization_urls else None,
                }

                # Send validation callback (synchronous)
                try:
                    callback_client.send_validation_sync(job_id, validation_data)
                    logger.info(f"✓ Validation callback sent for epoch {epochs}")
                except Exception as e:
                    logger.warning(f"Failed to send validation callback: {e}")
                    # Don't fail the job if validation callback fails

            except Exception as e:
                logger.warning(f"Failed to process validation results: {e}")
                logger.warning(traceback.format_exc())
                # Continue to completion callback even if validation processing fails

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
