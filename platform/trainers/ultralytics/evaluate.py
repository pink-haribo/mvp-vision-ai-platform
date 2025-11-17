#!/usr/bin/env python3
"""
Ultralytics YOLO Evaluator

Simple CLI script for evaluating trained YOLO models on test datasets with S3 integration and Backend callbacks.

Usage:
    python evaluate.py \
        --test-run-id 123 \
        --training-job-id 456 \
        --checkpoint-s3-uri s3://training-checkpoints/checkpoints/456/best.pt \
        --dataset-s3-uri s3://training-datasets/datasets/test-abc-123/ \
        --callback-url http://localhost:8000/api/v1/test_inference \
        --config '{"conf_threshold": 0.25, "iou_threshold": 0.45}'

Environment Variables (alternative to CLI args):
    TEST_RUN_ID, TRAINING_JOB_ID, CHECKPOINT_S3_URI, DATASET_S3_URI, CALLBACK_URL, CONFIG

Exit Codes:
    0 = Success
    1 = Evaluation failure
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
    parser = argparse.ArgumentParser(description='Ultralytics YOLO Evaluator')

    parser.add_argument('--test-run-id', type=str, help='Test run ID')
    parser.add_argument('--training-job-id', type=str, help='Original training job ID (for checkpoint reference)')
    parser.add_argument('--checkpoint-s3-uri', type=str, help='S3 URI to trained checkpoint')
    parser.add_argument('--dataset-s3-uri', type=str, help='S3 URI to test dataset')
    parser.add_argument('--callback-url', type=str, help='Backend API base URL')
    parser.add_argument('--config', type=str, help='Evaluation config JSON string')
    parser.add_argument('--config-file', type=str, help='Path to evaluation config JSON file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from environment or args (K8s Job compatible)"""
    # Priority: env vars > CLI args (K8s Job style)
    test_run_id = os.getenv('TEST_RUN_ID') or args.test_run_id
    training_job_id = os.getenv('TRAINING_JOB_ID') or args.training_job_id
    checkpoint_s3_uri = os.getenv('CHECKPOINT_S3_URI') or args.checkpoint_s3_uri
    dataset_s3_uri = os.getenv('DATASET_S3_URI') or args.dataset_s3_uri
    callback_url = os.getenv('CALLBACK_URL') or args.callback_url

    # Config priority: env var > config file > CLI arg
    if os.getenv('CONFIG'):
        config = json.loads(os.getenv('CONFIG'))
    elif args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    elif args.config:
        config = json.loads(args.config)
    else:
        config = {}

    # Validate required fields
    if not all([test_run_id, checkpoint_s3_uri, dataset_s3_uri, callback_url]):
        raise ValueError("Missing required arguments: test_run_id, checkpoint_s3_uri, dataset_s3_uri, callback_url")

    return {
        'test_run_id': test_run_id,
        'training_job_id': training_job_id,  # Optional
        'checkpoint_s3_uri': checkpoint_s3_uri,
        'dataset_s3_uri': dataset_s3_uri,
        'callback_url': callback_url,
        'config': config
    }


async def evaluate_model(
    test_run_id: str,
    training_job_id: Optional[str],
    checkpoint_s3_uri: str,
    dataset_s3_uri: str,
    callback_url: str,
    config: Dict[str, Any]
) -> int:
    """
    Main evaluation function

    Returns:
        Exit code (0 = success, 1 = evaluation failure, 2 = callback failure)
    """
    try:
        logger.info("=" * 80)
        logger.info(f"Ultralytics Evaluation Service - Test Run {test_run_id}")
        logger.info("=" * 80)
        logger.info(f"Checkpoint: {checkpoint_s3_uri}")
        logger.info(f"Test Dataset: {dataset_s3_uri}")
        logger.info(f"Callback URL: {callback_url}")
        logger.info(f"Evaluation config: {json.dumps(config, indent=2)}")
        logger.info("=" * 80)

        # Initialize clients
        storage = DualStorageClient()  # Automatically handles External/Internal storage routing
        callback_client = CallbackClient(callback_url)

        # ========================================================================
        # Step 1: Download checkpoint from Internal Storage (MinIO-Results)
        # ========================================================================
        logger.info(f"Downloading checkpoint from {checkpoint_s3_uri}")

        # Extract checkpoint path from S3 URI
        # s3://training-checkpoints/checkpoints/456/best.pt -> checkpoints/456/best.pt
        checkpoint_key = checkpoint_s3_uri.replace(f"s3://{storage.internal_client.bucket}/", "")
        checkpoint_path = Path(f"/tmp/evaluation/{test_run_id}/checkpoint/best.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Download from Internal Storage
        storage.internal_client.client.download_file(
            storage.internal_client.bucket,
            checkpoint_key,
            str(checkpoint_path)
        )
        logger.info(f"Checkpoint downloaded to {checkpoint_path}")

        # ========================================================================
        # Step 2: Download test dataset from External Storage (MinIO-Datasets)
        # ========================================================================
        # Extract dataset ID from S3 URI
        # s3://bucket/datasets/abc-123/ -> abc-123
        dataset_id = dataset_s3_uri.rstrip('/').split('/')[-1]

        # Download dataset (automatically uses External Storage)
        dataset_dir = Path(f"/tmp/evaluation/{test_run_id}/dataset")
        logger.info(f"Downloading test dataset from {dataset_s3_uri}")
        storage.download_dataset(dataset_id, dataset_dir)
        logger.info(f"Test dataset downloaded to {dataset_dir}")

        # ========================================================================
        # Step 3: Convert DICEFormat to YOLO if needed
        # ========================================================================
        split_config = config.get('split_config')
        convert_diceformat_to_yolo(dataset_dir, split_config)

        # ========================================================================
        # Step 4: Extract evaluation parameters
        # ========================================================================
        conf_threshold = config.get('conf_threshold', 0.25)
        iou_threshold = config.get('iou_threshold', 0.45)
        image_size = config.get('imgsz', 640)
        device = config.get('device', 'cpu')
        split = config.get('split', 'val')  # val or test

        logger.info(f"Evaluation parameters: conf={conf_threshold}, iou={iou_threshold}, imgsz={image_size}")

        # ========================================================================
        # Step 5: Load model from checkpoint
        # ========================================================================
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = YOLO(str(checkpoint_path))

        # ========================================================================
        # Step 6: Run validation
        # ========================================================================
        logger.info(f"Running validation on {split} split...")
        data_yaml = dataset_dir / "data.yaml"
        project_dir = Path(f"/tmp/evaluation/{test_run_id}/results")

        # Run validation
        val_results = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=image_size,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            project=str(project_dir),
            name='val',
            exist_ok=True,
            save_json=True,  # Save predictions in COCO format
            save_hybrid=True,  # Save hybrid labels
            plots=True,  # Generate plots
        )

        logger.info("Validation completed")

        # ========================================================================
        # Step 7: Extract metrics
        # ========================================================================
        metrics = {}
        per_class_metrics = []

        # Overall metrics
        if hasattr(val_results, 'box'):
            metrics = {
                'mAP50': float(val_results.box.map50),
                'mAP50-95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
            }
            logger.info(f"Overall metrics: {metrics}")

            # Per-class metrics
            if hasattr(val_results.box, 'maps') and val_results.box.maps is not None:
                # maps is per-class mAP50-95
                import yaml
                try:
                    with open(data_yaml, 'r') as f:
                        data_config = yaml.safe_load(f)
                        class_names = data_config.get('names', [])

                    for i, class_map in enumerate(val_results.box.maps):
                        if i < len(class_names):
                            per_class_metrics.append({
                                'class_id': i,
                                'class_name': class_names[i],
                                'mAP50-95': float(class_map),
                                'mAP50': float(val_results.box.ap50[i]) if hasattr(val_results.box, 'ap50') else None,
                            })
                except Exception as e:
                    logger.warning(f"Failed to extract per-class metrics: {e}")

        elif hasattr(val_results, 'seg'):
            # Segmentation metrics
            metrics = {
                'mask_mAP50': float(val_results.seg.map50),
                'mask_mAP50-95': float(val_results.seg.map),
                'mask_precision': float(val_results.seg.mp),
                'mask_recall': float(val_results.seg.mr),
                'box_mAP50': float(val_results.box.map50),
                'box_mAP50-95': float(val_results.box.map),
            }
            logger.info(f"Segmentation metrics: {metrics}")

        elif hasattr(val_results, 'pose'):
            # Pose estimation metrics
            metrics = {
                'pose_mAP50': float(val_results.pose.map50),
                'pose_mAP50-95': float(val_results.pose.map),
            }
            logger.info(f"Pose metrics: {metrics}")

        # ========================================================================
        # Step 8: Upload validation plots to Internal Storage
        # ========================================================================
        logger.info("Processing validation plots...")

        plots_dir = project_dir / "val"
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
                    # Upload to MinIO Internal Storage: s3://training-checkpoints/test-runs/{test_run_id}/plots/{plot_file}
                    s3_key = f"test-runs/{test_run_id}/plots/{plot_file}"
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

        # Upload predictions JSON if available
        predictions_json = plots_dir / "predictions.json"
        predictions_uri = None
        if predictions_json.exists():
            try:
                s3_key = f"test-runs/{test_run_id}/predictions.json"
                storage.internal_client.client.upload_file(
                    str(predictions_json),
                    storage.internal_client.bucket,
                    s3_key,
                    ExtraArgs={'ContentType': 'application/json'}
                )
                predictions_uri = f"s3://{storage.internal_client.bucket}/{s3_key}"
                logger.info(f"Uploaded predictions.json → {predictions_uri}")
            except Exception as e:
                logger.warning(f"Failed to upload predictions.json: {e}")

        # ========================================================================
        # Step 9: Extract class names and task type
        # ========================================================================
        import yaml
        class_names = None
        try:
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
                class_names = data_config.get('names', [])
        except Exception as e:
            logger.warning(f"Failed to extract class names: {e}")

        # Determine task type from model
        task_type = 'detection'  # Default
        if hasattr(val_results, 'seg'):
            task_type = 'segmentation'
        elif hasattr(val_results, 'pose'):
            task_type = 'pose'
        elif hasattr(val_results, 'names') and len(val_results.names) > 0:
            # Check if this is classification (no bbox)
            if not hasattr(val_results, 'box'):
                task_type = 'classification'

        # ========================================================================
        # Step 10: Send test completion callback
        # ========================================================================
        completion_data = {
            'test_run_id': int(test_run_id),
            'training_job_id': int(training_job_id) if training_job_id else None,
            'status': 'completed',
            'task_type': task_type,

            # Overall metrics
            'metrics': metrics,

            # Per-class metrics
            'per_class_metrics': per_class_metrics if per_class_metrics else None,

            # Metadata
            'class_names': class_names,
            'num_images': val_results.seen if hasattr(val_results, 'seen') else None,

            # Visualization
            'visualization_urls': visualization_urls if visualization_urls else None,
            'predictions_json_uri': predictions_uri,

            # Config
            'config': {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'image_size': image_size,
                'split': split,
            },

            'exit_code': 0,
        }

        try:
            await callback_client.send_test_completion(test_run_id, completion_data)
            logger.info("✓ Evaluation completed successfully")
            return 0  # Success

        except Exception as e:
            logger.error(f"Failed to send test completion callback: {e}")
            logger.error(traceback.format_exc())
            return 2  # Callback failure

    except Exception as e:
        logger.error(f"Evaluation failed for test run {test_run_id}")
        logger.error(traceback.format_exc())

        # Try to send error callback
        try:
            error_data = {
                'test_run_id': int(test_run_id),
                'training_job_id': int(training_job_id) if training_job_id else None,
                'status': 'failed',
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'exit_code': 1,
            }
            await callback_client.send_test_completion(test_run_id, error_data)
        except Exception as cb_error:
            logger.error(f"Failed to send error callback: {cb_error}")

        return 1  # Evaluation failure


def main():
    """Main entry point"""
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        # Load configuration
        cfg = load_config(args)

        # Run evaluation
        exit_code = asyncio.run(evaluate_model(
            test_run_id=cfg['test_run_id'],
            training_job_id=cfg['training_job_id'],
            checkpoint_s3_uri=cfg['checkpoint_s3_uri'],
            dataset_s3_uri=cfg['dataset_s3_uri'],
            callback_url=cfg['callback_url'],
            config=cfg['config']
        ))

        logger.info(f"Evaluation test run {cfg['test_run_id']} finished with exit code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
