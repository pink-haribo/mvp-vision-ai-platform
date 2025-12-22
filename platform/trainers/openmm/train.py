#!/usr/bin/env python3
"""
OpenMMLab Trainer Script

Supports:
- MMDetection (object detection)
- MMSegmentation (semantic/instance segmentation)
- MMPose (pose estimation)

Usage:
    python train.py

Environment Variables:
    CALLBACK_URL: Backend API URL for callbacks
    JOB_ID: Training job ID
    TASK_TYPE: detection, segmentation, pose
    MODEL_NAME: Model architecture (e.g., faster-rcnn, mask-rcnn, hrnet)
    DATASET_ID: Dataset ID to download
    SNAPSHOT_ID: Dataset snapshot ID
    DATASET_VERSION_HASH: Dataset version hash for caching
    CONFIG: JSON-encoded training configuration
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from trainer_sdk import TrainerSDK, ErrorType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_mmdet_config(model_name: str, data_root: str, num_classes: int, config: Dict[str, Any]):
    """
    Load MMDetection config for given model.

    Args:
        model_name: Model name (e.g., 'faster-rcnn', 'mask-rcnn', 'yolox')
        data_root: Dataset root directory
        num_classes: Number of classes
        config: Training configuration

    Returns:
        MMDetection config dict
    """
    from mmengine import Config

    # Map model names to MMDetection config files
    model_config_map = {
        'faster-rcnn': 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        'mask-rcnn': 'mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py',
        'retinanet': 'retinanet/retinanet_r50_fpn_1x_coco.py',
        'fcos': 'fcos/fcos_r50_fpn_1x_coco.py',
        'yolox': 'yolox/yolox_s_8xb8-300e_coco.py',
        'rtmdet': 'rtmdet/rtmdet_s_8xb32-300e_coco.py',
    }

    config_file = model_config_map.get(model_name)
    if not config_file:
        raise ValueError(f"Unsupported model: {model_name}")

    # Load base config from MMDetection
    cfg = Config.fromfile(f'configs/{config_file}')

    # Update dataset paths
    cfg.data_root = data_root
    cfg.train_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.data_root = data_root

    # Update number of classes
    cfg.model.roi_head.bbox_head.num_classes = num_classes
    if hasattr(cfg.model.roi_head, 'mask_head'):
        cfg.model.roi_head.mask_head.num_classes = num_classes

    # Update training parameters from config
    basic = config.get('basic', {})
    cfg.train_cfg.max_epochs = basic.get('epochs', 12)
    cfg.train_dataloader.batch_size = basic.get('batch', 2)
    cfg.optim_wrapper.optimizer.lr = basic.get('lr0', 0.02)

    return cfg


def main():
    """Main training function."""
    sdk = TrainerSDK()

    try:
        # Report training started
        config = sdk.get_full_config()
        basic_config = config['basic']
        total_epochs = basic_config.get('epochs', 12)

        sdk.report_started('training', total_epochs=total_epochs)
        logger.info(f"Training job {sdk.job_id} started")

        # Download dataset with caching
        logger.info("Downloading dataset...")
        dataset_dir = sdk.download_dataset_with_cache(
            snapshot_id=sdk.snapshot_id,
            dataset_id=sdk.dataset_id,
            dataset_version_hash=sdk.dataset_version_hash,
            dest_dir="/tmp/training"
        )
        logger.info(f"Dataset downloaded to: {dataset_dir}")

        # Convert dataset to COCO format (MMDetection uses COCO)
        logger.info("Converting dataset to COCO format...")
        dataset_dir = sdk.convert_dataset(
            dataset_dir=dataset_dir,
            source_format='dice',
            target_format='coco',
            task_type=sdk.task_type
        )

        # Parse annotation to get number of classes
        import json
        annotations_file = Path(dataset_dir) / "annotations_detection.json"
        with open(annotations_file) as f:
            data = json.load(f)

        categories = data.get('categories', [])
        # Filter out __background__ class
        real_categories = [cat for cat in categories if cat.get('name') != '__background__']
        num_classes = len(real_categories)
        class_names = [cat['name'] for cat in real_categories]

        logger.info(f"Dataset: {num_classes} classes - {class_names}")

        # Load MMDetection config
        logger.info(f"Loading {sdk.model_name} config...")
        cfg = load_mmdet_config(
            model_name=sdk.model_name,
            data_root=dataset_dir,
            num_classes=num_classes,
            config=config
        )

        # Import MMDetection
        from mmdet.apis import train_detector
        from mmdet.models import build_detector

        # Build model
        logger.info("Building model...")
        model = build_detector(cfg.model)
        model.init_weights()

        # Setup training hooks for progress reporting
        class SDKProgressHook:
            def __init__(self, sdk, total_epochs):
                self.sdk = sdk
                self.total_epochs = total_epochs

            def after_train_epoch(self, runner):
                epoch = runner.epoch + 1
                metrics = {
                    'loss': runner.message_hub.get_scalar('train/loss').current(),
                    'learning_rate': runner.optim_wrapper.get_lr()['lr'][0]
                }
                self.sdk.report_progress(epoch, self.total_epochs, metrics)

        # Add custom hook
        progress_hook = SDKProgressHook(sdk, total_epochs)
        cfg.custom_hooks = [
            dict(type='CustomHook', priority='NORMAL', hook=progress_hook)
        ]

        # Train model
        logger.info("Starting training...")
        train_detector(model, cfg.train_dataloader, cfg, validate=True)

        # Upload checkpoints
        logger.info("Uploading checkpoints...")
        work_dir = Path(cfg.work_dir)
        best_checkpoint = work_dir / "best_coco_bbox_mAP_epoch_*.pth"
        last_checkpoint = work_dir / f"epoch_{total_epochs}.pth"

        checkpoints = {}
        if best_checkpoint.exists():
            best_uri = sdk.upload_checkpoint(str(best_checkpoint), 'best')
            checkpoints['best'] = best_uri

        if last_checkpoint.exists():
            last_uri = sdk.upload_checkpoint(str(last_checkpoint), 'last')
            checkpoints['last'] = last_uri

        # Get final metrics from log
        final_metrics = {
            'loss': 0.0,  # TODO: Extract from logs
            'mAP50-95': 0.0,  # TODO: Extract from validation results
        }

        # Report completion
        sdk.report_completed(
            final_metrics=final_metrics,
            checkpoints=checkpoints,
            total_epochs=total_epochs
        )
        logger.info("Training completed successfully")

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Training failed: {error_msg}")
        logger.error(error_trace)

        # Determine error type
        if "dataset" in error_msg.lower():
            error_type = ErrorType.DATASET_ERROR
        elif "model" in error_msg.lower() or "config" in error_msg.lower():
            error_type = ErrorType.CONFIG_ERROR
        elif "cuda" in error_msg.lower() or "memory" in error_msg.lower():
            error_type = ErrorType.RESOURCE_ERROR
        else:
            error_type = ErrorType.FRAMEWORK_ERROR

        sdk.report_failed(
            error_type=error_type,
            message=error_msg,
            traceback=error_trace
        )
        sys.exit(1)

    finally:
        sdk.close()


if __name__ == "__main__":
    main()
