#!/usr/bin/env python3
"""
MMDetection Trainer (SDK Version)

Training script for MMDetection models using the Trainer SDK.
Supports Faster R-CNN, DETR, DINO, RTMDet, and other MMDet models.

Usage:
    python train.py \\
        --job-id 123 \\
        --model-name faster-rcnn_r50_fpn \\
        --dataset-s3-uri s3://bucket/datasets/abc-123/ \\
        --callback-url http://localhost:8000/api/v1/training \\
        --config '{"epochs": 12, "batch_size": 2}'

Environment Variables (alternative to CLI args):
    JOB_ID, MODEL_NAME, DATASET_S3_URI, CALLBACK_URL, CONFIG

Exit Codes:
    0 = Success
    1 = Training failure
    2 = Callback failure
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainerSDKLogHandler(logging.Handler):
    """Custom logging handler that forwards logs to Backend via TrainerSDK."""

    def __init__(self, sdk):
        super().__init__()
        self.sdk = sdk
        self._enabled = True

    def emit(self, record: logging.LogRecord):
        if not self._enabled:
            return
        try:
            level_map = {
                logging.DEBUG: 'DEBUG',
                logging.INFO: 'INFO',
                logging.WARNING: 'WARNING',
                logging.ERROR: 'ERROR',
                logging.CRITICAL: 'ERROR',
            }
            level = level_map.get(record.levelno, 'INFO')
            message = self.format(record)
            self.sdk.log(message, level=level, source='trainer')
        except Exception:
            pass

    def disable(self):
        self._enabled = False


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='MMDetection Trainer')

    parser.add_argument('--job-id', type=str, help='Training job ID')
    parser.add_argument('--model-name', type=str, help='Model name (e.g., faster-rcnn_r50_fpn)')
    parser.add_argument('--dataset-s3-uri', type=str, help='S3 URI to dataset')
    parser.add_argument('--callback-url', type=str, help='Backend API base URL')
    parser.add_argument('--config', type=str, help='Training config JSON string')
    parser.add_argument('--config-file', type=str, help='Path to training config JSON file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from args or environment"""
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

    if not all([job_id, model_name, dataset_s3_uri, callback_url]):
        raise ValueError("Missing required arguments: job_id, model_name, dataset_s3_uri, callback_url")

    os.environ['JOB_ID'] = str(job_id)
    os.environ['CALLBACK_URL'] = callback_url

    return {
        'job_id': job_id,
        'model_name': model_name,
        'dataset_s3_uri': dataset_s3_uri,
        'callback_url': callback_url,
        'config': config
    }


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration from capabilities.json"""
    capabilities_path = Path(__file__).parent / 'capabilities.json'
    with open(capabilities_path) as f:
        capabilities = json.load(f)

    for model in capabilities['models']:
        if model['model_name'] == model_name:
            return model

    raise ValueError(f"Unknown model: {model_name}")


def create_mmdet_config(
    model_name: str,
    dataset_dir: Path,
    work_dir: Path,
    config: Dict[str, Any],
    model_info: Dict[str, Any]
) -> str:
    """
    Create MMDetection config file dynamically.

    Args:
        model_name: Model name
        dataset_dir: Path to dataset
        work_dir: Working directory for outputs
        config: Training configuration
        model_info: Model info from capabilities.json

    Returns:
        Path to created config file
    """
    # Training parameters
    epochs = config.get('epochs', 12)
    batch_size = config.get('batch_size', 2)
    learning_rate = config.get('learning_rate', 0.02)
    warmup_iters = config.get('warmup_iters', 500)
    val_interval = config.get('val_interval', 1)
    img_scale = config.get('img_scale', (1333, 800))

    # Paths
    ann_file_train = str(dataset_dir / 'annotations_detection.json')
    ann_file_val = str(dataset_dir / 'annotations_detection.json')  # Same file, split handled internally
    img_prefix = str(dataset_dir / 'images')

    # Check for separate val annotations
    if (dataset_dir / 'annotations_val.json').exists():
        ann_file_val = str(dataset_dir / 'annotations_val.json')

    # Base config based on model
    base_config = model_info.get('config_file', 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')
    pretrained_url = model_info.get('pretrained')

    # Create config content
    config_content = f'''
# Auto-generated MMDetection config for {model_name}
# Generated by Vision AI Platform

_base_ = 'mmdet::detection/{base_config}'

# Dataset settings
data_root = '{dataset_dir}'

train_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='{ann_file_train}',
        data_prefix=dict(img='{img_prefix}'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale={img_scale}, keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='{ann_file_val}',
        data_prefix=dict(img='{img_prefix}'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale={img_scale}, keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]
    )
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file='{ann_file_val}',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# Training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={epochs}, val_interval={val_interval})
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr={learning_rate}, momentum=0.9, weight_decay=0.0001)
)

# Learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end={warmup_iters}),
    dict(type='MultiStepLR', by_epoch=True, milestones=[{int(epochs * 0.67)}, {int(epochs * 0.92)}], gamma=0.1)
]

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Load pretrained weights
'''
    if pretrained_url:
        config_content += f"load_from = '{pretrained_url}'\n"
    else:
        config_content += "load_from = None\n"

    config_content += f'''
# Work directory
work_dir = '{work_dir}'

# Randomness
randomness = dict(seed=42, deterministic=False)
'''

    # Write config file
    config_path = work_dir / 'train_config.py'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)

    return str(config_path)


def train_model(
    job_id: str,
    model_name: str,
    dataset_s3_uri: str,
    config: Dict[str, Any]
) -> int:
    """
    Main training function

    Returns:
        Exit code (0 = success, 1 = training failure, 2 = callback failure)
    """
    from trainer_sdk import TrainerSDK, ErrorType

    training_state = {
        'current_epoch': 0,
        'best_metric': 0.0
    }

    sdk = TrainerSDK()
    sdk_handler = TrainerSDKLogHandler(sdk)
    sdk_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sdk_handler)

    try:
        logger.info("=" * 80)
        logger.info(f"MMDetection Training Service - Job {job_id}")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Dataset: {dataset_s3_uri}")
        logger.info(f"Training config: {json.dumps(config, indent=2)}")
        logger.info("=" * 80)

        # Report started
        sdk.report_started('training')

        # Get model info
        model_info = get_model_config(model_name)
        logger.info(f"Model config: {model_info['display_name']}")

        # Setup directories
        job_working_dir = Path(f"/tmp/training/{job_id}")
        dataset_dir = job_working_dir / "dataset"
        work_dir = job_working_dir / "work_dir"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset
        dataset_id = dataset_s3_uri.rstrip('/').split('/')[-1]
        snapshot_id = os.getenv('SNAPSHOT_ID')
        dataset_version_hash = os.getenv('DATASET_VERSION_HASH')

        if snapshot_id and dataset_version_hash:
            logger.info(f"[Cache] Downloading dataset with caching")
            dataset_dir = Path(sdk.download_dataset_with_cache(
                snapshot_id=snapshot_id,
                dataset_id=dataset_id,
                dataset_version_hash=dataset_version_hash,
                dest_dir=str(job_working_dir)
            ))
        else:
            logger.info(f"Downloading dataset from {dataset_s3_uri}")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            sdk.download_dataset(dataset_id, str(dataset_dir))

        logger.info(f"Dataset ready at {dataset_dir}")

        # Convert DICE to COCO format if needed
        # MMDetection expects COCO format natively
        sdk.convert_dataset(str(dataset_dir), 'dice', 'coco')

        # Training parameters
        epochs = config.get('epochs', 12)

        # Create MMDetection config
        config_path = create_mmdet_config(
            model_name=model_name,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            config=config,
            model_info=model_info
        )
        logger.info(f"Config file created: {config_path}")

        # Import MMDetection components
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmengine.hooks import Hook

        # Load config
        cfg = Config.fromfile(config_path)

        # Build runner
        runner = Runner.from_cfg(cfg)

        # Custom hook for progress reporting (MMEngine Hook)
        class ProgressHook(Hook):
            """Custom hook for reporting training progress and uploading best checkpoints."""

            def __init__(self, sdk, total_epochs, training_state, checkpoint_dir):
                super().__init__()
                self.sdk = sdk
                self.total_epochs = total_epochs
                self.training_state = training_state
                self.checkpoint_dir = checkpoint_dir
                self.logged_epochs = set()

            def after_train_epoch(self, runner):
                epoch = runner.epoch + 1
                if epoch in self.logged_epochs:
                    return

                self.logged_epochs.add(epoch)
                self.training_state['current_epoch'] = epoch

                # Get metrics from message hub
                metrics = {}
                if hasattr(runner, 'message_hub'):
                    log_scalars = runner.message_hub.log_scalars
                    for key, value in log_scalars.items():
                        if isinstance(value, (int, float)):
                            metrics[key] = float(value)

                loss = metrics.get('loss', 0.0)

                # Send progress
                self.sdk.report_progress(
                    epoch=epoch,
                    total_epochs=self.total_epochs,
                    metrics={
                        'loss': loss,
                        'learning_rate': metrics.get('lr', 0.0),
                        **metrics
                    }
                )

            def after_val_epoch(self, runner, metrics=None):
                if metrics is None:
                    return

                epoch = runner.epoch + 1
                mAP = metrics.get('coco/bbox_mAP', 0.0)
                mAP_50 = metrics.get('coco/bbox_mAP_50', 0.0)

                is_best = mAP > self.training_state['best_metric']
                if is_best:
                    self.training_state['best_metric'] = mAP
                    logger.info(f"[BEST] New best mAP: {mAP:.4f} at epoch {epoch}")

                    # Upload best checkpoint immediately when updated
                    import glob as glob_module
                    import time
                    # Small delay to ensure checkpoint is fully written
                    time.sleep(0.5)
                    best_files = glob_module.glob(str(self.checkpoint_dir / 'best_*.pth'))
                    if best_files:
                        best_pt = sorted(best_files, key=lambda x: Path(x).stat().st_mtime)[-1]
                        logger.info(f"[BEST] Uploading best checkpoint: {best_pt}")
                        try:
                            self.sdk.upload_checkpoint(best_pt, 'best')
                            logger.info(f"[BEST] Best checkpoint uploaded successfully (epoch {epoch})")
                        except Exception as e:
                            logger.warning(f"[BEST] Failed to upload best checkpoint: {e}")

                # Report validation metrics
                self.sdk.report_validation(
                    epoch=epoch,
                    task_type='detection',
                    primary_metric=('bbox_mAP', mAP),
                    all_metrics={
                        'bbox_mAP': mAP,
                        'bbox_mAP_50': mAP_50,
                        'bbox_mAP_75': metrics.get('coco/bbox_mAP_75', 0.0),
                        'bbox_mAP_s': metrics.get('coco/bbox_mAP_s', 0.0),
                        'bbox_mAP_m': metrics.get('coco/bbox_mAP_m', 0.0),
                        'bbox_mAP_l': metrics.get('coco/bbox_mAP_l', 0.0),
                    },
                    is_best=is_best
                )

        # Register custom hook with runner
        progress_hook = ProgressHook(sdk, epochs, training_state, work_dir)
        runner.register_hook(progress_hook, priority='LOW')
        logger.info("Progress hook registered")

        # Start training
        logger.info(f"Starting training for {epochs} epochs")
        runner.train()

        logger.info("Training completed")

        # Find checkpoints
        best_ckpt = work_dir / 'best_coco_bbox_mAP_epoch_*.pth'
        last_ckpt = work_dir / f'epoch_{epochs}.pth'

        # Upload checkpoints
        checkpoints = {}
        import glob as glob_module

        best_files = glob_module.glob(str(work_dir / 'best_*.pth'))
        if best_files:
            best_pt = best_files[0]
            logger.info(f"Uploading best checkpoint: {best_pt}")
            checkpoints['best'] = sdk.upload_checkpoint(best_pt, 'best')

        if last_ckpt.exists():
            logger.info("Uploading last checkpoint")
            checkpoints['last'] = sdk.upload_checkpoint(str(last_ckpt), 'last')

        # Final metrics
        final_metrics = {
            'bbox_mAP': training_state['best_metric'],
            'epochs_completed': epochs
        }

        # Report completion
        sdk.report_completed(
            final_metrics=final_metrics,
            checkpoints=checkpoints if checkpoints else None,
            total_epochs=epochs
        )

        logger.info("Training completed successfully")

        sdk.flush_logs()
        sdk_handler.disable()
        logger.removeHandler(sdk_handler)
        sdk.close()
        return 0

    except Exception as e:
        logger.error(f"Training failed for job {job_id}")
        logger.error(traceback.format_exc())

        from trainer_sdk import ErrorType

        error_type = ErrorType.UNKNOWN_ERROR
        error_msg = str(e)

        if 'CUDA' in error_msg or 'memory' in error_msg.lower():
            error_type = ErrorType.RESOURCE_ERROR
        elif 'dataset' in error_msg.lower() or 'not found' in error_msg.lower():
            error_type = ErrorType.DATASET_ERROR
        elif 'config' in error_msg.lower():
            error_type = ErrorType.CONFIG_ERROR

        try:
            sdk.report_failed(
                error_type=error_type,
                message=error_msg,
                traceback=traceback.format_exc(),
                epochs_completed=training_state.get('current_epoch', 0)
            )
        except Exception as cb_error:
            logger.error(f"Failed to send error callback: {cb_error}")

        sdk.flush_logs()
        sdk_handler.disable()
        logger.removeHandler(sdk_handler)
        sdk.close()
        return 1


def main():
    """Main entry point"""
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        cfg = load_config(args)

        exit_code = train_model(
            job_id=cfg['job_id'],
            model_name=cfg['model_name'],
            dataset_s3_uri=cfg['dataset_s3_uri'],
            config=cfg['config']
        )

        logger.info(f"Training job {cfg['job_id']} finished with exit code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
