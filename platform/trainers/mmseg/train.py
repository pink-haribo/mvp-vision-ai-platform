#!/usr/bin/env python3
"""
MMSegmentation Trainer (SDK Version)

Training script for MMSegmentation semantic segmentation models.
Supports DeepLabV3+, SegFormer, Mask2Former, UperNet, and more.

Usage:
    python train.py \\
        --job-id 123 \\
        --model-name segformer_mit-b2 \\
        --dataset-s3-uri s3://bucket/datasets/abc-123/ \\
        --callback-url http://localhost:8000/api/v1/training \\
        --config '{"epochs": 80, "batch_size": 4}'

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
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

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
    parser = argparse.ArgumentParser(description='MMSegmentation Trainer')

    parser.add_argument('--job-id', type=str, help='Training job ID')
    parser.add_argument('--model-name', type=str, help='Model name (e.g., segformer_mit-b2)')
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
        raise ValueError("Missing required arguments")

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


def create_mmseg_config(
    model_name: str,
    dataset_dir: Path,
    work_dir: Path,
    config: Dict[str, Any],
    model_info: Dict[str, Any],
    num_classes: int,
    class_names: list
) -> str:
    """Create MMSegmentation config file dynamically."""

    # Training parameters
    epochs = config.get('epochs', 80)
    batch_size = config.get('batch_size', 4)
    learning_rate = config.get('learning_rate', 0.01)
    crop_size = config.get('crop_size', (512, 512))
    weight_decay = config.get('weight_decay', 0.0005)
    val_interval = config.get('val_interval', 1)

    # Base config
    base_config = model_info.get('config_file', 'deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_cityscapes-512x1024.py')
    pretrained_url = model_info.get('pretrained')

    # Convert epochs to iterations (rough estimate)
    # MMSeg typically uses iterations, not epochs
    # Assuming ~1000 images, batch_size=4 -> ~250 iters/epoch
    iters_per_epoch = 250  # Will be recalculated based on actual dataset
    max_iters = epochs * iters_per_epoch

    config_content = f'''
# Auto-generated MMSegmentation config for {model_name}
# Generated by Vision AI Platform

_base_ = 'mmseg::segmentation/{base_config}'

# Dataset settings
data_root = '{dataset_dir}'

# Override number of classes
num_classes = {num_classes}

# Model modifications for custom classes
model = dict(
    decode_head=dict(num_classes={num_classes}),
    auxiliary_head=dict(num_classes={num_classes}) if hasattr(_base_.model, 'auxiliary_head') else None
)

# Custom dataset
train_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='annotations/train'
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomResize', scale=(2048, {crop_size[0]}), ratio_range=(0.5, 2.0), keep_ratio=True),
            dict(type='RandomCrop', crop_size={crop_size}, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='annotations/val'
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, {crop_size[0]}), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]
    )
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# Training schedule (iteration-based)
train_cfg = dict(type='IterBasedTrainLoop', max_iters={max_iters}, val_interval={val_interval * iters_per_epoch})
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr={learning_rate}, momentum=0.9, weight_decay={weight_decay}),
    clip_grad=None
)

# Learning rate scheduler (poly schedule for segmentation)
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end={max_iters}, by_epoch=False)
]

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval={val_interval * iters_per_epoch}, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# Work directory
work_dir = '{work_dir}'

# Randomness
randomness = dict(seed=42)
'''

    if pretrained_url:
        config_content += f"\nload_from = '{pretrained_url}'\n"

    # Write config file
    config_path = work_dir / 'train_config.py'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)

    return str(config_path)


def parse_dataset_info(dataset_dir: Path) -> tuple:
    """Parse dataset to get class info."""
    # Try to read class info from annotations
    ann_file = dataset_dir / 'annotations_segmentation.json'
    if not ann_file.exists():
        ann_file = dataset_dir / 'annotations.json'

    if ann_file.exists():
        with open(ann_file) as f:
            data = json.load(f)
        categories = data.get('categories', [])
        class_names = [cat['name'] for cat in categories]
        num_classes = len(class_names)
        return num_classes, class_names

    # Fallback: count unique values in mask images
    mask_dir = dataset_dir / 'annotations' / 'train'
    if mask_dir.exists():
        import numpy as np
        from PIL import Image

        unique_classes = set()
        for mask_file in list(mask_dir.glob('*.png'))[:10]:  # Sample first 10
            mask = np.array(Image.open(mask_file))
            unique_classes.update(np.unique(mask).tolist())

        num_classes = max(unique_classes) + 1
        class_names = [f'class_{i}' for i in range(num_classes)]
        return num_classes, class_names

    raise ValueError("Cannot determine number of classes")


def train_model(
    job_id: str,
    model_name: str,
    dataset_s3_uri: str,
    config: Dict[str, Any]
) -> int:
    """Main training function"""
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
        logger.info(f"MMSegmentation Training Service - Job {job_id}")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Dataset: {dataset_s3_uri}")
        logger.info(f"Training config: {json.dumps(config, indent=2)}")
        logger.info("=" * 80)

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

        # Convert DICE to segmentation format if needed
        sdk.convert_dataset(str(dataset_dir), 'dice', 'segmentation')

        # Parse dataset info
        num_classes, class_names = parse_dataset_info(dataset_dir)
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Classes: {class_names}")

        # Training parameters
        epochs = config.get('epochs', 80)

        # Create config
        config_path = create_mmseg_config(
            model_name=model_name,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            config=config,
            model_info=model_info,
            num_classes=num_classes,
            class_names=class_names
        )
        logger.info(f"Config file created: {config_path}")

        # Import MMSegmentation
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmengine.hooks import Hook

        cfg = Config.fromfile(config_path)
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
                self.logged_iters = set()

            def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
                # Report progress every 100 iterations
                current_iter = runner.iter + 1
                if current_iter % 100 != 0 or current_iter in self.logged_iters:
                    return

                self.logged_iters.add(current_iter)

                # Get metrics from message hub
                metrics = {}
                if hasattr(runner, 'message_hub'):
                    log_scalars = runner.message_hub.log_scalars
                    for key, value in log_scalars.items():
                        if isinstance(value, (int, float)):
                            metrics[key] = float(value)

                loss = metrics.get('loss', 0.0)

                # Estimate epoch from iteration (rough estimate)
                iters_per_epoch = getattr(runner, '_max_iters', 20000) // self.total_epochs
                epoch = current_iter // max(iters_per_epoch, 1)

                # Send progress
                self.sdk.report_progress(
                    epoch=epoch,
                    total_epochs=self.total_epochs,
                    metrics={
                        'loss': loss,
                        'iteration': current_iter,
                        'learning_rate': metrics.get('lr', 0.0),
                        **metrics
                    }
                )

            def after_val_epoch(self, runner, metrics=None):
                if metrics is None:
                    return

                # Estimate epoch from iteration
                iters_per_epoch = getattr(runner, '_max_iters', 20000) // self.total_epochs
                epoch = runner.iter // max(iters_per_epoch, 1)

                mIoU = metrics.get('mIoU', 0.0)
                mDice = metrics.get('mDice', 0.0)
                mFscore = metrics.get('mFscore', 0.0)

                is_best = mIoU > self.training_state['best_metric']
                if is_best:
                    self.training_state['best_metric'] = mIoU
                    logger.info(f"[BEST] New best mIoU: {mIoU:.4f} at iter {runner.iter}")

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
                            logger.info(f"[BEST] Best checkpoint uploaded successfully")
                        except Exception as e:
                            logger.warning(f"[BEST] Failed to upload best checkpoint: {e}")

                # Report validation metrics
                self.sdk.report_validation(
                    epoch=epoch,
                    task_type='semantic_segmentation',
                    primary_metric=('mIoU', mIoU),
                    all_metrics={
                        'mIoU': mIoU,
                        'mDice': mDice,
                        'mFscore': mFscore,
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

        # Find and upload checkpoints
        checkpoints = {}
        import glob as glob_module

        best_files = glob_module.glob(str(work_dir / 'best_*.pth'))
        if best_files:
            best_pt = best_files[0]
            logger.info(f"Uploading best checkpoint: {best_pt}")
            checkpoints['best'] = sdk.upload_checkpoint(best_pt, 'best')

        # Find last checkpoint
        iter_files = sorted(glob_module.glob(str(work_dir / 'iter_*.pth')))
        if iter_files:
            last_pt = iter_files[-1]
            logger.info(f"Uploading last checkpoint: {last_pt}")
            checkpoints['last'] = sdk.upload_checkpoint(last_pt, 'last')

        # Final metrics
        final_metrics = {
            'mIoU': training_state['best_metric'],
            'epochs_completed': epochs
        }

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
