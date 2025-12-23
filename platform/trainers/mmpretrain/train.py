#!/usr/bin/env python3
"""
MMPreTrain Trainer (SDK Version)

Training script for MMPreTrain classification models.
Supports ResNet, Swin, ConvNeXt, ViT, EfficientNet, and more.

Usage:
    python train.py \\
        --job-id 123 \\
        --model-name resnet50 \\
        --dataset-s3-uri s3://bucket/datasets/abc-123/ \\
        --callback-url http://localhost:8000/api/v1/training \\
        --config '{"epochs": 100, "batch_size": 32}'

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
    parser = argparse.ArgumentParser(description='MMPreTrain Trainer')

    parser.add_argument('--job-id', type=str, help='Training job ID')
    parser.add_argument('--model-name', type=str, help='Model name (e.g., resnet50)')
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


def create_mmpretrain_config(
    model_name: str,
    dataset_dir: Path,
    work_dir: Path,
    config: Dict[str, Any],
    model_info: Dict[str, Any],
    num_classes: int
) -> str:
    """Create MMPreTrain config file dynamically."""

    # Training parameters
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.1)
    img_size = config.get('img_size', 224)
    warmup_epochs = config.get('warmup_epochs', 5)
    weight_decay = config.get('weight_decay', 0.0001)
    mixup_alpha = config.get('mixup_alpha', 0.0)
    cutmix_alpha = config.get('cutmix_alpha', 0.0)
    label_smoothing = config.get('label_smoothing', 0.0)

    # Base config
    base_config = model_info.get('config_file', 'resnet/resnet50_8xb32_in1k.py')
    pretrained_url = model_info.get('pretrained')

    # Train/val augmentations
    mixup_cutmix = ""
    if mixup_alpha > 0 or cutmix_alpha > 0:
        mixup_cutmix = f"""
# MixUp/CutMix augmentation
train_cfg = dict(
    augments=[
        dict(type='BatchMixup', alpha={mixup_alpha}, num_classes={num_classes}),
        dict(type='BatchCutMix', alpha={cutmix_alpha}, num_classes={num_classes}),
    ]
)
"""

    config_content = f'''
# Auto-generated MMPreTrain config for {model_name}
# Generated by Vision AI Platform

_base_ = 'mmpretrain::classification/{base_config}'

# Dataset settings
data_root = '{dataset_dir}'

# Override number of classes
model = dict(
    head=dict(
        num_classes={num_classes},
        loss=dict(type='LabelSmoothLoss', label_smooth_val={label_smoothing}, mode='original')
    )
)

train_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale={img_size}, backend='pillow'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size={batch_size * 2},
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale={img_size}, edge='short', backend='pillow'),
            dict(type='CenterCrop', crop_size={img_size}),
            dict(type='PackInputs')
        ]
    )
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = val_evaluator

# Training schedule
train_cfg = dict(by_epoch=True, max_epochs={epochs}, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr={learning_rate}, momentum=0.9, weight_decay={weight_decay})
)

# Learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end={warmup_epochs}),
    dict(type='CosineAnnealingLR', T_max={epochs - warmup_epochs}, by_epoch=True, begin={warmup_epochs}, end={epochs})
]

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='accuracy/top1', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False)
)

{mixup_cutmix}

# Work directory
work_dir = '{work_dir}'

# Randomness
randomness = dict(seed=42, deterministic=False)
'''

    if pretrained_url:
        config_content += f"\nload_from = '{pretrained_url}'\n"

    # Write config file
    config_path = work_dir / 'train_config.py'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)

    return str(config_path)


def count_classes(dataset_dir: Path) -> int:
    """Count number of classes from ImageFolder structure."""
    train_dir = dataset_dir / 'train'
    if train_dir.exists():
        classes = [d for d in train_dir.iterdir() if d.is_dir()]
        return len(classes)

    # Try reading from metadata
    meta_file = dataset_dir / 'meta' / 'classes.txt'
    if meta_file.exists():
        with open(meta_file) as f:
            return len([line.strip() for line in f if line.strip()])

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
        logger.info(f"MMPreTrain Training Service - Job {job_id}")
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

        # Convert DICE to ImageFolder if needed
        sdk.convert_dataset(str(dataset_dir), 'dice', 'imagefolder')

        # Count classes
        num_classes = count_classes(dataset_dir)
        logger.info(f"Number of classes: {num_classes}")

        # Training parameters
        epochs = config.get('epochs', 100)

        # Create config
        config_path = create_mmpretrain_config(
            model_name=model_name,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            config=config,
            model_info=model_info,
            num_classes=num_classes
        )
        logger.info(f"Config file created: {config_path}")

        # Import MMPreTrain
        from mmengine.config import Config
        from mmengine.runner import Runner

        cfg = Config.fromfile(config_path)
        runner = Runner.from_cfg(cfg)

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

        last_ckpt = work_dir / f'epoch_{epochs}.pth'
        if last_ckpt.exists():
            logger.info("Uploading last checkpoint")
            checkpoints['last'] = sdk.upload_checkpoint(str(last_ckpt), 'last')

        # Final metrics
        final_metrics = {
            'accuracy_top1': training_state['best_metric'],
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
