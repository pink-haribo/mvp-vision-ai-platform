#!/usr/bin/env python3
"""
OpenMMLab Model Evaluation Script

Evaluates trained models on validation/test datasets.

Environment Variables:
    CHECKPOINT_PATH: S3 URI of trained checkpoint
    DATASET_ID: Dataset ID for evaluation
    TASK_TYPE: detection, segmentation, pose
    MODEL_NAME: Model architecture
"""

import json
import logging
import sys
import traceback
from pathlib import Path

from trainer_sdk import TrainerSDK, ErrorType

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def evaluate_mmdet(model, dataset_dir: str):
    """
    Evaluate MMDetection model on COCO dataset.

    Args:
        model: MMDetection model
        dataset_dir: Dataset directory with COCO annotations

    Returns:
        Evaluation metrics
    """
    from mmdet.apis import single_gpu_test
    from mmdet.datasets import build_dataloader, build_dataset
    from mmengine import Config

    # Build validation dataset
    cfg = Config.fromfile('default_eval_config.py')
    cfg.data.val.ann_file = str(Path(dataset_dir) / "annotations_detection.json")
    cfg.data.val.img_prefix = str(Path(dataset_dir) / "images")

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False
    )

    # Run evaluation
    outputs = single_gpu_test(model, data_loader)

    # Calculate COCO metrics
    metric = dataset.evaluate(outputs)

    return {
        'mAP': metric.get('bbox_mAP', 0.0),
        'mAP50': metric.get('bbox_mAP_50', 0.0),
        'mAP75': metric.get('bbox_mAP_75', 0.0),
        'mAP_s': metric.get('bbox_mAP_s', 0.0),
        'mAP_m': metric.get('bbox_mAP_m', 0.0),
        'mAP_l': metric.get('bbox_mAP_l', 0.0),
    }


def main():
    """Main evaluation function."""
    sdk = TrainerSDK()

    try:
        logger.info("Starting evaluation...")

        # Download checkpoint
        checkpoint_s3_uri = os.getenv('CHECKPOINT_PATH')
        checkpoint_path = "/tmp/model.pth"
        sdk.download_checkpoint(checkpoint_s3_uri, checkpoint_path)
        logger.info(f"Checkpoint downloaded: {checkpoint_path}")

        # Download dataset
        dataset_dir = sdk.download_dataset(sdk.dataset_id, "/tmp/dataset")
        dataset_dir = sdk.convert_dataset(
            dataset_dir=dataset_dir,
            source_format='dice',
            target_format='coco',
            task_type=sdk.task_type
        )
        logger.info(f"Dataset prepared: {dataset_dir}")

        # Load model
        from mmdet.apis import init_detector
        config_file = f"configs/{sdk.model_name}/default.py"
        model = init_detector(config_file, checkpoint_path, device='cuda:0')
        logger.info("Model loaded")

        # Run evaluation
        logger.info("Running evaluation...")
        metrics = evaluate_mmdet(model, dataset_dir)

        logger.info(f"Evaluation results: {json.dumps(metrics, indent=2)}")
        print(f"mAP: {metrics['mAP']:.4f}")
        print(f"mAP50: {metrics['mAP50']:.4f}")
        print(f"mAP75: {metrics['mAP75']:.4f}")

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Evaluation failed: {error_msg}")
        logger.error(error_trace)
        sys.exit(1)

    finally:
        sdk.close()


if __name__ == "__main__":
    import os
    main()
