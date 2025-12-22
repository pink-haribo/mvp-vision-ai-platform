#!/usr/bin/env python3
"""
OpenMMLab Inference Script

Runs inference on trained MMDetection/MMSegmentation/MMPose models.

Environment Variables:
    CALLBACK_URL: Backend API URL
    JOB_ID: Inference job ID
    CHECKPOINT_PATH: S3 URI of trained checkpoint
    IMAGES_PATH: S3 URI of test images
    TASK_TYPE: detection, segmentation, pose
    MODEL_NAME: Model architecture
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from trainer_sdk import TrainerSDK, ErrorType

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_inference_mmdet(model, images: List[Path], threshold: float = 0.25):
    """
    Run MMDetection inference.

    Args:
        model: MMDetection model
        images: List of image paths
        threshold: Confidence threshold

    Returns:
        List of detection results
    """
    from mmdet.apis import inference_detector

    results = []
    for img_path in images:
        result = inference_detector(model, str(img_path))

        # Convert to standard format
        detections = []
        for class_id, bboxes in enumerate(result):
            for bbox in bboxes:
                if bbox[4] >= threshold:
                    detections.append({
                        'class_id': class_id,
                        'confidence': float(bbox[4]),
                        'bbox': [float(x) for x in bbox[:4]]  # [x1, y1, x2, y2]
                    })

        results.append({
            'image': img_path.name,
            'detections': detections,
            'num_detections': len(detections)
        })

    return results


def main():
    """Main inference function."""
    sdk = TrainerSDK()

    try:
        sdk.report_started('inference')
        logger.info(f"Inference job {sdk.job_id} started")

        # Download checkpoint
        checkpoint_s3_uri = os.getenv('CHECKPOINT_PATH')
        if not checkpoint_s3_uri:
            raise ValueError("CHECKPOINT_PATH environment variable required")

        checkpoint_path = "/tmp/model.pth"
        sdk.download_checkpoint(checkpoint_s3_uri, checkpoint_path)
        logger.info(f"Checkpoint downloaded: {checkpoint_path}")

        # Download test images
        images_s3_uri = os.getenv('IMAGES_PATH')
        if not images_s3_uri:
            raise ValueError("IMAGES_PATH environment variable required")

        # TODO: Implement image download from S3
        images_dir = Path("/tmp/test_images")
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        logger.info(f"Found {len(images)} test images")

        # Load model
        logger.info("Loading model...")
        from mmdet.apis import init_detector

        # Get config from checkpoint or use default
        config_file = f"configs/{sdk.model_name}/default.py"
        model = init_detector(config_file, checkpoint_path, device='cuda:0')
        logger.info("Model loaded successfully")

        # Run inference
        logger.info("Running inference...")
        start_time = time.time()
        results = run_inference_mmdet(model, images, threshold=0.25)
        total_time_ms = (time.time() - start_time) * 1000

        # Report completion
        sdk.report_inference_completed(
            total_images=len(images),
            total_time_ms=total_time_ms,
            results=results
        )
        logger.info("Inference completed successfully")

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Inference failed: {error_msg}")
        logger.error(error_trace)

        error_type = ErrorType.FRAMEWORK_ERROR
        if "checkpoint" in error_msg.lower():
            error_type = ErrorType.CHECKPOINT_ERROR

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
