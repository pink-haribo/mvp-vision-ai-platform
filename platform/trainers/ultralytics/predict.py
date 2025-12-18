#!/usr/bin/env python3
"""
Ultralytics YOLO Inference (SDK Version)

Simple CLI script for running inference with trained YOLO models using the Trainer SDK.
All observability is handled by Backend.

Usage:
    python predict.py \
        --inference-job-id 789 \
        --training-job-id 456 \
        --checkpoint-s3-uri s3://training-checkpoints/checkpoints/456/best.pt \
        --images-s3-uri s3://inference-images/batch-xyz/ \
        --callback-url http://localhost:8000/api/v1/test_inference \
        --config '{"conf": 0.25, "iou": 0.45, "save_txt": true, "save_crop": false}'

Environment Variables (alternative to CLI args):
    INFERENCE_JOB_ID, TRAINING_JOB_ID, CHECKPOINT_S3_URI, IMAGES_S3_URI, CALLBACK_URL, CONFIG

Exit Codes:
    0 = Success
    1 = Inference failure
    2 = Callback failure
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from ultralytics import YOLO

from trainer_sdk import ErrorType, TrainerSDK

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainerSDKLogHandler(logging.Handler):
    """
    Custom logging handler that forwards logs to Backend via TrainerSDK.

    This allows existing logger.info() calls to automatically send logs
    to Backend → WebSocket → Frontend without code changes.
    """

    def __init__(self, sdk: TrainerSDK):
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
            self.sdk.log(message, level=level, source='inference')
        except Exception:
            pass

    def disable(self):
        """Disable the handler (used during shutdown)"""
        self._enabled = False


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Ultralytics YOLO Inference')

    parser.add_argument('--inference-job-id', type=str, help='Inference job ID')
    parser.add_argument('--training-job-id', type=str, help='Original training job ID')
    parser.add_argument('--checkpoint-s3-uri', type=str, help='S3 URI to trained checkpoint')
    parser.add_argument('--images-s3-uri', type=str, help='S3 URI to input images')
    parser.add_argument('--callback-url', type=str, help='Backend API base URL')
    parser.add_argument('--config', type=str, help='Inference config JSON string')
    parser.add_argument('--config-file', type=str, help='Path to inference config JSON file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from environment or args"""
    # Priority: env vars > CLI args
    inference_job_id = os.getenv('INFERENCE_JOB_ID') or args.inference_job_id
    training_job_id = os.getenv('TRAINING_JOB_ID') or args.training_job_id
    checkpoint_s3_uri = os.getenv('CHECKPOINT_S3_URI') or args.checkpoint_s3_uri
    images_s3_uri = os.getenv('IMAGES_S3_URI') or args.images_s3_uri
    callback_url = os.getenv('CALLBACK_URL') or args.callback_url

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
    if not all([inference_job_id, checkpoint_s3_uri, images_s3_uri, callback_url]):
        raise ValueError("Missing required arguments: inference_job_id, checkpoint_s3_uri, images_s3_uri, callback_url")

    # Set environment variables for SDK
    os.environ['JOB_ID'] = str(inference_job_id)
    os.environ['CALLBACK_URL'] = callback_url

    return {
        'inference_job_id': inference_job_id,
        'training_job_id': training_job_id,
        'checkpoint_s3_uri': checkpoint_s3_uri,
        'images_s3_uri': images_s3_uri,
        'callback_url': callback_url,
        'config': config
    }


def run_inference(
    inference_job_id: str,
    training_job_id: Optional[str],
    checkpoint_s3_uri: str,
    images_s3_uri: str,
    config: Dict[str, Any]
) -> int:
    """
    Main inference function

    Returns:
        Exit code (0 = success, 1 = inference failure)
    """
    # Initialize SDK
    sdk = TrainerSDK()

    # Add SDK log handler to forward logs to Backend
    sdk_handler = TrainerSDKLogHandler(sdk)
    sdk_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sdk_handler)

    try:
        logger.info("=" * 80)
        logger.info(f"Ultralytics Inference Service - Job {inference_job_id}")
        logger.info("=" * 80)
        logger.info(f"Checkpoint: {checkpoint_s3_uri}")
        logger.info(f"Images: {images_s3_uri}")
        logger.info(f"Inference config: {json.dumps(config, indent=2)}")
        logger.info("=" * 80)

        # Report started
        sdk.report_started('inference')

        # ========================================================================
        # Step 1: Load checkpoint
        # ========================================================================
        if checkpoint_s3_uri == 'pretrained':
            model_name = os.getenv('MODEL_NAME', 'yolo11n')
            checkpoint_path = f"{model_name}.pt"
            logger.info(f"Using pretrained model: {checkpoint_path}")
        else:
            logger.info(f"Downloading checkpoint from {checkpoint_s3_uri}")
            checkpoint_path = Path(f"/tmp/inference/{inference_job_id}/checkpoint/best.pt")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            sdk.download_checkpoint(checkpoint_s3_uri, str(checkpoint_path))
            logger.info(f"Checkpoint downloaded to {checkpoint_path}")

        # ========================================================================
        # Step 2: Download input images
        # ========================================================================
        logger.info(f"Downloading images from {images_s3_uri}")
        images_dir = Path(f"/tmp/inference/{inference_job_id}/images")
        images_dir.mkdir(parents=True, exist_ok=True)

        # Parse S3 URI and download
        s3_parts = images_s3_uri.replace('s3://', '').split('/', 1)
        images_bucket = s3_parts[0]
        images_prefix = s3_parts[1].rstrip('/') if len(s3_parts) > 1 else ''

        # Determine storage type based on bucket
        if images_bucket in ['training-checkpoints', 'inference-data']:
            storage = sdk.internal_storage
        else:
            storage = sdk.external_storage

        logger.info(f"Downloading from bucket={images_bucket}, prefix={images_prefix}")

        # List and download images
        response = storage.client.list_objects_v2(
            Bucket=images_bucket,
            Prefix=images_prefix
        )

        if 'Contents' not in response:
            raise ValueError(f"No images found at {images_s3_uri}")

        image_count = 0
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue

            ext = Path(key).suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                continue

            local_path = images_dir / Path(key).name
            storage.client.download_file(images_bucket, key, str(local_path))
            image_count += 1

        logger.info(f"Downloaded {image_count} images to {images_dir}")

        # ========================================================================
        # Step 3: Extract inference parameters
        # ========================================================================
        conf_threshold = config.get('conf', 0.25)
        iou_threshold = config.get('iou', 0.45)
        image_size = config.get('imgsz', 640)
        device = config.get('device', 'cpu')
        save_txt = config.get('save_txt', True)
        save_crop = config.get('save_crop', False)
        save_conf = config.get('save_conf', True)
        max_det = config.get('max_det', 300)

        logger.info(f"Inference parameters: conf={conf_threshold}, iou={iou_threshold}")

        # ========================================================================
        # Step 4: Load model and run inference
        # ========================================================================
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = YOLO(str(checkpoint_path))

        logger.info(f"Running inference on {image_count} images...")
        results_dir = Path(f"/tmp/inference/{inference_job_id}/results")

        start_time = time.time()
        results = model.predict(
            source=str(images_dir),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            device=device,
            save=True,
            save_txt=save_txt,
            save_conf=save_conf,
            save_crop=save_crop,
            project=str(results_dir),
            name='predict',
            exist_ok=True,
            max_det=max_det,
        )
        total_inference_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Inference completed in {total_inference_time_ms:.1f}ms")

        # ========================================================================
        # Step 5: Extract per-image results
        # ========================================================================
        logger.info("Extracting per-image results...")

        image_results: List[Dict[str, Any]] = []
        total_detections = 0

        for result in results:
            image_path = Path(result.path)
            image_name = image_path.name
            image_inference_time_ms = result.speed.get('inference', 0) if hasattr(result, 'speed') else 0

            image_predictions: List[Dict[str, Any]] = []
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()

                    image_predictions.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': confidence,
                        'bbox': bbox,
                    })
                    total_detections += 1

            image_results.append({
                'image_path': str(image_path),
                'image_name': image_name,
                'predictions': image_predictions,
                'inference_time_ms': image_inference_time_ms,
                'num_detections': len(image_predictions)
            })

        avg_inference_time_ms = total_inference_time_ms / image_count if image_count > 0 else 0
        logger.info(f"Total detections: {total_detections} across {image_count} images")

        # ========================================================================
        # Step 6: Save and upload results
        # ========================================================================
        predictions_summary = {
            'total_images': image_count,
            'total_detections': total_detections,
            'avg_inference_time_ms': avg_inference_time_ms,
            'results': image_results,
        }

        predictions_json_path = results_dir / "predictions.json"
        with open(predictions_json_path, 'w') as f:
            json.dump(predictions_summary, f, indent=2)

        logger.info("Uploading results to S3...")

        result_urls: Dict[str, str] = {}

        # Upload annotated images
        predict_dir = results_dir / "predict"
        if predict_dir.exists():
            for img_file in predict_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    s3_key = f"inference-results/{inference_job_id}/images/{img_file.name}"
                    sdk.upload_file(
                        str(img_file),
                        s3_key,
                        content_type=f'image/{img_file.suffix[1:]}',
                        storage_type='internal'
                    )

            result_urls['annotated_images_prefix'] = f"s3://{sdk.internal_storage.bucket}/inference-results/{inference_job_id}/images/"

        # Upload labels
        labels_dir = predict_dir / "labels"
        if labels_dir.exists() and save_txt:
            for label_file in labels_dir.iterdir():
                if label_file.suffix == '.txt':
                    s3_key = f"inference-results/{inference_job_id}/labels/{label_file.name}"
                    sdk.upload_file(
                        str(label_file),
                        s3_key,
                        content_type='text/plain',
                        storage_type='internal'
                    )
            result_urls['labels_prefix'] = f"s3://{sdk.internal_storage.bucket}/inference-results/{inference_job_id}/labels/"

        # Upload predictions.json
        predictions_s3_key = f"inference-results/{inference_job_id}/predictions.json"
        predictions_uri = sdk.upload_file(
            str(predictions_json_path),
            predictions_s3_key,
            content_type='application/json',
            storage_type='internal'
        )
        result_urls['predictions_json'] = predictions_uri

        logger.info("All results uploaded to S3")

        # ========================================================================
        # Step 7: Send completion callback
        # ========================================================================
        sdk.report_inference_completed(
            total_images=image_count,
            total_time_ms=total_inference_time_ms,
            results=image_results,
            result_urls=result_urls
        )

        logger.info("Inference completed successfully")

        # Flush remaining logs and cleanup handler
        sdk.flush_logs()
        sdk_handler.disable()
        logger.removeHandler(sdk_handler)

        sdk.close()
        return 0

    except Exception as e:
        logger.error(f"Inference failed for job {inference_job_id}")
        logger.error(traceback.format_exc())

        error_type = ErrorType.UNKNOWN_ERROR
        error_msg = str(e)

        if 'checkpoint' in error_msg.lower():
            error_type = ErrorType.CHECKPOINT_ERROR
        elif 'image' in error_msg.lower() or 'not found' in error_msg.lower():
            error_type = ErrorType.DATASET_ERROR
        elif 'CUDA' in error_msg or 'memory' in error_msg.lower():
            error_type = ErrorType.RESOURCE_ERROR

        try:
            sdk.report_failed(
                error_type=error_type,
                message=error_msg,
                traceback=traceback.format_exc()
            )
        except Exception as cb_error:
            logger.error(f"Failed to send error callback: {cb_error}")

        # Flush remaining logs and cleanup handler
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

        exit_code = run_inference(
            inference_job_id=cfg['inference_job_id'],
            training_job_id=cfg['training_job_id'],
            checkpoint_s3_uri=cfg['checkpoint_s3_uri'],
            images_s3_uri=cfg['images_s3_uri'],
            config=cfg['config']
        )

        logger.info(f"Inference job {cfg['inference_job_id']} finished with exit code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
