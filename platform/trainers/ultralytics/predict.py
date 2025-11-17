#!/usr/bin/env python3
"""
Ultralytics YOLO Inference

Simple CLI script for running inference with trained YOLO models with S3 integration and Backend callbacks.

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
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

from ultralytics import YOLO

from utils import DualStorageClient, CallbackClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Ultralytics YOLO Inference')

    parser.add_argument('--inference-job-id', type=str, help='Inference job ID')
    parser.add_argument('--training-job-id', type=str, help='Original training job ID (for checkpoint reference)')
    parser.add_argument('--checkpoint-s3-uri', type=str, help='S3 URI to trained checkpoint')
    parser.add_argument('--images-s3-uri', type=str, help='S3 URI to input images (folder or prefix)')
    parser.add_argument('--callback-url', type=str, help='Backend API base URL')
    parser.add_argument('--config', type=str, help='Inference config JSON string')
    parser.add_argument('--config-file', type=str, help='Path to inference config JSON file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from environment or args (K8s Job compatible)"""
    # Priority: env vars > CLI args (K8s Job style)
    inference_job_id = os.getenv('INFERENCE_JOB_ID') or args.inference_job_id
    training_job_id = os.getenv('TRAINING_JOB_ID') or args.training_job_id
    checkpoint_s3_uri = os.getenv('CHECKPOINT_S3_URI') or args.checkpoint_s3_uri
    images_s3_uri = os.getenv('IMAGES_S3_URI') or args.images_s3_uri
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
    if not all([inference_job_id, checkpoint_s3_uri, images_s3_uri, callback_url]):
        raise ValueError("Missing required arguments: inference_job_id, checkpoint_s3_uri, images_s3_uri, callback_url")

    return {
        'inference_job_id': inference_job_id,
        'training_job_id': training_job_id,  # Optional
        'checkpoint_s3_uri': checkpoint_s3_uri,
        'images_s3_uri': images_s3_uri,
        'callback_url': callback_url,
        'config': config
    }


async def run_inference(
    inference_job_id: str,
    training_job_id: Optional[str],
    checkpoint_s3_uri: str,
    images_s3_uri: str,
    callback_url: str,
    config: Dict[str, Any]
) -> int:
    """
    Main inference function

    Returns:
        Exit code (0 = success, 1 = inference failure, 2 = callback failure)
    """
    try:
        logger.info("=" * 80)
        logger.info(f"Ultralytics Inference Service - Job {inference_job_id}")
        logger.info("=" * 80)
        logger.info(f"Checkpoint: {checkpoint_s3_uri}")
        logger.info(f"Images: {images_s3_uri}")
        logger.info(f"Callback URL: {callback_url}")
        logger.info(f"Inference config: {json.dumps(config, indent=2)}")
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
        checkpoint_path = Path(f"/tmp/inference/{inference_job_id}/checkpoint/best.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Download from Internal Storage
        storage.internal_client.client.download_file(
            storage.internal_client.bucket,
            checkpoint_key,
            str(checkpoint_path)
        )
        logger.info(f"Checkpoint downloaded to {checkpoint_path}")

        # ========================================================================
        # Step 2: Download input images
        # ========================================================================
        logger.info(f"Downloading images from {images_s3_uri}")

        # Determine source bucket (could be External Storage or a different bucket)
        # For now, assume External Storage
        # TODO: Support custom inference buckets
        images_dir = Path(f"/tmp/inference/{inference_job_id}/images")
        images_dir.mkdir(parents=True, exist_ok=True)

        # Parse S3 URI
        # s3://bucket/prefix/ -> bucket, prefix
        s3_parts = images_s3_uri.replace('s3://', '').split('/', 1)
        images_bucket = s3_parts[0]
        images_prefix = s3_parts[1].rstrip('/') if len(s3_parts) > 1 else ''

        # Download all images from prefix
        # Use External Storage client (same as datasets)
        logger.info(f"Downloading from bucket={images_bucket}, prefix={images_prefix}")

        # List objects with prefix
        response = storage.external_client.client.list_objects_v2(
            Bucket=images_bucket,
            Prefix=images_prefix
        )

        if 'Contents' not in response:
            raise ValueError(f"No images found at {images_s3_uri}")

        image_count = 0
        for obj in response['Contents']:
            key = obj['Key']
            # Skip directory markers
            if key.endswith('/'):
                continue

            # Check if it's an image file
            ext = Path(key).suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                logger.debug(f"Skipping non-image file: {key}")
                continue

            # Download to local
            local_path = images_dir / Path(key).name
            storage.external_client.client.download_file(images_bucket, key, str(local_path))
            image_count += 1
            logger.debug(f"Downloaded {key} → {local_path}")

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

        logger.info(f"Inference parameters: conf={conf_threshold}, iou={iou_threshold}, imgsz={image_size}")

        # ========================================================================
        # Step 4: Load model from checkpoint
        # ========================================================================
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = YOLO(str(checkpoint_path))

        # ========================================================================
        # Step 5: Run inference
        # ========================================================================
        logger.info(f"Running inference on {image_count} images...")
        results_dir = Path(f"/tmp/inference/{inference_job_id}/results")

        # Run prediction
        results = model.predict(
            source=str(images_dir),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            device=device,
            save=True,  # Save annotated images
            save_txt=save_txt,  # Save labels
            save_conf=save_conf,  # Save confidence in labels
            save_crop=save_crop,  # Save cropped predictions
            project=str(results_dir),
            name='predict',
            exist_ok=True,
            max_det=max_det,
        )

        logger.info("Inference completed")

        # ========================================================================
        # Step 6: Aggregate predictions
        # ========================================================================
        logger.info("Aggregating predictions...")

        predictions: List[Dict[str, Any]] = []
        class_counts: Dict[str, int] = {}
        total_detections = 0

        for result in results:
            # Extract image info
            image_path = Path(result.path)
            image_name = image_path.name

            # Extract predictions
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    predictions.append({
                        'image_name': image_name,
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': confidence,
                        'bbox': bbox,
                    })

                    # Count classes
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    total_detections += 1

        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Classes detected: {list(class_counts.keys())}")

        # ========================================================================
        # Step 7: Create predictions summary
        # ========================================================================
        predictions_summary = {
            'total_images': image_count,
            'total_detections': total_detections,
            'class_counts': class_counts,
            'avg_confidence': sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0.0,
            'predictions': predictions,
        }

        # Save predictions.json
        predictions_json_path = results_dir / "predictions.json"
        with open(predictions_json_path, 'w') as f:
            json.dump(predictions_summary, f, indent=2)

        logger.info(f"Saved predictions summary to {predictions_json_path}")

        # ========================================================================
        # Step 8: Upload results to Internal Storage
        # ========================================================================
        logger.info("Uploading results to S3...")

        result_urls: Dict[str, str] = {}

        # Upload annotated images
        predict_dir = results_dir / "predict"
        if predict_dir.exists():
            for img_file in predict_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    s3_key = f"inference-results/{inference_job_id}/images/{img_file.name}"
                    storage.internal_client.client.upload_file(
                        str(img_file),
                        storage.internal_client.bucket,
                        s3_key,
                        ExtraArgs={'ContentType': f'image/{img_file.suffix[1:]}'}
                    )
                    logger.debug(f"Uploaded {img_file.name}")

            result_urls['annotated_images_prefix'] = f"s3://{storage.internal_client.bucket}/inference-results/{inference_job_id}/images/"

        # Upload labels (if save_txt=True)
        labels_dir = predict_dir / "labels"
        if labels_dir.exists() and save_txt:
            for label_file in labels_dir.iterdir():
                if label_file.suffix == '.txt':
                    s3_key = f"inference-results/{inference_job_id}/labels/{label_file.name}"
                    storage.internal_client.client.upload_file(
                        str(label_file),
                        storage.internal_client.bucket,
                        s3_key,
                        ExtraArgs={'ContentType': 'text/plain'}
                    )

            result_urls['labels_prefix'] = f"s3://{storage.internal_client.bucket}/inference-results/{inference_job_id}/labels/"

        # Upload predictions.json
        predictions_s3_key = f"inference-results/{inference_job_id}/predictions.json"
        storage.internal_client.client.upload_file(
            str(predictions_json_path),
            storage.internal_client.bucket,
            predictions_s3_key,
            ExtraArgs={'ContentType': 'application/json'}
        )
        predictions_uri = f"s3://{storage.internal_client.bucket}/{predictions_s3_key}"
        result_urls['predictions_json'] = predictions_uri

        logger.info(f"All results uploaded to S3")

        # ========================================================================
        # Step 9: Determine task type from model
        # ========================================================================
        task_type = 'detection'  # Default
        if hasattr(model, 'task'):
            task_type = model.task
        elif 'seg' in str(checkpoint_path).lower():
            task_type = 'segmentation'
        elif 'pose' in str(checkpoint_path).lower():
            task_type = 'pose'
        elif 'cls' in str(checkpoint_path).lower() or 'classify' in str(checkpoint_path).lower():
            task_type = 'classification'

        # ========================================================================
        # Step 10: Send inference completion callback
        # ========================================================================
        completion_data = {
            'inference_job_id': int(inference_job_id),
            'training_job_id': int(training_job_id) if training_job_id else None,
            'status': 'completed',
            'task_type': task_type,

            # Summary
            'total_images': image_count,
            'total_detections': total_detections,
            'class_counts': class_counts,
            'avg_confidence': predictions_summary['avg_confidence'],

            # Results
            'predictions_json_uri': predictions_uri,
            'result_urls': result_urls,

            # Config
            'config': {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'image_size': image_size,
                'save_txt': save_txt,
                'save_crop': save_crop,
            },

            'exit_code': 0,
        }

        try:
            await callback_client.send_inference_completion(inference_job_id, completion_data)
            logger.info("✓ Inference completed successfully")
            return 0  # Success

        except Exception as e:
            logger.error(f"Failed to send inference completion callback: {e}")
            logger.error(traceback.format_exc())
            return 2  # Callback failure

    except Exception as e:
        logger.error(f"Inference failed for job {inference_job_id}")
        logger.error(traceback.format_exc())

        # Try to send error callback
        try:
            error_data = {
                'inference_job_id': int(inference_job_id),
                'training_job_id': int(training_job_id) if training_job_id else None,
                'status': 'failed',
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'exit_code': 1,
            }
            await callback_client.send_inference_completion(inference_job_id, error_data)
        except Exception as cb_error:
            logger.error(f"Failed to send error callback: {cb_error}")

        return 1  # Inference failure


def main():
    """Main entry point"""
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        # Load configuration
        cfg = load_config(args)

        # Run inference
        exit_code = asyncio.run(run_inference(
            inference_job_id=cfg['inference_job_id'],
            training_job_id=cfg['training_job_id'],
            checkpoint_s3_uri=cfg['checkpoint_s3_uri'],
            images_s3_uri=cfg['images_s3_uri'],
            callback_url=cfg['callback_url'],
            config=cfg['config']
        ))

        logger.info(f"Inference job {cfg['inference_job_id']} finished with exit code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
