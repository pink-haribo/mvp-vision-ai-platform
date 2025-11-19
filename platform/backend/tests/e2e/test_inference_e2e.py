#!/usr/bin/env python3
"""
E2E Inference Test Script

Tests the complete inference pipeline exactly as the frontend does:
1. Upload images to S3 (inference-data bucket)
2. Create InferenceJob with pretrained or trained checkpoint
3. Wait for completion
4. Fetch and display results

Usage:
    python test_inference_e2e.py --job-id 23 --pretrained --images test_images/*.jpg
    python test_inference_e2e.py --job-id 23 --images test_images/*.jpg  # Uses trained checkpoint
"""

import argparse
import json
import time
import sys
import glob
from pathlib import Path

import requests


def test_inference(
    api_url: str,
    training_job_id: int,
    image_paths: list,
    use_pretrained: bool = False,
    timeout: int = 60
):
    """Run complete E2E inference test"""

    print("=" * 80)
    print(f"E2E Inference Test - Job {training_job_id}")
    print("=" * 80)
    print(f"Images: {len(image_paths)}")
    print(f"Pretrained: {use_pretrained}")
    print("=" * 80)
    print()

    # Step 1: Get training job info
    print("[Step 1] Fetching training job info...")
    job_response = requests.get(f"{api_url}/training/jobs/{training_job_id}")

    if job_response.status_code == 404:
        # Try alternative endpoint
        print("  → Training job endpoint not found, using direct query...")
        # We'll proceed without job info for now
        model_name = "yolo11n"  # Default
        checkpoint_path = None
    else:
        job_data = job_response.json()
        model_name = job_data.get('model_name', 'yolo11n')
        checkpoint_path = job_data.get('best_checkpoint_path')
        print(f"  → Model: {model_name}")
        print(f"  → Checkpoint: {checkpoint_path}")

    print()

    # Step 2: Upload images
    print(f"[Step 2] Uploading {len(image_paths)} images to S3...")

    files = []
    for path in image_paths:
        files.append(('files', (Path(path).name, open(path, 'rb'), 'image/jpeg')))

    try:
        upload_response = requests.post(
            f"{api_url}/test_inference/inference/upload-images",
            params={'training_job_id': training_job_id},
            files=files
        )
        upload_response.raise_for_status()
        upload_data = upload_response.json()

        s3_prefix = upload_data['s3_prefix']
        print(f"  → Session ID: {upload_data['inference_session_id']}")
        print(f"  → S3 Prefix: {s3_prefix}")
        print(f"  → Uploaded: {upload_data['total_files']} files")

        for f in upload_data['uploaded_files']:
            print(f"     - {f['original_filename']} → {f['unique_filename']}")
    finally:
        for _, file_tuple in files:
            file_tuple[1].close()

    print()

    # Step 3: Create inference job
    print("[Step 3] Creating inference job...")

    # Determine checkpoint path (exactly like frontend)
    if use_pretrained:
        final_checkpoint = 'pretrained'
    else:
        final_checkpoint = checkpoint_path or 'pretrained'

    create_payload = {
        'training_job_id': training_job_id,
        'checkpoint_path': final_checkpoint,
        'inference_type': 'batch',
        'input_data': {
            'image_paths_s3': s3_prefix,
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
        }
    }

    print(f"  → Checkpoint: {final_checkpoint}")

    create_response = requests.post(
        f"{api_url}/test_inference/inference/jobs",
        json=create_payload
    )
    create_response.raise_for_status()
    job_data = create_response.json()

    inference_job_id = job_data['id']
    print(f"  → InferenceJob ID: {inference_job_id}")
    print(f"  → Status: {job_data['status']}")

    print()

    # Step 4: Wait for completion
    print(f"[Step 4] Waiting for completion (timeout: {timeout}s)...")

    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout:
        status_response = requests.get(
            f"{api_url}/test_inference/inference/jobs/detail/{inference_job_id}"
        )
        status_response.raise_for_status()
        status_data = status_response.json()

        current_status = status_data['status']

        if current_status != last_status:
            elapsed = time.time() - start_time
            print(f"  → [{elapsed:.1f}s] Status: {current_status}")
            last_status = current_status

        if current_status == 'completed':
            print(f"  → ✓ Completed in {time.time() - start_time:.1f}s")
            print(f"  → Total images: {status_data.get('total_images', 0)}")
            print(f"  → Total time: {status_data.get('total_inference_time_ms', 0):.1f}ms")
            print(f"  → Avg time: {status_data.get('avg_inference_time_ms', 0):.1f}ms/image")
            break
        elif current_status == 'failed':
            print(f"  → ✗ Failed: {status_data.get('error_message', 'Unknown error')}")
            return False

        time.sleep(1)
    else:
        print(f"  → ✗ Timeout after {timeout}s")
        return False

    print()

    # Step 5: Fetch results
    print("[Step 5] Fetching inference results...")

    results_response = requests.get(
        f"{api_url}/test_inference/inference/jobs/{inference_job_id}/results"
    )
    results_response.raise_for_status()
    results_data = results_response.json()

    print(f"  → Total results: {results_data['total_count']}")
    print()

    # Display per-image results
    print("=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)

    for i, result in enumerate(results_data['results']):
        print(f"\n[Image {i+1}] {result['image_name']}")
        print(f"  Inference time: {result.get('inference_time_ms', 0):.1f}ms")

        # Check predictions
        predictions = result.get('predictions') or []
        boxes = result.get('predicted_boxes') or []

        if predictions:
            print(f"  Predictions: {len(predictions)} detections")
            for j, pred in enumerate(predictions[:10]):  # Show first 10
                cls_name = pred.get('class_name', 'unknown')
                conf = pred.get('confidence', 0)
                bbox = pred.get('bbox', [])
                print(f"    [{j+1}] {cls_name}: {conf:.2%}")
                if bbox:
                    print(f"        bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        elif boxes:
            print(f"  Detections: {len(boxes)} boxes")
            for j, box in enumerate(boxes[:10]):
                label = box.get('label', 'unknown')
                conf = box.get('confidence', 0)
                print(f"    [{j+1}] {label}: {conf:.2%}")
        else:
            print("  Predictions: None")

    print()
    print("=" * 80)
    print("✓ E2E Test PASSED")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(description='E2E Inference Test')
    parser.add_argument('--api-url', default='http://localhost:8000/api/v1',
                        help='API base URL')
    parser.add_argument('--job-id', type=int, required=True,
                        help='Training job ID')
    parser.add_argument('--images', nargs='+', required=True,
                        help='Image paths (supports glob patterns)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights instead of trained checkpoint')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout in seconds')

    args = parser.parse_args()

    # Expand glob patterns
    image_paths = []
    for pattern in args.images:
        expanded = glob.glob(pattern)
        if expanded:
            image_paths.extend(expanded)
        else:
            image_paths.append(pattern)

    # Verify images exist
    for path in image_paths:
        if not Path(path).exists():
            print(f"Error: Image not found: {path}")
            sys.exit(1)

    # Run test
    success = test_inference(
        api_url=args.api_url,
        training_job_id=args.job_id,
        image_paths=image_paths,
        use_pretrained=args.pretrained,
        timeout=args.timeout
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
