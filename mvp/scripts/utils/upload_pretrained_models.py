"""
Upload pretrained model weights to R2.

Usage:
    python scripts/upload_pretrained_models.py

This script downloads common pretrained models and uploads them to R2
for use in Railway Training Services.
"""

import os
import sys
from pathlib import Path
import boto3
from dotenv import load_dotenv

# Load .env.r2 if exists
env_r2 = Path(__file__).parent.parent / '.env.r2'
if env_r2.exists():
    load_dotenv(env_r2)
    print(f"[OK] Loaded R2 credentials from {env_r2}")
else:
    print(f"[WARNING] .env.r2 not found, using environment variables")


def download_yolo_model(model_name: str) -> Path:
    """Download YOLO model using ultralytics."""
    print(f"\n[YOLO] Downloading {model_name}...")

    from ultralytics import YOLO

    # This will auto-download to ~/.cache/ultralytics/
    model = YOLO(f"{model_name}.pt")

    # Find downloaded file
    cache_path = Path.home() / ".cache" / "ultralytics" / f"{model_name}.pt"

    if not cache_path.exists():
        # Try alternative location
        cache_path = Path.home() / ".config" / "Ultralytics" / f"{model_name}.pt"

    if cache_path.exists():
        print(f"[OK] Downloaded to {cache_path}")
        return cache_path
    else:
        raise FileNotFoundError(f"Model downloaded but not found: {cache_path}")


def upload_to_r2(local_path: Path, framework: str, model_name: str):
    """Upload model to R2."""
    print(f"\n[R2] Uploading {model_name} to R2...")

    # Get R2 credentials
    endpoint = os.getenv('AWS_S3_ENDPOINT_URL')
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not all([endpoint, access_key, secret_key]):
        print("[ERROR] R2 credentials not set!")
        print("Required environment variables:")
        print("  - AWS_S3_ENDPOINT_URL")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        return False

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    bucket = 'vision-platform-prod'
    key = f'models/pretrained/{framework}/{model_name}.pt'

    try:
        s3.upload_file(str(local_path), bucket, key)
        print(f"[OK] Uploaded to s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return False


def main():
    """Upload all common pretrained models."""
    print("="*80)
    print("Pretrained Model Uploader for R2")
    print("="*80)

    # YOLO models to upload
    yolo_models = [
        "yolo11n",
        "yolo11s",
        "yolo11m",
        "yolov8n",
        "yolov8s",
    ]

    success_count = 0
    fail_count = 0

    for model_name in yolo_models:
        try:
            # Download
            local_path = download_yolo_model(model_name)

            # Upload to R2
            if upload_to_r2(local_path, "ultralytics", model_name):
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {model_name}: {e}")
            fail_count += 1

    print("\n" + "="*80)
    print(f"Upload complete: {success_count} success, {fail_count} failed")
    print("="*80)

    if success_count > 0:
        print("\n[INFO] Models uploaded successfully!")
        print("[INFO] Railway Training Services will now use these models from R2")


if __name__ == "__main__":
    main()
