"""
Upload sample datasets to R2.

Usage:
    python scripts/upload_sample_datasets.py

This script downloads sample datasets and uploads them to R2
for use in Railway Training Services.
"""

import os
import sys
import shutil
import tempfile
import zipfile
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


def download_coco8_dataset() -> Path:
    """Download COCO8 dataset using Ultralytics."""
    print(f"\n[COCO8] Downloading COCO8 dataset...")

    from ultralytics import YOLO

    # Create temporary directory for training (this triggers dataset download)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # YOLO will auto-download coco8 dataset when train() is called
        model = YOLO('yolov8n.pt')

        # Trigger dataset download by starting (and immediately stopping) training
        try:
            model.train(
                data='coco8.yaml',
                epochs=1,
                imgsz=64,
                batch=1,
                project=tmp_dir,
                exist_ok=True,
                verbose=False
            )
        except:
            # Training might fail, but dataset should be downloaded
            pass

    # Find downloaded dataset
    cache_path = Path.home() / ".cache" / "ultralytics" / "datasets" / "coco8"

    if not cache_path.exists():
        # Try alternative location
        cache_path = Path.home() / "datasets" / "coco8"

    if cache_path.exists():
        print(f"[OK] COCO8 downloaded to {cache_path}")
        return cache_path
    else:
        raise FileNotFoundError(f"COCO8 dataset not found after download")


def create_zip(dataset_path: Path, output_zip: Path):
    """Create zip file from dataset directory."""
    print(f"\n[ZIP] Creating zip file: {output_zip}")
    print(f"[ZIP] Source: {dataset_path}")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file():
                # Add file with relative path
                arcname = file_path.relative_to(dataset_path.parent)
                zipf.write(file_path, arcname)

    file_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[OK] Zip created: {output_zip} ({file_size_mb:.2f} MB)")


def upload_to_r2(local_zip: Path, dataset_id: str):
    """Upload dataset zip to R2."""
    print(f"\n[R2] Uploading {dataset_id} to R2...")

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
    key = f'datasets/{dataset_id}.zip'

    try:
        # Check if already exists
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"[WARNING] Dataset already exists in R2: s3://{bucket}/{key}")
            print(f"[INFO] Overwriting existing dataset...")
        except:
            # Object doesn't exist, proceed with upload
            pass

        print(f"[R2] Uploading to s3://{bucket}/{key}...")
        s3.upload_file(str(local_zip), bucket, key)

        file_size_mb = local_zip.stat().st_size / (1024 * 1024)
        print(f"[OK] Uploaded {file_size_mb:.2f} MB to s3://{bucket}/{key}")
        return True

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return False


def scan_datasets_directory(base_path: Path):
    """
    Scan base path for dataset directories.

    Returns:
        List of dataset paths (subdirectories)
    """
    if not base_path.exists():
        print(f"[ERROR] Directory not found: {base_path}")
        return []

    datasets = []
    for item in base_path.iterdir():
        if item.is_dir():
            # Skip hidden directories and __pycache__
            if not item.name.startswith('.') and item.name != '__pycache__':
                datasets.append(item)

    return datasets


def main():
    """Upload all datasets from C:/datasets to R2."""
    print("=" * 80)
    print("Sample Dataset Uploader for R2")
    print("=" * 80)

    # Scan C:/datasets directory
    base_path = Path("C:/datasets")
    datasets = scan_datasets_directory(base_path)

    if not datasets:
        print(f"\n[ERROR] No datasets found in {base_path}")
        print("Please ensure datasets exist in C:/datasets directory")
        return

    print(f"\n[INFO] Found {len(datasets)} datasets in {base_path}:")
    for dataset_path in datasets:
        print(f"  - {dataset_path.name}")

    print(f"\n[INFO] Starting upload of {len(datasets)} datasets to R2...")

    success_count = 0
    fail_count = 0

    # Create temporary directory for zip files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for dataset_path in datasets:
            dataset_name = dataset_path.name

            print(f"\n{'=' * 80}")
            print(f"Processing: {dataset_name}")
            print(f"Path: {dataset_path}")
            print('=' * 80)

            try:
                # Create zip file
                zip_path = tmp_path / f"{dataset_name}.zip"
                create_zip(dataset_path, zip_path)

                # Upload to R2
                if upload_to_r2(zip_path, dataset_name):
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                print(f"[ERROR] Failed to process {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1

    print("\n" + "=" * 80)
    print(f"Upload complete: {success_count} success, {fail_count} failed")
    print("=" * 80)

    if success_count > 0:
        print("\n[INFO] Sample datasets uploaded successfully!")
        print("[INFO] Railway Training Services can now download these datasets automatically")
        print(f"\nUploaded datasets ({success_count}):")
        print("These datasets are now available via get_dataset() in platform_sdk")


if __name__ == "__main__":
    main()
