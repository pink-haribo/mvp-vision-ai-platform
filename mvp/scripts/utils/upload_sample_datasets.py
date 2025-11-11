"""
Upload sample datasets to R2 with DB integration.

Usage:
    python scripts/upload_sample_datasets.py

This script:
1. Analyzes datasets using DatasetAnalyzer
2. Generates meta.json with real data statistics
3. Uploads dataset + meta.json to R2
4. Creates Dataset records in database
"""

import os
import sys
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
import boto3
from dotenv import load_dotenv

# Load .env files BEFORE importing app modules
project_root = Path(__file__).parent.parent

# Load main .env
env_file = project_root / 'mvp' / 'backend' / '.env'
if env_file.exists():
    load_dotenv(env_file)
    print(f"[OK] Loaded environment from {env_file}")

# Load .env.r2 (overrides if exists)
env_r2 = project_root / '.env.r2'
if env_r2.exists():
    load_dotenv(env_r2, override=True)
    print(f"[OK] Loaded R2 credentials from {env_r2}")
else:
    print(f"[WARNING] .env.r2 not found, using environment variables")

# Add mvp/backend to path for DB access
backend_path = project_root / 'mvp' / 'backend'
sys.path.insert(0, str(backend_path))

# Change working directory to mvp/backend for correct SQLite paths
os.chdir(backend_path)

# Now import app modules (after env is loaded)
from app.db.database import SessionLocal, init_db
from app.db.models import Dataset
from app.utils.dataset_analyzer import DatasetAnalyzer


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


def analyze_dataset(dataset_path: Path) -> dict:
    """
    Analyze dataset using DatasetAnalyzer and return metadata.

    Returns real dataset structure, not hardcoded values.
    """
    print(f"\n[ANALYZE] Analyzing dataset: {dataset_path.name}")

    analyzer = DatasetAnalyzer(dataset_path)

    # Detect format
    detected_format = analyzer.detect_format()
    if detected_format['format'] == 'unknown':
        print(f"[ERROR] Unknown dataset format: {dataset_path}")
        return None

    print(f"[OK] Format: {detected_format['format']} (confidence: {detected_format['confidence']})")

    # Collect statistics
    stats = analyzer.collect_statistics(detected_format['format'])

    # Build metadata
    metadata = {
        "format": detected_format['format'],
        "task_type": detected_format.get('task_type', 'unknown'),
        "num_classes": stats.get('structure', {}).get('num_classes', 0),
        "num_images": stats.get('statistics', {}).get('total_images', 0),
        "class_names": stats.get('structure', {}).get('classes', []),
        "statistics": stats.get('statistics', {}),
        "analyzed_at": datetime.utcnow().isoformat(),
    }

    print(f"[OK] Found {metadata['num_images']} images, {metadata['num_classes']} classes")

    return metadata


def create_dataset_record(db, dataset_id: str, dataset_name: str, metadata: dict, storage_path: str):
    """Create Dataset record in database."""
    print(f"\n[DB] Creating Dataset record: {dataset_id}")

    # Check if already exists
    existing = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if existing:
        print(f"[INFO] Dataset already exists, updating...")
        existing.name = dataset_name
        existing.format = metadata['format']
        existing.task_type = metadata['task_type']
        existing.num_classes = metadata['num_classes']
        existing.num_images = metadata['num_images']
        existing.class_names = metadata.get('class_names', [])
        existing.storage_path = storage_path
        existing.updated_at = datetime.utcnow()
        db.commit()
        print(f"[OK] Updated Dataset record")
        return existing

    # Create new dataset
    dataset = Dataset(
        id=dataset_id,
        name=dataset_name,
        description=f"Platform sample dataset for {metadata['task_type']}",
        owner_id=None,  # Public dataset, no specific owner
        visibility='public',  # Accessible to everyone
        tags=['platform-sample', metadata['task_type'], metadata['format']],
        storage_path=storage_path,
        storage_type='r2',
        format=metadata['format'],
        task_type=metadata['task_type'],
        num_classes=metadata['num_classes'],
        num_images=metadata['num_images'],
        class_names=metadata.get('class_names', []),
        version=1,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(dataset)
    db.commit()
    print(f"[OK] Created Dataset record in database")

    return dataset


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


def upload_metadata_to_r2(metadata: dict, dataset_id: str) -> bool:
    """Upload metadata JSON to R2."""
    print(f"\n[R2] Uploading metadata for {dataset_id}...")

    # Get R2 credentials
    endpoint = os.getenv('AWS_S3_ENDPOINT_URL')
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not all([endpoint, access_key, secret_key]):
        print("[ERROR] R2 credentials not set!")
        return False

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    bucket = 'vision-platform-prod'
    key = f'datasets/{dataset_id}.meta.json'

    try:
        # Upload JSON metadata
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )
        print(f"[OK] Uploaded metadata to s3://{bucket}/{key}")
        return True

    except Exception as e:
        print(f"[ERROR] Metadata upload failed: {e}")
        return False


def main():
    """Upload all datasets from C:/datasets to R2 with DB integration."""
    print("=" * 80)
    print("Sample Dataset Uploader for R2 (with DB integration)")
    print("=" * 80)

    # Initialize database session
    db = SessionLocal()

    try:
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
                dataset_id = dataset_path.name

                print(f"\n{'=' * 80}")
                print(f"Processing: {dataset_id}")
                print(f"Path: {dataset_path}")
                print('=' * 80)

                try:
                    # 1. Analyze dataset
                    metadata = analyze_dataset(dataset_path)
                    if not metadata:
                        print(f"[ERROR] Failed to analyze dataset")
                        fail_count += 1
                        continue

                    # 2. Create zip file
                    zip_path = tmp_path / f"{dataset_id}.zip"
                    create_zip(dataset_path, zip_path)

                    # 3. Upload zip to R2
                    if not upload_to_r2(zip_path, dataset_id):
                        fail_count += 1
                        continue

                    # 4. Upload metadata to R2
                    if not upload_metadata_to_r2(metadata, dataset_id):
                        print("[WARNING] Metadata upload failed, but continuing...")

                    # 5. Create Dataset record in DB
                    storage_path = f"datasets/{dataset_id}/"
                    dataset_name = f"{dataset_id.replace('-', ' ').title()}"
                    create_dataset_record(db, dataset_id, dataset_name, metadata, storage_path)

                    success_count += 1

                except Exception as e:
                    print(f"[ERROR] Failed to process {dataset_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    fail_count += 1

        print("\n" + "=" * 80)
        print(f"Upload complete: {success_count} success, {fail_count} failed")
        print("=" * 80)

        if success_count > 0:
            print("\n[INFO] Sample datasets uploaded successfully!")
            print("[INFO] Datasets are now:")
            print("  - Stored in R2 (zip + meta.json)")
            print("  - Registered in database (public visibility)")
            print("  - Available via /api/v1/datasets?visibility=public&tags=platform-sample")

    finally:
        db.close()


if __name__ == "__main__":
    main()
