"""
Utility helpers for Training Service

- DualStorageClient: Transparent dual storage access (Datasets + Results)
- S3Client: Low-level S3 operations
- CallbackClient: Send callbacks to Backend
- Dataset helpers: Format conversion
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ============================================================================
# Dual Storage Client
# ============================================================================

class DualStorageClient:
    """
    Dual Storage abstraction for Training Services.

    Automatically routes operations to the correct storage:
    - Dataset downloads: External Storage (MinIO-Datasets)
    - Checkpoint uploads: Internal Storage (MinIO-Results)

    Model developers don't need to worry about which storage to use.
    Just call download_dataset() and upload_checkpoint().

    Environment Variables:
        # External Storage (Datasets)
        EXTERNAL_STORAGE_ENDPOINT (default: http://localhost:9000)
        EXTERNAL_STORAGE_ACCESS_KEY (default: minioadmin)
        EXTERNAL_STORAGE_SECRET_KEY (default: minioadmin)
        EXTERNAL_BUCKET_DATASETS (default: training-datasets)

        # Internal Storage (Results)
        INTERNAL_STORAGE_ENDPOINT (default: http://localhost:9002)
        INTERNAL_STORAGE_ACCESS_KEY (default: minioadmin)
        INTERNAL_STORAGE_SECRET_KEY (default: minioadmin)
        INTERNAL_BUCKET_CHECKPOINTS (default: training-checkpoints)

        # Legacy fallback (uses same storage for both)
        S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET
    """

    def __init__(self):
        # External Storage (for datasets)
        external_endpoint = os.getenv('EXTERNAL_STORAGE_ENDPOINT') or os.getenv('S3_ENDPOINT', 'http://localhost:9000')
        external_access_key = os.getenv('EXTERNAL_STORAGE_ACCESS_KEY') or os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
        external_secret_key = os.getenv('EXTERNAL_STORAGE_SECRET_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')
        external_bucket = os.getenv('EXTERNAL_BUCKET_DATASETS', 'training-datasets')

        # Internal Storage (for checkpoints)
        internal_endpoint = os.getenv('INTERNAL_STORAGE_ENDPOINT', 'http://localhost:9002')
        internal_access_key = os.getenv('INTERNAL_STORAGE_ACCESS_KEY') or os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
        internal_secret_key = os.getenv('INTERNAL_STORAGE_SECRET_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')
        internal_bucket = os.getenv('INTERNAL_BUCKET_CHECKPOINTS', 'training-checkpoints')

        # Initialize S3 clients
        self.external_client = S3Client(
            endpoint=external_endpoint,
            access_key=external_access_key,
            secret_key=external_secret_key,
            bucket=external_bucket
        )

        self.internal_client = S3Client(
            endpoint=internal_endpoint,
            access_key=internal_access_key,
            secret_key=internal_secret_key,
            bucket=internal_bucket
        )

        logger.info(f"Dual Storage initialized:")
        logger.info(f"  External (Datasets): {external_endpoint} -> {external_bucket}")
        logger.info(f"  Internal (Results):  {internal_endpoint} -> {internal_bucket}")

    def download_dataset(self, dataset_id: str, dest_dir: Path) -> None:
        """
        Download dataset from External Storage (MinIO-Datasets).

        Model developers just call this without worrying about storage routing.
        """
        logger.info(f"[Dual Storage] Downloading dataset from External Storage")
        self.external_client.download_dataset(dataset_id, dest_dir)

    def upload_checkpoint(self, local_path: Path, job_id: str, filename: str = "best.pt") -> str:
        """
        Upload checkpoint to Internal Storage (MinIO-Results).

        Model developers just call this without worrying about storage routing.
        """
        logger.info(f"[Dual Storage] Uploading checkpoint to Internal Storage")
        return self.internal_client.upload_checkpoint(local_path, job_id, filename)


# ============================================================================
# S3 Client (Low-level)
# ============================================================================

class S3Client:
    """S3-compatible storage client (MinIO/S3/R2)"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        self.bucket = bucket
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def download_dataset(self, dataset_id: str, dest_dir: Path) -> None:
        """Download dataset from S3 and extract"""
        import zipfile
        import tempfile

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Download zip
        zip_path = Path(tempfile.gettempdir()) / f"{dataset_id}.zip"
        logger.info(f"Downloading dataset {dataset_id} from s3://{self.bucket}/datasets/{dataset_id}/")

        # List and download all files in dataset prefix
        prefix = f"datasets/{dataset_id}/"
        objects = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)

        if 'Contents' not in objects:
            raise ValueError(f"Dataset {dataset_id} not found in S3")

        # Download each file
        for obj in objects['Contents']:
            key = obj['Key']
            # Remove prefix to get relative path
            rel_path = key[len(prefix):]
            if not rel_path:  # Skip directory markers
                continue

            dest_file = dest_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Downloading {key} -> {dest_file}")
            self.client.download_file(self.bucket, key, str(dest_file))

        logger.info(f"Dataset downloaded to {dest_dir}")

    def upload_checkpoint(self, local_path: Path, job_id: str, filename: str = "best.pt") -> str:
        """Upload checkpoint to S3"""
        s3_key = f"checkpoints/{job_id}/{filename}"
        logger.info(f"Uploading {local_path} to s3://{self.bucket}/{s3_key}")

        self.client.upload_file(str(local_path), self.bucket, s3_key)

        s3_uri = f"s3://{self.bucket}/{s3_key}"
        logger.info(f"Checkpoint uploaded to {s3_uri}")
        return s3_uri


# ============================================================================
# Callback Client
# ============================================================================

class CallbackClient:
    """HTTP callback client for Backend communication"""

    def __init__(self, base_url: str, retry_attempts: int = 3, retry_delay: int = 2):
        self.base_url = base_url
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def send_progress(self, job_id: str, data: Dict[str, Any]) -> None:
        """Send progress callback"""
        url = f"{self.base_url}/jobs/{job_id}/callback/progress"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=data)
            response.raise_for_status()
            logger.debug(f"Progress callback sent: epoch {data.get('current_epoch')}/{data.get('total_epochs')}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def send_completion(self, job_id: str, data: Dict[str, Any]) -> None:
        """Send completion callback"""
        url = f"{self.base_url}/jobs/{job_id}/callback/completion"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=data)
            response.raise_for_status()
            logger.info(f"Completion callback sent: {data.get('status')}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def send_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """Send validation result callback"""
        url = f"{self.base_url}/jobs/{job_id}/validation"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=data)
            response.raise_for_status()
            logger.debug(f"Validation callback sent: epoch {data.get('epoch')}")

    # ========================================================================
    # Synchronous versions for use in non-async contexts (Ultralytics callbacks)
    # ========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def send_progress_sync(self, job_id: str, data: Dict[str, Any]) -> None:
        """Send progress callback (synchronous version for Ultralytics callbacks)"""
        url = f"{self.base_url}/jobs/{job_id}/callback/progress"

        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            logger.debug(f"Progress callback sent: epoch {data.get('current_epoch')}/{data.get('total_epochs')}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def send_completion_sync(self, job_id: str, data: Dict[str, Any]) -> None:
        """Send completion callback (synchronous version)"""
        url = f"{self.base_url}/jobs/{job_id}/callback/completion"

        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            logger.info(f"Completion callback sent: {data.get('status')}")


# ============================================================================
# Dataset Utilities
# ============================================================================

def convert_diceformat_to_yolo(dataset_dir: Path, split_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Convert DICEFormat (annotations.json) to YOLO format.

    Creates:
    - labels/*.txt (YOLO format labels)
    - train.txt, val.txt (image lists)
    - data.yaml (dataset config)

    Args:
        dataset_dir: Dataset directory containing annotations.json
        split_config: Optional split configuration from Backend
    """
    annotations_file = dataset_dir / "annotations.json"
    if not annotations_file.exists():
        logger.info("No annotations.json found, assuming YOLO format already")
        return

    logger.info("Converting DICEFormat → YOLO format")

    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])

    # Create image_id -> annotations mapping
    image_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Create category_id -> index mapping
    category_map = {cat['id']: idx for idx, cat in enumerate(categories)}

    # Create labels directory
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Convert annotations to YOLO format
    for img in images:
        img_id = img['id']
        img_file = Path(img['file_name'])

        # YOLO label file
        label_file = labels_dir / f"{img_file.stem}.txt"

        anns = image_annotations.get(img_id, [])

        with open(label_file, 'w') as f:
            for ann in anns:
                # Convert COCO bbox to YOLO format
                x, y, w, h = ann['bbox']
                img_w = img['width']
                img_h = img['height']

                # Normalize to 0-1
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width = w / img_w
                height = h / img_h

                class_id = category_map[ann['category_id']]

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    logger.info(f"Created YOLO labels in {labels_dir}")

    # Create train.txt and val.txt
    if split_config and 'splits' in split_config:
        logger.info("Using provided split configuration")
        splits = split_config['splits']
    else:
        logger.info("Creating default 80/20 train/val split")
        import random
        random.seed(42)
        shuffled = images.copy()
        random.shuffle(shuffled)
        train_count = int(len(shuffled) * 0.8)
        splits = {}
        for i, img in enumerate(shuffled):
            splits[str(img['id'])] = 'train' if i < train_count else 'val'

    train_images = []
    val_images = []

    for img in images:
        img_path = f"./images/{Path(img['file_name']).name}"
        split = splits.get(str(img['id']), 'train')

        if split == 'train':
            train_images.append(img_path)
        else:
            val_images.append(img_path)

    # Write train.txt and val.txt
    with open(dataset_dir / "train.txt", 'w') as f:
        f.write('\n'.join(train_images))

    with open(dataset_dir / "val.txt", 'w') as f:
        f.write('\n'.join(val_images))

    logger.info(f"Created train.txt ({len(train_images)} images) and val.txt ({len(val_images)} images)")

    # Create data.yaml
    class_names = [cat['name'] for cat in categories]

    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'train.txt',
        'val': 'val.txt',
        'nc': len(class_names),
        'names': class_names
    }

    import yaml
    with open(dataset_dir / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info(f"Created data.yaml with {len(class_names)} classes")
