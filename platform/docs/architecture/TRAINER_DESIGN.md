# Trainer Service Design

Complete design for framework-specific training services.

## Table of Contents

- [Overview](#overview)
- [API Contract](#api-contract)
- [Directory Structure](#directory-structure)
- [Trainer Implementation](#trainer-implementation)
- [HTTP Callbacks](#http-callbacks)
- [Storage Integration](#storage-integration)
- [Format Conversion Layer](#format-conversion-layer)
- [Split Handling](#split-handling)
- [Error Handling](#error-handling)
- [Framework-Specific Guides](#framework-specific-guides)

## Overview

Trainers are **isolated, stateless services** that execute model training. Each framework (Ultralytics, timm, HuggingFace) has its own trainer implementation.

**Key Principles**:
1. **Complete Isolation**: No shared code with backend
2. **Environment Variable Configuration**: All config via env vars
3. **HTTP-Only Communication**: Callbacks to backend API
4. **S3 Storage**: All data exchange via object storage
5. **No Sidecar**: Trainer handles all responsibilities (progress, checkpoints, callbacks)

**Execution Modes**:
- **Tier 1 (Subprocess)**: Backend spawns `python train.py` as subprocess
- **Tier 2/3 (Kubernetes)**: K8s Job creates pod with trainer container

## API Contract

All trainers MUST implement this contract for interoperability.

### Input: Environment Variables

```bash
# Job Identifiers
JOB_ID=550e8400-e29b-41d4-a716-446655440000
TRACE_ID=abc-def-ghi-jkl  # For distributed tracing

# Backend Communication
BACKEND_BASE_URL=https://api.example.com
CALLBACK_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Training Configuration
TASK_TYPE=object_detection  # or image_classification, instance_segmentation, etc.
MODEL_NAME=yolo11n
DATASET_ID=dataset-uuid
SNAPSHOT_ID=training-job-abc123  # Dataset snapshot for reproducibility
EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=0.001
IMAGE_SIZE=640

# Split Strategy (train/val split)
SPLIT_STRATEGY='{"method": "auto", "ratio": [0.8, 0.2], "seed": 42}'
# Or: '{"method": "use_dataset"}' to use dataset's split_config
# Or: '{"method": "custom", "custom_splits": {...}}'

# Storage (S3-compatible)
STORAGE_TYPE=r2  # or s3, minio
R2_ENDPOINT=https://xxx.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=xxx
R2_SECRET_ACCESS_KEY=xxx
BUCKET_NAME=vision-platform

# Optional
PRETRAINED_WEIGHT_URL=https://...  # Presigned URL for pretrained weight (if using pretrained)
DEVICE=cuda  # or cpu
NUM_WORKERS=4
CHECKPOINT_FREQUENCY=5  # Save every N epochs
```

### Output: HTTP Callbacks

**1. Heartbeat (per epoch)**
```http
POST {BACKEND_BASE_URL}/api/v1/jobs/{JOB_ID}/heartbeat
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}
Content-Type: application/json

{
  "epoch": 5,
  "progress": 10.0,
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.892,
    "learning_rate": 0.001
  },
  "timestamp": "2025-01-10T12:34:56Z"
}
```

**2. Event Notification**
```http
POST {BACKEND_BASE_URL}/api/v1/jobs/{JOB_ID}/event
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}
Content-Type: application/json

{
  "event_type": "checkpoint_saved",
  "message": "Checkpoint saved at epoch 10",
  "data": {
    "checkpoint_path": "s3://bucket/checkpoints/job-{JOB_ID}/epoch-10.pt",
    "epoch": 10,
    "metrics": {...}
  },
  "timestamp": "2025-01-10T12:35:00Z"
}
```

**3. Completion**
```http
POST {BACKEND_BASE_URL}/api/v1/jobs/{JOB_ID}/done
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}
Content-Type: application/json

{
  "status": "succeeded",  # or "failed"
  "final_metrics": {
    "best_epoch": 42,
    "best_loss": 0.123,
    "best_accuracy": 0.956,
    "total_time_seconds": 3600
  },
  "checkpoint_path": "s3://bucket/checkpoints/job-{JOB_ID}/best.pt",
  "error_message": null,
  "timestamp": "2025-01-10T13:00:00Z"
}
```

**4. Validation Result**
```http
POST {BACKEND_BASE_URL}/api/v1/jobs/{JOB_ID}/validation
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}
Content-Type: application/json

{
  "epoch": 10,
  "task_type": "object_detection",

  // Primary metric (from job configuration)
  "primary_metric_name": "mAP50-95",
  "primary_metric_value": 0.6523,

  // All metrics (standard + custom)
  "metrics": {
    "mAP50": 0.8721,
    "mAP50-95": 0.6523,
    "precision": 0.8912,
    "recall": 0.8234,
    "loss": 0.0345,

    // Custom metrics
    "inference_speed_fps": 45.2,
    "model_size_mb": 12.3
  },

  // Per-class metrics (optional)
  "per_class_metrics": [
    {
      "class_id": 0,
      "class_name": "cat",
      "precision": 0.90,
      "recall": 0.85,
      "ap": 0.88,
      "support": 120
    },
    {
      "class_id": 1,
      "class_name": "dog",
      "precision": 0.88,
      "recall": 0.79,
      "ap": 0.86,
      "support": 95
    }
  ],

  // Visualization data (optional)
  "confusion_matrix": [[110, 10], [4, 101]],  // Classification only
  "pr_curves": {  // Detection, Segmentation
    "cat": {
      "precision": [0.9, 0.88, 0.85, ...],
      "recall": [0.1, 0.2, 0.3, ...]
    },
    "dog": {
      "precision": [0.87, 0.85, 0.82, ...],
      "recall": [0.1, 0.2, 0.3, ...]
    }
  },

  // Per-image results (uploaded to S3 separately)
  "image_results_path": "s3://bucket/validation-results/job-abc123/epoch-10/images.json",
  "num_images_validated": 500,

  // Metadata
  "validation_time_seconds": 45.2,
  "timestamp": "2025-01-10T12:36:00Z"
}
```

**Response:**
```json
{
  "status": "ok",
  "is_best": true  // True if this is the new best epoch
}
```

### Storage: S3 Operations (All Tiers)

**Critical**: All tiers use S3-compatible API for complete code consistency.

**Tier-Specific Endpoints**:
- **Tier 1**: MinIO on `http://localhost:9000` (Docker Compose)
- **Tier 2**: MinIO on `http://minio.platform.svc.cluster.local:9000` (Kind)
- **Tier 3**: Cloudflare R2 or AWS S3 (Production)

**Download Dataset** (identical code across all tiers):
```python
import boto3
import os

# S3 client - endpoint changes per tier, code stays same
s3 = boto3.client(
    's3',
    endpoint_url=os.environ.get("S3_ENDPOINT"),  # Tier-specific
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
)

# Download dataset
s3.download_file(
    Bucket=os.environ.get("BUCKET_NAME", "vision-platform"),
    Key=f"datasets/{DATASET_ID}.zip",
    Filename="/workspace/dataset.zip"
)

# Unzip
import zipfile
with zipfile.ZipFile("/workspace/dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("/workspace/dataset")
```

**Upload Checkpoint** (identical code across all tiers):
```python
# Upload checkpoint - same code in all tiers
s3.upload_file(
    Filename="/workspace/runs/train/weights/best.pt",
    Bucket=os.environ.get("BUCKET_NAME", "vision-platform"),
    Key=f"checkpoints/job-{JOB_ID}/best.pt"
)
```

**Environment Variables Per Tier**:

```bash
# Tier 1 (MinIO via Docker Compose)
S3_ENDPOINT=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
BUCKET_NAME=vision-platform

# Tier 2 (MinIO in Kind)
S3_ENDPOINT=http://minio.platform.svc.cluster.local:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
BUCKET_NAME=vision-platform

# Tier 3 (Cloudflare R2)
S3_ENDPOINT=https://xxx.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=<r2-access-key>
AWS_SECRET_ACCESS_KEY=<r2-secret-key>
BUCKET_NAME=vision-platform-prod
```

**Why S3 API for All Tiers?**

1. ✅ **Complete Code Consistency**: Same boto3 code works everywhere
2. ✅ **Catch Bugs Early**: S3 permission issues caught in Tier 1
3. ✅ **Isolation Compliance**: No direct file system access
4. ✅ **Production Parity**: Local development uses same patterns as production

## Directory Structure

```
platform/trainers/
├── ultralytics/              # YOLO models
│   ├── train.py              # Main training script
│   ├── utils.py              # Helper functions (callbacks, storage)
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile            # Container image
│   └── README.md
│
├── timm/                     # PyTorch Image Models
│   ├── train.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── huggingface/              # Transformers (CLIP, DINO, etc.)
│   ├── train.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
└── base/                     # Shared utilities (optional)
    ├── callback_client.py    # HTTP callback helper
    ├── storage_client.py     # S3 helper
    └── metrics.py            # Common metric calculations
```

## Trainer Implementation

### Complete Example: Ultralytics Trainer

```python
# platform/trainers/ultralytics/train.py
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import requests
from datetime import datetime

# Import shared utilities
from utils import CallbackClient, StorageClient

def main():
    print("=== Ultralytics Trainer Starting ===")

    # 1. Read configuration from environment variables
    job_id = os.environ["JOB_ID"]
    trace_id = os.environ["TRACE_ID"]
    backend_url = os.environ["BACKEND_BASE_URL"]
    callback_token = os.environ["CALLBACK_TOKEN"]

    task_type = os.environ["TASK_TYPE"]
    model_name = os.environ["MODEL_NAME"]
    dataset_id = os.environ["DATASET_ID"]
    epochs = int(os.environ["EPOCHS"])
    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    imgsz = int(os.environ.get("IMAGE_SIZE", 640))
    pretrained_weight_url = os.environ.get("PRETRAINED_WEIGHT_URL")

    print(f"Job ID: {job_id}")
    print(f"Trace ID: {trace_id}")
    print(f"Model: {model_name}")
    print(f"Task: {task_type}")
    print(f"Epochs: {epochs}")

    # 2. Initialize clients
    callback_client = CallbackClient(
        backend_url=backend_url,
        job_id=job_id,
        trace_id=trace_id,
        callback_token=callback_token
    )

    storage_client = StorageClient()

    try:
        # 3. Download dataset from S3
        print("\n=== Downloading Dataset ===")
        dataset_path = storage_client.download_dataset(dataset_id)
        print(f"Dataset downloaded to: {dataset_path}")

        # Send event
        callback_client.send_event(
            event_type="dataset_downloaded",
            message="Dataset downloaded successfully"
        )

        # 4. Download pretrained weight (if provided)
        pretrained_weight_path = None
        if pretrained_weight_url:
            print("\n=== Downloading Pretrained Weight ===")
            pretrained_weight_path = storage_client.download_pretrained_weight(
                presigned_url=pretrained_weight_url,
                model_name=model_name
            )
            print(f"Pretrained weight downloaded to: {pretrained_weight_path}")

        # 5. Load model
        print("\n=== Loading Model ===")
        if pretrained_weight_path:
            model = YOLO(pretrained_weight_path)
            print(f"Model loaded from pretrained weight: {model_name}")
        else:
            model = YOLO(f"{model_name}.yaml")
            print(f"Model initialized from scratch: {model_name}")

        # 5. Setup custom callbacks for progress reporting
        def on_train_epoch_end(trainer):
            """Called after each epoch"""
            epoch = trainer.epoch + 1
            metrics = trainer.metrics

            # Calculate progress
            progress = (epoch / epochs) * 100

            # Send heartbeat to backend
            callback_client.send_heartbeat(
                epoch=epoch,
                progress=progress,
                metrics={
                    "loss": float(metrics.get("train/loss", 0)),
                    "box_loss": float(metrics.get("train/box_loss", 0)),
                    "cls_loss": float(metrics.get("train/cls_loss", 0)),
                    "dfl_loss": float(metrics.get("train/dfl_loss", 0)),
                    "precision": float(metrics.get("metrics/precision(B)", 0)),
                    "recall": float(metrics.get("metrics/recall(B)", 0)),
                    "mAP50": float(metrics.get("metrics/mAP50(B)", 0)),
                    "mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0)),
                }
            )

            print(f"Epoch {epoch}/{epochs} - Progress: {progress:.1f}% - Loss: {metrics.get('train/loss', 0):.4f}")

        def on_model_save(trainer):
            """Called when model checkpoint is saved"""
            epoch = trainer.epoch + 1

            # Upload checkpoint to S3
            checkpoint_local_path = trainer.best  # Path to best.pt
            checkpoint_s3_path = storage_client.upload_checkpoint(
                local_path=checkpoint_local_path,
                job_id=job_id,
                filename=f"epoch-{epoch}.pt"
            )

            # Send event
            callback_client.send_event(
                event_type="checkpoint_saved",
                message=f"Checkpoint saved at epoch {epoch}",
                data={
                    "checkpoint_path": checkpoint_s3_path,
                    "epoch": epoch,
                    "is_best": True
                }
            )

            print(f"Checkpoint uploaded: {checkpoint_s3_path}")

        # Add callbacks to model
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_model_save", on_model_save)

        # 6. Start training
        print("\n=== Starting Training ===")
        results = model.train(
            data=f"{dataset_path}/data.yaml",
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=os.environ.get("DEVICE", "0"),  # GPU 0 or CPU
            workers=int(os.environ.get("NUM_WORKERS", 4)),
            project="/workspace/runs",
            name="train",
            exist_ok=True,
            verbose=True
        )

        # 7. Upload final checkpoint
        print("\n=== Training Complete ===")
        best_checkpoint_path = model.trainer.best
        final_s3_path = storage_client.upload_checkpoint(
            local_path=best_checkpoint_path,
            job_id=job_id,
            filename="best.pt"
        )

        # 8. Send completion callback
        final_metrics = {
            "best_epoch": int(model.trainer.best_epoch),
            "mAP50": float(model.trainer.best_fitness),
            "total_time_seconds": model.trainer.epoch_time * epochs
        }

        callback_client.send_completion(
            status="succeeded",
            final_metrics=final_metrics,
            checkpoint_path=final_s3_path
        )

        print(f"Training succeeded! Best checkpoint: {final_s3_path}")
        sys.exit(0)

    except Exception as e:
        print(f"\n=== Training Failed ===")
        print(f"Error: {str(e)}")

        # Send failure callback
        callback_client.send_completion(
            status="failed",
            error_message=str(e)
        )

        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Shared Utilities

```python
# platform/trainers/ultralytics/utils.py
import os
import requests
import boto3
from pathlib import Path
from datetime import datetime
import zipfile

class CallbackClient:
    """HTTP client for sending callbacks to backend"""

    def __init__(self, backend_url: str, job_id: str, trace_id: str, callback_token: str):
        self.backend_url = backend_url.rstrip('/')
        self.job_id = job_id
        self.trace_id = trace_id
        self.headers = {
            "Authorization": f"Bearer {callback_token}",
            "X-Trace-ID": trace_id,
            "Content-Type": "application/json"
        }

    def send_heartbeat(self, epoch: int, progress: float, metrics: dict):
        """Send training progress update"""
        url = f"{self.backend_url}/api/v1/jobs/{self.job_id}/heartbeat"
        payload = {
            "epoch": epoch,
            "progress": progress,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to send heartbeat: {e}")
            # Don't raise - continue training even if callback fails

    def send_event(self, event_type: str, message: str, data: dict = None):
        """Send training event"""
        url = f"{self.backend_url}/api/v1/jobs/{self.job_id}/event"
        payload = {
            "event_type": event_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to send event: {e}")

    def send_completion(self, status: str, final_metrics: dict = None, checkpoint_path: str = None, error_message: str = None):
        """Send training completion"""
        url = f"{self.backend_url}/api/v1/jobs/{self.job_id}/done"
        payload = {
            "status": status,
            "final_metrics": final_metrics or {},
            "checkpoint_path": checkpoint_path,
            "error_message": error_message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Error: Failed to send completion: {e}")
            raise

    def send_validation(
        self,
        epoch: int,
        task_type: str,
        primary_metric_name: str,
        primary_metric_value: float,
        metrics: dict,
        per_class_metrics: list = None,
        confusion_matrix: list = None,
        pr_curves: dict = None,
        image_results_path: str = None,
        num_images_validated: int = 0,
        validation_time_seconds: float = None
    ):
        """
        Send validation result

        Called after validation run (usually per epoch)
        Backend automatically determines if this is the best checkpoint
        """
        url = f"{self.backend_url}/api/v1/jobs/{self.job_id}/validation"
        payload = {
            "epoch": epoch,
            "task_type": task_type,
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value,
            "metrics": metrics,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": confusion_matrix,
            "pr_curves": pr_curves,
            "image_results_path": image_results_path,
            "num_images_validated": num_images_validated,
            "validation_time_seconds": validation_time_seconds,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()

            result = response.json()
            if result.get("is_best"):
                print(f"[Validation] New best epoch: {epoch} - {primary_metric_name}={primary_metric_value:.4f}")
            else:
                print(f"[Validation] Epoch {epoch} - {primary_metric_name}={primary_metric_value:.4f}")

            return result
        except Exception as e:
            print(f"Warning: Failed to send validation result: {e}")
            # Don't raise - continue training even if validation callback fails


class StorageClient:
    """
    S3-compatible storage client for datasets and checkpoints.

    Works identically across all tiers:
    - Tier 1: MinIO (Docker Compose) on localhost:9000
    - Tier 2: MinIO (Kind) on minio.platform.svc:9000
    - Tier 3: Cloudflare R2 or AWS S3

    Only the S3_ENDPOINT environment variable changes.
    """

    def __init__(self):
        # Always use S3 API - no local file system mode
        endpoint = os.environ.get("S3_ENDPOINT")
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if not endpoint or not access_key or not secret_key:
            raise ValueError(
                "Missing required S3 credentials. "
                "Set S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
            )

        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        self.bucket = os.environ.get("BUCKET_NAME", "vision-platform")

    def download_dataset(self, dataset_id: str) -> str:
        """
        Download and extract dataset from S3.

        Works the same in all tiers - only endpoint differs.
        """
        download_path = "/workspace/dataset.zip"
        extract_path = "/workspace/dataset"

        print(f"Downloading dataset from s3://{self.bucket}/datasets/{dataset_id}.zip")

        self.s3_client.download_file(
            self.bucket,
            f"datasets/{dataset_id}.zip",
            download_path
        )

        print(f"Extracting dataset to {extract_path}")

        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        return extract_path

    def upload_checkpoint(self, local_path: str, job_id: str, filename: str) -> str:
        """
        Upload checkpoint to S3.

        Works the same in all tiers - only endpoint differs.
        """
        s3_key = f"checkpoints/job-{job_id}/{filename}"

        print(f"Uploading checkpoint to s3://{self.bucket}/{s3_key}")

        self.s3_client.upload_file(
            local_path,
            self.bucket,
            s3_key
        )

        return f"s3://{self.bucket}/{s3_key}"

    def download_pretrained_weight(self, presigned_url: str, model_name: str) -> str:
        """
        Download pretrained weight using presigned URL.

        Args:
            presigned_url: Presigned URL from backend API
            model_name: Model name for local filename

        Returns:
            Path to downloaded weight file
        """
        import requests

        local_path = f"/workspace/pretrained/{model_name}.pt"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading pretrained weight from presigned URL...")

        response = requests.get(presigned_url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Pretrained weight downloaded: {local_path}")

        return local_path
```

## HTTP Callbacks

### Callback Lifecycle

```
Training Start
    ↓
[Event: dataset_downloaded]
    ↓
Epoch 1
    ↓
[Heartbeat: epoch=1, progress=2%, metrics={...}]
    ↓
Epoch 2
    ↓
[Heartbeat: epoch=2, progress=4%, metrics={...}]
    ↓
...
    ↓
Epoch 5 (checkpoint)
    ↓
[Event: checkpoint_saved, checkpoint_path=s3://...]
    ↓
...
    ↓
Epoch 50
    ↓
[Heartbeat: epoch=50, progress=100%, metrics={...}]
    ↓
[Event: checkpoint_saved, checkpoint_path=s3://...]
    ↓
[Completion: status=succeeded, final_metrics={...}]
```

### Callback Error Handling

**Principle**: Training continues even if callbacks fail

```python
def send_heartbeat_safe(epoch, metrics):
    try:
        callback_client.send_heartbeat(epoch, progress, metrics)
    except Exception as e:
        # Log error but don't stop training
        print(f"Warning: Callback failed: {e}")
        # Continue training
```

**Exception**: Completion callback must succeed

```python
try:
    callback_client.send_completion(
        status="succeeded",
        final_metrics=final_metrics,
        checkpoint_path=checkpoint_path
    )
except Exception as e:
    print(f"CRITICAL: Failed to send completion: {e}")
    # Retry a few times
    for retry in range(3):
        try:
            callback_client.send_completion(...)
            break
        except:
            time.sleep(5)
    else:
        # Still failed - this is a problem
        sys.exit(1)
```

## Storage Integration

### Dataset Format Examples

**YOLO Format** (`data.yaml`):
```yaml
path: /workspace/dataset
train: images/train
val: images/val

nc: 2  # number of classes
names: ['cat', 'dog']
```

**ImageFolder Format**:
```
dataset/
├── train/
│   ├── cat/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── dog/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    ├── cat/
    └── dog/
```

### S3 Path Structure

```
vision-platform/  (bucket)
├── datasets/
│   ├── {dataset-id-1}.zip
│   ├── {dataset-id-2}.zip
│   └── ...
│
├── checkpoints/
│   ├── job-{job-id-1}/
│   │   ├── epoch-5.pt
│   │   ├── epoch-10.pt
│   │   └── best.pt
│   ├── job-{job-id-2}/
│   │   └── best.pt
│   └── ...
│
└── logs/
    ├── job-{job-id-1}/
    │   └── training.log
    └── ...
```

## Format Conversion Layer

Each trainer must convert the Platform Format to its framework-specific format.

### Platform Format (Input)

All datasets are stored in Platform Format (`annotations.json`):

```json
{
  "format_version": "1.0",
  "dataset_id": "dataset-abc123",
  "classes": [
    {"id": 0, "name": "cat"},
    {"id": 1, "name": "dog"}
  ],
  "images": [
    {
      "id": 1,
      "file_name": "cats/cat001.jpg",
      "width": 1920,
      "height": 1080,
      "annotation": {
        "class_id": 0,
        "class_name": "cat",
        "confidence": 1.0
      }
    }
  ]
}
```

### Converter Interface

```python
# platform/trainers/common/converter.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

class FormatConverter(ABC):
    """Base class for format converters"""

    @abstractmethod
    def convert(
        self,
        platform_annotations: Dict[str, Any],
        output_dir: Path,
        split_info: Dict[str, Any]
    ) -> Path:
        """
        Convert Platform Format to framework-specific format.

        Args:
            platform_annotations: Loaded annotations.json
            output_dir: Where to write converted data
            split_info: Split strategy from environment variable

        Returns:
            Path to framework-specific config file
        """
        pass
```

### Example: YOLO Converter

```python
# platform/trainers/ultralytics/converters/platform_to_yolo.py
import json
from pathlib import Path
import yaml

class PlatformToYOLOConverter:
    """Convert Platform Format to YOLO format"""

    def convert(self, platform_annotations, output_dir, split_info):
        """
        Convert to YOLO format with text file split.

        Directory structure:
          output_dir/
            ├── data.yaml
            ├── train.txt (paths to training images)
            ├── val.txt (paths to validation images)
            └── labels/
                ├── train/
                │   ├── img001.txt
                │   └── img003.txt
                └── val/
                    ├── img002.txt
                    └── img004.txt
        """

        # 1. Determine split
        splits = self._get_splits(platform_annotations, split_info)

        # 2. Split images
        train_images = [
            img for img in platform_annotations['images']
            if splits.get(img['id']) == 'train'
        ]
        val_images = [
            img for img in platform_annotations['images']
            if splits.get(img['id']) == 'val'
        ]

        # 3. Create label files
        labels_dir = output_dir / "labels"
        for split_name, images in [("train", train_images), ("val", val_images)]:
            split_label_dir = labels_dir / split_name
            split_label_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                label_path = split_label_dir / f"{Path(img['file_name']).stem}.txt"
                with open(label_path, 'w') as f:
                    # Get annotations for this image
                    for ann in self._get_annotations_for_image(
                        img['id'],
                        platform_annotations
                    ):
                        # Convert to YOLO format: <class_id> <x_center> <y_center> <width> <height>
                        yolo_line = self._to_yolo_format(ann, img)
                        f.write(yolo_line + '\n')

        # 4. Create train.txt and val.txt
        with open(output_dir / "train.txt", 'w') as f:
            for img in train_images:
                image_path = f"/workspace/dataset/images/{img['file_name']}"
                f.write(image_path + '\n')

        with open(output_dir / "val.txt", 'w') as f:
            for img in val_images:
                image_path = f"/workspace/dataset/images/{img['file_name']}"
                f.write(image_path + '\n')

        # 5. Create data.yaml
        data_yaml = {
            'path': '/workspace/dataset',
            'train': 'train.txt',
            'val': 'val.txt',
            'nc': len(platform_annotations['classes']),
            'names': [c['name'] for c in platform_annotations['classes']]
        }

        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)

        return yaml_path
```

### Example: timm Converter

```python
# platform/trainers/timm/converters/platform_to_imagefolder.py
from pathlib import Path
from PIL import Image
import shutil

class PlatformToImageFolderConverter:
    """Convert Platform Format to ImageFolder structure"""

    def convert(self, platform_annotations, output_dir, split_info):
        """
        Convert to ImageFolder format:

          output_dir/
            ├── train/
            │   ├── cat/
            │   │   ├── img001.jpg
            │   │   └── img003.jpg
            │   └── dog/
            │       ├── img005.jpg
            │       └── img007.jpg
            └── val/
                ├── cat/
                │   └── img002.jpg
                └── dog/
                    └── img006.jpg
        """

        # 1. Determine split
        splits = self._get_splits(platform_annotations, split_info)

        # 2. Create class directories
        for split_name in ['train', 'val']:
            for cls in platform_annotations['classes']:
                class_dir = output_dir / split_name / cls['name']
                class_dir.mkdir(parents=True, exist_ok=True)

        # 3. Symlink images to class folders
        for img in platform_annotations['images']:
            split = splits.get(img['id'], 'train')
            class_name = img['annotation']['class_name']

            src = Path(f"/workspace/dataset/images/{img['file_name']}")
            dst = output_dir / split / class_name / Path(img['file_name']).name

            # Symlink instead of copy (saves disk space)
            dst.symlink_to(src)

        return output_dir
```

---

## Split Handling

Trainers must handle train/val splits using the 3-level priority system.

### Reading Split Strategy

```python
# train.py
import os
import json

def get_split_info():
    """Get split information from environment variables"""

    # Priority 1: Job-level override
    split_strategy_str = os.environ.get('SPLIT_STRATEGY')
    if split_strategy_str:
        split_strategy = json.loads(split_strategy_str)

        if split_strategy['method'] != 'use_dataset':
            # Job overrides dataset split
            return split_strategy

    # Priority 2: Dataset-level split (from annotations.json)
    annotations = load_annotations()
    if 'split_config' in annotations:
        return {
            'method': 'use_dataset',
            'dataset_split_config': annotations['split_config']
        }

    # Priority 3: Default auto-split
    return {
        'method': 'auto',
        'ratio': [0.8, 0.2],
        'seed': 42
    }
```

### Applying Split

```python
def apply_split(images, split_info):
    """Apply split strategy to images"""

    if split_info['method'] == 'use_dataset':
        # Use dataset's predefined split
        dataset_splits = split_info['dataset_split_config']['splits']
        return dataset_splits

    elif split_info['method'] == 'auto':
        # Auto-generate split
        import random
        random.seed(split_info['seed'])

        shuffled = images.copy()
        random.shuffle(shuffled)

        ratio = split_info['ratio']
        split_idx = int(len(shuffled) * ratio[0])

        splits = {}
        for img in shuffled[:split_idx]:
            splits[img['id']] = 'train'
        for img in shuffled[split_idx:]:
            splits[img['id']] = 'val'

        return splits

    elif split_info['method'] == 'custom':
        # Use job-specific custom split
        return split_info['custom_splits']

    else:
        raise ValueError(f"Unknown split method: {split_info['method']}")
```

### Integration in Training

```python
# train.py main flow
def main():
    # 1. Load snapshot
    snapshot = load_snapshot_from_s3(dataset_id, snapshot_id)
    annotations = snapshot['annotations']

    # 2. Get split strategy
    split_info = get_split_info()

    # 3. Convert format with split
    converter = PlatformToYOLOConverter()
    data_yaml_path = converter.convert(
        platform_annotations=annotations,
        output_dir=Path("/workspace/yolo_data"),
        split_info=split_info
    )

    # 4. Train
    from ultralytics import YOLO
    model = YOLO(model_name)
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        # ... other params
    )
```

---

## Error Handling

### Error Categories

**1. Configuration Errors** (fail fast)
```python
# Missing required environment variable
if "JOB_ID" not in os.environ:
    print("ERROR: JOB_ID environment variable not set")
    sys.exit(1)

# Invalid configuration
epochs = int(os.environ.get("EPOCHS", 0))
if epochs <= 0:
    callback_client.send_completion(
        status="failed",
        error_message="Invalid EPOCHS value: must be > 0"
    )
    sys.exit(1)
```

**2. Dataset Errors**
```python
try:
    dataset_path = storage_client.download_dataset(dataset_id)
except Exception as e:
    callback_client.send_completion(
        status="failed",
        error_message=f"Failed to download dataset: {str(e)}"
    )
    sys.exit(1)

# Validate dataset structure
if not Path(dataset_path, "data.yaml").exists():
    callback_client.send_completion(
        status="failed",
        error_message="Invalid dataset: data.yaml not found"
    )
    sys.exit(1)
```

**3. Training Errors**
```python
try:
    results = model.train(...)
except torch.cuda.OutOfMemoryError:
    callback_client.send_completion(
        status="failed",
        error_message="Out of GPU memory. Try reducing batch_size."
    )
    sys.exit(1)
except Exception as e:
    callback_client.send_completion(
        status="failed",
        error_message=f"Training failed: {str(e)}"
    )
    sys.exit(1)
```

**4. Storage Errors**
```python
try:
    checkpoint_path = storage_client.upload_checkpoint(...)
except Exception as e:
    # Log error but don't fail training
    print(f"Warning: Failed to upload checkpoint: {e}")
    # Training can continue, but final upload must succeed
```

### Timeout Handling

Trainers must respect maximum training time:

```python
import signal

# Set timeout alarm (e.g., 24 hours)
MAX_TRAINING_TIME = int(os.environ.get("MAX_TRAINING_TIME_SECONDS", 86400))

def timeout_handler(signum, frame):
    print("Training timeout exceeded")
    callback_client.send_completion(
        status="failed",
        error_message="Training exceeded maximum time limit"
    )
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(MAX_TRAINING_TIME)

# Start training
model.train(...)

# Cancel alarm if completed
signal.alarm(0)
```

## Framework-Specific Guides

### Ultralytics (YOLO)

**Supported Models**:
- yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
- yolo11n-seg, yolo11s-seg (instance segmentation)
- yolo11n-pose, yolo11s-pose (pose estimation)
- yolo11n-cls (classification)
- yolo_world_v2_s, yolo_world_v2_m (open-vocabulary detection)
- sam2_t, sam2_s (Segment Anything Model 2)

**Task Types**:
- `object_detection`
- `instance_segmentation`
- `pose_estimation`
- `image_classification`

**Dataset Formats**: YOLO, COCO

**Requirements**:
```
ultralytics==8.3.0
torch>=2.0.0
torchvision
opencv-python
boto3
requests
```

### timm (PyTorch Image Models)

**Supported Models**:
- resnet18, resnet50
- efficientnet_b0, efficientnet_b1
- vit_base_patch16_224
- convnext_tiny

**Task Types**:
- `image_classification`

**Dataset Formats**: ImageFolder

**Requirements**:
```
timm==0.9.12
torch>=2.0.0
torchvision
boto3
requests
```

**Example** `train.py` structure:
```python
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(f"{dataset_path}/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load model
model = timm.create_model(model_name, pretrained=pretrained, num_classes=len(train_dataset.classes))
model = model.to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Send heartbeat
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)

    callback_client.send_heartbeat(
        epoch=epoch + 1,
        progress=(epoch + 1) / epochs * 100,
        metrics={
            "loss": avg_loss,
            "accuracy": accuracy
        }
    )

    # Save checkpoint
    if (epoch + 1) % checkpoint_frequency == 0:
        checkpoint_path = f"/workspace/checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        s3_path = storage_client.upload_checkpoint(checkpoint_path, job_id, f"epoch-{epoch+1}.pth")

        callback_client.send_event(
            event_type="checkpoint_saved",
            message=f"Checkpoint saved at epoch {epoch+1}",
            data={"checkpoint_path": s3_path, "epoch": epoch+1}
        )
```

### HuggingFace Transformers

**Supported Models**:
- clip-vit-base-patch32
- dino-vits16
- vit-base-patch16-224

**Task Types**:
- `image_classification`
- `zero_shot_classification` (CLIP)

**Dataset Formats**: ImageFolder, HuggingFace Datasets

**Requirements**:
```
transformers==4.36.0
torch>=2.0.0
datasets
pillow
boto3
requests
```

## Dockerfile Example

```dockerfile
# platform/trainers/ultralytics/Dockerfile
FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY train.py .
COPY utils.py .

# Run training
CMD ["python", "train.py"]
```

## Testing

### Unit Tests

```python
# tests/test_callbacks.py
import pytest
from unittest.mock import Mock, patch
from utils import CallbackClient

def test_callback_client_send_heartbeat():
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200

        client = CallbackClient(
            backend_url="http://test",
            job_id="test-job",
            trace_id="test-trace",
            callback_token="test-token"
        )

        client.send_heartbeat(epoch=1, progress=10.0, metrics={"loss": 0.5})

        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert args[0] == "http://test/api/v1/jobs/test-job/heartbeat"
        assert kwargs['headers']['Authorization'] == "Bearer test-token"
        assert kwargs['json']['epoch'] == 1
```

### Integration Tests

```python
# tests/test_training_integration.py
import pytest
import os
import subprocess

@pytest.mark.subprocess
def test_ultralytics_training_subprocess():
    """Test training in subprocess mode"""
    env = {
        "JOB_ID": "test-job-1",
        "TRACE_ID": "test-trace-1",
        "BACKEND_BASE_URL": "http://localhost:8000",
        "CALLBACK_TOKEN": "test-token",
        "TASK_TYPE": "object_detection",
        "MODEL_NAME": "yolo11n",
        "DATASET_ID": "test-dataset",
        "EPOCHS": "1",
        "STORAGE_TYPE": "local",
        "DATASET_PATH": "/tmp/test-dataset"
    }

    result = subprocess.run(
        ["python", "train.py"],
        env={**os.environ, **env},
        cwd="platform/trainers/ultralytics",
        capture_output=True,
        timeout=300
    )

    assert result.returncode == 0
    assert "Training Complete" in result.stdout.decode()
```

## Inference Execution

Trainers must support two types of inference:
1. **Test Run**: Inference on labeled dataset (with ground truth) → Compute metrics
2. **Inference Job**: Inference on unlabeled data (no ground truth) → Predictions only

Both support optional XAI (Explainable AI) with visualization and LLM-based natural language explanations.

### Inference Environment Variables

```bash
# Job Identifiers
JOB_ID=550e8400-e29b-41d4-a716-446655440000
INFERENCE_TYPE=test_run  # or "inference_job"
TRACE_ID=abc-def-ghi-jkl

# Backend Communication
BACKEND_BASE_URL=https://api.example.com
CALLBACK_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Configuration
TASK_TYPE=object_detection
FRAMEWORK=ultralytics
CHECKPOINT_PATH=s3://bucket/checkpoints/job-abc123/best.pt

# For test_run
DATASET_ID=dataset-uuid  # Labeled dataset
TEST_SPLIT=val  # Which split to use for testing

# For inference_job
IMAGE_PATHS='["s3://bucket/inference-jobs/abc/input/img1.jpg", "s3://bucket/inference-jobs/abc/input/img2.jpg"]'
# Or DATASET_ID for dataset-based inference

# Inference Settings
BATCH_SIZE=16
IMAGE_SIZE=640
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45  # For NMS in detection

# XAI (Explainable AI)
ENABLE_XAI=true
XAI_METHOD=gradcam  # gradcam, lime, shap, attention
XAI_TARGET_LAYERS='["model.24"]'  # Framework-specific layer names
ENABLE_LLM_EXPLANATION=true  # Natural language explanations
OPENAI_API_KEY=sk-...  # For LLM explanations

# Storage
S3_ENDPOINT=https://xxx.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
BUCKET_NAME=vision-platform

# Optional
DEVICE=cuda
NUM_WORKERS=4
SAVE_VISUALIZATIONS=true  # Save annotated images
```

### Inference Callbacks

**1. Progress Update**
```http
POST {BACKEND_BASE_URL}/api/v1/inference/{JOB_ID}/progress
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}

{
  "processed_images": 50,
  "total_images": 100,
  "progress_percent": 50.0,
  "avg_inference_time_ms": 25.3,
  "timestamp": "2025-01-10T12:34:56Z"
}
```

**2. Test Run Completion** (with metrics)
```http
POST {BACKEND_BASE_URL}/api/v1/test-runs/{JOB_ID}/done
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}

{
  "status": "succeeded",
  "overall_metrics": {
    "accuracy": 0.95,
    "mAP50-95": 0.6523,
    "precision": 0.89,
    "recall": 0.82
  },
  "per_class_metrics": [...],
  "confusion_matrix": [[110, 10], [4, 101]],
  "pr_curves": {...},
  "results_path": "s3://bucket/test-runs/abc123/results.json",
  "visualizations_path": "s3://bucket/test-runs/abc123/visualizations/",
  "xai_results_path": "s3://bucket/test-runs/abc123/xai/",
  "timestamp": "2025-01-10T12:35:00Z"
}
```

**3. Inference Job Completion** (predictions only)
```http
POST {BACKEND_BASE_URL}/api/v1/inference/jobs/{JOB_ID}/done
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}

{
  "status": "succeeded",
  "results_path": "s3://bucket/inference-jobs/abc123/results.json",
  "visualizations_path": "s3://bucket/inference-jobs/abc123/visualizations/",
  "xai_results_path": "s3://bucket/inference-jobs/abc123/xai/",
  "timestamp": "2025-01-10T12:35:00Z"
}
```

### Inference Implementation

```python
# platform/trainers/ultralytics/infer.py
import os
import sys
import json
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from datetime import datetime

from utils import CallbackClient, StorageClient, XAIGenerator

def main():
    print("=== Ultralytics Inference Starting ===")

    # 1. Read configuration
    job_id = os.environ["JOB_ID"]
    inference_type = os.environ["INFERENCE_TYPE"]  # "test_run" or "inference_job"
    task_type = os.environ["TASK_TYPE"]
    checkpoint_path = os.environ["CHECKPOINT_PATH"]

    enable_xai = os.environ.get("ENABLE_XAI", "false").lower() == "true"
    xai_method = os.environ.get("XAI_METHOD", "gradcam")
    enable_llm = os.environ.get("ENABLE_LLM_EXPLANATION", "false").lower() == "true"

    print(f"Job ID: {job_id}")
    print(f"Inference Type: {inference_type}")
    print(f"Task: {task_type}")
    print(f"XAI: {enable_xai} ({xai_method})")

    # 2. Initialize clients
    callback_client = CallbackClient(
        backend_url=os.environ["BACKEND_BASE_URL"],
        job_id=job_id,
        trace_id=os.environ["TRACE_ID"],
        callback_token=os.environ["CALLBACK_TOKEN"]
    )

    storage_client = StorageClient()

    try:
        # 3. Download checkpoint
        print("\n=== Downloading Checkpoint ===")
        local_checkpoint = storage_client.download_checkpoint(checkpoint_path)
        model = YOLO(local_checkpoint)
        print(f"Model loaded: {local_checkpoint}")

        # 4. Initialize XAI generator if enabled
        xai_generator = None
        if enable_xai:
            xai_generator = XAIGenerator(
                model=model,
                method=xai_method,
                enable_llm=enable_llm,
                task_type=task_type
            )
            print(f"XAI enabled: {xai_method}")

        # 5. Get images to process
        if inference_type == "test_run":
            # Test run: Use labeled dataset
            dataset_id = os.environ["DATASET_ID"]
            dataset_path = storage_client.download_dataset(dataset_id)

            # Load ground truth
            with open(f"{dataset_path}/annotations.json") as f:
                dataset_info = json.load(f)

            images_to_process = [
                {
                    "path": f"{dataset_path}/{img['file_name']}",
                    "ground_truth": img.get("annotation")
                }
                for img in dataset_info["images"]
                if img.get("split") == os.environ.get("TEST_SPLIT", "val")
            ]
        else:
            # Inference job: Use provided image paths
            image_paths = json.loads(os.environ["IMAGE_PATHS"])

            # Download images from S3
            images_to_process = []
            for s3_path in image_paths:
                local_path = storage_client.download_file(s3_path)
                images_to_process.append({
                    "path": local_path,
                    "ground_truth": None
                })

        total_images = len(images_to_process)
        print(f"\n=== Processing {total_images} Images ===")

        # 6. Run inference on all images
        results = []

        for idx, image_info in enumerate(images_to_process):
            image_path = image_info["path"]
            ground_truth = image_info["ground_truth"]

            # Run inference
            prediction = model(image_path)[0]

            # Extract predictions
            result = {
                "image_path": image_path,
                "predictions": extract_predictions(prediction, task_type),
                "ground_truth": ground_truth
            }

            # Compute metrics if ground truth available
            if ground_truth:
                result["metrics"] = compute_metrics(
                    result["predictions"],
                    ground_truth,
                    task_type
                )

            # Generate XAI if enabled
            if xai_generator:
                xai_result = xai_generator.generate(
                    image_path=image_path,
                    prediction=result["predictions"]
                )
                result["xai"] = xai_result

            results.append(result)

            # Send progress update every 10 images
            if (idx + 1) % 10 == 0 or (idx + 1) == total_images:
                callback_client.send_inference_progress(
                    processed_images=idx + 1,
                    total_images=total_images,
                    progress_percent=((idx + 1) / total_images) * 100
                )
                print(f"Progress: {idx + 1}/{total_images}")

        # 7. Compute overall metrics for test run
        overall_metrics = None
        per_class_metrics = None
        confusion_matrix = None

        if inference_type == "test_run":
            print("\n=== Computing Metrics ===")
            overall_metrics = compute_overall_metrics(results, task_type)
            per_class_metrics = compute_per_class_metrics(results, task_type)
            confusion_matrix = compute_confusion_matrix(results, task_type)

            print(f"Overall Accuracy: {overall_metrics.get('accuracy', 0):.4f}")
            print(f"mAP50-95: {overall_metrics.get('mAP50-95', 0):.4f}")

        # 8. Save results to S3
        print("\n=== Saving Results ===")

        results_filename = f"/workspace/results.json"
        with open(results_filename, "w") as f:
            json.dump({
                "job_id": job_id,
                "inference_type": inference_type,
                "task_type": task_type,
                "num_images": total_images,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "results": results
            }, f, indent=2)

        results_path = storage_client.upload_file(
            local_path=results_filename,
            s3_key=f"{inference_type}s/{job_id}/results.json"
        )

        # Upload visualizations if enabled
        visualizations_path = None
        if os.environ.get("SAVE_VISUALIZATIONS", "true").lower() == "true":
            visualizations_path = save_visualizations(results, job_id, inference_type, storage_client)

        # Upload XAI results if generated
        xai_results_path = None
        if enable_xai:
            xai_results_path = save_xai_results(results, job_id, inference_type, storage_client)

        # 9. Send completion callback
        if inference_type == "test_run":
            callback_client.send_test_run_completion(
                status="succeeded",
                overall_metrics=overall_metrics,
                per_class_metrics=per_class_metrics,
                confusion_matrix=confusion_matrix,
                results_path=results_path,
                visualizations_path=visualizations_path,
                xai_results_path=xai_results_path
            )
        else:
            callback_client.send_inference_completion(
                status="succeeded",
                results_path=results_path,
                visualizations_path=visualizations_path,
                xai_results_path=xai_results_path
            )

        print(f"\n=== Inference Complete ===")
        print(f"Results: {results_path}")
        sys.exit(0)

    except Exception as e:
        print(f"\n=== Inference Failed ===")
        print(f"Error: {str(e)}")

        # Send failure callback
        if inference_type == "test_run":
            callback_client.send_test_run_completion(
                status="failed",
                error_message=str(e)
            )
        else:
            callback_client.send_inference_completion(
                status="failed",
                error_message=str(e)
            )

        sys.exit(1)


def extract_predictions(result, task_type):
    """Extract predictions from YOLO result"""
    if task_type in ["object_detection", "instance_segmentation"]:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        return {
            "boxes": [
                {
                    "bbox": box.tolist(),
                    "class_id": int(class_id),
                    "class_name": result.names[class_id],
                    "confidence": float(score)
                }
                for box, score, class_id in zip(boxes, scores, class_ids)
            ]
        }

    elif task_type == "image_classification":
        top5_indices = result.probs.top5
        top5_confidences = result.probs.top5conf.cpu().numpy()

        return {
            "class_id": int(top5_indices[0]),
            "class_name": result.names[top5_indices[0]],
            "confidence": float(top5_confidences[0]),
            "top5": [
                {
                    "class_id": int(idx),
                    "class_name": result.names[idx],
                    "confidence": float(conf)
                }
                for idx, conf in zip(top5_indices, top5_confidences)
            ]
        }


def compute_metrics(prediction, ground_truth, task_type):
    """Compute per-image metrics"""
    # Implementation depends on task type
    # For detection: IoU, precision, recall
    # For classification: correct/incorrect
    pass


def compute_overall_metrics(results, task_type):
    """Compute overall metrics across all images"""
    # Aggregate per-image metrics
    pass


if __name__ == "__main__":
    main()
```

### XAI Integration

```python
# platform/trainers/ultralytics/xai_generator.py
import os
import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import openai

class XAIGenerator:
    """
    Generates XAI visualizations and explanations

    Supports:
    - Grad-CAM: Gradient-weighted Class Activation Mapping
    - LLM Explanations: Natural language explanations using GPT-4o-mini
    """

    def __init__(self, model, method="gradcam", enable_llm=False, task_type="object_detection"):
        self.model = model
        self.method = method
        self.enable_llm = enable_llm
        self.task_type = task_type

        if enable_llm:
            self.llm_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, image_path, prediction):
        """Generate XAI visualization and explanation"""

        # Generate visualization
        heatmap, activation_data = self._generate_gradcam(image_path, prediction)

        xai_result = {
            "method": self.method,
            "heatmap_path": None,  # Will be set after upload
            self.method: activation_data
        }

        # Generate LLM explanation if enabled
        if self.enable_llm:
            explanation = self._generate_llm_explanation(
                image_path=image_path,
                prediction=prediction,
                activation_data=activation_data
            )
            xai_result["explanation"] = explanation

        return xai_result, heatmap

    def _generate_gradcam(self, image_path, prediction):
        """Generate Grad-CAM heatmap"""

        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get target layers
        target_layers = self._get_target_layers()

        # Create Grad-CAM
        cam = GradCAM(model=self.model.model, target_layers=target_layers)

        # Generate heatmap
        grayscale_cam = cam(input_tensor=self._preprocess_image(image_rgb))

        # Overlay on image
        heatmap = show_cam_on_image(
            image_rgb / 255.0,
            grayscale_cam,
            use_rgb=True
        )

        # Compute activation statistics
        activation_data = {
            "max_activation": float(np.max(grayscale_cam)),
            "mean_activation": float(np.mean(grayscale_cam)),
            "top_regions": self._find_top_regions(grayscale_cam)
        }

        return heatmap, activation_data

    def _generate_llm_explanation(self, image_path, prediction, activation_data):
        """Generate natural language explanation using LLM"""

        # Build prompt based on task type
        if self.task_type == "image_classification":
            prompt = f"""Analyze this image classification result with Grad-CAM visualization.

**Prediction:**
- Predicted Class: {prediction['class_name']}
- Confidence: {prediction['confidence']:.1%}

**Grad-CAM Analysis:**
- Maximum Activation: {activation_data['max_activation']:.2f}
- Top Activation Regions: {activation_data['top_regions']}

Please provide:
1. Brief summary (1-2 sentences)
2. 3-4 key factors the model focused on
3. Confidence reasoning
4. Alternative considerations

Format as JSON:
{{
  "summary": "...",
  "key_factors": ["...", "..."],
  "confidence_reasoning": "...",
  "alternative_considerations": "..."
}}
"""

        # Call LLM
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI model explanation expert. Provide clear, concise explanations."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        # Parse response
        explanation_text = response.choices[0].message.content

        try:
            explanation = json.loads(explanation_text)
        except:
            # Fallback if not valid JSON
            explanation = {
                "summary": explanation_text,
                "key_factors": [],
                "confidence_reasoning": "",
                "alternative_considerations": ""
            }

        # Add metadata
        explanation["model"] = "gpt-4o-mini"
        explanation["generated_at"] = datetime.utcnow().isoformat() + "Z"

        return explanation

    def _get_target_layers(self):
        """Get target layers for Grad-CAM based on framework"""
        # Framework-specific layer selection
        layer_names = os.environ.get("XAI_TARGET_LAYERS", '["model.24"]')
        layers = json.loads(layer_names)

        target_layers = []
        for layer_name in layers:
            layer = eval(f"self.model.{layer_name}")
            target_layers.append(layer)

        return target_layers

    def _find_top_regions(self, heatmap, top_k=5):
        """Find top-k regions with highest activation"""
        # Threshold and find contours
        threshold = np.percentile(heatmap, 90)
        mask = (heatmap > threshold).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours[:top_k]:
            x, y, w, h = cv2.boundingRect(contour)
            score = np.mean(heatmap[y:y+h, x:x+w])

            regions.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "score": float(score)
            })

        # Sort by score
        regions.sort(key=lambda r: r["score"], reverse=True)

        return regions[:top_k]
```

### Callback Client Extensions for Inference

Add these methods to `CallbackClient` in `utils.py`:

```python
def send_inference_progress(
    self,
    processed_images: int,
    total_images: int,
    progress_percent: float
):
    """Send inference progress update"""
    url = f"{self.backend_url}/api/v1/inference/{self.job_id}/progress"
    payload = {
        "processed_images": processed_images,
        "total_images": total_images,
        "progress_percent": progress_percent,
        "avg_inference_time_ms": 0,  # Can calculate from timing
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    try:
        response = requests.post(url, json=payload, headers=self.headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Warning: Failed to send progress: {e}")


def send_test_run_completion(
    self,
    status: str,
    overall_metrics: dict = None,
    per_class_metrics: list = None,
    confusion_matrix: list = None,
    results_path: str = None,
    visualizations_path: str = None,
    xai_results_path: str = None,
    error_message: str = None
):
    """Send test run completion"""
    url = f"{self.backend_url}/api/v1/test-runs/{self.job_id}/done"
    payload = {
        "status": status,
        "overall_metrics": overall_metrics or {},
        "per_class_metrics": per_class_metrics or [],
        "confusion_matrix": confusion_matrix,
        "pr_curves": {},
        "results_path": results_path,
        "visualizations_path": visualizations_path,
        "xai_results_path": xai_results_path,
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    try:
        response = requests.post(url, json=payload, headers=self.headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error: Failed to send test run completion: {e}")
        raise


def send_inference_completion(
    self,
    status: str,
    results_path: str = None,
    visualizations_path: str = None,
    xai_results_path: str = None,
    error_message: str = None
):
    """Send inference job completion"""
    url = f"{self.backend_url}/api/v1/inference/jobs/{self.job_id}/done"
    payload = {
        "status": status,
        "results_path": results_path,
        "visualizations_path": visualizations_path,
        "xai_results_path": xai_results_path,
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    try:
        response = requests.post(url, json=payload, headers=self.headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error: Failed to send inference completion: {e}")
        raise
```

## Export Execution

Trainers must support model export to various deployment formats (ONNX, TensorRT, CoreML, TFLite, OpenVINO, TorchScript).

Each framework has different capabilities for export formats - see the Framework Capability Matrix in `EXPORT_DEPLOYMENT_DESIGN.md`.

### Export Environment Variables

```bash
# Job Identifiers
JOB_ID=550e8400-e29b-41d4-a716-446655440000
TRACE_ID=abc-def-ghi-jkl

# Backend Communication
BACKEND_BASE_URL=https://api.example.com
CALLBACK_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Export Configuration
EXPORT_FORMAT=onnx  # onnx, tensorrt, coreml, tflite, openvino, torchscript
TASK_TYPE=object_detection
FRAMEWORK=ultralytics
CHECKPOINT_PATH=s3://bucket/checkpoints/job-abc123/best.pt
MODEL_NAME=yolo11n

# Export Settings
INPUT_SIZE='[640, 640]'  # Model input size
OPTIMIZATION_LEVEL=3  # 0-3, framework-specific
INCLUDE_PREPROCESSING=true  # Embed preprocessing in model
INCLUDE_POSTPROCESSING=true  # Embed postprocessing in model

# Quantization (optional)
QUANTIZATION_TYPE=dynamic  # dynamic, static, none
QUANTIZATION_DTYPE=int8  # int8, int16, float16

# Validation (optional)
ENABLE_VALIDATION=true
VALIDATION_DATASET_ID=dataset-uuid
ACCURACY_DROP_MAX=0.02  # Fail if accuracy drops > 2%

# Storage
S3_ENDPOINT=https://xxx.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
BUCKET_NAME=vision-platform

# Optional
DEVICE=cuda
BATCH_SIZE=1  # For validation
```

### Export Callbacks

**1. Progress Update**
```http
POST {BACKEND_BASE_URL}/api/v1/export/jobs/{JOB_ID}/progress
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}

{
  "progress_percent": 50.0,
  "stage": "converting",  # downloading, converting, optimizing, validating, packaging
  "message": "Converting model to ONNX format...",
  "timestamp": "2025-01-10T12:34:56Z"
}
```

**2. Export Completion**
```http
POST {BACKEND_BASE_URL}/api/v1/export/jobs/{JOB_ID}/done
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}

{
  "status": "succeeded",  # or "failed"
  "export_path": "s3://bucket/export-jobs/abc123/model.onnx",
  "metadata_path": "s3://bucket/export-jobs/abc123/metadata.json",
  "package_path": "s3://bucket/export-jobs/abc123/package.zip",
  "model_info": {
    "model_size_mb": 22.5,
    "inference_speed_ms": 15.3,
    "input_shape": [1, 3, 640, 640],
    "output_shape": [1, 84, 8400]
  },
  "validation_metrics": {  # If validation enabled
    "accuracy": 0.94,
    "original_accuracy": 0.95,
    "accuracy_drop": 0.01,
    "passed": true
  },
  "error_message": null,
  "timestamp": "2025-01-10T12:35:00Z"
}
```

### Export Implementation

```python
# platform/trainers/ultralytics/export.py
import os
import sys
import json
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import datetime

from utils import CallbackClient, StorageClient, MetadataGenerator, RuntimeWrapperGenerator

def main():
    print("=== Ultralytics Export Starting ===")

    # 1. Read configuration
    job_id = os.environ["JOB_ID"]
    export_format = os.environ["EXPORT_FORMAT"]
    checkpoint_path = os.environ["CHECKPOINT_PATH"]
    task_type = os.environ["TASK_TYPE"]
    model_name = os.environ["MODEL_NAME"]
    input_size = json.loads(os.environ["INPUT_SIZE"])

    enable_validation = os.environ.get("ENABLE_VALIDATION", "false").lower() == "true"
    include_preprocessing = os.environ.get("INCLUDE_PREPROCESSING", "true").lower() == "true"
    include_postprocessing = os.environ.get("INCLUDE_POSTPROCESSING", "true").lower() == "true"

    print(f"Job ID: {job_id}")
    print(f"Format: {export_format}")
    print(f"Task: {task_type}")
    print(f"Input Size: {input_size}")

    # 2. Initialize clients
    callback_client = CallbackClient(
        backend_url=os.environ["BACKEND_BASE_URL"],
        job_id=job_id,
        trace_id=os.environ["TRACE_ID"],
        callback_token=os.environ["CALLBACK_TOKEN"]
    )

    storage_client = StorageClient()
    metadata_generator = MetadataGenerator()
    wrapper_generator = RuntimeWrapperGenerator()

    try:
        # 3. Download checkpoint
        callback_client.send_export_progress(
            progress_percent=10.0,
            stage="downloading",
            message="Downloading checkpoint..."
        )

        print("\n=== Downloading Checkpoint ===")
        local_checkpoint = storage_client.download_checkpoint(checkpoint_path)
        model = YOLO(local_checkpoint)
        print(f"Model loaded: {local_checkpoint}")

        # 4. Export model
        callback_client.send_export_progress(
            progress_percent=30.0,
            stage="converting",
            message=f"Converting to {export_format.upper()} format..."
        )

        print(f"\n=== Exporting to {export_format.upper()} ===")

        export_kwargs = {
            "format": export_format,
            "imgsz": input_size,
            "device": os.environ.get("DEVICE", "cpu")
        }

        # Add format-specific options
        if export_format == "onnx":
            export_kwargs["opset"] = 12
            export_kwargs["simplify"] = True
            export_kwargs["dynamic"] = False  # Fixed input size
        elif export_format == "tensorrt":
            export_kwargs["half"] = False  # FP32 by default
            export_kwargs["workspace"] = 4  # 4GB workspace
        elif export_format == "coreml":
            export_kwargs["nms"] = include_postprocessing  # Include NMS

        # Add quantization if enabled
        quantization_type = os.environ.get("QUANTIZATION_TYPE", "none")
        if quantization_type != "none":
            if export_format == "onnx":
                # ONNX quantization done post-export
                pass
            elif export_format == "tflite":
                export_kwargs["int8"] = (quantization_type == "static")

        # Export
        start_time = time.time()
        export_result = model.export(**export_kwargs)
        export_time = time.time() - start_time

        exported_model_path = Path(export_result)
        print(f"Export complete: {exported_model_path}")
        print(f"Export time: {export_time:.2f}s")

        # 5. Apply quantization (if ONNX + quantization enabled)
        if export_format == "onnx" and quantization_type != "none":
            callback_client.send_export_progress(
                progress_percent=50.0,
                stage="optimizing",
                message="Applying quantization..."
            )

            print("\n=== Applying Quantization ===")
            quantized_model_path = apply_onnx_quantization(
                model_path=exported_model_path,
                quantization_type=quantization_type,
                dtype=os.environ.get("QUANTIZATION_DTYPE", "int8")
            )
            exported_model_path = quantized_model_path
            print(f"Quantized model: {exported_model_path}")

        # 6. Validation (optional)
        validation_metrics = None
        if enable_validation:
            callback_client.send_export_progress(
                progress_percent=60.0,
                stage="validating",
                message="Validating exported model..."
            )

            print("\n=== Validating Export ===")
            validation_metrics = validate_export(
                original_model=model,
                exported_model_path=exported_model_path,
                export_format=export_format,
                dataset_id=os.environ.get("VALIDATION_DATASET_ID"),
                task_type=task_type,
                storage_client=storage_client
            )

            print(f"Original Accuracy: {validation_metrics['original_accuracy']:.4f}")
            print(f"Exported Accuracy: {validation_metrics['accuracy']:.4f}")
            print(f"Accuracy Drop: {validation_metrics['accuracy_drop']:.4f}")

            # Check threshold
            max_drop = float(os.environ.get("ACCURACY_DROP_MAX", 0.02))
            if validation_metrics['accuracy_drop'] > max_drop:
                raise ValueError(
                    f"Validation failed: Accuracy drop {validation_metrics['accuracy_drop']:.4f} "
                    f"exceeds threshold {max_drop}"
                )

            print("Validation passed!")

        # 7. Generate metadata
        callback_client.send_export_progress(
            progress_percent=75.0,
            stage="packaging",
            message="Generating metadata and runtime wrappers..."
        )

        print("\n=== Generating Metadata ===")

        metadata = metadata_generator.generate(
            model_path=exported_model_path,
            export_format=export_format,
            task_type=task_type,
            model_name=model_name,
            input_size=input_size,
            class_names=model.names,
            include_preprocessing=include_preprocessing,
            include_postprocessing=include_postprocessing,
            quantization_type=quantization_type if quantization_type != "none" else None,
            validation_metrics=validation_metrics
        )

        metadata_path = Path("/workspace/metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata generated: {metadata_path}")

        # 8. Generate runtime wrappers
        print("\n=== Generating Runtime Wrappers ===")

        wrappers_dir = Path("/workspace/runtimes")
        wrapper_generator.generate_all(
            export_format=export_format,
            task_type=task_type,
            metadata=metadata,
            output_dir=wrappers_dir
        )

        print(f"Runtime wrappers generated: {wrappers_dir}")

        # 9. Create deployment package
        print("\n=== Creating Deployment Package ===")

        package_path = create_deployment_package(
            model_path=exported_model_path,
            metadata_path=metadata_path,
            wrappers_dir=wrappers_dir,
            export_format=export_format,
            task_type=task_type
        )

        print(f"Package created: {package_path}")

        # 10. Upload to S3
        callback_client.send_export_progress(
            progress_percent=90.0,
            stage="uploading",
            message="Uploading exported model..."
        )

        print("\n=== Uploading to S3 ===")

        # Upload model file
        model_s3_path = storage_client.upload_file(
            local_path=str(exported_model_path),
            s3_key=f"export-jobs/{job_id}/model.{export_format}"
        )

        # Upload metadata
        metadata_s3_path = storage_client.upload_file(
            local_path=str(metadata_path),
            s3_key=f"export-jobs/{job_id}/metadata.json"
        )

        # Upload package
        package_s3_path = storage_client.upload_file(
            local_path=str(package_path),
            s3_key=f"export-jobs/{job_id}/package.zip"
        )

        print(f"Model uploaded: {model_s3_path}")
        print(f"Metadata uploaded: {metadata_s3_path}")
        print(f"Package uploaded: {package_s3_path}")

        # 11. Get model info
        model_size_mb = exported_model_path.stat().st_size / (1024 * 1024)
        input_shape = [1, 3] + input_size
        output_shape = get_output_shape(exported_model_path, export_format)

        model_info = {
            "model_size_mb": round(model_size_mb, 2),
            "inference_speed_ms": round(export_time * 1000 / 100, 2),  # Estimate
            "input_shape": input_shape,
            "output_shape": output_shape
        }

        # 12. Send completion callback
        callback_client.send_export_completion(
            status="succeeded",
            export_path=model_s3_path,
            metadata_path=metadata_s3_path,
            package_path=package_s3_path,
            model_info=model_info,
            validation_metrics=validation_metrics
        )

        print(f"\n=== Export Complete ===")
        print(f"Format: {export_format}")
        print(f"Model: {model_s3_path}")
        sys.exit(0)

    except Exception as e:
        print(f"\n=== Export Failed ===")
        print(f"Error: {str(e)}")

        # Send failure callback
        callback_client.send_export_completion(
            status="failed",
            error_message=str(e)
        )

        sys.exit(1)


def apply_onnx_quantization(model_path, quantization_type, dtype):
    """Apply quantization to ONNX model"""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, quantize_static

    output_path = model_path.parent / f"{model_path.stem}_quantized.onnx"

    if quantization_type == "dynamic":
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=getattr(onnx.TensorProto, dtype.upper())
        )
    elif quantization_type == "static":
        # Static quantization requires calibration data
        # For now, use dynamic quantization
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=getattr(onnx.TensorProto, dtype.upper())
        )

    return output_path


def validate_export(
    original_model,
    exported_model_path,
    export_format,
    dataset_id,
    task_type,
    storage_client
):
    """
    Validate exported model by comparing accuracy with original.

    Returns:
        dict: Validation metrics including accuracy_drop
    """
    # Download validation dataset
    dataset_path = storage_client.download_dataset(dataset_id)

    # Load validation data
    val_data = load_validation_data(dataset_path, task_type)

    # Run inference with original model
    print("Running inference with original model...")
    original_predictions = run_inference_batch(original_model, val_data)
    original_accuracy = compute_accuracy(original_predictions, val_data["labels"], task_type)

    # Run inference with exported model
    print("Running inference with exported model...")
    exported_model = load_exported_model(exported_model_path, export_format, task_type)
    exported_predictions = run_inference_batch(exported_model, val_data)
    exported_accuracy = compute_accuracy(exported_predictions, val_data["labels"], task_type)

    # Compute metrics
    accuracy_drop = abs(original_accuracy - exported_accuracy)

    return {
        "original_accuracy": original_accuracy,
        "accuracy": exported_accuracy,
        "accuracy_drop": accuracy_drop,
        "passed": True,
        "num_samples": len(val_data["images"])
    }


def create_deployment_package(
    model_path,
    metadata_path,
    wrappers_dir,
    export_format,
    task_type
):
    """
    Create deployment package with model, metadata, and runtime wrappers.

    Package structure:
      package.zip
        ├── model.{format}
        ├── metadata.json
        ├── README.md
        ├── runtimes/
        │   ├── python/
        │   │   ├── model_wrapper.py
        │   │   ├── requirements.txt
        │   │   └── example.py
        │   ├── cpp/
        │   │   ├── model_wrapper.cpp
        │   │   ├── model_wrapper.h
        │   │   ├── CMakeLists.txt
        │   │   └── example.cpp
        │   ├── swift/  (for CoreML)
        │   │   ├── ModelWrapper.swift
        │   │   ├── Package.swift
        │   │   └── Example.swift
        │   └── kotlin/  (for TFLite)
        │       ├── ModelWrapper.kt
        │       ├── build.gradle
        │       └── Example.kt
        └── Dockerfile (for container deployment)
    """
    import zipfile

    package_path = Path("/workspace/package.zip")

    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add model
        zf.write(model_path, f"model.{export_format}")

        # Add metadata
        zf.write(metadata_path, "metadata.json")

        # Add README
        readme_content = generate_readme(export_format, task_type, model_path)
        zf.writestr("README.md", readme_content)

        # Add runtime wrappers
        for wrapper_file in wrappers_dir.rglob("*"):
            if wrapper_file.is_file():
                rel_path = wrapper_file.relative_to(wrappers_dir.parent)
                zf.write(wrapper_file, str(rel_path))

        # Add Dockerfile for container deployment
        dockerfile_content = generate_dockerfile(export_format, task_type)
        zf.writestr("Dockerfile", dockerfile_content)

    return package_path


def get_output_shape(model_path, export_format):
    """Get model output shape"""
    # Implementation depends on format
    # For ONNX, can parse from model
    # For others, may need to run sample inference
    if export_format == "onnx":
        import onnx
        model = onnx.load(str(model_path))
        output = model.graph.output[0]
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        return shape
    else:
        # Default placeholder
        return [1, 84, 8400]


if __name__ == "__main__":
    main()
```

### Metadata Generation

```python
# platform/trainers/ultralytics/metadata_generator.py
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

class MetadataGenerator:
    """
    Generate comprehensive metadata for exported models.

    Metadata includes preprocessing, postprocessing, and runtime information.
    """

    def generate(
        self,
        model_path: Path,
        export_format: str,
        task_type: str,
        model_name: str,
        input_size: List[int],
        class_names: List[str],
        include_preprocessing: bool,
        include_postprocessing: bool,
        quantization_type: Optional[str] = None,
        validation_metrics: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate complete metadata"""

        metadata = {
            "format_version": "1.0",
            "export_format": export_format,
            "task_type": task_type,
            "model_name": model_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",

            # Model info
            "model": {
                "framework": "ultralytics",
                "architecture": model_name,
                "input_size": input_size,
                "num_classes": len(class_names),
                "class_names": class_names
            },

            # Preprocessing
            "preprocessing": self._generate_preprocessing_spec(
                task_type=task_type,
                input_size=input_size,
                embedded=include_preprocessing
            ),

            # Postprocessing
            "postprocessing": self._generate_postprocessing_spec(
                task_type=task_type,
                embedded=include_postprocessing
            ),

            # Runtime info
            "runtime": self._generate_runtime_info(
                export_format=export_format,
                task_type=task_type
            ),

            # Optimization
            "optimization": {
                "quantization": quantization_type,
                "include_preprocessing": include_preprocessing,
                "include_postprocessing": include_postprocessing
            }
        }

        # Add validation metrics if available
        if validation_metrics:
            metadata["validation"] = validation_metrics

        return metadata

    def _generate_preprocessing_spec(
        self,
        task_type: str,
        input_size: List[int],
        embedded: bool
    ) -> Dict[str, Any]:
        """Generate preprocessing specification"""

        if task_type in ["object_detection", "instance_segmentation"]:
            return {
                "resize": {
                    "method": "letterbox",  # Ultralytics uses letterbox
                    "size": input_size,
                    "fill_value": 114
                },
                "normalize": {
                    "method": "divide",
                    "mean": [0.0, 0.0, 0.0],
                    "std": [255.0, 255.0, 255.0]
                },
                "channel_order": "RGB",
                "data_type": "float32",
                "layout": "NCHW",  # Batch, Channel, Height, Width
                "embedded": embedded
            }
        elif task_type == "image_classification":
            return {
                "resize": {
                    "method": "resize_crop",
                    "size": input_size
                },
                "normalize": {
                    "method": "standardize",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                },
                "channel_order": "RGB",
                "data_type": "float32",
                "layout": "NCHW",
                "embedded": embedded
            }

    def _generate_postprocessing_spec(
        self,
        task_type: str,
        embedded: bool
    ) -> Dict[str, Any]:
        """Generate postprocessing specification"""

        if task_type == "object_detection":
            return {
                "nms": {
                    "method": "nms",
                    "iou_threshold": 0.45,
                    "confidence_threshold": 0.25,
                    "max_detections": 300
                },
                "coordinate_format": "xyxy",  # x1, y1, x2, y2
                "embedded": embedded
            }
        elif task_type == "instance_segmentation":
            return {
                "nms": {
                    "method": "nms",
                    "iou_threshold": 0.45,
                    "confidence_threshold": 0.25,
                    "max_detections": 300
                },
                "mask_threshold": 0.5,
                "coordinate_format": "xyxy",
                "embedded": embedded
            }
        elif task_type == "image_classification":
            return {
                "softmax": {
                    "apply": True,
                    "temperature": 1.0
                },
                "top_k": 5,
                "embedded": embedded
            }

    def _generate_runtime_info(
        self,
        export_format: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Generate runtime information"""

        runtimes = {
            "onnx": {
                "engines": ["onnxruntime", "opencv_dnn", "tensorrt"],
                "languages": ["python", "cpp", "csharp", "java"],
                "platforms": ["windows", "linux", "macos", "android", "ios"]
            },
            "tensorrt": {
                "engines": ["tensorrt"],
                "languages": ["python", "cpp"],
                "platforms": ["linux", "windows"]
            },
            "coreml": {
                "engines": ["coreml"],
                "languages": ["swift", "objective-c", "python"],
                "platforms": ["ios", "macos"]
            },
            "tflite": {
                "engines": ["tflite"],
                "languages": ["python", "kotlin", "java", "swift"],
                "platforms": ["android", "ios", "linux", "raspberry_pi"]
            },
            "openvino": {
                "engines": ["openvino"],
                "languages": ["python", "cpp"],
                "platforms": ["windows", "linux", "macos"]
            },
            "torchscript": {
                "engines": ["pytorch"],
                "languages": ["python", "cpp"],
                "platforms": ["windows", "linux", "macos", "android", "ios"]
            }
        }

        return runtimes.get(export_format, {})
```

### Runtime Wrapper Generation

```python
# platform/trainers/ultralytics/runtime_wrapper_generator.py
from pathlib import Path
from typing import Dict, Any

class RuntimeWrapperGenerator:
    """Generate runtime wrappers for different languages and platforms"""

    def generate_all(
        self,
        export_format: str,
        task_type: str,
        metadata: Dict[str, Any],
        output_dir: Path
    ):
        """Generate all applicable runtime wrappers"""

        output_dir.mkdir(parents=True, exist_ok=True)

        # Always generate Python wrapper
        self.generate_python_wrapper(
            export_format=export_format,
            task_type=task_type,
            metadata=metadata,
            output_dir=output_dir / "python"
        )

        # Always generate C++ wrapper
        self.generate_cpp_wrapper(
            export_format=export_format,
            task_type=task_type,
            metadata=metadata,
            output_dir=output_dir / "cpp"
        )

        # Format-specific wrappers
        if export_format == "coreml":
            self.generate_swift_wrapper(
                task_type=task_type,
                metadata=metadata,
                output_dir=output_dir / "swift"
            )

        if export_format == "tflite":
            self.generate_kotlin_wrapper(
                task_type=task_type,
                metadata=metadata,
                output_dir=output_dir / "kotlin"
            )

    def generate_python_wrapper(
        self,
        export_format: str,
        task_type: str,
        metadata: Dict[str, Any],
        output_dir: Path
    ):
        """Generate Python wrapper"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate model_wrapper.py
        wrapper_code = self._generate_python_wrapper_code(
            export_format=export_format,
            task_type=task_type,
            metadata=metadata
        )

        (output_dir / "model_wrapper.py").write_text(wrapper_code)

        # Generate requirements.txt
        requirements = self._generate_python_requirements(export_format)
        (output_dir / "requirements.txt").write_text(requirements)

        # Generate example.py
        example_code = self._generate_python_example(
            export_format=export_format,
            task_type=task_type
        )
        (output_dir / "example.py").write_text(example_code)

    def _generate_python_wrapper_code(
        self,
        export_format: str,
        task_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate Python wrapper code"""

        # Template for Python wrapper
        template = '''"""
Model Wrapper for {task_type} ({export_format})

Auto-generated wrapper for model inference with preprocessing and postprocessing.
"""

import json
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any

{imports}

class ModelWrapper:
    """Wrapper for {export_format} model"""

    def __init__(self, model_path: str, metadata_path: str = None):
        """
        Initialize model wrapper.

        Args:
            model_path: Path to model file
            metadata_path: Path to metadata.json (optional)
        """
        self.model_path = Path(model_path)

        # Load metadata
        if metadata_path:
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            # Try to find metadata.json in same directory
            metadata_file = self.model_path.parent / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    self.metadata = json.load(f)
            else:
                raise ValueError("metadata.json not found")

        # Load model
        {model_init}

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image according to metadata.

        Args:
            image: Input image (H, W, C) in BGR format (OpenCV)

        Returns:
            Preprocessed image tensor
        """
        preproc = self.metadata["preprocessing"]

        # Convert BGR to RGB
        if preproc["channel_order"] == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        resize_config = preproc["resize"]
        if resize_config["method"] == "letterbox":
            image = self._letterbox_resize(image, resize_config["size"])
        else:
            image = cv2.resize(image, tuple(resize_config["size"]))

        # Normalize
        norm_config = preproc["normalize"]
        if norm_config["method"] == "divide":
            image = image.astype(np.float32) / 255.0
        elif norm_config["method"] == "standardize":
            image = image.astype(np.float32) / 255.0
            mean = np.array(norm_config["mean"]).reshape(1, 1, 3)
            std = np.array(norm_config["std"]).reshape(1, 1, 3)
            image = (image - mean) / std

        # Change layout to NCHW
        if preproc["layout"] == "NCHW":
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, 0)  # Add batch dimension

        return image.astype(np.float32)

    def postprocess(self, outputs: Any) -> List[Dict]:
        """
        Postprocess model outputs according to metadata.

        Args:
            outputs: Raw model outputs

        Returns:
            Postprocessed predictions
        """
        postproc = self.metadata["postprocessing"]

        {postprocess_impl}

    def predict(self, image_path: str) -> List[Dict]:
        """
        Run inference on single image.

        Args:
            image_path: Path to input image

        Returns:
            List of predictions
        """
        # Load image
        image = cv2.imread(image_path)

        # Preprocess
        input_tensor = self.preprocess(image)

        # Run inference
        {inference_code}

        # Postprocess
        predictions = self.postprocess(outputs)

        return predictions

    def _letterbox_resize(self, image, target_size):
        """Resize with padding (letterbox)"""
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h))

        # Pad
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        return padded


# Helper functions
{helper_functions}
'''

        # Fill in template based on export format
        imports = self._get_python_imports(export_format)
        model_init = self._get_python_model_init(export_format)
        inference_code = self._get_python_inference_code(export_format)
        postprocess_impl = self._get_python_postprocess_impl(task_type)
        helper_functions = self._get_python_helper_functions(task_type)

        return template.format(
            task_type=task_type,
            export_format=export_format,
            imports=imports,
            model_init=model_init,
            inference_code=inference_code,
            postprocess_impl=postprocess_impl,
            helper_functions=helper_functions
        )

    def _get_python_imports(self, export_format: str) -> str:
        """Get format-specific imports"""
        if export_format == "onnx":
            return "import onnxruntime as ort\nimport cv2"
        elif export_format == "tensorrt":
            return "import tensorrt as trt\nimport pycuda.driver as cuda\nimport cv2"
        elif export_format == "torchscript":
            return "import torch\nimport cv2"
        else:
            return "import cv2"

    def _get_python_model_init(self, export_format: str) -> str:
        """Get model initialization code"""
        if export_format == "onnx":
            return '''self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name'''
        elif export_format == "torchscript":
            return '''self.model = torch.jit.load(str(self.model_path))
        self.model.eval()'''
        else:
            return "# Model initialization"

    def _get_python_inference_code(self, export_format: str) -> str:
        """Get inference code"""
        if export_format == "onnx":
            return "outputs = self.session.run(None, {self.input_name: input_tensor})"
        elif export_format == "torchscript":
            return '''with torch.no_grad():
            input_torch = torch.from_numpy(input_tensor)
            outputs = self.model(input_torch)
            outputs = [o.numpy() for o in outputs]'''
        else:
            return "# Inference code"

    def _get_python_postprocess_impl(self, task_type: str) -> str:
        """Get postprocessing implementation"""
        if task_type == "object_detection":
            return '''# NMS and formatting
        nms_config = postproc["nms"]

        # Parse outputs (depends on model architecture)
        boxes, scores, class_ids = self._parse_detection_outputs(outputs)

        # Apply NMS
        indices = self._apply_nms(
            boxes=boxes,
            scores=scores,
            iou_threshold=nms_config["iou_threshold"],
            confidence_threshold=nms_config["confidence_threshold"]
        )

        # Format results
        predictions = []
        for idx in indices:
            predictions.append({
                "bbox": boxes[idx].tolist(),
                "class_id": int(class_ids[idx]),
                "class_name": self.metadata["model"]["class_names"][int(class_ids[idx])],
                "confidence": float(scores[idx])
            })

        return predictions'''
        else:
            return "# Postprocessing implementation"

    def _get_python_helper_functions(self, task_type: str) -> str:
        """Get helper functions"""
        if task_type == "object_detection":
            return '''
def _parse_detection_outputs(outputs):
    """Parse model outputs to boxes, scores, class_ids"""
    # Implementation depends on model architecture
    pass

def _apply_nms(boxes, scores, iou_threshold, confidence_threshold):
    """Apply Non-Maximum Suppression"""
    import cv2
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        confidence_threshold,
        iou_threshold
    )
    return indices.flatten() if len(indices) > 0 else []
'''
        else:
            return ""

    def _generate_python_requirements(self, export_format: str) -> str:
        """Generate requirements.txt"""
        if export_format == "onnx":
            return "onnxruntime-gpu\nopencv-python\nnumpy"
        elif export_format == "tensorrt":
            return "tensorrt\npycuda\nopencv-python\nnumpy"
        elif export_format == "torchscript":
            return "torch\ntorchvision\nopencv-python\nnumpy"
        else:
            return "opencv-python\nnumpy"

    def _generate_python_example(
        self,
        export_format: str,
        task_type: str
    ) -> str:
        """Generate example usage"""
        return '''"""
Example usage of ModelWrapper
"""

from model_wrapper import ModelWrapper
import cv2

def main():
    # Initialize wrapper
    wrapper = ModelWrapper(
        model_path="model.{format}",
        metadata_path="metadata.json"
    )

    # Run inference
    predictions = wrapper.predict("test_image.jpg")

    # Print results
    for pred in predictions:
        print(pred)

    # Visualize (for detection)
    image = cv2.imread("test_image.jpg")
    for pred in predictions:
        bbox = pred["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        label = f"{{pred['class_name']}}: {{pred['confidence']:.2f}}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Predictions", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
'''.format(format=export_format)

    def generate_cpp_wrapper(
        self,
        export_format: str,
        task_type: str,
        metadata: Dict[str, Any],
        output_dir: Path
    ):
        """Generate C++ wrapper"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate model_wrapper.h
        header_code = self._generate_cpp_header(export_format, task_type)
        (output_dir / "model_wrapper.h").write_text(header_code)

        # Generate model_wrapper.cpp
        cpp_code = self._generate_cpp_implementation(export_format, task_type, metadata)
        (output_dir / "model_wrapper.cpp").write_text(cpp_code)

        # Generate CMakeLists.txt
        cmake_code = self._generate_cmake(export_format)
        (output_dir / "CMakeLists.txt").write_text(cmake_code)

    def _generate_cpp_header(self, export_format: str, task_type: str) -> str:
        """Generate C++ header file"""
        return '''// model_wrapper.h
#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct Prediction {
    std::vector<float> bbox;  // [x1, y1, x2, y2]
    int class_id;
    std::string class_name;
    float confidence;
};

class ModelWrapper {
public:
    ModelWrapper(const std::string& model_path, const std::string& metadata_path);
    ~ModelWrapper();

    std::vector<Prediction> predict(const std::string& image_path);

private:
    cv::Mat preprocess(const cv::Mat& image);
    std::vector<Prediction> postprocess(/* outputs */);

    // Model-specific members
    void* model_;  // Opaque pointer to model
    // Add format-specific members
};
'''

    def _generate_cpp_implementation(
        self,
        export_format: str,
        task_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate C++ implementation file"""
        return "// model_wrapper.cpp\n// C++ implementation\n"

    def _generate_cmake(self, export_format: str) -> str:
        """Generate CMakeLists.txt"""
        return f'''cmake_minimum_required(VERSION 3.10)
project(ModelWrapper)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# Add format-specific dependencies
# For {export_format}

add_library(model_wrapper model_wrapper.cpp)
target_link_libraries(model_wrapper ${{OpenCV_LIBS}})

add_executable(example example.cpp)
target_link_libraries(example model_wrapper)
'''

    def generate_swift_wrapper(
        self,
        task_type: str,
        metadata: Dict[str, Any],
        output_dir: Path
    ):
        """Generate Swift wrapper for CoreML"""
        output_dir.mkdir(parents=True, exist_ok=True)

        swift_code = '''// ModelWrapper.swift
import CoreML
import Vision
import UIKit

class ModelWrapper {
    private let model: VNCoreMLModel
    private let metadata: [String: Any]

    init(modelPath: String, metadataPath: String) throws {
        // Load model
        let compiledUrl = try MLModel.compileModel(at: URL(fileURLWithPath: modelPath))
        let mlModel = try MLModel(contentsOf: compiledUrl)
        self.model = try VNCoreMLModel(for: mlModel)

        // Load metadata
        let metadataData = try Data(contentsOf: URL(fileURLWithPath: metadataPath))
        self.metadata = try JSONSerialization.jsonObject(with: metadataData) as! [String: Any]
    }

    func predict(imagePath: String) -> [[String: Any]] {
        // Load image
        guard let image = UIImage(contentsOfFile: imagePath),
              let cgImage = image.cgImage else {
            return []
        }

        // Run inference
        var predictions: [[String: Any]] = []

        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                return
            }

            for result in results {
                let prediction: [String: Any] = [
                    "bbox": [
                        result.boundingBox.origin.x,
                        result.boundingBox.origin.y,
                        result.boundingBox.size.width,
                        result.boundingBox.size.height
                    ],
                    "class_name": result.labels.first?.identifier ?? "",
                    "confidence": result.confidence
                ]
                predictions.append(prediction)
            }
        }

        let handler = VNImageRequestHandler(cgImage: cgImage)
        try? handler.perform([request])

        return predictions
    }
}
'''

        (output_dir / "ModelWrapper.swift").write_text(swift_code)

    def generate_kotlin_wrapper(
        self,
        task_type: str,
        metadata: Dict[str, Any],
        output_dir: Path
    ):
        """Generate Kotlin wrapper for TFLite"""
        output_dir.mkdir(parents=True, exist_ok=True)

        kotlin_code = '''// ModelWrapper.kt
package com.example.modelwrapper

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ModelWrapper(context: Context, modelPath: String, metadataPath: String) {
    private val interpreter: Interpreter
    private val metadata: Map<String, Any>

    init {
        // Load model
        val model = loadModelFile(modelPath)
        interpreter = Interpreter(model)

        // Load metadata
        metadata = loadMetadata(metadataPath)
    }

    fun predict(bitmap: Bitmap): List<Prediction> {
        // Preprocess
        val inputTensor = preprocess(bitmap)

        // Run inference
        val outputTensor = Array(1) { FloatArray(84 * 8400) }
        interpreter.run(inputTensor, outputTensor)

        // Postprocess
        return postprocess(outputTensor)
    }

    private fun preprocess(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        // Resize and normalize
        // Return tensor in shape [1, 3, 640, 640]
        return Array(1) { Array(3) { Array(640) { FloatArray(640) } } }
    }

    private fun postprocess(outputs: Array<FloatArray>): List<Prediction> {
        // Apply NMS and format results
        return emptyList()
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = FileInputStream(modelPath).channel
        return fileDescriptor.map(FileChannel.MapMode.READ_ONLY, 0, fileDescriptor.size())
    }

    private fun loadMetadata(metadataPath: String): Map<String, Any> {
        // Load and parse JSON metadata
        return emptyMap()
    }
}

data class Prediction(
    val bbox: List<Float>,
    val classId: Int,
    val className: String,
    val confidence: Float
)
'''

        (output_dir / "ModelWrapper.kt").write_text(kotlin_code)


def generate_readme(export_format: str, task_type: str, model_path: Path) -> str:
    """Generate README for deployment package"""
    return f'''# Model Deployment Package

## Model Information

- **Format**: {export_format.upper()}
- **Task**: {task_type}
- **Model File**: model.{export_format}

## Contents

This package includes:
- `model.{export_format}`: Exported model file
- `metadata.json`: Model metadata (preprocessing, postprocessing, runtime info)
- `runtimes/`: Runtime wrappers for different languages
  - `python/`: Python wrapper with example
  - `cpp/`: C++ wrapper with CMake build files
  - `swift/`: Swift wrapper (CoreML only)
  - `kotlin/`: Kotlin wrapper (TFLite only)
- `Dockerfile`: Container deployment template
- `README.md`: This file

## Quick Start

### Python

```bash
cd runtimes/python
pip install -r requirements.txt
python example.py
```

### C++

```bash
cd runtimes/cpp
mkdir build && cd build
cmake ..
make
./example
```

### Container

```bash
docker build -t my-model .
docker run -v $(pwd)/test_images:/input my-model
```

## Runtime Wrappers

All runtime wrappers provide the same interface:
1. Load model and metadata
2. Preprocess input image
3. Run inference
4. Postprocess outputs

See individual README files in `runtimes/` for language-specific details.

## Model Metadata

The `metadata.json` file contains complete specifications for:
- Preprocessing (resize, normalization)
- Postprocessing (NMS, thresholds)
- Runtime requirements
- Class names and task information

Use this metadata to ensure correct inference in your application.

## Support

For issues or questions, please refer to the platform documentation.
'''


def generate_dockerfile(export_format: str, task_type: str) -> str:
    """Generate Dockerfile for container deployment"""
    if export_format == "onnx":
        return '''FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install onnxruntime-gpu opencv-python numpy

# Copy model and runtime
COPY model.onnx .
COPY metadata.json .
COPY runtimes/python/model_wrapper.py .

# Run inference
CMD ["python", "model_wrapper.py"]
'''
    else:
        return f'''FROM python:3.11-slim

WORKDIR /app

# Install dependencies for {export_format}
RUN pip install opencv-python numpy

# Copy model and runtime
COPY model.{export_format} .
COPY metadata.json .
COPY runtimes/python/model_wrapper.py .

# Run inference
CMD ["python", "model_wrapper.py"]
'''
```

### Callback Client Extensions for Export

Add these methods to `CallbackClient` in `utils.py`:

```python
def send_export_progress(
    self,
    progress_percent: float,
    stage: str,
    message: str
):
    """Send export progress update"""
    url = f"{self.backend_url}/api/v1/export/jobs/{self.job_id}/progress"
    payload = {
        "progress_percent": progress_percent,
        "stage": stage,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    try:
        response = requests.post(url, json=payload, headers=self.headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Warning: Failed to send export progress: {e}")


def send_export_completion(
    self,
    status: str,
    export_path: str = None,
    metadata_path: str = None,
    package_path: str = None,
    model_info: dict = None,
    validation_metrics: dict = None,
    error_message: str = None
):
    """Send export completion"""
    url = f"{self.backend_url}/api/v1/export/jobs/{self.job_id}/done"
    payload = {
        "status": status,
        "export_path": export_path,
        "metadata_path": metadata_path,
        "package_path": package_path,
        "model_info": model_info or {},
        "validation_metrics": validation_metrics,
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    try:
        response = requests.post(url, json=payload, headers=self.headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error: Failed to send export completion: {e}")
        raise
```

## Best Practices

1. **Always validate environment variables** at startup
2. **Send heartbeats regularly** - every epoch minimum
3. **Don't fail on callback errors** - log and continue
4. **Upload checkpoints incrementally** - don't wait until end
5. **Set timeouts** - prevent infinite training
6. **Use structured logging** - helps debugging
7. **Clean up temp files** - free disk space
8. **Handle GPU OOM gracefully** - suggest reducing batch size
9. **Generate comprehensive metadata** - include preprocessing/postprocessing specs
10. **Provide runtime wrappers** - make deployment easy for users

## References

- [Architecture Overview](./OVERVIEW.md)
- [Backend Design](./BACKEND_DESIGN.md)
- [Isolation Principles](./ISOLATION_DESIGN.md)
- [3-Tier Development](../development/3_TIER_DEVELOPMENT.md)
- [Trainers README](../../trainers/README.md)
