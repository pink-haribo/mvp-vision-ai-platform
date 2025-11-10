# Trainer Service Design

Complete design for framework-specific training services.

## Table of Contents

- [Overview](#overview)
- [API Contract](#api-contract)
- [Directory Structure](#directory-structure)
- [Trainer Implementation](#trainer-implementation)
- [HTTP Callbacks](#http-callbacks)
- [Storage Integration](#storage-integration)
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
EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=0.001
IMAGE_SIZE=640

# Storage (S3-compatible)
STORAGE_TYPE=r2  # or s3, minio
R2_ENDPOINT=https://xxx.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=xxx
R2_SECRET_ACCESS_KEY=xxx
BUCKET_NAME=vision-platform

# Optional
PRETRAINED=true
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
    pretrained = os.environ.get("PRETRAINED", "true").lower() == "true"

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

        # 4. Load model
        print("\n=== Loading Model ===")
        model = YOLO(f"{model_name}.pt" if pretrained else f"{model_name}.yaml")
        print(f"Model loaded: {model_name}")

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

## Best Practices

1. **Always validate environment variables** at startup
2. **Send heartbeats regularly** - every epoch minimum
3. **Don't fail on callback errors** - log and continue
4. **Upload checkpoints incrementally** - don't wait until end
5. **Set timeouts** - prevent infinite training
6. **Use structured logging** - helps debugging
7. **Clean up temp files** - free disk space
8. **Handle GPU OOM gracefully** - suggest reducing batch size

## References

- [Architecture Overview](./OVERVIEW.md)
- [Backend Design](./BACKEND_DESIGN.md)
- [Isolation Principles](./ISOLATION_DESIGN.md)
- [3-Tier Development](../development/3_TIER_DEVELOPMENT.md)
- [Trainers README](../../trainers/README.md)
