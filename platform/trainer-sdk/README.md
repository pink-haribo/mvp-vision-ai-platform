# Trainer SDK

Unified SDK for platform communication across all trainer implementations.

## Overview

This SDK provides a single, consistent interface for all trainers to communicate with the Vision AI Training Platform backend. Instead of maintaining duplicate copies in each trainer directory, all trainers share this unified SDK.

## Features

- **Lifecycle Management**: `report_started()`, `report_progress()`, `report_completed()`, `report_failed()`
- **Storage Operations**: Upload/download checkpoints, datasets with caching support
- **Dataset Conversion**: DICE to YOLO, DICE to COCO format conversion
- **Logging**: Buffered logging with automatic flush
- **ClearML Integration**: Optional metrics logging to ClearML (Phase 12.2)

## Supported Conversions

| Source | Target | Use Case |
|--------|--------|----------|
| dice | yolo | Ultralytics YOLO, mmyolo |
| dice | coco | MMDetection, VFM, mmpretrain, mmseg |

## Supported Annotation Formats (for `download_dataset_from_annotation`)

| Format | Description |
|--------|-------------|
| labelme | LabelMe format with `data_summary` containing S3 URIs |
| coco | COCO format (planned) |

## Usage

```python
from trainer_sdk import TrainerSDK, ErrorType

sdk = TrainerSDK()
sdk.report_started()

# Download and convert dataset
dataset_dir = sdk.download_dataset_with_cache(
    snapshot_id=sdk.snapshot_id,
    dataset_id=sdk.dataset_id,
    dataset_version_hash=sdk.dataset_version_hash,
    dest_dir='/tmp/training/123'
)
sdk.convert_dataset(dataset_dir, 'dice', 'yolo')  # or 'coco'

# OR: Download from annotation data (LabelMe format)
annotation_data = {
    "data_summary": [
        {"img_path": "s3://bucket/images/001.jpg", "label_path": "s3://bucket/labels/001.json"},
        {"img_path": "s3://bucket/images/002.jpg", "label_path": "s3://bucket/labels/002.json"},
    ]
}
dataset_dir = sdk.download_dataset_from_annotation(
    annotation_data=annotation_data,
    dest_dir='/tmp/training/123',
    format_type='labelme'
)

# Training loop
for epoch in range(1, epochs + 1):
    metrics = train_one_epoch(...)
    sdk.report_progress(epoch, epochs, metrics)

# Upload checkpoints
best_uri = sdk.upload_checkpoint('/tmp/best.pt', 'best')
sdk.report_completed(final_metrics, checkpoints={'best': best_uri})
```

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `CALLBACK_URL` | Backend API base URL |
| `JOB_ID` | Training/Inference/Export job ID |

### Job Context

| Variable | Description |
|----------|-------------|
| `TASK_TYPE` | Task type (detection, classification, segmentation, pose) |
| `MODEL_NAME` | Model name (e.g., yolo11n, vfm_v1_l) |
| `FRAMEWORK` | Framework name (ultralytics, vfm-v1, mmdet, etc.) |
| `DATASET_ID` | Dataset ID |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTERNAL_STORAGE_ENDPOINT` | `http://localhost:9000` | External S3 endpoint (datasets) |
| `INTERNAL_STORAGE_ENDPOINT` | `http://localhost:9002` | Internal S3 endpoint (checkpoints) |

### Dataset Caching

| Variable | Default | Description |
|----------|---------|-------------|
| `SNAPSHOT_ID` | - | Dataset snapshot ID |
| `DATASET_VERSION_HASH` | - | Dataset version hash |
| `DATASET_CACHE_DIR` | `/tmp/datasets` | Shared cache directory |
| `DATASET_CACHE_MAX_GB` | `50` | Max cache size in GB |

## Docker Build

All trainers must be built from the `platform/` directory:

```bash
cd platform/

# Ultralytics
docker build -f trainers/ultralytics/Dockerfile -t trainer-ultralytics:latest .

# VFM-v1
docker build -f trainers/vfm-v1/Dockerfile -t trainer-vfm-v1:latest .

# MMDetection
docker build -f trainers/mmdet/Dockerfile -t trainer-mmdet:latest .
```

## Directory Structure

```
platform/
├── trainer-sdk/
│   ├── trainer_sdk.py    # Unified SDK (this file)
│   └── README.md
│
└── trainers/
    ├── ultralytics/
    │   ├── Dockerfile    # COPY trainer-sdk/trainer_sdk.py .
    │   └── train.py
    ├── vfm-v1/
    │   ├── Dockerfile
    │   └── train.py
    └── ...
```

## Version History

- **v2.0.0** (2025-01-21): Unified SDK with multi-format dataset conversion
- **v1.0.0** (2025-11-19): Initial per-trainer SDK implementation
