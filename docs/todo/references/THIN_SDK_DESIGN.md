# Thin SDK Design Specification

**Version**: 1.0
**Date**: 2025-11-19
**Status**: Design Phase

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Current State Analysis](#current-state-analysis)
4. [SDK Function Specification](#sdk-function-specification)
5. [Data Schema Specification](#data-schema-specification)
6. [Storage Client Specification](#storage-client-specification)
7. [Implementation Plan](#implementation-plan)
8. [Migration Strategy](#migration-strategy)
9. [Testing Strategy](#testing-strategy)

---

## Overview

### Background

현재 Vision AI Training Platform의 trainer들은 Convention-based 방식으로 플랫폼과 통신합니다. 각 trainer가 `utils.py`의 `CallbackClient`와 `DualStorageClient`를 직접 사용하고 있으며, callback 데이터 형식이 trainer마다 다를 수 있는 구조입니다.

### Problem Statement

1. **통신 프로토콜 불일치**: trainer마다 다른 callback 데이터 형식 사용 가능
2. **에러 처리 불일치**: 구조화된 에러 타입 없이 string message만 전달
3. **메타데이터 불일치**: export metadata 필드명이 trainer마다 다를 수 있음
4. **플랫폼 복잡도**: Backend가 다양한 형식의 callback을 파싱해야 함
5. **중복 코드**: 각 trainer가 비슷한 통신 로직을 반복 구현

### Solution: Thin SDK

**"통신은 SDK로 강제, 로직은 자유롭게"**

- 최소 의존성 (requests/httpx, boto3만 필요)
- 단일 파일 (~500 lines)
- 모든 callback 형식 표준화
- 학습/추론 로직은 강제하지 않음

---

## Design Principles

### 1. Minimal Dependencies

```python
# Required dependencies
import httpx   # HTTP client with retry support
import boto3   # S3-compatible storage
import yaml    # Dataset configuration (PyYAML)
import mlflow  # Experiment tracking
```

### 2. Single File Distribution

SDK는 단일 파일로 제공되어 각 trainer에 복사해서 사용합니다.
버전 관리와 배포가 단순해집니다.

```
platform/trainers/ultralytics/
├── trainer_sdk.py      # Copy of SDK
├── train.py
├── predict.py
└── export.py
```

### 3. Convention over Configuration

- 환경변수 기반 설정 (K8s Job 호환)
- 표준화된 exit code
- 표준화된 callback URL 패턴

### 4. Backward Compatibility

기존 `utils.py`의 `CallbackClient`와 `DualStorageClient` 기능을 모두 포함하면서 표준화된 인터페이스를 제공합니다.

---

## Current State Analysis

### 현재 통신 패턴

#### Training Job (train.py)

```python
# 현재 구현
callback_client = CallbackClient(callback_url)

# 1. Progress reporting (매 epoch)
progress_data = {
    'job_id': int(job_id),
    'status': 'running',
    'current_epoch': epoch,
    'total_epochs': epochs,
    'progress_percent': (epoch / epochs) * 100,
    'metrics': {
        'extra_metrics': metrics
    }
}
callback_client.send_progress_sync(job_id, progress_data)

# 2. Validation results (학습 완료 후)
validation_data = {
    'job_id': int(job_id),
    'epoch': epochs,
    'task_type': task_type,
    'primary_metric_name': 'mAP50-95',
    'primary_metric_value': float,
    'metrics': final_metrics,
    'class_names': class_names,
    'visualization_urls': visualization_urls,
}
callback_client.send_validation_sync(job_id, validation_data)

# 3. Completion (성공/실패)
completion_data = {
    'job_id': int(job_id),
    'status': 'completed',  # or 'failed'
    'total_epochs_completed': epochs,
    'final_metrics': {'extra_metrics': final_metrics},
    'best_checkpoint_path': best_checkpoint_uri,
    'last_checkpoint_path': last_checkpoint_uri,
    'mlflow_run_id': run_id,
    'exit_code': 0,
}
await callback_client.send_completion(job_id, completion_data)
```

#### Inference Job (predict.py)

```python
# 현재 구현
completion_data = {
    'status': 'completed',
    'total_images': image_count,
    'total_inference_time_ms': total_time,
    'avg_inference_time_ms': avg_time,
    'results': image_results,  # List of per-image results
}
await callback_client.send_inference_completion(inference_job_id, completion_data)
```

#### Export Job (export.py)

```python
# 현재 구현 (export.py 분석 기반)
completion_data = {
    'status': 'completed',
    'export_format': format,
    'output_s3_uri': s3_uri,
    'file_size_bytes': size,
    'metadata': metadata_dict,
}
# POST to callback_url
```

### Callback URL Patterns

| Operation | Current URL Pattern |
|-----------|---------------------|
| Training Progress | `/training/jobs/{job_id}/callback/progress` |
| Training Completion | `/training/jobs/{job_id}/callback/completion` |
| Validation Results | `/validation/jobs/{job_id}/results` |
| Inference Completion | `/test_inference/inference/{job_id}/results` |
| Export Completion | `/export/{job_id}/callback` |

### Storage Operations

```python
storage = DualStorageClient()

# Dataset download (External Storage)
storage.download_dataset(dataset_id, dest_dir)

# Checkpoint upload (Internal Storage)
checkpoint_uri = storage.upload_checkpoint(local_path, job_id, filename)
```

---

## SDK Function Specification

### 1. Core Classes

#### TrainerSDK (Main Entry Point)

```python
class TrainerSDK:
    """
    Lightweight SDK for platform communication.

    Usage:
        sdk = TrainerSDK()
        sdk.report_started()
        # ... training logic ...
        sdk.report_progress(epoch=1, total_epochs=10, metrics={...})
        # ...
        sdk.report_completed(final_metrics={...})
    """

    def __init__(self):
        """Initialize SDK from environment variables."""
        pass
```

### 2. Lifecycle Reporting Functions

#### report_started()

```python
def report_started(self, operation_type: str = 'training') -> None:
    """
    Report job started.

    Args:
        operation_type: 'training', 'inference', or 'export'

    Callback:
        POST /training/jobs/{job_id}/callback/progress
        {
            "type": "started",
            "job_id": 123,
            "operation_type": "training",
            "timestamp": "2025-01-19T10:00:00Z"
        }
    """
```

#### report_progress()

```python
def report_progress(
    self,
    epoch: int,
    total_epochs: int,
    metrics: Dict[str, float],
    extra_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Report training progress.

    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        metrics: Standardized metrics dict
        extra_data: Additional framework-specific data

    Callback:
        POST /training/jobs/{job_id}/callback/progress
        {
            "type": "progress",
            "job_id": 123,
            "epoch": 1,
            "total_epochs": 10,
            "progress_percent": 10.0,
            "metrics": {
                "loss": 0.234,
                "accuracy": 0.876,
                "val_loss": 0.345,
                "val_accuracy": 0.812,
                "learning_rate": 0.001
            },
            "extra_data": {...},
            "timestamp": "2025-01-19T10:05:00Z"
        }

    Metrics Schema:
        Required keys: 'loss'
        Optional keys: 'accuracy', 'val_loss', 'val_accuracy', 'learning_rate',
                      'mAP50', 'mAP50-95', 'precision', 'recall', etc.
    """
```

#### report_validation()

```python
def report_validation(
    self,
    epoch: int,
    task_type: str,
    primary_metric: Tuple[str, float],
    all_metrics: Dict[str, float],
    class_names: Optional[List[str]] = None,
    visualization_urls: Optional[Dict[str, str]] = None,
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
) -> None:
    """
    Report validation results.

    Args:
        epoch: Validation epoch
        task_type: 'detection', 'classification', 'segmentation', 'pose'
        primary_metric: Tuple of (metric_name, metric_value)
        all_metrics: All validation metrics
        class_names: List of class names
        visualization_urls: S3 URIs of validation plots
        per_class_metrics: Per-class AP, precision, recall

    Callback:
        POST /validation/jobs/{job_id}/results
        {
            "type": "validation",
            "job_id": 123,
            "epoch": 10,
            "task_type": "detection",
            "primary_metric_name": "mAP50-95",
            "primary_metric_value": 0.567,
            "metrics": {...},
            "class_names": ["person", "car", ...],
            "visualization_urls": {
                "confusion_matrix": "s3://...",
                "pr_curve": "s3://..."
            },
            "per_class_metrics": {
                "person": {"AP": 0.78, "precision": 0.85, "recall": 0.72}
            },
            "timestamp": "2025-01-19T10:30:00Z"
        }
    """
```

#### report_completed()

```python
def report_completed(
    self,
    final_metrics: Dict[str, float],
    checkpoints: Optional[Dict[str, str]] = None,
    mlflow_run_id: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Report job completed successfully.

    Args:
        final_metrics: Final training/inference metrics
        checkpoints: Dict of checkpoint type to S3 URI
                    e.g., {'best': 's3://...', 'last': 's3://...'}
        mlflow_run_id: MLflow run ID for tracking
        extra_data: Additional data

    Callback:
        POST /training/jobs/{job_id}/callback/completion
        {
            "type": "completed",
            "job_id": 123,
            "status": "completed",
            "total_epochs_completed": 10,
            "final_metrics": {...},
            "checkpoints": {
                "best": "s3://training-checkpoints/checkpoints/123/best.pt",
                "last": "s3://training-checkpoints/checkpoints/123/last.pt"
            },
            "mlflow_run_id": "abc123",
            "exit_code": 0,
            "timestamp": "2025-01-19T11:00:00Z"
        }
    """
```

#### report_failed()

```python
def report_failed(
    self,
    error_type: str,
    message: str,
    traceback: Optional[str] = None,
    epochs_completed: int = 0
) -> None:
    """
    Report job failed.

    Args:
        error_type: Structured error type (see Error Types below)
        message: Human-readable error message
        traceback: Full traceback string
        epochs_completed: Number of epochs completed before failure

    Error Types:
        - 'DatasetError': Dataset not found, corrupted, invalid format
        - 'CheckpointError': Checkpoint not found, corrupted, incompatible
        - 'ConfigError': Invalid configuration, unsupported parameters
        - 'ResourceError': Out of memory, GPU not available
        - 'NetworkError': S3 connection failed, callback failed
        - 'FrameworkError': Framework-specific error (YOLO, timm, etc.)
        - 'ValidationError': Validation failed, NaN loss
        - 'UnknownError': Unexpected error

    Callback:
        POST /training/jobs/{job_id}/callback/completion
        {
            "type": "failed",
            "job_id": 123,
            "status": "failed",
            "error_type": "ResourceError",
            "error_message": "CUDA out of memory",
            "traceback": "...",
            "total_epochs_completed": 5,
            "exit_code": 1,
            "timestamp": "2025-01-19T10:45:00Z"
        }
    """
```

### 3. Inference-Specific Functions

#### report_inference_completed()

```python
def report_inference_completed(
    self,
    total_images: int,
    total_time_ms: float,
    results: List[Dict[str, Any]],
    result_urls: Optional[Dict[str, str]] = None
) -> None:
    """
    Report inference job completed.

    Args:
        total_images: Number of images processed
        total_time_ms: Total inference time in milliseconds
        results: Per-image results (see Results Schema below)
        result_urls: S3 URIs of outputs

    Results Schema (per image):
        {
            "image_name": "image001.jpg",
            "predictions": [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.95,
                    "bbox": [100, 200, 300, 400]  # xyxy
                }
            ],
            "inference_time_ms": 45.2,
            "num_detections": 3
        }

    Callback:
        POST /test_inference/inference/{job_id}/results
        {
            "type": "inference_completed",
            "job_id": 789,
            "status": "completed",
            "total_images": 100,
            "total_inference_time_ms": 4520.5,
            "avg_inference_time_ms": 45.2,
            "results": [...],
            "result_urls": {
                "predictions_json": "s3://...",
                "annotated_images": "s3://..."
            },
            "timestamp": "2025-01-19T12:00:00Z"
        }
    """
```

### 4. Export-Specific Functions

#### report_export_completed()

```python
def report_export_completed(
    self,
    export_format: str,
    output_s3_uri: str,
    file_size_bytes: int,
    metadata: Dict[str, Any]
) -> None:
    """
    Report export job completed.

    Args:
        export_format: 'onnx', 'tensorrt', 'coreml', etc.
        output_s3_uri: S3 URI of exported model
        file_size_bytes: Size of exported file
        metadata: Model metadata (see Metadata Schema)

    Callback:
        POST /export/{job_id}/callback
        {
            "type": "export_completed",
            "job_id": 456,
            "status": "completed",
            "export_format": "onnx",
            "output_s3_uri": "s3://training-checkpoints/exports/456/model.onnx",
            "file_size_bytes": 12345678,
            "metadata": {...},
            "timestamp": "2025-01-19T13:00:00Z"
        }
    """
```

#### create_export_metadata()

```python
@staticmethod
def create_export_metadata(
    framework: str,
    model_name: str,
    export_format: str,
    task_type: str,
    input_shape: List[int],
    output_shape: List[List[int]],
    class_names: Optional[List[str]] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
    postprocessing: Optional[Dict[str, Any]] = None,
    export_config: Optional[Dict[str, Any]] = None,
    **extra
) -> Dict[str, Any]:
    """
    Create standardized export metadata.

    Returns:
        {
            "framework": "ultralytics",
            "model_name": "yolo11n",
            "export_format": "onnx",
            "task_type": "detection",
            "input_shape": [640, 640, 3],
            "input_dtype": "float32",
            "output_shape": [[1, 84, 8400]],
            "class_names": ["person", "car", ...],
            "num_classes": 80,
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "color_space": "RGB"
            },
            "postprocessing": {
                "apply_nms": true,
                "confidence_threshold": 0.25
            },
            "export_config": {"opset_version": 17},
            "created_at": "2025-01-19T13:00:00Z"
        }
    """
```

### 5. Storage Functions

#### upload_checkpoint()

```python
def upload_checkpoint(
    self,
    local_path: str,
    checkpoint_type: str,
    metrics: Optional[Dict[str, float]] = None
) -> str:
    """
    Upload checkpoint to Internal Storage.

    Args:
        local_path: Local file path
        checkpoint_type: 'best', 'last', or 'epoch_{n}'
        metrics: Metrics at checkpoint time

    Returns:
        S3 URI of uploaded checkpoint

    Also reports checkpoint saved via callback:
        {
            "type": "checkpoint_saved",
            "job_id": 123,
            "checkpoint_type": "best",
            "s3_uri": "s3://training-checkpoints/checkpoints/123/best.pt",
            "metrics": {...},
            "timestamp": "..."
        }
    """
```

#### download_checkpoint()

```python
def download_checkpoint(self, s3_uri: str, local_path: str) -> str:
    """
    Download checkpoint from Internal Storage.

    Args:
        s3_uri: S3 URI of checkpoint
        local_path: Local destination path

    Returns:
        Local file path
    """
```

#### download_dataset()

```python
def download_dataset(self, dataset_id: str, dest_dir: str) -> str:
    """
    Download dataset from External Storage.

    Args:
        dataset_id: Dataset ID (from S3 URI)
        dest_dir: Local destination directory

    Returns:
        Local dataset directory path
    """
```

#### upload_file()

```python
def upload_file(
    self,
    local_path: str,
    s3_key: str,
    content_type: str = 'application/octet-stream',
    storage_type: str = 'internal'
) -> str:
    """
    Upload arbitrary file to storage.

    Args:
        local_path: Local file path
        s3_key: S3 key (path within bucket)
        content_type: MIME type
        storage_type: 'internal' or 'external'

    Returns:
        S3 URI of uploaded file
    """
```

### 6. MLflow Integration Functions

SDK는 MLflow 실험 추적을 표준화된 방식으로 제공합니다.

#### setup_mlflow()

```python
def setup_mlflow(self, experiment_name: Optional[str] = None) -> str:
    """
    Setup MLflow tracking.

    Args:
        experiment_name: MLflow experiment name.
                        Defaults to 'training-job-{job_id}'

    Returns:
        MLflow run ID

    Environment Variables:
        MLFLOW_TRACKING_URI: MLflow server URL (default: http://localhost:5000)

    Example:
        sdk = TrainerSDK()
        run_id = sdk.setup_mlflow()
        # MLflow run is now active
    """
```

#### log_metrics()

```python
def log_metrics(
    self,
    metrics: Dict[str, float],
    step: Optional[int] = None
) -> None:
    """
    Log metrics to MLflow.

    Automatically sanitizes metric names for MLflow compatibility.

    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number (epoch)

    Example:
        sdk.log_metrics({
            'loss': 0.234,
            'mAP50': 0.876,
            'mAP50-95': 0.654
        }, step=5)
    """
```

#### log_params()

```python
def log_params(self, params: Dict[str, Any]) -> None:
    """
    Log hyperparameters to MLflow.

    Args:
        params: Dictionary of parameter names and values

    Example:
        sdk.log_params({
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'model_name': 'yolo11n'
        })
    """
```

#### log_artifact()

```python
def log_artifact(
    self,
    local_path: str,
    artifact_path: Optional[str] = None
) -> None:
    """
    Log artifact file to MLflow.

    Args:
        local_path: Local file path
        artifact_path: Directory in artifact store

    Example:
        sdk.log_artifact('/tmp/confusion_matrix.png', 'plots')
        sdk.log_artifact('/tmp/model_summary.txt')
    """
```

#### end_mlflow_run()

```python
def end_mlflow_run(self, status: str = 'FINISHED') -> None:
    """
    End MLflow run.

    Args:
        status: Run status ('FINISHED', 'FAILED', 'KILLED')

    Note:
        Called automatically by report_completed() and report_failed().
    """
```

### 7. Logging Functions

SDK는 구조화된 로깅을 제공하여 모든 trainer가 일관된 로그 포맷을 사용합니다.

#### get_logger()

```python
def get_logger(self, name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name. Defaults to 'trainer.{job_id}'

    Returns:
        Configured logger with standard format

    Log Format:
        %(asctime)s - %(name)s - %(levelname)s - %(message)s

    Example:
        logger = sdk.get_logger()
        logger.info("Training started")
        logger.warning("Low GPU memory")
        logger.error("Dataset validation failed")
    """
```

#### log_event()

```python
def log_event(
    self,
    event_type: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    level: str = 'INFO'
) -> None:
    """
    Log structured event.

    Events are logged locally and optionally sent to Backend for
    real-time monitoring.

    Args:
        event_type: Event category
                   ('training', 'validation', 'checkpoint', 'error')
        message: Human-readable message
        data: Additional structured data
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

    Example:
        sdk.log_event(
            'checkpoint',
            'Saved best checkpoint',
            data={'epoch': 50, 'mAP50-95': 0.654, 's3_uri': '...'}
        )
    """
```

### 8. WebSocket Note

**WebSocket은 SDK가 아닌 Backend에서 처리합니다.**

```
Trainer (SDK)                    Backend                     Frontend
    |                               |                           |
    |-- HTTP callback ------------->|                           |
    |   (report_progress)           |-- WebSocket broadcast --->|
    |                               |   (training_metrics)      |
    |                               |                           |
```

- **SDK 역할**: HTTP callback으로 Backend에 상태/메트릭 전송
- **Backend 역할**: Callback 수신 → WebSocket으로 실시간 브로드캐스트
- **Frontend 역할**: WebSocket 연결 → 실시간 UI 업데이트

Trainer는 WebSocket을 직접 사용하지 않으므로 SDK에 WebSocket 기능이 없습니다.

---

## Data Schema Specification

### Metrics Schema

모든 trainer는 다음 키 이름을 사용해야 합니다:

#### Common Metrics

| Key | Type | Description |
|-----|------|-------------|
| `loss` | float | Training loss (required) |
| `val_loss` | float | Validation loss |
| `accuracy` | float | Training accuracy |
| `val_accuracy` | float | Validation accuracy |
| `learning_rate` | float | Current learning rate |

#### Detection Metrics

| Key | Type | Description |
|-----|------|-------------|
| `mAP50` | float | mAP at IoU=0.5 |
| `mAP50-95` | float | mAP at IoU=0.5:0.95 |
| `precision` | float | Precision |
| `recall` | float | Recall |
| `box_loss` | float | Box regression loss |
| `cls_loss` | float | Classification loss |
| `dfl_loss` | float | Distribution focal loss |

#### Classification Metrics

| Key | Type | Description |
|-----|------|-------------|
| `top1_accuracy` | float | Top-1 accuracy |
| `top5_accuracy` | float | Top-5 accuracy |

#### Segmentation Metrics

| Key | Type | Description |
|-----|------|-------------|
| `mask_mAP50` | float | Mask mAP at IoU=0.5 |
| `mask_mAP50-95` | float | Mask mAP at IoU=0.5:0.95 |
| `dice` | float | Dice coefficient |
| `iou` | float | Intersection over Union |

### Error Types

구조화된 에러 타입:

```python
class ErrorType:
    DATASET_ERROR = 'DatasetError'       # Dataset issues
    CHECKPOINT_ERROR = 'CheckpointError' # Checkpoint issues
    CONFIG_ERROR = 'ConfigError'         # Configuration issues
    RESOURCE_ERROR = 'ResourceError'     # GPU/Memory issues
    NETWORK_ERROR = 'NetworkError'       # S3/HTTP issues
    FRAMEWORK_ERROR = 'FrameworkError'   # Framework-specific
    VALIDATION_ERROR = 'ValidationError' # NaN, divergence
    UNKNOWN_ERROR = 'UnknownError'       # Unexpected
```

---

## Storage Client Specification

### Dual Storage Architecture

```
External Storage (MinIO-Datasets, port 9000)
├── training-datasets/
│   └── datasets/{dataset_id}/
│       ├── images/
│       ├── labels/
│       └── data.yaml

Internal Storage (MinIO-Results, port 9002)
├── training-checkpoints/
│   ├── checkpoints/{job_id}/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── exports/{job_id}/
│   │   ├── model.onnx
│   │   └── metadata.json
│   └── inference-results/{job_id}/
│       ├── images/
│       └── predictions.json
```

### Environment Variables

```bash
# External Storage (Datasets)
EXTERNAL_STORAGE_ENDPOINT=http://localhost:9000
EXTERNAL_STORAGE_ACCESS_KEY=minioadmin
EXTERNAL_STORAGE_SECRET_KEY=minioadmin
EXTERNAL_BUCKET_DATASETS=training-datasets

# Internal Storage (Results)
INTERNAL_STORAGE_ENDPOINT=http://localhost:9002
INTERNAL_STORAGE_ACCESS_KEY=minioadmin
INTERNAL_STORAGE_SECRET_KEY=minioadmin
INTERNAL_BUCKET_CHECKPOINTS=training-checkpoints
```

---

## Implementation Plan

### Phase 1: SDK Core Development (1주)

#### 1.1 SDK 파일 구조

```python
# platform/trainers/common/trainer_sdk.py

"""
Vision AI Training Platform - Trainer SDK

Lightweight SDK for platform communication.
Single file, minimal dependencies.
"""

__version__ = '1.0.0'

# Dependencies
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import httpx
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

# ... implementation
```

#### 1.2 구현 순서

1. **Environment 로딩** - 환경변수에서 설정 로드
2. **HTTP Client** - retry, timeout 설정
3. **S3 Client** - dual storage 지원
4. **Lifecycle functions** - report_* 함수들
5. **Metadata helpers** - create_export_metadata 등

#### 1.3 Deliverables

- [ ] `trainer_sdk.py` 단일 파일 구현
- [ ] Unit tests for all functions
- [ ] Integration tests with mock backend

### Phase 2: Ultralytics Migration (1주)

#### 2.1 Migration Steps

1. `trainer_sdk.py`를 `platform/trainers/ultralytics/`에 복사
2. `train.py` 수정: CallbackClient → SDK
3. `predict.py` 수정: CallbackClient → SDK
4. `export.py` 수정: CallbackClient → SDK
5. `utils.py`에서 사용하지 않는 코드 정리

#### 2.2 Code Comparison

**Before (train.py):**
```python
from utils import DualStorageClient, CallbackClient

storage = DualStorageClient()
callback_client = CallbackClient(callback_url)

# Progress
callback_client.send_progress_sync(job_id, progress_data)

# Completion
await callback_client.send_completion(job_id, completion_data)
```

**After (train.py):**
```python
from trainer_sdk import TrainerSDK

sdk = TrainerSDK()

# Progress
sdk.report_progress(
    epoch=epoch,
    total_epochs=epochs,
    metrics={
        'loss': loss_value,
        'mAP50': map50,
        'mAP50-95': map50_95
    }
)

# Completion
sdk.report_completed(
    final_metrics=final_metrics,
    checkpoints={
        'best': best_checkpoint_uri,
        'last': last_checkpoint_uri
    },
    mlflow_run_id=run_id
)
```

#### 2.3 Deliverables

- [ ] train.py migrated to SDK
- [ ] predict.py migrated to SDK
- [ ] export.py migrated to SDK
- [ ] evaluate.py migrated to SDK
- [ ] E2E tests passing

### Phase 3: Backend Simplification (1주)

#### 3.1 Backend Changes

현재 Backend가 다양한 callback 형식을 처리하고 있습니다. SDK 도입 후:

1. **Callback endpoint 통합**: 모든 callback이 동일한 스키마
2. **에러 처리 개선**: 구조화된 에러 타입으로 정확한 에러 분류
3. **TrainingSubprocessManager 단순화**: 중복 로직 제거

#### 3.2 Deliverables

- [ ] Callback API schema 표준화
- [ ] TrainingSubprocessManager 리팩토링
- [ ] Error handling 개선

### Phase 4: Documentation & Validation (1주)

#### 4.1 Documentation

- [ ] SDK API Reference
- [ ] Migration Guide for existing trainers
- [ ] New Trainer Development Guide
- [ ] Examples for each operation type

#### 4.2 Validation Tools

```python
# platform/trainers/common/validate_trainer.py

def validate_trainer_structure(trainer_dir: Path) -> ValidationResult:
    """Validate trainer directory has required files"""
    pass

def validate_sdk_usage(trainer_dir: Path) -> ValidationResult:
    """Validate trainer uses SDK correctly"""
    pass
```

#### 4.3 Deliverables

- [ ] Complete SDK documentation
- [ ] Validation script
- [ ] Example trainer implementation

---

## Migration Strategy

### Aggressive Migration (No Fallbacks)

**원칙: Fallback 없이 확실하게 마이그레이션**

기존 코드와의 호환성을 유지하려고 하면 어디서 문제가 발생하는지 파악하기 어렵습니다.
에러가 발생하면 그것을 수용하고 명확하게 수정합니다.

#### Migration Steps

1. **Phase 1**: SDK 구현 완료
2. **Phase 2**: Ultralytics trainer를 SDK로 완전 마이그레이션
   - `utils.py`의 `CallbackClient`, `DualStorageClient` → SDK로 대체
   - 기존 코드 제거 (fallback 없음)
   - 에러 발생 시 즉시 수정
3. **Phase 3**: Backend callback handler 단순화
   - SDK 표준 스키마만 처리
   - 레거시 형식 지원 제거
4. **Phase 4**: utils.py 정리
   - SDK로 이전된 기능 완전 제거
   - 남은 기능만 유지

#### Why No Fallbacks?

```python
# ❌ BAD: Fallback으로 문제 숨김
try:
    sdk.report_progress(...)
except:
    callback_client.send_progress_sync(...)  # 어디가 문제인지 모름

# ✅ GOOD: 에러 노출하여 명확하게 수정
sdk.report_progress(...)  # 에러 발생하면 바로 확인 가능
```

**장점:**
- 문제 발생 지점 명확
- 기술 부채 없음
- 깔끔한 코드베이스
- 테스트 신뢰성 향상

**단점 (수용 가능):**
- 마이그레이션 중 일시적 기능 중단 가능
- 즉각적인 에러 수정 필요

**마이그레이션은 개발 환경에서 진행하므로 일시적 중단 허용**

---

## Testing Strategy

### Unit Tests

```python
# tests/test_trainer_sdk.py

def test_report_progress():
    """Test progress reporting with all required fields"""
    pass

def test_report_failed_with_error_type():
    """Test error reporting with structured error types"""
    pass

def test_create_export_metadata():
    """Test metadata creation with required fields"""
    pass
```

### Integration Tests

```python
# tests/test_sdk_integration.py

def test_training_lifecycle():
    """Test complete training lifecycle: started → progress → completed"""
    pass

def test_inference_lifecycle():
    """Test inference lifecycle with result upload"""
    pass

def test_export_lifecycle():
    """Test export with metadata generation"""
    pass
```

### E2E Tests

```bash
# tests/e2e/test_sdk_training.sh

# 1. Start training job via API
# 2. Verify SDK callbacks received
# 3. Verify checkpoints uploaded
# 4. Verify completion callback
```

---

## Data Utility Functions

### Analysis: Dataset Utilities in SDK

현재 `utils.py`의 `convert_diceformat_to_yolo()` 함수를 분석한 결과:

#### 현재 구현

```python
def convert_diceformat_to_yolo(dataset_dir: Path, split_config: Optional[Dict] = None):
    """
    Convert DICEFormat (annotations.json) to YOLO format.

    Creates:
    - labels/*.txt (YOLO format labels)
    - train.txt, val.txt (image lists)
    - data.yaml (dataset config)
    """
```

#### 의존성 분석

| Library | Type | Usage |
|---------|------|-------|
| yaml (PyYAML) | External | data.yaml 생성 |
| json | Built-in | annotations.json 파싱 |
| random | Built-in | Train/val split |
| glob | Built-in | Cache 파일 삭제 |
| pathlib | Built-in | 파일 경로 처리 |

**PyYAML**만 외부 의존성이며, 이는 대부분의 ML 프레임워크에서 이미 사용 중입니다.

### Decision: Include Data Utilities (Required)

데이터 유틸리티를 SDK의 **필수 기능**으로 포함합니다.

#### Rationale

**장점:**
- 모든 trainer가 동일한 데이터 변환 로직 사용
- 데이터 포맷 지원 통일 (DICEFormat, COCO, YOLO)
- 중복 코드 제거
- Split 로직 표준화
- 통신 + 데이터 처리를 하나의 SDK로 통합

**의존성:**
- PyYAML → 모든 ML trainer에 이미 포함되어 있음 (필수)
- Framework별 다른 포맷 요구 → 공통 변환만 SDK에, framework-specific은 trainer에서

### Data Utility Functions Specification

#### convert_dataset()

```python
def convert_dataset(
    dataset_dir: str,
    source_format: str,
    target_format: str,
    split_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convert dataset between formats.

    Args:
        dataset_dir: Dataset directory path
        source_format: 'dice', 'coco', 'yolo', 'imagefolder'
        target_format: 'yolo', 'coco', 'imagefolder'
        split_config: Optional split configuration
            {
                'train_ratio': 0.8,
                'val_ratio': 0.2,
                'seed': 42,
                'splits': {'image_id': 'train'|'val'}  # Pre-defined splits
            }

    Returns:
        Path to converted dataset

    Supported Conversions:
        - dice → yolo (detection)
        - coco → yolo (detection)
        - imagefolder → yolo (classification)
        - yolo → coco (for evaluation tools)

    Example:
        sdk.convert_dataset(
            dataset_dir='/tmp/dataset',
            source_format='dice',
            target_format='yolo',
            split_config={'train_ratio': 0.8, 'val_ratio': 0.2}
        )
    """
```

#### create_data_yaml()

```python
def create_data_yaml(
    dataset_dir: str,
    class_names: List[str],
    train_path: str = 'train.txt',
    val_path: str = 'val.txt',
    test_path: Optional[str] = None
) -> str:
    """
    Create YOLO-format data.yaml file.

    Args:
        dataset_dir: Dataset directory path
        class_names: List of class names (order matters)
        train_path: Path to train image list (relative to data.yaml)
        val_path: Path to val image list
        test_path: Optional path to test image list

    Returns:
        Path to created data.yaml

    Output Format:
        train: train.txt
        val: val.txt
        nc: 3
        names: ['class1', 'class2', 'class3']
    """
```

#### split_dataset()

```python
def split_dataset(
    images: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
    stratify_by: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Split dataset into train/val/test.

    Args:
        images: List of image dictionaries with 'id' key
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        stratify_by: Optional key to stratify by (e.g., 'category_id')

    Returns:
        Dict mapping image_id to split name
        {'img001': 'train', 'img002': 'val', ...}

    Example:
        splits = sdk.split_dataset(
            images=dataset['images'],
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            stratify_by='category_id'
        )
    """
```

#### validate_dataset()

```python
def validate_dataset(
    dataset_dir: str,
    expected_format: str,
    min_images: int = 1,
    check_labels: bool = True
) -> Dict[str, Any]:
    """
    Validate dataset structure and content.

    Args:
        dataset_dir: Dataset directory path
        expected_format: 'yolo', 'coco', 'imagefolder'
        min_images: Minimum required images
        check_labels: Whether to validate labels exist

    Returns:
        {
            'valid': True/False,
            'issues': ['Missing labels for img001.jpg', ...],
            'stats': {
                'total_images': 1000,
                'total_labels': 998,
                'classes': ['person', 'car'],
                'class_distribution': {'person': 5000, 'car': 3000}
            }
        }

    Example:
        result = sdk.validate_dataset(
            dataset_dir='/tmp/dataset',
            expected_format='yolo',
            min_images=100
        )
        if not result['valid']:
            sdk.report_failed('DatasetError', f"Invalid dataset: {result['issues']}")
    """
```

#### clean_dataset_cache()

```python
def clean_dataset_cache(dataset_dir: str) -> int:
    """
    Remove stale cache files from dataset directory.

    YOLO creates .cache files that can cause issues when dataset is modified.
    This should be called after dataset conversion or modification.

    Args:
        dataset_dir: Dataset directory path

    Returns:
        Number of cache files deleted

    Example:
        sdk.convert_dataset('/tmp/dataset', 'dice', 'yolo')
        deleted = sdk.clean_dataset_cache('/tmp/dataset')
        logger.info(f"Deleted {deleted} cache files")
    """
```

### Format Specifications

#### DICEFormat (Platform Standard)

```json
{
  "images": [
    {"id": 1, "file_name": "images/001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}
  ],
  "categories": [
    {"id": 1, "name": "person"}
  ]
}
```

#### YOLO Format

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── train.txt
├── val.txt
└── data.yaml
```

#### COCO Format

```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...],
  "info": {...},
  "licenses": [...]
}
```

### SDK Structure (Final)

모든 기능이 포함된 SDK 구조:

```python
# platform/trainers/common/trainer_sdk.py

class TrainerSDK:
    """Main SDK class"""

    # Core functions (Lifecycle)
    def report_started(self): ...
    def report_progress(self): ...
    def report_completed(self): ...
    def report_failed(self): ...

    # Inference & Export
    def report_inference_completed(self): ...
    def report_export_completed(self): ...

    # Storage functions
    def upload_checkpoint(self): ...
    def download_checkpoint(self): ...
    def download_dataset(self): ...
    def upload_file(self): ...

    # MLflow integration
    def setup_mlflow(self): ...
    def log_metrics(self): ...
    def log_params(self): ...
    def log_artifact(self): ...
    def end_mlflow_run(self): ...

    # Logging functions
    def get_logger(self): ...
    def log_event(self): ...

    # Data utility functions
    def convert_dataset(self): ...
    def create_data_yaml(self): ...
    def split_dataset(self): ...
    def validate_dataset(self): ...
    def clean_dataset_cache(self): ...
```

### Required Dependencies

```python
# trainer_sdk.py

# All dependencies are required
import httpx     # HTTP client with retry
import boto3     # S3 storage
import yaml      # Dataset configuration (PyYAML)
import mlflow    # Experiment tracking
import logging   # Built-in
import json      # Built-in
import random    # Built-in
from pathlib import Path  # Built-in

class TrainerSDK:
    """
    Complete SDK with all functions available.
    No optional imports - all dependencies are required.
    """
    pass
```

**참고:** 모든 ML trainer는 이미 PyYAML과 MLflow를 사용하므로 추가 의존성 부담이 없습니다.

---

## Appendix

### A. Full SDK Example

```python
#!/usr/bin/env python3
"""Example trainer using SDK"""

from trainer_sdk import TrainerSDK

def main():
    sdk = TrainerSDK()

    try:
        # 1. Report started
        sdk.report_started()

        # 2. Download dataset
        dataset_dir = sdk.download_dataset(
            os.getenv('DATASET_ID'),
            '/tmp/dataset'
        )

        # 3. Training loop
        model = load_model(os.getenv('MODEL_NAME'))

        for epoch in range(1, epochs + 1):
            metrics = train_one_epoch(model, dataset_dir)

            # Report progress
            sdk.report_progress(
                epoch=epoch,
                total_epochs=epochs,
                metrics=metrics
            )

            # Save checkpoint
            if epoch % save_interval == 0:
                sdk.upload_checkpoint(
                    f'/tmp/checkpoints/epoch_{epoch}.pt',
                    f'epoch_{epoch}',
                    metrics
                )

        # 4. Report validation
        val_metrics = validate(model)
        sdk.report_validation(
            epoch=epochs,
            task_type='detection',
            primary_metric=('mAP50-95', val_metrics['mAP50-95']),
            all_metrics=val_metrics
        )

        # 5. Upload final checkpoints
        best_uri = sdk.upload_checkpoint('/tmp/best.pt', 'best', val_metrics)
        last_uri = sdk.upload_checkpoint('/tmp/last.pt', 'last', val_metrics)

        # 6. Report completed
        sdk.report_completed(
            final_metrics=val_metrics,
            checkpoints={'best': best_uri, 'last': last_uri}
        )

    except Exception as e:
        sdk.report_failed(
            error_type='FrameworkError',
            message=str(e),
            traceback=traceback.format_exc()
        )
        sys.exit(1)

if __name__ == '__main__':
    main()
```

### B. Related Documents

- [EXPORT_CONVENTION.md](../EXPORT_CONVENTION.md) - Export convention
- [TRAINER_MARKETPLACE_VISION.md](../planning/TRAINER_MARKETPLACE_VISION.md) - Marketplace design
- [DUAL_STORAGE.md](../DUAL_STORAGE.md) - Storage architecture
- [IMPLEMENTATION_TO_DO_LIST.md](../todo/IMPLEMENTATION_TO_DO_LIST.md) - TODO list

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-19 | Initial design specification |
