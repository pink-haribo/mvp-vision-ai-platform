#!/usr/bin/env python3
"""
Vision AI Training Platform - Unified Trainer SDK

Lightweight SDK for platform communication.
Single file, minimal dependencies.

Usage:
    from trainer_sdk import TrainerSDK

    sdk = TrainerSDK()
    sdk.report_started()

    # Download and convert dataset
    dataset_dir = sdk.download_dataset(dataset_id, '/tmp/dataset')
    sdk.convert_dataset(dataset_dir, 'dice', 'yolo')  # or 'coco'

    # Training loop
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(...)
        sdk.report_progress(epoch, epochs, metrics)

    # Upload checkpoints and report completion
    best_uri = sdk.upload_checkpoint('/tmp/best.pt', 'best')
    sdk.report_completed(final_metrics, checkpoints={'best': best_uri})

Dependencies:
    - httpx: HTTP client with retry support
    - boto3: S3-compatible storage
    - PyYAML: Dataset configuration

Supported Dataset Conversions:
    - dice -> yolo: For Ultralytics YOLO, mmyolo
    - dice -> coco: For MMDetection, VFM, mmpretrain, mmseg

Version: 2.0.0
Date: 2025-01-21
"""

__version__ = '2.0.0'

import glob as glob_module
import hashlib
import json
import logging
import os
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import httpx
import yaml
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

# ClearML (optional dependency - Phase 12.2)
try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Error Types
# =============================================================================

class ErrorType:
    """Structured error types for report_failed()"""
    DATASET_ERROR = 'DatasetError'       # Dataset not found, corrupted, invalid format
    CHECKPOINT_ERROR = 'CheckpointError' # Checkpoint not found, corrupted, incompatible
    CONFIG_ERROR = 'ConfigError'         # Invalid configuration, unsupported parameters
    RESOURCE_ERROR = 'ResourceError'     # Out of memory, GPU not available
    NETWORK_ERROR = 'NetworkError'       # S3 connection failed, callback failed
    FRAMEWORK_ERROR = 'FrameworkError'   # Framework-specific error (YOLO, timm, etc.)
    VALIDATION_ERROR = 'ValidationError' # Validation failed, NaN loss
    UNKNOWN_ERROR = 'UnknownError'       # Unexpected error


# =============================================================================
# Storage Clients
# =============================================================================

class StorageClient:
    """S3-compatible storage client for single bucket operations"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        self.endpoint = endpoint
        self.bucket = bucket
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(
                signature_version='s3v4',
                retries={'max_attempts': 3, 'mode': 'standard'}
            )
        )

    def download_file(self, s3_key: str, local_path: str) -> str:
        """Download file from S3"""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, s3_key, local_path)
        return local_path

    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        content_type: str = 'application/octet-stream'
    ) -> str:
        """Upload file to S3"""
        extra_args = {'ContentType': content_type}
        self.client.upload_file(local_path, self.bucket, s3_key, ExtraArgs=extra_args)
        return f"s3://{self.bucket}/{s3_key}"

    def download_directory(self, prefix: str, local_dir: str) -> str:
        """Download all files under a prefix"""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                relative_path = s3_key[len(prefix):].lstrip('/')
                if relative_path:  # Skip the prefix directory itself
                    local_path = local_dir / relative_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    self.client.download_file(self.bucket, s3_key, str(local_path))

        return str(local_dir)


# =============================================================================
# Main SDK Class
# =============================================================================

class TrainerSDK:
    """
    Unified SDK for platform communication.

    Features:
    - Minimal dependencies (httpx, boto3, yaml)
    - Backend-proxied observability (MLflow, Loki, Prometheus handled by Backend)
    - Standardized callbacks and error types
    - Dual storage support (External for datasets, Internal for checkpoints)
    - Dataset caching (Phase 12.9)
    - Multi-format dataset conversion (YOLO, COCO)

    Environment Variables:
        Required:
        - CALLBACK_URL: Backend API base URL
        - JOB_ID: Training/Inference/Export job ID

        Job Context:
        - TASK_TYPE: Task type (detection, classification, segmentation, pose)
        - MODEL_NAME: Model name (e.g., yolo11n, vfm_v1_l)
        - FRAMEWORK: Framework name (ultralytics, vfm-v1, mmdet, etc.)
        - DATASET_ID: Dataset ID

        External Storage (Datasets):
        - EXTERNAL_STORAGE_ENDPOINT: S3 endpoint (default: http://localhost:9000)
        - EXTERNAL_STORAGE_ACCESS_KEY: Access key (default: minioadmin)
        - EXTERNAL_STORAGE_SECRET_KEY: Secret key (default: minioadmin)
        - EXTERNAL_BUCKET_DATASETS: Bucket name (default: training-datasets)

        Internal Storage (Checkpoints/Results):
        - INTERNAL_STORAGE_ENDPOINT: S3 endpoint (default: http://localhost:9002)
        - INTERNAL_STORAGE_ACCESS_KEY: Access key (default: minioadmin)
        - INTERNAL_STORAGE_SECRET_KEY: Secret key (default: minioadmin)
        - INTERNAL_BUCKET_CHECKPOINTS: Bucket name (default: training-checkpoints)

        Dataset Caching (Phase 12.9):
        - SNAPSHOT_ID: Dataset snapshot ID (for cache key)
        - DATASET_VERSION_HASH: Dataset version hash (for cache verification)
        - DATASET_CACHE_DIR: Shared dataset cache directory (default: /tmp/datasets)
        - DATASET_CACHE_MAX_GB: Max cache size in GB (default: 50)
    """

    # Dataset caching defaults (can be overridden by environment variables)
    DEFAULT_DATASET_CACHE_DIR = "/tmp/datasets"
    DEFAULT_CACHE_MAX_SIZE_GB = 50

    def __init__(self):
        """Initialize SDK from environment variables"""
        # Required environment variables
        self.callback_url = os.getenv('CALLBACK_URL')
        self.job_id = os.getenv('JOB_ID')

        if not self.callback_url:
            raise ValueError("CALLBACK_URL environment variable is required")
        if not self.job_id:
            raise ValueError("JOB_ID environment variable is required")

        # Track operation type for proper callback routing
        self._operation_type = 'training'

        # HTTP client with extended timeout for K8s environments
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(
                connect=15.0,
                read=60.0,
                write=30.0,
                pool=10.0
            ),
            follow_redirects=True
        )

        # Retry configuration
        self._max_retries = int(os.getenv('CALLBACK_MAX_RETRIES', '3'))
        self._retry_base_delay = float(os.getenv('CALLBACK_RETRY_DELAY', '2.0'))

        # Initialize storage clients
        self._init_storage_clients()

        # Initialize dataset cache configuration (Phase 12.9)
        cache_dir = os.getenv('DATASET_CACHE_DIR', self.DEFAULT_DATASET_CACHE_DIR)
        self.SHARED_DATASET_CACHE = Path(cache_dir)
        self.CACHE_MAX_SIZE_GB = int(os.getenv('DATASET_CACHE_MAX_GB', self.DEFAULT_CACHE_MAX_SIZE_GB))
        self.CACHE_METADATA_FILE = self.SHARED_DATASET_CACHE / ".cache_metadata.json"

        # Initialize log buffer
        self._log_buffer: List[Dict[str, Any]] = []
        self._log_buffer_max = 50
        self._log_flush_interval = 5.0

        logger.info(f"TrainerSDK initialized for job {self.job_id}")
        logger.info(f"Callback URL: {self.callback_url}")

    # =========================================================================
    # SDK Properties (Training Job Context) - No hardcoded defaults
    # =========================================================================

    @property
    def task_type(self) -> str:
        """Get task type from environment variable"""
        return os.getenv('TASK_TYPE', 'detection')

    @property
    def model_name(self) -> str:
        """Get model name from environment variable (no default - must be set)"""
        return os.getenv('MODEL_NAME', '')

    @property
    def dataset_id(self) -> str:
        """Get dataset ID from environment variable"""
        return os.getenv('DATASET_ID', '')

    @property
    def snapshot_id(self) -> str:
        """Get dataset snapshot ID from environment variable (Phase 12.9)"""
        return os.getenv('SNAPSHOT_ID', '')

    @property
    def dataset_version_hash(self) -> str:
        """Get dataset version hash from environment variable (Phase 12.9)"""
        return os.getenv('DATASET_VERSION_HASH', '')

    @property
    def framework(self) -> str:
        """Get framework from environment variable (no default - must be set)"""
        return os.getenv('FRAMEWORK', '')

    # =========================================================================
    # Config Loading Methods
    # =========================================================================

    def get_basic_config(self) -> Dict[str, Any]:
        """
        Load basic config with priority: env vars > CONFIG JSON > defaults.

        Returns:
            Dict with basic config parameters
        """
        config_json = {}
        config_str = os.getenv('CONFIG', '{}')
        try:
            config_json = json.loads(config_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse CONFIG JSON: {e}")

        def get_config_value(direct_env: str, config_key: str, legacy_env: str, default: Any, value_type=str) -> Any:
            if direct_env and os.getenv(direct_env):
                value = os.getenv(direct_env)
            elif config_key and config_key in config_json:
                value = config_json[config_key]
            elif legacy_env and os.getenv(legacy_env):
                value = os.getenv(legacy_env)
            else:
                value = default

            if value_type == int:
                return int(value)
            elif value_type == float:
                return float(value)
            elif value_type == bool:
                return str(value).lower() == 'true' if isinstance(value, str) else bool(value)
            else:
                return str(value)

        return {
            'imgsz': get_config_value('IMGSZ', 'imgsz', 'CONFIG_IMGSZ', '640', int),
            'epochs': get_config_value('EPOCHS', 'epochs', 'CONFIG_EPOCHS', '100', int),
            'batch': get_config_value('BATCH_SIZE', 'batch', 'CONFIG_BATCH', '16', int),
            'lr0': get_config_value('LEARNING_RATE', 'learning_rate', 'CONFIG_LR0', '0.01', float),
            'optimizer': get_config_value('OPTIMIZER', 'optimizer', 'CONFIG_OPTIMIZER', 'auto', str),
            'augment': get_config_value('AUGMENT', 'augment', 'CONFIG_AUGMENT', 'True', bool),
            'device': get_config_value('DEVICE', 'device', 'CONFIG_DEVICE', '0', str),
            'workers': get_config_value('WORKERS', 'workers', 'CONFIG_WORKERS', '8', int),
            'patience': get_config_value('PATIENCE', 'patience', 'CONFIG_PATIENCE', '50', int),
        }

    def get_advanced_config(self) -> Dict[str, Any]:
        """Load advanced config from CONFIG JSON."""
        config_str = os.getenv('CONFIG', '{}')
        try:
            config_json = json.loads(config_str)
            if 'advanced_config' in config_json:
                return config_json['advanced_config']
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse CONFIG JSON: {e}")

        advanced_json = os.getenv('ADVANCED_CONFIG', '{}')
        try:
            return json.loads(advanced_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse ADVANCED_CONFIG: {e}")
            return {}

    def get_full_config(self) -> Dict[str, Any]:
        """Get merged basic and advanced config."""
        return {
            'basic': self.get_basic_config(),
            'advanced': self.get_advanced_config()
        }

    def _init_storage_clients(self):
        """Initialize dual storage clients"""
        self.external_storage = StorageClient(
            endpoint=os.getenv('EXTERNAL_STORAGE_ENDPOINT', 'http://localhost:9000'),
            access_key=os.getenv('EXTERNAL_STORAGE_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('EXTERNAL_STORAGE_SECRET_KEY', 'minioadmin'),
            bucket=os.getenv('EXTERNAL_BUCKET_DATASETS', 'training-datasets')
        )

        self.internal_storage = StorageClient(
            endpoint=os.getenv('INTERNAL_STORAGE_ENDPOINT', 'http://localhost:9002'),
            access_key=os.getenv('INTERNAL_STORAGE_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('INTERNAL_STORAGE_SECRET_KEY', 'minioadmin'),
            bucket=os.getenv('INTERNAL_BUCKET_CHECKPOINTS', 'training-checkpoints')
        )

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format"""
        return datetime.now(timezone.utc).isoformat()

    def _send_callback(self, endpoint: str, data: Dict[str, Any]) -> None:
        """Send callback to Backend with retry and exponential backoff."""
        url = f"{self.callback_url.rstrip('/')}/{endpoint.lstrip('/')}"
        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self.http_client.post(url, json=data)
                response.raise_for_status()
                logger.debug(f"Callback sent: {endpoint} -> {response.status_code}")
                return

            except httpx.HTTPStatusError as e:
                if 400 <= e.response.status_code < 500:
                    logger.error(f"Callback failed ({e.response.status_code}): {url}")
                    logger.error(f"Response: {e.response.text}")
                    raise

                last_exception = e
                logger.warning(
                    f"Callback failed ({e.response.status_code}), "
                    f"attempt {attempt + 1}/{self._max_retries + 1}: {url}"
                )

            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                last_exception = e
                logger.warning(
                    f"Callback request failed ({type(e).__name__}), "
                    f"attempt {attempt + 1}/{self._max_retries + 1}: {url}"
                )

            except httpx.RequestError as e:
                last_exception = e
                logger.warning(
                    f"Callback request error ({type(e).__name__}), "
                    f"attempt {attempt + 1}/{self._max_retries + 1}: {url}"
                )

            if attempt < self._max_retries:
                delay = self._retry_base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)

        logger.error(f"Callback failed after {self._max_retries + 1} attempts: {url}")
        raise last_exception

    # =========================================================================
    # Lifecycle Functions
    # =========================================================================

    def report_started(self, operation_type: str = 'training', total_epochs: Optional[int] = None) -> None:
        """Report job started."""
        self._operation_type = operation_type

        if operation_type == 'inference':
            data = {
                'type': 'started',
                'job_id': int(self.job_id),
                'operation_type': operation_type,
                'timestamp': self._get_timestamp()
            }
            endpoint = f'/test_inference/inference/jobs/{self.job_id}/callback/started'
        elif operation_type == 'export':
            data = {
                'type': 'started',
                'job_id': int(self.job_id),
                'operation_type': operation_type,
                'timestamp': self._get_timestamp()
            }
            endpoint = f'/export/jobs/{self.job_id}/callback/started'
        else:
            if total_epochs is None:
                config = self.get_basic_config()
                total_epochs = config.get('epochs', 100)

            self._total_epochs = total_epochs

            data = {
                'job_id': int(self.job_id),
                'status': 'running',
                'current_epoch': 0,
                'total_epochs': total_epochs,
                'progress_percent': 0.0,
                'metrics': None,
                'checkpoint_path': None
            }
            endpoint = f'/training/jobs/{self.job_id}/callback/progress'

        self._send_callback(endpoint, data)
        logger.info(f"Reported {operation_type} started for job {self.job_id}")

    def report_progress(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float],
        extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Report training progress."""
        progress_percent = (epoch / total_epochs) * 100

        callback_metrics = {
            'loss': metrics.get('loss'),
            'accuracy': metrics.get('accuracy') or metrics.get('mAP50-95'),
            'learning_rate': metrics.get('learning_rate') or metrics.get('lr'),
            'extra_metrics': metrics
        }

        data = {
            'job_id': int(self.job_id),
            'status': 'running',
            'current_epoch': epoch,
            'total_epochs': total_epochs,
            'progress_percent': progress_percent,
            'metrics': callback_metrics,
        }

        if extra_data:
            data['extra_data'] = extra_data

        self._send_callback(f'/training/jobs/{self.job_id}/callback/progress', data)
        logger.debug(f"Reported progress: epoch {epoch}/{total_epochs} ({progress_percent:.1f}%)")

        if CLEARML_AVAILABLE:
            try:
                task = Task.current_task()
                if task:
                    for metric_name, metric_value in metrics.items():
                        if metric_value is not None:
                            if '/' in metric_name:
                                series, title = metric_name.split('/', 1)
                            else:
                                title = metric_name
                                series = 'train'

                            task.get_logger().report_scalar(
                                title=title,
                                series=series,
                                value=float(metric_value),
                                iteration=epoch
                            )
                    logger.debug(f"Logged {len(metrics)} metrics to ClearML task")
            except Exception as e:
                logger.warning(f"Failed to log metrics to ClearML: {e}")

    def report_validation(
        self,
        epoch: int,
        task_type: str,
        primary_metric: Tuple[str, float],
        all_metrics: Dict[str, float],
        class_names: Optional[List[str]] = None,
        visualization_urls: Optional[Dict[str, str]] = None,
        per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        is_best: bool = False
    ) -> None:
        """Report validation results."""
        data = {
            'type': 'validation',
            'job_id': int(self.job_id),
            'epoch': epoch,
            'task_type': task_type,
            'primary_metric_name': primary_metric[0],
            'primary_metric_value': primary_metric[1],
            'metrics': all_metrics,
            'is_best': is_best,
            'timestamp': self._get_timestamp()
        }

        if class_names:
            data['class_names'] = class_names
        if visualization_urls:
            data['visualization_urls'] = visualization_urls
        if per_class_metrics:
            data['per_class_metrics'] = per_class_metrics

        self._send_callback(f'/validation/jobs/{self.job_id}/results', data)
        logger.info(f"Reported validation: {primary_metric[0]}={primary_metric[1]:.4f}")

    def report_completed(
        self,
        final_metrics: Dict[str, float],
        checkpoints: Optional[Dict[str, str]] = None,
        total_epochs: Optional[int] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Report job completed successfully."""
        callback_metrics = {
            'loss': final_metrics.get('loss'),
            'accuracy': final_metrics.get('accuracy') or final_metrics.get('mAP50-95'),
            'learning_rate': final_metrics.get('learning_rate'),
            'extra_metrics': final_metrics
        }

        data = {
            'job_id': int(self.job_id),
            'status': 'completed',
            'total_epochs_completed': total_epochs or 0,
            'final_metrics': callback_metrics,
            'exit_code': 0,
        }

        if checkpoints:
            data['best_checkpoint_path'] = checkpoints.get('best')
            data['last_checkpoint_path'] = checkpoints.get('last')

        if extra_data:
            data['extra_data'] = extra_data

        self._send_callback(f'/training/jobs/{self.job_id}/callback/completion', data)
        logger.info(f"Reported completion for job {self.job_id}")

    def report_failed(
        self,
        error_type: str,
        message: str,
        traceback: Optional[str] = None,
        epochs_completed: int = 0
    ) -> None:
        """Report job failed."""
        data = {
            'job_id': int(self.job_id),
            'status': 'failed',
            'total_epochs_completed': epochs_completed,
            'error_message': f"[{error_type}] {message}",
            'exit_code': 1,
        }

        if traceback:
            data['traceback'] = traceback

        if self._operation_type == 'inference':
            endpoint = f'/test_inference/inference/jobs/{self.job_id}/callback/completion'
        elif self._operation_type == 'export':
            endpoint = f'/export/jobs/{self.job_id}/callback/completion'
        else:
            endpoint = f'/training/jobs/{self.job_id}/callback/completion'

        self._send_callback(endpoint, data)
        logger.error(f"Reported failure: {error_type} - {message}")

    # =========================================================================
    # Inference & Export Functions
    # =========================================================================

    def report_inference_completed(
        self,
        total_images: int,
        total_time_ms: float,
        results: List[Dict[str, Any]],
        result_urls: Optional[Dict[str, str]] = None
    ) -> None:
        """Report inference job completed."""
        avg_time_ms = total_time_ms / total_images if total_images > 0 else 0

        data = {
            'type': 'inference_completed',
            'job_id': int(self.job_id),
            'status': 'completed',
            'total_images': total_images,
            'total_inference_time_ms': total_time_ms,
            'avg_inference_time_ms': avg_time_ms,
            'results': results,
            'timestamp': self._get_timestamp()
        }

        if result_urls:
            data['result_urls'] = result_urls

        self._send_callback(f'/test_inference/inference/jobs/{self.job_id}/callback/completion', data)
        logger.info(f"Reported inference completed: {total_images} images in {total_time_ms:.1f}ms")

    def report_export_completed(
        self,
        export_format: str,
        output_s3_uri: str,
        file_size_bytes: int,
        metadata: Dict[str, Any]
    ) -> None:
        """Report export job completed."""
        data = {
            'type': 'export_completed',
            'job_id': int(self.job_id),
            'status': 'completed',
            'export_format': export_format,
            'output_s3_uri': output_s3_uri,
            'file_size_bytes': file_size_bytes,
            'metadata': metadata,
            'timestamp': self._get_timestamp()
        }

        self._send_callback(f'/export/jobs/{self.job_id}/callback/completion', data)
        logger.info(f"Reported export completed: {export_format} ({file_size_bytes} bytes)")

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
        """Create standardized export metadata."""
        metadata = {
            'framework': framework,
            'model_name': model_name,
            'export_format': export_format,
            'task_type': task_type,
            'input_shape': input_shape,
            'input_dtype': 'float32',
            'output_shape': output_shape,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        if class_names:
            metadata['class_names'] = class_names
            metadata['num_classes'] = len(class_names)

        if preprocessing:
            metadata['preprocessing'] = preprocessing

        if postprocessing:
            metadata['postprocessing'] = postprocessing

        if export_config:
            metadata['export_config'] = export_config

        metadata.update(extra)
        return metadata

    # =========================================================================
    # Storage Functions
    # =========================================================================

    def upload_checkpoint(
        self,
        local_path: str,
        checkpoint_type: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Upload checkpoint to Internal Storage."""
        filename = Path(local_path).name
        s3_key = f"checkpoints/{self.job_id}/{checkpoint_type}.pt"

        content_type = 'application/octet-stream'
        if filename.endswith('.pt') or filename.endswith('.pth'):
            content_type = 'application/x-pytorch'

        s3_uri = self.internal_storage.upload_file(local_path, s3_key, content_type)
        logger.info(f"Uploaded checkpoint: {checkpoint_type} -> {s3_uri}")

        if metrics:
            self.log_event(
                'checkpoint',
                f'Saved {checkpoint_type} checkpoint',
                data={'s3_uri': s3_uri, 'metrics': metrics}
            )

        return s3_uri

    def download_checkpoint(self, s3_uri: str, local_path: str) -> str:
        """Download checkpoint from Internal Storage."""
        parts = s3_uri.replace('s3://', '').split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        bucket, key = parts
        return self.internal_storage.download_file(key, local_path)

    def download_dataset(self, dataset_id: str, dest_dir: str) -> str:
        """Download dataset from External Storage."""
        prefix = f"datasets/{dataset_id}/"
        local_dir = self.external_storage.download_directory(prefix, dest_dir)
        logger.info(f"Downloaded dataset {dataset_id} to {local_dir}")
        return local_dir

    def download_dataset_selective(self, dataset_id: str, dest_dir: str) -> str:
        """Download only images listed in annotations (Phase 12.9.2)."""
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)

        dataset_prefix = f"datasets/{dataset_id}/"
        logger.info(f"Listing annotation files in S3: {dataset_prefix}")

        try:
            response = self.external_storage.client.list_objects_v2(
                Bucket=self.external_storage.bucket,
                Prefix=dataset_prefix
            )
            objects = response.get('Contents', [])
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise

        annotation_keys = [
            obj['Key'] for obj in objects
            if obj['Key'].split('/')[-1].startswith('annotations')
            and obj['Key'].endswith('.json')
        ]

        if not annotation_keys:
            raise ValueError(
                f"No annotation files found in {dataset_prefix}. "
                f"Expected files like 'annotations_detection.json'"
            )

        logger.info(f"Found {len(annotation_keys)} annotation files")

        primary_annotation_path = None
        for annotation_key in annotation_keys:
            filename = annotation_key.split('/')[-1]
            local_path = dest_path / filename

            logger.info(f"Downloading annotation: {filename}")
            self.external_storage.download_file(annotation_key, str(local_path))

            if 'detection' in filename:
                primary_annotation_path = local_path
            elif primary_annotation_path is None:
                primary_annotation_path = local_path

        if primary_annotation_path is None:
            raise ValueError("No annotation file downloaded")

        with open(primary_annotation_path) as f:
            data = json.load(f)

        images_to_download = []
        for img in data['images']:
            images_to_download.append(img['file_name'])

        logger.info(f"Found {len(images_to_download)} labeled images to download")

        storage_info = data.get('storage_info', {})
        image_root = storage_info.get('image_root', f'datasets/{dataset_id}/images/')

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []

            for file_name in images_to_download:
                s3_key = f"{image_root}{file_name}"
                local_path = Path(dest_dir) / "images" / file_name
                local_path.parent.mkdir(parents=True, exist_ok=True)

                future = executor.submit(
                    self._download_single_file,
                    s3_key,
                    str(local_path)
                )
                futures.append((file_name, future))

            completed = 0
            for file_name, future in futures:
                try:
                    future.result()
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Downloaded {completed}/{len(images_to_download)} images")
                except Exception as e:
                    logger.error(f"Failed to download {file_name}: {e}")
                    raise

        logger.info(f"Downloaded {len(images_to_download)} images selectively")
        return dest_dir

    def _download_single_file(self, s3_key: str, local_path: str):
        """Download single file from S3"""
        self.external_storage.download_file(s3_key, local_path)

    def download_dataset_from_annotation(
        self,
        annotation_data: Dict[str, Any],
        dest_dir: str,
        format_type: str = 'labelme'
    ) -> str:
        """
        Download dataset based on annotation data content.

        This function downloads images and labels based on the paths specified
        in the annotation data, rather than from a predefined dataset structure.

        Args:
            annotation_data: Annotation data dict (parsed from annotation.json)
            dest_dir: Destination directory for downloaded files
            format_type: Annotation format type ('labelme', 'coco')

        Returns:
            Path to the downloaded dataset directory

        Raises:
            ValueError: If format_type is not supported
        """
        if format_type == 'labelme':
            return self._download_labelme_format(annotation_data, dest_dir)
        elif format_type == 'coco':
            return self._download_coco_format(annotation_data, dest_dir)
        else:
            raise ValueError(f"Unsupported annotation format: {format_type}")

    def _parse_s3_uri(self, s3_uri: str) -> Tuple[str, str]:
        """
        Parse S3 URI to extract bucket and key.

        Args:
            s3_uri: Full S3 URI (e.g., 's3://bucket-name/path/to/file')

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If URI format is invalid
        """
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

        path = s3_uri[5:]  # Remove 's3://'
        parts = path.split('/', 1)

        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format (missing key): {s3_uri}")

        bucket, key = parts
        return bucket, key

    def _download_labelme_format(
        self,
        annotation_data: Dict[str, Any],
        dest_dir: str
    ) -> str:
        """
        Download dataset from LabelMe format annotation data.

        LabelMe format:
        {
            "data_summary": [
                {"img_path": "s3://bucket/path/image.jpg", "label_path": "s3://bucket/path/label.json"},
                ...
            ]
        }

        Args:
            annotation_data: LabelMe annotation data dict
            dest_dir: Destination directory

        Returns:
            Path to the downloaded dataset directory
        """
        dest_path = Path(dest_dir)
        images_dir = dest_path / "images"
        labels_dir = dest_path / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        data_summary = annotation_data.get('data_summary', [])

        if not data_summary:
            raise ValueError("No data_summary found in annotation data")

        logger.info(f"Downloading LabelMe dataset: {len(data_summary)} items")

        download_tasks = []
        for item in data_summary:
            img_path = item.get('img_path')
            label_path = item.get('label_path')

            if img_path:
                _, img_key = self._parse_s3_uri(img_path)
                img_filename = Path(img_key).name
                local_img_path = images_dir / img_filename
                download_tasks.append(('image', img_key, str(local_img_path)))

            if label_path:
                _, label_key = self._parse_s3_uri(label_path)
                label_filename = Path(label_key).name
                local_label_path = labels_dir / label_filename
                download_tasks.append(('label', label_key, str(local_label_path)))

        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for task_type, s3_key, local_path in download_tasks:
                future = executor.submit(self._download_single_file, s3_key, local_path)
                futures[future] = (task_type, s3_key, local_path)

            for future in as_completed(futures):
                task_type, s3_key, local_path = futures[future]
                try:
                    future.result()
                    completed += 1
                    if completed % 20 == 0:
                        logger.info(f"Downloaded {completed}/{len(download_tasks)} files")
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to download {task_type} {s3_key}: {e}")

        if failed > 0:
            raise RuntimeError(f"Failed to download {failed} files")

        logger.info(
            f"LabelMe dataset download completed: "
            f"{len(data_summary)} items ({completed} files)"
        )

        return str(dest_path)

    def _download_coco_format(
        self,
        annotation_data: Dict[str, Any],
        dest_dir: str
    ) -> str:
        """
        Download dataset from COCO format annotation data.

        COCO format:
        {
            "images": [{"id": 1, "file_name": "s3://bucket/path/image.jpg", ...}, ...],
            "annotations": [...],
            "categories": [...]
        }

        Args:
            annotation_data: COCO annotation data dict
            dest_dir: Destination directory

        Returns:
            Path to the downloaded dataset directory

        Note:
            This is a placeholder for future implementation.
        """
        raise NotImplementedError(
            "COCO format download is not yet implemented. "
            "Use 'labelme' format_type for now."
        )

    # =========================================================================
    # Dataset Caching Methods (Phase 12.9)
    # =========================================================================

    def download_dataset_with_cache(
        self,
        snapshot_id: str,
        dataset_id: str,
        dataset_version_hash: str,
        dest_dir: str
    ) -> str:
        """Download dataset with caching support (Phase 12.9)."""
        cache_key = f"{dataset_version_hash[:16]}"
        cache_dir = self.SHARED_DATASET_CACHE / cache_key

        if self._is_cache_complete(cache_dir):
            logger.info(f"Cache HIT: {cache_key}")
            self._update_last_accessed(cache_key)
            return self._link_to_cache(cache_dir, dest_dir)

        logger.info(f"Cache MISS: {cache_key}, downloading...")

        if cache_dir.exists():
            logger.warning(f"Removing incomplete cache: {cache_key}")
            shutil.rmtree(cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)

        self.download_dataset_selective(dataset_id=dataset_id, dest_dir=str(cache_dir))

        self._mark_cache_complete(cache_dir)

        self._update_cache_metadata(cache_key, {
            'snapshot_id': snapshot_id,
            'dataset_id': dataset_id,
            'dataset_version_hash': dataset_version_hash,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_accessed': datetime.now(timezone.utc).isoformat(),
            'size_bytes': self._calculate_dir_size(cache_dir)
        })

        self._enforce_cache_size_limit()

        return self._link_to_cache(cache_dir, dest_dir)

    def _is_cache_complete(self, cache_dir: Path) -> bool:
        """Check if cache download completed successfully."""
        return (cache_dir / ".complete").exists()

    def _mark_cache_complete(self, cache_dir: Path):
        """Mark cache as complete after successful download."""
        (cache_dir / ".complete").touch()
        logger.info(f"Cache marked complete: {cache_dir.name}")

    def _link_to_cache(self, cache_dir: Path, dest_dir: str) -> str:
        """Setup job dataset directory from cache."""
        job_dataset_dir = Path(dest_dir) / "dataset"

        job_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

        if job_dataset_dir.exists():
            if job_dataset_dir.is_symlink():
                job_dataset_dir.unlink()
            else:
                shutil.rmtree(job_dataset_dir)

        job_dataset_dir.mkdir(parents=True, exist_ok=True)

        for ann_file in cache_dir.glob("annotations*.json"):
            shutil.copy2(ann_file, job_dataset_dir / ann_file.name)
            logger.debug(f"Copied annotation: {ann_file.name}")

        cache_images_dir = cache_dir / "images"
        job_images_dir = job_dataset_dir / "images"

        if cache_images_dir.exists():
            try:
                job_images_dir.symlink_to(cache_images_dir, target_is_directory=True)
                logger.info(f"Linked images: {job_images_dir} -> {cache_images_dir}")
            except (OSError, NotImplementedError) as e:
                logger.warning(f"Symlink failed ({e}), copying images directory")
                shutil.copytree(cache_images_dir, job_images_dir)
                logger.info(f"Copied images from cache")
        else:
            logger.warning(f"No images directory in cache: {cache_images_dir}")

        logger.info(f"Dataset setup complete: {job_dataset_dir}")
        return str(job_dataset_dir)

    def _update_cache_metadata(self, cache_key: str, metadata: dict):
        """Update cache metadata JSON file."""
        self.SHARED_DATASET_CACHE.mkdir(parents=True, exist_ok=True)

        if self.CACHE_METADATA_FILE.exists():
            with open(self.CACHE_METADATA_FILE) as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}

        all_metadata[cache_key] = metadata

        with open(self.CACHE_METADATA_FILE, 'w') as f:
            json.dump(all_metadata, f, indent=2)

    def _update_last_accessed(self, cache_key: str):
        """Update last_accessed timestamp for cache entry."""
        if not self.CACHE_METADATA_FILE.exists():
            return

        with open(self.CACHE_METADATA_FILE) as f:
            all_metadata = json.load(f)

        if cache_key in all_metadata:
            all_metadata[cache_key]['last_accessed'] = datetime.now(timezone.utc).isoformat()

            with open(self.CACHE_METADATA_FILE, 'w') as f:
                json.dump(all_metadata, f, indent=2)

    def _calculate_dir_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        for file in directory.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size

    def _enforce_cache_size_limit(self):
        """Enforce cache size limit using LRU eviction."""
        if not self.CACHE_METADATA_FILE.exists():
            return

        with open(self.CACHE_METADATA_FILE) as f:
            metadata = json.load(f)

        total_size_gb = sum(
            item['size_bytes'] for item in metadata.values()
        ) / (1024 ** 3)

        if total_size_gb <= self.CACHE_MAX_SIZE_GB:
            return

        logger.info(
            f"Cache size ({total_size_gb:.2f} GB) exceeds limit "
            f"({self.CACHE_MAX_SIZE_GB} GB), evicting LRU entries"
        )

        sorted_items = sorted(
            metadata.items(),
            key=lambda x: x[1]['last_accessed']
        )

        for cache_key, item in sorted_items:
            cache_dir = self.SHARED_DATASET_CACHE / cache_key

            if cache_dir.exists():
                logger.info(f"Evicting cache: {cache_key}")
                shutil.rmtree(cache_dir)

            del metadata[cache_key]

            total_size_gb = sum(
                item['size_bytes'] for item in metadata.values()
            ) / (1024 ** 3)

            if total_size_gb <= self.CACHE_MAX_SIZE_GB:
                break

        with open(self.CACHE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        content_type: str = 'application/octet-stream',
        storage_type: str = 'internal'
    ) -> str:
        """Upload arbitrary file to storage."""
        storage = self.internal_storage if storage_type == 'internal' else self.external_storage
        return storage.upload_file(local_path, s3_key, content_type)

    # =========================================================================
    # Logging Functions (with buffering)
    # =========================================================================

    def log(
        self,
        message: str,
        level: str = 'INFO',
        source: str = 'trainer',
        **metadata
    ) -> None:
        """Log message with buffering support."""
        log_entry = {
            'level': level,
            'message': message,
            'timestamp': self._get_timestamp(),
            'source': source,
            'metadata': metadata if metadata else {}
        }

        if level == 'ERROR':
            self._send_log_batch([log_entry])
            return

        self._log_buffer.append(log_entry)

        if len(self._log_buffer) >= self._log_buffer_max:
            self.flush_logs()

    def log_info(self, message: str, **metadata) -> None:
        """Log INFO level message"""
        self.log(message, level='INFO', **metadata)

    def log_warning(self, message: str, **metadata) -> None:
        """Log WARNING level message"""
        self.log(message, level='WARNING', **metadata)

    def log_error(self, message: str, **metadata) -> None:
        """Log ERROR level message (sent immediately)"""
        self.log(message, level='ERROR', **metadata)

    def log_debug(self, message: str, **metadata) -> None:
        """Log DEBUG level message"""
        self.log(message, level='DEBUG', **metadata)

    def flush_logs(self) -> None:
        """Flush buffered logs to Backend"""
        if not self._log_buffer:
            return

        logs_to_send = self._log_buffer.copy()
        self._log_buffer.clear()

        self._send_log_batch(logs_to_send)

    def _send_log_batch(self, logs: List[Dict[str, Any]]) -> None:
        """Send batch of logs to Backend"""
        if not logs:
            return

        sent_count = 0
        for log_entry in logs:
            callback_data = {
                'job_id': int(self.job_id),
                'event_type': log_entry.get('source', 'training'),
                'message': log_entry.get('message', ''),
                'level': log_entry.get('level', 'INFO'),
                'data': log_entry.get('metadata', {}),
                'timestamp': log_entry.get('timestamp')
            }

            try:
                self._send_callback(f'/training/jobs/{self.job_id}/callback/logs', callback_data)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send log: {e}")
                break

        if sent_count > 0:
            logger.debug(f"Sent {sent_count}/{len(logs)} logs to Backend")

    def log_event(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = 'INFO'
    ) -> None:
        """Log structured event to Backend (legacy method)."""
        callback_data = {
            'type': 'log_event',
            'job_id': int(self.job_id),
            'event_type': event_type,
            'message': message,
            'level': level,
            'timestamp': self._get_timestamp()
        }

        if data:
            callback_data['data'] = data

        try:
            self._send_callback(f'/training/jobs/{self.job_id}/callback/logs', callback_data)
        except Exception as e:
            logger.warning(f"Failed to send log event: {e}")

    # =========================================================================
    # Data Utility Functions - Unified Dataset Conversion
    # =========================================================================

    def convert_dataset(
        self,
        dataset_dir: str,
        source_format: str,
        target_format: str,
        split_config: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None
    ) -> str:
        """
        Convert dataset between formats.

        Supported conversions:
        - dice -> yolo: For Ultralytics YOLO, mmyolo (creates labels/*.txt, data.yaml)
        - dice -> coco: For MMDetection, VFM, mmpretrain (keeps COCO format annotations)

        Args:
            dataset_dir: Dataset directory path
            source_format: 'dice', 'coco', 'yolo', 'imagefolder'
            target_format: 'yolo', 'coco', 'imagefolder'
            split_config: Optional split configuration
            task_type: Task type ('detection', 'classification', 'segmentation', 'pose')

        Returns:
            Path to converted dataset
        """
        dataset_path = Path(dataset_dir)

        if task_type is None:
            task_type = self.task_type

        if source_format == 'dice' and target_format == 'yolo':
            return self._convert_dice_to_yolo(dataset_path, split_config, task_type)
        elif source_format == 'dice' and target_format == 'coco':
            return self._convert_dice_to_coco(dataset_path, task_type)
        else:
            raise NotImplementedError(f"Conversion {source_format} -> {target_format} not implemented")

    def _convert_dice_to_coco(
        self,
        dataset_dir: Path,
        task_type: str = 'detection'
    ) -> str:
        """
        Convert DICEFormat to COCO format for MMDetection/VFM.

        VFM and MMDetection use COCO format natively, so we just need to ensure
        the annotation file is in the right place with the right structure.

        Args:
            dataset_dir: Dataset directory path
            task_type: Task type for selecting annotation file

        Returns:
            Path to dataset directory
        """
        task_annotation_map = {
            'detection': 'annotations_detection.json',
            'object_detection': 'annotations_detection.json',
            'segmentation': 'annotations_segmentation.json',
            'instance_segmentation': 'annotations_segmentation.json',
        }

        annotation_filename = task_annotation_map.get(task_type, f'annotations_{task_type}.json')
        annotations_file = dataset_dir / annotation_filename

        if not annotations_file.exists():
            fallback_file = dataset_dir / "annotations.json"
            if fallback_file.exists():
                logger.info(f"Task-specific {annotation_filename} not found, using annotations.json")
                annotations_file = fallback_file
            else:
                logger.info(f"No annotation files found, skipping DICE conversion")
                return str(dataset_dir)

        logger.info(f"Using annotation file: {annotations_file.name} for task_type={task_type}")

        # Load and validate COCO format
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = data.get('images', [])
        annotations = data.get('annotations', [])
        categories = data.get('categories', [])

        logger.info(
            f"COCO dataset loaded: "
            f"{len(images)} images, {len(annotations)} annotations, "
            f"{len(categories)} categories"
        )

        return str(dataset_dir)

    def _convert_dice_to_yolo(
        self,
        dataset_dir: Path,
        split_config: Optional[Dict[str, Any]] = None,
        task_type: str = 'detection'
    ) -> str:
        """
        Convert DICEFormat (annotations.json) to YOLO format.

        Creates:
        - labels/*.txt (YOLO format labels)
        - train.txt, val.txt (image lists)
        - data.yaml (dataset config)

        Args:
            dataset_dir: Dataset directory path
            split_config: Split configuration
            task_type: Task type for selecting annotation file
        """
        task_annotation_map = {
            'detection': 'annotations_detection.json',
            'object_detection': 'annotations_detection.json',
            'classification': 'annotations_classification.json',
            'image_classification': 'annotations_classification.json',
            'segmentation': 'annotations_segmentation.json',
            'instance_segmentation': 'annotations_segmentation.json',
            'pose': 'annotations_pose.json',
            'pose_estimation': 'annotations_pose.json',
        }

        annotation_filename = task_annotation_map.get(task_type, f'annotations_{task_type}.json')
        annotations_file = dataset_dir / annotation_filename

        if not annotations_file.exists():
            fallback_file = dataset_dir / "annotations.json"
            if fallback_file.exists():
                logger.info(f"Task-specific {annotation_filename} not found, using annotations.json")
                annotations_file = fallback_file
            else:
                logger.info(f"No annotation files found, skipping DICE conversion")
                return str(dataset_dir)

        logger.info(f"Using annotation file: {annotations_file.name} for task_type={task_type}")

        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        storage_info = data.get('storage_info', {})
        image_root_s3 = storage_info.get('image_root', '')

        local_image_root = ''
        if image_root_s3:
            parts = image_root_s3.split('/')
            if len(parts) >= 3 and parts[0] == 'datasets':
                local_image_root = '/'.join(parts[2:])
                logger.info(f"[storage_info] S3 image_root: '{image_root_s3}'")
                logger.info(f"[storage_info] Extracted local_image_root: '{local_image_root}'")

        images = data.get('images', [])

        if not images:
            logger.warning("No images in annotations.json")
            return str(dataset_dir)

        if images and 'annotations' in images[0]:
            categories = data.get('classes', [])
            annotations = []
            for img in images:
                img_annotations = img.get('annotations', [])
                for ann in img_annotations:
                    if 'image_id' not in ann:
                        ann['image_id'] = img['id']
                    if 'class_id' in ann and 'category_id' not in ann:
                        ann['category_id'] = ann['class_id']
                    annotations.append(ann)
            logger.info(f"Converting custom nested format: {len(images)} images, {len(annotations)} annotations")
        else:
            annotations = data.get('annotations', [])
            categories = data.get('categories', [])
            logger.info(f"Converting COCO/DICE format: {len(images)} images, {len(annotations)} annotations")

        real_categories = [cat for cat in categories if cat.get('name') != '__background__']
        category_map = {cat['id']: idx for idx, cat in enumerate(real_categories)}
        class_names = [cat['name'] for cat in real_categories]

        background_cat_id = None
        for cat in categories:
            if cat.get('name') == '__background__':
                background_cat_id = cat['id']
                break

        image_annotations = {}
        for ann in annotations:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)

        labels_base_dir = dataset_dir / "labels"
        labels_base_dir.mkdir(parents=True, exist_ok=True)

        negative_samples = 0
        total_objects = 0

        for img in images:
            image_id = img['id']
            file_name = img['file_name']
            width = img['width']
            height = img['height']

            img_path = Path(file_name)

            if local_image_root and file_name.startswith(local_image_root):
                relative_path = file_name[len(local_image_root):]
                img_path_rel = Path(relative_path)
            else:
                img_path_rel = img_path

            label_subdir = labels_base_dir / img_path_rel.parent
            label_subdir.mkdir(parents=True, exist_ok=True)
            label_file = label_subdir / f"{img_path_rel.stem}.txt"

            img_anns = image_annotations.get(image_id, [])

            yolo_lines = []
            for ann in img_anns:
                ann_class_name = ann.get('class_name', '')
                category_id = ann.get('category_id')

                if ann_class_name == '__background__' or category_id == background_cat_id:
                    continue

                if 'bbox' not in ann:
                    logger.debug(f"Annotation {ann.get('id', 'unknown')} for image {image_id} missing bbox")
                    continue

                if category_id is None:
                    logger.warning(f"Annotation {ann.get('id', 'unknown')} for image {image_id} missing category_id")
                    continue

                if category_id not in category_map:
                    continue

                bbox = ann['bbox']
                x, y, w, h = bbox
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height

                class_idx = category_map[category_id]
                yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                total_objects += 1

            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))

            if len(yolo_lines) == 0:
                negative_samples += 1

        if split_config and 'splits' in split_config:
            splits = split_config['splits']
        else:
            train_ratio = split_config.get('train_ratio', 0.8) if split_config else 0.8
            val_ratio = split_config.get('val_ratio', 0.2) if split_config else 0.2
            seed = split_config.get('seed', 42) if split_config else 42

            splits = self.split_dataset(
                images,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed
            )

        train_images = []
        val_images = []

        for img in images:
            image_id = img['id']
            file_name = img['file_name']

            possible_paths = [
                dataset_dir / "images" / file_name,
                dataset_dir / file_name,
            ]
            if local_image_root:
                possible_paths.insert(0, dataset_dir / local_image_root / file_name)

            image_file = None
            for path in possible_paths:
                if path.exists():
                    image_file = path
                    break

            if not image_file:
                tried_paths = '\n  - '.join(str(p) for p in possible_paths)
                raise FileNotFoundError(
                    f"Image file not found: {file_name}\n"
                    f"Tried paths:\n  - {tried_paths}"
                )

            image_path = str(Path(image_file).absolute()).replace('\\', '/')

            split = splits.get(image_id, 'train')
            if split == 'train':
                train_images.append(image_path)
            else:
                val_images.append(image_path)

        with open(dataset_dir / "train.txt", 'w') as f:
            f.write('\n'.join(train_images))

        with open(dataset_dir / "val.txt", 'w') as f:
            f.write('\n'.join(val_images))

        logger.info(f"Split: {len(train_images)} train, {len(val_images)} val")

        self.create_data_yaml(
            str(dataset_dir),
            class_names,
            train_path='train.txt',
            val_path='val.txt'
        )

        self.clean_dataset_cache(str(dataset_dir))

        logger.info(
            f"DICE to YOLO conversion completed: "
            f"{len(images)} images, {total_objects} objects, "
            f"{negative_samples} negative samples, "
            f"{len(class_names)} classes"
        )
        return str(dataset_dir)

    def create_data_yaml(
        self,
        dataset_dir: str,
        class_names: List[str],
        train_path: str = 'train.txt',
        val_path: str = 'val.txt',
        test_path: Optional[str] = None
    ) -> str:
        """Create YOLO-format data.yaml file."""
        data_yaml_path = Path(dataset_dir) / "data.yaml"

        data_config = {
            'path': str(Path(dataset_dir).resolve()),
            'train': train_path,
            'val': val_path,
            'nc': len(class_names),
            'names': class_names
        }

        if test_path:
            data_config['test'] = test_path

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Created data.yaml with {len(class_names)} classes")
        return str(data_yaml_path)

    def split_dataset(
        self,
        images: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        stratify_by: Optional[str] = None
    ) -> Dict[str, str]:
        """Split dataset into train/val/test."""
        random.seed(seed)

        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        shuffled = images.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {}
        for i, img in enumerate(shuffled):
            image_id = img['id']
            if i < train_end:
                splits[image_id] = 'train'
            elif i < val_end:
                splits[image_id] = 'val'
            else:
                splits[image_id] = 'test'

        return splits

    def validate_dataset(
        self,
        dataset_dir: str,
        expected_format: str,
        min_images: int = 1,
        check_labels: bool = True
    ) -> Dict[str, Any]:
        """Validate dataset structure and content."""
        dataset_path = Path(dataset_dir)
        issues = []
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'classes': [],
            'class_distribution': {}
        }

        if expected_format == 'yolo':
            data_yaml = dataset_path / "data.yaml"
            if not data_yaml.exists():
                issues.append("Missing data.yaml")
            else:
                with open(data_yaml, 'r') as f:
                    config = yaml.safe_load(f)
                stats['classes'] = config.get('names', [])

            images_dir = dataset_path / "images"
            if images_dir.exists():
                image_files = list(images_dir.glob('**/*.jpg')) + \
                             list(images_dir.glob('**/*.jpeg')) + \
                             list(images_dir.glob('**/*.png'))
                stats['total_images'] = len(image_files)

            if check_labels:
                labels_dir = dataset_path / "labels"
                if labels_dir.exists():
                    label_files = list(labels_dir.glob('**/*.txt'))
                    stats['total_labels'] = len(label_files)

                    for label_file in label_files:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_idx = int(parts[0])
                                    class_name = stats['classes'][class_idx] if class_idx < len(stats['classes']) else str(class_idx)
                                    stats['class_distribution'][class_name] = \
                                        stats['class_distribution'].get(class_name, 0) + 1

        if stats['total_images'] < min_images:
            issues.append(f"Only {stats['total_images']} images found (minimum: {min_images})")

        if check_labels and stats['total_labels'] < stats['total_images']:
            missing = stats['total_images'] - stats['total_labels']
            issues.append(f"Missing labels for {missing} images")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }

    def clean_dataset_cache(self, dataset_dir: str) -> int:
        """Remove stale cache files from dataset directory."""
        cache_pattern = str(Path(dataset_dir) / "**/*.cache")
        cache_files = glob_module.glob(cache_pattern, recursive=True)

        deleted = 0
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                deleted += 1
                logger.debug(f"Deleted cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned {deleted} cache files from {dataset_dir}")

        return deleted

    # =========================================================================
    # Cleanup
    # =========================================================================

    def close(self):
        """Close HTTP client and cleanup resources"""
        self.flush_logs()
        self.http_client.close()
        logger.debug("TrainerSDK closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Convenience Functions
# =============================================================================

def get_sdk() -> TrainerSDK:
    """Get or create a singleton TrainerSDK instance."""
    if not hasattr(get_sdk, '_instance'):
        get_sdk._instance = TrainerSDK()
    return get_sdk._instance
