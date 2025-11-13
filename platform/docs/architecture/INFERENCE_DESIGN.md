# Inference & Testing System Design

Complete design for model inference and testing with XAI support.

## Table of Contents

- [Overview](#overview)
- [Design Principles](#design-principles)
- [Test Run vs Inference Job](#test-run-vs-inference-job)
- [Database Design](#database-design)
- [3-Tier Execution](#3-tier-execution)
- [Storage Strategy](#storage-strategy)
- [Callback Pattern](#callback-pattern)
- [XAI Integration](#xai-integration)
- [Inference Engine (Future)](#inference-engine-future)
- [API Endpoints](#api-endpoints)
- [Trainer Implementation](#trainer-implementation)
- [Task-Specific Examples](#task-specific-examples)

## Overview

The Inference & Testing System enables users to:
- **Test trained models** on labeled test datasets with metrics (evaluate)
- **Run inference** on new unlabeled images (predict)
- **Apply XAI techniques** for model explainability (Grad-CAM, LIME, SHAP)
- **Compare results** across different checkpoints
- **Export results** for downstream use

**Key Scenarios:**

| Scenario | Ground Truth | Output | Use Case |
|----------|--------------|--------|----------|
| **Test Run** | ✅ Has labels | Metrics + Predictions | Model evaluation, benchmarking |
| **Inference Job** | ❌ No labels | Predictions only | Real-world deployment, user uploads |

## Design Principles

### 1. Complete Isolation (No Shared Code)

**Critical**: Inference follows the same isolation principles as training.

```
Backend → HTTP API → Trainer (subprocess/K8s pod) → Callback → Backend
```

- ❌ No adapter pattern
- ❌ No shared Python modules
- ✅ Environment variables for configuration
- ✅ HTTP callbacks for progress/results
- ✅ S3 for data exchange

### 2. 3-Tier Development Parity

Inference must work identically across all tiers:

**Tier 1 (Subprocess)**:
```bash
# Backend spawns subprocess
python platform/trainers/ultralytics/infer.py
```

**Tier 2 (Kind)**:
```yaml
# Kind Job
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-job-abc123
spec:
  template:
    spec:
      containers:
      - name: ultralytics-inference
        image: vision-platform/ultralytics-trainer:latest
        command: ["python", "infer.py"]
```

**Tier 3 (Production K8s)**:
```yaml
# Same as Tier 2, different cluster
```

### 3. Task-Agnostic Design

All CV tasks use the same database schema and API:

```python
# Same structure for all tasks
inference_result = {
    "task_type": "object_detection",  # or "image_classification", etc.
    "predictions": {...},  # Task-specific format
    "xai_results": {...}   # Optional XAI explanations
}
```

### 4. Real-Time Progress Reporting

Like training, inference reports progress via HTTP callbacks:

```python
# Every N images
callback_client.send_inference_progress(
    processed_images=50,
    total_images=100,
    progress_percent=50.0,
    avg_inference_time_ms=25.3
)
```

### 5. XAI-Ready Architecture

Design supports XAI techniques from the start:

```python
inference_result = {
    "predictions": {...},
    "xai": {
        "method": "gradcam",
        "heatmap_path": "s3://bucket/xai/job-123/img001_gradcam.jpg",
        "importance_scores": [...]
    }
}
```

## Test Run vs Inference Job

### Test Run (Evaluate)

**Purpose**: Evaluate model performance on labeled test dataset

**Input**:
- Checkpoint path
- Test dataset (with annotations.json)
- Test split (usually "test")

**Output**:
- Predictions + Ground truth comparison
- Metrics (accuracy, mAP, mIoU, etc.)
- Per-class metrics
- Confusion matrix
- Per-image results with correctness

**Storage**:
- Database: `test_runs`, `test_image_results`
- S3: Per-image predictions + XAI results

**Use Cases**:
- Model evaluation before deployment
- Comparing different checkpoints
- Benchmarking on standard datasets (COCO, ImageNet)
- A/B testing models

**Example**:
```bash
# Backend → Trainer
JOB_ID=test-run-123
INFERENCE_TYPE=test_run
CHECKPOINT_PATH=s3://bucket/checkpoints/job-abc/best.pt
DATASET_ID=coco-test-2017
SPLIT=test  # Has annotations.json
COMPUTE_METRICS=true
```

### Inference Job (Predict)

**Purpose**: Run predictions on new unlabeled images

**Input**:
- Checkpoint path
- Image paths or dataset (no annotations)
- Inference type (single, batch, dataset)

**Output**:
- Predictions only (no metrics)
- Visualization images (bboxes, masks drawn)
- XAI explanations (if requested)
- Performance stats (inference time)

**Storage**:
- Database: `inference_jobs`, `inference_results`
- S3: Predictions + visualizations + XAI results

**Use Cases**:
- Real-world deployment
- User uploaded images
- Production API serving
- Ad-hoc testing on new data

**Example**:
```bash
# Backend → Trainer
JOB_ID=inference-job-456
INFERENCE_TYPE=batch
CHECKPOINT_PATH=s3://bucket/checkpoints/job-abc/best.pt
IMAGE_PATHS='["img1.jpg", "img2.jpg", "img3.jpg"]'
COMPUTE_METRICS=false
ENABLE_XAI=true
XAI_METHOD=gradcam
```

## Database Design

### Test Run Models

```python
# app/models/test_run.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class TestRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TestRun(Base):
    """Test run on labeled test dataset with metrics computation"""
    __tablename__ = "test_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False, index=True)

    # Configuration
    checkpoint_path = Column(String(500), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    dataset_split = Column(String(50), default="test")  # "test", "val", "train"

    # Task info
    task_type = Column(String(100), nullable=False, index=True)

    # Status
    status = Column(SQLEnum(TestRunStatus), default=TestRunStatus.PENDING, index=True)
    progress_percent = Column(Float, default=0.0)
    processed_images = Column(Integer, default=0)
    total_images = Column(Integer, default=0)

    # Primary Metric
    primary_metric_name = Column(String(50), nullable=True)
    primary_metric_value = Column(Float, nullable=True)

    # Metrics (task-agnostic)
    overall_metrics = Column(JSON, nullable=True)
    # {
    #   "accuracy": 0.95,
    #   "mAP50": 0.88,
    #   "precision": 0.89,
    #   "recall": 0.82
    # }

    per_class_metrics = Column(JSON, nullable=True)
    # [{"class_id": 0, "class_name": "cat", "precision": 0.9, "recall": 0.85}, ...]

    confusion_matrix = Column(JSON, nullable=True)
    # [[110, 10], [4, 101]]

    # Performance
    total_inference_time_seconds = Column(Float, nullable=True)
    avg_inference_time_ms = Column(Float, nullable=True)

    # Storage
    results_path = Column(String(500), nullable=True)
    # S3 path to detailed per-image results JSON

    # Error tracking
    error_message = Column(String(1000), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="test_runs")
    dataset = relationship("Dataset")
    image_results = relationship("TestImageResult", back_populates="test_run")
```

```python
# app/models/test_image_result.py
class TestImageResult(Base):
    """Per-image test result with ground truth comparison"""
    __tablename__ = "test_image_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_run_id = Column(UUID(as_uuid=True), ForeignKey("test_runs.id"), nullable=False, index=True)

    # Image info
    image_path = Column(String(500), nullable=False)
    image_name = Column(String(255), nullable=False, index=True)
    image_index = Column(Integer)

    # Task-agnostic predictions (JSON)
    ground_truth = Column(JSON, nullable=False)
    predictions = Column(JSON, nullable=False)
    # Format depends on task_type

    # Metrics
    is_correct = Column(Boolean, nullable=True)  # Classification
    iou = Column(Float, nullable=True)  # Detection, Segmentation
    confidence = Column(Float, nullable=True)

    # Performance
    inference_time_ms = Column(Float, nullable=True)

    # XAI results
    xai_results = Column(JSON, nullable=True)
    # {
    #   "method": "gradcam",
    #   "heatmap_path": "s3://bucket/xai/...",
    #   "importance_scores": [...]
    # }

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    test_run = relationship("TestRun", back_populates="image_results")

    # Indexes
    __table_args__ = (
        Index('idx_test_image_correct', 'test_run_id', 'is_correct'),
    )
```

### Inference Job Models

```python
# app/models/inference_job.py
class InferenceJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class InferenceType(str, enum.Enum):
    SINGLE = "single"
    BATCH = "batch"
    DATASET = "dataset"

class InferenceJob(Base):
    """Inference job on unlabeled images (no metrics)"""
    __tablename__ = "inference_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False, index=True)

    # Configuration
    checkpoint_path = Column(String(500), nullable=False)
    inference_type = Column(SQLEnum(InferenceType), nullable=False)

    # Input data
    input_config = Column(JSON, nullable=False)
    # For single: {"image_path": "..."}
    # For batch: {"image_paths": ["...", "..."]}
    # For dataset: {"dataset_id": "...", "image_dir": "..."}

    # Task info
    task_type = Column(String(100), nullable=False, index=True)

    # Status
    status = Column(SQLEnum(InferenceJobStatus), default=InferenceJobStatus.PENDING, index=True)
    progress_percent = Column(Float, default=0.0)
    processed_images = Column(Integer, default=0)
    total_images = Column(Integer, default=0)

    # XAI configuration
    enable_xai = Column(Boolean, default=False)
    xai_method = Column(String(50), nullable=True)  # "gradcam", "lime", "shap"

    # Performance
    total_inference_time_seconds = Column(Float, nullable=True)
    avg_inference_time_ms = Column(Float, nullable=True)

    # Storage
    results_path = Column(String(500), nullable=True)
    # S3 path to detailed results JSON

    # Error tracking
    error_message = Column(String(1000), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="inference_jobs")
    results = relationship("InferenceResult", back_populates="inference_job")
```

```python
# app/models/inference_result.py
class InferenceResult(Base):
    """Per-image inference result (no ground truth)"""
    __tablename__ = "inference_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    inference_job_id = Column(UUID(as_uuid=True), ForeignKey("inference_jobs.id"), nullable=False, index=True)

    # Image info
    image_path = Column(String(500), nullable=False)
    image_name = Column(String(255), nullable=False, index=True)
    image_index = Column(Integer)

    # Predictions (task-agnostic JSON)
    predictions = Column(JSON, nullable=False)
    # Classification: {"label": "cat", "confidence": 0.95, "top5": [...]}
    # Detection: {"boxes": [{"class": "cat", "bbox": [...], "confidence": 0.95}]}
    # Segmentation: {"mask_path": "s3://...", "classes": [...]}

    # Visualization
    visualization_path = Column(String(500), nullable=True)
    # S3 path to annotated image (with bboxes/masks drawn)

    # XAI results
    xai_results = Column(JSON, nullable=True)
    # {
    #   "method": "gradcam",
    #   "heatmap_path": "s3://bucket/xai/...",
    #   "importance_scores": [...],
    #   "top_regions": [...]
    # }

    # Performance
    inference_time_ms = Column(Float, nullable=True)
    preprocessing_time_ms = Column(Float, nullable=True)
    postprocessing_time_ms = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    inference_job = relationship("InferenceJob", back_populates="results")
```

### Updates to Existing Models

**TrainingJob Model** (add relationships):

```python
class TrainingJob(Base):
    # ... existing fields ...

    # Relationships
    test_runs = relationship("TestRun", back_populates="training_job")
    inference_jobs = relationship("InferenceJob", back_populates="training_job")
```

## 3-Tier Execution

### Tier 1: Subprocess

**Backend spawns inference process**:

```python
# app/services/inference_executor.py
import subprocess
import os

class InferenceExecutor:
    def execute_test_run(self, test_run: TestRun):
        """Execute test run in subprocess"""

        env = {
            **os.environ,
            "JOB_ID": str(test_run.id),
            "TRACE_ID": str(uuid.uuid4()),
            "INFERENCE_TYPE": "test_run",
            "BACKEND_BASE_URL": settings.BACKEND_BASE_URL,
            "CALLBACK_TOKEN": create_callback_token(str(test_run.id)),

            # Configuration
            "TASK_TYPE": test_run.task_type,
            "CHECKPOINT_PATH": test_run.checkpoint_path,
            "DATASET_ID": str(test_run.dataset_id),
            "SPLIT": test_run.dataset_split,
            "COMPUTE_METRICS": "true",

            # S3 credentials
            "S3_ENDPOINT": settings.S3_ENDPOINT,
            "AWS_ACCESS_KEY_ID": settings.AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": settings.AWS_SECRET_ACCESS_KEY,
            "BUCKET_NAME": settings.BUCKET_NAME
        }

        # Determine trainer script based on framework
        framework = test_run.training_job.framework
        script_path = f"platform/trainers/{framework}/infer.py"

        process = subprocess.Popen(
            ["python", script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        test_run.process_id = process.pid
        db.commit()

    def execute_inference_job(self, inference_job: InferenceJob):
        """Execute inference job in subprocess"""

        env = {
            **os.environ,
            "JOB_ID": str(inference_job.id),
            "TRACE_ID": str(uuid.uuid4()),
            "INFERENCE_TYPE": inference_job.inference_type,
            "BACKEND_BASE_URL": settings.BACKEND_BASE_URL,
            "CALLBACK_TOKEN": create_callback_token(str(inference_job.id)),

            # Configuration
            "TASK_TYPE": inference_job.task_type,
            "CHECKPOINT_PATH": inference_job.checkpoint_path,
            "INPUT_CONFIG": json.dumps(inference_job.input_config),
            "COMPUTE_METRICS": "false",

            # XAI
            "ENABLE_XAI": str(inference_job.enable_xai).lower(),
            "XAI_METHOD": inference_job.xai_method or "",

            # S3 credentials
            "S3_ENDPOINT": settings.S3_ENDPOINT,
            "AWS_ACCESS_KEY_ID": settings.AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": settings.AWS_SECRET_ACCESS_KEY,
            "BUCKET_NAME": settings.BUCKET_NAME
        }

        framework = inference_job.training_job.framework
        script_path = f"platform/trainers/{framework}/infer.py"

        process = subprocess.Popen(
            ["python", script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        inference_job.process_id = process.pid
        db.commit()
```

### Tier 2/3: Kubernetes Job

**Backend creates K8s Job**:

```python
# app/services/k8s_inference_executor.py
from kubernetes import client, config

class K8sInferenceExecutor:
    def execute_test_run(self, test_run: TestRun):
        """Execute test run as K8s Job"""

        framework = test_run.training_job.framework
        job_name = f"test-run-{test_run.id}"

        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                labels={
                    "app": "vision-platform",
                    "component": "inference",
                    "test-run-id": str(test_run.id)
                }
            ),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="inference",
                                image=f"vision-platform/{framework}-trainer:latest",
                                command=["python", "infer.py"],
                                env=[
                                    client.V1EnvVar(name="JOB_ID", value=str(test_run.id)),
                                    client.V1EnvVar(name="INFERENCE_TYPE", value="test_run"),
                                    client.V1EnvVar(name="BACKEND_BASE_URL", value=settings.BACKEND_BASE_URL),
                                    client.V1EnvVar(name="CALLBACK_TOKEN", value=create_callback_token(str(test_run.id))),
                                    client.V1EnvVar(name="TASK_TYPE", value=test_run.task_type),
                                    client.V1EnvVar(name="CHECKPOINT_PATH", value=test_run.checkpoint_path),
                                    client.V1EnvVar(name="DATASET_ID", value=str(test_run.dataset_id)),
                                    client.V1EnvVar(name="SPLIT", value=test_run.dataset_split),
                                    client.V1EnvVar(name="COMPUTE_METRICS", value="true"),

                                    # S3 credentials
                                    client.V1EnvVar(name="S3_ENDPOINT", value=settings.S3_ENDPOINT),
                                    client.V1EnvVar(name="AWS_ACCESS_KEY_ID", value_from=client.V1EnvVarSource(
                                        secret_key_ref=client.V1SecretKeySelector(name="s3-credentials", key="access-key")
                                    )),
                                    client.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value_from=client.V1EnvVarSource(
                                        secret_key_ref=client.V1SecretKeySelector(name="s3-credentials", key="secret-key")
                                    )),
                                    client.V1EnvVar(name="BUCKET_NAME", value=settings.BUCKET_NAME)
                                ],
                                resources=client.V1ResourceRequirements(
                                    requests={"cpu": "2", "memory": "4Gi"},
                                    limits={"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"}
                                )
                            )
                        ],
                        restart_policy="Never"
                    )
                ),
                backoff_limit=3
            )
        )

        batch_api = client.BatchV1Api()
        batch_api.create_namespaced_job(namespace="default", body=job)

        test_run.k8s_job_name = job_name
        db.commit()
```

## Storage Strategy

### Hybrid Approach: S3 + Database

**Database** (Quick queries):
- Test run / inference job metadata
- Overall metrics (test runs only)
- Per-class metrics (test runs only)
- Progress tracking
- Status and error messages

**S3** (Large data):
- Per-image predictions (detailed JSON)
- Visualization images (annotated)
- XAI heatmaps and explanations
- Raw inference outputs

### S3 Path Structure

```
vision-platform/  (bucket)
├── test-runs/
│   └── {test-run-id}/
│       ├── results.json              # All per-image results
│       ├── visualizations/           # Optional
│       │   ├── img001_pred.jpg
│       │   └── img002_pred.jpg
│       └── xai/                      # XAI results
│           ├── img001_gradcam.jpg
│           └── img002_gradcam.jpg
│
└── inference-jobs/
    └── {inference-job-id}/
        ├── results.json              # All per-image results
        ├── visualizations/           # Annotated images
        │   ├── img001_pred.jpg
        │   └── img002_pred.jpg
        └── xai/                      # XAI results
            ├── img001_gradcam.jpg
            ├── img001_lime.jpg
            └── img002_gradcam.jpg
```

### Per-Image Results JSON (S3)

**Test Run**:
```json
{
  "test_run_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_type": "object_detection",
  "num_images": 500,
  "images": [
    {
      "image_name": "val_001.jpg",
      "image_path": "datasets/abc/images/test/val_001.jpg",
      "ground_truth": {
        "boxes": [
          {"class_id": 0, "class_name": "cat", "bbox": [10, 20, 100, 150]}
        ]
      },
      "predictions": {
        "boxes": [
          {"class_id": 0, "class_name": "cat", "bbox": [12, 22, 98, 148], "confidence": 0.95}
        ]
      },
      "metrics": {
        "iou": 0.85,
        "precision": 1.0,
        "recall": 1.0
      },
      "xai": {
        "method": "gradcam",
        "heatmap_path": "s3://bucket/test-runs/abc123/xai/val_001_gradcam.jpg"
      },
      "inference_time_ms": 25.3
    }
  ]
}
```

**Inference Job**:
```json
{
  "inference_job_id": "660e9500-f30c-52e5-b827-557766551111",
  "task_type": "image_classification",
  "num_images": 10,
  "images": [
    {
      "image_name": "user_upload_001.jpg",
      "image_path": "user-uploads/user-123/user_upload_001.jpg",
      "predictions": {
        "label": "cat",
        "label_id": 0,
        "confidence": 0.95,
        "top5": [
          {"label": "cat", "confidence": 0.95},
          {"label": "dog", "confidence": 0.03},
          {"label": "bird", "confidence": 0.01}
        ]
      },
      "visualization_path": "s3://bucket/inference-jobs/xyz789/visualizations/user_upload_001_pred.jpg",
      "xai": {
        "method": "gradcam",
        "heatmap_path": "s3://bucket/inference-jobs/xyz789/xai/user_upload_001_gradcam.jpg",
        "importance_scores": [0.9, 0.8, 0.7, ...]
      },
      "inference_time_ms": 18.2
    }
  ]
}
```

## Callback Pattern

### Inference Progress Callback

```http
POST {BACKEND_BASE_URL}/api/v1/inference/{JOB_ID}/progress
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}
Content-Type: application/json

{
  "processed_images": 50,
  "total_images": 100,
  "progress_percent": 50.0,
  "avg_inference_time_ms": 25.3,
  "timestamp": "2025-01-10T12:34:56Z"
}
```

### Inference Completion Callback

**Test Run**:
```http
POST {BACKEND_BASE_URL}/api/v1/test-runs/{TEST_RUN_ID}/done
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}
Content-Type: application/json

{
  "status": "succeeded",  # or "failed"
  "overall_metrics": {
    "accuracy": 0.95,
    "mAP50": 0.88,
    "precision": 0.89,
    "recall": 0.82
  },
  "per_class_metrics": [
    {"class_id": 0, "class_name": "cat", "precision": 0.90, "recall": 0.85},
    {"class_id": 1, "class_name": "dog", "precision": 0.88, "recall": 0.79}
  ],
  "confusion_matrix": [[110, 10], [4, 101]],
  "results_path": "s3://bucket/test-runs/abc123/results.json",
  "total_inference_time_seconds": 12.5,
  "error_message": null,
  "timestamp": "2025-01-10T12:36:00Z"
}
```

**Inference Job**:
```http
POST {BACKEND_BASE_URL}/api/v1/inference-jobs/{JOB_ID}/done
Authorization: Bearer {CALLBACK_TOKEN}
X-Trace-ID: {TRACE_ID}
Content-Type: application/json

{
  "status": "succeeded",
  "total_images": 100,
  "results_path": "s3://bucket/inference-jobs/xyz789/results.json",
  "total_inference_time_seconds": 5.2,
  "avg_inference_time_ms": 52.0,
  "error_message": null,
  "timestamp": "2025-01-10T12:40:00Z"
}
```

## XAI Integration

### Supported Methods

| Method | Task Support | Output | Description |
|--------|--------------|--------|-------------|
| **Grad-CAM** | Classification, Detection | Heatmap | Class activation mapping |
| **LIME** | Classification | Superpixel importance | Local interpretable model |
| **SHAP** | Classification | Feature importance | SHapley Additive exPlanations |
| **Attention Maps** | Vision Transformers | Attention weights | Transformer attention visualization |

### XAI Configuration

```bash
# Environment variables for trainer
ENABLE_XAI=true
XAI_METHOD=gradcam  # or "lime", "shap", "attention"
XAI_TARGET_LAYER=layer4  # For Grad-CAM, framework-specific
XAI_NUM_SAMPLES=1000  # For LIME/SHAP
```

### XAI Result Format

```json
{
  "xai": {
    "method": "gradcam",
    "target_layer": "layer4",
    "heatmap_path": "s3://bucket/.../img001_gradcam.jpg",

    // Method-specific data
    "gradcam": {
      "activations_shape": [7, 7],
      "max_activation": 0.95,
      "top_regions": [
        {"x": 10, "y": 20, "w": 50, "h": 60, "score": 0.95},
        {"x": 100, "y": 80, "w": 40, "h": 45, "score": 0.87}
      ]
    },

    "lime": {
      "superpixels": 100,
      "num_samples": 1000,
      "importance_scores": [0.9, 0.8, 0.7, ...],
      "top_superpixels": [5, 12, 23, 45, 67]
    },

    "shap": {
      "base_value": 0.5,
      "shap_values": [0.3, 0.2, 0.1, ...],
      "feature_names": ["region_1", "region_2", ...]
    },

    // LLM-generated natural language explanation
    "explanation": {
      "summary": "The model classified this image as 'cat' with 95% confidence primarily based on distinctive facial features.",
      "key_factors": [
        "Strong activation detected in the ear region (top-left), indicating typical feline ear structure",
        "High attention on whisker patterns, which are characteristic of cats",
        "Facial structure alignment with learned feline features"
      ],
      "confidence_reasoning": "The high confidence score (95%) is justified by the presence of multiple distinctive cat features that are clearly visible and align with the model's learned representations.",
      "alternative_considerations": "The model also detected some dog-like features in the background, but they received much lower attention (5% confidence).",
      "model": "gpt-4o-mini",  // LLM used for explanation
      "generated_at": "2025-01-10T12:35:00Z"
    }
  }
}
```

### Trainer XAI Implementation Example

```python
# platform/trainers/ultralytics/xai_utils.py
import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class XAIGenerator:
    """Generate XAI explanations for model predictions"""

    def __init__(self, model, method: str, target_layer: str = None):
        self.model = model
        self.method = method

        if method == "gradcam":
            # Get target layer from model
            layer = self._get_layer(model, target_layer or "model.model[-2]")
            self.explainer = GradCAM(model=model, target_layers=[layer])

    def generate(self, image: np.ndarray, prediction: dict) -> dict:
        """Generate XAI explanation for an image"""

        if self.method == "gradcam":
            return self._generate_gradcam(image, prediction)
        elif self.method == "lime":
            return self._generate_lime(image, prediction)
        elif self.method == "shap":
            return self._generate_shap(image, prediction)
        else:
            raise ValueError(f"Unsupported XAI method: {self.method}")

    def _generate_gradcam(self, image: np.ndarray, prediction: dict) -> dict:
        """Generate Grad-CAM heatmap"""

        # Get predicted class
        target_class = prediction.get("class_id", 0)

        # Generate CAM
        cam = self.explainer(input_tensor=image, targets=[target_class])
        cam = cam[0, :]  # Get first image

        # Overlay on image
        visualization = show_cam_on_image(image, cam, use_rgb=True)

        # Find top activation regions
        top_regions = self._find_top_regions(cam, threshold=0.7)

        return {
            "method": "gradcam",
            "heatmap": cam,
            "visualization": visualization,
            "max_activation": float(cam.max()),
            "top_regions": top_regions
        }

    def _find_top_regions(self, cam: np.ndarray, threshold: float = 0.7) -> list:
        """Find regions with high activation"""

        # Threshold
        mask = (cam > threshold).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            score = float(cam[y:y+h, x:x+w].mean())

            regions.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "score": score
            })

        # Sort by score
        regions.sort(key=lambda r: r["score"], reverse=True)

        return regions[:5]  # Top 5 regions

    def _generate_explanation(
        self,
        image_path: str,
        prediction: dict,
        xai_result: dict
    ) -> dict:
        """
        Generate natural language explanation using LLM

        Args:
            image_path: Path to the input image
            prediction: Model prediction result
            xai_result: XAI analysis result (heatmap, regions, etc.)

        Returns:
            Dictionary with natural language explanation
        """
        import openai
        import os

        # Build prompt for LLM
        prompt = self._build_explanation_prompt(prediction, xai_result)

        # Call LLM
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI model explanation expert. Your job is to explain computer vision model predictions in clear, understandable language. Focus on what the model 'saw' and why it made its decision."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        explanation_text = response.choices[0].message.content

        # Parse structured explanation
        return self._parse_explanation(explanation_text)

    def _build_explanation_prompt(self, prediction: dict, xai_result: dict) -> str:
        """Build prompt for LLM explanation"""

        if self.method == "gradcam":
            top_regions = xai_result.get("top_regions", [])
            max_activation = xai_result.get("max_activation", 0)

            prompt = f"""Analyze this image classification result and explain it in natural language.

**Prediction:**
- Predicted Class: {prediction.get('class_name', 'Unknown')}
- Confidence: {prediction.get('confidence', 0):.1%}
- Top 5 Predictions: {prediction.get('top5', [])}

**Grad-CAM Analysis:**
- Method: Gradient-weighted Class Activation Mapping
- Maximum Activation Score: {max_activation:.2f}
- Top Activation Regions: {len(top_regions)} regions detected

Top 3 Regions:
"""
            for i, region in enumerate(top_regions[:3], 1):
                prompt += f"\n{i}. Region at ({region['x']}, {region['y']}), size {region['w']}x{region['h']}, activation score: {region['score']:.2f}"

            prompt += """

Please provide:
1. A brief summary (1-2 sentences) of why the model made this prediction
2. 3-4 key factors the model focused on (based on activation regions)
3. Confidence reasoning (why this confidence level?)
4. Alternative considerations (if any other classes had significant attention)

Format your response as JSON:
{
  "summary": "...",
  "key_factors": ["...", "...", "..."],
  "confidence_reasoning": "...",
  "alternative_considerations": "..."
}
"""

        elif self.method == "lime":
            top_superpixels = xai_result.get("top_superpixels", [])
            importance_scores = xai_result.get("importance_scores", [])

            prompt = f"""Analyze this image classification result using LIME explanation.

**Prediction:**
- Predicted Class: {prediction.get('class_name', 'Unknown')}
- Confidence: {prediction.get('confidence', 0):.1%}

**LIME Analysis:**
- Method: Local Interpretable Model-agnostic Explanations
- Number of Superpixels: {len(importance_scores)}
- Top Important Superpixels: {top_superpixels[:5]}

Explain in natural language why the model made this prediction based on the important image regions identified by LIME.

Format as JSON with: summary, key_factors, confidence_reasoning, alternative_considerations.
"""

        elif self.method == "shap":
            shap_values = xai_result.get("shap_values", [])

            prompt = f"""Analyze this image classification result using SHAP explanation.

**Prediction:**
- Predicted Class: {prediction.get('class_name', 'Unknown')}
- Confidence: {prediction.get('confidence', 0):.1%}

**SHAP Analysis:**
- Method: SHapley Additive exPlanations
- Feature Contributions: {len(shap_values)} features analyzed

Explain in natural language why the model made this prediction based on the SHAP feature importance values.

Format as JSON with: summary, key_factors, confidence_reasoning, alternative_considerations.
"""

        return prompt

    def _parse_explanation(self, explanation_text: str) -> dict:
        """Parse LLM response into structured explanation"""
        import json

        try:
            # Try to parse as JSON
            explanation = json.loads(explanation_text)

            # Ensure all required fields
            return {
                "summary": explanation.get("summary", ""),
                "key_factors": explanation.get("key_factors", []),
                "confidence_reasoning": explanation.get("confidence_reasoning", ""),
                "alternative_considerations": explanation.get("alternative_considerations", ""),
                "model": "gpt-4o-mini",
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        except json.JSONDecodeError:
            # Fallback: use raw text as summary
            return {
                "summary": explanation_text,
                "key_factors": [],
                "confidence_reasoning": "",
                "alternative_considerations": "",
                "model": "gpt-4o-mini",
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
```

## Inference Engine (Future)

**Note**: This section outlines future work. Inference Engine will be designed together with model export/deployment.

### Planned Features

1. **Model Export**:
   - ONNX export for framework-agnostic deployment
   - TensorRT optimization for NVIDIA GPUs
   - OpenVINO for Intel hardware
   - CoreML for Apple devices

2. **Deployment Options**:
   - REST API serving (FastAPI/TorchServe)
   - gRPC for high-performance
   - Edge deployment (TensorFlow Lite, ONNX Runtime Mobile)
   - Batch processing pipelines

3. **Inference Optimization**:
   - Model quantization (INT8, FP16)
   - Dynamic batching
   - Model caching and warm-up
   - Hardware-specific optimization

4. **Monitoring & Scaling**:
   - Request/response logging
   - Latency tracking
   - Throughput monitoring
   - Auto-scaling based on load

### Separation of Concerns

**Current Design** (this document):
- Inference execution using checkpoint files
- Test runs and ad-hoc inference jobs
- XAI integration
- Result storage and visualization

**Future Design** (separate document):
- Optimized inference engines
- Model serving infrastructure
- Production deployment patterns
- Performance optimization

## API Endpoints

### Test Run Endpoints

**POST /api/v1/training/{training_job_id}/test-runs**
```python
# app/api/v1/inference.py
from fastapi import APIRouter, Depends, BackgroundTasks
from app.schemas.test_run import TestRunCreate, TestRunResponse
from app.services.inference_executor import InferenceExecutor

router = APIRouter(prefix="/training", tags=["inference"])

@router.post("/{training_job_id}/test-runs", response_model=TestRunResponse)
async def create_test_run(
    training_job_id: UUID,
    test_run_data: TestRunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create and start a test run on labeled test dataset

    Computes metrics by comparing predictions to ground truth
    """
    # Verify job ownership
    training_job = await db.get(TrainingJob, training_job_id)
    if not training_job or training_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Create test run
    test_run = TestRun(
        training_job_id=training_job_id,
        checkpoint_path=test_run_data.checkpoint_path,
        dataset_id=test_run_data.dataset_id,
        dataset_split=test_run_data.dataset_split or "test",
        task_type=training_job.task_type,
        status=TestRunStatus.PENDING
    )

    db.add(test_run)
    await db.commit()
    await db.refresh(test_run)

    # Start execution (async)
    executor = InferenceExecutor()
    background_tasks.add_task(executor.execute_test_run, test_run.id)

    return test_run
```

**GET /api/v1/test-runs/{test_run_id}**
```python
@router.get("/test-runs/{test_run_id}", response_model=TestRunResponse)
async def get_test_run(
    test_run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get test run status and metrics"""
    test_run = await db.get(TestRun, test_run_id)

    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    # Verify ownership
    if test_run.training_job.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")

    return test_run
```

**GET /api/v1/test-runs/{test_run_id}/images**
```python
from app.schemas.test_run import TestImageResultResponse

@router.get("/test-runs/{test_run_id}/images", response_model=List[TestImageResultResponse])
async def get_test_image_results(
    test_run_id: UUID,
    skip: int = 0,
    limit: int = 20,
    filter_correct: Optional[bool] = None,  # True = correct only, False = incorrect only
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get per-image test results with filtering"""

    test_run = await db.get(TestRun, test_run_id)
    if not test_run or test_run.training_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")

    # Build query
    query = select(TestImageResult).filter(TestImageResult.test_run_id == test_run_id)

    if filter_correct is not None:
        query = query.filter(TestImageResult.is_correct == filter_correct)

    query = query.offset(skip).limit(limit)

    results = await db.execute(query)
    return results.scalars().all()
```

### Inference Job Endpoints

**POST /api/v1/training/{training_job_id}/inference**
```python
from app.schemas.inference_job import InferenceJobCreate, InferenceJobResponse

@router.post("/{training_job_id}/inference", response_model=InferenceJobResponse)
async def create_inference_job(
    training_job_id: UUID,
    inference_data: InferenceJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create and start an inference job on unlabeled images

    No metrics computation, only predictions
    """
    # Verify job ownership
    training_job = await db.get(TrainingJob, training_job_id)
    if not training_job or training_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Create inference job
    inference_job = InferenceJob(
        training_job_id=training_job_id,
        checkpoint_path=inference_data.checkpoint_path,
        inference_type=inference_data.inference_type,
        input_config=inference_data.input_config,
        task_type=training_job.task_type,
        enable_xai=inference_data.enable_xai or False,
        xai_method=inference_data.xai_method,
        status=InferenceJobStatus.PENDING
    )

    db.add(inference_job)
    await db.commit()
    await db.refresh(inference_job)

    # Start execution (async)
    executor = InferenceExecutor()
    background_tasks.add_task(executor.execute_inference_job, inference_job.id)

    return inference_job
```

**GET /api/v1/inference-jobs/{job_id}**
```python
@router.get("/inference-jobs/{job_id}", response_model=InferenceJobResponse)
async def get_inference_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get inference job status"""
    inference_job = await db.get(InferenceJob, job_id)

    if not inference_job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    # Verify ownership
    if inference_job.training_job.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")

    return inference_job
```

**GET /api/v1/inference-jobs/{job_id}/results**
```python
from app.schemas.inference_result import InferenceResultResponse

@router.get("/inference-jobs/{job_id}/results", response_model=List[InferenceResultResponse])
async def get_inference_results(
    job_id: UUID,
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get per-image inference results"""

    inference_job = await db.get(InferenceJob, job_id)
    if not inference_job or inference_job.training_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")

    query = select(InferenceResult).filter(
        InferenceResult.inference_job_id == job_id
    ).offset(skip).limit(limit)

    results = await db.execute(query)
    return results.scalars().all()
```

### Callback Endpoints (for Trainers)

**POST /api/v1/inference/{job_id}/progress**
```python
from app.schemas.inference import InferenceProgress

@router.post("/inference/{job_id}/progress")
async def report_inference_progress(
    job_id: UUID,
    progress: InferenceProgress,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive inference progress update from trainer

    Works for both test runs and inference jobs
    """
    # Try test run first
    test_run = await db.get(TestRun, job_id)
    if test_run:
        test_run.processed_images = progress.processed_images
        test_run.progress_percent = progress.progress_percent
        await db.commit()

        # Broadcast to WebSocket
        await broadcast_update(str(job_id), {
            "type": "inference_progress",
            "processed_images": progress.processed_images,
            "total_images": progress.total_images,
            "progress_percent": progress.progress_percent
        })

        return {"status": "ok"}

    # Try inference job
    inference_job = await db.get(InferenceJob, job_id)
    if inference_job:
        inference_job.processed_images = progress.processed_images
        inference_job.progress_percent = progress.progress_percent
        await db.commit()

        # Broadcast to WebSocket
        await broadcast_update(str(job_id), {
            "type": "inference_progress",
            "processed_images": progress.processed_images,
            "total_images": progress.total_images,
            "progress_percent": progress.progress_percent
        })

        return {"status": "ok"}

    raise HTTPException(status_code=404, detail="Job not found")
```

**POST /api/v1/test-runs/{test_run_id}/done**
```python
from app.schemas.test_run import TestRunComplete

@router.post("/test-runs/{test_run_id}/done")
async def test_run_complete(
    test_run_id: UUID,
    completion: TestRunComplete,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """Receive test run completion from trainer"""

    test_run = await db.get(TestRun, test_run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    # Update test run
    test_run.status = TestRunStatus.COMPLETED if completion.status == "succeeded" else TestRunStatus.FAILED
    test_run.overall_metrics = completion.overall_metrics
    test_run.per_class_metrics = completion.per_class_metrics
    test_run.confusion_matrix = completion.confusion_matrix
    test_run.primary_metric_value = completion.primary_metric_value
    test_run.results_path = completion.results_path
    test_run.total_inference_time_seconds = completion.total_inference_time_seconds
    test_run.completed_at = datetime.utcnow()

    if completion.status == "failed":
        test_run.error_message = completion.error_message

    await db.commit()

    # Broadcast to WebSocket
    await broadcast_update(str(test_run_id), {
        "type": "test_run_complete",
        "status": completion.status,
        "overall_metrics": completion.overall_metrics
    })

    return {"status": "ok"}
```

**POST /api/v1/inference-jobs/{job_id}/done**
```python
from app.schemas.inference_job import InferenceJobComplete

@router.post("/inference-jobs/{job_id}/done")
async def inference_job_complete(
    job_id: UUID,
    completion: InferenceJobComplete,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """Receive inference job completion from trainer"""

    inference_job = await db.get(InferenceJob, job_id)
    if not inference_job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    # Update inference job
    inference_job.status = InferenceJobStatus.COMPLETED if completion.status == "succeeded" else InferenceJobStatus.FAILED
    inference_job.results_path = completion.results_path
    inference_job.total_inference_time_seconds = completion.total_inference_time_seconds
    inference_job.avg_inference_time_ms = completion.avg_inference_time_ms
    inference_job.completed_at = datetime.utcnow()

    if completion.status == "failed":
        inference_job.error_message = completion.error_message

    await db.commit()

    # Broadcast to WebSocket
    await broadcast_update(str(job_id), {
        "type": "inference_complete",
        "status": completion.status,
        "total_images": completion.total_images
    })

    return {"status": "ok"}
```

## Trainer Implementation

### Inference Script Structure

```python
# platform/trainers/ultralytics/infer.py
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import json

from utils import CallbackClient, StorageClient, XAIGenerator

def main():
    print("=== Ultralytics Inference Starting ===")

    # 1. Read configuration
    job_id = os.environ["JOB_ID"]
    trace_id = os.environ["TRACE_ID"]
    inference_type = os.environ["INFERENCE_TYPE"]  # "test_run" or "single/batch/dataset"
    backend_url = os.environ["BACKEND_BASE_URL"]
    callback_token = os.environ["CALLBACK_TOKEN"]

    task_type = os.environ["TASK_TYPE"]
    checkpoint_path = os.environ["CHECKPOINT_PATH"]
    compute_metrics = os.environ.get("COMPUTE_METRICS", "false").lower() == "true"

    # XAI
    enable_xai = os.environ.get("ENABLE_XAI", "false").lower() == "true"
    xai_method = os.environ.get("XAI_METHOD")

    print(f"Job ID: {job_id}")
    print(f"Inference Type: {inference_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Compute Metrics: {compute_metrics}")
    print(f"Enable XAI: {enable_xai}")

    # 2. Initialize clients
    callback_client = CallbackClient(
        backend_url=backend_url,
        job_id=job_id,
        trace_id=trace_id,
        callback_token=callback_token
    )

    storage_client = StorageClient()

    try:
        # 3. Download checkpoint
        print("\n=== Downloading Checkpoint ===")
        local_checkpoint = storage_client.download_checkpoint(checkpoint_path)
        print(f"Checkpoint downloaded: {local_checkpoint}")

        # 4. Load model
        print("\n=== Loading Model ===")
        model = YOLO(local_checkpoint)
        print(f"Model loaded: {model.model}")

        # 5. Initialize XAI if enabled
        xai_generator = None
        if enable_xai:
            print(f"\n=== Initializing XAI ({xai_method}) ===")
            xai_generator = XAIGenerator(model, xai_method)

        # 6. Execute inference based on type
        if inference_type == "test_run":
            execute_test_run(
                model=model,
                job_id=job_id,
                callback_client=callback_client,
                storage_client=storage_client,
                xai_generator=xai_generator
            )
        else:
            execute_inference_job(
                model=model,
                job_id=job_id,
                inference_type=inference_type,
                callback_client=callback_client,
                storage_client=storage_client,
                xai_generator=xai_generator
            )

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


def execute_test_run(model, job_id, callback_client, storage_client, xai_generator):
    """Execute test run on labeled dataset with metrics"""

    # Get dataset
    dataset_id = os.environ["DATASET_ID"]
    split = os.environ.get("SPLIT", "test")

    print(f"\n=== Downloading Dataset (split={split}) ===")
    dataset_path = storage_client.download_dataset(dataset_id)

    # Run validation (includes metrics computation)
    results = model.val(
        data=f"{dataset_path}/data.yaml",
        split=split,
        save_json=True,
        save_hybrid=True
    )

    # Extract metrics
    overall_metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr)
    }

    per_class_metrics = []
    for i, class_name in enumerate(results.names.values()):
        per_class_metrics.append({
            "class_id": i,
            "class_name": class_name,
            "ap": float(results.box.ap[i]),
            "precision": float(results.box.p[i]),
            "recall": float(results.box.r[i])
        })

    # Process per-image results with XAI
    image_results = []
    total_images = len(results.pred)

    for idx, (pred, gt) in enumerate(zip(results.pred, results.gt)):
        # Extract predictions and ground truth
        pred_boxes = extract_boxes(pred)
        gt_boxes = extract_boxes(gt)

        # Compute per-image metrics
        iou = compute_iou(pred_boxes, gt_boxes)

        # Generate XAI if enabled
        xai_result = None
        if xai_generator:
            image = load_image(results.paths[idx])
            xai_result = xai_generator.generate(image, pred_boxes[0] if pred_boxes else None)

            # Upload XAI heatmap
            xai_path = storage_client.upload_xai_result(
                job_id=job_id,
                image_name=Path(results.paths[idx]).name,
                heatmap=xai_result["visualization"],
                method=xai_generator.method
            )
            xai_result["heatmap_path"] = xai_path

        image_results.append({
            "image_name": Path(results.paths[idx]).name,
            "image_path": results.paths[idx],
            "ground_truth": {"boxes": gt_boxes},
            "predictions": {"boxes": pred_boxes},
            "metrics": {"iou": iou},
            "xai": xai_result
        })

        # Send progress
        if (idx + 1) % 10 == 0:
            callback_client.send_inference_progress(
                processed_images=idx + 1,
                total_images=total_images,
                progress_percent=(idx + 1) / total_images * 100
            )

    # Upload results to S3
    results_path = storage_client.upload_test_results(
        job_id=job_id,
        image_results=image_results
    )

    # Send completion
    callback_client.send_test_run_completion(
        status="succeeded",
        overall_metrics=overall_metrics,
        per_class_metrics=per_class_metrics,
        results_path=results_path,
        total_inference_time_seconds=results.speed["inference"]
    )


def execute_inference_job(model, job_id, inference_type, callback_client, storage_client, xai_generator):
    """Execute inference on unlabeled images"""

    # Get input configuration
    input_config = json.loads(os.environ["INPUT_CONFIG"])

    if inference_type == "single":
        image_paths = [input_config["image_path"]]
    elif inference_type == "batch":
        image_paths = input_config["image_paths"]
    elif inference_type == "dataset":
        # Load all images from directory
        dataset_id = input_config["dataset_id"]
        dataset_path = storage_client.download_dataset(dataset_id)
        image_paths = list(Path(dataset_path).glob("**/*.jpg"))

    # Run inference
    image_results = []
    total_images = len(image_paths)

    for idx, image_path in enumerate(image_paths):
        # Run prediction
        results = model.predict(image_path, save=False)

        # Extract predictions
        pred_boxes = extract_boxes(results[0])

        # Generate XAI if enabled
        xai_result = None
        if xai_generator:
            image = load_image(image_path)
            xai_result = xai_generator.generate(image, pred_boxes[0] if pred_boxes else None)

            # Upload XAI heatmap
            xai_path = storage_client.upload_xai_result(
                job_id=job_id,
                image_name=Path(image_path).name,
                heatmap=xai_result["visualization"],
                method=xai_generator.method
            )
            xai_result["heatmap_path"] = xai_path

        # Generate visualization
        viz_path = storage_client.upload_visualization(
            job_id=job_id,
            image_name=Path(image_path).name,
            image=results[0].plot()  # Annotated image
        )

        image_results.append({
            "image_name": Path(image_path).name,
            "image_path": str(image_path),
            "predictions": {"boxes": pred_boxes},
            "visualization_path": viz_path,
            "xai": xai_result,
            "inference_time_ms": results[0].speed["inference"]
        })

        # Send progress
        if (idx + 1) % 5 == 0:
            callback_client.send_inference_progress(
                processed_images=idx + 1,
                total_images=total_images,
                progress_percent=(idx + 1) / total_images * 100
            )

    # Upload results to S3
    results_path = storage_client.upload_inference_results(
        job_id=job_id,
        image_results=image_results
    )

    # Calculate average inference time
    total_time = sum(r["inference_time_ms"] for r in image_results)
    avg_time = total_time / len(image_results)

    # Send completion
    callback_client.send_inference_completion(
        status="succeeded",
        total_images=total_images,
        results_path=results_path,
        total_inference_time_seconds=total_time / 1000,
        avg_inference_time_ms=avg_time
    )


if __name__ == "__main__":
    main()
```

## Task-Specific Examples

### Image Classification - Test Run

```python
# Environment variables
INFERENCE_TYPE=test_run
TASK_TYPE=image_classification
CHECKPOINT_PATH=s3://bucket/checkpoints/job-123/best.pt
DATASET_ID=imagenet-val
SPLIT=val
COMPUTE_METRICS=true
ENABLE_XAI=true
XAI_METHOD=gradcam

# Trainer output
{
  "overall_metrics": {
    "accuracy": 0.76,
    "top5_accuracy": 0.93,
    "loss": 0.89
  },
  "per_class_metrics": [
    {"class_id": 0, "class_name": "cat", "precision": 0.82, "recall": 0.79, "f1": 0.80},
    {"class_id": 1, "class_name": "dog", "precision": 0.78, "recall": 0.75, "f1": 0.76}
  ],
  "confusion_matrix": [[790, 210], [250, 750]]
}
```

### Object Detection - Inference Job

```python
# Environment variables
INFERENCE_TYPE=batch
TASK_TYPE=object_detection
CHECKPOINT_PATH=s3://bucket/checkpoints/job-456/best.pt
INPUT_CONFIG='{"image_paths": ["img1.jpg", "img2.jpg"]}'
COMPUTE_METRICS=false
ENABLE_XAI=true
XAI_METHOD=gradcam

# Trainer output
{
  "image_results": [
    {
      "image_name": "img1.jpg",
      "predictions": {
        "boxes": [
          {"class": "person", "bbox": [10, 20, 100, 200], "confidence": 0.95},
          {"class": "car", "bbox": [150, 50, 300, 180], "confidence": 0.87}
        ]
      },
      "visualization_path": "s3://bucket/inference-jobs/xyz/viz/img1_pred.jpg",
      "xai": {
        "method": "gradcam",
        "heatmap_path": "s3://bucket/inference-jobs/xyz/xai/img1_gradcam.jpg",
        "top_regions": [
          {"x": 10, "y": 20, "w": 90, "h": 180, "score": 0.95}
        ]
      }
    }
  ]
}
```

## References

- [Architecture Overview](./OVERVIEW.md)
- [Backend Design](./BACKEND_DESIGN.md)
- [Trainer Design](./TRAINER_DESIGN.md)
- [Validation & Metrics Design](./VALIDATION_METRICS_DESIGN.md)
- [Model Weight Management](./MODEL_WEIGHT_MANAGEMENT.md)
- [3-Tier Development](../development/3_TIER_DEVELOPMENT.md)
