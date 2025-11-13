# Validation & Metrics System Design

Complete design for task-agnostic validation and metrics tracking.

## Table of Contents

- [Overview](#overview)
- [Design Principles](#design-principles)
- [Database Design](#database-design)
- [Primary Metric System](#primary-metric-system)
- [Metrics Schema](#metrics-schema)
- [Trainer Integration](#trainer-integration)
- [Storage Strategy](#storage-strategy)
- [Frontend Integration](#frontend-integration)
- [Task-Specific Examples](#task-specific-examples)
- [Best Practices](#best-practices)

## Overview

The Validation & Metrics System provides a **task-agnostic** way to track, store, and visualize training validation results across all computer vision tasks.

**Key Features**:
- ✅ Task-agnostic design supporting all CV tasks
- ✅ Primary metric for automatic best checkpoint selection
- ✅ Flexible metrics storage (standard + custom)
- ✅ Per-class and per-image validation results
- ✅ Dependency isolation (Trainer → Backend via HTTP callbacks)
- ✅ Hybrid storage (DB for quick queries, S3 for detailed results)

**Supported Tasks**:
- Image Classification
- Object Detection
- Instance Segmentation
- Semantic Segmentation
- Pose Estimation
- Future: Image Captioning, VQA, etc.

## Design Principles

### 1. Task-Agnostic Architecture

All tasks use the same database schema and API:

```python
# Same code for all tasks
validation_result = TrainingValidationResult(
    task_type="object_detection",  # or "image_classification", etc.
    primary_metric_name="mAP50-95",
    primary_metric_value=0.8723,
    metrics={...}  # Task-specific metrics in JSON
)
```

### 2. Standard + Custom Metrics

**Standard Metrics**: Predefined, automatically visualized with beautiful charts
**Custom Metrics**: Any additional metrics, displayed in generic tables

```json
{
  "metrics": {
    // Standard metrics (Frontend knows how to visualize)
    "mAP50": 0.87,
    "precision": 0.89,

    // Custom metrics (displayed in generic table)
    "inference_speed_fps": 45.2,
    "my_custom_metric": 0.78
  }
}
```

### 3. Primary Metric for Best Checkpoint

Each training job defines a **primary metric** to automatically determine the best checkpoint:

```python
job = TrainingJob(
    primary_metric_name="mAP50-95",  # User selects or task default
    primary_metric_direction="maximize"  # or "minimize" for loss
)
```

Backend automatically tracks best checkpoint based on primary metric.

### 4. Dependency Isolation

Trainers report validation results via HTTP callbacks (no shared code):

```
Trainer (YOLO) → HTTP POST /api/v1/jobs/{job_id}/validation → Backend
```

### 5. Hybrid Storage

- **Database**: Overall & per-class metrics (fast queries, sorting, filtering)
- **S3**: Per-image results (large volume, less frequent access)

## Database Design

### TrainingValidationResult Model

```python
# app/models/training_validation_result.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class TrainingValidationResult(Base):
    """
    Training validation results (task-agnostic)

    Stores validation metrics for each epoch during training.
    Supports all CV tasks: classification, detection, segmentation, pose, etc.
    """
    __tablename__ = "training_validation_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)

    # Task identification
    task_type = Column(String(100), nullable=False, index=True)
    # "image_classification", "object_detection", "instance_segmentation", etc.

    # ===== Primary Metric (for best checkpoint selection) =====
    primary_metric_name = Column(String(50), nullable=False)
    # Examples: "accuracy", "mAP50", "mAP50-95", "mIoU", "OKS"

    primary_metric_value = Column(Float, nullable=False)
    # The value of the primary metric for this epoch

    is_best = Column(Boolean, default=False)
    # True if this is the best epoch for this job (based on primary metric)

    # ===== All Metrics (standard + custom) =====
    metrics = Column(JSON, nullable=False)
    # {
    #   // Standard metrics (Frontend knows these)
    #   "accuracy": 0.95,
    #   "loss": 0.234,
    #   "mAP50": 0.87,
    #   "mAP50-95": 0.65,
    #   "precision": 0.89,
    #   "recall": 0.82,
    #
    #   // Custom metrics (displayed in generic table)
    #   "inference_speed_fps": 45.2,
    #   "gpu_memory_gb": 8.2,
    #   "custom_metric_1": 0.78
    # }

    # ===== Per-Class Metrics =====
    per_class_metrics = Column(JSON, nullable=True)
    # [
    #   {
    #     "class_id": 0,
    #     "class_name": "cat",
    #     "precision": 0.90,
    #     "recall": 0.85,
    #     "f1": 0.87,
    #     "support": 120  // number of instances
    #   },
    #   {
    #     "class_id": 1,
    #     "class_name": "dog",
    #     "precision": 0.88,
    #     "recall": 0.79,
    #     "f1": 0.83,
    #     "support": 95
    #   }
    # ]

    # ===== Visualization Data =====
    confusion_matrix = Column(JSON, nullable=True)
    # Classification only
    # [[120, 5], [8, 95]]  // 2D array

    pr_curves = Column(JSON, nullable=True)
    # Detection, Segmentation
    # {"cat": {"precision": [...], "recall": [...]}, ...}

    # ===== Storage =====
    image_results_path = Column(String(500), nullable=True)
    # S3 path to detailed per-image results
    # "s3://bucket/validation-results/{job_id}/epoch-{epoch}/images.json"

    num_images_validated = Column(Integer, default=0)

    # ===== Metadata =====
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    validation_time_seconds = Column(Float, nullable=True)
    # How long validation took

    # Relationships
    training_job = relationship("TrainingJob", back_populates="validation_results")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_job_epoch', 'training_job_id', 'epoch'),
        Index('idx_job_best', 'training_job_id', 'is_best'),
    )
```

### TrainingValidationImageResult (S3 Storage)

Per-image validation results are stored in S3 due to large volume:

**S3 Path**: `s3://bucket/validation-results/{job_id}/epoch-{epoch}/images.json`

**JSON Format** (task-agnostic):
```json
{
  "format_version": "1.0",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "epoch": 42,
  "task_type": "object_detection",
  "num_images": 500,
  "images": [
    {
      "image_id": "val_001.jpg",
      "image_path": "datasets/abc/images/val/val_001.jpg",

      // Task-specific predictions
      "predictions": {
        "boxes": [
          {"class_id": 0, "class_name": "cat", "bbox": [10, 20, 100, 150], "confidence": 0.95},
          {"class_id": 1, "class_name": "dog", "bbox": [200, 50, 120, 180], "confidence": 0.87}
        ]
      },

      "ground_truth": {
        "boxes": [
          {"class_id": 0, "class_name": "cat", "bbox": [12, 22, 98, 148]},
          {"class_id": 1, "class_name": "dog", "bbox": [198, 48, 122, 182]}
        ]
      },

      // Per-image metrics
      "metrics": {
        "iou": 0.85,
        "precision": 0.90,
        "recall": 1.0,
        "num_tp": 2,
        "num_fp": 0,
        "num_fn": 0
      }
    },
    // ... 499 more images
  ]
}
```

### Updates to Existing Models

**TrainingJob Model** (add validation relationship):

```python
# app/models/training_job.py
class TrainingJob(Base):
    # ... existing fields ...

    # Primary metric configuration
    primary_metric_name = Column(String(50), nullable=False)
    # Set at job creation (user selects or uses task default)

    primary_metric_direction = Column(String(10), default="maximize")
    # "maximize" (for accuracy, mAP) or "minimize" (for loss)

    # Best validation result tracking
    best_epoch = Column(Integer, nullable=True)
    best_validation_id = Column(UUID(as_uuid=True), ForeignKey("training_validation_results.id"), nullable=True)

    # Relationships
    validation_results = relationship("TrainingValidationResult", back_populates="training_job")
    best_validation = relationship("TrainingValidationResult", foreign_keys=[best_validation_id])
```

## Primary Metric System

### Task-Specific Defaults

```python
# app/config/default_metrics.py
from enum import Enum
from typing import Dict, List

class TaskType(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POSE_ESTIMATION = "pose_estimation"

DEFAULT_PRIMARY_METRICS: Dict[TaskType, Dict] = {
    TaskType.IMAGE_CLASSIFICATION: {
        "name": "accuracy",
        "direction": "maximize",
        "alternatives": ["top5_accuracy", "f1_score", "loss"]
    },
    TaskType.OBJECT_DETECTION: {
        "name": "mAP50-95",
        "direction": "maximize",
        "alternatives": ["mAP50", "precision", "recall", "loss"]
    },
    TaskType.INSTANCE_SEGMENTATION: {
        "name": "mAP_mask",
        "direction": "maximize",
        "alternatives": ["mAP_bbox", "mIoU", "loss"]
    },
    TaskType.SEMANTIC_SEGMENTATION: {
        "name": "mIoU",
        "direction": "maximize",
        "alternatives": ["pixel_accuracy", "dice", "loss"]
    },
    TaskType.POSE_ESTIMATION: {
        "name": "OKS",
        "direction": "maximize",
        "alternatives": ["PCK", "mAP", "loss"]
    }
}

def get_default_primary_metric(task_type: TaskType) -> str:
    """Get default primary metric for a task type"""
    return DEFAULT_PRIMARY_METRICS[task_type]["name"]

def get_primary_metric_direction(metric_name: str) -> str:
    """Determine if metric should be maximized or minimized"""
    if "loss" in metric_name.lower() or "error" in metric_name.lower():
        return "minimize"
    return "maximize"
```

### Best Checkpoint Selection

```python
# app/services/validation_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.models.training_job import TrainingJob
from app.models.training_validation_result import TrainingValidationResult

class ValidationService:
    """Service for managing validation results and best checkpoint selection"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def save_validation_result(
        self,
        job_id: UUID,
        epoch: int,
        task_type: str,
        primary_metric_value: float,
        metrics: dict,
        per_class_metrics: list = None,
        confusion_matrix: list = None,
        pr_curves: dict = None
    ) -> TrainingValidationResult:
        """
        Save validation result and update best checkpoint if needed
        """
        # 1. Get training job
        job = await self.db.get(TrainingJob, job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")

        # 2. Create validation result
        validation_result = TrainingValidationResult(
            training_job_id=job_id,
            epoch=epoch,
            task_type=task_type,
            primary_metric_name=job.primary_metric_name,
            primary_metric_value=primary_metric_value,
            metrics=metrics,
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_matrix,
            pr_curves=pr_curves,
            is_best=False
        )

        self.db.add(validation_result)
        await self.db.flush()  # Get ID

        # 3. Check if this is the best epoch
        is_new_best = await self._is_new_best(job, validation_result)

        if is_new_best:
            # Mark previous best as not best
            if job.best_validation_id:
                await self.db.execute(
                    update(TrainingValidationResult)
                    .where(TrainingValidationResult.id == job.best_validation_id)
                    .values(is_best=False)
                )

            # Mark this as best
            validation_result.is_best = True
            job.best_epoch = epoch
            job.best_validation_id = validation_result.id
            job.best_metrics = {
                'epoch': epoch,
                'primary_metric_name': job.primary_metric_name,
                'primary_metric_value': primary_metric_value,
                'all_metrics': metrics
            }

            print(f"[Validation] New best at epoch {epoch}: "
                  f"{job.primary_metric_name}={primary_metric_value:.4f}")

        await self.db.commit()
        await self.db.refresh(validation_result)

        return validation_result

    async def _is_new_best(
        self,
        job: TrainingJob,
        new_result: TrainingValidationResult
    ) -> bool:
        """
        Determine if new validation result is better than current best
        """
        # No previous best
        if not job.best_validation_id:
            return True

        # Get current best
        current_best = await self.db.get(TrainingValidationResult, job.best_validation_id)
        if not current_best:
            return True

        # Compare based on direction
        new_value = new_result.primary_metric_value
        best_value = current_best.primary_metric_value

        if job.primary_metric_direction == "maximize":
            return new_value > best_value
        else:  # minimize
            return new_value < best_value
```

## Metrics Schema

### Standard Metrics by Task

These are **recommended** metrics that Frontend knows how to visualize beautifully. Trainers can send additional custom metrics.

#### Image Classification

| Metric | Description | Direction | Frontend Display |
|--------|-------------|-----------|------------------|
| `accuracy` | Overall accuracy | Maximize | Line chart, progress bar |
| `top5_accuracy` | Top-5 accuracy | Maximize | Line chart |
| `loss` | Cross-entropy loss | Minimize | Line chart |
| `precision` | Macro precision | Maximize | Line chart |
| `recall` | Macro recall | Maximize | Line chart |
| `f1_score` | Macro F1 | Maximize | Line chart |

**Per-class**: `precision`, `recall`, `f1`, `support`

**Visualization**: Confusion matrix, per-class metrics table

#### Object Detection

| Metric | Description | Direction | Frontend Display |
|--------|-------------|-----------|------------------|
| `mAP50` | mAP at IoU=0.5 | Maximize | Line chart, gauge |
| `mAP50-95` | mAP at IoU=0.5:0.95 | Maximize | Line chart, gauge |
| `precision` | Overall precision | Maximize | Line chart |
| `recall` | Overall recall | Maximize | Line chart |
| `loss` | Detection loss | Minimize | Line chart |

**Per-class**: `ap` (average precision), `precision`, `recall`

**Visualization**: PR curves, per-class AP table

#### Instance Segmentation

| Metric | Description | Direction | Frontend Display |
|--------|-------------|-----------|------------------|
| `mAP_mask` | mAP for masks | Maximize | Line chart, gauge |
| `mAP_bbox` | mAP for bboxes | Maximize | Line chart |
| `mIoU` | Mean IoU | Maximize | Line chart, gauge |
| `loss` | Segmentation loss | Minimize | Line chart |

**Per-class**: `ap_mask`, `ap_bbox`, `iou`

**Visualization**: PR curves, IoU heatmap

#### Semantic Segmentation

| Metric | Description | Direction | Frontend Display |
|--------|-------------|-----------|------------------|
| `mIoU` | Mean IoU | Maximize | Line chart, gauge |
| `pixel_accuracy` | Pixel accuracy | Maximize | Line chart |
| `dice` | Dice coefficient | Maximize | Line chart |
| `loss` | Segmentation loss | Minimize | Line chart |

**Per-class**: `iou`, `dice`, `pixel_count`

**Visualization**: IoU per class bar chart

#### Pose Estimation

| Metric | Description | Direction | Frontend Display |
|--------|-------------|-----------|------------------|
| `OKS` | Object Keypoint Similarity | Maximize | Line chart, gauge |
| `PCK` | Percentage of Correct Keypoints | Maximize | Line chart |
| `mAP` | Mean Average Precision | Maximize | Line chart |
| `loss` | Pose estimation loss | Minimize | Line chart |

**Per-class**: `oks`, `pck` (per keypoint type)

**Visualization**: Keypoint accuracy table

### Custom Metrics

Trainers can send **any additional metrics** beyond the standard ones:

```json
{
  "metrics": {
    // Standard
    "mAP50": 0.87,
    "precision": 0.89,

    // Custom (displayed in generic table)
    "inference_speed_fps": 45.2,
    "training_time_per_epoch_seconds": 120.5,
    "gpu_memory_usage_gb": 8.2,
    "model_size_mb": 25.6,
    "custom_business_metric": 0.78
  }
}
```

Frontend displays custom metrics in a generic "Additional Metrics" table.

## Trainer Integration

### Validation Callback

Trainers report validation results via HTTP POST:

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

  // Per-class metrics
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
  "pr_curves": {
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
  "validation_time_seconds": 45.2
}
```

### Backend API Endpoint

```python
# app/api/v1/training.py
from fastapi import APIRouter, Depends
from app.schemas.validation import ValidationResultCreate
from app.services.validation_service import ValidationService
from app.core.security import verify_callback_token

router = APIRouter(prefix="/training", tags=["training"])

@router.post("/jobs/{job_id}/validation")
async def report_validation_result(
    job_id: UUID,
    validation_data: ValidationResultCreate,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive validation result from trainer

    Called by trainer after each validation run (usually per epoch)
    Automatically determines if this is the best checkpoint
    """
    validation_service = ValidationService(db)

    result = await validation_service.save_validation_result(
        job_id=job_id,
        epoch=validation_data.epoch,
        task_type=validation_data.task_type,
        primary_metric_value=validation_data.primary_metric_value,
        metrics=validation_data.metrics,
        per_class_metrics=validation_data.per_class_metrics,
        confusion_matrix=validation_data.confusion_matrix,
        pr_curves=validation_data.pr_curves
    )

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), {
        "type": "validation_complete",
        "epoch": validation_data.epoch,
        "is_best": result.is_best,
        "primary_metric": {
            "name": result.primary_metric_name,
            "value": result.primary_metric_value
        },
        "metrics": validation_data.metrics
    })

    return {"status": "ok", "is_best": result.is_best}
```

### Trainer Implementation Example

```python
# platform/trainers/ultralytics/train.py
from utils import CallbackClient, StorageClient

def run_validation(model, val_loader, epoch):
    """Run validation and report results"""

    # 1. Run validation
    results = model.val()

    # 2. Extract metrics
    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "loss": float(results.box.loss)
    }

    # 3. Per-class metrics
    per_class_metrics = []
    for i, class_name in enumerate(results.names.values()):
        per_class_metrics.append({
            "class_id": i,
            "class_name": class_name,
            "ap": float(results.box.ap[i]),
            "precision": float(results.box.p[i]),
            "recall": float(results.box.r[i])
        })

    # 4. Upload per-image results to S3 (if available)
    image_results_path = None
    if results.save_json:
        image_results = extract_image_results(results)
        image_results_path = storage_client.upload_validation_images(
            job_id=job_id,
            epoch=epoch,
            image_results=image_results
        )

    # 5. Send validation callback
    callback_client.send_validation(
        epoch=epoch,
        task_type="object_detection",
        primary_metric_name=primary_metric_name,  # From env var
        primary_metric_value=metrics[primary_metric_name],
        metrics=metrics,
        per_class_metrics=per_class_metrics,
        image_results_path=image_results_path,
        num_images_validated=len(val_loader)
    )
```

## Storage Strategy

### Hybrid Approach: Database + S3

**Database** (Fast queries):
- Overall metrics
- Per-class metrics
- Primary metric value
- Best epoch tracking
- Confusion matrix (small)
- PR curves (compressed)

**S3** (Large data):
- Per-image validation results
- Visualization images (if generated)
- Detailed prediction outputs

### S3 Path Structure

```
vision-platform/  (bucket)
└── validation-results/
    ├── job-{job-id-1}/
    │   ├── epoch-5/
    │   │   ├── images.json          # Per-image results
    │   │   └── visualizations/      # Optional: prediction visualizations
    │   │       ├── img001_pred.jpg
    │   │       └── img002_pred.jpg
    │   ├── epoch-10/
    │   │   └── images.json
    │   └── epoch-15/
    │       └── images.json
    └── job-{job-id-2}/
        └── epoch-20/
            └── images.json
```

### Per-Image Results Format (S3)

```json
{
  "format_version": "1.0",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "epoch": 10,
  "task_type": "object_detection",
  "num_images": 500,
  "generated_at": "2025-01-10T12:34:56Z",

  "images": [
    {
      "image_id": "val_001.jpg",
      "image_path": "datasets/abc123/images/val/val_001.jpg",

      "predictions": {
        "boxes": [
          {
            "class_id": 0,
            "class_name": "cat",
            "bbox": [10, 20, 100, 150],
            "confidence": 0.95
          }
        ]
      },

      "ground_truth": {
        "boxes": [
          {
            "class_id": 0,
            "class_name": "cat",
            "bbox": [12, 22, 98, 148]
          }
        ]
      },

      "metrics": {
        "iou": 0.85,
        "precision": 1.0,
        "recall": 1.0,
        "num_tp": 1,
        "num_fp": 0,
        "num_fn": 0
      }
    }
    // ... 499 more images
  ]
}
```

## Frontend Integration

### Metrics Display Strategy

```typescript
// components/validation/MetricsDisplay.tsx
import { TaskType } from '@/types/training';

interface MetricsDisplayProps {
  validationResult: TrainingValidationResult;
  taskType: TaskType;
}

export function MetricsDisplay({ validationResult, taskType }: MetricsDisplayProps) {
  const knownMetrics = TASK_STANDARD_METRICS[taskType] || [];
  const allMetricKeys = Object.keys(validationResult.metrics);
  const customMetrics = allMetricKeys.filter(k => !knownMetrics.includes(k));

  return (
    <div className="space-y-6">
      {/* Primary Metric - Highlighted */}
      <PrimaryMetricCard
        name={validationResult.primary_metric_name}
        value={validationResult.primary_metric_value}
        isBest={validationResult.is_best}
        direction={getMetricDirection(validationResult.primary_metric_name)}
      />

      {/* Standard Metrics - Task-specific charts */}
      <StandardMetricsPanel
        taskType={taskType}
        metrics={validationResult.metrics}
        knownMetrics={knownMetrics}
      />

      {/* Per-Class Metrics */}
      {validationResult.per_class_metrics && (
        <PerClassMetricsTable
          classes={validationResult.per_class_metrics}
          taskType={taskType}
        />
      )}

      {/* Visualization */}
      {taskType === 'image_classification' && validationResult.confusion_matrix && (
        <ConfusionMatrixHeatmap matrix={validationResult.confusion_matrix} />
      )}

      {taskType === 'object_detection' && validationResult.pr_curves && (
        <PRCurvesChart curves={validationResult.pr_curves} />
      )}

      {/* Custom Metrics - Generic table */}
      {customMetrics.length > 0 && (
        <CustomMetricsSection>
          <h3>Additional Metrics</h3>
          <table>
            {customMetrics.map(key => (
              <MetricRow
                key={key}
                name={key}
                value={validationResult.metrics[key]}
              />
            ))}
          </table>
        </CustomMetricsSection>
      )}
    </div>
  );
}
```

### Standard Metrics Configuration

```typescript
// config/metrics.ts
export const TASK_STANDARD_METRICS: Record<TaskType, string[]> = {
  'image_classification': [
    'accuracy',
    'top5_accuracy',
    'loss',
    'precision',
    'recall',
    'f1_score'
  ],
  'object_detection': [
    'mAP50',
    'mAP50-95',
    'precision',
    'recall',
    'loss'
  ],
  'instance_segmentation': [
    'mAP_mask',
    'mAP_bbox',
    'mIoU',
    'loss'
  ],
  'semantic_segmentation': [
    'mIoU',
    'pixel_accuracy',
    'dice',
    'loss'
  ],
  'pose_estimation': [
    'OKS',
    'PCK',
    'mAP',
    'loss'
  ]
};

export const METRIC_DISPLAY_CONFIG: Record<string, MetricConfig> = {
  'accuracy': {
    label: 'Accuracy',
    format: 'percentage',
    chartType: 'line',
    color: 'blue'
  },
  'mAP50': {
    label: 'mAP@0.5',
    format: 'percentage',
    chartType: 'line',
    color: 'green'
  },
  'loss': {
    label: 'Loss',
    format: 'float',
    chartType: 'line',
    color: 'red',
    inverted: true  // Lower is better
  }
  // ... more configurations
};
```

## Task-Specific Examples

### Image Classification

```python
# Trainer callback
callback_client.send_validation(
    epoch=10,
    task_type="image_classification",
    primary_metric_name="accuracy",
    primary_metric_value=0.9523,
    metrics={
        "accuracy": 0.9523,
        "top5_accuracy": 0.9987,
        "loss": 0.1234,
        "precision": 0.9456,
        "recall": 0.9389,
        "f1_score": 0.9422
    },
    per_class_metrics=[
        {
            "class_id": 0,
            "class_name": "cat",
            "precision": 0.95,
            "recall": 0.92,
            "f1": 0.935,
            "support": 120
        },
        {
            "class_id": 1,
            "class_name": "dog",
            "precision": 0.94,
            "recall": 0.96,
            "f1": 0.950,
            "support": 105
        }
    ],
    confusion_matrix=[
        [110, 10],
        [4, 101]
    ]
)
```

### Object Detection

```python
# Trainer callback
callback_client.send_validation(
    epoch=20,
    task_type="object_detection",
    primary_metric_name="mAP50-95",
    primary_metric_value=0.6523,
    metrics={
        "mAP50": 0.8721,
        "mAP50-95": 0.6523,
        "precision": 0.8912,
        "recall": 0.8234,
        "loss": 0.0345
    },
    per_class_metrics=[
        {
            "class_id": 0,
            "class_name": "car",
            "ap": 0.92,
            "precision": 0.90,
            "recall": 0.88,
            "support": 350
        },
        {
            "class_id": 1,
            "class_name": "person",
            "ap": 0.87,
            "precision": 0.85,
            "recall": 0.82,
            "support": 420
        }
    ],
    pr_curves={
        "car": {
            "precision": [0.95, 0.93, 0.90, 0.87, ...],
            "recall": [0.1, 0.2, 0.3, 0.4, ...]
        },
        "person": {
            "precision": [0.92, 0.89, 0.85, 0.82, ...],
            "recall": [0.1, 0.2, 0.3, 0.4, ...]
        }
    }
)
```

### Instance Segmentation

```python
# Trainer callback
callback_client.send_validation(
    epoch=15,
    task_type="instance_segmentation",
    primary_metric_name="mAP_mask",
    primary_metric_value=0.7823,
    metrics={
        "mAP_mask": 0.7823,
        "mAP_bbox": 0.8012,
        "mIoU": 0.7654,
        "loss": 0.0456
    },
    per_class_metrics=[
        {
            "class_id": 0,
            "class_name": "person",
            "ap_mask": 0.85,
            "ap_bbox": 0.87,
            "iou": 0.82
        },
        {
            "class_id": 1,
            "class_name": "car",
            "ap_mask": 0.78,
            "ap_bbox": 0.81,
            "iou": 0.76
        }
    ]
)
```

## Best Practices

### For Trainer Developers

1. **Always send primary_metric_value**: Even if it's the same as a metric in the `metrics` dict
2. **Include standard metrics**: Use task-standard names for better visualization
3. **Add custom metrics freely**: Any metric not in standard list will be displayed generically
4. **Upload large data to S3**: Per-image results should go to S3, not in callback JSON
5. **Send validation regularly**: At least every checkpoint save

### For Backend Developers

1. **Never assume metric names**: Use JSON for flexibility
2. **Index by job + epoch**: Fast queries for validation history
3. **Track best checkpoint automatically**: Based on primary metric
4. **Broadcast WebSocket updates**: Real-time validation results to frontend
5. **Clean up old S3 data**: Implement retention policy for per-image results

### For Frontend Developers

1. **Handle unknown metrics gracefully**: Display in generic table
2. **Use task-specific visualizations**: Confusion matrix for classification, PR curves for detection
3. **Highlight primary metric**: Make it prominent in UI
4. **Show best epoch clearly**: Visual indicator for best validation result
5. **Lazy-load per-image data**: Only fetch from S3 when user requests detailed analysis

## References

- [Architecture Overview](./OVERVIEW.md)
- [Backend Design](./BACKEND_DESIGN.md)
- [Trainer Design](./TRAINER_DESIGN.md)
- [Model Weight Management](./MODEL_WEIGHT_MANAGEMENT.md)
- [3-Tier Development](../development/3_TIER_DEVELOPMENT.md)
