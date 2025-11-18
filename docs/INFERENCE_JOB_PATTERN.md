# Inference Job Pattern Architecture

**Last Updated**: 2025-11-18
**Status**: Design Approved - Implementation In Progress

## Overview

This document defines the unified job execution pattern for Training, Validation, and Inference operations. All three follow the same architecture for consistency and K8s Job compatibility.

## Design Principle

**"Train, Validate, Infer - Same Pattern"**

All job types use:
1. Storage-based image access (S3/MinIO)
2. Environment variable injection for configuration
3. CLI script execution (train.py, evaluate.py, predict.py)
4. Callback API for result reporting
5. Backend DB update from callback
6. Frontend DB query for display

## Unified Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│  - Upload images → S3                                            │
│  - Create Job (POST /api/v1/.../jobs)                           │
│  - Poll status (GET /api/v1/.../jobs/{id})                      │
│  - Fetch results from DB (GET /api/v1/.../jobs/{id}/results)   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  1. Create Job record in DB (status: pending)                   │
│  2. Launch background task / K8s Job                             │
│  3. Inject environment variables                                 │
│  4. Receive callback from trainer                                │
│  5. Update DB with results                                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Trainer (Subprocess/K8s Job)                   │
│  - Read env vars (JOB_ID, CALLBACK_URL, S3_PATHS, etc.)        │
│  - Download images from S3                                       │
│  - Execute task (train/validate/infer)                          │
│  - Send logs → Loki                                              │
│  - Send metrics → Callback API                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Job Type Comparison

| Aspect | Training | Validation | Inference |
|--------|----------|------------|-----------|
| **Script** | `train.py` | `evaluate.py` | `predict.py` |
| **Input Storage** | External (datasets) | External (datasets) | Internal (checkpoints) |
| **Output Storage** | Internal (checkpoints) | Internal (validation) | Internal (inference) |
| **Callback URL** | `/training/jobs/{id}/callback/progress` | `/validation/jobs/{id}/results` | `/inference/{id}/results` |
| **DB Model** | TrainingJob, TrainingMetric | ValidationResult | InferenceJob, InferenceResult |
| **Logs** | Loki | Loki | Loki |
| **Additional** | MLflow tracking | Confusion matrix | Per-image predictions |

## Storage Organization

### External Storage (training-datasets)
**Purpose**: User-uploaded datasets for training

```
s3://training-datasets/
└── datasets/
    └── {dataset_id}/
        ├── images/
        │   ├── img_001.jpg
        │   └── ...
        └── annotations/
            ├── img_001.json
            └── ...
```

**Access**: Both Training and Validation

### Internal Storage (training-checkpoints)
**Purpose**: Platform-generated artifacts (checkpoints, results)

```
s3://training-checkpoints/
├── checkpoints/
│   └── {job_id}/
│       ├── best.pt
│       └── last.pt
│
├── inference/
│   └── {inference_job_id}/
│       ├── img_001.jpg
│       └── img_002.jpg
│
└── validation/
    └── {job_id}/
        ├── confusion_matrix.png
        └── confusion_matrix_normalized.png
```

**Access**: Inference uses checkpoints + inference directories

## Environment Variables Pattern

### Training Job

```bash
# Job Identity
JOB_ID=23

# Callback
CALLBACK_URL=http://backend:8000/api/v1/training

# Data
DATASET_PATH=s3://training-datasets/datasets/{dataset_id}/
OUTPUT_PATH=s3://training-checkpoints/checkpoints/23/

# Config (from TrainingJob model)
MODEL_NAME=yolo11n
TASK=detection
EPOCHS=100
BATCH_SIZE=32
DEVICE=cpu
PRIMARY_METRIC=mAP50
PRIMARY_METRIC_MODE=max

# Observability
MLFLOW_TRACKING_URI=http://mlflow:5000
LOKI_URL=http://loki:3100
```

### Validation Job

```bash
# Job Identity
JOB_ID=23

# Callback
CALLBACK_URL=http://backend:8000/api/v1/validation/jobs/23/results

# Data
DATASET_PATH=s3://training-datasets/datasets/{dataset_id}/
CHECKPOINT_PATH=s3://training-checkpoints/checkpoints/23/best.pt
OUTPUT_PATH=s3://training-checkpoints/validation/23/

# Config
SPLIT=test
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45

# Observability
LOKI_URL=http://loki:3100
```

### Inference Job (NEW - To Implement)

```bash
# Job Identity
INFERENCE_JOB_ID=456

# Callback
CALLBACK_URL=http://backend:8000/api/v1/test_inference/inference/456/results

# Data
IMAGE_PATHS=s3://training-checkpoints/inference/456/
CHECKPOINT_PATH=s3://training-checkpoints/checkpoints/23/best.pt
OUTPUT_PATH=s3://training-checkpoints/inference/456/results/

# Config
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45
MAX_DETECTIONS=100

# Observability
LOKI_URL=http://loki:3100
```

## Callback API Flow

### 1. Training Progress Callback

**Endpoint**: `POST /api/v1/training/jobs/{job_id}/callback/progress`

**Payload** (from train.py epoch callback):
```json
{
  "epoch": 10,
  "total_epochs": 100,
  "status": "training",
  "metrics": {
    "train_loss": 0.234,
    "val_loss": 0.189,
    "mAP50": 0.756,
    "mAP50-95": 0.621
  },
  "learning_rate": 0.001,
  "eta_seconds": 1200
}
```

**Backend Action**:
- Create/Update `TrainingMetric` record
- Update `TrainingJob.status`
- Broadcast WebSocket event

### 2. Validation Results Callback

**Endpoint**: `POST /api/v1/validation/jobs/{job_id}/results`

**Payload** (from evaluate.py):
```json
{
  "status": "completed",
  "metrics": {
    "overall_loss": 0.234,
    "mAP50": 0.756,
    "mAP50-95": 0.621,
    "precision": 0.812,
    "recall": 0.789
  },
  "per_class_metrics": [...],
  "confusion_matrix": [...],
  "visualization_urls": {
    "confusion_matrix": "s3://...",
    "confusion_matrix_normalized": "s3://..."
  }
}
```

**Backend Action**:
- Create `ValidationResult` record
- Update validation metrics in TrainingJob

### 3. Inference Results Callback (NEW)

**Endpoint**: `POST /api/v1/test_inference/inference/{inference_job_id}/results`

**Payload** (from predict.py):
```json
{
  "status": "completed",
  "total_images": 10,
  "total_inference_time_ms": 2340.5,
  "avg_inference_time_ms": 234.05,
  "results": [
    {
      "image_path": "s3://training-checkpoints/inference/456/img_001.jpg",
      "predictions": [
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.85,
          "bbox": [100, 200, 300, 400]
        }
      ],
      "inference_time_ms": 218.5
    }
  ]
}
```

**Backend Action**:
- Update `InferenceJob` (status, total_images, avg_time)
- Create `InferenceResult` records for each image
- Store predictions in DB

## Database Models

### InferenceJob

```python
class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))
    checkpoint_path = Column(String(500))

    # Input
    inference_type = Column(String(20))  # single, batch, dataset
    input_data = Column(JSON)

    # Status
    status = Column(String(20))  # pending, running, completed, failed
    error_message = Column(Text)

    # Performance
    total_images = Column(Integer, default=0)
    total_inference_time_ms = Column(Float)
    avg_inference_time_ms = Column(Float)

    # Timestamps
    created_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
```

### InferenceResult

```python
class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True)
    inference_job_id = Column(Integer, ForeignKey("inference_jobs.id"))

    # Image
    image_path = Column(String(500))
    image_name = Column(String(200))

    # Predictions (JSON array)
    predictions = Column(JSON)

    # Performance
    inference_time_ms = Column(Float)
    num_detections = Column(Integer)

    # Timestamps
    created_at = Column(DateTime)
```

## Frontend Flow

### Current (Quick Inference - DEPRECATED)

```typescript
// ❌ OLD: Synchronous HTTP, no DB storage
const result = await fetch('/test_inference/inference/quick', {
  method: 'POST',
  body: params
})
const data = await result.json()  // Direct response
setImages(prev => [...prev, { result: data }])  // Memory only
```

**Problems**:
- No DB storage (volatile)
- No job tracking
- No K8s compatibility
- Page refresh loses data

### New (Inference Job Pattern)

```typescript
// ✅ NEW: Async job with DB storage
// Step 1: Upload images to S3
const uploadedPaths = await uploadImagesToS3(images)

// Step 2: Create InferenceJob
const response = await fetch('/api/v1/test_inference/inference/jobs', {
  method: 'POST',
  body: JSON.stringify({
    training_job_id: jobId,
    checkpoint_path: selectedEpoch.checkpoint_path,
    inference_type: 'batch',
    input_data: {
      image_paths: uploadedPaths,
      confidence_threshold: 0.25,
      iou_threshold: 0.45,
      max_detections: 100
    }
  })
})

const { id: inferenceJobId } = await response.json()

// Step 3: Poll status
const pollInterval = setInterval(async () => {
  const status = await fetch(`/api/v1/test_inference/inference/jobs/${inferenceJobId}`)
  const job = await status.json()

  if (job.status === 'completed') {
    // Step 4: Fetch results from DB
    const resultsResp = await fetch(
      `/api/v1/test_inference/inference/jobs/${inferenceJobId}/results`
    )
    const results = await resultsResp.json()

    // Display results
    setInferenceResults(results)
    clearInterval(pollInterval)
  } else if (job.status === 'failed') {
    setError(job.error_message)
    clearInterval(pollInterval)
  }
}, 1000)
```

**Benefits**:
- ✅ DB storage (persistent)
- ✅ Job tracking
- ✅ K8s compatible
- ✅ Page refresh safe

## K8s Job Deployment

### Local Development (Subprocess)

```python
# Backend: run_inference_task()
env = {
    'INFERENCE_JOB_ID': str(inference_job_id),
    'CALLBACK_URL': callback_url,
    'IMAGE_PATHS': image_paths,
    'CHECKPOINT_PATH': checkpoint_path,
    ...
}

subprocess.Popen(
    [trainer_python, predict_script],
    env=env
)
```

### Production (K8s Job)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-job-456
spec:
  template:
    spec:
      containers:
      - name: predict
        image: trainer-ultralytics:latest
        command: ["python", "predict.py"]
        env:
        - name: INFERENCE_JOB_ID
          value: "456"
        - name: CALLBACK_URL
          value: "http://backend:8000/api/v1/test_inference/inference/456/results"
        - name: IMAGE_PATHS
          value: "s3://training-checkpoints/inference/456/"
        - name: CHECKPOINT_PATH
          value: "s3://training-checkpoints/checkpoints/23/best.pt"
        - name: CONFIDENCE_THRESHOLD
          value: "0.25"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: s3-credentials
              key: access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: s3-credentials
              key: secret-key
      restartPolicy: Never
```

**Same CLI interface, different execution environment!**

## Implementation Checklist

### Phase 1: Backend Infrastructure
- [ ] Update InferenceJob model (add missing fields)
- [ ] Update InferenceResult model (add predictions JSON)
- [ ] Implement `run_inference_task()` function
- [ ] Add S3 image upload endpoint
- [ ] Implement inference results callback endpoint

### Phase 2: Trainer Script (predict.py)
- [ ] Add environment variable reading
- [ ] Add S3 image download logic
- [ ] Implement batch inference loop
- [ ] Add callback API integration
- [ ] Add Loki logging integration

### Phase 3: Frontend
- [ ] Add S3 image upload function
- [ ] Update InferenceJob creation API call
- [ ] Implement status polling
- [ ] Update result display to use DB data
- [ ] Remove old quick inference code

### Phase 4: Testing
- [ ] Test local subprocess execution
- [ ] Test callback flow
- [ ] Test DB storage/retrieval
- [ ] Test page refresh persistence
- [ ] Prepare K8s Job manifest

## Migration Path

### Deprecation of Quick Inference

**Old Endpoint**: `POST /test_inference/inference/quick`
**Status**: DEPRECATED - Will be removed in future version
**Replacement**: `POST /test_inference/inference/jobs` (InferenceJob pattern)

**Migration Steps**:
1. Implement new InferenceJob pattern
2. Update frontend to use new pattern
3. Mark old endpoint as deprecated (add warning log)
4. Remove old endpoint after 1 version

## Benefits of Unified Pattern

1. **Consistency**: Same pattern for train/validate/infer
2. **K8s Ready**: Environment variables → K8s Job compatible
3. **Scalable**: Background tasks, no request timeout
4. **Persistent**: DB storage, survives page refresh
5. **Observable**: Logs → Loki, Metrics → Callback
6. **Testable**: Same interface for local/K8s testing

## Related Documents

- [Training Subprocess Implementation](../platform/backend/app/utils/training_subprocess.py)
- [Validation Results Schema](../platform/backend/app/schemas/validation.py)
- [MVP to Platform Checklist](./planning/MVP_TO_PLATFORM_CHECKLIST.md)

## Changelog

- **2025-11-18**: Initial design document created
- **2025-11-18**: Architecture approved, implementation started
