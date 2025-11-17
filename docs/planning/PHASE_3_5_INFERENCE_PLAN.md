# Phase 3.5: Evaluation & Inference CLI Implementation

**목표**: train.py와 동일한 패턴으로 evaluate.py와 predict.py 구현

**예상 시간**: 3-4시간

## Architecture Overview

### Design Principles
1. **Consistent CLI Pattern**: train.py와 동일한 argparse 구조
2. **DualStorageClient Integration**: Checkpoint from Internal, Dataset from External
3. **Backend Callbacks**: Progress and completion callbacks
4. **Error Handling**: Exit codes (0=success, 1=failure, 2=callback failure)
5. **Framework Agnostic**: Ultralytics 먼저, 다른 프레임워크 확장 가능

---

## Part 1: evaluate.py (Model Evaluation)

### Purpose
학습된 모델을 test dataset으로 평가하고 상세한 메트릭 계산

### CLI Interface
```bash
python evaluate.py \
  --test-run-id 123 \
  --training-job-id 456 \
  --checkpoint-s3-uri s3://training-checkpoints/checkpoints/456/best.pt \
  --dataset-s3-uri s3://training-datasets/datasets/test-abc-123/ \
  --callback-url http://localhost:8000/api/v1/test_inference \
  --config '{"conf_threshold": 0.25, "iou_threshold": 0.45, "max_det": 300}'
```

### Arguments
- `--test-run-id`: Backend에서 생성한 Test Run ID
- `--training-job-id`: 원본 Training Job ID (metadata)
- `--checkpoint-s3-uri`: MinIO Internal Storage의 checkpoint 경로
- `--dataset-s3-uri`: MinIO External Storage의 test dataset 경로
- `--callback-url`: Backend API base URL
- `--config`: 평가 설정 (JSON)
  - `conf_threshold`: Confidence threshold (default: 0.25)
  - `iou_threshold`: IoU threshold for NMS (default: 0.45)
  - `max_det`: Maximum detections per image (default: 300)
  - `save_txt`: Save labels as txt (default: False)
  - `save_json`: Save results as JSON (default: True)

### Processing Flow

#### 1. Initialization
```python
def parse_args():
    parser.add_argument('--test-run-id', type=str, required=True)
    parser.add_argument('--training-job-id', type=str, required=True)
    parser.add_argument('--checkpoint-s3-uri', type=str, required=True)
    parser.add_argument('--dataset-s3-uri', type=str, required=True)
    parser.add_argument('--callback-url', type=str, required=True)
    parser.add_argument('--config', type=str, default='{}')
```

#### 2. Storage Operations
```python
# Initialize storage clients
storage = DualStorageClient()
callback_client = CallbackClient(callback_url)

# Download checkpoint from Internal Storage
checkpoint_dir = Path(f"/tmp/evaluate/{test_run_id}/checkpoint")
storage.internal_client.download_file(
    bucket="training-checkpoints",
    key="checkpoints/456/best.pt",
    dest=checkpoint_dir / "best.pt"
)

# Download test dataset from External Storage
dataset_dir = Path(f"/tmp/evaluate/{test_run_id}/dataset")
storage.external_client.download_dataset(
    dataset_id="test-abc-123",
    dest_dir=dataset_dir
)
```

#### 3. Dataset Preparation
```python
# Convert dataset format if needed (DICEFormat → YOLO)
if (dataset_dir / "annotations.json").exists():
    convert_diceformat_to_yolo(dataset_dir)
```

#### 4. Model Evaluation
```python
# Load model
model = YOLO(checkpoint_dir / "best.pt")

# Run validation
results = model.val(
    data=dataset_dir / "data.yaml",
    conf=config.get('conf_threshold', 0.25),
    iou=config.get('iou_threshold', 0.45),
    max_det=config.get('max_det', 300),
    save_json=True,
    save_txt=False,
    plots=True,  # Generate confusion matrix, PR curves
)
```

#### 5. Results Processing
```python
# Extract metrics
metrics = {
    'mAP50': results.box.map50,
    'mAP50-95': results.box.map,
    'precision': results.box.mp,
    'recall': results.box.mr,
    'f1_score': 2 * (precision * recall) / (precision + recall),
}

# Per-class metrics
per_class_metrics = []
for i, class_name in enumerate(results.names.values()):
    per_class_metrics.append({
        'class_name': class_name,
        'precision': results.box.p[i],
        'recall': results.box.r[i],
        'ap50': results.box.ap50[i],
        'ap': results.box.ap[i],
    })
```

#### 6. Upload Results
```python
# Upload plots to MinIO Internal Storage
plots_dir = Path("runs/detect/val")  # Ultralytics default
results_dir = f"test-runs/{test_run_id}"

for plot_file in ['confusion_matrix.png', 'PR_curve.png', 'F1_curve.png']:
    if (plots_dir / plot_file).exists():
        storage.internal_client.upload_file(
            local_path=plots_dir / plot_file,
            s3_key=f"{results_dir}/{plot_file}",
            content_type='image/png'
        )
```

#### 7. Backend Callback
```python
# Send evaluation results to Backend
callback_data = {
    'test_run_id': test_run_id,
    'status': 'completed',
    'metrics': metrics,
    'per_class_metrics': per_class_metrics,
    'visualization_urls': {
        'confusion_matrix': f"s3://training-checkpoints/{results_dir}/confusion_matrix.png",
        'pr_curve': f"s3://training-checkpoints/{results_dir}/PR_curve.png",
        'f1_curve': f"s3://training-checkpoints/{results_dir}/F1_curve.png",
    },
}

await callback_client.send_test_completion(test_run_id, callback_data)
```

### Backend API Integration

**Endpoint**: `POST /api/v1/test_inference/test/{test_run_id}/results`

**Request Body**:
```json
{
  "test_run_id": 123,
  "status": "completed",
  "metrics": {
    "mAP50": 0.678,
    "mAP50-95": 0.456,
    "precision": 0.789,
    "recall": 0.654,
    "f1_score": 0.715
  },
  "per_class_metrics": [
    {
      "class_name": "person",
      "precision": 0.85,
      "recall": 0.78,
      "ap50": 0.82,
      "ap": 0.76
    }
  ],
  "visualization_urls": {
    "confusion_matrix": "s3://...",
    "pr_curve": "s3://...",
    "f1_curve": "s3://..."
  }
}
```

### Exit Codes
- `0`: Evaluation completed successfully
- `1`: Evaluation failed (model loading, dataset error, etc.)
- `2`: Callback failed (network error, Backend unavailable)

---

## Part 2: predict.py (Inference Execution)

### Purpose
학습된 모델로 새로운 이미지에 대한 추론 실행

### CLI Interface
```bash
python predict.py \
  --inference-job-id 789 \
  --training-job-id 456 \
  --checkpoint-s3-uri s3://training-checkpoints/checkpoints/456/best.pt \
  --images-s3-uri s3://inference-images/batch-xyz/ \
  --callback-url http://localhost:8000/api/v1/test_inference \
  --config '{"conf": 0.25, "iou": 0.45, "max_det": 300, "save_txt": true, "save_crop": false}'
```

### Arguments
- `--inference-job-id`: Backend에서 생성한 Inference Job ID
- `--training-job-id`: 원본 Training Job ID (metadata)
- `--checkpoint-s3-uri`: MinIO Internal Storage의 checkpoint 경로
- `--images-s3-uri`: 추론할 이미지들의 S3 URI
- `--callback-url`: Backend API base URL
- `--config`: 추론 설정 (JSON)
  - `conf`: Confidence threshold (default: 0.25)
  - `iou`: IoU threshold for NMS (default: 0.45)
  - `max_det`: Maximum detections per image (default: 300)
  - `save_txt`: Save labels as txt (default: True)
  - `save_crop`: Save cropped detections (default: False)
  - `save_conf`: Save confidences in txt (default: True)
  - `line_width`: Bounding box line width (default: 2)

### Processing Flow

#### 1. Storage Operations
```python
# Download checkpoint
checkpoint_dir = Path(f"/tmp/predict/{inference_job_id}/checkpoint")
storage.internal_client.download_file(...)

# Download input images
images_dir = Path(f"/tmp/predict/{inference_job_id}/images")
storage.download_images(images_s3_uri, images_dir)
```

#### 2. Model Inference
```python
# Load model
model = YOLO(checkpoint_dir / "best.pt")

# Run inference
results = model.predict(
    source=images_dir,
    conf=config.get('conf', 0.25),
    iou=config.get('iou', 0.45),
    max_det=config.get('max_det', 300),
    save_txt=config.get('save_txt', True),
    save_crop=config.get('save_crop', False),
    save_conf=config.get('save_conf', True),
    line_width=config.get('line_width', 2),
    save=True,  # Save annotated images
)
```

#### 3. Results Processing
```python
# Aggregate predictions
predictions = []
for r in results:
    for box in r.boxes:
        predictions.append({
            'image_name': r.path.name,
            'class_id': int(box.cls),
            'class_name': r.names[int(box.cls)],
            'confidence': float(box.conf),
            'bbox': box.xyxy.tolist()[0],  # [x1, y1, x2, y2]
        })

# Create predictions summary
summary = {
    'total_images': len(results),
    'total_detections': len(predictions),
    'classes_detected': list(set(p['class_name'] for p in predictions)),
    'avg_confidence': sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0,
}
```

#### 4. Upload Results
```python
# Upload annotated images
results_dir = f"inference-results/{inference_job_id}"
output_dir = Path("runs/detect/predict")  # Ultralytics default

# Upload annotated images
for img_file in output_dir.glob("*.jpg"):
    storage.internal_client.upload_file(
        local_path=img_file,
        s3_key=f"{results_dir}/images/{img_file.name}",
        content_type='image/jpeg'
    )

# Upload labels (if save_txt=True)
labels_dir = output_dir / "labels"
if labels_dir.exists():
    for txt_file in labels_dir.glob("*.txt"):
        storage.internal_client.upload_file(
            local_path=txt_file,
            s3_key=f"{results_dir}/labels/{txt_file.name}",
            content_type='text/plain'
        )

# Upload predictions summary
import json
summary_file = Path(f"/tmp/predict/{inference_job_id}/summary.json")
with open(summary_file, 'w') as f:
    json.dump({'predictions': predictions, 'summary': summary}, f, indent=2)

storage.internal_client.upload_file(
    local_path=summary_file,
    s3_key=f"{results_dir}/predictions.json",
    content_type='application/json'
)
```

#### 5. Backend Callback
```python
callback_data = {
    'inference_job_id': inference_job_id,
    'status': 'completed',
    'summary': summary,
    'results_s3_uri': f"s3://training-checkpoints/{results_dir}/",
    'predictions_url': f"s3://training-checkpoints/{results_dir}/predictions.json",
}

await callback_client.send_inference_completion(inference_job_id, callback_data)
```

### Backend API Integration

**Endpoint**: `POST /api/v1/test_inference/inference/{inference_job_id}/results`

**Request Body**:
```json
{
  "inference_job_id": 789,
  "status": "completed",
  "summary": {
    "total_images": 25,
    "total_detections": 147,
    "classes_detected": ["person", "car", "dog"],
    "avg_confidence": 0.68
  },
  "results_s3_uri": "s3://training-checkpoints/inference-results/789/",
  "predictions_url": "s3://training-checkpoints/inference-results/789/predictions.json"
}
```

---

## Part 3: CallbackClient Extensions (utils.py)

### New Methods

```python
class CallbackClient:
    # ... existing methods ...

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(...))
    def send_test_completion_sync(self, test_run_id: str, data: Dict[str, Any]) -> None:
        """Send test evaluation completion callback"""
        url = f"{self.base_url}/test/{test_run_id}/results"

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            logger.info(f"Test completion callback sent: {test_run_id}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(...))
    def send_inference_completion_sync(self, inference_job_id: str, data: Dict[str, Any]) -> None:
        """Send inference completion callback"""
        url = f"{self.base_url}/inference/{inference_job_id}/results"

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            logger.info(f"Inference completion callback sent: {inference_job_id}")
```

---

## Part 4: Backend API Enhancements

### Required Backend Changes

#### 1. Update test_inference.py

**Add Result Callback Endpoint**:
```python
@router.post("/test/{test_run_id}/results", status_code=200)
async def receive_test_results(
    test_run_id: int,
    request: TestResultsRequest,
    db: Session = Depends(get_db)
):
    """
    Receive test evaluation results from evaluate.py

    Updates TestRun status and stores metrics in database.
    """
    test_run = db.query(models.TestRun).filter(
        models.TestRun.id == test_run_id
    ).first()

    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    # Update test run
    test_run.status = request.status
    test_run.metrics = request.metrics
    test_run.per_class_metrics = request.per_class_metrics
    test_run.visualization_urls = request.visualization_urls
    test_run.completed_at = datetime.utcnow()

    db.commit()

    logger.info(f"[TEST CALLBACK] Test run {test_run_id} completed")
    return {"status": "success"}


@router.post("/inference/{inference_job_id}/results", status_code=200)
async def receive_inference_results(
    inference_job_id: int,
    request: InferenceResultsRequest,
    db: Session = Depends(get_db)
):
    """
    Receive inference results from predict.py

    Updates InferenceJob status and stores result URLs.
    """
    inference_job = db.query(models.InferenceJob).filter(
        models.InferenceJob.id == inference_job_id
    ).first()

    if not inference_job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    # Update inference job
    inference_job.status = request.status
    inference_job.summary = request.summary
    inference_job.results_s3_uri = request.results_s3_uri
    inference_job.predictions_url = request.predictions_url
    inference_job.completed_at = datetime.utcnow()

    db.commit()

    logger.info(f"[INFERENCE CALLBACK] Inference job {inference_job_id} completed")
    return {"status": "success"}
```

#### 2. Add Schemas (test_inference.py or schemas/)

```python
class TestResultsRequest(BaseModel):
    test_run_id: int
    status: str  # "completed", "failed"
    metrics: Dict[str, float]
    per_class_metrics: List[Dict[str, Any]]
    visualization_urls: Dict[str, str]


class InferenceResultsRequest(BaseModel):
    inference_job_id: int
    status: str  # "completed", "failed"
    summary: Dict[str, Any]
    results_s3_uri: str
    predictions_url: str
```

---

## Implementation Checklist

### evaluate.py Implementation
- [ ] Create `platform/trainers/ultralytics/evaluate.py`
- [ ] CLI argument parsing (test_run_id, checkpoint, dataset, config)
- [ ] DualStorageClient integration (checkpoint from Internal, dataset from External)
- [ ] Checkpoint download and verification
- [ ] Dataset download and format conversion
- [ ] Model loading (YOLO)
- [ ] Evaluation execution (model.val())
- [ ] Metrics extraction (mAP, precision, recall, per-class)
- [ ] Results upload (confusion matrix, PR curves, F1 curves)
- [ ] Backend callback (POST /test/{test_run_id}/results)
- [ ] Error handling and exit codes
- [ ] Logging with timestamps

### predict.py Implementation
- [ ] Create `platform/trainers/ultralytics/predict.py`
- [ ] CLI argument parsing (inference_job_id, checkpoint, images, config)
- [ ] DualStorageClient integration
- [ ] Checkpoint download
- [ ] Images download
- [ ] Model loading (YOLO)
- [ ] Inference execution (model.predict())
- [ ] Predictions aggregation
- [ ] Results upload (annotated images, labels, JSON)
- [ ] Backend callback (POST /inference/{inference_job_id}/results)
- [ ] Error handling and exit codes
- [ ] Logging with timestamps

### CallbackClient Extensions (utils.py)
- [ ] Add `send_test_completion_sync()` method
- [ ] Add `send_inference_completion_sync()` method
- [ ] Add retry logic with exponential backoff
- [ ] Add logging for debugging

### Backend API (test_inference.py)
- [ ] Add `POST /test/{test_run_id}/results` endpoint
- [ ] Add `POST /inference/{inference_job_id}/results` endpoint
- [ ] Add `TestResultsRequest` schema
- [ ] Add `InferenceResultsRequest` schema
- [ ] Database updates (TestRun, InferenceJob models)
- [ ] Logging with [TEST CALLBACK], [INFERENCE CALLBACK] prefixes

### Testing
- [ ] E2E test: evaluate.py with trained model
- [ ] E2E test: predict.py with sample images
- [ ] Verify S3 uploads (plots, predictions)
- [ ] Verify Backend callbacks
- [ ] Verify database updates

---

## Expected Benefits

1. **Unified CLI Pattern**: evaluate.py와 predict.py가 train.py와 동일한 구조
2. **Dual Storage Integration**: 자동 라우팅으로 개발자 경험 개선
3. **Backend Integration**: 모든 작업이 Backend에 기록됨
4. **Error Resilience**: Retry logic과 명확한 exit codes
5. **Framework Agnostic**: 다른 프레임워크 (timm, HuggingFace)로 쉽게 확장 가능

---

## Timeline

- **evaluate.py 구현**: 1.5-2시간
- **predict.py 구현**: 1-1.5시간
- **Backend API 수정**: 30분
- **Testing & Debugging**: 30-45분

**Total**: 3-4시간
