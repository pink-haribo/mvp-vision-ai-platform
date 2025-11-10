# Training Services

**Framework-specific training implementations** following the API Contract.

## Directory Structure

```
trainers/
├── ultralytics/      # YOLO models
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
├── timm/            # PyTorch Image Models
├── huggingface/     # Transformers
└── base/            # Common utilities (optional)
```

## API Contract

All trainers must implement:

### Input (Environment Variables)
```bash
JOB_ID=123
TRACE_ID=abc-def-ghi
BACKEND_BASE_URL=https://api.example.com
CALLBACK_TOKEN=eyJhbGc...
TASK_TYPE=object_detection
MODEL_NAME=yolo11n
DATASET_ID=dataset-uuid
STORAGE_TYPE=r2
R2_ENDPOINT=https://...
R2_ACCESS_KEY_ID=xxx
R2_SECRET_ACCESS_KEY=xxx
```

### Output (HTTP Callbacks)
```python
# Heartbeat (per epoch)
POST {BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/heartbeat
{
  "trace_id": "abc-def",
  "epoch": 5,
  "progress": 50.0,
  "metrics": {"loss": 0.234}
}

# Events
POST {BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/event
{
  "trace_id": "abc-def",
  "event_type": "checkpoint_saved",
  "message": "Checkpoint saved"
}

# Completion
POST {BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/done
{
  "trace_id": "abc-def",
  "status": "succeeded",
  "final_metrics": {...}
}
```

### Storage (S3 API)
```python
import boto3

s3 = boto3.client('s3', endpoint_url=R2_ENDPOINT, ...)
s3.download_file(bucket, f"datasets/{DATASET_ID}.zip", "/tmp/dataset.zip")
s3.upload_file("/tmp/best.pt", bucket, f"checkpoints/job-{JOB_ID}/best.pt")
```

## Adding New Framework

1. Create directory: `trainers/yourframework/`
2. Implement API Contract (see above)
3. Create Dockerfile
4. Add to `infrastructure/helm/trainers/`
5. Optional: Copy `utils.py` from existing trainer

See `docs/k8s_refactoring/PLUGIN_GUIDE.md` for detailed guide.

## Development

```bash
# Build Docker image
cd trainers/ultralytics
docker build -t trainer-ultralytics:latest .

# Test locally
docker run --env-file .env trainer-ultralytics:latest
```
