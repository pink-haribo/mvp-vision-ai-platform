# Ultralytics YOLO Trainer

Simple CLI-based trainer for YOLO models (YOLOv8, YOLO11, etc.)

## Features

- ✅ CLI-based (no REST API complexity)
- ✅ S3 dataset download and checkpoint upload
- ✅ HTTP callbacks to Backend for progress/completion
- ✅ MLflow experiment tracking
- ✅ DICEFormat → YOLO auto-conversion
- ✅ Proper exit codes for K8s Job compatibility

## Files

```
ultralytics/
├── train.py           # Main training script (~330 lines)
├── utils.py           # S3Client, CallbackClient, dataset helpers (~270 lines)
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container image
└── README.md          # This file
```

**Total: ~600 lines of code** (vs 1000+ lines in old structure)

## Usage

### Local Development (Subprocess)

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py \
    --job-id 123 \
    --model-name yolov8n \
    --dataset-s3-uri s3://bucket/datasets/abc-123/ \
    --callback-url http://localhost:8000/api/v1/training \
    --config '{"epochs": 50, "batch": 16, "imgsz": 640}'
```

### Using Environment Variables

```bash
export JOB_ID=123
export MODEL_NAME=yolov8n
export DATASET_S3_URI=s3://bucket/datasets/abc-123/
export CALLBACK_URL=http://localhost:8000/api/v1/training
export CONFIG='{"epochs": 50, "batch": 16}'

python train.py
```

### Docker (K8s Job)

```bash
# Build image
docker build -t trainer-ultralytics:latest .

# Run container
docker run --env-file .env trainer-ultralytics:latest \
    python train.py --job-id 123 --model-name yolov8n ...
```

## Configuration

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--job-id` | Training job ID | `123` |
| `--model-name` | YOLO model name | `yolov8n`, `yolo11n` |
| `--dataset-s3-uri` | S3 URI to dataset | `s3://bucket/datasets/abc/` |
| `--callback-url` | Backend API base URL | `http://localhost:8000/api/v1/training` |
| `--config` | Training config JSON | `'{"epochs": 50}'` |

### Training Config Options

```json
{
  "epochs": 50,
  "batch": 16,
  "imgsz": 640,
  "device": "cpu",
  "task": "detect",
  "primary_metric": "mAP50-95",
  "split_config": {
    "splits": {
      "img_001": "train",
      "img_002": "val"
    }
  }
}
```

### Environment Variables

```bash
# Storage (S3-compatible)
S3_ENDPOINT=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
S3_BUCKET=training-datasets

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Callback settings
CALLBACK_INTERVAL=1  # Send progress every N epochs
```

## Exit Codes

- `0` = Training completed successfully
- `1` = Training failed (exception during training)
- `2` = Callback failed (training succeeded but notification failed)

## Callbacks

The trainer sends HTTP callbacks to Backend:

### Progress Callback (per epoch)

```http
POST {CALLBACK_URL}/jobs/{JOB_ID}/callback/progress
{
  "job_id": 123,
  "status": "running",
  "current_epoch": 5,
  "total_epochs": 50,
  "progress_percent": 10.0,
  "metrics": {
    "extra_metrics": {...}
  }
}
```

### Completion Callback (once)

```http
POST {CALLBACK_URL}/jobs/{JOB_ID}/callback/completion
{
  "job_id": 123,
  "status": "completed",
  "final_metrics": {
    "mAP50": 0.87,
    "mAP50-95": 0.65
  },
  "checkpoint_path": "s3://bucket/checkpoints/123/best.pt",
  "mlflow_run_id": "abc-def-ghi"
}
```

## Dataset Format

Supports two formats:

### 1. YOLO Format (native)

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### 2. DICEFormat (auto-converted)

```
dataset/
├── images/
└── annotations.json  # COCO-style annotations
```

The trainer automatically converts DICEFormat to YOLO format.

## Adding New Model

To add support for a new YOLO model (e.g., YOLO-World):

1. Ensure Ultralytics library supports it
2. Use model name in `--model-name` argument
3. That's it! No code changes needed.

Example:
```bash
python train.py --model-name yolo_world_v2_s ...
```

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with debug logging
python train.py --log-level DEBUG ...

# Test with small dataset
python train.py \
    --job-id test \
    --model-name yolov8n \
    --dataset-s3-uri s3://bucket/datasets/coco8/ \
    --callback-url http://localhost:8000/api/v1/training \
    --config '{"epochs": 2, "batch": 2}'
```

## Troubleshooting

**Model file not found:**
```bash
# Download model manually
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
```

**S3 connection error:**
- Check `S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Verify MinIO/S3 is running

**Callback 404 error:**
- Check Backend is running at `CALLBACK_URL`
- Verify Backend has `/callback/progress` and `/callback/completion` endpoints

## Plugin Development

To create a new trainer for a different framework (e.g., `timm`, `huggingface`):

1. Copy this directory: `cp -r ultralytics/ timm/`
2. Modify `train.py` to use your framework
3. Update `requirements.txt` with framework dependencies
4. Keep the same CLI interface and callbacks
5. Done!

The key is maintaining the **same CLI interface** so Backend can use any trainer interchangeably.
