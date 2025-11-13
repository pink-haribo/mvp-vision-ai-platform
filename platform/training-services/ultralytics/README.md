# Ultralytics Training Service

Training service for YOLO models (YOLOv11) with S3 integration and HTTP callbacks.

## Features

- ✅ YOLOv11 model training (detection, segmentation, pose)
- ✅ S3-only storage (MinIO for dev, R2/S3 for production)
- ✅ HTTP callbacks to Platform Backend
- ✅ Async operations with retry logic
- ✅ Background task execution

## Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your S3 credentials
```

### 3. Run Service

```bash
poetry run uvicorn app.main:app --reload --port 8001
```

The service will be available at:
- API: http://localhost:8001
- Docs: http://localhost:8001/docs
- Health: http://localhost:8001/health

## API Endpoints

### Models

- `GET /models` - List available YOLO models

### Training

- `POST /training/start` - Start training job

## Supported Models

### Detection
- `yolo11n` - Nano (2.6M params)
- `yolo11s` - Small (9.4M params)
- `yolo11m` - Medium (20.1M params)
- `yolo11l` - Large (25.3M params)
- `yolo11x` - Extra Large (56.9M params)

### Segmentation
- `yolo11n-seg` - Nano Segmentation
- `yolo11s-seg` - Small Segmentation

### Pose Estimation
- `yolo11n-pose` - Nano Pose
- `yolo11s-pose` - Small Pose

## Training Flow

1. Receive training request from Backend
2. Download dataset from S3
3. Train YOLO model
4. Send progress callbacks to Backend
5. Upload checkpoint to S3
6. Send completion callback

## Callback Format

The service sends callbacks to the Backend with this format:

```json
{
  "job_id": "uuid",
  "status": "running",
  "progress": 0.5,
  "message": "Training: epoch 25/50",
  "metrics": {
    "epoch": 25,
    "loss": 0.234
  }
}
```

Status values: `running`, `completed`, `failed`

## Development

### Run Tests

```bash
poetry run pytest
```

### Code Quality

```bash
poetry run black .
poetry run isort .
poetry run flake8 .
```

## Architecture

The service is completely isolated from the Backend:
- **Communication**: HTTP only (no shared filesystem)
- **Storage**: S3 APIs only (no local storage)
- **State**: Backend database (via callbacks)

See `../../docs/development/IMPLEMENTATION_PLAN.md` for detailed architecture.

## License

Copyright © 2025 Vision AI Platform Team
