# Training Services

**Framework-specific training implementations** following the API Contract.

## Directory Structure

```
trainers/
├── ultralytics/      # Ultralytics YOLO (YOLOv11, SAM2)
├── timm/             # PyTorch Image Models (Classification)
├── huggingface/      # HuggingFace Transformers
│
# OpenMMLab Framework
├── mmdet/            # MMDetection (Object Detection)
│   └── Faster R-CNN, DETR, DINO, RTMDet, Co-DETR
├── mmpretrain/       # MMPreTrain (Classification)
│   └── ResNet, Swin, ConvNeXt, ViT, EfficientNet
├── mmseg/            # MMSegmentation (Semantic Segmentation)
│   └── DeepLabV3+, SegFormer, Mask2Former, UperNet
└── mmyolo/           # MMYOLO (YOLO Series)
    └── YOLOv5-v8, RTMDet, PPYOLOE
```

## Framework Comparison

| Framework | Task Types | Key Models | Use Case |
|-----------|-----------|------------|----------|
| ultralytics | Detection, Segmentation, Pose | YOLO11, SAM2 | 빠른 배포, 통합 API |
| timm | Classification | ResNet, EfficientNet, ViT | 단순 분류 |
| mmdet | Detection | DINO, Co-DETR, Faster R-CNN | SOTA Detection |
| mmpretrain | Classification, SSL | Swin, MAE, BEiT | 고급 분류, 사전학습 |
| mmseg | Segmentation | Mask2Former, SegFormer | Semantic Segmentation |
| mmyolo | Detection | YOLOv5-v8, RTMDet | YOLO 계열 비교 연구 |

## OpenMMLab Shared Dependencies

All OpenMMLab trainers (mmdet, mmpretrain, mmseg, mmyolo) share:
- **mmengine**: Training engine and config system
- **mmcv**: Computer vision utilities (>=2.0.0)

Install via mim:
```bash
pip install openmim
mim install mmengine "mmcv>=2.0.0"
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
