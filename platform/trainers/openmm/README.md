# OpenMMLab Trainer

Vision AI Platform trainer for OpenMMLab frameworks (MMDetection, MMSegmentation, MMPose).

## Overview

This trainer provides integration with OpenMMLab ecosystem for computer vision tasks:

- **MMDetection**: Object detection and instance segmentation
- **MMSegmentation**: Semantic segmentation
- **MMPose**: Pose estimation and keypoint detection

## Supported Models

### Object Detection

| Model | Backbone | mAP (COCO) | Speed (FPS) | Params |
|-------|----------|------------|-------------|--------|
| Faster R-CNN | ResNet-50 | 37.4 | 18.2 | 41.5M |
| Mask R-CNN | ResNet-50 | 38.2 | 15.1 | 44.2M |
| RetinaNet | ResNet-50 | 36.5 | 20.3 | 37.7M |
| FCOS | ResNet-50 | 38.5 | 19.7 | 32.1M |
| YOLOX-S | CSPDarknet | 40.5 | 68.4 | 9.0M |
| YOLOX-M | CSPDarknet | 46.4 | 52.3 | 25.3M |
| RTMDet-S | CSPNeXt | 44.5 | 70.2 | 8.9M |
| RTMDet-M | CSPNeXt | 49.1 | 55.8 | 24.7M |

### Instance Segmentation

| Model | Backbone | mAP (box) | mAP (mask) | Speed (FPS) |
|-------|----------|-----------|------------|-------------|
| Mask R-CNN | ResNet-50 | 38.2 | 34.7 | 15.1 |
| Cascade Mask R-CNN | ResNet-50 | 41.2 | 35.7 | 10.2 |

## Features

- ✅ **20+ Pre-trained Models**: Faster R-CNN, Mask R-CNN, YOLOX, RTMDet, etc.
- ✅ **Multi-Task Support**: Detection, segmentation, pose estimation
- ✅ **COCO Format**: Native COCO dataset support
- ✅ **Auto Conversion**: Automatic DICE → COCO conversion
- ✅ **Export**: ONNX, TorchScript, TensorRT export
- ✅ **Mixed Precision**: AMP training for faster training
- ✅ **Data Augmentation**: Mosaic, Mixup, HSV, rotation, scaling

## Quick Start

### Training

```bash
python train.py
```

**Environment Variables**:
```bash
CALLBACK_URL=http://backend:8000
JOB_ID=123
TASK_TYPE=detection
MODEL_NAME=faster-rcnn-r50
DATASET_ID=ds_abc123
SNAPSHOT_ID=snap_xyz
DATASET_VERSION_HASH=1bb25f37...
CONFIG='{"basic": {"epochs": 12, "batch": 2, "lr0": 0.02}}'
```

### Inference

```bash
python predict.py
```

**Environment Variables**:
```bash
CALLBACK_URL=http://backend:8000
JOB_ID=456
CHECKPOINT_PATH=s3://bucket/checkpoints/model.pth
IMAGES_PATH=s3://bucket/test-images/
TASK_TYPE=detection
MODEL_NAME=faster-rcnn-r50
```

### Model Export

```bash
python export.py
```

**Environment Variables**:
```bash
CALLBACK_URL=http://backend:8000
JOB_ID=789
CHECKPOINT_PATH=s3://bucket/checkpoints/model.pth
EXPORT_FORMAT=onnx
EXPORT_CONFIG='{"opset_version": 17, "input_shape": [1, 3, 640, 640]}'
```

## Configuration

### Basic Config

Common parameters across all models:

```json
{
  "basic": {
    "epochs": 12,
    "batch": 2,
    "lr0": 0.02,
    "optimizer": "SGD",
    "device": "0",
    "workers": 4
  }
}
```

### Advanced Config

OpenMMLab-specific parameters:

```json
{
  "advanced": {
    "lr_scheduler": "step",
    "warmup_epochs": 1,
    "warmup_ratio": 0.001,
    "mosaic": false,
    "mixup": false,
    "amp": false,
    "sync_bn": false,
    "val_interval": 1,
    "checkpoint_interval": 1
  }
}
```

## Dataset Format

### Input Format (DICE)

Platform automatically converts DICE format to COCO:

```json
{
  "categories": [
    {"id": 1, "name": "cat"},
    {"id": 2, "name": "dog"}
  ],
  "images": [
    {
      "id": 1,
      "file_name": "images/001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 180]
    }
  ]
}
```

### Output Format (COCO)

MMDetection uses standard COCO format internally.

## Architecture

```
┌─────────────────┐
│   train.py      │ ← Main training script
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
┌────────▼────────┐  ┌──────▼──────┐
│  trainer_sdk.py │  │ MMDetection │
│                 │  │   Engine    │
└────────┬────────┘  └──────┬──────┘
         │                  │
         └──────────┬───────┘
                    │
         ┌──────────▼──────────┐
         │   Backend API       │
         │   (Callbacks)       │
         └─────────────────────┘
```

## Docker Build

```bash
cd platform/trainers/openmm
docker build -t trainer-openmm:latest .
```

**Run Container**:
```bash
docker run \
  -e CALLBACK_URL=http://backend:8000 \
  -e JOB_ID=123 \
  -e MODEL_NAME=faster-rcnn-r50 \
  -e TASK_TYPE=detection \
  --gpus all \
  trainer-openmm:latest
```

## Troubleshooting

### CUDA Out of Memory

**Solution**: Reduce batch size or use smaller model variant.

```json
{
  "basic": {
    "batch": 1  // Reduce from 2 to 1
  }
}
```

### Config File Not Found

**Error**: `Config file not found: configs/faster_rcnn/...`

**Solution**: MMDetection configs are installed with package. Check model name matches exactly.

### Low mAP

**Solution**:
1. Increase training epochs (12 → 24)
2. Enable data augmentation (mosaic, mixup)
3. Use larger model variant (S → M → L)
4. Check dataset quality and annotations

## Performance Tips

1. **Use AMP**: Enable `amp: true` for 30-40% speedup
2. **Batch Size**: Maximize batch size for GPU utilization
3. **Workers**: Set `workers` to number of CPU cores
4. **Model Selection**:
   - Fast inference: YOLOX, RTMDet
   - High accuracy: Faster R-CNN, Cascade R-CNN
   - Balanced: RetinaNet, FCOS

## References

- [MMDetection Documentation](https://mmdetection.readthedocs.io/)
- [OpenMMLab GitHub](https://github.com/open-mmlab)
- [Model Zoo](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/model_zoo.md)

## License

Apache 2.0 (OpenMMLab)

---

**Maintained by**: Vision AI Platform Team
**Last Updated**: 2025-12-22
