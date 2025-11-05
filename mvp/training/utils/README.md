# Training Utilities

This directory contains utility scripts for managing the Vision AI Training Platform.

## Upload Pretrained Weights

### Purpose

The `upload_pretrained_weights.py` script proactively uploads pretrained model weights to R2 storage. This ensures:

- **Faster training starts**: No need to download from external sources on first run
- **Consistent versions**: All deployments use the same pretrained weights
- **Reduced external dependencies**: Works even if original sources are slow/unavailable
- **Cost optimization**: Shared cache across all training jobs

### R2 Path Structure

Pretrained weights are stored using this standardized path:

```
r2://vision-platform-prod/models/pretrained/{framework}/{model_name}.{ext}
```

**Examples:**
- `r2://vision-platform-prod/models/pretrained/ultralytics/yolo11n.pt`
- `r2://vision-platform-prod/models/pretrained/timm/tf_efficientnetv2_s.in1k.pth`

### Usage

#### Prerequisites

1. **Set R2 credentials** in your environment or `.env` file:
   ```bash
   AWS_S3_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

2. **Install dependencies**:
   ```bash
   # For Ultralytics
   cd mvp/training
   source venv-ultralytics/Scripts/activate  # Windows
   # source venv-ultralytics/bin/activate      # Linux/Mac
   pip install ultralytics boto3

   # For timm
   source venv-timm/Scripts/activate  # Windows
   # source venv-timm/bin/activate      # Linux/Mac
   pip install timm torch boto3
   ```

#### Commands

**Dry run** (list models without uploading):
```bash
python utils/upload_pretrained_weights.py --framework all --dry-run
```

**Upload Ultralytics weights**:
```bash
cd mvp/training
source venv-ultralytics/Scripts/activate
python utils/upload_pretrained_weights.py --framework ultralytics
```

**Upload timm weights**:
```bash
cd mvp/training
source venv-timm/Scripts/activate
python utils/upload_pretrained_weights.py --framework timm
```

**Upload all weights**:
```bash
# Run ultralytics first
cd mvp/training
source venv-ultralytics/Scripts/activate
python utils/upload_pretrained_weights.py --framework ultralytics
deactivate

# Then run timm
source venv-timm/Scripts/activate
python utils/upload_pretrained_weights.py --framework timm
```

### Currently Registered Models

#### Ultralytics (5 models)
- `yolo11n` - YOLOv11 Nano (object detection)
- `yolo11n-seg` - YOLOv11 Nano Segmentation
- `yolo11n-pose` - YOLOv11 Nano Pose Estimation
- `yolo_world_v2_s` - YOLO-World v2 Small (zero-shot detection)
- `sam2_t` - SAM2 Tiny (zero-shot segmentation)

#### timm (3 models)
- `tf_efficientnetv2_s.in1k` - EfficientNetV2-Small
- `convnext_tiny` - ConvNeXt-Tiny
- `vit_base_patch16_224` - Vision Transformer Base

### How It Works

1. **Download**: The script loads each model using the respective library (Ultralytics/timm), which triggers automatic download to local cache
2. **Locate**: Finds the downloaded weight file in the library's cache directory
3. **Upload**: Uploads to R2 using the standardized path structure
4. **Skip**: If weights already exist in R2, they are skipped

### Integration with Training

The training platform's `platform_sdk/storage.py` uses a **fallback strategy**:

1. **Local cache** (`/workspace/data/.cache/models/{framework}/`) - fastest
2. **R2 storage** (`r2://vision-platform-prod/models/pretrained/`) - fast, shared
3. **Original source** (Ultralytics/timm servers) - slow, triggers auto-upload to R2

This script pre-populates R2 (step 2), so training jobs never need to hit external servers.

### Troubleshooting

**"R2 credentials not configured"**
- Check that environment variables are set correctly
- Verify the endpoint URL format (should start with `https://`)

**"Could not locate downloaded weights"**
- Ultralytics weights are typically cached in `~/.cache/ultralytics/` or `~/.cache/torch/hub/ultralytics/`
- timm weights are downloaded on model creation and saved manually

**"Already exists in R2"**
- This is normal! The script skips uploading if weights are already present
- To force re-upload, delete the existing R2 object first

**"Model weight file mapping unknown"**
- The model's weight filename might differ from the registry name
- Check `MODEL_WEIGHT_MAPPING` in the script and update if needed

### Adding New Models

When adding new models to the registry:

1. **Add to model registry** (`model_registry/ultralytics_models.py` or `timm_models.py`)
2. **Update weight mapping** (if needed) in `upload_pretrained_weights.py`
3. **Run upload script**:
   ```bash
   python utils/upload_pretrained_weights.py --framework <framework>
   ```

### Notes

- **Storage cost**: Each weight file is typically 5-200 MB. Total storage for all models: ~1-2 GB
- **Upload time**: First upload takes 5-15 minutes depending on internet speed
- **Idempotent**: Safe to run multiple times - skips existing weights
- **Framework isolation**: Ultralytics and timm weights are uploaded separately (different venvs)
