# Dynamic Model Registration

## Overview

This document explains how models are dynamically registered in the production environment, eliminating the need for manual/hardcoded model definitions in the Backend service.

## Architecture

### Development (Local) Mode

```
Backend API (/models/list)
    ↓
Direct import: from model_registry import get_all_models()
    ↓
Returns: All models from TIMM_MODEL_REGISTRY + ULTRALYTICS_MODEL_REGISTRY + HUGGINGFACE_MODEL_REGISTRY
```

### Production (Railway) Mode

```
Backend API (/models/list)
    ↓
HTTP GET → Training Services (/models/list)
    ├─→ timm-service.railway.app/models/list
    ├─→ ultralytics-service.railway.app/models/list
    └─→ huggingface-service.railway.app/models/list
    ↓
Merge results from all services
    ↓
Returns: Combined list of all available models
```

## Implementation Details

### Training Service (api_server.py)

Each Training Service exposes two model endpoints:

**GET /models/list**
```json
{
  "framework": "ultralytics",
  "model_count": 5,
  "models": [
    {
      "framework": "ultralytics",
      "model_name": "yolo11n",
      "display_name": "YOLOv11 Nano",
      "description": "...",
      "params": "2.6M",
      "task_types": ["object_detection"],
      ...
    },
    ...
  ]
}
```

**GET /models/{model_name}**
```json
{
  "framework": "ultralytics",
  "model_name": "yolo11n",
  "display_name": "YOLOv11 Nano",
  ...
}
```

### Backend Service (models.py)

The Backend service uses a fallback strategy:

1. **Try Training Services first** (Production mode)
   - Fetch models from `$TIMM_SERVICE_URL/models/list`
   - Fetch models from `$ULTRALYTICS_SERVICE_URL/models/list`
   - Fetch models from `$HUGGINGFACE_SERVICE_URL/models/list`
   - Merge results

2. **Fall back to local import** (Development mode)
   - If `model_registry` can be imported, use it directly
   - This happens when training code is available locally

3. **Fall back to static definitions** (Last resort)
   - If both above methods fail, use hardcoded `STATIC_MODELS`
   - Provides minimal functionality for degraded state

## Configuration

### Environment Variables

Training Services need `FRAMEWORK` environment variable:
```bash
# Set in Dockerfile
ENV FRAMEWORK=timm          # For timm service
ENV FRAMEWORK=ultralytics   # For ultralytics service
ENV FRAMEWORK=huggingface   # For huggingface service
```

Backend needs Training Service URLs:
```bash
# Railway environment variables
TIMM_SERVICE_URL=https://timm-service-production-xxxx.up.railway.app
ULTRALYTICS_SERVICE_URL=https://ultralytics-service-production-xxxx.up.railway.app
HUGGINGFACE_SERVICE_URL=https://huggingface-service-production-xxxx.up.railway.app
```

## Benefits

### Automatic Registration
- ✅ Models are automatically registered when added to `model_registry/*.py`
- ✅ No manual updates needed in Backend
- ✅ No hardcoded model lists
- ✅ No risk of mismatched definitions

### Framework Isolation
- ✅ Each Training Service knows exactly which models it supports
- ✅ No dependency on Backend for model definitions
- ✅ Training Services can be updated independently

### Graceful Degradation
- ✅ Works in development (local model_registry)
- ✅ Works in production (HTTP API)
- ✅ Falls back to static definitions if services unavailable

## Adding New Models

### Step 1: Update Model Registry

Add model to `mvp/training/model_registry/ultralytics_models.py`:

```python
ULTRALYTICS_MODEL_REGISTRY = {
    # ...existing models...

    "yolo11l": {  # New model
        "display_name": "YOLOv11 Large",
        "description": "High-accuracy YOLO model",
        "params": "25.3M",
        "task_types": ["object_detection"],
        "priority": 1,
        # ...more metadata...
    }
}
```

### Step 2: Deploy

Deploy the Training Service:
```bash
git add mvp/training/model_registry/ultralytics_models.py
git commit -m "feat: add YOLO11l model"
git push
```

### Step 3: Verify

The model appears automatically:
```bash
curl https://ultralytics-service.railway.app/models/list
# Returns: [..., {"model_name": "yolo11l", ...}]

curl https://backend.railway.app/api/v1/models/list?framework=ultralytics
# Returns: [..., {"model_name": "yolo11l", ...}]
```

**No Backend changes needed!** 🎉

## Troubleshooting

### Issue: Models not appearing in Backend

**Check:**
1. Training Service URLs are set in Backend environment variables
2. Training Services are deployed and running
3. Health check: `curl https://ultralytics-service.railway.app/health`
4. Model endpoint: `curl https://ultralytics-service.railway.app/models/list`

**Verify:**
```bash
# In Railway shell for Backend
echo $ULTRALYTICS_SERVICE_URL
# Should show: https://ultralytics-service-production-xxxx.up.railway.app
```

### Issue: Wrong models appearing

**Check:**
1. Training Service `FRAMEWORK` environment variable is set correctly
2. Dockerfile has: `ENV FRAMEWORK=ultralytics`
3. Restart Training Service if environment changed

**Verify:**
```bash
# In Railway shell for Training Service
echo $FRAMEWORK
# Should show: ultralytics
```

### Issue: Backend shows static models

**Cause:** Training Services are unavailable

**Fix:**
1. Check Training Service deployment status in Railway
2. Check Training Service health endpoints
3. Verify environment variables are set in Backend

**Temporary workaround:** Update `STATIC_MODELS` in `mvp/backend/app/api/models.py` as a last resort.

## References

- Implementation Plan: `docs/planning/DOCKER_IMPLEMENTATION_PLAN.md`
- Training Service API: `mvp/training/api_server.py`
- Backend Models API: `mvp/backend/app/api/models.py`
- Model Registries: `mvp/training/model_registry/*.py`
