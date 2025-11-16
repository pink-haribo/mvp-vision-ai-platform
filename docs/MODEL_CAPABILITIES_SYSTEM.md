# Model Capabilities System

**Status**: ✅ Implemented (Phase 3.6)
**Updated**: 2025-01-17

## Overview

Model Capabilities System provides a convention-based registry for supported models across all training frameworks (Ultralytics, timm, HuggingFace, etc.). This enables the backend to expose available models without breaking dependency isolation.

**Key Principle**: CLI-based trainers (not HTTP APIs) + Static JSON capabilities + GitHub Actions automation

## Design Background

### Problem

**Initial (Wrong) Design**:
- Backend tried to fetch models from Training Services via HTTP (e.g., `http://localhost:8001/models/list`)
- But Training Services are **CLI-based**, not HTTP servers!
- Resulted in connection errors and fallback to hardcoded static models

**Root Cause**: Confusion between design modes
- ❌ Training Services as HTTP APIs (old MVP approach)
- ✅ Training Services as CLI tools in K8s Jobs (correct platform design)

### Solution: Convention-Based Capabilities

Similar to ConfigSchema, use static JSON files + GitHub Actions automation.

**Architecture**:
```
platform/trainers/{framework}/capabilities.json  (Static JSON file)
    ↓ (GitHub Actions on push/PR)
R2/S3: config-schemas/model-capabilities/{framework}.json
    ↓ (Backend loads on request)
GET /api/v1/models/list
```

**Why This Works**:
1. ✅ **Dependency Isolation**: Backend doesn't need trainer code/dependencies
2. ✅ **No HTTP API Required**: Trainers remain pure CLI tools
3. ✅ **Version Control**: Capabilities tracked in Git
4. ✅ **Auto-Deploy**: GitHub Actions uploads on merge
5. ✅ **No Redeployment**: Backend loads from R2 dynamically

## File Structure

### capabilities.json Schema

Each trainer directory contains `capabilities.json`:

```
platform/trainers/
├── ultralytics/
│   ├── capabilities.json  ← Model registry
│   ├── train.py
│   └── export.py
├── timm/
│   ├── capabilities.json
│   └── train.py
└── huggingface/
    ├── capabilities.json
    └── train.py
```

### JSON Structure

```json
{
  "framework": "ultralytics",
  "display_name": "Ultralytics YOLO",
  "description": "YOLO series models for detection, segmentation, pose",
  "version": "1.0.0",
  "models": [
    {
      "model_name": "yolo11n",
      "display_name": "YOLO11 Nano",
      "task_types": ["detection"],
      "description": "Fastest YOLO11 model for real-time object detection",
      "parameters": {
        "min": 1.8,
        "macs": 6.5
      },
      "supported": true
    }
  ],
  "task_types": [
    {
      "name": "detection",
      "display_name": "Object Detection",
      "description": "Detect and localize objects with bounding boxes",
      "supported": true
    }
  ],
  "dataset_formats": [
    {
      "name": "yolo",
      "display_name": "YOLO Format",
      "description": "Native YOLO format with images/ and labels/ directories",
      "supported": true
    }
  ]
}
```

**Required Fields**:
- `framework`: Unique identifier (e.g., "ultralytics", "timm")
- `display_name`: User-friendly name
- `description`: Framework description
- `version`: Capability schema version (semver)
- `models`: Array of model definitions
- `task_types`: Array of supported task types
- `dataset_formats`: Array of supported dataset formats

**Model Object**:
- `model_name`: Unique identifier (used in training commands)
- `display_name`: User-friendly name
- `task_types`: Array of supported tasks
- `description`: Model description
- `supported`: Boolean indicating availability
- `parameters`: Optional model-specific parameters (size, MACs, etc.)

## GitHub Actions Automation

### Workflow: upload-model-capabilities.yml

**Trigger Conditions**:
- Push to `main`, `production`, `mvp-to-platform-migration`
- PR that modifies:
  - `platform/trainers/*/capabilities.json`
  - `platform/scripts/upload_model_capabilities.py`
- Manual trigger via `workflow_dispatch`

**PR Workflow (Validation)**:
```bash
python platform/scripts/upload_model_capabilities.py --all --dry-run
```
- ✅ Validates JSON syntax
- ✅ Checks required fields
- ✅ Posts validation result as PR comment
- ❌ Does NOT upload to R2

**Main/Production Workflow (Upload)**:
```bash
python platform/scripts/upload_model_capabilities.py --all
```
- ✅ Validates all capabilities
- ✅ Uploads to Cloudflare R2: `config-schemas/model-capabilities/{framework}.json`
- ✅ Generates upload summary

**Environment Variables** (GitHub Secrets):
- `R2_ENDPOINT_URL`: Cloudflare R2 endpoint
- `R2_ACCESS_KEY_ID`: R2 access key
- `R2_SECRET_ACCESS_KEY`: R2 secret key
- `INTERNAL_BUCKET_SCHEMAS`: Bucket name (default: `config-schemas`)

### Local Development

**Validate capabilities**:
```bash
cd platform
python scripts/upload_model_capabilities.py --all --dry-run
```

**Upload to local MinIO** (requires environment variables):
```bash
# Set environment variables
export INTERNAL_STORAGE_ENDPOINT=http://localhost:9002
export INTERNAL_STORAGE_ACCESS_KEY=minioadmin
export INTERNAL_STORAGE_SECRET_KEY=minioadmin
export INTERNAL_BUCKET_SCHEMAS=config-schemas

# Upload
python scripts/upload_model_capabilities.py --all
```

**Upload single framework**:
```bash
python scripts/upload_model_capabilities.py --framework ultralytics
```

## Backend Implementation

### Loading Capabilities

**File**: `platform/backend/app/api/models.py`

```python
def load_framework_capabilities(framework: str) -> Optional[Dict[str, Any]]:
    """Load model capabilities from R2/S3."""
    from app.utils.dual_storage import dual_storage

    # Load from R2: config-schemas/model-capabilities/{framework}.json
    capabilities_bytes = dual_storage.get_capabilities(framework)

    if not capabilities_bytes:
        return None

    return json.loads(capabilities_bytes.decode('utf-8'))
```

**DualStorageClient Method**:
```python
# platform/backend/app/utils/dual_storage.py

def get_capabilities(self, framework: str) -> Optional[bytes]:
    """Get model capabilities from internal storage."""
    capabilities_key = f"model-capabilities/{framework}.json"
    response = self.internal_client.get_object(
        Bucket=self.internal_bucket_schemas,  # config-schemas
        Key=capabilities_key
    )
    return response['Body'].read()
```

### API Endpoints

**List all models**:
```http
GET /api/v1/models/list
GET /api/v1/models/list?framework=ultralytics
GET /api/v1/models/list?task_type=detection
GET /api/v1/models/list?supported_only=false
```

**Get specific model**:
```http
GET /api/v1/models/get?framework=ultralytics&model_name=yolo11n
GET /api/v1/models/{framework}/{model_name}
```

**Get framework capabilities**:
```http
GET /api/v1/models/capabilities/ultralytics
```

### Error Handling

**No Fallback** - Clear error messages instead:

**404 - Framework not found**:
```json
{
  "detail": "Model capabilities for framework 'ultralytics' not found. Available frameworks: ultralytics, timm, huggingface. Capabilities are uploaded via GitHub Actions from platform/trainers/*/capabilities.json"
}
```

**503 - No capabilities available**:
```json
{
  "detail": "No model capabilities available. Model capabilities must be uploaded via GitHub Actions from platform/trainers/*/capabilities.json. Check that workflows/.github/workflows/upload-model-capabilities.yml has run successfully."
}
```

**Why No Fallback**:
- ✅ Makes problems visible immediately
- ✅ Forces proper configuration
- ✅ Prevents silent failures
- ✅ Easier debugging

## Adding New Trainers

### Step 1: Create capabilities.json

```bash
cd platform/trainers/{new_framework}
vim capabilities.json
```

**Template**:
```json
{
  "framework": "new_framework",
  "display_name": "New Framework",
  "description": "Framework description",
  "version": "1.0.0",
  "models": [
    {
      "model_name": "model_v1",
      "display_name": "Model V1",
      "task_types": ["classification"],
      "description": "Model description",
      "supported": true
    }
  ],
  "task_types": [
    {
      "name": "classification",
      "display_name": "Image Classification",
      "description": "Classify images into categories",
      "supported": true
    }
  ],
  "dataset_formats": [
    {
      "name": "imagefolder",
      "display_name": "ImageFolder",
      "description": "PyTorch ImageFolder structure",
      "supported": true
    }
  ]
}
```

### Step 2: Validate Locally

```bash
python platform/scripts/upload_model_capabilities.py --framework new_framework --dry-run
```

### Step 3: Commit and Push

```bash
git add platform/trainers/new_framework/capabilities.json
git commit -m "feat(trainers): add capabilities for new_framework"
git push origin feature/new-framework
```

### Step 4: Create PR

GitHub Actions will:
1. Validate capabilities.json
2. Post validation result as comment
3. On merge: Auto-upload to R2

### Step 5: Verify in Production

```bash
# Check API
curl https://api.yourplatform.com/api/v1/models/list?framework=new_framework

# Check R2
aws s3 ls s3://config-schemas/model-capabilities/ --endpoint-url=$R2_ENDPOINT
```

## Current Capabilities

### Ultralytics (22 models)

**Detection** (5 models): yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
**Segmentation** (9 models): yolo11n-seg, yolo11s-seg, ..., sam2_t, sam2_s, sam2_b, sam2_l
**Pose** (5 models): yolo11n-pose, yolo11s-pose, yolo11m-pose, yolo11l-pose, yolo11x-pose
**Open-Vocabulary Detection** (3 models): yolo_world_v2_s, yolo_world_v2_m, yolo_world_v2_l

**Task Types**: detection, segmentation, pose, open_vocabulary_detection
**Dataset Formats**: yolo, coco, dice

### timm (Planned)

TBD - Add when implementing timm trainer

### HuggingFace (Planned)

TBD - Add when implementing HuggingFace trainer

## Comparison: ConfigSchema vs Model Capabilities

| Aspect | ConfigSchema | Model Capabilities |
|--------|--------------|-------------------|
| **Purpose** | Training parameters | Available models |
| **Source** | `config_schema.py` (Python) | `capabilities.json` (JSON) |
| **Generation** | Python function call | Static file |
| **Upload Path** | `{framework}.json` | `model-capabilities/{framework}.json` |
| **Backend Loader** | `dual_storage.get_schema()` | `dual_storage.get_capabilities()` |
| **API Endpoint** | `/training/config-schema` | `/models/list` |
| **Update Frequency** | On parameter changes | On new model support |
| **GitHub Workflow** | `upload-config-schemas.yml` | `upload-model-capabilities.yml` |

**Shared Infrastructure**:
- Both stored in `config-schemas` bucket
- Both use DualStorageClient (Internal Storage)
- Both auto-uploaded via GitHub Actions
- Both support dry-run validation

## Troubleshooting

### "No model capabilities available" Error

**Symptom**: Frontend shows "No models available" or similar error.

**Cause**: capabilities.json not uploaded to R2.

**Solution**:
1. Check GitHub Actions workflow ran successfully
2. Verify R2 bucket contains files:
   ```bash
   aws s3 ls s3://config-schemas/model-capabilities/ --endpoint-url=$R2_ENDPOINT
   ```
3. Manual upload if needed:
   ```bash
   python platform/scripts/upload_model_capabilities.py --all
   ```

### Validation Fails in PR

**Symptom**: GitHub Actions posts "Validation Failed" comment.

**Common Issues**:
- ❌ Missing required field (`framework`, `models`, `task_types`, etc.)
- ❌ Invalid JSON syntax
- ❌ Model missing required field (`model_name`, `display_name`, `task_types`, `description`, `supported`)

**Solution**:
1. Check workflow logs for specific error
2. Fix capabilities.json
3. Run local validation:
   ```bash
   python platform/scripts/upload_model_capabilities.py --framework {name} --dry-run
   ```
4. Push fix

### Backend Returns 404 for Specific Framework

**Symptom**: `/models/list?framework=timm` returns 404.

**Cause**: capabilities.json for timm not uploaded.

**Solution**:
1. Create `platform/trainers/timm/capabilities.json`
2. Follow "Adding New Trainers" steps above

### Models Not Showing in Frontend

**Symptom**: TrainingConfigPanel model selector is empty.

**Possible Causes**:
1. Backend not loading capabilities (check logs)
2. Frontend API call failing (check network tab)
3. Filter excluding all models (check `supported_only`, `task_type` params)

**Debug Steps**:
```bash
# 1. Check backend can load capabilities
curl http://localhost:8000/api/v1/models/list

# 2. Check specific framework
curl http://localhost:8000/api/v1/models/capabilities/ultralytics

# 3. Check backend logs
grep "models" platform/backend/logs/app.log
```

## Migration Notes

### From Old Static Models

**Before (Hardcoded)**:
```python
STATIC_MODELS = [
    {"framework": "ultralytics", "model_name": "yolo11n", ...},
    {"framework": "ultralytics", "model_name": "yolo11s", ...},
]
```

**After (Dynamic from R2)**:
```python
capabilities = dual_storage.get_capabilities("ultralytics")
models = capabilities["models"]  # Loaded from R2
```

**Migration Checklist**:
- [x] Create `platform/trainers/ultralytics/capabilities.json`
- [x] Remove `STATIC_MODELS` from `app/api/models.py`
- [x] Remove Training Services HTTP API calls
- [x] Add `dual_storage.get_capabilities()` method
- [x] Update `/models/list` endpoint to use capabilities
- [x] Create GitHub Actions workflow
- [x] Update documentation

### Breaking Changes

**API Response Schema Changed**:

**Old**:
```json
{
  "framework": "ultralytics",
  "model_name": "yolo11n",
  "params": "2.6M",           // ← Removed
  "input_size": 640,           // ← Removed
  "pretrained_available": true, // ← Removed
  "recommended_batch_size": 16, // ← Removed
  "recommended_lr": 0.01,      // ← Removed
  "tags": ["fast"],            // ← Removed
  "priority": 1                // ← Removed
}
```

**New**:
```json
{
  "framework": "ultralytics",
  "model_name": "yolo11n",
  "display_name": "YOLO11 Nano",
  "task_types": ["detection"],
  "description": "Fastest YOLO11 model...",
  "supported": true,
  "parameters": {              // ← New: Framework-specific
    "min": 1.8,
    "macs": 6.5
  }
}
```

**Impact**: Frontend components using old schema fields need updates.

## Future Enhancements

### Planned Features

1. **Model Benchmarks**: Add performance metrics to capabilities
   ```json
   "benchmarks": {
     "coco_map50": 0.534,
     "inference_time_ms": 12.3
   }
   ```

2. **Hardware Requirements**: Specify minimum VRAM, compute
   ```json
   "requirements": {
     "min_vram_gb": 4,
     "supports_cpu": true,
     "supports_mps": false
   }
   ```

3. **Dynamic Capability Discovery**: Auto-generate from trainer code
   ```bash
   python train.py --list-capabilities > capabilities.json
   ```

4. **Capability Versioning**: Track schema changes
   ```json
   "schema_version": "2.0.0",
   "deprecated": ["old_field"]
   ```

## References

- **Related Docs**:
  - [CONFIG_SCHEMA_SYSTEM.md](./CONFIG_SCHEMA_SYSTEM.md) - Configuration schema system
  - [EXPORT_CONVENTION.md](./EXPORT_CONVENTION.md) - Export convention (similar pattern)
  - [DUAL_STORAGE.md](./DUAL_STORAGE.md) - Dual storage architecture

- **Implementation Files**:
  - `platform/trainers/ultralytics/capabilities.json`
  - `platform/scripts/upload_model_capabilities.py`
  - `platform/backend/app/api/models.py`
  - `platform/backend/app/utils/dual_storage.py`
  - `.github/workflows/upload-model-capabilities.yml`

- **GitHub Actions**:
  - [Upload Config Schemas Workflow](../.github/workflows/upload-config-schemas.yml)
  - [Upload Model Capabilities Workflow](../.github/workflows/upload-model-capabilities.yml)
