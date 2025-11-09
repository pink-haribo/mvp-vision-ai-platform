# Training Configuration Schema Distribution

**Date**: 2025-11-08
**Status**: Implemented
**Architecture Decision**: Storage-based schema distribution with hybrid automation

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Design Principles](#design-principles)
- [Implementation](#implementation)
- [Automation Strategy](#automation-strategy)
- [Roles and Responsibilities](#roles-and-responsibilities)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Problem Statement

Before Kubernetes Job migration, Backend communicated with Training Services via HTTP to fetch configuration schemas. After migrating to K8s Job-based training:

**Challenge**: How does Backend get framework-specific configuration schemas without creating runtime dependencies on Training code?

**Critical Requirements**:
1. Complete dependency isolation between Backend and Training
2. Support for plugin architecture (model developers can add frameworks without Backend changes)
3. Prepare for separate GPU cluster deployment
4. Dynamic schema updates without redeployment

### Solution: Storage-based Schema Distribution

Training frameworks upload their configuration schemas to shared object storage (MinIO/S3). Backend reads schemas from storage at runtime.

```
Backend (CPU Cluster)          Training (GPU Cluster)
      │                              │
      └──► MinIO/S3 Storage ◄────────┘
           (schemas/)

   ✅ Complete dependency isolation
   ✅ Plugin architecture ready
   ✅ Separate cluster deployment supported
```

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Schema Definition (Training Code)                            │
│                                                                  │
│   mvp/training/config_schemas.py:                                │
│   def get_ultralytics_schema():                                  │
│       return ConfigSchema(                                       │
│           fields=[...],  # 24 fields                             │
│           presets={...}  # easy, medium, advanced                │
│       )                                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Schema Upload (Automated)                                    │
│                                                                  │
│   Local: dev-start.ps1 → upload_schema_to_storage.py            │
│   Production: GitHub Actions → CI/CD workflow                   │
│                                                                  │
│   Output: schemas/ultralytics.json (MinIO/S3)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Schema Consumption (Backend API)                             │
│                                                                  │
│   GET /api/v1/training/config-schema?framework=ultralytics      │
│   → storage.get_file_content("schemas/ultralytics.json")        │
│   → Parse JSON                                                   │
│   → Return to Frontend                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Dynamic UI Rendering (Frontend)                              │
│                                                                  │
│   DynamicConfigPanel:                                            │
│   - Group fields by category (optimizer, scheduler, augment)    │
│   - Render appropriate input type (int, float, bool, select)    │
│   - Apply presets (easy/medium/advanced)                        │
│   - Show/hide advanced options                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Storage Structure

```
training-results/
└── schemas/
    ├── ultralytics.json     # 7.8 KB, 24 fields, 3 presets
    ├── timm.json            # 8.3 KB, 25 fields, 3 presets
    ├── huggingface.json     # (future)
    └── custom-framework.json # (extensible)
```

### Schema Format

```json
{
  "framework": "ultralytics",
  "description": "Ultralytics Training Configuration",
  "version": "1.0",
  "fields": [
    {
      "name": "optimizer_type",
      "type": "select",
      "default": "Adam",
      "description": "Optimizer algorithm",
      "required": false,
      "options": ["Adam", "AdamW", "SGD", "RMSprop"],
      "min": null,
      "max": null,
      "step": null,
      "group": "optimizer",
      "advanced": false
    },
    {
      "name": "weight_decay",
      "type": "float",
      "default": 0.0005,
      "description": "Weight decay (L2 regularization)",
      "required": false,
      "options": null,
      "min": 0.0,
      "max": 0.01,
      "step": 0.0001,
      "group": "optimizer",
      "advanced": true
    }
    // ... 22 more fields
  ],
  "presets": {
    "easy": {
      "mosaic": 1.0,
      "fliplr": 0.5,
      "amp": true
    },
    "medium": {
      "mosaic": 1.0,
      "mixup": 0.1,
      "fliplr": 0.5,
      "degrees": 10
    },
    "advanced": {
      "mosaic": 1.0,
      "mixup": 0.15,
      "copy_paste": 0.1,
      "shear": 5.0
    }
  }
}
```

---

## Design Principles

### 1. No Shortcuts, Proper Implementation

**Forbidden**:
```python
# ❌ Hardcoded data (temporary workaround)
STATIC_MODELS = [
    {"model_name": "yolo11n", "framework": "ultralytics"},
]

# ❌ Copying Training code to Backend
from mvp.training.config_schemas import get_ultralytics_schema
```

**Correct**:
```python
# ✅ Dynamic loading from storage
schema_bytes = storage.get_file_content("schemas/ultralytics.json")
schema = json.loads(schema_bytes)
```

### 2. Dependency Isolation

**Build-time**: Backend and Training use separate Docker images, no shared code dependencies.

**Runtime**: Backend never imports or calls Training code directly. Communication only through storage.

```
Backend Dependencies          Training Dependencies
- FastAPI                     - PyTorch
- SQLAlchemy                  - Ultralytics
- Boto3 (S3 client)          - timm
                              - Boto3 (S3 client)

Shared Interface: MinIO/S3 Storage
```

### 3. Plugin Architecture

Model developers can add new frameworks without Backend code changes:

```bash
# 1. Define schema in Training code
# mvp/training/config_schemas.py
def get_new_framework_schema():
    return ConfigSchema(...)

# 2. Upload schema to storage
python scripts/upload_schema_to_storage.py --framework new_framework

# 3. Backend automatically supports new framework
# (no code changes, no redeployment needed)
```

### 4. Environment Parity

Same source code works in both local and production:

| Environment | Storage Endpoint | Bucket | Auth |
|-------------|-----------------|--------|------|
| **Local** | `http://localhost:30900` | `training-results` | MinIO admin |
| **Production** | `https://pub-xxxxx.r2.dev` | `training-results` | R2 credentials |

Only difference: environment variables in `.env` vs Railway Variables.

---

## Implementation

### Backend API

**File**: `mvp/backend/app/api/training.py`

```python
@router.get("/config-schema")
async def get_config_schema(framework: str, task_type: str = None):
    """
    Get configuration schema for a framework.

    Loads schema from storage (uploaded by Training services).
    """
    logger.info(f"[config-schema] Requested framework={framework}")

    try:
        storage = get_storage_client()
        schema_key = f"schemas/{framework}.json"

        logger.info(f"[config-schema] Loading schema from storage: {schema_key}")

        # Get schema from storage
        schema_bytes = storage.get_file_content(
            schema_key,
            bucket=storage.bucket_results  # schemas stored in results bucket
        )

        if not schema_bytes:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration schema for framework '{framework}' not found. "
                       f"Please run: mvp/training/scripts/upload_schema_to_storage.py --framework {framework}"
            )

        # Parse JSON
        schema_dict = json.loads(schema_bytes.decode('utf-8'))

        logger.info(f"[config-schema] Schema loaded: {len(schema_dict.get('fields', []))} fields, "
                   f"{len(schema_dict.get('presets', {}))} presets")

        return schema_dict

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"[config-schema] Invalid JSON in schema file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schema file is corrupted. Please re-upload.")
    except Exception as e:
        logger.error(f"[config-schema] Error loading schema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load configuration schema: {str(e)}")
```

### Schema Upload Script

**File**: `mvp/training/scripts/upload_schema_to_storage.py`

```python
#!/usr/bin/env python3
"""
Upload training configuration schemas to storage.

Usage:
    python upload_schema_to_storage.py --all
    python upload_schema_to_storage.py --framework ultralytics
    python upload_schema_to_storage.py --dry-run
"""

def extract_schema(framework: str) -> dict:
    """Extract schema from framework."""
    if framework == 'ultralytics':
        schema = get_ultralytics_schema()
    elif framework == 'timm':
        schema = get_timm_schema()
    else:
        raise ValueError(f"Unknown framework: {framework}")

    schema_dict = {
        'framework': framework,
        'description': f"{framework.title()} Training Configuration",
        'version': '1.0',
        'fields': [field.to_dict() for field in schema.fields],
        'presets': schema.presets
    }

    return schema_dict

def upload_schema(s3_client, framework: str, schema_dict: dict, bucket: str):
    """Upload schema to storage."""
    schema_key = f"schemas/{framework}.json"
    schema_json = json.dumps(schema_dict, indent=2)

    s3_client.put_object(
        Bucket=bucket,
        Key=schema_key,
        Body=schema_json.encode('utf-8'),
        ContentType='application/json'
    )
```

### Frontend Dynamic UI

**File**: `mvp/frontend/components/training/DynamicConfigPanel.tsx`

```typescript
export default function DynamicConfigPanel({ framework, taskType, config, onChange }) {
  const [schema, setSchema] = useState<ConfigSchema | null>(null)

  // Fetch schema from Backend API
  useEffect(() => {
    const fetchSchema = async () => {
      const params = new URLSearchParams({ framework })
      if (taskType) params.append('task_type', taskType)

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/training/config-schema?${params}`
      )

      const data = await response.json()
      // Backend returns { framework, fields, presets } directly
      setSchema({
        fields: data.fields || [],
        presets: data.presets || {}
      })
    }

    fetchSchema()
  }, [framework, taskType])

  // Group fields by category
  const groupedFields = schema.fields.reduce((acc, field) => {
    const group = field.group || 'general'
    if (!acc[group]) acc[group] = []
    acc[group].push(field)
    return acc
  }, {})

  // Render field based on type
  const renderField = (field: ConfigField) => {
    switch (field.type) {
      case 'bool':
        return <Checkbox {...field} />
      case 'int':
      case 'float':
        return <NumberInput {...field} />
      case 'select':
        return <Select options={field.options} {...field} />
      default:
        return <TextInput {...field} />
    }
  }

  return (
    <div>
      {Object.entries(groupedFields).map(([group, fields]) => (
        <Accordion key={group} title={group}>
          {fields.map(field => renderField(field))}
        </Accordion>
      ))}
    </div>
  )
}
```

---

## Automation Strategy

### Hybrid Approach

**Local Development**: Automated via `dev-start.ps1`
**Production**: Automated via CI/CD (GitHub Actions)

### Local Development (dev-start.ps1)

**Setup** (Platform Operator - 1 time):

```powershell
# Add to dev-start.ps1 (after MinIO is ready)

Write-Host "Uploading training configuration schemas..." -ForegroundColor Yellow

# Run upload script from Backend container (has Python + boto3)
kubectl exec -n backend deployment/backend -- bash -c "
    export AWS_S3_ENDPOINT_URL=http://minio.storage.svc.cluster.local:9000
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin
    export S3_BUCKET_RESULTS=training-results

    cd /tmp && \
    curl -O https://raw.githubusercontent.com/your-org/repo/main/mvp/training/scripts/upload_schema_to_storage.py && \
    curl -O https://raw.githubusercontent.com/your-org/repo/main/mvp/training/config_schemas.py && \
    python upload_schema_to_storage.py --all
"

Write-Host "✓ Schemas uploaded to storage" -ForegroundColor Green
```

**Alternative** (simpler for now):

```powershell
# Manual upload after dev-start.ps1
Write-Host "⚠ Run schema upload manually:" -ForegroundColor Yellow
Write-Host "  cd mvp/training" -ForegroundColor Cyan
Write-Host "  python scripts/upload_schema_to_storage.py --all" -ForegroundColor Cyan
```

**Model Developer Workflow**:

```bash
# 1. Edit schema
vim mvp/training/config_schemas.py

# 2. Upload to local MinIO
cd mvp/training
export AWS_S3_ENDPOINT_URL=http://localhost:30900
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
python scripts/upload_schema_to_storage.py --framework ultralytics

# 3. Test in Frontend immediately
# Open http://localhost:3000 → Create Training → Advanced Config
```

### Production (GitHub Actions)

**Setup** (Platform Operator - 1 time):

```yaml
# .github/workflows/upload-schemas.yml
name: Upload Training Schemas

on:
  push:
    branches: [main, production]
    paths:
      - 'mvp/training/config_schemas.py'
      - 'mvp/training/scripts/upload_schema_to_storage.py'
  pull_request:
    paths:
      - 'mvp/training/config_schemas.py'

jobs:
  validate-and-upload:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          cd mvp/training
          pip install boto3 pydantic

      # PR: Validate only (dry-run)
      - name: Validate schemas (PR)
        if: github.event_name == 'pull_request'
        run: |
          cd mvp/training
          python scripts/upload_schema_to_storage.py --all --dry-run

      # Main/Production: Upload to R2
      - name: Upload schemas to Cloudflare R2
        if: github.event_name == 'push'
        env:
          AWS_S3_ENDPOINT_URL: ${{ secrets.R2_ENDPOINT_URL }}
          AWS_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
          S3_BUCKET_RESULTS: ${{ secrets.S3_BUCKET_RESULTS }}
        run: |
          cd mvp/training
          python scripts/upload_schema_to_storage.py --all

      - name: Summary
        run: |
          echo "### Schema Upload Summary ✅" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "Uploaded schemas:" >> $GITHUB_STEP_SUMMARY
          echo "- ultralytics.json" >> $GITHUB_STEP_SUMMARY
          echo "- timm.json" >> $GITHUB_STEP_SUMMARY
```

**GitHub Secrets Configuration**:

```
Repository → Settings → Secrets and variables → Actions

Required secrets:
  R2_ENDPOINT_URL:         https://xxxxx.r2.cloudflarestorage.com
  R2_ACCESS_KEY_ID:        your-r2-access-key
  R2_SECRET_ACCESS_KEY:    your-r2-secret-key
  S3_BUCKET_RESULTS:       training-results
```

**Model Developer Workflow**:

```bash
# 1. Create branch
git checkout -b feat/add-resnet101-schema

# 2. Edit schema
vim mvp/training/config_schemas.py

# 3. Test locally
python scripts/upload_schema_to_storage.py --dry-run

# 4. Commit and push
git add mvp/training/config_schemas.py
git commit -m "feat(training): add ResNet-101 config schema"
git push origin feat/add-resnet101-schema

# 5. Create PR
# GitHub Actions will:
#   - Validate schema (--dry-run)
#   - Run tests
#   - Comment on PR with validation results

# 6. After PR merge to main
# GitHub Actions will:
#   - Upload schemas to R2
#   - Schemas immediately available in production
#   - No Backend redeployment needed!
```

---

## Roles and Responsibilities

### Platform Operator

**Initial Setup (1 time)**:
1. Create storage buckets (`training-results`, `training-datasets`, `training-checkpoints`)
2. Configure GitHub Secrets for R2 credentials
3. Add schema upload step to `dev-start.ps1` (optional)
4. Set up CI/CD workflow (`.github/workflows/upload-schemas.yml`)

**Ongoing**:
- Review schema-related PRs (field types, defaults, validation)
- Monitor GitHub Actions for upload failures
- No manual intervention needed (fully automated)

### Model/Training Developer

**Adding New Framework**:

```python
# 1. Define schema in config_schemas.py
def get_detectron2_schema() -> ConfigSchema:
    return ConfigSchema(
        fields=[
            ConfigField(
                name='solver_type',
                type='select',
                default='SGD',
                description='Optimizer type',
                options=['SGD', 'Adam', 'AdamW'],
                group='optimizer',
                advanced=False
            ),
            # ... more fields
        ],
        presets={
            'easy': {...},
            'medium': {...},
            'advanced': {...}
        }
    )

# 2. Register in upload script
# mvp/training/scripts/upload_schema_to_storage.py
if framework == 'detectron2':
    schema = get_detectron2_schema()
```

**Local Testing**:

```bash
cd mvp/training
python scripts/upload_schema_to_storage.py --framework detectron2 --dry-run
python scripts/upload_schema_to_storage.py --framework detectron2
```

**Production Deployment**:

```bash
git checkout -b feat/add-detectron2-schema
git add mvp/training/config_schemas.py
git add mvp/training/scripts/upload_schema_to_storage.py
git commit -m "feat(training): add Detectron2 config schema"
git push origin feat/add-detectron2-schema
# Create PR → Auto-upload on merge
```

**No Backend Changes Needed**: Frontend automatically renders new schema!

---

## Usage Guide

### For Platform Operators

**Check Uploaded Schemas**:

```bash
# Local (MinIO)
kubectl exec -n storage deployment/minio -- mc ls local/training-results/schemas/

# Production (R2)
aws s3 ls s3://training-results/schemas/ \
  --endpoint-url https://xxxxx.r2.cloudflarestorage.com
```

**Manual Upload** (emergency):

```bash
cd mvp/training
export AWS_S3_ENDPOINT_URL=https://xxxxx.r2.cloudflarestorage.com
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
python scripts/upload_schema_to_storage.py --all
```

**Verify Schema**:

```bash
# Download and inspect
kubectl exec -n storage deployment/minio -- \
  mc cat local/training-results/schemas/ultralytics.json | jq .

# Check field count
kubectl exec -n storage deployment/minio -- \
  mc cat local/training-results/schemas/ultralytics.json | jq '.fields | length'
# Output: 24

# Check presets
kubectl exec -n storage deployment/minio -- \
  mc cat local/training-results/schemas/ultralytics.json | jq '.presets | keys'
# Output: ["easy", "medium", "advanced"]
```

### For Model Developers

**Testing Schema Locally**:

```bash
# 1. Dry run (no upload)
python scripts/upload_schema_to_storage.py --framework ultralytics --dry-run

# 2. Upload to local MinIO
export AWS_S3_ENDPOINT_URL=http://localhost:30900
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
python scripts/upload_schema_to_storage.py --framework ultralytics

# 3. Test in Frontend
# Open http://localhost:3000
# Project → Create Training → Advanced Config
# Should see new fields immediately
```

**Schema Validation Checklist**:

- [ ] All fields have valid types (`int`, `float`, `bool`, `select`, `str`)
- [ ] Required fields are marked correctly
- [ ] Min/max ranges are sensible
- [ ] Default values are within ranges
- [ ] Group names are consistent (`optimizer`, `scheduler`, `augmentation`, etc.)
- [ ] Advanced fields are marked with `advanced=True`
- [ ] Presets contain valid field names and values
- [ ] Description is clear and concise

**Example Schema Definition**:

```python
ConfigField(
    name='learning_rate',
    type='float',
    default=0.001,
    description='Initial learning rate',
    required=False,
    min=0.0,
    max=1.0,
    step=0.0001,
    group='optimizer',
    advanced=False
)
```

---

## Troubleshooting

### Frontend Shows Empty Advanced Config

**Symptom**: Click "Advanced Config" button, modal opens but shows nothing.

**Diagnosis**:

```bash
# 1. Check Backend logs
kubectl logs -n backend deployment/backend --tail=50 | grep config-schema

# 2. Check if schema exists in storage
kubectl exec -n storage deployment/minio -- \
  mc ls local/training-results/schemas/

# 3. Test API directly
curl "http://localhost:8000/api/v1/training/config-schema?framework=ultralytics" | jq .
```

**Solutions**:

```bash
# If schema missing in storage:
cd mvp/training
python scripts/upload_schema_to_storage.py --framework ultralytics

# If schema exists but API returns error:
# Check Backend logs for details
kubectl logs -n backend deployment/backend --tail=100

# If Frontend parsing error:
# Check browser console (F12)
# Look for JSON parsing errors or network failures
```

### Schema Upload Fails (Connection Refused)

**Symptom**: `ConnectionError: [Errno 111] Connection refused`

**Diagnosis**:

```bash
# 1. Check if MinIO is running
kubectl get pods -n storage

# 2. Check MinIO service
kubectl get svc -n storage

# 3. Test MinIO connectivity
kubectl exec -n storage deployment/minio -- mc admin info local
```

**Solutions**:

```bash
# If MinIO not running:
kubectl rollout restart deployment/minio -n storage

# If credentials wrong:
# Check .env file
cat mvp/training/.env | grep AWS

# If endpoint wrong:
# Update AWS_S3_ENDPOINT_URL
export AWS_S3_ENDPOINT_URL=http://localhost:30900  # Local
export AWS_S3_ENDPOINT_URL=http://minio.storage.svc.cluster.local:9000  # In-cluster
```

### GitHub Actions Upload Fails

**Symptom**: CI/CD workflow fails at "Upload schemas to Cloudflare R2" step.

**Diagnosis**:

```bash
# Check GitHub Actions logs:
# Repo → Actions → Upload Training Schemas → [failed run]

# Common errors:
# - "AWS_S3_ENDPOINT_URL not set" → Secret not configured
# - "Access Denied" → Wrong credentials
# - "Bucket not found" → Bucket doesn't exist in R2
```

**Solutions**:

```bash
# 1. Verify GitHub Secrets are set
# Repository → Settings → Secrets → Check all 4 secrets exist

# 2. Test credentials locally
cd mvp/training
export AWS_S3_ENDPOINT_URL=$PROD_R2_ENDPOINT
export AWS_ACCESS_KEY_ID=$PROD_R2_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$PROD_R2_SECRET_KEY
python scripts/upload_schema_to_storage.py --framework ultralytics

# 3. Create bucket if missing
# Cloudflare Dashboard → R2 → Create Bucket: "training-results"

# 4. Re-run failed workflow
# GitHub Actions → [failed run] → Re-run all jobs
```

### Schema Validation Errors

**Symptom**: `ValueError: Invalid field type: xyz`

**Diagnosis**:

```python
# Check schema definition
# mvp/training/config_schemas.py

# Valid types:
VALID_TYPES = ['int', 'float', 'bool', 'select', 'str', 'multiselect']
```

**Solutions**:

```python
# Fix invalid type
ConfigField(
    name='optimizer_type',
    type='xyz',  # ❌ Invalid
    type='select',  # ✅ Correct
    options=['Adam', 'SGD']
)

# Test validation
python scripts/upload_schema_to_storage.py --framework ultralytics --dry-run
```

---

## References

### Related Documentation

- [K8s Training FAQ](./K8S_TRAINING_FAQ.md) - Kubernetes Job architecture
- [Development Workflow Setup](./20251107_development_workflow_setup.md) - Local development setup
- [Backend Storage Utils](../../mvp/backend/app/utils/s3_storage.py) - S3 client implementation
- [Frontend Dynamic Config](../../mvp/frontend/components/training/DynamicConfigPanel.tsx) - UI rendering

### External Links

- [Cloudflare R2 Documentation](https://developers.cloudflare.com/r2/)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [MinIO mc CLI](https://min.io/docs/minio/linux/reference/minio-mc.html)

### Commit History

- `feat(training): implement storage-based config schema distribution` (2025-11-08)
- `refactor(storage): use separate MinIO buckets for datasets/checkpoints/results` (2025-11-08)
- `fix(frontend): parse config schema response correctly` (2025-11-08)

---

## Appendix: Architecture Benefits

### Achieved Goals

✅ **Complete Dependency Isolation**
- Backend never imports Training code
- Training never imports Backend code
- Communication only via storage interface

✅ **Plugin Architecture**
- Model developers add frameworks independently
- No Backend code changes required
- No Frontend code changes required (dynamic UI)

✅ **Separate Cluster Support**
- Backend runs in CPU-only cluster
- Training runs in GPU cluster (future)
- Only storage connection needed

✅ **Hot Reload Capability**
- Upload new schema → Available immediately
- No Backend restart needed
- No Frontend rebuild needed

✅ **Environment Parity**
- Same code in local and production
- Only environment variables differ
- Reduces deployment issues

### Future Enhancements

**Schema Versioning**:
```
schemas/
├── ultralytics/
│   ├── v1.0.json
│   ├── v1.1.json
│   └── latest.json (symlink)
```

**Schema Registry Service**:
```python
# Dedicated microservice for schema management
GET /schemas?framework=ultralytics&version=1.1
POST /schemas/validate  # Validate before upload
GET /schemas/diff?from=1.0&to=1.1  # Show changes
```

**Schema Evolution Support**:
```json
{
  "framework": "ultralytics",
  "version": "2.0",
  "migrations": {
    "from_1.0": {
      "renamed_fields": {"lr": "learning_rate"},
      "removed_fields": ["old_option"],
      "added_fields": ["new_option"]
    }
  }
}
```

**Multi-tenancy**:
```
schemas/
├── public/
│   ├── ultralytics.json
│   └── timm.json
└── tenants/
    └── company-xyz/
        └── custom-yolo.json
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-08
**Contributors**: Platform Team, Training Team
