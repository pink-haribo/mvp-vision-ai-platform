# Advanced Configuration Schema System

## Overview

The Vision AI Training Platform uses a **Distributed Schema Pattern** for managing framework-specific training configurations. This architecture enables:

- **Zero-downtime schema updates**: Upload new schemas → Frontend UI updates automatically
- **Plugin-friendly development**: New trainers just add `config_schema.py`
- **Version-controlled schemas**: Schemas stored in Git, uploaded to S3/R2
- **Dynamic UI generation**: Frontend renders forms from schema definitions
- **Framework independence**: Each trainer owns its configuration structure

## Architecture

### Design Principles

1. **Schema Ownership**: Each trainer defines its own configuration schema
2. **Distributed Storage**: Schemas stored in object storage (S3/R2), not embedded in Backend
3. **Auto-Discovery**: Scripts automatically find all trainers with `config_schema.py`
4. **API-Driven**: Backend serves schemas via REST API
5. **Frontend Agnostic**: Schema format works with any UI framework

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Schema Lifecycle                          │
└─────────────────────────────────────────────────────────────┘

1. Developer adds config_schema.py to trainer
      ↓
2. GitHub Actions validates schema on PR
      ↓
3. On merge to main, script uploads to S3/R2
      ↓
4. Backend fetches schema from S3: GET /config-schema?framework=X
      ↓
5. Frontend renders dynamic form from schema
      ↓
6. User submits config → Backend → Trainer receives JSON
```

### Storage Architecture

**Tier-0 (Local Development):**
- Internal Storage (MinIO-Results, port 9002)
- Bucket: `config-schemas`
- Schema path: `s3://config-schemas/{framework}.json`

**Production:**
- Cloudflare R2
- Bucket: `vision-platform-results` (or configured bucket)
- Schema path: `s3://{bucket}/schemas/{framework}.json`

## Schema Definition

### File Structure

Each trainer must include:

```
platform/trainers/{framework}/
├── train.py
├── config_schema.py    # Schema definition
├── utils.py
└── README.md
```

### config_schema.py Template

```python
"""
{Framework} Training Configuration Schema

Defines configuration fields for dynamic UI generation.
"""

from typing import Dict, Any, List


def get_config_schema() -> Dict[str, Any]:
    """
    Return configuration schema for {Framework} models.

    Returns:
        Dict with framework, description, version, fields, and presets
    """
    fields = [
        # ========== Optimizer Settings ==========
        {
            "name": "optimizer_type",
            "type": "select",
            "default": "Adam",
            "options": ["Adam", "AdamW", "SGD", "RMSprop"],
            "description": "Optimizer algorithm",
            "group": "optimizer",
            "required": False,
            "advanced": False  # Show by default
        },
        {
            "name": "learning_rate",
            "type": "float",
            "default": 0.001,
            "min": 0.0,
            "max": 1.0,
            "step": 0.0001,
            "description": "Initial learning rate",
            "group": "optimizer",
            "required": False,
            "advanced": True  # Hide by default (show in "Advanced" toggle)
        },
        # ... more fields
    ]

    presets = {
        "easy": {
            "learning_rate": 0.001,
            "epochs": 50,
        },
        "medium": {
            "learning_rate": 0.0005,
            "epochs": 100,
            "weight_decay": 0.0001,
        },
        "advanced": {
            "learning_rate": 0.0001,
            "epochs": 200,
            "weight_decay": 0.001,
            "scheduler": "cosine",
        }
    }

    return {
        "framework": "{framework}",
        "description": "{Framework} Training Configuration",
        "version": "1.0",
        "fields": fields,
        "presets": presets
    }


if __name__ == "__main__":
    """Test schema generation"""
    import json
    schema = get_config_schema()
    print(json.dumps(schema, indent=2))
    print(f"\nTotal fields: {len(schema['fields'])}")
    print(f"Presets: {list(schema['presets'].keys())}")
```

### Field Types

| Type | Description | Required Attributes | Optional Attributes |
|------|-------------|---------------------|---------------------|
| `int` | Integer input | `name`, `type`, `default` | `min`, `max`, `step`, `description`, `group`, `required`, `advanced` |
| `float` | Float input | `name`, `type`, `default` | `min`, `max`, `step`, `description`, `group`, `required`, `advanced` |
| `bool` | Boolean toggle | `name`, `type`, `default` | `description`, `group`, `required`, `advanced` |
| `select` | Dropdown menu | `name`, `type`, `default`, `options` | `description`, `group`, `required`, `advanced` |
| `string` | Text input | `name`, `type`, `default` | `description`, `group`, `required`, `advanced` |

### Field Grouping

Organize fields into logical groups for better UI presentation:

- `optimizer`: Optimizer settings (learning rate, weight decay, etc.)
- `scheduler`: Learning rate scheduler settings
- `augmentation`: Data augmentation parameters
- `optimization`: Training optimizations (mixed precision, gradient clipping)
- `validation`: Validation settings (interval, metrics)

### Presets

Presets provide quick-start configurations:

- `easy`: Minimal configuration for beginners
- `medium`: Balanced configuration for intermediate users
- `advanced`: Aggressive configuration for experts

Presets only need to specify non-default values.

## Upload Script

### Usage

**Validate schemas (dry-run):**
```bash
python platform/scripts/upload_config_schemas.py --all --dry-run
```

**Upload single schema:**
```bash
python platform/scripts/upload_config_schemas.py --framework ultralytics
```

**Upload all schemas:**
```bash
python platform/scripts/upload_config_schemas.py --all
```

### Environment Variables

**Tier-0 (Local Development):**
```bash
INTERNAL_STORAGE_ENDPOINT=http://localhost:9002
INTERNAL_STORAGE_ACCESS_KEY=minioadmin
INTERNAL_STORAGE_SECRET_KEY=minioadmin
INTERNAL_BUCKET_SCHEMAS=config-schemas
```

**Production (Cloudflare R2):**
```bash
AWS_S3_ENDPOINT_URL=https://{account-id}.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID={r2-access-key}
AWS_SECRET_ACCESS_KEY={r2-secret-key}
S3_BUCKET_RESULTS=vision-platform-results
```

### How It Works

1. **Auto-Discovery**: Scans `platform/trainers/` for directories with `config_schema.py`
2. **Import & Validate**: Dynamically imports each schema, calls `get_config_schema()`
3. **Validation**: Checks required keys (`framework`, `description`, `version`, `fields`, `presets`)
4. **Upload**: Uploads to S3/R2 with `application/json` content type and 5-minute cache

### Example Output

```bash
================================================================================
Config Schema Upload Script
================================================================================
Platform directory: /path/to/platform
Trainers directory: /path/to/platform/trainers
Target bucket: config-schemas
Dry run: False
================================================================================
[OK] Found trainer: ultralytics

Processing ultralytics...
[OK] Extracted schema for ultralytics
  - Fields: 24
  - Presets: ['easy', 'medium', 'advanced']
[OK] Uploaded schema to s3://config-schemas/ultralytics.json

================================================================================
Summary
================================================================================
[OK] Success: 1/1
[ERROR] Failed: 0/1

[OK] All schemas uploaded successfully!
```

## GitHub Actions Integration

### Workflow Triggers

The workflow runs on:
- **Pull Request**: Validates schemas (dry-run)
- **Push to main/production**: Uploads schemas to Cloudflare R2
- **Manual Trigger**: `workflow_dispatch` for manual runs

### Configuration

**Required GitHub Secrets:**
- `R2_ENDPOINT_URL`: Cloudflare R2 endpoint
- `R2_ACCESS_KEY_ID`: R2 access key
- `R2_SECRET_ACCESS_KEY`: R2 secret key
- `S3_BUCKET_RESULTS`: Production bucket name

### PR Validation

When a PR modifies `config_schema.py`:
1. Workflow validates all schemas with `--dry-run`
2. Posts comment on PR with validation results
3. Prevents merge if validation fails

**Success Comment:**
```
✅ Training Configuration Schema Validation

All schemas validated successfully!

Next Steps:
- Schemas will be automatically uploaded to production storage when merged
- No Backend redeployment needed
```

**Failure Comment:**
```
❌ Training Configuration Schema Validation Failed

One or more schemas failed validation. Check logs for details.

Common Issues:
- Missing required keys
- Invalid field structure
- get_config_schema() function not found
```

### Production Upload

On merge to `main` or `production`:
1. Workflow uploads all schemas to Cloudflare R2
2. Generates upload summary in workflow logs
3. Schemas become available immediately via Backend API

## Backend API Integration

### Endpoint

```http
GET /api/v1/training/config-schema?framework={framework}
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/training/config-schema?framework=ultralytics"
```

**Response:**
```json
{
  "framework": "ultralytics",
  "description": "Ultralytics YOLO Training Configuration",
  "version": "1.0",
  "fields": [
    {
      "name": "optimizer_type",
      "type": "select",
      "default": "Adam",
      "options": ["Adam", "AdamW", "SGD", "RMSprop"],
      "description": "Optimizer algorithm",
      "group": "optimizer",
      "required": false,
      "advanced": false
    },
    ...
  ],
  "presets": {
    "easy": {...},
    "medium": {...},
    "advanced": {...}
  }
}
```

### Backend Implementation

The Backend fetches schemas from S3 on-demand:

```python
@router.get("/config-schema")
async def get_config_schema(framework: str):
    """Fetch configuration schema from S3."""
    s3_key = f"{framework}.json"

    try:
        response = s3_client.get_object(
            Bucket=settings.internal_bucket_schemas,
            Key=s3_key
        )
        schema = json.loads(response['Body'].read())
        return schema

    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Schema not found")
```

**Benefits:**
- No schema caching needed (S3 caching handles it)
- Schemas update without Backend redeployment
- Low latency (S3 5-minute cache + CDN if needed)

## Frontend Integration

### Dynamic Form Rendering

The Frontend fetches schemas and renders forms dynamically:

```typescript
import useSWR from 'swr';

function DynamicConfigPanel({ framework }: { framework: string }) {
  const { data: schema, error } = useSWR(
    `/training/config-schema?framework=${framework}`
  );

  if (!schema) return <Loading />;

  return (
    <div>
      <h2>{schema.description}</h2>

      {/* Group fields by category */}
      {Object.entries(groupByCategory(schema.fields)).map(([group, fields]) => (
        <FieldGroup key={group} name={group}>
          {fields.map(field => (
            <DynamicField key={field.name} field={field} />
          ))}
        </FieldGroup>
      ))}

      {/* Preset selector */}
      <PresetSelector presets={schema.presets} />
    </div>
  );
}

function DynamicField({ field }: { field: Field }) {
  switch (field.type) {
    case 'int':
    case 'float':
      return <NumberInput field={field} />;
    case 'bool':
      return <BooleanToggle field={field} />;
    case 'select':
      return <SelectDropdown field={field} />;
    default:
      return <TextInput field={field} />;
  }
}
```

### MVP Implementation

The MVP already includes a complete implementation:
- File: `mvp/frontend/components/training/DynamicConfigPanel.tsx`
- Features:
  - Fetches schema from Backend
  - Renders fields by type
  - Groups fields by category
  - Shows/hides advanced fields
  - Applies presets

This component can be reused in the Platform Frontend with minimal changes.

## Adding a New Framework

### Step-by-Step Guide

**1. Create Trainer Structure**

```bash
cd platform/trainers
cp -r ultralytics/ {new-framework}/
```

**2. Create config_schema.py**

```bash
cd {new-framework}/
nano config_schema.py
```

Define your schema following the template above.

**3. Validate Schema Locally**

```bash
python config_schema.py
```

Verify output is valid JSON with all required keys.

**4. Test Upload (Dry-Run)**

```bash
python ../../scripts/upload_config_schemas.py --framework {new-framework} --dry-run
```

**5. Create Pull Request**

```bash
git checkout -b feature/add-{new-framework}-schema
git add platform/trainers/{new-framework}/config_schema.py
git commit -m "feat(trainers): add configuration schema for {new-framework}"
git push origin feature/add-{new-framework}-schema
```

**6. Wait for PR Validation**

GitHub Actions will:
- Validate your schema
- Post comment with validation results
- Block merge if validation fails

**7. Merge to Main**

On merge:
- Schema automatically uploads to Cloudflare R2
- Available immediately via Backend API
- Frontend can render your configuration UI

## Best Practices

### Schema Design

1. **Start Simple**: Begin with 5-10 essential fields
2. **Group Logically**: Use groups to organize related settings
3. **Provide Defaults**: Every field should have a sensible default
4. **Use Advanced Flag**: Hide advanced settings by default
5. **Add Descriptions**: Help users understand what each field does

### Field Naming

- Use snake_case: `learning_rate`, not `learningRate`
- Be descriptive: `warmup_epochs`, not `we`
- Match framework conventions: Use framework's native parameter names when possible

### Presets

- **Easy**: Minimal config, fast training, good for prototyping
- **Medium**: Balanced config, recommended for most users
- **Advanced**: Aggressive config, best performance, longer training

### Validation

- Run `--dry-run` before committing
- Test schema import: `python config_schema.py`
- Check JSON is valid: `python config_schema.py | jq .`

## Troubleshooting

### Common Errors

**1. Schema validation failed: Missing required key**

```
[ERROR] Failed to extract schema for ultralytics: Schema missing required key: fields
```

**Fix:** Ensure your schema returns all required keys:
- `framework`
- `description`
- `version`
- `fields`
- `presets`

**2. get_config_schema() function not found**

```
[ERROR] ultralytics/config_schema.py must have get_config_schema() function
```

**Fix:** Ensure your module exports `get_config_schema()` function.

**3. Bucket not found**

```
[ERROR] Bucket not found: config-schemas
```

**Fix:** Ensure Internal Storage (MinIO-Results) is running:
```bash
docker-compose -f platform/infrastructure/docker-compose.tier0.yaml up -d
```

**4. Invalid field structure**

```
[ERROR] Field 'optimizer_type' missing required attribute: type
```

**Fix:** Every field must have at minimum:
- `name`
- `type`
- `default`

### Debugging

**Check schema output:**
```bash
python platform/trainers/{framework}/config_schema.py
```

**Validate JSON:**
```bash
python platform/trainers/{framework}/config_schema.py | jq .
```

**Test upload locally:**
```bash
python platform/scripts/upload_config_schemas.py --framework {framework} --dry-run
```

**Check S3 storage:**
```bash
# MinIO CLI (mc)
mc ls local/config-schemas/

# AWS CLI
aws --endpoint-url http://localhost:9002 s3 ls s3://config-schemas/
```

## Migration Guide

### From Hardcoded Config to Schema

**Before (Hardcoded in Backend):**
```python
# backend/app/models/training_config.py
ULTRALYTICS_CONFIG = {
    "mosaic": {"type": "float", "default": 1.0},
    "mixup": {"type": "float", "default": 0.0},
    ...
}
```

**After (Schema-Driven):**
```python
# platform/trainers/ultralytics/config_schema.py
def get_config_schema():
    return {
        "framework": "ultralytics",
        "fields": [
            {"name": "mosaic", "type": "float", "default": 1.0},
            {"name": "mixup", "type": "float", "default": 0.0},
            ...
        ],
        "presets": {...}
    }
```

**Benefits:**
- Config lives with trainer (single source of truth)
- Updates without Backend redeployment
- Version controlled in Git
- Framework maintainers control their own config

## References

### Implementation Files

- **Schema Definition**: `platform/trainers/ultralytics/config_schema.py`
- **Upload Script**: `platform/scripts/upload_config_schemas.py`
- **GitHub Actions**: `.github/workflows/upload-config-schemas.yml`
- **Backend API**: `platform/backend/app/api/training.py` (config-schema endpoint)
- **Frontend Component**: `mvp/frontend/components/training/DynamicConfigPanel.tsx` (reference)

### Related Documentation

- [Ultralytics Trainer README](../platform/trainers/ultralytics/README.md) - Advanced Config section
- [MVP Plan](./planning/MVP_PLAN.md) - Phase 3.2 details
- [Checklist](./planning/MVP_TO_PLATFORM_CHECKLIST.md) - Phase 3.2 implementation status

### External Resources

- [Ultralytics YOLO Training Args](https://docs.ultralytics.com/modes/train/) - Reference for YOLO parameters
- [Cloudflare R2 Docs](https://developers.cloudflare.com/r2/) - R2 configuration guide
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets) - Secret management

## FAQ

**Q: Do I need to restart Backend after uploading schemas?**
A: No. Backend fetches schemas from S3 on-demand. Changes are available immediately.

**Q: Can I have different schemas for different model variants?**
A: Yes. Use query parameters: `/config-schema?framework=ultralytics&model=yolov8n`

**Q: What happens if schema upload fails?**
A: GitHub Actions workflow will fail, preventing merge. Fix validation errors and re-push.

**Q: How do I version schemas?**
A: Include `version` field in schema. Backend can support multiple versions with versioned S3 keys.

**Q: Can Frontend cache schemas?**
A: Yes. Use SWR with 5-minute revalidation to match S3 cache TTL.

**Q: How do I test schema changes locally?**
A: Use `--dry-run` to validate, then upload to local MinIO for testing with Backend/Frontend.

---

**Last Updated**: 2025-11-14
**Status**: ✅ Production-Ready (Phase 3.2 Complete)
