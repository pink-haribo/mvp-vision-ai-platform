# Dual Storage Architecture

## Overview

The platform uses a **Dual Storage Architecture** to optimize costs and performance:

- **Internal Storage** (MinIO - Backend location): Small, frequently accessed files
- **External Storage** (S3/R2 - Cloud): Large, infrequently accessed files

## Storage Distribution

### Internal Storage (MinIO - Same location as Backend)

**Purpose**: Low-latency access to small, frequently used files

**Buckets**:
- `model-weights`: Pretrained model weights (~500MB each)
- `training-checkpoints`: Training checkpoints (~500MB per checkpoint)
- `config-schemas`: Framework configuration schemas (~10KB each)

**Characteristics**:
- Low latency (< 1ms from Backend)
- Small total size (~tens of GB)
- Private network only
- No external access needed

### External Storage (Cloudflare R2 / AWS S3 - Cloud)

**Purpose**: Cost-effective storage for large datasets

**Buckets**:
- `training-datasets`: Training images and annotations (GB ~ TB scale)

**Characteristics**:
- High latency (~100-500ms)
- Large total size (TB scale)
- Public URL support (presigned URLs)
- Free egress (R2)

## Environment Configuration

### Local Development (.env)

```bash
# Both use same MinIO instance
INTERNAL_STORAGE_ENDPOINT=http://localhost:30900
INTERNAL_STORAGE_ACCESS_KEY=minioadmin
INTERNAL_STORAGE_SECRET_KEY=minioadmin

EXTERNAL_STORAGE_ENDPOINT=http://localhost:30900
EXTERNAL_STORAGE_ACCESS_KEY=minioadmin
EXTERNAL_STORAGE_SECRET_KEY=minioadmin

# Internal buckets
INTERNAL_BUCKET_WEIGHTS=model-weights
INTERNAL_BUCKET_CHECKPOINTS=training-checkpoints
INTERNAL_BUCKET_SCHEMAS=config-schemas

# External buckets
EXTERNAL_BUCKET_DATASETS=training-datasets
```

### Production (Railway Variables)

```bash
# Internal Storage (MinIO in Backend cluster)
INTERNAL_STORAGE_ENDPOINT=http://minio.backend.svc.cluster.local:9000
INTERNAL_STORAGE_ACCESS_KEY=${MINIO_ACCESS_KEY}
INTERNAL_STORAGE_SECRET_KEY=${MINIO_SECRET_KEY}

# External Storage (Cloudflare R2)
EXTERNAL_STORAGE_ENDPOINT=https://xxxxx.r2.cloudflarestorage.com
EXTERNAL_STORAGE_ACCESS_KEY=${R2_ACCESS_KEY}
EXTERNAL_STORAGE_SECRET_KEY=${R2_SECRET_KEY}
```

## Usage

### Import DualStorageClient

```python
from app.utils.dual_storage import dual_storage
```

### Internal Storage Operations

```python
# Upload pretrained weight
dual_storage.upload_weight(
    file_path=Path("resnet50.pth"),
    weight_key="resnet50.pth"
)

# Download weight
dual_storage.download_weight(
    weight_key="resnet50.pth",
    dest_path=Path("/tmp/resnet50.pth")
)

# Upload checkpoint
dual_storage.upload_checkpoint(
    file_path=Path("checkpoint.pth"),
    checkpoint_key="jobs/123/checkpoint_epoch_10.pth"
)

# Get config schema
schema_bytes = dual_storage.get_schema("ultralytics")
schema_dict = json.loads(schema_bytes)
```

### External Storage Operations

```python
# Upload dataset
with open("dataset.zip", "rb") as f:
    dual_storage.upload_dataset(
        file_obj=f,
        dataset_key="datasets/abc-123.zip"
    )

# Download dataset
dual_storage.download_dataset(
    dataset_key="datasets/abc-123.zip",
    dest_path=Path("/tmp/dataset.zip")
)

# Generate presigned URL (for Frontend download)
url = dual_storage.generate_dataset_presigned_url(
    dataset_key="datasets/abc-123.zip",
    expiration=3600  # 1 hour
)
```

## Cost Comparison

### Current (Single Storage - All in R2)

```
Storage: Cloudflare R2
- Datasets: 100GB × $0.015/GB = $1.50/month
- Weights: 10GB × $0.015/GB = $0.15/month
- Checkpoints: 50GB × $0.015/GB = $0.75/month
→ Total: $2.40/month
```

### Dual Storage

```
Internal MinIO (Backend location):
- Weights: 10GB
- Checkpoints: 50GB
- Schemas: 1MB
→ Cost: $0/month (included in Backend server)

External R2 (Datasets only):
- Datasets: 100GB × $0.015/GB = $1.50/month
→ Total: $1.50/month (37% savings!)
```

## Benefits

1. **Cost Optimization**: Store large datasets in cheap cloud storage, small files locally
2. **Performance**: Low-latency access to frequently used weights and checkpoints
3. **Scalability**: Cloud storage handles unlimited dataset growth
4. **Security**: Internal storage never exposed to internet
5. **Flexibility**: Easy to swap External storage provider (R2 → S3 → GCS)

## Migration from Legacy Storage

The system maintains backward compatibility with legacy storage configuration:

```python
# Legacy (still works)
AWS_S3_ENDPOINT_URL=http://localhost:30900
S3_BUCKET_DATASETS=training-datasets

# New (preferred)
INTERNAL_STORAGE_ENDPOINT=http://localhost:30900
EXTERNAL_STORAGE_ENDPOINT=http://localhost:30900
```

Code automatically falls back to legacy configuration if dual storage variables are not set.

## Related Documentation

- [Config Schema Distribution](../../../../docs/k8s/20251108_config_schema_distribution.md)
- [Storage Architecture](../../../../docs/architecture/STORAGE_ARCHITECTURE.md) (TODO)
