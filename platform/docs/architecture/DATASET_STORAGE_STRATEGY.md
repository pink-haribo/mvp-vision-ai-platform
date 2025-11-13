# Dataset Storage Strategy

**Date**: 2025-01-10
**Status**: Production Design

Complete strategy for dataset storage, versioning, and integrity management.

## Table of Contents

- [Overview](#overview)
- [Core Principles](#core-principles)
- [Storage Structure](#storage-structure)
- [Snapshot System](#snapshot-system)
- [Integrity Management](#integrity-management)
- [API Patterns](#api-patterns)
- [Implementation Guide](#implementation-guide)

---

## Overview

The platform uses **individual file storage with meta-based snapshots** to provide:

- ✅ Labeling tool support (presigned URLs)
- ✅ Incremental updates (add/remove single files)
- ✅ Space-efficient versioning (99% storage savings)
- ✅ Training reproducibility (snapshot-based)
- ✅ Integrity tracking (broken/repair states)

### Key Decision: Individual Files

**Why not zip files?**

Zip files are incompatible with browser-based labeling tools that need direct image access via presigned URLs. Individual files enable real-time labeling workflows while maintaining production-grade versioning through lightweight snapshots.

---

## Core Principles

### 1. Images Are Always Latest (HEAD)

**Rule**: The `images/` folder always contains the current state of the dataset.

```
datasets/{dataset-id}/
├── images/               ← Always HEAD (latest)
│   ├── img001.jpg       ← Current version
│   ├── img002.jpg       ← Current version
│   └── ...
```

**Implications**:
- ✅ Adding images: Just upload to `images/`
- ✅ Deleting images: Remove from `images/` (with impact analysis)
- ✅ Modifying images: Delete + re-upload (triggers integrity checks)

### 2. Snapshots Are Lightweight Metadata

**Rule**: Snapshots only store `annotations.json` + `meta.json`, NOT images.

```
datasets/{dataset-id}/
├── images/                        ← Shared by all snapshots (10GB)
└── snapshots/
    ├── training-job-123.json     ← Snapshot metadata (500KB)
    ├── training-job-456.json     ← Snapshot metadata (500KB)
    └── v1.json                   ← Named version (500KB)
```

**Storage savings**:
```
Traditional versioning: 10GB + 10GB + 10GB = 30GB
Meta-based snapshots: 10GB + 500KB + 500KB + 500KB = 10.0015GB
Savings: 99.95%
```

### 3. Automatic Snapshots on Training

**Rule**: Every training job creates an automatic snapshot for reproducibility.

```python
# Backend: Before starting training
snapshot_id = f"training-job-{job.id}"
create_snapshot(dataset_id, snapshot_id)

# TrainingJob record
job.dataset_snapshot_id = snapshot_id  # For reproducibility
```

### 4. task_type Is NOT a Dataset Property

**Wrong concept**:
```python
# ❌ Dataset has task_type
class Dataset:
    task_type: str  # classification, detection, etc.
```

**Correct concept**:
```python
# ✅ Dataset: Just images + annotations
class Dataset:
    id: str
    num_images: int
    labeled: bool

# ✅ TrainingJob: Defines task_type
class TrainingJob:
    dataset_id: str
    task_type: str  # How to use this dataset
```

**Reason**: Same images can be used for classification, detection, segmentation, etc.

---

## Storage Structure

### Complete S3 Layout

```
s3://bucket/datasets/{dataset-id}/
│
├── meta.json                      ← Dataset metadata (created when labeled)
├── annotations.json               ← Platform Format annotations (HEAD)
│
├── images/                        ← All images (shared by all snapshots)
│   ├── cats/cat001.jpg           ← Preserves folder structure
│   ├── dogs/dog001.jpg
│   └── subdir/bird001.jpg
│
└── snapshots/                     ← Snapshot metadata only
    ├── training-job-abc123.json  ← Auto-created during training
    ├── training-job-def456.json
    └── v1.json                   ← User-created named version
```

### Snapshot Metadata Structure

**File**: `snapshots/training-job-abc123.json`

```json
{
  "snapshot_id": "training-job-abc123",
  "snapshot_type": "training",  // "training" | "manual"
  "created_at": "2025-01-10T10:00:00Z",
  "created_by": "user-123",
  "parent_dataset_id": "dataset-xyz",

  "annotations": {
    /* Complete annotations.json content at snapshot time */
    "format_version": "1.0",
    "dataset_id": "dataset-xyz",
    "classes": [
      {"id": 0, "name": "cat"},
      {"id": 1, "name": "dog"}
    ],
    "images": [
      {
        "id": 1,
        "file_name": "cats/cat001.jpg",
        "annotation": {...}
      }
    ]
  },

  "integrity": {
    "total_images": 100,
    "annotations_hash": "sha256:abc123...",
    "image_checksums": {
      "cats/cat001.jpg": "sha256:def456...",
      "dogs/dog001.jpg": "sha256:ghi789..."
    }
  },

  "metadata": {
    "num_classes": 2,
    "class_distribution": {
      "cat": 60,
      "dog": 40
    }
  }
}
```

### Dataset Lifecycle States

```
unlabeled → labeled → snapshot-valid → snapshot-broken
                           ↓
                      snapshot-repaired
```

**State definitions**:
- `unlabeled`: Only images, no annotations.json
- `labeled`: Has annotations.json (HEAD)
- `snapshot-valid`: All referenced images exist
- `snapshot-broken`: Some referenced images missing
- `snapshot-repaired`: Was broken, annotation updated to exclude missing images

---

## Snapshot System

### Automatic Training Snapshot

**Backend workflow**:

```python
# backend/app/services/training_service.py

async def start_training(job_id: str, dataset_id: str):
    """Start training job with automatic snapshot"""

    # 1. Create snapshot
    snapshot_id = f"training-job-{job_id}"
    snapshot = await create_snapshot(
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        snapshot_type="training",
        created_by=job.user_id
    )

    # 2. Record snapshot in training job
    job.dataset_snapshot_id = snapshot_id
    job.snapshot_status_at_start = "valid"
    await db.commit()

    # 3. Pass snapshot info to trainer
    env = {
        "DATASET_ID": dataset_id,
        "SNAPSHOT_ID": snapshot_id,
        "ANNOTATIONS_PATH": f"datasets/{dataset_id}/snapshots/{snapshot_id}.json",
        "IMAGES_PATH": f"datasets/{dataset_id}/images/",
        # ... other env vars
    }

    await start_trainer_job(job_id, env)
```

**Snapshot creation**:

```python
async def create_snapshot(
    dataset_id: str,
    snapshot_id: str,
    snapshot_type: str,
    created_by: str
) -> Snapshot:
    """Create snapshot metadata file"""

    # 1. Load current annotations
    annotations = await s3.get_object(
        bucket,
        f"datasets/{dataset_id}/annotations.json"
    )

    # 2. Calculate image checksums (for integrity)
    checksums = {}
    for img in annotations['images']:
        file_path = f"datasets/{dataset_id}/images/{img['file_name']}"
        checksum = await calculate_s3_etag(bucket, file_path)
        checksums[img['file_name']] = checksum

    # 3. Create snapshot metadata
    snapshot_data = {
        "snapshot_id": snapshot_id,
        "snapshot_type": snapshot_type,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "created_by": created_by,
        "parent_dataset_id": dataset_id,
        "annotations": annotations,  # Full annotations at snapshot time
        "integrity": {
            "total_images": len(annotations['images']),
            "annotations_hash": hashlib.sha256(
                json.dumps(annotations).encode()
            ).hexdigest(),
            "image_checksums": checksums
        }
    }

    # 4. Upload snapshot metadata
    await s3.put_object(
        bucket,
        f"datasets/{dataset_id}/snapshots/{snapshot_id}.json",
        json.dumps(snapshot_data)
    )

    # 5. Record in database
    snapshot = Snapshot(
        id=snapshot_id,
        dataset_id=dataset_id,
        snapshot_type=snapshot_type,
        status="valid",
        created_at=datetime.utcnow()
    )
    db.add(snapshot)
    await db.commit()

    return snapshot
```

### Manual Named Versions

**User creates explicit version**:

```python
# API: POST /api/v1/datasets/{dataset_id}/versions
async def create_version(dataset_id: str, version_name: str):
    """Create user-named version (e.g., v1, v2, production)"""

    snapshot_id = f"{dataset_id}-{version_name}"

    snapshot = await create_snapshot(
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        snapshot_type="manual",
        created_by=current_user.id
    )

    # Add version tag
    snapshot.version_tag = version_name
    await db.commit()

    return {
        "snapshot_id": snapshot_id,
        "version_tag": version_name,
        "created_at": snapshot.created_at
    }
```

**UI display**:

```
Dataset: Pet Classification
├─ HEAD (current)          150 images    [Create Version]
│
└─ Versions:
   ├─ v2 (2025-01-10)      150 images    [Restore] [Delete]
   ├─ v1 (2025-01-05)      100 images    [Restore] [Delete]
   └─ Training Snapshots:
      ├─ job-abc123        100 images    [View Job]
      └─ job-def456        120 images    [View Job]
```

---

## Integrity Management

### Image Deletion Impact Analysis

**Problem**: Deleting an image may break past snapshots that reference it.

**Solution**: Analyze impact before deletion, offer options.

```python
# API: DELETE /api/v1/datasets/{dataset_id}/images/{filename}
async def delete_image_with_impact_analysis(
    dataset_id: str,
    filename: str
):
    """Delete image with impact analysis"""

    # 1. Find affected snapshots
    affected = []
    snapshots = await get_all_snapshots(dataset_id)

    for snapshot in snapshots:
        snapshot_data = await load_snapshot_metadata(snapshot.id)
        image_files = [
            img['file_name']
            for img in snapshot_data['annotations']['images']
        ]

        if filename in image_files:
            affected.append({
                "snapshot_id": snapshot.id,
                "snapshot_type": snapshot.snapshot_type,
                "created_at": snapshot.created_at,
                "version_tag": snapshot.version_tag
            })

    # 2. Return impact report
    if affected:
        return {
            "status": "confirmation_required",
            "message": f"This image is used in {len(affected)} snapshot(s)",
            "affected_snapshots": affected,
            "options": [
                {
                    "action": "mark_broken",
                    "description": "Delete image, mark snapshots as broken (cannot reproduce training)"
                },
                {
                    "action": "auto_repair",
                    "description": "Delete image, remove from snapshot annotations (modifies history)"
                },
                {
                    "action": "cancel",
                    "description": "Cancel deletion"
                }
            ]
        }

    # 3. No impact - safe to delete
    await s3.delete_object(bucket, f"datasets/{dataset_id}/images/{filename}")

    # Update HEAD annotations
    annotations = await load_annotations(dataset_id)
    annotations['images'] = [
        img for img in annotations['images']
        if img['file_name'] != filename
    ]
    await save_annotations(dataset_id, annotations)

    return {"status": "deleted", "affected_snapshots": []}
```

### Option 1: Mark Snapshots as Broken

```python
async def delete_image_mark_broken(dataset_id: str, filename: str):
    """Delete image and mark affected snapshots as broken"""

    # 1. Delete image
    await s3.delete_object(
        bucket,
        f"datasets/{dataset_id}/images/{filename}"
    )

    # 2. Mark snapshots as broken
    for snapshot in affected_snapshots:
        snapshot.status = "broken"
        snapshot.integrity_status = {
            "total_images": snapshot.total_images,
            "missing_images": [filename],
            "missing_count": 1,
            "broken_at": datetime.utcnow().isoformat() + "Z"
        }
        await db.commit()

    return {"status": "deleted", "broken_snapshots": len(affected_snapshots)}
```

**Result**:
- Image deleted from HEAD
- Snapshots marked as `broken`
- Training jobs using those snapshots cannot be reproduced
- Records preserved for audit trail

### Option 2: Auto-Repair Snapshots

```python
async def delete_image_auto_repair(dataset_id: str, filename: str):
    """Delete image and update snapshot annotations"""

    # 1. Delete image
    await s3.delete_object(
        bucket,
        f"datasets/{dataset_id}/images/{filename}"
    )

    # 2. Update each affected snapshot
    for snapshot in affected_snapshots:
        # Load snapshot metadata
        snapshot_data = await load_snapshot_metadata(snapshot.id)

        # Remove image from annotations
        annotations = snapshot_data['annotations']
        image_id = None

        # Find and remove image
        annotations['images'] = [
            img for img in annotations['images']
            if img['file_name'] != filename or (image_id := img['id']) is None
        ]

        # Remove annotations for that image
        if image_id:
            annotations['annotations'] = [
                ann for ann in annotations.get('annotations', [])
                if ann.get('image_id') != image_id
            ]

        # Update snapshot
        snapshot_data['annotations'] = annotations
        snapshot_data['integrity']['removed_images'] = [filename]

        await save_snapshot_metadata(snapshot.id, snapshot_data)

        # Update DB
        snapshot.status = "valid"
        snapshot.integrity_status = {
            "repaired_at": datetime.utcnow().isoformat() + "Z",
            "removed_images": [filename]
        }
        await db.commit()

    return {"status": "deleted", "repaired_snapshots": len(affected_snapshots)}
```

**Result**:
- Image deleted from HEAD
- Snapshot annotations updated (image removed)
- Snapshots remain `valid`
- Training jobs can still be reproduced with modified dataset

### Periodic Integrity Check

```python
# Background task: Run daily
@celery.task
async def check_snapshot_integrity():
    """Check all valid snapshots for missing images"""

    snapshots = await db.query(Snapshot).filter(
        Snapshot.status == "valid"
    ).all()

    for snapshot in snapshots:
        snapshot_data = await load_snapshot_metadata(snapshot.id)
        missing = []

        for img in snapshot_data['annotations']['images']:
            image_path = f"datasets/{snapshot.dataset_id}/images/{img['file_name']}"

            if not await s3.object_exists(bucket, image_path):
                missing.append(img['file_name'])

        if missing:
            # Mark as broken
            snapshot.status = "broken"
            snapshot.integrity_status = {
                "total_images": len(snapshot_data['annotations']['images']),
                "missing_images": missing,
                "missing_count": len(missing),
                "detected_at": datetime.utcnow().isoformat() + "Z"
            }
            await db.commit()

            # Notify admin
            await notify_admin(
                f"Snapshot {snapshot.id} broken: {len(missing)} images missing"
            )
```

---

## API Patterns

### Upload Dataset (Labeled)

```python
# POST /api/v1/datasets/upload
async def upload_dataset(files: List[UploadFile]):
    """Upload dataset with folder structure + annotations"""

    # 1. Generate dataset ID
    dataset_id = f"dataset-{uuid.uuid4()}"

    # 2. Parse uploaded files
    annotations_file = find_file(files, "annotations.json")
    image_files = [f for f in files if f.filename.startswith("images/")]

    # 3. Validate annotations format
    annotations = json.loads(await annotations_file.read())
    validate_platform_format(annotations)

    # 4. Upload images (preserve folder structure)
    for img_file in image_files:
        await s3.upload_fileobj(
            img_file.file,
            bucket,
            f"datasets/{dataset_id}/{img_file.filename}"
        )

    # 5. Upload annotations
    await s3.put_object(
        bucket,
        f"datasets/{dataset_id}/annotations.json",
        json.dumps(annotations)
    )

    # 6. Create meta.json
    meta = {
        "dataset_id": dataset_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_images": len(annotations['images']),
        "num_classes": len(annotations['classes'])
    }
    await s3.put_object(
        bucket,
        f"datasets/{dataset_id}/meta.json",
        json.dumps(meta)
    )

    # 7. Save to database
    dataset = Dataset(
        id=dataset_id,
        name=annotations.get('dataset_name', 'Untitled'),
        labeled=True,
        num_images=len(annotations['images']),
        num_classes=len(annotations['classes']),
        storage_path=f"datasets/{dataset_id}/"
    )
    db.add(dataset)
    await db.commit()

    return {"dataset_id": dataset_id}
```

### Get Image for Labeling Tool

```python
# GET /api/v1/datasets/{dataset_id}/images/{filename}/url
async def get_image_presigned_url(dataset_id: str, filename: str):
    """Generate presigned URL for labeling tool"""

    # 1. Verify access
    dataset = await db.get(Dataset, dataset_id)
    if not has_permission(current_user, dataset):
        raise PermissionError()

    # 2. Generate presigned URL (1 hour expiry)
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': bucket,
            'Key': f'datasets/{dataset_id}/images/{filename}'
        },
        ExpiresIn=3600
    )

    return {"url": url, "expires_in": 3600}
```

### Trainer: Load Dataset from Snapshot

```python
# Training Service: train.py
def load_dataset_from_snapshot(dataset_id: str, snapshot_id: str):
    """Load dataset using specific snapshot"""

    # 1. Download snapshot metadata
    snapshot_path = f"datasets/{dataset_id}/snapshots/{snapshot_id}.json"
    snapshot_data = s3.get_object(bucket, snapshot_path)

    # 2. Extract annotations
    annotations = snapshot_data['annotations']

    # 3. Download images (shared by all snapshots)
    for img in annotations['images']:
        s3.download_file(
            bucket,
            f"datasets/{dataset_id}/images/{img['file_name']}",
            f"/workspace/dataset/images/{img['file_name']}"
        )

    # 4. Save annotations locally
    with open("/workspace/dataset/annotations.json", 'w') as f:
        json.dump(annotations, f)

    return annotations
```

---

## Implementation Guide

### Database Schema

```python
# platform/backend/app/db/models.py

class Dataset(Base):
    __tablename__ = "datasets"

    # Basic info
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    # Status
    labeled = Column(Boolean, default=False)
    num_images = Column(Integer)
    num_classes = Column(Integer, nullable=True)
    class_names = Column(JSON, nullable=True)

    # Storage
    storage_type = Column(String, default="s3")  # "s3" (MinIO/R2)
    storage_path = Column(String)  # "datasets/{id}/"

    # Metadata
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


class Snapshot(Base):
    __tablename__ = "snapshots"

    # Identity
    id = Column(String, primary_key=True)  # "training-job-123" or "dataset-xyz-v1"
    dataset_id = Column(String, ForeignKey("datasets.id"))

    # Type
    snapshot_type = Column(String)  # "training" | "manual"
    version_tag = Column(String, nullable=True)  # "v1", "v2", etc.

    # Integrity
    status = Column(String, default="valid")  # "valid" | "broken" | "repairing"
    integrity_status = Column(JSON, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))

    # Relationships
    dataset = relationship("Dataset", back_populates="snapshots")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"))
    dataset_snapshot_id = Column(String, ForeignKey("snapshots.id"))
    snapshot_status_at_start = Column(String)  # "valid" | "broken"

    task_type = Column(String)  # "classification" | "detection" | etc.
    model_name = Column(String)
    # ... other training config
```

### Environment Variables (Trainer)

```bash
# Tier 1: Subprocess (local dev)
DATASET_ID=dataset-abc123
SNAPSHOT_ID=training-job-456
S3_ENDPOINT=http://localhost:9000
BUCKET_NAME=vision-platform-dev

# Tier 2: Kind
S3_ENDPOINT=http://minio.platform.svc:9000
BUCKET_NAME=vision-platform-dev

# Tier 3: Production
S3_ENDPOINT=https://xxx.r2.cloudflarestorage.com
BUCKET_NAME=vision-platform-prod
```

### Consistency Across Tiers

**Same code, different endpoints**:

```python
# Works identically in all tiers
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['S3_ENDPOINT'],
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

# Download snapshot
s3.download_file(
    os.environ['BUCKET_NAME'],
    f"datasets/{dataset_id}/snapshots/{snapshot_id}.json",
    "/tmp/snapshot.json"
)

# Download images
for img in annotations['images']:
    s3.download_file(
        os.environ['BUCKET_NAME'],
        f"datasets/{dataset_id}/images/{img['file_name']}",
        f"/workspace/images/{img['file_name']}"
    )
```

---

## Benefits

### Space Efficiency

```
Traditional approach (copy images per version):
  v1: 10GB images + 500KB annotations = 10.0005GB
  v2: 10GB images + 500KB annotations = 10.0005GB
  v3: 10GB images + 500KB annotations = 10.0005GB
  Total: 30.0015GB

Meta-based approach (shared images):
  images/: 10GB (shared)
  snapshot v1: 500KB
  snapshot v2: 500KB
  snapshot v3: 500KB
  Total: 10.0015GB

Savings: 66.7% (3x reduction)
```

### Performance

**Upload speed** (1000 images):
- Individual files: ~30 seconds (parallel upload)
- Metadata snapshot: <1 second

**Training start time**:
- Download snapshot metadata: ~1 second
- Download images: ~20 seconds (same for all approaches)

**Incremental update** (add 10 images):
- Upload 10 files: ~3 seconds
- Update annotations.json: <1 second
- Total: ~4 seconds

### Flexibility

- ✅ Labeling tool support (presigned URLs)
- ✅ Incremental updates (no full re-upload)
- ✅ Per-job reproducibility (automatic snapshots)
- ✅ Named versions (user-created)
- ✅ Integrity tracking (broken/repair)
- ✅ Multi-user workflows (different snapshot per job)

---

## References

- [ISOLATION_DESIGN.md](./ISOLATION_DESIGN.md) - Backend ↔ Trainer isolation
- [3_TIER_DEVELOPMENT.md](../development/3_TIER_DEVELOPMENT.md) - S3 API consistency
- [DATASET_SPLIT_STRATEGY.md](./DATASET_SPLIT_STRATEGY.md) - Train/val split handling
- [TRAINER_DESIGN.md](./TRAINER_DESIGN.md) - Trainer implementation

---

**Last Updated**: 2025-01-10
**Status**: Production Design
