# Dataset Storage Strategy Comparison

**Date**: 2025-01-10
**Status**: Decision Pending

This document compares two storage strategies for datasets: **Zip File Storage** vs **Individual File Storage**.

## Table of Contents

- [Overview](#overview)
- [Storage Structures](#storage-structures)
- [Comparison Matrix](#comparison-matrix)
- [Scenario Analysis](#scenario-analysis)
- [Cost Analysis](#cost-analysis)
- [Implementation Complexity](#implementation-complexity)
- [Recommendations](#recommendations)

---

## Overview

### Zip File Storage

Store entire dataset as a single compressed archive:

```
s3://bucket/datasets/{dataset-id}/
└── dataset.zip
    ├── annotations.json
    ├── meta.json
    └── images/
        ├── img001.jpg
        ├── img002.jpg
        └── ...
```

### Individual File Storage

Store each file separately with directory structure:

```
s3://bucket/datasets/{dataset-id}/
├── annotations.json
├── meta.json
└── images/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

---

## Storage Structures

### Option A: Zip File Storage

#### S3 Structure
```
datasets/
├── platform-imagenet-{uuid}/
│   └── dataset.zip                    (100 MB compressed)
│       ├── annotations.json           (500 KB)
│       ├── meta.json                  (5 KB)
│       └── images/
│           ├── img001.jpg             (1 MB)
│           ├── img002.jpg             (1 MB)
│           └── ... (100 images)
└── platform-coco-{uuid}/
    └── dataset.zip                    (500 MB compressed)
```

#### Workflow
```
User Upload:
1. User zips dataset locally
2. Upload single zip file → Backend
3. Backend validates format
4. Backend uploads to S3

Training:
1. Trainer downloads zip from S3
2. Extract to /workspace/dataset/
3. Train using extracted files
4. Delete extracted files after training
```

#### Pros
- ✅ **Simple structure**: Single file to manage
- ✅ **Faster upload**: One HTTP request
- ✅ **Faster download**: One HTTP request
- ✅ **Space efficient**: Compression reduces size
- ✅ **Atomic operations**: Upload/download is all-or-nothing
- ✅ **Easy versioning**: Just copy one file

#### Cons
- ❌ **Cannot access individual files**: Must download entire zip
- ❌ **Cannot update single image**: Must re-upload entire dataset
- ❌ **No incremental updates**: Any change requires full re-upload
- ❌ **Labeling tool unfriendly**: Can't load images directly from S3
- ❌ **Disk space during extraction**: Needs 2x space (zip + extracted)

---

### Option B: Individual File Storage

#### S3 Structure
```
datasets/
├── platform-imagenet-{uuid}/
│   ├── annotations.json               (500 KB)
│   ├── meta.json                      (5 KB)
│   └── images/
│       ├── img001.jpg                 (1 MB)
│       ├── img002.jpg                 (1 MB)
│       └── ... (100 images)
└── platform-coco-{uuid}/
    ├── annotations.json               (2 MB)
    ├── meta.json                      (5 KB)
    └── images/
        ├── img001.jpg                 (1 MB)
        └── ... (500 images)
```

#### Workflow
```
User Upload:
1. Frontend extracts zip (if uploaded as zip)
2. Upload each file individually → Backend
3. Backend uploads to S3 (preserving structure)

Training:
1. Trainer lists files in S3
2. Download each file individually
3. Train using downloaded files
4. Delete files after training

Alternative Training (with snapshot):
1. Backend creates snapshot.zip on-demand
2. Trainer downloads snapshot.zip
3. Extract and train
```

#### Pros
- ✅ **Individual file access**: Can download single image
- ✅ **Incremental updates**: Add/remove/modify single files
- ✅ **Labeling tool friendly**: Direct image loading via presigned URLs
- ✅ **No extraction needed**: Files ready to use
- ✅ **Partial downloads**: Download only needed files

#### Cons
- ❌ **Complex structure**: Many files to manage
- ❌ **Slower upload**: N HTTP requests (100-1000+)
- ❌ **Slower download**: N HTTP requests for training
- ❌ **More S3 operations**: Higher API call costs
- ❌ **Consistency issues**: Partial upload failures

---

## Comparison Matrix

| Aspect | Zip File Storage | Individual File Storage | Winner |
|--------|-----------------|------------------------|--------|
| **Upload Speed** | Fast (1 request) | Slow (N requests) | Zip |
| **Upload Reliability** | Atomic | Partial failure possible | Zip |
| **Download Speed (Training)** | Fast (1 request) | Slow (N requests) | Zip |
| **Labeling Tool Support** | ❌ Cannot load images directly | ✅ Load via presigned URLs | Individual |
| **Incremental Updates** | ❌ Must re-upload all | ✅ Update single file | Individual |
| **Storage Space** | ✅ Compressed (~30% savings) | ❌ Uncompressed | Zip |
| **S3 API Calls** | Low (1-2 per operation) | High (N per operation) | Zip |
| **Versioning** | Easy (copy 1 file) | Complex (copy N files) | Zip |
| **Disk Usage During Training** | 2x (zip + extracted) | 1x (files only) | Individual |
| **Consistency** | ✅ All-or-nothing | ⚠️ Partial uploads | Zip |
| **Flexibility** | ❌ Rigid | ✅ Flexible | Individual |

---

## Scenario Analysis

### Scenario 1: Complete Dataset Upload (No Labeling Tool)

**User has pre-labeled dataset, wants to train immediately**

#### Zip Approach
```
User:
  1. Zip dataset locally (5 seconds)
  2. Upload dataset.zip (30 seconds for 100MB)
  Total: 35 seconds

Backend:
  1. Validate zip contents (2 seconds)
  2. Upload to S3 (5 seconds)
  Total: 7 seconds

Training:
  1. Download zip (5 seconds)
  2. Extract (3 seconds)
  3. Train
  Total overhead: 8 seconds
```

**Total time to start training: ~50 seconds**

#### Individual File Approach
```
User:
  1. Frontend unzips (if needed) (5 seconds)
  2. Upload 100 images + annotations (2 minutes via multi-part)
  Total: ~2 minutes

Backend:
  1. Validate each file (10 seconds)
  2. Upload to S3 (30 seconds for 100 files)
  Total: 40 seconds

Training:
  1. List S3 files (1 second)
  2. Download 100 images (20 seconds)
  3. Train
  Total overhead: 21 seconds
```

**Total time to start training: ~3 minutes**

**Winner: Zip File Storage** (6x faster)

---

### Scenario 2: Using Platform Labeling Tool

**User wants to label images using our web-based labeling tool**

#### Zip Approach
```
❌ Not feasible without workarounds:

Option A: Download all images
  1. User uploads zip
  2. Backend extracts zip to S3 (individual files)
  3. Generate presigned URLs
  4. Labeling tool loads images
  Problem: Zip storage becomes individual storage

Option B: Generate URLs from zip
  1. Backend extracts zip to temp directory
  2. Generate presigned URLs for temp files
  Problem: Expensive, temp files expire

Conclusion: Zip storage incompatible with labeling tool
```

#### Individual File Approach
```
✅ Native support:

1. User uploads images individually (or via batch)
2. Backend stores as individual files
3. Labeling tool requests image list
4. Backend generates presigned URLs (1 hour expiry)
5. Frontend loads images directly from S3
6. User labels images
7. Frontend saves annotations to backend
8. Backend updates annotations.json

Performance:
- Image load: 200-500ms per image
- Annotation save: <100ms
- No backend relay needed
```

**Winner: Individual File Storage** (Only viable option)

---

### Scenario 3: Incremental Dataset Updates

**User wants to add 10 more images to existing dataset**

#### Zip Approach
```
❌ Inefficient:

1. User downloads current dataset.zip (100MB, 30 seconds)
2. User extracts locally
3. User adds 10 new images
4. User re-zips dataset (110MB)
5. User uploads new dataset.zip (110MB, 35 seconds)
6. Backend replaces old zip

Total: ~2 minutes + manual work
Cost: Full dataset re-upload (110MB)
```

#### Individual File Approach
```
✅ Efficient:

1. User uploads 10 new images
2. Backend uploads to S3 images/ folder
3. Backend updates annotations.json
   (download 500KB → modify → upload 500KB)

Total: ~10 seconds
Cost: Only new files (10MB) + annotation update (1MB)
```

**Winner: Individual File Storage** (12x faster, 11x less bandwidth)

---

### Scenario 4: Dataset Versioning

**User wants to create v2 of dataset after modifications**

#### Zip Approach
```
✅ Simple:

1. Backend copies dataset.zip to dataset-v2.zip
   S3 server-side copy: 1-2 seconds
2. No download/upload needed

Cost: 1 S3 copy operation
Storage: 2x dataset size
```

#### Individual File Approach
```
❌ Complex:

1. Backend copies all files:
   - annotations.json
   - meta.json
   - images/img001.jpg
   - images/img002.jpg
   - ... (N files)

   S3 server-side copy: 100+ operations, 10-30 seconds

Cost: N S3 copy operations
Storage: 2x dataset size

Alternative: Copy-on-write with annotations only
- Only copy annotations.json + meta.json
- Share images/ folder between versions
- Requires careful reference tracking
```

**Winner: Zip File Storage** (Much simpler)

---

### Scenario 5: Multi-Tier Consistency (Local Dev → Production)

**Developer testing locally, then deploying to production**

#### Zip Approach
```
✅ Consistent across all tiers:

Tier 1 (Subprocess):
  - MinIO stores: datasets/{id}/dataset.zip
  - Trainer downloads zip from MinIO
  - Extract → Train

Tier 2 (Kind):
  - MinIO stores: datasets/{id}/dataset.zip
  - Trainer downloads zip from MinIO
  - Extract → Train

Tier 3 (Production):
  - R2 stores: datasets/{id}/dataset.zip
  - Trainer downloads zip from R2
  - Extract → Train

Same code, same logic, only endpoint differs
```

#### Individual File Approach
```
✅ Also consistent:

Tier 1 (Subprocess):
  - MinIO stores: datasets/{id}/images/*.jpg
  - Trainer downloads files from MinIO

Tier 2 (Kind):
  - MinIO stores: datasets/{id}/images/*.jpg
  - Trainer downloads files from MinIO

Tier 3 (Production):
  - R2 stores: datasets/{id}/images/*.jpg
  - Trainer downloads files from R2

Same code, same logic, only endpoint differs
```

**Winner: Tie** (Both work consistently across tiers)

---

## Cost Analysis

### S3/R2 Storage Pricing (Cloudflare R2 as example)

- **Storage**: $0.015 per GB/month
- **Class A operations** (PUT, POST): $4.50 per million
- **Class B operations** (GET, LIST): Free (R2) or $0.40 per million (S3)

### Cost Comparison: 100 Images (100MB total)

#### Zip File Storage

**Upload:**
- 1 PUT operation: $0.0000045
- 100MB storage: $0.0015/month
- Total upload: **$0.0000045**

**Training (download):**
- 1 GET operation: $0 (R2 free)
- Total per training: **$0**

**Monthly cost** (1 upload, 10 trainings):
- Storage: $0.0015
- Operations: $0.0000045
- **Total: $0.0015/month**

#### Individual File Storage

**Upload:**
- 102 PUT operations (100 images + annotations + meta): $0.00046
- 100MB storage: $0.0015/month
- Total upload: **$0.00046**

**Training (download):**
- 102 GET operations: $0 (R2 free)
- Total per training: **$0**

**Incremental update** (add 1 image):
- 1 PUT (new image): $0.0000045
- 2 GET + 1 PUT (annotations): $0.0000045
- Total: **$0.000009**

**Monthly cost** (1 upload, 10 trainings, 5 updates):
- Storage: $0.0015
- Operations: $0.00046 + (5 × $0.000009) = $0.00051
- **Total: $0.0021/month**

#### Cost Summary (per dataset per month)

| Scenario | Zip | Individual | Savings |
|----------|-----|------------|---------|
| Upload only | $0.0015 | $0.0021 | Zip -29% |
| With 10 trainings | $0.0015 | $0.0021 | Zip -29% |
| With 10 updates | $0.0015 | $0.0021 | Zip -29% |

**Winner: Zip File Storage** (30% cheaper)

**Note**: Cost difference is minimal at small scale. At 10,000 datasets with frequent updates, individual file storage could cost 30% more (~$6/month vs $15/month).

---

## Implementation Complexity

### Zip File Storage

#### Backend Implementation
```python
# Upload API
@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile):
    # 1. Validate zip
    with zipfile.ZipFile(file.file) as z:
        validate_structure(z)

    # 2. Upload to S3
    await s3.upload_fileobj(
        file.file,
        bucket,
        f"datasets/{dataset_id}/dataset.zip"
    )

    # 3. Save metadata to DB
    dataset = Dataset(
        id=dataset_id,
        storage_path=f"datasets/{dataset_id}/dataset.zip"
    )
    db.add(dataset)

    return {"dataset_id": dataset_id}
```

**Complexity**: Low (50 lines)

#### Trainer Implementation
```python
# Training Service
def load_dataset(dataset_id: str):
    # 1. Download zip
    s3.download_file(
        bucket,
        f"datasets/{dataset_id}/dataset.zip",
        "/tmp/dataset.zip"
    )

    # 2. Extract
    with zipfile.ZipFile("/tmp/dataset.zip") as z:
        z.extractall("/workspace/dataset")

    # 3. Load annotations
    with open("/workspace/dataset/annotations.json") as f:
        annotations = json.load(f)

    return annotations
```

**Complexity**: Low (30 lines)

---

### Individual File Storage

#### Backend Implementation
```python
# Upload API
@router.post("/datasets/upload")
async def upload_dataset(files: List[UploadFile]):
    # 1. Parse structure
    file_tree = build_file_tree(files)

    # 2. Validate structure
    validate_structure(file_tree)

    # 3. Upload each file
    for file in files:
        await s3.upload_fileobj(
            file.file,
            bucket,
            f"datasets/{dataset_id}/{file.filename}"
        )

    # 4. Save metadata
    dataset = Dataset(
        id=dataset_id,
        storage_path=f"datasets/{dataset_id}/"
    )
    db.add(dataset)

    return {"dataset_id": dataset_id}

# Incremental update API
@router.post("/datasets/{id}/images")
async def add_images(dataset_id: str, files: List[UploadFile]):
    # Upload new images
    for file in files:
        await s3.upload_fileobj(
            file.file,
            bucket,
            f"datasets/{dataset_id}/images/{file.filename}"
        )

    # Update annotations (complex!)
    annotations = await s3.get_object(
        bucket,
        f"datasets/{dataset_id}/annotations.json"
    )
    # ... modify annotations
    await s3.put_object(
        bucket,
        f"datasets/{dataset_id}/annotations.json",
        json.dumps(annotations)
    )

# Presigned URL API (for labeling tool)
@router.get("/datasets/{id}/images/{filename}/url")
async def get_presigned_url(dataset_id: str, filename: str):
    url = s3.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': bucket,
            'Key': f'datasets/{dataset_id}/images/{filename}'
        },
        ExpiresIn=3600
    )
    return {"url": url}
```

**Complexity**: High (200+ lines with error handling)

#### Trainer Implementation
```python
# Training Service
def load_dataset(dataset_id: str):
    # 1. Download annotations
    s3.download_file(
        bucket,
        f"datasets/{dataset_id}/annotations.json",
        "/tmp/annotations.json"
    )

    with open("/tmp/annotations.json") as f:
        annotations = json.load(f)

    # 2. Download all images
    for img in annotations['images']:
        s3.download_file(
            bucket,
            f"datasets/{dataset_id}/images/{img['file_name']}",
            f"/workspace/dataset/images/{img['file_name']}"
        )

    return annotations

# Alternative: Download snapshot
def load_dataset_from_snapshot(dataset_id: str):
    # Backend creates snapshot on-demand
    snapshot_path = await backend.create_snapshot(dataset_id)

    # Download and extract
    s3.download_file(bucket, snapshot_path, "/tmp/snapshot.zip")
    with zipfile.ZipFile("/tmp/snapshot.zip") as z:
        z.extractall("/workspace/dataset")
```

**Complexity**: Medium-High (100+ lines)

---

## Recommendations

### Hybrid Approach (Recommended)

Combine both strategies based on use case:

```
Platform Storage Structure:
datasets/{dataset-id}/
├── dataset.zip                    ← For completed datasets (training)
├── annotations.json               ← For active labeling (mutable)
├── meta.json                      ← Metadata
└── images/                        ← For incremental updates (mutable)
    ├── img001.jpg
    └── ...
```

#### Workflow

**1. Complete Dataset Upload** (External labeling, ready to train)
```
User uploads zip → Backend stores as dataset.zip
Training uses: dataset.zip
```

**2. Incremental Dataset (Platform labeling tool)**
```
User uploads images individually → Stored in images/
User labels via web tool → Updates annotations.json
Training time: Backend creates snapshot.zip from images/ + annotations.json
Training uses: snapshot.zip
```

**3. Dataset Update**
```
If dataset.zip exists:
  - Download, modify, re-upload
  OR
  - Extract to images/, switch to incremental mode

If images/ exists:
  - Add/remove files directly
  - Update annotations.json
  - Create new snapshot.zip when needed
```

#### Storage Strategy per Dataset Type

| Dataset Type | Storage Method | Training Method |
|--------------|----------------|-----------------|
| **Static** (no updates expected) | `dataset.zip` | Direct zip download |
| **Active** (frequent labeling/updates) | `images/` + `annotations.json` | Snapshot on-demand |
| **Versioned** | `snapshots/v1.zip`, `snapshots/v2.zip` | Version-specific zip |

### Implementation Priority

**Phase 1: Zip-Only (Current MVP Focus)**
- [x] Upload zip file
- [x] Store as dataset.zip
- [x] Training downloads zip
- ✅ Simple, works for most use cases

**Phase 2: Add Individual File Support (When labeling tool needed)**
- [ ] Upload individual files
- [ ] Generate presigned URLs
- [ ] Labeling tool integration
- [ ] Snapshot creation on-demand

**Phase 3: Hybrid Optimization**
- [ ] Auto-detect upload type
- [ ] Smart storage selection
- [ ] Snapshot caching
- [ ] Version management

---

## Decision Criteria

Choose **Zip File Storage** if:
- ✅ Datasets are pre-labeled (no platform labeling tool)
- ✅ Datasets rarely change after upload
- ✅ Simplicity and speed are priorities
- ✅ Storage cost optimization matters
- ✅ MVP/early stage

Choose **Individual File Storage** if:
- ✅ Platform labeling tool is required
- ✅ Frequent incremental updates
- ✅ Need to access individual files
- ✅ Collaborative labeling workflows

Choose **Hybrid Approach** if:
- ✅ Both use cases needed
- ✅ Long-term flexibility
- ✅ Different user personas (some label, some bring complete datasets)

---

## References

- [PLATFORM_DATASET_FORMAT.md](../../docs/datasets/PLATFORM_DATASET_FORMAT.md)
- [STORAGE_ACCESS_PATTERNS.md](../../docs/datasets/STORAGE_ACCESS_PATTERNS.md)
- [DATASET_MANAGEMENT_DESIGN.md](../../docs/datasets/DATASET_MANAGEMENT_DESIGN.md)
- [3_TIER_DEVELOPMENT.md](../development/3_TIER_DEVELOPMENT.md)

---

**Last Updated**: 2025-01-10
**Decision Status**: ⏳ Pending (review this document and decide)
