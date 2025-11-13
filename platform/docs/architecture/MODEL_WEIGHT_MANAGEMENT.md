# Model Weight Management Design

Complete design for pretrained weights and checkpoint management in the Vision AI Training Platform.

## Table of Contents

- [Overview](#overview)
- [Dual Storage Strategy](#dual-storage-strategy)
- [Pretrained Weight Management](#pretrained-weight-management)
- [Checkpoint Management](#checkpoint-management)
- [Organization & Quota Management](#organization--quota-management)
- [Storage Structure](#storage-structure)
- [API Specifications](#api-specifications)
- [Implementation Guidelines](#implementation-guidelines)

## Overview

This document defines the complete strategy for managing model weights and checkpoints, including:

1. **Dual Storage**: Separate storage for datasets (large, user-owned) vs model weights (platform-managed)
2. **Pretrained Weights**: Public model caching, custom weight uploads, access control
3. **Checkpoints**: Training checkpoint storage, retention policies, cleanup strategies
4. **Organization**: Multi-tenant support with company/division-based quotas

**Key Principles**:
- Dataset storage: User-controlled, presigned URL access
- Model weights: Platform-controlled, S3-compatible storage
- Access control: DB-based permissions (not folder-based)
- Retention: Usage + time-based cleanup
- Organization: Company + Division mapping

## Dual Storage Strategy

### Storage Types

**Dataset Storage** (User-Controlled):
- Large image datasets (10GB - 1TB+)
- User can choose storage location
- Platform accesses via presigned URLs
- User pays storage costs

**Model Weight Storage** (Platform-Controlled):
- Pretrained weights (10MB - 500MB)
- Training checkpoints (5MB - 100MB)
- Platform manages in S3-compatible storage
- Platform pays storage costs

### Dataset Storage Flow

**Upload (Frontend → Backend → S3):**
```python
# Frontend
const formData = new FormData();
formData.append('file', datasetZipFile);
formData.append('name', 'my-dataset');

await fetch('/api/v1/datasets/upload', {
  method: 'POST',
  body: formData
});

# Backend
@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    # 1. Create dataset record
    dataset = Dataset(
        user_id=current_user.id,
        name=name,
        status=DatasetStatus.UPLOADING
    )
    db.add(dataset)
    await db.commit()

    # 2. Upload to S3 with multipart support
    s3_key = f"datasets/{dataset.id}/dataset.zip"
    s3_client.upload_fileobj(
        file.file,
        bucket=BUCKET_NAME,
        key=s3_key,
        Config=TransferConfig(
            multipart_threshold=100 * 1024 * 1024,  # 100MB
            max_concurrency=10,
            use_threads=True
        )
    )

    # 3. Update dataset status
    dataset.storage_path = f"s3://{BUCKET_NAME}/{s3_key}"
    dataset.status = DatasetStatus.READY
    await db.commit()

    return dataset
```

**Download (Trainer → Presigned URL → S3):**
```python
# Trainer (train.py)
def download_dataset_snapshot(dataset_id: str, snapshot_id: str):
    """Download dataset using presigned URLs"""

    # 1. Request presigned URLs from backend
    response = requests.get(
        f"{BACKEND_BASE_URL}/api/v1/datasets/{dataset_id}/snapshots/{snapshot_id}/download-urls",
        headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"}
    )

    snapshot_data = response.json()

    # Check snapshot status
    if snapshot_data["status"] == "broken":
        raise RuntimeError(
            f"Snapshot is broken. Missing {len(snapshot_data['missing_images'])} images. "
            f"Please use a different snapshot or re-upload the dataset."
        )

    # 2. Download snapshot metadata
    annotations = snapshot_data["annotations"]

    # 3. Download all images using presigned URLs
    os.makedirs("/workspace/dataset/images", exist_ok=True)

    for image_info in snapshot_data["image_urls"]:
        filename = image_info["filename"]
        presigned_url = image_info["url"]

        print(f"[Dataset] Downloading {filename}...")
        response = requests.get(presigned_url, stream=True)

        local_path = f"/workspace/dataset/images/{filename}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    # 4. Save annotations
    with open("/workspace/dataset/annotations.json", "w") as f:
        json.dump(annotations, f)

    return "/workspace/dataset"

# Backend API
@router.get("/datasets/{dataset_id}/snapshots/{snapshot_id}/download-urls")
async def get_snapshot_download_urls(
    dataset_id: UUID,
    snapshot_id: str,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """Generate presigned URLs for all images in snapshot"""

    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Load snapshot from S3
    snapshot_key = f"datasets/{dataset_id}/snapshots/{snapshot_id}.json"
    snapshot_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=snapshot_key)
    snapshot = json.loads(snapshot_obj['Body'].read())

    # Check integrity
    if snapshot.get("status") == "broken":
        return {
            "status": "broken",
            "missing_images": snapshot["integrity"]["missing_images"],
            "message": "This snapshot has missing images and cannot be used for training"
        }

    # Generate presigned URLs for each image
    image_urls = []
    for image in snapshot["annotations"]["images"]:
        filename = image["file_name"]
        image_key = f"datasets/{dataset_id}/images/{filename}"

        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': image_key},
            ExpiresIn=3600  # 1 hour
        )

        image_urls.append({
            "filename": filename,
            "url": presigned_url
        })

    return {
        "status": "valid",
        "annotations": snapshot["annotations"],
        "image_urls": image_urls
    }
```

**Why Presigned URLs?**
- ✅ Backend doesn't proxy large files (reduced load)
- ✅ Trainer downloads directly from S3 (faster)
- ✅ Temporary URLs (security - expires in 1 hour)
- ✅ No permanent credentials in trainer

## Pretrained Weight Management

### Naming Convention

Format: `{base_model}_{variant}_{pretraining_dataset}`

**Examples:**
```
yolo11n_det_coco              # YOLO11n for detection, pretrained on COCO
yolo11n_seg_coco              # YOLO11n for segmentation, pretrained on COCO
yolo11n_seg_ade20k            # YOLO11n for segmentation, pretrained on ADE20K
yolo11s_det_objects365        # YOLO11s for detection, pretrained on Objects365
resnet50_cls_imagenet         # ResNet-50 for classification, pretrained on ImageNet
resnet50_cls_imagenet22k      # ResNet-50 for classification, pretrained on ImageNet-22K
custom_detector_internal_v2   # Custom model, internal dataset, version 2
```

### Storage Structure

```
s3://vision-platform/pretrained-weights/
├── timm/
│   ├── resnet50_cls_imagenet.pth
│   ├── resnet50_cls_imagenet22k.pth
│   ├── efficientnet_b0_cls_imagenet.pth
│   └── vit_base_patch16_224_cls_imagenet.pth
│
├── ultralytics/
│   ├── yolo11n_det_coco.pt
│   ├── yolo11n_seg_coco.pt
│   ├── yolo11n_seg_ade20k.pt
│   ├── yolo11s_det_coco.pt
│   └── yolo_world_v2_s_det_objects365.pt
│
├── huggingface/
│   ├── clip_vit_base_patch32.bin
│   ├── dino_vits16.bin
│   └── vit_base_patch16_224.bin
│
└── custom/
    ├── company_detector_coco_v1.pt
    ├── company_segmenter_internal_v3.pt
    └── user_experiment_yolo_v2.pt
```

**Folder Structure:**
- **Framework-based only**: `timm/`, `ultralytics/`, `huggingface/`, `custom/`
- **No visibility folders**: No `public/`, `private/`, `organization/` separation
- **Permissions in DB**: Access control via `PretrainedWeight` model

### Database Model

```python
# app/models/pretrained_weight.py
from sqlalchemy import Column, String, Integer, BigInteger, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class PretrainedWeight(Base):
    """Registry of pretrained model weights"""
    __tablename__ = "pretrained_weights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Naming
    framework = Column(String(100), nullable=False, index=True)  # timm, ultralytics, huggingface, custom
    model_name = Column(String(255), nullable=False, unique=True, index=True)  # yolo11n_seg_coco

    # Metadata
    base_architecture = Column(String(100))  # yolo11n, resnet50, clip
    task_types = Column(JSON, nullable=False)  # ["object_detection"], ["image_classification"]
    pretraining_dataset = Column(String(255))  # COCO, ImageNet, ADE20K, Objects365
    description = Column(String(1000))

    # Model info
    num_parameters = Column(String(50))  # "11.2M", "25.6M"
    input_size = Column(JSON)  # [640, 640] or [224, 224]
    supported_formats = Column(JSON)  # ["yolo", "coco"]

    # Storage
    storage_path = Column(String(500), nullable=False)  # s3://bucket/pretrained-weights/ultralytics/yolo11n_seg_coco.pt
    size_bytes = Column(BigInteger, nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA256

    # Access control (DB-based, not folder-based)
    visibility = Column(String(50), default="private", nullable=False, index=True)  # public, private, organization
    owner_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)

    # Usage tracking
    download_count = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    owner = relationship("User", foreign_keys=[owner_user_id])
    organization = relationship("Organization", back_populates="pretrained_weights")

    # Indexes
    __table_args__ = (
        Index('idx_framework_visibility', 'framework', 'visibility'),
    )
```

### Public Weight Caching

**Problem**: Each trainer downloading weights from external URLs (HuggingFace, Ultralytics) is slow and expensive.

**Solution**: Cache public weights in platform storage on first use.

```python
# app/services/pretrained_weight_service.py
from pathlib import Path
import hashlib
import requests
from sqlalchemy.ext.asyncio import AsyncSession

class PretrainedWeightService:
    """Service for managing pretrained weights"""

    OFFICIAL_URLS = {
        "ultralytics": {
            "yolo11n_det_coco": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            "yolo11n_seg_coco": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
            # ... more models
        },
        "timm": {
            "resnet50_cls_imagenet": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            # ... more models
        }
    }

    async def get_or_cache_weight(
        self,
        framework: str,
        model_name: str,
        db: AsyncSession
    ) -> str:
        """
        Get pretrained weight path. Download and cache if not exists.

        Returns:
            S3 path to weight file
        """
        # 1. Check if already in platform storage
        weight = await db.query(PretrainedWeight).filter_by(
            framework=framework,
            model_name=model_name
        ).first()

        if weight:
            # Update usage tracking
            weight.download_count += 1
            weight.last_used_at = datetime.utcnow()
            await db.commit()
            return weight.storage_path

        # 2. Not cached - download from official source
        official_url = self.OFFICIAL_URLS.get(framework, {}).get(model_name)
        if not official_url:
            raise ValueError(f"Unknown model: {framework}/{model_name}")

        print(f"[PretrainedWeight] Downloading {model_name} from official source (first use)")

        # Download to temp file
        temp_path = f"/tmp/{model_name}"
        response = requests.get(official_url, stream=True)
        response.raise_for_status()

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Calculate checksum
        checksum = self._calculate_sha256(temp_path)
        file_size = Path(temp_path).stat().st_size

        # 3. Upload to platform storage
        s3_key = f"pretrained-weights/{framework}/{model_name}.pt"
        s3_client.upload_file(temp_path, BUCKET_NAME, s3_key)
        storage_path = f"s3://{BUCKET_NAME}/{s3_key}"

        # 4. Register in database
        weight = PretrainedWeight(
            framework=framework,
            model_name=model_name,
            storage_path=storage_path,
            size_bytes=file_size,
            checksum=checksum,
            visibility="public",  # Official weights are public
            download_count=1,
            last_used_at=datetime.utcnow()
        )
        db.add(weight)
        await db.commit()

        print(f"[PretrainedWeight] Cached {model_name} in platform storage")

        return storage_path

    def _calculate_sha256(self, filepath: str) -> str:
        """Calculate SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
```

### Custom Weight Upload

**Use Cases:**
1. Model developer uploads custom pretrained weights
2. Organization shares fine-tuned weights internally
3. User uploads experiment checkpoint as pretrained weight

**Upload API:**
```python
# app/api/v1/models.py
from fastapi import APIRouter, UploadFile, File, Form, Depends

router = APIRouter(prefix="/models", tags=["models"])

@router.post("/pretrained-weights", response_model=PretrainedWeightResponse)
async def upload_pretrained_weight(
    file: UploadFile = File(...),
    framework: str = Form(...),
    model_name: str = Form(...),
    base_architecture: str = Form(...),
    task_types: str = Form(...),  # JSON string: ["object_detection"]
    pretraining_dataset: str = Form(None),
    description: str = Form(None),
    visibility: str = Form("private"),  # public, private, organization
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload custom pretrained weight

    Access control:
    - private: Only owner can use
    - organization: All users in owner's organization can use
    - public: All users can use (requires admin approval)
    """

    # Validate
    if visibility not in ["private", "organization", "public"]:
        raise HTTPException(status_code=400, detail="Invalid visibility")

    if visibility == "public" and not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Only admins can create public pretrained weights"
        )

    # Check name uniqueness
    existing = await db.query(PretrainedWeight).filter_by(
        model_name=model_name
    ).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Model name '{model_name}' already exists"
        )

    # Upload to S3
    file_content = await file.read()
    file_size = len(file_content)

    # Calculate checksum
    checksum = hashlib.sha256(file_content).hexdigest()

    # Determine S3 path based on framework
    s3_key = f"pretrained-weights/{framework}/{model_name}.pt"

    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=file_content
    )

    storage_path = f"s3://{BUCKET_NAME}/{s3_key}"

    # Create database record
    weight = PretrainedWeight(
        framework=framework,
        model_name=model_name,
        base_architecture=base_architecture,
        task_types=json.loads(task_types),
        pretraining_dataset=pretraining_dataset,
        description=description,
        storage_path=storage_path,
        size_bytes=file_size,
        checksum=checksum,
        visibility=visibility,
        owner_user_id=current_user.id,
        organization_id=current_user.organization_id if visibility == "organization" else None
    )

    db.add(weight)
    await db.commit()
    await db.refresh(weight)

    return weight
```

### Access Control

**Permission Check:**
```python
# app/services/pretrained_weight_service.py
async def check_access(
    weight: PretrainedWeight,
    user: User
) -> bool:
    """Check if user can access pretrained weight"""

    # Public weights: everyone can access
    if weight.visibility == "public":
        return True

    # Private weights: only owner can access
    if weight.visibility == "private":
        return weight.owner_user_id == user.id

    # Organization weights: users in same organization can access
    if weight.visibility == "organization":
        return (
            user.organization_id is not None and
            user.organization_id == weight.organization_id
        )

    return False

# Usage in trainer
async def get_pretrained_weight_for_training(
    model_name: str,
    user_id: UUID,
    db: AsyncSession
) -> str:
    """Get pretrained weight path for training job"""

    weight = await db.query(PretrainedWeight).filter_by(
        model_name=model_name
    ).first()

    if not weight:
        raise ValueError(f"Pretrained weight not found: {model_name}")

    user = await db.get(User, user_id)

    if not await check_access(weight, user):
        raise PermissionError(
            f"You don't have access to pretrained weight: {model_name}"
        )

    # Generate presigned URL for download
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': weight.storage_path.replace(f"s3://{BUCKET_NAME}/", "")},
        ExpiresIn=3600
    )

    return presigned_url
```

## Checkpoint Management

### Storage Strategy: Hybrid (S3 + MLflow)

**Approach:**
1. **Primary storage**: S3 (fast, cheap, scalable)
2. **Metadata**: MLflow (experiment tracking, lineage)
3. **Quick access**: Database (for UI queries)

**Why Hybrid?**
- ✅ S3: Direct access for inference/fine-tuning
- ✅ MLflow: Experiment tracking integration
- ✅ Database: Fast queries without S3 API calls

### Checkpoint Types

**Best Checkpoint**: Model with best validation metric
**Last Checkpoint**: Most recent epoch (for resume training)
**Intermediate Checkpoints** (optional): Periodic saves (e.g., every 10 epochs)

### Storage Structure

```
s3://vision-platform/checkpoints/
├── jobs/
│   ├── {job-id-1}/
│   │   ├── best.pt          # Best validation performance
│   │   ├── last.pt          # Most recent epoch
│   │   ├── epoch-10.pt      # Optional intermediate
│   │   ├── epoch-20.pt
│   │   └── metadata.json    # Job metadata
│   └── {job-id-2}/
│       ├── best.pt
│       └── last.pt
```

### Checkpoint Save Implementation

```python
# platform/trainers/common/checkpoint_manager.py
import os
import json
from pathlib import Path
from typing import Optional, Dict
import mlflow

class CheckpointManager:
    """Unified checkpoint management for all trainers"""

    def __init__(self, job_id: str, project_id: Optional[str] = None):
        self.job_id = job_id
        self.project_id = project_id
        self.s3_client = get_s3_client()
        self.bucket = os.environ.get("BUCKET_NAME", "vision-platform")

    def save_checkpoint(
        self,
        epoch: int,
        checkpoint_type: str,  # "best", "last", "intermediate"
        local_path: str,
        metrics: Dict[str, float]
    ) -> str:
        """
        Save checkpoint using hybrid approach

        Returns:
            S3 path to checkpoint
        """
        # 1. Upload to S3
        checkpoint_name = f"{checkpoint_type}.pt" if checkpoint_type in ["best", "last"] else f"epoch-{epoch}.pt"
        s3_key = f"checkpoints/jobs/{self.job_id}/{checkpoint_name}"

        print(f"[Checkpoint] Uploading {checkpoint_name} to S3...")
        self.s3_client.upload_file(local_path, self.bucket, s3_key)

        s3_path = f"s3://{self.bucket}/{s3_key}"
        print(f"[Checkpoint] Uploaded: {s3_path}")

        # 2. Log to MLflow
        mlflow.log_param(f"checkpoint_{checkpoint_type}_path", s3_path)
        mlflow.log_param(f"checkpoint_{checkpoint_type}_epoch", epoch)

        # Log metrics with checkpoint
        for key, value in metrics.items():
            mlflow.log_metric(f"checkpoint_{checkpoint_type}_{key}", value, step=epoch)

        # Optionally log small checkpoints as MLflow artifacts (< 50MB)
        file_size = Path(local_path).stat().st_size
        if file_size < 50 * 1024 * 1024:
            mlflow.log_artifact(local_path, artifact_path="checkpoints")

        # 3. Update database
        self._update_database(epoch, checkpoint_type, s3_path, metrics)

        return s3_path

    def _update_database(
        self,
        epoch: int,
        checkpoint_type: str,
        s3_path: str,
        metrics: Dict[str, float]
    ):
        """Update database with checkpoint info"""
        import sqlite3

        db_path = Path(__file__).parent.parent.parent / "data" / "db" / "vision_platform.db"

        if not db_path.exists():
            print(f"[Checkpoint WARNING] Database not found: {db_path}")
            return

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Update ValidationResult
            cursor.execute("""
                UPDATE validation_results
                SET checkpoint_path = ?, checkpoint_type = ?
                WHERE job_id = ? AND epoch = ?
            """, (s3_path, checkpoint_type, self.job_id, epoch))

            # Also update TrainingJob if this is best checkpoint
            if checkpoint_type == "best":
                cursor.execute("""
                    UPDATE training_jobs
                    SET checkpoint_path = ?, best_epoch = ?
                    WHERE id = ?
                """, (s3_path, epoch, self.job_id))

            conn.commit()
            conn.close()

            print(f"[Checkpoint] Database updated for epoch {epoch}")

        except Exception as e:
            print(f"[Checkpoint WARNING] Failed to update database: {e}")

    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        Cleanup old intermediate checkpoints, keep only:
        - best.pt
        - last.pt
        - Last N intermediate checkpoints
        """
        try:
            # List all checkpoints for this job
            prefix = f"checkpoints/jobs/{self.job_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return

            checkpoints = []
            for obj in response['Contents']:
                key = obj['Key']
                filename = Path(key).name

                # Keep best and last
                if filename in ["best.pt", "last.pt", "metadata.json"]:
                    continue

                # Extract epoch number from intermediate checkpoints
                if filename.startswith("epoch-"):
                    epoch_num = int(filename.replace("epoch-", "").replace(".pt", ""))
                    checkpoints.append((epoch_num, key))

            # Sort by epoch (descending)
            checkpoints.sort(reverse=True)

            # Delete old intermediate checkpoints
            for epoch_num, key in checkpoints[keep_last_n:]:
                print(f"[Checkpoint] Deleting old checkpoint: {key}")
                self.s3_client.delete_object(Bucket=self.bucket, Key=key)

        except Exception as e:
            print(f"[Checkpoint WARNING] Failed to cleanup old checkpoints: {e}")
```

### Database Schema Extensions

```python
# app/models/training_job.py
class TrainingJob(Base):
    # ... existing fields ...

    # Checkpoint tracking
    checkpoint_path = Column(String(500), nullable=True)  # S3 path to best checkpoint
    best_epoch = Column(Integer, nullable=True)
    last_epoch = Column(Integer, nullable=True)

    # Checkpoint metadata
    checkpoint_count = Column(Integer, default=0)  # Total checkpoints saved
    checkpoint_storage_bytes = Column(BigInteger, default=0)  # Total checkpoint size

# app/models/validation_result.py
class ValidationResult(Base):
    __tablename__ = "validation_results"

    # ... existing fields ...

    # Checkpoint info
    checkpoint_path = Column(String(500), nullable=True)  # S3 path if checkpoint saved
    checkpoint_type = Column(String(50), nullable=True)  # "best", "last", "intermediate"
    checkpoint_size_bytes = Column(BigInteger, nullable=True)
```

### Checkpoint Retention Policy

**Policy: Usage + Time Based**

**Rules:**
1. **Best checkpoint**: Keep for 90 days
2. **Last checkpoint**: Keep for 30 days
3. **Intermediate checkpoints**: Keep for 7 days

**Protection from deletion:**
- Currently used for inference (within last 30 days)
- Used as base for fine-tuning
- User marked as "important"

**Implementation:**
```python
# app/services/checkpoint_cleanup_service.py
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

class CheckpointCleanupService:
    """Service for cleaning up expired checkpoints"""

    RETENTION_DAYS = {
        "best": 90,
        "last": 30,
        "intermediate": 7
    }

    def __init__(self, db: AsyncSession):
        self.db = db
        self.s3_client = get_s3_client()

    async def cleanup_expired_checkpoints(self):
        """
        Run cleanup job to delete expired checkpoints

        Should be run daily via cron or Temporal scheduled workflow
        """
        now = datetime.utcnow()
        deleted_count = 0
        freed_bytes = 0

        # Find all checkpoints with paths
        results = await self.db.execute(
            select(ValidationResult).filter(
                ValidationResult.checkpoint_path.isnot(None)
            )
        )
        checkpoints = results.scalars().all()

        for checkpoint in checkpoints:
            # Check if protected
            if await self._is_protected(checkpoint):
                continue

            # Check retention period
            retention_days = self._get_retention_days(checkpoint)
            age_days = (now - checkpoint.created_at).days

            if age_days > retention_days:
                # Delete checkpoint
                success = await self._delete_checkpoint(checkpoint)
                if success:
                    deleted_count += 1
                    freed_bytes += checkpoint.checkpoint_size_bytes or 0

        print(f"[Cleanup] Deleted {deleted_count} checkpoints, freed {freed_bytes / (1024**3):.2f} GB")

    async def _is_protected(self, checkpoint: ValidationResult) -> bool:
        """Check if checkpoint is protected from deletion"""

        # 1. Check if user marked as important
        if checkpoint.is_important:
            return True

        # 2. Check if used for inference recently (last 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        inference_result = await self.db.execute(
            select(Inference).filter(
                Inference.checkpoint_path == checkpoint.checkpoint_path,
                Inference.created_at > cutoff_date
            ).limit(1)
        )
        if inference_result.scalar_one_or_none():
            return True

        # 3. Check if used as base for fine-tuning
        finetuning_result = await self.db.execute(
            select(TrainingJob).filter(
                TrainingJob.pretrained_checkpoint_path == checkpoint.checkpoint_path
            ).limit(1)
        )
        if finetuning_result.scalar_one_or_none():
            return True

        return False

    def _get_retention_days(self, checkpoint: ValidationResult) -> int:
        """Get retention period for checkpoint based on type"""
        checkpoint_type = checkpoint.checkpoint_type or "intermediate"
        return self.RETENTION_DAYS.get(checkpoint_type, self.RETENTION_DAYS["intermediate"])

    async def _delete_checkpoint(self, checkpoint: ValidationResult) -> bool:
        """Delete checkpoint from S3 and database"""
        try:
            # Parse S3 path
            s3_path = checkpoint.checkpoint_path  # s3://bucket/key
            bucket, key = s3_path.replace("s3://", "").split("/", 1)

            # Delete from S3
            self.s3_client.delete_object(Bucket=bucket, Key=key)

            # Update database
            checkpoint.checkpoint_path = None
            checkpoint.checkpoint_type = None
            await self.db.commit()

            print(f"[Cleanup] Deleted checkpoint: {key}")
            return True

        except Exception as e:
            print(f"[Cleanup ERROR] Failed to delete checkpoint {checkpoint.id}: {e}")
            return False
```

**Manual Protection:**
```python
# API endpoint for user to mark checkpoint as important
@router.post("/validation-results/{result_id}/protect")
async def protect_checkpoint(
    result_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Mark checkpoint as important (protected from auto-deletion)"""

    result = await db.get(ValidationResult, result_id)
    if not result:
        raise HTTPException(status_code=404)

    # Check ownership
    job = await db.get(TrainingJob, result.job_id)
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403)

    result.is_important = True
    await db.commit()

    return {"message": "Checkpoint protected from deletion"}
```

## Organization & Quota Management

### Organization Model

```python
# app/models/organization.py
from sqlalchemy import Column, String, Integer, BigInteger, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class Organization(Base):
    """Organization for multi-tenant support"""
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Company + Division
    company = Column(String(255), nullable=False, index=True)
    division = Column(String(255), nullable=True, index=True)

    # Display
    display_name = Column(String(500))  # "ABC Corporation - AI Research"

    # Storage quotas (in GB)
    checkpoint_storage_quota_gb = Column(Integer, default=500)
    pretrained_weight_storage_quota_gb = Column(Integer, default=100)

    # Usage tracking (in bytes)
    checkpoint_storage_used_bytes = Column(BigInteger, default=0)
    pretrained_weight_storage_used_bytes = Column(BigInteger, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="organization")
    pretrained_weights = relationship("PretrainedWeight", back_populates="organization")

    # Unique constraint on company + division
    __table_args__ = (
        UniqueConstraint('company', 'division', name='uq_company_division'),
    )

    @property
    def checkpoint_storage_used_gb(self) -> float:
        """Get checkpoint storage used in GB"""
        return self.checkpoint_storage_used_bytes / (1024 ** 3)

    @property
    def checkpoint_storage_remaining_gb(self) -> float:
        """Get remaining checkpoint storage in GB"""
        return self.checkpoint_storage_quota_gb - self.checkpoint_storage_used_gb

    @property
    def is_checkpoint_quota_exceeded(self) -> bool:
        """Check if checkpoint storage quota exceeded"""
        return self.checkpoint_storage_used_gb >= self.checkpoint_storage_quota_gb
```

### User Model Extension

```python
# app/models/user.py
class User(Base):
    # ... existing fields ...

    # Organization info
    company = Column(String(255), nullable=True)
    division = Column(String(255), nullable=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True, index=True)

    # Relationships
    organization = relationship("Organization", back_populates="users")
```

### Organization Auto-Creation

```python
# app/services/auth_service.py
async def register_user(
    email: str,
    password: str,
    full_name: str,
    company: str,
    division: Optional[str],
    db: AsyncSession
) -> User:
    """Register new user and auto-create/join organization"""

    # 1. Find or create organization
    org = await db.execute(
        select(Organization).filter_by(
            company=company,
            division=division
        )
    )
    organization = org.scalar_one_or_none()

    if not organization:
        # Create new organization
        display_name = f"{company}"
        if division:
            display_name += f" - {division}"

        organization = Organization(
            company=company,
            division=division,
            display_name=display_name
        )
        db.add(organization)
        await db.flush()

        print(f"[Auth] Created new organization: {display_name}")

    # 2. Create user
    user = User(
        email=email,
        hashed_password=get_password_hash(password),
        full_name=full_name,
        company=company,
        division=division,
        organization_id=organization.id
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user
```

### Quota Enforcement

```python
# app/services/quota_service.py
class QuotaService:
    """Service for managing storage quotas"""

    async def check_checkpoint_quota(
        self,
        user: User,
        checkpoint_size_bytes: int,
        db: AsyncSession
    ) -> bool:
        """Check if saving checkpoint would exceed quota"""

        if not user.organization_id:
            # No organization = no quota enforcement
            return True

        org = await db.get(Organization, user.organization_id)

        quota_bytes = org.checkpoint_storage_quota_gb * (1024 ** 3)
        used_bytes = org.checkpoint_storage_used_bytes

        if used_bytes + checkpoint_size_bytes > quota_bytes:
            return False

        return True

    async def track_checkpoint_usage(
        self,
        user: User,
        checkpoint_size_bytes: int,
        db: AsyncSession
    ):
        """Track checkpoint storage usage"""

        if not user.organization_id:
            return

        org = await db.get(Organization, user.organization_id)
        org.checkpoint_storage_used_bytes += checkpoint_size_bytes
        await db.commit()

    async def track_checkpoint_deletion(
        self,
        user: User,
        checkpoint_size_bytes: int,
        db: AsyncSession
    ):
        """Update usage when checkpoint is deleted"""

        if not user.organization_id:
            return

        org = await db.get(Organization, user.organization_id)
        org.checkpoint_storage_used_bytes -= checkpoint_size_bytes
        org.checkpoint_storage_used_bytes = max(0, org.checkpoint_storage_used_bytes)
        await db.commit()
```

## Storage Structure

### Complete S3 Bucket Layout

```
s3://vision-platform/
├── datasets/                          # User datasets (large)
│   ├── {dataset-id-1}/
│   │   ├── images/
│   │   │   ├── img001.jpg
│   │   │   └── img002.jpg
│   │   ├── annotations.json          # HEAD version
│   │   └── snapshots/
│   │       ├── training-job-123.json  # Training snapshot
│   │       └── v1.json                # Manual version
│   └── {dataset-id-2}/
│       └── ...
│
├── pretrained-weights/                # Model weights (platform-managed)
│   ├── timm/
│   │   ├── resnet50_cls_imagenet.pth
│   │   └── efficientnet_b0_cls_imagenet.pth
│   ├── ultralytics/
│   │   ├── yolo11n_det_coco.pt
│   │   └── yolo11n_seg_coco.pt
│   ├── huggingface/
│   │   └── clip_vit_base_patch32.bin
│   └── custom/
│       ├── my_custom_model.pt
│       └── company_detector.pt
│
└── checkpoints/                       # Training checkpoints
    └── jobs/
        ├── {job-id-1}/
        │   ├── best.pt                # Best validation metric
        │   ├── last.pt                # Most recent epoch
        │   ├── epoch-10.pt            # Optional intermediate
        │   └── metadata.json
        └── {job-id-2}/
            ├── best.pt
            └── last.pt
```

## API Specifications

### Pretrained Weight APIs

**List Pretrained Weights:**
```http
GET /api/v1/models/pretrained-weights
Query params:
  - framework: string (optional)
  - visibility: string (optional)
  - task_type: string (optional)

Response:
{
  "items": [
    {
      "id": "uuid",
      "framework": "ultralytics",
      "model_name": "yolo11n_seg_coco",
      "task_types": ["instance_segmentation"],
      "visibility": "public",
      "size_bytes": 12345678,
      "download_count": 42
    }
  ],
  "total": 100
}
```

**Upload Custom Weight:**
```http
POST /api/v1/models/pretrained-weights
Content-Type: multipart/form-data

Form fields:
  - file: binary
  - framework: string
  - model_name: string
  - base_architecture: string
  - task_types: string (JSON array)
  - visibility: string (private|organization|public)

Response:
{
  "id": "uuid",
  "model_name": "my_custom_model",
  "storage_path": "s3://bucket/pretrained-weights/custom/my_custom_model.pt"
}
```

**Get Download URL:**
```http
GET /api/v1/models/pretrained-weights/{model_name}/download-url

Response:
{
  "download_url": "https://s3.../presigned-url",
  "expires_in": 3600
}
```

### Checkpoint APIs

**List Job Checkpoints:**
```http
GET /api/v1/training/jobs/{job_id}/checkpoints

Response:
{
  "checkpoints": [
    {
      "epoch": 45,
      "checkpoint_type": "best",
      "checkpoint_path": "s3://bucket/checkpoints/jobs/123/best.pt",
      "metrics": {
        "loss": 0.234,
        "mAP50": 0.856
      },
      "size_bytes": 23456789,
      "created_at": "2025-01-10T12:00:00Z",
      "is_protected": false
    }
  ]
}
```

**Protect Checkpoint:**
```http
POST /api/v1/validation-results/{result_id}/protect

Response:
{
  "message": "Checkpoint protected from deletion"
}
```

**Get Checkpoint Download URL:**
```http
GET /api/v1/validation-results/{result_id}/checkpoint/download-url

Response:
{
  "download_url": "https://s3.../presigned-url",
  "expires_in": 3600
}
```

### Organization APIs

**Get Organization Quota:**
```http
GET /api/v1/organizations/me/quota

Response:
{
  "organization": {
    "company": "ABC Corporation",
    "division": "AI Research"
  },
  "checkpoint_storage": {
    "quota_gb": 500,
    "used_gb": 234.5,
    "remaining_gb": 265.5,
    "usage_percent": 46.9
  },
  "pretrained_weight_storage": {
    "quota_gb": 100,
    "used_gb": 12.3,
    "remaining_gb": 87.7,
    "usage_percent": 12.3
  }
}
```

## Implementation Guidelines

### Phase 1: Basic Infrastructure (Week 1-2)

1. **Database migrations**
   - Create `pretrained_weights` table
   - Create `organizations` table
   - Add organization fields to `users`
   - Add checkpoint fields to `validation_results`

2. **S3 bucket setup**
   - Create folder structure
   - Set up bucket policies

3. **Basic APIs**
   - Pretrained weight upload/list
   - Checkpoint download URL
   - Organization quota check

### Phase 2: Integration (Week 3-4)

1. **Trainer integration**
   - Modify trainers to use presigned URLs
   - Implement CheckpointManager in all trainers
   - Add quota checks before checkpoint save

2. **Public weight caching**
   - Implement PretrainedWeightService
   - Cache YOLO, timm, HF weights on first use

3. **Frontend updates**
   - Pretrained weight upload UI
   - Checkpoint browser
   - Organization quota display

### Phase 3: Advanced Features (Week 5-6)

1. **Retention policy**
   - Implement CheckpointCleanupService
   - Set up daily cleanup job (Temporal scheduled workflow)
   - Add checkpoint protection API

2. **Fine-tuning support**
   - API to start training from existing checkpoint
   - Track checkpoint lineage in database

3. **Monitoring & alerts**
   - Storage usage metrics
   - Quota exceeded alerts
   - Cleanup job monitoring

## References

- [Dataset Storage Strategy](./DATASET_STORAGE_STRATEGY.md)
- [Trainer Design](./TRAINER_DESIGN.md)
- [Backend Design](./BACKEND_DESIGN.md)
- [3-Tier Development](../development/3_TIER_DEVELOPMENT.md)
- [Checkpoint Management (MVP)](../../../docs/training/20251105_checkpoint_management_and_r2_upload_policy.md)
