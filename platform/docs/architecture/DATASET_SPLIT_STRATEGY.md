# Dataset Train/Validation Split Strategy

**Date**: 2025-01-10
**Status**: Production Design

Complete strategy for managing train/validation splits across different use cases and frameworks.

## Table of Contents

- [Overview](#overview)
- [3-Level Priority System](#3-level-priority-system)
- [Dataset-Level Split](#dataset-level-split)
- [Job-Level Split Override](#job-level-split-override)
- [Runtime Auto-Split](#runtime-auto-split)
- [Framework Implementation](#framework-implementation)
- [Scenarios](#scenarios)

---

## Overview

### The Problem

Users need flexible control over train/validation splits:

- **Researcher A**: Wants 80/20 split with seed=42 for reproducibility
- **Researcher B**: Wants 70/30 split on the same dataset
- **Production team**: Wants fixed split defined at dataset level
- **New user**: Just wants it to work (auto-split)

### The Solution: 3-Level Priority

```
Priority 1: Job-Level Override     (Highest - per-training flexibility)
    ↓ (if not specified)
Priority 2: Dataset-Level Metadata (Middle - shared split definition)
    ↓ (if not specified)
Priority 3: Runtime Auto-Split     (Lowest - default 80/20)
```

**Benefits**:
- ✅ Flexibility: Each job can customize split
- ✅ Consistency: Datasets can define default split
- ✅ Simplicity: Auto-split for beginners
- ✅ Reproducibility: Seed-based deterministic splits

---

## 3-Level Priority System

### Architecture

```
Training Job Created
       ↓
[1] Check job.split_strategy
       ↓ (exists?)
    YES → Use job-level split ✅
       ↓ (not specified)
    NO → [2] Check dataset.split_config
       ↓ (exists?)
    YES → Use dataset-level split ✅
       ↓ (not specified)
    NO → [3] Runtime auto-split (80/20) ✅
```

### Decision Logic

```python
# Backend: training_service.py
def get_split_strategy(job: TrainingJob, dataset: Dataset) -> SplitInfo:
    """Determine which split strategy to use"""

    # Priority 1: Job-level override
    if job.split_strategy:
        return apply_job_split(job.split_strategy)

    # Priority 2: Dataset-level metadata
    if dataset.split_config:
        return apply_dataset_split(dataset.split_config)

    # Priority 3: Runtime auto-split (default)
    return apply_auto_split(ratio=[0.8, 0.2], seed=42)
```

---

## Dataset-Level Split

### Storage Location

Split configuration is stored in `annotations.json`:

```json
{
  "format_version": "1.0",
  "dataset_id": "dataset-abc123",

  "split_config": {
    "method": "manual",  // "manual" | "auto" | "none"
    "default_ratio": [0.8, 0.2],  // [train, val]
    "seed": 42,
    "created_at": "2025-01-10T10:00:00Z",
    "created_by": "user-123",
    "splits": {
      "image_001": "train",
      "image_002": "val",
      "image_003": "train",
      "image_004": null  // Will be auto-assigned
    }
  },

  "classes": [...],
  "images": [...],
  "annotations": [...]
}
```

### Split Methods

#### Method 1: Manual Split

User manually assigns each image to train/val:

```json
{
  "method": "manual",
  "splits": {
    "image_001": "train",
    "image_002": "train",
    "image_003": "val",
    "image_004": "train",
    "image_005": "val"
  }
}
```

**Use case**: Production datasets with carefully curated splits

#### Method 2: Auto Split

System automatically assigns with specified ratio:

```json
{
  "method": "auto",
  "default_ratio": [0.8, 0.2],
  "seed": 42,
  "splits": {
    // Auto-generated, stored for reproducibility
    "image_001": "train",
    "image_002": "train",
    "image_003": "val",
    ...
  }
}
```

**Use case**: Research datasets with reproducible random splits

#### Method 3: Partial Split

Some images assigned, rest auto-filled:

```json
{
  "method": "manual",
  "default_ratio": [0.8, 0.2],
  "seed": 42,
  "splits": {
    "image_001": "train",  // Manually assigned
    "image_002": "val",    // Manually assigned
    "image_003": null,     // Will be auto-assigned
    "image_004": null      // Will be auto-assigned
  }
}
```

**Use case**: Hybrid approach - important samples manually split, rest automatic

### Database Schema

```python
# platform/backend/app/db/models.py

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String)

    # Split configuration (stored in annotations.json, cached in DB)
    split_config = Column(JSON, nullable=True)
    # {
    #   "method": "manual",
    #   "default_ratio": [0.8, 0.2],
    #   "seed": 42
    # }
```

### API: Create/Update Split

```python
# POST /api/v1/datasets/{dataset_id}/split
async def update_dataset_split(
    dataset_id: str,
    split_config: SplitConfig
):
    """Create or update dataset split configuration"""

    # 1. Load annotations
    annotations = await load_annotations(dataset_id)

    # 2. Apply split
    if split_config.method == "manual":
        splits = split_config.splits
    elif split_config.method == "auto":
        splits = auto_generate_splits(
            images=annotations['images'],
            ratio=split_config.default_ratio,
            seed=split_config.seed
        )

    # 3. Update annotations.json
    annotations['split_config'] = {
        "method": split_config.method,
        "default_ratio": split_config.default_ratio,
        "seed": split_config.seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "created_by": current_user.id,
        "splits": splits
    }

    await save_annotations(dataset_id, annotations)

    # 4. Update database cache
    dataset = await db.get(Dataset, dataset_id)
    dataset.split_config = annotations['split_config']
    await db.commit()

    return {"status": "updated", "splits": len(splits)}
```

---

## Job-Level Split Override

### API Schema

```python
# platform/backend/app/schemas/training.py

class SplitStrategy(BaseModel):
    method: Literal["use_dataset", "auto", "custom"]
    ratio: Optional[List[float]] = None  # [train, val], e.g., [0.7, 0.3]
    seed: Optional[int] = 42
    custom_splits: Optional[Dict[str, str]] = None  # {"image_id": "train"|"val"}


class TrainingJobCreate(BaseModel):
    dataset_id: str
    model_name: str
    task_type: str

    # Split override (optional)
    split_strategy: Optional[SplitStrategy] = None

    # Other training config
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
```

### Example Requests

#### Use Dataset Split

```bash
POST /api/v1/training/jobs
{
  "dataset_id": "dataset-abc123",
  "model_name": "yolo11n",
  "task_type": "object_detection",
  "split_strategy": {
    "method": "use_dataset"  # Use dataset.split_config
  }
}
```

#### Override with Different Ratio

```bash
POST /api/v1/training/jobs
{
  "dataset_id": "dataset-abc123",
  "model_name": "resnet50",
  "task_type": "image_classification",
  "split_strategy": {
    "method": "auto",
    "ratio": [0.7, 0.3],  # 70/30 instead of dataset's 80/20
    "seed": 123
  }
}
```

#### Custom Split for Specific Job

```bash
POST /api/v1/training/jobs
{
  "dataset_id": "dataset-abc123",
  "model_name": "efficientnet-b0",
  "split_strategy": {
    "method": "custom",
    "custom_splits": {
      "image_001": "train",
      "image_002": "val",
      "image_003": "train",
      ...
    }
  }
}
```

### Backend Processing

```python
# platform/backend/app/services/training_service.py

async def start_training(job_create: TrainingJobCreate) -> TrainingJob:
    """Start training with split strategy"""

    # 1. Determine split strategy
    dataset = await db.get(Dataset, job_create.dataset_id)

    if job_create.split_strategy:
        split_info = job_create.split_strategy
    elif dataset.split_config:
        split_info = SplitStrategy(method="use_dataset")
    else:
        split_info = SplitStrategy(method="auto", ratio=[0.8, 0.2], seed=42)

    # 2. Create training job
    job = TrainingJob(
        dataset_id=job_create.dataset_id,
        model_name=job_create.model_name,
        task_type=job_create.task_type,
        split_strategy=split_info.dict(),
        status="pending"
    )
    db.add(job)
    await db.commit()

    # 3. Pass split info to trainer via environment variables
    env = {
        "JOB_ID": str(job.id),
        "DATASET_ID": job_create.dataset_id,
        "MODEL_NAME": job_create.model_name,
        "SPLIT_STRATEGY": json.dumps(split_info.dict()),
        # ... other env vars
    }

    await launch_trainer(job.id, env)

    return job
```

---

## Runtime Auto-Split

### Default Behavior

If no split strategy is specified at job or dataset level:

```python
def auto_split(
    images: List[dict],
    ratio: List[float] = [0.8, 0.2],
    seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """Auto-generate train/val split"""

    import random
    random.seed(seed)

    # Shuffle images
    shuffled = images.copy()
    random.shuffle(shuffled)

    # Split
    split_idx = int(len(shuffled) * ratio[0])
    train_images = shuffled[:split_idx]
    val_images = shuffled[split_idx:]

    return train_images, val_images
```

**Parameters**:
- `ratio`: [0.8, 0.2] (80% train, 20% val)
- `seed`: 42 (reproducible)
- `method`: Random shuffle

### Stratified Split (Future)

For classification tasks, ensure class balance:

```python
def stratified_split(
    images: List[dict],
    annotations: dict,
    ratio: List[float] = [0.8, 0.2],
    seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """Split with class distribution preservation"""

    from sklearn.model_selection import train_test_split

    # Extract labels
    image_ids = [img['id'] for img in images]
    labels = [
        ann['category_id']
        for ann in annotations['annotations']
        if ann['image_id'] in image_ids
    ]

    # Stratified split
    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=ratio[1],
        stratify=labels,
        random_state=seed
    )

    train_images = [img for img in images if img['id'] in train_ids]
    val_images = [img for img in images if img['id'] in val_ids]

    return train_images, val_images
```

---

## Framework Implementation

### Text File Approach (Recommended)

Instead of copying images to `train/` and `val/` folders, use text files with image paths.

**Benefits**:
- ✅ No image duplication (saves disk space)
- ✅ Faster conversion (no file copying)
- ✅ Consistent with snapshot system (shared images)

### YOLO (Ultralytics)

**Directory structure**:
```
/workspace/dataset/
├── images/                  ← Original images (from S3)
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── labels/
│   ├── train/              ← Label files for training
│   │   ├── img001.txt
│   │   └── img003.txt
│   └── val/                ← Label files for validation
│       ├── img002.txt
│       └── img004.txt
├── train.txt               ← List of training image paths
├── val.txt                 ← List of validation image paths
└── data.yaml
```

**train.txt**:
```
/workspace/dataset/images/img001.jpg
/workspace/dataset/images/img003.jpg
/workspace/dataset/images/img005.jpg
```

**val.txt**:
```
/workspace/dataset/images/img002.jpg
/workspace/dataset/images/img004.jpg
```

**data.yaml**:
```yaml
path: /workspace/dataset
train: train.txt  # Not 'images/train'
val: val.txt      # Not 'images/val'

nc: 2
names: ['cat', 'dog']
```

**Converter implementation**:

```python
# platform/trainers/ultralytics/converters/platform_to_yolo.py

def convert_with_split(
    annotations: dict,
    split_info: SplitStrategy,
    output_dir: Path
):
    """Convert Platform Format to YOLO with split"""

    # 1. Determine split
    if split_info.method == "use_dataset":
        splits = annotations['split_config']['splits']
    elif split_info.method == "auto":
        train_imgs, val_imgs = auto_split(
            annotations['images'],
            split_info.ratio,
            split_info.seed
        )
        splits = {
            img['id']: 'train' for img in train_imgs
        } | {
            img['id']: 'val' for img in val_imgs
        }
    elif split_info.method == "custom":
        splits = split_info.custom_splits

    # 2. Create label files
    for img in annotations['images']:
        split = splits.get(img['id'], 'train')
        label_dir = output_dir / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)

        # Create YOLO label file
        label_path = label_dir / f"{Path(img['file_name']).stem}.txt"
        with open(label_path, 'w') as f:
            for ann in get_annotations_for_image(img['id'], annotations):
                # Convert to YOLO format
                yolo_line = to_yolo_format(ann, img)
                f.write(yolo_line + '\n')

    # 3. Create train.txt and val.txt
    train_images = [
        img for img in annotations['images']
        if splits.get(img['id']) == 'train'
    ]
    val_images = [
        img for img in annotations['images']
        if splits.get(img['id']) == 'val'
    ]

    with open(output_dir / "train.txt", 'w') as f:
        for img in train_images:
            image_path = f"/workspace/dataset/images/{img['file_name']}"
            f.write(image_path + '\n')

    with open(output_dir / "val.txt", 'w') as f:
        for img in val_images:
            image_path = f"/workspace/dataset/images/{img['file_name']}"
            f.write(image_path + '\n')

    # 4. Create data.yaml
    data_yaml = {
        'path': '/workspace/dataset',
        'train': 'train.txt',
        'val': 'val.txt',
        'nc': len(annotations['classes']),
        'names': [c['name'] for c in annotations['classes']]
    }

    with open(output_dir / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f)
```

### PyTorch (timm)

**Using ImageFolder with custom split**:

```python
# platform/trainers/timm/loaders/platform_loader.py

def load_dataset_with_split(
    dataset_path: Path,
    annotations: dict,
    split_info: SplitStrategy,
    transform=None
):
    """Load Platform Format dataset with split"""

    # 1. Determine split
    splits = get_splits(annotations, split_info)

    # 2. Filter images by split
    train_images = [
        img for img in annotations['images']
        if splits.get(img['id']) == 'train'
    ]
    val_images = [
        img for img in annotations['images']
        if splits.get(img['id']) == 'val'
    ]

    # 3. Create datasets
    train_dataset = PlatformDataset(
        images=train_images,
        annotations=annotations,
        dataset_path=dataset_path,
        transform=transform
    )

    val_dataset = PlatformDataset(
        images=val_images,
        annotations=annotations,
        dataset_path=dataset_path,
        transform=transform
    )

    return train_dataset, val_dataset


class PlatformDataset(Dataset):
    def __init__(self, images, annotations, dataset_path, transform=None):
        self.images = images
        self.annotations = annotations
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_path = self.dataset_path / "images" / img_info['file_name']

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Get label
        ann = get_annotation_for_image(img_info['id'], self.annotations)
        label = ann['class_id']

        if self.transform:
            image = self.transform(image)

        return image, label
```

### HuggingFace Transformers

```python
# platform/trainers/huggingface/loaders/platform_loader.py

from datasets import Dataset, DatasetDict

def load_dataset_with_split(
    annotations: dict,
    dataset_path: Path,
    split_info: SplitStrategy
):
    """Load Platform Format as HuggingFace Dataset"""

    # 1. Determine split
    splits = get_splits(annotations, split_info)

    # 2. Prepare data
    data = {
        'image': [],
        'label': [],
        'split': []
    }

    for img in annotations['images']:
        image_path = str(dataset_path / "images" / img['file_name'])
        ann = get_annotation_for_image(img['id'], annotations)

        data['image'].append(image_path)
        data['label'].append(ann['class_id'])
        data['split'].append(splits.get(img['id'], 'train'))

    # 3. Create HF Dataset
    dataset = Dataset.from_dict(data)

    # 4. Split
    train_dataset = dataset.filter(lambda x: x['split'] == 'train')
    val_dataset = dataset.filter(lambda x: x['split'] == 'val')

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
```

---

## Scenarios

### Scenario 1: Use Dataset Default Split

```
Dataset: split_config with 80/20, seed=42
Job: No split_strategy specified
Result: Use dataset's 80/20 split ✅
```

### Scenario 2: Override Dataset Split

```
Dataset: split_config with 80/20
Job: split_strategy = {"method": "auto", "ratio": [0.7, 0.3], "seed": 123}
Result: Use job's 70/30 split ✅
```

### Scenario 3: No Split Defined

```
Dataset: No split_config
Job: No split_strategy
Result: Auto-split 80/20 with seed=42 ✅
```

### Scenario 4: Multiple Users, Same Dataset

```
User A: split_strategy = {"ratio": [0.8, 0.2], "seed": 42}
User B: split_strategy = {"ratio": [0.7, 0.3], "seed": 99}
Result: Each job gets different split ✅
```

### Scenario 5: Partial Dataset Split

```
Dataset: split_config.splits = {"img1": "train", "img2": "val", "img3": null}
Job: No split_strategy
Result: Use specified splits, auto-assign img3 ✅
```

---

## Environment Variables (Trainer)

```bash
# Backend passes split info to trainer
SPLIT_STRATEGY='{"method": "auto", "ratio": [0.8, 0.2], "seed": 42}'
DATASET_SPLIT_CONFIG='{"method": "manual", "splits": {...}}'  # If dataset has split
```

**Trainer reads**:

```python
import os
import json

split_strategy = json.loads(os.environ.get('SPLIT_STRATEGY', '{}'))
dataset_split_config = json.loads(os.environ.get('DATASET_SPLIT_CONFIG', '{}'))

# Apply priority logic
if split_strategy and split_strategy.get('method') != 'use_dataset':
    # Use job-level override
    splits = apply_split_strategy(split_strategy)
elif dataset_split_config:
    # Use dataset-level split
    splits = dataset_split_config['splits']
else:
    # Auto-split
    splits = auto_split(annotations['images'], ratio=[0.8, 0.2], seed=42)
```

---

## Future Enhancements

### 1. Stratified Split

Preserve class distribution in train/val:

```python
split_strategy = {
    "method": "auto",
    "ratio": [0.8, 0.2],
    "stratify": True,  # Ensure balanced class distribution
    "seed": 42
}
```

### 2. Three-Way Split

Support train/val/test:

```python
split_strategy = {
    "method": "auto",
    "ratio": [0.7, 0.2, 0.1],  # train/val/test
    "seed": 42
}
```

### 3. K-Fold Cross Validation

```python
split_strategy = {
    "method": "kfold",
    "n_folds": 5,
    "current_fold": 1,
    "seed": 42
}
```

### 4. Time-Based Split

For temporal data:

```python
split_strategy = {
    "method": "temporal",
    "train_end_date": "2024-12-31",
    "val_start_date": "2025-01-01"
}
```

---

## References

- [DATASET_STORAGE_STRATEGY.md](./DATASET_STORAGE_STRATEGY.md) - Dataset versioning and snapshots
- [TRAINER_DESIGN.md](./TRAINER_DESIGN.md) - Trainer implementation patterns
- [BACKEND_DESIGN.md](./BACKEND_DESIGN.md) - API schemas and database models

---

**Last Updated**: 2025-01-10
**Status**: Production Design
