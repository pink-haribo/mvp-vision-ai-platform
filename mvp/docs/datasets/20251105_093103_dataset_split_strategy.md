# Dataset Train/Validation Split Strategy

**Date**: 2025-11-05 09:31
**Status**: Proposed
**Related Issues**: DICE→YOLO converter validation split, Training validation metrics

## Overview

This document defines the platform-wide strategy for managing train/validation dataset splits across different user scenarios and frameworks. The strategy implements a 3-level priority system: Job-level override → Dataset-level metadata → Runtime auto-split.

## Background / Context

### Problem Discovery

During YOLO training implementation, we discovered that the DICE→YOLO converter was creating datasets where train and validation pointed to the same data:

```yaml
# dice_to_yolo.py output (INCORRECT)
train: images/train  # 32 images
val: images/train    # SAME 32 images! ❌
```

This made all validation metrics (mAP50, precision, recall) meaningless since the model was being evaluated on the same data it trained on.

### Root Questions

1. **When should split be defined?**
   - At dataset creation (labeling phase)?
   - At training job creation?
   - At runtime (auto-split)?

2. **How to handle multiple users with different split preferences?**
   - User A wants 80/20 split
   - User B wants 70/30 split on the same dataset

3. **How to handle partially-labeled splits?**
   - Some images labeled as "train"
   - Some images labeled as "val"
   - Remaining images unlabeled

## Current State

### DICE Format Structure

Current `annotations.json` format:

```json
{
  "version": "1.0",
  "dataset_info": {...},
  "categories": [...],
  "images": [],  // Often empty, reconstructed from files
  "annotations": [
    {
      "id": 1,
      "image_id": "image_1",
      "category_id": 1,
      "bbox": [x, y, w, h]
      // NO SPLIT INFO ❌
    }
  ]
}
```

**Key Finding**: No split metadata exists in current DICE format.

### Current Converter Behavior

**DICE → YOLO Converter** (BEFORE fix):
- Copied all images to `images/train/`
- Set `val: images/train` in data.yaml
- Result: Train and val used identical data

**DICE → YOLO Converter** (AFTER fix - commit 1c24aa6):
- Random shuffle with seed=42
- 80/20 split
- Separate `images/train/` and `images/val/` folders
- Result: Proper train/val separation

## Proposed Solution: 3-Level Split Strategy

### Architecture Overview

```
Priority 1: Job-Level Split Override
  ↓ (if not specified)
Priority 2: Dataset-Level Split Metadata
  ↓ (if not specified)
Priority 3: Runtime Auto-Split (80/20 default)
```

### Level 1: Dataset-Level Split Metadata

**DICE Format Extension** - Add optional `split_config` to `annotations.json`:

```json
{
  "version": "1.0",
  "dataset_info": {...},
  "split_config": {
    "method": "manual",  // "manual" | "auto" | "none"
    "default_ratio": [0.8, 0.2],  // [train, val]
    "created_at": "2025-11-05T09:30:00Z",
    "created_by": "user_123",
    "splits": {
      "image_1": "train",
      "image_2": "val",
      "image_3": "train",
      "image_4": null  // Will be auto-assigned
    }
  },
  "categories": [...],
  "annotations": [...]
}
```

**Alternative Approach** - Annotation-level split:

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": "image_1",
      "split": "train",  // ← Per-annotation split info
      "category_id": 1,
      "bbox": [...]
    }
  ]
}
```

**Recommendation**: Use top-level `split_config` for better separation of concerns.

### Level 2: Job-Level Split Override

**Training Job Schema Extension**:

```python
# Backend: app/schemas/training.py
class TrainingJobCreate(BaseModel):
    dataset_id: str
    model_name: str

    # NEW: Split strategy override
    split_strategy: Optional[SplitStrategy] = None

class SplitStrategy(BaseModel):
    method: Literal["use_dataset", "auto", "custom"]
    ratio: Optional[List[float]] = None  # [train, val] e.g., [0.7, 0.3]
    seed: Optional[int] = 42  # For reproducibility

    # For advanced use: explicit image assignment
    custom_splits: Optional[Dict[str, str]] = None
```

**Example API Call**:

```bash
POST /api/v1/training/jobs
{
  "dataset_id": "40acbb9d-e381-4d58-82d4-0391042963fb",
  "model_name": "yolo11n",
  "split_strategy": {
    "method": "auto",
    "ratio": [0.7, 0.3],  // User wants 70/30 instead of dataset default
    "seed": 42
  }
}
```

### Level 3: Runtime Auto-Split (Fallback)

If no split information is available at Level 1 or 2:

```python
# Current implementation in _auto_split_dataset()
import random

def auto_split(images, ratio=0.8, seed=42):
    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    return train_images, val_images
```

**Default Behavior**:
- Ratio: 80% train, 20% val
- Seed: 42 (reproducible)
- Method: Random shuffle

## Implementation Plan

### Phase 1: Immediate (Current Sprint)

**Goal**: Fix validation split for YOLO training

- [x] Fix DICE→YOLO converter to create separate train/val folders
- [x] Enable YOLO validation (`val: True`)
- [ ] Change to text-file-based split (avoid image duplication)
- [ ] Update converter to use `train.txt` / `val.txt` instead of folder copying

**Text File Approach**:

```python
# DICE → YOLO converter (optimized)
def convert_dice_to_yolo_txt_split(dice_dir, output_dir, split_info=None):
    # Don't copy images - use original DICE images
    # Only create label files and txt file lists

    train_images, val_images = get_split(images, split_info)

    # Create label files in labels/train/ and labels/val/
    for img in train_images:
        create_label_file(img, labels_dir / "train")
    for img in val_images:
        create_label_file(img, labels_dir / "val")

    # Create train.txt with absolute paths to original images
    with open(output_dir / "train.txt", 'w') as f:
        for img in train_images:
            f.write(f"{dice_dir}/images/{img.filename}\n")

    # Create val.txt
    with open(output_dir / "val.txt", 'w') as f:
        for img in val_images:
            f.write(f"{dice_dir}/images/{img.filename}\n")

    # data.yaml
    return {
        'train': 'train.txt',  # Not 'images/train'
        'val': 'val.txt'
    }
```

**Benefits**:
- ✅ No image duplication (saves disk space)
- ✅ Faster conversion (no file copying)
- ✅ Consistent with `_auto_split_dataset()` approach

### Phase 2: Dataset-Level Split (Future)

**Goal**: Allow users to define splits in Dataset Manager

- [ ] Extend DICE format with `split_config`
- [ ] Add split UI in Dataset Manager
- [ ] Update Backend API to read/write split metadata
- [ ] Update all converters to respect dataset split

**Backend Schema**:

```python
# app/db/models.py
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String)
    format = Column(String)  # "dice", "coco", "yolo"

    # NEW: Split configuration
    split_config = Column(JSON, nullable=True)
    # {
    #   "method": "manual",
    #   "splits": {"image_1": "train", ...}
    # }
```

**Frontend UI** (Dataset Manager):

```tsx
// DatasetPanel.tsx
<SplitManager
  datasetId={dataset.id}
  images={images}
  onSplitChange={(splits) => updateSplitConfig(splits)}
/>

// Features:
// - Drag-and-drop images to train/val buckets
// - Auto-split with custom ratio slider
// - Import/export split CSV
```

### Phase 3: Job-Level Override (Future)

**Goal**: Allow per-job split customization

- [ ] Add `split_strategy` to TrainingJobCreate schema
- [ ] Update TrainingManager to pass split override to converters
- [ ] Handle priority: Job > Dataset > Auto

**Converter Integration**:

```python
# train.py
def start_training(job_config):
    # Priority check
    if job_config.get('split_strategy'):
        split_info = job_config['split_strategy']
    elif dataset_has_split_config(job_config['dataset_id']):
        split_info = load_dataset_split_config(job_config['dataset_id'])
    else:
        split_info = {'method': 'auto', 'ratio': [0.8, 0.2]}

    # Pass to converter
    convert_dice_to_yolo(
        dice_dir=dataset_path,
        output_dir=output_dir,
        split_override=split_info
    )
```

## Scenario Handling

### Scenario 1: Dataset has split, Job uses it

```
Dataset: split_config.splits = {"img1": "train", "img2": "val", ...}
Job: split_strategy = None (or method="use_dataset")
Result: Use dataset split as-is ✅
```

### Scenario 2: Dataset has split, Job overrides

```
Dataset: split_config with 80/20 split
Job: split_strategy = {"method": "auto", "ratio": [0.7, 0.3]}
Result: Ignore dataset split, use 70/30 auto-split ✅
```

### Scenario 3: Dataset has no split

```
Dataset: No split_config
Job: split_strategy = None
Result: Auto-split 80/20 ✅
```

### Scenario 4: Partial dataset split

```
Dataset: split_config.splits = {"img1": "train", "img2": "val", "img3": null}
Job: split_strategy = None
Result: Use specified splits, auto-assign remaining images ✅
```

### Scenario 5: Multiple users, same dataset

```
User A Job: split_strategy = {"ratio": [0.8, 0.2], "seed": 42}
User B Job: split_strategy = {"ratio": [0.7, 0.3], "seed": 123}
Result: Each job gets its own split ✅
```

## Technical Details

### YOLO Split Methods Comparison

| Method | Implementation | Pros | Cons |
|--------|---------------|------|------|
| **Folder Structure** | `images/train/`, `images/val/` | Clear separation, easy to visualize | Duplicates images, wastes disk |
| **Text Files** | `train.txt`, `val.txt` with paths | No duplication, faster | Less intuitive, requires abs paths |

**Recommendation**: Use text files for production.

### Data Structures

**SplitConfig (Database)**:

```python
{
  "method": "manual",  # How split was created
  "default_ratio": [0.8, 0.2],
  "created_at": "2025-11-05T09:30:00Z",
  "created_by": "user_123",
  "seed": 42,
  "splits": {
    "image_abc123": "train",
    "image_def456": "val",
    "image_ghi789": "train"
  }
}
```

**SplitStrategy (Job Config)**:

```python
{
  "method": "auto",  # "use_dataset" | "auto" | "custom"
  "ratio": [0.7, 0.3],  # Train/val ratio
  "seed": 42,
  "custom_splits": {  # For method="custom"
    "image_abc123": "train",
    "image_def456": "val"
  }
}
```

### Framework-Specific Implementation

**YOLO (Ultralytics)**:

```yaml
# data.yaml
path: /dataset
train: train.txt
val: val.txt
```

**PyTorch (timm)**:

```python
# Use ImageFolder with custom split
train_dataset = ImageFolder(
    root=dataset_path,
    transform=train_transform,
    is_valid_file=lambda path: path in train_image_paths
)
```

**HuggingFace Transformers**:

```python
# Use datasets library
from datasets import Dataset

dataset = Dataset.from_dict({
    'image': image_paths,
    'label': labels,
    'split': splits  # 'train' or 'val'
})

split_dataset = dataset.train_test_split(
    test_size=0.2,
    stratify_by_column='label'
)
```

## Migration Path

### For Existing Datasets

1. **No action required** - Auto-split will continue to work
2. **Optional**: Add split metadata to `annotations.json`
3. **Optional**: Re-upload dataset with split configuration

### For Existing Training Jobs

1. **No breaking changes** - Jobs without split_strategy will auto-split
2. **Recommended**: Clear cached YOLO datasets to force re-conversion:
   ```bash
   rm -rf workspace/data/.cache/datasets/*_yolo/
   ```

### Backward Compatibility

- ✅ Old datasets without split_config: Auto-split (80/20)
- ✅ Old jobs without split_strategy: Use dataset split or auto-split
- ✅ Existing converters: Add split parameter as optional

## Alternatives Considered

### Alternative 1: Always use dataset-level split

**Pros**:
- Single source of truth
- Simpler implementation

**Cons**:
- ❌ Not flexible for per-job customization
- ❌ Multi-user scenarios problematic
- ❌ Can't experiment with different splits

**Why rejected**: Too restrictive for research use cases.

### Alternative 2: Always auto-split at runtime

**Pros**:
- Simple, no metadata needed
- Works for all datasets

**Cons**:
- ❌ Not reproducible across jobs (unless same seed)
- ❌ Can't preserve user-defined splits
- ❌ No control for users

**Why rejected**: Lacks control and reproducibility.

### Alternative 3: Store split in training job only

**Pros**:
- Maximum flexibility
- No dataset modification needed

**Cons**:
- ❌ Loses split information if job deleted
- ❌ Can't share splits across users
- ❌ Duplicates split logic for every job

**Why rejected**: 3-level approach provides better balance.

## References

### Related Files

- `mvp/training/converters/dice_to_yolo.py` - DICE→YOLO converter
- `mvp/training/adapters/ultralytics_adapter.py` - YOLO training adapter
- `mvp/backend/app/db/models.py` - Database models
- `mvp/backend/app/schemas/training.py` - API schemas

### Related Commits

- `1c24aa6` - Fix DICE→YOLO train/val split (folder approach)
- `31a81be` - Enable YOLO validation during training
- `6915831` - Add comprehensive frontend logging to UltralyticsAdapter

### External Resources

- [Ultralytics YOLO Data Format](https://docs.ultralytics.com/datasets/)
- [COCO Dataset Format](https://cocodataset.org/#format-data)

## Notes

### Open Questions

1. **Stratification**: Should auto-split stratify by class distribution?
   - Current: Random shuffle
   - Alternative: Ensure balanced class distribution in train/val

2. **Test split**: Should we support train/val/test 3-way split?
   - Current: Only train/val
   - Future: Add optional test split

3. **Cross-validation**: Support k-fold CV splits?
   - Current: Single train/val split
   - Future: Store multiple fold definitions

### Future Considerations

1. **Split versioning**: Track changes to dataset splits over time
2. **Split templates**: Predefined split strategies (stratified, random, time-based)
3. **Split validation**: Ensure no data leakage between train/val
4. **Split analytics**: Show class distribution, dataset statistics per split

### Performance Implications

**Text file approach** (recommended):
- Conversion time: ~2-5s for 1000 images
- Disk usage: Original images only (no duplication)
- Memory: Minimal (only metadata)

**Folder approach** (current):
- Conversion time: ~10-30s for 1000 images (file copying)
- Disk usage: 2x original (images copied to train/ and val/)
- Memory: Minimal

**Recommendation**: Migrate to text file approach in Phase 1.
