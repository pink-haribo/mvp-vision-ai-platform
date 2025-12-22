# DICE Annotation Format Specification

**Version**: 1.0.0
**Date**: 2025-12-22
**Status**: Stable
**Audience**: Labeler Team, Platform Developers

---

## Overview

**DICE Format** is Vision AI Platform's standard annotation format for computer vision datasets. It is **COCO-compatible** with Platform-specific extensions for enhanced flexibility.

### Design Goals

- ✅ **COCO Compatibility**: Standard COCO format works out-of-the-box
- ✅ **Multi-Task Support**: Detection, Classification, Segmentation, Pose Estimation
- ✅ **Negative Samples**: First-class support for images without annotations
- ✅ **Flexible Storage**: S3, R2, local filesystem
- ✅ **Nested Format**: Optional simplified structure for Labeler frontend

---

## File Naming Convention

Annotation files are named by task type:

| Task Type | Filename | Description |
|-----------|----------|-------------|
| Object Detection | `annotations_detection.json` | Bounding boxes |
| Image Classification | `annotations_classification.json` | Image-level labels |
| Instance Segmentation | `annotations_segmentation.json` | Polygons/masks |
| Pose Estimation | `annotations_pose.json` | Keypoints |
| Generic | `annotations.json` | Fallback if task-specific file missing |

**Labeler Backend** should generate task-specific files based on project type.

---

## Format Variants

### Variant 1: COCO/DICE Standard Format (Recommended)

Fully compatible with COCO dataset format.

```json
{
  "categories": [
    {
      "id": 1,
      "name": "broken",
      "supercategory": "defect"
    },
    {
      "id": 2,
      "name": "normal"
    }
  ],

  "images": [
    {
      "id": 1,
      "file_name": "images/wood/scratch/000.png",
      "width": 640,
      "height": 480,
      "date_captured": "2025-11-20T10:00:00Z"
    },
    {
      "id": 2,
      "file_name": "images/zipper/good/001.png",
      "width": 640,
      "height": 480
    }
  ],

  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 180],
      "area": 36000,
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 1,
      "bbox": [300, 200, 150, 120],
      "area": 18000,
      "iscrowd": 0
    }
  ],

  "storage_info": {
    "image_root": "datasets/ds_c75023ca76d7448b/images/",
    "storage_type": "s3"
  }
}
```

**Key Fields**:
- `categories`: List of classes (required)
- `images`: List of images (required)
- `annotations`: List of annotations (can be empty)
- `storage_info`: Storage metadata (Platform extension)

---

### Variant 2: Platform Nested Format (Optional)

Simplified format with annotations nested inside images.

```json
{
  "classes": [
    { "id": 1, "name": "cat" },
    { "id": 2, "name": "dog" }
  ],

  "images": [
    {
      "id": 1,
      "file_name": "images/001.jpg",
      "width": 800,
      "height": 600,
      "annotations": [
        {
          "id": 1,
          "class_id": 1,
          "bbox": [50, 60, 120, 140]
        },
        {
          "id": 2,
          "class_id": 2,
          "bbox": [200, 100, 180, 200]
        }
      ]
    },
    {
      "id": 2,
      "file_name": "images/002.jpg",
      "width": 800,
      "height": 600,
      "annotations": []
    }
  ],

  "storage_info": {
    "image_root": "datasets/ds_xyz/images/"
  }
}
```

**Differences from Standard**:
- `classes` instead of `categories`
- `annotations` nested inside each `image`
- `class_id` instead of `category_id`
- Automatically converted to standard format by Platform

**Use Case**: Labeler frontend can use this for simpler data binding.

---

## Special Features

### 1. Negative Samples (No Objects)

Images without any objects are marked with `__background__` category:

```json
{
  "categories": [
    { "id": 0, "name": "__background__" },
    { "id": 1, "name": "defect" }
  ],

  "images": [
    {
      "id": 42,
      "file_name": "images/good/sample.png",
      "width": 640,
      "height": 480
    }
  ],

  "annotations": [
    {
      "id": 100,
      "image_id": 42,
      "category_id": 0
      // No bbox field = negative sample
    }
  ]
}
```

**Platform Behavior**:
- `__background__` class is filtered out during training
- Creates empty YOLO label file (`labels/sample.txt` with 0 bytes)
- Helps models learn "no object" cases

---

### 2. Storage Info

`storage_info` field provides S3/R2 path mapping:

```json
{
  "storage_info": {
    "image_root": "datasets/ds_c75023ca76d7448b/images/",
    "annotation_root": "datasets/ds_c75023ca76d7448b/annotations/",
    "storage_type": "s3"
  }
}
```

**Purpose**:
- Trainer SDK uses this to map `file_name` to actual S3 keys
- Example: `file_name: "wood/scratch/000.png"` → S3 key: `datasets/ds_abc/images/wood/scratch/000.png`

---

### 3. Bbox Format

Bounding boxes use **COCO format** (top-left + width/height):

```
bbox: [x, y, width, height]

Example: [100, 150, 200, 180]
  x = 100       (left edge)
  y = 150       (top edge)
  width = 200
  height = 180
```

**NOT** `[x1, y1, x2, y2]` (YOLO format) - Platform converts automatically.

---

## Task-Specific Schemas

### Object Detection

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 180],
      "area": 36000,
      "iscrowd": 0
    }
  ]
}
```

### Instance Segmentation

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 180],
      "segmentation": [[x1,y1,x2,y2,x3,y3,...]],
      "area": 36000,
      "iscrowd": 0
    }
  ]
}
```

### Pose Estimation

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 180],
      "keypoints": [x1,y1,v1, x2,y2,v2, ...],
      "num_keypoints": 17
    }
  ]
}
```

**Keypoint visibility values**:
- `v = 0`: Not labeled (not visible)
- `v = 1`: Labeled but not visible (occluded)
- `v = 2`: Labeled and visible

### Image Classification

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 2
      // No bbox for classification
    }
  ]
}
```

---

## Validation Rules

### Required Fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `categories[].id` | int | ✅ | Unique, >= 0 |
| `categories[].name` | string | ✅ | Non-empty |
| `images[].id` | int | ✅ | Unique, >= 1 |
| `images[].file_name` | string | ✅ | Relative path |
| `images[].width` | int | ✅ | > 0 |
| `images[].height` | int | ✅ | > 0 |
| `annotations[].id` | int | ✅ | Unique, >= 1 |
| `annotations[].image_id` | int | ✅ | Valid reference |
| `annotations[].category_id` | int | ✅ | Valid reference |
| `annotations[].bbox` | array | ⚠️ | Required for detection (except `__background__`) |

### Validation Examples

**✅ Valid**:
```json
{
  "categories": [{"id": 1, "name": "cat"}],
  "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0,0,10,10]}]
}
```

**❌ Invalid** (missing bbox for non-background):
```json
{
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1}]
}
```

**✅ Valid** (negative sample with `__background__`):
```json
{
  "categories": [{"id": 0, "name": "__background__"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 0}]
}
```

---

## Compatibility Matrix

| Tool/Framework | COCO Standard | Nested Format | Notes |
|----------------|---------------|---------------|-------|
| Platform Trainer SDK | ✅ | ✅ | Auto-converts both |
| COCO API | ✅ | ❌ | Standard only |
| Labelme | ✅ | ❌ | Export as COCO |
| CVAT | ✅ | ❌ | Export as COCO |
| Labeler Frontend | ✅ | ✅ | Can use either |

---

## Migration Guide

### From COCO to DICE

COCO datasets work as-is. Optionally add `storage_info`:

```json
{
  "storage_info": {
    "image_root": "datasets/YOUR_DATASET_ID/images/"
  }
}
```

### From YOLO to DICE

Use Platform's conversion API:

```bash
POST /api/v1/datasets/convert
{
  "source_format": "yolo",
  "target_format": "dice",
  "dataset_path": "s3://bucket/datasets/yolo-dataset/"
}
```

---

## JSON Schema Reference

**Full JSON Schema**: `platform/schemas/dice_format.schema.json`

Validate your annotations:

```bash
# Using ajv-cli
ajv validate -s dice_format.schema.json -d annotations_detection.json

# Using Python jsonschema
python -c "
import json, jsonschema
schema = json.load(open('dice_format.schema.json'))
data = json.load(open('annotations_detection.json'))
jsonschema.validate(data, schema)
print('✅ Valid')
"
```

---

## Implementation Reference

DICE format conversion is implemented in Trainer SDK:

**File**: `platform/trainers/{framework}/trainer_sdk.py`
**Method**: `_convert_dice_to_yolo()` (lines 1500-1820)

**Conversion Flow**:
1. Select annotation file by task type (`annotations_{task_type}.json`)
2. Parse DICE format (supports both COCO standard and nested format)
3. Convert to YOLO format:
   - Create `labels/*.txt` files
   - Generate `train.txt`, `val.txt` split files
   - Create `data.yaml` config
4. Filter out `__background__` class
5. Handle nested directory structures

---

## FAQ

### Q: Can I use both formats in the same dataset?

**A**: No. Choose either COCO standard or nested format per annotation file.

### Q: How do I mark unlabeled images?

**A**: Don't include them in `images` array, or use `__background__` annotation.

### Q: What if an image has no annotations?

**A**: Include the image in `images` array but no corresponding entries in `annotations` array.

### Q: Can I have multiple annotation files?

**A**: Yes! Use task-specific files:
- `annotations_detection.json`
- `annotations_segmentation.json`
- etc.

Platform SDK will load the correct one based on `TASK_TYPE` environment variable.

### Q: What's the difference between `category_id` and `class_id`?

**A**: They're aliases. `category_id` is COCO standard, `class_id` is Platform nested format. Both are supported.

### Q: How are image paths resolved?

**A**: `file_name` is relative to `storage_info.image_root`.

Example:
- `storage_info.image_root`: `"datasets/ds_abc/images/"`
- `file_name`: `"wood/scratch/000.png"`
- Final S3 key: `"datasets/ds_abc/images/wood/scratch/000.png"`

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-22 | Initial specification |

---

## Contact

**Platform Team**: platform@example.com
**Labeler Team**: labeler@example.com
**Slack**: #dataset-format

---

## See Also

- [Labeler Dataset API Requirements](../cowork/LABELER_DATASET_API_REQUIREMENTS.md)
- [Dataset Management Architecture](../architecture/DATASET_MANAGEMENT_ARCHITECTURE.md)
- [Trainer SDK Documentation](../../platform/trainers/README.md)

---

**License**: Internal Use Only
**Copyright**: Vision AI Platform Team
