# Export Convention for Multi-Framework Trainers

**Version**: 1.0
**Date**: 2025-11-16
**Status**: Approved Design

## Table of Contents

1. [Design Background](#design-background)
2. [Architecture Decision](#architecture-decision)
3. [Export Script Convention](#export-script-convention)
4. [Metadata Schema](#metadata-schema)
5. [Implementation Guide](#implementation-guide)
6. [Validation & Testing](#validation--testing)

---

## Design Background

### Problem: Dependency Isolation vs Code Reusability

**Requirement**: Each trainer runs as an isolated service with independent dependencies.

```
Architecture Overview:
├── ultralytics-trainer:8001 (Docker: ultralytics + PyTorch)
├── timm-trainer:8002        (Docker: timm + PyTorch)
└── huggingface-trainer:8003 (Docker: transformers + PyTorch)
```

**Challenge**: How to avoid code duplication without breaking dependency isolation?

### Rejected Approach: Shared Base Module

```python
# ❌ This breaks dependency isolation
from trainers.common.export.base import BaseExportAdapter
```

**Why rejected**:
1. **Dependency coupling**: All trainers depend on `trainers.common`
2. **Deployment complexity**: Need to copy `common/` to each Docker image
3. **Version sync issues**: Different trainers may have different `common` versions
4. **PyPI package overhead**: Requires private package repository or public publishing

### Analysis: What Actually Needs Sharing?

Looking at `platform/trainers/ultralytics/export.py` (606 lines):
- **CLI parsing** (30 lines): Standard argparse - no sharing needed
- **S3 upload** (50 lines): Already handled by Backend `export_subprocess.py`
- **Metadata generation** (50 lines): Simple dict construction - can be templated
- **Export execution** (400 lines): Framework-specific - cannot be shared
- **Validation** (76 lines): Framework-specific - cannot be shared

**Insight**: Only ~10% of code is truly duplicatable, and it's not worth the dependency coupling.

---

## Architecture Decision

### ✅ Adopted: Convention-Based Approach

**Core Principle**: Define a standard interface (convention), provide templates, let each trainer implement independently.

**Benefits**:
- ✅ **Complete dependency isolation**: Each trainer is fully independent
- ✅ **Simple deployment**: No shared modules to sync
- ✅ **Version flexibility**: Each trainer controls its own export implementation
- ✅ **Clear contract**: Backend only needs to know the convention
- ✅ **Low coupling**: Convention is documentation, not code dependency

**Trade-off**:
- ❌ Code duplication across trainers (acceptable because trainers are isolated services)

---

## Export Script Convention

All trainers **MUST** implement an `export.py` script following this convention.

### 1. CLI Interface (Required)

```bash
python export.py \
  --checkpoint_path <path>     # Path to trained checkpoint
  --export_format <format>     # onnx | tensorrt | coreml | tflite | torchscript | openvino
  --output_dir <path>          # Output directory for exported files
  --export_config <json>       # JSON string with format-specific config (optional)
```

**Example**:
```bash
python export.py \
  --checkpoint_path /data/checkpoints/yolo11n_epoch50.pt \
  --export_format onnx \
  --output_dir /data/exports/job_123 \
  --export_config '{"opset_version": 17, "dynamic_axes": true}'
```

### 2. Output Files (Required)

The script **MUST** produce these files in `output_dir`:

```
output_dir/
├── model.{format}      # Exported model file (e.g., model.onnx, model.engine)
└── metadata.json       # Metadata following standard schema
```

**File naming convention**:
- `model.onnx` (ONNX format)
- `model.engine` (TensorRT format)
- `model.mlpackage/` (CoreML format - directory)
- `model.tflite` (TFLite format)
- `model.torchscript` (TorchScript format)
- `model.xml` + `model.bin` (OpenVINO format)

### 3. Exit Codes (Required)

```
0  - Export successful
1  - Export failed (model loading, export execution failed)
2  - Validation failed (export completed but validation detected issues)
3  - Configuration error (invalid arguments, unsupported format)
```

### 4. Logging (Recommended)

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Starting export to ONNX format")
logger.error("Export failed: Invalid checkpoint format")
```

**Why logging matters**: Backend's `export_subprocess.py` captures stdout/stderr for monitoring.

### 5. Environment Variables (Optional)

If using S3 upload within trainer (not recommended - let Backend handle it):

```python
import os
S3_BUCKET = os.getenv('S3_BUCKET', 'vision-platform-exports')
S3_PREFIX = os.getenv('S3_PREFIX', 'exports')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
```

---

## Metadata Schema

### Standard Metadata (Required Fields)

```json
{
  "framework": "ultralytics",
  "model_name": "yolo11n",
  "export_format": "onnx",
  "task_type": "detection",
  "input_shape": [640, 640, 3],
  "input_dtype": "float32",
  "output_shape": [[1, 84, 8400]],
  "class_names": ["person", "bicycle", "car", ...],
  "num_classes": 80,
  "created_at": "2025-11-16T10:30:00Z",
  "export_config": {
    "opset_version": 17,
    "dynamic_axes": true
  }
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `framework` | string | ✅ | Framework name (ultralytics, timm, huggingface) |
| `model_name` | string | ✅ | Model architecture name (yolo11n, resnet50, etc.) |
| `export_format` | string | ✅ | Export format (onnx, tensorrt, coreml, etc.) |
| `task_type` | string | ✅ | Task type (detection, classification, segmentation, pose) |
| `input_shape` | array | ✅ | Model input shape [H, W, C] or [C, H, W] |
| `input_dtype` | string | ✅ | Input data type (float32, uint8) |
| `output_shape` | array | ✅ | Model output shape(s) |
| `class_names` | array | ⚠️ | Class names (required for classification/detection) |
| `num_classes` | integer | ⚠️ | Number of classes (required for classification/detection) |
| `created_at` | string | ✅ | ISO 8601 timestamp |
| `export_config` | object | ⚠️ | Format-specific export configuration |

### Task-Specific Metadata

#### Detection Models
```json
{
  "task_type": "detection",
  "input_shape": [640, 640, 3],
  "output_shape": [[1, 84, 8400]],
  "class_names": ["person", "car", ...],
  "num_classes": 80,
  "anchor_based": false,
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45
}
```

#### Classification Models
```json
{
  "task_type": "classification",
  "input_shape": [224, 224, 3],
  "output_shape": [[1, 1000]],
  "class_names": ["tench", "goldfish", ...],
  "num_classes": 1000,
  "top_k": 5
}
```

#### Segmentation Models
```json
{
  "task_type": "segmentation",
  "input_shape": [640, 640, 3],
  "output_shape": [[1, 80, 160, 160], [1, 32, 160, 160]],
  "class_names": ["background", "person", ...],
  "num_classes": 80,
  "mask_resolution": [160, 160]
}
```

#### Pose Estimation Models
```json
{
  "task_type": "pose",
  "input_shape": [640, 640, 3],
  "output_shape": [[1, 56, 8400]],
  "num_keypoints": 17,
  "keypoint_names": ["nose", "left_eye", "right_eye", ...],
  "skeleton": [[16, 14], [14, 12], ...]
}
```

### Preprocessing Metadata (Recommended)

```json
{
  "preprocessing": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "color_space": "RGB",
    "resize_mode": "letterbox",
    "normalize": true
  }
}
```

### Postprocessing Metadata (Recommended)

```json
{
  "postprocessing": {
    "apply_nms": true,
    "nms_method": "hard",
    "coordinate_format": "xyxy"
  }
}
```

---

## Implementation Guide

### Step 1: Copy Template

```bash
cp docs/examples/export_template.py platform/trainers/your_framework/export.py
```

### Step 2: Implement Framework-Specific Functions

```python
def load_model(checkpoint_path: str):
    """
    Load your framework's model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is corrupted
    """
    # Example for Ultralytics:
    from ultralytics import YOLO
    return YOLO(checkpoint_path)
```

```python
def get_metadata(model) -> dict:
    """
    Extract metadata from model.

    Returns:
        dict: Metadata following standard schema
    """
    return {
        'framework': 'your_framework',
        'model_name': model.architecture_name,
        'task_type': model.task,
        'input_shape': list(model.input_shape),
        'input_dtype': 'float32',
        'output_shape': [list(s) for s in model.output_shapes],
        'class_names': model.class_names,
        'num_classes': len(model.class_names),
        'preprocessing': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'color_space': 'RGB'
        }
    }
```

```python
def export_onnx(model, config: dict) -> str:
    """
    Export model to ONNX format.

    Args:
        model: Loaded model object
        config: Export configuration dict

    Returns:
        str: Path to exported ONNX file

    Raises:
        RuntimeError: If export fails
    """
    # Call framework's native export
    output_path = model.export(
        format='onnx',
        opset=config.get('opset_version', 17),
        dynamic=config.get('dynamic_axes', False),
        simplify=config.get('simplify', True)
    )
    return output_path
```

### Step 3: Test Your Implementation

```bash
# Test basic export
python export.py \
  --checkpoint_path tests/fixtures/model.pt \
  --export_format onnx \
  --output_dir /tmp/test_export

# Verify outputs
ls /tmp/test_export/
# Expected: model.onnx, metadata.json

# Validate metadata
cat /tmp/test_export/metadata.json | python -m json.tool
```

### Step 4: Add Format-Specific Exports

Implement all supported formats:
- `export_onnx()`
- `export_tensorrt()` (if supported)
- `export_coreml()` (if supported)
- `export_tflite()` (if supported)
- `export_torchscript()` (if supported)
- `export_openvino()` (if supported)

If a format is not supported, raise `NotImplementedError`:

```python
def export_tensorrt(model, config: dict) -> str:
    raise NotImplementedError(
        "TensorRT export not supported for this framework"
    )
```

Backend will handle this gracefully and mark the format as unavailable.

---

## Validation & Testing

### Backend Integration Test

The Backend's `export_subprocess.py` will:
1. Execute: `python export.py --checkpoint_path ... --export_format onnx --output_dir ...`
2. Check exit code (0 = success)
3. Verify `output_dir/model.onnx` exists
4. Parse `output_dir/metadata.json`
5. Validate metadata schema
6. Upload to S3 with metadata

### Manual Testing Checklist

- [ ] Script accepts all required CLI arguments
- [ ] Script produces `model.{format}` file
- [ ] Script produces `metadata.json` file
- [ ] Metadata includes all required fields
- [ ] Metadata follows standard schema
- [ ] Exit code is 0 on success
- [ ] Exit code is non-zero on failure
- [ ] Logs are written to stdout/stderr
- [ ] Script works in Docker container
- [ ] Exported model loads in ONNX Runtime (for ONNX)

### Automated Testing Template

```python
# tests/test_export.py
import subprocess
import json
from pathlib import Path

def test_export_onnx(tmp_path):
    """Test ONNX export convention compliance"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Run export
    result = subprocess.run([
        'python', 'export.py',
        '--checkpoint_path', 'tests/fixtures/model.pt',
        '--export_format', 'onnx',
        '--output_dir', str(output_dir),
        '--export_config', '{"opset_version": 17}'
    ], capture_output=True, text=True)

    # Check exit code
    assert result.returncode == 0, f"Export failed: {result.stderr}"

    # Check model file exists
    model_file = output_dir / 'model.onnx'
    assert model_file.exists(), "model.onnx not found"

    # Check metadata file exists
    metadata_file = output_dir / 'metadata.json'
    assert metadata_file.exists(), "metadata.json not found"

    # Validate metadata schema
    with open(metadata_file) as f:
        metadata = json.load(f)

    required_fields = [
        'framework', 'model_name', 'export_format', 'task_type',
        'input_shape', 'output_shape', 'created_at'
    ]
    for field in required_fields:
        assert field in metadata, f"Missing required field: {field}"

    # Validate export format matches
    assert metadata['export_format'] == 'onnx'
```

---

## Format-Specific Guidelines

### ONNX Export

**Recommended config**:
```json
{
  "opset_version": 17,
  "dynamic_axes": true,
  "simplify": true
}
```

**Validation**: Load with ONNX Runtime and run dummy inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
output = session.run(None, {input_name: dummy_input})
```

### TensorRT Export

**Recommended config**:
```json
{
  "fp16": true,
  "int8": false,
  "max_batch_size": 1,
  "workspace_size": 4096
}
```

**Requirements**: NVIDIA GPU, TensorRT installed

### CoreML Export

**Recommended config**:
```json
{
  "minimum_deployment_target": "iOS15",
  "compute_units": "ALL"
}
```

**Output**: `model.mlpackage/` directory (not a file)

### TFLite Export

**Recommended config**:
```json
{
  "quantize": false,
  "int8": false
}
```

**Validation**: Load with TFLite interpreter

---

## Reference Implementations

### Ultralytics (Current)

Location: `platform/trainers/ultralytics/export.py` (606 lines)

**Key features**:
- Supports all 6 formats (ONNX, TensorRT, CoreML, TFLite, TorchScript, OpenVINO)
- Validation with ONNX Runtime
- Comprehensive metadata generation
- Error handling and logging

### Timm (Planned)

Location: `platform/trainers/timm/export.py`

**Expected formats**:
- ONNX (primary)
- TorchScript (secondary)
- CoreML (via coremltools)

**Implementation**: ~200-300 lines (simpler than Ultralytics)

### HuggingFace (Planned)

Location: `platform/trainers/huggingface/export.py`

**Expected formats**:
- ONNX (via optimum)
- TensorRT (via optimum)

**Implementation**: ~250-350 lines

---

## FAQ

### Q1: Can I use shared utility functions?

**A**: Yes, but only within your trainer's directory. Do NOT import from `trainers.common` or other trainers.

Good:
```python
from .utils import generate_metadata  # Same trainer
```

Bad:
```python
from trainers.common.utils import generate_metadata  # ❌ Breaks isolation
from trainers.ultralytics.utils import ...  # ❌ Cross-trainer dependency
```

### Q2: What if my framework doesn't support a format?

**A**: Raise `NotImplementedError` with a clear message:

```python
def export_tensorrt(model, config):
    raise NotImplementedError(
        "TensorRT export is not available for timm models. "
        "Please use ONNX format and convert to TensorRT externally."
    )
```

Backend will detect this and mark the format as unavailable in `/export/capabilities`.

### Q3: Should I handle S3 upload in export.py?

**A**: No. Let Backend's `export_subprocess.py` handle S3 upload. Your script only needs to:
1. Save exported model to `output_dir/model.{format}`
2. Save metadata to `output_dir/metadata.json`

Backend will zip and upload to S3.

### Q4: Can I add custom fields to metadata?

**A**: Yes! Add them under `extra_metadata`:

```json
{
  "framework": "ultralytics",
  "model_name": "yolo11n",
  ...
  "extra_metadata": {
    "custom_field": "value",
    "framework_version": "8.0.200"
  }
}
```

Required fields must still be present.

### Q5: How do I handle validation errors?

**A**: Return exit code 2 and log the error:

```python
try:
    validate_export(output_path)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    sys.exit(2)
```

Backend will mark the export job as `failed` with the error message.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-16 | Initial convention based on Ultralytics implementation |

---

## See Also

- [Export Template](../examples/export_template.py) - Reference implementation
- [Export API Specification](EXPORT_DEPLOYMENT_DESIGN.md) - Backend API design
- [Trainer Development Guide](TRAINER_DEVELOPMENT_GUIDE.md) - How to add new trainers
