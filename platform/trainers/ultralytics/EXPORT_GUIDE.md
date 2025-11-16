# Ultralytics Model Export Guide

Complete guide for exporting trained Ultralytics models to production formats.

## Table of Contents

- [Quick Start](#quick-start)
- [Export Script Usage](#export-script-usage)
- [Supported Formats](#supported-formats)
- [Format-Specific Configurations](#format-specific-configurations)
- [Metadata Schema](#metadata-schema)
- [Runtime Wrappers](#runtime-wrappers)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Export to ONNX

```bash
python export.py \
  --checkpoint_path /data/checkpoints/yolo11n_epoch50.pt \
  --export_format onnx \
  --output_dir /data/exports/job_123 \
  --export_config '{"opset_version": 17}'
```

**Output**:
```
/data/exports/job_123/
├── model.onnx       # Exported ONNX model
└── metadata.json    # Model metadata (input shape, class names, preprocessing, etc.)
```

### With Validation

```bash
python export.py \
  --checkpoint_path /data/checkpoints/yolo11n_epoch50.pt \
  --export_format onnx \
  --output_dir /data/exports/job_123 \
  --export_config '{"opset_version": 17, "include_validation": true}'
```

Validation runs inference with ONNX Runtime and compares outputs with original model.

## Export Script Usage

### Command-Line Interface

```bash
python export.py \
  --checkpoint_path <path>     # Required: Path to trained checkpoint (.pt file)
  --export_format <format>     # Required: onnx|tensorrt|coreml|tflite|torchscript|openvino
  --output_dir <path>          # Required: Output directory for exported files
  --export_config <json>       # Optional: JSON string with format-specific config
```

### Environment Variables (K8s Deployment)

When running in Kubernetes, the script reads from environment variables:

```bash
export CHECKPOINT_PATH=/data/checkpoints/model.pt
export EXPORT_FORMAT=onnx
export OUTPUT_DIR=/data/exports
export EXPORT_CONFIG='{"opset_version": 17}'

python export.py
```

This allows the Backend to launch export jobs as K8s Jobs without modifying the script.

## Supported Formats

| Format | File Extension | Primary Use Case | Hardware Support |
|--------|---------------|------------------|------------------|
| ONNX | `.onnx` | Cross-platform deployment, cloud inference | CPU, GPU (CUDA, OpenCL) |
| TensorRT | `.engine` | NVIDIA GPU optimized inference | NVIDIA GPU only |
| CoreML | `.mlpackage/` | iOS/macOS deployment | Apple Silicon, ANE |
| TFLite | `.tflite` | Mobile & embedded devices | CPU, EdgeTPU, GPU |
| TorchScript | `.torchscript` | PyTorch native deployment | CPU, GPU (CUDA) |
| OpenVINO | `.xml` + `.bin` | Intel hardware optimized | Intel CPU, iGPU, VPU |

### Capability Matrix by Model

| Model | ONNX | TensorRT | CoreML | TFLite | TorchScript | OpenVINO |
|-------|------|----------|--------|--------|-------------|----------|
| YOLOv8n | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLOv8s | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLOv8m | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLO11n | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLO11n-seg | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLO11n-pose | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLO-World-v2-s | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| SAM2-t | ✅ | ⚠️ | ❌ | ❌ | ✅ | ⚠️ |

✅ = Fully supported
⚠️ = Partially supported (may have limitations)
❌ = Not supported

## Format-Specific Configurations

### ONNX

**Recommended for**: Cross-platform deployment, cloud inference, production servers

```json
{
  "opset_version": 17,
  "dynamic_axes": true,
  "simplify": true,
  "half": false
}
```

**Configuration Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `opset_version` | int | 17 | ONNX opset version (13-18 supported) |
| `dynamic_axes` | bool | false | Enable dynamic batch/height/width |
| `simplify` | bool | true | Simplify ONNX graph (recommended) |
| `half` | bool | false | Export as FP16 (may reduce accuracy) |

**Example**:
```bash
python export.py \
  --checkpoint_path model.pt \
  --export_format onnx \
  --output_dir ./exports \
  --export_config '{
    "opset_version": 17,
    "dynamic_axes": true,
    "simplify": true
  }'
```

**Output Shape** (YOLOv8n detection):
- Input: `[batch, 3, 640, 640]` (or dynamic if `dynamic_axes=true`)
- Output: `[batch, 84, 8400]` (84 = 4 bbox + 80 classes)

### TensorRT

**Recommended for**: NVIDIA GPU inference, edge devices with NVIDIA GPU

```json
{
  "fp16": true,
  "int8": false,
  "workspace": 4,
  "max_batch_size": 1
}
```

**Configuration Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fp16` | bool | false | Enable FP16 precision (2x faster, minimal accuracy loss) |
| `int8` | bool | false | Enable INT8 quantization (requires calibration) |
| `workspace` | int | 4 | Workspace size in GB (higher = faster, more memory) |
| `max_batch_size` | int | 1 | Maximum batch size for optimization |

**Example**:
```bash
python export.py \
  --checkpoint_path model.pt \
  --export_format tensorrt \
  --output_dir ./exports \
  --export_config '{
    "fp16": true,
    "workspace": 8
  }'
```

**Requirements**:
- NVIDIA GPU with CUDA support
- TensorRT installed (`pip install tensorrt`)
- Export must run on GPU machine

### CoreML

**Recommended for**: iOS/macOS deployment, on-device inference

```json
{
  "minimum_deployment_target": "iOS15",
  "nms": true
}
```

**Configuration Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `minimum_deployment_target` | str | "iOS15" | iOS13, iOS14, iOS15, iOS16, iOS17 |
| `nms` | bool | true | Include NMS in CoreML model |

**Example**:
```bash
python export.py \
  --checkpoint_path model.pt \
  --export_format coreml \
  --output_dir ./exports \
  --export_config '{
    "minimum_deployment_target": "iOS15",
    "nms": true
  }'
```

**Output**: `model.mlpackage/` directory (not a single file)

**Deployment**:
- Use Swift runtime wrapper in `runtimes/swift/`
- See [Swift Integration Guide](./runtimes/swift/README.md)

### TFLite

**Recommended for**: Android devices, embedded systems, edge TPU

```json
{
  "int8": false,
  "tflite_model": "default"
}
```

**Configuration Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `int8` | bool | false | Enable INT8 quantization |
| `tflite_model` | str | "default" | "default" or "edgetpu" for Coral TPU |

**Example**:
```bash
python export.py \
  --checkpoint_path model.pt \
  --export_format tflite \
  --output_dir ./exports \
  --export_config '{
    "int8": false
  }'
```

**Deployment**:
- Use Kotlin runtime wrapper in `runtimes/kotlin/`
- See [Kotlin Integration Guide](./runtimes/kotlin/README.md)

### TorchScript

**Recommended for**: PyTorch deployment, C++ integration

```json
{
  "optimize": true
}
```

**Configuration Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `optimize` | bool | true | Apply TorchScript optimizations |

**Example**:
```bash
python export.py \
  --checkpoint_path model.pt \
  --export_format torchscript \
  --output_dir ./exports \
  --export_config '{
    "optimize": true
  }'
```

**Usage in Python**:
```python
import torch
model = torch.jit.load('model.torchscript')
model.eval()
output = model(input_tensor)
```

### OpenVINO

**Recommended for**: Intel CPU/GPU/VPU, edge devices

```json
{
  "half": false
}
```

**Configuration Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `half` | bool | false | Export as FP16 |

**Example**:
```bash
python export.py \
  --checkpoint_path model.pt \
  --export_format openvino \
  --output_dir ./exports \
  --export_config '{}'
```

**Output**: `model.xml` (IR definition) + `model.bin` (weights)

## Metadata Schema

Every export produces a `metadata.json` file with standardized structure:

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
  },
  "preprocessing": {
    "mean": [0.0, 0.0, 0.0],
    "std": [255.0, 255.0, 255.0],
    "color_space": "RGB",
    "resize_mode": "letterbox",
    "normalize": true
  },
  "postprocessing": {
    "apply_nms": true,
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "coordinate_format": "xyxy"
  }
}
```

### Task-Specific Metadata

#### Detection Models (YOLOv8, YOLO11)

```json
{
  "task_type": "detection",
  "input_shape": [640, 640, 3],
  "output_shape": [[1, 84, 8400]],
  "class_names": ["person", "car", ...],
  "num_classes": 80,
  "anchor_based": false
}
```

#### Segmentation Models (YOLOv8-seg, YOLO11-seg)

```json
{
  "task_type": "segmentation",
  "input_shape": [640, 640, 3],
  "output_shape": [[1, 116, 8400], [1, 32, 160, 160]],
  "class_names": ["person", "car", ...],
  "num_classes": 80,
  "mask_resolution": [160, 160]
}
```

#### Pose Estimation Models (YOLOv8-pose, YOLO11-pose)

```json
{
  "task_type": "pose",
  "input_shape": [640, 640, 3],
  "output_shape": [[1, 56, 8400]],
  "num_keypoints": 17,
  "keypoint_names": ["nose", "left_eye", "right_eye", ...],
  "skeleton": [[16, 14], [14, 12], [12, 6], ...]
}
```

#### Open-Vocabulary Detection (YOLO-World-v2)

```json
{
  "task_type": "detection",
  "model_variant": "open_vocabulary",
  "input_shape": [640, 640, 3],
  "text_input_enabled": true,
  "max_text_length": 77
}
```

## Runtime Wrappers

Pre-built runtime wrappers for easy integration in production environments.

### Python (ONNX Runtime)

**Location**: `runtimes/python/`

**Usage**:
```python
from runtimes.python.inference import YOLOInference

# Initialize
model = YOLOInference(
    model_path='model.onnx',
    metadata_path='metadata.json'
)

# Run inference
results = model.predict(
    'image.jpg',
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Access detections
for det in results.detections:
    print(f"{det.class_name}: {det.confidence:.2f}")
    print(f"BBox: {det.bbox}")
```

**Features**:
- Automatic preprocessing (resize, normalize)
- NMS postprocessing
- Multi-batch support
- GPU acceleration (if available)

**See**: [Python Runtime Guide](./runtimes/python/README.md)

### C++ (ONNX Runtime + OpenCV)

**Location**: `runtimes/cpp/`

**Usage**:
```cpp
#include "onnx_inference.h"

// Initialize
ONNXInference model("model.onnx", "metadata.json");

// Run inference
cv::Mat image = cv::imread("image.jpg");
auto results = model.predict(image, 0.25, 0.45);

// Access detections
for (const auto& det : results.detections) {
    std::cout << det.class_name << ": " << det.confidence << std::endl;
}
```

**Build**:
```bash
cd runtimes/cpp
mkdir build && cd build
cmake ..
make
```

**See**: [C++ Runtime Guide](./runtimes/cpp/README.md)

### Swift (CoreML)

**Location**: `runtimes/swift/`

**Usage**:
```swift
import CoreML
import Vision

let model = try YOLOModel(contentsOf: modelURL)
let inference = YOLOInference(model: model)

let results = try inference.predict(image: uiImage, confThreshold: 0.25)
for detection in results.detections {
    print("\(detection.className): \(detection.confidence)")
}
```

**Integration**: iOS/macOS apps with Xcode

**See**: [Swift Runtime Guide](./runtimes/swift/README.md)

### Kotlin (TFLite)

**Location**: `runtimes/kotlin/`

**Usage**:
```kotlin
import com.platform.tflite.YOLOInference

val model = YOLOInference(context, "model.tflite", "metadata.json")
val results = model.predict(bitmap, confThreshold = 0.25f)

results.detections.forEach { detection ->
    println("${detection.className}: ${detection.confidence}")
}
```

**Integration**: Android apps with Android Studio

**See**: [Kotlin Runtime Guide](./runtimes/kotlin/README.md)

## Validation

Export script includes optional validation to verify exported model correctness.

### How Validation Works

1. Load original PyTorch model and exported model (ONNX Runtime)
2. Generate random input tensor matching input shape
3. Run inference on both models
4. Compare outputs (bounding boxes, confidence scores)
5. Check if differences are within acceptable threshold

### Enable Validation

```bash
python export.py \
  --checkpoint_path model.pt \
  --export_format onnx \
  --output_dir ./exports \
  --export_config '{"opset_version": 17, "include_validation": true}'
```

### Validation Criteria

- **Bounding Box IoU**: > 0.95 (95% overlap)
- **Confidence Score Difference**: < 0.05 (5% tolerance)
- **Class Prediction**: Must match exactly

### Exit Codes

- **0**: Export successful, validation passed
- **1**: Export failed (model loading, export execution failed)
- **2**: Export successful, but validation failed
- **3**: Configuration error (invalid arguments, unsupported format)

## Troubleshooting

### Export Fails with "CUDA out of memory"

**Cause**: TensorRT export requires GPU and may use lots of VRAM

**Solution**:
1. Reduce `workspace` size in config: `"workspace": 2`
2. Close other GPU processes
3. Use smaller model (e.g., YOLOv8n instead of YOLOv8x)
4. Export on machine with more VRAM

### ONNX Model Has Different Output Shape

**Cause**: Dynamic axes not properly configured

**Solution**:
1. Check `dynamic_axes` in config
2. Verify input shape in metadata.json
3. For fixed batch size, set `"dynamic_axes": false`

### CoreML Export Fails on Linux

**Cause**: CoreML export requires macOS (coremltools limitation)

**Solution**:
1. Export ONNX first on Linux
2. Convert ONNX → CoreML on macOS:
   ```bash
   pip install coremltools
   python convert_onnx_to_coreml.py model.onnx model.mlpackage
   ```

### TFLite Model Has Low Accuracy

**Cause**: INT8 quantization without calibration dataset

**Solution**:
1. Export without quantization: `"int8": false`
2. Or provide calibration dataset (advanced, see Ultralytics docs)
3. Use FP16 instead (if TFLite GPU delegate supports it)

### Export Takes Too Long

**Cause**: Validation running on large model

**Solution**:
1. Disable validation: Remove `"include_validation": true`
2. Or set timeout in Backend's export_subprocess.py

### Validation Always Fails

**Cause**: Numerical precision differences between PyTorch and ONNX Runtime

**Solution**:
1. This is often acceptable for production use
2. Check actual accuracy degradation is minimal
3. Adjust validation thresholds (requires modifying export.py)
4. Or disable validation and test manually

### File Not Found Error

**Cause**: Checkpoint path doesn't exist or incorrect format

**Solution**:
1. Verify checkpoint file exists: `ls -la $CHECKPOINT_PATH`
2. Check it's a `.pt` file (not `.pth` or `.safetensors`)
3. Use absolute path instead of relative path

## Advanced Usage

### Batch Export Multiple Formats

```bash
for format in onnx tensorrt coreml tflite; do
  python export.py \
    --checkpoint_path model.pt \
    --export_format $format \
    --output_dir ./exports/$format
done
```

### Custom Export Configuration

Create a config file `export_config.json`:

```json
{
  "onnx": {
    "opset_version": 17,
    "dynamic_axes": true,
    "simplify": true
  },
  "tensorrt": {
    "fp16": true,
    "workspace": 8
  },
  "coreml": {
    "minimum_deployment_target": "iOS15"
  }
}
```

Use in script:
```python
import json

with open('export_config.json') as f:
    configs = json.load(f)

for format, config in configs.items():
    # Run export with config
    ...
```

### Programmatic Export

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# Export to ONNX
model.export(
    format='onnx',
    opset=17,
    dynamic=True,
    simplify=True
)

# Export to TensorRT
model.export(
    format='engine',
    half=True,
    workspace=8
)
```

## See Also

- [Export Convention](../../../docs/EXPORT_CONVENTION.md) - Convention-based export design
- [Export Template](../../../docs/examples/export_template.py) - Reference implementation
- [Export & Deployment Design](../../docs/architecture/EXPORT_DEPLOYMENT_DESIGN.md) - Full system design
- [Python Runtime Guide](./runtimes/python/README.md)
- [C++ Runtime Guide](./runtimes/cpp/README.md)
- [Swift Runtime Guide](./runtimes/swift/README.md)
- [Kotlin Runtime Guide](./runtimes/kotlin/README.md)
