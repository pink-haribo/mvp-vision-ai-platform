# Model Export & Deployment Design

Complete design for model export and deployment system.

## Table of Contents

- [Overview](#overview)
- [Export Job Design](#export-job-design)
- [Supported Export Formats](#supported-export-formats)
- [Framework Capability Matrix](#framework-capability-matrix)
- [Pre/Post Processing Strategy](#prepost-processing-strategy)
- [Optimization Strategies](#optimization-strategies)
- [Deployment Strategies](#deployment-strategies)
- [Backend Design](#backend-design)
- [Trainer Implementation](#trainer-implementation)
- [3-Tier Execution](#3-tier-execution)
- [Storage Strategy](#storage-strategy)
- [Frontend Implementation](#frontend-implementation)
- [Use Cases](#use-cases)
- [Implementation Phases](#implementation-phases)

## Overview

The export/deployment system converts trained model checkpoints into production-ready formats and provides various deployment options.

**Two-Phase Approach**:
1. **Export**: Convert checkpoint → Optimized format (ONNX, TensorRT, etc.)
2. **Deployment**: Deploy exported model → Production environment

**Key Principles**:
- **Complete Isolation**: Export jobs run in separate processes/containers
- **3-Tier Compatible**: Works in subprocess, Kind, and production K8s
- **S3 Storage**: All artifacts stored in S3
- **Framework Agnostic**: Supports all training frameworks (Ultralytics, timm, HuggingFace)
- **User Permission Based**: Export limits based on user tier/subscription

## Export Job Design

### What is an Export Job?

Export Job converts a trained checkpoint into a deployment-ready format with optional optimizations.

**Input**: Training checkpoint (`.pt`, `.pth`, `.safetensors`)
**Output**: Exported model in target format + metadata + runtime wrappers

**Workflow**:
```
TrainingJob (checkpoint.pt)
    ↓
ExportJob (format=onnx, optimize=quantize_int8)
    ↓
Exported Model Package
  ├── model.onnx
  ├── metadata.json (pre/post processing specs)
  └── runtimes/ (Python, C++, Swift, Kotlin wrappers)
    ↓
Deployment (download, platform_endpoint, edge_package, container)
```

### Export Job Types

**1. Format Conversion**
- Convert checkpoint to target format
- No optimization, pure format conversion
- Fast, lossless conversion

**2. Optimized Export**
- Format conversion + optimization
- Quantization (INT8, FP16)
- Pruning (structured/unstructured)
- Slower, may have accuracy trade-offs
- Optional validation after export

**3. Edge Deployment Export**
- Format conversion + edge-specific optimization
- Mobile-optimized (CoreML, TFLite)
- Hardware-specific (TensorRT for NVIDIA, OpenVINO for Intel)
- Size and speed prioritized

## Supported Export Formats

### ONNX (Open Neural Network Exchange)

**Target**: Cross-platform deployment, cloud inference
**Pros**: Framework-agnostic, wide hardware support, well-established
**Cons**: May not support latest model architectures

```python
# Export configuration
{
  "format": "onnx",
  "opset_version": 17,
  "dynamic_axes": {
    "input": [0, 2, 3],  # Batch, height, width
    "output": [0]
  },
  "optimize_for_inference": true
}
```

**Use Cases**:
- Cloud inference (Triton Inference Server, ONNX Runtime)
- Multi-framework compatibility
- CPU inference

### TensorRT

**Target**: NVIDIA GPU inference
**Pros**: Fastest inference on NVIDIA GPUs, automatic optimization
**Cons**: NVIDIA-only, platform-specific

```python
{
  "format": "tensorrt",
  "precision": "fp16",  # fp32, fp16, int8
  "max_batch_size": 32,
  "workspace_size_gb": 4
}
```

**Use Cases**:
- High-throughput GPU inference
- Real-time applications
- Edge devices with NVIDIA GPUs (Jetson)

### CoreML

**Target**: Apple devices (iOS, macOS)
**Pros**: Native iOS/macOS integration, optimized for Apple Silicon
**Cons**: Apple ecosystem only

```python
{
  "format": "coreml",
  "minimum_deployment_target": "iOS15",
  "compute_units": "all"  # all, cpu_only, cpu_and_gpu
}
```

**Use Cases**:
- Mobile apps (iPhone, iPad)
- macOS applications
- Apple Silicon Macs

### TensorFlow Lite

**Target**: Mobile and edge devices (Android, Raspberry Pi)
**Pros**: Small model size, fast inference on mobile
**Cons**: Limited operator support

```python
{
  "format": "tflite",
  "quantization": "dynamic",  # none, dynamic, full_integer
  "supported_ops": ["TFLITE_BUILTINS", "SELECT_TF_OPS"]
}
```

**Use Cases**:
- Android apps
- Raspberry Pi
- Embedded systems

### OpenVINO

**Target**: Intel hardware (CPU, integrated GPU, VPU)
**Pros**: Optimized for Intel hardware, broad device support
**Cons**: Intel ecosystem focused

```python
{
  "format": "openvino",
  "precision": "FP16",
  "model_optimizer_args": {
    "data_type": "FP16",
    "reverse_input_channels": true
  }
}
```

**Use Cases**:
- Intel CPU inference
- Edge devices with Intel hardware
- Industrial applications

### TorchScript

**Target**: PyTorch production deployment
**Pros**: Full PyTorch compatibility, easy to use
**Cons**: PyTorch-only

```python
{
  "format": "torchscript",
  "method": "trace",  # trace or script
  "optimize_for_mobile": false
}
```

**Use Cases**:
- PyTorch-based serving
- LibTorch C++ deployment
- Custom PyTorch inference servers

## Framework Capability Matrix

Each framework supports different export formats with varying quality levels.

### Support Matrix

| Framework | ONNX | TensorRT | CoreML | TFLite | OpenVINO | TorchScript |
|-----------|------|----------|--------|--------|----------|-------------|
| **Ultralytics** | ✅ Excellent<br/>Native | ✅ Excellent<br/>Native | ✅ Excellent<br/>Native | ✅ Excellent<br/>Native | ✅ Excellent<br/>Native | ✅ Excellent<br/>Native |
| **timm** | ✅ Excellent<br/>Native | ⚠️ Good<br/>Via ONNX | ⚠️ Good<br/>Via ONNX | ⚠️ Fair<br/>Via ONNX | ⚠️ Good<br/>Via ONNX | ✅ Excellent<br/>Native |
| **HuggingFace** | ✅ Excellent<br/>Native | ⚠️ Fair<br/>Via ONNX | ❌ Not supported | ⚠️ Fair<br/>Via ONNX | ✅ Excellent<br/>Native | ✅ Excellent<br/>Native |

**Legend**:
- ✅ Excellent: Native support, production-ready, well-tested
- ⚠️ Good/Fair: Via conversion, may have limitations
- ❌ Not supported: Currently not available

### API Endpoint: Get Capabilities

```python
# GET /api/v1/export/capabilities?framework={framework}

Response:
{
  "framework": "ultralytics",
  "formats": {
    "onnx": {
      "supported": true,
      "quality": "excellent",
      "native": true,
      "preprocessing_embedded": true
    },
    "tensorrt": {
      "supported": true,
      "quality": "excellent",
      "native": true,
      "preprocessing_embedded": true
    },
    "coreml": {
      "supported": true,
      "quality": "excellent",
      "native": true,
      "preprocessing_embedded": true
    },
    "tflite": {
      "supported": true,
      "quality": "excellent",
      "native": true,
      "preprocessing_embedded": false  # Needs wrapper
    }
  }
}
```

### Two-Stage Export for Non-Native Formats

For frameworks without native support, use two-stage conversion:

```python
# Example: timm → TensorRT
# Stage 1: timm → ONNX (native)
onnx_model = export_timm_to_onnx(checkpoint)

# Stage 2: ONNX → TensorRT (via tool)
tensorrt_engine = convert_onnx_to_tensorrt(onnx_model)
```

## Pre/Post Processing Strategy

**Critical Decision**: How to handle preprocessing/postprocessing in exported models?

### Three-Tier Approach (Hybrid)

#### Tier 1: Embedded Pre/Post Processing

**Embed preprocessing/postprocessing into model graph (where possible)**

```python
# Export with embedded preprocessing
class ModelWithPreprocessing(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Preprocessing (embedded in ONNX graph)
        x = x.float() / 255.0  # Normalize to [0, 1]
        x = (x - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])

        # Model inference
        outputs = self.model(x)

        # Postprocessing (embedded)
        predictions = self.apply_nms(outputs)

        return predictions

# Export
torch.onnx.export(ModelWithPreprocessing(model), ...)
```

**Pros**:
- Self-contained model
- No separate preprocessing code needed
- Guaranteed consistency

**Cons**:
- Limited by ONNX operators
- Debugging harder
- Not always possible (e.g., NMS in some formats)

**Applicability**: ONNX, TorchScript

#### Tier 2: Runtime Wrappers

**Provide framework-specific wrapper code**

Every export includes runtime wrappers in `runtimes/` folder:

```
export-package/
├── model.onnx
├── metadata.json
└── runtimes/
    ├── python/
    │   └── model_wrapper.py
    ├── cpp/
    │   └── model_wrapper.cpp
    ├── swift/
    │   └── ModelWrapper.swift
    └── kotlin/
        └── ModelWrapper.kt
```

**Python Wrapper Example**:
```python
# runtimes/python/model_wrapper.py
import numpy as np
import onnxruntime as ort
import cv2
import json

class ModelWrapper:
    def __init__(self, model_path, metadata_path):
        self.session = ort.InferenceSession(model_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)

    def preprocess(self, image):
        """Preprocessing exactly as used in training"""
        # Read specs from metadata
        preproc = self.metadata["preprocessing"]

        # Resize
        size = preproc["resize"]["size"]
        image = cv2.resize(image, tuple(size))

        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array(preproc["normalize"]["mean"])
        std = np.array(preproc["normalize"]["std"])
        image = (image - mean) / std

        # Format: HWC → NCHW
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image

    def postprocess(self, outputs):
        """Postprocessing (NMS, threshold, etc.)"""
        postproc = self.metadata["postprocessing"]

        boxes, scores, classes = self.parse_outputs(outputs)

        # Apply NMS
        indices = self.nms(
            boxes, scores,
            iou_threshold=postproc["nms"]["iou_threshold"]
        )

        # Filter by confidence
        conf_threshold = postproc["nms"]["confidence_threshold"]
        mask = scores[indices] > conf_threshold

        predictions = [
            {
                "class_id": int(classes[i]),
                "class_name": self.metadata["classes"][int(classes[i])]["name"],
                "confidence": float(scores[i]),
                "bbox": boxes[i].tolist()
            }
            for i in indices[mask]
        ]

        return predictions

    def predict(self, image_path):
        """End-to-end prediction"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        outputs = self.session.run(None, {"input": input_tensor})

        # Postprocess
        predictions = self.postprocess(outputs[0])

        return predictions


# Usage
wrapper = ModelWrapper("model.onnx", "metadata.json")
predictions = wrapper.predict("image.jpg")
```

**Pros**:
- Works for all formats
- Easy to customize
- Clear separation of concerns

**Cons**:
- User must use wrapper (not pure model file)
- Slight integration effort

**Applicability**: All formats

#### Tier 3: Metadata-Driven

**Comprehensive metadata describing pre/post processing**

```json
// metadata.json
{
  "model_info": {
    "framework": "ultralytics",
    "task_type": "object_detection",
    "model_name": "yolo11n",
    "export_format": "onnx"
  },

  "preprocessing": {
    "resize": {
      "method": "letterbox",
      "size": [640, 640],
      "fill_value": 114,
      "stride": 32
    },
    "normalize": {
      "method": "standard",
      "mean": [0.0, 0.0, 0.0],  # ImageNet mean
      "std": [255.0, 255.0, 255.0]
    },
    "format": {
      "input_layout": "HWC",
      "output_layout": "NCHW",
      "color_space": "RGB",
      "dtype": "float32"
    }
  },

  "postprocessing": {
    "nms": {
      "enabled": true,
      "iou_threshold": 0.45,
      "confidence_threshold": 0.25,
      "max_detections": 300
    },
    "output_format": {
      "bbox_format": "xyxy",  # or xywh, cxcywh
      "coordinates": "absolute"  # or normalized
    }
  },

  "input_spec": {
    "name": "input",
    "shape": [1, 3, 640, 640],
    "dtype": "float32",
    "format": "NCHW"
  },

  "output_spec": {
    "detection": {
      "name": "output",
      "shape": [1, 25200, 85],
      "format": "xyxy_conf_classes",
      "description": "25200 anchors, 85 values per anchor (4 bbox + 1 conf + 80 classes)"
    }
  },

  "classes": [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "bicycle"},
    {"id": 2, "name": "car"}
  ],

  "performance": {
    "inference_latency_ms": 25.3,
    "throughput_fps": 39.5,
    "hardware": "NVIDIA RTX 4090",
    "batch_size": 1
  },

  "runtime_wrappers": {
    "python": "runtimes/python/model_wrapper.py",
    "cpp": "runtimes/cpp/model_wrapper.cpp",
    "swift": "runtimes/swift/ModelWrapper.swift",
    "kotlin": "runtimes/kotlin/ModelWrapper.kt"
  }
}
```

**Pros**:
- Complete specification
- Machine-readable
- Enables automatic wrapper generation

**Cons**:
- Requires comprehensive metadata generation

**Applicability**: All formats, always included

### Implementation Strategy

**All three tiers combined**:

1. **Try Embedded** (Tier 1): If format supports it (ONNX, TorchScript)
2. **Always provide Runtime Wrappers** (Tier 2): For all formats
3. **Always include Metadata** (Tier 3): Comprehensive specs

**Result**: Users can choose:
- Use embedded preprocessing (if available) → Just load model
- Use runtime wrappers → Copy wrapper code
- Read metadata → Implement custom wrapper

## Optimization Strategies

### Quantization

**Reduces model size and inference time by using lower precision**

**Dynamic Quantization** (Easiest)
- Weights: INT8
- Activations: FP32 (computed at runtime)
- No calibration data needed
- 2-4x smaller models, 1.5-2x faster

```python
{
  "optimization": {
    "quantization": {
      "method": "dynamic",
      "dtype": "qint8"
    }
  }
}
```

**Static Quantization** (Best accuracy)
- Weights: INT8
- Activations: INT8
- Requires calibration dataset
- 4x smaller models, 2-4x faster

```python
{
  "optimization": {
    "quantization": {
      "method": "static",
      "dtype": "qint8",
      "calibration_dataset_id": "dataset-uuid",
      "calibration_samples": 100
    }
  }
}
```

**Quantization-Aware Training** (Best performance)
- Train with quantization simulation
- Requires retraining
- Best accuracy at INT8
- Future implementation (Phase 3)

### Pruning

**Removes redundant weights to reduce model size**

**Structured Pruning**
- Removes entire channels/filters
- Hardware-friendly (actual speedup)
- 30-50% smaller models

```python
{
  "optimization": {
    "pruning": {
      "method": "structured",
      "sparsity": 0.5,  # Remove 50% of channels
      "pruning_schedule": "iterative"
    }
  }
}
```

**Unstructured Pruning**
- Removes individual weights
- Sparse tensors (needs sparse inference support)
- 50-90% smaller models

### Validation After Export (Optional)

**Test model accuracy after optimization**

```python
{
  "validation": {
    "enabled": true,
    "dataset_id": "validation-dataset-uuid",
    "metrics": ["accuracy", "mAP50-95"],
    "fail_threshold": {
      "accuracy_drop_max": 0.02  # Fail if accuracy drops > 2%
    }
  }
}
```

**Process**:
1. Export + optimize model
2. Run inference on validation dataset
3. Compare metrics to original checkpoint
4. Report accuracy drop
5. Fail export if threshold exceeded (optional)

**Output**:
```json
{
  "validation_metrics": {
    "accuracy": 0.94,
    "original_accuracy": 0.95,
    "accuracy_drop": 0.01,
    "mAP50-95": 0.6423,
    "original_mAP50-95": 0.6523,
    "passed": true
  }
}
```

## Deployment Strategies

### 1. Self-Hosted Download

**Simplest deployment: User downloads exported model**

**Process**:
1. User completes export job
2. Platform generates presigned download URL
3. User downloads model package (model + metadata + wrappers)
4. User deploys on their own infrastructure

**Pros**:
- No platform infrastructure needed
- User has full control
- Simple implementation

**Cons**:
- No inference endpoint provided
- User responsible for serving

**Implementation**:
```python
# Backend generates presigned URL
download_url = storage_client.generate_presigned_url(
    bucket="vision-platform",
    key=f"exports/{export_job_id}/export-package.zip",
    expiration=86400  # 24 hours
)
```

**Package Contents**:
```
export-package.zip
├── model.onnx (or other format)
├── metadata.json
├── README.md
└── runtimes/
    ├── python/
    │   ├── model_wrapper.py
    │   ├── requirements.txt
    │   └── example.py
    ├── cpp/
    │   ├── model_wrapper.cpp
    │   ├── model_wrapper.h
    │   └── CMakeLists.txt
    ├── swift/
    │   ├── ModelWrapper.swift
    │   └── Package.swift
    └── kotlin/
        ├── ModelWrapper.kt
        └── build.gradle
```

### 2. Platform Inference Endpoint ⭐ **CRITICAL**

**Platform hosts exported model and provides REST API endpoint**

**Why This is Critical**:
- Users want instant deployment without DevOps
- Pay-as-you-go inference (no infrastructure management)
- Auto-scaling for variable workloads
- Built-in monitoring and analytics

**Process**:
1. User completes export job
2. Platform deploys to inference server (Triton/TorchServe)
3. Platform provides API endpoint + API key
4. User calls endpoint for predictions

**Architecture**:
```
Exported Model (ONNX)
    ↓
Triton Inference Server (K8s Deployment)
  - Auto-scaling (HPA: 1-10 replicas)
  - GPU support (optional)
  - Model versioning
    ↓
REST API Endpoint: https://api.platform.com/v1/infer/{deployment_id}
```

**API Request**:
```http
POST /v1/infer/{deployment_id}
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "confidence_threshold": 0.5,
  "iou_threshold": 0.45
}

Response:
{
  "predictions": [
    {
      "class": "cat",
      "class_id": 0,
      "confidence": 0.95,
      "bbox": [10, 20, 100, 150]
    }
  ],
  "latency_ms": 25,
  "model_version": "v1"
}
```

**Pricing Model** (User Permission Based):
```python
# Tier-based pricing
PRICING_TIERS = {
    "free": {
        "requests_per_month": 1000,
        "rate_limit_per_minute": 10,
        "gpu_access": False
    },
    "pro": {
        "requests_per_month": 100000,
        "rate_limit_per_minute": 100,
        "gpu_access": True,
        "price_per_1000_requests": 0.50
    },
    "enterprise": {
        "requests_per_month": "unlimited",
        "rate_limit_per_minute": 1000,
        "gpu_access": True,
        "dedicated_instance": True,
        "price_per_month": 500
    }
}
```

**Implementation**: See Backend Design section

### 3. Edge Deployment Package

**Package model + runtime for edge devices**

**Process**:
1. User completes export job (CoreML/TFLite)
2. Platform creates deployment package
3. Package includes:
   - Optimized model
   - Runtime wrapper code
   - Configuration
   - Sample integration code

**Mobile Deployment Package Structure**:
```
vision-ai-mobile-package-{export_job_id}.zip
├── model.tflite (or model.mlmodel)
├── labels.txt
├── metadata.json
├── sample_code/
│   ├── android/
│   │   ├── ModelWrapper.kt
│   │   ├── MainActivity.kt (example)
│   │   └── build.gradle
│   └── ios/
│       ├── ModelWrapper.swift
│       ├── ViewController.swift (example)
│       └── Package.swift
└── README.md
```

### 4. Container Deployment

**Docker container with model + serving runtime**

**Hybrid Approach**:
- **Phase 1**: Dockerfile template + build script (MVP)
- **Phase 2**: Platform-managed Docker registry

**Phase 1 Implementation** (Dockerfile Template):

```
container-package.zip
├── Dockerfile
├── model.onnx
├── metadata.json
├── config.pbtxt (Triton config)
├── build.sh
└── README.md
```

**Dockerfile**:
```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Copy model
COPY model.onnx /models/model/1/model.onnx
COPY config.pbtxt /models/model/config.pbtxt

# Expose port
EXPOSE 8000 8001 8002

# Run Triton
CMD ["tritonserver", "--model-repository=/models"]
```

**Usage**:
```bash
# Build
docker build -t my-model:latest .

# Run
docker run -p 8000:8000 my-model:latest

# Test
curl -X POST http://localhost:8000/v2/models/model/infer \
  -H "Content-Type: application/json" \
  -d @input.json
```

**Phase 2 Implementation** (Platform Registry):
```python
# Platform automatically builds and pushes to registry
platform.registry.com/users/{user_id}/models/{export_job_id}:latest

# User just pulls
docker pull platform.registry.com/users/{user_id}/models/{export_job_id}:latest
docker run -p 8000:8000 platform.registry.com/users/{user_id}/models/{export_job_id}:latest
```

## Backend Design

### ExportJob Model

```python
# app/models/export_job.py
from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class ExportFormat(str, enum.Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    COREML = "coreml"
    TFLITE = "tflite"
    OPENVINO = "openvino"
    TORCHSCRIPT = "torchscript"

class ExportJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExportJob(Base):
    """Export Job: Convert checkpoint to deployment format"""
    __tablename__ = "export_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False)

    # Version management
    version = Column(Integer, nullable=False)  # Auto-increment per training_job
    version_tag = Column(String(100), nullable=True)  # "production", "staging", "v1.0"
    is_default = Column(Boolean, default=False)

    # Identifiers
    trace_id = Column(UUID(as_uuid=True), default=uuid.uuid4, index=True)
    callback_token = Column(String(500), nullable=False)
    k8s_job_name = Column(String(255), nullable=True)
    process_id = Column(Integer, nullable=True)

    # Configuration
    export_format = Column(SQLEnum(ExportFormat), nullable=False)
    framework = Column(String(100), nullable=False)
    task_type = Column(String(100), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)

    # Export settings
    export_config = Column(JSON, nullable=True)
    # {
    #   "opset_version": 17,
    #   "dynamic_axes": {...},
    #   "embed_preprocessing": true
    # }

    # Optimization settings
    optimization_config = Column(JSON, nullable=True)
    # {
    #   "quantization": {
    #     "method": "static",
    #     "dtype": "qint8",
    #     "calibration_dataset_id": "uuid"
    #   },
    #   "pruning": {
    #     "method": "structured",
    #     "sparsity": 0.5
    #   }
    # }

    # Validation settings (optional)
    validation_config = Column(JSON, nullable=True)
    # {
    #   "enabled": true,
    #   "dataset_id": "uuid",
    #   "fail_threshold": {"accuracy_drop_max": 0.02}
    # }

    # Status
    status = Column(SQLEnum(ExportJobStatus), default=ExportJobStatus.PENDING, index=True)
    progress_percent = Column(Float, default=0.0)

    # Results
    exported_model_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/exports/{export_job_id}/model.onnx

    export_package_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/exports/{export_job_id}/export-package.zip

    model_size_bytes = Column(BigInteger, nullable=True)

    optimization_stats = Column(JSON, nullable=True)
    # {
    #   "original_size_mb": 50.2,
    #   "exported_size_mb": 12.5,
    #   "compression_ratio": 4.0
    # }

    # Validation metrics (if validation enabled)
    validation_metrics = Column(JSON, nullable=True)
    # {
    #   "accuracy": 0.94,
    #   "original_accuracy": 0.95,
    #   "accuracy_drop": 0.01,
    #   "passed": true
    # }

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Error tracking
    error_message = Column(String(1000), nullable=True)

    # Relationships
    user = relationship("User", back_populates="export_jobs")
    training_job = relationship("TrainingJob", back_populates="export_jobs")
    deployment_targets = relationship("DeploymentTarget", back_populates="export_job")

    # Indexes
    __table_args__ = (
        Index('idx_training_version', 'training_job_id', 'version'),
        Index('idx_training_default', 'training_job_id', 'is_default'),
    )
```

### DeploymentTarget Model

```python
# app/models/deployment_target.py
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class DeploymentType(str, enum.Enum):
    DOWNLOAD = "download"
    PLATFORM_ENDPOINT = "platform_endpoint"
    EDGE_PACKAGE = "edge_package"
    CONTAINER = "container"

class DeploymentStatus(str, enum.Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"

class DeploymentTarget(Base):
    """Deployment Target: How exported model is deployed"""
    __tablename__ = "deployment_targets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    export_job_id = Column(UUID(as_uuid=True), ForeignKey("export_jobs.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Deployment configuration
    deployment_type = Column(SQLEnum(DeploymentType), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)

    # Status
    status = Column(SQLEnum(DeploymentStatus), default=DeploymentStatus.PENDING)

    # Deployment-specific data
    deployment_config = Column(JSON, nullable=True)
    # For DOWNLOAD:
    # {
    #   "download_url": "https://...",
    #   "expires_at": "2025-01-10T15:00:00Z"
    # }
    #
    # For PLATFORM_ENDPOINT:
    # {
    #   "endpoint_url": "https://api.platform.com/v1/infer/{id}",
    #   "api_key": "pk_...",
    #   "inference_framework": "triton",
    #   "replicas": 2,
    #   "gpu_enabled": true,
    #   "auto_scaling": {"min": 1, "max": 10, "target_utilization": 70}
    # }
    #
    # For EDGE_PACKAGE:
    # {
    #   "package_url": "s3://bucket/edge-packages/{id}.zip",
    #   "target_platform": "ios",
    #   "includes_sample_code": true
    # }
    #
    # For CONTAINER:
    # {
    #   "dockerfile_url": "s3://bucket/container-packages/{id}.zip",
    #   "registry": "platform.registry.com",  # Phase 2
    #   "image_name": "exports/{id}",
    #   "image_tag": "latest"
    # }

    # Usage tracking (for platform_endpoint)
    request_count = Column(Integer, default=0)
    last_request_at = Column(DateTime, nullable=True)
    total_inference_time_ms = Column(BigInteger, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    deployed_at = Column(DateTime, nullable=True)
    deactivated_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    export_job = relationship("ExportJob", back_populates="deployment_targets")
    user = relationship("User", back_populates="deployment_targets")
    history = relationship("DeploymentHistory", back_populates="deployment_target")
```

### DeploymentHistory Model

```python
# app/models/deployment_history.py
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class DeploymentHistory(Base):
    """Deployment event history"""
    __tablename__ = "deployment_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deployment_target_id = Column(UUID(as_uuid=True), ForeignKey("deployment_targets.id"), nullable=False, index=True)

    # Event tracking
    event_type = Column(String(50), nullable=False)
    # "deployed", "updated", "scaled", "deactivated", "reactivated"

    event_data = Column(JSON, nullable=True)
    # {
    #   "replicas": 3,
    #   "version": "v2",
    #   "config_changes": {...}
    # }

    # Status before/after
    status_before = Column(String(50), nullable=True)
    status_after = Column(String(50), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    # Relationships
    deployment_target = relationship("DeploymentTarget", back_populates="history")
```

### API Endpoints

See BACKEND_DESIGN.md for complete endpoint specifications.

**Key Endpoints**:
- `GET /api/v1/export/capabilities` - Get framework export capabilities
- `POST /api/v1/export/jobs` - Create export job
- `GET /api/v1/training/{id}/exports` - List exports for training job
- `POST /api/v1/export/{id}/set-default` - Set default export
- `POST /api/v1/deployments` - Create deployment
- `POST /v1/infer/{deployment_id}` - Platform inference endpoint
- `GET /api/v1/deployments/{id}/history` - Get deployment history

## Trainer Implementation

See TRAINER_DESIGN.md for complete implementation.

**Export Script** (`platform/trainers/{framework}/export.py`):
- Download checkpoint from S3
- Apply optimizations (quantization, pruning)
- Export to target format
- Generate metadata.json
- Generate runtime wrappers
- Create export package (zip)
- Upload to S3
- Send completion callback

**Key Features**:
- Embedded preprocessing (where possible)
- Validation after export (optional)
- Framework capability checking
- Two-stage conversion for non-native formats

## 3-Tier Execution

Export jobs follow the same 3-tier pattern as training and inference.

### Tier 1: Subprocess
```python
process = subprocess.Popen(
    ["python", f"platform/trainers/{framework}/export.py"],
    env={
        "JOB_ID": str(export_job.id),
        "EXPORT_FORMAT": export_job.export_format,
        "CHECKPOINT_PATH": export_job.checkpoint_path,
        "EXPORT_CONFIG": json.dumps(export_job.export_config),
        "OPTIMIZATION_CONFIG": json.dumps(export_job.optimization_config),
        # ... S3 credentials
    }
)
```

### Tier 2/3: Kubernetes Job
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: export-job-{export-job-id}
spec:
  template:
    spec:
      containers:
      - name: export
        image: vision-platform/{framework}-trainer:latest
        command: ["python", "export.py"]
        env:
        - name: JOB_ID
          value: "{export-job-id}"
        - name: EXPORT_FORMAT
          value: "onnx"
        # ... other env vars
```

## Storage Strategy

### S3 Structure

```
vision-platform/
├── exports/
│   ├── {export-job-id-1}/
│   │   ├── model.onnx
│   │   ├── metadata.json
│   │   ├── runtimes/
│   │   │   ├── python/model_wrapper.py
│   │   │   ├── cpp/model_wrapper.cpp
│   │   │   ├── swift/ModelWrapper.swift
│   │   │   └── kotlin/ModelWrapper.kt
│   │   ├── export-package.zip (all above files)
│   │   └── validation_results.json (if validation enabled)
│   └── ...
│
├── edge-packages/
│   ├── {deployment-id-1}.zip (iOS)
│   ├── {deployment-id-2}.zip (Android)
│   └── ...
│
├── container-packages/
│   ├── {deployment-id-1}.zip (Dockerfile + model)
│   └── ...
│
└── triton-models/ (Platform endpoints)
    ├── {deployment-id-1}/
    │   ├── 1/model.onnx
    │   └── config.pbtxt
    └── ...
```

## Frontend Implementation

### Page Structure

```
/training/{job_id}/export-deploy
├── Export Section
│   ├── Export History List (table)
│   ├── Create Export Button → 3-step wizard
│   │   ├── Step 1: Format Selection
│   │   ├── Step 2: Optimization Options
│   │   └── Step 3: Review & Submit
│   └── Export Detail Modal
│
└── Deployment Section
    ├── Active Deployments List (cards)
    ├── Create Deployment Button
    └── Deployment History Timeline
```

### Key Components

**1. ExportHistoryList**: Table showing all exports
**2. CreateExportModal**: 3-step wizard for creating exports
**3. ExportFormatSelector**: Grid of format cards with capability badges
**4. OptimizationConfig**: Accordion panels for quantization/pruning/validation
**5. DeploymentList**: Cards showing active deployments
**6. PlatformEndpointCard**: Endpoint URL, API key, usage stats
**7. DeploymentHistoryTimeline**: Event timeline for deployment

See detailed component implementations in the Frontend Implementation section above.

## Use Cases

### Use Case 1: Mobile App Deployment (iOS)

**Scenario**: Deploy object detection model to iOS app

**Steps**:
1. Train YOLO11n model → TrainingJob
2. Export to CoreML with INT8 quantization → ExportJob
3. Create edge deployment package → DeploymentTarget (type=edge_package)
4. Download package with Swift wrapper and sample code
5. Integrate into Xcode project

**Export Configuration**:
```json
{
  "export_format": "coreml",
  "export_config": {
    "minimum_deployment_target": "iOS15",
    "compute_units": "all"
  },
  "optimization_config": {
    "quantization": {
      "method": "static",
      "dtype": "qint8",
      "calibration_dataset_id": "dataset-uuid"
    }
  },
  "validation_config": {
    "enabled": true,
    "dataset_id": "validation-dataset-uuid",
    "fail_threshold": {"accuracy_drop_max": 0.02}
  }
}
```

### Use Case 2: Platform-Hosted Inference API

**Scenario**: Instant deployment without infrastructure

**Steps**:
1. Train model → TrainingJob
2. Export to ONNX → ExportJob
3. Deploy to platform endpoint → DeploymentTarget (type=platform_endpoint)
4. Get API endpoint and key
5. Call API from application

**Export Configuration**:
```json
{
  "export_format": "onnx",
  "export_config": {
    "opset_version": 17,
    "dynamic_axes": {"input": [0]},
    "embed_preprocessing": true
  }
}
```

**Deployment Configuration**:
```json
{
  "deployment_type": "platform_endpoint",
  "name": "Production API",
  "config": {
    "gpu_enabled": true,
    "auto_scaling": {
      "min_replicas": 2,
      "max_replicas": 10,
      "target_cpu_utilization": 70
    }
  }
}
```

**Usage**:
```python
import requests

response = requests.post(
    "https://api.platform.com/v1/infer/abc123",
    headers={"Authorization": "Bearer pk_..."},
    json={
        "image": base64_encoded_image,
        "confidence_threshold": 0.5
    }
)

predictions = response.json()["predictions"]
```

### Use Case 3: Self-Hosted with Docker

**Scenario**: Deploy on own infrastructure with Docker

**Steps**:
1. Train model → TrainingJob
2. Export to ONNX → ExportJob
3. Create container deployment → DeploymentTarget (type=container)
4. Download Dockerfile package
5. Build and run Docker container

**Contents of download**:
```
container-package.zip
├── Dockerfile
├── model.onnx
├── metadata.json
├── config.pbtxt
├── build.sh
└── README.md
```

**Usage**:
```bash
# Build
./build.sh

# Run
docker run -p 8000:8000 my-model:latest

# Test
curl -X POST http://localhost:8000/v2/models/model/infer \
  -d @input.json
```

### Use Case 4: Edge Device (Raspberry Pi)

**Scenario**: Deploy to Raspberry Pi for real-time inference

**Steps**:
1. Train lightweight model (YOLO11n) → TrainingJob
2. Export to TFLite with full integer quantization → ExportJob
3. Download export package
4. Run on Raspberry Pi with TFLite runtime

**Export Configuration**:
```json
{
  "export_format": "tflite",
  "export_config": {
    "quantization": "full_integer",
    "supported_ops": ["TFLITE_BUILTINS"]
  },
  "optimization_config": {
    "quantization": {
      "method": "static",
      "dtype": "qint8",
      "calibration_dataset_id": "dataset-uuid",
      "calibration_samples": 100
    }
  }
}
```

## Implementation Phases

### Phase 1: Core Export & Platform Endpoints (MVP)

**Timeline**: 3-4 weeks

**Goals**:
- Export to ONNX, TensorRT, CoreML, TFLite, TorchScript
- Self-hosted download deployment
- **Platform inference endpoints** (Triton/TorchServe)
- Basic quantization (dynamic)
- Runtime wrappers (Python, C++, Swift, Kotlin)
- Comprehensive metadata
- 3-tier execution support
- Framework capability matrix

**Deliverables**:
- ExportJob, DeploymentTarget, DeploymentHistory models
- Export API endpoints
- Platform endpoint infrastructure
- Export scripts for all frameworks
- Frontend: Export wizard + Deployment dashboard

### Phase 2: Advanced Features

**Timeline**: 4-5 weeks

**Goals**:
- Static quantization with calibration
- Structured pruning
- Validation after export
- Container deployment (Platform registry)
- Usage analytics for platform endpoints
- Deployment scaling UI

**Deliverables**:
- Advanced optimization algorithms
- Validation pipeline
- Docker registry integration
- Enhanced monitoring

### Phase 3: Production Optimization

**Timeline**: 6-8 weeks

**Goals**:
- Quantization-Aware Training (QAT)
- Knowledge distillation
- Unstructured pruning
- Multi-model deployments
- A/B testing for platform endpoints

**Deliverables**:
- Retraining infrastructure for QAT
- Distillation workflow
- Advanced deployment strategies

## Implementation Status

**Phase 1 Implementation**: ✅ **COMPLETED** (2025-11-16)

**Progress**: 89/100 tasks (89%) - Core implementation complete

### Completed Components

#### Backend (100%)
- ✅ **Database Models** (`platform/backend/app/db/models.py`):
  - `ExportJob` model with 6 enums (ExportFormat, ExportJobStatus, ExportStrategy, OptimizationLevel, QuantizationType, PruningType)
  - `DeploymentTarget` model with usage tracking and resource management
  - `DeploymentHistory` model for audit logs

- ✅ **API Endpoints** (`platform/backend/app/api/export.py`, `inference.py`):
  - `POST /api/v1/export/jobs` - Create export job
  - `GET /api/v1/export/jobs/{id}` - Get export job status
  - `GET /api/v1/export/training/{id}/exports` - List exports for training job
  - `GET /api/v1/export/{id}/download` - Generate presigned download URL
  - `DELETE /api/v1/export/jobs/{id}` - Delete export job
  - `GET /api/v1/export/capabilities` - Get framework export capabilities
  - `POST /api/v1/deployments` - Create deployment
  - `GET /api/v1/deployments/{id}` - Get deployment details
  - `GET /api/v1/deployments/training/{id}` - List deployments for training job
  - `PATCH /api/v1/deployments/{id}/activate` - Activate deployment
  - `PATCH /api/v1/deployments/{id}/deactivate` - Deactivate deployment
  - `DELETE /api/v1/deployments/{id}` - Delete deployment
  - `POST /v1/infer/{deployment_id}` - Real-time inference with Bearer token auth
  - `GET /v1/deployments/{deployment_id}/health` - Health check
  - `GET /v1/deployments/{deployment_id}/usage` - Usage statistics

- ✅ **Export Subprocess Integration** (`platform/backend/app/utils/export_subprocess.py`):
  - Subprocess execution with monitoring
  - K8s Job execution (planned)
  - S3 upload after export
  - Error handling and logging

- ✅ **Inference Engine** (`platform/backend/app/utils/inference_engine.py`):
  - ONNX Runtime integration with GPU support (CUDA + CPU providers)
  - Image preprocessing (letterbox resize, normalization)
  - Postprocessing (NMS for detection)
  - Model caching by deployment_id
  - Usage tracking

#### Trainer (75% - Ultralytics Complete)
- ✅ **Ultralytics Export Script** (`platform/trainers/ultralytics/export.py` - 606 lines):
  - CLI with env var support (K8s compatible)
  - Multi-format export: ONNX, TensorRT, CoreML, TFLite, TorchScript, OpenVINO
  - Metadata generation with preprocessing/postprocessing specs
  - Validation with ONNX Runtime
  - S3 upload integration

- ✅ **Runtime Wrappers** (`platform/trainers/ultralytics/runtimes/`):
  - **Python** (670 lines): Complete YOLOInference class with preprocessing, inference, postprocessing
  - **C++** (850 lines): ONNXInference class with OpenCV integration
  - **Swift** (750 lines): CoreML wrapper for iOS deployment
  - **Kotlin** (700 lines): TFLite wrapper for Android deployment

- ⏸️ **Timm Export** (Planned): ONNX, TorchScript, CoreML
- ⏸️ **HuggingFace Export** (Planned): ONNX via optimum, TensorRT

#### Frontend (100%)
- ✅ **TrainingPanel Integration** (`platform/frontend/components/TrainingPanel.tsx`):
  - New "📦 Export & Deploy" tab
  - Modal state management

- ✅ **Export Components** (`platform/frontend/components/export/`):
  - `ExportJobCard.tsx` (205 lines): Status badges, format badges, download/deploy/delete actions
  - `ExportJobList.tsx` (189 lines): Auto-refresh polling (3s), download handler, delete confirmation
  - `CreateExportModal.tsx` (700+ lines): 3-step wizard (Format → Options → Review)

- ✅ **Deployment Components**:
  - `DeploymentCard.tsx` (205 lines): Type-specific display, copy-to-clipboard, usage stats
  - `DeploymentList.tsx` (200+ lines): Filter by type/status, activate/deactivate handlers
  - `CreateDeploymentModal.tsx` (650+ lines): 3-step wizard (Export → Type → Config)

- ✅ **Inference Test**:
  - `InferenceTestPanel.tsx` (390 lines): Drag & drop upload, sliders, canvas visualization

#### Documentation (100% Core Design)
- ✅ **EXPORT_CONVENTION.md** (`docs/EXPORT_CONVENTION.md` - 450+ lines):
  - **CRITICAL**: Convention-based export design for dependency isolation
  - Design rationale: Why not shared base module
  - Export script convention (CLI, output files, exit codes, metadata schema)
  - Implementation guide for new trainers (50-100 lines of work)
  - Format-specific guidelines
  - FAQ for common questions

- ✅ **export_template.py** (`docs/examples/export_template.py` - 400+ lines):
  - Copy-paste ready reference implementation
  - Framework-specific function stubs with examples
  - Standard workflow following convention

### Remaining Work (11%)

#### Documentation (3 tasks)
- [ ] Update this document with implementation status ← **IN PROGRESS**
- [ ] Create `platform/trainers/ultralytics/EXPORT_GUIDE.md`
- [ ] Update CLAUDE.md with export/deployment section

#### Testing (11 tasks)
- [ ] Unit tests (ExportJob CRUD, DeploymentTarget CRUD, capability detection)
- [ ] Integration tests (Export workflow, Platform endpoint deployment)
- [ ] E2E tests (UI → Backend → Trainer → S3 flow)

#### K8s Job Templates (3 tasks)
- [ ] Export job K8s template with resource limits
- [ ] Deployment job K8s template
- [ ] Validation job K8s template

### Key Design Decisions

#### 1. Convention-Based Export (CRITICAL)

**Decision**: Use convention + template instead of shared base module

**Rationale**:
- Each trainer runs in isolated Docker container with independent dependencies
- Shared base module would break dependency isolation
- Only ~10% of export code is truly duplicatable
- Convention-based approach maintains complete isolation while reducing duplication

**Impact**:
- New trainer implementation: 50-100 lines (vs 600 lines without template)
- Complete dependency isolation preserved
- See `docs/EXPORT_CONVENTION.md` for full specification

#### 2. Platform Inference Endpoint

**Decision**: ONNX Runtime with Bearer token authentication

**Implementation**:
- Backend serves as inference server (not separate Triton/TorchServe)
- Model caching by deployment_id for performance
- GPU support via CUDA providers
- Usage tracking built-in

**Benefits**:
- Simpler architecture (no separate inference server)
- Direct integration with deployment system
- Easier authentication and access control

#### 3. Export Subprocess Isolation

**Decision**: Export runs in subprocess with monitoring (K8s in production)

**3-Tier Execution**:
- Subprocess (local): Direct python subprocess execution
- Kind (local K8s): K8s Job with resource limits
- Production K8s: Same K8s Job with production resources

**Benefits**:
- Consistent behavior across environments
- Resource isolation
- Easy monitoring and timeout control

### API Examples

#### Create Export Job

```bash
curl -X POST http://localhost:8000/api/v1/export/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "training_job_id": 123,
    "export_format": "onnx",
    "include_validation": true,
    "export_config": {
      "opset_version": 17,
      "dynamic_axes": true
    }
  }'
```

#### Create Platform Endpoint Deployment

```bash
curl -X POST http://localhost:8000/api/v1/deployments \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "export_job_id": 456,
    "deployment_type": "platform_endpoint",
    "deployment_config": {
      "auto_activate": true
    }
  }'
```

#### Run Inference

```bash
curl -X POST http://localhost:8000/v1/infer/789 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "image": "base64_encoded_image_data",
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "max_detections": 100
  }'
```

### Troubleshooting

#### Export Job Fails with "Format not supported"

**Cause**: Framework's export script doesn't implement the requested format

**Solution**:
1. Check `/export/capabilities?training_job_id={id}` for supported formats
2. Use a different format that's supported
3. Implement the format in trainer's `export.py` (see `export_template.py`)

#### Deployment Inactive After Creation

**Cause**: `auto_activate` was set to `false` or activation failed

**Solution**:
1. Check deployment status: `GET /deployments/{id}`
2. Check error_message if status is `failed`
3. Manually activate: `PATCH /deployments/{id}/activate`

#### Inference Returns 401 Unauthorized

**Cause**: Invalid or missing API key

**Solution**:
1. Get correct API key from deployment details: `GET /deployments/{id}`
2. Use Bearer token: `Authorization: Bearer {api_key}`
3. Check deployment is `active` status

#### Export Validation Fails

**Cause**: Exported model produces different outputs than original

**Solution**:
1. Check trainer's validation logic in `export.py`
2. Review export config (opset version, quantization may affect accuracy)
3. Disable validation if acceptable: `"include_validation": false`

#### Model Cache Not Working

**Cause**: Model file changed but cache key didn't update

**Solution**:
1. Clear cache: `POST /v1/deployments/{id}/cache/clear`
2. Cache is keyed by deployment_id, so recreating deployment will use new cache

## References

- [Export Convention](../../EXPORT_CONVENTION.md) - **CRITICAL**: Convention-based export design
- [Export Template](../../examples/export_template.py) - Reference implementation template
- [Training Design](./TRAINER_DESIGN.md)
- [Inference Design](./INFERENCE_DESIGN.md)
- [Backend Design](./BACKEND_DESIGN.md)
- [3-Tier Development](../development/3_TIER_DEVELOPMENT.md)
- [Validation & Metrics Design](./VALIDATION_METRICS_DESIGN.md)
