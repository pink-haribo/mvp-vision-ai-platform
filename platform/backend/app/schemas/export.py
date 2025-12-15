"""
Model Export and Deployment API schemas.

Pydantic models for:
- Export: Converting trained checkpoints to production formats (ONNX, TensorRT, etc.)
- Deployment: Deploying exported models (download, platform endpoint, edge, container)
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ========== Export Capabilities ==========

class ExportFormatCapability(BaseModel):
    """Export format capability for a specific framework."""
    format: str = Field(..., description="Export format (onnx, tensorrt, coreml, tflite, torchscript, openvino)")
    supported: bool = Field(..., description="Whether this format is supported by the framework")
    native_support: bool = Field(..., description="Whether framework has native export support")
    requires_conversion: bool = Field(False, description="Whether conversion is needed")
    optimization_options: List[str] = Field(default_factory=list, description="Available optimizations (quantization, pruning, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "format": "onnx",
                "supported": True,
                "native_support": True,
                "requires_conversion": False,
                "optimization_options": ["dynamic_quantization", "static_quantization"]
            }
        }


class ExportCapabilitiesResponse(BaseModel):
    """Export capabilities for a framework."""
    framework: str
    task_type: str
    supported_formats: List[ExportFormatCapability]
    default_format: str = Field(..., description="Recommended default export format")

    class Config:
        json_schema_extra = {
            "example": {
                "framework": "ultralytics",
                "task_type": "object_detection",
                "supported_formats": [
                    {
                        "format": "onnx",
                        "supported": True,
                        "native_support": True,
                        "requires_conversion": False,
                        "optimization_options": ["dynamic_quantization"]
                    }
                ],
                "default_format": "onnx"
            }
        }


# ========== Export Request/Response ==========

class ExportConfig(BaseModel):
    """Export configuration (format-specific settings)."""
    opset_version: Optional[int] = Field(None, description="ONNX opset version (default: 11)")
    dynamic_axes: Optional[Dict[str, List[int]]] = Field(None, description="Dynamic axes for ONNX export")
    embed_preprocessing: bool = Field(False, description="Embed preprocessing in exported model")
    simplify: bool = Field(True, description="Simplify ONNX model (requires onnx-simplifier)")

    # TensorRT specific
    workspace_size_gb: Optional[int] = Field(None, description="TensorRT workspace size in GB")
    fp16: bool = Field(False, description="Use FP16 precision (TensorRT)")
    int8: bool = Field(False, description="Use INT8 precision (TensorRT)")

    # CoreML specific
    minimum_deployment_target: Optional[str] = Field(None, description="iOS/macOS minimum deployment target")

    class Config:
        json_schema_extra = {
            "example": {
                "opset_version": 13,
                "dynamic_axes": {"input": [0, 2, 3]},
                "embed_preprocessing": True,
                "simplify": True
            }
        }


class OptimizationConfig(BaseModel):
    """Optimization configuration."""
    quantization_type: Optional[str] = Field(None, description="Quantization type (dynamic, static, qat)")
    calibration_dataset_size: Optional[int] = Field(100, description="Number of images for calibration (static quantization)")
    pruning_amount: Optional[float] = Field(None, ge=0.0, le=1.0, description="Pruning amount (0.0-1.0)")

    class Config:
        json_schema_extra = {
            "example": {
                "quantization_type": "dynamic",
                "calibration_dataset_size": 100
            }
        }


class ValidationConfig(BaseModel):
    """Validation configuration for export."""
    validate_outputs: bool = Field(True, description="Validate exported model outputs against original")
    tolerance: float = Field(1e-3, description="Tolerance for output validation")
    num_samples: int = Field(10, ge=1, description="Number of samples for validation")

    class Config:
        json_schema_extra = {
            "example": {
                "validate_outputs": True,
                "tolerance": 0.001,
                "num_samples": 10
            }
        }


class ExportJobRequest(BaseModel):
    """Request to export a trained model."""
    training_job_id: int = Field(..., description="Training job ID")
    export_format: str = Field(..., description="Export format (onnx, tensorrt, coreml, tflite, torchscript, openvino)")
    checkpoint_path: Optional[str] = Field(None, description="Checkpoint path (uses best if not specified)")

    export_config: Optional[ExportConfig] = Field(None, description="Format-specific export configuration")
    optimization_config: Optional[OptimizationConfig] = Field(None, description="Optimization settings")
    validation_config: Optional[ValidationConfig] = Field(None, description="Validation settings")

    set_as_default: bool = Field(False, description="Set this export as default for deployment")

    class Config:
        json_schema_extra = {
            "example": {
                "training_job_id": 123,
                "export_format": "onnx",
                "checkpoint_path": "/workspace/checkpoints/best_model.pth",
                "export_config": {
                    "opset_version": 13,
                    "simplify": True
                },
                "set_as_default": True
            }
        }


class ExportJobResponse(BaseModel):
    """Export job status and results."""
    id: int
    training_job_id: int

    # Export configuration
    export_format: str
    checkpoint_path: str
    export_path: Optional[str] = None

    # Version management
    version: int
    is_default: bool

    # Framework info
    framework: str
    task_type: str
    model_name: str

    # Configuration
    export_config: Optional[Dict[str, Any]] = None
    optimization_config: Optional[Dict[str, Any]] = None
    validation_config: Optional[Dict[str, Any]] = None

    # Status
    status: str  # pending, running, completed, failed, cancelled
    error_message: Optional[str] = None

    # Results
    export_results: Optional[Dict[str, Any]] = None
    file_size_mb: Optional[float] = None
    validation_passed: Optional[bool] = None

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ExportJobListResponse(BaseModel):
    """List of export jobs for a training job."""
    training_job_id: int
    total_count: int
    export_jobs: List[ExportJobResponse]


# ========== Deployment Request/Response ==========

class DeploymentConfig(BaseModel):
    """Deployment configuration (type-specific)."""
    # Platform endpoint specific
    auto_scale: Optional[bool] = Field(None, description="Enable auto-scaling (platform endpoint)")
    min_replicas: Optional[int] = Field(None, ge=1, description="Minimum replicas (platform endpoint)")
    max_replicas: Optional[int] = Field(None, ge=1, description="Maximum replicas (platform endpoint)")

    # Edge package specific
    target_platform: Optional[str] = Field(None, description="Target platform (ios, android, embedded)")
    runtime_wrapper: Optional[str] = Field(None, description="Runtime wrapper language (python, cpp, swift, kotlin)")

    # Container specific
    base_image: Optional[str] = Field(None, description="Base Docker image")
    port: Optional[int] = Field(None, description="Container port")

    class Config:
        json_schema_extra = {
            "example": {
                "auto_scale": True,
                "min_replicas": 1,
                "max_replicas": 5
            }
        }


class DeploymentRequest(BaseModel):
    """Request to deploy an exported model."""
    export_job_id: int = Field(..., description="Export job ID")
    deployment_type: str = Field(..., description="Deployment type (download, platform_endpoint, edge_package, container)")
    deployment_name: Optional[str] = Field(None, max_length=200, description="User-friendly deployment name (auto-generated if not provided)")
    deployment_config: Optional[DeploymentConfig] = Field(None, description="Type-specific deployment configuration")

    # Resource configuration (platform endpoint)
    cpu_limit: Optional[str] = Field(None, description="CPU limit (e.g., '2000m')")
    memory_limit: Optional[str] = Field(None, description="Memory limit (e.g., '4Gi')")
    gpu_enabled: bool = Field(False, description="Enable GPU acceleration")

    class Config:
        json_schema_extra = {
            "example": {
                "export_job_id": 456,
                "deployment_type": "platform_endpoint",
                "deployment_name": "YOLOv11n Detection API",
                "deployment_config": {
                    "auto_scale": True,
                    "min_replicas": 1,
                    "max_replicas": 3
                },
                "cpu_limit": "2000m",
                "memory_limit": "4Gi",
                "gpu_enabled": False
            }
        }


class DeploymentResponse(BaseModel):
    """Deployment status and configuration."""
    id: int
    export_job_id: int
    training_job_id: int

    # Deployment configuration
    deployment_type: str
    deployment_name: str
    deployment_config: Optional[Dict[str, Any]] = None

    # Platform endpoint specific
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None

    # Container specific
    container_image: Optional[str] = None
    container_registry: Optional[str] = None

    # Edge package specific
    package_path: Optional[str] = None
    runtime_wrapper_language: Optional[str] = None

    # Status
    status: str  # pending, deploying, active, deactivated, failed
    error_message: Optional[str] = None

    # Usage tracking
    request_count: int
    total_inference_time_ms: float
    avg_latency_ms: Optional[float] = None
    last_request_at: Optional[datetime] = None

    # Resource usage
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    gpu_enabled: bool

    # Timestamps
    created_at: datetime
    deployed_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DeploymentListResponse(BaseModel):
    """List of deployments for a training job or export job."""
    total_count: int
    deployments: List[DeploymentResponse]


class DeploymentUpdateRequest(BaseModel):
    """Request to update deployment configuration."""
    deployment_config: Optional[DeploymentConfig] = Field(None, description="Updated deployment configuration")
    cpu_limit: Optional[str] = Field(None, description="Updated CPU limit")
    memory_limit: Optional[str] = Field(None, description="Updated memory limit")


# ========== Deployment History ==========

class DeploymentHistoryResponse(BaseModel):
    """Deployment event history."""
    id: int
    deployment_id: int

    # Event information
    event_type: str  # deployed, scaled, deactivated, reactivated, updated, error
    message: str
    details: Optional[Dict[str, Any]] = None

    # User tracking
    triggered_by: Optional[int] = None

    created_at: datetime

    class Config:
        from_attributes = True


class DeploymentHistoryListResponse(BaseModel):
    """List of deployment events."""
    deployment_id: int
    total_count: int
    events: List[DeploymentHistoryResponse]


# ========== Platform Inference Endpoint ==========

class InferenceRequest(BaseModel):
    """Request to run inference on deployed model."""
    image: str = Field(..., description="Base64 encoded image")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold (detection only)")

    class Config:
        json_schema_extra = {
            "example": {
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45
            }
        }


class BoundingBox(BaseModel):
    """Bounding box for object detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_id: int
    confidence: float


class KeypointPrediction(BaseModel):
    """Keypoint prediction for pose estimation."""
    x: float
    y: float
    confidence: float
    visible: bool


class InferenceResult(BaseModel):
    """Inference result from deployed model."""
    task_type: str

    # Classification
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    top5: Optional[List[Dict[str, Any]]] = None

    # Detection
    boxes: Optional[List[BoundingBox]] = None

    # Segmentation
    mask_url: Optional[str] = None

    # Pose
    keypoints: Optional[List[KeypointPrediction]] = None

    # Performance
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float


class InferenceResponse(BaseModel):
    """Response from platform inference endpoint."""
    deployment_id: int
    deployment_name: str
    results: InferenceResult

    class Config:
        json_schema_extra = {
            "example": {
                "deployment_id": 789,
                "deployment_name": "YOLOv11n Detection API",
                "results": {
                    "task_type": "object_detection",
                    "boxes": [
                        {
                            "x1": 100.5,
                            "y1": 200.3,
                            "x2": 350.7,
                            "y2": 450.2,
                            "class_name": "person",
                            "class_id": 0,
                            "confidence": 0.92
                        }
                    ],
                    "inference_time_ms": 15.3,
                    "preprocessing_time_ms": 2.1,
                    "postprocessing_time_ms": 1.8
                }
            }
        }
