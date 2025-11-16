"""Schemas for platform inference API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import base64


class InferenceRequest(BaseModel):
    """Request schema for platform inference."""

    image: str = Field(..., description="Base64 encoded image")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    max_detections: int = Field(300, ge=1, le=1000, description="Maximum number of detections")

    @validator('image')
    def validate_base64(cls, v):
        """Validate base64 encoding."""
        try:
            base64.b64decode(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}")


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class Detection(BaseModel):
    """Detection result."""

    class_id: int = Field(..., description="Class ID")
    class_name: Optional[str] = Field(None, description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box")


class Keypoint(BaseModel):
    """Keypoint for pose estimation."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Keypoint confidence")


class PoseDetection(BaseModel):
    """Pose detection result."""

    bbox: BoundingBox = Field(..., description="Bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    keypoints: List[Keypoint] = Field(..., description="Pose keypoints")


class ClassificationResult(BaseModel):
    """Classification result."""

    class_id: int = Field(..., description="Predicted class ID")
    class_name: Optional[str] = Field(None, description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    top_k: Optional[List[Dict[str, Any]]] = Field(None, description="Top-K predictions")


class InferenceResponse(BaseModel):
    """Response schema for platform inference."""

    deployment_id: int = Field(..., description="Deployment ID")
    task_type: str = Field(..., description="Task type (detect, segment, pose, classify)")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")

    # Task-specific results
    detections: Optional[List[Detection]] = Field(None, description="Object detections")
    poses: Optional[List[PoseDetection]] = Field(None, description="Pose detections")
    classification: Optional[ClassificationResult] = Field(None, description="Classification result")

    # Metadata
    model_info: Dict[str, Any] = Field(..., description="Model metadata")


class InferenceError(BaseModel):
    """Error response for inference."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    deployment_id: Optional[int] = Field(None, description="Deployment ID")


class UsageStats(BaseModel):
    """Usage statistics for a deployment."""

    deployment_id: int
    request_count: int
    total_inference_time_ms: float
    avg_latency_ms: float
    last_request_at: Optional[str] = None
