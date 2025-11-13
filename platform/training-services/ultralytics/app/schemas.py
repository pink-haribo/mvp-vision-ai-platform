"""
Request/Response Schemas
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    """Request schema for starting training."""

    job_id: str = Field(..., description="Unique job ID from Backend")
    config: Dict[str, Any] = Field(
        ...,
        description="Training configuration",
        examples=[
            {
                "model": "yolo11n",
                "task": "detect",
                "epochs": 50,
                "batch": 16,
                "imgsz": 640,
            }
        ],
    )
    dataset_s3_uri: str = Field(
        ...,
        description="S3 URI to dataset",
        examples=["s3://vision-platform/datasets/coco8/"],
    )
    callback_url: str = Field(
        ...,
        description="Backend callback URL for status updates",
        examples=["http://localhost:8000/api/v1/training/jobs/{job_id}/callback"],
    )


class TrainingStartResponse(BaseModel):
    """Response when training starts."""

    status: str = "started"
    job_id: str
    message: str = "Training started in background"
