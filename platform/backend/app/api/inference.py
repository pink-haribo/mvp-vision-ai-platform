"""
Platform Inference API

Provides real-time inference for deployed models.
"""

import logging
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models
from app.schemas import inference as schemas
from app.utils.inference_engine import get_inference_engine
from app.utils.dual_storage import dual_storage

logger = logging.getLogger(__name__)

router = APIRouter()


async def verify_api_key(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> models.DeploymentTarget:
    """
    Verify API key and return associated deployment.

    Args:
        authorization: Authorization header (Bearer token)
        db: Database session

    Returns:
        DeploymentTarget instance

    Raises:
        HTTPException: If authentication fails
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    api_key = authorization.replace("Bearer ", "")

    # Find deployment by API key
    deployment = db.query(models.DeploymentTarget).filter(
        models.DeploymentTarget.api_key == api_key
    ).first()

    if not deployment:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if deployment.status != models.DeploymentStatus.ACTIVE:
        raise HTTPException(
            status_code=403,
            detail=f"Deployment is not active (status: {deployment.status})"
        )

    return deployment


@router.post(
    "/v1/infer/{deployment_id}",
    response_model=schemas.InferenceResponse,
    summary="Run inference on deployed model",
    description="Submit an image for real-time inference using a deployed model"
)
async def infer(
    deployment_id: int,
    request: schemas.InferenceRequest,
    deployment: models.DeploymentTarget = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Run inference on deployed model.

    **Authentication**: Bearer token (API key from deployment)

    **Request**:
    - `image`: Base64 encoded image
    - `confidence_threshold`: Confidence threshold (default: 0.25)
    - `iou_threshold`: IoU threshold for NMS (default: 0.45)
    - `max_detections`: Maximum number of detections (default: 300)

    **Response**:
    - `detections`: Array of detected objects (for detection task)
    - `poses`: Array of poses (for pose task)
    - `classification`: Classification result (for classification task)
    - `inference_time_ms`: Inference time in milliseconds
    - `model_info`: Model metadata

    **Usage Tracking**:
    - Increments `request_count`
    - Updates `total_inference_time_ms`
    - Updates `avg_latency_ms`
    - Sets `last_request_at`
    """
    # Verify deployment ID matches API key
    if deployment.id != deployment_id:
        raise HTTPException(
            status_code=403,
            detail="Deployment ID does not match API key"
        )

    logger.info(f"[Inference] Request for deployment {deployment_id}")

    try:
        # Get export job
        export_job = db.query(models.ExportJob).filter(
            models.ExportJob.id == deployment.export_job_id
        ).first()

        if not export_job:
            raise HTTPException(status_code=404, detail="Export job not found")

        if export_job.status != models.ExportJobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Export job is not completed (status: {export_job.status})"
            )

        # Check export format
        if export_job.export_format not in [models.ExportFormat.ONNX, models.ExportFormat.TORCHSCRIPT]:
            raise HTTPException(
                status_code=400,
                detail=f"Export format {export_job.export_format} not supported for platform inference. "
                       f"Please use ONNX or TorchScript format."
            )

        # Download and extract export package
        model_path, metadata = await download_and_extract_model(
            deployment_id,
            export_job.export_path,
            export_job.export_format
        )

        # Run inference
        inference_engine = get_inference_engine()

        result = inference_engine.infer(
            deployment_id=deployment_id,
            model_path=model_path,
            metadata=metadata,
            image_base64=request.image,
            conf_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            max_detections=request.max_detections
        )

        # Update usage statistics
        deployment.request_count += 1
        deployment.total_inference_time_ms += result['inference_time_ms']
        deployment.avg_latency_ms = deployment.total_inference_time_ms / deployment.request_count
        deployment.last_request_at = datetime.utcnow()

        db.commit()

        logger.info(
            f"[Inference] Completed for deployment {deployment_id} "
            f"in {result['inference_time_ms']:.2f}ms"
        )

        # Format response
        response = schemas.InferenceResponse(
            deployment_id=deployment_id,
            task_type=result['task_type'],
            inference_time_ms=result['inference_time_ms'],
            detections=[
                schemas.Detection(
                    class_id=d['class_id'],
                    class_name=d['class_name'],
                    confidence=d['confidence'],
                    bbox=schemas.BoundingBox(**d['bbox'])
                )
                for d in result.get('detections', [])
            ] if 'detections' in result else None,
            poses=None,  # TODO: Add pose support
            classification=None,  # TODO: Add classification support
            model_info=result['model_info']
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Inference] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


async def download_and_extract_model(
    deployment_id: int,
    export_path: str,
    export_format: models.ExportFormat
) -> tuple[Path, dict]:
    """
    Download and extract model from S3.

    Args:
        deployment_id: Deployment ID for caching
        export_path: S3 URI to export package
        export_format: Export format

    Returns:
        model_path: Path to extracted model file
        metadata: Model metadata dictionary
    """
    storage_client = dual_storage

    # Create cache directory
    cache_dir = Path(tempfile.gettempdir()) / "platform_inference_cache" / str(deployment_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    model_files = list(cache_dir.glob("*.onnx")) + list(cache_dir.glob("*.pt"))
    metadata_file = cache_dir / "metadata.json"

    if model_files and metadata_file.exists():
        logger.info(f"[Inference] Using cached model for deployment {deployment_id}")
        model_path = model_files[0]
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return model_path, metadata

    # Download export package
    logger.info(f"[Inference] Downloading export package from: {export_path}")

    # Parse S3 URI: s3://bucket/key
    s3_uri_parts = export_path.replace("s3://", "").split("/", 1)
    bucket = s3_uri_parts[0]
    key = s3_uri_parts[1]

    # Download to temp file
    temp_zip = cache_dir / "export_package.zip"

    storage_client.internal_client.client.download_file(
        Bucket=bucket,
        Key=key,
        Filename=str(temp_zip)
    )

    logger.info(f"[Inference] Downloaded export package: {temp_zip.stat().st_size / 1024 / 1024:.2f} MB")

    # Extract zip
    with zipfile.ZipFile(temp_zip, 'r') as zipf:
        zipf.extractall(cache_dir)

    # Remove zip file
    temp_zip.unlink()

    # Find model file
    model_extensions = {
        models.ExportFormat.ONNX: "*.onnx",
        models.ExportFormat.TORCHSCRIPT: "*.pt"
    }

    pattern = model_extensions.get(export_format, "*.onnx")
    model_files = list(cache_dir.glob(pattern))

    if not model_files:
        raise FileNotFoundError(f"No model file found with pattern: {pattern}")

    model_path = model_files[0]

    # Load metadata
    if not metadata_file.exists():
        raise FileNotFoundError("metadata.json not found in export package")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    logger.info(f"[Inference] Model extracted: {model_path}")

    return model_path, metadata


@router.get(
    "/v1/deployments/{deployment_id}/health",
    summary="Health check for deployment",
    description="Check if deployment is active and ready for inference"
)
async def health_check(
    deployment_id: int,
    db: Session = Depends(get_db)
):
    """
    Health check endpoint for deployment.

    Returns:
    - `status`: "healthy" or "unhealthy"
    - `deployment_id`: Deployment ID
    - `deployment_status`: Current deployment status
    - `model_loaded`: Whether model is loaded in cache
    """
    deployment = db.query(models.DeploymentTarget).filter(
        models.DeploymentTarget.id == deployment_id
    ).first()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # Check if model is in cache
    inference_engine = get_inference_engine()
    model_loaded = deployment_id in inference_engine.model_cache

    is_healthy = deployment.status == models.DeploymentStatus.ACTIVE

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "deployment_id": deployment_id,
        "deployment_status": deployment.status.value,
        "model_loaded": model_loaded,
        "request_count": deployment.request_count,
        "avg_latency_ms": deployment.avg_latency_ms
    }


@router.post(
    "/v1/deployments/{deployment_id}/cache/clear",
    summary="Clear model cache for deployment",
    description="Clear cached model from memory (admin only)"
)
async def clear_cache(
    deployment_id: int,
    db: Session = Depends(get_db)
):
    """
    Clear model cache for deployment.

    This endpoint is useful for:
    - Freeing up memory
    - Forcing model reload after update
    - Troubleshooting inference issues
    """
    deployment = db.query(models.DeploymentTarget).filter(
        models.DeploymentTarget.id == deployment_id
    ).first()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    inference_engine = get_inference_engine()
    inference_engine.clear_cache(deployment_id)

    logger.info(f"[Inference] Cache cleared for deployment {deployment_id}")

    return {
        "message": f"Cache cleared for deployment {deployment_id}",
        "deployment_id": deployment_id
    }


@router.get(
    "/v1/deployments/{deployment_id}/usage",
    response_model=schemas.UsageStats,
    summary="Get usage statistics for deployment",
    description="Get detailed usage statistics including request count and latency"
)
async def get_usage_stats(
    deployment_id: int,
    deployment: models.DeploymentTarget = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get usage statistics for deployment."""
    if deployment.id != deployment_id:
        raise HTTPException(
            status_code=403,
            detail="Deployment ID does not match API key"
        )

    return schemas.UsageStats(
        deployment_id=deployment_id,
        request_count=deployment.request_count,
        total_inference_time_ms=deployment.total_inference_time_ms or 0.0,
        avg_latency_ms=deployment.avg_latency_ms or 0.0,
        last_request_at=deployment.last_request_at.isoformat() if deployment.last_request_at else None
    )
