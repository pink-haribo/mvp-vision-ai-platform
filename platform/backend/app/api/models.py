"""Model registry API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


# ============================================================
# Helper Functions
# ============================================================

def load_framework_capabilities(framework: str) -> Optional[Dict[str, Any]]:
    """
    Load model capabilities for a specific framework from R2/S3.

    Capabilities are stored in Internal Storage (Results MinIO) in the config-schemas bucket
    with prefix 'model-capabilities/'.

    Uploaded by GitHub Actions workflow (.github/workflows/upload-model-capabilities.yml)
    from platform/trainers/*/capabilities.json files.

    Args:
        framework: Framework name (e.g., "ultralytics", "timm", "huggingface")

    Returns:
        Capabilities dict, or None if not found
    """
    from app.utils.dual_storage import dual_storage

    try:
        logger.info(f"[models] Loading capabilities from Internal Storage: {framework}")

        # Get capabilities from Internal Storage (config-schemas bucket, model-capabilities/ prefix)
        capabilities_bytes = dual_storage.get_capabilities(framework)

        if not capabilities_bytes:
            logger.warning(f"[models] Capabilities not found: {framework}")
            return None

        # Parse JSON
        capabilities_dict = json.loads(capabilities_bytes.decode('utf-8'))

        logger.info(
            f"[models] Capabilities loaded: {framework} - "
            f"{len(capabilities_dict.get('models', []))} models, "
            f"{len(capabilities_dict.get('task_types', []))} task types"
        )

        return capabilities_dict

    except json.JSONDecodeError as e:
        logger.error(f"[models] Invalid JSON in capabilities file for {framework}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"[models] Error loading capabilities for {framework}: {str(e)}")
        return None


def get_all_models() -> List[Dict[str, Any]]:
    """
    Get all models from all available frameworks.

    Loads capabilities from R2/S3 for each known framework.
    If a framework's capabilities are not found, it's skipped with a warning.

    Returns:
        List of all model dictionaries
    """
    known_frameworks = ["ultralytics", "timm", "huggingface"]
    all_models = []

    for framework in known_frameworks:
        capabilities = load_framework_capabilities(framework)
        if capabilities and "models" in capabilities:
            # Add framework to each model
            for model in capabilities["models"]:
                model_copy = model.copy()
                model_copy["framework"] = framework
                all_models.append(model_copy)

    logger.info(f"[models] Loaded {len(all_models)} models from {len(known_frameworks)} frameworks")

    return all_models


def get_model_info_by_name(framework: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get specific model info by framework and model name.

    Args:
        framework: Framework name (e.g., "ultralytics")
        model_name: Model name (e.g., "yolo11n")

    Returns:
        Model info dict, or None if not found
    """
    capabilities = load_framework_capabilities(framework)
    if not capabilities or "models" not in capabilities:
        return None

    for model in capabilities["models"]:
        if model["model_name"] == model_name:
            return model

    return None


# ============================================================
# Pydantic Models
# ============================================================

class ModelInfo(BaseModel):
    """Model information schema."""
    framework: str
    model_name: str
    display_name: str
    task_types: List[str]
    description: str
    supported: bool
    parameters: Optional[Dict[str, Any]] = None  # Model-specific parameters (min, macs, etc.)


class ModelGuide(BaseModel):
    """Complete model guide information."""
    model: ModelInfo
    use_cases: Optional[List[str]] = None
    pros: Optional[List[str]] = None
    cons: Optional[List[str]] = None
    when_to_use: Optional[str] = None
    when_not_to_use: Optional[str] = None
    alternatives: Optional[List[Dict[str, str]]] = None
    recommended_settings: Optional[Dict[str, Any]] = None


# ============================================================
# API Endpoints
# ============================================================

@router.get("/list", response_model=List[ModelInfo])
async def list_models(
    framework: Optional[str] = Query(None, description="Filter by framework (ultralytics, timm, huggingface)"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    supported_only: bool = Query(True, description="Show only supported models")
):
    """
    List all available models with optional filtering.

    Model capabilities are loaded from R2/S3 storage.
    Uploaded via GitHub Actions from platform/trainers/*/capabilities.json files.

    Examples:
    - /models/list - All supported models
    - /models/list?framework=ultralytics - Only Ultralytics models
    - /models/list?task_type=detection - Only detection models
    - /models/list?supported_only=false - Include unsupported models

    Raises:
        HTTPException: If no capabilities are available for any framework
    """
    # Get all models or single framework
    if framework:
        capabilities = load_framework_capabilities(framework)
        if not capabilities:
            raise HTTPException(
                status_code=404,
                detail=f"Model capabilities for framework '{framework}' not found. "
                       f"Available frameworks: ultralytics, timm, huggingface. "
                       f"Capabilities are uploaded via GitHub Actions from platform/trainers/*/capabilities.json"
            )

        all_models_data = []
        for model in capabilities.get("models", []):
            model_copy = model.copy()
            model_copy["framework"] = framework
            all_models_data.append(model_copy)
    else:
        all_models_data = get_all_models()

    if not all_models_data:
        raise HTTPException(
            status_code=503,
            detail="No model capabilities available. "
                   "Model capabilities must be uploaded via GitHub Actions from platform/trainers/*/capabilities.json. "
                   "Check that workflows/.github/workflows/upload-model-capabilities.yml has run successfully."
        )

    # Filter models
    models = []
    for model_data in all_models_data:
        # Filter by task type
        if task_type:
            model_task_types = model_data.get("task_types", [])
            if task_type not in model_task_types:
                continue

        # Filter by supported status
        if supported_only and not model_data.get("supported", False):
            continue

        # Create ModelInfo
        try:
            model_info = ModelInfo(
                framework=model_data["framework"],
                model_name=model_data["model_name"],
                display_name=model_data["display_name"],
                task_types=model_data["task_types"],
                description=model_data["description"],
                supported=model_data.get("supported", False),
                parameters=model_data.get("parameters")
            )
            models.append(model_info)
        except Exception as e:
            logger.warning(f"Failed to create ModelInfo for {model_data.get('model_name')}: {e}")
            continue

    return models


@router.get("/get", response_model=Dict[str, Any])
async def get_model_by_query(
    framework: str = Query(..., description="Framework name (ultralytics, timm, huggingface)"),
    model_name: str = Query(..., description="Model name (e.g., yolo11n, resnet50)")
):
    """
    Get detailed information for a specific model using query parameters.

    Examples:
    - /models/get?framework=ultralytics&model_name=yolo11n
    - /models/get?framework=timm&model_name=resnet50

    Args:
        framework: Framework name
        model_name: Model name

    Returns:
        Complete model information

    Raises:
        HTTPException: If model not found or capabilities not available
    """
    model_info = get_model_info_by_name(framework, model_name)

    if not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in framework '{framework}'. "
                   f"Check /models/list?framework={framework} for available models."
        )

    # Add framework to response
    response = {
        "framework": framework,
        **model_info
    }

    return response


@router.get("/{framework}/{model_name:path}", response_model=Dict[str, Any])
async def get_model(framework: str, model_name: str):
    """
    Get detailed information for a specific model using path parameters.

    Args:
        framework: Framework name (ultralytics, timm, huggingface)
        model_name: Model name (e.g., yolo11n, resnet50)

    Returns:
        Complete model information

    Raises:
        HTTPException: If model not found or capabilities not available
    """
    model_info = get_model_info_by_name(framework, model_name)

    if not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in framework '{framework}'. "
                   f"Check /models/list?framework={framework} for available models."
        )

    # Add framework to response
    response = {
        "framework": framework,
        **model_info
    }

    return response


@router.get("/capabilities/{framework}")
async def get_framework_capabilities(framework: str):
    """
    Get complete capabilities for a specific framework.

    Returns all information including models, task types, and dataset formats.

    Args:
        framework: Framework name (ultralytics, timm, huggingface)

    Returns:
        Complete framework capabilities

    Raises:
        HTTPException: If capabilities not found
    """
    capabilities = load_framework_capabilities(framework)

    if not capabilities:
        raise HTTPException(
            status_code=404,
            detail=f"Capabilities for framework '{framework}' not found. "
                   f"Available frameworks: ultralytics, timm, huggingface. "
                   f"Capabilities are uploaded via GitHub Actions from platform/trainers/*/capabilities.json"
        )

    return capabilities
