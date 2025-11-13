"""Model registry API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import requests
import os

# Import model registries (optional in production API mode)
import sys
from pathlib import Path

# Try to import model_registry (only available in local/subprocess mode)
try:
    # Add training directory to path
    # In development: project_root/mvp/training
    training_dir = Path(__file__).parent.parent.parent.parent / "training"
    if training_dir.exists():
        sys.path.insert(0, str(training_dir))

    from model_registry import (
        TIMM_MODEL_REGISTRY,
        ULTRALYTICS_MODEL_REGISTRY,
        HUGGINGFACE_MODEL_REGISTRY,
        get_all_models,
        get_model_info as get_registry_model_info
    )
    MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    # Production API mode: model_registry not available
    # Will fetch models from Training Services instead
    MODEL_REGISTRY_AVAILABLE = False

    # Static model definitions for production (minimal set)
    STATIC_MODELS = [
        # timm models
        {
            "framework": "timm",
            "model_name": "resnet50",
            "display_name": "ResNet-50",
            "description": "Deep residual network with 50 layers",
            "params": "25.6M",
            "input_size": 224,
            "task_types": ["image_classification"],
            "pretrained_available": True,
            "recommended_batch_size": 32,
            "recommended_lr": 0.001,
            "tags": ["cnn", "popular"],
            "priority": 1
        },
        {
            "framework": "timm",
            "model_name": "resnet18",
            "display_name": "ResNet-18",
            "description": "Lightweight residual network with 18 layers",
            "params": "11.7M",
            "input_size": 224,
            "task_types": ["image_classification"],
            "pretrained_available": True,
            "recommended_batch_size": 64,
            "recommended_lr": 0.001,
            "tags": ["cnn", "fast"],
            "priority": 1
        },
        {
            "framework": "timm",
            "model_name": "efficientnet_b0",
            "display_name": "EfficientNet-B0",
            "description": "Efficient neural network with balanced scaling",
            "params": "5.3M",
            "input_size": 224,
            "task_types": ["image_classification"],
            "pretrained_available": True,
            "recommended_batch_size": 32,
            "recommended_lr": 0.001,
            "tags": ["efficient", "popular"],
            "priority": 1
        },
        # ultralytics models
        {
            "framework": "ultralytics",
            "model_name": "yolo11n",
            "display_name": "YOLO11n (Nano)",
            "description": "Ultra-fast YOLO model for real-time detection",
            "params": "2.6M",
            "input_size": 640,
            "task_types": ["object_detection", "instance_segmentation"],
            "pretrained_available": True,
            "recommended_batch_size": 16,
            "recommended_lr": 0.01,
            "tags": ["fast", "realtime"],
            "priority": 1
        },
        {
            "framework": "ultralytics",
            "model_name": "yolo11s",
            "display_name": "YOLO11s (Small)",
            "description": "Balanced YOLO model for accuracy and speed",
            "params": "9.4M",
            "input_size": 640,
            "task_types": ["object_detection", "instance_segmentation"],
            "pretrained_available": True,
            "recommended_batch_size": 16,
            "recommended_lr": 0.01,
            "tags": ["balanced", "popular"],
            "priority": 1
        },
        {
            "framework": "ultralytics",
            "model_name": "yolo11m",
            "display_name": "YOLO11m (Medium)",
            "description": "High-accuracy YOLO model",
            "params": "20.1M",
            "input_size": 640,
            "task_types": ["object_detection", "instance_segmentation"],
            "pretrained_available": True,
            "recommended_batch_size": 8,
            "recommended_lr": 0.01,
            "tags": ["accurate"],
            "priority": 2
        },
    ]

    def fetch_models_from_service(service_url: str, timeout: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch models from a Training Service.

        Args:
            service_url: Base URL of the Training Service
            timeout: Request timeout in seconds

        Returns:
            List of model dictionaries, empty list if service unavailable
        """
        try:
            response = requests.get(f"{service_url}/models/list", timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            print(f"[WARNING] Failed to fetch models from {service_url}: {e}")

        return []

    def get_all_models():
        """
        Get all models from Training Services or static definitions.

        In production (Railway), fetches models from Training Services via HTTP.
        Falls back to static definitions if services unavailable.
        """
        models = []

        # Try to fetch from Training Services
        training_services = {
            "timm": os.getenv("TIMM_SERVICE_URL"),
            "ultralytics": os.getenv("ULTRALYTICS_SERVICE_URL"),
            "huggingface": os.getenv("HUGGINGFACE_SERVICE_URL"),
        }

        services_available = False
        for framework, service_url in training_services.items():
            if service_url:
                service_models = fetch_models_from_service(service_url)
                if service_models:
                    models.extend(service_models)
                    services_available = True

        # Fallback to static models if no services available
        if not services_available:
            print("[INFO] No Training Services available, using static model definitions")
            models = STATIC_MODELS

        return models

    def get_registry_model_info(framework: str, model_name: str):
        """
        Get specific model info by framework and name.

        Tries Training Services first, falls back to static definitions.
        """
        # Try to fetch from Training Service
        training_services = {
            "timm": os.getenv("TIMM_SERVICE_URL"),
            "ultralytics": os.getenv("ULTRALYTICS_SERVICE_URL"),
            "huggingface": os.getenv("HUGGINGFACE_SERVICE_URL"),
        }

        service_url = training_services.get(framework)
        if service_url:
            try:
                response = requests.get(f"{service_url}/models/{model_name}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # Remove framework and model_name from response (they're redundant)
                    data.pop("framework", None)
                    data.pop("model_name", None)
                    return data
            except Exception as e:
                print(f"[WARNING] Failed to fetch model {model_name} from {service_url}: {e}")

        # Fallback to static definitions
        for model in STATIC_MODELS:
            if model["framework"] == framework and model["model_name"] == model_name:
                return model

        return None

router = APIRouter(prefix="/models", tags=["models"])


# ============================================================
# Pydantic Models
# ============================================================

class ModelInfo(BaseModel):
    """Model information schema."""
    framework: str
    model_name: str
    display_name: str
    description: str
    params: str
    input_size: int
    task_types: List[str]  # Changed from task_type to task_types (plural, array)
    pretrained_available: bool
    recommended_batch_size: int
    recommended_lr: float
    tags: List[str]
    priority: int


class ModelBenchmark(BaseModel):
    """Model benchmark performance."""
    # Will contain different fields based on task type
    pass


class ModelGuide(BaseModel):
    """Complete model guide information."""
    model: ModelInfo
    benchmark: Dict[str, Any]
    use_cases: List[str]
    pros: List[str]
    cons: List[str]
    when_to_use: str
    when_not_to_use: Optional[str] = None
    alternatives: List[Dict[str, str]]
    recommended_settings: Dict[str, Any]
    real_world_examples: Optional[List[Dict[str, Any]]] = None
    special_features: Optional[Dict[str, Any]] = None  # For YOLO-World, etc.


# ============================================================
# API Endpoints
# ============================================================

@router.get("/list", response_model=List[ModelInfo])
async def list_models(
    framework: Optional[str] = Query(None, description="Filter by framework (timm, ultralytics, huggingface)"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    priority: Optional[int] = Query(None, ge=0, le=2, description="Filter by priority (0=P0, 1=P1, 2=P2)")
):
    """
    List all available models with optional filtering.

    Examples:
    - /models/list - All models
    - /models/list?framework=timm - Only timm models
    - /models/list?framework=huggingface - Only HuggingFace models
    - /models/list?task_type=object_detection - Only detection models
    - /models/list?task_type=super_resolution - Only super-resolution models
    - /models/list?tags=p0,latest - P0 and latest models
    - /models/list?priority=0 - Only P0 models
    """
    all_models_data = get_all_models()

    # Convert to ModelInfo objects
    models = []
    for model_data in all_models_data:
        # Filter by framework
        if framework and model_data["framework"] != framework:
            continue

        # Filter by task type
        if task_type:
            model_task_types = model_data.get("task_types", [])
            if task_type not in model_task_types:
                continue

        # Filter by priority
        if priority is not None and model_data.get("priority") != priority:
            continue

        # Filter by tags
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
            model_tags = model_data.get("tags", [])
            if not any(tag in model_tags for tag in tag_list):
                continue

        # Create ModelInfo
        try:
            model_info = ModelInfo(
                framework=model_data["framework"],
                model_name=model_data["model_name"],
                display_name=model_data["display_name"],
                description=model_data["description"],
                params=model_data["params"],
                input_size=model_data["input_size"],
                task_types=model_data["task_types"],
                pretrained_available=model_data["pretrained_available"],
                recommended_batch_size=model_data["recommended_batch_size"],
                recommended_lr=model_data["recommended_lr"],
                tags=model_data["tags"],
                priority=model_data.get("priority", 2)
            )
            models.append(model_info)
        except Exception as e:
            print(f"Warning: Failed to create ModelInfo for {model_data.get('model_name')}: {e}")
            continue

    return models


@router.get("/get", response_model=Dict[str, Any])
async def get_model_by_query(
    framework: str = Query(..., description="Framework name (timm, ultralytics, huggingface)"),
    model_name: str = Query(..., description="Model name (e.g., resnet50, yolo11n, google/vit-base-patch16-224)")
):
    """
    Get detailed information for a specific model using query parameters.

    This endpoint is preferred for HuggingFace models which contain '/' in their names.

    Examples:
    - /models/get?framework=timm&model_name=resnet50
    - /models/get?framework=huggingface&model_name=google/vit-base-patch16-224

    Args:
        framework: Framework name
        model_name: Model name

    Returns:
        Complete model information including all metadata
    """
    model_info = get_registry_model_info(framework, model_name)

    if not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in framework '{framework}'"
        )

    # Add framework and model_name to response
    response = {
        "framework": framework,
        "model_name": model_name,
        **model_info
    }

    return response


@router.get("/{framework}/{model_name:path}", response_model=Dict[str, Any])
async def get_model(framework: str, model_name: str):
    """
    Get detailed information for a specific model using path parameters.

    NOTE: For HuggingFace models with '/' in names, use /models/get endpoint instead.

    Args:
        framework: Framework name (timm, ultralytics, huggingface)
        model_name: Model name (e.g., resnet50, yolo11n, google/vit-base-patch16-224)

    Returns:
        Complete model information including all metadata
    """
    model_info = get_registry_model_info(framework, model_name)

    if not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in framework '{framework}'"
        )

    # Add framework and model_name to response
    response = {
        "framework": framework,
        "model_name": model_name,
        **model_info
    }

    return response


@router.get("/guide", response_model=Dict[str, Any])
async def get_model_guide_by_query(
    framework: str = Query(..., description="Framework name"),
    model_name: str = Query(..., description="Model name")
):
    """
    Get complete guide information for a model using query parameters.

    This endpoint is preferred for HuggingFace models which contain '/' in their names.

    Examples:
    - /models/guide?framework=timm&model_name=resnet50
    - /models/guide?framework=huggingface&model_name=google/vit-base-patch16-224

    Returns:
        Complete guide information including benchmarks, use cases, pros/cons, etc.
    """
    model_info = get_registry_model_info(framework, model_name)

    if not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in framework '{framework}'"
        )

    # Build guide response
    guide = {
        "model": {
            "framework": framework,
            "model_name": model_name,
            "display_name": model_info["display_name"],
            "description": model_info["description"],
            "params": model_info["params"],
            "input_size": model_info["input_size"],
            "task_types": model_info["task_types"],
            "tags": model_info["tags"],
        },
        "benchmark": model_info.get("benchmark", {}),
        "use_cases": model_info.get("use_cases", []),
        "pros": model_info.get("pros", []),
        "cons": model_info.get("cons", []),
        "when_to_use": model_info.get("when_to_use", ""),
        "when_not_to_use": model_info.get("when_not_to_use"),
        "alternatives": model_info.get("alternatives", []),
        "recommended_settings": model_info.get("recommended_settings", {}),
        "real_world_examples": model_info.get("real_world_examples", []),
        "special_features": model_info.get("special_features"),  # For YOLO-World
    }

    return guide


@router.get("/{framework}/{model_name:path}/guide", response_model=Dict[str, Any])
async def get_model_guide(framework: str, model_name: str):
    """
    Get complete guide information for a model (for ModelGuideDrawer UI).

    NOTE: For HuggingFace models with '/' in names, use /models/guide endpoint instead.

    This includes all fields needed for the guide panel:
    - Quick stats (benchmark)
    - Usage guidance (pros, cons, when to use)
    - Similar models (alternatives)
    - Recommended settings
    - Real-world examples
    - Special features (for YOLO-World, etc.)

    Args:
        framework: Framework name
        model_name: Model name

    Returns:
        Complete guide information
    """
    model_info = get_registry_model_info(framework, model_name)

    if not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in framework '{framework}'"
        )

    # Build guide response
    guide = {
        "model": {
            "framework": framework,
            "model_name": model_name,
            "display_name": model_info["display_name"],
            "description": model_info["description"],
            "params": model_info["params"],
            "input_size": model_info["input_size"],
            "task_types": model_info["task_types"],
            "tags": model_info["tags"],
        },
        "benchmark": model_info.get("benchmark", {}),
        "use_cases": model_info.get("use_cases", []),
        "pros": model_info.get("pros", []),
        "cons": model_info.get("cons", []),
        "when_to_use": model_info.get("when_to_use", ""),
        "when_not_to_use": model_info.get("when_not_to_use"),
        "alternatives": model_info.get("alternatives", []),
        "recommended_settings": model_info.get("recommended_settings", {}),
        "real_world_examples": model_info.get("real_world_examples", []),
        "special_features": model_info.get("special_features"),  # For YOLO-World
    }

    return guide


@router.get("/compare")
async def compare_models(
    models: str = Query(..., description="Comma-separated model specifications (framework:model_name)")
):
    """
    Compare multiple models side-by-side.

    Examples:
    - /models/compare?models=timm:resnet50,ultralytics:yolo11n,timm:efficientnetv2_s
    - /models/compare?models=timm:resnet50,huggingface:google/vit-base-patch16-224

    Returns:
        Comparison data for specified models
    """
    model_specs = models.split(",")
    comparison_data = []

    for spec in model_specs:
        try:
            framework, model_name = spec.strip().split(":")
            model_info = get_registry_model_info(framework, model_name)

            if model_info:
                comparison_data.append({
                    "framework": framework,
                    "model_name": model_name,
                    "display_name": model_info["display_name"],
                    "params": model_info["params"],
                    "task_types": model_info["task_types"],
                    "benchmark": model_info.get("benchmark", {}),
                    "tags": model_info["tags"],
                })
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model specification: '{spec}'. Expected format: 'framework:model_name'"
            )

    if not comparison_data:
        raise HTTPException(
            status_code=404,
            detail="No valid models found for comparison"
        )

    return {
        "models": comparison_data,
        "count": len(comparison_data)
    }
