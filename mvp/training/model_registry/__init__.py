"""Model registry for all supported models."""

from .timm_models import TIMM_MODEL_REGISTRY, get_timm_model_info
from .ultralytics_models import ULTRALYTICS_MODEL_REGISTRY, get_ultralytics_model_info
from .huggingface_models import HUGGINGFACE_MODEL_REGISTRY, get_huggingface_model

__all__ = [
    "TIMM_MODEL_REGISTRY",
    "ULTRALYTICS_MODEL_REGISTRY",
    "HUGGINGFACE_MODEL_REGISTRY",
    "get_timm_model_info",
    "get_ultralytics_model_info",
    "get_huggingface_model",
]


def get_all_models():
    """Get all models from all registries."""
    all_models = []

    # Add timm models
    for model_name, info in TIMM_MODEL_REGISTRY.items():
        all_models.append({
            "framework": "timm",
            "model_name": model_name,
            **info
        })

    # Add ultralytics models
    for model_name, info in ULTRALYTICS_MODEL_REGISTRY.items():
        all_models.append({
            "framework": "ultralytics",
            "model_name": model_name,
            **info
        })

    # Add huggingface models
    for model_name, info in HUGGINGFACE_MODEL_REGISTRY.items():
        all_models.append({
            "framework": "huggingface",
            "model_name": model_name,
            **info
        })

    return all_models


def get_model_info(framework: str, model_name: str):
    """Get model info by framework and model name."""
    if framework == "timm":
        return get_timm_model_info(model_name)
    elif framework == "ultralytics":
        return get_ultralytics_model_info(model_name)
    elif framework == "huggingface":
        return get_huggingface_model(model_name)
    else:
        return None
