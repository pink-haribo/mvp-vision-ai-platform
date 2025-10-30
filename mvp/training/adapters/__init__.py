"""Training adapters for different frameworks."""

from platform_sdk import (
    TrainingAdapter,
    TaskType,
    DatasetFormat,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    MetricsResult,
)

# Conditional imports for Docker compatibility
# Each Docker image only has its framework-specific adapter
TimmAdapter = None
UltralyticsAdapter = None
TransformersAdapter = None

try:
    from .timm_adapter import TimmAdapter
except ImportError:
    pass

try:
    from .ultralytics_adapter import UltralyticsAdapter
except ImportError:
    pass

try:
    from .transformers_adapter import TransformersAdapter
except ImportError:
    pass

# Adapter registry (only includes successfully imported adapters)
ADAPTER_REGISTRY = {}
if TimmAdapter is not None:
    ADAPTER_REGISTRY['timm'] = TimmAdapter
if UltralyticsAdapter is not None:
    ADAPTER_REGISTRY['ultralytics'] = UltralyticsAdapter
if TransformersAdapter is not None:
    ADAPTER_REGISTRY['huggingface'] = TransformersAdapter

__all__ = [
    "TrainingAdapter",
    "TaskType",
    "DatasetFormat",
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "MetricsResult",
    "TimmAdapter",
    "UltralyticsAdapter",
    "TransformersAdapter",
    "ADAPTER_REGISTRY",
]
