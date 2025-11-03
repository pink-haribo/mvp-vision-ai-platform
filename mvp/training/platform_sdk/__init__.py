"""Vision Platform Training SDK - Common utilities for all frameworks."""

from .base import (
    TrainingAdapter,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    MetricsResult,
    TaskType,
    DatasetFormat,
    ConfigSchema,
    ConfigField,
    InferenceResult,
)
from .logger import TrainingLogger
from .storage import get_model_weights, get_dataset

__version__ = "0.1.0"
__all__ = [
    "TrainingAdapter",
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "MetricsResult",
    "TaskType",
    "DatasetFormat",
    "ConfigSchema",
    "ConfigField",
    "InferenceResult",
    "TrainingLogger",
    "get_model_weights",
    "get_dataset",
]
