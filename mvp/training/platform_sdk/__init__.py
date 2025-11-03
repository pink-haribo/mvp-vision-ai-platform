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
]
