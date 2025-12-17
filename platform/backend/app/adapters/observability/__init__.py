"""
Observability Adapters for Phase 13

Provides pluggable adapters for multiple observability backends (ClearML, MLflow, TensorBoard, Database).
"""

from .base import ObservabilityAdapter, MetricsResult
from .database_adapter import DatabaseAdapter

# ClearMLAdapter is lazily imported to avoid requiring clearml package
# Use: from app.adapters.observability.clearml_adapter import ClearMLAdapter

# MLflowAdapter is lazily imported to avoid requiring mlflow package
# Use: from app.adapters.observability.mlflow_adapter import MLflowAdapter

__all__ = [
    "ObservabilityAdapter",
    "MetricsResult",
    "DatabaseAdapter",
]
