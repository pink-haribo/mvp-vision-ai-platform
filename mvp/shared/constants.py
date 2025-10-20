"""Shared constants."""

# Supported models
SUPPORTED_MODELS = [
    "resnet50",
]

# Supported tasks
SUPPORTED_TASKS = [
    "classification",
]

# Training status
class TrainingStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# WebSocket event types
class WSEventType:
    PROGRESS = "progress"
    METRIC = "metric"
    COMPLETED = "completed"
    ERROR = "error"
