"""
Training Service Configuration

K8s Job-Ready Design:
- All config loaded from environment variables (12-factor app)
- Supports both Service mode (long-running) and Job mode (one-time execution)
- Backend API callback URL for status updates
- MLflow integration for experiment tracking
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "Ultralytics Trainer"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

    # Execution Mode (K8s Job support)
    EXECUTION_MODE: str = "service"  # "service" (long-running API) or "job" (one-time K8s Job)

    # Backend API (for callbacks)
    BACKEND_API_URL: str = "http://localhost:8000"  # Backend API base URL
    BACKEND_API_KEY: str = ""  # Optional API key for Backend authentication

    # S3 Storage (MinIO/S3/R2)
    S3_ENDPOINT: str = "http://localhost:9000"  # MinIO endpoint
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"
    BUCKET_NAME: str = "vision-platform"
    S3_REGION: str = "us-east-1"  # For AWS S3

    # MLflow Integration
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"  # MLflow server URL
    MLFLOW_EXPERIMENT_NAME: str = "vision-training"  # Default experiment name
    MLFLOW_ENABLE: bool = True  # Enable/disable MLflow logging

    # Training
    WORKSPACE_DIR: str = "/tmp/training"  # Temporary workspace for downloads
    CHECKPOINT_INTERVAL: int = 5  # Save checkpoint every N epochs
    CALLBACK_INTERVAL: int = 1  # Send progress update every N epochs
    CALLBACK_RETRY_ATTEMPTS: int = 3  # Retry callback requests N times
    CALLBACK_RETRY_DELAY: int = 2  # Initial retry delay in seconds (exponential backoff)
    DEFAULT_BATCH_SIZE: int = 16  # Default batch size for training
    DEFAULT_IMAGE_SIZE: int = 640  # Default image size for training

    # K8s Job specific (only used in Job mode)
    POD_NAME: str = ""  # Injected by K8s (metadata.name)
    POD_NAMESPACE: str = "default"  # Injected by K8s (metadata.namespace)
    JOB_NAME: str = ""  # Injected by K8s Job


settings = Settings()
