"""
Training Service Configuration
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

    # S3 Storage
    S3_ENDPOINT: str = "http://localhost:9000"  # MinIO endpoint
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"
    BUCKET_NAME: str = "vision-platform"

    # Training
    WORKSPACE_DIR: str = "/tmp/training"  # Temporary workspace for downloads


settings = Settings()
