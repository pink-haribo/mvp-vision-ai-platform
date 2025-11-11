"""
Application Configuration

Uses pydantic-settings for environment variable management.
"""

from typing import List

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
    APP_NAME: str = "Platform Backend"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./platform.db"  # Default to SQLite for dev

    # S3 Storage (MinIO for dev, R2/S3 for production)
    S3_ENDPOINT: str = "http://localhost:9000"  # MinIO endpoint
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"
    BUCKET_NAME: str = "vision-platform"

    # Training Service
    TRAINING_MODE: str = "subprocess"  # subprocess | kubernetes
    TRAINER_SERVICE_URL: str = "http://localhost:8001"  # For kubernetes mode
    TRAINER_SUBPROCESS_PATH: str = "../training-services/ultralytics"  # For subprocess mode

    # Kubernetes (for K8s mode)
    KUBE_NAMESPACE: str = "platform"
    KUBE_CONFIG_PATH: str | None = None  # None = use in-cluster config

    # Backend URL (for callbacks)
    BACKEND_BASE_URL: str = "http://localhost:8000"

    # Authentication (JWT)
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production-please"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60


settings = Settings()
