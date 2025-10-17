"""Application configuration."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Vision AI Training Platform - MVP"

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # Database
    DATABASE_URL: str = "sqlite:///./mvp/data/db/vision_platform.db"

    # LLM
    GOOGLE_API_KEY: str
    LLM_MODEL: str = "gemini-2.0-flash-exp"
    LLM_TEMPERATURE: float = 0.0

    # Storage Paths
    UPLOAD_DIR: str = "./mvp/data/uploads"
    OUTPUT_DIR: str = "./mvp/data/outputs"
    MODEL_DIR: str = "./mvp/data/models"
    LOG_DIR: str = "./mvp/data/logs"

    # Training Defaults
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 50
    DEFAULT_LEARNING_RATE: float = 0.001

    class Config:
        env_file = ".env.mvp"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
