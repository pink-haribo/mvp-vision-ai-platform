"""Application configuration."""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Vision AI Training Platform - MVP"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def BACKEND_CORS_ORIGINS(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    # Database
    DATABASE_URL: str = "sqlite:///./mvp/data/db/vision_platform.db"

    # LLM
    GOOGLE_API_KEY: str
    LLM_MODEL: str = "gemini-2.0-flash-exp"
    LLM_TEMPERATURE: float = 0.0
    # Security & Authentication
    JWT_SECRET: str = "your-secret-key-change-this-in-production-use-openssl-rand-hex-32"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

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
