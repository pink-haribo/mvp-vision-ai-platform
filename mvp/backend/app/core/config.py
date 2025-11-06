"""Application configuration."""

import os
from pathlib import Path
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

    # Storage Paths (will be converted to absolute paths)
    # Note: Paths are relative to project root (mvp-vision-ai-platform/)
    UPLOAD_DIR: str = "./data/uploads"
    OUTPUT_DIR: str = "./data/outputs"
    MODEL_DIR: str = "./data/models"
    LOG_DIR: str = "./data/logs"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Convert relative paths to absolute paths based on mvp directory
        # __file__ = mvp/backend/app/core/config.py
        # parent x3 = mvp/backend -> mvp/
        project_root = Path(__file__).parent.parent.parent.absolute()

        for attr in ['UPLOAD_DIR', 'OUTPUT_DIR', 'MODEL_DIR', 'LOG_DIR']:
            path_value = getattr(self, attr)
            if not os.path.isabs(path_value):
                # Convert relative path to absolute
                abs_path = project_root / path_value.lstrip('./')
                setattr(self, attr, str(abs_path))

        # Also handle DATABASE_URL if it contains relative path
        if self.DATABASE_URL.startswith('sqlite:///./'):
            rel_path = self.DATABASE_URL.replace('sqlite:///./', '')
            abs_path = project_root / rel_path
            self.DATABASE_URL = f'sqlite:///{abs_path}'

    # Training Service URLs
    TIMM_SERVICE_URL: str = "http://localhost:8001"
    ULTRALYTICS_SERVICE_URL: str = "http://localhost:8002"
    HUGGINGFACE_SERVICE_URL: str = "http://localhost:8003"

    # Training Defaults
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 50
    DEFAULT_LEARNING_RATE: float = 0.001

    class Config:
        env_file = ".env.mvp"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
