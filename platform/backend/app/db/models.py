"""
Database Models

SQLAlchemy models for the Platform backend.
All models follow the design from IMPLEMENTATION_PLAN.md.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import Column, String, DateTime, Float, Integer, Text, Enum, JSON, Boolean

from app.db.session import Base


class JobStatus(PyEnum):
    """Training job status enum."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class User(Base):
    """
    User Model

    Represents a user account in the platform.
    Supports authentication and authorization.
    """

    __tablename__ = "users"

    # Primary key (UUID stored as string for SQLite compatibility)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)

    # Profile
    full_name = Column(String(255), nullable=True)

    # Permissions
    is_active = Column(Boolean, nullable=False, default=True)
    is_superuser = Column(Boolean, nullable=False, default=False)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class Project(Base):
    """
    Project Model

    Represents a project that groups datasets and training jobs.
    Each project belongs to a user.
    """

    __tablename__ = "projects"

    # Primary key (UUID stored as string for SQLite compatibility)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Project info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Owner
    owner_id = Column(String(36), nullable=False, index=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name}, owner_id={self.owner_id})>"


class TrainingJob(Base):
    """
    Training Job Model

    Represents a training job submitted by a user.
    The job is executed by a Training Service (via HTTP or K8s).
    """

    __tablename__ = "training_jobs"

    # Primary key (UUID stored as string for SQLite compatibility)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # User info
    user_id = Column(String(255), nullable=False, index=True)

    # Project (optional - jobs can be created without a project)
    project_id = Column(String(36), nullable=True, index=True)

    # Model config
    model_name = Column(String(255), nullable=False)  # e.g., "yolo11n", "resnet50"
    framework = Column(String(100), nullable=False)  # e.g., "ultralytics", "timm"

    # Dataset (S3 URI only, no local paths!)
    dataset_s3_uri = Column(String(1024), nullable=False)

    # Training config (JSON)
    config = Column(JSON, nullable=True)  # Framework-specific config

    # Status
    status = Column(
        Enum(JobStatus),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
    )
    progress = Column(Float, nullable=False, default=0.0)  # 0.0 to 1.0
    message = Column(Text, nullable=True)  # Status message

    # Callback URL (for Training Service to send updates)
    callback_url = Column(String(1024), nullable=False)

    # Results
    checkpoint_s3_uri = Column(String(1024), nullable=True)  # Best model checkpoint
    error = Column(Text, nullable=True)  # Error message if failed

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<TrainingJob(id={self.id}, model={self.model_name}, status={self.status})>"


class TrainingMetric(Base):
    """
    Training Metrics Model

    Stores metrics reported during training (loss, accuracy, etc.).
    Sent via callbacks from Training Service.
    """

    __tablename__ = "training_metrics"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to TrainingJob (UUID stored as string)
    job_id = Column(String(36), nullable=False, index=True)

    # Metric data
    epoch = Column(Integer, nullable=False)
    step = Column(Integer, nullable=True)
    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)

    # Additional metrics (JSON for flexibility)
    metrics = Column(JSON, nullable=True)  # e.g., {"val_loss": 0.5, "map": 0.7}

    # Timestamp
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<TrainingMetric(job_id={self.job_id}, epoch={self.epoch}, loss={self.loss})>"
