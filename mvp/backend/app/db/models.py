"""Database models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.db.database import Base


class Project(Base):
    """Project model for organizing experiments."""

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    task_type = Column(String(50), nullable=True)  # Optional: image_classification, object_detection, etc.

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    experiments = relationship("TrainingJob", back_populates="project", cascade="all, delete-orphan")


class Session(Base):
    """Chat session model."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    """Chat message model."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("Session", back_populates="messages")


class TrainingJob(Base):
    """Training job model (also serves as Experiment)."""

    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)  # NEW: Link to project (nullable for backward compatibility)

    # Experiment metadata (NEW)
    experiment_name = Column(String(200), nullable=True)  # Optional experiment name
    tags = Column(JSON, nullable=True)  # Tags for categorization
    notes = Column(Text, nullable=True)  # User notes about this experiment
    mlflow_run_id = Column(String(100), nullable=True)  # Link to MLflow run

    # Training configuration
    framework = Column(String(50), nullable=False, default="timm")  # NEW: timm, ultralytics, transformers
    model_name = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=False)
    num_classes = Column(Integer, nullable=True)  # Changed to nullable for non-classification tasks
    dataset_path = Column(String(500), nullable=False)
    dataset_format = Column(String(50), nullable=False, default="imagefolder")  # NEW: imagefolder, coco, yolo, etc.
    output_dir = Column(String(500), nullable=False)

    # Hyperparameters
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)

    # Status and results
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    process_id = Column(Integer, nullable=True)

    # Training results
    final_accuracy = Column(Float, nullable=True)
    best_checkpoint_path = Column(String(500), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    session = relationship("Session", back_populates="training_jobs")
    project = relationship("Project", back_populates="experiments")
    metrics = relationship("TrainingMetric", back_populates="job", cascade="all, delete-orphan")
    logs = relationship("TrainingLog", back_populates="job", cascade="all, delete-orphan")


class TrainingMetric(Base):
    """Training metric model."""

    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)

    epoch = Column(Integer, nullable=False)
    step = Column(Integer, nullable=True)

    # Metrics
    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)

    # Additional metrics (stored as JSON)
    extra_metrics = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    job = relationship("TrainingJob", back_populates="metrics")


class TrainingLog(Base):
    """Training log model for capturing stdout/stderr."""

    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)

    log_type = Column(String(10), nullable=False)  # 'stdout' or 'stderr'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    job = relationship("TrainingJob", back_populates="logs")
