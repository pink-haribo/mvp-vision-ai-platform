"""Database models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship

from app.db.database import Base


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)

    company = Column(String(100), nullable=True)
    company_custom = Column(String(255), nullable=True)
    division = Column(String(100), nullable=True)
    division_custom = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)

    phone_number = Column(String(50), nullable=True)
    bio = Column(Text, nullable=True)

    system_role = Column(String(50), nullable=False, default='guest')
    is_active = Column(Boolean, nullable=False, default=True)
    badge_color = Column(String(20), nullable=True)  # Avatar badge color (e.g., 'blue', 'green', 'purple')

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    owned_projects = relationship("Project", back_populates="owner", foreign_keys="[Project.user_id]")
    sessions = relationship("Session", back_populates="user")
    created_training_jobs = relationship("TrainingJob", back_populates="creator", foreign_keys="[TrainingJob.created_by]")
    project_memberships = relationship("ProjectMember", back_populates="user", foreign_keys="[ProjectMember.user_id]")
    invited_members = relationship("ProjectMember", back_populates="inviter", foreign_keys="[ProjectMember.invited_by]")


class ProjectMember(Base):
    """Project member model for collaboration."""

    __tablename__ = "project_members"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    role = Column(String(20), nullable=False, default='member')
    invited_by = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    joined_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_memberships", foreign_keys=[user_id])
    inviter = relationship("User", back_populates="invited_members", foreign_keys=[invited_by])


class Project(Base):
    """Project model for organizing experiments."""

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    task_type = Column(String(50), nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="owned_projects", foreign_keys=[user_id])
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    experiments = relationship("TrainingJob", back_populates="project", cascade="all, delete-orphan")


class Session(Base):
    """Chat session model."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    state = Column(String(50), nullable=False, default="initial", index=True)
    temp_data = Column(JSON, nullable=False, default={})

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    """Chat message model."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="messages")


class TrainingJob(Base):
    """Training job model (also serves as Experiment)."""

    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    experiment_name = Column(String(200), nullable=True)
    tags = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    mlflow_run_id = Column(String(100), nullable=True)

    framework = Column(String(50), nullable=False, default="timm")
    model_name = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=False)
    num_classes = Column(Integer, nullable=True)
    dataset_path = Column(String(500), nullable=False)
    dataset_format = Column(String(50), nullable=False, default="imagefolder")
    output_dir = Column(String(500), nullable=False)

    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)

    status = Column(String(20), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    process_id = Column(Integer, nullable=True)

    final_accuracy = Column(Float, nullable=True)
    best_checkpoint_path = Column(String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    creator = relationship("User", back_populates="created_training_jobs", foreign_keys=[created_by])
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

    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)

    extra_metrics = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="metrics")


class TrainingLog(Base):
    """Training log model for capturing stdout/stderr."""

    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)

    log_type = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="logs")
