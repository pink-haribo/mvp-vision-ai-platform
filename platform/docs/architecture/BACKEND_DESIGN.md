# Backend Service Design

Complete design for the FastAPI backend service.

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Database Design](#database-design)
- [API Endpoints](#api-endpoints)
- [LLM Integration](#llm-integration)
- [Temporal Integration](#temporal-integration)
- [Real-Time Updates](#real-time-updates)
- [Authentication](#authentication)
- [Storage Abstraction](#storage-abstraction)
- [Observability](#observability)

## Overview

The Backend service is a FastAPI application that serves as the central orchestration layer for the platform.

**Responsibilities**:
- REST API for all client operations
- LLM-based natural language intent parsing
- Temporal workflow management
- Dataset validation and storage
- Real-time WebSocket updates
- Authentication and authorization
- Training job lifecycle management

**Port**: 8000 (configurable)

**Base URL**: `http://localhost:8000` (dev) or `https://api.example.com` (prod)

## Technology Stack

### Core Framework
- **FastAPI 0.109+**: Modern async web framework
- **Python 3.11**: Language version
- **Uvicorn**: ASGI server
- **Pydantic 2.5+**: Data validation

### Database
- **SQLAlchemy 2.0**: ORM
- **Alembic**: Database migrations
- **PostgreSQL 16**: Primary database
- **asyncpg**: Async PostgreSQL driver

### Async & Queue
- **Redis 7.2**: Caching, pub/sub
- **redis-py**: Async Redis client

### Workflow
- **Temporal Python SDK**: Workflow orchestration
- **temporalio**: Client library

### LLM
- **LangChain**: LLM framework
- **Anthropic Claude**: Primary LLM
- **OpenAI GPT-4**: Fallback LLM

### Storage
- **boto3**: S3-compatible storage client
- **Supports**: MinIO, Cloudflare R2, AWS S3

### Auth
- **python-jose**: JWT tokens
- **passlib**: Password hashing
- **bcrypt**: Hashing algorithm

### Observability
- **OpenTelemetry**: Distributed tracing
- **Prometheus Client**: Metrics
- **structlog**: Structured logging

### Testing
- **pytest**: Test framework
- **pytest-asyncio**: Async test support
- **httpx**: Async HTTP client for tests
- **pytest-cov**: Coverage reporting

## Project Structure

```
platform/backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Settings (Pydantic BaseSettings)
│   │
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── deps.py             # Dependency injection
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py       # API router aggregation
│   │   │   ├── chat.py         # POST /v1/chat/message
│   │   │   ├── training.py     # Training job endpoints
│   │   │   ├── datasets.py     # Dataset management
│   │   │   ├── models.py       # Model listing
│   │   │   └── auth.py         # Login, register
│   │   └── websocket.py        # WebSocket endpoints
│   │
│   ├── models/                 # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── user.py             # User model
│   │   ├── training_job.py     # TrainingJob model
│   │   ├── dataset.py          # Dataset model
│   │   └── model_registry.py   # ModelRegistration model
│   │
│   ├── schemas/                # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py             # UserCreate, UserResponse
│   │   ├── training.py         # TrainingJobCreate, TrainingJobResponse
│   │   ├── dataset.py          # DatasetCreate, DatasetResponse
│   │   ├── chat.py             # ChatMessage, ChatResponse
│   │   └── common.py           # Shared schemas
│   │
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── llm_parser.py       # LLM intent parsing
│   │   ├── training_executor.py # Training execution (subprocess/k8s)
│   │   ├── storage.py          # Storage abstraction
│   │   ├── dataset_service.py  # Dataset validation
│   │   └── auth_service.py     # Authentication logic
│   │
│   ├── temporal/               # Temporal client
│   │   ├── __init__.py
│   │   ├── client.py           # Temporal client wrapper
│   │   └── activities.py       # Shared activities (if any)
│   │
│   ├── db/                     # Database utilities
│   │   ├── __init__.py
│   │   ├── base.py             # SQLAlchemy Base
│   │   ├── session.py          # Database session management
│   │   └── init_db.py          # Database initialization
│   │
│   ├── core/                   # Core utilities
│   │   ├── __init__.py
│   │   ├── security.py         # JWT, password hashing
│   │   ├── logging.py          # Logging configuration
│   │   └── telemetry.py        # OpenTelemetry setup
│   │
│   └── utils/                  # Helper utilities
│       ├── __init__.py
│       └── validation.py       # Custom validators
│
├── alembic/                    # Database migrations
│   ├── env.py
│   ├── versions/
│   └── alembic.ini
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── unit/                   # Unit tests
│   │   ├── test_llm_parser.py
│   │   ├── test_storage.py
│   │   └── test_auth.py
│   └── integration/            # Integration tests
│       ├── test_api_chat.py
│       ├── test_api_training.py
│       └── test_training_flow.py
│
├── Dockerfile                  # Container image
├── pyproject.toml              # Poetry dependencies
├── poetry.lock
├── .env.example                # Environment variables template
└── README.md                   # Service documentation
```

## Database Design

### User Model

```python
# app/models/user.py
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class UserRole(str, enum.Enum):
    """User role with hierarchical permissions"""
    ADMIN = "admin"              # All permissions, user management
    MANAGER = "manager"          # Can grant permissions below manager
    ENGINEER_II = "engineer_ii"  # Advanced training features
    ENGINEER_I = "engineer_i"    # Basic training features
    GUEST = "guest"              # Limited: 1 project, 1 dataset, no collaboration

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)  # Deprecated, use role=ADMIN instead

    # Role-based permissions
    role = Column(SQLEnum(UserRole), default=UserRole.GUEST, nullable=False, index=True)

    # Organization and department
    company = Column(String(255), nullable=True)
    division = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)  # NEW: Department within organization
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True, index=True)

    # Avatar information (for consistent UI representation)
    avatar_name = Column(String(100), nullable=True)  # "John D", "JD", etc.
    badge_color = Column(String(20), nullable=True)   # "#4F46E5", "indigo", etc.

    # User preferences
    timezone = Column(String(50), default="UTC")
    language = Column(String(10), default="en")  # en, ko, ja, etc.
    notification_enabled = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships (existing)
    organization = relationship("Organization", back_populates="users")
    training_jobs = relationship("TrainingJob", back_populates="user")
    datasets = relationship("Dataset", back_populates="user")  # Deprecated, use owned_datasets
    test_runs = relationship("TestRun", back_populates="user")
    inference_jobs = relationship("InferenceJob", back_populates="user")
    export_jobs = relationship("ExportJob", back_populates="user")
    deployment_targets = relationship("DeploymentTarget", back_populates="user")

    # Relationships (NEW - from PROJECT_MEMBERSHIP_DESIGN.md)
    owned_projects = relationship("Project", back_populates="owner", foreign_keys="Project.owner_id")
    project_memberships = relationship("ProjectMember", back_populates="user")
    owned_datasets = relationship("Dataset", back_populates="owner", foreign_keys="Dataset.owner_id")
    dataset_memberships = relationship("DatasetMember", back_populates="user")
    starred_experiments = relationship("ExperimentStar", back_populates="user")
    experiment_notes = relationship("ExperimentNote", back_populates="user")
    sent_invitations = relationship("Invitation", back_populates="inviter", foreign_keys="Invitation.inviter_id")
    received_invitation = relationship("Invitation", back_populates="invitee_user", foreign_keys="Invitation.invitee_user_id")

    # Relationships (NEW - from USER_ANALYTICS_DESIGN.md)
    sessions = relationship("UserSession", back_populates="user")
    usage_stats = relationship("UserUsageStats", back_populates="user", uselist=False)
    usage_timeseries = relationship("UserUsageTimeSeries", back_populates="user")
    activity_events = relationship("ActivityEvent", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
```

### TrainingJob Model

```python
# app/models/training_job.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)

    # Project and Experiment (NEW - from PROJECT_MEMBERSHIP_DESIGN.md)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False, index=True)
    mlflow_run_id = Column(String(255), nullable=True)  # Maps to MLflow Run ID
    mlflow_run_name = Column(String(500), nullable=True)

    # Dataset snapshot (for reproducibility)
    dataset_snapshot_id = Column(String(255), ForeignKey("snapshots.id"), nullable=True)
    snapshot_status_at_start = Column(String(50), nullable=True)  # "valid" | "broken"

    # Identifiers
    trace_id = Column(UUID(as_uuid=True), default=uuid.uuid4, index=True)
    callback_token = Column(String(500), nullable=False)  # JWT for trainer callbacks
    temporal_workflow_id = Column(String(255), nullable=True)
    k8s_job_name = Column(String(255), nullable=True)
    process_id = Column(Integer, nullable=True)  # For subprocess mode

    # Configuration
    task_type = Column(String(100), nullable=False)  # image_classification, object_detection, etc.
    framework = Column(String(100), nullable=False)  # ultralytics, timm, huggingface
    model_name = Column(String(255), nullable=False)
    config = Column(JSON, nullable=False)  # Full training configuration

    # Split strategy (train/val split)
    split_strategy = Column(JSON, nullable=True)
    # {
    #   "method": "auto",  // "use_dataset" | "auto" | "custom"
    #   "ratio": [0.8, 0.2],
    #   "seed": 42,
    #   "custom_splits": {...}  // For method="custom"
    # }

    # Status
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, index=True)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, nullable=False)
    progress_percent = Column(Float, default=0.0)

    # Metrics
    current_metrics = Column(JSON, nullable=True)  # Latest epoch metrics
    final_metrics = Column(JSON, nullable=True)  # Final results
    best_metrics = Column(JSON, nullable=True)  # Best epoch metrics

    # Primary Metric (for best checkpoint selection)
    primary_metric_name = Column(String(50), nullable=False)  # "accuracy", "mAP50", "mIoU", etc.
    primary_metric_direction = Column(String(10), default="maximize")  # "maximize" or "minimize"

    # Best validation tracking
    best_epoch = Column(Integer, nullable=True)
    best_validation_id = Column(UUID(as_uuid=True), ForeignKey("training_validation_results.id"), nullable=True)

    # Storage
    checkpoint_path = Column(String(500), nullable=True)  # S3 path to best checkpoint
    logs_path = Column(String(500), nullable=True)  # S3 path to logs

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Error tracking
    error_message = Column(String(1000), nullable=True)

    # Relationships
    user = relationship("User", back_populates="training_jobs")
    dataset = relationship("Dataset", back_populates="training_jobs")
    project = relationship("Project", back_populates="training_jobs")  # NEW
    experiment = relationship("Experiment", back_populates="training_jobs")  # NEW
    snapshot = relationship("Snapshot", foreign_keys=[dataset_snapshot_id])
    validation_results = relationship("TrainingValidationResult", back_populates="training_job", foreign_keys="TrainingValidationResult.training_job_id")
    best_validation = relationship("TrainingValidationResult", foreign_keys=[best_validation_id])
    test_runs = relationship("TestRun", back_populates="training_job")
    inference_jobs = relationship("InferenceJob", back_populates="training_job")
    export_jobs = relationship("ExportJob", back_populates="training_job")
```

### Dataset Model

```python
# app/models/dataset.py
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class DatasetFormat(str, enum.Enum):
    YOLO = "yolo"
    COCO = "coco"
    IMAGEFOLDER = "imagefolder"
    PASCAL_VOC = "pascal_voc"

class DatasetStatus(str, enum.Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class DatasetVisibility(str, enum.Enum):
    """Dataset visibility"""
    PUBLIC = "public"    # Accessible to all users (read-only for non-members)
    PRIVATE = "private"  # Only accessible to owner and members

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)  # Deprecated, use owner_id

    # Ownership and visibility (NEW - from PROJECT_MEMBERSHIP_DESIGN.md)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True, index=True)
    visibility = Column(SQLEnum(DatasetVisibility), default=DatasetVisibility.PRIVATE, nullable=False, index=True)

    name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)

    # Format and structure
    format = Column(SQLEnum(DatasetFormat), nullable=False)
    labeled = Column(Boolean, default=False)  # Has annotations.json
    num_images = Column(Integer, nullable=False, default=0)
    num_classes = Column(Integer, nullable=True)
    num_train_samples = Column(Integer, nullable=True)
    num_val_samples = Column(Integer, nullable=True)
    class_names = Column(JSON, nullable=True)  # List of class names

    # Split configuration (cached from annotations.json)
    split_config = Column(JSON, nullable=True)
    # {
    #   "method": "manual",  // "manual" | "auto" | "none"
    #   "default_ratio": [0.8, 0.2],
    #   "seed": 42
    # }

    # Storage
    storage_path = Column(String(500), nullable=False)  # S3 path to dataset folder
    size_bytes = Column(Integer, nullable=True)

    # Status
    status = Column(SQLEnum(DatasetStatus), default=DatasetStatus.UPLOADING)
    validation_errors = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="datasets")  # Deprecated, use owner
    owner = relationship("User", back_populates="owned_datasets", foreign_keys=[owner_id])  # NEW
    organization = relationship("Organization", back_populates="datasets")  # NEW
    members = relationship("DatasetMember", back_populates="dataset")  # NEW
    training_jobs = relationship("TrainingJob", back_populates="dataset")
    snapshots = relationship("Snapshot", back_populates="dataset")
    test_runs = relationship("TestRun", back_populates="dataset")
    inference_jobs = relationship("InferenceJob", back_populates="dataset")
```

### Snapshot Model

```python
# app/models/snapshot.py
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base

class Snapshot(Base):
    """Dataset snapshot for training reproducibility"""
    __tablename__ = "snapshots"

    id = Column(String(255), primary_key=True)  # e.g., "training-job-abc123" or "dataset-xyz-v1"
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)

    # Snapshot type
    snapshot_type = Column(String(50), nullable=False)  # "training" | "manual"
    version_tag = Column(String(100), nullable=True)  # "v1", "v2", "production", etc. (for manual snapshots)

    # Integrity
    status = Column(String(50), default="valid")  # "valid" | "broken" | "repairing"
    integrity_status = Column(JSON, nullable=True)
    # {
    #   "total_images": 100,
    #   "missing_images": ["img001.jpg", ...],  // If status="broken"
    #   "missing_count": 2,
    #   "broken_at": "2025-01-10T12:00:00Z",
    #   "repaired_at": "2025-01-10T13:00:00Z"  // If repaired
    # }

    # Metadata (cached from snapshot JSON file)
    num_images = Column(Integer, nullable=True)
    annotations_hash = Column(String(64), nullable=True)  # SHA256 of annotations

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="snapshots")
    training_jobs = relationship("TrainingJob", foreign_keys="TrainingJob.dataset_snapshot_id")
```

**Note**: The actual snapshot data (annotations.json, image checksums) is stored in S3 at `datasets/{dataset_id}/snapshots/{snapshot_id}.json`. The database only stores metadata for quick querying.

### Organization Model

```python
# app/models/organization.py
from sqlalchemy import Column, String, Integer, BigInteger, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class Organization(Base):
    """Organization for multi-tenant support"""
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Company + Division
    company = Column(String(255), nullable=False, index=True)
    division = Column(String(255), nullable=True, index=True)

    # Display
    display_name = Column(String(500))  # "ABC Corporation - AI Research"

    # Storage quotas (in GB)
    checkpoint_storage_quota_gb = Column(Integer, default=500)
    pretrained_weight_storage_quota_gb = Column(Integer, default=100)

    # Usage tracking (in bytes)
    checkpoint_storage_used_bytes = Column(BigInteger, default=0)
    pretrained_weight_storage_used_bytes = Column(BigInteger, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="organization")
    pretrained_weights = relationship("PretrainedWeight", back_populates="organization")
    projects = relationship("Project", back_populates="organization")  # NEW
    datasets = relationship("Dataset", back_populates="organization")  # NEW

    # Unique constraint on company + division
    __table_args__ = (
        UniqueConstraint('company', 'division', name='uq_company_division'),
    )
```

### Pretrained Weight Model

```python
# app/models/pretrained_weight.py
from sqlalchemy import Column, String, Integer, BigInteger, DateTime, ForeignKey, JSON, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class PretrainedWeight(Base):
    """Registry of pretrained model weights"""
    __tablename__ = "pretrained_weights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Naming
    framework = Column(String(100), nullable=False, index=True)  # timm, ultralytics, huggingface, custom
    model_name = Column(String(255), nullable=False, unique=True, index=True)  # yolo11n_seg_coco

    # Metadata
    base_architecture = Column(String(100))  # yolo11n, resnet50, clip
    task_types = Column(JSON, nullable=False)  # ["object_detection"], ["image_classification"]
    pretraining_dataset = Column(String(255))  # COCO, ImageNet, ADE20K, Objects365
    description = Column(String(1000))

    # Model info
    num_parameters = Column(String(50))  # "11.2M", "25.6M"
    input_size = Column(JSON)  # [640, 640] or [224, 224]
    supported_formats = Column(JSON)  # ["yolo", "coco"]

    # Storage
    storage_path = Column(String(500), nullable=False)  # s3://bucket/pretrained-weights/ultralytics/yolo11n_seg_coco.pt
    size_bytes = Column(BigInteger, nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA256

    # Access control (DB-based, not folder-based)
    visibility = Column(String(50), default="private", nullable=False, index=True)  # public, private, organization
    owner_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)

    # Usage tracking
    download_count = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    owner = relationship("User", foreign_keys=[owner_user_id])
    organization = relationship("Organization", back_populates="pretrained_weights")

    # Indexes
    __table_args__ = (
        Index('idx_framework_visibility', 'framework', 'visibility'),
    )
```

### Model Registry Model

```python
# app/models/model_registry.py
from sqlalchemy import Column, String, DateTime, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.db.base import Base

class ModelRegistration(Base):
    """Registry of available models from training services"""
    __tablename__ = "model_registry"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    framework = Column(String(100), nullable=False, index=True)  # ultralytics, timm, etc.
    model_name = Column(String(255), nullable=False, index=True)
    task_types = Column(JSON, nullable=False)  # List of supported task types

    # Model metadata
    description = Column(String(1000), nullable=True)
    input_size = Column(JSON, nullable=True)  # e.g., [640, 640]
    num_parameters = Column(String(50), nullable=True)  # e.g., "11.2M"
    pretrained_available = Column(Boolean, default=True)

    # Configuration
    default_config = Column(JSON, nullable=True)  # Default hyperparameters
    supported_formats = Column(JSON, nullable=True)  # Compatible dataset formats

    # Service metadata
    service_url = Column(String(500), nullable=True)  # Training service URL
    docker_image = Column(String(500), nullable=True)  # Trainer Docker image

    # Timestamps
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### Training Validation Result Model

```python
# app/models/training_validation_result.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class TrainingValidationResult(Base):
    """
    Training validation results (task-agnostic)

    Stores validation metrics for each epoch during training.
    Supports all CV tasks: classification, detection, segmentation, pose, etc.
    """
    __tablename__ = "training_validation_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)

    # Task identification
    task_type = Column(String(100), nullable=False, index=True)
    # "image_classification", "object_detection", "instance_segmentation", etc.

    # Primary Metric (for best checkpoint selection)
    primary_metric_name = Column(String(50), nullable=False)
    # Examples: "accuracy", "mAP50", "mAP50-95", "mIoU", "OKS"

    primary_metric_value = Column(Float, nullable=False)
    # The value of the primary metric for this epoch

    is_best = Column(Boolean, default=False)
    # True if this is the best epoch for this job (based on primary metric)

    # All Metrics (standard + custom)
    metrics = Column(JSON, nullable=False)
    # {
    #   // Standard metrics (Frontend knows these)
    #   "accuracy": 0.95,
    #   "loss": 0.234,
    #   "mAP50": 0.87,
    #   "mAP50-95": 0.65,
    #   "precision": 0.89,
    #   "recall": 0.82,
    #
    #   // Custom metrics (displayed in generic table)
    #   "inference_speed_fps": 45.2,
    #   "gpu_memory_gb": 8.2,
    #   "custom_metric_1": 0.78
    # }

    # Per-Class Metrics
    per_class_metrics = Column(JSON, nullable=True)
    # [
    #   {
    #     "class_id": 0,
    #     "class_name": "cat",
    #     "precision": 0.90,
    #     "recall": 0.85,
    #     "f1": 0.87,
    #     "support": 120
    #   },
    #   ...
    # ]

    # Visualization Data
    confusion_matrix = Column(JSON, nullable=True)
    # Classification only: [[120, 5], [8, 95]]

    pr_curves = Column(JSON, nullable=True)
    # Detection, Segmentation: {"cat": {"precision": [...], "recall": [...]}, ...}

    # Storage
    image_results_path = Column(String(500), nullable=True)
    # S3 path to detailed per-image results
    # "s3://bucket/validation-results/{job_id}/epoch-{epoch}/images.json"

    num_images_validated = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    validation_time_seconds = Column(Float, nullable=True)
    # How long validation took

    # Relationships
    training_job = relationship("TrainingJob", back_populates="validation_results", foreign_keys=[training_job_id])

    # Indexes for common queries
    __table_args__ = (
        Index('idx_job_epoch', 'training_job_id', 'epoch'),
        Index('idx_job_best', 'training_job_id', 'is_best'),
    )
```

**Per-Image Validation Results** (stored in S3):

Per-image validation results are stored in S3 due to large volume. The database only stores the S3 path in `image_results_path`.

**S3 Path**: `s3://bucket/validation-results/{job_id}/epoch-{epoch}/images.json`

**JSON Format** (task-agnostic):
```json
{
  "format_version": "1.0",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "epoch": 42,
  "task_type": "object_detection",
  "num_images": 500,
  "generated_at": "2025-01-10T12:34:56Z",
  "images": [
    {
      "image_id": "val_001.jpg",
      "image_path": "datasets/abc/images/val/val_001.jpg",
      "predictions": {
        "boxes": [
          {"class_id": 0, "class_name": "cat", "bbox": [10, 20, 100, 150], "confidence": 0.95}
        ]
      },
      "ground_truth": {
        "boxes": [
          {"class_id": 0, "class_name": "cat", "bbox": [12, 22, 98, 148]}
        ]
      },
      "metrics": {
        "iou": 0.85,
        "precision": 1.0,
        "recall": 1.0
      }
    }
  ]
}
```

### Test Run Model

```python
# app/models/test_run.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class TestRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TestRun(Base):
    """
    Test Run: Inference on labeled dataset (with ground truth)
    Computes metrics by comparing predictions to ground truth
    """
    __tablename__ = "test_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)

    # Identifiers
    trace_id = Column(UUID(as_uuid=True), default=uuid.uuid4, index=True)
    callback_token = Column(String(500), nullable=False)
    k8s_job_name = Column(String(255), nullable=True)
    process_id = Column(Integer, nullable=True)  # For subprocess mode

    # Configuration
    task_type = Column(String(100), nullable=False)
    framework = Column(String(100), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)  # S3 path to checkpoint
    config = Column(JSON, nullable=True)  # Test configuration (batch_size, etc.)

    # XAI (Explainable AI)
    enable_xai = Column(Boolean, default=False)
    xai_method = Column(String(50), nullable=True)  # "gradcam", "lime", "shap", "attention"
    xai_config = Column(JSON, nullable=True)  # XAI-specific configuration

    # Status
    status = Column(SQLEnum(TestRunStatus), default=TestRunStatus.PENDING, index=True)
    progress_percent = Column(Float, default=0.0)
    processed_images = Column(Integer, default=0)
    total_images = Column(Integer, default=0)

    # Results
    overall_metrics = Column(JSON, nullable=True)
    # {
    #   "accuracy": 0.95,
    #   "mAP50-95": 0.6523,
    #   "precision": 0.89,
    #   "recall": 0.82
    # }

    per_class_metrics = Column(JSON, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)
    pr_curves = Column(JSON, nullable=True)

    # Storage
    results_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/test-runs/{test_run_id}/results.json
    visualizations_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/test-runs/{test_run_id}/visualizations/
    xai_results_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/test-runs/{test_run_id}/xai/

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Error tracking
    error_message = Column(String(1000), nullable=True)

    # Relationships
    user = relationship("User", back_populates="test_runs")
    training_job = relationship("TrainingJob", back_populates="test_runs")
    dataset = relationship("Dataset", back_populates="test_runs")
```

### Inference Job Model

```python
# app/models/inference_job.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class InferenceType(str, enum.Enum):
    SINGLE = "single"      # Single image
    BATCH = "batch"        # Multiple images (uploaded)
    DATASET = "dataset"    # Existing dataset (no labels)

class InferenceJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class InferenceJob(Base):
    """
    Inference Job: Inference on unlabeled data (no ground truth)
    Only produces predictions, no metrics computation
    """
    __tablename__ = "inference_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=True)
    # Can be null for inference on pretrained models without training

    # Identifiers
    trace_id = Column(UUID(as_uuid=True), default=uuid.uuid4, index=True)
    callback_token = Column(String(500), nullable=False)
    k8s_job_name = Column(String(255), nullable=True)
    process_id = Column(Integer, nullable=True)  # For subprocess mode

    # Configuration
    inference_type = Column(SQLEnum(InferenceType), nullable=False)
    task_type = Column(String(100), nullable=False)
    framework = Column(String(100), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)  # S3 path to checkpoint
    config = Column(JSON, nullable=True)  # Inference configuration

    # Input
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=True)
    # For inference_type="dataset"

    image_paths = Column(JSON, nullable=True)
    # For inference_type="single" or "batch"
    # ["s3://bucket/inference-jobs/{job_id}/input/img001.jpg", ...]

    # XAI (Explainable AI)
    enable_xai = Column(Boolean, default=False)
    xai_method = Column(String(50), nullable=True)  # "gradcam", "lime", "shap", "attention"
    xai_config = Column(JSON, nullable=True)
    enable_llm_explanation = Column(Boolean, default=False)  # Natural language explanation

    # Status
    status = Column(SQLEnum(InferenceJobStatus), default=InferenceJobStatus.PENDING, index=True)
    progress_percent = Column(Float, default=0.0)
    processed_images = Column(Integer, default=0)
    total_images = Column(Integer, default=0)

    # Storage
    results_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/inference-jobs/{job_id}/results.json
    visualizations_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/inference-jobs/{job_id}/visualizations/
    xai_results_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/inference-jobs/{job_id}/xai/

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Error tracking
    error_message = Column(String(1000), nullable=True)

    # Relationships
    user = relationship("User", back_populates="inference_jobs")
    training_job = relationship("TrainingJob", back_populates="inference_jobs")
    dataset = relationship("Dataset", back_populates="inference_jobs")
```

### ExportJob Model

```python
# app/models/export_job.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class ExportFormat(str, enum.Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    COREML = "coreml"
    TFLITE = "tflite"
    OPENVINO = "openvino"
    TORCHSCRIPT = "torchscript"

class ExportJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExportJob(Base):
    """
    Export Job: Convert trained model to deployment formats
    """
    __tablename__ = "export_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False)

    # Version management
    version = Column(Integer, nullable=False)  # Auto-increment per training_job (1, 2, 3, ...)
    version_tag = Column(String(100), nullable=True)  # User-friendly tags: "production", "staging", "v1.0"
    is_default = Column(Boolean, default=False)  # Only one default export per training_job

    # Identifiers
    trace_id = Column(UUID(as_uuid=True), default=uuid.uuid4, index=True)
    callback_token = Column(String(500), nullable=False)
    k8s_job_name = Column(String(255), nullable=True)
    process_id = Column(Integer, nullable=True)  # For subprocess mode

    # Configuration
    format = Column(SQLEnum(ExportFormat), nullable=False, index=True)
    task_type = Column(String(100), nullable=False)
    framework = Column(String(100), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)  # S3 path to source checkpoint
    export_config = Column(JSON, nullable=True)
    # {
    #   "quantization": {"type": "dynamic", "dtype": "int8"},
    #   "optimization_level": 3,
    #   "input_size": [640, 640],
    #   "include_preprocessing": true,
    #   "include_postprocessing": true
    # }

    # Validation (optional)
    validation_config = Column(JSON, nullable=True)
    # {
    #   "enabled": true,
    #   "dataset_id": "uuid",
    #   "fail_threshold": {"accuracy_drop_max": 0.02}
    # }

    validation_metrics = Column(JSON, nullable=True)
    # {
    #   "accuracy": 0.94,
    #   "original_accuracy": 0.95,
    #   "accuracy_drop": 0.01,
    #   "passed": true
    # }

    # Status
    status = Column(SQLEnum(ExportJobStatus), default=ExportJobStatus.PENDING, index=True)
    progress_percent = Column(Float, default=0.0)

    # Results
    export_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/export-jobs/{job_id}/model.{format}

    metadata_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/export-jobs/{job_id}/metadata.json

    package_path = Column(String(500), nullable=True)
    # S3 path: s3://bucket/export-jobs/{job_id}/package.zip (includes model + runtimes + metadata)

    model_info = Column(JSON, nullable=True)
    # {
    #   "model_size_mb": 22.5,
    #   "inference_speed_ms": 15.3,
    #   "input_shape": [1, 3, 640, 640],
    #   "output_shape": [1, 84, 8400]
    # }

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Error tracking
    error_message = Column(String(1000), nullable=True)

    # Relationships
    user = relationship("User", back_populates="export_jobs")
    training_job = relationship("TrainingJob", back_populates="export_jobs")
    deployment_targets = relationship("DeploymentTarget", back_populates="export_job")
```

### DeploymentTarget Model

```python
# app/models/deployment_target.py
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class DeploymentStrategy(str, enum.Enum):
    DOWNLOAD = "download"              # User downloads package
    PLATFORM_ENDPOINT = "platform_endpoint"  # Platform-hosted inference API
    EDGE_PACKAGE = "edge_package"      # Edge deployment package
    CONTAINER = "container"            # Docker container

class DeploymentStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPLOYING = "deploying"
    FAILED = "failed"

class DeploymentTarget(Base):
    """
    Deployment Target: Deployed model endpoint or package
    """
    __tablename__ = "deployment_targets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    export_job_id = Column(UUID(as_uuid=True), ForeignKey("export_jobs.id"), nullable=False)

    # Configuration
    deployment_strategy = Column(SQLEnum(DeploymentStrategy), nullable=False, index=True)
    name = Column(String(255), nullable=False)  # User-friendly name
    description = Column(String(1000), nullable=True)

    # Platform Endpoint specific
    endpoint_url = Column(String(500), nullable=True)
    # "https://api.platform.com/v1/infer/{deployment_id}"

    api_key = Column(String(500), nullable=True)  # For platform endpoint auth

    tier = Column(String(50), nullable=True)  # "free", "pro", "enterprise"

    usage_stats = Column(JSON, nullable=True)
    # {
    #   "requests_this_month": 1523,
    #   "total_requests": 45892,
    #   "avg_latency_ms": 28.5
    # }

    # Container specific
    docker_image = Column(String(500), nullable=True)
    # "platform.registry.com/users/{user_id}/models/{export_job_id}:latest"

    # Status
    status = Column(SQLEnum(DeploymentStatus), default=DeploymentStatus.ACTIVE, index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    deployed_at = Column(DateTime, nullable=True)
    last_accessed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="deployment_targets")
    export_job = relationship("ExportJob", back_populates="deployment_targets")
    deployment_history = relationship("DeploymentHistory", back_populates="deployment_target")
```

### DeploymentHistory Model

```python
# app/models/deployment_history.py
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class DeploymentHistory(Base):
    """
    Deployment History: Event log for deployment lifecycle
    """
    __tablename__ = "deployment_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deployment_target_id = Column(UUID(as_uuid=True), ForeignKey("deployment_targets.id"), nullable=False, index=True)

    # Event tracking
    event_type = Column(String(50), nullable=False, index=True)
    # "deployed", "updated", "scaled", "deactivated", "reactivated", "deleted"

    event_data = Column(JSON, nullable=True)
    # {
    #   "previous_tier": "free",
    #   "new_tier": "pro",
    #   "scaling_config": {...}
    # }

    status_before = Column(String(50), nullable=True)
    status_after = Column(String(50), nullable=True)

    message = Column(String(1000), nullable=True)  # Human-readable event description

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    deployment_target = relationship("DeploymentTarget", back_populates="deployment_history")
```

## API Endpoints

### Authentication Endpoints

**POST /api/v1/auth/register**
```python
# app/api/v1/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.user import UserCreate, UserResponse
from app.services.auth_service import create_user
from app.api.deps import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    user = await create_user(db, user_data)
    return user
```

**POST /api/v1/auth/login**
```python
from datetime import timedelta
from fastapi.security import OAuth2PasswordRequestForm
from app.core.security import create_access_token, verify_password
from app.schemas.auth import Token

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login and get JWT token"""
    user = await get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": access_token, "token_type": "bearer"}
```

### Chat Endpoints

**POST /api/v1/chat/message**
```python
# app/api/v1/chat.py
from fastapi import APIRouter, Depends
from app.schemas.chat import ChatMessage, ChatResponse
from app.services.llm_parser import parse_training_intent
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/message", response_model=ChatResponse)
async def send_chat_message(
    message: ChatMessage,
    current_user: User = Depends(get_current_user)
):
    """
    Parse natural language training request using LLM

    Returns either:
    - Complete training configuration (ready to start)
    - Clarification questions (need more info from user)
    """
    result = await parse_training_intent(
        message.content,
        conversation_history=message.history,
        user_id=str(current_user.id)
    )

    if result.is_complete:
        return ChatResponse(
            message="I understand. Here's the training configuration:",
            training_config=result.training_config,
            needs_clarification=False
        )
    else:
        return ChatResponse(
            message="I need a bit more information:",
            questions=result.questions,
            needs_clarification=True
        )
```

### Training Job Endpoints

**POST /api/v1/training/jobs**
```python
# app/api/v1/training.py
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.training import TrainingJobCreate, TrainingJobResponse
from app.services.training_executor import TrainingExecutor
from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.training_job import TrainingJob, JobStatus
from app.core.security import create_callback_token

router = APIRouter(prefix="/training", tags=["training"])

@router.post("/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    job_data: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create and start a new training job"""

    # Create database record
    job = TrainingJob(
        user_id=current_user.id,
        dataset_id=job_data.dataset_id,
        task_type=job_data.task_type,
        framework=job_data.framework,
        model_name=job_data.model_name,
        total_epochs=job_data.epochs,
        config=job_data.dict(),
        callback_token=create_callback_token(job_id=str(job.id)),
        status=JobStatus.PENDING
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Start training (async via Temporal or subprocess)
    executor = TrainingExecutor()
    background_tasks.add_task(executor.start_training, job.id)

    return job
```

**GET /api/v1/training/jobs/{job_id}**
```python
@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get training job status and metrics"""
    job = await db.get(TrainingJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")

    return job
```

**POST /api/v1/training/jobs/{job_id}/cancel**
```python
@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Cancel a running training job"""
    job = await db.get(TrainingJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Job is not running")

    # Send cancellation signal to Temporal workflow
    if job.temporal_workflow_id:
        await temporal_client.get_handle(job.temporal_workflow_id).signal("cancel")

    job.status = JobStatus.CANCELLED
    await db.commit()

    return {"message": "Job cancelled"}
```

### Callback Endpoints (for Trainers)

**POST /api/v1/jobs/{job_id}/heartbeat**
```python
from app.core.security import verify_callback_token
from app.schemas.training import TrainingHeartbeat

@router.post("/jobs/{job_id}/heartbeat")
async def training_heartbeat(
    job_id: UUID,
    heartbeat: TrainingHeartbeat,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive training progress update from trainer

    Called by trainer every epoch
    """
    job = await db.get(TrainingJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Update job status
    job.status = JobStatus.RUNNING
    job.current_epoch = heartbeat.epoch
    job.progress_percent = heartbeat.progress
    job.current_metrics = heartbeat.metrics
    job.updated_at = datetime.utcnow()

    await db.commit()

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), {
        "type": "progress",
        "epoch": heartbeat.epoch,
        "progress": heartbeat.progress,
        "metrics": heartbeat.metrics
    })

    return {"status": "ok"}
```

**POST /api/v1/jobs/{job_id}/event**
```python
from app.schemas.training import TrainingEvent

@router.post("/jobs/{job_id}/event")
async def training_event(
    job_id: UUID,
    event: TrainingEvent,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive training event from trainer

    Events: checkpoint_saved, validation_complete, etc.
    """
    job = await db.get(TrainingJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Handle specific events
    if event.event_type == "checkpoint_saved":
        job.checkpoint_path = event.data.get("checkpoint_path")

    await db.commit()

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), {
        "type": "event",
        "event_type": event.event_type,
        "message": event.message,
        "data": event.data
    })

    return {"status": "ok"}
```

**POST /api/v1/jobs/{job_id}/done**
```python
from app.schemas.training import TrainingComplete

@router.post("/jobs/{job_id}/done")
async def training_complete(
    job_id: UUID,
    completion: TrainingComplete,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive training completion from trainer
    """
    job = await db.get(TrainingJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Update job status
    job.status = JobStatus.COMPLETED if completion.status == "succeeded" else JobStatus.FAILED
    job.final_metrics = completion.final_metrics
    job.completed_at = datetime.utcnow()

    if completion.status == "failed":
        job.error_message = completion.error_message

    await db.commit()

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), {
        "type": "complete",
        "status": completion.status,
        "final_metrics": completion.final_metrics
    })

    return {"status": "ok"}
```

**POST /api/v1/jobs/{job_id}/validation**
```python
from app.schemas.validation import ValidationResultCreate
from app.services.validation_service import ValidationService

@router.post("/jobs/{job_id}/validation")
async def report_validation_result(
    job_id: UUID,
    validation_data: ValidationResultCreate,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive validation result from trainer

    Called by trainer after each validation run (usually per epoch)
    Automatically determines if this is the best checkpoint
    """
    job = await db.get(TrainingJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Save validation result and update best checkpoint if needed
    validation_service = ValidationService(db)

    result = await validation_service.save_validation_result(
        job_id=job_id,
        epoch=validation_data.epoch,
        task_type=validation_data.task_type,
        primary_metric_value=validation_data.primary_metric_value,
        metrics=validation_data.metrics,
        per_class_metrics=validation_data.per_class_metrics,
        confusion_matrix=validation_data.confusion_matrix,
        pr_curves=validation_data.pr_curves
    )

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), {
        "type": "validation_complete",
        "epoch": validation_data.epoch,
        "is_best": result.is_best,
        "primary_metric": {
            "name": result.primary_metric_name,
            "value": result.primary_metric_value
        },
        "metrics": validation_data.metrics
    })

    return {"status": "ok", "is_best": result.is_best}
```

### Test Run Endpoints

**POST /api/v1/test-runs**
```python
# app/api/v1/test_runs.py
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.inference import TestRunCreate, TestRunResponse
from app.services.inference_executor import InferenceExecutor
from app.models.test_run import TestRun, TestRunStatus
from app.core.security import create_callback_token

router = APIRouter(prefix="/test-runs", tags=["inference"])

@router.post("", response_model=TestRunResponse, status_code=status.HTTP_201_CREATED)
async def create_test_run(
    test_data: TestRunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create and start a test run

    Test run performs inference on a labeled dataset and computes metrics
    """

    # Verify training job exists and user has access
    training_job = await db.get(TrainingJob, test_data.training_job_id)
    if not training_job or training_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Create test run record
    test_run = TestRun(
        user_id=current_user.id,
        training_job_id=test_data.training_job_id,
        dataset_id=test_data.dataset_id,
        task_type=training_job.task_type,
        framework=training_job.framework,
        checkpoint_path=training_job.checkpoint_path,
        config=test_data.config,
        enable_xai=test_data.enable_xai,
        xai_method=test_data.xai_method,
        xai_config=test_data.xai_config,
        callback_token=create_callback_token(job_id=str(test_run.id)),
        status=TestRunStatus.PENDING
    )

    db.add(test_run)
    await db.commit()
    await db.refresh(test_run)

    # Start test run (async)
    executor = InferenceExecutor()
    background_tasks.add_task(executor.start_test_run, test_run.id)

    return test_run
```

**GET /api/v1/test-runs/{test_run_id}**
```python
@router.get("/{test_run_id}", response_model=TestRunResponse)
async def get_test_run(
    test_run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get test run status and results"""
    test_run = await db.get(TestRun, test_run_id)

    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    if test_run.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")

    return test_run
```

**POST /api/v1/test-runs/{test_run_id}/cancel**
```python
@router.post("/{test_run_id}/cancel")
async def cancel_test_run(
    test_run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Cancel a running test run"""
    test_run = await db.get(TestRun, test_run_id)

    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    if test_run.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if test_run.status not in [TestRunStatus.PENDING, TestRunStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Test run is not running")

    # Send cancellation signal
    if test_run.k8s_job_name:
        # Cancel K8s job
        pass
    elif test_run.process_id:
        # Kill subprocess
        pass

    test_run.status = TestRunStatus.CANCELLED
    await db.commit()

    return {"message": "Test run cancelled"}
```

### Inference Job Endpoints

**POST /api/v1/inference/jobs**
```python
# app/api/v1/inference.py
from fastapi import APIRouter, Depends, BackgroundTasks, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.inference import InferenceJobCreate, InferenceJobResponse
from app.services.inference_executor import InferenceExecutor
from app.models.inference_job import InferenceJob, InferenceJobStatus, InferenceType
from app.core.security import create_callback_token
from app.services.storage import get_storage

router = APIRouter(prefix="/inference", tags=["inference"])

@router.post("/jobs", response_model=InferenceJobResponse, status_code=status.HTTP_201_CREATED)
async def create_inference_job(
    job_data: InferenceJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create and start an inference job

    Inference job performs inference on unlabeled data (no metrics computation)
    """

    # Verify checkpoint exists
    if job_data.training_job_id:
        training_job = await db.get(TrainingJob, job_data.training_job_id)
        if not training_job or training_job.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Training job not found")
        checkpoint_path = training_job.checkpoint_path
        task_type = training_job.task_type
        framework = training_job.framework
    else:
        # Using pretrained model
        checkpoint_path = job_data.checkpoint_path
        task_type = job_data.task_type
        framework = job_data.framework

    # Create inference job record
    job = InferenceJob(
        user_id=current_user.id,
        training_job_id=job_data.training_job_id,
        inference_type=job_data.inference_type,
        task_type=task_type,
        framework=framework,
        checkpoint_path=checkpoint_path,
        dataset_id=job_data.dataset_id,
        image_paths=job_data.image_paths,
        config=job_data.config,
        enable_xai=job_data.enable_xai,
        xai_method=job_data.xai_method,
        xai_config=job_data.xai_config,
        enable_llm_explanation=job_data.enable_llm_explanation,
        callback_token=create_callback_token(job_id=str(job.id)),
        status=InferenceJobStatus.PENDING
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Start inference (async)
    executor = InferenceExecutor()
    background_tasks.add_task(executor.start_inference_job, job.id)

    return job
```

**POST /api/v1/inference/upload**
```python
@router.post("/upload", response_model=InferenceJobResponse)
async def upload_and_infer(
    files: List[UploadFile] = File(...),
    training_job_id: UUID = Form(...),
    enable_xai: bool = Form(False),
    xai_method: Optional[str] = Form(None),
    enable_llm_explanation: bool = Form(False),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload images and start inference

    Convenience endpoint for single/batch inference
    """

    # Verify training job
    training_job = await db.get(TrainingJob, training_job_id)
    if not training_job or training_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Upload files to S3
    storage = get_storage()
    image_paths = []

    for file in files:
        # Create job first to get ID for path
        job_id = uuid.uuid4()
        s3_path = await storage.upload_file(
            file,
            f"inference-jobs/{job_id}/input/{file.filename}"
        )
        image_paths.append(s3_path)

    # Create inference job
    job = InferenceJob(
        user_id=current_user.id,
        training_job_id=training_job_id,
        inference_type=InferenceType.BATCH if len(files) > 1 else InferenceType.SINGLE,
        task_type=training_job.task_type,
        framework=training_job.framework,
        checkpoint_path=training_job.checkpoint_path,
        image_paths=image_paths,
        enable_xai=enable_xai,
        xai_method=xai_method,
        enable_llm_explanation=enable_llm_explanation,
        callback_token=create_callback_token(job_id=str(job.id)),
        status=InferenceJobStatus.PENDING
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Start inference
    executor = InferenceExecutor()
    background_tasks.add_task(executor.start_inference_job, job.id)

    return job
```

**GET /api/v1/inference/jobs/{job_id}**
```python
@router.get("/jobs/{job_id}", response_model=InferenceJobResponse)
async def get_inference_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get inference job status and results"""
    job = await db.get(InferenceJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    if job.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")

    return job
```

### Inference Callback Endpoints (for Trainers)

**POST /api/v1/inference/{job_id}/progress**
```python
from app.schemas.inference import InferenceProgress

@router.post("/inference/{job_id}/progress")
async def inference_progress(
    job_id: UUID,
    progress: InferenceProgress,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive inference progress update

    Called by inference runner for real-time progress updates
    """
    # Determine if it's test run or inference job
    test_run = await db.get(TestRun, job_id)
    inference_job = await db.get(InferenceJob, job_id)

    job = test_run or inference_job

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Update progress
    job.status = test_run and TestRunStatus.RUNNING or InferenceJobStatus.RUNNING
    job.processed_images = progress.processed_images
    job.total_images = progress.total_images
    job.progress_percent = progress.progress_percent
    job.updated_at = datetime.utcnow()

    await db.commit()

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), {
        "type": "inference_progress",
        "processed_images": progress.processed_images,
        "total_images": progress.total_images,
        "progress_percent": progress.progress_percent
    })

    return {"status": "ok"}
```

**POST /api/v1/test-runs/{test_run_id}/done**
```python
from app.schemas.inference import TestRunComplete

@router.post("/test-runs/{test_run_id}/done")
async def test_run_complete(
    test_run_id: UUID,
    completion: TestRunComplete,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive test run completion
    """
    test_run = await db.get(TestRun, test_run_id)

    if not test_run:
        raise HTTPException(status_code=404, detail="Test run not found")

    # Update status
    test_run.status = TestRunStatus.COMPLETED if completion.status == "succeeded" else TestRunStatus.FAILED
    test_run.overall_metrics = completion.overall_metrics
    test_run.per_class_metrics = completion.per_class_metrics
    test_run.confusion_matrix = completion.confusion_matrix
    test_run.pr_curves = completion.pr_curves
    test_run.results_path = completion.results_path
    test_run.visualizations_path = completion.visualizations_path
    test_run.xai_results_path = completion.xai_results_path
    test_run.completed_at = datetime.utcnow()

    if completion.status == "failed":
        test_run.error_message = completion.error_message

    await db.commit()

    # Broadcast to WebSocket clients
    await broadcast_update(str(test_run_id), {
        "type": "test_run_complete",
        "status": completion.status,
        "overall_metrics": completion.overall_metrics
    })

    return {"status": "ok"}
```

**POST /api/v1/inference/jobs/{job_id}/done**
```python
from app.schemas.inference import InferenceJobComplete

@router.post("/inference/jobs/{job_id}/done")
async def inference_job_complete(
    job_id: UUID,
    completion: InferenceJobComplete,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive inference job completion
    """
    job = await db.get(InferenceJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    # Update status
    job.status = InferenceJobStatus.COMPLETED if completion.status == "succeeded" else InferenceJobStatus.FAILED
    job.results_path = completion.results_path
    job.visualizations_path = completion.visualizations_path
    job.xai_results_path = completion.xai_results_path
    job.completed_at = datetime.utcnow()

    if completion.status == "failed":
        job.error_message = completion.error_message

    await db.commit()

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), {
        "type": "inference_complete",
        "status": completion.status,
        "results_path": completion.results_path
    })

    return {"status": "ok"}
```

### Export Job Endpoints

**GET /api/v1/export/capabilities**
```python
# app/api/v1/export.py
from fastapi import APIRouter, Depends
from app.schemas.export import ExportCapabilitiesResponse

router = APIRouter(prefix="/export", tags=["export"])

@router.get("/capabilities", response_model=ExportCapabilitiesResponse)
async def get_export_capabilities(
    framework: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """
    Get available export formats and capabilities by framework

    Returns framework capability matrix showing which formats are supported
    """

    capabilities = {
        "ultralytics": {
            "formats": {
                "onnx": {"supported": True, "quality": "excellent", "native": True},
                "tensorrt": {"supported": True, "quality": "excellent", "native": True},
                "coreml": {"supported": True, "quality": "excellent", "native": True},
                "tflite": {"supported": True, "quality": "excellent", "native": True},
                "openvino": {"supported": True, "quality": "excellent", "native": True},
                "torchscript": {"supported": True, "quality": "excellent", "native": True}
            }
        },
        "timm": {
            "formats": {
                "onnx": {"supported": True, "quality": "excellent", "native": True},
                "tensorrt": {"supported": True, "quality": "good", "native": False, "via": "onnx"},
                "coreml": {"supported": True, "quality": "good", "native": False, "via": "onnx"},
                "tflite": {"supported": True, "quality": "fair", "native": False, "via": "onnx"},
                "openvino": {"supported": True, "quality": "good", "native": False, "via": "onnx"},
                "torchscript": {"supported": True, "quality": "excellent", "native": True}
            }
        },
        "huggingface": {
            "formats": {
                "onnx": {"supported": True, "quality": "excellent", "native": True},
                "tensorrt": {"supported": True, "quality": "fair", "native": False, "via": "onnx"},
                "coreml": {"supported": False, "quality": "none"},
                "tflite": {"supported": True, "quality": "fair", "native": False, "via": "onnx"},
                "openvino": {"supported": True, "quality": "excellent", "native": True},
                "torchscript": {"supported": True, "quality": "excellent", "native": True}
            }
        }
    }

    if framework:
        return {"framework": framework, "capabilities": capabilities.get(framework, {})}

    return {"frameworks": capabilities}
```

**POST /api/v1/export/jobs**
```python
from app.schemas.export import ExportJobCreate, ExportJobResponse
from app.services.export_executor import ExportExecutor
from app.models.export_job import ExportJob, ExportJobStatus
from app.core.security import create_callback_token

@router.post("/jobs", response_model=ExportJobResponse, status_code=status.HTTP_201_CREATED)
async def create_export_job(
    job_data: ExportJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create and start an export job

    Converts trained checkpoint to specified deployment format
    """

    # Verify training job exists and user has access
    training_job = await db.get(TrainingJob, job_data.training_job_id)
    if not training_job or training_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Get next version number
    result = await db.execute(
        select(func.max(ExportJob.version))
        .where(ExportJob.training_job_id == job_data.training_job_id)
    )
    max_version = result.scalar() or 0
    next_version = max_version + 1

    # Create export job record
    export_job = ExportJob(
        user_id=current_user.id,
        training_job_id=job_data.training_job_id,
        version=next_version,
        version_tag=job_data.version_tag,
        is_default=job_data.is_default,
        format=job_data.format,
        task_type=training_job.task_type,
        framework=training_job.framework,
        checkpoint_path=training_job.checkpoint_path,
        export_config=job_data.export_config,
        validation_config=job_data.validation_config,
        callback_token=create_callback_token(job_id=str(export_job.id)),
        status=ExportJobStatus.PENDING
    )

    # If is_default=True, unset other defaults
    if job_data.is_default:
        await db.execute(
            update(ExportJob)
            .where(ExportJob.training_job_id == job_data.training_job_id)
            .where(ExportJob.id != export_job.id)
            .values(is_default=False)
        )

    db.add(export_job)
    await db.commit()
    await db.refresh(export_job)

    # Start export (async)
    executor = ExportExecutor()
    background_tasks.add_task(executor.start_export_job, export_job.id)

    return export_job
```

**GET /api/v1/training/{training_job_id}/exports**
```python
@router.get("/training/{training_job_id}/exports", response_model=List[ExportJobResponse])
async def list_training_exports(
    training_job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all export jobs for a training job

    Returns exports sorted by version (descending)
    """

    # Verify training job access
    training_job = await db.get(TrainingJob, training_job_id)
    if not training_job or training_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Get all exports
    result = await db.execute(
        select(ExportJob)
        .where(ExportJob.training_job_id == training_job_id)
        .order_by(ExportJob.version.desc())
    )

    exports = result.scalars().all()

    return exports
```

**GET /api/v1/export/jobs/{export_job_id}**
```python
@router.get("/jobs/{export_job_id}", response_model=ExportJobResponse)
async def get_export_job(
    export_job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get export job status and results"""

    export_job = await db.get(ExportJob, export_job_id)

    if not export_job:
        raise HTTPException(status_code=404, detail="Export job not found")

    if export_job.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")

    return export_job
```

**POST /api/v1/export/jobs/{export_job_id}/set-default**
```python
@router.post("/jobs/{export_job_id}/set-default")
async def set_default_export(
    export_job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Set an export as the default for its training job

    Unsets all other exports as default
    """

    export_job = await db.get(ExportJob, export_job_id)

    if not export_job:
        raise HTTPException(status_code=404, detail="Export job not found")

    if export_job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Unset all other defaults
    await db.execute(
        update(ExportJob)
        .where(ExportJob.training_job_id == export_job.training_job_id)
        .where(ExportJob.id != export_job_id)
        .values(is_default=False)
    )

    # Set this as default
    export_job.is_default = True
    await db.commit()

    return {"message": "Default export updated"}
```

**POST /api/v1/export/jobs/{export_job_id}/cancel**
```python
@router.post("/jobs/{export_job_id}/cancel")
async def cancel_export_job(
    export_job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Cancel a running export job"""

    export_job = await db.get(ExportJob, export_job_id)

    if not export_job:
        raise HTTPException(status_code=404, detail="Export job not found")

    if export_job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if export_job.status not in [ExportJobStatus.PENDING, ExportJobStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Export job is not running")

    # Send cancellation signal
    if export_job.k8s_job_name:
        # Cancel K8s job
        pass
    elif export_job.process_id:
        # Kill subprocess
        pass

    export_job.status = ExportJobStatus.CANCELLED
    await db.commit()

    return {"message": "Export job cancelled"}
```

### Export Callback Endpoints (for Export Runners)

**POST /api/v1/export/jobs/{job_id}/progress**
```python
from app.schemas.export import ExportProgress

@router.post("/export/jobs/{job_id}/progress")
async def export_progress(
    job_id: UUID,
    progress: ExportProgress,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive export progress update

    Called by export runner for real-time progress updates
    """

    export_job = await db.get(ExportJob, job_id)

    if not export_job:
        raise HTTPException(status_code=404, detail="Export job not found")

    # Update progress
    export_job.status = ExportJobStatus.RUNNING
    export_job.progress_percent = progress.progress_percent
    export_job.updated_at = datetime.utcnow()

    await db.commit()

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), {
        "type": "export_progress",
        "progress_percent": progress.progress_percent,
        "stage": progress.stage
    })

    return {"status": "ok"}
```

**POST /api/v1/export/jobs/{job_id}/done**
```python
from app.schemas.export import ExportJobComplete

@router.post("/export/jobs/{job_id}/done")
async def export_job_complete(
    job_id: UUID,
    completion: ExportJobComplete,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Receive export job completion
    """

    export_job = await db.get(ExportJob, job_id)

    if not export_job:
        raise HTTPException(status_code=404, detail="Export job not found")

    # Update status
    export_job.status = ExportJobStatus.COMPLETED if completion.status == "succeeded" else ExportJobStatus.FAILED
    export_job.export_path = completion.export_path
    export_job.metadata_path = completion.metadata_path
    export_job.package_path = completion.package_path
    export_job.model_info = completion.model_info
    export_job.validation_metrics = completion.validation_metrics
    export_job.completed_at = datetime.utcnow()

    if completion.status == "failed":
        export_job.error_message = completion.error_message

    await db.commit()

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), {
        "type": "export_complete",
        "status": completion.status,
        "export_path": completion.export_path
    })

    return {"status": "ok"}
```

### Deployment Endpoints

**POST /api/v1/deployments**
```python
# app/api/v1/deployments.py
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.deployment import DeploymentCreate, DeploymentResponse
from app.services.deployment_service import DeploymentService
from app.models.deployment_target import DeploymentTarget, DeploymentStatus

router = APIRouter(prefix="/deployments", tags=["deployment"])

@router.post("", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
async def create_deployment(
    deployment_data: DeploymentCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new deployment target

    Supports multiple deployment strategies:
    - download: User downloads package
    - platform_endpoint: Platform-hosted inference API
    - edge_package: Edge deployment package
    - container: Docker container
    """

    # Verify export job exists and user has access
    export_job = await db.get(ExportJob, deployment_data.export_job_id)
    if not export_job or export_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Export job not found")

    # Create deployment target
    deployment = DeploymentTarget(
        user_id=current_user.id,
        export_job_id=deployment_data.export_job_id,
        deployment_strategy=deployment_data.deployment_strategy,
        name=deployment_data.name,
        description=deployment_data.description,
        tier=deployment_data.tier,
        status=DeploymentStatus.DEPLOYING
    )

    db.add(deployment)
    await db.commit()
    await db.refresh(deployment)

    # Process deployment based on strategy
    service = DeploymentService(db)

    if deployment_data.deployment_strategy == "platform_endpoint":
        # Deploy to platform inference service
        background_tasks.add_task(
            service.deploy_to_platform_endpoint,
            deployment.id,
            deployment_data.tier
        )
    elif deployment_data.deployment_strategy == "container":
        # Build and push Docker image
        background_tasks.add_task(
            service.build_docker_image,
            deployment.id
        )
    else:
        # For download and edge_package, just mark as active
        deployment.status = DeploymentStatus.ACTIVE
        deployment.deployed_at = datetime.utcnow()
        await db.commit()

    return deployment
```

**GET /api/v1/deployments/{deployment_id}**
```python
@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get deployment target details"""

    deployment = await db.get(DeploymentTarget, deployment_id)

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if deployment.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")

    return deployment
```

**GET /api/v1/deployments/{deployment_id}/history**
```python
from app.schemas.deployment import DeploymentHistoryResponse

@router.get("/{deployment_id}/history", response_model=List[DeploymentHistoryResponse])
async def get_deployment_history(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get deployment history events"""

    # Verify deployment access
    deployment = await db.get(DeploymentTarget, deployment_id)
    if not deployment or deployment.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # Get history
    result = await db.execute(
        select(DeploymentHistory)
        .where(DeploymentHistory.deployment_target_id == deployment_id)
        .order_by(DeploymentHistory.created_at.desc())
    )

    history = result.scalars().all()

    return history
```

**POST /api/v1/deployments/{deployment_id}/deactivate**
```python
@router.post("/{deployment_id}/deactivate")
async def deactivate_deployment(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Deactivate a deployment target"""

    deployment = await db.get(DeploymentTarget, deployment_id)

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if deployment.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Create history event
    history = DeploymentHistory(
        deployment_target_id=deployment_id,
        event_type="deactivated",
        status_before=deployment.status,
        status_after="inactive",
        message="Deployment deactivated by user"
    )

    deployment.status = DeploymentStatus.INACTIVE

    db.add(history)
    await db.commit()

    return {"message": "Deployment deactivated"}
```

### Platform Inference Endpoint (Public API)

**POST /v1/infer/{deployment_id}**
```python
# app/api/v1/inference_api.py
from fastapi import APIRouter, Depends, Header, HTTPException
from app.schemas.inference_api import InferenceRequest, InferenceResponse
from app.services.inference_api_service import InferenceAPIService

router = APIRouter(prefix="/v1/infer", tags=["inference-api"])

@router.post("/{deployment_id}", response_model=InferenceResponse)
async def platform_inference(
    deployment_id: UUID,
    request: InferenceRequest,
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Platform-hosted inference endpoint

    This is a public API endpoint that clients call for inference
    Requires API key authentication (tier-based rate limiting)
    """

    # Verify API key
    api_key = authorization.replace("Bearer ", "")

    deployment = await db.get(DeploymentTarget, deployment_id)

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if deployment.api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if deployment.status != DeploymentStatus.ACTIVE:
        raise HTTPException(status_code=503, detail="Deployment is not active")

    # Check rate limits based on tier
    service = InferenceAPIService(db)

    if not await service.check_rate_limit(deployment.user_id, deployment.tier):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Perform inference
    try:
        predictions = await service.run_inference(
            deployment_id=deployment_id,
            image=request.image,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold
        )

        # Update usage stats
        await service.update_usage_stats(deployment_id, predictions.latency_ms)

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Dataset Endpoints

**POST /api/v1/datasets/upload**
```python
# app/api/v1/datasets.py
from fastapi import APIRouter, UploadFile, File, Form, Depends
from app.schemas.dataset import DatasetResponse
from app.services.storage import get_storage
from app.services.dataset_service import validate_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    format: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload and validate a dataset"""

    # Create dataset record
    dataset = Dataset(
        user_id=current_user.id,
        name=name,
        format=format,
        status=DatasetStatus.UPLOADING
    )
    db.add(dataset)
    await db.commit()

    # Upload to storage
    storage = get_storage()
    storage_path = await storage.upload_dataset(file, str(dataset.id))

    # Validate dataset structure
    validation_result = await validate_dataset(storage_path, format)

    dataset.storage_path = storage_path
    dataset.num_classes = validation_result.num_classes
    dataset.num_train_samples = validation_result.num_train
    dataset.num_val_samples = validation_result.num_val
    dataset.class_names = validation_result.class_names
    dataset.status = DatasetStatus.READY if validation_result.is_valid else DatasetStatus.ERROR
    dataset.validation_errors = validation_result.errors

    await db.commit()

    return dataset
```

## LLM Integration

### Intent Parser Service

```python
# app/services/llm_parser.py
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional

class TrainingIntent(BaseModel):
    task_type: str
    model_name: Optional[str] = None
    dataset_id: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None

class ParseResult(BaseModel):
    is_complete: bool
    training_config: Optional[TrainingIntent] = None
    questions: Optional[List[str]] = None

async def parse_training_intent(
    message: str,
    conversation_history: List[dict],
    user_id: str
) -> ParseResult:
    """
    Parse natural language training request using LLM

    Examples:
    - "Train a YOLO model for object detection"
    - "Fine-tune ResNet-50 on my cat-dog dataset for 10 epochs"
    - "Train YOLOv8n with learning rate 0.001"
    """

    # Build conversation history
    messages = []
    for msg in conversation_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # System prompt
    system_prompt = """
    You are an AI training configuration assistant. Parse the user's natural language
    request and extract training parameters.

    Available task types: image_classification, object_detection, instance_segmentation, pose_estimation

    Available models:
    - Ultralytics: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x (+ variants: -seg, -pose, -cls)
    - timm: resnet50, efficientnet_b0, vit_base_patch16_224
    - huggingface: clip-vit-base-patch32, dino-vits16

    If the user's request is incomplete, ask clarifying questions.
    If you have all information, return a complete training configuration.

    Respond in JSON format:
    {
        "is_complete": true/false,
        "training_config": {...} or null,
        "questions": [...] or null
    }
    """

    messages.append({"role": "user", "content": message})

    # Call LLM
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        api_key=settings.ANTHROPIC_API_KEY
    )

    response = await llm.ainvoke(messages, system=system_prompt)

    # Parse response
    result = ParseResult.parse_raw(response.content)

    return result
```

## Temporal Integration

### Temporal Client Wrapper

```python
# app/temporal/client.py
from temporalio.client import Client
from temporalio.common import RetryPolicy
from app.config import settings

class TemporalClientWrapper:
    def __init__(self):
        self.client: Optional[Client] = None

    async def connect(self):
        """Connect to Temporal server"""
        self.client = await Client.connect(
            settings.TEMPORAL_HOST,
            namespace=settings.TEMPORAL_NAMESPACE
        )

    async def start_training_workflow(self, job_id: str, config: dict) -> str:
        """Start training workflow"""
        handle = await self.client.start_workflow(
            "TrainingWorkflow",
            config,
            id=f"training-{job_id}",
            task_queue="training-tasks",
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval_seconds=10
            )
        )

        return handle.workflow_id

# Global instance
temporal_client = TemporalClientWrapper()
```

## Real-Time Updates

### WebSocket Connection Handler

```python
# app/api/websocket.py
from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set
from uuid import UUID

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)

    def disconnect(self, job_id: str, websocket: WebSocket):
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)

    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass  # Connection closed

manager = ConnectionManager()

@app.websocket("/ws/training/{job_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    job_id: UUID,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """
    WebSocket endpoint for real-time training updates

    Usage:
    const ws = new WebSocket(`ws://backend/ws/training/${jobId}?token=${jwt}`)
    """

    # Verify JWT token
    try:
        user = verify_access_token(token)
    except:
        await websocket.close(code=1008)  # Policy violation
        return

    # Verify job ownership
    job = await db.get(TrainingJob, job_id)
    if not job or (job.user_id != user.id and not user.is_superuser):
        await websocket.close(code=1008)
        return

    # Connect
    await manager.connect(str(job_id), websocket)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(str(job_id), websocket)

async def broadcast_update(job_id: str, message: dict):
    """Helper function to broadcast updates to all connected clients"""
    await manager.broadcast(job_id, message)
```

## Authentication

### JWT Security

```python
# app/core/security.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM
    )

    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)

def create_callback_token(job_id: str) -> str:
    """Create token for trainer callbacks"""
    return create_access_token(
        data={"sub": job_id, "type": "callback"},
        expires_delta=timedelta(hours=48)  # Longer expiry for training jobs
    )

def verify_callback_token(token: str) -> dict:
    """Verify callback token from trainer"""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        if payload.get("type") != "callback":
            raise HTTPException(status_code=403, detail="Invalid token type")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

## Storage Abstraction

See [3_TIER_DEVELOPMENT.md](../development/3_TIER_DEVELOPMENT.md#storage-abstraction) for complete storage abstraction implementation.

## Observability

### OpenTelemetry Setup

```python
# app/core/telemetry.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

def setup_telemetry(app):
    """Setup OpenTelemetry tracing"""

    # Create tracer provider
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # Add OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.OTEL_COLLECTOR_ENDPOINT
    )
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Instrument SQLAlchemy
    SQLAlchemyInstrumentor().instrument()

# Usage in main.py
from app.core.telemetry import setup_telemetry

app = FastAPI(title="Vision AI Training Platform")

setup_telemetry(app)
```

### Prometheus Metrics

```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Training metrics
training_jobs_total = Counter(
    'training_jobs_total',
    'Total training jobs',
    ['status', 'framework']
)

training_jobs_active = Gauge(
    'training_jobs_active',
    'Currently running training jobs'
)

training_duration_seconds = Histogram(
    'training_duration_seconds',
    'Training job duration',
    ['framework', 'model_name']
)
```

## Configuration

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Vision AI Training Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Execution
    EXECUTION_MODE: str = "subprocess"  # subprocess | kubernetes
    STORAGE_TYPE: str = "local"  # local | minio | r2 | s3

    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str

    # Temporal
    TEMPORAL_HOST: str = "localhost:7233"
    TEMPORAL_NAMESPACE: str = "default"

    # LLM
    ANTHROPIC_API_KEY: str
    OPENAI_API_KEY: Optional[str] = None

    # JWT
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Storage
    S3_ENDPOINT: Optional[str] = None
    R2_ENDPOINT: Optional[str] = None
    R2_ACCESS_KEY_ID: Optional[str] = None
    R2_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # Observability
    OTEL_COLLECTOR_ENDPOINT: str = "localhost:4317"

    # Backend
    BACKEND_BASE_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"

settings = Settings()
```

## Entry Point

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.api.websocket import websocket_endpoint
from app.core.telemetry import setup_telemetry
from app.temporal.client import temporal_client
from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v1")

# WebSocket
app.add_websocket_route("/ws/training/{job_id}", websocket_endpoint)

# Setup telemetry
setup_telemetry(app)

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    await temporal_client.connect()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## References

- [Architecture Overview](./OVERVIEW.md)
- [Trainer Design](./TRAINER_DESIGN.md)
- [Workflows Design](./WORKFLOWS_DESIGN.md)
- [3-Tier Development](../development/3_TIER_DEVELOPMENT.md)
- [API Specification (old)](../../../docs/api/API_SPECIFICATION.md)
