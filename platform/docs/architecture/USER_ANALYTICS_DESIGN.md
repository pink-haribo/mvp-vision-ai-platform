# User Analytics & Usage Tracking Design

## Overview

This document defines the **User Analytics** and **Usage Tracking** system for comprehensive monitoring of platform usage, resource consumption, and user behavior patterns.

### Key Concepts

1. **Session Tracking**: Login activity, active time, idle time
2. **Resource Usage**: GPU/CPU hours, storage, job counts
3. **Behavioral Analytics**: Feature usage, model preferences, experiment patterns
4. **Cost Tracking**: Estimated costs, trends, efficiency metrics
5. **Time Series Analysis**: Hourly, daily, monthly aggregations
6. **KPI Monitoring**: Success rates, utilization, collaboration metrics

### Architecture

```
User Activity → Event Collection → Aggregation → Analytics API
    ↓                    ↓              ↓              ↓
Sessions            Raw Events    Time Series    Dashboards
API Calls          Job Metrics    Aggregates      Reports
Resource Usage     Cost Data      Insights        KPIs
```

---

## Database Models

### UserSession Model

```python
# app/models/user_session.py
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import uuid

from app.db.base import Base

class UserSession(Base):
    """User login sessions with activity tracking"""
    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # User reference
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Session info
    session_token = Column(String(500), unique=True, nullable=False, index=True)
    refresh_token = Column(String(500), unique=True, nullable=True)

    # Device/Client info
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(50), nullable=True)
    device_type = Column(String(50), nullable=True)  # "desktop", "mobile", "tablet"
    browser = Column(String(100), nullable=True)
    os = Column(String(100), nullable=True)

    # Timestamps
    login_at = Column(DateTime, default=datetime.utcnow, index=True)
    logout_at = Column(DateTime, nullable=True, index=True)
    last_activity_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Activity tracking
    active_duration_seconds = Column(Integer, default=0)  # Actual active time
    total_duration_seconds = Column(Integer, default=0)   # Login to logout
    idle_duration_seconds = Column(Integer, default=0)    # Idle time

    # Activity counts
    api_calls_count = Column(Integer, default=0)
    page_views_count = Column(Integer, default=0)

    # Status
    is_active = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    @property
    def active_duration_hours(self) -> float:
        """Active duration in hours"""
        return self.active_duration_seconds / 3600

    @property
    def total_duration_hours(self) -> float:
        """Total duration in hours"""
        return self.total_duration_seconds / 3600

    def update_activity(self):
        """Update last activity timestamp"""
        now = datetime.utcnow()
        time_since_last_activity = (now - self.last_activity_at).total_seconds()

        # If activity within 5 minutes, count as active time
        if time_since_last_activity < 300:  # 5 minutes
            self.active_duration_seconds += int(time_since_last_activity)
        else:
            # Count as idle time
            self.idle_duration_seconds += int(time_since_last_activity)

        self.last_activity_at = now

        if self.login_at:
            self.total_duration_seconds = int((now - self.login_at).total_seconds())
```

### UserUsageStats Model

```python
# app/models/user_usage_stats.py
from sqlalchemy import Column, String, Integer, Float, BigInteger, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class UserUsageStats(Base):
    """Aggregated usage statistics per user"""
    __tablename__ = "user_usage_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # User reference
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True, index=True)

    # ==================== Session Statistics ====================

    # Total sessions
    total_sessions = Column(Integer, default=0)
    total_login_time_seconds = Column(BigInteger, default=0)
    total_active_time_seconds = Column(BigInteger, default=0)
    total_idle_time_seconds = Column(BigInteger, default=0)

    # Last activity
    last_login_at = Column(DateTime, nullable=True)
    last_activity_at = Column(DateTime, nullable=True)

    # ==================== Resource Usage ====================

    # Compute resources
    total_gpu_seconds = Column(BigInteger, default=0)  # Total GPU time used
    total_cpu_seconds = Column(BigInteger, default=0)  # Total CPU time used

    # Storage (in bytes)
    total_storage_used_bytes = Column(BigInteger, default=0)
    dataset_storage_bytes = Column(BigInteger, default=0)
    checkpoint_storage_bytes = Column(BigInteger, default=0)
    export_storage_bytes = Column(BigInteger, default=0)

    # ==================== Job Counts ====================

    # Training
    total_training_jobs = Column(Integer, default=0)
    completed_training_jobs = Column(Integer, default=0)
    failed_training_jobs = Column(Integer, default=0)
    cancelled_training_jobs = Column(Integer, default=0)

    # Inference
    total_test_runs = Column(Integer, default=0)
    total_inference_jobs = Column(Integer, default=0)

    # Export/Deploy
    total_export_jobs = Column(Integer, default=0)
    total_deployments = Column(Integer, default=0)

    # ==================== Feature Usage ====================

    feature_usage_counts = Column(JSON, default=dict)
    # {
    #   "training": 152,
    #   "inference": 89,
    #   "export": 23,
    #   "xai": 12,
    #   "hyperparameter_tuning": 8
    # }

    # Model usage
    model_usage_counts = Column(JSON, default=dict)
    # {
    #   "yolo11n": 45,
    #   "resnet50": 30,
    #   "efficientnet-b0": 20
    # }

    # Framework usage
    framework_usage_counts = Column(JSON, default=dict)
    # {
    #   "ultralytics": 65,
    #   "timm": 40,
    #   "huggingface": 15
    # }

    # ==================== Collaboration Metrics ====================

    # Projects
    projects_created = Column(Integer, default=0)
    projects_joined = Column(Integer, default=0)
    projects_owned = Column(Integer, default=0)

    # Datasets
    datasets_created = Column(Integer, default=0)
    datasets_shared = Column(Integer, default=0)
    datasets_owned = Column(Integer, default=0)

    # Invitations
    users_invited = Column(Integer, default=0)

    # ==================== Cost Estimates ====================

    # Estimated costs (in USD)
    estimated_total_cost_usd = Column(Float, default=0.0)
    estimated_gpu_cost_usd = Column(Float, default=0.0)
    estimated_storage_cost_usd = Column(Float, default=0.0)
    estimated_inference_cost_usd = Column(Float, default=0.0)

    # ==================== Efficiency Metrics ====================

    # Success rates
    training_success_rate = Column(Float, default=0.0)  # completed / total
    avg_training_time_hours = Column(Float, default=0.0)

    # Resource efficiency
    avg_gpu_utilization = Column(Float, default=0.0)  # 0.0 - 1.0

    # ==================== API Usage ====================

    total_api_calls = Column(BigInteger, default=0)
    api_error_count = Column(Integer, default=0)

    api_calls_by_endpoint = Column(JSON, default=dict)
    # {
    #   "/training": 2100,
    #   "/inference": 1500,
    #   "/datasets": 800
    # }

    # ==================== Timestamps ====================

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="usage_stats")

    # ==================== Computed Properties ====================

    @property
    def total_gpu_hours(self) -> float:
        """Total GPU usage in hours"""
        return self.total_gpu_seconds / 3600

    @property
    def total_cpu_hours(self) -> float:
        """Total CPU usage in hours"""
        return self.total_cpu_seconds / 3600

    @property
    def total_login_hours(self) -> float:
        """Total login time in hours"""
        return self.total_login_time_seconds / 3600

    @property
    def total_active_hours(self) -> float:
        """Total active time in hours"""
        return self.total_active_time_seconds / 3600

    @property
    def total_storage_gb(self) -> float:
        """Total storage in GB"""
        return self.total_storage_used_bytes / (1024 ** 3)

    @property
    def avg_session_duration_hours(self) -> float:
        """Average session duration in hours"""
        if self.total_sessions == 0:
            return 0.0
        return self.total_login_hours / self.total_sessions

    @property
    def api_error_rate(self) -> float:
        """API error rate (0.0 - 1.0)"""
        if self.total_api_calls == 0:
            return 0.0
        return self.api_error_count / self.total_api_calls
```

### UserUsageTimeSeries Model

```python
# app/models/user_usage_timeseries.py
from sqlalchemy import Column, String, Integer, Float, BigInteger, DateTime, Date, ForeignKey, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, date
import uuid
import enum

from app.db.base import Base

class TimeSeriesGranularity(str, enum.Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class UserUsageTimeSeries(Base):
    """Time-series usage data for trend analysis"""
    __tablename__ = "user_usage_timeseries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # User and time
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    granularity = Column(String(20), nullable=False, index=True)  # TimeSeriesGranularity
    timestamp = Column(DateTime, nullable=False, index=True)  # Start of period
    date = Column(Date, nullable=False, index=True)  # For daily/weekly/monthly queries

    # Session metrics
    sessions_count = Column(Integer, default=0)
    active_time_seconds = Column(Integer, default=0)
    api_calls_count = Column(Integer, default=0)

    # Resource usage
    gpu_seconds = Column(Integer, default=0)
    storage_bytes = Column(BigInteger, default=0)

    # Job counts
    training_jobs_started = Column(Integer, default=0)
    training_jobs_completed = Column(Integer, default=0)
    training_jobs_failed = Column(Integer, default=0)

    # Cost
    estimated_cost_usd = Column(Float, default=0.0)

    # Metrics summary
    metrics_summary = Column(JSON, nullable=True)
    # {
    #   "avg_training_time_hours": 2.3,
    #   "peak_concurrent_jobs": 5,
    #   "top_model": "yolo11n"
    # }

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")

    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_user_granularity_timestamp', 'user_id', 'granularity', 'timestamp', unique=True),
        Index('idx_granularity_date', 'granularity', 'date'),
    )
```

### ActivityEvent Model

```python
# app/models/activity_event.py
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class ActivityEvent(Base):
    """Raw activity events for detailed tracking"""
    __tablename__ = "activity_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # User reference
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.id"), nullable=True, index=True)

    # Event type
    event_type = Column(String(100), nullable=False, index=True)
    # "training_started", "inference_completed", "dataset_uploaded",
    # "api_call", "page_view", "export_completed", etc.

    # Event category
    category = Column(String(50), nullable=False, index=True)
    # "training", "inference", "dataset", "api", "ui", "export", "deployment"

    # Resource references
    resource_type = Column(String(50), nullable=True)  # "training_job", "dataset", "project"
    resource_id = Column(String(255), nullable=True, index=True)

    # Event data
    event_data = Column(JSON, nullable=True)
    # {
    #   "model_name": "yolo11n",
    #   "duration_seconds": 3600,
    #   "gpu_hours": 1.0,
    #   "success": true
    # }

    # Request info (for API events)
    endpoint = Column(String(500), nullable=True)
    method = Column(String(10), nullable=True)  # GET, POST, etc.
    status_code = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User")
    session = relationship("UserSession")

    # Partitioning hint: Consider partitioning by timestamp (monthly)
    # for large-scale deployments
```

### AuditLog Model

```python
# app/models/audit_log.py
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Text, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class AuditAction(str, enum.Enum):
    """Audit log action types"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    INVITE = "invite"
    REMOVE = "remove"
    GRANT_ROLE = "grant_role"
    REVOKE_ROLE = "revoke_role"

class AuditEntityType(str, enum.Enum):
    """Types of entities being audited"""
    USER = "user"
    PROJECT = "project"
    EXPERIMENT = "experiment"
    DATASET = "dataset"
    TRAINING_JOB = "training_job"
    PROJECT_MEMBER = "project_member"
    DATASET_MEMBER = "dataset_member"
    ORGANIZATION = "organization"

class AuditLog(Base):
    """
    Audit log for compliance, security, and debugging

    Tracks all important changes: who did what, when, and how
    """
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Who performed the action
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    user_email = Column(String(255), nullable=False)  # Cached for historical record
    user_role = Column(String(50), nullable=False)     # Role at time of action

    # What entity was affected
    entity_type = Column(String(50), nullable=False, index=True)  # AuditEntityType
    entity_id = Column(String(255), nullable=False, index=True)
    entity_name = Column(String(500), nullable=True)  # Cached for readability

    # What action was performed
    action = Column(String(50), nullable=False, index=True)  # AuditAction

    # Changes (for UPDATE actions)
    changes = Column(JSON, nullable=True)
    # {
    #   "field_name": {
    #     "old": "previous_value",
    #     "new": "new_value"
    #   }
    # }

    # Additional context
    context = Column(JSON, nullable=True)
    # {
    #   "project_id": "uuid",  // For project-scoped actions
    #   "reason": "User request",
    #   "ip_address": "192.168.1.1"
    # }

    # Description (human-readable)
    description = Column(Text, nullable=True)
    # "User john@example.com created project 'Object Detection Research'"

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User")

    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_entity_action', 'entity_type', 'entity_id', 'action'),
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_timestamp_desc', 'timestamp desc'),
    )

    @classmethod
    def log_create(
        cls,
        user: "User",
        entity_type: str,
        entity_id: str,
        entity_name: str = None,
        context: dict = None
    ):
        """Create audit log for CREATE action"""
        return cls(
            user_id=user.id,
            user_email=user.email,
            user_role=user.role,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            action=AuditAction.CREATE,
            context=context,
            description=f"User {user.email} created {entity_type} '{entity_name or entity_id}'"
        )

    @classmethod
    def log_update(
        cls,
        user: "User",
        entity_type: str,
        entity_id: str,
        entity_name: str = None,
        changes: dict = None,
        context: dict = None
    ):
        """Create audit log for UPDATE action"""

        # Format changes description
        if changes:
            changes_desc = ", ".join([f"{field}: {v['old']} → {v['new']}" for field, v in changes.items()])
        else:
            changes_desc = ""

        return cls(
            user_id=user.id,
            user_email=user.email,
            user_role=user.role,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            action=AuditAction.UPDATE,
            changes=changes,
            context=context,
            description=f"User {user.email} updated {entity_type} '{entity_name or entity_id}': {changes_desc}"
        )

    @classmethod
    def log_delete(
        cls,
        user: "User",
        entity_type: str,
        entity_id: str,
        entity_name: str = None,
        context: dict = None
    ):
        """Create audit log for DELETE action"""
        return cls(
            user_id=user.id,
            user_email=user.email,
            user_role=user.role,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            action=AuditAction.DELETE,
            context=context,
            description=f"User {user.email} deleted {entity_type} '{entity_name or entity_id}'"
        )

    @classmethod
    def log_invite(
        cls,
        user: "User",
        entity_type: str,
        entity_id: str,
        entity_name: str = None,
        invitee_email: str = None,
        role: str = None,
        context: dict = None
    ):
        """Create audit log for INVITE action"""
        return cls(
            user_id=user.id,
            user_email=user.email,
            user_role=user.role,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            action=AuditAction.INVITE,
            context={**(context or {}), "invitee_email": invitee_email, "role": role},
            description=f"User {user.email} invited {invitee_email} to {entity_type} '{entity_name or entity_id}' as {role}"
        )

    @classmethod
    def log_grant_role(
        cls,
        user: "User",
        target_user_id: str,
        target_user_email: str,
        old_role: str,
        new_role: str,
        context: dict = None
    ):
        """Create audit log for role grant"""
        return cls(
            user_id=user.id,
            user_email=user.email,
            user_role=user.role,
            entity_type=AuditEntityType.USER,
            entity_id=target_user_id,
            entity_name=target_user_email,
            action=AuditAction.GRANT_ROLE,
            changes={"role": {"old": old_role, "new": new_role}},
            context=context,
            description=f"User {user.email} changed role of {target_user_email} from {old_role} to {new_role}"
        )
```

---

## Audit Log Service

### AuditLogger Service

```python
# app/services/audit_logger.py
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.audit_log import AuditLog, AuditAction, AuditEntityType

class AuditLogger:
    """Centralized audit logging service"""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== User Actions ====================

    async def log_user_registered(
        self,
        user: User,
        invitation_id: str = None
    ):
        """Log user registration"""
        audit = AuditLog.log_create(
            user=user,  # Self-registration
            entity_type=AuditEntityType.USER,
            entity_id=str(user.id),
            entity_name=user.email,
            context={
                "invitation_id": invitation_id,
                "role": user.role
            }
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_user_deleted(
        self,
        admin_user: User,
        deleted_user: User,
        reason: str = None
    ):
        """Log user deletion"""
        audit = AuditLog.log_delete(
            user=admin_user,
            entity_type=AuditEntityType.USER,
            entity_id=str(deleted_user.id),
            entity_name=deleted_user.email,
            context={"reason": reason}
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_role_changed(
        self,
        admin_user: User,
        target_user: User,
        old_role: str,
        new_role: str
    ):
        """Log role change"""
        audit = AuditLog.log_grant_role(
            user=admin_user,
            target_user_id=str(target_user.id),
            target_user_email=target_user.email,
            old_role=old_role,
            new_role=new_role
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_user_updated(
        self,
        user: User,
        changes: Dict[str, Dict[str, Any]]
    ):
        """Log user profile update"""
        audit = AuditLog.log_update(
            user=user,  # Self-update
            entity_type=AuditEntityType.USER,
            entity_id=str(user.id),
            entity_name=user.email,
            changes=changes
        )
        self.db.add(audit)
        await self.db.commit()

    # ==================== Project Actions ====================

    async def log_project_created(
        self,
        user: User,
        project: "Project"
    ):
        """Log project creation"""
        audit = AuditLog.log_create(
            user=user,
            entity_type=AuditEntityType.PROJECT,
            entity_id=str(project.id),
            entity_name=project.name
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_project_updated(
        self,
        user: User,
        project: "Project",
        changes: Dict[str, Dict[str, Any]]
    ):
        """Log project update"""
        audit = AuditLog.log_update(
            user=user,
            entity_type=AuditEntityType.PROJECT,
            entity_id=str(project.id),
            entity_name=project.name,
            changes=changes
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_project_deleted(
        self,
        user: User,
        project: "Project"
    ):
        """Log project deletion"""
        audit = AuditLog.log_delete(
            user=user,
            entity_type=AuditEntityType.PROJECT,
            entity_id=str(project.id),
            entity_name=project.name,
            context={
                "num_experiments": project.num_experiments,
                "num_members": project.num_members
            }
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_project_member_invited(
        self,
        user: User,
        project: "Project",
        invitee_email: str,
        role: str
    ):
        """Log project invitation"""
        audit = AuditLog.log_invite(
            user=user,
            entity_type=AuditEntityType.PROJECT,
            entity_id=str(project.id),
            entity_name=project.name,
            invitee_email=invitee_email,
            role=role
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_project_member_removed(
        self,
        user: User,
        project: "Project",
        removed_user: User
    ):
        """Log project member removal"""
        audit = AuditLog(
            user_id=user.id,
            user_email=user.email,
            user_role=user.role,
            entity_type=AuditEntityType.PROJECT_MEMBER,
            entity_id=f"{project.id}:{removed_user.id}",
            entity_name=f"{project.name} - {removed_user.email}",
            action=AuditAction.REMOVE,
            context={
                "project_id": str(project.id),
                "removed_user_id": str(removed_user.id)
            },
            description=f"User {user.email} removed {removed_user.email} from project '{project.name}'"
        )
        self.db.add(audit)
        await self.db.commit()

    # ==================== Experiment Actions ====================

    async def log_experiment_created(
        self,
        user: User,
        experiment: "Experiment"
    ):
        """Log experiment creation"""
        audit = AuditLog.log_create(
            user=user,
            entity_type=AuditEntityType.EXPERIMENT,
            entity_id=str(experiment.id),
            entity_name=experiment.name,
            context={
                "project_id": str(experiment.project_id),
                "mlflow_experiment_id": experiment.mlflow_experiment_id
            }
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_experiment_deleted(
        self,
        user: User,
        experiment: "Experiment"
    ):
        """Log experiment deletion"""
        audit = AuditLog.log_delete(
            user=user,
            entity_type=AuditEntityType.EXPERIMENT,
            entity_id=str(experiment.id),
            entity_name=experiment.name,
            context={
                "project_id": str(experiment.project_id),
                "num_runs": experiment.num_runs
            }
        )
        self.db.add(audit)
        await self.db.commit()

    # ==================== Dataset Actions ====================

    async def log_dataset_created(
        self,
        user: User,
        dataset: "Dataset"
    ):
        """Log dataset creation"""
        audit = AuditLog.log_create(
            user=user,
            entity_type=AuditEntityType.DATASET,
            entity_id=str(dataset.id),
            entity_name=dataset.name,
            context={
                "visibility": dataset.visibility,
                "num_images": dataset.num_images
            }
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_dataset_updated(
        self,
        user: User,
        dataset: "Dataset",
        changes: Dict[str, Dict[str, Any]]
    ):
        """Log dataset update"""
        audit = AuditLog.log_update(
            user=user,
            entity_type=AuditEntityType.DATASET,
            entity_id=str(dataset.id),
            entity_name=dataset.name,
            changes=changes
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_dataset_deleted(
        self,
        user: User,
        dataset: "Dataset"
    ):
        """Log dataset deletion"""
        audit = AuditLog.log_delete(
            user=user,
            entity_type=AuditEntityType.DATASET,
            entity_id=str(dataset.id),
            entity_name=dataset.name,
            context={
                "num_images": dataset.num_images,
                "size_bytes": dataset.size_bytes
            }
        )
        self.db.add(audit)
        await self.db.commit()

    async def log_dataset_member_invited(
        self,
        user: User,
        dataset: "Dataset",
        invitee_email: str,
        role: str
    ):
        """Log dataset invitation"""
        audit = AuditLog.log_invite(
            user=user,
            entity_type=AuditEntityType.DATASET,
            entity_id=str(dataset.id),
            entity_name=dataset.name,
            invitee_email=invitee_email,
            role=role
        )
        self.db.add(audit)
        await self.db.commit()

    # ==================== Query Methods ====================

    async def get_entity_history(
        self,
        entity_type: str,
        entity_id: str,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit history for a specific entity"""
        result = await self.db.execute(
            select(AuditLog)
            .where(AuditLog.entity_type == entity_type)
            .where(AuditLog.entity_id == entity_id)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def get_user_actions(
        self,
        user_id: UUID,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get all actions performed by a user"""
        result = await self.db.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()
```

---

## Audit Log API Endpoints

```python
# app/api/v1/audit.py
from fastapi import APIRouter, Depends, Query, HTTPException
from datetime import date, datetime, timedelta
from typing import List, Optional

from app.models.user import User, UserRole
from app.models.audit_log import AuditLog
from app.services.audit_logger import AuditLogger

router = APIRouter(prefix="/audit", tags=["audit"])

@router.get("/me")
async def get_my_audit_logs(
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    action: Optional[str] = Query(None),
    entity_type: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get audit logs for current user's actions

    Users can see their own actions
    """

    query = select(AuditLog).where(AuditLog.user_id == current_user.id)

    if action:
        query = query.where(AuditLog.action == action)

    if entity_type:
        query = query.where(AuditLog.entity_type == entity_type)

    if start_date:
        query = query.where(AuditLog.timestamp >= datetime.combine(start_date, datetime.min.time()))

    if end_date:
        query = query.where(AuditLog.timestamp <= datetime.combine(end_date, datetime.max.time()))

    query = query.order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    logs = result.scalars().all()

    return {
        "logs": [
            {
                "id": log.id,
                "action": log.action,
                "entity_type": log.entity_type,
                "entity_name": log.entity_name,
                "description": log.description,
                "changes": log.changes,
                "timestamp": log.timestamp
            }
            for log in logs
        ],
        "total": await get_audit_count(current_user.id),
        "limit": limit,
        "offset": offset
    }


@router.get("/entity/{entity_type}/{entity_id}")
async def get_entity_audit_logs(
    entity_type: str,
    entity_id: str,
    limit: int = Query(100, le=1000),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get audit logs for a specific entity

    Requires permission to view the entity
    """

    # Verify permission to view entity
    await verify_entity_access(current_user, entity_type, entity_id, db)

    logger = AuditLogger(db)
    logs = await logger.get_entity_history(entity_type, entity_id, limit)

    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "logs": [
            {
                "id": log.id,
                "user_email": log.user_email,
                "user_role": log.user_role,
                "action": log.action,
                "description": log.description,
                "changes": log.changes,
                "timestamp": log.timestamp
            }
            for log in logs
        ]
    }


@router.get("/project/{project_id}")
async def get_project_audit_logs(
    project_id: UUID,
    limit: int = Query(100, le=1000),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all audit logs related to a project

    Includes project actions, member changes, experiment actions
    """

    # Verify project access
    project = await get_project_or_404(project_id, db)
    if not await can_view_project(current_user, project):
        raise HTTPException(status_code=403, detail="Not authorized")

    # Query all logs related to this project
    result = await db.execute(
        select(AuditLog)
        .where(
            (AuditLog.entity_id == str(project_id)) |
            (AuditLog.context['project_id'].astext == str(project_id))
        )
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
    )

    logs = result.scalars().all()

    return {
        "project_id": project_id,
        "logs": logs
    }


@router.get("/organization")
async def get_organization_audit_logs(
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    action: Optional[str] = Query(None),
    entity_type: Optional[str] = Query(None),
    user_email: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get organization-wide audit logs

    Requires ADMIN or MANAGER role
    """

    if current_user.role not in [UserRole.ADMIN, UserRole.MANAGER]:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Query logs for all users in organization
    query = (
        select(AuditLog)
        .join(User, AuditLog.user_id == User.id)
        .where(User.organization_id == current_user.organization_id)
    )

    if action:
        query = query.where(AuditLog.action == action)

    if entity_type:
        query = query.where(AuditLog.entity_type == entity_type)

    if user_email:
        query = query.where(AuditLog.user_email == user_email)

    if start_date:
        query = query.where(AuditLog.timestamp >= datetime.combine(start_date, datetime.min.time()))

    if end_date:
        query = query.where(AuditLog.timestamp <= datetime.combine(end_date, datetime.max.time()))

    query = query.order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    logs = result.scalars().all()

    return {
        "organization_id": current_user.organization_id,
        "logs": logs,
        "limit": limit,
        "offset": offset
    }
```

---

## Usage Tracking Service

### ActivityTracker Service

```python
# app/services/activity_tracker.py
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.user import User
from app.models.user_session import UserSession
from app.models.user_usage_stats import UserUsageStats
from app.models.activity_event import ActivityEvent

class ActivityTracker:
    """Centralized activity tracking service"""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== Session Tracking ====================

    async def track_login(
        self,
        user_id: UUID,
        user_agent: str = None,
        ip_address: str = None
    ) -> UserSession:
        """Track user login"""

        # Parse user agent
        device_info = self._parse_user_agent(user_agent)

        session = UserSession(
            user_id=user_id,
            session_token=self._generate_session_token(),
            user_agent=user_agent,
            ip_address=ip_address,
            device_type=device_info.get("device_type"),
            browser=device_info.get("browser"),
            os=device_info.get("os"),
            login_at=datetime.utcnow()
        )

        self.db.add(session)

        # Update user stats
        stats = await self._get_or_create_stats(user_id)
        stats.total_sessions += 1
        stats.last_login_at = datetime.utcnow()

        # Log activity event
        await self._log_event(
            user_id=user_id,
            session_id=session.id,
            event_type="user_login",
            category="session",
            event_data={
                "device_type": device_info.get("device_type"),
                "browser": device_info.get("browser")
            }
        )

        await self.db.commit()

        return session

    async def track_logout(self, session_id: UUID):
        """Track user logout"""

        session = await self.db.get(UserSession, session_id)
        if not session:
            return

        now = datetime.utcnow()
        session.logout_at = now
        session.is_active = False

        # Calculate durations
        if session.login_at:
            session.total_duration_seconds = int((now - session.login_at).total_seconds())

        # Update user stats
        stats = await self._get_or_create_stats(session.user_id)
        stats.total_login_time_seconds += session.total_duration_seconds
        stats.total_active_time_seconds += session.active_duration_seconds
        stats.total_idle_time_seconds += session.idle_duration_seconds
        stats.last_activity_at = now

        # Log event
        await self._log_event(
            user_id=session.user_id,
            session_id=session.id,
            event_type="user_logout",
            category="session",
            event_data={
                "duration_hours": session.total_duration_hours,
                "active_hours": session.active_duration_hours
            }
        )

        await self.db.commit()

    async def track_activity(self, session_id: UUID):
        """Track user activity (heartbeat)"""

        session = await self.db.get(UserSession, session_id)
        if not session:
            return

        session.update_activity()

        # Update user stats
        stats = await self._get_or_create_stats(session.user_id)
        stats.last_activity_at = datetime.utcnow()

        await self.db.commit()

    # ==================== Resource Usage Tracking ====================

    async def track_training_job(
        self,
        user_id: UUID,
        training_job_id: UUID,
        event_type: str,  # "started", "completed", "failed", "cancelled"
        gpu_seconds: int = 0,
        model_name: str = None,
        framework: str = None
    ):
        """Track training job lifecycle"""

        stats = await self._get_or_create_stats(user_id)

        if event_type == "started":
            stats.total_training_jobs += 1

        elif event_type == "completed":
            stats.completed_training_jobs += 1
            stats.total_gpu_seconds += gpu_seconds

            # Update feature usage
            if "feature_usage_counts" not in stats.feature_usage_counts:
                stats.feature_usage_counts = {}
            stats.feature_usage_counts["training"] = stats.feature_usage_counts.get("training", 0) + 1

            # Update model usage
            if model_name:
                if "model_usage_counts" not in stats.model_usage_counts:
                    stats.model_usage_counts = {}
                stats.model_usage_counts[model_name] = stats.model_usage_counts.get(model_name, 0) + 1

            # Update framework usage
            if framework:
                if "framework_usage_counts" not in stats.framework_usage_counts:
                    stats.framework_usage_counts = {}
                stats.framework_usage_counts[framework] = stats.framework_usage_counts.get(framework, 0) + 1

        elif event_type == "failed":
            stats.failed_training_jobs += 1

        elif event_type == "cancelled":
            stats.cancelled_training_jobs += 1

        # Update success rate
        total = stats.completed_training_jobs + stats.failed_training_jobs
        if total > 0:
            stats.training_success_rate = stats.completed_training_jobs / total

        # Log event
        await self._log_event(
            user_id=user_id,
            event_type=f"training_{event_type}",
            category="training",
            resource_type="training_job",
            resource_id=str(training_job_id),
            event_data={
                "model_name": model_name,
                "framework": framework,
                "gpu_seconds": gpu_seconds
            }
        )

        await self.db.commit()

    async def track_api_call(
        self,
        user_id: UUID,
        session_id: UUID,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int
    ):
        """Track API call"""

        stats = await self._get_or_create_stats(user_id)
        stats.total_api_calls += 1

        if status_code >= 400:
            stats.api_error_count += 1

        # Update endpoint counts
        if "api_calls_by_endpoint" not in stats.api_calls_by_endpoint:
            stats.api_calls_by_endpoint = {}
        stats.api_calls_by_endpoint[endpoint] = stats.api_calls_by_endpoint.get(endpoint, 0) + 1

        # Update session
        session = await self.db.get(UserSession, session_id)
        if session:
            session.api_calls_count += 1

        # Log event
        await self._log_event(
            user_id=user_id,
            session_id=session_id,
            event_type="api_call",
            category="api",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms
        )

        await self.db.commit()

    async def track_storage_usage(
        self,
        user_id: UUID,
        storage_type: str,  # "dataset", "checkpoint", "export"
        bytes_delta: int
    ):
        """Track storage usage changes"""

        stats = await self._get_or_create_stats(user_id)

        stats.total_storage_used_bytes += bytes_delta

        if storage_type == "dataset":
            stats.dataset_storage_bytes += bytes_delta
        elif storage_type == "checkpoint":
            stats.checkpoint_storage_bytes += bytes_delta
        elif storage_type == "export":
            stats.export_storage_bytes += bytes_delta

        await self.db.commit()

    # ==================== Cost Estimation ====================

    async def update_cost_estimates(self, user_id: UUID):
        """Update cost estimates based on usage"""

        stats = await self._get_or_create_stats(user_id)

        # GPU cost: $1.50/hour (example rate)
        gpu_cost = stats.total_gpu_hours * 1.50

        # Storage cost: $0.02/GB/month (example rate)
        storage_cost = stats.total_storage_gb * 0.02

        # Inference cost: $0.01/1000 requests (example rate)
        inference_cost = stats.total_inference_jobs * 0.00001

        stats.estimated_gpu_cost_usd = gpu_cost
        stats.estimated_storage_cost_usd = storage_cost
        stats.estimated_inference_cost_usd = inference_cost
        stats.estimated_total_cost_usd = gpu_cost + storage_cost + inference_cost

        await self.db.commit()

    # ==================== Helper Methods ====================

    async def _get_or_create_stats(self, user_id: UUID) -> UserUsageStats:
        """Get or create user usage stats"""

        result = await self.db.execute(
            select(UserUsageStats).where(UserUsageStats.user_id == user_id)
        )
        stats = result.scalar_one_or_none()

        if not stats:
            stats = UserUsageStats(user_id=user_id)
            self.db.add(stats)

        return stats

    async def _log_event(
        self,
        user_id: UUID,
        event_type: str,
        category: str,
        session_id: UUID = None,
        resource_type: str = None,
        resource_id: str = None,
        event_data: Dict[str, Any] = None,
        endpoint: str = None,
        method: str = None,
        status_code: int = None,
        response_time_ms: int = None
    ):
        """Log activity event"""

        event = ActivityEvent(
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            category=category,
            resource_type=resource_type,
            resource_id=resource_id,
            event_data=event_data,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            timestamp=datetime.utcnow()
        )

        self.db.add(event)

    def _parse_user_agent(self, user_agent: str) -> Dict[str, str]:
        """Parse user agent string"""
        # Use user-agents library or similar
        # Simplified example
        return {
            "device_type": "desktop",
            "browser": "Chrome",
            "os": "Windows"
        }

    def _generate_session_token(self) -> str:
        """Generate unique session token"""
        import secrets
        return secrets.token_urlsafe(32)
```

---

## Time Series Aggregation

### Aggregation Service

```python
# app/services/timeseries_aggregator.py
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.user_usage_timeseries import UserUsageTimeSeries, TimeSeriesGranularity
from app.models.activity_event import ActivityEvent
from app.models.training_job import TrainingJob

class TimeSeriesAggregator:
    """Aggregate usage data into time series"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def aggregate_hourly(self, user_id: UUID, hour: datetime):
        """Aggregate usage data for a specific hour"""

        # Round to hour
        hour_start = hour.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)

        # Query events in this hour
        result = await self.db.execute(
            select(ActivityEvent)
            .where(ActivityEvent.user_id == user_id)
            .where(ActivityEvent.timestamp >= hour_start)
            .where(ActivityEvent.timestamp < hour_end)
        )
        events = result.scalars().all()

        # Aggregate metrics
        sessions_count = len({e.session_id for e in events if e.session_id})
        api_calls_count = len([e for e in events if e.category == "api"])

        # Training jobs
        training_events = [e for e in events if e.category == "training"]
        training_started = len([e for e in training_events if e.event_type == "training_started"])
        training_completed = len([e for e in training_events if e.event_type == "training_completed"])
        training_failed = len([e for e in training_events if e.event_type == "training_failed"])

        # GPU usage
        gpu_seconds = sum(
            e.event_data.get("gpu_seconds", 0)
            for e in events
            if e.event_data and "gpu_seconds" in e.event_data
        )

        # Cost
        cost = gpu_seconds / 3600 * 1.50  # $1.50/hour

        # Create or update time series record
        result = await self.db.execute(
            select(UserUsageTimeSeries)
            .where(UserUsageTimeSeries.user_id == user_id)
            .where(UserUsageTimeSeries.granularity == TimeSeriesGranularity.HOURLY)
            .where(UserUsageTimeSeries.timestamp == hour_start)
        )
        ts = result.scalar_one_or_none()

        if not ts:
            ts = UserUsageTimeSeries(
                user_id=user_id,
                granularity=TimeSeriesGranularity.HOURLY,
                timestamp=hour_start,
                date=hour_start.date()
            )
            self.db.add(ts)

        # Update metrics
        ts.sessions_count = sessions_count
        ts.api_calls_count = api_calls_count
        ts.training_jobs_started = training_started
        ts.training_jobs_completed = training_completed
        ts.training_jobs_failed = training_failed
        ts.gpu_seconds = gpu_seconds
        ts.estimated_cost_usd = cost

        await self.db.commit()

    async def aggregate_daily(self, user_id: UUID, day: date):
        """Aggregate hourly data into daily"""

        day_start = datetime.combine(day, datetime.min.time())
        day_end = day_start + timedelta(days=1)

        # Query hourly records for this day
        result = await self.db.execute(
            select(UserUsageTimeSeries)
            .where(UserUsageTimeSeries.user_id == user_id)
            .where(UserUsageTimeSeries.granularity == TimeSeriesGranularity.HOURLY)
            .where(UserUsageTimeSeries.timestamp >= day_start)
            .where(UserUsageTimeSeries.timestamp < day_end)
        )
        hourly_records = result.scalars().all()

        # Aggregate
        daily_ts = UserUsageTimeSeries(
            user_id=user_id,
            granularity=TimeSeriesGranularity.DAILY,
            timestamp=day_start,
            date=day,
            sessions_count=sum(r.sessions_count for r in hourly_records),
            api_calls_count=sum(r.api_calls_count for r in hourly_records),
            gpu_seconds=sum(r.gpu_seconds for r in hourly_records),
            training_jobs_started=sum(r.training_jobs_started for r in hourly_records),
            training_jobs_completed=sum(r.training_jobs_completed for r in hourly_records),
            training_jobs_failed=sum(r.training_jobs_failed for r in hourly_records),
            estimated_cost_usd=sum(r.estimated_cost_usd for r in hourly_records)
        )

        self.db.add(daily_ts)
        await self.db.commit()

    async def aggregate_all_pending(self):
        """Aggregate all pending time series data"""

        # Run hourly aggregation for past 24 hours
        now = datetime.utcnow()
        for hour_offset in range(24):
            hour = now - timedelta(hours=hour_offset)
            # Aggregate for all users (batch job)
            # ... implementation

        # Run daily aggregation for yesterday
        yesterday = (now - timedelta(days=1)).date()
        # ... implementation
```

---

## Analytics API Endpoints

### User Analytics

```python
# app/api/v1/analytics.py
from fastapi import APIRouter, Depends, Query
from datetime import date, datetime, timedelta
from typing import List, Optional

from app.models.user import User
from app.services.activity_tracker import ActivityTracker
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/me")
async def get_my_analytics(
    period: str = Query("30d", regex="^(7d|30d|90d|1y|all)$"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current user's analytics summary

    Period options: 7d, 30d, 90d, 1y, all
    """

    service = AnalyticsService(db)

    # Get overall stats
    stats = await service.get_user_stats(current_user.id)

    # Get time series data
    end_date = date.today()
    if period == "7d":
        start_date = end_date - timedelta(days=7)
    elif period == "30d":
        start_date = end_date - timedelta(days=30)
    elif period == "90d":
        start_date = end_date - timedelta(days=90)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    else:  # all
        start_date = stats.created_at.date()

    timeseries = await service.get_timeseries(
        user_id=current_user.id,
        start_date=start_date,
        end_date=end_date,
        granularity="daily"
    )

    # Get insights
    insights = await service.generate_insights(current_user.id, period)

    return {
        "user_id": current_user.id,
        "period": period,
        "summary": {
            "total_sessions": stats.total_sessions,
            "total_active_hours": stats.total_active_hours,
            "total_training_jobs": stats.total_training_jobs,
            "training_success_rate": stats.training_success_rate,
            "total_gpu_hours": stats.total_gpu_hours,
            "total_storage_gb": stats.total_storage_gb,
            "estimated_cost_usd": stats.estimated_total_cost_usd,
            "projects_owned": stats.projects_owned,
            "datasets_owned": stats.datasets_owned,
            "users_invited": stats.users_invited
        },
        "timeseries": timeseries,
        "insights": insights,
        "top_models": get_top_items(stats.model_usage_counts, 5),
        "top_features": get_top_items(stats.feature_usage_counts, 5),
        "api_usage": {
            "total_calls": stats.total_api_calls,
            "error_rate": stats.api_error_rate,
            "top_endpoints": get_top_items(stats.api_calls_by_endpoint, 10)
        }
    }


@router.get("/sessions")
async def get_session_history(
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """Get user's session history"""

    result = await db.execute(
        select(UserSession)
        .where(UserSession.user_id == current_user.id)
        .order_by(UserSession.login_at.desc())
        .limit(limit)
        .offset(offset)
    )
    sessions = result.scalars().all()

    return {
        "sessions": [
            {
                "id": s.id,
                "login_at": s.login_at,
                "logout_at": s.logout_at,
                "duration_hours": s.total_duration_hours,
                "active_hours": s.active_duration_hours,
                "device_type": s.device_type,
                "browser": s.browser,
                "api_calls": s.api_calls_count,
                "is_active": s.is_active
            }
            for s in sessions
        ],
        "total": await get_session_count(current_user.id),
        "limit": limit,
        "offset": offset
    }


@router.get("/activity-timeline")
async def get_activity_timeline(
    start_date: date = Query(...),
    end_date: date = Query(...),
    event_types: Optional[List[str]] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed activity timeline

    Useful for activity heatmaps and detailed analysis
    """

    query = select(ActivityEvent).where(
        ActivityEvent.user_id == current_user.id
    ).where(
        ActivityEvent.timestamp >= datetime.combine(start_date, datetime.min.time())
    ).where(
        ActivityEvent.timestamp <= datetime.combine(end_date, datetime.max.time())
    )

    if event_types:
        query = query.where(ActivityEvent.event_type.in_(event_types))

    query = query.order_by(ActivityEvent.timestamp.desc()).limit(1000)

    result = await db.execute(query)
    events = result.scalars().all()

    return {
        "events": [
            {
                "id": e.id,
                "type": e.event_type,
                "category": e.category,
                "timestamp": e.timestamp,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
                "data": e.event_data
            }
            for e in events
        ],
        "count": len(events)
    }


@router.get("/cost-breakdown")
async def get_cost_breakdown(
    period: str = Query("30d"),
    current_user: User = Depends(get_current_user)
):
    """Get detailed cost breakdown"""

    stats = await get_user_stats(current_user.id)

    return {
        "period": period,
        "total_cost_usd": stats.estimated_total_cost_usd,
        "breakdown": {
            "gpu": {
                "cost_usd": stats.estimated_gpu_cost_usd,
                "usage_hours": stats.total_gpu_hours,
                "rate_per_hour": 1.50
            },
            "storage": {
                "cost_usd": stats.estimated_storage_cost_usd,
                "usage_gb": stats.total_storage_gb,
                "rate_per_gb_month": 0.02
            },
            "inference": {
                "cost_usd": stats.estimated_inference_cost_usd,
                "requests": stats.total_inference_jobs,
                "rate_per_1000_requests": 0.01
            }
        },
        "trend": await get_cost_trend(current_user.id, period)
    }


# ==================== Admin Analytics ====================

@router.get("/organization")
async def get_organization_analytics(
    period: str = Query("30d"),
    current_user: User = Depends(get_current_user)
):
    """
    Get organization-wide analytics

    Requires ADMIN or MANAGER role
    """

    if current_user.role not in [UserRole.ADMIN, UserRole.MANAGER]:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Get all users in organization
    result = await db.execute(
        select(User).where(User.organization_id == current_user.organization_id)
    )
    users = result.scalars().all()

    # Aggregate organization metrics
    org_stats = await aggregate_organization_stats(users)

    return {
        "organization_id": current_user.organization_id,
        "period": period,
        "summary": org_stats,
        "top_users": await get_top_users_by_usage(users, limit=10),
        "resource_utilization": await get_resource_utilization(users),
        "cost_allocation": await get_cost_allocation(users)
    }


def get_top_items(counts_dict: dict, limit: int) -> List[dict]:
    """Get top N items from counts dictionary"""
    if not counts_dict:
        return []

    sorted_items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
    return [
        {"name": name, "count": count}
        for name, count in sorted_items[:limit]
    ]
```

---

## KPI Definitions

### Platform-level KPIs

```python
# Key Performance Indicators

KPIs = {
    # User Engagement
    "daily_active_users": "Number of users active in past 24 hours",
    "weekly_active_users": "Number of users active in past 7 days",
    "monthly_active_users": "Number of users active in past 30 days",
    "avg_session_duration": "Average session duration (hours)",
    "retention_rate_30d": "% of users returning after 30 days",

    # Resource Utilization
    "total_gpu_hours": "Total GPU hours consumed",
    "avg_gpu_utilization": "Average GPU utilization rate (0-1)",
    "total_storage_tb": "Total storage used (TB)",
    "storage_growth_rate": "Storage growth rate (GB/day)",

    # Training Metrics
    "total_training_jobs": "Total training jobs started",
    "training_success_rate": "% of training jobs completed successfully",
    "avg_training_time": "Average training job duration (hours)",
    "failed_job_rate": "% of training jobs failed",

    # Collaboration
    "total_projects": "Total projects created",
    "avg_project_members": "Average members per project",
    "shared_datasets": "Number of shared datasets",
    "total_invitations_sent": "Total user invitations sent",

    # Cost & Efficiency
    "total_cost_usd": "Total estimated platform cost",
    "cost_per_user": "Average cost per user",
    "cost_per_training_job": "Average cost per training job",
    "cost_growth_rate": "Cost growth rate (%/month)",

    # API Usage
    "total_api_calls": "Total API calls",
    "api_error_rate": "API error rate (%)",
    "avg_api_response_time": "Average API response time (ms)",

    # Feature Adoption
    "export_adoption_rate": "% of users using export feature",
    "xai_adoption_rate": "% of users using XAI features",
    "platform_endpoint_usage": "Number of platform endpoint deployments"
}
```

### KPI Calculation Service

```python
# app/services/kpi_service.py
from datetime import datetime, timedelta, date
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

class KPIService:
    """Calculate platform KPIs"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def calculate_dau(self, target_date: date = None) -> int:
        """Calculate Daily Active Users"""
        if not target_date:
            target_date = date.today()

        day_start = datetime.combine(target_date, datetime.min.time())
        day_end = day_start + timedelta(days=1)

        result = await self.db.execute(
            select(func.count(func.distinct(UserSession.user_id)))
            .where(UserSession.last_activity_at >= day_start)
            .where(UserSession.last_activity_at < day_end)
        )

        return result.scalar()

    async def calculate_training_success_rate(self, period_days: int = 30) -> float:
        """Calculate training job success rate"""
        start_date = datetime.utcnow() - timedelta(days=period_days)

        result = await self.db.execute(
            select(
                func.count(TrainingJob.id).label("total"),
                func.sum(
                    func.case((TrainingJob.status == "completed", 1), else_=0)
                ).label("completed")
            )
            .where(TrainingJob.created_at >= start_date)
        )

        row = result.first()
        if row.total == 0:
            return 0.0

        return row.completed / row.total

    async def calculate_all_kpis(self) -> dict:
        """Calculate all KPIs"""

        return {
            "engagement": {
                "dau": await self.calculate_dau(),
                "wau": await self.calculate_wau(),
                "mau": await self.calculate_mau()
            },
            "training": {
                "success_rate": await self.calculate_training_success_rate(),
                "avg_duration": await self.calculate_avg_training_duration()
            },
            # ... more KPIs
        }
```

---

## Dashboard Design

### Analytics Dashboard Structure

```typescript
// frontend/components/analytics/AnalyticsDashboard.tsx
interface AnalyticsDashboard {
  // Overview Cards
  overview: {
    totalSessions: number;
    activeHours: number;
    trainingJobs: number;
    successRate: number;
    gpuHours: number;
    estimatedCost: number;
  };

  // Charts
  charts: {
    usageOverTime: TimeSeriesChart;      // Line chart: sessions, API calls over time
    resourceUsage: StackedAreaChart;     // GPU, storage usage trends
    costBreakdown: PieChart;             // GPU, storage, inference costs
    trainingMetrics: BarChart;           // Success/fail counts
    activityHeatmap: HeatmapChart;       // Hour-of-day × day-of-week activity
  };

  // Top Lists
  topModels: TopItem[];                  // Most used models
  topFeatures: TopItem[];                // Most used features
  topProjects: Project[];                // Most active projects

  // Recent Activity
  recentActivity: ActivityEvent[];       // Timeline of recent actions

  // Insights
  insights: Insight[];                   // AI-generated insights
}
```

### Example Dashboard API Response

```json
{
  "user_id": "uuid",
  "period": "30d",
  "summary": {
    "total_sessions": 45,
    "total_active_hours": 87.5,
    "total_training_jobs": 152,
    "training_success_rate": 0.87,
    "total_gpu_hours": 234.5,
    "total_storage_gb": 156.8,
    "estimated_cost_usd": 385.20,
    "projects_owned": 5,
    "datasets_owned": 12,
    "users_invited": 8
  },
  "timeseries": [
    {
      "date": "2025-01-01",
      "sessions": 2,
      "api_calls": 145,
      "gpu_hours": 8.5,
      "training_jobs": 5,
      "cost_usd": 12.75
    }
    // ... more days
  ],
  "top_models": [
    {"name": "yolo11n", "count": 45},
    {"name": "resnet50", "count": 30},
    {"name": "efficientnet-b0", "count": 20}
  ],
  "top_features": [
    {"name": "training", "count": 152},
    {"name": "inference", "count": 89},
    {"name": "export", "count": 23}
  ],
  "insights": [
    {
      "type": "cost_alert",
      "severity": "warning",
      "message": "Your GPU usage increased 25% this week",
      "action": "Consider optimizing batch sizes"
    },
    {
      "type": "success_improvement",
      "severity": "info",
      "message": "Training success rate improved from 82% to 87%",
      "action": "Keep up the good work!"
    }
  ]
}
```

---

## Scheduled Jobs

### Background Tasks

```python
# app/tasks/analytics_tasks.py
from celery import Celery
from datetime import datetime, timedelta

app = Celery('analytics_tasks')

@app.task
def aggregate_hourly_stats():
    """
    Run every hour to aggregate usage data

    Schedule: 0 * * * * (every hour at :00)
    """
    from app.services.timeseries_aggregator import TimeSeriesAggregator

    current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    # Aggregate for all users
    # ... implementation

@app.task
def aggregate_daily_stats():
    """
    Run daily to aggregate hourly data

    Schedule: 0 2 * * * (every day at 2:00 AM)
    """
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()

    # Aggregate for all users
    # ... implementation

@app.task
def update_cost_estimates():
    """
    Update cost estimates for all users

    Schedule: 0 3 * * * (every day at 3:00 AM)
    """
    from app.services.activity_tracker import ActivityTracker

    # Update for all users
    # ... implementation

@app.task
def generate_weekly_reports():
    """
    Generate weekly usage reports

    Schedule: 0 9 * * 1 (every Monday at 9:00 AM)
    """
    # Send email reports to users
    # ... implementation
```

---

## Best Practices

1. **Track everything** - Log all user actions for comprehensive analytics
2. **Aggregate frequently** - Run hourly/daily aggregations to keep queries fast
3. **Use time series tables** - Separate raw events from aggregated data
4. **Partition large tables** - Consider partitioning ActivityEvent by month
5. **Cache expensive queries** - Use Redis for frequently accessed stats
6. **Privacy-aware** - Only admins can see organization-wide analytics
7. **Cost transparency** - Show users estimated costs to encourage efficiency
8. **Actionable insights** - Generate insights that users can act on
9. **Performance monitoring** - Track API response times and error rates
10. **Audit trail** - Keep raw events for compliance and debugging

---

## References

- [PROJECT_MEMBERSHIP_DESIGN.md](./PROJECT_MEMBERSHIP_DESIGN.md) - User roles and permissions
- [BACKEND_DESIGN.md](./BACKEND_DESIGN.md) - Core database models
- [ISOLATION_DESIGN.md](./ISOLATION_DESIGN.md) - Security principles
