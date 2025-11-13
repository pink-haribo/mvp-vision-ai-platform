# Project & Membership Design

## Overview

This document defines the **Project** and **Membership** system for the Vision AI Training Platform.

### Key Concepts

1. **Project**: Collaborative workspace for organizing experiments
2. **Experiment**: MLflow-tracked experiment group containing training runs
3. **Membership**: Permission-based access control for Projects and Datasets
4. **Role-based Access Control**: 5-tier user permission system

### Architecture Decision

**Option B (Selected)**: Platform Project as grouping layer, Experiment as MLflow tracking unit

```
Organization
    └── User (with Role)
        └── Project (collaboration unit)
            └── Experiment → MLflow Experiment
                    └── TrainingJob → MLflow Run
                    └── TestRun → MLflow Run
                    └── InferenceJob → MLflow Run
```

**Rationale**:
- Platform Project = Collaboration + Organization
- Experiment = Experiment Tracking + Metrics
- Clear separation of concerns
- MLflow integration without tight coupling

---

## User Role System

### UserRole Enum

```python
# app/models/user.py
class UserRole(str, enum.Enum):
    """5-tier user permission system"""
    ADMIN = "admin"              # All permissions
    MANAGER = "manager"          # Can grant permissions below manager
    ENGINEER_II = "engineer_ii"  # Advanced training features
    ENGINEER_I = "engineer_i"    # Basic training features
    GUEST = "guest"              # Limited: 1 project, 1 dataset, no collaboration
```

### Permission Matrix

| Feature | GUEST | ENGINEER_I | ENGINEER_II | MANAGER | ADMIN |
|---------|-------|------------|-------------|---------|-------|
| **Projects** |
| Create projects | 1 max | ✅ | ✅ | ✅ | ✅ |
| Invite to projects | ❌ | ✅ | ✅ | ✅ | ✅ |
| Delete own projects | ✅ | ✅ | ✅ | ✅ | ✅ |
| Delete others' projects | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Datasets** |
| Create datasets | 1 max | ✅ | ✅ | ✅ | ✅ |
| Upload datasets | ✅ | ✅ | ✅ | ✅ | ✅ |
| Share datasets | ❌ | ✅ | ✅ | ✅ | ✅ |
| Delete own datasets | ✅ | ✅ | ✅ | ✅ | ✅ |
| Delete others' datasets | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Training** |
| Basic training | ✅ | ✅ | ✅ | ✅ | ✅ |
| Advanced features* | ❌ | ❌ | ✅ | ✅ | ✅ |
| Distributed training | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Export/Deploy** |
| Export models | ❌ | ✅ | ✅ | ✅ | ✅ |
| Deploy to platform | ❌ | ✅ | ✅ | ✅ | ✅ |
| **User Management** |
| Invite users | ❌ | ❌ | ❌ | ✅† | ✅ |
| Grant roles | ❌ | ❌ | ❌ | ✅† | ✅ |
| Delete users | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Organization** |
| View org stats | ❌ | ❌ | ❌ | ✅ | ✅ |
| Manage quotas | ❌ | ❌ | ❌ | ❌ | ✅ |

**\* Advanced features**: XAI, advanced optimizers, custom loss functions, etc.
**† Manager**: Can only grant GUEST, ENGINEER_I, ENGINEER_II roles (not MANAGER or ADMIN)

### Role-based Quotas

```python
ROLE_QUOTAS = {
    "guest": {
        "max_projects": 1,
        "max_datasets": 1,
        "max_concurrent_jobs": 1,
        "storage_quota_gb": 10,
        "gpu_hours_per_month": 10
    },
    "engineer_i": {
        "max_projects": 10,
        "max_datasets": 20,
        "max_concurrent_jobs": 3,
        "storage_quota_gb": 100,
        "gpu_hours_per_month": 100
    },
    "engineer_ii": {
        "max_projects": 50,
        "max_datasets": 100,
        "max_concurrent_jobs": 10,
        "storage_quota_gb": 500,
        "gpu_hours_per_month": 500
    },
    "manager": {
        "max_projects": "unlimited",
        "max_datasets": "unlimited",
        "max_concurrent_jobs": 20,
        "storage_quota_gb": 1000,
        "gpu_hours_per_month": "unlimited"
    },
    "admin": {
        # All unlimited
    }
}
```

---

## Database Models

### Updated User Model

```python
# app/models/user.py
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    ENGINEER_II = "engineer_ii"
    ENGINEER_I = "engineer_i"
    GUEST = "guest"

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)

    # Organization info
    company = Column(String(255), nullable=True)
    division = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)  # NEW
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True, index=True)

    # Role and permissions
    role = Column(SQLEnum(UserRole), default=UserRole.GUEST, nullable=False, index=True)

    # Status
    is_active = Column(Boolean, default=True)
    is_email_verified = Column(Boolean, default=False)

    # Avatar (for consistent UI representation)
    avatar_name = Column(String(100), nullable=True)  # "John D", "JD", etc.
    badge_color = Column(String(20), nullable=True)   # "#4F46E5", "indigo", etc.

    # Settings
    timezone = Column(String(50), default="UTC")
    language = Column(String(10), default="en")
    notification_enabled = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="users")
    owned_projects = relationship("Project", back_populates="owner", foreign_keys="Project.owner_id")
    project_memberships = relationship("ProjectMember", back_populates="user")
    owned_datasets = relationship("Dataset", back_populates="owner", foreign_keys="Dataset.owner_id")
    dataset_memberships = relationship("DatasetMember", back_populates="user")

    # Job relationships
    training_jobs = relationship("TrainingJob", back_populates="user")
    test_runs = relationship("TestRun", back_populates="user")
    inference_jobs = relationship("InferenceJob", back_populates="user")
    export_jobs = relationship("ExportJob", back_populates="user")
    deployment_targets = relationship("DeploymentTarget", back_populates="user")

    # Experiment interactions
    starred_experiments = relationship("ExperimentStar", back_populates="user")
    experiment_notes = relationship("ExperimentNote", back_populates="user")

    # Invitations
    sent_invitations = relationship("Invitation", back_populates="inviter", foreign_keys="Invitation.inviter_id")
    received_invitation = relationship("Invitation", back_populates="invitee", foreign_keys="Invitation.invitee_id", uselist=False)

    def can_create_project(self) -> bool:
        """Check if user can create a new project"""
        if self.role in [UserRole.ADMIN, UserRole.MANAGER, UserRole.ENGINEER_II, UserRole.ENGINEER_I]:
            return True

        if self.role == UserRole.GUEST:
            # Check if user already has 1 project
            return len(self.owned_projects) < 1

        return False

    def can_create_dataset(self) -> bool:
        """Check if user can create a new dataset"""
        if self.role in [UserRole.ADMIN, UserRole.MANAGER, UserRole.ENGINEER_II, UserRole.ENGINEER_I]:
            return True

        if self.role == UserRole.GUEST:
            # Check if user already has 1 dataset
            return len(self.owned_datasets) < 1

        return False

    def can_grant_role(self, target_role: UserRole) -> bool:
        """Check if user can grant a specific role"""
        if self.role == UserRole.ADMIN:
            return True

        if self.role == UserRole.MANAGER:
            # Manager can grant GUEST, ENGINEER_I, ENGINEER_II
            return target_role in [UserRole.GUEST, UserRole.ENGINEER_I, UserRole.ENGINEER_II]

        return False

    def has_advanced_features(self) -> bool:
        """Check if user can access advanced training features"""
        return self.role in [UserRole.ADMIN, UserRole.MANAGER, UserRole.ENGINEER_II]
```

### Project Model

```python
# app/models/project.py
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class Project(Base):
    """
    Project: Collaborative workspace for organizing experiments

    A project contains multiple experiments and can be shared with team members.
    """
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic info
    name = Column(String(255), nullable=False)
    description = Column(String(2000), nullable=True)

    # Ownership
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True, index=True)

    # Settings
    tags = Column(JSON, nullable=True)  # ["computer-vision", "object-detection"]
    color = Column(String(20), nullable=True)  # For UI representation
    icon = Column(String(50), nullable=True)  # Icon name or emoji

    # Status
    is_active = Column(Boolean, default=True)
    is_archived = Column(Boolean, default=False)

    # Statistics (cached for performance)
    num_experiments = Column(Integer, default=0)
    num_members = Column(Integer, default=1)  # Owner counts as member

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity_at = Column(DateTime, nullable=True)

    # Relationships
    owner = relationship("User", back_populates="owned_projects", foreign_keys=[owner_id])
    organization = relationship("Organization", back_populates="projects")
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="project", cascade="all, delete-orphan")
```

### ProjectMember Model

```python
# app/models/project_member.py
from sqlalchemy import Column, String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class ProjectRole(str, enum.Enum):
    """Project-level roles"""
    OWNER = "owner"    # Can delete project, manage members, modify settings
    MEMBER = "member"  # Can view experiments, create runs

class ProjectMember(Base):
    """Project membership with role-based permissions"""
    __tablename__ = "project_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # References
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Role
    role = Column(String(20), nullable=False, default=ProjectRole.MEMBER)

    # Invitation tracking
    invited_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # Timestamps
    joined_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_memberships")
    invited_by = relationship("User", foreign_keys=[invited_by_id])

    # Unique constraint: one user per project
    __table_args__ = (
        Index('idx_project_user', 'project_id', 'user_id', unique=True),
    )
```

### Experiment Model

```python
# app/models/experiment.py
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class Experiment(Base):
    """
    Experiment: MLflow-tracked experiment group

    Maps to MLflow Experiment. Contains multiple training/test/inference runs.
    """
    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Project relationship
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)

    # MLflow integration
    mlflow_experiment_id = Column(String(255), unique=True, nullable=False, index=True)
    mlflow_experiment_name = Column(String(500), nullable=False)

    # Basic info
    name = Column(String(255), nullable=False)
    description = Column(String(2000), nullable=True)

    # Metadata
    tags = Column(JSON, nullable=True)  # {"model": "yolo11n", "task": "detection"}

    # Statistics (cached)
    num_runs = Column(Integer, default=0)
    num_completed_runs = Column(Integer, default=0)
    num_failed_runs = Column(Integer, default=0)

    # Best metrics (cached from runs)
    best_metrics = Column(JSON, nullable=True)
    # {
    #   "accuracy": 0.95,
    #   "mAP50": 0.87,
    #   "best_run_id": "uuid"
    # }

    # Status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_run_at = Column(DateTime, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="experiments")
    training_jobs = relationship("TrainingJob", back_populates="experiment")
    test_runs = relationship("TestRun", back_populates="experiment")
    inference_jobs = relationship("InferenceJob", back_populates="experiment")

    # User interactions
    stars = relationship("ExperimentStar", back_populates="experiment", cascade="all, delete-orphan")
    notes = relationship("ExperimentNote", back_populates="experiment", cascade="all, delete-orphan")
```

### ExperimentStar Model

```python
# app/models/experiment_star.py
from sqlalchemy import Column, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class ExperimentStar(Base):
    """Starred/favorited experiments"""
    __tablename__ = "experiment_stars"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # References
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Timestamp
    starred_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="stars")
    user = relationship("User", back_populates="starred_experiments")

    # Unique constraint
    __table_args__ = (
        Index('idx_experiment_user_star', 'experiment_id', 'user_id', unique=True),
    )
```

### ExperimentNote Model

```python
# app/models/experiment_note.py
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class ExperimentNote(Base):
    """Notes/comments on experiments"""
    __tablename__ = "experiment_notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # References
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Content
    title = Column(String(255), nullable=True)
    content = Column(String(10000), nullable=False)  # Markdown supported

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="notes")
    user = relationship("User", back_populates="experiment_notes")
```

### Updated Dataset Model

```python
# app/models/dataset.py
# ... existing fields ...

class DatasetVisibility(str, enum.Enum):
    PUBLIC = "public"    # Accessible to all users
    PRIVATE = "private"  # Only accessible to owner and members

class Dataset(Base):
    __tablename__ = "datasets"

    # ... existing fields ...

    # Ownership
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Visibility
    visibility = Column(SQLEnum(DatasetVisibility), default=DatasetVisibility.PRIVATE, nullable=False, index=True)

    # ... rest of fields ...

    # Relationships
    owner = relationship("User", back_populates="owned_datasets", foreign_keys=[owner_id])
    members = relationship("DatasetMember", back_populates="dataset", cascade="all, delete-orphan")
```

### DatasetMember Model

```python
# app/models/dataset_member.py
from sqlalchemy import Column, String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.db.base import Base

class DatasetRole(str, enum.Enum):
    """Dataset-level roles"""
    OWNER = "owner"    # Can delete dataset, manage members, modify metadata
    MEMBER = "member"  # Can use dataset in training, view metadata

class DatasetMember(Base):
    """Dataset membership with role-based permissions"""
    __tablename__ = "dataset_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # References
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Role
    role = Column(String(20), nullable=False, default=DatasetRole.MEMBER)

    # Invitation tracking
    invited_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # Timestamps
    joined_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="members")
    user = relationship("User", back_populates="dataset_memberships")
    invited_by = relationship("User", foreign_keys=[invited_by_id])

    # Unique constraint
    __table_args__ = (
        Index('idx_dataset_user', 'dataset_id', 'user_id', unique=True),
    )
```

### Updated TrainingJob Model

```python
# app/models/training_job.py
# Add experiment relationship

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    # ... existing fields ...

    # NEW: Experiment relationship
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=True, index=True)

    # MLflow Run integration
    mlflow_run_id = Column(String(255), unique=True, nullable=True, index=True)

    # ... rest of fields ...

    # Relationships
    experiment = relationship("Experiment", back_populates="training_jobs")
```

---

## Authentication & Invitation System

### Invitation Model

```python
# app/models/invitation.py
from sqlalchemy import Column, String, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import uuid
import enum
import secrets

from app.db.base import Base

class InvitationType(str, enum.Enum):
    ORGANIZATION = "organization"
    PROJECT = "project"
    DATASET = "dataset"

class InvitationStatus(str, enum.Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"

class Invitation(Base):
    """Invitations for user signup and resource access"""
    __tablename__ = "invitations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Token for invitation link
    token = Column(String(255), unique=True, nullable=False, index=True)

    # Type and target
    invitation_type = Column(SQLEnum(InvitationType), nullable=False)

    # For organization invitations
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)

    # For project invitations
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)
    project_role = Column(String(20), nullable=True)  # ProjectRole

    # For dataset invitations
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=True)
    dataset_role = Column(String(20), nullable=True)  # DatasetRole

    # Inviter and invitee
    inviter_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    invitee_email = Column(String(255), nullable=False, index=True)
    invitee_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)  # Set when accepted
    invitee_role = Column(SQLEnum(UserRole), nullable=True)  # Target user role

    # Status
    status = Column(SQLEnum(InvitationStatus), default=InvitationStatus.PENDING, nullable=False, index=True)

    # Message
    message = Column(String(1000), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)  # Default: 7 days
    accepted_at = Column(DateTime, nullable=True)

    # Relationships
    inviter = relationship("User", back_populates="sent_invitations", foreign_keys=[inviter_id])
    invitee = relationship("User", back_populates="received_invitation", foreign_keys=[invitee_id])
    organization = relationship("Organization")
    project = relationship("Project")
    dataset = relationship("Dataset")

    @classmethod
    def create_invitation(
        cls,
        invitation_type: InvitationType,
        inviter_id: UUID,
        invitee_email: str,
        expires_in_days: int = 7,
        **kwargs
    ):
        """Create a new invitation with unique token"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        return cls(
            token=token,
            invitation_type=invitation_type,
            inviter_id=inviter_id,
            invitee_email=invitee_email,
            expires_at=expires_at,
            **kwargs
        )

    def is_expired(self) -> bool:
        """Check if invitation has expired"""
        return datetime.utcnow() > self.expires_at
```

### Authentication Flow

#### 1. User Signup (No Invitation)

```python
# POST /api/v1/auth/signup
async def signup(
    email: str,
    password: str,
    full_name: str,
    company: str,
    division: str = None,
    department: str = None
):
    """
    Standard signup flow

    - Creates User with GUEST role by default
    - Sends email verification
    - Auto-generates avatar_name and badge_color
    """

    # 1. Validate email not exists
    # 2. Hash password
    # 3. Find or create Organization (company + division)
    # 4. Generate avatar
    avatar_name = generate_avatar_name(full_name)  # "John D"
    badge_color = generate_badge_color(email)  # Deterministic color from email hash

    # 5. Create user
    user = User(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
        company=company,
        division=division,
        department=department,
        organization_id=organization.id,
        role=UserRole.GUEST,
        avatar_name=avatar_name,
        badge_color=badge_color,
        is_email_verified=False
    )

    # 6. Send verification email
    # 7. Return JWT token
```

#### 2. Invitation-based Signup

```python
# GET /api/v1/invitations/{token}
async def get_invitation_info(token: str):
    """Get invitation details for display"""
    invitation = await get_invitation_by_token(token)

    if not invitation or invitation.is_expired():
        raise HTTPException(status_code=404, detail="Invalid or expired invitation")

    return {
        "type": invitation.invitation_type,
        "inviter_name": invitation.inviter.full_name,
        "inviter_company": invitation.inviter.company,
        "project_name": invitation.project.name if invitation.project else None,
        "dataset_name": invitation.dataset.name if invitation.dataset else None,
        "role": invitation.invitee_role,
        "message": invitation.message,
        "expires_at": invitation.expires_at
    }


# POST /api/v1/auth/signup-with-invitation
async def signup_with_invitation(
    token: str,
    email: str,
    password: str,
    full_name: str,
    department: str = None
):
    """
    Signup via invitation link

    - Validates invitation token
    - Creates user with specified role
    - Auto-adds to project/dataset/organization
    """

    # 1. Validate invitation
    invitation = await get_invitation_by_token(token)

    if not invitation or invitation.is_expired():
        raise HTTPException(status_code=404, detail="Invalid or expired invitation")

    if invitation.invitee_email != email:
        raise HTTPException(status_code=400, detail="Email mismatch")

    # 2. Create user with specified role
    user = User(
        email=email,
        hashed_password=hash_password(password),
        full_name=full_name,
        company=invitation.inviter.company,
        division=invitation.inviter.division,
        department=department,
        organization_id=invitation.organization_id,
        role=invitation.invitee_role or UserRole.ENGINEER_I,  # Default to ENGINEER_I
        avatar_name=generate_avatar_name(full_name),
        badge_color=generate_badge_color(email),
        is_email_verified=True  # Auto-verified via invitation
    )

    # 3. Add to resources based on invitation type
    if invitation.invitation_type == InvitationType.PROJECT:
        project_member = ProjectMember(
            project_id=invitation.project_id,
            user_id=user.id,
            role=invitation.project_role,
            invited_by_id=invitation.inviter_id
        )
        await db.add(project_member)

    elif invitation.invitation_type == InvitationType.DATASET:
        dataset_member = DatasetMember(
            dataset_id=invitation.dataset_id,
            user_id=user.id,
            role=invitation.dataset_role,
            invited_by_id=invitation.inviter_id
        )
        await db.add(dataset_member)

    # 4. Update invitation status
    invitation.status = InvitationStatus.ACCEPTED
    invitation.invitee_id = user.id
    invitation.accepted_at = datetime.utcnow()

    # 5. Return JWT token
```

#### 3. Invite Existing User

```python
# POST /api/v1/projects/{project_id}/invite
async def invite_to_project(
    project_id: UUID,
    email: str,
    role: ProjectRole,
    message: str = None,
    current_user: User = Depends(get_current_user)
):
    """
    Invite existing or new user to project

    - If user exists: Send notification, auto-add to project
    - If user doesn't exist: Send invitation email with signup link
    """

    # 1. Check permissions
    project = await get_project_or_404(project_id)
    if not await has_project_permission(current_user, project, "invite"):
        raise HTTPException(status_code=403, detail="No permission to invite")

    # 2. Check if user exists
    existing_user = await get_user_by_email(email)

    if existing_user:
        # User exists - add directly to project
        member = ProjectMember(
            project_id=project_id,
            user_id=existing_user.id,
            role=role,
            invited_by_id=current_user.id
        )
        await db.add(member)

        # Send notification
        await send_project_invitation_notification(
            user=existing_user,
            project=project,
            inviter=current_user,
            message=message
        )

        return {"status": "added", "user_id": existing_user.id}

    else:
        # User doesn't exist - create invitation
        invitation = Invitation.create_invitation(
            invitation_type=InvitationType.PROJECT,
            inviter_id=current_user.id,
            invitee_email=email,
            project_id=project_id,
            project_role=role,
            invitee_role=UserRole.ENGINEER_I,  # Default role for new users
            message=message
        )
        await db.add(invitation)

        # Send invitation email
        await send_invitation_email(
            email=email,
            invitation_token=invitation.token,
            inviter=current_user,
            project=project
        )

        return {"status": "invited", "invitation_id": invitation.id}
```

#### 4. JWT Authentication

```python
# JWT Token Structure
{
    "sub": "user-uuid",
    "email": "user@example.com",
    "role": "engineer_ii",
    "organization_id": "org-uuid",
    "exp": 1234567890,
    "iat": 1234567890
}

# Token Generation
def create_access_token(user: User) -> str:
    """Create JWT access token"""
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "organization_id": str(user.organization_id) if user.organization_id else None,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

# Token Validation
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")

        user = await db.get(User, user_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid user")

        return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

## Permission System

### Option C: Independent Management with Validation

**Principle**: Project and Dataset are independent resources. Permission is validated at action time.

### Permission Checking

```python
# app/services/permission_service.py
from typing import Optional
from app.models.user import User, UserRole
from app.models.project import Project
from app.models.dataset import Dataset, DatasetVisibility

class PermissionService:
    """Centralized permission checking"""

    # ==================== Project Permissions ====================

    @staticmethod
    async def can_view_project(user: User, project: Project) -> bool:
        """Check if user can view project"""
        # Admin can view all
        if user.role == UserRole.ADMIN:
            return True

        # Owner can view
        if project.owner_id == user.id:
            return True

        # Member can view
        membership = await get_project_membership(user.id, project.id)
        return membership is not None

    @staticmethod
    async def can_edit_project(user: User, project: Project) -> bool:
        """Check if user can edit project settings"""
        # Admin can edit all
        if user.role == UserRole.ADMIN:
            return True

        # Only owner can edit
        if project.owner_id == user.id:
            return True

        # Check if user is project owner (multiple owners allowed)
        membership = await get_project_membership(user.id, project.id)
        return membership and membership.role == ProjectRole.OWNER

    @staticmethod
    async def can_delete_project(user: User, project: Project) -> bool:
        """Check if user can delete project"""
        # Admin can delete all
        if user.role == UserRole.ADMIN:
            return True

        # Only owner can delete
        return project.owner_id == user.id

    @staticmethod
    async def can_invite_to_project(user: User, project: Project) -> bool:
        """Check if user can invite others to project"""
        # Guest cannot invite
        if user.role == UserRole.GUEST:
            return False

        # Admin can invite to all
        if user.role == UserRole.ADMIN:
            return True

        # Owner can invite
        membership = await get_project_membership(user.id, project.id)
        return membership and membership.role == ProjectRole.OWNER

    # ==================== Dataset Permissions ====================

    @staticmethod
    async def can_view_dataset(user: User, dataset: Dataset) -> bool:
        """Check if user can view dataset"""
        # Public datasets are viewable by all
        if dataset.visibility == DatasetVisibility.PUBLIC:
            return True

        # Admin can view all
        if user.role == UserRole.ADMIN:
            return True

        # Owner can view
        if dataset.owner_id == user.id:
            return True

        # Member can view
        membership = await get_dataset_membership(user.id, dataset.id)
        return membership is not None

    @staticmethod
    async def can_use_dataset(user: User, dataset: Dataset) -> bool:
        """Check if user can use dataset in training"""
        # Same as view permission
        return await PermissionService.can_view_dataset(user, dataset)

    @staticmethod
    async def can_edit_dataset(user: User, dataset: Dataset) -> bool:
        """Check if user can edit dataset metadata"""
        # Admin can edit all
        if user.role == UserRole.ADMIN:
            return True

        # Only owner can edit
        membership = await get_dataset_membership(user.id, dataset.id)
        return membership and membership.role == DatasetRole.OWNER

    @staticmethod
    async def can_delete_dataset(user: User, dataset: Dataset) -> bool:
        """Check if user can delete dataset"""
        # Admin can delete all
        if user.role == UserRole.ADMIN:
            return True

        # Only owner can delete
        return dataset.owner_id == user.id

    # ==================== Training Job Validation ====================

    @staticmethod
    async def validate_training_job_creation(
        user: User,
        project_id: UUID,
        dataset_id: UUID
    ) -> tuple[bool, Optional[str]]:
        """
        Validate if user can create training job

        Returns: (allowed, error_message)
        """

        # 1. Check project permission
        project = await get_project_or_404(project_id)
        if not await PermissionService.can_view_project(user, project):
            return False, "No access to project"

        # 2. Check dataset permission
        dataset = await get_dataset_or_404(dataset_id)
        if not await PermissionService.can_use_dataset(user, dataset):
            return False, "No access to dataset"

        # 3. Check role-based features
        # ... (quota checking, advanced features, etc.)

        return True, None
```

### Usage Example

```python
# POST /api/v1/experiments/{experiment_id}/training
async def create_training_job(
    experiment_id: UUID,
    dataset_id: UUID,
    config: TrainingConfig,
    current_user: User = Depends(get_current_user)
):
    """Create training job in experiment"""

    # 1. Get experiment and project
    experiment = await get_experiment_or_404(experiment_id)
    project = experiment.project

    # 2. Validate permissions (Option C: independent validation)
    allowed, error = await PermissionService.validate_training_job_creation(
        user=current_user,
        project_id=project.id,
        dataset_id=dataset_id
    )

    if not allowed:
        raise HTTPException(status_code=403, detail=error)

    # 3. Create training job
    training_job = TrainingJob(
        user_id=current_user.id,
        experiment_id=experiment.id,
        dataset_id=dataset_id,
        # ... rest of config
    )

    # Permission validated at this point:
    # - User has access to project (view permission)
    # - User has access to dataset (use permission)
    # - If dataset deleted later, training job still references it (historical record)
```

---

## Experiment Comparison Feature

### ExperimentComparison Model

```python
# app/models/experiment_comparison.py
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base

class ExperimentComparison(Base):
    """Saved experiment comparisons"""
    __tablename__ = "experiment_comparisons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Owner
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)

    # Comparison info
    name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)

    # Experiments being compared
    experiment_ids = Column(JSON, nullable=False)  # List of UUIDs

    # Comparison configuration
    metrics_to_compare = Column(JSON, nullable=True)  # ["accuracy", "mAP50", "loss"]

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User")
    project = relationship("Project")
```

### Comparison API

```python
# GET /api/v1/experiments/compare
async def compare_experiments(
    experiment_ids: List[UUID],
    metrics: List[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Compare multiple experiments

    Returns aggregated metrics, charts, and insights
    """

    # 1. Validate all experiments accessible
    experiments = []
    for exp_id in experiment_ids:
        exp = await get_experiment_or_404(exp_id)
        project = exp.project

        if not await PermissionService.can_view_project(current_user, project):
            raise HTTPException(status_code=403, detail=f"No access to experiment {exp_id}")

        experiments.append(exp)

    # 2. Fetch metrics from MLflow
    comparison_data = []

    for exp in experiments:
        mlflow_runs = mlflow.search_runs(
            experiment_ids=[exp.mlflow_experiment_id],
            order_by=["metrics.accuracy DESC"],
            max_results=1
        )

        if not mlflow_runs.empty:
            best_run = mlflow_runs.iloc[0]

            comparison_data.append({
                "experiment_id": exp.id,
                "experiment_name": exp.name,
                "best_run_id": best_run["run_id"],
                "metrics": {
                    metric: best_run[f"metrics.{metric}"]
                    for metric in (metrics or ["accuracy", "loss"])
                    if f"metrics.{metric}" in best_run
                },
                "params": best_run["params"],
                "num_runs": exp.num_runs
            })

    # 3. Generate comparison insights
    insights = generate_comparison_insights(comparison_data)

    return {
        "experiments": comparison_data,
        "insights": insights,
        "compared_at": datetime.utcnow().isoformat()
    }


# POST /api/v1/experiments/comparisons
async def save_comparison(
    comparison: ExperimentComparisonCreate,
    current_user: User = Depends(get_current_user)
):
    """Save experiment comparison for later reference"""

    # Validate all experiments in same project
    experiments = [await get_experiment_or_404(id) for id in comparison.experiment_ids]
    project_ids = {exp.project_id for exp in experiments}

    if len(project_ids) > 1:
        raise HTTPException(status_code=400, detail="All experiments must be in same project")

    saved_comparison = ExperimentComparison(
        user_id=current_user.id,
        project_id=list(project_ids)[0],
        name=comparison.name,
        description=comparison.description,
        experiment_ids=comparison.experiment_ids,
        metrics_to_compare=comparison.metrics
    )

    await db.add(saved_comparison)
    await db.commit()

    return saved_comparison
```

---

## MLflow Integration

### Experiment Creation

```python
# app/services/mlflow_service.py
import mlflow
from mlflow.tracking import MlflowClient

class MLflowService:
    """MLflow integration service"""

    def __init__(self):
        self.client = MlflowClient()

    async def create_experiment(
        self,
        project: Project,
        experiment_name: str,
        tags: dict = None
    ) -> tuple[str, str]:
        """
        Create MLflow experiment

        Returns: (mlflow_experiment_id, mlflow_experiment_name)
        """

        # Generate unique MLflow experiment name
        mlflow_name = f"{project.name}/{experiment_name}"

        # Create MLflow experiment
        mlflow_experiment_id = mlflow.create_experiment(
            name=mlflow_name,
            tags={
                "project_id": str(project.id),
                "project_name": project.name,
                "organization_id": str(project.organization_id) if project.organization_id else None,
                **(tags or {})
            }
        )

        return mlflow_experiment_id, mlflow_name

    async def start_run(
        self,
        experiment: Experiment,
        training_job: TrainingJob,
        user: User
    ) -> str:
        """
        Start MLflow run for training job

        Returns: mlflow_run_id
        """

        with mlflow.start_run(
            experiment_id=experiment.mlflow_experiment_id,
            run_name=f"training-{training_job.id}",
            tags={
                "training_job_id": str(training_job.id),
                "user_id": str(user.id),
                "user_name": user.full_name,
                "model_name": training_job.model_name,
                "framework": training_job.framework,
                "task_type": training_job.task_type
            }
        ) as run:
            # Log parameters
            mlflow.log_params(training_job.config)

            return run.info.run_id
```

---

## API Endpoints Summary

### Projects

- `GET /api/v1/projects` - List user's projects
- `POST /api/v1/projects` - Create project
- `GET /api/v1/projects/{id}` - Get project details
- `PATCH /api/v1/projects/{id}` - Update project
- `DELETE /api/v1/projects/{id}` - Delete project
- `POST /api/v1/projects/{id}/invite` - Invite user to project
- `GET /api/v1/projects/{id}/members` - List project members
- `DELETE /api/v1/projects/{id}/members/{user_id}` - Remove member

### Experiments

- `GET /api/v1/projects/{project_id}/experiments` - List experiments
- `POST /api/v1/projects/{project_id}/experiments` - Create experiment
- `GET /api/v1/experiments/{id}` - Get experiment details
- `PATCH /api/v1/experiments/{id}` - Update experiment
- `DELETE /api/v1/experiments/{id}` - Delete experiment
- `POST /api/v1/experiments/{id}/star` - Star experiment
- `DELETE /api/v1/experiments/{id}/star` - Unstar experiment
- `GET /api/v1/experiments/{id}/notes` - List notes
- `POST /api/v1/experiments/{id}/notes` - Add note
- `PATCH /api/v1/experiments/notes/{note_id}` - Update note
- `DELETE /api/v1/experiments/notes/{note_id}` - Delete note
- `GET /api/v1/experiments/compare` - Compare experiments
- `POST /api/v1/experiments/comparisons` - Save comparison

### Datasets

- `GET /api/v1/datasets` - List accessible datasets
- `POST /api/v1/datasets` - Create dataset
- `GET /api/v1/datasets/{id}` - Get dataset details
- `PATCH /api/v1/datasets/{id}` - Update dataset
- `DELETE /api/v1/datasets/{id}` - Delete dataset
- `POST /api/v1/datasets/{id}/invite` - Invite user to dataset
- `GET /api/v1/datasets/{id}/members` - List dataset members
- `DELETE /api/v1/datasets/{id}/members/{user_id}` - Remove member
- `PATCH /api/v1/datasets/{id}/visibility` - Change visibility

### Authentication

- `POST /api/v1/auth/signup` - Standard signup
- `POST /api/v1/auth/signup-with-invitation` - Signup via invitation
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/refresh` - Refresh token
- `POST /api/v1/auth/verify-email` - Verify email
- `POST /api/v1/auth/forgot-password` - Request password reset
- `POST /api/v1/auth/reset-password` - Reset password

### Invitations

- `GET /api/v1/invitations/{token}` - Get invitation details
- `GET /api/v1/invitations` - List sent invitations
- `DELETE /api/v1/invitations/{id}` - Cancel invitation

---

## Frontend Integration

### Consistent Avatar Display

```typescript
// utils/avatar.ts
interface AvatarProps {
  name: string;
  color: string;
}

export function generateAvatarProps(user: User): AvatarProps {
  return {
    name: user.avatar_name || generateAvatarName(user.full_name),
    color: user.badge_color || generateBadgeColor(user.email)
  };
}

function generateAvatarName(fullName: string): string {
  const parts = fullName.trim().split(' ');
  if (parts.length === 1) {
    return parts[0].substring(0, 2).toUpperCase();
  }
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function generateBadgeColor(email: string): string {
  // Deterministic color generation from email hash
  const colors = [
    '#4F46E5', // indigo
    '#7C3AED', // purple
    '#DC2626', // red
    '#EA580C', // orange
    '#CA8A04', // yellow
    '#16A34A', // green
    '#0891B2', // cyan
    '#2563EB'  // blue
  ];

  const hash = email.split('').reduce((acc, char) => {
    return char.charCodeAt(0) + ((acc << 5) - acc);
  }, 0);

  return colors[Math.abs(hash) % colors.length];
}

// Component usage
function UserAvatar({ user }: { user: User }) {
  const { name, color } = generateAvatarProps(user);

  return (
    <div
      className="flex items-center justify-center w-10 h-10 rounded-full text-white font-semibold"
      style={{ backgroundColor: color }}
    >
      {name}
    </div>
  );
}
```

---

## Best Practices

1. **Always validate permissions** at action time, not at query time
2. **Use JWT tokens** for authentication, refresh tokens for session management
3. **Log all permission changes** for audit trail
4. **Invitation tokens expire** after 7 days by default
5. **Email verification required** for standard signup
6. **Project/Dataset membership is independent** - validate both at training time
7. **MLflow experiment names** must be unique - use project/experiment hierarchy
8. **Avatar colors are deterministic** - same user gets same color across sessions
9. **Guest users are restricted** - enforce quotas at API level
10. **Admin can do everything** - but log admin actions for compliance

---

## References

- [BACKEND_DESIGN.md](./BACKEND_DESIGN.md) - Core database models
- [USER_ANALYTICS_DESIGN.md](./USER_ANALYTICS_DESIGN.md) - Usage tracking and analytics
- [ISOLATION_DESIGN.md](./ISOLATION_DESIGN.md) - Security isolation principles
