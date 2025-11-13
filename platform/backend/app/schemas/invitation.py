"""Schemas for Invitation API."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

from app.db.models import InvitationType, InvitationStatus, UserRole


# ==================== Base Schemas ====================

class InvitationBase(BaseModel):
    """Base invitation schema."""
    invitee_email: EmailStr = Field(..., description="Email of person being invited")
    invitee_role: UserRole = Field(UserRole.GUEST, description="Role to assign upon acceptance")
    message: Optional[str] = Field(None, description="Optional personal message")


class InvitationCreate(InvitationBase):
    """Schema for creating an invitation."""
    invitation_type: InvitationType = Field(..., description="Type of invitation")

    # One of these should be provided based on invitation_type
    organization_id: Optional[int] = Field(None, description="Organization ID (for ORGANIZATION invites)")
    project_id: Optional[int] = Field(None, description="Project ID (for PROJECT invites)")
    dataset_id: Optional[str] = Field(None, description="Dataset ID (for DATASET invites)")


class InvitationResponse(InvitationBase):
    """Schema for invitation response."""
    id: int
    token: str
    invitation_type: InvitationType
    status: InvitationStatus

    organization_id: Optional[int] = None
    project_id: Optional[int] = None
    dataset_id: Optional[str] = None

    inviter_id: int
    invitee_id: Optional[int] = None

    expires_at: datetime
    created_at: datetime
    accepted_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ==================== Public Invitation Info ====================

class InvitationInfoResponse(BaseModel):
    """Public invitation information (for unauthenticated users)."""
    token: str
    invitation_type: InvitationType
    inviter_name: str
    invitee_email: EmailStr
    invitee_role: UserRole

    # Entity information
    entity_name: str  # Name of organization, project, or dataset
    organization_name: Optional[str] = None  # For project/dataset invites

    message: Optional[str] = None
    expires_at: datetime
    is_expired: bool


# ==================== Invitation Acceptance ====================

class AcceptInvitationRequest(BaseModel):
    """Schema for accepting an invitation."""
    token: str = Field(..., description="Invitation token")

    # User registration data (if not already registered)
    full_name: Optional[str] = Field(None, description="User's full name")
    password: Optional[str] = Field(None, min_length=8, description="Password for new account")

    # Optional profile data
    phone_number: Optional[str] = None
    department: Optional[str] = None


class DeclineInvitationRequest(BaseModel):
    """Schema for declining an invitation."""
    token: str = Field(..., description="Invitation token")


# ==================== Invitation List ====================

class InvitationListResponse(BaseModel):
    """Schema for paginated invitation list."""
    invitations: list[InvitationResponse]
    total: int
    skip: int
    limit: int
