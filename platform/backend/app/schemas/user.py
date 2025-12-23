"""User schemas for API requests and responses.

Note: User authentication is handled by Keycloak SSO.
Password-related schemas (UserCreate, Token, ForgotPassword, etc.) have been removed.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr
from app.schemas.enums import SystemRole


class UserBase(BaseModel):
    """Base user schema."""

    email: EmailStr
    full_name: Optional[str] = None

    # Organization hierarchy: 회사 → 사업부 → 부서
    company: Optional[str] = None           # 회사 (삼성전자, 협력사, 직접입력)
    company_custom: Optional[str] = None    # 직접입력 시 회사명
    division: Optional[str] = None          # 사업부 (생산기술연구소, MX, VD, DA, SR, 직접입력)
    division_custom: Optional[str] = None   # 직접입력 시 사업부명
    department: Optional[str] = None        # 부서 (자유 입력)

    phone_number: Optional[str] = None      # 전화번호
    bio: Optional[str] = None               # 소개


class UserUpdate(BaseModel):
    """Schema for updating user profile information.

    Note: Password changes are handled by Keycloak, not this schema.
    """

    full_name: Optional[str] = None
    company: Optional[str] = None
    company_custom: Optional[str] = None
    division: Optional[str] = None
    division_custom: Optional[str] = None
    department: Optional[str] = None
    phone_number: Optional[str] = None
    bio: Optional[str] = None


class UserResponse(UserBase):
    """Schema for user response."""

    id: int
    system_role: SystemRole  # 시스템 권한
    is_active: bool
    badge_color: Optional[str] = None  # Avatar badge color
    avatar_name: Optional[str] = None  # Random avatar name
    organization_id: Optional[int] = None  # Organization membership
    created_at: datetime

    class Config:
        from_attributes = True
