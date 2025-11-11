"""
Authentication Schemas

Pydantic models for authentication requests and responses.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ==================== Request Schemas ====================


class UserRegister(BaseModel):
    """User registration request."""

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request."""

    username: str  # Can be username or email
    password: str


# ==================== Response Schemas ====================


class UserResponse(BaseModel):
    """User information response."""

    id: str
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse


class LoginResponse(TokenResponse):
    """Login response (alias for TokenResponse)."""

    pass
