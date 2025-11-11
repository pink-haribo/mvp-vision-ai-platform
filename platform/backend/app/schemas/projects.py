"""
Project Schemas

Pydantic models for project requests and responses.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ==================== Request Schemas ====================


class ProjectCreate(BaseModel):
    """Project creation request."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectUpdate(BaseModel):
    """Project update request."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


# ==================== Response Schemas ====================


class ProjectResponse(BaseModel):
    """Project information response."""

    id: str
    name: str
    description: Optional[str]
    owner_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
