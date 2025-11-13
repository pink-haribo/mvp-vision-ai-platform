"""Project-related Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class ProjectCreate(BaseModel):
    """Schema for creating a project."""

    name: str = Field(..., min_length=1, max_length=200, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    task_type: Optional[str] = Field(None, description="Primary task type (optional)")


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""

    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    task_type: Optional[str] = Field(None, description="Primary task type")


class ProjectResponse(BaseModel):
    """Schema for project response."""

    id: int
    name: str
    description: Optional[str] = None
    task_type: Optional[str] = None
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectWithExperimentsResponse(BaseModel):
    """Schema for project with experiments count."""

    id: int
    name: str
    description: Optional[str] = None
    task_type: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    experiment_count: int = Field(0, description="Number of experiments in this project")
    user_role: str = Field("owner", description="Current user's role in this project (owner or member)")

    class Config:
        from_attributes = True
