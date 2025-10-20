"""Common Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None


class TimestampMixin(BaseModel):
    """Timestamp mixin."""

    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True
