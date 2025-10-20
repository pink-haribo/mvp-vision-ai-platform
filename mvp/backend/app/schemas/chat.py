"""Chat-related Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class MessageCreate(BaseModel):
    """Schema for creating a new message."""

    content: str = Field(..., min_length=1, max_length=10000)


class MessageResponse(BaseModel):
    """Schema for message response."""

    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class SessionCreate(BaseModel):
    """Schema for creating a new session."""

    pass  # No fields needed for creation


class SessionResponse(BaseModel):
    """Schema for session response."""

    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """Schema for chat request."""

    session_id: Optional[int] = None
    message: str = Field(..., min_length=1, max_length=10000)


class ChatResponse(BaseModel):
    """Schema for chat response."""

    session_id: int
    user_message: MessageResponse
    assistant_message: MessageResponse
    parsed_intent: Optional[dict] = None
