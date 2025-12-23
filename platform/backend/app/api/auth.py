"""Authentication API endpoints.

Note: User authentication is handled by Keycloak SSO.
This module only contains endpoints for:
- Getting current user info
- Updating user profile
- Service-to-service tokens (Platform → Labeler SSO)
"""

import random
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import create_service_token
from app.db.database import get_user_db
from app.db import models
from app.schemas import user as user_schemas
from app.utils.dependencies import get_current_user

router = APIRouter()

# Avatar badge colors (excluding light colors like yellow, amber, lime)
BADGE_COLORS = [
    'red', 'orange', 'green', 'emerald',
    'teal', 'cyan', 'sky', 'blue',
    'indigo', 'violet', 'purple', 'fuchsia', 'pink'
]

# Avatar name adjectives and nouns for random generation
AVATAR_ADJECTIVES = [
    'happy', 'brave', 'clever', 'wise', 'swift',
    'bright', 'cool', 'kind', 'bold', 'calm',
    'eager', 'fair', 'gentle', 'keen', 'noble'
]

AVATAR_NOUNS = [
    'panda', 'tiger', 'eagle', 'dolphin', 'fox',
    'wolf', 'lion', 'bear', 'hawk', 'owl',
    'falcon', 'lynx', 'otter', 'raven', 'phoenix'
]


def generate_avatar_name() -> str:
    """Generate a random avatar name in the format 'adjective-noun-number'."""
    adjective = random.choice(AVATAR_ADJECTIVES)
    noun = random.choice(AVATAR_NOUNS)
    number = random.randint(100, 999)
    return f"{adjective}-{noun}-{number}"


@router.get("/me", response_model=user_schemas.UserResponse)
def get_current_user_info(
    current_user: models.User = Depends(get_current_user)
):
    """
    Get current user information.

    Args:
        current_user: Current authenticated user (via Keycloak token)

    Returns:
        Current user object
    """
    return current_user


@router.put("/me", response_model=user_schemas.UserResponse)
def update_current_user(
    user_update: user_schemas.UserUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_user_db)
):
    """
    Update current user profile information.

    Note: Password changes are handled by Keycloak, not this endpoint.

    Args:
        user_update: User update data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Updated user object
    """
    # Update fields if provided
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name
    if user_update.company is not None:
        current_user.company = user_update.company
    if user_update.company_custom is not None:
        current_user.company_custom = user_update.company_custom
    if user_update.division is not None:
        current_user.division = user_update.division
    if user_update.division_custom is not None:
        current_user.division_custom = user_update.division_custom
    if user_update.department is not None:
        current_user.department = user_update.department
    if user_update.phone_number is not None:
        current_user.phone_number = user_update.phone_number
    if user_update.bio is not None:
        current_user.bio = user_update.bio

    current_user.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(current_user)

    return current_user


@router.post("/labeler-token")
def get_labeler_token(
    current_user: models.User = Depends(get_current_user)
):
    """
    Generate a service JWT token for SSO to Labeler service.

    This endpoint is used when the frontend redirects users from Platform
    to Labeler for dataset management. The token is short-lived (5 minutes)
    and contains user identity information for automatic login.

    Args:
        current_user: Authenticated user from Platform

    Returns:
        dict: Contains service_token for SSO to Labeler

    Example:
        Frontend flow:
        1. User clicks "데이터셋" in Platform sidebar
        2. Frontend calls /api/v1/auth/labeler-token
        3. Frontend redirects to: {LABELER_URL}/sso?token={service_token}
        4. Labeler validates token and creates user session
    """
    service_token = create_service_token({
        "user_id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "system_role": current_user.system_role,
        "badge_color": current_user.badge_color
    })

    return {
        "service_token": service_token,
        "expires_in": 300  # 5 minutes in seconds
    }
