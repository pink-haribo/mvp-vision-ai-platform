"""Authentication API endpoints."""

import random
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token
)
from app.db.database import get_db
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


@router.post("/register", response_model=user_schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    user_in: user_schemas.UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user.

    Args:
        user_in: User registration data
        db: Database session

    Returns:
        Created user object

    Raises:
        HTTPException: If email already registered
    """
    # Check if user already exists
    existing_user = db.query(models.User).filter(models.User.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user with default system_role='guest' and random badge color
    hashed_password = get_password_hash(user_in.password)
    db_user = models.User(
        email=user_in.email,
        hashed_password=hashed_password,
        full_name=user_in.full_name,
        company=user_in.company,
        company_custom=user_in.company_custom,
        division=user_in.division,
        division_custom=user_in.division_custom,
        department=user_in.department,
        phone_number=user_in.phone_number,
        bio=user_in.bio,
        system_role='guest',  # New users start as guest
        is_active=True,
        badge_color=random.choice(BADGE_COLORS)  # Assign random badge color
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@router.post("/login", response_model=user_schemas.Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login with email and password.

    Args:
        form_data: OAuth2 form data (username=email, password)
        db: Database session

    Returns:
        Access and refresh tokens

    Raises:
        HTTPException: If credentials are invalid
    """
    # Find user by email (OAuth2 uses 'username' field)
    user = db.query(models.User).filter(models.User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    # Create tokens
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=user_schemas.Token)
def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token.

    Args:
        refresh_token: Refresh token
        db: Database session

    Returns:
        New access and refresh tokens

    Raises:
        HTTPException: If refresh token is invalid
    """
    try:
        payload = decode_token(refresh_token)
        user_id = payload.get("sub")
        token_type = payload.get("type")

        if user_id is None or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        # Verify user exists
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )

        # Create new tokens
        new_access_token = create_access_token(data={"sub": user.id})
        new_refresh_token = create_refresh_token(data={"sub": user.id})

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {str(e)}"
        )


@router.get("/me", response_model=user_schemas.UserResponse)
def get_current_user_info(
    current_user: models.User = Depends(get_current_user)
):
    """
    Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        Current user object
    """
    return current_user


@router.put("/me", response_model=user_schemas.UserResponse)
def update_current_user(
    user_update: user_schemas.UserUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update current user information.

    Args:
        user_update: User update data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Updated user object
    """
    from datetime import datetime

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

    # Update password if provided
    if user_update.password is not None:
        current_user.hashed_password = get_password_hash(user_update.password)

    current_user.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(current_user)

    return current_user


@router.post("/logout")
def logout():
    """
    Logout (client-side token deletion).

    Note: Since we're using JWT tokens, logout is handled client-side
    by deleting the tokens. This endpoint is a placeholder for future
    token blacklisting implementation.

    Returns:
        Success message
    """
    return {"message": "Successfully logged out"}
