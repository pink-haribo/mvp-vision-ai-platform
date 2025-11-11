"""
Authentication API Endpoints

Provides user registration, login, and profile management.
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_active_user
from app.auth.jwt import create_access_token
from app.auth.password import hash_password, verify_password
from app.config import settings
from app.db.models import User
from app.db.session import get_db
from app.schemas.auth import (
    UserRegister,
    UserLogin,
    UserResponse,
    LoginResponse,
)

router = APIRouter()


# ==================== Public Endpoints ====================


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user.

    Args:
        user_data: User registration data
        db: Database session

    Returns:
        Created user information

    Raises:
        HTTPException: If email or username already exists
    """
    # Check if user already exists (email or username)
    result = await db.execute(
        select(User).where(
            or_(
                User.email == user_data.email,
                User.username == user_data.username,
            )
        )
    )
    existing_user = result.scalar_one_or_none()

    if existing_user:
        if existing_user.email == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken",
            )

    # Create new user
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
        company=user_data.company,
        division=user_data.division,
        department=user_data.department,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return new_user


@router.post("/login", response_model=LoginResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """
    Login with username/email and password.

    Args:
        credentials: Login credentials
        db: Database session

    Returns:
        JWT access token and user information

    Raises:
        HTTPException: If credentials are invalid
    """
    # Find user by username or email
    result = await db.execute(
        select(User).where(
            or_(
                User.username == credentials.username,
                User.email == credentials.username,
            )
        )
    )
    user = result.scalar_one_or_none()

    # Verify user exists and password is correct
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id},
        expires_delta=access_token_expires,
    )

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
        user=UserResponse.model_validate(user),
    )


# ==================== Protected Endpoints ====================


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
):
    """
    Get current authenticated user information.

    Args:
        current_user: Current authenticated user (from JWT token)

    Returns:
        User information
    """
    return current_user


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user),
):
    """
    Logout current user.

    Note: Since we're using stateless JWT, logout is handled client-side
    by removing the token. This endpoint exists for consistency and
    future enhancements (e.g., token blacklisting).

    Args:
        current_user: Current authenticated user

    Returns:
        Success message
    """
    return {"message": "Successfully logged out"}
