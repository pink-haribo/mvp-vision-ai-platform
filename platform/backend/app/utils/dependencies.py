"""Authentication dependencies for FastAPI endpoints.

Keycloak OIDC 인증을 사용합니다.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.keycloak_auth import (
    verify_keycloak_token,
    map_keycloak_to_system_role,
    KeycloakTokenPayload,
)
from app.db.database import get_db, get_user_db
from app.db import models

# HTTP Bearer 스킴 (Keycloak 토큰용)
http_bearer = HTTPBearer(
    scheme_name="Keycloak Bearer",
    description="Keycloak에서 발급받은 Access Token",
    auto_error=True
)


async def get_current_user(
    bearer: HTTPAuthorizationCredentials = Depends(http_bearer),
    user_db: Session = Depends(get_user_db)
) -> models.User:
    """
    Keycloak 토큰에서 현재 인증된 사용자 반환

    1. Keycloak 토큰 검증 (RS256, JWKS)
    2. keycloak_id로 사용자 조회
    3. 없으면 이메일로 조회 후 keycloak_id 연결
    4. 그래도 없으면 JIT Provisioning (자동 사용자 생성)

    Args:
        bearer: Authorization 헤더에서 추출한 Bearer 토큰
        user_db: Shared User database session

    Returns:
        Current user object

    Raises:
        HTTPException: 토큰 검증 실패 또는 사용자 비활성
    """
    token = bearer.credentials

    # 1. Keycloak 토큰 검증
    try:
        payload: KeycloakTokenPayload = await verify_keycloak_token(token)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Keycloak token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. keycloak_id로 사용자 조회
    user = user_db.query(models.User).filter(
        models.User.keycloak_id == payload.sub
    ).first()

    # 3. 없으면 이메일로 기존 사용자 조회 (마이그레이션 케이스)
    if not user and payload.email:
        user = user_db.query(models.User).filter(
            models.User.email == payload.email
        ).first()

        if user:
            # 기존 사용자에 keycloak_id 연결
            user.keycloak_id = payload.sub
            user_db.commit()

    # 4. 그래도 없으면 새 사용자 생성 (JIT Provisioning)
    if not user:
        user = models.User(
            keycloak_id=payload.sub,
            email=payload.email or f"{payload.sub}@keycloak.local",
            hashed_password=None,  # Keycloak 사용자는 비밀번호 없음
            full_name=payload.name or payload.preferred_username or "Unknown",
            system_role=map_keycloak_to_system_role(payload),
            is_active=True,
            # 커스텀 속성 (Keycloak User Attributes에서 매핑된 경우)
            company=payload.company,
            department=payload.department,
        )
        user_db.add(user)
        user_db.commit()
        user_db.refresh(user)

    # 5. 활성 사용자 확인
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return user


def get_current_active_user(
    current_user: models.User = Depends(get_current_user)
) -> models.User:
    """
    Get the current active user (is_active=True).

    Args:
        current_user: Current user from get_current_user

    Returns:
        Active user object

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def get_current_superuser(
    current_user: models.User = Depends(get_current_user)
) -> models.User:
    """
    Get the current admin user.

    Args:
        current_user: Current user from get_current_user

    Returns:
        Admin user object

    Raises:
        HTTPException: If user is not an admin
    """
    if current_user.system_role != models.UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin role required."
        )
    return current_user


def check_project_permission(project_id: int, user_id: int, db: Session) -> bool:
    """
    Check if user has permission to access a project.

    Args:
        project_id: Project ID to check
        user_id: User ID to check
        db: Database session

    Returns:
        True if user has permission, False otherwise
    """
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        return False

    # Check if user is the owner
    if project.user_id == user_id:
        return True

    # Check if user is a member
    member = db.query(models.ProjectMember).filter(
        models.ProjectMember.project_id == project_id,
        models.ProjectMember.user_id == user_id
    ).first()

    return member is not None
