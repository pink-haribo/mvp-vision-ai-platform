"""
Admin API Endpoints

Provides admin-only endpoints for user and project management.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_active_user
from app.db.models import User, Project, TrainingJob
from app.db.session import get_db

router = APIRouter(tags=["admin"])


# ==================== Schemas ====================


class UserRoleUpdate(BaseModel):
    """Schema for updating user role."""
    system_role: str


class AdminUserUpdate(BaseModel):
    """Schema for admin to update user info."""
    email: Optional[str] = None
    full_name: Optional[str] = None
    company: Optional[str] = None
    division: Optional[str] = None
    department: Optional[str] = None


# ==================== Dependencies ====================


async def require_admin(current_user: User = Depends(get_current_active_user)):
    """Dependency to check if user is admin."""
    if current_user.system_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


async def require_admin_or_manager(current_user: User = Depends(get_current_active_user)):
    """Dependency to check if user is admin or manager."""
    if current_user.system_role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Manager privileges required"
        )
    return current_user


# ==================== Endpoints ====================


@router.get("/users")
async def list_all_users(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin_or_manager)
):
    """List all users in the system (admin or manager)."""

    # Query all users with project counts
    query = (
        select(
            User,
            func.count(Project.id).label("project_count")
        )
        .outerjoin(Project, Project.owner_id == User.id)
        .group_by(User.id)
    )

    # If user is manager, exclude other managers and admins
    if current_user.system_role == "manager":
        query = query.where(User.system_role.not_in(["manager", "admin"]))

    result = await db.execute(query)
    users = result.all()

    # Format response
    user_list = []
    for user, project_count in users:
        user_list.append({
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "company": user.company,
            "division": user.division,
            "department": user.department,
            "system_role": user.system_role,
            "is_active": user.is_active,
            "badge_color": user.badge_color,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "project_count": project_count,
        })

    return user_list


@router.patch("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role_update: UserRoleUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin_or_manager)
):
    """Update a user's system role (admin or manager)."""

    # Validate role (5-tier system)
    valid_roles = ["guest", "engineer_i", "engineer_ii", "manager", "admin"]
    if role_update.system_role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}"
        )

    # If user is manager, they cannot set roles to manager or admin
    if current_user.system_role == "manager" and role_update.system_role in ["manager", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Managers cannot assign manager or admin roles"
        )

    # Get user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Prevent self-demotion
    if user.id == current_user.id and role_update.system_role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot demote yourself"
        )

    # Update role
    user.system_role = role_update.system_role
    await db.commit()
    await db.refresh(user)

    return {
        "message": "Role updated successfully",
        "user_id": user.id,
        "new_role": user.system_role
    }


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    user_update: AdminUserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin_or_manager)
):
    """Update a user's information (admin or manager)."""

    # Get user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Update fields if provided
    if user_update.email is not None:
        # Check if email is already taken by another user
        stmt = select(User).where(User.email == user_update.email, User.id != user_id)
        result = await db.execute(stmt)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        user.email = user_update.email

    if user_update.full_name is not None:
        user.full_name = user_update.full_name
    if user_update.company is not None:
        user.company = user_update.company
    if user_update.division is not None:
        user.division = user_update.division
    if user_update.department is not None:
        user.department = user_update.department

    await db.commit()
    await db.refresh(user)

    return {
        "message": "User updated successfully",
        "user_id": user.id,
        "email": user.email,
        "full_name": user.full_name
    }


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Delete a user (admin only). Fails if user has projects."""

    # Get user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Prevent self-deletion
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself"
        )

    # Check if user has projects
    result = await db.execute(
        select(func.count(Project.id)).where(Project.owner_id == user_id)
    )
    project_count = result.scalar()

    if project_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete user with {project_count} project(s). Delete or reassign projects first."
        )

    # Delete user
    await db.delete(user)
    await db.commit()

    return {
        "message": "User deleted successfully",
        "user_id": user_id
    }


@router.get("/projects")
async def list_all_projects(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """List all projects in the system (admin only)."""

    # Query all projects with user info and training job counts
    query = (
        select(
            Project,
            User,
            func.count(TrainingJob.id).label("job_count")
        )
        .outerjoin(User, User.id == Project.owner_id)
        .outerjoin(TrainingJob, TrainingJob.project_id == Project.id)
        .group_by(Project.id, User.id)
    )

    result = await db.execute(query)
    projects = result.all()

    # Format response
    project_list = []
    for project, user, job_count in projects:
        project_list.append({
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "job_count": job_count,
            "owner_id": project.owner_id,
            "owner_name": user.full_name if user else None,
            "owner_email": user.email if user else None,
        })

    return project_list
