"""Admin API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel

from app.db.database import get_db
from app.db.models import Project, TrainingJob, User
from app.schemas.project import ProjectResponse, ProjectUpdate
from app.schemas.user import UserResponse
from app.utils.dependencies import get_current_active_user

router = APIRouter(tags=["admin"])


class UserRoleUpdate(BaseModel):
    """Schema for updating user role."""
    system_role: str


class AdminProjectUpdate(BaseModel):
    """Schema for admin to update any project."""
    name: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None
    user_id: Optional[int] = None  # Allow admin to change owner


def require_admin(current_user: User = Depends(get_current_active_user)):
    """Dependency to check if user is admin or superadmin."""
    if current_user.system_role not in ["admin", "superadmin"]:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    return current_user


@router.get("/projects")
def list_all_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """List all projects in the system (admin only)."""

    # Query all projects with user info and experiment counts
    projects = (
        db.query(
            Project,
            User,
            func.count(TrainingJob.id).label("experiment_count")
        )
        .outerjoin(User, User.id == Project.user_id)
        .outerjoin(TrainingJob, TrainingJob.project_id == Project.id)
        .group_by(Project.id)
        .all()
    )

    # Format response
    result = []
    for project, user, exp_count in projects:
        result.append({
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "task_type": project.task_type,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "experiment_count": exp_count,
            "owner_id": project.user_id,
            "owner_name": user.full_name if user else None,
            "owner_email": user.email if user else None,
        })

    return result


@router.get("/users")
def list_all_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """List all users in the system (admin only)."""

    # Query all users with project counts
    users = (
        db.query(
            User,
            func.count(Project.id).label("project_count")
        )
        .outerjoin(Project, Project.user_id == User.id)
        .group_by(User.id)
        .all()
    )

    # Format response
    result = []
    for user, project_count in users:
        result.append({
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "company": user.company,
            "company_custom": user.company_custom,
            "division": user.division,
            "division_custom": user.division_custom,
            "department": user.department,
            "phone_number": user.phone_number,
            "system_role": user.system_role,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "project_count": project_count,
        })

    return result


@router.patch("/users/{user_id}/role")
def update_user_role(
    user_id: int,
    role_update: UserRoleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Update a user's system role (admin only)."""

    # Validate role
    if role_update.system_role not in ["guest", "admin", "superadmin"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    # Get user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent self-demotion
    if user.id == current_user.id and role_update.system_role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=400, detail="Cannot demote yourself")

    # Update role
    user.system_role = role_update.system_role
    db.commit()
    db.refresh(user)

    return {"message": "Role updated successfully", "user_id": user.id, "new_role": user.system_role}


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Delete a user (admin only). Fails if user has projects."""

    # Get user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent self-deletion
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    # Check if user has projects
    project_count = db.query(Project).filter(Project.user_id == user_id).count()
    if project_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete user with {project_count} project(s). Delete or reassign projects first."
        )

    # Delete user
    db.delete(user)
    db.commit()

    return {"message": "User deleted successfully", "user_id": user_id}


@router.put("/projects/{project_id}")
def update_project(
    project_id: int,
    project_update: AdminProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Update a project (admin only)."""

    # Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Update fields
    if project_update.name is not None:
        project.name = project_update.name
    if project_update.description is not None:
        project.description = project_update.description
    if project_update.task_type is not None:
        project.task_type = project_update.task_type
    if project_update.user_id is not None:
        # Verify new owner exists
        new_owner = db.query(User).filter(User.id == project_update.user_id).first()
        if not new_owner:
            raise HTTPException(status_code=404, detail="New owner user not found")
        project.user_id = project_update.user_id

    db.commit()
    db.refresh(project)

    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "task_type": project.task_type,
        "user_id": project.user_id,
        "updated_at": project.updated_at
    }


@router.delete("/projects/{project_id}")
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Delete a project and all its experiments (admin only)."""

    # Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get experiment count
    exp_count = db.query(TrainingJob).filter(TrainingJob.project_id == project_id).count()

    # Delete project (cascade will delete experiments)
    db.delete(project)
    db.commit()

    return {
        "message": f"Project '{project.name}' deleted successfully",
        "project_id": project_id,
        "deleted_experiments": exp_count
    }
