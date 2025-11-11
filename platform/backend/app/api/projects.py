"""
Project API Endpoints

Provides CRUD operations for projects.
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_active_user
from app.db.models import Project, User
from app.db.session import get_db
from app.schemas.projects import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
)

router = APIRouter()


# ==================== Project CRUD ====================


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new project.

    Args:
        project_data: Project creation data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Created project information
    """
    # Create new project
    new_project = Project(
        name=project_data.name,
        description=project_data.description,
        owner_id=current_user.id,
    )

    db.add(new_project)
    await db.commit()
    await db.refresh(new_project)

    return new_project


@router.get("", response_model=List[ProjectResponse])
async def list_projects(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all projects for the current user.

    Args:
        current_user: Current authenticated user
        db: Database session

    Returns:
        List of projects
    """
    query = (
        select(Project)
        .where(Project.owner_id == current_user.id)
        .order_by(Project.created_at.desc())
    )

    result = await db.execute(query)
    projects = result.scalars().all()

    return projects


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific project by ID.

    Args:
        project_id: Project ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Project information

    Raises:
        HTTPException: If project not found or not owned by user
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Check ownership
    if project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this project",
        )

    return project


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a project.

    Args:
        project_id: Project ID
        project_data: Project update data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Updated project information

    Raises:
        HTTPException: If project not found or not owned by user
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Check ownership
    if project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to update this project",
        )

    # Update fields
    if project_data.name is not None:
        project.name = project_data.name
    if project_data.description is not None:
        project.description = project_data.description

    project.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(project)

    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a project.

    Args:
        project_id: Project ID
        current_user: Current authenticated user
        db: Database session

    Raises:
        HTTPException: If project not found or not owned by user
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Check ownership
    if project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this project",
        )

    await db.delete(project)
    await db.commit()

    return None
