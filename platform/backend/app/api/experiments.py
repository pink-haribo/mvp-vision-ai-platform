"""Experiments API endpoints for MLflow integration."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.database import get_db
from app.db.models import User, Experiment, ExperimentStar, ExperimentNote
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
    ExperimentSummary,
    ExperimentStarCreate,
    ExperimentStarResponse,
    ExperimentNoteCreate,
    ExperimentNoteUpdate,
    ExperimentNoteResponse,
    ExperimentListResponse,
    ExperimentSearchRequest,
    MLflowRunData,
    MLflowRunMetrics,
    TrainingJobSummary,
)
from app.utils.dependencies import get_current_active_user
from app.services.mlflow_service import MLflowService

router = APIRouter(prefix="/experiments", tags=["experiments"])


# ==================== Experiment CRUD ====================

@router.post("", response_model=ExperimentResponse)
def create_experiment(
    experiment: ExperimentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new experiment for a project.

    Requires project access (owner or member).
    """
    mlflow_service = MLflowService(db)

    # TODO: Add project access check (owner or member)

    try:
        exp = mlflow_service.create_or_get_experiment(
            project_id=experiment.project_id,
            name=experiment.name,
            description=experiment.description,
            tags=experiment.tags
        )
        return exp
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentSummary)
def get_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get experiment details with training jobs.

    Requires project access (owner or member).
    """
    mlflow_service = MLflowService(db)

    experiment = mlflow_service.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # TODO: Add project access check

    summary = mlflow_service.get_experiment_summary(experiment_id)
    return summary


@router.get("", response_model=ExperimentListResponse)
def list_experiments(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    List experiments with pagination.

    Can filter by project_id.
    """
    mlflow_service = MLflowService(db)

    experiments = mlflow_service.list_experiments(
        project_id=project_id,
        skip=skip,
        limit=limit
    )

    # Count total
    query = db.query(Experiment)
    if project_id:
        query = query.filter(Experiment.project_id == project_id)
    total = query.count()

    return ExperimentListResponse(
        experiments=experiments,
        total=total,
        skip=skip,
        limit=limit
    )


@router.put("/{experiment_id}", response_model=ExperimentResponse)
def update_experiment(
    experiment_id: int,
    update_data: ExperimentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update experiment details.

    Requires project access (owner or member).
    """
    mlflow_service = MLflowService(db)

    experiment = mlflow_service.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # TODO: Add project access check

    updated_experiment = mlflow_service.update_experiment(
        experiment_id=experiment_id,
        name=update_data.name,
        description=update_data.description,
        tags=update_data.tags
    )

    return updated_experiment


@router.delete("/{experiment_id}")
def delete_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete an experiment.

    Requires project owner permissions.
    This will also delete all associated training jobs, stars, and notes.
    """
    mlflow_service = MLflowService(db)

    experiment = mlflow_service.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # TODO: Add project owner check

    success = mlflow_service.delete_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete experiment")

    return {"message": "Experiment deleted successfully"}


# ==================== Search and Filter ====================

@router.post("/search", response_model=List[ExperimentResponse])
def search_experiments(
    search: ExperimentSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Search experiments by name, description, or tags.

    Project ID is required for search.
    """
    if not search.project_id:
        raise HTTPException(status_code=400, detail="project_id is required for search")

    mlflow_service = MLflowService(db)

    # TODO: Add project access check

    experiments = mlflow_service.search_experiments(
        project_id=search.project_id,
        query=search.query,
        tags=search.tags
    )

    return experiments


# ==================== MLflow Data ====================

@router.get("/{experiment_id}/runs", response_model=List[MLflowRunData])
def get_experiment_runs(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get all MLflow runs for an experiment.

    Requires project access (owner or member).
    """
    mlflow_service = MLflowService(db)

    experiment = mlflow_service.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # TODO: Add project access check

    runs = mlflow_service.get_experiment_runs(experiment_id)
    return runs


@router.get("/{experiment_id}/runs/{run_id}/metrics", response_model=MLflowRunMetrics)
def get_run_metrics(
    experiment_id: int,
    run_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get detailed metrics for a specific MLflow run.

    Requires project access (owner or member).
    """
    mlflow_service = MLflowService(db)

    experiment = mlflow_service.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # TODO: Add project access check

    metrics = mlflow_service.get_run_metrics(run_id)
    return MLflowRunMetrics(metrics=metrics)


@router.post("/{experiment_id}/sync")
def sync_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Sync experiment data from MLflow (update run counts and best metrics).

    Requires project access (owner or member).
    """
    mlflow_service = MLflowService(db)

    experiment = mlflow_service.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # TODO: Add project access check

    success = mlflow_service.sync_experiment_from_mlflow(experiment_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to sync experiment")

    return {"message": "Experiment synced successfully"}


# ==================== Stars (Favorites) ====================

@router.post("/{experiment_id}/star", response_model=ExperimentStarResponse)
def star_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Star (favorite) an experiment.
    """
    # Check if experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Check if already starred
    existing_star = db.query(ExperimentStar).filter(
        ExperimentStar.experiment_id == experiment_id,
        ExperimentStar.user_id == current_user.id
    ).first()

    if existing_star:
        return existing_star

    # Create star
    star = ExperimentStar(
        experiment_id=experiment_id,
        user_id=current_user.id
    )
    db.add(star)
    db.commit()
    db.refresh(star)

    return star


@router.delete("/{experiment_id}/star")
def unstar_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Unstar (remove favorite) an experiment.
    """
    star = db.query(ExperimentStar).filter(
        ExperimentStar.experiment_id == experiment_id,
        ExperimentStar.user_id == current_user.id
    ).first()

    if not star:
        raise HTTPException(status_code=404, detail="Star not found")

    db.delete(star)
    db.commit()

    return {"message": "Experiment unstarred successfully"}


@router.get("/starred/list", response_model=List[ExperimentResponse])
def list_starred_experiments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    List all experiments starred by the current user.
    """
    starred = db.query(ExperimentStar).filter(
        ExperimentStar.user_id == current_user.id
    ).all()

    experiment_ids = [s.experiment_id for s in starred]

    experiments = db.query(Experiment).filter(
        Experiment.id.in_(experiment_ids)
    ).all()

    return experiments


# ==================== Notes ====================

@router.post("/{experiment_id}/notes", response_model=ExperimentNoteResponse)
def create_note(
    experiment_id: int,
    note: ExperimentNoteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a note for an experiment.

    Requires project access (owner or member).
    """
    # Check if experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # TODO: Add project access check

    db_note = ExperimentNote(
        experiment_id=experiment_id,
        user_id=current_user.id,
        title=note.title,
        content=note.content
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)

    return db_note


@router.get("/{experiment_id}/notes", response_model=List[ExperimentNoteResponse])
def list_notes(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    List all notes for an experiment.

    Requires project access (owner or member).
    """
    # Check if experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # TODO: Add project access check

    notes = db.query(ExperimentNote).filter(
        ExperimentNote.experiment_id == experiment_id
    ).order_by(ExperimentNote.created_at.desc()).all()

    return notes


@router.put("/notes/{note_id}", response_model=ExperimentNoteResponse)
def update_note(
    note_id: int,
    update_data: ExperimentNoteUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update a note.

    Only the note author can update.
    """
    note = db.query(ExperimentNote).filter(ExperimentNote.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Check if user is the author
    if note.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this note")

    if update_data.title is not None:
        note.title = update_data.title
    if update_data.content is not None:
        note.content = update_data.content

    from datetime import datetime
    note.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(note)

    return note


@router.delete("/notes/{note_id}")
def delete_note(
    note_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a note.

    Only the note author can delete.
    """
    note = db.query(ExperimentNote).filter(ExperimentNote.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Check if user is the author
    if note.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this note")

    db.delete(note)
    db.commit()

    return {"message": "Note deleted successfully"}
