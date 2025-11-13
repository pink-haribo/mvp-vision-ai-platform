"""Schemas for Experiment API."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ==================== Base Schemas ====================

class ExperimentBase(BaseModel):
    """Base experiment schema."""
    name: str = Field(..., min_length=1, max_length=200, description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    tags: Optional[List[str]] = Field(default=[], description="List of tags")


class ExperimentCreate(ExperimentBase):
    """Schema for creating an experiment."""
    project_id: int = Field(..., description="Project ID")


class ExperimentUpdate(BaseModel):
    """Schema for updating an experiment."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class ExperimentResponse(ExperimentBase):
    """Schema for experiment response."""
    id: int
    project_id: int
    mlflow_experiment_id: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    num_runs: int = 0
    num_completed_runs: int = 0
    best_metrics: Optional[Dict[str, float]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ==================== Training Job Summary ====================

class TrainingJobSummary(BaseModel):
    """Summary of a training job for experiment view."""
    id: int
    status: str
    model_name: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ==================== Experiment Summary ====================

class ExperimentSummary(ExperimentResponse):
    """Extended experiment response with training jobs."""
    training_jobs: List[TrainingJobSummary] = []


# ==================== Experiment Star ====================

class ExperimentStarCreate(BaseModel):
    """Schema for starring an experiment."""
    experiment_id: int


class ExperimentStarResponse(BaseModel):
    """Schema for experiment star response."""
    id: int
    experiment_id: int
    user_id: int
    starred_at: datetime

    class Config:
        from_attributes = True


# ==================== Experiment Note ====================

class ExperimentNoteBase(BaseModel):
    """Base experiment note schema."""
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)


class ExperimentNoteCreate(ExperimentNoteBase):
    """Schema for creating an experiment note."""
    experiment_id: int


class ExperimentNoteUpdate(BaseModel):
    """Schema for updating an experiment note."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)


class ExperimentNoteResponse(ExperimentNoteBase):
    """Schema for experiment note response."""
    id: int
    experiment_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ==================== MLflow Data ====================

class MLflowRunData(BaseModel):
    """Schema for MLflow run data."""
    run_id: str
    status: str
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    metrics: Dict[str, float] = {}
    params: Dict[str, str] = {}
    tags: Dict[str, str] = {}
    artifact_uri: Optional[str] = None


class MLflowMetricHistory(BaseModel):
    """Schema for MLflow metric history."""
    step: int
    value: float
    timestamp: int


class MLflowRunMetrics(BaseModel):
    """Schema for detailed MLflow run metrics."""
    metrics: Dict[str, List[MLflowMetricHistory]] = {}


# ==================== Search and Filter ====================

class ExperimentSearchRequest(BaseModel):
    """Schema for experiment search request."""
    query: Optional[str] = Field(None, description="Text query for name/description")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    project_id: Optional[int] = Field(None, description="Filter by project")


class ExperimentListResponse(BaseModel):
    """Schema for paginated experiment list."""
    experiments: List[ExperimentResponse]
    total: int
    skip: int
    limit: int
