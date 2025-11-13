"""MLflow integration service for experiment tracking."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session

from app.db.models import Experiment, TrainingJob, Project
from app.utils.mlflow_client import get_mlflow_client

logger = logging.getLogger(__name__)


class MLflowService:
    """Service for managing MLflow integration with Experiments."""

    def __init__(self, db: Session):
        """Initialize MLflow service."""
        self.db = db
        self.mlflow_client = get_mlflow_client()

    # ==================== Experiment Management ====================

    def create_or_get_experiment(
        self,
        project_id: int,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Experiment:
        """
        Create a new experiment or get existing one by name.

        Args:
            project_id: Project ID
            name: Experiment name
            description: Experiment description
            tags: List of tags

        Returns:
            Experiment object
        """
        # Check if experiment already exists for this project
        experiment = self.db.query(Experiment).filter(
            Experiment.project_id == project_id,
            Experiment.name == name
        ).first()

        if experiment:
            logger.info(f"[MLflow] Using existing experiment: {name} (ID: {experiment.id})")
            return experiment

        # Create MLflow experiment
        mlflow_experiment_name = f"project_{project_id}_{name}"
        mlflow_experiment_id = None

        if self.mlflow_client.available:
            try:
                import mlflow
                mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
                logger.info(f"[MLflow] Created MLflow experiment: {mlflow_experiment_name}")
            except Exception as e:
                logger.warning(f"[MLflow] Failed to create experiment: {e}")

        # Create database experiment
        experiment = Experiment(
            project_id=project_id,
            name=name,
            description=description,
            tags=tags or [],
            mlflow_experiment_id=mlflow_experiment_id,
            mlflow_experiment_name=mlflow_experiment_name,
            num_runs=0,
            num_completed_runs=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)

        logger.info(f"[MLflow] Created experiment: {name} (ID: {experiment.id})")
        return experiment

    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment object or None
        """
        return self.db.query(Experiment).filter(Experiment.id == experiment_id).first()

    def list_experiments(
        self,
        project_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Experiment]:
        """
        List experiments with optional filters.

        Args:
            project_id: Filter by project ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of Experiment objects
        """
        query = self.db.query(Experiment)

        if project_id:
            query = query.filter(Experiment.project_id == project_id)

        return query.order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()

    def update_experiment(
        self,
        experiment_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Experiment]:
        """
        Update experiment details.

        Args:
            experiment_id: Experiment ID
            name: New name
            description: New description
            tags: New tags

        Returns:
            Updated Experiment object or None
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        if name is not None:
            experiment.name = name
        if description is not None:
            experiment.description = description
        if tags is not None:
            experiment.tags = tags

        experiment.updated_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(experiment)

        logger.info(f"[MLflow] Updated experiment: {experiment.name} (ID: {experiment_id})")
        return experiment

    def delete_experiment(self, experiment_id: int) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted, False if not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False

        # Delete from MLflow if available
        if experiment.mlflow_experiment_id and self.mlflow_client.available:
            try:
                import mlflow
                mlflow.delete_experiment(experiment.mlflow_experiment_id)
                logger.info(f"[MLflow] Deleted MLflow experiment: {experiment.mlflow_experiment_id}")
            except Exception as e:
                logger.warning(f"[MLflow] Failed to delete MLflow experiment: {e}")

        # Delete from database (cascade deletes training_jobs, stars, notes)
        self.db.delete(experiment)
        self.db.commit()

        logger.info(f"[MLflow] Deleted experiment: {experiment.name} (ID: {experiment_id})")
        return True

    # ==================== Training Job Integration ====================

    def link_training_job_to_experiment(
        self,
        training_job_id: int,
        experiment_id: int
    ) -> bool:
        """
        Link a training job to an experiment.

        Args:
            training_job_id: Training job ID
            experiment_id: Experiment ID

        Returns:
            True if successful, False otherwise
        """
        job = self.db.query(TrainingJob).filter(TrainingJob.id == training_job_id).first()
        experiment = self.get_experiment(experiment_id)

        if not job or not experiment:
            return False

        job.experiment_id = experiment_id

        # Update experiment run counts
        experiment.num_runs += 1
        experiment.updated_at = datetime.utcnow()

        self.db.commit()

        logger.info(f"[MLflow] Linked job {training_job_id} to experiment {experiment_id}")
        return True

    def update_experiment_run_status(
        self,
        experiment_id: int,
        job_id: int,
        status: str
    ) -> None:
        """
        Update experiment run status counters.

        Args:
            experiment_id: Experiment ID
            job_id: Training job ID
            status: New status ('completed', 'failed', etc.)
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return

        if status == "completed":
            experiment.num_completed_runs += 1

        experiment.updated_at = datetime.utcnow()
        self.db.commit()

        logger.debug(f"[MLflow] Updated experiment {experiment_id} run status")

    def update_experiment_best_metrics(
        self,
        experiment_id: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Update experiment's best metrics.

        Args:
            experiment_id: Experiment ID
            metrics: Dictionary of metric names and values
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return

        # Initialize or update best_metrics
        if not experiment.best_metrics:
            experiment.best_metrics = {}

        # Update with new metrics (could add logic to compare and keep best)
        experiment.best_metrics.update(metrics)
        experiment.updated_at = datetime.utcnow()

        self.db.commit()

        logger.info(f"[MLflow] Updated best metrics for experiment {experiment_id}")

    # ==================== MLflow Data Retrieval ====================

    def get_experiment_runs(self, experiment_id: int) -> List[Dict[str, Any]]:
        """
        Get all runs for an experiment from MLflow.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of run data dictionaries
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment or not experiment.mlflow_experiment_id:
            return []

        if not self.mlflow_client.available:
            logger.warning("[MLflow] Client not available")
            return []

        try:
            runs = self.mlflow_client.client.search_runs(
                experiment_ids=[experiment.mlflow_experiment_id],
                order_by=["start_time DESC"]
            )

            return [
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags),
                    "artifact_uri": run.info.artifact_uri,
                }
                for run in runs
            ]
        except Exception as e:
            logger.error(f"[MLflow] Error getting experiment runs: {e}")
            return []

    def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """
        Get detailed metrics for a specific run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary of metric data
        """
        if not self.mlflow_client.available:
            return {}

        try:
            run = self.mlflow_client.client.get_run(run_id)

            # Get metric history
            metrics_data = {}
            for metric_key in run.data.metrics.keys():
                try:
                    metric_history = self.mlflow_client.client.get_metric_history(run_id, metric_key)
                    metrics_data[metric_key] = [
                        {
                            "step": m.step,
                            "value": m.value,
                            "timestamp": m.timestamp,
                        }
                        for m in metric_history
                    ]
                except Exception as e:
                    logger.debug(f"[MLflow] Error getting metric history for {metric_key}: {e}")
                    metrics_data[metric_key] = []

            return metrics_data
        except Exception as e:
            logger.error(f"[MLflow] Error getting run metrics: {e}")
            return {}

    def sync_experiment_from_mlflow(self, experiment_id: int) -> bool:
        """
        Sync experiment data from MLflow (update run counts, best metrics).

        Args:
            experiment_id: Experiment ID

        Returns:
            True if successful, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment or not experiment.mlflow_experiment_id:
            return False

        runs = self.get_experiment_runs(experiment_id)
        if not runs:
            return False

        # Update run counts
        experiment.num_runs = len(runs)
        experiment.num_completed_runs = sum(1 for r in runs if r["status"] == "FINISHED")

        # Calculate best metrics across all runs
        # (This is a simple implementation - could be enhanced)
        all_metrics = {}
        for run in runs:
            for metric_name, value in run["metrics"].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        best_metrics = {}
        for metric_name, values in all_metrics.items():
            # Assume "loss" and "error" metrics should be minimized, others maximized
            if "loss" in metric_name.lower() or "error" in metric_name.lower():
                best_metrics[f"best_{metric_name}"] = min(values)
            else:
                best_metrics[f"best_{metric_name}"] = max(values)

        experiment.best_metrics = best_metrics
        experiment.updated_at = datetime.utcnow()

        self.db.commit()

        logger.info(f"[MLflow] Synced experiment {experiment_id} from MLflow")
        return True

    # ==================== Search and Query ====================

    def search_experiments(
        self,
        project_id: int,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Experiment]:
        """
        Search experiments by name or tags.

        Args:
            project_id: Project ID
            query: Text query to match against name/description
            tags: List of tags to filter by

        Returns:
            List of matching Experiment objects
        """
        db_query = self.db.query(Experiment).filter(Experiment.project_id == project_id)

        if query:
            search_filter = f"%{query}%"
            db_query = db_query.filter(
                (Experiment.name.ilike(search_filter)) |
                (Experiment.description.ilike(search_filter))
            )

        if tags:
            # Filter by tags (JSON field contains any of the specified tags)
            for tag in tags:
                db_query = db_query.filter(Experiment.tags.contains([tag]))

        return db_query.order_by(Experiment.created_at.desc()).all()

    def get_experiment_summary(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get comprehensive summary of an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary with experiment details, run stats, and best metrics
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {}

        # Get training jobs for this experiment
        jobs = self.db.query(TrainingJob).filter(
            TrainingJob.experiment_id == experiment_id
        ).all()

        return {
            "id": experiment.id,
            "project_id": experiment.project_id,
            "name": experiment.name,
            "description": experiment.description,
            "tags": experiment.tags,
            "num_runs": experiment.num_runs,
            "num_completed_runs": experiment.num_completed_runs,
            "best_metrics": experiment.best_metrics or {},
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "updated_at": experiment.updated_at.isoformat() if experiment.updated_at else None,
            "mlflow_experiment_id": experiment.mlflow_experiment_id,
            "training_jobs": [
                {
                    "id": job.id,
                    "status": job.status,
                    "model_name": job.model_name,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                }
                for job in jobs
            ],
        }
