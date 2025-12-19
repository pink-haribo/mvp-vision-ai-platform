"""Training Callback Service - Centralized handler for training callbacks."""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.db import models
from app.schemas import training
from app.services.clearml_service import ClearMLService
from app.services.websocket_manager import get_websocket_manager
from app.core.config import settings

# MLflow Adapter (optional)
try:
    from app.adapters.observability.mlflow_adapter import MLflowAdapter, MLFLOW_AVAILABLE
except ImportError:
    MLflowAdapter = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingCallbackService:
    """
    Service for handling training callbacks from Training SDK.

    Consolidates common logic across progress, completion, and log callbacks:
    - Job validation
    - Database updates
    - ClearML integration (Phase 12.2)
    - MLflow integration (Phase 13)
    - WebSocket broadcasting
    """

    def __init__(self, db: Session):
        """
        Initialize callback service.

        Args:
            db: Database session
        """
        self.db = db
        self.clearml_service = ClearMLService(db)
        self.ws_manager = get_websocket_manager()

        # Initialize MLflow adapter
        self.mlflow_adapter: Optional[MLflowAdapter] = None
        if MLFLOW_AVAILABLE:
            try:
                self.mlflow_adapter = MLflowAdapter()
                self.mlflow_adapter.initialize({
                    "tracking_uri": settings.MLFLOW_TRACKING_URI
                })
                logger.info(f"[CALLBACK] MLflow adapter initialized (tracking_uri={settings.MLFLOW_TRACKING_URI})")
            except Exception as e:
                logger.warning(f"[CALLBACK] MLflow adapter initialization failed (non-critical): {e}")
                self.mlflow_adapter = None

    def _get_job_or_404(self, job_id: int) -> models.TrainingJob:
        """
        Get training job by ID or raise 404.

        Args:
            job_id: Training job ID

        Returns:
            TrainingJob instance

        Raises:
            HTTPException: 404 if job not found
        """
        job = self.db.query(models.TrainingJob).filter(
            models.TrainingJob.id == job_id
        ).first()

        if not job:
            logger.error(f"[CALLBACK] Job {job_id} not found")
            raise HTTPException(status_code=404, detail="Training job not found")

        return job

    def _log_metrics_to_clearml(
        self,
        job: models.TrainingJob,
        metrics: Optional[Any],
        iteration: Optional[int] = None
    ) -> bool:
        """
        Log metrics to ClearML.

        Args:
            job: Training job
            metrics: Metrics data
            iteration: Training iteration/epoch

        Returns:
            True if successful, False otherwise
        """
        if not job.clearml_task_id or not metrics:
            return False

        try:
            metrics_dict = metrics.dict() if hasattr(metrics, 'dict') else {}

            success = self.clearml_service.log_metrics(
                task_id=job.clearml_task_id,
                metrics=metrics_dict,
                iteration=iteration
            )

            if success:
                logger.info(f"[CALLBACK] Logged metrics to ClearML for iteration {iteration}")
            else:
                logger.warning(f"[CALLBACK] Failed to log metrics to ClearML")

            return success

        except Exception as e:
            logger.warning(f"[CALLBACK] ClearML logging error: {e}")
            return False

    def _get_or_create_mlflow_run(
        self,
        job: models.TrainingJob
    ) -> Optional[str]:
        """
        Get existing MLflow run_id or create a new one.

        Args:
            job: Training job

        Returns:
            MLflow run_id or None if MLflow is not available
        """
        if not self.mlflow_adapter or not self.mlflow_adapter.is_enabled():
            return None

        # Check if run already exists in observability_experiment_ids
        experiment_ids = job.observability_experiment_ids or {}
        if "mlflow" in experiment_ids:
            return experiment_ids["mlflow"]

        # Create new MLflow run
        try:
            # Get project name for grouping
            project = self.db.query(models.Project).filter(
                models.Project.id == job.project_id
            ).first()
            project_name = project.name if project else "default"

            run_id = self.mlflow_adapter.create_experiment(
                job_id=job.id,
                project_name=project_name,
                experiment_name=f"Job {job.id} - {job.model_name}",
                tags=[job.task_type, job.framework] if job.task_type and job.framework else None,
                hyperparameters=job.config if isinstance(job.config, dict) else None
            )

            # Store run_id in observability_experiment_ids
            experiment_ids["mlflow"] = run_id
            job.observability_experiment_ids = experiment_ids

            logger.info(f"[CALLBACK] Created MLflow run {run_id} for job {job.id}")
            return run_id

        except Exception as e:
            logger.warning(f"[CALLBACK] Failed to create MLflow run: {e}")
            return None

    def _log_metrics_to_mlflow(
        self,
        job: models.TrainingJob,
        metrics: Optional[Any],
        step: Optional[int] = None
    ) -> bool:
        """
        Log metrics to MLflow.

        Args:
            job: Training job
            metrics: Metrics data
            step: Training step/epoch

        Returns:
            True if successful, False otherwise
        """
        if not self.mlflow_adapter or not self.mlflow_adapter.is_enabled():
            return False

        if not metrics:
            return False

        # Get or create MLflow run
        run_id = self._get_or_create_mlflow_run(job)
        if not run_id:
            return False

        try:
            # Extract metrics dict
            metrics_dict = metrics.dict() if hasattr(metrics, 'dict') else {}
            extra_metrics = metrics_dict.get('extra_metrics', {})

            # Flatten metrics for MLflow
            flat_metrics = {}
            for key, value in metrics_dict.items():
                if key != 'extra_metrics' and value is not None:
                    flat_metrics[key] = float(value) if isinstance(value, (int, float)) else None

            # Add extra_metrics
            for key, value in extra_metrics.items():
                if value is not None:
                    try:
                        flat_metrics[key] = float(value)
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric values

            # Remove None values
            flat_metrics = {k: v for k, v in flat_metrics.items() if v is not None}

            if flat_metrics:
                self.mlflow_adapter.log_metrics(
                    experiment_id=run_id,
                    metrics=flat_metrics,
                    step=step or 0
                )
                logger.debug(f"[CALLBACK] Logged {len(flat_metrics)} metrics to MLflow for step {step}")
                return True

            return False

        except Exception as e:
            logger.warning(f"[CALLBACK] MLflow logging error: {e}")
            return False

    def _finalize_mlflow_run(
        self,
        job: models.TrainingJob,
        status: str,
        final_metrics: Optional[Any] = None
    ) -> bool:
        """
        Finalize MLflow run (mark as completed/failed).

        Args:
            job: Training job
            status: Final status ("completed" or "failed")
            final_metrics: Optional final metrics

        Returns:
            True if successful, False otherwise
        """
        if not self.mlflow_adapter or not self.mlflow_adapter.is_enabled():
            return False

        experiment_ids = job.observability_experiment_ids or {}
        run_id = experiment_ids.get("mlflow")

        if not run_id:
            return False

        try:
            # Extract final metrics
            final_metrics_dict = None
            if final_metrics:
                metrics_data = final_metrics.dict() if hasattr(final_metrics, 'dict') else {}
                extra_metrics = metrics_data.get('extra_metrics', {})

                final_metrics_dict = {}
                for key, value in {**metrics_data, **extra_metrics}.items():
                    if key != 'extra_metrics' and value is not None:
                        try:
                            final_metrics_dict[key] = float(value)
                        except (ValueError, TypeError):
                            pass

            self.mlflow_adapter.finalize_experiment(
                experiment_id=run_id,
                status=status,
                final_metrics=final_metrics_dict
            )

            logger.info(f"[CALLBACK] Finalized MLflow run {run_id} with status '{status}'")
            return True

        except Exception as e:
            logger.warning(f"[CALLBACK] Failed to finalize MLflow run: {e}")
            return False

    async def handle_progress(
        self,
        job_id: int,
        callback: training.TrainingProgressCallback
    ) -> training.TrainingCallbackResponse:
        """
        Handle training progress callback.

        Updates job status, stores metrics, logs to MLflow, and broadcasts via WebSocket.

        Args:
            job_id: Training job ID
            callback: Progress callback data

        Returns:
            Callback response

        Raises:
            HTTPException: On validation or processing errors
        """
        try:
            job = self._get_job_or_404(job_id)

            logger.info(
                f"[CALLBACK] Progress update for job {job_id}: "
                f"epoch {callback.current_epoch}/{callback.total_epochs}, "
                f"status={callback.status}"
            )

            # Update job status
            job.status = callback.status

            # Update started_at if this is the first progress update
            if job.started_at is None and callback.status == "running":
                job.started_at = datetime.utcnow()

            # Handle completion or failure
            if callback.status in ["completed", "failed"]:
                job.completed_at = datetime.utcnow()
                if callback.error_message:
                    job.error_message = callback.error_message

            # Store metrics in database if provided
            if callback.metrics:
                metrics_dict = callback.metrics.dict() if hasattr(callback.metrics, 'dict') else {}
                extra_metrics = metrics_dict.get('extra_metrics', {})

                metric = models.TrainingMetric(
                    job_id=job_id,
                    epoch=callback.current_epoch,
                    step=None,
                    loss=metrics_dict.get('loss') or extra_metrics.get('loss'),
                    accuracy=metrics_dict.get('accuracy') or extra_metrics.get('accuracy'),
                    learning_rate=metrics_dict.get('learning_rate') or extra_metrics.get('learning_rate') or extra_metrics.get('lr'),
                    extra_metrics=extra_metrics if extra_metrics else metrics_dict,
                    checkpoint_path=callback.checkpoint_path,
                )
                self.db.add(metric)

                logger.info(f"[CALLBACK] Created TrainingMetric record for epoch {callback.current_epoch}")

            # Update best checkpoint path if provided
            if callback.best_checkpoint_path:
                job.best_checkpoint_path = callback.best_checkpoint_path

            # ClearML integration (Phase 12.2)
            try:
                self._log_metrics_to_clearml(job, callback.metrics, callback.current_epoch)
            except Exception as e:
                logger.warning(f"[CALLBACK] ClearML integration error (non-critical): {e}")

            # MLflow integration (Phase 13)
            try:
                self._log_metrics_to_mlflow(job, callback.metrics, callback.current_epoch)
            except Exception as e:
                logger.warning(f"[CALLBACK] MLflow integration error (non-critical): {e}")

            self.db.commit()
            logger.info(f"[CALLBACK] Successfully updated job {job_id}")

            # Broadcast to WebSocket clients
            await self.ws_manager.broadcast_to_job(job_id, {
                "type": "training_progress",
                "job_id": job_id,
                "status": callback.status,
                "current_epoch": callback.current_epoch,
                "total_epochs": callback.total_epochs,
                "progress_percent": callback.progress_percent,
                "metrics": callback.metrics.dict() if callback.metrics else None,
                "checkpoint_path": callback.checkpoint_path,
                "best_checkpoint_path": callback.best_checkpoint_path,
            })

            return training.TrainingCallbackResponse(
                success=True,
                message=f"Progress update received for epoch {callback.current_epoch}",
                job_status=job.status
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.error(f"[CALLBACK] Error processing progress update: {str(e)}")
            logger.error(traceback.format_exc())
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process progress callback: {str(e)}"
            )

    async def handle_completion(
        self,
        job_id: int,
        callback: training.TrainingCompletionCallback
    ) -> training.TrainingCallbackResponse:
        """
        Handle training completion callback.

        Args:
            job_id: Training job ID
            callback: Completion callback data

        Returns:
            Callback response

        Raises:
            HTTPException: On validation or processing errors
        """
        try:
            job = self._get_job_or_404(job_id)

            logger.info(
                f"[CALLBACK] Completion callback for job {job_id}: "
                f"status={callback.status}, "
                f"epochs_completed={callback.total_epochs_completed}, "
                f"exit_code={callback.exit_code}"
            )

            # Update job status
            # K8s Job compatibility: Use exit_code if provided
            if callback.exit_code is not None:
                if callback.exit_code == 0:
                    job.status = "completed"
                else:
                    job.status = "failed"
                    if not callback.error_message:
                        callback.error_message = f"Training failed with exit code {callback.exit_code}"
            else:
                job.status = callback.status

            # Update completion time
            job.completed_at = datetime.utcnow()

            # Update job final accuracy if available (from final_metrics)
            if callback.final_metrics:
                # Extract metrics dynamically to support any trainer's metric structure
                metrics_dict = callback.final_metrics.dict() if hasattr(callback.final_metrics, 'dict') else {}
                extra_metrics = metrics_dict.get('extra_metrics', {})
                job.final_accuracy = metrics_dict.get('accuracy') or extra_metrics.get('accuracy')

            # NOTE: Do NOT create duplicate TrainingMetric record here
            # Metrics are already saved by epoch callbacks during training
            # Completion callback only updates checkpoint paths and final status

            # Store best metrics if provided
            if callback.best_metrics and callback.best_epoch:
                # Update existing metric record for best epoch
                best_metric = self.db.query(models.TrainingMetric).filter(
                    models.TrainingMetric.job_id == job_id,
                    models.TrainingMetric.epoch == callback.best_epoch
                ).first()

                if best_metric:
                    best_metric.extra_metrics = best_metric.extra_metrics or {}
                    best_metric.extra_metrics["is_best"] = True
                    best_metric.checkpoint_path = callback.best_checkpoint_path

            # Update checkpoint paths
            if callback.best_checkpoint_path:
                job.best_checkpoint_path = callback.best_checkpoint_path
            if callback.last_checkpoint_path:
                job.last_checkpoint_path = callback.last_checkpoint_path

            # Store error information if failed
            if callback.status == "failed" or (callback.exit_code and callback.exit_code != 0):
                job.error_message = callback.error_message or "Training failed (no error message provided)"

                # Log traceback if available
                if callback.traceback:
                    logger.error(f"[CALLBACK] Job {job_id} traceback:\n{callback.traceback}")
                    # Store traceback in logs
                    log_entry = models.TrainingLog(
                        job_id=job_id,
                        log_type="stderr",
                        content=f"TRACEBACK:\n{callback.traceback}"
                    )
                    self.db.add(log_entry)

            # ClearML integration (Phase 12.2) - Mark task as completed/failed
            try:
                if job.clearml_task_id:
                    # Log final metrics if provided
                    if callback.final_metrics:
                        final_metrics_dict = callback.final_metrics.dict() if hasattr(callback.final_metrics, 'dict') else {}

                        success = self.clearml_service.log_metrics(
                            task_id=job.clearml_task_id,
                            metrics=final_metrics_dict,
                            iteration=callback.total_epochs_completed or 0
                        )

                        if success:
                            logger.info(f"[CALLBACK] Logged final metrics to ClearML")

                    # Mark task as completed or failed
                    if job.status == "completed":
                        self.clearml_service.mark_completed(job.clearml_task_id)
                        logger.info(f"[CALLBACK] Marked ClearML task {job.clearml_task_id} as completed")
                    else:
                        self.clearml_service.mark_failed(
                            job.clearml_task_id,
                            error_message=job.error_message
                        )
                        logger.info(f"[CALLBACK] Marked ClearML task {job.clearml_task_id} as failed")

            except Exception as e:
                # Graceful degradation - don't fail the callback if ClearML fails
                logger.warning(f"[CALLBACK] ClearML task update error (non-critical): {str(e)}")

            # MLflow integration (Phase 13) - Finalize run
            try:
                self._finalize_mlflow_run(job, job.status, callback.final_metrics)
            except Exception as e:
                logger.warning(f"[CALLBACK] MLflow finalization error (non-critical): {str(e)}")

            self.db.commit()

            logger.info(
                f"[CALLBACK] Successfully processed completion for job {job_id} "
                f"(status={job.status}, final_accuracy={job.final_accuracy})"
            )

            # Broadcast completion to WebSocket clients
            await self.ws_manager.broadcast_to_job(job_id, {
                "type": "training_complete" if callback.status == "completed" else "training_error",
                "job_id": job_id,
                "status": callback.status,
                "total_epochs_completed": callback.total_epochs_completed,
                "final_metrics": callback.final_metrics.dict() if callback.final_metrics else None,
                "best_metrics": callback.best_metrics.dict() if callback.best_metrics else None,
                "best_epoch": callback.best_epoch,
                "final_checkpoint_path": callback.final_checkpoint_path,
                "best_checkpoint_path": callback.best_checkpoint_path,
                "clearml_task_id": job.clearml_task_id,  # Phase 12.2: ClearML
                "mlflow_run_id": (job.observability_experiment_ids or {}).get("mlflow"),  # Phase 13: MLflow
                "error_message": callback.error_message,
            })

            return training.TrainingCallbackResponse(
                success=True,
                message=f"Training {callback.status} after {callback.total_epochs_completed} epochs",
                job_status=job.status
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.error(f"[CALLBACK] Error processing completion callback: {str(e)}")
            logger.error(traceback.format_exc())
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process completion callback: {str(e)}"
            )

    async def handle_log(
        self,
        job_id: int,
        callback: training.LogEventCallback
    ) -> training.TrainingCallbackResponse:
        """
        Handle training log callback.

        Args:
            job_id: Training job ID
            callback: Log callback data

        Returns:
            Callback response

        Raises:
            HTTPException: On validation or processing errors
        """
        try:
            job = self._get_job_or_404(job_id)

            logger.info(
                f"[LOG] Event for job {job_id}: [{callback.level}] {callback.event_type} - {callback.message}"
            )

            # Store in database
            log_entry = models.TrainingLog(
                job_id=job_id,
                log_type=callback.event_type,
                content=f"[{callback.level}] {callback.message}"
            )
            self.db.add(log_entry)

            # Log rotation: Keep only latest 2000 logs per job
            log_count = self.db.query(models.TrainingLog).filter(
                models.TrainingLog.job_id == job_id
            ).count()

            if log_count >= 2000:
                # Delete oldest logs to maintain 2000 limit
                # Keep 1900 logs + new one = 1901 total (leaves room for next batch)
                logs_to_keep = 1900
                oldest_logs = self.db.query(models.TrainingLog).filter(
                    models.TrainingLog.job_id == job_id
                ).order_by(models.TrainingLog.id.asc()).offset(logs_to_keep).all()

                if oldest_logs:
                    for old_log in oldest_logs:
                        self.db.delete(old_log)
                    logger.info(f"[LOG] Rotated {len(oldest_logs)} old logs for job {job_id} (kept {logs_to_keep})")

            self.db.commit()

            # Broadcast to WebSocket clients
            await self.ws_manager.broadcast_to_job(job_id, {
                "type": "training_log",
                "job_id": job_id,
                "level": callback.level,
                "event_type": callback.event_type,
                "message": callback.message,
                "timestamp": datetime.utcnow().isoformat(),
            })

            return training.TrainingCallbackResponse(
                success=True,
                message=f"Log event received: {callback.event_type}",
                job_status=job.status
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.error(f"[LOG] Error processing log callback: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process log callback: {str(e)}"
            )
