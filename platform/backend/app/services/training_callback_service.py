"""Training Callback Service - Centralized handler for training callbacks."""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.db import models
from app.schemas import training
from app.services.mlflow_service import MLflowService
from app.services.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)


class TrainingCallbackService:
    """
    Service for handling training callbacks from Training SDK.

    Consolidates common logic across progress, completion, and log callbacks:
    - Job validation
    - Database updates
    - MLflow integration
    - WebSocket broadcasting
    """

    def __init__(self, db: Session):
        """
        Initialize callback service.

        Args:
            db: Database session
        """
        self.db = db
        self.mlflow_service = MLflowService(db)
        self.ws_manager = get_websocket_manager()

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

    def _create_mlflow_run_if_needed(self, job: models.TrainingJob) -> Optional[str]:
        """
        Create MLflow run if not exists.

        Args:
            job: Training job

        Returns:
            MLflow run ID or None
        """
        if not self.mlflow_service.mlflow_client.available:
            return None

        if job.mlflow_run_id:
            return job.mlflow_run_id

        try:
            experiment_name = f"project-{job.project_id}"
            run_name = f"job-{job.id}-{job.model_name}"
            tags = {
                "job_id": str(job.id),
                "project_id": str(job.project_id),
                "model_name": job.model_name,
                "task_type": job.task_type,
                "framework": job.framework or "unknown"
            }

            job.mlflow_run_id = self.mlflow_service.create_mlflow_run(
                experiment_name=experiment_name,
                run_name=run_name,
                tags=tags
            )

            # Log training parameters on first run
            if job.mlflow_run_id and job.training_config:
                self.mlflow_service.log_training_params(
                    run_id=job.mlflow_run_id,
                    params=job.training_config
                )

            logger.info(f"[CALLBACK] Created MLflow run: {job.mlflow_run_id}")
            return job.mlflow_run_id

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
        if not job.mlflow_run_id or not metrics:
            return False

        try:
            metrics_dict = metrics.dict() if hasattr(metrics, 'dict') else {}

            success = self.mlflow_service.log_training_metrics(
                run_id=job.mlflow_run_id,
                metrics=metrics_dict,
                step=step
            )

            if success:
                logger.info(f"[CALLBACK] Logged metrics to MLflow for step {step}")
            else:
                logger.warning(f"[CALLBACK] Failed to log metrics to MLflow")

            return success

        except Exception as e:
            logger.warning(f"[CALLBACK] MLflow logging error: {e}")
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

            # MLflow integration
            try:
                self._create_mlflow_run_if_needed(job)
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

            # Store MLflow run ID
            if callback.mlflow_run_id:
                job.mlflow_run_id = callback.mlflow_run_id
                logger.info(f"[CALLBACK] Stored MLflow run ID: {callback.mlflow_run_id}")

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

            # MLflow integration - End the run with final status
            try:
                if job.mlflow_run_id and self.mlflow_service.mlflow_client.available:
                    # Log final metrics if provided
                    if callback.final_metrics:
                        final_metrics_dict = callback.final_metrics.dict() if hasattr(callback.final_metrics, 'dict') else {}

                        success = self.mlflow_service.log_training_metrics(
                            run_id=job.mlflow_run_id,
                            metrics=final_metrics_dict,
                            step=callback.total_epochs_completed or 0
                        )

                        if success:
                            logger.info(f"[CALLBACK] Logged final metrics to MLflow")

                    # Determine MLflow run status
                    mlflow_status = "FINISHED" if job.status == "completed" else "FAILED"

                    # End the MLflow run
                    self.mlflow_service.end_mlflow_run(
                        run_id=job.mlflow_run_id,
                        status=mlflow_status
                    )

                    logger.info(f"[CALLBACK] Ended MLflow run {job.mlflow_run_id} with status {mlflow_status}")

            except Exception as e:
                # Graceful degradation - don't fail the callback if MLflow fails
                logger.warning(f"[CALLBACK] MLflow end_run error (non-critical): {str(e)}")

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
                "mlflow_run_id": callback.mlflow_run_id,
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
