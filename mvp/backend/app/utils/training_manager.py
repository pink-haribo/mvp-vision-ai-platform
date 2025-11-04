"""Training process manager using Training Service API."""

import os
from typing import Optional

from sqlalchemy.orm import Session

from app.db import models


class TrainingManager:
    """Manage training execution via Training Service API.

    This manager always delegates training to the Training Service via HTTP API.
    This ensures Dev-Prod parity: local Docker Compose setup works identically to Railway deployment.
    """

    def __init__(self, db: Session):
        """
        Initialize training manager.

        Args:
            db: Database session
        """
        self.db = db
        print("[TrainingManager] Using Training Service API mode (Dev-Prod parity)")

    def start_training(self, job_id: int, checkpoint_path: Optional[str] = None, resume: bool = False) -> bool:
        """
        Start training via Training Service API.

        Args:
            job_id: Training job ID
            checkpoint_path: Optional path to checkpoint to load
            resume: If True, resume training from checkpoint (restore optimizer/scheduler state)

        Returns:
            True if training started successfully
        """
        return self._start_training_api(job_id, checkpoint_path, resume)

    def stop_training(self, job_id: int) -> bool:
        """
        Stop a running training job via Training Service API.

        Args:
            job_id: Training job ID

        Returns:
            True if stopped successfully
        """
        from app.utils.training_client import TrainingServiceClient

        # Get job from database
        job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if not job:
            print(f"[TrainingManager] Job {job_id} not found")
            return False

        try:
            # Initialize Training Service client
            client = TrainingServiceClient(framework=job.framework)

            # Stop training via API
            success = client.stop_training(job_id)

            if success:
                # Update job status in Backend DB
                job.status = "cancelled"
                self.db.commit()
                print(f"[TrainingManager] Training stopped successfully for job {job_id}")
                return True
            else:
                print(f"[TrainingManager] Failed to stop training via API for job {job_id}")
                return False

        except Exception as e:
            print(f"[TrainingManager] Error stopping training: {e}")
            return False

    def _start_training_api(self, job_id: int, checkpoint_path: Optional[str] = None, resume: bool = False) -> bool:
        """
        Start training via Training Service API (Production mode).

        Args:
            job_id: Training job ID
            checkpoint_path: Optional checkpoint path
            resume: If True, resume from checkpoint

        Returns:
            True if training started successfully
        """
        from app.utils.training_client import TrainingServiceClient

        # Get job from database
        job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if not job:
            print(f"[TrainingManager] Job {job_id} not found")
            return False

        if job.status != "pending":
            print(f"[TrainingManager] Job {job_id} status is {job.status}, expected 'pending'")
            return False

        # Prepare training config
        job_config = {
            "job_id": job_id,
            "framework": job.framework,
            "task_type": job.task_type,
            "model_name": job.model_name,
            "dataset_path": job.dataset_path,
            "dataset_format": job.dataset_format,
            "num_classes": job.num_classes or 1000,
            "output_dir": job.output_dir,
            "epochs": job.epochs,
            "batch_size": job.batch_size,
            "learning_rate": job.learning_rate,
            "optimizer": "adam",
            "device": "cpu",  # Railway doesn't have GPU
            "image_size": 224,
            "pretrained": True,
            "checkpoint_path": checkpoint_path,
            "resume": resume,
            "advanced_config": job.advanced_config  # Pass advanced_config from Backend DB
        }

        print(f"[TrainingManager] Starting training via API for job {job_id}")
        print(f"[TrainingManager] Config: {job_config}")

        try:
            # Initialize Training Service client with framework-specific routing
            client = TrainingServiceClient(framework=job.framework)

            # Check if Training Service is healthy
            if not client.health_check():
                raise Exception(f"Training Service for framework '{job.framework}' is not healthy")

            # Start training
            client.start_training(job_config)

            # Update job status
            job.status = "running"
            self.db.commit()

            print(f"[TrainingManager] Training started successfully for job {job_id}")
            return True

        except Exception as e:
            print(f"[TrainingManager] Failed to start training: {e}")
            job.status = "failed"
            job.error_message = f"Failed to start training: {str(e)}"
            self.db.commit()
            return False
