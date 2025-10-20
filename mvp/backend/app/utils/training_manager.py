"""Training process manager using subprocess."""

import json
import os
import subprocess
import threading
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.db import models
from app.utils.metrics import update_training_metrics, clear_training_metrics, active_training_jobs


class TrainingManager:
    """Manage training subprocess and collect metrics."""

    def __init__(self, db: Session):
        """
        Initialize training manager.

        Args:
            db: Database session
        """
        self.db = db
        self.processes = {}  # job_id -> process

    def start_training(self, job_id: int) -> bool:
        """
        Start training for a job.

        Args:
            job_id: Training job ID

        Returns:
            True if training started successfully
        """
        # Get job from database
        job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if not job:
            return False

        if job.status != "pending":
            return False

        # Get Python executable path for training environment
        training_python = self._get_training_python()

        # Get project root directory
        # __file__ = .../mvp/backend/app/utils/training_manager.py
        # We need to go up to .../mvp-vision-ai-platform/
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../mvp/backend
        mvp_dir = os.path.dirname(backend_dir)  # .../mvp
        project_root = os.path.dirname(mvp_dir)  # .../mvp-vision-ai-platform

        # Get absolute path to train.py
        train_script = os.path.join(project_root, "mvp", "training", "train.py")

        # Build command with absolute paths
        cmd = [
            training_python,
            train_script,
            "--dataset_path", job.dataset_path,
            "--output_dir", job.output_dir,
            "--num_classes", str(job.num_classes),
            "--epochs", str(job.epochs),
            "--batch_size", str(job.batch_size),
            "--learning_rate", str(job.learning_rate),
            "--job_id", str(job.id),
        ]

        try:
            # Debug logging
            print(f"[DEBUG] Project root: {project_root}")
            print(f"[DEBUG] Train script: {train_script}")
            print(f"[DEBUG] Training python: {training_python}")
            print(f"[DEBUG] Command: {' '.join(cmd)}")
            print(f"[DEBUG] CWD: {project_root}")

            # Start subprocess (run from project root)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=project_root,  # Set working directory to project root
            )

            # Store process
            self.processes[job_id] = process

            # Update job status
            job.status = "running"
            job.started_at = datetime.utcnow()
            job.process_id = process.pid
            self.db.commit()

            # Update Prometheus metrics
            active_training_jobs.inc()
            update_training_metrics(
                job_id=job.id,
                model_name="resnet18",  # TODO: Get from job config
                dataset_name=os.path.basename(job.dataset_path),
                status="running",
            )

            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_training,
                args=(job_id, process),
                daemon=True,
            )
            monitor_thread.start()

            return True

        except Exception as e:
            job.status = "failed"
            job.error_message = f"Failed to start training: {str(e)}"
            self.db.commit()
            return False

    def _monitor_training(self, job_id: int, process: subprocess.Popen):
        """
        Monitor training process and parse stdout.

        Args:
            job_id: Training job ID
            process: Subprocess instance
        """
        # Import here to avoid circular dependency
        from app.db.database import SessionLocal

        # Use a dedicated DB session for this thread
        db = SessionLocal()

        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                # Save log to database
                self._save_log(job_id, line, "stdout")

                # Parse metrics from stdout
                self._parse_and_save_metrics(job_id, line)

            # Wait for process to complete
            process.wait()

            # Update job status
            job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
            if job:
                if process.returncode == 0:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()

                    # Get final accuracy from latest metric
                    latest_metric = (
                        db.query(models.TrainingMetric)
                        .filter(models.TrainingMetric.job_id == job_id)
                        .order_by(models.TrainingMetric.epoch.desc())
                        .first()
                    )
                    if latest_metric:
                        job.final_accuracy = latest_metric.accuracy
                        job.best_checkpoint_path = os.path.join(job.output_dir, "best_model.pth")

                    # Update Prometheus metrics
                    update_training_metrics(
                        job_id=job_id,
                        model_name="resnet18",
                        dataset_name=os.path.basename(job.dataset_path),
                        status="completed",
                    )
                else:
                    job.status = "failed"
                    job.error_message = f"Training process exited with code {process.returncode}"
                    job.completed_at = datetime.utcnow()

                    # Update Prometheus metrics
                    update_training_metrics(
                        job_id=job_id,
                        model_name="resnet18",
                        dataset_name=os.path.basename(job.dataset_path),
                        status="failed",
                    )

                db.commit()

                # Decrement active jobs counter
                active_training_jobs.dec()

        except Exception as e:
            print(f"Exception in _monitor_training: {e}")
            try:
                job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
                if job:
                    job.status = "failed"
                    job.error_message = f"Monitoring error: {str(e)}"
                    job.completed_at = datetime.utcnow()
                    db.commit()

                    # Update Prometheus metrics
                    update_training_metrics(
                        job_id=job_id,
                        model_name="resnet18",
                        dataset_name=os.path.basename(job.dataset_path),
                        status="failed",
                    )

                    # Decrement active jobs counter
                    active_training_jobs.dec()
            except Exception as inner_e:
                print(f"Error updating job status after exception: {inner_e}")
                db.rollback()

        finally:
            # Close the dedicated DB session
            db.close()

            # Remove from active processes
            if job_id in self.processes:
                del self.processes[job_id]

    def _save_log(self, job_id: int, content: str, log_type: str):
        """
        Save a log entry to the database.

        Args:
            job_id: Training job ID
            content: Log content
            log_type: Type of log ('stdout' or 'stderr')
        """
        # Import here to avoid circular dependency
        from app.db.database import SessionLocal

        # Use a new session for each log save to avoid session conflicts
        db = SessionLocal()
        try:
            log = models.TrainingLog(
                job_id=job_id,
                log_type=log_type,
                content=content,
            )
            db.add(log)
            db.commit()
        except Exception as e:
            print(f"Error saving log: {e}")
            db.rollback()
            # Don't fail training if logging fails
            pass
        finally:
            db.close()

    def _parse_and_save_metrics(self, job_id: int, line: str):
        """
        Parse stdout line and save metrics if found.

        Args:
            job_id: Training job ID
            line: Output line from training process
        """
        # Look for [METRICS] tag
        if "[METRICS]" in line:
            # Import here to avoid circular dependency
            from app.db.database import SessionLocal

            db = SessionLocal()
            try:
                # Extract JSON from line
                json_start = line.find("{")
                json_str = line[json_start:]
                metrics_data = json.loads(json_str)

                # Use val_loss/val_accuracy if available, otherwise use train_loss/train_accuracy
                loss = metrics_data.get("val_loss") or metrics_data.get("train_loss")
                accuracy = metrics_data.get("val_accuracy") or metrics_data.get("train_accuracy")

                # Save to database (only if we have a loss value)
                if loss is not None:
                    metric = models.TrainingMetric(
                        job_id=job_id,
                        epoch=metrics_data.get("epoch"),
                        loss=loss,
                        accuracy=accuracy,
                        learning_rate=metrics_data.get("learning_rate"),
                        extra_metrics={
                            "train_loss": metrics_data.get("train_loss"),
                            "train_accuracy": metrics_data.get("train_accuracy"),
                            "val_loss": metrics_data.get("val_loss"),
                            "val_accuracy": metrics_data.get("val_accuracy"),
                            "epoch_time": metrics_data.get("epoch_time"),
                            "batch": metrics_data.get("batch"),
                            "total_batches": metrics_data.get("total_batches"),
                        },
                    )
                    db.add(metric)
                    db.commit()

                # Export to Prometheus (if we have metrics to export)
                if loss is not None:
                    # Accuracy might be in 0-1 range, convert to percentage
                    acc_percentage = accuracy * 100 if accuracy and accuracy <= 1.0 else (accuracy or 0)

                    update_training_metrics(
                        job_id=job_id,
                        model_name="resnet18",  # TODO: Get from job config
                        loss=loss,
                        accuracy=acc_percentage,
                        epoch=metrics_data.get("epoch"),
                    )

            except json.JSONDecodeError:
                pass  # Ignore lines that aren't valid JSON
            except Exception as e:
                print(f"Error parsing metrics: {e}")
                db.rollback()
            finally:
                db.close()

    def stop_training(self, job_id: int) -> bool:
        """
        Stop a running training job.

        Args:
            job_id: Training job ID

        Returns:
            True if stopped successfully
        """
        if job_id not in self.processes:
            return False

        # Import here to avoid circular dependency
        from app.db.database import SessionLocal

        db = SessionLocal()
        try:
            process = self.processes[job_id]
            process.terminate()

            # Wait for termination (with timeout)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

            # Update job status
            job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
            if job:
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()
                db.commit()

            return True

        except Exception as e:
            print(f"Error stopping training: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def _get_training_python(self) -> str:
        """
        Get Python executable for training environment.

        Returns:
            Absolute path to Python executable
        """
        # For MVP, we assume training venv is at mvp/training/venv
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../mvp/backend
        mvp_dir = os.path.dirname(backend_dir)  # .../mvp
        project_root = os.path.dirname(mvp_dir)  # .../mvp-vision-ai-platform

        # Get absolute path to training python
        training_venv_python = os.path.join(project_root, "mvp", "training", "venv", "Scripts", "python.exe")

        if os.path.exists(training_venv_python):
            return training_venv_python

        # Fallback to system python
        return "python"
