"""Training process manager using subprocess."""

import json
import os
import subprocess
import threading
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.db import models


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

        # Build command
        cmd = [
            training_python,
            "./mvp/training/train.py",
            "--dataset_path", job.dataset_path,
            "--output_dir", job.output_dir,
            "--num_classes", str(job.num_classes),
            "--epochs", str(job.epochs),
            "--batch_size", str(job.batch_size),
            "--learning_rate", str(job.learning_rate),
            "--job_id", str(job.id),
        ]

        try:
            # Start subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Store process
            self.processes[job_id] = process

            # Update job status
            job.status = "running"
            job.started_at = datetime.utcnow()
            job.process_id = process.pid
            self.db.commit()

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
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                # Parse metrics from stdout
                self._parse_and_save_metrics(job_id, line)

            # Wait for process to complete
            process.wait()

            # Update job status
            job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
            if job:
                if process.returncode == 0:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()

                    # Get final accuracy from latest metric
                    latest_metric = (
                        self.db.query(models.TrainingMetric)
                        .filter(models.TrainingMetric.job_id == job_id)
                        .order_by(models.TrainingMetric.epoch.desc())
                        .first()
                    )
                    if latest_metric:
                        job.final_accuracy = latest_metric.accuracy
                        job.best_checkpoint_path = os.path.join(job.output_dir, "best_model.pth")
                else:
                    job.status = "failed"
                    job.error_message = f"Training process exited with code {process.returncode}"
                    job.completed_at = datetime.utcnow()

                self.db.commit()

        except Exception as e:
            job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = f"Monitoring error: {str(e)}"
                job.completed_at = datetime.utcnow()
                self.db.commit()

        finally:
            # Remove from active processes
            if job_id in self.processes:
                del self.processes[job_id]

    def _parse_and_save_metrics(self, job_id: int, line: str):
        """
        Parse stdout line and save metrics if found.

        Args:
            job_id: Training job ID
            line: Output line from training process
        """
        # Look for [METRICS] tag
        if "[METRICS]" in line:
            try:
                # Extract JSON from line
                json_start = line.find("{")
                json_str = line[json_start:]
                metrics_data = json.loads(json_str)

                # Save to database
                metric = models.TrainingMetric(
                    job_id=job_id,
                    epoch=metrics_data.get("epoch"),
                    loss=metrics_data.get("val_loss"),
                    accuracy=metrics_data.get("val_accuracy"),
                    learning_rate=metrics_data.get("learning_rate"),
                    extra_metrics={
                        "train_loss": metrics_data.get("train_loss"),
                        "train_accuracy": metrics_data.get("train_accuracy"),
                        "epoch_time": metrics_data.get("epoch_time"),
                    },
                )
                self.db.add(metric)
                self.db.commit()

            except json.JSONDecodeError:
                pass  # Ignore lines that aren't valid JSON
            except Exception as e:
                print(f"Error parsing metrics: {e}")

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

        try:
            process = self.processes[job_id]
            process.terminate()

            # Wait for termination (with timeout)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

            # Update job status
            job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
            if job:
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()
                self.db.commit()

            return True

        except Exception:
            return False

    def _get_training_python(self) -> str:
        """
        Get Python executable for training environment.

        Returns:
            Path to Python executable
        """
        # For MVP, we assume training venv is at mvp/training/venv
        # In production, this would be more sophisticated
        training_venv_python = "./mvp/training/venv/Scripts/python.exe"

        if os.path.exists(training_venv_python):
            return training_venv_python

        # Fallback to system python
        return "python"
