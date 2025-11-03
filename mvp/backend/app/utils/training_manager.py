"""Training process manager with Docker and subprocess support."""

import json
import os
import subprocess
import threading
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy.orm import Session

from app.db import models
from app.utils.metrics import update_training_metrics, clear_training_metrics, active_training_jobs


class ExecutionMode(Enum):
    """Training execution mode."""
    SUBPROCESS = "subprocess"  # Local subprocess (MVP compatible)
    DOCKER = "docker"          # Docker container
    API = "api"                 # Separate Training Service (Production)


class TrainingManager:
    """Manage training execution (subprocess or Docker)."""

    # Docker image mapping
    IMAGE_MAP = {
        "timm": "vision-platform-timm:latest",
        "ultralytics": "vision-platform-ultralytics:latest",
        "huggingface": "vision-platform-huggingface:latest",
    }

    def __init__(self, db: Session, execution_mode: Optional[ExecutionMode] = None):
        """
        Initialize training manager.

        Args:
            db: Database session
            execution_mode: Execution mode (auto-detect if None)
        """
        self.db = db
        self.processes = {}  # job_id -> process

        # Auto-detect execution mode if not specified
        if execution_mode is None:
            execution_mode = self._detect_execution_mode()

        self.execution_mode = execution_mode
        print(f"[TrainingManager] Execution mode: {self.execution_mode.value}")

    def _detect_execution_mode(self) -> ExecutionMode:
        """
        Auto-detect best execution mode.

        Returns:
            ExecutionMode.DOCKER if Docker available, else SUBPROCESS
        """
        # Check environment variable first
        env_mode = os.getenv("TRAINING_EXECUTION_MODE", "auto").lower()

        if env_mode == "subprocess":
            return ExecutionMode.SUBPROCESS
        elif env_mode == "docker":
            return ExecutionMode.DOCKER
        elif env_mode == "api":
            return ExecutionMode.API

        # Auto-detect: check if Docker is available
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            print("[TrainingManager] Docker detected, using Docker mode")
            return ExecutionMode.DOCKER
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("[TrainingManager] Docker not available, using subprocess mode")
            return ExecutionMode.SUBPROCESS

    def start_training(self, job_id: int, checkpoint_path: Optional[str] = None, resume: bool = False) -> bool:
        """
        Start training using configured execution mode.

        Args:
            job_id: Training job ID
            checkpoint_path: Optional path to checkpoint to load
            resume: If True, resume training from checkpoint (restore optimizer/scheduler state)

        Returns:
            True if training started successfully
        """
        if self.execution_mode == ExecutionMode.SUBPROCESS:
            return self._start_training_subprocess(job_id, checkpoint_path, resume)
        elif self.execution_mode == ExecutionMode.DOCKER:
            return self._start_training_docker(job_id, checkpoint_path, resume)
        elif self.execution_mode == ExecutionMode.API:
            return self._start_training_api(job_id, checkpoint_path, resume)
        else:
            raise ValueError(f"Unsupported execution mode: {self.execution_mode}")

    def _start_training_docker(self, job_id: int, checkpoint_path: Optional[str] = None, resume: bool = False) -> bool:
        """
        Start training in Docker container.

        Args:
            job_id: Training job ID
            checkpoint_path: Optional checkpoint path
            resume: If True, resume from checkpoint

        Returns:
            True if training started successfully
        """
        # Get job from database
        job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if not job:
            return False

        if job.status != "pending":
            return False

        # Select Docker image
        image = self.IMAGE_MAP.get(job.framework)
        if not image:
            job.status = "failed"
            job.error_message = f"No Docker image for framework: {job.framework}"
            self.db.commit()
            return False

        # Get absolute paths
        dataset_path = os.path.abspath(job.dataset_path)
        output_dir = os.path.abspath(job.output_dir)

        # Get DB path for volume mount
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../mvp/backend
        mvp_dir = os.path.dirname(backend_dir)  # .../mvp
        project_root = os.path.dirname(mvp_dir)  # .../mvp-vision-ai-platform
        db_file_path = os.path.join(project_root, "mvp", "data", "db", "vision_platform.db")
        db_dir = os.path.dirname(db_file_path)

        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(db_dir, exist_ok=True)

        # Clean up any existing container with same name
        container_name = f"training-job-{job_id}"
        try:
            print(f"[INFO] Cleaning up existing container: {container_name}")
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10
            )
        except Exception as e:
            print(f"[DEBUG] No existing container to remove: {e}")

        # Build Docker command
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container when done
            "--name", f"training-job-{job_id}",
        ]

        # Add GPU support if available (Linux only)
        if os.name != 'nt':  # Not Windows
            docker_cmd.extend(["--gpus", "all"])

        # Volume mounts
        docker_cmd.extend([
            "-v", f"{dataset_path}:/workspace/dataset:ro",  # Read-only
            "-v", f"{output_dir}:/workspace/output:rw",     # Read-write
            "-v", f"{db_file_path}:/opt/data/db/vision_platform.db:rw",  # Database file
        ])

        # Environment variables
        docker_cmd.extend([
            "-e", f"JOB_ID={job_id}",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", "MLFLOW_TRACKING_URI=http://localhost:5000",  # Use host's MLflow server (host network mode)
            "-e", "DATABASE_URL=sqlite:////opt/data/db/vision_platform.db",  # Database path in container
        ])

        # Network (use host for MLflow tracking)
        docker_cmd.extend(["--network", "host"])

        # Image
        docker_cmd.append(image)

        # Training command
        docker_cmd.extend([
            "python", "/opt/vision-platform/train.py",
            "--framework", job.framework,
            "--task_type", job.task_type,
            "--model_name", job.model_name,
            "--dataset_path", "/workspace/dataset",
            "--dataset_format", job.dataset_format,
            "--output_dir", "/workspace/output",
            "--epochs", str(job.epochs),
            "--batch_size", str(job.batch_size),
            "--learning_rate", str(job.learning_rate),
            "--job_id", str(job_id),
        ])

        # Add num_classes if set
        if job.num_classes is not None:
            docker_cmd.extend(["--num_classes", str(job.num_classes)])

        # Add checkpoint args
        if checkpoint_path:
            docker_cmd.extend(["--checkpoint_path", checkpoint_path])
            if resume:
                docker_cmd.append("--resume")

        try:
            print(f"[DEBUG] Docker command: {' '.join(docker_cmd)}")

            # Prepare environment
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            # Start container
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
                env=env,
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
                model_name=job.model_name,
                dataset_name=os.path.basename(job.dataset_path),
                status="running",
            )

            # Start monitoring thread (same as subprocess)
            monitor_thread = threading.Thread(
                target=self._monitor_training,
                args=(job_id, process),
                daemon=True,
            )
            monitor_thread.start()

            return True

        except Exception as e:
            job.status = "failed"
            job.error_message = f"Failed to start Docker training: {str(e)}"
            self.db.commit()
            return False

    def _start_training_subprocess(self, job_id: int, checkpoint_path: Optional[str] = None, resume: bool = False) -> bool:
        """
        Start training using subprocess (existing MVP implementation).

        Args:
            job_id: Training job ID
            checkpoint_path: Optional checkpoint path
            resume: If True, resume from checkpoint

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
            "--framework", job.framework,
            "--task_type", job.task_type,
            "--model_name", job.model_name,
            "--dataset_path", job.dataset_path,
            "--dataset_format", job.dataset_format,
            "--output_dir", job.output_dir,
            "--epochs", str(job.epochs),
            "--batch_size", str(job.batch_size),
            "--learning_rate", str(job.learning_rate),
            "--job_id", str(job.id),
        ]

        # Add num_classes only if it's set (required for classification tasks)
        if job.num_classes is not None:
            cmd.extend(["--num_classes", str(job.num_classes)])

        # Add checkpoint parameters if provided
        if checkpoint_path:
            cmd.extend(["--checkpoint_path", checkpoint_path])
            if resume:
                cmd.append("--resume")

        try:
            # Debug logging
            print(f"[DEBUG] Project root: {project_root}")
            print(f"[DEBUG] Train script: {train_script}")
            print(f"[DEBUG] Training python: {training_python}")
            print(f"[DEBUG] Command: {' '.join(cmd)}")
            print(f"[DEBUG] CWD: {project_root}")

            # Prepare environment with unbuffered Python output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            # Add -u flag to python command for unbuffered output
            if training_python.endswith('python.exe') or training_python.endswith('python'):
                cmd_with_unbuffered = [training_python, '-u'] + cmd[1:]
            else:
                cmd_with_unbuffered = cmd

            print(f"[DEBUG] Command with unbuffered: {' '.join(cmd_with_unbuffered)}")

            # Start subprocess (run from project root)
            process = subprocess.Popen(
                cmd_with_unbuffered,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',  # Force UTF-8 encoding to handle emojis
                errors='replace',  # Replace invalid characters instead of crashing
                cwd=project_root,  # Set working directory to project root
                env=env,  # Use environment with PYTHONUNBUFFERED
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
                model_name=job.model_name,
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
                        model_name=job.model_name,
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
                        model_name=job.model_name,
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
                        model_name=job.model_name,
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

                print(f"[DEBUG] Parsed metrics: {metrics_data}")

                # Use val_loss/val_accuracy if available, otherwise use train_loss/train_accuracy
                loss = metrics_data.get("val_loss") or metrics_data.get("train_loss")
                accuracy = metrics_data.get("val_accuracy") or metrics_data.get("train_accuracy")

                print(f"[DEBUG] Loss: {loss}, Accuracy: {accuracy}")

                # Save to database (only if we have a loss value)
                if loss is not None:
                    print(f"[DEBUG] Saving metric to database for job {job_id}, epoch {metrics_data.get('epoch')}")
                    metric = models.TrainingMetric(
                        job_id=job_id,
                        epoch=metrics_data.get("epoch"),
                        loss=loss,
                        accuracy=accuracy,
                        learning_rate=metrics_data.get("learning_rate"),
                        checkpoint_path=metrics_data.get("checkpoint_path"),
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
                    print(f"[DEBUG] Metric saved successfully!")
                else:
                    print(f"[DEBUG] Skipping metric save: loss is None")

                # Export to Prometheus (if we have metrics to export)
                if loss is not None:
                    # Get job info for model_name
                    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()

                    # Accuracy might be in 0-1 range, convert to percentage
                    acc_percentage = accuracy * 100 if accuracy and accuracy <= 1.0 else (accuracy or 0)

                    update_training_metrics(
                        job_id=job_id,
                        model_name=job.model_name if job else "unknown",
                        loss=loss,
                        accuracy=acc_percentage,
                        epoch=metrics_data.get("epoch"),
                    )

            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON decode error: {e}")
                print(f"[DEBUG] Tried to parse: {line}")
            except Exception as e:
                print(f"[DEBUG] Error parsing/saving metrics: {e}")
                import traceback
                traceback.print_exc()
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

            # Stop Docker container first
            container_name = f"training-job-{job_id}"
            try:
                print(f"[INFO] Stopping Docker container: {container_name}")
                subprocess.run(
                    ["docker", "stop", container_name],
                    capture_output=True,
                    timeout=30
                )
                print(f"[INFO] Removing Docker container: {container_name}")
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    timeout=10
                )
            except Exception as e:
                print(f"[WARNING] Failed to stop/remove Docker container: {e}")

            # Terminate the subprocess
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
