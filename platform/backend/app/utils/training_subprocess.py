"""
Training Subprocess Manager

Manages training jobs by executing Training Service CLI via subprocess.
This approach maintains consistency between local dev and K8s Job production.

Architecture:
- Backend executes Training Service's train.py using its dedicated venv
- Training Service has its own dependencies (ultralytics, torch, etc.)
- Same execution model for both local subprocess and K8s Job
"""

import asyncio
import io
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TrainingSubprocessManager:
    """
    Manages training subprocesses for local development.

    Each framework (ultralytics, timm, huggingface) has its own Training Service
    with dedicated venv and dependencies.
    """

    def __init__(self):
        # Get Trainers base directory
        self.backend_dir = Path(__file__).parent.parent.parent
        self.platform_dir = self.backend_dir.parent
        self.trainers_dir = self.platform_dir / "trainers"

        logger.info(f"[TrainingSubprocess] Trainers directory: {self.trainers_dir}")

        # Store running processes
        self.processes: Dict[int, subprocess.Popen] = {}

    def get_python_executable(self, framework: str) -> Path:
        """
        Get Python executable for a specific framework's Trainer.

        Args:
            framework: Framework name (ultralytics, timm, huggingface)

        Returns:
            Path to Python executable in Trainer's venv

        Raises:
            FileNotFoundError: If Trainer venv not found
        """
        trainer_dir = self.trainers_dir / framework

        # Check Windows path first
        python_exe = trainer_dir / "venv" / "Scripts" / "python.exe"
        if python_exe.exists():
            return python_exe

        # Check Linux/macOS path
        python_exe = trainer_dir / "venv" / "bin" / "python"
        if python_exe.exists():
            return python_exe

        raise FileNotFoundError(
            f"Trainer venv not found for {framework}. "
            f"Expected at: {trainer_dir / 'venv'}. "
            f"Please create venv: cd {trainer_dir} && python -m venv venv && venv/Scripts/pip install -r requirements.txt"
        )

    def get_trainer_directory(self, framework: str) -> Path:
        """Get Trainer directory for a framework."""
        trainer_dir = self.trainers_dir / framework

        if not trainer_dir.exists():
            raise FileNotFoundError(
                f"Trainer directory not found for {framework}: {trainer_dir}"
            )

        return trainer_dir

    async def start_training(
        self,
        job_id: int,
        framework: str,
        model_name: str,
        dataset_s3_uri: str,
        callback_url: str,
        config: Dict[str, Any],
    ) -> subprocess.Popen:
        """
        Start training subprocess.

        Args:
            job_id: Training job ID
            framework: Framework name (ultralytics, timm, huggingface)
            model_name: Model name to train
            dataset_s3_uri: S3 URI of dataset
            callback_url: Backend API callback URL
            config: Training configuration dictionary

        Returns:
            subprocess.Popen instance

        Raises:
            FileNotFoundError: If Training Service not found
            subprocess.CalledProcessError: If subprocess fails to start
        """
        try:
            # Get paths
            python_exe = self.get_python_executable(framework)
            trainer_dir = self.get_trainer_directory(framework)

            logger.info(f"[TrainingSubprocess] Starting job {job_id}")
            logger.info(f"[TrainingSubprocess]   Framework: {framework}")
            logger.info(f"[TrainingSubprocess]   Trainer dir: {trainer_dir}")
            logger.info(f"[TrainingSubprocess]   Python: {python_exe}")
            logger.info(f"[TrainingSubprocess]   Model: {model_name}")

            # Prepare command
            cmd = [
                str(python_exe),
                "train.py",
                "--job-id", str(job_id),
                "--model-name", model_name,
                "--dataset-s3-uri", dataset_s3_uri,
                "--callback-url", callback_url,
                "--config", json.dumps(config),
                "--log-level", "INFO",
            ]

            logger.info(f"[TrainingSubprocess] Command: {' '.join(cmd)}")

            # Prepare environment (inherit current env + Trainer will load its own .env)
            env = os.environ.copy()

            # Start subprocess in background
            process = subprocess.Popen(
                cmd,
                cwd=str(trainer_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",  # Replace decode errors instead of failing
            )

            # Store process
            self.processes[job_id] = process

            logger.info(f"[TrainingSubprocess] Job {job_id} started (PID: {process.pid})")

            # Start async log monitoring (non-blocking)
            asyncio.create_task(self._monitor_process_logs(job_id, process))

            return process

        except FileNotFoundError as e:
            logger.error(f"[TrainingSubprocess] Setup error: {e}")
            raise
        except Exception as e:
            logger.error(f"[TrainingSubprocess] Failed to start job {job_id}: {e}")
            raise

    async def _monitor_process_logs(self, job_id: int, process: subprocess.Popen):
        """
        Monitor subprocess logs in background.

        This is a simple implementation that logs to console.
        For production, you should store logs in database or forward to Loki.
        """
        try:
            # Wrap stdout/stderr with explicit UTF-8 encoding to avoid Windows cp949 issues
            # Even though we set encoding="utf-8" on Popen, iteration over the stream
            # still uses system default encoding, so we need to wrap it explicitly
            stdout_reader = io.TextIOWrapper(
                process.stdout.buffer,
                encoding='utf-8',
                errors='replace'  # Replace undecodable bytes instead of failing
            )
            stderr_reader = io.TextIOWrapper(
                process.stderr.buffer,
                encoding='utf-8',
                errors='replace'
            )

            # Read stdout
            for line in stdout_reader:
                if line.strip():
                    logger.info(f"[JOB {job_id}] {line.rstrip()}")

            # Read stderr
            for line in stderr_reader:
                if line.strip():
                    logger.error(f"[JOB {job_id}] {line.rstrip()}")

            # Wait for process to complete
            exit_code = process.wait()

            logger.info(f"[TrainingSubprocess] Job {job_id} finished (exit code: {exit_code})")

            # Remove from active processes
            if job_id in self.processes:
                del self.processes[job_id]

        except Exception as e:
            logger.error(f"[TrainingSubprocess] Error monitoring job {job_id}: {e}")

    def stop_training(self, job_id: int) -> bool:
        """
        Stop a running training subprocess.

        Args:
            job_id: Training job ID to stop

        Returns:
            True if stopped successfully, False if not found or already stopped
        """
        try:
            if job_id not in self.processes:
                logger.warning(f"[TrainingSubprocess] Job {job_id} not found in active processes")
                return False

            process = self.processes[job_id]

            if process.poll() is None:
                # Process still running
                logger.info(f"[TrainingSubprocess] Terminating job {job_id} (PID: {process.pid})")
                process.terminate()

                # Wait up to 5 seconds for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"[TrainingSubprocess] Job {job_id} didn't terminate gracefully, killing")
                    process.kill()
                    process.wait()

                logger.info(f"[TrainingSubprocess] Job {job_id} stopped")
            else:
                logger.info(f"[TrainingSubprocess] Job {job_id} already finished")

            # Remove from active processes
            del self.processes[job_id]
            return True

        except Exception as e:
            logger.error(f"[TrainingSubprocess] Error stopping job {job_id}: {e}")
            return False

    def get_process_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """
        Get status of a training subprocess.

        Args:
            job_id: Training job ID

        Returns:
            Status dictionary or None if not found
        """
        if job_id not in self.processes:
            return None

        process = self.processes[job_id]
        exit_code = process.poll()

        return {
            "job_id": job_id,
            "pid": process.pid,
            "running": exit_code is None,
            "exit_code": exit_code,
        }


# Global singleton instance
_training_subprocess_manager: Optional[TrainingSubprocessManager] = None


def get_training_subprocess_manager() -> TrainingSubprocessManager:
    """Get or create global TrainingSubprocessManager instance."""
    global _training_subprocess_manager

    if _training_subprocess_manager is None:
        _training_subprocess_manager = TrainingSubprocessManager()

    return _training_subprocess_manager
