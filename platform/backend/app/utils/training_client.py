"""
Training Service HTTP Client

Communicates with framework-specific Training Services for executing training jobs.
"""

import os
import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TrainingServiceClient:
    """Client for Training Service API with framework-specific routing."""

    # Framework to service URL mapping
    FRAMEWORK_SERVICES = {
        "timm": "TIMM_SERVICE_URL",
        "ultralytics": "ULTRALYTICS_SERVICE_URL",
        "huggingface": "HUGGINGFACE_SERVICE_URL",
    }

    def __init__(self, framework: Optional[str] = None):
        """
        Initialize Training Service client.

        Args:
            framework: Framework name (timm, ultralytics, huggingface)
                      If None, uses default TRAINING_SERVICE_URL
        """
        # Get framework-specific URL or fallback to default
        if framework and framework in self.FRAMEWORK_SERVICES:
            env_var = self.FRAMEWORK_SERVICES[framework]
            self.base_url = os.getenv(env_var)

            if not self.base_url:
                logger.warning(f"[TrainingClient] {env_var} not set, using default")
                self.base_url = os.getenv("TRAINING_SERVICE_URL", "http://localhost:8001")
        else:
            self.base_url = os.getenv("TRAINING_SERVICE_URL", "http://localhost:8001")

        logger.info(f"[TrainingClient] Framework: {framework}, URL: {self.base_url}")

    def start_training(self, job_config: Dict[str, Any]) -> bool:
        """
        Start training job on Training Service.

        Args:
            job_config: Training configuration dict

        Returns:
            True if training started successfully

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        try:
            url = f"{self.base_url}/training/start"
            logger.info(f"[TrainingClient] Sending training request to {url}")
            logger.debug(f"[TrainingClient] Job config: {job_config}")

            response = requests.post(
                url,
                json=job_config,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            logger.info(f"[TrainingClient] Training started: {result}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"[TrainingClient] Failed to start training: {e}")
            raise

    def get_job_status(self, job_id: int) -> Dict[str, Any]:
        """
        Get training job status from Training Service.

        Args:
            job_id: Job ID

        Returns:
            Job status dict

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        try:
            url = f"{self.base_url}/training/status/{job_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"[TrainingClient] Failed to get job status: {e}")
            raise

    def stop_training(self, job_id: int) -> bool:
        """
        Stop training job on Training Service.

        Args:
            job_id: Job ID to stop

        Returns:
            True if training stopped successfully
        """
        try:
            url = f"{self.base_url}/training/stop/{job_id}"
            logger.info(f"[TrainingClient] Sending stop request to {url}")

            response = requests.post(url, timeout=10)
            response.raise_for_status()

            result = response.json()
            logger.info(f"[TrainingClient] Training stopped: {result}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"[TrainingClient] Failed to stop training: {e}")
            return False

    def health_check(self) -> bool:
        """
        Check if Training Service is healthy.

        Returns:
            True if service is healthy
        """
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            result = response.json()
            return result.get("status") == "healthy"

        except requests.exceptions.RequestException:
            return False
