"""Training Manager Abstraction.

Phase 12: Temporal Orchestration & Backend Modernization

This module provides an abstract base class for training execution,
allowing seamless switching between subprocess (Tier 0) and Kubernetes (Tier 1+)
based on the TRAINING_MODE environment variable.

Architecture:
- TrainingManager: Abstract base class defining the interface
- SubprocessTrainingManager: Executes training in local subprocess (Tier 0)
- KubernetesTrainingManager: Executes training in K8s Job (Tier 1+)
- get_training_manager(): Factory function selecting implementation based on TRAINING_MODE
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TrainingManager(ABC):
    """
    Abstract base class for training execution.

    Implementations:
    - SubprocessTrainingManager: Local subprocess execution (Tier 0)
    - KubernetesTrainingManager: K8s Job execution (Tier 1+)

    This abstraction allows the same code to work across different deployment tiers
    by simply changing the TRAINING_MODE environment variable.
    """

    @abstractmethod
    async def start_training(
        self,
        job_id: int,
        framework: str,
        model_name: str,
        dataset_s3_uri: str,
        callback_url: str,
        config: Dict[str, Any],
        snapshot_id: Optional[str] = None,
        dataset_version_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a training job.

        Args:
            job_id: TrainingJob ID
            framework: Framework name (ultralytics, timm, huggingface)
            model_name: Model name to train
            dataset_s3_uri: S3 URI of dataset
            callback_url: Backend API callback URL for status updates
            config: Training configuration dictionary
            snapshot_id: Dataset snapshot ID (for caching)
            dataset_version_hash: Dataset version hash (for caching)

        Returns:
            Dict containing execution metadata (k8s_job_name, status, etc.)

        Raises:
            RuntimeError: If training fails to start
            FileNotFoundError: If required files/resources not found
        """
        pass

    @abstractmethod
    def stop_training(self, job_id: int) -> bool:
        """
        Stop a running training job.

        Args:
            job_id: TrainingJob ID

        Returns:
            True if successfully stopped, False otherwise
        """
        pass

    @abstractmethod
    def get_training_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """
        Get current status of a training job.

        Args:
            job_id: TrainingJob ID

        Returns:
            Dict with status information, or None if job not found

        Example return:
            {
                "job_id": 123,
                "status": "running",  # pending, running, completed, failed
                "k8s_job_name": "training-123-abc123",
                "started_at": "2025-01-27T10:00:00Z",
            }
        """
        pass

    @abstractmethod
    async def start_evaluation(
        self,
        test_run_id: int,
        training_job_id: Optional[int],
        framework: str,
        checkpoint_s3_uri: str,
        dataset_s3_uri: str,
        callback_url: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Start model evaluation.

        Args:
            test_run_id: Test run ID
            training_job_id: Original training job ID (optional)
            framework: Framework name (ultralytics, timm, huggingface)
            checkpoint_s3_uri: S3 URI to model checkpoint
            dataset_s3_uri: S3 URI to test dataset
            callback_url: Backend API callback URL
            config: Evaluation configuration dictionary

        Returns:
            Dict containing execution metadata

        Raises:
            RuntimeError: If evaluation fails to start
        """
        pass

    @abstractmethod
    async def start_inference(
        self,
        inference_job_id: int,
        training_job_id: Optional[int],
        framework: str,
        checkpoint_s3_uri: str,
        images_s3_uri: str,
        callback_url: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run inference with trained model.

        Args:
            inference_job_id: Inference job ID
            training_job_id: Original training job ID (optional)
            framework: Framework name (ultralytics, timm, huggingface)
            checkpoint_s3_uri: S3 URI to model checkpoint
            images_s3_uri: S3 URI to input images
            callback_url: Backend API callback URL
            config: Inference configuration dictionary

        Returns:
            Dict containing execution metadata

        Raises:
            RuntimeError: If inference fails to start
        """
        pass

    @abstractmethod
    async def start_export(
        self,
        export_job_id: int,
        training_job_id: int,
        framework: str,
        checkpoint_s3_uri: str,
        export_format: str,
        callback_url: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Export trained model to different format.

        Args:
            export_job_id: Export job ID
            training_job_id: Original training job ID
            framework: Framework name (ultralytics, timm, huggingface)
            checkpoint_s3_uri: S3 URI to model checkpoint
            export_format: Target format (onnx, tensorrt, coreml, tflite, etc.)
            callback_url: Backend API callback URL
            config: Export configuration dictionary

        Returns:
            Dict containing execution metadata

        Raises:
            RuntimeError: If export fails to start
        """
        pass

    @abstractmethod
    def cleanup_resources(self, job_id: int) -> None:
        """
        Clean up resources for a training job.

        This includes:
        - Temporary files
        - Running processes/pods
        - GPU allocations

        Args:
            job_id: TrainingJob ID
        """
        pass


def get_training_manager() -> TrainingManager:
    """
    Factory function to get TrainingManager implementation.

    Returns appropriate implementation based on TRAINING_MODE environment variable:
    - "subprocess": SubprocessTrainingManager (Tier 0)
    - "kubernetes": KubernetesTrainingManager (Tier 1+)

    Returns:
        TrainingManager instance

    Raises:
        ValueError: If TRAINING_MODE is invalid

    Example:
        >>> from app.core.config import settings
        >>> manager = get_training_manager()
        >>> # Returns SubprocessTrainingManager if settings.TRAINING_MODE == "subprocess"
        >>> # Returns KubernetesTrainingManager if settings.TRAINING_MODE == "kubernetes"
    """
    from app.core.config import settings

    mode = settings.TRAINING_MODE.lower()

    if mode == "subprocess":
        from app.core.training_managers.subprocess_manager import SubprocessTrainingManager
        logger.info("[TrainingManager] Using SubprocessTrainingManager (Tier 0)")
        return SubprocessTrainingManager()

    elif mode == "kubernetes":
        from app.core.training_managers.kubernetes_manager import KubernetesTrainingManager
        logger.info("[TrainingManager] Using KubernetesTrainingManager (Tier 1+)")
        return KubernetesTrainingManager()

    else:
        raise ValueError(
            f"Invalid TRAINING_MODE: {mode}. "
            f"Expected 'subprocess' or 'kubernetes'"
        )
