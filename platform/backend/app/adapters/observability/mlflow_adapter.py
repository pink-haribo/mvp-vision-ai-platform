"""
MLflowAdapter for Phase 13

Adapter for MLflow observability backend.
Provides experiment tracking, metrics logging, and artifact management via MLflow.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .base import ObservabilityAdapter, MetricsResult

# Optional MLflow import
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


class MLflowAdapter(ObservabilityAdapter):
    """
    MLflow-based observability adapter.

    Provides experiment tracking, metrics logging, and artifact management via MLflow.

    Design Principles:
    - Direct MLflow SDK usage for metrics logging
    - Graceful degradation: Failures don't stop training
    - External service: Requires MLflow tracking server
    """

    # Default experiment name for vision training jobs
    DEFAULT_EXPERIMENT_NAME = "vision-training"

    def __init__(self):
        """
        Initialize MLflowAdapter.
        """
        super().__init__(name="mlflow")
        self.client: Optional[MlflowClient] = None
        self.tracking_uri: Optional[str] = None
        self._active_runs: Dict[str, str] = {}  # experiment_id -> run_id mapping

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize MLflow client.

        Args:
            config: Configuration dictionary containing:
                - tracking_uri: MLflow tracking server URL (default: http://localhost:5000)

        Raises:
            Exception: If MLflow initialization fails
        """
        if not MLFLOW_AVAILABLE:
            logger.error("[MLflowAdapter] MLflow package not installed")
            self.disable()
            raise RuntimeError("MLflow package not installed")

        try:
            self.tracking_uri = config.get(
                "tracking_uri",
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )

            # Set tracking URI globally
            mlflow.set_tracking_uri(self.tracking_uri)

            # Create client for API operations
            self.client = MlflowClient(tracking_uri=self.tracking_uri)

            # Test connection
            try:
                import requests
                response = requests.get(f"{self.tracking_uri}/health", timeout=5)
                if response.status_code != 200:
                    raise RuntimeError(f"MLflow server not healthy: {response.status_code}")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Cannot connect to MLflow server: {e}")

            logger.info(f"[MLflowAdapter] Initialized successfully (tracking_uri={self.tracking_uri})")

        except Exception as e:
            logger.error(f"[MLflowAdapter] Initialization failed: {e}")
            self.disable()
            raise

    def create_experiment(
        self,
        job_id: int,
        project_name: str,
        experiment_name: str,
        tags: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create MLflow experiment and run.

        Args:
            job_id: Training job ID
            project_name: Project name (used as experiment name prefix)
            experiment_name: Human-readable experiment name
            tags: Optional tags for the run
            hyperparameters: Hyperparameters to log

        Returns:
            run_id: MLflow run ID

        Raises:
            Exception: If experiment/run creation fails
        """
        if not self.client:
            raise RuntimeError("MLflowAdapter not initialized")

        try:
            # Use consistent experiment name for all vision training jobs
            mlflow_experiment_name = self.DEFAULT_EXPERIMENT_NAME

            # Get or create experiment
            experiment = self.client.get_experiment_by_name(mlflow_experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(mlflow_experiment_name)
                logger.info(f"[MLflowAdapter] Created experiment '{mlflow_experiment_name}'")
            else:
                experiment_id = experiment.experiment_id

            # Create run with job-specific name
            run_name = f"job-{job_id}"
            run = self.client.create_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags={
                    "job_id": str(job_id),
                    "project_name": project_name,
                    "experiment_name": experiment_name,
                }
            )
            run_id = run.info.run_id

            # Add custom tags if provided
            if tags:
                for i, tag in enumerate(tags):
                    self.client.set_tag(run_id, f"custom_tag_{i}", tag)

            # Log hyperparameters
            if hyperparameters:
                for key, value in hyperparameters.items():
                    # MLflow params must be strings
                    self.client.log_param(run_id, key, str(value))

            # Store run_id for this experiment
            self._active_runs[run_id] = run_id

            logger.info(f"[MLflowAdapter] Created run {run_id} for job {job_id}")
            return run_id

        except Exception as e:
            logger.error(f"[MLflowAdapter] Failed to create experiment: {e}")
            raise

    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: int
    ) -> None:
        """
        Log metrics to MLflow run.

        Args:
            experiment_id: MLflow run ID
            metrics: Dictionary of metric names to values
            step: Training step/epoch number

        Raises:
            Exception: If logging fails
        """
        if not self.client:
            raise RuntimeError("MLflowAdapter not initialized")

        run_id = experiment_id  # experiment_id is actually run_id for MLflow

        try:
            # Log each metric
            for metric_name, value in metrics.items():
                if value is not None:
                    # Sanitize metric name (MLflow doesn't allow certain characters)
                    safe_name = metric_name.replace("/", "_").replace(" ", "_")
                    self.client.log_metric(run_id, safe_name, float(value), step=step)

            logger.debug(
                f"[MLflowAdapter] Logged {len(metrics)} metrics for run {run_id}, step {step}"
            )

        except Exception as e:
            logger.error(f"[MLflowAdapter] Failed to log metrics: {e}")
            raise

    def log_hyperparameters(
        self,
        experiment_id: str,
        params: Dict[str, Any]
    ) -> None:
        """
        Log hyperparameters to MLflow run.

        Args:
            experiment_id: MLflow run ID
            params: Hyperparameters dictionary

        Raises:
            Exception: If logging fails
        """
        if not self.client:
            raise RuntimeError("MLflowAdapter not initialized")

        run_id = experiment_id

        try:
            for key, value in params.items():
                # MLflow params must be strings
                self.client.log_param(run_id, key, str(value))

            logger.info(f"[MLflowAdapter] Logged hyperparameters for run {run_id}")

        except Exception as e:
            logger.error(f"[MLflowAdapter] Failed to log hyperparameters: {e}")
            raise

    def get_metrics(
        self,
        experiment_id: str,
        metric_names: Optional[List[str]] = None
    ) -> MetricsResult:
        """
        Retrieve metrics from MLflow run.

        Args:
            experiment_id: MLflow run ID
            metric_names: Optional list of specific metrics

        Returns:
            MetricsResult: Metrics data

        Raises:
            Exception: If retrieval fails
        """
        if not self.client:
            raise RuntimeError("MLflowAdapter not initialized")

        run_id = experiment_id

        try:
            run = self.client.get_run(run_id)

            # Get all metric keys
            all_metrics = run.data.metrics
            if metric_names:
                metric_keys = [k for k in all_metrics.keys() if k in metric_names]
            else:
                metric_keys = list(all_metrics.keys())

            # Get metric history
            metrics_dict: Dict[str, List[float]] = {}
            all_steps: set = set()

            for metric_key in metric_keys:
                history = self.client.get_metric_history(run_id, metric_key)
                metrics_dict[metric_key] = [m.value for m in history]
                all_steps.update(m.step for m in history)

            # Sort steps
            steps = sorted(all_steps)

            logger.debug(
                f"[MLflowAdapter] Retrieved metrics for run {run_id}: "
                f"{len(metrics_dict)} metrics, {len(steps)} steps"
            )

            return MetricsResult(
                metrics=metrics_dict,
                steps=steps,
                total_count=len(steps)
            )

        except Exception as e:
            logger.error(f"[MLflowAdapter] Failed to retrieve metrics: {e}")
            raise

    def finalize_experiment(
        self,
        experiment_id: str,
        status: str,
        final_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Mark MLflow run as completed/failed.

        Args:
            experiment_id: MLflow run ID
            status: Final status ("completed", "failed", "stopped")
            final_metrics: Optional final metrics to log

        Raises:
            Exception: If finalization fails
        """
        if not self.client:
            raise RuntimeError("MLflowAdapter not initialized")

        run_id = experiment_id

        try:
            # Log final metrics if provided
            if final_metrics:
                # Get current step (max step from existing metrics)
                run = self.client.get_run(run_id)
                max_step = 0
                for metric_key in run.data.metrics.keys():
                    history = self.client.get_metric_history(run_id, metric_key)
                    if history:
                        max_step = max(max_step, max(m.step for m in history))

                # Log final metrics at max_step + 1
                self.log_metrics(run_id, final_metrics, step=max_step + 1)

            # Map status to MLflow status
            mlflow_status_map = {
                "completed": "FINISHED",
                "failed": "FAILED",
                "stopped": "KILLED",
            }
            mlflow_status = mlflow_status_map.get(status, "FINISHED")

            # End run with status
            self.client.set_terminated(run_id, status=mlflow_status)

            # Remove from active runs
            if run_id in self._active_runs:
                del self._active_runs[run_id]

            logger.info(
                f"[MLflowAdapter] Finalized run {run_id} with status '{status}' "
                f"(MLflow status: {mlflow_status})"
            )

        except Exception as e:
            logger.error(f"[MLflowAdapter] Failed to finalize experiment: {e}")
            # Don't raise - finalization failure shouldn't stop training

    def get_experiment_url(self, experiment_id: str) -> Optional[str]:
        """
        Get MLflow Web UI URL for the run.

        Args:
            experiment_id: MLflow run ID

        Returns:
            URL string for MLflow run UI
        """
        if not self.client or not self.tracking_uri:
            return None

        try:
            run = self.client.get_run(experiment_id)
            exp_id = run.info.experiment_id

            # MLflow UI URL format: {tracking_uri}/#/experiments/{exp_id}/runs/{run_id}
            url = f"{self.tracking_uri}/#/experiments/{exp_id}/runs/{experiment_id}"

            logger.debug(f"[MLflowAdapter] Experiment URL: {url}")
            return url

        except Exception as e:
            logger.error(f"[MLflowAdapter] Failed to get experiment URL: {e}")
            return None
