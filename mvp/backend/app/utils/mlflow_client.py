"""MLflow client utility for fetching experiment tracking data."""

import os

from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Metric, Param, RunTag
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowClientWrapper:
    """Wrapper for MLflow client with error handling."""

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """Initialize MLflow client."""
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow is not installed")

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def get_run_by_job_id(self, job_id: int) -> Optional[Any]:
        """
        Get MLflow run for a training job.

        Args:
            job_id: Training job ID

        Returns:
            MLflow run object or None if not found
        """
        try:
            # Experiment name matches the one created in TrainingCallbacks
            experiment_name = f"job_{job_id}"

            # Get experiment
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                return None

            # Search for runs in this experiment (most recent first)
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )

            if not runs:
                return None

            return runs[0]
        except Exception as e:
            print(f"Error getting MLflow run: {e}")
            return None

    def get_run_metrics(self, job_id: int) -> Dict[str, Any]:
        """
        Get metrics for a training job from MLflow.

        Args:
            job_id: Training job ID

        Returns:
            Dictionary containing metrics data
        """
        run = self.get_run_by_job_id(job_id)
        if not run:
            return {
                "found": False,
                "run_id": None,
                "metrics": {},
                "params": {},
                "status": None,
            }

        # Get metric history
        metrics_data = {}
        for metric_key in run.data.metrics.keys():
            try:
                metric_history = self.client.get_metric_history(run.info.run_id, metric_key)
                metrics_data[metric_key] = [
                    {
                        "step": m.step,
                        "value": m.value,
                        "timestamp": m.timestamp,
                    }
                    for m in metric_history
                ]
            except Exception as e:
                print(f"Error getting metric history for {metric_key}: {e}")
                metrics_data[metric_key] = []

        return {
            "found": True,
            "run_id": run.info.run_id,
            "status": run.info.status,
            "metrics": metrics_data,
            "params": dict(run.data.params),
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }

    def get_run_summary(self, job_id: int) -> Dict[str, Any]:
        """
        Get summary information for a training job.

        Args:
            job_id: Training job ID

        Returns:
            Dictionary containing summary data
        """
        run = self.get_run_by_job_id(job_id)
        if not run:
            return {
                "found": False,
                "run_id": None,
            }

        # Get latest metrics (final values)
        latest_metrics = dict(run.data.metrics)

        # Calculate best metrics
        best_val_accuracy = latest_metrics.get("best_val_accuracy")
        best_val_loss = latest_metrics.get("best_val_loss")

        return {
            "found": True,
            "run_id": run.info.run_id,
            "status": run.info.status,
            "latest_metrics": latest_metrics,
            "params": dict(run.data.params),
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
        }


# Global instance
_mlflow_client: Optional[MLflowClientWrapper] = None


def get_mlflow_client() -> MLflowClientWrapper:
    """Get or create MLflow client instance."""
    global _mlflow_client

    if _mlflow_client is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        _mlflow_client = MLflowClientWrapper(tracking_uri=tracking_uri)

    return _mlflow_client
