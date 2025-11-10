"""Training Logger for Platform SDK.

This module provides status update functionality to Backend API.
Metrics and logs are handled by MLflow and Loki respectively.
"""

import os
import requests
from datetime import datetime
from typing import Optional
import warnings


class TrainingLogger:
    """
    Send status updates to Backend API.

    Metrics are logged to MLflow (via TrainingCallbacks).
    Logs are collected by Promtail/Loki (via stdout).
    This class only handles job status updates.

    Usage:
        logger = TrainingLogger(job_id=1)
        logger.update_status("running")
        logger.update_status("completed")
    """

    def __init__(
        self,
        job_id: int,
        backend_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Training Logger.

        Args:
            job_id: Training job ID
            backend_url: Backend API base URL (defaults to BACKEND_API_URL env var)
            auth_token: Internal auth token (defaults to INTERNAL_AUTH_TOKEN env var)
            enabled: Enable/disable logging (useful for testing)
        """
        self.job_id = job_id
        self.enabled = enabled

        # Get Backend URL from env or parameter
        self.backend_url = backend_url or os.environ.get("BACKEND_API_URL")
        if not self.backend_url:
            warnings.warn(
                "BACKEND_API_URL not set. Status updates disabled.",
                UserWarning
            )
            self.enabled = False

        # Get auth token from env or parameter
        self.auth_token = auth_token or os.environ.get("INTERNAL_AUTH_TOKEN")
        if not self.auth_token and self.enabled:
            warnings.warn(
                "INTERNAL_AUTH_TOKEN not set. Status updates may fail.",
                UserWarning
            )

        # Request headers
        self.headers = {
            "Content-Type": "application/json",
            "X-Internal-Auth": self.auth_token or ""
        }

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Timeout for API calls (fail fast)
        self.timeout = 5

    def update_status(
        self,
        status: str,
        error: Optional[str] = None,
        progress: Optional[float] = None
    ) -> bool:
        """
        Update training job status.

        Args:
            status: Job status (running, completed, failed)
            error: Error message (if failed)
            progress: Training progress 0-100 (optional)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            payload = {
                "status": status,
                "error": error,
                "progress": progress,
                "updated_at": datetime.utcnow().isoformat()
            }

            response = self.session.patch(
                f"{self.backend_url}/internal/training/{self.job_id}/status",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            return True

        except requests.exceptions.Timeout:
            warnings.warn(f"[TrainingLogger] Timeout updating status", UserWarning)
            return False
        except requests.exceptions.RequestException as e:
            warnings.warn(f"[TrainingLogger] Failed to update status: {e}", UserWarning)
            return False
        except Exception as e:
            warnings.warn(f"[TrainingLogger] Unexpected error: {e}", UserWarning)
            return False

    def close(self):
        """Close the underlying HTTP session."""
        if hasattr(self, 'session'):
            self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
