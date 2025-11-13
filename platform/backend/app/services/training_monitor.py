"""
Background Training Job Monitor.

Monitors active Kubernetes training jobs and updates their status in the database.
Sends WebSocket notifications for status changes and metrics updates.
"""

import asyncio
import logging
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import TrainingJob
from app.services.vm_controller import VMController

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Background service that monitors active training jobs.

    Runs as an async task in the backend, polling Kubernetes API
    and updating job status in the database.
    """

    def __init__(
        self,
        poll_interval: int = 10,
        namespace: str = "training",
    ):
        """
        Initialize training monitor.

        Args:
            poll_interval: Seconds between status checks
            namespace: Kubernetes namespace
        """
        self.poll_interval = poll_interval
        self.namespace = namespace
        self.running = False
        self.vm_controller = VMController(namespace=namespace)

        # WebSocket manager (will be injected)
        self.ws_manager = None

        logger.info(f"[TrainingMonitor] Initialized (poll interval: {poll_interval}s)")

    def set_websocket_manager(self, ws_manager):
        """Set WebSocket manager for notifications"""
        self.ws_manager = ws_manager
        logger.info("[TrainingMonitor] WebSocket manager set")

    async def start(self):
        """Start monitoring loop"""
        self.running = True
        logger.info("[TrainingMonitor] Starting monitoring loop...")

        while self.running:
            try:
                await self.check_active_jobs()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"[TrainingMonitor] Error in monitoring loop: {e}")
                import traceback
                traceback.print_exc()
                # Continue running despite errors
                await asyncio.sleep(self.poll_interval)

    async def stop(self):
        """Stop monitoring loop"""
        self.running = False
        logger.info("[TrainingMonitor] Stopping monitoring loop...")

    async def check_active_jobs(self):
        """Check status of all active jobs"""
        db = SessionLocal()

        try:
            # Get all active jobs (pending, running)
            active_jobs = db.query(TrainingJob).filter(
                TrainingJob.status.in_(["pending", "running"]),
                TrainingJob.executor_type == "kubernetes",
            ).all()

            if not active_jobs:
                return

            logger.debug(f"[TrainingMonitor] Checking {len(active_jobs)} active jobs")

            for job in active_jobs:
                await self._check_job(job, db)

        except Exception as e:
            logger.error(f"[TrainingMonitor] Error checking active jobs: {e}")
        finally:
            db.close()

    async def _check_job(self, job: TrainingJob, db: Session):
        """
        Check status of a single job.

        Args:
            job: TrainingJob model
            db: Database session
        """
        if not job.execution_id:
            logger.warning(f"[TrainingMonitor] Job {job.id} has no execution_id")
            return

        try:
            # Query Kubernetes for job status
            k8s_status = self.vm_controller.get_job_status(job.execution_id)

            logger.debug(f"[TrainingMonitor] Job {job.id}: {job.status} -> {k8s_status}")

            # Update database if status changed
            if k8s_status != job.status:
                old_status = job.status
                job.status = k8s_status

                # Update timestamps
                if k8s_status == "running" and not job.started_at:
                    job.started_at = datetime.utcnow()
                elif k8s_status in ["completed", "failed"]:
                    job.completed_at = datetime.utcnow()

                db.commit()

                logger.info(f"[TrainingMonitor] Job {job.id} status changed: {old_status} -> {k8s_status}")

                # Send WebSocket notification
                await self._send_status_notification(job, old_status, k8s_status)

            # Collect metrics if job is running
            if k8s_status == "running":
                await self._collect_metrics(job)

        except Exception as e:
            logger.error(f"[TrainingMonitor] Error checking job {job.id}: {e}")

    async def _send_status_notification(
        self,
        job: TrainingJob,
        old_status: str,
        new_status: str,
    ):
        """Send WebSocket notification for status change"""
        if not self.ws_manager:
            return

        try:
            message = {
                "type": "training_status_change",
                "job_id": job.id,
                "old_status": old_status,
                "new_status": new_status,
                "timestamp": datetime.utcnow().isoformat(),
                "execution_id": job.execution_id,
            }

            # Broadcast to all connected clients (or filter by user/session)
            await self.ws_manager.broadcast(message)

            logger.debug(f"[TrainingMonitor] Sent WebSocket notification for job {job.id}")

        except Exception as e:
            logger.error(f"[TrainingMonitor] Error sending WebSocket notification: {e}")

    async def _collect_metrics(self, job: TrainingJob):
        """
        Collect training metrics from Training Logger API.

        The training pod sends metrics to Backend API via TrainingLogger.
        Here we can optionally fetch and cache them for quick access.
        """
        try:
            # Metrics are already stored by TrainingLogger
            # This is a placeholder for additional metric collection if needed
            pass

        except Exception as e:
            logger.error(f"[TrainingMonitor] Error collecting metrics for job {job.id}: {e}")


# Global monitor instance
_monitor_instance: Optional[TrainingMonitor] = None


def get_monitor() -> TrainingMonitor:
    """Get or create global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = TrainingMonitor()
    return _monitor_instance


async def start_monitoring():
    """Start background monitoring (called at app startup)"""
    monitor = get_monitor()
    await monitor.start()


async def stop_monitoring():
    """Stop background monitoring (called at app shutdown)"""
    monitor = get_monitor()
    await monitor.stop()
