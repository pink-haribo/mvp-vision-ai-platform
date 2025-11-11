"""
Prometheus metrics for training monitoring.

This module defines all Prometheus metrics used throughout the application.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# ================================
# Training Metrics
# ================================

# Training job status (0=failed, 1=pending, 2=running, 3=completed)
training_status = Gauge(
    "training_status",
    "Current status of training job",
    labelnames=["job_id", "model_name", "dataset_name"],
)

# Training loss
training_loss = Gauge(
    "training_loss",
    "Current training loss",
    labelnames=["job_id", "model_name"],
)

# Training accuracy
training_accuracy = Gauge(
    "training_accuracy",
    "Current training accuracy (percentage 0-100)",
    labelnames=["job_id", "model_name"],
)

# Current epoch
training_epoch = Gauge(
    "training_epoch",
    "Current training epoch",
    labelnames=["job_id", "model_name"],
)

# GPU utilization
training_gpu_utilization = Gauge(
    "training_gpu_utilization",
    "GPU utilization percentage",
    labelnames=["job_id"],
)

# Memory utilization
training_memory_utilization = Gauge(
    "training_memory_utilization",
    "Memory utilization percentage",
    labelnames=["job_id"],
)

# Samples per second
training_samples_per_second = Gauge(
    "training_samples_per_second",
    "Number of samples processed per second",
    labelnames=["job_id"],
)

# ETA in seconds
training_eta_seconds = Gauge(
    "training_eta_seconds",
    "Estimated time to completion in seconds",
    labelnames=["job_id"],
)

# ================================
# API Metrics
# ================================

# API request counter
api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    labelnames=["method", "endpoint", "status_code"],
)

# API request duration
api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# LLM request counter
llm_requests_total = Counter(
    "llm_requests_total",
    "Total number of LLM API requests",
    labelnames=["model", "status"],
)

# LLM request duration
llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    labelnames=["model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

# ================================
# System Metrics
# ================================

# Active training jobs
active_training_jobs = Gauge(
    "active_training_jobs",
    "Number of currently active training jobs",
)

# Total training jobs counter
total_training_jobs = Counter(
    "total_training_jobs",
    "Total number of training jobs created",
    labelnames=["status"],
)


# ================================
# Helper Functions
# ================================


def update_training_metrics(
    job_id: int,
    model_name: str,
    dataset_name: str = "",
    status: str | None = None,
    loss: float | None = None,
    accuracy: float | None = None,
    epoch: int | None = None,
    gpu_util: float | None = None,
    mem_util: float | None = None,
    samples_per_sec: float | None = None,
    eta_seconds: float | None = None,
):
    """
    Update training metrics for a specific job.

    Args:
        job_id: Training job ID
        model_name: Name of the model being trained
        dataset_name: Name of the dataset
        status: Job status (pending/running/completed/failed)
        loss: Current loss value
        accuracy: Current accuracy (0-100)
        epoch: Current epoch number
        gpu_util: GPU utilization percentage (0-100)
        mem_util: Memory utilization percentage (0-100)
        samples_per_sec: Samples processed per second
        eta_seconds: Estimated time to completion in seconds
    """
    labels = {"job_id": str(job_id), "model_name": model_name}

    if status is not None:
        status_map = {"pending": 1, "running": 2, "completed": 3, "failed": 0}
        status_labels = {**labels, "dataset_name": dataset_name}
        training_status.labels(**status_labels).set(status_map.get(status, 0))

    if loss is not None:
        training_loss.labels(**labels).set(loss)

    if accuracy is not None:
        training_accuracy.labels(**labels).set(accuracy)

    if epoch is not None:
        training_epoch.labels(**labels).set(epoch)

    job_labels = {"job_id": str(job_id)}

    if gpu_util is not None:
        training_gpu_utilization.labels(**job_labels).set(gpu_util)

    if mem_util is not None:
        training_memory_utilization.labels(**job_labels).set(mem_util)

    if samples_per_sec is not None:
        training_samples_per_second.labels(**job_labels).set(samples_per_sec)

    if eta_seconds is not None:
        training_eta_seconds.labels(**job_labels).set(eta_seconds)


def clear_training_metrics(job_id: int, model_name: str):
    """
    Clear metrics for a completed/failed training job.

    This helps prevent stale metrics from appearing in Prometheus.
    """
    # Note: Prometheus client doesn't support removing specific label values
    # We set them to 0 instead
    labels = {"job_id": str(job_id), "model_name": model_name}
    job_labels = {"job_id": str(job_id)}

    training_loss.labels(**labels).set(0)
    training_accuracy.labels(**labels).set(0)
    training_epoch.labels(**labels).set(0)
    training_gpu_utilization.labels(**job_labels).set(0)
    training_memory_utilization.labels(**job_labels).set(0)
    training_samples_per_second.labels(**job_labels).set(0)
    training_eta_seconds.labels(**job_labels).set(0)
