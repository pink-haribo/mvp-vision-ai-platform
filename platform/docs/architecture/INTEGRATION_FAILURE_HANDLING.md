# Integration Failure Handling Design

**Date**: 2025-01-11
**Status**: Production Design
**Priority**: P0 - Critical for production deployment

Complete strategy for handling integration failures across all platform services.

---

## Table of Contents

- [Overview](#overview)
- [Integration Map](#integration-map)
- [Backend ↔ Trainer Service](#backend--trainer-service)
- [Backend ↔ MLflow](#backend--mlflow)
- [Backend ↔ Temporal](#backend--temporal)
- [Backend ↔ Database](#backend--database)
- [Backend ↔ S3 Storage](#backend--s3-storage)
- [Backend ↔ LLM Services](#backend--llm-services)
- [Trainer ↔ S3 Storage](#trainer--s3-storage)
- [Frontend ↔ Backend](#frontend--backend)
- [Graceful Degradation Strategies](#graceful-degradation-strategies)
- [Health Check Implementation](#health-check-implementation)
- [Testing Strategy](#testing-strategy)

---

## Overview

### Design Principles

1. **Fail Independently**: Failures in one integration don't cascade to others
2. **Timeout Everything**: All external calls have explicit timeouts
3. **Retry Intelligently**: Retry transient failures with exponential backoff
4. **Degrade Gracefully**: Continue with reduced functionality when possible
5. **Monitor Actively**: Track all integration health metrics

### Integration Reliability Matrix

| Integration | Criticality | SLA Target | Failure Impact | Mitigation |
|-------------|-------------|------------|----------------|------------|
| **Trainer Service** | CRITICAL | 99.5% | Training fails | Retry, queue |
| **Database** | CRITICAL | 99.9% | All operations fail | Connection pool, replica |
| **S3 Storage** | CRITICAL | 99.9% | Data access fails | Retry, cache |
| **Temporal** | HIGH | 99.5% | Workflows fail | Retry, manual recovery |
| **MLflow** | MEDIUM | 95% | Metrics lost | Buffer, offline mode |
| **LLM (Claude/GPT)** | MEDIUM | 95% | Chat unavailable | Fallback, retry |

---

## Integration Map

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                    │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Chat    │  │ Training │  │ Datasets │  │Analytics │  │
│  │  Panel   │  │Dashboard │  │  Upload  │  │Dashboard │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└───────┼─────────────┼─────────────┼─────────────┼─────────┘
        │             │             │             │
        │    HTTP/WS  │    HTTP/WS  │    HTTP     │    HTTP
        └─────────────┴─────────────┴─────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         Backend (FastAPI)               │
        │                                         │
        │  ┌──────────────────────────────────┐  │
        │  │   API Gateway + Auth Middleware  │  │
        │  └──────────────────────────────────┘  │
        │                                         │
        │  ┌─────────┐  ┌─────────┐  ┌────────┐ │
        │  │Training │  │ Dataset │  │ User   │ │
        │  │Service  │  │Service  │  │Service │ │
        │  └────┬────┘  └────┬────┘  └───┬────┘ │
        └───────┼────────────┼────────────┼──────┘
                │            │            │
        ┌───────┴────┬───────┴────┬───────┴──────┐
        │            │            │              │
        ▼            ▼            ▼              ▼
┌─────────────┐ ┌────────┐ ┌─────────┐ ┌──────────────┐
│  Trainer    │ │  S3    │ │Database │ │   Temporal   │
│  Services   │ │Storage │ │(Postgres│ │  (Workflows) │
│             │ │(MinIO/ │ │  +Redis)│ │              │
│ ┌─────────┐ │ │  R2)   │ │         │ │              │
│ │Ultralyt.│ │ └────────┘ └─────────┘ └──────────────┘
│ ├─────────┤ │      ▲
│ │  timm   │ │      │
│ ├─────────┤ │      │ Artifacts
│ │Hugging  │ │      │
│ │  Face   │ │      │
│ └─────────┘ │      │
└─────┬───────┘      │
      │              │
      │   Callback   │
      │   (HTTP)     │
      └──────────────┘
              │
              ▼
      ┌─────────────┐
      │   MLflow    │
      │  (Tracking) │
      └─────────────┘
```

---

## Backend ↔ Trainer Service

### 1. Integration Overview

**Communication Pattern**: HTTP API + Callback

**Critical Operations**:
- `POST /training/start` - Start training job
- `POST /training/stop` - Stop running job
- `GET /models/list` - List available models
- `GET /health` - Service health check

**Failure Modes**:
- Service unavailable (not started, crashed)
- Network timeout
- Training process crash
- Callback unreachable from trainer

### 2. Timeout Configuration

```python
# backend/app/services/trainer_client.py
import httpx
from typing import Optional, Dict, Any

class TrainerClient:
    """HTTP client for trainer service communication with comprehensive error handling"""

    # Timeout configuration (in seconds)
    TIMEOUTS = {
        "start_training": 30,      # Training start is quick (just spawns process)
        "stop_training": 10,       # Stopping should be immediate
        "list_models": 5,          # Model list is cached
        "health_check": 2,         # Health checks must be fast
        "default": 10,
    }

    def __init__(self, base_url: str, service_name: str = "trainer"):
        self.base_url = base_url
        self.service_name = service_name
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=ServiceUnavailableError,
        )

    @retry_trainer_service()
    async def start_training(
        self,
        config: Dict[str, Any],
        callback_url: str,
        dataset_s3_uri: str,
    ) -> Dict[str, Any]:
        """Start training job with retry and timeout"""
        try:
            async with httpx.AsyncClient(
                timeout=self.TIMEOUTS["start_training"]
            ) as client:
                response = await client.post(
                    f"{self.base_url}/training/start",
                    json={
                        "config": config,
                        "callback_url": callback_url,
                        "dataset_s3_uri": dataset_s3_uri,
                        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
                        "s3_endpoint_url": os.getenv("S3_ENDPOINT_URL"),
                    },
                    headers={
                        "X-Request-ID": str(uuid.uuid4()),
                        "Authorization": f"Bearer {self._get_internal_token()}",
                    },
                )

                if response.status_code == 503:
                    raise ServiceUnavailableError(
                        service=self.service_name,
                        retry_after=int(response.headers.get("Retry-After", 60)),
                    )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException as e:
            raise TimeoutError(
                operation=f"{self.service_name}.start_training",
                timeout_seconds=self.TIMEOUTS["start_training"],
                is_retryable=True,
            ) from e

        except httpx.ConnectError as e:
            raise NetworkError(
                service=self.service_name,
                original=e,
            ) from e

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise ServiceUnavailableError(service=self.service_name)
            elif e.response.status_code == 422:
                raise ConfigurationError(
                    field="training_config",
                    issue=e.response.json().get("detail", "Invalid configuration"),
                )
            else:
                raise UnexpectedError(e) from e

    async def stop_training(self, job_id: str) -> Dict[str, Any]:
        """Stop training job (best effort, don't retry)"""
        try:
            async with httpx.AsyncClient(
                timeout=self.TIMEOUTS["stop_training"]
            ) as client:
                response = await client.post(
                    f"{self.base_url}/training/stop",
                    json={"job_id": job_id},
                )
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            # Stopping timed out, but training might still stop
            logger.warning(f"Stop training timed out for job {job_id}")
            return {"status": "stop_requested", "confirmed": False}

        except Exception as e:
            logger.error(f"Failed to stop training job {job_id}: {e}")
            # Don't raise - stopping is best effort
            return {"status": "stop_failed", "error": str(e)}

    @circuit_breaker_protected
    async def list_models(self, framework: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available models with circuit breaker protection"""
        try:
            async with httpx.AsyncClient(
                timeout=self.TIMEOUTS["list_models"]
            ) as client:
                response = await client.get(
                    f"{self.base_url}/models/list",
                    params={"framework": framework} if framework else {},
                )
                response.raise_for_status()
                return response.json()["models"]

        except Exception as e:
            # Return cached models if service is down
            logger.warning(f"Failed to fetch models from {self.service_name}, using cache")
            return self._get_cached_models(framework)

    async def health_check(self) -> bool:
        """Check if trainer service is healthy"""
        try:
            async with httpx.AsyncClient(
                timeout=self.TIMEOUTS["health_check"]
            ) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200 and response.json().get("status") == "healthy"
        except Exception:
            return False
```

### 3. Failure Scenarios and Handling

#### Scenario 1: Trainer Service Not Started

**Symptoms**:
- Connection refused errors
- Health check fails

**Detection**:
```python
# backend/app/services/training_service.py
async def create_training_job(config: TrainingConfig) -> TrainingJob:
    # Pre-flight health check
    trainer_client = get_trainer_client(config.framework)

    if not await trainer_client.health_check():
        raise ServiceUnavailableError(
            service=f"{config.framework} trainer",
            retry_after=30,
        )

    # Proceed with job creation
    ...
```

**Resolution**:
- Return 503 Service Unavailable to user
- Retry after 30 seconds (user or automated)
- Alert ops team if service down > 5 minutes

#### Scenario 2: Training Start Times Out

**Symptoms**:
- POST /training/start takes > 30 seconds

**Handling**:
```python
try:
    result = await trainer_client.start_training(...)
except TimeoutError as e:
    # Mark job as failed, don't retry (training may have started)
    job.status = JobStatus.FAILED
    job.error_message = "Training failed to start within timeout"
    job.error_details = e.to_dict()
    await db.commit()

    # Alert ops team
    await alert_service.send_alert(
        severity="high",
        title="Training start timeout",
        message=f"Trainer {config.framework} timed out starting job {job.id}",
    )
```

#### Scenario 3: Callback Unreachable from Trainer

**Symptoms**:
- Training starts but no callbacks received
- Job stuck in RUNNING state

**Detection**:
```python
# backend/app/tasks/job_monitor.py
async def monitor_stale_jobs():
    """Check for jobs with no progress updates"""
    stale_jobs = await db.query(TrainingJob).filter(
        TrainingJob.status == JobStatus.RUNNING,
        TrainingJob.updated_at < datetime.utcnow() - timedelta(minutes=10),
    ).all()

    for job in stale_jobs:
        logger.warning(f"Job {job.id} has no updates for 10 minutes")

        # Check if process still alive (subprocess mode)
        if job.process_id and not psutil.pid_exists(job.process_id):
            job.status = JobStatus.FAILED
            job.error_message = "Training process terminated unexpectedly"
            await db.commit()

        # Send reminder alert
        if datetime.utcnow() - job.updated_at > timedelta(minutes=30):
            await alert_service.send_alert(
                severity="high",
                title="Training job stale",
                message=f"Job {job.id} has no updates for 30+ minutes",
            )
```

**Resolution**:
- After 30 minutes with no updates, mark job as failed
- Send alert to ops team
- In subprocess mode: check if process still alive
- In K8s mode: check if pod still running

#### Scenario 4: Trainer Process Crashes

**Symptoms**:
- Callback with status="failed" received
- Exit code != 0 in subprocess mode
- Pod CrashLoopBackOff in K8s

**Handling**:
```python
# backend/app/api/training.py
@router.post("/training-jobs/{job_id}/callback")
async def training_callback(job_id: str, update: TrainingUpdate):
    job = await db.get_job(job_id)

    if update.status == "failed":
        job.status = JobStatus.FAILED
        job.error_message = update.error.get("message", "Training failed")
        job.error_details = update.error
        job.completed_at = datetime.utcnow()

        # Classify error for potential retry
        error_code = update.error.get("code")
        if error_code in ["CUDA_OOM", "NETWORK_ERROR"]:
            # User can retry with different config
            job.retryable = True
        else:
            job.retryable = False

        await db.commit()

        # Send WebSocket update to frontend
        await websocket_manager.send_to_user(
            user_id=job.user_id,
            message={
                "type": "training_failed",
                "job_id": job.id,
                "error": update.error,
                "retryable": job.retryable,
            },
        )
```

### 4. Fallback Strategies

**Strategy 1: Trainer Queue**
```python
# If trainer service is busy, queue the request
if response.status_code == 503:
    await training_queue.enqueue(job_id, priority=job.priority)
    job.status = JobStatus.QUEUED
    await db.commit()

    return {
        "status": "queued",
        "message": "Training service is busy. Your job is queued.",
        "position": await training_queue.get_position(job_id),
    }
```

**Strategy 2: Model List Cache**
```python
# Cache model list to survive trainer service downtime
@cached(ttl=3600)  # 1 hour
async def get_available_models(framework: str):
    try:
        return await trainer_client.list_models(framework)
    except Exception:
        # Return last known good models
        return FALLBACK_MODELS[framework]
```

---

## Backend ↔ MLflow

### 1. Integration Overview

**Communication Pattern**: HTTP API (MLflow REST API)

**Critical Operations**:
- Create experiment
- Log metrics
- Log artifacts
- Query metrics

**Failure Modes**:
- MLflow server down
- S3 backend (artifact store) unreachable
- Metric logging timeout
- Disk full (local artifact store)

### 2. Timeout Configuration

```python
# backend/app/services/mlflow_client.py
import mlflow
from mlflow.tracking import MlflowClient

class ResilientMLflowClient:
    """MLflow client with error handling and offline mode"""

    TIMEOUTS = {
        "create_experiment": 5,
        "log_metric": 2,
        "log_artifact": 30,
        "search_runs": 10,
    }

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri)
        self.offline_mode = False
        self.metric_buffer = []

    @retry_mlflow()
    async def create_experiment(self, name: str, tags: Dict[str, str]) -> str:
        """Create MLflow experiment with retry"""
        try:
            with timeout(self.TIMEOUTS["create_experiment"]):
                experiment_id = self.client.create_experiment(name, tags=tags)
                return experiment_id
        except Exception as e:
            logger.warning(f"Failed to create MLflow experiment: {e}")
            # Generate local experiment ID, will sync later
            return f"local-{uuid.uuid4()}"

    async def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: Optional[int] = None,
    ):
        """Log metric with buffering for offline mode"""
        if self.offline_mode:
            self.metric_buffer.append({
                "run_id": run_id,
                "key": key,
                "value": value,
                "step": step,
                "timestamp": datetime.utcnow(),
            })
            return

        try:
            with timeout(self.TIMEOUTS["log_metric"]):
                self.client.log_metric(run_id, key, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metric to MLflow: {e}")
            # Enter offline mode
            self.offline_mode = True
            self.metric_buffer.append({
                "run_id": run_id,
                "key": key,
                "value": value,
                "step": step,
                "timestamp": datetime.utcnow(),
            })

    async def flush_buffered_metrics(self):
        """Flush buffered metrics when MLflow comes back online"""
        if not self.metric_buffer:
            return

        logger.info(f"Flushing {len(self.metric_buffer)} buffered metrics to MLflow")

        success_count = 0
        for metric in self.metric_buffer:
            try:
                self.client.log_metric(
                    metric["run_id"],
                    metric["key"],
                    metric["value"],
                    step=metric["step"],
                    timestamp=int(metric["timestamp"].timestamp() * 1000),
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to flush metric: {e}")

        if success_count == len(self.metric_buffer):
            self.metric_buffer.clear()
            self.offline_mode = False
            logger.info("All buffered metrics flushed successfully")
        else:
            logger.warning(f"Only flushed {success_count}/{len(self.metric_buffer)} metrics")
```

### 3. Failure Scenarios and Handling

#### Scenario 1: MLflow Server Down

**Detection**:
```python
async def check_mlflow_health() -> bool:
    """Check if MLflow is accessible"""
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{MLFLOW_TRACKING_URI}/health")
            return response.status_code == 200
    except Exception:
        return False

# Run periodic health check
@scheduler.task(interval=60)  # Every 60 seconds
async def monitor_mlflow_health():
    is_healthy = await check_mlflow_health()

    if not is_healthy:
        logger.warning("MLflow server is down, entering offline mode")
        mlflow_client.offline_mode = True

        # Alert if down > 5 minutes
        if not hasattr(monitor_mlflow_health, 'down_since'):
            monitor_mlflow_health.down_since = datetime.utcnow()
        elif datetime.utcnow() - monitor_mlflow_health.down_since > timedelta(minutes=5):
            await alert_service.send_alert(
                severity="medium",
                title="MLflow server down",
                message="MLflow has been unavailable for 5+ minutes",
            )
    else:
        if mlflow_client.offline_mode:
            logger.info("MLflow server is back online, flushing buffered metrics")
            await mlflow_client.flush_buffered_metrics()

        if hasattr(monitor_mlflow_health, 'down_since'):
            del monitor_mlflow_health.down_since
```

**Handling**:
- Enter offline mode (buffer metrics)
- Training continues without MLflow
- Flush metrics when MLflow recovers
- Alert if down > 5 minutes

#### Scenario 2: Artifact Upload Fails

**Handling**:
```python
async def log_artifact_with_retry(
    run_id: str,
    local_path: str,
    artifact_path: str,
):
    """Log artifact with retry and fallback to direct S3 upload"""
    try:
        await mlflow_client.log_artifact(run_id, local_path, artifact_path)
    except Exception as e:
        logger.warning(f"MLflow artifact upload failed, uploading directly to S3: {e}")

        # Fallback: upload directly to S3
        s3_key = f"mlflow/{run_id}/artifacts/{artifact_path}"
        await s3_client.upload_file(local_path, MLFLOW_BUCKET, s3_key)

        # Store S3 path in database for later retrieval
        await db.execute(
            "INSERT INTO artifact_fallback (run_id, artifact_path, s3_key) VALUES (?, ?, ?)",
            (run_id, artifact_path, s3_key),
        )
```

### 4. Graceful Degradation

**Level 1: Offline Metric Logging**
- Buffer metrics in memory
- Continue training
- Flush when MLflow recovers

**Level 2: Database Fallback**
- Store critical metrics in PostgreSQL
- Training dashboard still works
- Sync to MLflow later

**Level 3: Training Continues**
- Even if MLflow is completely down
- Training completes successfully
- Metrics can be manually recovered from logs

---

## Backend ↔ Temporal

### 1. Integration Overview

**Communication Pattern**: gRPC (Temporal SDK)

**Critical Operations**:
- Start workflow
- Send signal to workflow
- Query workflow state
- Cancel workflow

**Failure Modes**:
- Temporal server unreachable
- Workflow worker disconnected
- Workflow timeout
- Activity failure

### 2. Timeout Configuration

```python
# backend/app/workflows/training_workflow.py
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from datetime import timedelta

@workflow.defn
class TrainingWorkflow:
    """Training workflow with comprehensive error handling"""

    @workflow.run
    async def run(self, job_id: str, config: TrainingConfig) -> TrainingResult:
        # Activity timeouts and retry policies
        validate_activity_options = {
            "start_to_close_timeout": timedelta(minutes=10),
            "retry_policy": RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
                backoff_coefficient=2.0,
            ),
        }

        training_activity_options = {
            "start_to_close_timeout": timedelta(hours=24),
            "heartbeat_timeout": timedelta(minutes=5),
            "retry_policy": RetryPolicy(
                maximum_attempts=1,  # Don't retry training
            ),
        }

        # Step 1: Validate dataset
        try:
            validation_result = await workflow.execute_activity(
                validate_dataset_activity,
                args=[config.dataset_id],
                **validate_activity_options,
            )

            if not validation_result.valid:
                raise workflow.ApplicationError(
                    f"Dataset validation failed: {validation_result.error}"
                )
        except workflow.ActivityError as e:
            # Activity failed after retries
            raise workflow.ApplicationError(f"Dataset validation failed: {e}")

        # Step 2: Start training (long-running, callback-driven)
        training_task = asyncio.create_task(
            workflow.execute_activity(
                start_training_activity,
                args=[job_id, config],
                **training_activity_options,
            )
        )

        # Step 3: Wait for completion via signal or timeout
        try:
            await workflow.wait_condition(
                lambda: self.training_completed or self.training_failed,
                timeout=timedelta(hours=24),
            )
        except asyncio.TimeoutError:
            # Training took too long, cancel it
            await workflow.execute_activity(
                cancel_training_activity,
                args=[job_id],
                start_to_close_timeout=timedelta(seconds=30),
            )
            raise workflow.ApplicationError("Training timeout exceeded 24 hours")

        # Step 4: Cleanup (always execute)
        try:
            await workflow.execute_activity(
                cleanup_training_activity,
                args=[job_id],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
        except Exception as e:
            workflow.logger.error(f"Cleanup failed (non-critical): {e}")

        if self.training_failed:
            raise workflow.ApplicationError(f"Training failed: {self.error_message}")

        return self.training_result

    @workflow.signal
    def training_progress_update(self, update: Dict[str, Any]):
        """Receive progress updates from trainer via backend callback"""
        if update["status"] == "completed":
            self.training_completed = True
            self.training_result = update["result"]
        elif update["status"] == "failed":
            self.training_failed = True
            self.error_message = update["error"]["message"]
```

### 3. Failure Scenarios and Handling

#### Scenario 1: Temporal Server Unreachable

**Detection & Handling**:
```python
# backend/app/services/workflow_service.py
from temporalio.client import Client, WorkflowFailureError

class WorkflowService:
    """Service for managing Temporal workflows"""

    async def start_training_workflow(self, job_id: str, config: TrainingConfig):
        """Start training workflow with error handling"""
        try:
            client = await Client.connect(
                os.getenv("TEMPORAL_HOST"),
                namespace="default",
                timeout=timedelta(seconds=10),
            )

            handle = await client.start_workflow(
                TrainingWorkflow.run,
                args=[job_id, config],
                id=f"training-{job_id}",
                task_queue="training-tasks",
                execution_timeout=timedelta(hours=25),
            )

            return handle.id

        except asyncio.TimeoutError:
            # Temporal server unreachable
            logger.error("Failed to connect to Temporal server")

            # Fallback: Start training directly (bypass Temporal)
            await self._start_training_without_workflow(job_id, config)

            # Alert ops team
            await alert_service.send_alert(
                severity="high",
                title="Temporal server unreachable",
                message="Starting training without workflow orchestration",
            )

            raise ServiceUnavailableError(
                service="Temporal",
                retry_after=60,
            )

    async def _start_training_without_workflow(self, job_id: str, config: TrainingConfig):
        """Fallback: start training directly without Temporal"""
        # Directly call trainer service
        trainer_client = get_trainer_client(config.framework)
        await trainer_client.start_training(
            config=config.dict(),
            callback_url=f"{BACKEND_URL}/api/training-jobs/{job_id}/callback",
            dataset_s3_uri=f"s3://{BUCKET}/datasets/{config.dataset_id}/",
        )

        # Update job to indicate no workflow
        job = await db.get_job(job_id)
        job.temporal_workflow_id = None
        job.fallback_mode = True
        await db.commit()
```

#### Scenario 2: Workflow Worker Disconnected

**Detection**:
```python
# Workflow workers send heartbeats
@activity.defn
async def start_training_activity(job_id: str, config: TrainingConfig):
    """Training activity with heartbeat"""
    # Start training via trainer service
    await trainer_client.start_training(...)

    # Send heartbeats while training runs
    while not training_completed:
        activity.heartbeat({"progress": get_training_progress(job_id)})
        await asyncio.sleep(60)  # Heartbeat every 60s
```

**Handling**:
- Activity fails if no heartbeat for 5 minutes
- Workflow cancels training and retries
- Alert if multiple workers disconnecting

#### Scenario 3: Activity Timeout

**Handling**:
```python
try:
    result = await workflow.execute_activity(
        some_activity,
        start_to_close_timeout=timedelta(minutes=10),
    )
except workflow.ActivityError as e:
    if "timeout" in str(e).lower():
        # Activity timed out
        workflow.logger.error(f"Activity timed out: {e}")

        # Decide: retry, skip, or fail workflow
        if is_critical_activity:
            raise  # Fail workflow
        else:
            # Skip non-critical activity
            workflow.logger.warning("Skipping non-critical activity")
            result = None
```

### 4. Callback-First Integration

**Resolution of Temporal vs Callback contradiction**:

```python
# Backend callback handler sends signal to Temporal
@router.post("/training-jobs/{job_id}/callback")
async def training_callback(job_id: str, update: TrainingUpdate):
    # 1. Update database (source of truth)
    job = await db.get_job(job_id)
    job.status = update.status
    job.current_epoch = update.progress.get("epoch")
    job.current_metrics = update.progress.get("metrics")
    job.updated_at = datetime.utcnow()
    await db.commit()

    # 2. Send signal to Temporal workflow (if exists)
    if job.temporal_workflow_id:
        try:
            client = await get_temporal_client()
            handle = client.get_workflow_handle(job.temporal_workflow_id)
            await handle.signal(
                "training_progress_update",
                update.dict(),
            )
        except Exception as e:
            # Workflow might not exist (fallback mode), that's OK
            logger.warning(f"Failed to signal workflow: {e}")

    # 3. Send WebSocket update to frontend
    await websocket_manager.send_to_user(
        user_id=job.user_id,
        message={"type": "training_update", **update.dict()},
    )
```

**Key principle**: Database is source of truth, Temporal reacts to signals.

---

## Backend ↔ Database

### 1. Integration Overview

**Communication Pattern**: SQL over TCP (asyncpg for PostgreSQL, aioredis for Redis)

**Critical Operations**:
- All CRUD operations
- Transaction management
- Connection pooling

**Failure Modes**:
- Connection refused (database not started)
- Connection timeout (network issue)
- Connection pool exhausted
- Deadlock
- Disk full

### 2. Connection Pool Configuration

```python
# backend/app/db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

def create_engine():
    """Create database engine with connection pooling"""
    return create_async_engine(
        DATABASE_URL,
        # Connection pool settings
        poolclass=QueuePool,
        pool_size=20,              # Max connections in pool
        max_overflow=10,           # Additional connections when pool exhausted
        pool_timeout=30,           # Wait up to 30s for available connection
        pool_recycle=3600,         # Recycle connections after 1 hour
        pool_pre_ping=True,        # Verify connections before use
        # Timeout settings
        connect_args={
            "timeout": 10,         # Connection timeout (seconds)
            "command_timeout": 30, # Query timeout (seconds)
        },
        # Error handling
        echo=False,
        future=True,
    )

engine = create_engine()
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

### 3. Failure Scenarios and Handling

#### Scenario 1: Database Connection Failed

**Handling**:
```python
# backend/app/db/session.py
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def get_db_session() -> AsyncSession:
    """Get database session with retry"""
    try:
        async with AsyncSessionLocal() as session:
            # Test connection
            await session.execute("SELECT 1")
            yield session
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise DatabaseError(operation="connect", original=e)

# Startup check
async def verify_database_connection():
    """Verify database is accessible on startup"""
    max_retries = 10
    for i in range(max_retries):
        try:
            async with AsyncSessionLocal() as session:
                await session.execute("SELECT 1")
                logger.info("Database connection successful")
                return
        except Exception as e:
            if i == max_retries - 1:
                logger.critical("Failed to connect to database after retries")
                raise
            logger.warning(f"Database not ready, retrying in 5s... ({i+1}/{max_retries})")
            await asyncio.sleep(5)
```

#### Scenario 2: Connection Pool Exhausted

**Detection & Handling**:
```python
from prometheus_client import Gauge

# Metrics
db_pool_size = Gauge('db_pool_size', 'Database connection pool size')
db_pool_available = Gauge('db_pool_available', 'Available database connections')

@scheduler.task(interval=10)
async def monitor_connection_pool():
    """Monitor connection pool health"""
    pool = engine.pool
    db_pool_size.set(pool.size())
    db_pool_available.set(pool.size() - pool.checkedout())

    utilization = pool.checkedout() / pool.size()
    if utilization > 0.9:
        logger.warning(f"Connection pool {utilization*100:.0f}% utilized")
        await alert_service.send_alert(
            severity="medium",
            title="Database connection pool near capacity",
            message=f"{pool.checkedout()}/{pool.size()} connections in use",
        )

# Graceful handling when pool exhausted
try:
    async with get_db_session() as session:
        ...
except asyncio.TimeoutError:
    raise ServiceUnavailableError(
        service="Database",
        retry_after=5,
    )
```

#### Scenario 3: Query Timeout

**Handling**:
```python
# Set statement timeout for long queries
async def execute_with_timeout(session: AsyncSession, query, timeout_seconds: int = 30):
    """Execute query with timeout"""
    try:
        await session.execute(f"SET statement_timeout = {timeout_seconds * 1000}")
        result = await session.execute(query)
        await session.execute("SET statement_timeout = 0")  # Reset
        return result
    except asyncpg.QueryCanceledError:
        raise TimeoutError(
            operation="database_query",
            timeout_seconds=timeout_seconds,
            is_retryable=False,
        )
```

#### Scenario 4: Database Disk Full

**Detection**:
```python
@scheduler.task(interval=300)  # Every 5 minutes
async def check_database_disk_space():
    """Check database disk space"""
    async with get_db_session() as session:
        result = await session.execute(
            "SELECT pg_database_size('visionai_platform') as size"
        )
        db_size_bytes = result.scalar()

        # Check if disk space > 80% of quota
        if db_size_bytes > DATABASE_QUOTA_BYTES * 0.8:
            await alert_service.send_alert(
                severity="high",
                title="Database disk space running low",
                message=f"Database size: {db_size_bytes / 1e9:.2f}GB",
            )
```

---

## Backend ↔ S3 Storage

### 1. Integration Overview

**Communication Pattern**: HTTP (S3 API via boto3/aioboto3)

**Critical Operations**:
- Upload file (PUT object)
- Download file (GET object)
- List objects
- Generate presigned URL
- Delete object

**Failure Modes**:
- S3 endpoint unreachable
- Upload timeout (large files)
- Bucket quota exceeded
- Access denied (credentials invalid)

### 2. Timeout Configuration

```python
# backend/app/storage/s3_client.py
import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

class S3Client:
    """S3 client with comprehensive error handling"""

    def __init__(self):
        self.config = Config(
            # Timeouts
            connect_timeout=5,
            read_timeout=60,
            # Retries
            retries={
                'max_attempts': 3,
                'mode': 'adaptive',
            },
        )

        self.session = aioboto3.Session()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=120,
        )

    @retry_s3()
    async def upload_file(
        self,
        file_path: str,
        bucket: str,
        key: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload file to S3 with retry"""
        try:
            async with self.session.client(
                's3',
                endpoint_url=os.getenv('S3_ENDPOINT_URL'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                config=self.config,
            ) as s3:
                with open(file_path, 'rb') as f:
                    await s3.upload_fileobj(
                        f,
                        bucket,
                        key,
                        ExtraArgs={'ContentType': content_type} if content_type else None,
                    )

                return f"s3://{bucket}/{key}"

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')

            if error_code == 'NoSuchBucket':
                raise ConfigurationError(
                    field="s3_bucket",
                    issue=f"Bucket '{bucket}' does not exist",
                )
            elif error_code == 'AccessDenied':
                raise ConfigurationError(
                    field="s3_credentials",
                    issue="S3 access denied. Check credentials.",
                )
            elif error_code == 'QuotaExceeded':
                raise QuotaExceededError(
                    resource="storage",
                    limit=STORAGE_QUOTA_GB,
                    current=await self._get_storage_usage(bucket),
                )
            else:
                raise StorageError(
                    operation="upload",
                    bucket=bucket,
                    key=key,
                    original=e,
                )

        except BotoCoreError as e:
            raise NetworkError(service="S3", original=e)

    @circuit_breaker_protected
    async def download_file(
        self,
        bucket: str,
        key: str,
        local_path: str,
    ):
        """Download file from S3 with circuit breaker"""
        try:
            async with self.session.client(
                's3',
                endpoint_url=os.getenv('S3_ENDPOINT_URL'),
                config=self.config,
            ) as s3:
                await s3.download_file(bucket, key, local_path)

        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'NoSuchKey':
                raise ResourceNotFoundError(
                    resource_type="file",
                    resource_id=f"s3://{bucket}/{key}",
                )
            else:
                raise StorageError(
                    operation="download",
                    bucket=bucket,
                    key=key,
                    original=e,
                )

    async def generate_presigned_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate presigned URL for direct browser access"""
        try:
            async with self.session.client(
                's3',
                endpoint_url=os.getenv('S3_ENDPOINT_URL'),
                config=self.config,
            ) as s3:
                url = await s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': key},
                    ExpiresIn=expires_in,
                )
                return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            # Fallback: return direct S3 URL (requires public access)
            return f"{os.getenv('S3_ENDPOINT_URL')}/{bucket}/{key}"
```

### 3. Failure Scenarios

#### Scenario 1: S3 Endpoint Unreachable

**Handling**:
- Retry 3 times with exponential backoff
- Circuit breaker opens after 5 failures
- Return cached data if available
- Alert if S3 down > 5 minutes

#### Scenario 2: Upload Timeout (Large Files)

**Handling**:
```python
# Use multipart upload for large files
async def upload_large_file(file_path: str, bucket: str, key: str):
    """Upload large file using multipart upload"""
    file_size = os.path.getsize(file_path)

    # Use multipart if file > 100MB
    if file_size > 100 * 1024 * 1024:
        async with self.session.client('s3', ...) as s3:
            # Create multipart upload
            response = await s3.create_multipart_upload(Bucket=bucket, Key=key)
            upload_id = response['UploadId']

            # Upload parts (5MB chunks)
            part_size = 5 * 1024 * 1024
            parts = []

            with open(file_path, 'rb') as f:
                part_number = 1
                while True:
                    data = f.read(part_size)
                    if not data:
                        break

                    response = await s3.upload_part(
                        Bucket=bucket,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=data,
                    )
                    parts.append({
                        'PartNumber': part_number,
                        'ETag': response['ETag'],
                    })
                    part_number += 1

            # Complete multipart upload
            await s3.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts},
            )
    else:
        # Use regular upload
        await self.upload_file(file_path, bucket, key)
```

---

## Graceful Degradation Strategies

### Overall Strategy Matrix

| Service Down | Critical? | Degradation Strategy | User Impact |
|--------------|-----------|---------------------|-------------|
| **Trainer** | YES | Queue requests | Training delayed |
| **Database** | YES | No degradation | Service unavailable |
| **S3** | YES | Cache, retry | Reduced functionality |
| **Temporal** | NO | Bypass workflows | Training works, no orchestration |
| **MLflow** | NO | Buffer metrics | Training works, metrics delayed |
| **LLM** | NO | Disable chat | Training unaffected |

### Implementation

```python
# backend/app/core/degradation.py
from enum import Enum

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"

class PlatformStatus:
    """Track overall platform health and degradation state"""

    def __init__(self):
        self.services = {
            "trainer": ServiceStatus.HEALTHY,
            "database": ServiceStatus.HEALTHY,
            "s3": ServiceStatus.HEALTHY,
            "temporal": ServiceStatus.HEALTHY,
            "mlflow": ServiceStatus.HEALTHY,
            "llm": ServiceStatus.HEALTHY,
        }

    def set_service_status(self, service: str, status: ServiceStatus):
        """Update service status and determine overall health"""
        self.services[service] = status

        # Determine overall platform status
        if self.services["database"] == ServiceStatus.DOWN:
            self.overall_status = ServiceStatus.DOWN
        elif self.services["trainer"] == ServiceStatus.DOWN:
            self.overall_status = ServiceStatus.DOWN
        elif ServiceStatus.DEGRADED in self.services.values():
            self.overall_status = ServiceStatus.DEGRADED
        else:
            self.overall_status = ServiceStatus.HEALTHY

    def get_capabilities(self) -> Dict[str, bool]:
        """Return what features are currently available"""
        return {
            "training": self.services["trainer"] != ServiceStatus.DOWN,
            "chat": self.services["llm"] != ServiceStatus.DOWN,
            "metrics": self.services["mlflow"] != ServiceStatus.DOWN,
            "workflows": self.services["temporal"] != ServiceStatus.DOWN,
            "storage": self.services["s3"] != ServiceStatus.DOWN,
        }

platform_status = PlatformStatus()

# API endpoint to expose platform status
@router.get("/health/detailed")
async def get_platform_health():
    """Get detailed platform health status"""
    return {
        "status": platform_status.overall_status,
        "services": platform_status.services,
        "capabilities": platform_status.get_capabilities(),
        "timestamp": datetime.utcnow().isoformat(),
    }
```

---

## Health Check Implementation

### 1. Service Health Checks

```python
# backend/app/health/checks.py
from typing import Dict, Any

class HealthChecker:
    """Comprehensive health checking for all integrations"""

    async def check_all(self) -> Dict[str, Any]:
        """Check all services in parallel"""
        results = await asyncio.gather(
            self.check_database(),
            self.check_s3(),
            self.check_trainer_services(),
            self.check_temporal(),
            self.check_mlflow(),
            self.check_llm(),
            return_exceptions=True,
        )

        return {
            "database": results[0],
            "s3": results[1],
            "trainers": results[2],
            "temporal": results[3],
            "mlflow": results[4],
            "llm": results[5],
        }

    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        start_time = time.time()
        try:
            async with get_db_session() as session:
                await session.execute("SELECT 1")

            latency = (time.time() - start_time) * 1000
            return {
                "status": "healthy",
                "latency_ms": latency,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def check_s3(self) -> Dict[str, Any]:
        """Check S3 connectivity"""
        try:
            async with aioboto3.Session().client('s3', ...) as s3:
                await s3.head_bucket(Bucket=os.getenv('S3_BUCKET'))

            return {"status": "healthy"}
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def check_trainer_services(self) -> Dict[str, Any]:
        """Check all trainer services"""
        trainers = ["ultralytics", "timm", "huggingface"]
        results = {}

        for trainer in trainers:
            try:
                url = os.getenv(f"{trainer.upper()}_SERVICE_URL")
                async with httpx.AsyncClient(timeout=2) as client:
                    response = await client.get(f"{url}/health")
                    results[trainer] = {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                    }
            except Exception as e:
                results[trainer] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        return results

health_checker = HealthChecker()

# Kubernetes liveness probe
@router.get("/health/live")
async def liveness():
    """Kubernetes liveness probe - is the service running?"""
    return {"status": "ok"}

# Kubernetes readiness probe
@router.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe - is the service ready to accept traffic?"""
    # Check critical dependencies
    db_ok = (await health_checker.check_database())["status"] == "healthy"
    s3_ok = (await health_checker.check_s3())["status"] == "healthy"

    if db_ok and s3_ok:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Not ready")

# Comprehensive health check
@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check for all services"""
    results = await health_checker.check_all()

    # Determine overall status
    all_healthy = all(
        r.get("status") == "healthy"
        for r in results.values()
        if isinstance(r, dict)
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": results,
        "timestamp": datetime.utcnow().isoformat(),
    }
```

### 2. Prometheus Metrics

```python
# backend/app/health/metrics.py
from prometheus_client import Gauge, Histogram

# Service health metrics
service_health = Gauge(
    'service_health',
    'Service health status (1=healthy, 0=unhealthy)',
    ['service']
)

# Integration latency
integration_latency = Histogram(
    'integration_latency_seconds',
    'Integration call latency',
    ['integration', 'operation'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
)

# Circuit breaker status
circuit_breaker_status = Gauge(
    'circuit_breaker_status',
    'Circuit breaker status (0=closed, 1=open, 0.5=half-open)',
    ['service']
)

@scheduler.task(interval=30)
async def update_health_metrics():
    """Update Prometheus health metrics"""
    health_results = await health_checker.check_all()

    for service, result in health_results.items():
        if isinstance(result, dict):
            status = 1 if result.get("status") == "healthy" else 0
            service_health.labels(service=service).set(status)
```

---

## Testing Strategy

### 1. Unit Tests

```python
# tests/test_integration_errors.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_trainer_timeout_handling():
    """Test that trainer timeouts are handled correctly"""
    trainer_client = TrainerClient("http://trainer:8001")

    with patch.object(trainer_client, '_make_request', side_effect=asyncio.TimeoutError):
        with pytest.raises(TimeoutError) as exc_info:
            await trainer_client.start_training(config, callback_url, dataset_uri)

        assert exc_info.value.error_type == ErrorType.TRANSIENT
        assert exc_info.value.is_retryable

@pytest.mark.asyncio
async def test_mlflow_offline_mode():
    """Test that MLflow offline mode works"""
    mlflow_client = ResilientMLflowClient("http://mlflow:5000")

    # Simulate MLflow being down
    mlflow_client.offline_mode = True

    # Log metrics should buffer
    await mlflow_client.log_metric("run_id", "loss", 0.5, step=1)
    assert len(mlflow_client.metric_buffer) == 1

    # Simulate MLflow recovery
    mlflow_client.offline_mode = False
    await mlflow_client.flush_buffered_metrics()

    assert len(mlflow_client.metric_buffer) == 0
```

### 2. Integration Tests

```python
# tests/integration/test_failure_scenarios.py
import pytest

@pytest.mark.asyncio
async def test_training_continues_when_mlflow_down(
    client: AsyncClient,
    mock_mlflow_server,
):
    """Test that training continues even if MLflow is down"""
    # Stop MLflow
    mock_mlflow_server.stop()

    # Start training
    response = await client.post("/api/v1/training-jobs", json={...})
    job_id = response.json()["job_id"]

    # Wait for training to complete
    await asyncio.sleep(60)

    # Check job status
    response = await client.get(f"/api/v1/training-jobs/{job_id}")
    job = response.json()

    # Training should complete successfully
    assert job["status"] == "completed"
    assert job["mlflow_offline_mode"] == True

@pytest.mark.asyncio
async def test_temporal_fallback():
    """Test that training works without Temporal"""
    with patch('temporalio.client.Client.connect', side_effect=asyncio.TimeoutError):
        response = await client.post("/api/v1/training-jobs", json={...})
        job_id = response.json()["job_id"]

        # Check that job was started in fallback mode
        job = await db.get_job(job_id)
        assert job.temporal_workflow_id is None
        assert job.fallback_mode is True
```

### 3. Chaos Engineering Tests

```bash
# tests/chaos/scenarios/trainer_down.sh
#!/bin/bash

# Stop trainer service
docker stop ultralytics-trainer

# Submit training job (should fail gracefully)
curl -X POST http://localhost:8000/api/v1/training-jobs \
  -H "Content-Type: application/json" \
  -d '{"model": "yolo11n", "dataset_id": "test", "epochs": 100}'

# Check response code (should be 503)
# Check that error message is user-friendly

# Start trainer service
docker start ultralytics-trainer

# Wait for recovery
sleep 10

# Retry training job (should succeed)
curl -X POST http://localhost:8000/api/v1/training-jobs \
  -H "Content-Type: application/json" \
  -d '{"model": "yolo11n", "dataset_id": "test", "epochs": 100}'
```

---

## Summary: Integration Failure Matrix

| Integration | Timeout | Max Retries | Circuit Breaker | Fallback | Critical |
|-------------|---------|-------------|-----------------|----------|----------|
| **Trainer → Start** | 30s | 3 | Yes (5/60s) | Queue | YES |
| **Trainer → Stop** | 10s | 0 | No | Log only | NO |
| **MLflow → Log** | 2s | 5 | No | Buffer | NO |
| **Temporal → Start** | 10s | 5 | Yes (5/120s) | Direct call | NO |
| **Database → Query** | 30s | 2 | Yes (3/30s) | None | YES |
| **S3 → Upload** | 60s | 3 | Yes (5/120s) | None | YES |
| **S3 → Download** | 60s | 3 | Yes (5/120s) | Cache | YES |
| **LLM → Generate** | 30s | 3 | Yes (5/300s) | Disable chat | NO |

**Key Takeaways**:
- ✅ All integrations have explicit timeouts
- ✅ Transient failures retry automatically
- ✅ Circuit breakers prevent cascading failures
- ✅ Non-critical services degrade gracefully
- ✅ Critical services (DB, S3, Trainer) fail fast

---

**End of Document**
