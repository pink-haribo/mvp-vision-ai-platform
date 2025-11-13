# Error Handling Design

**Date**: 2025-01-11
**Status**: Production Design
**Priority**: P0 - Critical for production deployment

Complete error handling strategy for the Vision AI Training Platform.

---

## Table of Contents

- [Overview](#overview)
- [Error Taxonomy](#error-taxonomy)
- [Retry Policies](#retry-policies)
- [Error Propagation Flow](#error-propagation-flow)
- [Error Tracking and Monitoring](#error-tracking-and-monitoring)
- [User-Facing Error Messages](#user-facing-error-messages)
- [Implementation Guide](#implementation-guide)
- [Testing Strategy](#testing-strategy)

---

## Overview

### Design Principles

1. **Fail Fast, Recover Smart**: Detect errors quickly, classify correctly, retry intelligently
2. **Observability First**: All errors logged with context, tracked centrally, alerted appropriately
3. **User-Centric**: Technical errors translated to actionable user messages
4. **Isolation-Aware**: Errors in one component don't cascade to others

### Key Goals

- ✅ **Classify** all errors into actionable categories
- ✅ **Retry** transient failures automatically with backoff
- ✅ **Propagate** errors to appropriate handlers (user, ops team, logs)
- ✅ **Track** errors centrally for debugging and analytics
- ✅ **Communicate** errors clearly to users with suggested actions

---

## Error Taxonomy

### 1. Error Classification System

All errors in the platform are classified into **4 primary types**:

```python
# backend/app/core/errors.py
from enum import Enum
from typing import Optional, Dict, Any

class ErrorType(str, Enum):
    """Error classification for automated handling"""

    TRANSIENT = "transient"      # Retry automatically (network glitch, service busy)
    PERMANENT = "permanent"      # Don't retry (invalid config, not found)
    USER_ERROR = "user_error"    # Notify user (validation failed, quota exceeded)
    SYSTEM_ERROR = "system_error" # Alert ops team (bug, database corruption)

class ErrorSeverity(str, Enum):
    """Impact severity for alerting"""

    LOW = "low"           # Informational, no action needed
    MEDIUM = "medium"     # Warning, investigate later
    HIGH = "high"         # Error, investigate soon
    CRITICAL = "critical" # Failure, immediate action required

class PlatformError(Exception):
    """Base exception for all platform errors"""

    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_message: Optional[str] = None,
        suggested_action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.user_message = user_message or self._default_user_message()
        self.suggested_action = suggested_action
        self.context = context or {}
        self.original_exception = original_exception

    def _default_user_message(self) -> str:
        """Generate user-friendly message based on error type"""
        messages = {
            ErrorType.TRANSIENT: "A temporary issue occurred. Please try again.",
            ErrorType.PERMANENT: "Unable to complete the operation. Please check your configuration.",
            ErrorType.USER_ERROR: "There was a problem with your request. Please review and try again.",
            ErrorType.SYSTEM_ERROR: "An unexpected error occurred. Our team has been notified.",
        }
        return messages.get(self.error_type, "An error occurred.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error for API responses and logging"""
        return {
            "error": {
                "type": self.error_type.value,
                "severity": self.severity.value,
                "message": self.message,
                "user_message": self.user_message,
                "suggested_action": self.suggested_action,
                "context": self.context,
            }
        }
```

### 2. Specific Error Classes

```python
# backend/app/core/errors.py

# ============================================================================
# TRANSIENT ERRORS (Auto-retry)
# ============================================================================

class NetworkError(PlatformError):
    """Network connectivity issues"""
    def __init__(self, service: str, original: Exception):
        super().__init__(
            message=f"Network error communicating with {service}",
            error_type=ErrorType.TRANSIENT,
            severity=ErrorSeverity.MEDIUM,
            user_message=f"Unable to connect to {service}. Retrying...",
            suggested_action="Please wait while we retry the connection.",
            context={"service": service},
            original_exception=original,
        )

class ServiceUnavailableError(PlatformError):
    """External service temporarily unavailable"""
    def __init__(self, service: str, retry_after: Optional[int] = None):
        super().__init__(
            message=f"{service} is temporarily unavailable",
            error_type=ErrorType.TRANSIENT,
            severity=ErrorSeverity.HIGH,
            user_message=f"{service} is currently busy. We'll retry automatically.",
            suggested_action=f"Wait {retry_after}s" if retry_after else "Please try again later",
            context={"service": service, "retry_after": retry_after},
        )

class TimeoutError(PlatformError):
    """Operation timed out (may be transient or permanent)"""
    def __init__(self, operation: str, timeout_seconds: int, is_retryable: bool = True):
        super().__init__(
            message=f"{operation} timed out after {timeout_seconds}s",
            error_type=ErrorType.TRANSIENT if is_retryable else ErrorType.PERMANENT,
            severity=ErrorSeverity.MEDIUM,
            user_message=f"The operation took too long and was cancelled.",
            suggested_action="Please try again. If the issue persists, try reducing dataset size or model complexity.",
            context={"operation": operation, "timeout_seconds": timeout_seconds},
        )

# ============================================================================
# PERMANENT ERRORS (Don't retry)
# ============================================================================

class ConfigurationError(PlatformError):
    """Invalid configuration that won't work with retries"""
    def __init__(self, field: str, issue: str):
        super().__init__(
            message=f"Invalid configuration: {field} - {issue}",
            error_type=ErrorType.PERMANENT,
            severity=ErrorSeverity.MEDIUM,
            user_message=f"Configuration error: {issue}",
            suggested_action=f"Please correct the '{field}' field in your configuration.",
            context={"field": field, "issue": issue},
        )

class ResourceNotFoundError(PlatformError):
    """Requested resource does not exist"""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            error_type=ErrorType.PERMANENT,
            severity=ErrorSeverity.LOW,
            user_message=f"The {resource_type} you requested does not exist.",
            suggested_action=f"Please check the {resource_type} ID and try again.",
            context={"resource_type": resource_type, "resource_id": resource_id},
        )

class InsufficientPermissionsError(PlatformError):
    """User lacks required permissions"""
    def __init__(self, action: str, required_role: str):
        super().__init__(
            message=f"Insufficient permissions for {action}. Required role: {required_role}",
            error_type=ErrorType.PERMANENT,
            severity=ErrorSeverity.LOW,
            user_message=f"You don't have permission to {action}.",
            suggested_action=f"Contact your administrator to request '{required_role}' role access.",
            context={"action": action, "required_role": required_role},
        )

# ============================================================================
# USER ERRORS (Validation, quota, business logic)
# ============================================================================

class ValidationError(PlatformError):
    """User input validation failed"""
    def __init__(self, field: str, constraint: str, value: Any):
        super().__init__(
            message=f"Validation failed: {field} {constraint} (got: {value})",
            error_type=ErrorType.USER_ERROR,
            severity=ErrorSeverity.LOW,
            user_message=f"Invalid value for {field}: {constraint}",
            suggested_action=f"Please provide a valid value for '{field}'.",
            context={"field": field, "constraint": constraint, "value": value},
        )

class QuotaExceededError(PlatformError):
    """User exceeded resource quota"""
    def __init__(self, resource: str, limit: int, current: int):
        super().__init__(
            message=f"Quota exceeded for {resource}: {current}/{limit}",
            error_type=ErrorType.USER_ERROR,
            severity=ErrorSeverity.MEDIUM,
            user_message=f"You've reached your limit of {limit} {resource}.",
            suggested_action=f"Please delete unused {resource} or upgrade your plan.",
            context={"resource": resource, "limit": limit, "current": current},
        )

class DatasetValidationError(PlatformError):
    """Dataset format or content validation failed"""
    def __init__(self, issue: str, details: Dict[str, Any]):
        super().__init__(
            message=f"Dataset validation failed: {issue}",
            error_type=ErrorType.USER_ERROR,
            severity=ErrorSeverity.MEDIUM,
            user_message=f"Dataset validation error: {issue}",
            suggested_action="Please check your dataset format and try uploading again.",
            context={"issue": issue, **details},
        )

# ============================================================================
# SYSTEM ERRORS (Bugs, infrastructure issues)
# ============================================================================

class DatabaseError(PlatformError):
    """Database operation failed"""
    def __init__(self, operation: str, original: Exception):
        super().__init__(
            message=f"Database error during {operation}: {str(original)}",
            error_type=ErrorType.SYSTEM_ERROR,
            severity=ErrorSeverity.CRITICAL,
            user_message="A system error occurred. Our team has been notified.",
            suggested_action="Please try again later. If the issue persists, contact support.",
            context={"operation": operation},
            original_exception=original,
        )

class StorageError(PlatformError):
    """S3/storage operation failed"""
    def __init__(self, operation: str, bucket: str, key: str, original: Exception):
        super().__init__(
            message=f"Storage error during {operation}: {bucket}/{key}",
            error_type=ErrorType.SYSTEM_ERROR,
            severity=ErrorSeverity.HIGH,
            user_message="Unable to access storage. Our team has been notified.",
            suggested_action="Please try again later.",
            context={"operation": operation, "bucket": bucket, "key": key},
            original_exception=original,
        )

class UnexpectedError(PlatformError):
    """Uncaught exception (bug)"""
    def __init__(self, original: Exception):
        super().__init__(
            message=f"Unexpected error: {type(original).__name__}: {str(original)}",
            error_type=ErrorType.SYSTEM_ERROR,
            severity=ErrorSeverity.CRITICAL,
            user_message="An unexpected error occurred. Our team has been notified.",
            suggested_action="Please contact support if the issue persists.",
            original_exception=original,
        )
```

---

## Retry Policies

### 1. Retry Configuration

```python
# backend/app/core/retry.py
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Retry Decorators by Service
# ============================================================================

def retry_on_transient_error():
    """Retry on transient errors with exponential backoff"""
    return retry(
        retry=retry_if_exception_type((NetworkError, ServiceUnavailableError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

def retry_trainer_service():
    """Retry policy for trainer service calls"""
    return retry(
        retry=retry_if_exception_type((NetworkError, ServiceUnavailableError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

def retry_mlflow():
    """Retry policy for MLflow API calls"""
    return retry(
        retry=retry_if_exception_type((NetworkError, ServiceUnavailableError)),
        stop=stop_after_attempt(5),  # More retries for logging (non-critical)
        wait=wait_exponential(multiplier=1, min=1, max=5),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=False,  # Don't fail training if MLflow is down
    )

def retry_s3():
    """Retry policy for S3 operations"""
    return retry(
        retry=retry_if_exception_type((NetworkError, ServiceUnavailableError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

def retry_temporal():
    """Retry policy for Temporal operations"""
    return retry(
        retry=retry_if_exception_type(NetworkError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        reraise=True,
    )
```

### 2. Circuit Breaker Pattern

```python
# backend/app/core/circuit_breaker.py
from datetime import datetime, timedelta
from typing import Callable, Optional
import asyncio

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise ServiceUnavailableError(
                    service=func.__name__,
                    retry_after=self.recovery_timeout,
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try resetting"""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)

    def _on_success(self):
        """Reset circuit breaker on successful call"""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Increment failure count and open circuit if threshold exceeded"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures. "
                f"Retry in {self.recovery_timeout}s"
            )

# Usage example
trainer_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=ServiceUnavailableError,
)
```

### 3. Integration-Specific Retry Policies

| Integration | Max Attempts | Initial Wait | Max Wait | Circuit Breaker |
|-------------|-------------|--------------|----------|-----------------|
| **Trainer Service** | 3 | 4s | 30s | Yes (5 failures, 60s) |
| **MLflow** | 5 | 1s | 5s | No (non-critical) |
| **S3 Storage** | 3 | 2s | 10s | No (highly available) |
| **Temporal** | 5 | 5s | 60s | Yes (5 failures, 120s) |
| **Database** | 2 | 1s | 5s | Yes (3 failures, 30s) |
| **LLM (Claude/GPT)** | 3 | 5s | 30s | Yes (5 failures, 300s) |

---

## Error Propagation Flow

### 1. Error Flow Architecture

```
Trainer → Backend → Frontend
   ↓         ↓          ↓
Logs    Database    User UI
   ↓         ↓
Sentry   Alerting
```

### 2. Trainer → Backend Error Propagation

**Scenario 1: Training Failure**

```python
# Trainer: trainers/ultralytics/train.py
try:
    model.train(...)
except torch.cuda.OutOfMemoryError as e:
    # Send error via callback
    requests.post(callback_url, json={
        "status": "failed",
        "error": {
            "type": "system_error",
            "code": "CUDA_OOM",
            "message": "GPU out of memory",
            "details": {
                "allocated_memory": torch.cuda.memory_allocated(),
                "max_memory": torch.cuda.max_memory_allocated(),
                "batch_size": config.batch_size,
            },
        },
    })
    sys.exit(1)

# Backend: app/api/training.py
@router.post("/training-jobs/{job_id}/callback")
async def training_callback(job_id: str, update: TrainingUpdate):
    if update.status == "failed":
        error_info = update.error

        # Classify error
        if error_info["code"] == "CUDA_OOM":
            platform_error = SystemError(
                message=f"GPU out of memory during training",
                severity=ErrorSeverity.HIGH,
                user_message="Training failed due to insufficient GPU memory.",
                suggested_action="Try reducing batch size or model size.",
                context=error_info["details"],
            )

        # Update database
        job = await db.get_job(job_id)
        job.status = JobStatus.FAILED
        job.error_message = platform_error.message
        job.error_details = platform_error.to_dict()
        await db.commit()

        # Send alert if critical
        if platform_error.severity == ErrorSeverity.CRITICAL:
            await alert_service.send_alert(platform_error)

        # Track in Sentry
        sentry_sdk.capture_exception(platform_error)
```

### 3. Backend → Frontend Error Propagation

**REST API Error Response Format:**

```python
# backend/app/core/responses.py
from fastapi import HTTPException
from fastapi.responses import JSONResponse

def error_response(error: PlatformError, status_code: int = 500) -> JSONResponse:
    """Standardized error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error.error_type.value,
                "severity": error.severity.value,
                "message": error.user_message,
                "suggested_action": error.suggested_action,
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": get_trace_id(),  # For debugging
            }
        },
    )

# Example API endpoint
@router.post("/training-jobs")
async def create_training_job(config: TrainingConfig):
    try:
        job = await training_service.create_job(config)
        return {"job_id": job.id}
    except QuotaExceededError as e:
        return error_response(e, status_code=429)
    except ValidationError as e:
        return error_response(e, status_code=400)
    except ConfigurationError as e:
        return error_response(e, status_code=422)
    except DatabaseError as e:
        sentry_sdk.capture_exception(e)
        return error_response(e, status_code=500)
```

**WebSocket Error Propagation:**

```python
# backend/app/websocket/training.py
@router.websocket("/ws/training-jobs/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()

    try:
        # Stream training updates
        async for update in training_stream(job_id):
            await websocket.send_json(update)
    except Exception as e:
        error = UnexpectedError(e)
        await websocket.send_json({
            "type": "error",
            "error": error.to_dict(),
        })
        await websocket.close(code=1011, reason=error.user_message)
```

### 4. Frontend Error Handling

```typescript
// frontend/src/services/api.ts
export class ApiError extends Error {
  type: string;
  severity: string;
  suggestedAction?: string;
  traceId?: string;

  constructor(response: ErrorResponse) {
    super(response.error.message);
    this.type = response.error.type;
    this.severity = response.error.severity;
    this.suggestedAction = response.error.suggested_action;
    this.traceId = response.error.trace_id;
  }
}

// frontend/src/hooks/useTrainingJob.ts
export function useTrainingJob(jobId: string) {
  const [error, setError] = useState<ApiError | null>(null);

  const handleError = (apiError: ApiError) => {
    setError(apiError);

    // Show user-friendly toast notification
    if (apiError.type === 'user_error') {
      toast.warning(apiError.message, {
        description: apiError.suggestedAction,
      });
    } else if (apiError.type === 'system_error') {
      toast.error('An unexpected error occurred', {
        description: 'Our team has been notified. Please try again later.',
      });
    }

    // Track in frontend error monitoring
    Sentry.captureException(apiError);
  };

  return { error, handleError };
}
```

---

## Error Tracking and Monitoring

### 1. Sentry Integration

```python
# backend/app/core/monitoring.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

def init_sentry():
    """Initialize Sentry for error tracking"""
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        environment=os.getenv("ENV_NAME"),  # subprocess, kind, k8s
        traces_sample_rate=0.1,  # 10% of transactions
        profiles_sample_rate=0.1,
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
        ],
        before_send=filter_sensitive_data,
    )

def filter_sensitive_data(event, hint):
    """Remove sensitive data from Sentry events"""
    # Remove password fields
    if 'request' in event:
        if 'data' in event['request']:
            data = event['request']['data']
            if isinstance(data, dict):
                data.pop('password', None)
                data.pop('api_key', None)
    return event

# Context enrichment
def capture_error_with_context(error: PlatformError, extra: Dict[str, Any] = None):
    """Capture error in Sentry with rich context"""
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("error_type", error.error_type.value)
        scope.set_tag("severity", error.severity.value)
        scope.set_level(error.severity.value)

        scope.set_context("error_details", {
            "user_message": error.user_message,
            "suggested_action": error.suggested_action,
            **error.context,
        })

        if extra:
            scope.set_context("additional_context", extra)

        sentry_sdk.capture_exception(error)
```

### 2. Error Metrics (Prometheus)

```python
# backend/app/core/metrics.py
from prometheus_client import Counter, Histogram

# Error counters by type
errors_total = Counter(
    'platform_errors_total',
    'Total errors by type and severity',
    ['error_type', 'severity', 'component']
)

# Training job failures
training_job_failures = Counter(
    'training_job_failures_total',
    'Training job failures by error code',
    ['error_code', 'framework']
)

# API error responses
api_errors = Counter(
    'api_errors_total',
    'API errors by status code and endpoint',
    ['status_code', 'endpoint', 'method']
)

# Error handling latency
error_handling_latency = Histogram(
    'error_handling_seconds',
    'Time spent handling errors',
    ['error_type']
)

# Usage in error handler
def track_error(error: PlatformError, component: str):
    """Track error in Prometheus"""
    errors_total.labels(
        error_type=error.error_type.value,
        severity=error.severity.value,
        component=component,
    ).inc()
```

### 3. Alert Rules

```yaml
# monitoring/alertmanager/rules.yml
groups:
  - name: error_alerts
    interval: 1m
    rules:
      # Critical system errors
      - alert: HighCriticalErrorRate
        expr: rate(platform_errors_total{severity="critical"}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High rate of critical errors"
          description: "{{ $value }} critical errors per second in the last 5 minutes"

      # Training job failures
      - alert: HighTrainingFailureRate
        expr: rate(training_job_failures_total[10m]) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High training job failure rate"
          description: "{{ $value }} training jobs failing per second"

      # API error spike
      - alert: APIErrorSpike
        expr: rate(api_errors_total{status_code="500"}[5m]) > 1
        for: 3m
        labels:
          severity: high
        annotations:
          summary: "API 500 error spike detected"
          description: "{{ $value }} 500 errors per second"

      # Database errors
      - alert: DatabaseErrors
        expr: platform_errors_total{error_type="system_error", component="database"} > 10
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database errors detected"
          description: "Multiple database errors in the last minute"
```

---

## User-Facing Error Messages

### 1. Error Message Guidelines

**Principles:**
- ✅ Clear and concise (under 100 characters)
- ✅ Explain WHAT happened (not technical details)
- ✅ Suggest WHAT TO DO (actionable)
- ✅ Avoid blame ("failed to upload" not "you failed")
- ✅ Provide context when helpful

### 2. Error Message Templates

```python
# backend/app/core/error_messages.py
ERROR_MESSAGES = {
    # Training errors
    "CUDA_OOM": {
        "message": "Training failed due to insufficient GPU memory.",
        "action": "Try reducing batch size (currently {batch_size}) or use a smaller model.",
    },
    "DATASET_NOT_FOUND": {
        "message": "The dataset you selected no longer exists.",
        "action": "Please select a different dataset or upload a new one.",
    },
    "INVALID_CONFIG": {
        "message": "Training configuration has errors: {details}",
        "action": "Please review your configuration and correct the highlighted fields.",
    },
    "TRAINER_UNREACHABLE": {
        "message": "Unable to start training at this time.",
        "action": "Our system is temporarily busy. Please try again in a few minutes.",
    },

    # Dataset errors
    "DATASET_INVALID_FORMAT": {
        "message": "Dataset format is invalid: {details}",
        "action": "Please ensure your dataset follows the {format} format specification.",
    },
    "DATASET_TOO_LARGE": {
        "message": "Dataset size ({size_gb}GB) exceeds your limit of {limit_gb}GB.",
        "action": "Delete unused datasets or upgrade your plan for more storage.",
    },

    # Permission errors
    "INSUFFICIENT_PERMISSIONS": {
        "message": "You don't have permission to {action}.",
        "action": "Contact your project owner to request access.",
    },
    "QUOTA_EXCEEDED": {
        "message": "You've reached your limit of {limit} {resource}.",
        "action": "Delete unused {resource} or upgrade to increase your limit.",
    },

    # System errors
    "SYSTEM_ERROR": {
        "message": "An unexpected error occurred (ID: {trace_id}).",
        "action": "Our team has been notified. Please try again or contact support with this error ID.",
    },
}

def format_error_message(error_code: str, **kwargs) -> tuple[str, str]:
    """Format error message with context"""
    template = ERROR_MESSAGES.get(error_code, ERROR_MESSAGES["SYSTEM_ERROR"])
    message = template["message"].format(**kwargs)
    action = template["action"].format(**kwargs)
    return message, action
```

### 3. Frontend Error Display

```typescript
// frontend/src/components/ErrorAlert.tsx
interface ErrorAlertProps {
  error: ApiError;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export function ErrorAlert({ error, onRetry, onDismiss }: ErrorAlertProps) {
  const getIcon = () => {
    switch (error.severity) {
      case 'critical': return <AlertCircle className="text-red-600" />;
      case 'high': return <AlertTriangle className="text-orange-600" />;
      case 'medium': return <Info className="text-yellow-600" />;
      default: return <Info className="text-blue-600" />;
    }
  };

  return (
    <Alert variant={error.severity === 'critical' ? 'destructive' : 'warning'}>
      {getIcon()}
      <AlertTitle>{error.message}</AlertTitle>
      <AlertDescription>
        {error.suggestedAction}
      </AlertDescription>
      <div className="mt-4 flex gap-2">
        {onRetry && error.type === 'transient' && (
          <Button onClick={onRetry} variant="outline" size="sm">
            Try Again
          </Button>
        )}
        <Button onClick={onDismiss} variant="ghost" size="sm">
          Dismiss
        </Button>
      </div>
    </Alert>
  );
}
```

---

## Implementation Guide

### 1. Backend Implementation Checklist

- [ ] Create error classes in `app/core/errors.py`
- [ ] Implement retry decorators in `app/core/retry.py`
- [ ] Add circuit breaker in `app/core/circuit_breaker.py`
- [ ] Integrate Sentry in `app/core/monitoring.py`
- [ ] Add Prometheus metrics in `app/core/metrics.py`
- [ ] Create error response handler in `app/core/responses.py`
- [ ] Update all API endpoints to use error classes
- [ ] Add error tracking to training callback handler
- [ ] Configure alert rules in Prometheus/Alertmanager

### 2. Trainer Implementation Checklist

- [ ] Catch all exceptions in training loop
- [ ] Send structured errors via callback
- [ ] Include error context (GPU memory, batch size, etc.)
- [ ] Exit with appropriate exit codes
- [ ] Log errors to stderr in JSON format

### 3. Frontend Implementation Checklist

- [ ] Create `ApiError` class
- [ ] Add error handling to API client
- [ ] Implement `ErrorAlert` component
- [ ] Add error boundaries for React components
- [ ] Integrate Sentry for frontend errors
- [ ] Show user-friendly error messages
- [ ] Provide retry buttons for transient errors

---

## Testing Strategy

### 1. Unit Tests

```python
# tests/test_errors.py
import pytest
from app.core.errors import *

def test_transient_error_classification():
    error = NetworkError(service="trainer", original=Exception("Connection refused"))
    assert error.error_type == ErrorType.TRANSIENT
    assert error.severity == ErrorSeverity.MEDIUM

def test_error_serialization():
    error = ValidationError(field="epochs", constraint="must be > 0", value=-5)
    error_dict = error.to_dict()
    assert error_dict["error"]["type"] == "user_error"
    assert "epochs" in error_dict["error"]["message"]

def test_retry_policy():
    @retry_on_transient_error()
    def flaky_function():
        # Simulate transient failure then success
        if not hasattr(flaky_function, 'attempts'):
            flaky_function.attempts = 0
        flaky_function.attempts += 1
        if flaky_function.attempts < 3:
            raise NetworkError(service="test", original=Exception())
        return "success"

    result = flaky_function()
    assert result == "success"
    assert flaky_function.attempts == 3
```

### 2. Integration Tests

```python
# tests/integration/test_error_handling.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_training_job_error_propagation(client: AsyncClient):
    """Test that trainer errors are properly propagated to frontend"""

    # Create training job
    response = await client.post("/api/v1/training-jobs", json={
        "model": "yolo11n",
        "dataset_id": "invalid-dataset",  # Will fail
        "epochs": 100,
    })
    job_id = response.json()["job_id"]

    # Wait for failure
    await asyncio.sleep(5)

    # Check job status
    response = await client.get(f"/api/v1/training-jobs/{job_id}")
    job = response.json()

    assert job["status"] == "failed"
    assert job["error"]["type"] == "permanent"
    assert "dataset" in job["error"]["message"].lower()
    assert job["error"]["suggested_action"] is not None

@pytest.mark.asyncio
async def test_transient_error_retry(client: AsyncClient, mock_trainer_service):
    """Test that transient errors are retried automatically"""

    # Mock trainer service to fail twice then succeed
    mock_trainer_service.fail_count = 2

    response = await client.post("/api/v1/training-jobs", json={
        "model": "yolo11n",
        "dataset_id": "test-dataset",
        "epochs": 100,
    })

    # Should succeed after retries
    assert response.status_code == 200
    assert mock_trainer_service.call_count == 3  # Initial + 2 retries
```

### 3. Chaos Testing

```python
# tests/chaos/test_failure_scenarios.py
import pytest
from chaos_toolkit import run_experiment

def test_trainer_service_down():
    """Test behavior when trainer service is unavailable"""
    experiment = {
        "title": "Trainer Service Down",
        "method": [
            {
                "type": "action",
                "name": "Stop trainer service",
                "provider": {
                    "type": "process",
                    "path": "docker",
                    "arguments": ["stop", "ultralytics-trainer"],
                },
            },
            {
                "type": "probe",
                "name": "Submit training job",
                "provider": {
                    "type": "http",
                    "url": "http://backend:8000/api/v1/training-jobs",
                    "method": "POST",
                    "expected_status": [503, 429],  # Service unavailable or rate limited
                },
            },
        ],
    }
    result = run_experiment(experiment)
    assert result["steady_state_met"]
```

---

## Appendix: Error Code Reference

### Training Errors

| Code | Type | Severity | Description |
|------|------|----------|-------------|
| `CUDA_OOM` | SYSTEM | HIGH | GPU out of memory |
| `TRAINER_UNREACHABLE` | TRANSIENT | MEDIUM | Cannot connect to trainer service |
| `TRAINING_TIMEOUT` | PERMANENT | MEDIUM | Training exceeded max time limit |
| `INVALID_MODEL` | PERMANENT | LOW | Model name not supported |
| `INVALID_CONFIG` | PERMANENT | LOW | Training configuration invalid |

### Dataset Errors

| Code | Type | Severity | Description |
|------|------|----------|-------------|
| `DATASET_NOT_FOUND` | PERMANENT | LOW | Dataset does not exist |
| `DATASET_INVALID_FORMAT` | USER_ERROR | MEDIUM | Dataset format validation failed |
| `DATASET_TOO_LARGE` | USER_ERROR | MEDIUM | Dataset exceeds size quota |
| `DATASET_BROKEN` | PERMANENT | HIGH | Dataset integrity check failed |

### Permission Errors

| Code | Type | Severity | Description |
|------|------|----------|-------------|
| `INSUFFICIENT_PERMISSIONS` | PERMANENT | LOW | User lacks required role |
| `QUOTA_EXCEEDED` | USER_ERROR | MEDIUM | Resource quota exceeded |
| `INVALID_TOKEN` | PERMANENT | LOW | Authentication token invalid/expired |

### System Errors

| Code | Type | Severity | Description |
|------|------|----------|-------------|
| `DATABASE_ERROR` | SYSTEM | CRITICAL | Database operation failed |
| `STORAGE_ERROR` | SYSTEM | HIGH | S3 storage operation failed |
| `MLFLOW_UNAVAILABLE` | TRANSIENT | LOW | MLflow server unreachable |
| `TEMPORAL_ERROR` | SYSTEM | CRITICAL | Temporal workflow error |

---

**End of Document**
