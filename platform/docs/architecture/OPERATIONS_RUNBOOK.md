# OPERATIONS_RUNBOOK.md

**작성일**: 2025-01-11
**대상**: Platform Operations Team, SRE, On-Call Engineers
**목적**: Vision AI Training Platform 운영 중 발생하는 장애 대응, 일상 운영 절차, 긴급 복구 가이드

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Health Check and Monitoring](#health-check-and-monitoring)
3. [Incident Response Procedures](#incident-response-procedures)
4. [Common Issues and Solutions](#common-issues-and-solutions)
5. [Restart and Upgrade Procedures](#restart-and-upgrade-procedures)
6. [On-Call Playbook](#on-call-playbook)
7. [Disaster Recovery](#disaster-recovery)
8. [Appendix](#appendix)

---

## 1. System Overview

### 1.1 System Architecture

```
┌──────────────┐
│   Frontend   │ (Next.js on Port 3000)
└──────┬───────┘
       │
┌──────▼───────┐
│   Backend    │ (FastAPI on Port 8000)
└──┬───┬───┬───┘
   │   │   │
   │   │   └──────────────────┐
   │   │                      │
┌──▼───▼────┐  ┌─────────┐  ┌▼────────┐
│ Trainer   │  │ MLflow  │  │Temporal │
│ Services  │  │ Server  │  │ Server  │
└───────────┘  └─────────┘  └─────────┘
   │
┌──▼──────────┐
│ PostgreSQL  │
└─────────────┘
   │
┌──▼──────────┐
│  S3/MinIO   │
└─────────────┘
```

**Critical Services**:
- **Backend** (FastAPI): API Gateway, core business logic
- **Trainer Services**: Framework-specific training (timm, ultralytics, huggingface)
- **PostgreSQL**: Source of truth for all job metadata
- **S3/MinIO**: Dataset and checkpoint storage
- **MLflow**: Experiment tracking (non-critical, has offline mode)
- **Temporal**: Workflow orchestration (optional, callback-first pattern)

**Dependency Chain**:
```
Frontend → Backend → Database (critical)
                  ↓
                  Trainer Services (critical)
                  ↓
                  S3 Storage (critical)
                  ↓
                  MLflow (non-critical)
                  ↓
                  Temporal (non-critical)
```

### 1.2 Service URLs and Ports

**Production**:
- Frontend: https://vision-ai.example.com
- Backend API: https://api.vision-ai.example.com
- MLflow UI: https://mlflow.vision-ai.example.com
- Temporal UI: https://temporal.vision-ai.example.com (internal)
- Prometheus: https://metrics.vision-ai.example.com (internal)
- Grafana: https://grafana.vision-ai.example.com (internal)

**Local Development**:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- MLflow: http://localhost:5000
- Temporal UI: http://localhost:8233
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

### 1.3 Data Flow

**Training Job Lifecycle**:
1. User creates job via Frontend → Backend API
2. Backend validates config → Saves to PostgreSQL
3. Backend calls Trainer Service `/training/start` with callback URL
4. Trainer downloads dataset from S3 → Starts training
5. Trainer sends progress updates to Backend callback endpoint
6. Backend updates database → Sends WebSocket update to Frontend
7. Trainer uploads checkpoints to S3
8. Trainer sends final results to Backend callback
9. Backend marks job as completed

**Critical Path**: Frontend → Backend → Database → Trainer → S3

---

## 2. Health Check and Monitoring

### 2.1 Health Check Endpoints

**Backend Health Checks**:

```bash
# Liveness probe (is process alive?)
curl http://localhost:8000/health

# Expected response:
{"status": "healthy", "timestamp": "2025-01-11T10:30:00Z"}

# Readiness probe (ready to serve traffic?)
curl http://localhost:8000/health/ready

# Expected response:
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "s3": "ok",
    "trainer_services": "ok",
    "mlflow": "degraded"  # non-critical
  }
}

# Detailed health check (for debugging)
curl http://localhost:8000/health/detailed

# Expected response:
{
  "status": "healthy",
  "components": {
    "database": {
      "status": "ok",
      "latency_ms": 5,
      "pool_size": 20,
      "active_connections": 8
    },
    "s3": {
      "status": "ok",
      "latency_ms": 120,
      "bucket_accessible": true
    },
    "trainer_services": {
      "timm": {"status": "ok", "latency_ms": 15},
      "ultralytics": {"status": "ok", "latency_ms": 12},
      "huggingface": {"status": "degraded", "error": "timeout"}
    },
    "mlflow": {
      "status": "offline",
      "offline_mode_enabled": true,
      "buffered_metrics": 245
    },
    "temporal": {
      "status": "ok",
      "active_workflows": 3
    }
  }
}
```

**Trainer Service Health Checks**:

```bash
# Timm service
curl http://localhost:8001/health

# Ultralytics service
curl http://localhost:8002/health

# HuggingFace service
curl http://localhost:8003/health

# Expected response from each:
{
  "status": "healthy",
  "service": "timm-trainer",
  "version": "1.0.0",
  "gpu_available": true,
  "active_jobs": 2,
  "max_concurrent_jobs": 4
}
```

**MLflow Health Check**:

```bash
curl http://localhost:5000/health

# If MLflow is down, Backend should continue in offline mode
```

**Temporal Health Check**:

```bash
curl http://localhost:7233/health

# Backend should work even if Temporal is down (callback-first pattern)
```

### 2.2 Monitoring Dashboards

**Prometheus Metrics**:

```bash
# Query active training jobs
http://localhost:9090/graph?g0.expr=training_jobs_active

# Query error rates
http://localhost:9090/graph?g0.expr=rate(http_requests_total{status=~"5.."}[5m])

# Query circuit breaker status
http://localhost:9090/graph?g0.expr=circuit_breaker_state{service="trainer"}
```

**Key Metrics to Monitor**:

| Metric | Alert Threshold | Description |
|--------|----------------|-------------|
| `http_request_duration_seconds` | p95 > 5s | API response time |
| `training_jobs_failed_total` | rate > 0.1/min | Job failure rate |
| `database_connection_pool_active` | > 18/20 | DB connection exhaustion |
| `s3_upload_errors_total` | > 0 | Storage upload failures |
| `circuit_breaker_state{state="open"}` | > 0 | Service degradation |
| `mlflow_offline_mode` | 1 | MLflow unavailable |
| `job_status_inconsistency_total` | > 0 | State sync issues |

**Grafana Dashboards**:

1. **Platform Overview Dashboard**:
   - Active jobs count
   - Error rate (last 1h, 24h)
   - Service health status
   - Resource utilization (CPU, Memory, GPU)

2. **Training Jobs Dashboard**:
   - Jobs by status (pending, running, completed, failed)
   - Average job duration
   - Job success rate
   - Training metrics (loss, accuracy)

3. **Infrastructure Dashboard**:
   - Database connection pool utilization
   - S3 bandwidth and latency
   - Circuit breaker states
   - Integration timeout rates

### 2.3 Log Locations

**Backend Logs**:
```bash
# Production (Docker/K8s)
kubectl logs deployment/backend -n vision-ai --tail=100 -f

# Local development
tail -f /var/log/vision-ai/backend.log

# Log structure
{
  "timestamp": "2025-01-11T10:30:00Z",
  "level": "ERROR",
  "service": "backend",
  "request_id": "req_abc123",
  "user_id": "user_123",
  "message": "Failed to start training job",
  "error": {
    "type": "NetworkError",
    "service": "timm-trainer",
    "original": "Connection refused"
  },
  "context": {
    "job_id": "job_456",
    "retry_count": 2
  }
}
```

**Trainer Service Logs**:
```bash
# Timm trainer
kubectl logs deployment/timm-trainer -n vision-ai --tail=100 -f

# Ultralytics trainer
kubectl logs deployment/ultralytics-trainer -n vision-ai --tail=100 -f
```

**Application Log Levels**:
- **DEBUG**: Detailed debugging info (only in development)
- **INFO**: Normal operations (job started, completed)
- **WARNING**: Degraded state (MLflow offline, retry attempts)
- **ERROR**: Operation failed (job failed, integration timeout)
- **CRITICAL**: System-wide failure (database down, cannot start jobs)

**Structured Logging Query Examples**:

```bash
# Find all errors for a specific job
cat backend.log | jq 'select(.context.job_id == "job_456" and .level == "ERROR")'

# Find all network errors in the last hour
cat backend.log | jq 'select(.error.type == "NetworkError" and .timestamp > "2025-01-11T09:30:00Z")'

# Count errors by service
cat backend.log | jq -r 'select(.level == "ERROR") | .error.service' | sort | uniq -c
```

### 2.4 Alerting Rules

**Critical Alerts (P0 - Immediate Response)**:

```yaml
# Backend service down
alert: BackendServiceDown
expr: up{job="backend"} == 0
for: 1m
severity: critical
description: "Backend service is not responding"
action: "Check backend logs, restart backend deployment"

# Database unreachable
alert: DatabaseUnreachable
expr: database_connection_errors_total > 0
for: 30s
severity: critical
description: "Cannot connect to PostgreSQL database"
action: "Check database status, check network connectivity"

# S3 storage unavailable
alert: S3StorageUnavailable
expr: s3_health_check_failures_total > 0
for: 1m
severity: critical
description: "Cannot access S3 storage"
action: "Check S3 service status, check credentials"

# High error rate
alert: HighErrorRate
expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
for: 5m
severity: critical
description: "Error rate exceeds 10%"
action: "Check error logs, investigate failing endpoints"
```

**Warning Alerts (P1 - Response within 30min)**:

```yaml
# Trainer service degraded
alert: TrainerServiceDegraded
expr: circuit_breaker_state{service="trainer", state="open"} == 1
for: 5m
severity: warning
description: "Trainer service circuit breaker is open"
action: "Check trainer service health, investigate recent errors"

# High database connection usage
alert: HighDatabaseConnectionUsage
expr: database_connection_pool_active / database_connection_pool_size > 0.9
for: 10m
severity: warning
description: "Database connection pool near capacity"
action: "Investigate slow queries, consider increasing pool size"

# MLflow offline mode
alert: MLflowOfflineMode
expr: mlflow_offline_mode == 1
for: 15m
severity: warning
description: "MLflow is unavailable, running in offline mode"
action: "Check MLflow service, metrics are being buffered"
```

**Info Alerts (P2 - Response within 24h)**:

```yaml
# High job failure rate
alert: HighJobFailureRate
expr: rate(training_jobs_failed_total[1h]) > 0.05
for: 1h
severity: info
description: "Job failure rate exceeds 5%"
action: "Investigate failed jobs, check for common patterns"
```

---

## 3. Incident Response Procedures

### 3.1 Incident Severity Levels

| Severity | Response Time | Escalation | Examples |
|----------|--------------|------------|----------|
| **P0 (Critical)** | Immediate (< 5min) | Page on-call + manager | Backend down, database down, all jobs failing |
| **P1 (High)** | 30 minutes | Page on-call | Single service degraded, high error rate |
| **P2 (Medium)** | 2 hours | Email on-call | MLflow down, specific job types failing |
| **P3 (Low)** | Next business day | Email team | Performance degradation, non-critical feature issues |

### 3.2 Incident: Backend Service Unresponsive

**Symptoms**:
- Health check endpoint `/health` returns 5xx or times out
- Frontend shows "Cannot connect to server"
- Prometheus alert `BackendServiceDown` firing

**Investigation Steps**:

```bash
# Step 1: Check if backend process is running
kubectl get pods -n vision-ai | grep backend

# Step 2: Check backend logs
kubectl logs deployment/backend -n vision-ai --tail=100

# Step 3: Check resource utilization
kubectl top pod -n vision-ai | grep backend

# Step 4: Check database connectivity
psql -h db.vision-ai.internal -U admin -d vision_platform -c "SELECT 1;"

# Step 5: Check application logs for errors
kubectl logs deployment/backend -n vision-ai | grep -i "error\|critical"
```

**Common Causes and Solutions**:

| Cause | Detection | Solution |
|-------|-----------|----------|
| **OOM (Out of Memory)** | `OOMKilled` in pod status | Increase memory limits, investigate memory leak |
| **Database connection pool exhausted** | Logs show `QueuePool limit exceeded` | Restart backend, increase pool size in config |
| **Crash loop** | Pod restarts repeatedly | Check logs for unhandled exceptions, rollback if recent deploy |
| **Deadlock** | Process alive but unresponsive | Get thread dump, restart backend |
| **Dependency failure** | Health check shows database/S3 down | Fix dependency first, backend auto-recovers |

**Resolution Procedure**:

```bash
# Quick restart (if no data loss risk)
kubectl rollout restart deployment/backend -n vision-ai

# Watch rollout progress
kubectl rollout status deployment/backend -n vision-ai

# Verify health after restart
curl https://api.vision-ai.example.com/health

# Check if jobs resumed
curl https://api.vision-ai.example.com/api/v1/training-jobs?status=running
```

**Post-Incident**:
1. Check Sentry for crash reports
2. Review logs for root cause
3. Create incident report
4. If caused by recent deploy, rollback and investigate before redeploying

### 3.3 Incident: Training Job Stuck

**Symptoms**:
- Job status remains `running` for > 24 hours
- No progress updates received
- User reports job not completing

**Investigation Steps**:

```bash
# Step 1: Check job details in database
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT id, status, created_at, updated_at, trainer_job_id FROM training_jobs WHERE id = 'job_123';"

# Step 2: Check if trainer service received the job
curl http://timm-trainer:8001/training/status/trainer_job_456

# Step 3: Check trainer logs for the job
kubectl logs deployment/timm-trainer -n vision-ai | grep "job_456"

# Step 4: Check for callback failures
cat backend.log | jq 'select(.context.job_id == "job_123" and .message | contains("callback"))'

# Step 5: Check Temporal workflow status (if using Temporal)
temporal workflow describe --workflow-id job_123 --namespace vision-ai
```

**Common Causes and Solutions**:

| Cause | Detection | Solution |
|-------|-----------|----------|
| **Trainer crashed silently** | Trainer service has no record of job | Restart job, improve trainer error handling |
| **Callback URL unreachable** | Trainer logs show callback errors | Check network connectivity, verify callback URL |
| **Job legitimately long-running** | Logs show normal progress | No action, inform user of expected duration |
| **GPU hang** | GPU utilization 0% but job running | Kill job, restart trainer pod, investigate GPU issue |
| **Dataset download stuck** | Logs show S3 download timeout | Cancel job, check S3 connectivity, retry |

**Resolution Procedure**:

```bash
# Option 1: Stop job gracefully
curl -X POST https://api.vision-ai.example.com/api/v1/training-jobs/job_123/stop

# Option 2: Force stop via trainer service
curl -X POST http://timm-trainer:8001/training/stop/trainer_job_456

# Option 3: Kill trainer pod (if unresponsive)
kubectl delete pod <trainer-pod-name> -n vision-ai

# Option 4: Mark job as failed in database (last resort)
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "UPDATE training_jobs SET status = 'failed', error_message = 'Job stuck, manually terminated' WHERE id = 'job_123';"

# Verify job status updated
curl https://api.vision-ai.example.com/api/v1/training-jobs/job_123
```

**Prevention**:
1. Implement job timeout (24h max)
2. Add heartbeat mechanism (every 5min)
3. Auto-terminate jobs without progress updates

### 3.4 Incident: MLflow Server Down

**Symptoms**:
- MLflow UI unreachable at https://mlflow.vision-ai.example.com
- Backend logs show `MLflow offline mode enabled`
- Prometheus alert `MLflowOfflineMode` firing

**Impact Assessment**:
- **Severity**: P2 (Medium) - Non-critical service
- **User Impact**: Users cannot view training metrics in real-time
- **System Behavior**: Backend buffers metrics, training continues normally

**Investigation Steps**:

```bash
# Step 1: Check MLflow service status
kubectl get pods -n vision-ai | grep mlflow

# Step 2: Check MLflow logs
kubectl logs deployment/mlflow -n vision-ai --tail=100

# Step 3: Check MLflow database (PostgreSQL tracking store)
psql -h db.vision-ai.internal -U mlflow -d mlflow_tracking -c "SELECT COUNT(*) FROM experiments;"

# Step 4: Check MLflow artifact store (S3)
aws s3 ls s3://vision-ai-mlflow-artifacts/

# Step 5: Verify backend offline mode
curl https://api.vision-ai.example.com/health/detailed | jq '.components.mlflow'
```

**Common Causes and Solutions**:

| Cause | Detection | Solution |
|-------|-----------|----------|
| **MLflow pod crashed** | Pod status `CrashLoopBackOff` | Check logs, restart pod |
| **Database connection issue** | Logs show `Cannot connect to database` | Check MLflow database credentials |
| **S3 artifact store issue** | Logs show S3 access errors | Check S3 credentials, bucket permissions |
| **Resource limits** | Pod OOMKilled | Increase memory limits |

**Resolution Procedure**:

```bash
# Step 1: Restart MLflow service
kubectl rollout restart deployment/mlflow -n vision-ai

# Step 2: Wait for MLflow to be ready
kubectl wait --for=condition=ready pod -l app=mlflow -n vision-ai --timeout=300s

# Step 3: Verify MLflow is accessible
curl http://mlflow:5000/health

# Step 4: Trigger metric flush from backend
curl -X POST https://api.vision-ai.example.com/internal/mlflow/flush-buffer

# Step 5: Verify metrics appeared in MLflow UI
open https://mlflow.vision-ai.example.com
```

**Post-Recovery**:
1. Backend auto-flushes buffered metrics to MLflow
2. Check MLflow UI to verify metrics are visible
3. Inform users that metrics are now available

**User Communication Template**:
```
Subject: MLflow Service Restored

The MLflow tracking service is now available. Training metrics that were
collected during the outage have been uploaded and are now visible in the
MLflow UI.

Training jobs were not affected - all jobs continued running normally during
the MLflow outage.
```

### 3.5 Incident: Database Connection Issues

**Symptoms**:
- Backend health check shows database unhealthy
- Logs show `QueuePool limit exceeded` or `FATAL: too many connections`
- High latency on all API endpoints
- Prometheus alert `HighDatabaseConnectionUsage` or `DatabaseUnreachable`

**Severity**: P0 (Critical) - Database is critical dependency

**Investigation Steps**:

```bash
# Step 1: Check database server status
pg_isready -h db.vision-ai.internal -U admin

# Step 2: Check current connection count
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# Step 3: Check connection limits
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT setting FROM pg_settings WHERE name = 'max_connections';"

# Step 4: Identify connection hogs
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT pid, usename, application_name, state, query_start, state_change \
   FROM pg_stat_activity WHERE state = 'idle in transaction' ORDER BY query_start;"

# Step 5: Check for long-running queries
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT pid, now() - pg_stat_activity.query_start AS duration, query \
   FROM pg_stat_activity WHERE state = 'active' ORDER BY duration DESC LIMIT 10;"

# Step 6: Check backend connection pool status
curl https://api.vision-ai.example.com/health/detailed | jq '.components.database'
```

**Common Causes and Solutions**:

| Cause | Detection | Solution |
|-------|-----------|----------|
| **Connection leak** | Active connections keep increasing | Restart backend, fix connection handling in code |
| **Long-running queries** | Queries running > 5min | Kill slow queries, optimize query or add timeout |
| **Too many backend replicas** | Connection count = replicas × pool_size | Reduce replicas or pool_size |
| **Database server overload** | CPU/Memory at 100% | Scale database, optimize queries |
| **Idle transactions** | Many `idle in transaction` connections | Kill idle connections, fix application code |

**Emergency Resolution**:

```bash
# Option 1: Kill idle connections (safe)
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
   WHERE state = 'idle in transaction' AND state_change < now() - interval '5 minutes';"

# Option 2: Kill specific long-running query
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT pg_terminate_backend(<pid>);"

# Option 3: Restart backend to reset connection pool
kubectl rollout restart deployment/backend -n vision-ai

# Option 4: Increase database max_connections (temporary)
# Edit postgresql.conf
max_connections = 200  # increase from 100
# Restart PostgreSQL (causes brief downtime)
kubectl rollout restart statefulset/postgres -n vision-ai
```

**Permanent Fix**:

1. **Adjust Backend Connection Pool**:
```python
# backend/app/db/database.py
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,        # reduce from 20
    max_overflow=5,      # reduce from 10
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
)
```

2. **Add Connection Pool Monitoring**:
```python
@app.get("/metrics")
async def metrics():
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "checked_in": pool.checkedin(),
    }
```

3. **Set Statement Timeout**:
```sql
ALTER DATABASE vision_platform SET statement_timeout = '30s';
```

### 3.6 Incident: S3 Storage Failures

**Symptoms**:
- Dataset upload fails with "Cannot upload to S3"
- Checkpoint upload fails during training
- Backend health check shows S3 unhealthy
- Logs show `S3UploadError` or `ClientError: NoSuchBucket`

**Severity**: P0 (Critical) - Storage is critical dependency

**Investigation Steps**:

```bash
# Step 1: Check S3 service status (AWS)
aws s3 ls s3://vision-ai-datasets/ --profile production

# Step 2: Check MinIO status (local/private cloud)
mc admin info minio-server

# Step 3: Check backend S3 health
curl https://api.vision-ai.example.com/health/detailed | jq '.components.s3'

# Step 4: Verify credentials
aws sts get-caller-identity --profile production

# Step 5: Check bucket permissions
aws s3api get-bucket-acl --bucket vision-ai-datasets --profile production

# Step 6: Check for quota exceeded
aws s3api list-buckets --profile production
# Check total storage used
```

**Common Causes and Solutions**:

| Cause | Detection | Solution |
|-------|-----------|----------|
| **Invalid credentials** | `InvalidAccessKeyId` error | Rotate credentials, update in backend config |
| **Bucket doesn't exist** | `NoSuchBucket` error | Create bucket or fix bucket name in config |
| **Permission denied** | `AccessDenied` error | Update IAM policy, grant s3:PutObject permission |
| **Network timeout** | Upload times out after 60s | Check network connectivity, increase timeout |
| **Quota exceeded** | Bucket size limit reached | Delete old data, increase quota |
| **Multipart upload incomplete** | Orphaned multipart uploads | Clean up incomplete uploads |

**Resolution Procedure**:

```bash
# Fix 1: Verify bucket exists and is accessible
aws s3 mb s3://vision-ai-datasets --region us-west-2 --profile production

# Fix 2: Update bucket policy to allow uploads
cat > bucket-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::123456789012:role/vision-ai-backend"},
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::vision-ai-datasets/*"
    }
  ]
}
EOF
aws s3api put-bucket-policy --bucket vision-ai-datasets --policy file://bucket-policy.json

# Fix 3: Clean up incomplete multipart uploads (reclaim space)
aws s3api list-multipart-uploads --bucket vision-ai-datasets --profile production
aws s3api abort-multipart-upload --bucket vision-ai-datasets --key <key> --upload-id <id>

# Fix 4: Rotate credentials
aws iam create-access-key --user-name vision-ai-backend
# Update backend deployment with new credentials
kubectl create secret generic s3-credentials \
  --from-literal=access-key-id=NEW_KEY \
  --from-literal=secret-access-key=NEW_SECRET \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment/backend -n vision-ai

# Fix 5: Verify uploads work
curl -X POST https://api.vision-ai.example.com/api/v1/datasets/upload \
  -F "file=@test-image.jpg" \
  -H "Authorization: Bearer $TOKEN"
```

**Prevention**:
1. Set up S3 bucket monitoring (CloudWatch alarms)
2. Implement automatic cleanup of old datasets
3. Use lifecycle policies to move old data to Glacier
4. Set up quota alerts

### 3.7 Incident: GPU Node Unresponsive

**Symptoms**:
- Training jobs stuck at 0% progress
- GPU utilization shows 0% in monitoring
- Trainer pod logs show CUDA errors
- nvidia-smi command hangs or times out

**Severity**: P1 (High) - Affects new training jobs

**Investigation Steps**:

```bash
# Step 1: Check GPU node status
kubectl get nodes -l gpu=true

# Step 2: Check NVIDIA driver on node
kubectl exec -it <trainer-pod> -- nvidia-smi

# Step 3: Check for GPU errors in dmesg
kubectl exec -it <trainer-pod> -- dmesg | grep -i nvidia

# Step 4: Check GPU memory usage
kubectl exec -it <trainer-pod> -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Step 5: Check for zombie processes holding GPU
kubectl exec -it <trainer-pod> -- fuser -v /dev/nvidia*
```

**Common Causes and Solutions**:

| Cause | Detection | Solution |
|-------|-----------|----------|
| **GPU driver crash** | nvidia-smi hangs | Restart node (drains jobs first) |
| **CUDA OOM** | `CUDA out of memory` in logs | Kill GPU process, reduce batch size |
| **GPU process hung** | Process stuck, GPU 0% | Kill process, restart pod |
| **ECC errors** | `ECC errors detected` in dmesg | Mark node unschedulable, replace GPU |
| **Thermal throttling** | GPU temp > 85°C | Check cooling, reduce workload |

**Resolution Procedure**:

```bash
# Step 1: Drain node (prevents new jobs from scheduling)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Step 2: Cordon node (mark unschedulable)
kubectl cordon <node-name>

# Step 3: Kill hung GPU processes
kubectl exec -it <trainer-pod> -- pkill -9 python

# Step 4: Restart trainer pod
kubectl delete pod <trainer-pod> -n vision-ai

# Step 5: If driver issue, restart node
ssh <node-name>
sudo systemctl restart nvidia-persistenced
sudo systemctl restart kubelet

# Step 6: Verify GPU is healthy
kubectl exec -it <new-trainer-pod> -- nvidia-smi

# Step 7: Uncordon node (allow scheduling again)
kubectl uncordon <node-name>
```

**Post-Incident**:
1. Re-run failed jobs: `kubectl get pods -n vision-ai | grep Error | awk '{print $1}' | xargs kubectl delete pod`
2. Check GPU health metrics in Grafana
3. If recurring, consider replacing GPU hardware

### 3.8 Incident: Temporal Workflow Failures

**Symptoms**:
- Temporal UI shows workflows in "Failed" state
- Backend logs show `TemporalError: Workflow timeout`
- Jobs stuck in database but Temporal shows no workflow

**Severity**: P2 (Medium) - Backend uses callback-first pattern, can operate without Temporal

**Investigation Steps**:

```bash
# Step 1: Check Temporal server status
kubectl get pods -n temporal-system

# Step 2: Check Temporal UI
open https://temporal.vision-ai.example.com

# Step 3: List failed workflows
temporal workflow list --query 'ExecutionStatus="Failed"' --namespace vision-ai

# Step 4: Describe specific workflow
temporal workflow describe --workflow-id job_123 --namespace vision-ai

# Step 5: Check Temporal worker logs
kubectl logs deployment/temporal-worker -n vision-ai --tail=100
```

**Common Causes and Solutions**:

| Cause | Detection | Solution |
|-------|-----------|----------|
| **Workflow timeout** | Workflow exceeded 24h | Increase timeout or optimize workflow |
| **Worker crashed** | No active workers | Restart worker deployment |
| **Activity failure** | Activity retries exhausted | Check activity logs, fix issue, retry workflow |
| **Temporal server down** | Cannot connect to server | Restart Temporal server |
| **Database inconsistency** | Job exists in DB but not Temporal | Manually signal workflow or rely on callback |

**Resolution Procedure**:

```bash
# Option 1: Retry failed workflow
temporal workflow reset --workflow-id job_123 --namespace vision-ai --reason "Retry after fixing issue"

# Option 2: Terminate stuck workflow
temporal workflow terminate --workflow-id job_123 --namespace vision-ai --reason "Manual termination"

# Option 3: Restart Temporal worker
kubectl rollout restart deployment/temporal-worker -n vision-ai

# Option 4: Rely on callback-first pattern (if Temporal unavailable)
# Backend continues to work normally, Temporal is optional
# Jobs update via callback, database is source of truth
```

**Callback-First Pattern Fallback**:

Since the platform uses callback-first pattern, if Temporal is completely down:

1. **Training continues normally**: Trainer sends callbacks to Backend
2. **Database is source of truth**: Job status updates in PostgreSQL
3. **Frontend gets updates**: WebSocket updates from Backend
4. **Temporal catches up**: When Temporal recovers, signal workflows with current state

**No immediate user impact** - system designed to work without Temporal.

---

## 4. Common Issues and Solutions

### 4.1 Job Status Inconsistencies

**Symptom**: Job shows `running` in frontend but trainer service says `completed`

**Root Cause**: Callback delivery failure or database update race condition

**Detection**:
```bash
# Check job status in database
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT id, status, updated_at FROM training_jobs WHERE id = 'job_123';"

# Check trainer service status
curl http://timm-trainer:8001/training/status/trainer_job_456
```

**Solution**:
```bash
# Step 1: Manually trigger callback from trainer
curl -X POST http://timm-trainer:8001/training/resend-callback/trainer_job_456

# Step 2: If that fails, manually update database
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "UPDATE training_jobs SET status = 'completed', updated_at = NOW() WHERE id = 'job_123';"

# Step 3: Send WebSocket update to frontend
curl -X POST https://api.vision-ai.example.com/internal/websocket/broadcast \
  -H "X-Internal-Auth: $INTERNAL_AUTH_TOKEN" \
  -d '{"user_id": "user_789", "message": {"type": "job_status_changed", "job_id": "job_123", "status": "completed"}}'
```

**Prevention**:
- Implement idempotent callback handling
- Add callback retry mechanism in trainer
- Add status reconciliation job (runs every 5min)

### 4.2 Callback Delivery Failures

**Symptom**: Trainer service logs show `Failed to send callback: Connection refused`

**Root Cause**: Backend unreachable from trainer, network issue, or incorrect callback URL

**Detection**:
```bash
# Check trainer logs
kubectl logs deployment/timm-trainer -n vision-ai | grep "callback"

# Check if backend is reachable from trainer
kubectl exec -it <trainer-pod> -- curl -v http://backend:8000/health
```

**Solution**:
```bash
# Step 1: Verify callback URL is correct
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT id, callback_url FROM training_jobs WHERE id = 'job_123';"

# Should be: http://backend:8000/api/v1/training-jobs/job_123/callback

# Step 2: Check network connectivity
kubectl exec -it <trainer-pod> -- nslookup backend
kubectl exec -it <trainer-pod> -- curl -v http://backend:8000/health

# Step 3: If backend is down, fix backend first (see Section 3.2)

# Step 4: Trigger manual callback resend
curl -X POST http://timm-trainer:8001/training/resend-callback/trainer_job_456
```

**Prevention**:
- Use service DNS names (not IP addresses)
- Implement callback retry with exponential backoff
- Add callback queue (persist callbacks, retry later)

### 4.3 Resource Exhaustion

**Symptom**: New jobs cannot start, logs show `Resource quota exceeded` or `Insufficient GPU`

**Detection**:
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n vision-ai

# Check resource quotas
kubectl describe quota -n vision-ai

# Check GPU availability
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .status.allocatable["nvidia.com/gpu"]}'
```

**Solution**:
```bash
# Option 1: Stop idle jobs
kubectl get pods -n vision-ai | grep Completed | awk '{print $1}' | xargs kubectl delete pod

# Option 2: Increase resource quotas
kubectl edit quota vision-ai-quota -n vision-ai
# Increase limits.memory, limits.cpu, requests.nvidia.com/gpu

# Option 3: Scale down non-critical services
kubectl scale deployment/mlflow --replicas=1 -n vision-ai

# Option 4: Add more GPU nodes (if on-prem)
# Or increase node count (if cloud)
```

**Prevention**:
- Set job resource limits (max 1 GPU per job)
- Implement job queue (max 4 concurrent jobs per user)
- Auto-scale GPU nodes based on demand
- Add resource usage monitoring and alerts

### 4.4 Storage Quota Exceeded

**Symptom**: Dataset upload fails with `QuotaExceeded` or `InsufficientStorage`

**Detection**:
```bash
# Check S3 bucket size
aws s3 ls s3://vision-ai-datasets/ --recursive --summarize --human-readable | grep "Total Size"

# Check disk usage (if using MinIO)
df -h /mnt/minio-data

# Check per-user storage quotas
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT user_id, SUM(size_bytes) as total_storage FROM datasets GROUP BY user_id ORDER BY total_storage DESC LIMIT 10;"
```

**Solution**:
```bash
# Option 1: Delete old datasets
# Identify datasets not used in 30 days
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "SELECT id, name, created_at, size_bytes FROM datasets WHERE updated_at < NOW() - INTERVAL '30 days';"

# Delete from S3
aws s3 rm s3://vision-ai-datasets/<dataset-id>/ --recursive

# Delete from database
psql -h db.vision-ai.internal -U admin -d vision_platform -c \
  "DELETE FROM datasets WHERE id = 'dataset_123';"

# Option 2: Implement storage lifecycle policies
aws s3api put-bucket-lifecycle-configuration --bucket vision-ai-datasets --lifecycle-configuration file://lifecycle.json

# lifecycle.json:
{
  "Rules": [
    {
      "Id": "DeleteOldCheckpoints",
      "Status": "Enabled",
      "Prefix": "checkpoints/",
      "Expiration": {"Days": 90}
    },
    {
      "Id": "MoveOldDatasetsToGlacier",
      "Status": "Enabled",
      "Prefix": "datasets/",
      "Transitions": [{"Days": 30, "StorageClass": "GLACIER"}]
    }
  ]
}

# Option 3: Increase storage quota (if cloud)
# Contact AWS support or increase MinIO disk size
```

**Prevention**:
- Set per-user storage quotas (e.g., 100GB per user)
- Implement automatic cleanup of old checkpoints
- Add storage usage monitoring
- Alert when storage > 80% capacity

### 4.5 Integration Timeouts

**Symptom**: Logs show `TimeoutError: Operation timed out after 30s`

**Detection**:
```bash
# Check timeout errors by service
cat backend.log | jq -r 'select(.error.type == "TimeoutError") | .error.service' | sort | uniq -c

# Check average response times
cat backend.log | jq -r 'select(.message == "HTTP request completed") | .latency_ms' | awk '{sum+=$1; count++} END {print sum/count}'
```

**Common Timeouts**:

| Operation | Timeout | Symptom | Solution |
|-----------|---------|---------|----------|
| Trainer `/start` | 30s | Slow dataset download | Increase timeout to 60s, optimize S3 transfer |
| Database query | 30s | Slow query | Optimize query, add index, set statement_timeout |
| S3 upload | 60s | Large file upload | Use multipart upload, increase timeout to 300s |
| MLflow log | 5s | MLflow slow | Enter offline mode, buffer metrics |

**Solution**:
```python
# Increase timeout for specific operation
class TrainerClient:
    TIMEOUTS = {
        "start_training": 60,  # increased from 30s
        "stop_training": 10,
        "list_models": 5,
    }
```

**Prevention**:
- Set appropriate timeouts per operation
- Implement timeout monitoring
- Optimize slow operations
- Use async operations for long-running tasks

### 4.6 Circuit Breaker Tripped

**Symptom**: Logs show `CircuitBreakerOpen: Service is unavailable`, requests fail immediately without retry

**Detection**:
```bash
# Check circuit breaker status
curl https://api.vision-ai.example.com/health/detailed | jq '.components.trainer_services'

# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=circuit_breaker_state{service="timm-trainer"}
```

**Solution**:
```bash
# Step 1: Check underlying service health
curl http://timm-trainer:8001/health

# Step 2: If service is healthy, manually close circuit breaker
curl -X POST https://api.vision-ai.example.com/internal/circuit-breaker/reset \
  -H "X-Internal-Auth: $INTERNAL_AUTH_TOKEN" \
  -d '{"service": "timm-trainer"}'

# Step 3: If service is unhealthy, fix service first
# Circuit breaker will auto-recover after service is healthy

# Step 4: Monitor circuit breaker state
watch -n 5 'curl -s https://api.vision-ai.example.com/health/detailed | jq .components.trainer_services'
```

**Circuit Breaker States**:
- **CLOSED**: Normal operation, requests allowed
- **OPEN**: Service degraded, requests fail immediately (no retry)
- **HALF_OPEN**: Testing recovery, limited requests allowed

**Auto-Recovery**: Circuit breaker automatically transitions OPEN → HALF_OPEN → CLOSED after 60s recovery timeout if service is healthy.

**Prevention**:
- Monitor service health proactively
- Set up alerts for circuit breaker state changes
- Implement graceful degradation (continue without non-critical services)

---

## 5. Restart and Upgrade Procedures

### 5.1 Safe Backend Restart

**When to restart**:
- After configuration changes
- After code deployment
- To recover from memory leak
- After database migration

**Preparation**:
```bash
# Step 1: Check current status
kubectl get pods -n vision-ai | grep backend

# Step 2: Check active jobs (warn users if any)
curl https://api.vision-ai.example.com/api/v1/training-jobs?status=running | jq '.total'

# Step 3: Check database migration status
cd backend
alembic current
alembic history
```

**Rolling Restart (Zero Downtime)**:
```bash
# Step 1: Ensure multiple replicas (for zero downtime)
kubectl get deployment/backend -n vision-ai -o jsonpath='{.spec.replicas}'
# If replicas < 2, scale up first
kubectl scale deployment/backend --replicas=3 -n vision-ai

# Step 2: Perform rolling restart
kubectl rollout restart deployment/backend -n vision-ai

# Step 3: Watch rollout progress
kubectl rollout status deployment/backend -n vision-ai

# Step 4: Verify pods are ready
kubectl get pods -n vision-ai | grep backend

# Step 5: Check health
curl https://api.vision-ai.example.com/health
curl https://api.vision-ai.example.com/health/ready

# Step 6: Verify active jobs still running
curl https://api.vision-ai.example.com/api/v1/training-jobs?status=running
```

**Full Restart (Brief Downtime)**:
```bash
# Step 1: Delete all backend pods
kubectl delete pods -l app=backend -n vision-ai

# Step 2: Wait for new pods to start
kubectl wait --for=condition=ready pod -l app=backend -n vision-ai --timeout=300s

# Step 3: Verify health
curl https://api.vision-ai.example.com/health
```

**Rollback (If Issues After Deployment)**:
```bash
# Step 1: Check rollout history
kubectl rollout history deployment/backend -n vision-ai

# Step 2: Rollback to previous version
kubectl rollout undo deployment/backend -n vision-ai

# Step 3: Rollback to specific revision
kubectl rollout undo deployment/backend -n vision-ai --to-revision=3

# Step 4: Verify rollback succeeded
kubectl rollout status deployment/backend -n vision-ai
```

### 5.2 Trainer Service Upgrade

**Preparation**:
```bash
# Step 1: Check current version
curl http://timm-trainer:8001/health | jq '.version'

# Step 2: Check active training jobs
curl http://timm-trainer:8001/training/active | jq '.total'

# Step 3: Review changelog and breaking changes
cat CHANGELOG.md
```

**Upgrade Procedure (Graceful)**:
```bash
# Step 1: Prevent new jobs from starting
kubectl annotate deployment/timm-trainer -n vision-ai drain="true"

# Step 2: Wait for active jobs to complete (or timeout after 24h)
while [ $(curl -s http://timm-trainer:8001/training/active | jq '.total') -gt 0 ]; do
  echo "Waiting for jobs to complete..."
  sleep 300  # Check every 5 minutes
done

# Step 3: Deploy new version
kubectl set image deployment/timm-trainer timm-trainer=vision-ai/timm-trainer:v2.0.0 -n vision-ai

# Step 4: Wait for rollout to complete
kubectl rollout status deployment/timm-trainer -n vision-ai

# Step 5: Verify new version
curl http://timm-trainer:8001/health | jq '.version'

# Step 6: Run smoke tests
curl -X POST http://timm-trainer:8001/training/start \
  -H "Content-Type: application/json" \
  -d '{"config": {...}, "callback_url": "..."}'

# Step 7: Re-enable job scheduling
kubectl annotate deployment/timm-trainer -n vision-ai drain-
```

**Forced Upgrade (If Jobs Can Be Interrupted)**:
```bash
# Step 1: Stop all active jobs
curl -X POST http://timm-trainer:8001/training/stop-all

# Step 2: Deploy new version immediately
kubectl set image deployment/timm-trainer timm-trainer=vision-ai/timm-trainer:v2.0.0 -n vision-ai

# Step 3: Verify deployment
kubectl rollout status deployment/timm-trainer -n vision-ai

# Step 4: Re-run interrupted jobs
# Users need to manually restart failed jobs
```

**Version Compatibility Check**:
```bash
# Ensure trainer API version compatible with backend
curl http://timm-trainer:8001/api/version
# Expected: {"api_version": "v1", "compatible_backend_versions": ["1.0.0", "1.1.0"]}
```

### 5.3 Database Migration Procedures

**Before Migration**:
```bash
# Step 1: Backup database
pg_dump -h db.vision-ai.internal -U admin -d vision_platform -F c -f backup_$(date +%Y%m%d_%H%M%S).dump

# Step 2: Test migration on staging
psql -h db-staging.vision-ai.internal -U admin -d vision_platform < migration.sql

# Step 3: Review migration script
cat alembic/versions/abc123_add_new_column.py
```

**Migration Procedure (No Downtime)**:
```bash
# Step 1: Put backend in read-only mode (optional)
kubectl set env deployment/backend READ_ONLY=true -n vision-ai

# Step 2: Run migration
cd backend
poetry run alembic upgrade head

# Step 3: Verify migration succeeded
poetry run alembic current
# Should show latest revision

# Step 4: Deploy new backend code (if schema-dependent)
kubectl set image deployment/backend backend=vision-ai/backend:v2.0.0 -n vision-ai

# Step 5: Remove read-only mode
kubectl set env deployment/backend READ_ONLY- -n vision-ai

# Step 6: Verify application works
curl https://api.vision-ai.example.com/api/v1/training-jobs | jq '.total'
```

**Migration Rollback**:
```bash
# Step 1: Check current revision
poetry run alembic current

# Step 2: Rollback one revision
poetry run alembic downgrade -1

# Step 3: Rollback to specific revision
poetry run alembic downgrade abc123

# Step 4: Restore from backup (if migration corrupted data)
pg_restore -h db.vision-ai.internal -U admin -d vision_platform -c backup_20250111_103000.dump

# Step 5: Rollback backend deployment
kubectl rollout undo deployment/backend -n vision-ai
```

**Zero-Downtime Migration Strategy**:

1. **Backward-compatible schema change**:
   - Add new column with default value
   - Deploy new code (uses new column if exists, falls back to old)
   - Run migration to populate new column
   - Deploy new code (uses new column only)
   - Remove old column in next migration

2. **Example**:
```python
# Migration 1: Add new column
def upgrade():
    op.add_column('training_jobs', sa.Column('framework', sa.String(), nullable=True))

# Deploy code that uses `framework` if present, else derives from model_name

# Migration 2: Populate new column
def upgrade():
    op.execute("UPDATE training_jobs SET framework = 'timm' WHERE model_name LIKE 'resnet%'")

# Deploy code that requires `framework`

# Migration 3: Make column non-nullable
def upgrade():
    op.alter_column('training_jobs', 'framework', nullable=False)
```

### 5.4 Temporal Workflow Migration

**When to migrate**:
- Workflow definition changes
- Activity signature changes
- Temporal server upgrade

**Preparation**:
```bash
# Step 1: Deploy new workflow version side-by-side
kubectl apply -f temporal-worker-v2.yaml

# Step 2: Route new workflows to v2
# Update backend to use new workflow name: TrainingWorkflowV2
```

**Migration Procedure**:
```bash
# Step 1: Stop sending new workflows to v1
kubectl annotate deployment/backend -n vision-ai workflow-version="v2"

# Step 2: Wait for all v1 workflows to complete
temporal workflow list --query 'WorkflowType="TrainingWorkflowV1" AND ExecutionStatus="Running"' --namespace vision-ai

# Step 3: Once v1 workflows complete, remove v1 worker
kubectl delete deployment/temporal-worker-v1 -n vision-ai

# Step 4: Verify v2 workflows working
temporal workflow list --query 'WorkflowType="TrainingWorkflowV2"' --namespace vision-ai
```

**Note**: Platform uses callback-first pattern, so Temporal migration has no user impact. Backend and Trainer communicate directly, Temporal is supplementary.

### 5.5 MLflow Upgrade

**Preparation**:
```bash
# Step 1: Backup MLflow tracking database
pg_dump -h db.vision-ai.internal -U mlflow -d mlflow_tracking -F c -f mlflow_backup_$(date +%Y%m%d).dump

# Step 2: Backup MLflow artifacts (S3)
aws s3 sync s3://vision-ai-mlflow-artifacts/ s3://vision-ai-mlflow-artifacts-backup-$(date +%Y%m%d)/

# Step 3: Test upgrade on staging
```

**Upgrade Procedure**:
```bash
# Step 1: Enable offline mode in backend (prevents new metric writes)
kubectl set env deployment/backend MLFLOW_OFFLINE=true -n vision-ai

# Step 2: Stop MLflow server
kubectl scale deployment/mlflow --replicas=0 -n vision-ai

# Step 3: Run MLflow database migration
kubectl run mlflow-upgrade --rm -it --image=vision-ai/mlflow:v2.0.0 -- mlflow db upgrade $MLFLOW_TRACKING_URI

# Step 4: Deploy new MLflow version
kubectl set image deployment/mlflow mlflow=vision-ai/mlflow:v2.0.0 -n vision-ai
kubectl scale deployment/mlflow --replicas=1 -n vision-ai

# Step 5: Wait for MLflow to be ready
kubectl wait --for=condition=ready pod -l app=mlflow -n vision-ai --timeout=300s

# Step 6: Verify MLflow is accessible
curl http://mlflow:5000/health

# Step 7: Disable offline mode (backend reconnects to MLflow)
kubectl set env deployment/backend MLFLOW_OFFLINE- -n vision-ai

# Step 8: Flush buffered metrics
curl -X POST https://api.vision-ai.example.com/internal/mlflow/flush-buffer
```

---

## 6. On-Call Playbook

### 6.1 On-Call Responsibilities

**Primary On-Call Engineer**:
- Respond to P0 alerts within 5 minutes
- Respond to P1 alerts within 30 minutes
- Perform initial triage and mitigation
- Escalate to secondary if needed
- Create incident reports for P0/P1 incidents

**Secondary On-Call Engineer**:
- Backup for primary on-call
- Respond if primary doesn't respond within 10min
- Assist with complex incidents
- Conduct post-incident reviews

**On-Call Schedule**: 1-week rotations, Mon 9am - Mon 9am

### 6.2 Alert Severity and Response Times

| Severity | Response Time | Examples | Action |
|----------|--------------|----------|--------|
| **P0 (Critical)** | 5 minutes | Backend down, database down, all jobs failing | Immediate investigation and mitigation |
| **P1 (High)** | 30 minutes | Trainer service down, high error rate, circuit breaker open | Investigate and resolve within 1 hour |
| **P2 (Medium)** | 2 hours | MLflow down, specific job types failing | Investigate during business hours |
| **P3 (Low)** | Next business day | Performance degradation, non-critical feature issues | Add to backlog |

### 6.3 Escalation Procedures

**Escalation Path**:
1. **L1**: Primary On-Call Engineer
2. **L2**: Secondary On-Call Engineer
3. **L3**: Platform Team Lead
4. **L4**: Engineering Manager
5. **L5**: CTO (for major outages affecting all users)

**When to escalate**:
- P0 incident not resolved within 30 minutes
- P1 incident not resolved within 2 hours
- Incident requires specialized knowledge (database, GPU, Temporal)
- Incident affects > 50% of users
- Incident requires urgent stakeholder communication

**Escalation Contact Info**:

| Role | Name | Phone | Slack | Specialty |
|------|------|-------|-------|-----------|
| Platform Lead | Alice Smith | +1-555-0101 | @alice | Architecture, Temporal |
| Backend Lead | Bob Johnson | +1-555-0102 | @bob | FastAPI, Database |
| ML Lead | Carol Davis | +1-555-0103 | @carol | Training, GPU |
| DevOps Lead | David Lee | +1-555-0104 | @david | Kubernetes, Infrastructure |
| Eng Manager | Eve Wilson | +1-555-0105 | @eve | Escalation, Stakeholders |

### 6.4 Communication Templates

**P0 Incident - Initial Notification** (within 15min of detection):
```
Subject: [P0] Vision AI Platform - Backend Service Down

Status: INVESTIGATING
Impact: All users unable to access platform
Started: 2025-01-11 10:15 UTC
ETA: Under investigation

We are aware of an issue preventing users from accessing the Vision AI Platform.
Our team is actively investigating and will provide updates every 15 minutes.

Current status: Backend health check failing, investigating database connectivity.

Next update: 10:30 UTC
```

**P0 Incident - Progress Update** (every 15min):
```
Subject: [P0 UPDATE] Vision AI Platform - Backend Service Down

Status: MITIGATING
Impact: All users unable to access platform
Started: 2025-01-11 10:15 UTC
ETA: 10:45 UTC

Update: Root cause identified - database connection pool exhausted.
Action: Restarting backend deployment to reset connection pool.

Next update: 10:45 UTC or when resolved
```

**P0 Incident - Resolution**:
```
Subject: [RESOLVED] Vision AI Platform - Backend Service Down

Status: RESOLVED
Impact: All users unable to access platform
Started: 2025-01-11 10:15 UTC
Resolved: 2025-01-11 10:42 UTC
Duration: 27 minutes

The issue has been resolved. The platform is now fully operational.

Root Cause: Database connection pool exhausted due to long-running queries.

Mitigation: Restarted backend deployment, killed long-running queries.

Permanent Fix: Will implement query timeout and connection pool monitoring.

Post-Incident Review: Scheduled for 2025-01-12 14:00 UTC

We apologize for the disruption and appreciate your patience.
```

**P1 Incident - Notification**:
```
Subject: [P1] Vision AI Platform - Trainer Service Degraded

Status: INVESTIGATING
Impact: New training jobs for ResNet models failing
Started: 2025-01-11 11:00 UTC
ETA: Under investigation

Some users may experience issues starting new training jobs for ResNet models.
Existing running jobs are not affected.

Next update: 11:30 UTC
```

### 6.5 Common On-Call Scenarios

**Scenario 1: PagerDuty Alert - Backend Service Down**

```
Alert: BackendServiceDown
Time: 02:30 AM
Severity: P0
```

**Response**:
1. Acknowledge alert in PagerDuty (within 5min)
2. Check Grafana dashboard: https://grafana.vision-ai.example.com/d/platform-overview
3. Check backend health: `curl https://api.vision-ai.example.com/health`
4. Check logs: `kubectl logs deployment/backend -n vision-ai --tail=100`
5. If database issue, see Section 3.5
6. If backend crashed, see Section 3.2
7. Send initial notification to #incidents Slack channel
8. Resolve and send resolution notification

**Scenario 2: User Report - Training Job Stuck**

```
User report: "My job has been running for 3 days, is it stuck?"
Job ID: job_abc123
Severity: P2
```

**Response**:
1. Check job in database: `psql ... "SELECT * FROM training_jobs WHERE id = 'job_abc123';"`
2. Check trainer service status: `curl http://timm-trainer:8001/training/status/...`
3. Check trainer logs: `kubectl logs deployment/timm-trainer | grep job_abc123`
4. If stuck, follow Section 3.3
5. If legitimately long-running, inform user of expected duration
6. Update user via email or Slack

**Scenario 3: Monitoring Alert - High Error Rate**

```
Alert: HighErrorRate
Time: 10:00 AM
Severity: P1
Error rate: 15% (threshold: 10%)
```

**Response**:
1. Check error breakdown in Grafana
2. Check Sentry for error details: https://sentry.io/vision-ai/
3. Identify common error type (e.g., `NetworkError: trainer service`)
4. Check affected service health
5. If circuit breaker open, see Section 4.6
6. If service down, see relevant incident response section
7. Send notification if user-facing impact

### 6.6 Post-Incident Review Process

**Timing**: Within 48 hours of P0/P1 incident resolution

**Attendees**:
- On-call engineer(s)
- Service owner(s)
- Platform lead
- Engineering manager (for P0)

**Agenda**:
1. **Timeline**: What happened and when?
2. **Root Cause**: Why did it happen?
3. **Impact**: How many users affected? For how long?
4. **Detection**: How did we detect the issue? Could we detect it faster?
5. **Response**: What went well? What could be improved?
6. **Prevention**: How do we prevent this from happening again?
7. **Action Items**: Who will do what by when?

**Document Template**:
```markdown
# Incident Report: Backend Service Down - 2025-01-11

## Summary
Backend service was unavailable for 27 minutes due to database connection pool exhaustion.

## Impact
- All users unable to access platform
- 15 active training jobs unaffected (continued running)
- No data loss

## Timeline (UTC)
- 10:15 - Alert fired: BackendServiceDown
- 10:16 - On-call engineer acknowledged alert
- 10:18 - Initial investigation: health check failing
- 10:22 - Root cause identified: DB connection pool exhausted
- 10:25 - Mitigation started: Restarting backend deployment
- 10:30 - Backend pods restarted
- 10:35 - Health checks passing
- 10:42 - Declared resolved, monitoring for 10min

## Root Cause
A long-running query (5+ minutes) held database connections open. With 20 backend
replicas and pool_size=20, this exhausted all 400 connections. New requests timed
out waiting for available connections.

## Detection
Prometheus alert fired after 1 minute of failed health checks. Good detection time.

## Response
On-call engineer responded within 1 minute. Mitigation applied within 10 minutes.
Communication sent to users within 15 minutes.

## What Went Well
- Fast detection (1min)
- Fast response (1min)
- Clear runbook followed (Section 3.5)
- Rolling restart prevented data loss

## What Could Be Improved
- No query timeout was set (long-running queries not killed)
- No connection pool monitoring (couldn't see exhaustion coming)
- No automatic circuit breaker for database (failed all requests)

## Action Items
1. [P0] Set statement_timeout=30s on database (@bob - 2025-01-12)
2. [P1] Add connection pool metrics to Grafana (@alice - 2025-01-13)
3. [P1] Implement database circuit breaker (@bob - 2025-01-15)
4. [P2] Add alert for connection pool >80% (@david - 2025-01-18)
5. [P3] Review all slow queries and add indexes (@bob - 2025-01-25)
```

---

## 7. Disaster Recovery

### 7.1 Backup Procedures

**Database Backups (PostgreSQL)**:
```bash
# Daily automated backup (cron job)
0 2 * * * pg_dump -h db.vision-ai.internal -U admin -d vision_platform -F c -f /backups/vision_platform_$(date +\%Y\%m\%d).dump

# Retention: 7 daily, 4 weekly, 12 monthly backups

# Verify backup
pg_restore --list /backups/vision_platform_20250111.dump

# Upload to S3
aws s3 cp /backups/vision_platform_20250111.dump s3://vision-ai-backups/database/
```

**MLflow Backups**:
```bash
# Backup tracking database (PostgreSQL)
pg_dump -h db.vision-ai.internal -U mlflow -d mlflow_tracking -F c -f /backups/mlflow_tracking_$(date +\%Y\%m\%d).dump

# Backup artifacts (S3)
aws s3 sync s3://vision-ai-mlflow-artifacts/ s3://vision-ai-backups/mlflow-artifacts/
```

**Configuration Backups**:
```bash
# Backup Kubernetes manifests
kubectl get all -n vision-ai -o yaml > backup_k8s_vision-ai_$(date +\%Y\%m\%d).yaml

# Backup Temporal workflows
temporal workflow export --namespace vision-ai > backup_temporal_$(date +\%Y\%m\%d).json
```

**Backup Verification** (Monthly):
```bash
# Restore to staging environment
pg_restore -h db-staging.vision-ai.internal -U admin -d vision_platform_test -c /backups/vision_platform_20250111.dump

# Verify data
psql -h db-staging.vision-ai.internal -U admin -d vision_platform_test -c "SELECT COUNT(*) FROM training_jobs;"
```

### 7.2 Recovery Time Objective (RTO) and Recovery Point Objective (RPO)

| Component | RPO | RTO | Backup Frequency |
|-----------|-----|-----|------------------|
| **PostgreSQL** | 24 hours | 1 hour | Daily at 2 AM UTC |
| **MLflow Tracking** | 24 hours | 2 hours | Daily at 3 AM UTC |
| **S3 Datasets** | 0 (versioned) | 0 (immediate) | Continuous (S3 versioning) |
| **S3 Checkpoints** | 24 hours | 0 (immediate) | Continuous (S3 versioning) |
| **Kubernetes Config** | 24 hours | 30 minutes | Daily at 4 AM UTC |
| **Application Code** | 0 (Git) | 30 minutes | Continuous (Git) |

### 7.3 Database Disaster Recovery

**Scenario**: PostgreSQL database corrupted or lost

**Recovery Procedure**:
```bash
# Step 1: Stop all services writing to database
kubectl scale deployment/backend --replicas=0 -n vision-ai

# Step 2: Drop corrupted database (if exists)
psql -h db.vision-ai.internal -U postgres -c "DROP DATABASE vision_platform;"

# Step 3: Create new database
psql -h db.vision-ai.internal -U postgres -c "CREATE DATABASE vision_platform OWNER admin;"

# Step 4: Restore from latest backup
pg_restore -h db.vision-ai.internal -U admin -d vision_platform -c /backups/vision_platform_latest.dump

# Step 5: Verify data integrity
psql -h db.vision-ai.internal -U admin -d vision_platform <<EOF
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM training_jobs;
SELECT COUNT(*) FROM datasets;
SELECT MAX(created_at) FROM training_jobs;  -- Should match backup time
EOF

# Step 6: Run migrations (if database schema changed since backup)
cd backend
poetry run alembic upgrade head

# Step 7: Restart backend
kubectl scale deployment/backend --replicas=3 -n vision-ai

# Step 8: Verify application works
curl https://api.vision-ai.example.com/api/v1/training-jobs | jq '.total'

# Step 9: Communicate data loss window to users
# Example: "Data from 2025-01-10 02:00 UTC to 2025-01-11 10:00 UTC may be lost"
```

**Data Loss Assessment**:
```sql
-- Find jobs created after backup time
SELECT id, status, created_at FROM training_jobs WHERE created_at > '2025-01-10 02:00:00';

-- Notify affected users
SELECT DISTINCT user_id FROM training_jobs WHERE created_at > '2025-01-10 02:00:00';
```

### 7.4 Full Platform Disaster Recovery

**Scenario**: Entire Kubernetes cluster lost (data center failure, catastrophic failure)

**Prerequisites**:
- Database backups in S3
- S3 datasets/checkpoints replicated to another region
- Kubernetes manifests in Git
- Docker images in container registry

**Recovery Procedure** (Estimated time: 4-6 hours):

```bash
# Step 1: Provision new Kubernetes cluster (30min)
# Using cloud provider (AWS, GCP, Azure) or on-prem

# Step 2: Install infrastructure components (30min)
helm install postgres bitnami/postgresql -n vision-ai
helm install redis bitnami/redis -n vision-ai
helm install temporal temporalio/temporal -n temporal-system

# Step 3: Restore database from backup (20min)
kubectl exec -it postgres-0 -n vision-ai -- pg_restore -U admin -d vision_platform -c /backups/vision_platform_latest.dump

# Step 4: Deploy application services (20min)
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/trainer-services-deployment.yaml
kubectl apply -f k8s/mlflow-deployment.yaml

# Step 5: Verify infrastructure (10min)
kubectl get pods -n vision-ai
kubectl get svc -n vision-ai

# Step 6: Restore configuration (10min)
kubectl apply -f backup_k8s_vision-ai_20250111.yaml

# Step 7: Verify health checks (5min)
curl https://api.vision-ai.example.com/health
curl https://api.vision-ai.example.com/health/ready

# Step 8: Verify data access (5min)
curl https://api.vision-ai.example.com/api/v1/training-jobs | jq '.total'

# Step 9: Run smoke tests (15min)
pytest tests/smoke/

# Step 10: Communicate to users (5min)
# Send email: "Platform restored, services operational"
```

**Post-Recovery Validation**:
1. Verify all users can log in
2. Verify existing jobs are visible
3. Create and run test training job
4. Verify MLflow metrics visible
5. Verify dataset upload works
6. Verify checkpoint download works

### 7.5 Data Corruption Recovery

**Scenario**: Database has corrupted data (e.g., job status inconsistencies)

**Detection**:
```bash
# Find status inconsistencies
psql -h db.vision-ai.internal -U admin -d vision_platform <<EOF
-- Jobs marked 'running' but updated >24h ago
SELECT id, status, updated_at FROM training_jobs
WHERE status = 'running' AND updated_at < NOW() - INTERVAL '24 hours';

-- Jobs with invalid status
SELECT id, status FROM training_jobs
WHERE status NOT IN ('pending', 'running', 'completed', 'failed', 'stopped');

-- Jobs without associated user
SELECT id FROM training_jobs WHERE user_id NOT IN (SELECT id FROM users);
EOF
```

**Recovery**:
```bash
# Option 1: Fix data inconsistencies with SQL
psql -h db.vision-ai.internal -U admin -d vision_platform <<EOF
-- Mark old 'running' jobs as 'failed'
UPDATE training_jobs SET status = 'failed', error_message = 'Job timeout'
WHERE status = 'running' AND updated_at < NOW() - INTERVAL '24 hours';

-- Delete orphaned jobs
DELETE FROM training_jobs WHERE user_id NOT IN (SELECT id FROM users);
EOF

# Option 2: Restore from backup (if corruption widespread)
# Follow Section 7.3

# Option 3: Restore specific table from backup
pg_restore -h db.vision-ai.internal -U admin -d vision_platform -t training_jobs -c /backups/vision_platform_latest.dump
```

**Prevention**:
1. Add database constraints (foreign keys, check constraints)
2. Implement data validation in application layer
3. Run data integrity checks daily (cron job)
4. Enable database query logging for auditing

---

## 8. Appendix

### 8.1 Useful Commands Cheat Sheet

**Kubernetes**:
```bash
# Get all resources
kubectl get all -n vision-ai

# Get pods by label
kubectl get pods -l app=backend -n vision-ai

# Describe pod (shows events)
kubectl describe pod <pod-name> -n vision-ai

# Get logs
kubectl logs <pod-name> -n vision-ai --tail=100 -f

# Get logs from all pods in deployment
kubectl logs -l app=backend -n vision-ai --tail=100

# Execute command in pod
kubectl exec -it <pod-name> -n vision-ai -- /bin/bash

# Port-forward to pod
kubectl port-forward pod/<pod-name> 8000:8000 -n vision-ai

# Scale deployment
kubectl scale deployment/backend --replicas=5 -n vision-ai

# Restart deployment
kubectl rollout restart deployment/backend -n vision-ai

# Check rollout status
kubectl rollout status deployment/backend -n vision-ai

# Rollback deployment
kubectl rollout undo deployment/backend -n vision-ai

# Cordon node (prevent scheduling)
kubectl cordon <node-name>

# Drain node (evict all pods)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Uncordon node
kubectl uncordon <node-name>

# Get resource usage
kubectl top nodes
kubectl top pods -n vision-ai

# Get events
kubectl get events -n vision-ai --sort-by='.lastTimestamp'
```

**PostgreSQL**:
```bash
# Connect to database
psql -h db.vision-ai.internal -U admin -d vision_platform

# List databases
\l

# List tables
\dt

# Describe table
\d training_jobs

# Show running queries
SELECT pid, usename, state, query FROM pg_stat_activity WHERE state = 'active';

# Kill query
SELECT pg_terminate_backend(<pid>);

# Show table sizes
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

# Show index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;

# Vacuum analyze (reclaim space, update stats)
VACUUM ANALYZE training_jobs;
```

**Docker**:
```bash
# List running containers
docker ps

# View logs
docker logs <container-id> --tail=100 -f

# Execute command in container
docker exec -it <container-id> /bin/bash

# Inspect container
docker inspect <container-id>

# View resource usage
docker stats

# Prune unused resources
docker system prune -a

# View images
docker images

# Remove image
docker rmi <image-id>
```

**AWS S3**:
```bash
# List buckets
aws s3 ls

# List objects in bucket
aws s3 ls s3://vision-ai-datasets/ --recursive --human-readable

# Copy file to S3
aws s3 cp local-file.txt s3://vision-ai-datasets/file.txt

# Copy file from S3
aws s3 cp s3://vision-ai-datasets/file.txt local-file.txt

# Sync directory to S3
aws s3 sync /local/dir s3://vision-ai-datasets/dir/

# Delete object
aws s3 rm s3://vision-ai-datasets/file.txt

# Delete all objects with prefix
aws s3 rm s3://vision-ai-datasets/old-data/ --recursive

# Get bucket size
aws s3 ls s3://vision-ai-datasets/ --recursive --summarize --human-readable

# List incomplete multipart uploads
aws s3api list-multipart-uploads --bucket vision-ai-datasets

# Abort multipart upload
aws s3api abort-multipart-upload --bucket vision-ai-datasets --key <key> --upload-id <id>
```

**Temporal**:
```bash
# List workflows
temporal workflow list --namespace vision-ai

# Describe workflow
temporal workflow describe --workflow-id job_123 --namespace vision-ai

# Terminate workflow
temporal workflow terminate --workflow-id job_123 --namespace vision-ai

# Retry workflow
temporal workflow reset --workflow-id job_123 --namespace vision-ai

# Query workflow
temporal workflow query --workflow-id job_123 --query-type getStatus --namespace vision-ai

# Signal workflow
temporal workflow signal --workflow-id job_123 --signal-name stop --namespace vision-ai
```

**MLflow**:
```bash
# List experiments
mlflow experiments list --tracking-uri http://mlflow:5000

# Get experiment details
mlflow experiments describe --experiment-id 1 --tracking-uri http://mlflow:5000

# List runs
mlflow runs list --experiment-id 1 --tracking-uri http://mlflow:5000

# Download artifacts
mlflow artifacts download --run-id <run-id> --artifact-path model --tracking-uri http://mlflow:5000
```

### 8.2 Monitoring Queries

**Prometheus PromQL**:
```promql
# API request rate (req/sec)
rate(http_requests_total[5m])

# API error rate (errors/sec)
rate(http_requests_total{status=~"5.."}[5m])

# API error percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100

# p95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Active training jobs
training_jobs_active

# Training job completion rate
rate(training_jobs_completed_total[1h])

# Training job failure rate
rate(training_jobs_failed_total[1h])

# Database connection pool usage
database_connection_pool_active / database_connection_pool_size * 100

# Circuit breaker state (1=open, 0=closed)
circuit_breaker_state{state="open"}

# MLflow offline mode (1=offline, 0=online)
mlflow_offline_mode

# S3 upload errors
rate(s3_upload_errors_total[5m])
```

**Grafana Dashboard Queries**:
```sql
-- See MONITORING_DESIGN.md for complete Grafana dashboard JSON definitions
```

### 8.3 Log Query Examples

**jq queries for structured logs**:
```bash
# Filter by log level
cat backend.log | jq 'select(.level == "ERROR")'

# Filter by time range
cat backend.log | jq 'select(.timestamp > "2025-01-11T10:00:00Z" and .timestamp < "2025-01-11T11:00:00Z")'

# Filter by user
cat backend.log | jq 'select(.user_id == "user_123")'

# Filter by job
cat backend.log | jq 'select(.context.job_id == "job_456")'

# Count errors by type
cat backend.log | jq -r 'select(.level == "ERROR") | .error.type' | sort | uniq -c

# Get error messages
cat backend.log | jq -r 'select(.level == "ERROR") | .message'

# Get average latency
cat backend.log | jq -r 'select(.latency_ms) | .latency_ms' | awk '{sum+=$1; count++} END {print sum/count}'

# Find slow requests (>5s)
cat backend.log | jq 'select(.latency_ms > 5000)'
```

**grep patterns**:
```bash
# Find all errors
grep -i "error\|critical" backend.log

# Find timeouts
grep -i "timeout" backend.log

# Find database errors
grep -i "database\|postgresql" backend.log

# Find S3 errors
grep -i "s3\|bucket" backend.log

# Find job failures
grep "job_failed" backend.log
```

### 8.4 Network Diagnostics

```bash
# Test connectivity
ping db.vision-ai.internal
telnet db.vision-ai.internal 5432
curl -v http://timm-trainer:8001/health

# DNS resolution
nslookup db.vision-ai.internal
dig db.vision-ai.internal

# Trace route
traceroute db.vision-ai.internal

# Check open ports
netstat -tuln | grep LISTEN
ss -tuln | grep LISTEN

# Check firewall rules (iptables)
sudo iptables -L -n

# Test from pod
kubectl exec -it <pod-name> -n vision-ai -- curl -v http://backend:8000/health
kubectl exec -it <pod-name> -n vision-ai -- nslookup backend
kubectl exec -it <pod-name> -n vision-ai -- ping backend

# Check network policies
kubectl get networkpolicies -n vision-ai
kubectl describe networkpolicy <policy-name> -n vision-ai
```

### 8.5 Performance Profiling

**Backend Performance**:
```bash
# Python profiling (cProfile)
python -m cProfile -o profile.stats backend/app/main.py
python -m pstats profile.stats
# In pstats: sort time, stats 20

# Memory profiling (memory_profiler)
python -m memory_profiler backend/app/main.py

# Async profiling (py-spy)
py-spy record -o profile.svg -d 60 -p <pid>

# Request profiling (pyinstrument)
pyinstrument backend/app/main.py
```

**Database Performance**:
```sql
-- Enable query logging
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries >1s
SELECT pg_reload_conf();

-- View slow queries
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active' AND now() - query_start > interval '1 second'
ORDER BY duration DESC;

-- Explain query
EXPLAIN ANALYZE SELECT * FROM training_jobs WHERE status = 'running';

-- View index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0 AND indexname NOT LIKE '%pkey';

-- Vacuum statistics
SELECT schemaname, tablename, last_vacuum, last_autovacuum, last_analyze, last_autoanalyze
FROM pg_stat_user_tables;
```

### 8.6 Security Audit

```bash
# Check exposed services
kubectl get services -n vision-ai

# Check secrets
kubectl get secrets -n vision-ai

# Decode secret
kubectl get secret <secret-name> -n vision-ai -o jsonpath='{.data.<key>}' | base64 -d

# Check RBAC
kubectl get rolebindings -n vision-ai
kubectl get clusterrolebindings | grep vision-ai

# Check pod security policies
kubectl get psp

# Check network policies
kubectl get networkpolicies -n vision-ai

# Audit container images for vulnerabilities
trivy image vision-ai/backend:latest

# Check for privileged containers
kubectl get pods -n vision-ai -o json | jq '.items[] | select(.spec.containers[].securityContext.privileged == true) | .metadata.name'
```

### 8.7 Capacity Planning

**Resource Usage Metrics**:
```bash
# Current resource requests/limits
kubectl describe nodes | grep -A 5 "Allocated resources"

# Pod resource usage
kubectl top pods -n vision-ai --sort-by=memory
kubectl top pods -n vision-ai --sort-by=cpu

# Node resource usage
kubectl top nodes

# Predict future capacity needs
# Rule of thumb: Provision 20% headroom above peak usage
```

**Database Capacity**:
```sql
-- Table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Database size
SELECT pg_size_pretty(pg_database_size('vision_platform'));

-- Row counts
SELECT schemaname, tablename, n_live_tup
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;

-- Growth rate (compare with last week)
-- Estimate: If 100k jobs/month, ~3.3k/day, assume 1KB/row = 3.3MB/day = 100MB/month
```

**S3 Storage Capacity**:
```bash
# Current usage
aws s3 ls s3://vision-ai-datasets/ --recursive --summarize --human-readable

# Estimate growth rate
# Example: 100 jobs/day, 1GB/dataset = 100GB/day = 3TB/month
```

---

## Document Metadata

**Version**: 1.0
**Last Updated**: 2025-01-11
**Maintained By**: Platform Operations Team
**Review Frequency**: Quarterly
**Related Documents**:
- [ERROR_HANDLING_DESIGN.md](./ERROR_HANDLING_DESIGN.md)
- [INTEGRATION_FAILURE_HANDLING.md](./INTEGRATION_FAILURE_HANDLING.md)
- [MONITORING_DESIGN.md](./MONITORING_DESIGN.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [BACKEND_DESIGN.md](./BACKEND_DESIGN.md)

**Change Log**:
- 2025-01-11: Initial version created as P0 document for production readiness
