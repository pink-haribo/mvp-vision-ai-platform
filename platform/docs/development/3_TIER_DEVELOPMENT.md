# 3-Tier Development Strategy - Detailed Implementation Guide

**Last Updated**: 2025-01-12
**Status**: Implementation Guide
**Related**: [TIER_STRATEGY.md](./TIER_STRATEGY.md), [OVERVIEW.md](../architecture/OVERVIEW.md)

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tier 1: Kind + Subprocess Training](#tier-1-kind--subprocess-training)
- [Tier 2: Fully Kind](#tier-2-fully-kind)
- [Tier 3: Production Kubernetes](#tier-3-production-kubernetes)
- [Backend Training Mode Implementation](#backend-training-mode-implementation)
- [Migration Between Tiers](#migration-between-tiers)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Design Philosophy

The 3-tier strategy provides a **progressive path from development to production** while maintaining environment parity:

```
Development Speed ←→ Production Similarity

Tier 1: Fast iteration with production-like infrastructure
Tier 2: Full production simulation in local environment
Tier 3: Actual production deployment
```

### Key Principle: Same Code, Different Execution

**Critical Design Decision**: Training code remains identical across all tiers. Only the execution environment changes.

```python
# trainers/ultralytics/train.py
# This EXACT code runs in:
# - Tier 1: Python subprocess on host machine
# - Tier 2: K8s Job in Kind cluster
# - Tier 3: K8s Job in production cluster

import os
import requests

JOB_ID = os.environ["JOB_ID"]
BACKEND_URL = os.environ["BACKEND_URL"]
CALLBACK_TOKEN = os.environ["CALLBACK_TOKEN"]

# Training loop
for epoch in range(epochs):
    train_one_epoch()

    # Heartbeat to Backend (works via K8s DNS or localhost)
    requests.post(
        f"{BACKEND_URL}/api/v1/training/{JOB_ID}/heartbeat",
        headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
        json={"epoch": epoch, "metrics": metrics}
    )
```

**How it works**:
- **Tier 1**: Backend sets `BACKEND_URL=http://localhost:30080` for subprocess
- **Tier 2/3**: Backend sets `BACKEND_URL=http://backend.platform.svc.cluster.local:8000` (K8s DNS)

---

## Tier 1: Kind + Subprocess Training

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Machine (Windows)                    │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Kind Cluster (Docker)                      │ │
│  │                                                          │ │
│  │  Namespace: platform                                    │ │
│  │  ├── Backend (FastAPI)                                  │ │
│  │  ├── Frontend (Next.js)                                 │ │
│  │  ├── PostgreSQL                                         │ │
│  │  ├── Redis                                              │ │
│  │  └── MinIO                                              │ │
│  │                                                          │ │
│  │  Namespace: mlflow                                      │ │
│  │  └── MLflow Server                                      │ │
│  │                                                          │ │
│  │  Namespace: observability                               │ │
│  │  ├── Prometheus                                         │ │
│  │  ├── Grafana                                            │ │
│  │  └── Loki                                               │ │
│  │                                                          │ │
│  │  Namespace: temporal                                    │ │
│  │  ├── Temporal Server                                    │ │
│  │  ├── Temporal UI                                        │ │
│  │  └── Temporal Worker                                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Training Process (Subprocess)                   │ │
│  │                                                          │ │
│  │  python trainers/ultralytics/train.py \                │ │
│  │    --job-id=job-123 \                                  │ │
│  │    --backend-url=http://localhost:30080 \              │ │
│  │    --callback-token=secret                             │ │
│  │                                                          │ │
│  │  GPU: Direct access to host GPU (CUDA, ROCm)           │ │
│  │  Logs: stdout → Backend captures in real-time          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Why This Design?

**Why Kind for infrastructure?**
1. **Production Parity**: K8s DNS, networking, service discovery work identically
2. **Observability Stack**: Prometheus, Grafana, Loki available for development
3. **Temporal Workflows**: Workflow definitions work exactly as in production
4. **Service Communication**: Backend → MLflow, Backend → MinIO use same URLs

**Why Subprocess for training?**
1. **Instant Start**: `< 1 second` vs `10-30 seconds` for K8s Pod scheduling
2. **Direct Debugging**: Use IDE breakpoints, `pdb`, `print()` statements
3. **No Rebuild**: Change code → run immediately (no Docker build/push/load)
4. **Full GPU Access**: Direct CUDA access without container configuration

**Trade-off**: Tier 1 doesn't test pod lifecycle, resource limits, or K8s-specific issues. That's what Tier 2 is for.

### 1.3 Setup Instructions

#### Prerequisites

```bash
# Windows
winget install Kubernetes.kind
winget install Kubernetes.kubectl

# Verify installation
kind version
kubectl version --client
```

#### Step 1: Create Kind Cluster

Create `platform/infrastructure/kind-config.yaml`:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: platform-dev
nodes:
- role: control-plane
  extraPortMappings:
  # Application Services
  - containerPort: 30080  # Backend API
    hostPort: 30080
    protocol: TCP
  - containerPort: 30300  # Frontend
    hostPort: 30300
    protocol: TCP

  # Data Services
  - containerPort: 30900  # MinIO API
    hostPort: 30900
    protocol: TCP
  - containerPort: 30901  # MinIO Console
    hostPort: 30901
    protocol: TCP

  # ML Services
  - containerPort: 30500  # MLflow
    hostPort: 30500
    protocol: TCP

  # Observability
  - containerPort: 30090  # Prometheus
    hostPort: 30090
    protocol: TCP
  - containerPort: 30030  # Grafana
    hostPort: 30030
    protocol: TCP
  - containerPort: 30100  # Loki
    hostPort: 30100
    protocol: TCP

  # Orchestration
  - containerPort: 30233  # Temporal UI
    hostPort: 30233
    protocol: TCP
  - containerPort: 30700  # Temporal gRPC
    hostPort: 30700
    protocol: TCP
```

Create cluster:

```bash
cd platform/infrastructure
kind create cluster --config kind-config.yaml
kubectl cluster-info --context kind-platform-dev
```

#### Step 2: Create Namespaces

```bash
kubectl create namespace platform
kubectl create namespace mlflow
kubectl create namespace observability
kubectl create namespace temporal
```

#### Step 3: Deploy Infrastructure Services

**PostgreSQL**:

```yaml
# k8s/platform/postgres.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: platform
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: platform
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:16-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: platform
        - name: POSTGRES_USER
          value: admin
        - name: POSTGRES_PASSWORD
          value: devpass
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: platform
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
    nodePort: 30543
  type: NodePort
```

**Redis, MinIO, MLflow, Temporal, Prometheus, Grafana, Loki**: Similar manifests (see full examples in `platform/infrastructure/k8s/`)

Apply all:

```bash
kubectl apply -f k8s/platform/
kubectl apply -f k8s/mlflow/
kubectl apply -f k8s/observability/
kubectl apply -f k8s/temporal/
```

#### Step 4: Build and Deploy Backend/Frontend Images

```bash
# Build Backend
cd platform/backend
docker build -t platform-backend:latest .
kind load docker-image platform-backend:latest --name platform-dev

# Build Frontend
cd platform/frontend
docker build -t platform-frontend:latest .
kind load docker-image platform-frontend:latest --name platform-dev

# Deploy
kubectl apply -f k8s/platform/backend.yaml
kubectl apply -f k8s/platform/frontend.yaml
```

#### Step 5: Wait for Ready

```bash
kubectl wait --for=condition=ready pod -l app=backend -n platform --timeout=300s
kubectl wait --for=condition=ready pod -l app=frontend -n platform --timeout=300s
kubectl wait --for=condition=ready pod -l app=mlflow -n mlflow --timeout=300s
```

#### Step 6: Configure Backend for Subprocess Mode

Backend deployment ConfigMap:

```yaml
# k8s/platform/backend-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: backend-config
  namespace: platform
data:
  TRAINING_MODE: "subprocess"
  DATABASE_URL: "postgresql://admin:devpass@postgres:5432/platform"
  REDIS_URL: "redis://redis:6379"
  MINIO_ENDPOINT: "http://minio:9000"
  MLFLOW_TRACKING_URI: "http://mlflow.mlflow:5000"
  TEMPORAL_HOST: "temporal.temporal:7233"
  BACKEND_URL: "http://localhost:30080"  # For subprocess callbacks
  TRAINERS_BASE_PATH: "/workspace/trainers"  # Mounted from host
  PYTHON_EXECUTABLE: "python"
```

**Important**: Backend runs in Kind, but spawns subprocess on host machine via exec into Backend pod!

#### Step 7: Access Services

All services are accessible via NodePort:

```
Frontend:  http://localhost:30300
Backend:   http://localhost:30080
MinIO API: http://localhost:30900
MinIO UI:  http://localhost:30901
MLflow:    http://localhost:30500
Grafana:   http://localhost:30030
Prometheus: http://localhost:30090
Temporal UI: http://localhost:30233
```

### 1.4 Daily Development Workflow

**Services are already running!** Just use the platform:

1. **Access Frontend**: http://localhost:30300
2. **Create training job**: Chat interface → "ResNet-50로 고양이 개 분류 학습해줘"
3. **Backend receives request**:
   - Parses intent via LLM
   - Creates Temporal workflow
   - Workflow executes training step
4. **Backend spawns subprocess**:
   ```bash
   python C:/Users/flyto/.../platform/trainers/timm/train.py \
     --job-id=job-123 \
     --backend-url=http://localhost:30080 \
     --callback-token=eyJ0eXAiOiJKV1QiLCJhbGc...
   ```
5. **Training runs on host**:
   - Direct GPU access (CUDA)
   - Logs appear in subprocess stdout
   - Backend captures logs in real-time
   - Sends heartbeats to Backend API
6. **Monitor progress**:
   - Frontend: Real-time WebSocket updates
   - MLflow: http://localhost:30500 (experiment tracking)
   - Grafana: http://localhost:30030 (system metrics)

**To restart Backend** (after code changes):

```bash
kubectl rollout restart deployment/backend -n platform
```

**To view logs**:

```bash
# Backend logs (includes subprocess spawn messages)
kubectl logs -n platform deployment/backend -f

# Training logs (subprocess stdout)
# Captured by Backend and sent to Frontend via WebSocket
```

---

## Tier 2: Fully Kind

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Machine (Windows)                    │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Kind Cluster (Docker)                      │ │
│  │                                                          │ │
│  │  Namespace: platform                                    │ │
│  │  ├── Backend (FastAPI)                                  │ │
│  │  ├── Frontend (Next.js)                                 │ │
│  │  ├── PostgreSQL                                         │ │
│  │  ├── Redis                                              │ │
│  │  └── MinIO                                              │ │
│  │                                                          │ │
│  │  Namespace: training (NEW!)                             │ │
│  │  └── Training Jobs (K8s Job resources)                 │ │
│  │      ├── trainer-job-123 (Pod)                         │ │
│  │      │   └── Container: trainer-ultralytics:latest     │ │
│  │      ├── trainer-job-124 (Pod)                         │ │
│  │      │   └── Container: trainer-timm:latest            │ │
│  │      └── ...                                            │ │
│  │                                                          │ │
│  │  Namespace: mlflow, observability, temporal (same)     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Why Tier 2?

**Purpose**: Final validation before production deployment.

**What Tier 2 tests that Tier 1 doesn't**:
1. **Pod Lifecycle**: Init containers, readiness/liveness probes, graceful shutdown
2. **Resource Limits**: Memory/CPU limits, eviction scenarios
3. **Image Build**: Dockerfile correctness, dependencies, entrypoint
4. **K8s Job**: Job completion, retry logic, TTL after completion
5. **Network Policies**: Pod-to-pod communication restrictions
6. **Volume Mounts**: Dataset downloads, checkpoint uploads via volumes

**When to use Tier 2**:
- Before deploying to production (final validation)
- Testing resource limit configurations
- Debugging K8s-specific issues
- CI/CD pipeline integration

### 2.3 Migration from Tier 1 to Tier 2

#### Step 1: Build Trainer Images

```bash
# Build Ultralytics trainer
cd platform/trainers/ultralytics
docker build -t trainer-ultralytics:latest .

# Build timm trainer
cd platform/trainers/timm
docker build -t trainer-timm:latest .

# Load into Kind
kind load docker-image trainer-ultralytics:latest --name platform-dev
kind load docker-image trainer-timm:latest --name platform-dev
```

#### Step 2: Create Training Namespace

```bash
kubectl create namespace training

# Create resource quotas
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: training-quota
  namespace: training
spec:
  hard:
    requests.cpu: "16"
    requests.memory: 32Gi
    limits.cpu: "32"
    limits.memory: 64Gi
    persistentvolumeclaims: "10"
EOF
```

#### Step 3: Update Backend Configuration

Update Backend ConfigMap to use `TRAINING_MODE=kubernetes`:

```bash
kubectl patch configmap backend-config -n platform --type merge -p '{"data":{"TRAINING_MODE":"kubernetes"}}'
kubectl rollout restart deployment/backend -n platform
```

Or update `k8s/platform/backend-config.yaml`:

```yaml
data:
  TRAINING_MODE: "kubernetes"  # Changed from "subprocess"
  TRAINING_NAMESPACE: "training"
  TRAINING_IMAGE_ULTRALYTICS: "trainer-ultralytics:latest"
  TRAINING_IMAGE_TIMM: "trainer-timm:latest"
```

#### Step 4: Grant Backend RBAC Permissions

Backend needs permission to create Jobs in `training` namespace:

```yaml
# k8s/platform/backend-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: backend-sa
  namespace: platform
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: training-job-manager
  namespace: training
rules:
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "get", "list", "watch", "delete"]
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: backend-training-manager
  namespace: training
subjects:
- kind: ServiceAccount
  name: backend-sa
  namespace: platform
roleRef:
  kind: Role
  name: training-job-manager
  apiGroup: rbac.authorization.k8s.io
```

Apply:

```bash
kubectl apply -f k8s/platform/backend-rbac.yaml
```

Update Backend deployment to use ServiceAccount:

```yaml
# k8s/platform/backend.yaml
spec:
  template:
    spec:
      serviceAccountName: backend-sa  # Add this
      containers:
      - name: backend
        # ... rest of spec
```

#### Step 5: Test K8s Job Creation

Trigger training via Frontend:

1. Frontend → Backend: "YOLO11n으로 객체 탐지 학습해줘"
2. Backend creates K8s Job:
   ```yaml
   apiVersion: batch/v1
   kind: Job
   metadata:
     name: trainer-job-123
     namespace: training
   spec:
     template:
       spec:
         restartPolicy: Never
         containers:
         - name: trainer
           image: trainer-ultralytics:latest
           command: ["python", "train.py"]
           env:
           - name: JOB_ID
             value: "job-123"
           - name: BACKEND_URL
             value: "http://backend.platform.svc.cluster.local:8000"
           - name: CALLBACK_TOKEN
             value: "eyJ0eXAiOiJKV1QiLCJhbGc..."
           - name: S3_ENDPOINT
             value: "http://minio.platform.svc.cluster.local:9000"
           resources:
             requests:
               memory: "4Gi"
               cpu: "2"
             limits:
               memory: "8Gi"
               cpu: "4"
     backoffLimit: 3
     ttlSecondsAfterFinished: 3600
   ```
3. K8s schedules Pod in `training` namespace
4. Pod runs training code (same code as Tier 1!)
5. Training sends heartbeats to `http://backend.platform.svc.cluster.local:8000` (K8s DNS)
6. Backend receives heartbeats, updates Frontend via WebSocket

**Verify**:

```bash
kubectl get jobs -n training
kubectl get pods -n training
kubectl logs -n training <pod-name> -f
```

### 2.4 Differences from Tier 1

| Aspect | Tier 1 | Tier 2 |
|--------|--------|--------|
| **Training Start Time** | < 1 second | 10-30 seconds |
| **Debugging** | IDE breakpoints work | Need `kubectl exec` or logs |
| **Code Changes** | Instant (no rebuild) | Rebuild + push + load |
| **GPU Access** | Direct host GPU | Requires `nvidia.com/gpu: 1` |
| **Logs** | Direct stdout capture | `kubectl logs` |
| **Resource Isolation** | None (host resources) | K8s limits enforced |
| **Production Similarity** | 80% | 95% |

---

## Tier 3: Production Kubernetes

### 3.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                 Cloud K8s Cluster (Railway)                  │
│                                                               │
│  Namespace: platform-prod                                    │
│  ├── Backend (Deployment, replicas: 3)                      │
│  ├── Frontend (Deployment, replicas: 2)                     │
│  ├── Redis (StatefulSet, replicas: 1)                       │
│  └── MinIO (StatefulSet, replicas: 1)                       │
│                                                               │
│  Namespace: training-prod                                    │
│  └── Training Jobs (K8s Job resources)                      │
│      └── Uses same images as Tier 2!                        │
│                                                               │
│  Namespace: mlflow-prod, observability-prod, temporal-prod  │
│                                                               │
│  External Services (Managed)                                 │
│  ├── PostgreSQL (Railway PostgreSQL)                        │
│  ├── R2 (Cloudflare R2 / S3)                                │
│  └── Monitoring (Railway built-in)                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Key Differences from Tier 2

| Component | Tier 2 (Kind) | Tier 3 (Production) |
|-----------|---------------|---------------------|
| **PostgreSQL** | StatefulSet in Kind | Managed DB (Railway PostgreSQL) |
| **Object Storage** | MinIO in Kind | Cloudflare R2 / AWS S3 |
| **Ingress** | NodePort | Ingress Controller + TLS |
| **Replicas** | 1 for all services | Multi-replica (Backend: 3, Frontend: 2) |
| **Autoscaling** | None | HPA for Backend/Frontend |
| **Secrets** | ConfigMap | Kubernetes Secrets + External Secrets Operator |
| **Monitoring** | Local Prometheus/Grafana | Railway metrics + Prometheus remote write |

### 3.3 Deployment Process

#### Prerequisites

```bash
# Railway CLI
npm install -g @railway/cli
railway login

# Kubernetes context
railway kubernetes context <project-id>
```

#### Step 1: Create Production Secrets

```bash
kubectl create secret generic backend-secrets -n platform-prod \
  --from-literal=DATABASE_URL='postgresql://user:pass@host:5432/db' \
  --from-literal=REDIS_URL='redis://host:6379' \
  --from-literal=R2_ENDPOINT='https://account.r2.cloudflarestorage.com' \
  --from-literal=R2_ACCESS_KEY='...' \
  --from-literal=R2_SECRET_KEY='...' \
  --from-literal=ANTHROPIC_API_KEY='sk-ant-...' \
  --from-literal=JWT_SECRET='production-secret-key'
```

#### Step 2: Build and Push Images

```bash
# Build with multi-arch support
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t registry.railway.app/platform-backend:latest \
  --push \
  platform/backend/

# Same for Frontend, Trainers
docker buildx build --platform linux/amd64,linux/arm64 \
  -t registry.railway.app/trainer-ultralytics:latest \
  --push \
  platform/trainers/ultralytics/
```

#### Step 3: Deploy to Production

```bash
# Apply production manifests
kubectl apply -f k8s/production/platform/
kubectl apply -f k8s/production/training/
kubectl apply -f k8s/production/mlflow/
kubectl apply -f k8s/production/observability/

# Verify rollout
kubectl rollout status deployment/backend -n platform-prod
kubectl rollout status deployment/frontend -n platform-prod
```

#### Step 4: Configure Ingress

```yaml
# k8s/production/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: platform-ingress
  namespace: platform-prod
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.vision-platform.com
    - app.vision-platform.com
    secretName: platform-tls
  rules:
  - host: api.vision-platform.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8000
  - host: app.vision-platform.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 3000
```

### 3.4 Environment Variables (Production)

Backend ConfigMap (loaded via Kubernetes Secret):

```yaml
data:
  TRAINING_MODE: "kubernetes"
  DATABASE_URL: "postgresql://user:pass@railway-postgres:5432/platform_prod"
  REDIS_URL: "redis://redis.platform-prod.svc.cluster.local:6379"
  MLFLOW_TRACKING_URI: "http://mlflow.mlflow-prod.svc.cluster.local:5000"
  TEMPORAL_HOST: "temporal.temporal-prod.svc.cluster.local:7233"
  BACKEND_URL: "http://backend.platform-prod.svc.cluster.local:8000"
  R2_ENDPOINT: "https://account-id.r2.cloudflarestorage.com"
  R2_BUCKET: "platform-prod-datasets"
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
```

**Training code remains IDENTICAL** - only environment variables change!

---

## Backend Training Mode Implementation

### 4.1 Training Manager Architecture

```python
# platform/backend/app/services/training_manager.py
from enum import Enum
from typing import Protocol
import subprocess
import os

class TrainingMode(str, Enum):
    SUBPROCESS = "subprocess"
    KUBERNETES = "kubernetes"

class TrainingExecutor(Protocol):
    """Protocol for training executors."""

    async def start_training(
        self,
        job_id: str,
        trainer_type: str,  # "ultralytics", "timm", etc.
        config: dict,
        callback_token: str
    ) -> str:
        """Start training job. Returns execution ID."""
        ...

    async def get_status(self, execution_id: str) -> dict:
        """Get training job status."""
        ...

    async def stop_training(self, execution_id: str):
        """Stop training job."""
        ...

    async def get_logs(self, execution_id: str) -> str:
        """Get training logs."""
        ...
```

### 4.2 Subprocess Executor (Tier 1)

```python
# platform/backend/app/services/executors/subprocess_executor.py
import subprocess
import asyncio
import os
from pathlib import Path
from typing import Dict, Optional

class SubprocessExecutor:
    """Executes training as host subprocess."""

    def __init__(self, config: dict):
        self.trainers_path = Path(config["TRAINERS_BASE_PATH"])
        self.python_executable = config["PYTHON_EXECUTABLE"]
        self.backend_url = config["BACKEND_URL"]
        self.processes: Dict[str, subprocess.Popen] = {}

    async def start_training(
        self,
        job_id: str,
        trainer_type: str,
        config: dict,
        callback_token: str
    ) -> str:
        """Start training subprocess."""

        # Build command
        trainer_script = self.trainers_path / trainer_type / "train.py"

        cmd = [
            self.python_executable,
            str(trainer_script),
            "--job-id", job_id,
            "--backend-url", self.backend_url,
            "--callback-token", callback_token,
        ]

        # Add config as JSON
        import json
        config_json = json.dumps(config)
        cmd.extend(["--config", config_json])

        # Environment variables
        env = os.environ.copy()
        env.update({
            "JOB_ID": job_id,
            "BACKEND_URL": self.backend_url,
            "CALLBACK_TOKEN": callback_token,
            "S3_ENDPOINT": self.backend_url.replace(":30080", ":30900"),  # MinIO
            "MLFLOW_TRACKING_URI": self.backend_url.replace(":30080", ":30500"),
        })

        # Start subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1
        )

        self.processes[job_id] = process

        # Stream logs asynchronously
        asyncio.create_task(self._stream_logs(job_id, process))

        return job_id

    async def _stream_logs(self, job_id: str, process: subprocess.Popen):
        """Stream subprocess logs to database/websocket."""
        for line in process.stdout:
            # Send to WebSocket clients
            await self.send_log_to_clients(job_id, line)

            # Parse metrics if present
            metrics = self.parse_metrics(line)
            if metrics:
                await self.update_job_metrics(job_id, metrics)

    async def stop_training(self, execution_id: str):
        """Stop subprocess."""
        process = self.processes.get(execution_id)
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    async def get_status(self, execution_id: str) -> dict:
        """Get subprocess status."""
        process = self.processes.get(execution_id)
        if not process:
            return {"status": "not_found"}

        poll = process.poll()
        if poll is None:
            return {"status": "running", "pid": process.pid}
        elif poll == 0:
            return {"status": "completed", "exit_code": 0}
        else:
            return {"status": "failed", "exit_code": poll}
```

### 4.3 Kubernetes Executor (Tier 2 & 3)

```python
# platform/backend/app/services/executors/k8s_executor.py
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import yaml

class KubernetesExecutor:
    """Executes training as K8s Job."""

    def __init__(self, k8s_config: dict):
        config.load_incluster_config()  # When running in cluster
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()
        self.namespace = k8s_config["TRAINING_NAMESPACE"]
        self.images = {
            "ultralytics": k8s_config["TRAINING_IMAGE_ULTRALYTICS"],
            "timm": k8s_config["TRAINING_IMAGE_TIMM"],
        }

    async def start_training(
        self,
        job_id: str,
        trainer_type: str,
        config: dict,
        callback_token: str
    ) -> str:
        """Create K8s Job for training."""

        # Build Job manifest
        job_manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"trainer-{job_id}",
                "namespace": self.namespace,
                "labels": {
                    "app": "trainer",
                    "job_id": job_id,
                    "trainer_type": trainer_type
                }
            },
            "spec": {
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "trainer",
                            "job_id": job_id
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "trainer",
                            "image": self.images[trainer_type],
                            "command": ["python", "train.py"],
                            "env": [
                                {"name": "JOB_ID", "value": job_id},
                                {"name": "BACKEND_URL", "value": "http://backend.platform.svc.cluster.local:8000"},
                                {"name": "CALLBACK_TOKEN", "value": callback_token},
                                {"name": "S3_ENDPOINT", "value": "http://minio.platform.svc.cluster.local:9000"},
                                {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow.mlflow.svc.cluster.local:5000"},
                                {"name": "CONFIG_JSON", "value": json.dumps(config)}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": config.get("memory", "4Gi"),
                                    "cpu": config.get("cpu", "2")
                                },
                                "limits": {
                                    "memory": config.get("memory_limit", "8Gi"),
                                    "cpu": config.get("cpu_limit", "4")
                                }
                            }
                        }]
                    }
                },
                "backoffLimit": 3,
                "ttlSecondsAfterFinished": 3600
            }
        }

        # Create Job
        try:
            self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job_manifest
            )
            return f"trainer-{job_id}"
        except ApiException as e:
            raise RuntimeError(f"Failed to create K8s Job: {e}")

    async def get_status(self, execution_id: str) -> dict:
        """Get Job status."""
        try:
            job = self.batch_v1.read_namespaced_job(
                name=execution_id,
                namespace=self.namespace
            )

            status = job.status
            if status.succeeded:
                return {"status": "completed"}
            elif status.failed:
                return {"status": "failed", "reason": status.conditions}
            elif status.active:
                return {"status": "running"}
            else:
                return {"status": "pending"}
        except ApiException as e:
            return {"status": "not_found"}

    async def get_logs(self, execution_id: str) -> str:
        """Get Pod logs."""
        # Find pod for this job
        pods = self.core_v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"job-name={execution_id}"
        )

        if not pods.items:
            return ""

        pod_name = pods.items[0].metadata.name
        return self.core_v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=self.namespace
        )
```

### 4.4 Training Manager Factory

```python
# platform/backend/app/services/training_manager.py
from app.services.executors.subprocess_executor import SubprocessExecutor
from app.services.executors.k8s_executor import KubernetesExecutor
from app.core.config import settings

class TrainingManager:
    """Factory for training executors."""

    def __init__(self):
        self.mode = TrainingMode(settings.TRAINING_MODE)

        if self.mode == TrainingMode.SUBPROCESS:
            self.executor = SubprocessExecutor({
                "TRAINERS_BASE_PATH": settings.TRAINERS_BASE_PATH,
                "PYTHON_EXECUTABLE": settings.PYTHON_EXECUTABLE,
                "BACKEND_URL": settings.BACKEND_URL,
            })
        elif self.mode == TrainingMode.KUBERNETES:
            self.executor = KubernetesExecutor({
                "TRAINING_NAMESPACE": settings.TRAINING_NAMESPACE,
                "TRAINING_IMAGE_ULTRALYTICS": settings.TRAINING_IMAGE_ULTRALYTICS,
                "TRAINING_IMAGE_TIMM": settings.TRAINING_IMAGE_TIMM,
            })

    async def start_training(self, job_id: str, trainer_type: str, config: dict) -> str:
        """Start training (delegates to executor)."""
        callback_token = self.generate_callback_token(job_id)
        return await self.executor.start_training(
            job_id, trainer_type, config, callback_token
        )

    # ... other methods delegate to self.executor
```

---

## Migration Between Tiers

### 5.1 Tier 1 → Tier 2 Migration Checklist

- [ ] Build all trainer Docker images
- [ ] Load images into Kind cluster
- [ ] Create `training` namespace
- [ ] Apply RBAC for Backend ServiceAccount
- [ ] Update Backend ConfigMap: `TRAINING_MODE=kubernetes`
- [ ] Update Backend ConfigMap: Add image names
- [ ] Restart Backend deployment
- [ ] Test: Create training job via Frontend
- [ ] Verify: `kubectl get jobs -n training` shows job
- [ ] Verify: Job completes successfully
- [ ] Verify: Logs available via `kubectl logs`
- [ ] Verify: MLflow tracks experiment

### 5.2 Tier 2 → Tier 3 Migration Checklist

- [ ] Provision production K8s cluster (Railway)
- [ ] Set up managed PostgreSQL
- [ ] Set up R2/S3 bucket
- [ ] Create production secrets
- [ ] Build and push multi-arch images
- [ ] Update image references to registry URLs
- [ ] Apply production manifests
- [ ] Configure Ingress with TLS
- [ ] Update DNS records
- [ ] Run smoke tests
- [ ] Monitor metrics (Prometheus/Grafana)
- [ ] Verify: Training jobs complete in prod cluster
- [ ] Verify: R2 storage works correctly

---

## Troubleshooting

### Common Issues - Tier 1

**Problem**: Backend can't reach MinIO
```
Error: Connection refused to http://minio.platform:9000
```

**Solution**: Check MinIO pod is running
```bash
kubectl get pods -n platform
kubectl logs -n platform -l app=minio
```

**Problem**: Subprocess can't import dependencies
```
ModuleNotFoundError: No module named 'torch'
```

**Solution**: Install dependencies in host Python environment
```bash
cd platform/trainers/ultralytics
pip install -r requirements.txt
```

**Problem**: Training logs not appearing
```
Backend shows "Training started" but no progress logs
```

**Solution**: Check subprocess stdout is being captured
```python
# In subprocess_executor.py, add debug logging:
for line in process.stdout:
    print(f"[DEBUG] Captured: {line}")  # Should appear in Backend logs
    await self.send_log_to_clients(job_id, line)
```

### Common Issues - Tier 2

**Problem**: Job stuck in Pending state
```
kubectl get jobs -n training
NAME            COMPLETIONS   DURATION   AGE
trainer-job-1   0/1           5m         5m
```

**Solution**: Check Pod events
```bash
kubectl describe pod -n training <pod-name>
# Look for: Insufficient memory, ImagePullBackOff, etc.
```

**Problem**: ImagePullBackOff
```
Failed to pull image "trainer-ultralytics:latest": not found
```

**Solution**: Verify image is loaded into Kind
```bash
docker images | grep trainer
kind load docker-image trainer-ultralytics:latest --name platform-dev
```

**Problem**: Job fails immediately
```
Error: Back-off restarting failed container
```

**Solution**: Check logs for startup errors
```bash
kubectl logs -n training <pod-name>
# Common: Missing environment variables, wrong entrypoint
```

### Common Issues - Tier 3

**Problem**: Training can't reach Backend
```
Error: requests.post("http://backend.platform.svc.cluster.local:8000"): Name or service not known
```

**Solution**: Check DNS and Service
```bash
kubectl get svc -n platform-prod
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- sh
# Inside pod:
nslookup backend.platform-prod.svc.cluster.local
curl http://backend.platform-prod.svc.cluster.local:8000/health
```

**Problem**: R2 upload fails
```
Error: S3 Forbidden (403)
```

**Solution**: Verify R2 credentials and bucket policy
```bash
# Test credentials locally
aws s3 ls s3://platform-prod-datasets \
  --endpoint-url https://account-id.r2.cloudflarestorage.com \
  --profile r2
```

---

## Summary

| Tier | Use Case | Training Execution | Setup Time | Iteration Speed |
|------|----------|-------------------|------------|-----------------|
| **Tier 1** | Daily development | Subprocess on host | 30 min (once) | ⚡️ Instant |
| **Tier 2** | Pre-production validation | K8s Job in Kind | 10 min (images) | 🐢 30 sec |
| **Tier 3** | Production | K8s Job in cloud | 1-2 hours | 🐢 30 sec |

**Key Takeaway**: Spend 95% of time in Tier 1, validate in Tier 2 before each production deployment, deploy to Tier 3 with confidence.

---

**Next Steps**: [Infrastructure Setup Scripts](../../infrastructure/README.md)
