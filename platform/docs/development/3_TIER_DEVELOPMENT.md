# 3-Tier Development Process

Complete guide to developing across three environments: Subprocess → Kind → Production K8s.

## Table of Contents

- [Overview](#overview)
- [Tier Comparison](#tier-comparison)
- [Configuration Strategy](#configuration-strategy)
- [Development Workflow](#development-workflow)
- [Testing Strategy](#testing-strategy)
- [Migration Between Tiers](#migration-between-tiers)
- [Troubleshooting](#troubleshooting)

## Overview

The platform supports three development tiers with **the same source code** but **different configurations**:

```
Tier 1: Subprocess (Local Dev)
  ↓ Works? Deploy to →
Tier 2: Kind (Local K8s)
  ↓ Works? Deploy to →
Tier 3: Production (K8s Cluster)
```

### Key Principle

**One Codebase, Three Environments**

The ONLY differences between tiers are environment variables. The application code is identical.

## Tier Comparison

### Tier 1: Hybrid Development Mode

**Purpose**: Rapid development with production-like monitoring

**Training Execution**: Backend spawns Python subprocess

**Architecture**: Hybrid approach combining the best of local development and K8s infrastructure

**Infrastructure**:
```
Developer Machine
├── Local Processes (Fast iteration):
│   ├── Frontend (pnpm dev) - localhost:3000
│   ├── Backend (uvicorn --reload) - localhost:8000
│   └── Trainer (python train.py) - subprocess
│
├── Docker Compose (Lightweight services):
│   ├── PostgreSQL - localhost:5432
│   ├── Redis - localhost:6379
│   └── MinIO - localhost:9000
│
└── Kind Cluster (Monitoring stack - set up once, reuse):
    ├── MLflow - localhost:5000
    ├── Prometheus - localhost:9090
    ├── Grafana - localhost:3001
    ├── Loki - localhost:3100
    └── Temporal - localhost:7233
```

**Environment Variables**:
```bash
# .env.tier1
EXECUTION_MODE=subprocess
STORAGE_TYPE=minio

# Database & Cache (Docker Compose)
DATABASE_URL=postgresql://postgres:devpass@localhost:5432/platform
REDIS_URL=redis://localhost:6379/0

# Storage (Docker Compose - S3-compatible)
S3_ENDPOINT=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
BUCKET_NAME=vision-platform

# Monitoring (Kind cluster)
MLFLOW_TRACKING_URI=http://localhost:5000
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3001

# Temporal (Kind cluster)
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default

# LLM APIs
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Backend
BACKEND_BASE_URL=http://localhost:8000
```

**Pros**:
- Fastest iteration cycle (services run as processes)
- Easy debugging (VS Code, breakpoints, hot reload)
- Production-like monitoring (MLflow, Grafana, Prometheus)
- Temporal workflow debugging via Temporal UI
- Real experiment tracking
- One-time setup for monitoring stack

**Cons**:
- Initial setup requires Kind cluster
- Monitoring stack uses ~2GB RAM (but reusable)

**When to Use**:
- Daily feature development
- Bug fixing with full observability
- Testing with real MLflow tracking
- Debugging Temporal workflows
- Any development that needs monitoring

**Setup Once**:
```bash
# 1. Create Kind cluster with monitoring stack
./scripts/setup-monitoring-stack.sh

# 2. Start Docker Compose services
docker-compose -f infrastructure/docker-compose.dev.yml up -d
```

**Daily Usage**:
```bash
# Start backend (hot reload enabled)
cd platform/backend
poetry run uvicorn app.main:app --reload --port 8000

# Start frontend (in another terminal)
cd platform/frontend
pnpm dev

# Monitoring is already running in Kind cluster
# Open http://localhost:3001 for Grafana
# Open http://localhost:5000 for MLflow
```

### Tier 2: Kind (Local Kubernetes)

**Purpose**: K8s testing before production deploy

**Training Execution**: K8s Job in Kind cluster

**Infrastructure**:
```
Developer Machine
└── Kind Cluster
    ├── Namespace: platform
    │   ├── frontend Pod
    │   ├── backend Pod
    │   ├── postgres Pod
    │   ├── redis Pod
    │   └── minio Pod
    └── Namespace: training
        └── Training Job (created on-demand)
            └── trainer Pod
```

**Environment Variables** (in K8s ConfigMap/Secret):
```yaml
# ConfigMap: platform-config
EXECUTION_MODE: "kubernetes"
STORAGE_TYPE: "minio"
DATABASE_URL: "postgresql://postgres:5432/platform"
REDIS_URL: "redis://redis:6379/0"
S3_ENDPOINT: "http://minio:9000"
```

**Pros**:
- Tests K8s manifests
- Tests networking and DNS
- Tests resource limits
- Identical to production environment

**Cons**:
- Slower iteration (build → push → deploy)
- Harder to debug
- More complex setup

**When to Use**:
- Integration testing
- K8s manifest validation
- Pre-production testing
- CI/CD pipeline

### Tier 3: Production (Kubernetes Cluster)

**Purpose**: Live production environment

**Training Execution**: K8s Job in cloud cluster

**Infrastructure**:
```
Cloud Provider (Railway / AWS / On-Premise)
├── K8s Cluster
│   ├── Namespace: platform
│   │   ├── frontend Deployment
│   │   ├── backend Deployment
│   │   └── redis Deployment
│   └── Namespace: training-{model-id}
│       └── Training Jobs (isolated)
├── Managed PostgreSQL
└── S3/R2 Storage
```

**Environment Variables** (in K8s Secret):
```yaml
# Secret: platform-secrets
EXECUTION_MODE: "kubernetes"
STORAGE_TYPE: "r2"  # or s3
DATABASE_URL: "postgresql://user:pass@db.example.com:5432/platform"
REDIS_URL: "redis://redis.platform.svc.cluster.local:6379/0"
R2_ENDPOINT: "https://xxx.r2.cloudflarestorage.com"
R2_ACCESS_KEY_ID: "xxx"
R2_SECRET_ACCESS_KEY: "xxx"
```

**Pros**:
- Real user traffic
- Actual cloud resources
- Full observability stack
- Auto-scaling

**Cons**:
- Costs money
- Can't easily rollback
- Harder to debug

**When to Use**:
- Production deployment
- Load testing
- User acceptance testing

## Configuration Strategy

### The EXECUTION_MODE Pattern

**Core Abstraction**: The `EXECUTION_MODE` environment variable controls how training jobs are executed.

```python
# platform/backend/app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    EXECUTION_MODE: str = "subprocess"  # subprocess | kubernetes
    STORAGE_TYPE: str = "local"         # local | minio | r2 | s3

    DATABASE_URL: str
    REDIS_URL: str

    # Storage credentials (conditional based on STORAGE_TYPE)
    S3_ENDPOINT: str | None = None
    R2_ENDPOINT: str | None = None
    R2_ACCESS_KEY_ID: str | None = None
    R2_SECRET_ACCESS_KEY: str | None = None

    class Config:
        env_file = ".env"
```

### Training Execution Abstraction

```python
# platform/backend/app/services/training_executor.py
import subprocess
from kubernetes import client as k8s_client
from app.config import settings

class TrainingExecutor:
    async def start_training(self, job_config: dict) -> str:
        if settings.EXECUTION_MODE == "subprocess":
            return await self._start_subprocess(job_config)
        elif settings.EXECUTION_MODE == "kubernetes":
            return await self._start_kubernetes_job(job_config)
        else:
            raise ValueError(f"Unknown EXECUTION_MODE: {settings.EXECUTION_MODE}")

    async def _start_subprocess(self, job_config: dict) -> str:
        """Tier 1: Spawn subprocess"""
        env = {
            "JOB_ID": str(job_config["job_id"]),
            "TRACE_ID": job_config["trace_id"],
            "BACKEND_BASE_URL": "http://localhost:8000",
            "CALLBACK_TOKEN": job_config["callback_token"],
            "TASK_TYPE": job_config["task_type"],
            "MODEL_NAME": job_config["model_name"],
            "DATASET_PATH": f"/tmp/datasets/{job_config['dataset_id']}",
            # Storage not needed in subprocess mode - uses local files
        }

        trainer_script = f"platform/trainers/{job_config['framework']}/train.py"
        process = subprocess.Popen(
            ["python", trainer_script],
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        return f"subprocess-{process.pid}"

    async def _start_kubernetes_job(self, job_config: dict) -> str:
        """Tier 2 & 3: Create K8s Job"""
        storage_env = self._get_storage_env()

        job_manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"training-job-{job_config['job_id']}",
                "namespace": "training"
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "trainer",
                            "image": f"trainer-{job_config['framework']}:latest",
                            "env": [
                                {"name": "JOB_ID", "value": str(job_config["job_id"])},
                                {"name": "TRACE_ID", "value": job_config["trace_id"]},
                                {"name": "BACKEND_BASE_URL", "value": os.environ["BACKEND_BASE_URL"]},
                                {"name": "CALLBACK_TOKEN", "value": job_config["callback_token"]},
                                {"name": "TASK_TYPE", "value": job_config["task_type"]},
                                {"name": "MODEL_NAME", "value": job_config["model_name"]},
                                {"name": "DATASET_ID", "value": job_config["dataset_id"]},
                                *storage_env,  # R2/S3 credentials
                            ],
                            "resources": {
                                "requests": {"memory": "4Gi", "cpu": "2"},
                                "limits": {"memory": "8Gi", "cpu": "4"}
                            }
                        }],
                        "restartPolicy": "Never"
                    }
                }
            }
        }

        api = k8s_client.BatchV1Api()
        await api.create_namespaced_job("training", job_manifest)

        return f"training-job-{job_config['job_id']}"

    def _get_storage_env(self) -> list[dict]:
        """
        Get S3 storage credentials.

        All tiers use S3-compatible API:
        - Tier 1: MinIO (Docker Compose)
        - Tier 2: MinIO (Kind)
        - Tier 3: R2 or S3 (Production)
        """
        return [
            {"name": "S3_ENDPOINT", "value": settings.S3_ENDPOINT},
            {"name": "AWS_ACCESS_KEY_ID", "value": settings.AWS_ACCESS_KEY_ID},
            {"name": "AWS_SECRET_ACCESS_KEY", "value": settings.AWS_SECRET_ACCESS_KEY},
            {"name": "BUCKET_NAME", "value": settings.BUCKET_NAME},
        ]
```

### Storage Abstraction

**All tiers use S3-compatible API** - only the endpoint changes.

```python
# platform/backend/app/services/storage.py
import boto3
import os
from pathlib import Path
import zipfile

class S3Storage:
    """
    S3-compatible storage client.

    Works identically across all tiers:
    - Tier 1: MinIO (Docker Compose) - http://localhost:9000
    - Tier 2: MinIO (Kind) - http://minio.platform.svc:9000
    - Tier 3: Cloudflare R2 or AWS S3
    """

    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=os.environ["S3_ENDPOINT"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )
        self.bucket = os.environ.get("BUCKET_NAME", "vision-platform")

    async def upload_dataset(self, file_path: str, dataset_id: str) -> str:
        """Upload dataset to S3"""
        s3_key = f"datasets/{dataset_id}.zip"

        self.client.upload_file(
            file_path,
            self.bucket,
            s3_key
        )

        return f"s3://{self.bucket}/{s3_key}"

    async def download_dataset(self, dataset_id: str, dest_path: str) -> str:
        """Download and extract dataset from S3"""
        zip_path = f"{dest_path}/dataset.zip"

        # Download from S3
        self.client.download_file(
            self.bucket,
            f"datasets/{dataset_id}.zip",
            zip_path
        )

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)

        return dest_path

    async def upload_checkpoint(self, source_path: str, job_id: str) -> str:
        """Upload checkpoint to S3"""
        filename = Path(source_path).name
        s3_key = f"checkpoints/job-{job_id}/{filename}"

        self.client.upload_file(
            source_path,
            self.bucket,
            s3_key
        )

        return f"s3://{self.bucket}/{s3_key}"

def get_storage() -> S3Storage:
    """Get S3 storage client - works for all tiers"""
    return S3Storage()
```

**Key Point**: The SAME boto3 code works in all tiers. Only `S3_ENDPOINT` differs:

```python
# Tier 1 (.env)
S3_ENDPOINT=http://localhost:9000

# Tier 2 (K8s ConfigMap)
S3_ENDPOINT: "http://minio.platform.svc.cluster.local:9000"

# Tier 3 (K8s Secret)
S3_ENDPOINT: "https://xxx.r2.cloudflarestorage.com"
```

## Development Workflow

### Day-to-Day Development (Tier 1)

```bash
# 1. Start infrastructure
docker-compose up -d  # postgres, redis, minio (optional)

# 2. Start backend
cd platform/backend
cp .env.tier1.example .env
poetry install
poetry run uvicorn app.main:app --reload --port 8000

# 3. Start frontend
cd platform/frontend
cp .env.tier1.example .env.local
pnpm install
pnpm dev  # localhost:3000

# 4. Make changes
# Edit code in VS Code
# Backend auto-reloads
# Frontend auto-reloads

# 5. Test
poetry run pytest tests/unit/
pnpm test

# 6. Run training (subprocess mode)
# Just use the UI - backend will spawn subprocess
# Or test directly:
cd platform/trainers/ultralytics
python train.py  # Uses env vars from .env
```

### Pre-Deployment Testing (Tier 2)

```bash
# 1. Create Kind cluster
kind create cluster --name vision-platform

# 2. Build Docker images
cd platform/backend
docker build -t backend:latest .

cd platform/frontend
docker build -t frontend:latest .

cd platform/trainers/ultralytics
docker build -t trainer-ultralytics:latest .

# 3. Load images into Kind
kind load docker-image backend:latest --name vision-platform
kind load docker-image frontend:latest --name vision-platform
kind load docker-image trainer-ultralytics:latest --name vision-platform

# 4. Deploy with Helm
cd platform/infrastructure/helm

# Create namespace
kubectl create namespace platform

# Install PostgreSQL
helm install postgres bitnami/postgresql \
  --namespace platform \
  -f values/postgres-kind.yaml

# Install Redis
helm install redis bitnami/redis \
  --namespace platform \
  -f values/redis-kind.yaml

# Install MinIO
helm install minio bitnami/minio \
  --namespace platform \
  -f values/minio-kind.yaml

# Install Backend
helm install backend ./backend \
  --namespace platform \
  -f values/backend-kind.yaml

# Install Frontend
helm install frontend ./frontend \
  --namespace platform \
  -f values/frontend-kind.yaml

# 5. Port-forward for access
kubectl port-forward -n platform svc/frontend 3000:3000
kubectl port-forward -n platform svc/backend 8000:8000

# 6. Run integration tests
cd platform/backend
poetry run pytest tests/integration/ --kind

# 7. Test training job
# Use UI or API to create training job
# Verify K8s Job is created:
kubectl get jobs -n training

# Watch logs:
kubectl logs -n training job/training-job-{id} -f

# 8. Cleanup
kind delete cluster --name vision-platform
```

### Production Deployment (Tier 3)

```bash
# 1. Tag and push images
docker tag backend:latest registry.example.com/backend:v1.0.0
docker push registry.example.com/backend:v1.0.0

docker tag frontend:latest registry.example.com/frontend:v1.0.0
docker push registry.example.com/frontend:v1.0.0

docker tag trainer-ultralytics:latest registry.example.com/trainer-ultralytics:v1.0.0
docker push registry.example.com/trainer-ultralytics:v1.0.0

# 2. Create production namespace
kubectl create namespace platform --context=production

# 3. Create secrets
kubectl create secret generic platform-secrets \
  --from-literal=DATABASE_URL=postgresql://... \
  --from-literal=R2_ACCESS_KEY_ID=xxx \
  --from-literal=R2_SECRET_ACCESS_KEY=xxx \
  --namespace platform \
  --context=production

# 4. Deploy with Helm
helm install backend ./infrastructure/helm/backend \
  --namespace platform \
  -f values/backend-production.yaml \
  --context=production

helm install frontend ./infrastructure/helm/frontend \
  --namespace platform \
  -f values/frontend-production.yaml \
  --context=production

# 5. Verify deployment
kubectl get pods -n platform --context=production
kubectl get svc -n platform --context=production

# 6. Monitor
# Access Grafana dashboard
# Check logs in Loki
# View traces in Grafana Tempo
```

## Testing Strategy

### Unit Tests (Tier 1)

**Scope**: Individual functions, classes

**Environment**: No external dependencies (mocked)

```bash
cd platform/backend
poetry run pytest tests/unit/ -v

cd platform/frontend
pnpm test
```

**Example**:
```python
# tests/unit/test_storage.py
from unittest.mock import patch, MagicMock
from app.services.storage import get_storage, LocalStorage, S3Storage
from app.config import Settings

def test_get_storage_local():
    with patch('app.config.settings', Settings(STORAGE_TYPE="local")):
        storage = get_storage()
        assert isinstance(storage, LocalStorage)

def test_get_storage_s3():
    with patch('app.config.settings', Settings(STORAGE_TYPE="s3")):
        storage = get_storage()
        assert isinstance(storage, S3Storage)
```

### Integration Tests (Tier 1 + Tier 2)

**Scope**: Multiple components together

**Tier 1 Environment**: PostgreSQL, Redis (Docker)

```bash
# Start dependencies
docker-compose up -d postgres redis minio

# Run tests
cd platform/backend
poetry run pytest tests/integration/ -v --tier=subprocess
```

**Tier 2 Environment**: Full Kind cluster

```bash
# Tests run in CI/CD pipeline
poetry run pytest tests/integration/ -v --tier=kind
```

**Example**:
```python
# tests/integration/test_training_flow.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_training_job_subprocess(db_session):
    """Test creating training job in subprocess mode"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/training/jobs",
            json={
                "task_type": "image_classification",
                "model_name": "resnet50",
                "dataset_id": "test-dataset",
                "epochs": 1
            },
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 201
        job = response.json()
        assert job["status"] == "pending"

        # Wait for subprocess to start
        await asyncio.sleep(2)

        # Check job status
        response = await client.get(f"/api/v1/training/jobs/{job['id']}")
        assert response.json()["status"] in ["running", "completed"]

@pytest.mark.k8s
@pytest.mark.asyncio
async def test_create_training_job_kubernetes(k8s_client):
    """Test creating training job in K8s mode"""
    # Similar test but verifies K8s Job is created
    ...
```

### End-to-End Tests (Tier 2)

**Scope**: Full user workflow

**Environment**: Kind cluster with all services

```bash
cd platform/frontend
pnpm test:e2e
```

**Example** (Playwright):
```typescript
// tests/e2e/training-flow.spec.ts
import { test, expect } from '@playwright/test';

test('complete training workflow', async ({ page }) => {
  // Login
  await page.goto('http://localhost:3000/login');
  await page.fill('input[name="email"]', 'test@example.com');
  await page.fill('input[name="password"]', 'password');
  await page.click('button[type="submit"]');

  // Navigate to training
  await page.click('text=New Training');

  // Configure via chat
  await page.fill('textarea[name="message"]',
    'Train a ResNet-50 on my cat-dog dataset for 5 epochs');
  await page.click('button:has-text("Send")');

  // Wait for LLM response
  await expect(page.locator('.chat-message')).toContainText('I will train');

  // Start training
  await page.click('button:has-text("Start Training")');

  // Wait for progress updates
  await expect(page.locator('.training-status')).toContainText('running', {
    timeout: 30000
  });

  // Verify Grafana dashboard link
  await expect(page.locator('a[href*="grafana"]')).toBeVisible();
});
```

### Testing Matrix

| Test Type | Tier 1 | Tier 2 | Tier 3 |
|-----------|--------|--------|--------|
| Unit Tests | ✓ | - | - |
| Integration Tests (subprocess) | ✓ | - | - |
| Integration Tests (K8s) | - | ✓ | - |
| E2E Tests | - | ✓ | - |
| Smoke Tests | - | - | ✓ |
| Load Tests | - | - | ✓ |

## Migration Between Tiers

### Tier 1 → Tier 2 Checklist

- [ ] All unit tests pass
- [ ] Integration tests pass (subprocess mode)
- [ ] Docker images build successfully
- [ ] Kubernetes manifests valid (`kubectl apply --dry-run`)
- [ ] ConfigMaps and Secrets created
- [ ] Environment variables match

**Common Issues**:
- **DNS resolution**: Use service names (e.g., `postgres` not `localhost`)
- **File paths**: Use `/workspace` in containers, not local paths
- **Ports**: Services exposed on different ports in K8s
- **Permissions**: FS permissions differ in containers

### Tier 2 → Tier 3 Checklist

- [ ] Integration tests pass (Kind)
- [ ] E2E tests pass
- [ ] Resource limits defined
- [ ] Production secrets created
- [ ] Ingress/LoadBalancer configured
- [ ] Monitoring dashboards created
- [ ] Backup strategy in place

**Common Issues**:
- **Image registry**: Images must be in accessible registry (not local)
- **Storage**: R2/S3 credentials must be valid
- **Database**: Managed database connection string differs
- **Networking**: LoadBalancer/Ingress setup

## Troubleshooting

### Tier 1 Issues

**Problem**: `ModuleNotFoundError`
```bash
# Solution: Install dependencies
cd platform/backend
poetry install
```

**Problem**: Port 8000 already in use
```bash
# Solution: Kill existing process
lsof -ti:8000 | xargs kill -9
# Or use different port
uvicorn app.main:app --port 8001
```

**Problem**: PostgreSQL connection refused
```bash
# Solution: Start Docker services
docker-compose up -d postgres
# Verify
docker ps | grep postgres
```

### Tier 2 Issues

**Problem**: Image not found in Kind
```bash
# Solution: Load image into Kind
kind load docker-image backend:latest --name vision-platform
```

**Problem**: Pods in `ImagePullBackOff`
```bash
# Solution: Check image name and load into Kind
kubectl describe pod <pod-name> -n platform
kind load docker-image <image-name> --name vision-platform
```

**Problem**: Backend can't connect to PostgreSQL
```bash
# Solution: Verify service name in DATABASE_URL
# Should be: postgresql://postgres:5432/platform
# NOT: postgresql://localhost:5432/platform

# Check service exists
kubectl get svc -n platform
```

**Problem**: Training Job not starting
```bash
# Check Job status
kubectl get jobs -n training
kubectl describe job <job-name> -n training

# Check Pod logs
kubectl get pods -n training
kubectl logs <pod-name> -n training

# Common issue: Trainer image not loaded
kind load docker-image trainer-ultralytics:latest --name vision-platform
```

### Tier 3 Issues

**Problem**: Pods in `CrashLoopBackOff`
```bash
# Check logs
kubectl logs <pod-name> -n platform --previous

# Common cause: Missing environment variables
kubectl get secret platform-secrets -n platform -o yaml
```

**Problem**: 502 Bad Gateway
```bash
# Check backend is running
kubectl get pods -n platform
kubectl logs -l app=backend -n platform

# Check Ingress
kubectl get ingress -n platform
kubectl describe ingress platform-ingress -n platform
```

**Problem**: Training jobs fail immediately
```bash
# Check trainer logs
kubectl logs job/training-job-{id} -n training

# Common causes:
# - Invalid R2/S3 credentials
# - Dataset not found
# - Insufficient resources

# Verify secrets
kubectl get secret platform-secrets -n platform -o yaml | grep R2
```

## Environment Variable Reference

### Complete .env.tier1 (Subprocess)

```bash
# Execution
EXECUTION_MODE=subprocess
STORAGE_TYPE=local

# Database
DATABASE_URL=postgresql://admin:devpass@localhost:5432/platform

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...

# JWT
JWT_SECRET=your-super-secret-dev-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Backend
BACKEND_BASE_URL=http://localhost:8000

# Storage (optional for subprocess)
# S3_ENDPOINT=http://localhost:9000
# AWS_ACCESS_KEY_ID=minioadmin
# AWS_SECRET_ACCESS_KEY=minioadmin
```

### Complete ConfigMap for Tier 2 (Kind)

```yaml
# platform-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: platform-config
  namespace: platform
data:
  EXECUTION_MODE: "kubernetes"
  STORAGE_TYPE: "minio"
  DATABASE_URL: "postgresql://postgres:postgres@postgres:5432/platform"
  REDIS_URL: "redis://redis:6379/0"
  BACKEND_BASE_URL: "http://backend:8000"
  S3_ENDPOINT: "http://minio:9000"
  JWT_ALGORITHM: "HS256"
  ACCESS_TOKEN_EXPIRE_MINUTES: "60"
```

```yaml
# platform-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: platform-secrets
  namespace: platform
type: Opaque
stringData:
  JWT_SECRET: "your-super-secret-dev-key"
  ANTHROPIC_API_KEY: "sk-ant-api03-..."
  OPENAI_API_KEY: "sk-..."
  AWS_ACCESS_KEY_ID: "minioadmin"
  AWS_SECRET_ACCESS_KEY: "minioadmin"
```

### Complete Secret for Tier 3 (Production)

```yaml
# platform-secrets-prod.yaml
apiVersion: v1
kind: Secret
metadata:
  name: platform-secrets
  namespace: platform
type: Opaque
stringData:
  # Execution
  EXECUTION_MODE: "kubernetes"
  STORAGE_TYPE: "r2"

  # Database (Managed PostgreSQL)
  DATABASE_URL: "postgresql://user:pass@db.railway.app:5432/railway"

  # Redis
  REDIS_URL: "redis://redis.platform.svc.cluster.local:6379/0"

  # LLM
  ANTHROPIC_API_KEY: "sk-ant-api03-..."
  OPENAI_API_KEY: "sk-..."

  # JWT
  JWT_SECRET: "production-secret-from-1password"
  JWT_ALGORITHM: "HS256"
  ACCESS_TOKEN_EXPIRE_MINUTES: "60"

  # Backend
  BACKEND_BASE_URL: "https://api.example.com"

  # R2 Storage
  R2_ENDPOINT: "https://xxx.r2.cloudflarestorage.com"
  R2_ACCESS_KEY_ID: "xxx"
  R2_SECRET_ACCESS_KEY: "xxx"
```

## Best Practices

1. **Always test in Tier 1 first** - Fastest feedback loop
2. **Test K8s changes in Tier 2** - Before pushing to production
3. **Use same environment variable names** - Across all tiers
4. **Keep tier-specific config minimal** - Only what's necessary
5. **Version control .env.example files** - Not actual .env files
6. **Use helper scripts** - For switching between tiers
7. **Document tier-specific quirks** - In code comments

## Summary

| Aspect | Tier 1 | Tier 2 | Tier 3 |
|--------|--------|--------|--------|
| **Code** | Same | Same | Same |
| **Config** | .env file | K8s ConfigMap/Secret | K8s Secret |
| **Training** | Subprocess | K8s Job | K8s Job |
| **Storage** | Local FS or MinIO | MinIO | R2/S3 |
| **Database** | Local PostgreSQL | K8s PostgreSQL | Managed PostgreSQL |
| **Iteration Speed** | Fast (seconds) | Medium (minutes) | Slow (deploy) |
| **Fidelity** | Low (no K8s) | High (same as prod) | Production |

The key insight: **One codebase, three configurations, identical behavior.**

## References

- [Architecture Overview](../architecture/OVERVIEW.md)
- [Backend Design](../architecture/BACKEND_DESIGN.md)
- [Trainer Design](../architecture/TRAINER_DESIGN.md)
- [Infrastructure Design](../architecture/INFRASTRUCTURE_DESIGN.md)
