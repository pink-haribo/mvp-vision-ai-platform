# 3-Tier Development Strategy (Correct)

**Last Updated**: 2025-01-12

## 🎯 Strategy Overview

```
Tier 1: Kind + Subprocess Training
├── Backend, Frontend, Infrastructure → Kind (K8s)
└── Training → Subprocess (Fast iteration)

Tier 2: Fully Kind
├── All services → Kind (K8s)
└── Training → K8s Job

Tier 3: Production K8s
└── Everything in cloud K8s cluster
```

---

## Tier Comparison Table

| Component | Tier 1 | Tier 2 | Tier 3 |
|-----------|--------|--------|--------|
| **Backend** | Kind Pod | Kind Pod | K8s Deployment |
| **Frontend** | Kind Pod | Kind Pod | K8s Deployment |
| **PostgreSQL** | Kind Pod | Kind Pod | Managed DB |
| **Redis** | Kind Pod | Kind Pod | K8s Pod |
| **MinIO** | Kind Pod | Kind Pod | R2/S3 |
| **MLflow** | Kind Pod | Kind Pod | K8s Pod |
| **Prometheus** | Kind Pod | Kind Pod | K8s Pod |
| **Grafana** | Kind Pod | Kind Pod | K8s Pod |
| **Temporal** | Kind Pod | Kind Pod | K8s Deployment |
| **Training** | **Subprocess** 🚀 | **K8s Job** | **K8s Job** |

---

## Why This Strategy?

### Tier 1: Kind + Subprocess

**Why Kind for infra?**
- ✅ Production-like environment (K8s DNS, networking)
- ✅ MLflow, Prometheus, Grafana available
- ✅ Temporal workflows work identically

**Why Subprocess for training?**
- ✅ Instant start (< 1 second)
- ✅ Direct logs in terminal
- ✅ Use IDE debugger (breakpoints, pdb)
- ✅ No container rebuild needed

**Result**: Best of both worlds - production-like infrastructure with dev-friendly training.

### Tier 2: Fully Kind

**Purpose**: Final validation before production

- Training runs as K8s Job (exactly like production)
- Tests resource limits, pod lifecycle
- Validates all K8s manifests

### Tier 3: Production

**Only differences from Tier 2**:
- Managed PostgreSQL (not pod)
- R2/S3 (not MinIO)
- More replicas
- Ingress/LoadBalancer

---

## Environment Variables

**Tier 1** (Kind + Subprocess):
```bash
TRAINING_MODE=subprocess
BACKEND_URL=http://backend.platform.svc.cluster.local:8000  # Kind DNS
MINIO_ENDPOINT=http://minio.platform.svc.cluster.local:9000
```

**Tier 2** (Fully Kind):
```bash
TRAINING_MODE=kubernetes
BACKEND_URL=http://backend.platform.svc.cluster.local:8000
MINIO_ENDPOINT=http://minio.platform.svc.cluster.local:9000
```

**Tier 3** (Production):
```bash
TRAINING_MODE=kubernetes
BACKEND_URL=http://backend.platform.svc.cluster.local:8000
R2_ENDPOINT=https://your-account.r2.cloudflarestorage.com
```

---

## Quick Start

### Tier 1 Setup (One-time)

```bash
# 1. Create Kind cluster
cd platform/infrastructure
./scripts/setup-kind-cluster.sh

# 2. Deploy all services to Kind
kubectl apply -f k8s/platform/
kubectl apply -f k8s/mlflow/
kubectl apply -f k8s/observability/

# 3. Wait for ready
kubectl wait --for=condition=ready pod -l app=backend -n platform --timeout=300s

# 4. Port-forward for access
kubectl port-forward -n platform svc/backend 30080:8000 &
kubectl port-forward -n platform svc/frontend 30300:3000 &
kubectl port-forward -n platform svc/minio 30900:9000 &
```

### Daily Development (Tier 1)

```bash
# Services are already running in Kind!
# Just trigger training via API/UI

# Backend spawns subprocess automatically
# Logs appear in Backend pod logs:
kubectl logs -n platform deployment/backend -f
```

### Switch to Tier 2 (K8s Training)

```bash
# 1. Build trainer image
cd platform/trainers/ultralytics
docker build -t trainer-ultralytics:latest .

# 2. Load into Kind
kind load docker-image trainer-ultralytics:latest

# 3. Update Backend config
kubectl set env deployment/backend -n platform TRAINING_MODE=kubernetes
```

---

## Port Mappings

```yaml
# kind-config.yaml
extraPortMappings:
- containerPort: 30080  # Backend
  hostPort: 30080
- containerPort: 30300  # Frontend  
  hostPort: 30300
- containerPort: 30900  # MinIO API
  hostPort: 30900
- containerPort: 30901  # MinIO Console
  hostPort: 30901
- containerPort: 30500  # MLflow
  hostPort: 30500
- containerPort: 30090  # Prometheus
  hostPort: 30090
- containerPort: 30030  # Grafana
  hostPort: 30030
- containerPort: 30233  # Temporal UI
  hostPort: 30233
```

**Access URLs**:
- Frontend: http://localhost:30300
- Backend API: http://localhost:30080
- MinIO Console: http://localhost:30901
- MLflow: http://localhost:30500
- Grafana: http://localhost:30030
- Prometheus: http://localhost:30090
- Temporal UI: http://localhost:30233

---

## Training Code (Works in Both Tiers)

```python
# trainers/ultralytics/train.py
import os
import requests
import boto3

# Environment variables (same for subprocess and K8s)
JOB_ID = os.environ["JOB_ID"]
BACKEND_URL = os.environ["BACKEND_URL"]
CALLBACK_TOKEN = os.environ["CALLBACK_TOKEN"]
S3_ENDPOINT = os.environ["S3_ENDPOINT"]

# Download dataset from MinIO/R2
s3 = boto3.client('s3', endpoint_url=S3_ENDPOINT, ...)
s3.download_file('datasets', f'{dataset_id}.zip', '/tmp/data.zip')

# Training loop
for epoch in range(epochs):
    train_one_epoch()
    
    # Send heartbeat to Backend
    requests.post(
        f"{BACKEND_URL}/api/v1/jobs/{JOB_ID}/heartbeat",
        headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
        json={"epoch": epoch, "metrics": metrics}
    )
```

**Same code, different execution**:
- **Tier 1**: Runs as subprocess on host
- **Tier 2/3**: Runs in K8s Pod

---

## References

- Full details: [3_TIER_DEVELOPMENT.md](./3_TIER_DEVELOPMENT.md)
- Architecture: [../architecture/OVERVIEW.md](../architecture/OVERVIEW.md)
- Infrastructure setup: [../../infrastructure/README.md](../../infrastructure/README.md)
