# Development Workflow for Fast Training Iteration

**Date**: 2025-11-07
**Status**: Implemented
**Related Files**:
- `dev-train-local.ps1`
- `dev-train-k8s.ps1`
- `dev-start.ps1`, `dev-stop.ps1`, `dev-status.ps1`
- `mvp/backend/app/utils/training_manager.py`
- `mvp/training/api_server.py`

## Overview

Established a 3-tier development workflow that eliminates the need to rebuild Docker images for every code change. Developers can iterate on training code in seconds using local Python execution, test in Kubernetes when needed, and only build images for final deployment.

## Background / Context

**Problem Statement**: "실제 training job을 실행해서 테스트하면 수정 사항이 많이 발생할거야. 이때는 어떻게 돼? 매번 빌드를 해야하는건가?"

Rebuilding Docker images for every training code change takes 10-15 minutes, which makes iterative development extremely slow and frustrating.

**User Need**: Full-stack integration testing (Frontend ↔ Backend ↔ Training ↔ MLflow ↔ MinIO) during daily development.

## Current State

### Infrastructure
- **Kind Cluster**: Local Kubernetes cluster running MLflow and MinIO
- **MLflow**: Experiment tracking with SQLite backend + MinIO artifact store
- **MinIO**: S3-compatible object storage (replaces R2 for local dev)
- **Sample Dataset**: 50 images (cats/dogs) at `mvp/data/datasets/sample_dataset/`

### Architecture
```
Frontend (localhost:3000)
    ↓
Backend API (localhost:8000)
    ↓
Training Service API (api_server.py)
    ↓
Training Execution (subprocess or K8s Job)
    ↓
MLflow (Kind:30500) + MinIO (Kind:30900)
```

## Proposed Solution / Decision

### Three-Tier Development Approach

#### Tier 1: Local Development (99% of time) ⚡⚡⚡
**Execution**: subprocess (direct Python)
**Speed**: 5-30 seconds
**Use case**: Daily development, rapid iteration

```powershell
# Backend runs training via subprocess
subprocess.Popen([
    "venv-{framework}/python", "train.py",
    "--framework", "ultralytics",
    "--job_id", "123",
    # ... other params
])
```

#### Tier 2: K8s Testing (before deployment) ⚡⚡
**Execution**: K8s Job with ConfigMap code injection
**Speed**: 1-3 minutes
**Use case**: Validate K8s environment, final integration testing

```powershell
# Inject code via ConfigMap (no image rebuild)
.\dev-train-k8s.ps1 -Watch
```

#### Tier 3: Production Deployment ⚡
**Execution**: K8s Job with built Docker images
**Speed**: 10-15 minutes (build time)
**Use case**: Production deployment only

### Key Design Choices

1. **subprocess for Local Development**
   - **Rationale**: Eliminates Docker build overhead, enables fast iteration
   - **Trade-offs**: Local environment differs slightly from production (acceptable for dev)
   - **Implementation**: `api_server.py:86-139` executes `subprocess.Popen()`

2. **Framework-Specific Virtual Environments**
   - **Rationale**: Prevents dependency conflicts between frameworks (timm, ultralytics, huggingface)
   - **Trade-offs**: Multiple venvs use more disk space (~500MB each)
   - **Implementation**: `venv-timm/`, `venv-ultralytics/`, `venv-huggingface/`

3. **ConfigMap Code Injection for K8s Testing**
   - **Rationale**: Test K8s environment without rebuilding images
   - **Trade-offs**: ConfigMap has size limits (~1MB), not suitable for large codebases
   - **Implementation**: `dev-train-k8s.ps1` creates ConfigMap and mounts at `/code/train.py`

4. **MinIO for Local Object Storage**
   - **Rationale**: S3-compatible, no internet/credentials needed, free
   - **Trade-offs**: Not identical to R2/S3 (acceptable for dev)
   - **Implementation**: Kind deployment at `mvp/k8s/minio-config.yaml`

## Implementation Plan

### Phase 1: Infrastructure Setup ✅
- [x] Deploy MLflow to Kind cluster with SQLite + MinIO
- [x] Deploy MinIO with persistent storage (20Gi PVC)
- [x] Create automation scripts (`dev-start.ps1`, `dev-stop.ps1`, `dev-status.ps1`)
- [x] Setup port forwarding (MLflow:30500, MinIO:30900/30901)

### Phase 2: Training Execution ✅
- [x] Implement subprocess execution in `api_server.py`
- [x] Create framework-specific virtual environments
- [x] Add environment variable auto-configuration in `dev-train-local.ps1`
- [x] Implement ConfigMap injection in `dev-train-k8s.ps1`

### Phase 3: Documentation ✅
- [x] Create `GETTING_STARTED.md` (5-minute quickstart)
- [x] Create `DEV_WORKFLOW.md` (detailed workflow guide)
- [x] Create `QUICK_DEV_GUIDE.md` (one-page reference)
- [x] Update `README.md` with quickstart link

## Technical Details

### Environment Variables (Auto-set by dev-train-local.ps1)

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:30500"
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:30900"
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin"
$env:MLFLOW_S3_IGNORE_TLS = "true"
$env:JOB_ID = "local-20251107-143000"
$env:MODEL_NAME = "yolo11n"
$env:FRAMEWORK = "ultralytics"
$env:NUM_EPOCHS = "10"
```

### Training Service Framework Routing

```python
# training_client.py:19-23
FRAMEWORK_SERVICES = {
    "timm": "TIMM_SERVICE_URL",
    "ultralytics": "ULTRALYTICS_SERVICE_URL",
    "huggingface": "HUGGINGFACE_SERVICE_URL",
}

# api_server.py:98-106
venv_python = f"venv-{request.framework}/Scripts/python.exe"
if os.path.exists(venv_python):
    python_exe = venv_python  # Use framework-specific venv
else:
    python_exe = "python"  # Fallback to system python
```

### ConfigMap Code Injection

```yaml
# dev-train-k8s.ps1 creates:
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-code-dev-20251107-143000
data:
  train.py: |
    # Actual train.py content

---
apiVersion: batch/v1
kind: Job
spec:
  template:
    spec:
      containers:
      - name: trainer
        volumeMounts:
        - name: training-code
          mountPath: /code/train.py
          subPath: train.py
      volumes:
      - name: training-code
        configMap:
          name: training-code-dev-20251107-143000
```

## Alternatives Considered

### 1. Docker Volume Mounts
**Approach**: Mount local code into Docker container
```bash
docker run -v ./mvp/training:/code trainer-image
```
**Pros**:
- Fast iteration (no rebuild)
- Closer to production environment

**Cons**:
- Requires Docker Compose setup
- More complex than subprocess
- Still slower than native Python

**Why Rejected**: subprocess is simpler and faster for local dev

### 2. Hot Reload in Container
**Approach**: Use file watcher to reload code in running container

**Pros**:
- No container restart needed
- Production-like environment

**Cons**:
- Complex to implement
- Not suitable for training scripts (stateful)
- Overkill for batch jobs

**Why Rejected**: Training is batch process, not a web server

### 3. Always Use K8s Jobs (Even Locally)
**Approach**: Run K8s Jobs for every training, even during development

**Pros**:
- Identical to production
- Tests K8s environment constantly

**Cons**:
- 1-3 minute startup time (Pod scheduling, image pull)
- Harder to debug (logs in K8s)
- Slower iteration

**Why Rejected**: Too slow for daily development (99% use case)

## Migration Path

### From Current State to This Workflow

**Step 1: One-time Setup**
```powershell
# Start K8s infrastructure
.\dev-start.ps1 -SkipBuild

# Setup Python environment
cd mvp/training
python -m venv venv-ultralytics
.\venv-ultralytics\Scripts\activate
pip install -r requirements.txt
```

**Step 2: Daily Development**
```powershell
# Terminal 1: Backend
cd mvp/backend
.\venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd mvp/frontend
npm run dev

# Terminal 3: Browser
start http://localhost:3000
```

**Step 3: Iterate on Training Code**
```powershell
# Edit training code
vim mvp/training/train.py

# Training runs automatically via Backend API (subprocess)
# → Check results in MLflow: http://localhost:30500
```

**Step 4: Final K8s Validation (Before Deployment)**
```powershell
# Test in K8s environment (ConfigMap injection)
.\dev-train-k8s.ps1 -Watch
```

## Development Speed Comparison

| Method | Time | When to Use |
|--------|------|-------------|
| **Subprocess (Tier 1)** | 5-30s | Daily development (99%) |
| **ConfigMap K8s (Tier 2)** | 1-3min | Pre-deployment testing |
| **Docker Build (Tier 3)** | 10-15min | Final deployment only |

### Typical Development Day

```
Edit code (10 iterations):
  - Old way: 10 × 15min = 150 minutes
  - New way: 10 × 30sec = 5 minutes

Time saved: 145 minutes per day (2.4 hours!)
```

## References

### Related Files
- **Scripts**: `dev-start.ps1`, `dev-stop.ps1`, `dev-status.ps1`, `dev-train-local.ps1`, `dev-train-k8s.ps1`
- **Backend**: `mvp/backend/app/utils/training_manager.py`, `mvp/backend/app/utils/training_client.py`
- **Training**: `mvp/training/api_server.py`, `mvp/training/train.py`
- **K8s**: `mvp/k8s/mlflow-config.yaml`, `mvp/k8s/minio-config.yaml`

### Related Documentation
- `GETTING_STARTED.md` - 5-minute quickstart guide
- `DEV_WORKFLOW.md` - Detailed workflow explanation
- `QUICK_DEV_GUIDE.md` - One-page reference
- `mvp/k8s/MLFLOW_SETUP.md` - MLflow usage guide
- `mvp/k8s/MINIO_SETUP.md` - MinIO setup guide
- `mvp/k8s/DOCKER_VS_K8S.md` - Environment comparison
- `mvp/k8s/DATA_PERSISTENCE.md` - Data persistence strategy

### Key Decisions Log
- **2025-11-07**: Replaced R2 with MinIO for local development
- **2025-11-07**: Implemented 3-tier development workflow
- **2025-11-07**: Created automation scripts for environment management
- **2025-11-07**: Documented ConfigMap injection approach for K8s testing

## Notes

### Open Questions
- Should we add a `dev-train-hybrid.ps1` that starts Backend+Frontend automatically?
- Consider adding health check endpoint to Training Service for better monitoring
- Future: Add Docker Compose as alternative to Kind for users who prefer it

### Future Considerations
1. **Production Migration**: When deploying to production, switch from subprocess to K8s Jobs
   - Add environment variable: `EXECUTION_MODE=k8s` vs `EXECUTION_MODE=subprocess`
   - Implement K8s Job creation in `training_manager.py`

2. **Multi-Framework Support**: Currently supports timm, ultralytics, huggingface
   - Framework-specific venvs work well
   - Consider adding more frameworks (MMDetection, Detectron2)

3. **Distributed Training**: Current subprocess approach is single-machine
   - K8s Jobs support distributed training (multi-GPU)
   - Consider adding distributed training support in future

### User Feedback
- **Initial confusion**: "가이드가 너무 많고 복잡해" → Created this session doc for clarity
- **Clarification needed**: "개발시에는 subprocess로 개발하기 때문인거야?" → Confirmed, documented rationale
- **Framework separation**: "프레임워크별로 각각의 training job을 실행할거야" → Already implemented and documented

### Performance Metrics
- **Environment startup**: 2-3 minutes (with `-SkipBuild`)
- **Local training execution**: 5-30 seconds (depends on model/epochs)
- **K8s Job startup**: 1-3 minutes (Pod scheduling + ConfigMap mount)
- **Docker image build**: 10-15 minutes (full rebuild)

### Infrastructure Resources
- **MLflow PVC**: 5Gi (SQLite database + metadata)
- **MinIO PVC**: 20Gi (training datasets, checkpoints, results)
- **Kind Cluster**: ~2GB RAM, 2 CPU cores
- **Framework venvs**: ~500MB each (timm, ultralytics, huggingface)
