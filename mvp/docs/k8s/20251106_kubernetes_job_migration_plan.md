# Kubernetes Job Migration Plan - Training Service Architecture Transition

**Date**: 2025-11-06
**Status**: Proposed
**Phase**: Phase 2 - Production Infrastructure
**Related Issues**: Training Service scalability, Resource isolation, Cloud GPU support

## Overview

This document outlines the migration plan from subprocess-based training execution to Kubernetes Job-based architecture. The migration enables containerized training workloads with resource isolation, automatic recovery, and cloud GPU support while maintaining 100% compatibility with existing code (train.py, adapters, Platform SDK).

**Key Insight**: The current architecture is already well-prepared for K8s migration. `train.py` serves as a framework-agnostic orchestrator that requires NO changes for K8s deployment.

## Background / Context

### Current Architecture (Phase 1 MVP)

**Execution Model**: HTTP API-based Training Services + Python subprocess

```
Frontend → Backend API → Training Services (FastAPI, Port 8002+)
                              ↓ HTTP POST /training/start
                         subprocess.Popen(["python", "train.py", ...])
                              ↓
                         Training Process
                              ↓
                         R2 Storage (Cloudflare R2)
```

**Current Components:**
- `mvp/backend/app/utils/training_manager.py` - Training orchestration
- `mvp/backend/app/utils/training_client.py` - HTTP client for Training Services
- `mvp/training/api_server.py` - Training Service API (FastAPI)
- `mvp/training/train.py` - Main training script (entry point)
- `mvp/training/adapters/` - Framework-specific implementations
- `mvp/training/platform_sdk/storage.py` - R2 integration

**Limitations:**
- ❌ No isolation between training jobs (same process space)
- ❌ No resource limits (CPU/GPU/memory)
- ❌ No automatic recovery on failure
- ❌ Limited to single machine capacity
- ❌ GPU idle time during non-training periods

### Target Architecture (Phase 2)

**Execution Model**: Kubernetes Jobs with containerized workloads

```
Frontend → Backend API → VM Controller (K8s Client)
                              ↓ K8s API
                         Kubernetes Cluster
                              ↓
                         Training Pod (Job)
                         ├─ Container: trainer (train.py)
                         └─ Container: sidecar (optional)
                              ↓
                         R2 Storage
```

**Benefits:**
- ✅ Full isolation (containers)
- ✅ Resource limits (CPU/GPU/memory quotas)
- ✅ Automatic recovery (pod restart policies)
- ✅ Horizontal scalability (multiple nodes)
- ✅ Better monitoring (K8s metrics, logs)
- ✅ Cloud GPU support (AWS ECS/EKS, GCP GKE)

## Current State Analysis

### train.py Structure

**Location**: `mvp/training/train.py` (419 lines)

**Input Interface**:
- **Command-line arguments** (30+ args via argparse):
  - `--framework` (timm, ultralytics, transformers)
  - `--model_name` (resnet18, yolo11n, etc.)
  - `--task_type` (image_classification, object_detection, etc.)
  - `--dataset_path` (dataset ID or local path)
  - `--epochs`, `--batch_size`, `--learning_rate`
  - `--job_id`, `--project_id`
  - `--advanced_config` (JSON string)
- **Environment variables** (.env file):
  - `AWS_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
  - `DATABASE_URL` (PostgreSQL connection)
  - `BACKEND_API_URL` (for TrainingLogger)
  - `MLFLOW_TRACKING_URI`

**Execution Flow**:
```python
1. Parse arguments (argparse)
2. Load advanced_config from CLI or database
3. Download dataset from R2 (if dataset_id provided)
   └─ get_dataset(dataset_id) → auto-download to /workspace/data/.cache/
4. Create config objects (ModelConfig, DatasetConfig, TrainingConfig)
5. Select adapter from registry (ADAPTER_REGISTRY[framework])
6. Initialize TrainingLogger (Backend API communication)
7. Create adapter instance
8. Execute training: adapter.train(...)
9. Upload checkpoints to R2 (via Platform SDK)
10. Update status via TrainingLogger
```

**Key Dependencies**:
- `platform_sdk` - Storage, configs, logger
- `adapters` - Framework-specific implementations
- `boto3` - R2 (S3-compatible) storage
- Framework packages (ultralytics, timm, transformers)

### Adapter Pattern Implementation

**Base Class**: `TrainingAdapter` (ABC)

```python
class TrainingAdapter(ABC):
    @abstractmethod
    def train(self, start_epoch, checkpoint_path, resume_training) -> List[MetricsResult]:
        """Execute training (framework-specific)"""
        pass

    @abstractmethod
    def validate(self) -> MetricsResult:
        """Execute validation"""
        pass

    @classmethod
    @abstractmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Return framework-specific config schema"""
        pass
```

**Current Implementations**:
- `UltralyticsAdapter` - ✅ Implemented (YOLO models)
- `TimmAdapter` - ⏳ Planned (ResNet, EfficientNet)
- `TransformersAdapter` - ⏳ Planned (HuggingFace models)

**Registry Pattern** (train.py:281):
```python
ADAPTER_REGISTRY = {
    "timm": TimmAdapter,
    "ultralytics": UltralyticsAdapter,
    "transformers": TransformersAdapter,
}
```

### Platform SDK Integration

**Storage Management** (`platform_sdk/storage.py`):

**3-tier fallback strategy**:
1. Local cache (`/workspace/data/.cache/`)
2. R2 (platform shared cache)
3. Original source (Ultralytics, timm, HuggingFace) → auto-upload to R2

**Key Functions**:
```python
# Dataset management
get_dataset(dataset_id, download_fn) -> str
    # Downloads from R2 to /workspace/data/.cache/datasets/{dataset_id}/

# Model weights management
get_model_weights(model_name, framework, download_fn) -> str
    # Downloads from R2 to /workspace/data/.cache/models/{framework}/{model_name}.pt

# Checkpoint management
upload_checkpoint(checkpoint_path, job_id, checkpoint_name, project_id) -> bool
    # Uploads to s3://bucket/checkpoints/projects/{project_id}/jobs/{job_id}/{checkpoint_name}

download_checkpoint(checkpoint_path, dest_path) -> str
    # Downloads checkpoint from R2 URL (r2://bucket/key)
```

**Storage Structure**:
```
s3://vision-platform-prod/
├── datasets/
│   └── {dataset_id}/
│       ├── images/
│       └── annotations/
├── models/
│   └── pretrained/
│       └── {framework}/
│           └── {model_name}.pt
└── checkpoints/
    ├── projects/{project_id}/jobs/{job_id}/
    │   ├── best.pt
    │   └── last.pt
    └── test-jobs/job_{job_id}/
        ├── best.pt
        └── last.pt
```

### Backend Communication

**TrainingLogger** (`platform_sdk/__init__.py`):
```python
class TrainingLogger:
    def update_status(self, status: str):
        # POST {backend_url}/api/v1/internal/training/jobs/{job_id}/status
        pass

    def log_metrics(self, epoch: int, metrics: dict):
        # POST {backend_url}/api/v1/internal/training/jobs/{job_id}/metrics
        pass

    def log_message(self, message: str, level: str):
        # POST {backend_url}/api/v1/internal/training/jobs/{job_id}/logs
        pass
```

**Used throughout training**:
- Status updates: `pending` → `running` → `completed`/`failed`
- Real-time metrics: every epoch
- Log messages: INFO, WARNING, ERROR

## Proposed Solution: Kubernetes Job Architecture

### Design Principles

1. **Zero Changes to train.py** - K8s Job runs train.py as-is
2. **Framework Independence** - Each framework has its own Docker image
3. **Adapter Pattern Preserved** - Adapters remain the abstraction layer
4. **Storage Consistency** - R2 integration works identically in containers
5. **Backend Communication Maintained** - TrainingLogger continues to work
6. **Incremental Migration** - Can run K8s and subprocess simultaneously

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Backend API                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ POST /api/v1/training/jobs/{id}/start                      │ │
│  │   ↓                                                         │ │
│  │ TrainingManager.start_training()                           │ │
│  │   ↓                                                         │ │
│  │ VMController.create_training_job(job_config)               │ │
│  │   - Build K8s Job manifest                                 │ │
│  │   - Submit to K8s API                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ K8s API
┌───────────────────────────▼─────────────────────────────────────┐
│                    Kubernetes Cluster                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Job: training-job-123                                      │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Pod: training-job-123-xxxxx                          │ │ │
│  │  │  ┌────────────────────────────────────────────────┐  │ │ │
│  │  │  │ Container: trainer                             │  │ │ │
│  │  │  │  - Image: vision-platform/trainer-{framework} │  │ │ │
│  │  │  │  - Command: python train.py                   │  │ │ │
│  │  │  │  - Args: [--framework, --model_name, ...]     │  │ │ │
│  │  │  │  - Env: R2 credentials, Backend URL           │  │ │ │
│  │  │  │  - Resources: GPU=1, Memory=16Gi              │  │ │ │
│  │  │  │                                                 │  │ │ │
│  │  │  │  Execution:                                    │  │ │ │
│  │  │  │  1. Download dataset from R2                   │  │ │ │
│  │  │  │  2. Download model weights from R2             │  │ │ │
│  │  │  │  3. Run training (adapter.train())             │  │ │ │
│  │  │  │  4. Upload checkpoints to R2                   │  │ │ │
│  │  │  │  5. Report status to Backend API               │  │ │ │
│  │  │  └────────────────────────────────────────────────┘  │ │ │
│  │  │                                                       │ │ │
│  │  │  ┌────────────────────────────────────────────────┐  │ │ │
│  │  │  │ Container: sidecar (OPTIONAL)                  │  │ │ │
│  │  │  │  - Monitor stdout for metrics                  │  │ │ │
│  │  │  │  - Watch for checkpoint files                  │  │ │ │
│  │  │  │  - Collect GPU/memory metrics                  │  │ │ │
│  │  │  │  - Send telemetry to Backend                   │  │ │ │
│  │  │  └────────────────────────────────────────────────┘  │ │ │
│  │  └───────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  GPU Node Pool (NVIDIA GPU Plugin)                             │
│  - Node selector: accelerator=nvidia-gpu                        │
│  - Resource limits: nvidia.com/gpu=1                           │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    R2 Storage (Cloudflare)                       │
│  - Datasets: datasets/{dataset_id}/                             │
│  - Pretrained Models: models/pretrained/{framework}/            │
│  - Checkpoints: checkpoints/projects/{project_id}/jobs/{job_id}/│
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

#### 1. Kubernetes Job vs Deployment

**Decision**: Use Kubernetes **Job** (not Deployment)

**Rationale**:
- Training is a batch workload (runs to completion)
- Job automatically marks as succeeded/failed
- `restartPolicy: Never` prevents infinite retry loops
- `ttlSecondsAfterFinished` for automatic cleanup
- `backoffLimit` for controlled retry attempts

**Job Spec**:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job-{{ job_id }}
spec:
  ttlSecondsAfterFinished: 86400  # 24h cleanup
  backoffLimit: 3                 # Max 3 retries
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        # ... container spec
```

**Trade-offs**:
- ✅ Simpler lifecycle management
- ✅ Automatic success/failure detection
- ✅ No need for manual cleanup
- ❌ No built-in long-running process support (use long timeout instead)

#### 2. Container Image Strategy: Framework-Specific Images

**Decision**: Separate Docker images per framework

**Rationale**:
- Different dependency sets (ultralytics vs timm vs transformers)
- Smaller image sizes (no unnecessary packages)
- Faster pull times
- Independent versioning

**Image Structure**:
```
vision-platform/trainer-ultralytics:v1.0
├── Base: nvidia/cuda:11.8.0-cudnn8-runtime
├── Python 3.11
├── ultralytics==8.3.0
├── torch, torchvision
├── boto3 (R2)
├── mlflow
├── COPY mvp/training/ → /app/training/
└── ENTRYPOINT: python train.py

vision-platform/trainer-timm:v1.0
├── Base: nvidia/cuda:11.8.0-cudnn8-runtime
├── Python 3.11
├── timm==1.0.0
├── torch, torchvision, albumentations
├── boto3, mlflow
├── COPY mvp/training/ → /app/training/
└── ENTRYPOINT: python train.py
```

**Selection Logic** (Backend):
```python
def get_training_image(framework: str) -> str:
    images = {
        "ultralytics": "vision-platform/trainer-ultralytics:v1.0",
        "timm": "vision-platform/trainer-timm:v1.0",
        "transformers": "vision-platform/trainer-transformers:v1.0",
    }
    return images[framework]
```

**Trade-offs**:
- ✅ Framework independence
- ✅ Smaller images (~2-3 GB each vs 8+ GB monolithic)
- ✅ Faster deployment
- ❌ Need to maintain multiple Dockerfiles
- ❌ More CI/CD pipelines

#### 3. Configuration Management: Hybrid Args + Environment Variables

**Decision**: Keep hybrid approach (no change from current)

**Rationale**:
- Command-line args for training config (framework, model, hyperparams)
- Environment variables for infrastructure (R2 credentials, API URLs)
- Matches current train.py interface
- K8s Secrets for sensitive data

**K8s Job Manifest**:
```yaml
containers:
- name: trainer
  command: ["python", "train.py"]
  args:
    - "--framework=ultralytics"
    - "--model_name=yolo11n"
    - "--task_type=object_detection"
    - "--dataset_path=abc123"      # R2 dataset ID
    - "--epochs=50"
    - "--batch_size=32"
    - "--learning_rate=0.001"
    - "--job_id=123"
    - "--project_id=5"
    - "--output_dir=/workspace/output"
    - "--advanced_config={{ json_string }}"

  env:
    # R2 Credentials (from Secret)
    - name: AWS_S3_ENDPOINT_URL
      valueFrom:
        secretKeyRef:
          name: r2-credentials
          key: endpoint
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: r2-credentials
          key: access-key
    - name: AWS_SECRET_ACCESS_KEY
      valueFrom:
        secretKeyRef:
          name: r2-credentials
          key: secret-key

    # Backend API (from ConfigMap)
    - name: BACKEND_API_URL
      valueFrom:
        configMapKeyRef:
          name: backend-config
          key: api-url

    # MLflow
    - name: MLFLOW_TRACKING_URI
      value: "http://mlflow-service:5000"
```

**Trade-offs**:
- ✅ Zero changes to train.py
- ✅ Secrets managed by K8s (not in code)
- ✅ Easy to update credentials (kubectl edit secret)
- ❌ Slightly verbose manifest (but generated programmatically)

#### 4. Storage Strategy: Keep R2 Integration

**Decision**: Continue using R2 (Cloudflare R2) with Platform SDK

**Rationale**:
- train.py already implements R2 auto-download/upload
- Works identically in containers (boto3 + env vars)
- No need for PersistentVolumes for datasets
- Cost-effective (Cloudflare R2 cheaper than EBS/EFS)

**Flow in Container**:
```python
# train.py:209-227
if is_dataset_id(args.dataset_path):
    # Download from R2 to /workspace/data/.cache/datasets/{dataset_id}/
    local_dataset_path = get_dataset(dataset_id=args.dataset_path)
else:
    local_dataset_path = args.dataset_path

# Later: adapter.train() uploads checkpoints to R2
upload_checkpoint(checkpoint_path, job_id, project_id)
```

**Container Volume**:
```yaml
volumes:
  - name: workspace
    emptyDir: {}  # Ephemeral storage, auto-cleaned after job

volumeMounts:
  - name: workspace
    mountPath: /workspace
```

**Trade-offs**:
- ✅ No changes to train.py or Platform SDK
- ✅ Works in any K8s cluster (no EBS/EFS dependency)
- ✅ Automatic cleanup (emptyDir deleted after pod)
- ✅ Shared cache across jobs (R2 central storage)
- ❌ Network transfer time for datasets (mitigated by R2 cache)

#### 5. Sidecar Pattern: Optional (Not Required)

**Decision**: Sidecar is OPTIONAL, current implementation sufficient

**Rationale**:
- train.py already reports to Backend via TrainingLogger
- Adapters already upload checkpoints via Platform SDK
- stdout/stderr captured by K8s logs (kubectl logs)
- Sidecar only needed for advanced monitoring (GPU metrics, log parsing)

**Current Flow (No Sidecar)**:
```python
# train.py already does:
logger = TrainingLogger(job_id=123, backend_url=BACKEND_API_URL)
logger.update_status("running")
logger.log_metrics(epoch=10, metrics={"loss": 0.234})
logger.update_status("completed")

# Adapter already does:
upload_checkpoint(checkpoint_path, job_id, project_id)
```

**Sidecar Use Cases (Future)**:
- Parse stdout with regex to extract metrics (when logger not available)
- Auto-upload checkpoints without adapter changes
- Collect GPU/memory metrics for cost optimization
- Send heartbeats to Temporal workflows

**Trade-offs**:
- ✅ Simpler pod spec (single container)
- ✅ No inter-container communication needed
- ✅ Faster pod startup
- ❌ Miss advanced monitoring capabilities (acceptable for Phase 2)

## Implementation Plan

### Phase 1: Local Development Setup (Week 1-2)

**Goal**: Get K8s Job working locally with Kind/Minikube

#### Tasks

- [ ] **1.1 Install Local K8s Cluster**
  ```bash
  # Option A: Kind with GPU support
  kind create cluster --config kind-gpu.yaml

  # Option B: Minikube with GPU
  minikube start --driver=docker --gpus=all
  ```

- [ ] **1.2 Build Framework Docker Images**
  ```bash
  # Create Dockerfiles
  mvp/training/docker/
  ├── Dockerfile.ultralytics
  ├── Dockerfile.timm
  └── Dockerfile.base  # Shared base layer

  # Build images
  docker build -f Dockerfile.ultralytics -t trainer-ultralytics:v1.0 .
  docker build -f Dockerfile.timm -t trainer-timm:v1.0 .

  # Load into Kind
  kind load docker-image trainer-ultralytics:v1.0
  ```

- [ ] **1.3 Create K8s Secrets and ConfigMaps**
  ```bash
  # R2 credentials
  kubectl create secret generic r2-credentials \
    --from-literal=endpoint=$AWS_S3_ENDPOINT_URL \
    --from-literal=access-key=$AWS_ACCESS_KEY_ID \
    --from-literal=secret-key=$AWS_SECRET_ACCESS_KEY

  # Backend API URL
  kubectl create configmap backend-config \
    --from-literal=api-url=http://backend-service:8000
  ```

- [ ] **1.4 Create K8s Job Template (Python/Jinja2)**
  ```python
  # mvp/backend/app/templates/training_job.yaml.j2
  apiVersion: batch/v1
  kind: Job
  metadata:
    name: training-job-{{ job_id }}
  spec:
    template:
      spec:
        containers:
        - name: trainer
          image: {{ image }}
          command: ["python", "train.py"]
          args: {{ args | tojson }}
          env: {{ env | tojson }}
  ```

- [ ] **1.5 Test Manual Job Submission**
  ```bash
  # Render manifest
  python render_job_manifest.py --job-id=123 --framework=ultralytics

  # Submit to K8s
  kubectl apply -f training-job-123.yaml

  # Monitor
  kubectl get jobs
  kubectl logs -f training-job-123-xxxxx

  # Check R2 for uploaded checkpoints
  aws s3 ls s3://vision-platform-prod/checkpoints/test-jobs/job_123/
  ```

**Deliverables**:
- ✅ Working local K8s cluster
- ✅ Docker images for ultralytics and timm
- ✅ K8s Secret/ConfigMap setup
- ✅ Job manifest template
- ✅ Manual job execution validated

### Phase 2: Backend Integration (Week 3-4)

**Goal**: Backend API can create and monitor K8s Jobs

#### Tasks

- [ ] **2.1 Implement VMController**
  ```python
  # mvp/backend/app/services/vm_controller.py

  from kubernetes import client, config

  class VMController:
      def __init__(self):
          config.load_incluster_config()  # or load_kube_config() for local
          self.batch_v1 = client.BatchV1Api()
          self.core_v1 = client.CoreV1Api()

      def create_training_job(self, job_config: TrainingJobConfig) -> str:
          """
          Create K8s Job for training

          Returns:
              job_name: K8s Job name (e.g., "training-job-123")
          """
          # 1. Select image based on framework
          image = self._get_image(job_config.framework)

          # 2. Build command args
          args = self._build_args(job_config)

          # 3. Create Job manifest
          job_manifest = self._build_job_manifest(
              job_id=job_config.job_id,
              image=image,
              args=args,
              resources=job_config.resources
          )

          # 4. Submit to K8s
          job = self.batch_v1.create_namespaced_job(
              namespace="training",
              body=job_manifest
          )

          return job.metadata.name

      def get_job_status(self, job_name: str) -> JobStatus:
          """Get K8s Job status"""
          job = self.batch_v1.read_namespaced_job_status(
              name=job_name,
              namespace="training"
          )

          if job.status.succeeded:
              return JobStatus.COMPLETED
          elif job.status.failed:
              return JobStatus.FAILED
          elif job.status.active:
              return JobStatus.RUNNING
          else:
              return JobStatus.PENDING

      def get_job_logs(self, job_name: str) -> str:
          """Get logs from training pod"""
          pods = self.core_v1.list_namespaced_pod(
              namespace="training",
              label_selector=f"job-name={job_name}"
          )

          if not pods.items:
              return ""

          pod_name = pods.items[0].metadata.name
          logs = self.core_v1.read_namespaced_pod_log(
              name=pod_name,
              namespace="training"
          )

          return logs

      def delete_job(self, job_name: str):
          """Delete K8s Job and associated pods"""
          self.batch_v1.delete_namespaced_job(
              name=job_name,
              namespace="training",
              propagation_policy="Foreground"  # Delete pods first
          )
  ```

- [ ] **2.2 Update TrainingManager**
  ```python
  # mvp/backend/app/utils/training_manager.py

  class TrainingManager:
      def __init__(self):
          self.vm_controller = VMController()

      async def start_training(self, job: TrainingJob) -> str:
          """
          Start training job in K8s

          Returns:
              execution_id: K8s Job name
          """
          # Build job config
          job_config = TrainingJobConfig(
              job_id=job.id,
              framework=job.framework,
              model_name=job.model_name,
              task_type=job.task_type,
              dataset_path=job.dataset_id,  # R2 dataset ID
              dataset_format=job.dataset_format,
              num_classes=job.num_classes,
              epochs=job.epochs,
              batch_size=job.batch_size,
              learning_rate=job.learning_rate,
              advanced_config=job.advanced_config,
              project_id=job.project_id,
              resources={
                  "gpu": 1,
                  "memory": "16Gi",
                  "cpu": "4"
              }
          )

          # Create K8s Job
          job_name = self.vm_controller.create_training_job(job_config)

          # Update database
          job.execution_id = job_name
          job.executor_type = "kubernetes"
          job.status = "pending"
          db.commit()

          return job_name
  ```

- [ ] **2.3 Implement Job Status Monitor (Background Task)**
  ```python
  # mvp/backend/app/services/training_monitor.py

  import asyncio

  class TrainingMonitor:
      """Background service to monitor K8s Jobs"""

      async def start(self):
          while True:
              await self.check_active_jobs()
              await asyncio.sleep(10)  # Check every 10s

      async def check_active_jobs(self):
          # Get all active jobs from DB
          jobs = db.query(TrainingJob).filter(
              TrainingJob.status.in_(["pending", "running"]),
              TrainingJob.executor_type == "kubernetes"
          ).all()

          for job in jobs:
              # Query K8s for status
              k8s_status = vm_controller.get_job_status(job.execution_id)

              # Update DB if status changed
              if k8s_status != job.status:
                  job.status = k8s_status
                  db.commit()

                  # Send WebSocket notification
                  await notify_job_status(job.id, k8s_status)
  ```

- [ ] **2.4 Add API Endpoints**
  ```python
  # mvp/backend/app/api/training.py

  @router.post("/training/jobs/{job_id}/start")
  async def start_training_job(job_id: int):
      job = get_job(job_id)
      execution_id = await training_manager.start_training(job)
      return {"execution_id": execution_id}

  @router.get("/training/jobs/{job_id}/logs")
  async def get_training_logs(job_id: int):
      job = get_job(job_id)
      logs = vm_controller.get_job_logs(job.execution_id)
      return {"logs": logs}

  @router.post("/training/jobs/{job_id}/stop")
  async def stop_training_job(job_id: int):
      job = get_job(job_id)
      vm_controller.delete_job(job.execution_id)
      job.status = "stopped"
      db.commit()
      return {"status": "stopped"}
  ```

- [ ] **2.5 Integration Testing**
  ```bash
  # Test full flow
  curl -X POST http://localhost:8000/api/v1/training/jobs/123/start

  # Check K8s
  kubectl get jobs -n training
  kubectl logs -f training-job-123-xxxxx -n training

  # Verify checkpoint uploaded to R2
  aws s3 ls s3://vision-platform-prod/checkpoints/projects/5/jobs/123/
  ```

**Deliverables**:
- ✅ VMController implementation
- ✅ TrainingManager K8s integration
- ✅ Background job monitor
- ✅ API endpoints for job control
- ✅ End-to-end test passing

### Phase 3: Production Deployment (Week 5-6)

**Goal**: Deploy to production K8s cluster (Railway, AWS EKS, or GCP GKE)

#### Tasks

- [ ] **3.1 Setup Production K8s Cluster**
  ```bash
  # Option A: Railway (managed K8s)
  railway up

  # Option B: AWS EKS
  eksctl create cluster --name=vision-platform-training --nodes=2 --node-type=g4dn.xlarge

  # Option C: GCP GKE
  gcloud container clusters create vision-platform-training --num-nodes=2 --machine-type=n1-standard-4 --accelerator=type=nvidia-tesla-t4,count=1
  ```

- [ ] **3.2 Configure GPU Support**
  ```bash
  # Install NVIDIA Device Plugin
  kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml

  # Verify GPU nodes
  kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
  ```

- [ ] **3.3 Deploy Backend to K8s**
  ```yaml
  # backend-deployment.yaml
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: backend-api
  spec:
    replicas: 2
    selector:
      matchLabels:
        app: backend-api
    template:
      spec:
        containers:
        - name: backend
          image: vision-platform/backend:v1.0
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
          ports:
            - containerPort: 8000
  ```

- [ ] **3.4 CI/CD Pipeline**
  ```yaml
  # .github/workflows/deploy.yml
  name: Deploy Training Images

  on:
    push:
      branches: [main]
      paths:
        - 'mvp/training/**'

  jobs:
    build-and-push:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3

        - name: Build Ultralytics Image
          run: |
            docker build -f Dockerfile.ultralytics -t ${{ secrets.REGISTRY }}/trainer-ultralytics:${{ github.sha }} .
            docker push ${{ secrets.REGISTRY }}/trainer-ultralytics:${{ github.sha }}

        - name: Build Timm Image
          run: |
            docker build -f Dockerfile.timm -t ${{ secrets.REGISTRY }}/trainer-timm:${{ github.sha }} .
            docker push ${{ secrets.REGISTRY }}/trainer-timm:${{ github.sha }}

        - name: Update K8s Deployment
          run: |
            kubectl set image deployment/backend-api backend=${{ secrets.REGISTRY }}/backend:${{ github.sha }}
  ```

- [ ] **3.5 Monitoring & Alerting**
  ```bash
  # Install Prometheus + Grafana
  helm install prometheus prometheus-community/kube-prometheus-stack

  # Configure alerts
  kubectl apply -f training-job-alerts.yaml
  ```

**Deliverables**:
- ✅ Production K8s cluster with GPU nodes
- ✅ Backend deployed to K8s
- ✅ CI/CD pipeline for image builds
- ✅ Monitoring and alerting setup
- ✅ Production validation complete

### Phase 4: Advanced Features (Week 7-8, Optional)

**Goal**: Add Temporal orchestration and cloud GPU support

#### Tasks

- [ ] **4.1 Temporal Integration**
  ```python
  # mvp/backend/app/workflows/training_workflow.py

  @workflow.defn
  class TrainingWorkflow:
      @workflow.run
      async def run(self, job_config: TrainingJobConfig) -> WorkflowResult:
          # Step 1: Validate dataset
          dataset_info = await workflow.execute_activity(
              validate_dataset,
              job_config.dataset_id,
              start_to_close_timeout=timedelta(minutes=5)
          )

          # Step 2: Create K8s Job
          job_name = await workflow.execute_activity(
              create_k8s_training_job,
              job_config,
              start_to_close_timeout=timedelta(minutes=10)
          )

          # Step 3: Monitor training (with heartbeat)
          result = await workflow.execute_activity(
              monitor_training_job,
              job_name,
              start_to_close_timeout=timedelta(hours=24),
              heartbeat_timeout=timedelta(minutes=5)
          )

          # Step 4: Cleanup
          await workflow.execute_activity(
              cleanup_k8s_job,
              job_name,
              start_to_close_timeout=timedelta(minutes=5)
          )

          return result
  ```

- [ ] **4.2 Cloud GPU Executors**
  ```python
  # mvp/backend/app/services/executors/aws_executor.py

  class AWSExecutor(ExecutionStrategy):
      """Launch training on AWS ECS/EKS"""

      async def submit_job(self, job_config):
          # Launch ECS task with GPU
          # or
          # Create EKS pod
          pass

  # Job Dispatcher selects executor
  class JobDispatcher:
      async def select_executor(self, job_config):
          if job_config.executor == "local":
              return LocalExecutor()
          elif job_config.executor == "aws":
              return AWSExecutor()
          else:
              # Auto-select based on cost/availability
              return self._auto_select(job_config)
  ```

**Deliverables**:
- ✅ Temporal workflow orchestration
- ✅ Cloud GPU support (AWS ECS/EKS)
- ✅ Cost tracking and optimization

## Technical Details

### Kubernetes Job Manifest (Full Example)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job-123
  namespace: training
  labels:
    app: training-job
    job-id: "123"
    framework: ultralytics
    project-id: "5"
spec:
  # Job completion and cleanup
  ttlSecondsAfterFinished: 86400  # 24 hours
  backoffLimit: 3                  # Max 3 retries
  activeDeadlineSeconds: 86400     # Max 24h runtime

  template:
    metadata:
      labels:
        app: training-job
        job-id: "123"

    spec:
      restartPolicy: Never

      # Container specification
      containers:
      - name: trainer
        image: ghcr.io/myorg/trainer-ultralytics:v1.0
        imagePullPolicy: IfNotPresent

        # Entry point
        command: ["python", "train.py"]

        # Training arguments
        args:
          - "--framework=ultralytics"
          - "--task_type=object_detection"
          - "--model_name=yolo11n"
          - "--dataset_path=abc123"           # R2 dataset ID
          - "--dataset_format=yolo"
          - "--num_classes=80"
          - "--epochs=50"
          - "--batch_size=32"
          - "--learning_rate=0.001"
          - "--optimizer=adam"
          - "--image_size=640"
          - "--pretrained"
          - "--output_dir=/workspace/output"
          - "--job_id=123"
          - "--project_id=5"
          - "--advanced_config={\"optimizer\":{\"type\":\"Adam\",\"weight_decay\":0.0005},\"scheduler\":{\"type\":\"cosine\",\"warmup_epochs\":3},\"augmentation\":{\"enabled\":true,\"mosaic\":1.0,\"mixup\":0.0}}"

        # Environment variables
        env:
          # R2 Storage (from Secret)
          - name: AWS_S3_ENDPOINT_URL
            valueFrom:
              secretKeyRef:
                name: r2-credentials
                key: endpoint
          - name: AWS_ACCESS_KEY_ID
            valueFrom:
              secretKeyRef:
                name: r2-credentials
                key: access-key
          - name: AWS_SECRET_ACCESS_KEY
            valueFrom:
              secretKeyRef:
                name: r2-credentials
                key: secret-key

          # Backend API (from ConfigMap)
          - name: BACKEND_API_URL
            valueFrom:
              configMapKeyRef:
                name: backend-config
                key: api-url

          # MLflow
          - name: MLFLOW_TRACKING_URI
            value: "http://mlflow-service.mlflow.svc.cluster.local:5000"

          # CUDA
          - name: CUDA_VISIBLE_DEVICES
            value: "0"

        # Resource limits
        resources:
          limits:
            nvidia.com/gpu: 1      # 1 GPU
            memory: "16Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"

        # Volume mounts
        volumeMounts:
          - name: workspace
            mountPath: /workspace
          - name: dshm                # Shared memory for DataLoader
            mountPath: /dev/shm

      # GPU node selection
      nodeSelector:
        accelerator: nvidia-gpu
        gpu-type: t4               # Optional: specific GPU type

      # Tolerations (if using tainted nodes)
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

      # Volumes
      volumes:
        - name: workspace
          emptyDir: {}              # Ephemeral storage
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 2Gi
```

### Dockerfile Templates

**Dockerfile.base** (Shared base layer):
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install common dependencies
RUN pip install \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 \
    boto3==1.34.0 \
    mlflow==2.9.0 \
    numpy==1.24.3 \
    pillow==10.1.0 \
    pyyaml==6.0.1 \
    requests==2.31.0 \
    python-dotenv==1.0.0

# Create workspace
RUN mkdir -p /workspace/data/.cache /workspace/output
WORKDIR /app

# Copy platform SDK
COPY mvp/training/platform_sdk /app/platform_sdk

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
```

**Dockerfile.ultralytics**:
```dockerfile
FROM vision-platform/trainer-base:latest

# Install Ultralytics
RUN pip install \
    ultralytics==8.3.0 \
    opencv-python-headless==4.8.1.78

# Copy training code
COPY mvp/training/train.py /app/train.py
COPY mvp/training/adapters /app/adapters

# Entry point
ENTRYPOINT ["python", "train.py"]
```

**Dockerfile.timm**:
```dockerfile
FROM vision-platform/trainer-base:latest

# Install timm and augmentation libraries
RUN pip install \
    timm==1.0.0 \
    albumentations==1.3.1 \
    opencv-python-headless==4.8.1.78

# Copy training code
COPY mvp/training/train.py /app/train.py
COPY mvp/training/adapters /app/adapters

# Entry point
ENTRYPOINT ["python", "train.py"]
```

### VMController Implementation

```python
# mvp/backend/app/services/vm_controller.py

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from typing import Dict, Any, Optional
import os
import json

class VMController:
    """Kubernetes Job controller for training workloads"""

    def __init__(self):
        """Initialize K8s client"""
        try:
            # Try in-cluster config first (production)
            config.load_incluster_config()
        except:
            # Fallback to local kubeconfig (development)
            config.load_kube_config()

        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()
        self.namespace = os.getenv("K8S_TRAINING_NAMESPACE", "training")

    def create_training_job(self, job_config: TrainingJobConfig) -> str:
        """
        Create Kubernetes Job for training

        Args:
            job_config: Training job configuration

        Returns:
            job_name: Kubernetes Job name (e.g., "training-job-123")

        Raises:
            ApiException: If job creation fails
        """
        job_name = f"training-job-{job_config.job_id}"

        # Select Docker image based on framework
        image = self._get_image(job_config.framework)

        # Build command arguments for train.py
        args = self._build_training_args(job_config)

        # Build Job manifest
        job_manifest = self._build_job_manifest(
            job_name=job_name,
            image=image,
            args=args,
            resources=job_config.resources,
            labels={
                "app": "training-job",
                "job-id": str(job_config.job_id),
                "framework": job_config.framework,
                "project-id": str(job_config.project_id) if job_config.project_id else "none",
            }
        )

        # Create Job in Kubernetes
        try:
            job = self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job_manifest
            )
            print(f"[K8S] Created Job: {job_name}")
            return job_name

        except ApiException as e:
            print(f"[K8S ERROR] Failed to create Job: {e}")
            raise

    def get_job_status(self, job_name: str) -> str:
        """
        Get Kubernetes Job status

        Returns:
            status: "pending" | "running" | "completed" | "failed"
        """
        try:
            job = self.batch_v1.read_namespaced_job_status(
                name=job_name,
                namespace=self.namespace
            )

            if job.status.succeeded:
                return "completed"
            elif job.status.failed:
                return "failed"
            elif job.status.active:
                return "running"
            else:
                return "pending"

        except ApiException as e:
            if e.status == 404:
                return "not_found"
            raise

    def get_job_logs(self, job_name: str, tail_lines: int = 100) -> str:
        """
        Get logs from training pod

        Args:
            job_name: Kubernetes Job name
            tail_lines: Number of lines to return from end of log

        Returns:
            logs: Pod logs (stdout/stderr)
        """
        try:
            # Find pod for this job
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}"
            )

            if not pods.items:
                return ""

            pod_name = pods.items[0].metadata.name

            # Get logs
            logs = self.core_v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                tail_lines=tail_lines
            )

            return logs

        except ApiException as e:
            print(f"[K8S ERROR] Failed to get logs: {e}")
            return ""

    def delete_job(self, job_name: str):
        """
        Delete Kubernetes Job and associated pods

        Args:
            job_name: Kubernetes Job name
        """
        try:
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Foreground"  # Delete pods first
            )
            print(f"[K8S] Deleted Job: {job_name}")

        except ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                print(f"[K8S ERROR] Failed to delete Job: {e}")
                raise

    # Helper methods

    def _get_image(self, framework: str) -> str:
        """Get Docker image for framework"""
        registry = os.getenv("DOCKER_REGISTRY", "ghcr.io/myorg")
        version = os.getenv("TRAINER_IMAGE_VERSION", "v1.0")

        images = {
            "ultralytics": f"{registry}/trainer-ultralytics:{version}",
            "timm": f"{registry}/trainer-timm:{version}",
            "transformers": f"{registry}/trainer-transformers:{version}",
        }

        return images.get(framework, images["ultralytics"])

    def _build_training_args(self, job_config: TrainingJobConfig) -> list:
        """Build command-line arguments for train.py"""
        args = [
            f"--framework={job_config.framework}",
            f"--task_type={job_config.task_type}",
            f"--model_name={job_config.model_name}",
            f"--dataset_path={job_config.dataset_path}",
            f"--dataset_format={job_config.dataset_format}",
            f"--epochs={job_config.epochs}",
            f"--batch_size={job_config.batch_size}",
            f"--learning_rate={job_config.learning_rate}",
            f"--optimizer={job_config.optimizer}",
            f"--output_dir=/workspace/output",
            f"--job_id={job_config.job_id}",
        ]

        # Optional args
        if job_config.num_classes:
            args.append(f"--num_classes={job_config.num_classes}")

        if job_config.project_id:
            args.append(f"--project_id={job_config.project_id}")

        if job_config.image_size:
            args.append(f"--image_size={job_config.image_size}")

        if job_config.pretrained:
            args.append("--pretrained")

        if job_config.advanced_config:
            # Serialize advanced_config as JSON string
            args.append(f"--advanced_config={json.dumps(job_config.advanced_config)}")

        return args

    def _build_job_manifest(
        self,
        job_name: str,
        image: str,
        args: list,
        resources: Dict[str, Any],
        labels: Dict[str, str]
    ) -> client.V1Job:
        """Build Kubernetes Job manifest"""

        # Container spec
        container = client.V1Container(
            name="trainer",
            image=image,
            image_pull_policy="IfNotPresent",
            command=["python", "train.py"],
            args=args,
            env=[
                # R2 Credentials (from Secret)
                client.V1EnvVar(
                    name="AWS_S3_ENDPOINT_URL",
                    value_from=client.V1EnvVarSource(
                        secret_key_ref=client.V1SecretKeySelector(
                            name="r2-credentials",
                            key="endpoint"
                        )
                    )
                ),
                client.V1EnvVar(
                    name="AWS_ACCESS_KEY_ID",
                    value_from=client.V1EnvVarSource(
                        secret_key_ref=client.V1SecretKeySelector(
                            name="r2-credentials",
                            key="access-key"
                        )
                    )
                ),
                client.V1EnvVar(
                    name="AWS_SECRET_ACCESS_KEY",
                    value_from=client.V1EnvVarSource(
                        secret_key_ref=client.V1SecretKeySelector(
                            name="r2-credentials",
                            key="secret-key"
                        )
                    )
                ),
                # Backend API (from ConfigMap)
                client.V1EnvVar(
                    name="BACKEND_API_URL",
                    value_from=client.V1EnvVarSource(
                        config_map_key_ref=client.V1ConfigMapKeySelector(
                            name="backend-config",
                            key="api-url"
                        )
                    )
                ),
                # MLflow
                client.V1EnvVar(
                    name="MLFLOW_TRACKING_URI",
                    value="http://mlflow-service:5000"
                ),
            ],
            resources=client.V1ResourceRequirements(
                limits={
                    "nvidia.com/gpu": str(resources.get("gpu", 1)),
                    "memory": resources.get("memory", "16Gi"),
                    "cpu": str(resources.get("cpu", 4)),
                },
                requests={
                    "nvidia.com/gpu": str(resources.get("gpu", 1)),
                    "memory": resources.get("memory_request", "8Gi"),
                    "cpu": str(resources.get("cpu_request", 2)),
                }
            ),
            volume_mounts=[
                client.V1VolumeMount(
                    name="workspace",
                    mount_path="/workspace"
                ),
                client.V1VolumeMount(
                    name="dshm",
                    mount_path="/dev/shm"
                ),
            ]
        )

        # Pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[container],
                node_selector={
                    "accelerator": "nvidia-gpu"
                },
                volumes=[
                    client.V1Volume(
                        name="workspace",
                        empty_dir=client.V1EmptyDirVolumeSource()
                    ),
                    client.V1Volume(
                        name="dshm",
                        empty_dir=client.V1EmptyDirVolumeSource(
                            medium="Memory",
                            size_limit="2Gi"
                        )
                    ),
                ]
            )
        )

        # Job spec
        job_spec = client.V1JobSpec(
            ttl_seconds_after_finished=86400,  # 24 hours
            backoff_limit=3,
            active_deadline_seconds=86400,  # Max 24h runtime
            template=pod_template
        )

        # Job manifest
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels=labels
            ),
            spec=job_spec
        )

        return job
```

### TrainingJobConfig Schema

```python
# mvp/backend/app/schemas/training.py

from pydantic import BaseModel
from typing import Optional, Dict, Any

class TrainingJobConfig(BaseModel):
    """Configuration for creating K8s training job"""

    # Job metadata
    job_id: int
    project_id: Optional[int] = None

    # Framework and model
    framework: str  # "timm", "ultralytics", "transformers"
    task_type: str  # "image_classification", "object_detection", etc.
    model_name: str  # "resnet18", "yolo11n", etc.

    # Dataset
    dataset_path: str  # Dataset ID (for R2) or local path
    dataset_format: str  # "imagefolder", "yolo", "coco", etc.
    num_classes: Optional[int] = None

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    image_size: Optional[int] = None
    pretrained: bool = True

    # Advanced configuration
    advanced_config: Optional[Dict[str, Any]] = None

    # Resource allocation
    resources: Dict[str, Any] = {
        "gpu": 1,
        "memory": "16Gi",
        "cpu": 4,
        "memory_request": "8Gi",
        "cpu_request": 2,
    }

    # Execution options
    executor: Optional[str] = None  # "kubernetes", "aws", "gcp", None (auto-select)
```

## Alternatives Considered

### Alternative 1: Continue with Subprocess Execution

**Pros**:
- ✅ Already implemented and working
- ✅ Simple architecture
- ✅ No K8s infrastructure needed
- ✅ Faster development iteration

**Cons**:
- ❌ No resource isolation
- ❌ No automatic recovery
- ❌ Limited scalability
- ❌ GPU idle time waste

**Why rejected**: Doesn't meet production scalability and reliability requirements.

### Alternative 2: Use Kubernetes Deployment instead of Job

**Pros**:
- ✅ Built-in liveness/readiness probes
- ✅ Rolling updates
- ✅ Service discovery

**Cons**:
- ❌ Training is not a long-running service
- ❌ Need manual cleanup after completion
- ❌ More complex lifecycle management
- ❌ Doesn't fit batch workload pattern

**Why rejected**: Kubernetes Job is designed for batch workloads like training.

### Alternative 3: Single Monolithic Docker Image (All Frameworks)

**Pros**:
- ✅ Single image to manage
- ✅ Simpler CI/CD pipeline
- ✅ No framework selection logic needed

**Cons**:
- ❌ Large image size (~10+ GB)
- ❌ Slower pull times
- ❌ Dependency conflicts between frameworks
- ❌ Wastes storage (unused packages)

**Why rejected**: Framework-specific images provide better isolation and efficiency.

### Alternative 4: Use PersistentVolumes for Dataset Storage

**Pros**:
- ✅ Faster dataset access (no download)
- ✅ Shared across pods

**Cons**:
- ❌ Requires cluster-specific storage (EBS, EFS, GCE PD)
- ❌ Manual dataset management
- ❌ Doesn't work across cloud providers
- ❌ More expensive than object storage

**Why rejected**: R2 provides cloud-agnostic, cost-effective storage with automatic caching.

### Alternative 5: Implement Sidecar Pattern from Start

**Pros**:
- ✅ Advanced monitoring capabilities
- ✅ Automatic metric extraction from logs
- ✅ GPU/memory metrics

**Cons**:
- ❌ More complex pod spec
- ❌ Inter-container communication overhead
- ❌ TrainingLogger already provides needed functionality

**Why rejected**: Current implementation (TrainingLogger + Platform SDK) is sufficient for Phase 2. Sidecar can be added later if needed.

## Migration Path

### Migration Strategy: Incremental Rollout

**Phase 1: Parallel Execution**
- Keep subprocess-based execution working
- Add K8s execution as opt-in (`executor="kubernetes"`)
- Both modes coexist

```python
# Backend supports both
if job.executor == "kubernetes":
    vm_controller.create_training_job(job_config)
else:
    training_client.start_training(job_config)  # Subprocess
```

**Phase 2: Gradual Migration**
- Test all frameworks on K8s
- Migrate production jobs incrementally
- Monitor for issues

**Phase 3: Full Migration**
- Make K8s default executor
- Deprecate subprocess mode
- Remove Training Service API servers

### Rollback Plan

If K8s migration encounters critical issues:

1. **Immediate Rollback**:
   ```python
   # Change default executor to subprocess
   DEFAULT_EXECUTOR = "subprocess"  # Was "kubernetes"
   ```

2. **Job-level Rollback**:
   ```python
   # Retry failed K8s job with subprocess
   if job.status == "failed" and job.executor == "kubernetes":
       job.executor = "subprocess"
       training_client.start_training(job_config)
   ```

3. **Zero Downtime**:
   - Subprocess mode remains available during entire migration
   - Can switch back without code deployment

## References

### Related Files

**Backend**:
- `mvp/backend/app/utils/training_manager.py` - Training orchestration (needs update)
- `mvp/backend/app/utils/training_client.py` - HTTP client (will be replaced)
- `mvp/backend/app/api/training.py` - Training API endpoints
- `mvp/backend/app/db/models.py` - TrainingJob model

**Training**:
- `mvp/training/train.py` - Main training script (NO CHANGES NEEDED)
- `mvp/training/adapters/` - Framework adapters (NO CHANGES NEEDED)
- `mvp/training/platform_sdk/storage.py` - R2 integration (NO CHANGES NEEDED)
- `mvp/training/api_server.py` - Training Service API (will be deprecated)

**Infrastructure**:
- `docker-compose.yml` - Local development services
- `mvp/docker/` - Docker configurations

### Related Documentation

- [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) - Overall system architecture
- [CLOUD_GPU_ARCHITECTURE.md](../architecture/CLOUD_GPU_ARCHITECTURE.md) - Cloud GPU execution design
- [ADAPTER_DESIGN.md](../architecture/ADAPTER_DESIGN.md) - Adapter pattern design

### External Resources

- [Kubernetes Jobs Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Kubernetes Python Client](https://github.com/kubernetes-client/python)
- [NVIDIA Device Plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin)
- [Kind - Local Kubernetes Clusters](https://kind.sigs.k8s.io/)
- [GPU Operator for Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)

## Notes

### Open Questions

1. **Q: Should we implement Temporal orchestration in Phase 2 or Phase 4?**
   - **A**: Phase 4 (optional). K8s Job provides sufficient orchestration for now. Temporal adds value for complex workflows (multi-stage training, hyperparameter tuning).

2. **Q: How to handle very large datasets (>100GB)?**
   - **A**: R2's download strategy works, but consider:
     - PersistentVolume for frequently used datasets
     - Dataset streaming (progressive download during training)
     - Dataset sharding for distributed training

3. **Q: Should we use Spot Instances for cost savings?**
   - **A**: Yes, for non-critical jobs. Add to Phase 4. Requires:
     - Checkpoint resume capability (already implemented)
     - Job retry logic (already implemented with backoffLimit)
     - Cost tracking

4. **Q: How to handle framework version upgrades?**
   - **A**: Use image tags with version numbers:
     - `trainer-ultralytics:v1.0` (ultralytics 8.3.0)
     - `trainer-ultralytics:v1.1` (ultralytics 8.4.0)
     - Pin versions in job creation, allow user to select

### Future Considerations

1. **Multi-GPU Training**: Add support for distributed training
   ```yaml
   resources:
     limits:
       nvidia.com/gpu: 4  # 4 GPUs
   ```

2. **Auto-scaling**: K8s Cluster Autoscaler for dynamic node provisioning

3. **Job Priority Classes**: Prioritize production jobs over test jobs
   ```yaml
   priorityClassName: high-priority
   ```

4. **Pod Preemption**: Allow critical jobs to preempt lower-priority jobs

5. **Training Queuing System**: If all GPUs busy, queue jobs and run when available

### Success Metrics

**Technical Metrics**:
- ✅ Job success rate > 95%
- ✅ Average job start time < 2 minutes (image pull + pod scheduling)
- ✅ Resource utilization > 80% (GPU not idle)
- ✅ No data loss (all checkpoints uploaded to R2)

**Operational Metrics**:
- ✅ Zero downtime migration
- ✅ Rollback capability maintained throughout migration
- ✅ Monitoring and alerting operational

**Cost Metrics**:
- ✅ GPU cost per training job tracked
- ✅ Storage cost optimization (R2 vs PV)
- ✅ Spot instance savings measured (Phase 4)

---

**Document Status**: Proposed
**Next Review**: 2025-11-13 (1 week)
**Owner**: Development Team
**Approvers**: TBD
