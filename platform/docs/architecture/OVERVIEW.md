# Architecture Overview

Complete system architecture for the production-ready Vision AI Training Platform.

## Table of Contents

- [System Architecture](#system-architecture)
- [Component Responsibilities](#component-responsibilities)
- [Data Flow](#data-flow)
- [Communication Patterns](#communication-patterns)
- [Technology Stack](#technology-stack)
- [Deployment Architecture](#deployment-architecture)

## System Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Interface                             │
│                    (Next.js Frontend + WebSocket)                    │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ HTTP/WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Backend Service (FastAPI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐         │
│  │ Chat API     │  │ Training API │  │ Dataset API       │         │
│  │ (LLM Parser) │  │              │  │                   │         │
│  └──────────────┘  └──────────────┘  └───────────────────┘         │
│                             │                                        │
│                             ▼                                        │
│                  ┌──────────────────────┐                           │
│                  │  Temporal Client     │                           │
│                  └──────────────────────┘                           │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ Start Workflow
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Temporal Server (Orchestrator)                    │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              TrainingWorkflow                         │           │
│  │  ┌────────┐  ┌──────────┐  ┌─────────┐  ┌────────┐ │           │
│  │  │Validate│→ │Create Job│→ │Monitor  │→ │Cleanup │ │           │
│  │  │Dataset │  │(K8s)     │  │Training │  │        │ │           │
│  │  └────────┘  └──────────┘  └─────────┘  └────────┘ │           │
│  └──────────────────────────────────────────────────────┘           │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ Create K8s Job
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                                │
│  ┌───────────────────────────────────────────────────────┐          │
│  │              Training Job Pod                          │          │
│  │  ┌─────────────────────┐  ┌──────────────────────┐   │          │
│  │  │  Trainer Container  │  │  (No Sidecar)        │   │          │
│  │  │                     │  │                      │   │          │
│  │  │  - train.py         │  │  Trainer handles:    │   │          │
│  │  │  - Load from S3     │  │  - Progress updates  │   │          │
│  │  │  - Train model      │  │  - Checkpoint upload │   │          │
│  │  │  - Upload to S3     │  │  - Callback to API   │   │          │
│  │  │  - HTTP callbacks   │  │                      │   │          │
│  │  └─────────────────────┘  └──────────────────────┘   │          │
│  └───────────────────────────────────────────────────────┘          │
└─────────────────────────────┬─────────────────────────────────────────┘
                             │ Heartbeat & Events
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Layer                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │PostgreSQL│  │  Redis   │  │ S3/R2    │  │ MLflow   │           │
│  │(Metadata)│  │  (Cache) │  │(Storage) │  │(Tracking)│           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Flow

1. **User → Frontend**: User configures training via chat interface
2. **Frontend → Backend**: Sends natural language or structured request
3. **Backend → Temporal**: Creates training workflow
4. **Temporal → K8s**: Provisions training pod with environment variables
5. **Trainer → Backend**: Sends heartbeats, events, and completion via HTTP callbacks
6. **Backend → Frontend**: Broadcasts real-time updates via WebSocket
7. **Trainer → S3**: Downloads dataset, uploads checkpoints

## Component Responsibilities

### Frontend (Next.js)
**Location**: `platform/frontend/`

**Responsibilities**:
- Chat-based training configuration UI
- Real-time training monitoring (WebSocket)
- Dataset upload and management
- Model selection and configuration
- Training history and results visualization

**Key Features**:
- Server-side rendering for SEO
- Real-time WebSocket updates
- JWT authentication
- Responsive design

**Technology**: Next.js 14, React 18, TypeScript, TailwindCSS, Zustand

**See**: [FRONTEND_DESIGN.md](./FRONTEND_DESIGN.md)

### Backend Service (FastAPI)
**Location**: `platform/backend/`

**Responsibilities**:
- REST API endpoints for all operations
- LLM-based intent parsing (chat)
- Training workflow management (Temporal client)
- Dataset validation and metadata storage
- Real-time WebSocket server
- Authentication & authorization (JWT)
- OpenTelemetry tracing

**Key Features**:
- Async/await throughout
- SQLAlchemy ORM with PostgreSQL
- Redis for caching and pub/sub
- S3-compatible storage abstraction
- Temporal workflow triggers

**Technology**: FastAPI, SQLAlchemy, PostgreSQL, Redis, Temporal Python SDK

**See**: [BACKEND_DESIGN.md](./BACKEND_DESIGN.md)

### Training Services (Trainers)
**Location**: `platform/trainers/`

**Responsibilities**:
- Execute actual model training
- Download datasets from S3
- Upload checkpoints to S3
- Send progress updates via HTTP callbacks
- Report errors and completion status

**Key Features**:
- Framework-specific implementations (ultralytics, timm, huggingface)
- Stateless - all state via environment variables
- HTTP-only communication with backend
- No shared file system
- No direct imports from backend

**Technology**: PyTorch, framework-specific libraries, boto3

**See**: [TRAINER_DESIGN.md](./TRAINER_DESIGN.md)

### Temporal Workflows
**Location**: `platform/workflows/`

**Responsibilities**:
- Orchestrate training lifecycle
- Handle retries and timeouts
- Coordinate activities (validate, create, monitor, cleanup)
- Maintain execution history
- Support cancellation and error recovery

**Key Features**:
- Automatic retries with exponential backoff
- Activity timeouts and heartbeats
- Workflow versioning
- Durable execution (survives server restarts)

**Technology**: Temporal Python SDK, AsyncIO

**See**: [WORKFLOWS_DESIGN.md](./WORKFLOWS_DESIGN.md)

### Infrastructure (Kubernetes + Helm)
**Location**: `platform/infrastructure/`

**Responsibilities**:
- Kubernetes manifests for all services
- Helm charts for deployment
- Environment-specific configurations
- Resource quotas and limits
- Network policies

**Key Features**:
- Cloud-agnostic (works on Railway, AWS, on-premise)
- Namespace isolation per model/user
- Job templates for trainers
- Auto-scaling configurations

**Technology**: Kubernetes, Helm, Terraform

**See**: [INFRASTRUCTURE_DESIGN.md](./INFRASTRUCTURE_DESIGN.md)

### Observability Stack
**Location**: `platform/observability/`

**Responsibilities**:
- Metrics collection (Prometheus)
- Log aggregation (Loki)
- Distributed tracing (OpenTelemetry)
- Dashboards (Grafana)
- Alerts and notifications

**Key Features**:
- Trace ID propagation across services
- Training-specific dashboards
- Error rate alerts
- Resource usage monitoring

**Technology**: Prometheus, Loki, Grafana, OpenTelemetry

**See**: [OBSERVABILITY_DESIGN.md](./OBSERVABILITY_DESIGN.md)

## Data Flow

### Training Workflow Data Flow

```
1. User Input (Chat)
   ↓
2. LLM Intent Parsing
   ↓
3. TrainingConfig Creation
   ↓
4. Database: TrainingJob Record (status=pending)
   ↓
5. Temporal: Start TrainingWorkflow
   ↓
6. Activity: validate_dataset
   - Check dataset exists in S3
   - Validate format
   ↓
7. Activity: create_training_job
   - Create K8s Job manifest
   - Set environment variables:
     * JOB_ID
     * TRACE_ID
     * BACKEND_BASE_URL
     * CALLBACK_TOKEN
     * TASK_TYPE
     * MODEL_NAME
     * DATASET_ID
     * STORAGE_TYPE
     * R2_* credentials
   - Apply to K8s cluster
   ↓
8. K8s: Schedule Pod
   ↓
9. Trainer Container Starts
   - Reads environment variables
   - Downloads dataset from S3
   - Starts training loop
   ↓
10. Training Loop (per epoch)
    - Train epoch
    - Calculate metrics
    - HTTP POST {BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/heartbeat
      * Updates TrainingJob.current_epoch, metrics
      * Temporal activity heartbeat
    - Save checkpoint
    - Upload checkpoint to S3
    ↓
11. Training Completion
    - HTTP POST {BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/done
      * status: succeeded/failed
      * final_metrics
    - Update database: TrainingJob.status = completed
    - Temporal activity returns result
    ↓
12. Activity: cleanup_training_job
    - Delete K8s Job
    - Optional: cleanup temp files
    ↓
13. Workflow Completes
    - Return final result
    - Close WebSocket connections
```

### Real-Time Updates Flow

```
Trainer                Backend              Frontend
  │                       │                    │
  ├─ POST /heartbeat ────>│                    │
  │                       ├─ Update DB         │
  │                       ├─ Redis PUBLISH ───>│
  │                       │                    ├─ WebSocket message
  │                       │                    ├─ Update UI (progress bar)
  │                       │                    │
  ├─ POST /event ───────>│                    │
  │  (checkpoint_saved)   ├─ Redis PUBLISH ───>│
  │                       │                    ├─ Show notification
  │                       │                    │
  ├─ POST /done ────────>│                    │
  │  (status=succeeded)   ├─ Update DB         │
  │                       ├─ Redis PUBLISH ───>│
  │                       │                    ├─ Show completion message
  │                       │                    ├─ Navigate to results
```

## Communication Patterns

### 1. Frontend ↔ Backend

**HTTP REST API** (Request/Response)
- Authentication: JWT Bearer token
- Content-Type: application/json
- Endpoints: `/api/v1/*`

**WebSocket** (Real-time updates)
- Connection: `ws://backend/ws/training/{job_id}?token={jwt}`
- Messages: JSON format
- Auto-reconnect on disconnect

Example:
```typescript
// Frontend
const response = await fetch('/api/v1/training/jobs', {
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(config)
});

const ws = new WebSocket(`ws://backend/ws/training/${jobId}?token=${token}`);
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Update UI with training progress
};
```

### 2. Backend ↔ Temporal

**Temporal Client** (Python SDK)
- Workflow execution: `await client.start_workflow(TrainingWorkflow, config)`
- Workflow queries: `await handle.query(GetStatusQuery)`
- Workflow signals: `await handle.signal(CancelSignal)`

Example:
```python
# Backend
handle = await temporal_client.start_workflow(
    TrainingWorkflow.run,
    training_config,
    id=f"training-{job_id}",
    task_queue="training-tasks"
)

# Query status
status = await handle.query(GetStatusQuery)
```

### 3. Temporal ↔ Kubernetes

**Kubernetes Python Client**
- Create Job: `await k8s_client.create_namespaced_job(namespace, job_manifest)`
- Watch Job: `await k8s_client.read_namespaced_job_status(name, namespace)`
- Delete Job: `await k8s_client.delete_namespaced_job(name, namespace)`

Example:
```python
# Temporal Activity
@activity.defn
async def create_training_job(config: TrainingConfig) -> str:
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "trainer",
                        "image": f"trainer-{config.framework}:latest",
                        "env": [
                            {"name": "JOB_ID", "value": str(config.job_id)},
                            {"name": "CALLBACK_URL", "value": callback_url},
                            # ... more env vars
                        ]
                    }]
                }
            }
        }
    }
    await k8s_client.create_namespaced_job("training", job_manifest)
    return f"training-job-{config.job_id}"
```

### 4. Trainer ↔ Backend

**HTTP Callbacks** (Trainer → Backend)
- Authentication: CALLBACK_TOKEN in environment variable
- Content-Type: application/json
- All requests include trace_id for distributed tracing

Example:
```python
# Trainer
import requests
import os

JOB_ID = os.environ["JOB_ID"]
BACKEND_BASE_URL = os.environ["BACKEND_BASE_URL"]
CALLBACK_TOKEN = os.environ["CALLBACK_TOKEN"]
TRACE_ID = os.environ["TRACE_ID"]

# Heartbeat
requests.post(
    f"{BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/heartbeat",
    headers={
        "Authorization": f"Bearer {CALLBACK_TOKEN}",
        "X-Trace-ID": TRACE_ID
    },
    json={
        "epoch": current_epoch,
        "progress": progress_percent,
        "metrics": {"loss": loss_value, "accuracy": acc_value}
    }
)

# Completion
requests.post(
    f"{BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/done",
    headers={
        "Authorization": f"Bearer {CALLBACK_TOKEN}",
        "X-Trace-ID": TRACE_ID
    },
    json={
        "status": "succeeded",
        "final_metrics": {...}
    }
)
```

### 5. All Services ↔ S3 Storage

**S3 API** (boto3)
- Abstraction layer supports: MinIO (local), R2 (Railway), S3 (AWS)
- Same code works across all environments

Example:
```python
# Backend / Trainer
import boto3

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.environ.get("R2_ENDPOINT"),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"]
    )

s3 = get_s3_client()

# Download dataset
s3.download_file(
    "datasets",
    f"{dataset_id}/data.zip",
    "/tmp/dataset.zip"
)

# Upload checkpoint
s3.upload_file(
    "/workspace/checkpoints/best.pt",
    "checkpoints",
    f"job-{job_id}/best.pt"
)
```

## Technology Stack

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript 5.3
- **UI**: TailwindCSS, shadcn/ui
- **State**: Zustand
- **API Client**: React Query (TanStack Query)
- **WebSocket**: native WebSocket API
- **Forms**: React Hook Form + Zod
- **Testing**: Jest, React Testing Library, Playwright

### Backend
- **Framework**: FastAPI 0.109
- **Language**: Python 3.11
- **ORM**: SQLAlchemy 2.0
- **Validation**: Pydantic 2.5
- **Auth**: python-jose (JWT)
- **WebSocket**: FastAPI WebSocket
- **LLM**: LangChain + Anthropic/OpenAI
- **Testing**: pytest, pytest-asyncio

### Orchestration
- **Workflow Engine**: Temporal 1.22
- **Language**: Python (Temporal SDK)
- **Worker**: AsyncIO-based

### Infrastructure
- **Container Orchestration**: Kubernetes 1.28+
- **Package Manager**: Helm 3.12
- **IaC**: Terraform 1.6
- **Container Runtime**: containerd

### Data Layer
- **Database**: PostgreSQL 16
- **Cache**: Redis 7.2
- **Object Storage**: S3 API (MinIO/R2/S3)
- **ML Tracking**: MLflow 2.9

### Observability
- **Metrics**: Prometheus 2.48
- **Logs**: Loki 2.9
- **Traces**: OpenTelemetry 1.21
- **Dashboards**: Grafana 10.2

### Training
- **Deep Learning**: PyTorch 2.1
- **Frameworks**:
  - Ultralytics 8.1 (YOLO)
  - timm 0.9 (Image Models)
  - HuggingFace Transformers 4.36

## Deployment Architecture

### Environment Tiers

**Tier 1: Local Development (Subprocess)**
```
Developer Machine
├── Frontend (localhost:3000)
├── Backend (localhost:8000)
│   └── Spawns subprocess for training
├── PostgreSQL (Docker, localhost:5432)
├── Redis (Docker, localhost:6379)
└── MinIO (Docker, localhost:9000)
```

**Tier 2: Local Kubernetes (Kind)**
```
Developer Machine
├── Kind Cluster
│   ├── frontend Pod
│   ├── backend Pod
│   ├── temporal Pod
│   ├── postgres Pod
│   ├── redis Pod
│   └── minio Pod
└── Training Jobs (batch/v1/Job)
    └── trainer Pod (created dynamically)
```

**Tier 3: Production (K8s Cluster)**
```
Cloud Provider (Railway/AWS/On-Premise)
├── Kubernetes Cluster
│   ├── Namespace: platform
│   │   ├── frontend Deployment
│   │   ├── backend Deployment
│   │   ├── temporal Deployment
│   │   └── redis Deployment
│   ├── Namespace: training-{model-id}
│   │   └── Training Jobs (isolated per model)
│   └── Namespace: observability
│       ├── prometheus
│       ├── loki
│       └── grafana
├── Managed PostgreSQL
└── S3/R2 Storage
```

### Cross-Tier Consistency

**Same Code, Different Configuration**:

```python
# backend/app/config.py
class Settings(BaseSettings):
    # Changes per tier
    EXECUTION_MODE: str = "subprocess"  # subprocess | kubernetes
    STORAGE_TYPE: str = "local"         # local | minio | r2 | s3
    DATABASE_URL: str

    # Same across tiers
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"

# Tier 1: .env
EXECUTION_MODE=subprocess
STORAGE_TYPE=local
DATABASE_URL=postgresql://localhost:5432/platform

# Tier 2: Kind ConfigMap
EXECUTION_MODE=kubernetes
STORAGE_TYPE=minio
DATABASE_URL=postgresql://postgres:5432/platform

# Tier 3: K8s Secret
EXECUTION_MODE=kubernetes
STORAGE_TYPE=r2
DATABASE_URL=postgresql://db.example.com:5432/platform
```

**See**: [3_TIER_DEVELOPMENT.md](../development/3_TIER_DEVELOPMENT.md)

## Key Design Decisions

### 1. No Sidecar Pattern
**Decision**: Trainer containers handle all responsibilities (progress, checkpoints, callbacks) without a sidecar.

**Rationale**:
- Simpler deployment (single container)
- Less resource overhead
- Easier debugging
- Framework code can natively handle callbacks

**Tradeoff**: Trainer must implement callback logic, but this is minimal overhead.

### 2. API Contract Over Shared Code
**Decision**: Backend and Trainers communicate only via HTTP APIs and environment variables.

**Rationale**:
- Complete dependency isolation
- Can deploy trainers independently
- Supports multiple languages
- Clear API boundaries

**Tradeoff**: Cannot share Python code, but isolation benefits outweigh this.

**See**: [ISOLATION_DESIGN.md](./ISOLATION_DESIGN.md)

### 3. Temporal for Orchestration
**Decision**: Use Temporal workflows instead of custom state machines or Celery.

**Rationale**:
- Built-in retries, timeouts, error handling
- Durable execution (survives crashes)
- Excellent visibility and debugging
- Versioning support

**Tradeoff**: Additional infrastructure component, but benefits are substantial.

### 4. Callback Pattern for Progress
**Decision**: Trainers push updates to Backend via HTTP callbacks instead of Backend polling.

**Rationale**:
- Real-time updates
- Lower latency
- Reduced load on backend
- Trainer controls update frequency

**Tradeoff**: Requires trainer to have network access, which is standard in K8s.

### 5. Environment Variables for Configuration
**Decision**: Pass all configuration via environment variables, not config files or databases.

**Rationale**:
- 12-factor app principles
- Works across all environments
- K8s native (Secrets, ConfigMaps)
- No file system dependencies

**Tradeoff**: Limited to string values, but this is sufficient for our needs.

## Security Considerations

1. **Authentication**: JWT tokens for API access
2. **Authorization**: Role-based access control (RBAC)
3. **Secrets**: Kubernetes Secrets for sensitive data
4. **Network**: NetworkPolicies for pod-to-pod communication
5. **Storage**: Pre-signed URLs for S3 access (optional)
6. **Callbacks**: Token-based authentication for trainer callbacks
7. **Tracing**: Trace ID propagation for audit trails

## Scalability

1. **Backend**: Horizontal scaling (multiple replicas)
2. **Temporal**: Separate workers for different task queues
3. **Training**: Isolated K8s Jobs per training run
4. **Database**: Connection pooling, read replicas
5. **Storage**: S3 auto-scales
6. **WebSocket**: Redis pub/sub for multi-replica support

## References

- [Backend Design](./BACKEND_DESIGN.md)
- [Trainer Design](./TRAINER_DESIGN.md)
- [Workflows Design](./WORKFLOWS_DESIGN.md)
- [Frontend Design](./FRONTEND_DESIGN.md)
- [Infrastructure Design](./INFRASTRUCTURE_DESIGN.md)
- [Observability Design](./OBSERVABILITY_DESIGN.md)
- [Isolation Principles](./ISOLATION_DESIGN.md)
- [3-Tier Development](../development/3_TIER_DEVELOPMENT.md)
- [MVP Migration Plan](../migration/MVP_MIGRATION.md)
