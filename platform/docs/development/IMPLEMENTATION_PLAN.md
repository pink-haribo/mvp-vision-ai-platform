# Platform Implementation Plan

**Created**: 2025-01-11
**Status**: Draft
**Target**: Platform MVP (3-tier environment with complete isolation)

---

## Overview

This document provides a detailed, phase-by-phase implementation plan for the Vision AI Training Platform. The plan is based on the final design review and follows strict isolation principles and 3-tier environment parity.

### Prerequisites Completed

- ✅ All architecture design documents
- ✅ P0 operational documents:
  - ERROR_HANDLING_DESIGN.md
  - INTEGRATION_FAILURE_HANDLING.md
  - OPERATIONS_RUNBOOK.md
- ✅ Design review with P0/P1/P2 action items
- ✅ Repository reorganization (MVP → mvp/, Platform → platform/)

### Implementation Goals

**Primary Goal**: Build a production-ready Platform that enforces complete isolation between Backend and Training Services across all 3 tiers.

**Success Criteria**:
1. Same code works in Subprocess, Kind, and Production K8s
2. Backend ↔ Training Services communicate only via HTTP APIs
3. All storage uses S3-compatible APIs (MinIO for dev/Kind, R2/S3 for prod)
4. Complete error handling and retry logic implemented
5. Full observability (logs, metrics, traces)

---

## Phase Overview

```
Phase 1: Foundation (Week 1-2)          ← Core infrastructure
  ├── Backend skeleton (FastAPI + DB)
  ├── Training Service skeleton
  └── HTTP communication

Phase 2: Storage & Integration (Week 3-4)  ← S3-only + Callbacks
  ├── MinIO setup (all tiers)
  ├── S3 client abstraction
  └── Callback pattern implementation

Phase 3: 3-Tier Environment (Week 5-6)  ← Environment parity
  ├── Tier 1: Subprocess mode
  ├── Tier 2: Kind cluster
  └── Tier 3: Production (stub)

Phase Frontend: UI Development (Week 3-8)  ← Parallel with Phase 2-4
  ├── Design system setup
  ├── MVP migration & refinement
  └── Backend integration

Phase 4: Observability & Operations (Week 7-8)  ← Production readiness
  ├── Prometheus + Grafana
  ├── Centralized logging
  └── Error handling

Phase 5: Testing & Polish (Week 9-10)  ← Quality assurance
  ├── Integration tests
  ├── E2E tests
  └── Documentation
```

**Total Timeline**: 10 weeks (~2.5 months)
**Note**: Frontend development (Phase Frontend) runs in parallel with Phases 2-4

---

## Phase 1: Foundation (Week 1-2)

### Objective
Build the basic skeleton of Backend and Training Services with HTTP-only communication.

### Week 1.1: Backend Skeleton

**Goal**: FastAPI app with database models and basic CRUD APIs

**Tasks**:

1. **Project Structure** (1 day)
   ```
   platform/
   ├── backend/
   │   ├── app/
   │   │   ├── main.py              # FastAPI app
   │   │   ├── config.py            # Settings (pydantic-settings)
   │   │   ├── db/
   │   │   │   ├── models.py        # SQLAlchemy models
   │   │   │   └── session.py       # Async session
   │   │   ├── api/
   │   │   │   └── training.py      # Training job APIs
   │   │   └── schemas/
   │   │       └── training.py      # Pydantic schemas
   │   ├── pyproject.toml           # Poetry dependencies
   │   └── README.md
   ```

2. **Database Models** (1 day)
   ```python
   # app/db/models.py
   class TrainingJob(Base):
       id: UUID
       user_id: str
       model_name: str
       framework: str
       dataset_s3_uri: str       # ← S3 URI, not local path
       status: JobStatus
       callback_url: str
       created_at: datetime
       updated_at: datetime

   class TrainingMetric(Base):
       id: int
       job_id: UUID
       epoch: int
       loss: float
       accuracy: float
       created_at: datetime
   ```

3. **CRUD APIs** (2 days)
   ```python
   # app/api/training.py
   POST   /training/jobs              # Create job
   GET    /training/jobs              # List jobs
   GET    /training/jobs/{id}         # Get job
   POST   /training/jobs/{id}/callback  # Callback from trainer
   DELETE /training/jobs/{id}         # Cancel job
   ```

4. **Configuration** (0.5 day)
   ```python
   # app/config.py
   class Settings(BaseSettings):
       DATABASE_URL: str
       S3_ENDPOINT: str           # MinIO or R2
       AWS_ACCESS_KEY_ID: str
       AWS_SECRET_ACCESS_KEY: str
       BUCKET_NAME: str = "vision-platform"

       TRAINER_SERVICE_URL: str   # HTTP URL to training service
   ```

**Deliverables**:
- ✅ FastAPI app running on localhost:8000
- ✅ PostgreSQL database with TrainingJob table
- ✅ Basic CRUD APIs (no actual training yet)

**Testing**:
```bash
# Start backend
cd platform/backend
poetry install
poetry run uvicorn app.main:app --reload

# Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "model_name": "yolo11n",
    "framework": "ultralytics",
    "dataset_s3_uri": "s3://datasets/test/my-dataset/"
  }'
```

---

### Week 1.2: Training Service Skeleton

**Goal**: Framework-specific training service (start with Ultralytics)

**Tasks**:

1. **Project Structure** (1 day)
   ```
   platform/
   ├── training-services/
   │   └── ultralytics/
   │       ├── app/
   │       │   ├── main.py          # FastAPI app
   │       │   ├── config.py        # Settings
   │       │   ├── api/
   │       │   │   └── training.py  # Training endpoints
   │       │   └── trainer/
   │       │       └── train.py     # Actual training logic
   │       ├── Dockerfile
   │       ├── pyproject.toml
   │       └── README.md
   ```

2. **Training API** (1 day)
   ```python
   # app/api/training.py
   POST /training/start
     Body: {
       "job_id": "uuid",
       "config": {...},
       "dataset_s3_uri": "s3://...",
       "callback_url": "http://backend:8000/training/jobs/{id}/callback"
     }

   GET  /models                      # List available models
   GET  /health                      # Health check
   ```

3. **Stub Training Logic** (1 day)
   ```python
   # app/trainer/train.py
   async def train(job_id: str, config: dict, dataset_s3_uri: str, callback_url: str):
       # TODO: Download from S3, train model, upload checkpoints

       # For now, just send progress callbacks
       async with httpx.AsyncClient() as client:
           await client.post(callback_url, json={
               "job_id": job_id,
               "status": "running",
               "progress": 0.5,
               "message": "Training in progress..."
           })
   ```

4. **Dockerfile** (0.5 day)
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY pyproject.toml poetry.lock ./
   RUN pip install poetry && poetry install --no-dev
   COPY app/ ./app/
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
   ```

**Deliverables**:
- ✅ Training service running on localhost:8001
- ✅ `/training/start` endpoint receives config and dataset S3 URI
- ✅ Sends callback to Backend (stub implementation)

**Testing**:
```bash
# Start training service
cd platform/training-services/ultralytics
poetry install
poetry run uvicorn app.main:app --reload --port 8001

# Test API
curl http://localhost:8001/health
curl -X POST http://localhost:8001/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "123",
    "config": {"model": "yolo11n", "epochs": 10},
    "dataset_s3_uri": "s3://datasets/test/coco/",
    "callback_url": "http://localhost:8000/training/jobs/123/callback"
  }'
```

---

### Week 2: HTTP Communication

**Goal**: Backend triggers training via HTTP, receives callbacks

**Tasks**:

1. **TrainerClient in Backend** (1 day)
   ```python
   # platform/backend/app/clients/trainer.py
   class TrainerClient:
       def __init__(self, base_url: str):
           self.base_url = base_url
           self.session = httpx.AsyncClient(timeout=60.0)

       async def start_training(
           self,
           job_id: str,
           config: dict,
           dataset_s3_uri: str,
           callback_url: str
       ):
           response = await self.session.post(
               f"{self.base_url}/training/start",
               json={
                   "job_id": job_id,
                   "config": config,
                   "dataset_s3_uri": dataset_s3_uri,
                   "callback_url": callback_url
               }
           )
           response.raise_for_status()
           return response.json()
   ```

2. **Start Training Flow** (1 day)
   ```python
   # platform/backend/app/api/training.py
   @router.post("/training/jobs/{job_id}/start")
   async def start_training(job_id: UUID, db: AsyncSession):
       job = await db.get(TrainingJob, job_id)
       if not job:
           raise HTTPException(404)

       # Construct callback URL
       callback_url = f"{settings.BACKEND_BASE_URL}/training/jobs/{job_id}/callback"

       # Call Training Service
       trainer_client = TrainerClient(settings.TRAINER_SERVICE_URL)
       await trainer_client.start_training(
           job_id=str(job_id),
           config=job.config,
           dataset_s3_uri=job.dataset_s3_uri,
           callback_url=callback_url
       )

       # Update status
       job.status = JobStatus.RUNNING
       await db.commit()
       return {"status": "started"}
   ```

3. **Callback Endpoint** (1 day)
   ```python
   # platform/backend/app/api/training.py
   @router.post("/training/jobs/{job_id}/callback")
   async def training_callback(job_id: UUID, update: TrainingUpdate, db: AsyncSession):
       job = await db.get(TrainingJob, job_id)
       if not job:
           raise HTTPException(404)

       # Update job status
       job.status = update.status
       job.progress = update.progress
       job.message = update.message
       job.updated_at = datetime.utcnow()

       # Store metrics if provided
       if update.metrics:
           metric = TrainingMetric(
               job_id=job_id,
               epoch=update.metrics.epoch,
               loss=update.metrics.loss,
               accuracy=update.metrics.accuracy
           )
           db.add(metric)

       await db.commit()
       return {"status": "ok"}
   ```

4. **Integration Test** (1 day)
   ```python
   # tests/integration/test_training_flow.py
   async def test_full_training_flow():
       # 1. Create job
       response = await client.post("/training/jobs", json={...})
       job_id = response.json()["id"]

       # 2. Start training
       await client.post(f"/training/jobs/{job_id}/start")

       # 3. Wait for callback
       await asyncio.sleep(2)

       # 4. Check job status
       response = await client.get(f"/training/jobs/{job_id}")
       assert response.json()["status"] == "running"
   ```

**Deliverables**:
- ✅ Backend can trigger training via HTTP API
- ✅ Training service sends callbacks to Backend
- ✅ Job status updates correctly
- ✅ Integration test passes

**Milestone**: 🎯 **End-to-end HTTP communication working**

---

## Phase 2: Storage & Integration (Week 3-4)

### Objective
Implement S3-only storage strategy and complete callback pattern.

### Week 3.1: MinIO Setup

**Goal**: Run MinIO in all 3 tiers, replace all local file access with S3 APIs

**Tasks**:

1. **Docker Compose for MinIO** (0.5 day)
   ```yaml
   # platform/infrastructure/docker-compose.dev.yml
   services:
     minio:
       image: minio/minio:latest
       ports:
         - "9000:9000"  # API
         - "9001:9001"  # Console
       environment:
         MINIO_ROOT_USER: minioadmin
         MINIO_ROOT_PASSWORD: minioadmin
       command: server /data --console-address ":9001"
       volumes:
         - minio-data:/data

   volumes:
     minio-data:
   ```

2. **S3 Client Abstraction** (1 day)
   ```python
   # platform/backend/app/storage/s3.py
   import aioboto3

   class S3Client:
       def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
           self.endpoint = endpoint
           self.bucket = bucket
           self.session = aioboto3.Session(
               aws_access_key_id=access_key,
               aws_secret_access_key=secret_key
           )

       async def upload_file(self, local_path: Path, s3_key: str):
           async with self.session.client(
               "s3",
               endpoint_url=self.endpoint
           ) as s3:
               await s3.upload_file(str(local_path), self.bucket, s3_key)

       async def download_file(self, s3_key: str, local_path: Path):
           async with self.session.client(
               "s3",
               endpoint_url=self.endpoint
           ) as s3:
               await s3.download_file(self.bucket, s3_key, str(local_path))

       def get_s3_uri(self, key: str) -> str:
           return f"s3://{self.bucket}/{key}"
   ```

3. **Dataset Upload API** (1 day)
   ```python
   # platform/backend/app/api/datasets.py
   @router.post("/datasets/upload")
   async def upload_dataset(
       file: UploadFile,
       user_id: str,
       dataset_name: str,
       s3: S3Client = Depends(get_s3_client)
   ):
       # Save temporarily (in-memory or /tmp)
       temp_path = Path(f"/tmp/{file.filename}")
       with open(temp_path, "wb") as f:
           f.write(await file.read())

       # Upload to S3
       s3_key = f"users/{user_id}/datasets/{dataset_name}/{file.filename}"
       await s3.upload_file(temp_path, s3_key)

       # Clean up
       temp_path.unlink()

       return {
           "s3_uri": s3.get_s3_uri(s3_key),
           "size": file.size
       }
   ```

4. **Training Service S3 Download** (1.5 days)
   ```python
   # platform/training-services/ultralytics/app/trainer/train.py
   async def train(job_id: str, config: dict, dataset_s3_uri: str, callback_url: str):
       # Parse S3 URI
       bucket, key = parse_s3_uri(dataset_s3_uri)

       # Download dataset from S3 to /tmp
       local_dataset_dir = Path(f"/tmp/datasets/{job_id}")
       await s3_client.download_directory(bucket, key, local_dataset_dir)

       # Train model
       model = YOLO(config["model"])
       results = model.train(
           data=str(local_dataset_dir / "data.yaml"),
           epochs=config["epochs"]
       )

       # Upload checkpoints to S3
       checkpoint_s3_key = f"users/{user_id}/jobs/{job_id}/best.pt"
       await s3_client.upload_file(
           local_path=results.save_dir / "weights/best.pt",
           s3_key=checkpoint_s3_key
       )

       # Send completion callback
       await send_callback(callback_url, {
           "status": "completed",
           "checkpoint_s3_uri": f"s3://{bucket}/{checkpoint_s3_key}"
       })
   ```

**Deliverables**:
- ✅ MinIO running in Docker Compose
- ✅ Backend uploads datasets to S3
- ✅ Training service downloads from S3, trains, uploads checkpoints
- ✅ Zero local file system usage (except /tmp for downloads)

**Testing**:
```bash
# Start MinIO
docker-compose -f platform/infrastructure/docker-compose.dev.yml up -d minio

# Access MinIO console
open http://localhost:9001  # minioadmin/minioadmin

# Test upload
curl -X POST http://localhost:8000/datasets/upload \
  -F "file=@sample.zip" \
  -F "user_id=test" \
  -F "dataset_name=my-dataset"
```

---

### Week 3.2: Callback Pattern Completion

**Goal**: Implement complete callback flow with retries and error handling

**Tasks**:

1. **Callback Schema** (0.5 day)
   ```python
   # platform/backend/app/schemas/training.py
   class TrainingUpdate(BaseModel):
       job_id: str
       status: JobStatus  # running, completed, failed
       progress: float = 0.0
       message: str = ""
       metrics: Optional[Dict[str, float]] = None
       checkpoint_s3_uri: Optional[str] = None
       error: Optional[str] = None
   ```

2. **Retry Logic in Trainer** (1 day)
   ```python
   # platform/training-services/ultralytics/app/utils/callback.py
   from tenacity import retry, stop_after_attempt, wait_exponential

   @retry(
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=2, min=4, max=30),
       reraise=True
   )
   async def send_callback(url: str, data: dict):
       async with httpx.AsyncClient(timeout=10.0) as client:
           response = await client.post(url, json=data)
           response.raise_for_status()
   ```

3. **Progress Callbacks** (1 day)
   ```python
   # platform/training-services/ultralytics/app/trainer/train.py
   async def train(job_id: str, config: dict, dataset_s3_uri: str, callback_url: str):
       # Send start callback
       await send_callback(callback_url, {
           "status": "running",
           "progress": 0.0,
           "message": "Starting training..."
       })

       # Training loop with progress updates
       for epoch in range(config["epochs"]):
           # ... training code ...

           # Send progress every epoch
           await send_callback(callback_url, {
               "status": "running",
               "progress": (epoch + 1) / config["epochs"],
               "metrics": {"epoch": epoch, "loss": loss, "map": map}
           })

       # Send completion callback
       await send_callback(callback_url, {
           "status": "completed",
           "progress": 1.0,
           "checkpoint_s3_uri": checkpoint_uri
       })
   ```

4. **Error Handling** (1 day)
   ```python
   # platform/training-services/ultralytics/app/trainer/train.py
   async def train_with_error_handling(job_id: str, ...):
       try:
           await train(job_id, config, dataset_s3_uri, callback_url)
       except Exception as e:
           logger.exception(f"Training failed for job {job_id}")

           # Send error callback
           try:
               await send_callback(callback_url, {
                   "status": "failed",
                   "error": str(e)
               })
           except Exception as callback_error:
               logger.error(f"Failed to send error callback: {callback_error}")
               # TODO: Write to dead letter queue
   ```

**Deliverables**:
- ✅ Progress callbacks sent every epoch
- ✅ Completion callback with checkpoint S3 URI
- ✅ Error callbacks on failure
- ✅ Retry logic for callback failures

**Milestone**: 🎯 **Complete training workflow with S3 storage**

---

## Phase 3: 3-Tier Environment (Week 5-6)

### Objective
Make the same code work in Subprocess, Kind, and Production K8s.

### Week 5: Tier 1 (Subprocess Mode)

**Goal**: Backend runs training in subprocess (no K8s)

**Tasks**:

1. **Subprocess Trainer Manager** (1 day)
   ```python
   # platform/backend/app/services/subprocess_trainer.py
   import asyncio

   class SubprocessTrainerManager:
       async def start_training(self, job_id: str, config: dict, dataset_s3_uri: str):
           # Construct callback URL (localhost)
           callback_url = f"http://localhost:8000/training/jobs/{job_id}/callback"

           # Start training service as subprocess
           process = await asyncio.create_subprocess_exec(
               "python", "-m", "uvicorn",
               "app.main:app",
               "--host", "127.0.0.1",
               "--port", "8001",
               cwd="/path/to/training-services/ultralytics",
               stdout=asyncio.subprocess.PIPE,
               stderr=asyncio.subprocess.PIPE
           )

           # Trigger training via HTTP
           async with httpx.AsyncClient() as client:
               await client.post(
                   "http://localhost:8001/training/start",
                   json={
                       "job_id": job_id,
                       "config": config,
                       "dataset_s3_uri": dataset_s3_uri,
                       "callback_url": callback_url
                   }
               )
   ```

2. **Configuration for Subprocess** (0.5 day)
   ```python
   # platform/backend/app/config.py
   class Settings(BaseSettings):
       TRAINING_MODE: str = "subprocess"  # subprocess | kubernetes

       # Subprocess mode
       TRAINER_SUBPROCESS_PATH: Optional[Path] = None

       # Kubernetes mode
       KUBE_NAMESPACE: Optional[str] = None
   ```

3. **End-to-End Test** (1 day)
   ```bash
   # Start MinIO
   docker-compose up -d minio

   # Start Backend (subprocess mode)
   cd platform/backend
   TRAINING_MODE=subprocess poetry run uvicorn app.main:app --reload

   # Test full flow
   # 1. Upload dataset → MinIO
   # 2. Create training job → PostgreSQL
   # 3. Start training → Subprocess
   # 4. Training downloads from MinIO, trains, uploads to MinIO
   # 5. Callbacks update job status
   ```

**Deliverables**:
- ✅ Tier 1 (Subprocess) working end-to-end
- ✅ Same S3 client code used (MinIO endpoint)

---

### Week 6: Tier 2 (Kind Cluster)

**Goal**: Deploy Backend + Training Service to Kind, same code works

**Tasks**:

1. **Kubernetes Manifests** (1.5 days)
   ```yaml
   # platform/infrastructure/k8s/backend-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: backend
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: backend
     template:
       metadata:
         labels:
           app: backend
       spec:
         containers:
         - name: backend
           image: backend:latest
           env:
           - name: DATABASE_URL
             value: "postgresql://..."
           - name: S3_ENDPOINT
             value: "http://minio.platform.svc:9000"
           - name: TRAINING_MODE
             value: "kubernetes"
           - name: KUBE_NAMESPACE
             value: "platform"

   ---
   # platform/infrastructure/k8s/minio-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: minio
   spec:
     replicas: 1
     template:
       spec:
         containers:
         - name: minio
           image: minio/minio:latest
           command: ["server", "/data", "--console-address", ":9001"]

   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: minio
   spec:
     selector:
       app: minio
     ports:
     - name: api
       port: 9000
     - name: console
       port: 9001
   ```

2. **Kubernetes Trainer Manager** (1.5 days)
   ```python
   # platform/backend/app/services/k8s_trainer.py
   from kubernetes import client, config

   class KubernetesTrainerManager:
       def __init__(self):
           config.load_incluster_config()  # When running in K8s
           self.batch_v1 = client.BatchV1Api()

       async def start_training(self, job_id: str, config: dict, dataset_s3_uri: str):
           # Construct callback URL (internal K8s service)
           callback_url = f"http://backend.platform.svc:8000/training/jobs/{job_id}/callback"

           # Create Kubernetes Job
           job_manifest = client.V1Job(
               api_version="batch/v1",
               kind="Job",
               metadata=client.V1ObjectMeta(name=f"training-{job_id}"),
               spec=client.V1JobSpec(
                   template=client.V1PodTemplateSpec(
                       spec=client.V1PodSpec(
                           containers=[
                               client.V1Container(
                                   name="trainer",
                                   image="ultralytics-trainer:latest",
                                   env=[
                                       client.V1EnvVar(name="JOB_ID", value=job_id),
                                       client.V1EnvVar(name="CONFIG", value=json.dumps(config)),
                                       client.V1EnvVar(name="DATASET_S3_URI", value=dataset_s3_uri),
                                       client.V1EnvVar(name="CALLBACK_URL", value=callback_url),
                                       client.V1EnvVar(name="S3_ENDPOINT", value="http://minio.platform.svc:9000"),
                                   ]
                               )
                           ],
                           restart_policy="Never"
                       )
                   )
               )
           )

           self.batch_v1.create_namespaced_job(
               namespace="platform",
               body=job_manifest
           )
   ```

3. **Kind Setup Script** (1 day)
   ```bash
   # platform/scripts/kind-setup.sh
   #!/bin/bash

   # Create Kind cluster
   kind create cluster --name platform --config kind-config.yaml

   # Load images
   kind load docker-image backend:latest --name platform
   kind load docker-image ultralytics-trainer:latest --name platform

   # Apply manifests
   kubectl apply -f platform/infrastructure/k8s/

   # Wait for services
   kubectl wait --for=condition=available --timeout=300s \
     deployment/backend deployment/minio -n platform
   ```

4. **Integration Test** (1 day)
   ```bash
   # Deploy to Kind
   ./platform/scripts/kind-setup.sh

   # Port-forward to test
   kubectl port-forward svc/backend 8000:8000 -n platform

   # Test API (same curl commands as Tier 1)
   curl http://localhost:8000/health
   ```

**Deliverables**:
- ✅ Tier 2 (Kind) working end-to-end
- ✅ Same Backend code, different TRAINING_MODE env var
- ✅ MinIO running in Kind cluster
- ✅ Training Jobs created in K8s

**Milestone**: 🎯 **Environment parity achieved (Tier 1 & 2)**

---

## Phase Frontend: UI Development (Week 3-8)

**Note**: This phase runs in parallel with Phases 2-4

### Objective
Build a consistent, accessible, production-ready frontend by refining MVP components and integrating with the Platform backend.

---

### Week 3: Design System Setup

**Goal**: Establish design tokens, typography, and component foundations

**Tasks**:

1. **Install Dependencies** (0.5 day)
   ```bash
   # platform/frontend/
   pnpm install next@15 react@18 react-dom@18
   pnpm install -D tailwindcss@3.4 postcss autoprefixer
   pnpm install -D typescript @types/react @types/node
   pnpm install class-variance-authority clsx tailwind-merge
   pnpm install @radix-ui/react-alert-dialog @radix-ui/react-select
   pnpm install lucide-react  # Icons
   ```

2. **Design Tokens Configuration** (1 day)
   ```typescript
   // platform/frontend/app/globals.css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;

   @layer base {
     :root {
       --background: 0 0% 100%;
       --foreground: 222.2 84% 4.9%;
       --primary: 222.2 47.4% 11.2%;
       --success: 142 76% 36%;         /* Added */
       --warning: 38 92% 50%;          /* Added */
       --info: 199 89% 48%;            /* Added */
       --destructive: 0 84.2% 60.2%;
       --radius: 0.5rem;
     }
   }
   ```

3. **Font Setup** (1 day)
   ```typescript
   // platform/frontend/app/layout.tsx
   import { Inter } from "next/font/google";
   import localFont from "next/font/local";

   const inter = Inter({
     subsets: ["latin"],
     variable: "--font-inter",
   });

   const suit = localFont({
     src: "../fonts/SUIT-Variable.woff2",
     variable: "--font-suit",
     weight: "100 900",
   });

   export default function RootLayout({ children }: { children: React.ReactNode }) {
     return (
       <html lang="ko" className={`${inter.variable} ${suit.variable}`}>
         <body className="font-sans">{children}</body>
       </html>
     );
   }
   ```

   ```typescript
   // platform/frontend/tailwind.config.ts
   fontFamily: {
     sans: [
       "var(--font-suit)",        // Korean (SUIT)
       "var(--font-inter)",       // Latin
       "system-ui",
       "sans-serif",
     ],
   }
   ```

4. **Base Component Setup** (1.5 days)
   ```bash
   # Install shadcn/ui CLI
   pnpm dlx shadcn-ui@latest init

   # Add core components
   pnpm dlx shadcn-ui@latest add button
   pnpm dlx shadcn-ui@latest add card
   pnpm dlx shadcn-ui@latest add alert
   pnpm dlx shadcn-ui@latest add input
   pnpm dlx shadcn-ui@latest add label
   pnpm dlx shadcn-ui@latest add select
   pnpm dlx shadcn-ui@latest add skeleton
   pnpm dlx shadcn-ui@latest add toast
   ```

**Deliverables**:
- ✅ Design tokens configured (including success/warning/info)
- ✅ Korean font (SUIT) + Latin font (Inter) working
- ✅ shadcn/ui components installed
- ✅ Storybook setup for component development (optional)

**Testing**:
```bash
cd platform/frontend
pnpm dev  # http://localhost:3000

# Create test page to verify fonts and colors
# app/design-test/page.tsx
```

---

### Week 4: Component Library & Patterns

**Goal**: Build common UI patterns and state components

**Tasks**:

1. **Loading States** (1 day)
   ```tsx
   // components/ui/loading-card.tsx
   import { Skeleton } from "@/components/ui/skeleton";
   import { Card, CardContent, CardHeader } from "@/components/ui/card";

   export function LoadingCard() {
     return (
       <Card>
         <CardHeader>
           <Skeleton className="h-6 w-40" />
         </CardHeader>
         <CardContent className="space-y-2">
           <Skeleton className="h-4 w-full" />
           <Skeleton className="h-4 w-3/4" />
         </CardContent>
       </Card>
     );
   }
   ```

2. **Error States** (1 day)
   ```tsx
   // components/ui/error-alert.tsx
   import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
   import { AlertCircle } from "lucide-react";

   interface ErrorAlertProps {
     title?: string;
     message: string;
     variant?: "destructive" | "warning";
   }

   export function ErrorAlert({ title = "Error", message, variant = "destructive" }: ErrorAlertProps) {
     return (
       <Alert variant={variant}>
         <AlertCircle className="h-4 w-4" />
         <AlertTitle>{title}</AlertTitle>
         <AlertDescription>{message}</AlertDescription>
       </Alert>
     );
   }
   ```

3. **Empty States** (1 day)
   ```tsx
   // components/ui/empty-state.tsx
   import { Card, CardContent } from "@/components/ui/card";
   import { Button } from "@/components/ui/button";

   interface EmptyStateProps {
     icon: React.ReactNode;
     title: string;
     description: string;
     action?: {
       label: string;
       onClick: () => void;
     };
   }

   export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
     return (
       <Card>
         <CardContent className="flex flex-col items-center justify-center py-12">
           <div className="text-muted-foreground mb-4">{icon}</div>
           <h3 className="text-lg font-semibold mb-2">{title}</h3>
           <p className="text-sm text-muted-foreground mb-6 text-center max-w-sm">
             {description}
           </p>
           {action && (
             <Button onClick={action.onClick}>{action.label}</Button>
           )}
         </CardContent>
       </Card>
     );
   }
   ```

4. **Form Patterns** (1 day)
   ```tsx
   // components/forms/form-field.tsx
   import { Label } from "@/components/ui/label";
   import { Input } from "@/components/ui/input";

   interface FormFieldProps {
     id: string;
     label: string;
     required?: boolean;
     error?: string;
     description?: string;
     children: React.ReactNode;
   }

   export function FormField({ id, label, required, error, description, children }: FormFieldProps) {
     return (
       <div className="space-y-2">
         <Label htmlFor={id}>
           {label}
           {required && <span className="text-destructive ml-1" aria-label="required">*</span>}
         </Label>
         {children}
         {description && !error && (
           <p id={`${id}-description`} className="text-sm text-muted-foreground">
             {description}
           </p>
         )}
         {error && (
           <p id={`${id}-error`} className="text-sm text-destructive" role="alert">
             {error}
           </p>
         )}
       </div>
     );
   }
   ```

**Deliverables**:
- ✅ Loading, Error, Empty State components
- ✅ Accessible form field pattern
- ✅ Component documentation
- ✅ Visual regression tests (optional)

---

### Week 5: MVP Component Migration

**Goal**: Copy and refine MVP components to Platform frontend

**Tasks**:

1. **Analyze MVP Components** (0.5 day)
   ```bash
   # Review mvp/frontend/components/ structure
   # Identify components to migrate:
   # - ChatPanel → platform/frontend/components/chat/
   # - TrainingPanel → platform/frontend/components/training/
   # - DatasetPanel → platform/frontend/components/datasets/
   # - Sidebar → platform/frontend/components/layout/
   ```

2. **Migrate & Refactor Components** (2 days)
   ```tsx
   // platform/frontend/components/training/training-card.tsx
   import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
   import { Badge } from "@/components/ui/badge";
   import { Button } from "@/components/ui/button";

   interface TrainingCardProps {
     job: {
       id: string;
       model_name: string;
       status: "pending" | "running" | "completed" | "failed";
       progress: number;
     };
     onView: (id: string) => void;
   }

   export function TrainingCard({ job, onView }: TrainingCardProps) {
     const statusColors = {
       pending: "secondary",
       running: "info",
       completed: "success",
       failed: "destructive",
     } as const;

     return (
       <Card>
         <CardHeader className="flex flex-row items-center justify-between">
           <CardTitle className="text-base">{job.model_name}</CardTitle>
           <Badge variant={statusColors[job.status]}>
             {job.status}
           </Badge>
         </CardHeader>
         <CardContent>
           <div className="space-y-2">
             <div className="flex justify-between text-sm">
               <span className="text-muted-foreground">Progress</span>
               <span className="font-medium">{Math.round(job.progress * 100)}%</span>
             </div>
             <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
               <div
                 className="bg-primary h-full transition-all duration-300"
                 style={{ width: `${job.progress * 100}%` }}
               />
             </div>
             <Button
               variant="outline"
               size="sm"
               className="w-full mt-4"
               onClick={() => onView(job.id)}
             >
               View Details
             </Button>
           </div>
         </CardContent>
       </Card>
     );
   }
   ```

3. **Fix Design Inconsistencies** (1 day)
   - Replace hardcoded colors (text-green-600 → text-success)
   - Add missing ARIA labels
   - Ensure consistent spacing (use design tokens)
   - Add keyboard navigation

4. **Responsive Design Improvements** (0.5 day)
   ```tsx
   // Example: Responsive grid layout
   <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
     {jobs.map(job => <TrainingCard key={job.id} job={job} />)}
   </div>
   ```

**Deliverables**:
- ✅ All MVP components migrated to Platform
- ✅ Design tokens used consistently
- ✅ Responsive design working
- ✅ Accessibility improved

---

### Week 6: Backend API Integration

**Goal**: Connect frontend to Platform Backend APIs

**Tasks**:

1. **API Client Setup** (1 day)
   ```typescript
   // platform/frontend/lib/api/client.ts
   import axios from "axios";

   const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

   export const apiClient = axios.create({
     baseURL: API_BASE_URL,
     timeout: 30000,
     headers: {
       "Content-Type": "application/json",
     },
   });

   // Request interceptor for auth
   apiClient.interceptors.request.use((config) => {
     const token = localStorage.getItem("auth_token");
     if (token) {
       config.headers.Authorization = `Bearer ${token}`;
     }
     return config;
   });

   // Response interceptor for error handling
   apiClient.interceptors.response.use(
     (response) => response,
     (error) => {
       if (error.response?.status === 401) {
         // Redirect to login
         window.location.href = "/login";
       }
       return Promise.reject(error);
     }
   );
   ```

2. **Training Job API** (1.5 days)
   ```typescript
   // platform/frontend/lib/api/training.ts
   import { apiClient } from "./client";

   export interface TrainingJob {
     id: string;
     user_id: string;
     model_name: string;
     framework: string;
     dataset_s3_uri: string;
     status: "pending" | "running" | "completed" | "failed";
     progress: number;
     created_at: string;
     updated_at: string;
   }

   export const trainingApi = {
     async listJobs(): Promise<TrainingJob[]> {
       const response = await apiClient.get("/training/jobs");
       return response.data;
     },

     async getJob(id: string): Promise<TrainingJob> {
       const response = await apiClient.get(`/training/jobs/${id}`);
       return response.data;
     },

     async createJob(data: {
       model_name: string;
       framework: string;
       dataset_s3_uri: string;
     }): Promise<TrainingJob> {
       const response = await apiClient.post("/training/jobs", data);
       return response.data;
     },

     async startJob(id: string): Promise<void> {
       await apiClient.post(`/training/jobs/${id}/start`);
     },

     async cancelJob(id: string): Promise<void> {
       await apiClient.delete(`/training/jobs/${id}`);
     },
   };
   ```

3. **React Query Integration** (1.5 days)
   ```tsx
   // platform/frontend/lib/hooks/useTrainingJobs.ts
   import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
   import { trainingApi } from "@/lib/api/training";

   export function useTrainingJobs() {
     return useQuery({
       queryKey: ["training-jobs"],
       queryFn: () => trainingApi.listJobs(),
       refetchInterval: 5000, // Poll every 5 seconds
     });
   }

   export function useCreateTrainingJob() {
     const queryClient = useQueryClient();

     return useMutation({
       mutationFn: trainingApi.createJob,
       onSuccess: () => {
         queryClient.invalidateQueries({ queryKey: ["training-jobs"] });
       },
     });
   }

   // Usage in component
   function TrainingList() {
     const { data: jobs, isLoading, error } = useTrainingJobs();
     const createJob = useCreateTrainingJob();

     if (isLoading) return <LoadingCard />;
     if (error) return <ErrorAlert message={error.message} />;
     if (!jobs?.length) return <EmptyState title="No training jobs" />;

     return (
       <div className="space-y-4">
         {jobs.map(job => <TrainingCard key={job.id} job={job} />)}
       </div>
     );
   }
   ```

**Deliverables**:
- ✅ API client configured with auth
- ✅ Training job API integrated
- ✅ React Query setup for data fetching
- ✅ Loading/error states working

---

### Week 7: State Management & Real-time Updates

**Goal**: Add global state and WebSocket for real-time training updates

**Tasks**:

1. **Zustand Store** (1 day)
   ```typescript
   // platform/frontend/lib/store/training.ts
   import { create } from "zustand";
   import type { TrainingJob } from "@/lib/api/training";

   interface TrainingStore {
     activeJobs: Map<string, TrainingJob>;
     updateJob: (job: TrainingJob) => void;
     removeJob: (id: string) => void;
   }

   export const useTrainingStore = create<TrainingStore>((set) => ({
     activeJobs: new Map(),
     updateJob: (job) =>
       set((state) => {
         const newJobs = new Map(state.activeJobs);
         newJobs.set(job.id, job);
         return { activeJobs: newJobs };
       }),
     removeJob: (id) =>
       set((state) => {
         const newJobs = new Map(state.activeJobs);
         newJobs.delete(id);
         return { activeJobs: newJobs };
       }),
   }));
   ```

2. **WebSocket Integration** (2 days)
   ```typescript
   // platform/frontend/lib/websocket/training.ts
   import { useEffect } from "react";
   import { useTrainingStore } from "@/lib/store/training";

   export function useTrainingWebSocket(jobId: string) {
     const updateJob = useTrainingStore((state) => state.updateJob);

     useEffect(() => {
       const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
       const ws = new WebSocket(`${WS_URL}/ws/training/${jobId}`);

       ws.onmessage = (event) => {
         const update = JSON.parse(event.data);
         updateJob(update);
       };

       ws.onerror = (error) => {
         console.error("WebSocket error:", error);
       };

       return () => {
         ws.close();
       };
     }, [jobId, updateJob]);
   }
   ```

3. **Toast Notifications** (1 day)
   ```tsx
   // platform/frontend/components/training/training-monitor.tsx
   import { useEffect } from "react";
   import { useToast } from "@/components/ui/use-toast";
   import { useTrainingStore } from "@/lib/store/training";

   export function TrainingMonitor() {
     const { toast } = useToast();
     const activeJobs = useTrainingStore((state) => state.activeJobs);

     useEffect(() => {
       activeJobs.forEach((job) => {
         if (job.status === "completed") {
           toast({
             title: "Training Complete",
             description: `${job.model_name} finished successfully`,
             variant: "success",
           });
         } else if (job.status === "failed") {
           toast({
             title: "Training Failed",
             description: `${job.model_name} encountered an error`,
             variant: "destructive",
           });
         }
       });
     }, [activeJobs, toast]);

     return null;
   }
   ```

**Deliverables**:
- ✅ Zustand store for global state
- ✅ WebSocket connection for real-time updates
- ✅ Toast notifications for training events
- ✅ Automatic UI updates when job status changes

---

### Week 8: Testing & Polish

**Goal**: Ensure quality and accessibility

**Tasks**:

1. **Unit Tests** (1.5 days)
   ```tsx
   // platform/frontend/__tests__/components/training-card.test.tsx
   import { render, screen } from "@testing-library/react";
   import userEvent from "@testing-library/user-event";
   import { TrainingCard } from "@/components/training/training-card";

   describe("TrainingCard", () => {
     it("renders job information correctly", () => {
       const job = {
         id: "123",
         model_name: "yolo11n",
         status: "running" as const,
         progress: 0.5,
       };

       render(<TrainingCard job={job} onView={jest.fn()} />);

       expect(screen.getByText("yolo11n")).toBeInTheDocument();
       expect(screen.getByText("running")).toBeInTheDocument();
       expect(screen.getByText("50%")).toBeInTheDocument();
     });

     it("calls onView when button clicked", async () => {
       const onView = jest.fn();
       const job = { id: "123", model_name: "test", status: "completed" as const, progress: 1 };

       render(<TrainingCard job={job} onView={onView} />);

       await userEvent.click(screen.getByText("View Details"));
       expect(onView).toHaveBeenCalledWith("123");
     });
   });
   ```

2. **Accessibility Audit** (1 day)
   ```bash
   # Install axe-core for accessibility testing
   pnpm add -D @axe-core/react

   # Run lighthouse audit
   pnpm build
   pnpm start
   # Use Chrome DevTools Lighthouse for accessibility score (target: 95+)
   ```

3. **E2E Tests** (1.5 days)
   ```typescript
   // platform/frontend/e2e/training-flow.spec.ts
   import { test, expect } from "@playwright/test";

   test("user can create and monitor training job", async ({ page }) => {
     await page.goto("http://localhost:3000/training");

     // Create job
     await page.click("text=New Training Job");
     await page.fill('input[name="model_name"]', "yolo11n");
     await page.click("text=Start Training");

     // Wait for job to appear
     await expect(page.locator("text=yolo11n")).toBeVisible();
     await expect(page.locator("text=running")).toBeVisible();

     // Monitor progress
     await page.waitForSelector("text=completed", { timeout: 60000 });
   });
   ```

**Deliverables**:
- ✅ Unit tests for all components (80%+ coverage)
- ✅ Accessibility score 95+ (Lighthouse)
- ✅ E2E tests for critical user flows
- ✅ Performance optimization (Lighthouse: 90+)

**Milestone**: 🎯 **Production-ready frontend with backend integration**

---

## Phase 4: Observability & Operations (Week 7-8)

### Objective
Add monitoring, logging, and error handling for production readiness.

### Week 7: Monitoring

**Tasks**:

1. **Prometheus Metrics** (1 day)
   ```python
   # platform/backend/app/monitoring/metrics.py
   from prometheus_client import Counter, Histogram, Gauge

   training_jobs_total = Counter(
       "training_jobs_total",
       "Total number of training jobs created",
       ["framework", "status"]
   )

   training_duration_seconds = Histogram(
       "training_duration_seconds",
       "Training job duration",
       ["framework"]
   )

   active_training_jobs = Gauge(
       "active_training_jobs",
       "Number of currently active training jobs"
   )
   ```

2. **Grafana Dashboards** (1 day)
   - Import from platform/docs/architecture/OBSERVABILITY_DESIGN.md

3. **Centralized Logging** (2 days)
   - Loki for log aggregation
   - Structured logging in all services

**Deliverables**:
- ✅ Prometheus metrics exposed
- ✅ Grafana dashboard deployed
- ✅ Centralized logging working

---

### Week 8: Error Handling

**Tasks**:

1. **Implement Error Taxonomy** (1 day)
   - Based on ERROR_HANDLING_DESIGN.md

2. **Retry Policies** (1 day)
   - Based on INTEGRATION_FAILURE_HANDLING.md

3. **Circuit Breakers** (1 day)
   - For Trainer Service calls

4. **Sentry Integration** (1 day)
   - Error tracking and alerting

**Deliverables**:
- ✅ Error handling implemented per design docs
- ✅ Sentry capturing errors
- ✅ Retry logic tested

**Milestone**: 🎯 **Production-ready observability**

---

## Phase 5: Testing & Polish (Week 9-10)

### Objective
Ensure quality through comprehensive testing.

### Week 9: Integration & E2E Tests

**Tasks**:

1. **Integration Tests** (2 days)
   ```python
   # tests/integration/test_full_workflow.py
   async def test_tier1_full_workflow():
       # Upload dataset
       # Create job
       # Start training (subprocess)
       # Wait for completion
       # Verify checkpoint in MinIO

   async def test_tier2_full_workflow():
       # Same test, but in Kind cluster
   ```

2. **E2E Tests** (2 days)
   ```python
   # tests/e2e/test_user_journey.py
   async def test_user_trains_model():
       # User uploads dataset via API
       # User creates training job
       # System trains model
       # User downloads checkpoint
   ```

3. **Load Tests** (1 day)
   ```python
   # tests/load/locustfile.py
   # Simulate 10 concurrent training jobs
   ```

**Deliverables**:
- ✅ 90%+ test coverage
- ✅ All integration tests passing
- ✅ E2E tests passing in both Tier 1 & 2

---

### Week 10: Documentation & Polish

**Tasks**:

1. **Update Documentation** (2 days)
   - README for each component
   - API documentation (OpenAPI)
   - Deployment guides

2. **Code Review & Refactor** (2 days)
   - Clean up code
   - Remove TODOs
   - Consistent naming

3. **Final Testing** (1 day)
   - Smoke tests in all 3 tiers
   - Security scan

**Deliverables**:
- ✅ Complete documentation
- ✅ Clean codebase
- ✅ Ready for code review

**Milestone**: 🎯 **Platform MVP complete**

---

## Success Metrics

### Technical Metrics

- [ ] Same code runs in all 3 tiers (no `if environment == ...`)
- [ ] Zero local file system usage (100% S3 APIs)
- [ ] Backend ↔ Training Services communicate only via HTTP
- [ ] 90%+ test coverage
- [ ] All integration tests passing
- [ ] < 5 second callback latency (p95)
- [ ] Training jobs complete successfully 95%+ of time

### Operational Metrics

- [ ] Prometheus metrics implemented
- [ ] Grafana dashboards deployed
- [ ] Centralized logging working
- [ ] Error handling tested (retry, circuit breaker)
- [ ] Runbook procedures documented

### Documentation

- [ ] All design documents updated
- [ ] API documentation complete (OpenAPI)
- [ ] Deployment guides for all 3 tiers
- [ ] Operations runbook tested

---

## Risk Management

### High Risks

1. **S3 Performance in Subprocess Mode**
   - Risk: MinIO in Docker may be slow
   - Mitigation: Use local cache for frequently accessed files

2. **Callback Failures**
   - Risk: Network issues → lost callbacks
   - Mitigation: Dead letter queue + polling fallback

3. **Database Migrations**
   - Risk: Breaking changes during development
   - Mitigation: Alembic migrations, careful schema design

### Medium Risks

1. **Kind Cluster Stability**
   - Risk: Kind cluster crashes during development
   - Mitigation: Automated setup scripts

2. **Dependency Conflicts**
   - Risk: Different Python versions in Backend vs Trainer
   - Mitigation: Use same base image (python:3.11-slim)

---

## Next Steps

After completing this plan, the following can be tackled:

1. **Tier 3 (Production K8s)**: Deploy to AWS/GCP
2. **Temporal Integration**: Replace subprocess/K8s job with Temporal workflows
3. **More Frameworks**: Add timm, HuggingFace trainers
4. **Frontend**: Build UI for training management
5. **MLflow Integration**: Complete integration with tracking
6. **Advanced Features**: Multi-GPU, distributed training

---

**Document Version**: 1.0
**Last Updated**: 2025-01-11
**Status**: Ready for Implementation
