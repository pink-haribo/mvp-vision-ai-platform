# Complete Dependency Isolation Design

Complete isolation between Backend and Trainers - the foundation of the platform architecture.

## Table of Contents

- [Overview](#overview)
- [Why Isolation is Critical](#why-isolation-is-critical)
- [Isolation Principles](#isolation-principles)
- [Communication Patterns](#communication-patterns)
- [Forbidden Patterns](#forbidden-patterns)
- [Verification Tests](#verification-tests)

## Overview

The platform enforces **complete dependency isolation** between Backend and Training services. This means:
- No shared code
- No shared file system
- No direct imports
- Only HTTP API communication
- All data exchange via S3 storage

## Why Isolation is Critical

### 1. Independent Deployment
```
Backend updates → Deploy backend only
Trainer updates → Deploy trainers only
No coordination needed
```

###2. Language Flexibility
```
Backend: Python (FastAPI)
Trainer: ANY language (Python, Rust, C++, Go)
Only requirement: HTTP callbacks
```

### 3. Version Independence
```
Backend v2.0 + Trainer v1.0 = Works
Backend v1.0 + Trainer v2.0 = Works
Only API contract matters
```

### 4. Multi-Tenant Safety
```
User A's training → Isolated pod/process
User B's training → Isolated pod/process
No shared state, no interference
```

## Isolation Principles

### Principle 1: No Shared File System

❌ **Forbidden**:
```python
# Backend writes file
with open("/shared/config.json", "w") as f:
    json.dump(config, f)

# Trainer reads file
with open("/shared/config.json", "r") as f:
    config = json.load(f)
```

✅ **Correct**:
```python
# Backend: Upload to S3
s3.upload_file("config.json", "configs/job-123.json")

# Trainer: Download from S3
s3.download_file("configs/job-123.json", "/tmp/config.json")
```

### Principle 2: No Direct Imports

❌ **Forbidden**:
```python
# Trainer code
from backend.app.db.models import TrainingJob  # NEVER!
from backend.app.utils.metrics import calculate_loss  # NEVER!

job = TrainingJob.query.get(job_id)
```

✅ **Correct**:
```python
# Trainer code - reads config from environment variables
JOB_ID = os.environ["JOB_ID"]
MODEL_NAME = os.environ["MODEL_NAME"]
CALLBACK_URL = os.environ["CALLBACK_URL"]

# No backend imports!
```

### Principle 3: API-Only Communication

❌ **Forbidden**:
```python
# Direct database access from trainer
engine = create_engine(DATABASE_URL)
session = Session(engine)
job = session.query(TrainingJob).get(job_id)
job.status = "completed"
session.commit()
```

✅ **Correct**:
```python
# HTTP callback to backend
requests.post(
    f"{BACKEND_URL}/api/v1/jobs/{JOB_ID}/done",
    headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
    json={"status": "succeeded", "metrics": {...}}
)
```

### Principle 4: Storage as Data Exchange

❌ **Forbidden**:
```python
# Backend passes data via environment variable (size limit!)
os.environ["DATASET_SAMPLES"] = json.dumps(huge_list)  # Too large!
```

✅ **Correct**:
```python
# Backend: Upload dataset to S3
s3.upload_file("dataset.zip", f"datasets/{dataset_id}.zip")

# Trainer: Download from S3
dataset_id = os.environ["DATASET_ID"]
s3.download_file(f"datasets/{dataset_id}.zip", "/tmp/dataset.zip")
```

## Communication Patterns

### Pattern 1: Configuration via Environment Variables

**Backend creates training job:**
```python
env = {
    "JOB_ID": str(job.id),
    "TRACE_ID": str(job.trace_id),
    "BACKEND_BASE_URL": settings.BACKEND_BASE_URL,
    "CALLBACK_TOKEN": create_callback_token(job.id),
    "MODEL_NAME": job.model_name,
    "DATASET_ID": str(job.dataset_id),
    "EPOCHS": str(job.epochs),
    "BATCH_SIZE": str(job.batch_size),
    "STORAGE_TYPE": "r2",
    "R2_ENDPOINT": settings.R2_ENDPOINT,
    "R2_ACCESS_KEY_ID": settings.R2_ACCESS_KEY_ID,
    "R2_SECRET_ACCESS_KEY": settings.R2_SECRET_ACCESS_KEY,
}

# Subprocess mode
subprocess.Popen(["python", "train.py"], env=env)

# Kubernetes mode
create_job(env_vars=env)
```

**Trainer reads configuration:**
```python
import os

job_id = os.environ["JOB_ID"]
model_name = os.environ["MODEL_NAME"]
dataset_id = os.environ["DATASET_ID"]
epochs = int(os.environ["EPOCHS"])
# ... etc
```

### Pattern 2: Progress Updates via HTTP Callbacks

**Trainer sends progress:**
```python
def send_heartbeat(epoch, metrics):
    requests.post(
        f"{BACKEND_URL}/api/v1/jobs/{JOB_ID}/heartbeat",
        headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
        json={
            "epoch": epoch,
            "progress": (epoch / total_epochs) * 100,
            "metrics": metrics
        }
    )

# Called every epoch
for epoch in range(epochs):
    metrics = train_epoch()
    send_heartbeat(epoch + 1, metrics)
```

**Backend receives progress:**
```python
@router.post("/jobs/{job_id}/heartbeat")
async def receive_heartbeat(
    job_id: UUID,
    heartbeat: HeartbeatData,
    token: str = Depends(verify_callback_token),
    db: AsyncSession = Depends(get_db)
):
    job = await db.get(TrainingJob, job_id)
    job.current_epoch = heartbeat.epoch
    job.progress_percent = heartbeat.progress
    job.current_metrics = heartbeat.metrics
    await db.commit()

    # Broadcast to WebSocket clients
    await broadcast_update(str(job_id), heartbeat.dict())

    return {"status": "ok"}
```

### Pattern 3: Data Exchange via S3

**Backend uploads dataset:**
```python
# Backend prepares dataset
dataset_path = prepare_dataset(dataset_id)
zip_dataset(dataset_path, "/tmp/dataset.zip")

# Upload to S3
s3.upload_file(
    "/tmp/dataset.zip",
    "datasets",
    f"{dataset_id}.zip"
)

# Pass dataset_id to trainer (not the file!)
env["DATASET_ID"] = str(dataset_id)
```

**Trainer downloads dataset:**
```python
dataset_id = os.environ["DATASET_ID"]
s3_endpoint = os.environ.get("R2_ENDPOINT")

# Download from S3
s3 = boto3.client('s3', endpoint_url=s3_endpoint, ...)
s3.download_file(
    "datasets",
    f"{dataset_id}.zip",
    "/workspace/dataset.zip"
)

# Extract
unzip("/workspace/dataset.zip", "/workspace/dataset")
```

**Trainer uploads checkpoint:**
```python
# Train and save checkpoint
checkpoint_path = "/workspace/checkpoints/best.pt"
save_checkpoint(model, checkpoint_path)

# Upload to S3
s3.upload_file(
    checkpoint_path,
    "checkpoints",
    f"job-{JOB_ID}/best.pt"
)

# Notify backend
requests.post(
    f"{BACKEND_URL}/api/v1/jobs/{JOB_ID}/event",
    json={
        "event_type": "checkpoint_saved",
        "data": {
            "checkpoint_path": f"s3://checkpoints/job-{JOB_ID}/best.pt"
        }
    }
)
```

## Forbidden Patterns

### ❌ Pattern 1: Shared Python Code

**Don't create a shared utils package:**
```
# BAD STRUCTURE
shared/
├── utils/
│   ├── metrics.py       # Used by both backend and trainer
│   └── storage.py       # Used by both backend and trainer
backend/
└── app/
    └── main.py          # imports from shared/
trainers/
└── ultralytics/
    └── train.py         # imports from shared/  ← FORBIDDEN!
```

**Why it's forbidden:**
- Creates tight coupling
- Backend and trainer must be deployed together
- Can't use different languages for trainers
- Updates to shared code break isolation

**What to do instead:**
```
# CORRECT STRUCTURE
backend/
└── app/
    └── utils/
        ├── metrics.py   # Backend-specific utils
        └── storage.py
trainers/
└── ultralytics/
    └── utils.py         # Trainer-specific utils (can duplicate code!)
```

Code duplication is **OK** when it maintains isolation!

### ❌ Pattern 2: Shared Database

**Don't access the database from trainers:**
```python
# Trainer code - FORBIDDEN!
from sqlalchemy import create_engine
from backend.app.db.models import TrainingJob

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Update job status directly
job = session.query(TrainingJob).filter_by(id=job_id).first()
job.status = "running"
session.commit()
```

**Why it's forbidden:**
- Database is backend's internal implementation
- Trainer shouldn't know backend's schema
- Schema changes break trainers
- No transaction isolation

**What to do instead:**
```python
# Trainer code - CORRECT
requests.post(
    f"{BACKEND_URL}/api/v1/jobs/{JOB_ID}/status",
    json={"status": "running"}
)
```

### ❌ Pattern 3: Shared Configuration Files

**Don't use shared config files:**
```yaml
# config.yaml (shared file) - FORBIDDEN!
backend:
  port: 8000
  database_url: postgresql://...

trainer:
  batch_size: 16
  learning_rate: 0.001
```

```python
# Backend
with open("/shared/config.yaml") as f:
    config = yaml.load(f)

# Trainer
with open("/shared/config.yaml") as f:  # ← File system dependency!
    config = yaml.load(f)
```

**Why it's forbidden:**
- Requires shared file system
- Doesn't work in Kubernetes (different pods)
- Config changes require restart of both services

**What to do instead:**
```python
# Backend: Environment variables
backend_port = int(os.environ["BACKEND_PORT"])

# Trainer: Environment variables (passed by backend)
batch_size = int(os.environ["BATCH_SIZE"])
learning_rate = float(os.environ["LEARNING_RATE"])
```

## Verification Tests

### Test 1: No Direct Imports

```python
# tests/isolation/test_no_imports.py
import ast
import os
from pathlib import Path

def test_trainer_does_not_import_backend():
    """Verify trainer code doesn't import from backend"""

    trainer_dir = Path("trainers/")
    forbidden_imports = [
        "from backend",
        "import backend",
        "from app",  # Backend app module
    ]

    for trainer_file in trainer_dir.rglob("*.py"):
        with open(trainer_file) as f:
            content = f.read()

        for forbidden in forbidden_imports:
            assert forbidden not in content, \
                f"{trainer_file} contains forbidden import: {forbidden}"

def test_trainer_has_no_backend_dependency():
    """Verify trainer requirements.txt doesn't list backend packages"""

    for req_file in Path("trainers/").rglob("requirements.txt"):
        with open(req_file) as f:
            requirements = f.read()

        # Backend-specific packages
        forbidden_packages = [
            "fastapi",
            "sqlalchemy",
            "alembic",
            "psycopg2",
        ]

        for pkg in forbidden_packages:
            assert pkg not in requirements.lower(), \
                f"{req_file} contains backend package: {pkg}"
```

### Test 2: Communication via API Only

```python
# tests/isolation/test_api_communication.py
import subprocess
import time
import requests

def test_trainer_communicates_via_http():
    """Verify trainer sends callbacks to backend HTTP API"""

    # Start mock backend server
    backend_proc = subprocess.Popen([
        "python", "-m", "tests.mocks.backend_server"
    ])

    try:
        time.sleep(2)  # Wait for server to start

        # Start trainer
        trainer_proc = subprocess.Popen(
            ["python", "trainers/ultralytics/train.py"],
            env={
                "JOB_ID": "test-job",
                "BACKEND_BASE_URL": "http://localhost:9999",
                "CALLBACK_TOKEN": "test-token",
                "EPOCHS": "1",
                # ... other env vars
            }
        )

        # Wait for trainer to send heartbeat
        for _ in range(30):  # Wait up to 30 seconds
            response = requests.get("http://localhost:9999/received_callbacks")
            if response.json()["count"] > 0:
                break
            time.sleep(1)

        # Verify callbacks were received
        response = requests.get("http://localhost:9999/received_callbacks")
        callbacks = response.json()["callbacks"]

        assert len(callbacks) > 0, "No callbacks received from trainer"
        assert callbacks[0]["endpoint"] == "/api/v1/jobs/test-job/heartbeat"

    finally:
        backend_proc.terminate()
        trainer_proc.terminate()
```

### Test 3: No Shared File System

```python
# tests/isolation/test_no_shared_fs.py
import os
import subprocess
import tempfile
from pathlib import Path

def test_trainer_uses_s3_not_filesystem():
    """Verify trainer downloads dataset from S3, not local FS"""

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run trainer WITHOUT creating local dataset
        # If it tries to read from local FS, it will fail

        trainer_proc = subprocess.run(
            ["python", "trainers/ultralytics/train.py"],
            env={
                "JOB_ID": "test-job",
                "DATASET_ID": "test-dataset",
                "STORAGE_TYPE": "minio",
                "S3_ENDPOINT": "http://localhost:9000",
                # ... other S3 env vars
            },
            cwd=tmpdir,  # Run in empty temp dir
            capture_output=True,
            timeout=60
        )

        # Trainer should NOT try to access local files
        # Should only access S3
        assert "FileNotFoundError" not in trainer_proc.stderr.decode()
        assert "No such file or directory" not in trainer_proc.stderr.decode()
```

### Test 4: Environment Variables Only

```python
# tests/isolation/test_env_vars_only.py
import os
import subprocess

def test_trainer_gets_config_from_env():
    """Verify trainer reads configuration from environment variables"""

    # Run trainer with minimal env vars
    required_env = {
        "JOB_ID": "test-job",
        "MODEL_NAME": "yolo11n",
        "DATASET_ID": "test-dataset",
        "EPOCHS": "1",
        "BACKEND_BASE_URL": "http://localhost:8000",
        "CALLBACK_TOKEN": "test-token",
    }

    # Should NOT require any config files
    trainer_proc = subprocess.run(
        ["python", "trainers/ultralytics/train.py"],
        env=required_env,
        capture_output=True,
        timeout=10
    )

    # Should start successfully with only env vars
    assert trainer_proc.returncode == 0 or "Missing required" in trainer_proc.stderr.decode()
    assert "config.yaml not found" not in trainer_proc.stderr.decode()
    assert "Cannot read config file" not in trainer_proc.stderr.decode()
```

## Benefits of Complete Isolation

### 1. Framework Flexibility

Can implement trainers in any language:
```
trainers/
├── ultralytics/   (Python)
├── timm/          (Python)
├── rust-trainer/  (Rust - faster training!)
├── cpp-trainer/   (C++ - CUDA optimizations)
└── go-trainer/    (Go - easier concurrency)
```

All work with same backend, as long as they follow HTTP callback contract.

### 2. Independent Scaling

```
Backend: 3 replicas (handle 1000 req/s)
Trainers: 100 pods (one per training job)

Backend scaled based on API traffic
Trainers scaled based on training workload
Completely independent!
```

### 3. Security

```
Backend: Full database access, secrets
Trainer: ONLY environment variables + S3

Trainer compromised?
→ Can't access database
→ Can't access other users' data
→ Limited blast radius
```

### 4. Testing

```
Backend tests: Mock trainer HTTP callbacks
Trainer tests: Mock backend HTTP server

No need to run both together!
Each can be tested in complete isolation.
```

## Summary

**The Golden Rule**: Pretend backend and trainer are different companies with a public API contract.

✅ **Do**:
- Communicate via HTTP APIs
- Exchange data via S3 storage
- Pass configuration via environment variables
- Duplicate code if needed to maintain isolation

❌ **Don't**:
- Share Python packages
- Access database directly
- Share file system
- Import code from each other

**Remember**: Isolation is not just a nice-to-have, it's the **foundation** of the entire platform architecture. Never compromise it for convenience.

## References

- [Architecture Overview](./OVERVIEW.md)
- [Backend Design](./BACKEND_DESIGN.md)
- [Trainer Design](./TRAINER_DESIGN.md)
- [3-Tier Development](../development/3_TIER_DEVELOPMENT.md)
