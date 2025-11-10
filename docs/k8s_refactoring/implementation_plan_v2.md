# Trainer 구현 계획 v2 (API Contract 반영)

## 전체 타임라인 (12-18시간, 8 Phase)

| Phase | 작업 | 예상 시간 | 핵심 목표 |
|-------|------|----------|----------|
| Phase 1 | Backend API Contract 구현 | 2-3h | Callback 엔드포인트 (heartbeat, event, done) |
| Phase 2 | JWT 인증 구현 | 1-2h | 콜백 토큰 발급/검증 |
| Phase 3 | 상태머신 + Trace ID | 1-2h | DB 스키마 확장, 상태 관리 |
| Phase 4 | trainer-ultralytics utils.py | 1-2h | Storage, Callback, Env 유틸 |
| Phase 5 | trainer-ultralytics train.py | 3-4h | 학습 메인 + 새 API 사용 |
| Phase 6 | Backend 연동 + K8s Job | 2-3h | training_manager_k8s 수정 |
| Phase 7 | 통합 테스트 | 1-2h | End-to-end 검증 |
| Phase 8 | 문서화 + 정리 | 1h | PLUGIN_GUIDE 업데이트 |

**Total: 12-18시간**

---

## Phase 1: Backend API Contract 구현 (2-3시간)

### 목표
- 새 Callback API 엔드포인트 3개 구현
- 기존 `/internal/training/` 엔드포인트 마이그레이션

### 작업

#### 1. 새 API 라우터 생성

```bash
touch mvp/backend/app/api/v1_jobs.py
```

```python
# mvp/backend/app/api/v1_jobs.py
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.db.database import get_db
from app.db.models import TrainingJob, JobEvent
from app.auth import verify_callback_token

router = APIRouter(prefix="/v1/jobs", tags=["jobs-v1"])


# ===== Schemas =====
class HeartbeatRequest(BaseModel):
    trace_id: str
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    progress: Optional[float] = None  # 0-100
    metrics: Dict[str, Any] = {}


class EventRequest(BaseModel):
    trace_id: str
    level: str  # info | warning | error
    event_type: str
    message: str
    data: Optional[Dict[str, Any]] = None


class DoneRequest(BaseModel):
    trace_id: str
    status: str  # succeeded | failed
    error: Optional[str] = None
    final_metrics: Dict[str, Any] = {}
    artifacts: Dict[str, str] = {}
    mlflow_run_id: Optional[str] = None
    training_time_seconds: Optional[int] = None


# ===== Endpoints =====
@router.post("/{job_id}/heartbeat")
async def heartbeat(
    job_id: int,
    req: HeartbeatRequest,
    db: Session = Depends(get_db),
    token: dict = Depends(verify_callback_token)
):
    """Receive heartbeat from trainer."""
    # 토큰 job_id 검증
    if token["job_id"] != job_id:
        raise HTTPException(401, "Token job_id mismatch")

    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    # Trace ID 검증
    if job.trace_id != req.trace_id:
        raise HTTPException(400, "Trace ID mismatch")

    # 진행률 업데이트
    job.current_epoch = req.epoch
    job.progress = req.progress
    job.last_heartbeat_at = datetime.utcnow()

    # Metrics 저장 (기존 training_metrics 테이블 재사용)
    if req.metrics:
        metric = TrainingMetric(
            job_id=job_id,
            epoch=req.epoch or 0,
            step=req.step or 0,
            extra_metrics=req.metrics
        )
        db.add(metric)

    db.commit()

    # 중단 요청 확인
    should_continue = job.status != "cancelled"

    return {"status": "ok", "should_continue": should_continue}


@router.post("/{job_id}/event")
async def event(
    job_id: int,
    req: EventRequest,
    db: Session = Depends(get_db),
    token: dict = Depends(verify_callback_token)
):
    """Receive event from trainer."""
    if token["job_id"] != job_id:
        raise HTTPException(401, "Token job_id mismatch")

    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    # 이벤트 저장
    event_obj = JobEvent(
        job_id=job_id,
        trace_id=req.trace_id,
        level=req.level,
        event_type=req.event_type,
        message=req.message,
        data=req.data
    )
    db.add(event_obj)
    db.commit()

    # WebSocket으로 Frontend에 실시간 중계 (나중에 구현)
    # await websocket_manager.broadcast(job_id, event_obj.dict())

    return {"status": "ok"}


@router.post("/{job_id}/done")
async def done(
    job_id: int,
    req: DoneRequest,
    db: Session = Depends(get_db),
    token: dict = Depends(verify_callback_token)
):
    """Receive final result from trainer."""
    if token["job_id"] != job_id:
        raise HTTPException(401, "Token job_id mismatch")

    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    # 최종 상태 업데이트
    job.status = req.status
    job.error_message = req.error
    job.completed_at = datetime.utcnow()
    job.mlflow_run_id = req.mlflow_run_id
    job.training_time_seconds = req.training_time_seconds

    # Artifacts 저장 (JSON)
    job.artifacts = req.artifacts

    # Final metrics 저장
    if req.final_metrics:
        final_metric = TrainingMetric(
            job_id=job_id,
            epoch=-1,  # -1 = final
            extra_metrics=req.final_metrics
        )
        db.add(final_metric)

    db.commit()

    return {"status": "ok"}
```

#### 2. DB 스키마 확장

```python
# mvp/backend/app/db/models.py 추가

class JobEvent(Base):
    """학습 중 발생한 이벤트"""
    __tablename__ = "job_events"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"))
    trace_id = Column(String(255), index=True)
    level = Column(String(20))  # info, warning, error
    event_type = Column(String(50))  # checkpoint_saved, validation_complete, ...
    message = Column(Text)
    data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="events")


class TrainingJob(Base):
    # 기존 필드 +
    trace_id = Column(String(255), unique=True, nullable=False)
    current_epoch = Column(Integer, default=0)
    progress = Column(Float, default=0.0)  # 0-100
    last_heartbeat_at = Column(DateTime, nullable=True)
    training_time_seconds = Column(Integer, nullable=True)
    artifacts = Column(JSON, nullable=True)  # {"best": "s3://...", "last": "s3://..."}

    events = relationship("JobEvent", back_populates="job", cascade="all, delete-orphan")
```

#### 3. Migration 스크립트

```bash
cd mvp/backend
alembic revision -m "Add trace_id, events, heartbeat fields"
```

### 완료 조건
- [ ] 3개 엔드포인트 구현 (heartbeat, event, done)
- [ ] DB 스키마 확장 (JobEvent, trace_id, ...)
- [ ] Migration 적용
- [ ] Postman/curl 테스트 통과

---

## Phase 2: JWT 인증 구현 (1-2시간)

### 목표
- Job 생성 시 JWT 토큰 발급
- Callback API에서 토큰 검증

### 작업

#### 1. JWT 유틸리티

```python
# mvp/backend/app/auth.py
import os
import jwt
from datetime import datetime, timedelta
from fastapi import Header, HTTPException

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"


def create_callback_token(job_id: int, expires_hours: int = 6) -> str:
    """Create JWT token for job callback."""
    payload = {
        "job_id": job_id,
        "scope": "training-callback",
        "exp": datetime.utcnow() + timedelta(hours=expires_hours),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_callback_token(authorization: str = Header(...)) -> dict:
    """Verify callback JWT token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header")

    token = authorization[7:]  # Remove "Bearer "

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Scope 검증
        if payload.get("scope") != "training-callback":
            raise HTTPException(403, "Invalid token scope")

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid token: {e}")
```

#### 2. Job 생성 시 토큰 발급

```python
# mvp/backend/app/api/training.py
from app.auth import create_callback_token

@router.post("/jobs")
async def create_training_job(...):
    # Job 생성
    job = TrainingJob(
        trace_id=str(uuid.uuid4()),
        ...
    )
    db.add(job)
    db.commit()

    # JWT 토큰 발급
    callback_token = create_callback_token(job.id, expires_hours=24)

    # Trainer 환경변수로 전달 (나중에 training_manager에서 사용)
    return {
        "job_id": job.id,
        "callback_token": callback_token,  # Frontend는 모르고, Backend만 사용
        ...
    }
```

### 완료 조건
- [ ] JWT 발급/검증 함수 구현
- [ ] Callback API에 인증 적용
- [ ] 토큰 만료 테스트
- [ ] job_id mismatch 테스트

---

## Phase 3: 상태머신 + Trace ID (1-2시간)

### 목표
- 상태 전이 규칙 명확화
- Trace ID 추가

### 작업

#### 1. 상태 Enum

```python
# mvp/backend/app/db/models.py
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"      # DB 생성됨
    QUEUED = "queued"        # K8s Job 제출됨
    RUNNING = "running"      # 학습 중
    SUCCEEDED = "succeeded"  # 완료
    FAILED = "failed"        # 실패
    CANCELLED = "cancelled"  # 취소


class TrainingJob(Base):
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    queued_at = Column(DateTime, nullable=True)
    # ...
```

#### 2. 상태 전이 함수

```python
# mvp/backend/app/services/job_state_machine.py
def transition_status(job: TrainingJob, new_status: JobStatus) -> bool:
    """Validate and transition job status."""
    VALID_TRANSITIONS = {
        JobStatus.PENDING: [JobStatus.QUEUED, JobStatus.FAILED, JobStatus.CANCELLED],
        JobStatus.QUEUED: [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED],
        JobStatus.RUNNING: [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED],
    }

    current = job.status
    if new_status not in VALID_TRANSITIONS.get(current, []):
        raise ValueError(f"Invalid transition: {current} → {new_status}")

    job.status = new_status

    # 타임스탬프 업데이트
    if new_status == JobStatus.QUEUED:
        job.queued_at = datetime.utcnow()
    elif new_status == JobStatus.RUNNING:
        job.started_at = datetime.utcnow()
    elif new_status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
        job.completed_at = datetime.utcnow()

    return True
```

### 완료 조건
- [ ] JobStatus Enum 정의
- [ ] transition_status() 함수 구현
- [ ] 잘못된 전이 테스트

---

## Phase 4: trainer-ultralytics utils.py (1-2시간)

### 목표
- Storage, Callback, Env 유틸리티 구현

### 작업

```bash
mkdir -p mvp/trainer-ultralytics
cd mvp/trainer-ultralytics
touch utils.py
```

```python
# mvp/trainer-ultralytics/utils.py (~150줄)
import os
import boto3
import requests
from typing import Optional, Dict, Any

# ===== Storage =====
def get_storage_client():
    """Get S3 client (MinIO or R2)."""
    storage_type = os.getenv("STORAGE_TYPE", "minio")

    if storage_type == "minio":
        return boto3.client(
            's3',
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin")
        )
    else:  # r2
        return boto3.client(
            's3',
            endpoint_url=os.getenv("R2_ENDPOINT"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY")
        )


def download_dataset(dataset_id: str, output_dir: str = "/tmp/dataset") -> str:
    """Download and extract dataset."""
    import zipfile
    from pathlib import Path

    s3 = get_storage_client()
    bucket = os.getenv("STORAGE_BUCKET")
    zip_path = f"{output_dir}/dataset.zip"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[STORAGE] Downloading s3://{bucket}/datasets/{dataset_id}.zip")
    s3.download_file(bucket, f"datasets/{dataset_id}.zip", zip_path)

    print(f"[STORAGE] Extracting to {output_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)
    return output_dir


def upload_checkpoint(local_path: str, remote_path: str) -> str:
    """Upload checkpoint to storage."""
    s3 = get_storage_client()
    bucket = os.getenv("STORAGE_BUCKET")

    print(f"[STORAGE] Uploading {local_path} → s3://{bucket}/{remote_path}")
    s3.upload_file(local_path, bucket, remote_path)
    return f"s3://{bucket}/{remote_path}"


# ===== Callback API =====
def _get_callback_headers() -> Dict[str, str]:
    """Get headers for callback requests."""
    return {
        "Authorization": f"Bearer {os.getenv('CALLBACK_TOKEN')}",
        "Content-Type": "application/json"
    }


def _callback_url(endpoint: str) -> str:
    """Build callback URL."""
    base = os.getenv("BACKEND_BASE_URL")
    job_id = os.getenv("JOB_ID")
    return f"{base}/v1/jobs/{job_id}/{endpoint}"


def send_heartbeat(
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
    progress: Optional[float] = None,
    metrics: Dict[str, Any] = None
) -> bool:
    """Send heartbeat to backend."""
    try:
        response = requests.post(
            _callback_url("heartbeat"),
            headers=_get_callback_headers(),
            json={
                "trace_id": os.getenv("TRACE_ID"),
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "total_steps": total_steps,
                "progress": progress,
                "metrics": metrics or {}
            },
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return data.get("should_continue", True)
    except Exception as e:
        print(f"[WARN] Heartbeat failed: {e}")
        return True  # Continue on error


def send_event(
    event_type: str,
    message: str,
    level: str = "info",
    data: Optional[Dict[str, Any]] = None
):
    """Send event to backend."""
    try:
        requests.post(
            _callback_url("event"),
            headers=_get_callback_headers(),
            json={
                "trace_id": os.getenv("TRACE_ID"),
                "level": level,
                "event_type": event_type,
                "message": message,
                "data": data
            },
            timeout=5
        )
    except Exception as e:
        print(f"[WARN] Event failed: {e}")


def send_done(
    status: str,
    error: Optional[str] = None,
    final_metrics: Dict[str, Any] = None,
    artifacts: Dict[str, str] = None,
    mlflow_run_id: Optional[str] = None,
    training_time_seconds: Optional[int] = None
):
    """Send final result to backend."""
    try:
        requests.post(
            _callback_url("done"),
            headers=_get_callback_headers(),
            json={
                "trace_id": os.getenv("TRACE_ID"),
                "status": status,
                "error": error,
                "final_metrics": final_metrics or {},
                "artifacts": artifacts or {},
                "mlflow_run_id": mlflow_run_id,
                "training_time_seconds": training_time_seconds
            },
            timeout=5
        )
    except Exception as e:
        print(f"[ERROR] Done callback failed: {e}")


# ===== Environment =====
def check_required_env(*keys):
    """Check required environment variables."""
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")


def get_env_int(key: str, default: Optional[int] = None, required: bool = False) -> int:
    """Get integer from env."""
    value = os.getenv(key)
    if value is None:
        if required:
            raise ValueError(f"Required env var {key} not set")
        return default
    return int(value)


def get_env_float(key: str, default: Optional[float] = None, required: bool = False) -> float:
    """Get float from env."""
    value = os.getenv(key)
    if value is None:
        if required:
            raise ValueError(f"Required env var {key} not set")
        return default
    return float(value)
```

### 완료 조건
- [ ] utils.py 구현 완료
- [ ] `python -m py_compile utils.py` 통과
- [ ] 단위 테스트 (Storage, Callback)

---

## Phase 5: trainer-ultralytics train.py (3-4시간)

### 목표
- 새 API Contract 사용
- Heartbeat, Event, Done 호출

### 구현 (핵심만)

```python
# mvp/trainer-ultralytics/train.py
import os
import sys
import signal
import time
from pathlib import Path
from ultralytics import YOLO
import mlflow

from utils import (
    check_required_env,
    get_env_int,
    get_env_float,
    download_dataset,
    upload_checkpoint,
    send_heartbeat,
    send_event,
    send_done
)

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n[SIGNAL] Received {signum}, shutting down...")
    send_done(status="failed", error="Interrupted by signal")
    sys.exit(1)

def main():
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()

    print("="*80)
    print("YOLO Training Service")
    print("="*80)

    # 환경변수 검증
    check_required_env(
        "JOB_ID", "TRACE_ID", "BACKEND_BASE_URL", "CALLBACK_TOKEN",
        "DATASET_ID", "MODEL_NAME", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE"
    )

    job_id = os.getenv("JOB_ID")
    trace_id = os.getenv("TRACE_ID")
    dataset_id = os.getenv("DATASET_ID")
    model_name = os.getenv("MODEL_NAME")
    epochs = get_env_int("EPOCHS")
    batch_size = get_env_int("BATCH_SIZE")
    learning_rate = get_env_float("LEARNING_RATE")

    print(f"[CONFIG] Job ID: {job_id}")
    print(f"[CONFIG] Trace ID: {trace_id}")
    print(f"[CONFIG] Model: {model_name}, Epochs: {epochs}")

    try:
        # Step 1: Dataset 다운로드
        print("\n[1/4] Downloading dataset...")
        send_event("dataset_download_started", "Downloading dataset from storage")
        dataset_dir = download_dataset(dataset_id)
        send_event("dataset_loaded", f"Dataset loaded: {dataset_dir}", data={"path": dataset_dir})

        # Step 2: MLflow 초기화
        print("\n[2/4] Initializing MLflow...")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(f"project-{os.getenv('PROJECT_ID', 'default')}")
        run = mlflow.start_run(run_name=f"job-{job_id}")
        mlflow_run_id = run.info.run_id
        send_event("mlflow_initialized", f"MLflow run: {mlflow_run_id}", data={"run_id": mlflow_run_id})

        mlflow.log_params({
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        })

        # Step 3: 모델 초기화
        print("\n[3/4] Initializing model...")
        model = YOLO(f"{model_name}.pt")
        send_event("model_initialized", f"Model loaded: {model_name}")

        # Step 4: 학습 (YOLO callback 등록)
        print("\n[4/4] Starting training...")
        print("-" * 80)

        def on_train_epoch_end(trainer):
            """Called after each epoch."""
            epoch = trainer.epoch + 1  # 1-indexed
            metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}

            # Heartbeat
            progress = (epoch / epochs) * 100
            should_continue = send_heartbeat(
                epoch=epoch,
                total_epochs=epochs,
                progress=progress,
                metrics={
                    "loss": float(metrics.get("train/loss", 0)),
                    "lr": float(trainer.optimizer.param_groups[0]['lr'])
                }
            )

            # Backend에서 중단 요청 시
            if not should_continue:
                print("[WARN] Backend requested cancellation")
                raise KeyboardInterrupt("Cancelled by backend")

            # MLflow 로깅
            mlflow.log_metrics({
                "train_loss": float(metrics.get("train/loss", 0)),
            }, step=epoch)

        # Callback 등록
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # 학습 시작
        results = model.train(
            data=f"{dataset_dir}/data.yaml",
            epochs=epochs,
            batch=batch_size,
            lr0=learning_rate,
            imgsz=640,
            project="/tmp/runs",
            name=f"job-{job_id}",
            save=True,
            save_period=5,  # 5 epoch마다 저장
            val=True,
        )

        print("-" * 80)
        print("[4/4] Training completed")

        # Checkpoint 업로드
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        best_url = upload_checkpoint(str(best_pt), f"checkpoints/job-{job_id}/best.pt")
        send_event("checkpoint_saved", f"Best checkpoint uploaded", data={"path": best_url})

        last_pt = Path(results.save_dir) / "weights" / "last.pt"
        last_url = upload_checkpoint(str(last_pt), f"checkpoints/job-{job_id}/last.pt")

        # 최종 완료
        training_time = int(time.time() - start_time)
        send_done(
            status="succeeded",
            final_metrics={
                "best_mAP50": float(results.maps[0]) if results.maps else 0,
            },
            artifacts={
                "best_checkpoint": best_url,
                "last_checkpoint": last_url,
            },
            mlflow_run_id=mlflow_run_id,
            training_time_seconds=training_time
        )

        print(f"\n[SUCCESS] Training completed in {training_time}s")

    except KeyboardInterrupt:
        print("\n[CANCELLED] Training cancelled")
        send_done(status="failed", error="Cancelled by user or backend")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()

        send_done(status="failed", error=str(e))
        sys.exit(1)

    finally:
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    main()
```

### 완료 조건
- [ ] train.py 구현 완료
- [ ] 로컬 테스트 (환경변수 설정)
- [ ] Heartbeat 5회 이상 호출
- [ ] Event 3회 이상 호출 (dataset_loaded, checkpoint_saved, ...)
- [ ] Done 1회 호출

---

## Phase 6: Backend 연동 + K8s Job (2-3시간)

### 목표
- training_manager_k8s 수정
- JWT 토큰 주입
- K8s Job 템플릿

### 작업

```python
# mvp/backend/app/utils/training_manager_k8s.py
from app.auth import create_callback_token

def _start_training_subprocess(self, job_id):
    job = self.db.query(TrainingJob).filter_by(id=job_id).first()

    # Trainer 경로
    trainer_dir = project_root / f"trainer-{job.framework}"
    train_script = trainer_dir / "train.py"

    # JWT 토큰 발급
    callback_token = create_callback_token(job.id, expires_hours=24)

    # 환경변수 빌드
    env = self._build_env_vars(job, callback_token)

    # 실행
    cmd = ["python", str(train_script)]
    process = subprocess.Popen(cmd, env=env, ...)

    # QUEUED 상태로 전이
    transition_status(job, JobStatus.QUEUED)
    job.process_id = process.pid
    self.db.commit()

def _build_env_vars(self, job, callback_token):
    env = os.environ.copy()

    env.update({
        "JOB_ID": str(job.id),
        "TRACE_ID": job.trace_id,
        "BACKEND_BASE_URL": os.getenv("BACKEND_INTERNAL_URL", "http://localhost:8000"),
        "CALLBACK_TOKEN": callback_token,
        "DATASET_ID": job.dataset_id,
        "MODEL_NAME": job.model_name,
        "EPOCHS": str(job.epochs),
        ...
    })

    return env
```

### 완료 조건
- [ ] training_manager_k8s 수정
- [ ] 토큰 주입 확인
- [ ] 상태 전이 확인 (PENDING → QUEUED → RUNNING)

---

## Phase 7: 통합 테스트 (1-2시간)

### 테스트 시나리오

1. **정상 완료**
   - Frontend에서 학습 시작
   - Heartbeat 수신 확인
   - Event 수신 확인
   - Done 수신 확인
   - 상태: SUCCEEDED

2. **중간 취소**
   - 학습 중 Cancel 버튼
   - Trainer가 should_continue=false 받음
   - 상태: CANCELLED

3. **에러 발생**
   - 잘못된 DATASET_ID
   - Done(status=failed) 호출
   - 상태: FAILED

4. **토큰 만료**
   - 24시간 후 heartbeat 호출
   - 401 Unauthorized

### 완료 조건
- [ ] 4가지 시나리오 통과
- [ ] Frontend 실시간 업데이트 확인
- [ ] Loki 로그 수집 확인

---

## Phase 8: 문서화 (1시간)

- [ ] PLUGIN_GUIDE.md 최종 검토
- [ ] README.md 업데이트
- [ ] 예제 코드 추가

---

## 총괄 체크리스트

### Backend
- [ ] 3개 Callback API 구현
- [ ] JWT 발급/검증
- [ ] 상태머신
- [ ] DB Migration

### Trainer
- [ ] utils.py
- [ ] train.py
- [ ] Heartbeat, Event, Done 호출
- [ ] MLflow 통합

### 통합
- [ ] End-to-end 테스트 4가지
- [ ] 문서화 완료
