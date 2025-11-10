# Trainer Plugin 개발 가이드

## 개요

새로운 프레임워크(framework)의 Trainer를 추가하려면 다음 **API Contract**만 준수하면 됩니다.

**내부 구현은 자유:**
- utils.py 복사 ✅
- 직접 requests/boto3 호출 ✅
- 자체 라이브러리 사용 ✅

---

## 필수 API Contract

### 1. 입력: 환경변수

Trainer는 다음 환경변수를 읽어야 합니다:

```bash
# ===== Job 식별 =====
JOB_ID=123                              # 필수: Training job ID
TRACE_ID=abc-def-ghi                    # 필수: 분산 추적 ID
PROJECT_ID=proj-456                     # 선택: 프로젝트 ID

# ===== Backend 통신 =====
BACKEND_BASE_URL=https://api.example.com  # 필수: Backend API URL
CALLBACK_TOKEN=eyJhbGc...                # 필수: JWT 콜백 인증 토큰

# ===== Training 설정 =====
TASK_TYPE=object_detection              # 필수: 작업 타입
MODEL_NAME=yolo11n                      # 필수: 모델 이름
DATASET_ID=dataset-uuid                 # 필수: 데이터셋 ID
EPOCHS=10                               # 필수: Epoch 수
BATCH_SIZE=16                           # 필수: Batch size
LEARNING_RATE=0.001                     # 필수: Learning rate
IMAGE_SIZE=640                          # 선택: 이미지 크기

# ===== Storage (R2/MinIO) =====
STORAGE_TYPE=r2                         # 필수: minio | r2
STORAGE_BUCKET=vision-platform          # 필수: Bucket 이름
R2_ENDPOINT=https://xxx.r2.cloudflarestorage.com  # R2 엔드포인트
R2_ACCESS_KEY_ID=xxx                    # R2 액세스 키
R2_SECRET_ACCESS_KEY=xxx                # R2 시크릿 키

# ===== MLflow =====
MLFLOW_TRACKING_URI=https://mlflow.example.com  # 선택: MLflow 서버
```

---

### 2. 출력: HTTP Callback API

Trainer는 다음 엔드포인트를 호출해야 합니다:

#### A. Heartbeat (Epoch마다 또는 5-10초 간격)

**목적:** 진행률 업데이트, Frontend 실시간 표시

```http
POST {BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/heartbeat
Authorization: Bearer {CALLBACK_TOKEN}
Content-Type: application/json

{
  "trace_id": "abc-def-ghi",
  "epoch": 5,
  "total_epochs": 10,
  "step": 1250,
  "total_steps": 2500,
  "progress": 50.0,
  "metrics": {
    "loss": 0.234,
    "learning_rate": 0.0001,
    "gpu_mem_mb": 8192
  }
}
```

**응답:**
```json
{
  "status": "ok",
  "should_continue": true  # false면 학습 중단
}
```

---

#### B. Event (중요 이벤트 발생 시)

**목적:** 체크포인트 저장, 에러, 경고 등

```http
POST {BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/event
Authorization: Bearer {CALLBACK_TOKEN}
Content-Type: application/json

{
  "trace_id": "abc-def-ghi",
  "level": "info",  # info | warning | error
  "event_type": "checkpoint_saved",
  "message": "Checkpoint saved at epoch 10",
  "data": {
    "checkpoint_path": "s3://bucket/checkpoints/job-123/epoch-10.pt",
    "file_size_mb": 245
  }
}
```

**이벤트 타입 예시:**
- `checkpoint_saved` - 체크포인트 저장됨
- `validation_complete` - Validation 완료
- `dataset_loaded` - 데이터셋 로드 완료
- `model_initialized` - 모델 초기화 완료
- `error` - 에러 발생 (복구 가능)

---

#### C. Done (학습 완료 시, 1회)

**목적:** 최종 결과 전달

```http
POST {BACKEND_BASE_URL}/v1/jobs/{JOB_ID}/done
Authorization: Bearer {CALLBACK_TOKEN}
Content-Type: application/json

{
  "trace_id": "abc-def-ghi",
  "status": "succeeded",  # succeeded | failed
  "error": null,  # 실패 시 에러 메시지
  "final_metrics": {
    "best_mAP50": 0.856,
    "best_mAP50-95": 0.723,
    "final_loss": 0.123
  },
  "artifacts": {
    "best_checkpoint": "s3://bucket/checkpoints/job-123/best.pt",
    "last_checkpoint": "s3://bucket/checkpoints/job-123/last.pt",
    "tensorboard_logs": "s3://bucket/logs/job-123/"
  },
  "mlflow_run_id": "abc123",
  "training_time_seconds": 3600
}
```

---

### 3. 출력: Storage (S3 API)

#### Dataset 다운로드

```python
import boto3

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv("R2_ENDPOINT"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY")
)

bucket = os.getenv("STORAGE_BUCKET")
dataset_id = os.getenv("DATASET_ID")

# 다운로드
s3.download_file(bucket, f"datasets/{dataset_id}.zip", "/tmp/dataset.zip")

# 압축 해제
import zipfile
with zipfile.ZipFile("/tmp/dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("/tmp/dataset")
```

#### Checkpoint 업로드

```python
# 체크포인트 저장
s3.upload_file(
    "/tmp/runs/weights/best.pt",
    bucket,
    f"checkpoints/job-{job_id}/best.pt"
)

# Event로 알림
requests.post(
    f"{backend_url}/v1/jobs/{job_id}/event",
    headers={"Authorization": f"Bearer {callback_token}"},
    json={
        "event_type": "checkpoint_saved",
        "data": {"checkpoint_path": f"s3://{bucket}/checkpoints/job-{job_id}/best.pt"}
    }
)
```

---

### 4. 출력: stdout 로그

모든 진행 상황은 stdout으로 출력해야 합니다 (Loki 수집용):

```python
print(f"[TRACE:{trace_id}] [EPOCH:{epoch}] Training started")
print(f"[TRACE:{trace_id}] [STEP:{step}] Loss: {loss:.4f}")
print(f"[TRACE:{trace_id}] [INFO] Checkpoint saved: {path}")
```

**로그 포맷 권장:**
```
[TRACE:{trace_id}] [LEVEL] Message
```

---

### 5. 출력: MLflow (선택)

```python
import mlflow

# 초기화
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(f"project-{project_id}")

# Run 시작
with mlflow.start_run(run_name=f"job-{job_id}"):
    # 파라미터 로깅
    mlflow.log_params({
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size
    })

    # 메트릭 로깅 (epoch마다)
    mlflow.log_metrics({
        "train_loss": loss,
        "val_mAP50": map50
    }, step=epoch)

    # Artifact 로깅
    mlflow.log_artifact("/tmp/runs/weights/best.pt")

    # Run ID 저장 (done 콜백에 포함)
    run_id = mlflow.active_run().info.run_id
```

---

## 구현 예시

### 옵션 A: 기존 utils.py 재사용 (권장)

```bash
cp trainer-ultralytics/utils.py trainer-myframework/utils.py
```

```python
# trainer-myframework/train.py
from utils import (
    check_required_env,
    get_env_int,
    download_dataset,
    upload_checkpoint,
    send_heartbeat,
    send_event,
    send_done
)

def main():
    # 환경변수 검증
    check_required_env("JOB_ID", "DATASET_ID", "CALLBACK_TOKEN")

    job_id = os.getenv("JOB_ID")
    trace_id = os.getenv("TRACE_ID")
    epochs = get_env_int("EPOCHS")

    # 데이터셋 다운로드
    dataset_dir = download_dataset(os.getenv("DATASET_ID"))

    # 학습
    for epoch in range(epochs):
        # ... training ...

        # Heartbeat
        send_heartbeat(epoch, total_epochs, metrics={...})

        # Checkpoint 저장 & 이벤트
        if epoch % 5 == 0:
            checkpoint_path = upload_checkpoint(...)
            send_event("checkpoint_saved", data={"path": checkpoint_path})

    # 완료
    send_done(status="succeeded", final_metrics={...})
```

---

### 옵션 B: 직접 구현

```python
# trainer-myframework/train.py
import os
import requests
import boto3

def send_heartbeat(epoch, metrics):
    """Heartbeat 전송"""
    requests.post(
        f"{os.getenv('BACKEND_BASE_URL')}/v1/jobs/{os.getenv('JOB_ID')}/heartbeat",
        headers={"Authorization": f"Bearer {os.getenv('CALLBACK_TOKEN')}"},
        json={
            "trace_id": os.getenv("TRACE_ID"),
            "epoch": epoch,
            "metrics": metrics
        },
        timeout=5
    )

def main():
    job_id = os.getenv("JOB_ID")

    # S3 클라이언트
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv("R2_ENDPOINT"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY")
    )

    # 데이터셋 다운로드
    s3.download_file(
        os.getenv("STORAGE_BUCKET"),
        f"datasets/{os.getenv('DATASET_ID')}.zip",
        "/tmp/dataset.zip"
    )

    # 학습
    for epoch in range(int(os.getenv("EPOCHS"))):
        # ... training ...
        send_heartbeat(epoch, {"loss": 0.5})
```

---

## 테스트

### 로컬 테스트

```bash
# 환경변수 설정
export JOB_ID=999
export TRACE_ID=test-trace-123
export BACKEND_BASE_URL=http://localhost:8000
export CALLBACK_TOKEN=test-token
export DATASET_ID=test-dataset
export STORAGE_TYPE=minio
export MINIO_ENDPOINT=http://localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export STORAGE_BUCKET=vision-platform

# 실행
python train.py

# Backend 로그 확인
# → POST /v1/jobs/999/heartbeat 200 OK
# → POST /v1/jobs/999/event 200 OK
# → POST /v1/jobs/999/done 200 OK
```

---

## 체크리스트

새 Trainer를 추가할 때 확인:

- [ ] 모든 필수 환경변수 읽기
- [ ] Heartbeat API 호출 (Epoch마다 또는 5-10초)
- [ ] Event API 호출 (중요 이벤트)
- [ ] Done API 호출 (1회, 완료 시)
- [ ] Dataset S3에서 다운로드
- [ ] Checkpoint S3에 업로드
- [ ] stdout 로그 출력 (TRACE ID 포함)
- [ ] MLflow 로깅 (선택)
- [ ] JWT 토큰으로 인증
- [ ] 에러 시 status="failed"로 done 호출

---

## FAQ

**Q: utils.py를 반드시 사용해야 하나요?**

A: 아니요. utils.py는 참고 구현일 뿐입니다. API Contract만 지키면 어떻게 구현하든 상관없습니다.

**Q: Python이 아닌 다른 언어로 작성할 수 있나요?**

A: 네. Go, Rust, C++ 등 어떤 언어든 가능합니다. HTTP/S3 API만 호출하면 됩니다.

**Q: Heartbeat를 건너뛰면 어떻게 되나요?**

A: Backend가 타임아웃으로 인식하고 Job을 실패 처리할 수 있습니다 (권장: Epoch마다 1회).

**Q: MLflow는 필수인가요?**

A: 아니요. 선택 사항입니다. 하지만 실험 추적을 위해 권장합니다.

**Q: Callback API 인증이 실패하면?**

A: 401 Unauthorized 응답이 옵니다. CALLBACK_TOKEN이 올바른지 확인하세요.
