# Trainer 공통 기능 분석

## 모든 스크립트에서 사용되는 공통 기능

### 1. Storage 다운로드/업로드 (모든 스크립트)
```python
# Dataset 다운로드
download_from_storage(dataset_id) → local_path

# Checkpoint 다운로드
download_checkpoint(checkpoint_path) → local_path

# Checkpoint 업로드
upload_checkpoint(local_path, s3_path)

# Artifacts 업로드 (이미지, JSON 등)
upload_artifact(local_path, s3_path)
```

### 2. Backend Callback (train.py만)
```python
# 상태 업데이트
notify_status(status, error=None)

# Validation 결과 전송
send_validation_result(epoch, metrics, per_class_metrics, ...)

# Training metrics 전송
send_training_metrics(epoch, loss, accuracy, lr, ...)
```

### 3. MLflow 로깅 (train.py, evaluate.py)
```python
# 실험 시작
start_experiment(experiment_name, run_name)

# 메트릭 로깅
log_metrics(metrics_dict, step)

# 파라미터 로깅
log_params(params_dict)

# Artifact 업로드
log_artifact(file_path)
```

### 4. 환경변수 파싱 (모든 스크립트)
```python
# 필수 환경변수 체크
required_env_vars = ["JOB_ID", "CALLBACK_URL", ...]
check_required_env(required_env_vars)

# 환경변수 파싱 (타입 변환)
epochs = get_env_int("EPOCHS", default=10)
learning_rate = get_env_float("LEARNING_RATE", default=0.001)
```

---

## 공통 기능 구현 전략

### 옵션 A: 각 Trainer에 utils.py 복사 (현재 선호)

```
trainer-ultralytics/
├── train.py
├── predict.py
├── utils.py          # 공통 함수 (100줄 정도)
└── requirements.txt

trainer-timm/
├── train.py
├── predict.py
├── utils.py          # 동일한 코드 복사
└── requirements.txt
```

**장점:**
- ✅ 완전한 독립성
- ✅ 각 trainer가 자유롭게 수정 가능
- ✅ 배포 간단 (의존성 없음)
- ✅ 한 trainer 수정이 다른 것에 영향 없음

**단점:**
- ❌ 코드 중복 (~100줄)
- ❌ 공통 버그 수정 시 모든 trainer 수정 필요

**결론:** 100줄 정도는 중복해도 괜찮음. 완전 독립성이 더 중요.

---

### 옵션 B: Minimal Shared Package

```
mvp/
├── shared/
│   └── trainer_utils/        # Minimal package
│       ├── __init__.py
│       ├── storage.py        # 40줄
│       ├── callback.py       # 30줄
│       └── env.py            # 20줄
├── trainer-ultralytics/
│   ├── train.py
│   └── requirements.txt      # trainer-utils @ file://../../shared/trainer_utils
```

**장점:**
- ✅ 코드 중복 없음
- ✅ 공통 버그 수정 한 번에

**단점:**
- ❌ 상대 경로 의존성
- ❌ Docker 빌드 복잡 (COPY ../shared)
- ❌ 독립성 감소

**결론:** MVP에는 과함. 나중에 필요하면 고려.

---

### 옵션 C: Inline 구현 (공통 함수 없이)

각 스크립트에 필요한 기능을 직접 구현

```python
# train.py 내부에 직접 구현
def download_dataset():
    s3 = boto3.client('s3', endpoint_url=os.getenv("STORAGE_ENDPOINT"))
    s3.download_file(bucket, key, local_path)
    extract_zip(local_path)
```

**장점:**
- ✅ 의존성 제로
- ✅ 코드 파악 쉬움 (모든 코드가 한 파일에)

**단점:**
- ❌ 파일이 너무 길어짐 (500줄+)
- ❌ 가독성 떨어짐

**결론:** train.py는 메인 로직에 집중하고, utils는 분리하는 게 나음.

---

## 최종 결정: **옵션 A (각 trainer에 utils.py 복사)**

### utils.py 구조 (100줄 정도)

```python
# trainer-ultralytics/utils.py
import os
import boto3
import requests
from pathlib import Path

# ==================== Storage ====================
def get_storage_client():
    """Get S3-compatible storage client (MinIO or R2)."""
    storage_type = os.getenv("STORAGE_TYPE", "minio")

    if storage_type == "minio":
        return boto3.client(
            's3',
            endpoint_url=os.getenv("MINIO_ENDPOINT"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
        )
    else:  # r2
        return boto3.client(
            's3',
            endpoint_url=os.getenv("R2_ENDPOINT"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY")
        )

def download_dataset(dataset_id, output_dir="/tmp/dataset"):
    """Download and extract dataset from storage."""
    s3 = get_storage_client()
    bucket = os.getenv("STORAGE_BUCKET", "vision-platform")
    zip_path = f"{output_dir}/dataset.zip"

    # Download
    s3.download_file(bucket, f"datasets/{dataset_id}.zip", zip_path)

    # Extract
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)
    return output_dir

def upload_checkpoint(local_path, remote_path):
    """Upload checkpoint to storage."""
    s3 = get_storage_client()
    bucket = os.getenv("STORAGE_BUCKET", "vision-platform")
    s3.upload_file(local_path, bucket, remote_path)
    return f"s3://{bucket}/{remote_path}"

# ==================== Callback ====================
def get_callback_headers():
    """Get headers for Backend API calls."""
    return {
        "X-Internal-Auth": os.getenv("INTERNAL_AUTH_TOKEN"),
        "Content-Type": "application/json"
    }

def notify_status(status, error=None):
    """Notify Backend of status change."""
    callback_url = os.getenv("CALLBACK_URL")
    if not callback_url:
        return

    try:
        requests.patch(
            f"{callback_url}/status",
            json={"status": status, "error": error},
            headers=get_callback_headers(),
            timeout=5
        )
    except Exception as e:
        print(f"[WARN] Failed to notify status: {e}")

def send_validation_result(epoch, metrics):
    """Send validation result to Backend."""
    callback_url = os.getenv("CALLBACK_URL")
    if not callback_url:
        return

    try:
        requests.post(
            f"{callback_url}/validation-results",
            json={"epoch": epoch, "metrics": metrics, ...},
            headers=get_callback_headers(),
            timeout=5
        )
    except Exception as e:
        print(f"[WARN] Failed to send validation result: {e}")

# ==================== Environment ====================
def get_env_int(key, default=None, required=False):
    """Get integer from environment variable."""
    value = os.getenv(key)
    if value is None:
        if required:
            raise ValueError(f"Required env var {key} not set")
        return default
    return int(value)

def get_env_float(key, default=None, required=False):
    """Get float from environment variable."""
    value = os.getenv(key)
    if value is None:
        if required:
            raise ValueError(f"Required env var {key} not set")
        return default
    return float(value)

def check_required_env(*keys):
    """Check required environment variables are set."""
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise ValueError(f"Missing required env vars: {missing}")
```

### train.py에서 사용

```python
from utils import (
    download_dataset,
    upload_checkpoint,
    notify_status,
    send_validation_result,
    get_env_int,
    check_required_env
)

# Validation
check_required_env("JOB_ID", "CALLBACK_URL", "DATASET_ID")

# Parsing
epochs = get_env_int("EPOCHS", default=10)

# Storage
dataset_dir = download_dataset(os.getenv("DATASET_ID"))

# Training loop
for epoch in range(epochs):
    # ... training ...
    send_validation_result(epoch, metrics)
    upload_checkpoint(f"checkpoint_{epoch}.pt", f"checkpoints/job-{job_id}/epoch-{epoch}.pt")

# Done
notify_status("completed")
```

---

## 중복 코드 정리

### trainer-ultralytics와 trainer-timm의 차이

| 파일 | ultralytics | timm | 중복 여부 |
|------|-------------|------|----------|
| **utils.py** | 100% 동일 | 100% 동일 | ✅ 복사 |
| **train.py** | YOLO 학습 로직 | Timm 학습 로직 | ❌ 다름 |
| **predict.py** | YOLO 추론 | Timm 추론 | ❌ 다름 |
| **requirements.txt** | ultralytics, torch | timm, torch | ❌ 다름 |

**결론: utils.py만 중복 (~100줄). 허용 가능한 수준.**
