# Trainer 실행 환경 비교

## 공통 사항 (환경 무관)

### 입력
- 환경변수로 모든 설정 받음
- argparse는 사용하지 않음 (K8s에서 env가 더 편함)

### 출력
- stdout으로 로그 출력 (Loki 수집)
- Callback URL로 상태/메트릭 전송
- Storage에 checkpoint/artifacts 업로드
- MLflow에 실험 메트릭 기록

### 필수 환경변수
```bash
# Job 식별
JOB_ID=1
PROJECT_ID=abc123

# Backend 통신
CALLBACK_URL=http://backend/internal/training/1
INTERNAL_AUTH_TOKEN=secret

# Training 설정
TASK_TYPE=object_detection
MODEL_NAME=yolo11n
DATASET_ID=dataset-uuid
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=0.001

# Storage (환경별로 다름)
STORAGE_TYPE=minio|r2
# ... storage credentials

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=project-abc123
```

---

## 환경별 차이점

| 항목 | Local (Subprocess) | Local (Kind) | Production (Railway) |
|------|-------------------|--------------|---------------------|
| **실행 방식** | subprocess.Popen() | K8s Job | Railway K8s Job |
| **네트워크** | localhost | K8s services | Public HTTPS |
| **Storage** | MinIO (localhost:9000) | MinIO (minio-service:9000) | R2 (HTTPS) |
| **Database** | N/A (Callback만 사용) | N/A | N/A |
| **MLflow** | http://localhost:5000 | http://mlflow-service:5000 | https://mlflow.railway.app |
| **로그 수집** | 파일 → Promtail → Loki | kubectl logs → Loki | Railway logs → Loki |
| **GPU** | CUDA_VISIBLE_DEVICES | nodeSelector + limits | Railway GPU node |
| **이미지** | 로컬 빌드 | docker build + kind load | GHCR push |

---

## 환경 감지 로직 (Trainer 내부)

```python
# train.py 내부에서 자동 감지
import os

STORAGE_TYPE = os.getenv("STORAGE_TYPE", "minio")  # minio or r2

if STORAGE_TYPE == "minio":
    # Local MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
    )
elif STORAGE_TYPE == "r2":
    # Production R2
    s3_client = boto3.client(
        's3',
        endpoint_url=os.getenv("R2_ENDPOINT"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY")
    )
```

---

## Backend에서 환경변수 주입

### training_manager_k8s.py

```python
def _build_env_vars(self, job):
    """Build environment variables for trainer."""
    env = {
        # Job info
        "JOB_ID": str(job.id),
        "PROJECT_ID": job.project_id or "default",

        # Backend callback
        "CALLBACK_URL": f"{self.backend_url}/internal/training/{job.id}",
        "INTERNAL_AUTH_TOKEN": os.getenv("INTERNAL_AUTH_TOKEN"),

        # Training config
        "TASK_TYPE": job.task_type,
        "MODEL_NAME": job.model_name,
        "DATASET_ID": job.dataset_id,
        "EPOCHS": str(job.epochs),
        "BATCH_SIZE": str(job.batch_size),
        "LEARNING_RATE": str(job.learning_rate),

        # MLflow
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        "MLFLOW_EXPERIMENT_NAME": f"project-{job.project_id}",
    }

    # Storage (환경별 분기)
    if self.executor == "subprocess":
        # Local MinIO
        env.update({
            "STORAGE_TYPE": "minio",
            "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
            "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY"),
            "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY"),
        })
    elif self.executor == "kubernetes":
        # Production R2
        env.update({
            "STORAGE_TYPE": "r2",
            "R2_ENDPOINT": os.getenv("R2_ENDPOINT"),
            "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID"),
            "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY"),
            "R2_BUCKET": os.getenv("R2_BUCKET"),
        })

    return env
```
