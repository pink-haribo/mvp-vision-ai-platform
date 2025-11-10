# Trainer 아키텍처 설계

## 전체 디렉토리 구조

```
mvp/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── internal.py           # Callback endpoints
│   │   │   └── training.py           # Training job API
│   │   └── utils/
│   │       └── training_manager_k8s.py  # Trainer 실행 관리
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── trainer-ultralytics/              # ✅ NEW
│   ├── train.py                      # 학습 메인 (~300줄)
│   ├── predict.py                    # 추론 (~150줄)
│   ├── utils.py                      # 공통 유틸 (~120줄)
│   ├── requirements.txt              # ultralytics, torch, boto3, requests, mlflow
│   ├── Dockerfile
│   ├── .env.example
│   └── README.md
│
├── trainer-timm/                     # ✅ NEW (나중에)
│   ├── train.py
│   ├── predict.py
│   ├── utils.py                      # ultralytics/utils.py 복사
│   ├── requirements.txt              # timm, torch, boto3, requests, mlflow
│   ├── Dockerfile
│   └── .env.example
│
├── frontend/
│   └── ... (변경 없음)
│
└── docker-compose.yml
```

---

## trainer-ultralytics/ 파일 상세

### 1. utils.py (~120줄)

공통 유틸리티 함수 모음:

```python
# Storage 관련
get_storage_client() → boto3.client  # MinIO/R2 자동 분기
download_dataset(dataset_id) → local_path
upload_checkpoint(local_path, remote_path) → s3_url

# Backend Callback
notify_status(status, error=None)
send_validation_result(epoch, metrics)
send_training_metrics(epoch, loss, lr, ...)

# 환경변수 파싱
check_required_env(*keys)
get_env_int(key, default, required)
get_env_float(key, default, required)
```

### 2. train.py (~300줄)

학습 메인 스크립트:

```python
def main():
    # 1. 환경변수 검증
    check_required_env("JOB_ID", "DATASET_ID", ...)
    
    # 2. 상태 알림
    notify_status("running")
    
    # 3. 데이터셋 다운로드
    dataset_dir = download_dataset(dataset_id)
    
    # 4. MLflow 초기화
    mlflow.set_tracking_uri(...)
    mlflow.start_run(...)
    
    # 5. 모델 초기화
    model = YOLO(f"{model_name}.pt")
    
    # 6. 학습 (YOLO callback 등록)
    model.add_callback("on_val_end", send_validation_result)
    results = model.train(...)
    
    # 7. Checkpoint 업로드
    upload_checkpoint("best.pt", "checkpoints/job-X/best.pt")
    
    # 8. 완료 알림
    notify_status("completed")
```

### 3. predict.py (~150줄)

추론 스크립트:

```python
def main():
    # 1. Checkpoint 다운로드
    checkpoint_path = download_checkpoint(...)
    
    # 2. 모델 로드
    model = YOLO(checkpoint_path)
    
    # 3. 이미지 다운로드
    image_path = download_image(...)
    
    # 4. 추론
    results = model.predict(image_path, conf=0.5)
    
    # 5. 결과 포맷팅 및 출력
    print(json.dumps({
        "predictions": [...],
        "count": len(results)
    }))
```

### 4. requirements.txt

```txt
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
boto3>=1.26.0
requests>=2.28.0
mlflow>=2.8.0
```

### 5. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py predict.py utils.py ./

CMD ["python", "train.py"]
```

### 6. .env.example

```bash
# Job Configuration
JOB_ID=1
TASK_TYPE=object_detection
MODEL_NAME=yolo11n
DATASET_ID=dataset-uuid

# Training Hyperparameters
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=0.001

# Backend Callback
CALLBACK_URL=http://backend:8000/internal/training/1
INTERNAL_AUTH_TOKEN=your-secret

# Storage (Local MinIO)
STORAGE_TYPE=minio
STORAGE_BUCKET=vision-platform
MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
```

---

## Backend 변경사항

### training_manager_k8s.py

```python
def _start_training_subprocess(self, job_id):
    """Start training via subprocess."""
    job = self.db.query(TrainingJob).filter_by(id=job_id).first()
    
    # Determine trainer directory
    project_root = Path(__file__).parent.parent.parent.parent
    trainer_dir = project_root / f"trainer-{job.framework}"  # trainer-ultralytics
    train_script = trainer_dir / "train.py"
    
    if not train_script.exists():
        raise FileNotFoundError(f"Trainer not found: {trainer_dir}")
    
    # Build environment variables
    env = self._build_env_vars(job)
    
    # Execute
    cmd = ["python", str(train_script)]
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    
    # Update job
    job.status = "running"
    job.process_id = process.pid
    self.db.commit()

def _build_env_vars(self, job):
    """Build environment variables for trainer."""
    env = os.environ.copy()
    
    # Job config
    env.update({
        "JOB_ID": str(job.id),
        "TASK_TYPE": job.task_type,
        "MODEL_NAME": job.model_name,
        "DATASET_ID": job.dataset_id,
        "EPOCHS": str(job.epochs),
        "BATCH_SIZE": str(job.batch_size),
        "LEARNING_RATE": str(job.learning_rate),
    })
    
    # Callback
    backend_url = os.getenv("BACKEND_INTERNAL_URL", "http://localhost:8000")
    env["CALLBACK_URL"] = f"{backend_url}/internal/training/{job.id}"
    env["INTERNAL_AUTH_TOKEN"] = os.getenv("INTERNAL_AUTH_TOKEN")
    
    # Storage (local = MinIO, prod = R2)
    if os.getenv("ENVIRONMENT") == "production":
        env.update({
            "STORAGE_TYPE": "r2",
            "R2_ENDPOINT": os.getenv("R2_ENDPOINT"),
            "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID"),
            "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY"),
        })
    else:
        env.update({
            "STORAGE_TYPE": "minio",
            "MINIO_ENDPOINT": "http://localhost:9000",
            "MINIO_ACCESS_KEY": "minioadmin",
            "MINIO_SECRET_KEY": "minioadmin",
        })
    
    # MLflow
    env["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    return env
```

---

## 핵심 원칙

1. **완전한 독립성**: trainer-ultralytics는 ultralytics + requests + boto3만 의존
2. **환경변수 기반**: argparse 없음, 모든 설정은 env
3. **Backend는 DB만**: Trainer는 Callback URL로만 통신
4. **Storage 추상화**: STORAGE_TYPE=minio|r2로 자동 분기
