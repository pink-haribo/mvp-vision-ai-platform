# Trainer 구현 계획

## 전체 타임라인 (9-15시간, 8 Phase)

| Phase | 작업 | 예상 시간 | 의존성 |
|-------|------|----------|--------|
| Phase 1 | trainer-ultralytics 기본 구조 | 1-2h | - |
| Phase 2 | train.py 기본 구현 | 2-3h | Phase 1 |
| Phase 3 | Backend 연동 | 1-2h | Phase 2 |
| Phase 4 | Validation Callback 연동 | 2-3h | Phase 3 |
| Phase 5 | predict.py 구현 | 1-2h | Phase 2 |
| Phase 6 | MLflow 메트릭 연동 | 1h | Phase 4 |
| Phase 7 | 에러 처리 및 정리 | 1-2h | Phase 6 |
| Phase 8 | Docker 및 K8s 준비 | 나중에 | - |

---

## Phase 1: trainer-ultralytics 기본 구조 (1-2시간)

### 목표
- `mvp/training/` 삭제
- `mvp/trainer-ultralytics/` 생성
- 기본 파일 완성

### 작업

```bash
# 1. 삭제
git rm -rf mvp/training/
git commit -m "refactor: remove old training/ directory"

# 2. 생성
mkdir -p mvp/trainer-ultralytics
cd mvp/trainer-ultralytics
touch utils.py train.py predict.py
touch requirements.txt Dockerfile .env.example README.md
```

### 파일 작성

1. **utils.py** (~120줄)
   - `get_storage_client()` - MinIO/R2 클라이언트
   - `download_dataset()` - 데이터셋 다운로드/압축 해제
   - `upload_checkpoint()` - 체크포인트 업로드
   - `notify_status()` - 상태 알림
   - `send_validation_result()` - Validation 결과
   - `check_required_env()` - 환경변수 검증
   - `get_env_int()`, `get_env_float()` - 파싱

2. **requirements.txt**
   ```txt
   ultralytics>=8.0.0
   torch>=2.0.0
   boto3>=1.26.0
   requests>=2.28.0
   mlflow>=2.8.0
   ```

3. **.env.example** (환경변수 템플릿)

### 완료 조건
- [  ] `mvp/training/` 삭제됨
- [  ] `mvp/trainer-ultralytics/` 생성됨
- [  ] `utils.py` 구현 완료
- [  ] `requirements.txt` 작성
- [  ] `.env.example` 작성
- [  ] `python -m py_compile utils.py` 통과

---

## Phase 2: train.py 기본 구현 (2-3시간)

### 목표
- 최소 기능 train.py 완성
- Backend callback 연동
- MLflow 로깅

### 구현

```python
def main():
    # 1. 환경변수 검증
    check_required_env("JOB_ID", "DATASET_ID", ...)
    
    # 2. 상태 알림
    notify_status("running")
    
    try:
        # 3. 데이터셋 다운로드
        dataset_dir = download_dataset(dataset_id)
        
        # 4. MLflow 초기화
        mlflow.set_tracking_uri(...)
        mlflow.start_run(...)
        
        # 5. YOLO 모델 초기화
        model = YOLO(f"{model_name}.pt")
        
        # 6. 학습 (간단 버전)
        results = model.train(
            data=f"{dataset_dir}/data.yaml",
            epochs=epochs,
            batch=batch_size,
        )
        
        # 7. Checkpoint 업로드
        upload_checkpoint("best.pt", f"checkpoints/job-{job_id}/best.pt")
        
        # 8. 완료 알림
        notify_status("completed")
        
    except Exception as e:
        notify_status("failed", error=str(e))
```

### 완료 조건
- [  ] `train.py` 기본 구조 완성
- [  ] 환경변수로 실행 가능
- [  ] MinIO에서 데이터셋 다운로드
- [  ] YOLO 학습 실행
- [  ] Checkpoint MinIO 업로드
- [  ] Backend callback (status) 동작

---

## Phase 3: Backend 연동 (1-2시간)

### 목표
- Backend `training_manager_k8s.py` 수정
- 환경변수 주입
- subprocess 실행 테스트

### 변경사항

**training_manager_k8s.py:**
```python
def _start_training_subprocess(self, job_id):
    job = self.db.query(TrainingJob).filter_by(id=job_id).first()
    
    # trainer-{framework}/train.py 경로
    trainer_dir = project_root / f"trainer-{job.framework}"
    train_script = trainer_dir / "train.py"
    
    # 환경변수 빌드
    env = self._build_env_vars(job)
    
    # 실행
    cmd = ["python", str(train_script)]
    process = subprocess.Popen(cmd, env=env, ...)
```

### 완료 조건
- [  ] `training_manager_k8s.py` 경로 수정
- [  ] `_build_env_vars()` 구현
- [  ] Frontend에서 학습 시작 가능
- [  ] subprocess 로그 확인

---

## Phase 4: Validation Callback 연동 (2-3시간)

### 목표
- Epoch마다 validation 결과 Backend 전송
- Frontend에 실시간 메트릭 표시

### 구현

**train.py에 YOLO Callback:**
```python
from ultralytics.utils.callbacks import Callbacks

def on_val_end(validator):
    """Called after validation."""
    epoch = validator.epoch
    metrics = validator.metrics
    
    send_validation_result(
        epoch=epoch,
        metrics={
            "mAP50": metrics.get("mAP50"),
            "mAP50-95": metrics.get("mAP50-95"),
        }
    )

# Register callback
model.add_callback("on_val_end", on_val_end)
```

### 완료 조건
- [  ] YOLO callback 등록
- [  ] Epoch마다 Backend에 메트릭 전송
- [  ] Frontend에 실시간 메트릭 표시

---

## Phase 5: predict.py 구현 (1-2시간)

### 목표
- 추론 스크립트 완성
- Backend API 연동

### 구현

```python
def main():
    # Checkpoint 다운로드
    checkpoint_path = download_checkpoint(...)
    
    # 모델 로드
    model = YOLO(checkpoint_path)
    
    # 추론
    results = model.predict(input_path, conf=0.5)
    
    # 결과 출력
    print(json.dumps({
        "predictions": [...],
        "count": len(results)
    }))
```

### 완료 조건
- [  ] `predict.py` 구현 완료
- [  ] Backend 추론 API 구현
- [  ] Frontend에서 추론 테스트

---

## Phase 6: MLflow 메트릭 연동 (1시간)

### 목표
- MLflow에 학습 메트릭 자동 로깅

### 구현

```python
with mlflow.start_run(run_name=f"job-{job_id}"):
    # Log params
    mlflow.log_params({"model": model_name, ...})
    
    # Training loop
    for epoch in range(epochs):
        mlflow.log_metrics({"train_loss": loss}, step=epoch)
    
    # Log final checkpoint
    mlflow.log_artifact("best.pt")
```

### 완료 조건
- [  ] MLflow에 메트릭 자동 로깅
- [  ] MLflow Run ID 저장
- [  ] Frontend에서 MLflow 링크 제공

---

## Phase 7: 에러 처리 및 정리 (1-2시간)

### 목표
- Graceful shutdown
- 리소스 정리
- 에러 핸들링 강화

### 구현

```python
import signal

def signal_handler(signum, frame):
    print(f"[SIGNAL] Received {signum}")
    notify_status("cancelled")
    cleanup_temp_files()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

### 완료 조건
- [  ] Signal 핸들링 구현
- [  ] 임시 파일 정리
- [  ] Ctrl+C 정상 종료
- [  ] Backend stop 요청 정상 처리

---

## Phase 8: Docker 및 K8s (나중에)

- Dockerfile 최적화
- K8s Job 템플릿
- Railway 배포 설정

---

## 시작하기

### Step 1: 현재 코드 삭제

```bash
git rm -rf mvp/training/
git commit -m "refactor: remove old training/ for clean rebuild"
```

### Step 2: Phase 1 시작

```bash
mkdir -p mvp/trainer-ultralytics
cd mvp/trainer-ultralytics
touch utils.py train.py predict.py requirements.txt .env.example Dockerfile README.md
```

### Step 3: utils.py부터 구현

가장 먼저 utils.py를 완성하면 train.py/predict.py에서 바로 사용 가능
