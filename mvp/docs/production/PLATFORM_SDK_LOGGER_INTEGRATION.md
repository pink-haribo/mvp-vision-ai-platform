# Platform SDK Logger Integration Guide

**작성일**: 2025-01-18
**목적**: Adapter에 TrainingLogger 통합 및 train.py 수정 가이드

---

## 완료된 작업

### 1. Platform SDK에 TrainingLogger 추가 ✅

**파일**: `mvp/training/platform_sdk/logger.py`

**기능**:
- `log_metric()`: 메트릭 전송 (train_loss, val_accuracy 등)
- `log_message()`: 로그 메시지 전송
- `update_status()`: 학습 상태 업데이트
- `log_epoch_summary()`: 에폭 요약 전송

**환경변수**:
- `BACKEND_API_URL`: Backend API URL
- `INTERNAL_AUTH_TOKEN`: 인증 토큰

### 2. Backend Internal API 구현 ✅

**파일**: `mvp/backend/app/api/internal.py`

**엔드포인트**:
- `POST /internal/training/{job_id}/metrics`: 메트릭 저장
- `POST /internal/training/{job_id}/logs`: 로그 저장
- `PATCH /internal/training/{job_id}/status`: 상태 업데이트

**보안**: `X-Internal-Auth` 헤더 검증

### 3. advanced_config API 전달 ✅

**변경 사항**:
- `api_server.py`: `TrainingRequest.advanced_config` 추가
- `training_manager.py`: Backend가 advanced_config 포함하여 전송
- `train.py`: `--advanced_config` 커맨드라인 인자 추가

**효과**: Training Service가 더 이상 Backend DB에 직접 접근하지 않음 (의존성 격리)

### 4. TrainingAdapter에 logger 파라미터 추가 ✅

**파일**: `mvp/training/platform_sdk/base.py`

```python
def __init__(
    self,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    output_dir: str,
    job_id: int,
    logger: Optional['TrainingLogger'] = None  # ← 추가됨
):
    self.logger = logger
    # ...
```

---

## 남은 작업

### 5. train.py에서 TrainingLogger 초기화

**파일**: `mvp/training/train.py`

**수정 위치**: `main()` 함수에서 adapter 생성 직전

```python
def main():
    """Main training function."""
    args = parse_args()

    # ... (existing code) ...

    # Initialize TrainingLogger
    from platform_sdk import TrainingLogger

    logger = TrainingLogger(
        job_id=args.job_id,
        enabled=True  # Railway에서는 환경변수가 있으면 자동 활성화
    )

    logger.log_message(f"Starting training: {args.model_name}", level="INFO")
    logger.update_status("running")

    # Create adapter instance with logger
    adapter = adapter_class(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        output_dir=args.output_dir,
        job_id=args.job_id,
        logger=logger  # ← logger 전달
    )

    # Train
    try:
        metrics = adapter.train(...)

        # Final status update
        logger.update_status("completed")
        logger.log_message("Training completed successfully", level="INFO")

    except Exception as e:
        logger.update_status("failed", error=str(e))
        logger.log_message(f"Training failed: {e}", level="ERROR")
        raise
```

### 6. TimmAdapter에서 logger 사용

**파일**: `mvp/training/adapters/timm_adapter.py`

**수정 위치**: `train_epoch()` 메서드

```python
def train_epoch(self, epoch: int) -> MetricsResult:
    """Train for one epoch."""
    self.model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(self.train_loader):
        # ... training logic ...

        # Log batch metrics (optional, every 100 steps)
        if self.logger and batch_idx % 100 == 0:
            self.logger.log_metric(
                "batch_loss",
                loss.item(),
                epoch=epoch,
                step=batch_idx
            )

    # Calculate epoch metrics
    avg_loss = total_loss / len(self.train_loader)
    accuracy = correct / total

    # Log epoch metrics
    if self.logger:
        self.logger.log_metric("train_loss", avg_loss, epoch=epoch)
        self.logger.log_metric("train_accuracy", accuracy, epoch=epoch)
        self.logger.log_message(
            f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}",
            level="INFO",
            epoch=epoch
        )

    # Also return metrics for MLflow
    return MetricsResult(
        train_loss=avg_loss,
        train_acc=accuracy,
        # ...
    )
```

### 7. UltralyticsAdapter에서 logger 사용

**파일**: `mvp/training/adapters/ultralytics_adapter.py`

Ultralytics는 자체 콜백 시스템을 사용하므로, custom callback을 추가해야 합니다:

```python
def train(self, ...):
    """Train YOLO model."""

    # Custom callback for logging
    def on_epoch_end(trainer):
        epoch = trainer.epoch
        metrics = trainer.metrics

        if self.logger:
            # Log all metrics
            for key, value in metrics.items():
                self.logger.log_metric(key, value, epoch=epoch)

    # Add callback
    model.add_callback("on_train_epoch_end", on_epoch_end)

    # Start training
    results = model.train(...)

    return results
```

### 8. Railway 환경변수 설정

**각 Training Service (timm-service, ultralytics-service)에 추가:**

```bash
# Backend API URL
BACKEND_API_URL=https://backend-service-production.up.railway.app

# Internal auth token (Backend와 동일한 값)
INTERNAL_AUTH_TOKEN=<generate-random-token-here>

# MLflow (이미 추가함)
MLFLOW_TRACKING_URI=https://mlflow-service-production.up.railway.app
AWS_ACCESS_KEY_ID=<r2-access-key>
AWS_SECRET_ACCESS_KEY=<r2-secret-key>
AWS_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
```

**Backend에도 동일한 토큰 추가:**

```bash
INTERNAL_AUTH_TOKEN=<same-random-token-here>
```

**토큰 생성 방법 (PowerShell):**

```powershell
# 랜덤 64자 토큰 생성
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | % {[char]$_})
```

---

## 테스트 계획

### 1. 로컬 테스트 (선택적)

```bash
# Backend 시작
cd mvp/backend
python -m uvicorn app.main:app --reload --port 8000

# Training Service 시작 (다른 터미널)
cd mvp/training
$env:BACKEND_API_URL="http://localhost:8000"
$env:INTERNAL_AUTH_TOKEN="test-token-local"
python api_server.py

# 학습 실행
# Frontend에서 학습 시작 → Backend 로그에서 Internal API 호출 확인
```

### 2. Railway 배포 테스트

**단계**:
1. 모든 코드 변경사항 commit & push
2. Railway 자동 재배포 대기 (또는 수동 트리거)
3. 환경변수 설정 (BACKEND_API_URL, INTERNAL_AUTH_TOKEN)
4. Frontend에서 학습 시작
5. 실시간 로그 확인:
   - Frontend UI에 로그 표시되는지
   - Backend 로그에 Internal API 호출 보이는지
   - MLflow UI에 메트릭 기록되는지

**확인 사항**:
- ✅ Frontend에서 실시간 로그 표시
- ✅ Backend DB에 TrainingLog, TrainingMetric 저장
- ✅ MLflow에 실험 기록
- ✅ R2에 artifacts 업로드
- ✅ 학습 완료 시 status='completed'

---

## 예상 로그 흐름

**Training Service (train.py)**:
```
[CONFIG] Advanced config loaded from API parameter  ← DB 접근 안 함!
[INFO] Starting training: yolo11n
[TrainingLogger] Sending metric to Backend: train_loss=0.234
```

**Backend (Internal API)**:
```
POST /internal/training/3/metrics → 200 OK
POST /internal/training/3/logs → 200 OK
```

**Frontend**:
```
Epoch 1: Loss=0.234, Acc=0.89  ← 실시간 표시
```

---

## 문제 해결

### 문제: Internal API 401/403 에러

**원인**: INTERNAL_AUTH_TOKEN 불일치

**해결**:
```bash
# Backend와 Training Services의 토큰이 정확히 일치하는지 확인
railway variables get INTERNAL_AUTH_TOKEN --service backend-service
railway variables get INTERNAL_AUTH_TOKEN --service timm-service
```

### 문제: Logger가 "disabled" 상태

**원인**: BACKEND_API_URL 미설정

**해결**:
```bash
railway variables set BACKEND_API_URL=https://backend-service-production.up.railway.app --service timm-service
```

### 문제: 로그는 보이지만 MLflow는 404

**원인**: MLFLOW_TRACKING_URI 미설정 또는 잘못됨

**해결**:
```bash
# URI 확인 (TRACKING_URI, not TRACKING_API!)
railway variables get MLFLOW_TRACKING_URI --service timm-service
```

---

## 다음 단계

1. ✅ Platform SDK Logger 구현
2. ✅ Backend Internal API 구현
3. ✅ advanced_config API 전달
4. ✅ TrainingAdapter에 logger 파라미터 추가
5. ⬜ train.py에서 TrainingLogger 초기화 및 전달
6. ⬜ TimmAdapter/UltralyticsAdapter에서 logger 사용
7. ⬜ Railway 환경변수 설정
8. ⬜ 통합 테스트

**현재 단계**: 5번 진행 중

---

**작성자**: Claude Code
**최종 수정**: 2025-01-18
