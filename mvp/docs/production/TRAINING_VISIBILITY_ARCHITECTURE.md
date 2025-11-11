# Training Visibility Architecture Issue & Fix

**작성일**: 2025-01-18
**문제**: Railway에서 API 모드로 학습 실행 시 Backend가 학습 진행 상황을 추적하지 못함
**상태**: 분석 완료, 수정 필요

---

## 문제 요약

Railway 환경에서 Training Service API를 통해 학습을 시작하면 학습이 정상적으로 실행되지만, Backend는 학습 진행 상황을 전혀 알 수 없습니다.

**증상:**
- ✅ 학습 시작 성공 (`POST /training/start` → 200 OK)
- ❌ Backend DB에 로그/메트릭 저장 안 됨
- ❌ Frontend에서 학습 상태 조회 불가
- ❌ MLflow에서 404 에러 (experiment not found)

---

## 아키텍처 분석

### Subprocess 모드 (로컬 환경 - 정상 작동)

```
Backend
  └─ TrainingManager.start_subprocess()
       ├─ subprocess.Popen(["python", "train.py", ...])
       ├─ 실시간 stdout/stderr 캡처
       ├─ 로그 파싱 (정규식)
       │    └─ "Epoch 1, Loss: 0.234" → TrainingLog 생성
       ├─ DB에 저장 (TrainingLog, TrainingMetric)
       └─ job.status = "running"/"completed" 업데이트

Frontend
  └─ GET /api/v1/training/{job_id}/logs
       └─ Backend DB에서 실시간 조회 ✅
```

**동작 원리:**
1. Backend가 train.py를 subprocess로 직접 실행
2. stdout을 실시간으로 캡처하여 파싱
3. 파싱된 로그/메트릭을 Backend DB에 저장
4. Frontend가 Backend DB를 조회하여 실시간 업데이트

### API 모드 (Railway 환경 - 현재 문제 상황)

```
Backend
  └─ TrainingClient.start_training()
       ├─ POST http://timm-service/training/start
       └─ Response: {"job_id": 2, "status": "started"} ✅

Training Service (timm-service)
  └─ run_training() [background task]
       ├─ subprocess.run(["python", "/workspace/training/train.py", ...])
       ├─ stdout 캡처하지만 아무것도 안 함 ❌
       ├─ 로그 파싱 안 함 ❌
       ├─ Backend DB 업데이트 안 함 ❌
       └─ 결과만 in-memory job_status에 저장 ❌

Backend
  └─ Training Service와 연결 끊김 ❌
       └─ 학습 진행 상황 알 수 없음 ❌

Frontend
  └─ GET /api/v1/training/{job_id}/logs
       └─ Backend DB가 비어있음 → 빈 배열 반환 ❌
```

**문제점:**
1. Training Service의 `run_training()` 함수가 stdout을 캡처만 하고 파싱하지 않음
2. 파싱된 로그/메트릭을 Backend DB에 저장하지 않음 (API 호출 없음)
3. Backend는 Training Service에 학습 시작만 요청하고 이후 상태를 모름
4. Frontend는 Backend DB를 조회하지만 데이터가 없음

---

## 코드 레벨 분석

### train.py (mvp/training/train.py)

```python
# Lines 25-30: MLflow 환경 변수 (로컬 개발용 기본값)
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")  # ❌ Railway에서는 변경 필요
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")  # ❌ R2 사용 시 변경 필요

# Lines 87-146: DB에서 advanced_config 읽기만 함
def load_advanced_config_from_db(job_id: int):
    """Read advanced_config from database."""
    # ✅ DB 읽기는 함
    # ❌ 로그/메트릭 쓰기는 안 함
```

**관찰:**
- train.py는 Backend DB에서 `advanced_config`를 읽기만 함
- `TrainingLog`, `TrainingMetric` 모델을 import하지 않음 (DB 쓰기 안 함)
- MLflow에만 로그/메트릭 저장 (adapter.train() 내부에서 mlflow.log_metric() 호출)
- stdout에 로그 출력 (`print()` 사용)

### api_server.py (mvp/training/api_server.py)

```python
# Lines 61-101: run_training() 함수
def run_training(request: TrainingRequest):
    """Execute training in background."""
    job_id = request.job_id

    try:
        job_status[job_id] = {"status": "running", "error": None}  # ❌ In-memory only!

        # Build command
        cmd = ["python", "/workspace/training/train.py", ...]

        # Execute training
        result = subprocess.run(
            cmd,
            capture_output=True,  # ✅ stdout/stderr 캡처함
            text=True,
            timeout=3600
        )

        if result.returncode == 0:
            job_status[job_id] = {"status": "completed", "error": None}  # ❌ In-memory only!
        else:
            job_status[job_id] = {"status": "failed", "error": result.stderr}

        # ❌ 문제:
        # 1. result.stdout를 파싱하지 않음
        # 2. Backend API 호출 안 함 (로그/메트릭 저장 안 함)
        # 3. job_status는 in-memory dict → 서비스 재시작 시 사라짐
        # 4. Backend는 이 정보를 알 수 없음

    except Exception as e:
        job_status[job_id] = {"status": "failed", "error": str(e)}
```

**문제점:**
1. `capture_output=True`로 stdout을 캡처하지만 **아무것도 안 함**
2. Backend DB 업데이트 없음 (HTTP API 호출 없음)
3. In-memory `job_status`만 업데이트 (휘발성)
4. Backend는 `/training/status/{job_id}` 엔드포인트로 조회할 수 있지만, **실시간이 아님**

---

## 해결 방안

### Option A: Training Service가 Backend API 호출 (권장)

Training Service의 `run_training()` 함수를 개선하여 stdout을 파싱하고 Backend API를 호출합니다.

**장점:**
- ✅ 기존 아키텍처 유지 (train.py 수정 불필요)
- ✅ Training Service가 모든 프레임워크에 대해 일관된 로그 파싱 제공
- ✅ Backend와의 결합도 낮음 (HTTP API만 호출)

**단점:**
- ❌ 구현 복잡도 증가 (로그 파싱 로직 필요)
- ❌ 네트워크 오버헤드 (각 로그마다 HTTP 요청)

**구현 예시:**

```python
# mvp/training/api_server.py

import re
import requests
from typing import Optional

# Backend API URL (환경 변수에서 읽기)
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")

def parse_training_logs(stdout: str, job_id: int):
    """Parse training logs and send to Backend API."""
    # 정규식 패턴 (subprocess 모드와 동일)
    epoch_pattern = re.compile(r'Epoch (\d+)/(\d+)')
    loss_pattern = re.compile(r'Loss: ([\d.]+)')
    metric_pattern = re.compile(r'(\w+): ([\d.]+)')

    lines = stdout.split('\n')
    current_epoch = None

    for line in lines:
        # Epoch 파싱
        if match := epoch_pattern.search(line):
            current_epoch = int(match.group(1))

        # Loss 파싱
        if match := loss_pattern.search(line):
            loss_value = float(match.group(1))
            # Backend API 호출: POST /internal/training/{job_id}/logs
            send_log_to_backend(job_id, current_epoch, "train_loss", loss_value)

        # 기타 메트릭 파싱
        # ...

def send_log_to_backend(job_id: int, epoch: Optional[int], metric_name: str, value: float):
    """Send training log to Backend API."""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/internal/training/{job_id}/logs",
            json={
                "epoch": epoch,
                "metric_name": metric_name,
                "value": value,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={"X-Internal-Auth": os.environ.get("INTERNAL_AUTH_TOKEN")},
            timeout=5
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Failed to send log to Backend: {e}")

def run_training(request: TrainingRequest):
    """Execute training in background."""
    job_id = request.job_id

    try:
        # Update status: running
        update_backend_status(job_id, "running")

        # Build command
        cmd = [...]

        # Execute training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Parse stdout and send to Backend
        parse_training_logs(result.stdout, job_id)  # ✅ 로그 파싱 및 전송

        if result.returncode == 0:
            update_backend_status(job_id, "completed")
        else:
            update_backend_status(job_id, "failed", error=result.stderr)

    except Exception as e:
        update_backend_status(job_id, "failed", error=str(e))

def update_backend_status(job_id: int, status: str, error: Optional[str] = None):
    """Update training job status in Backend DB."""
    try:
        response = requests.patch(
            f"{BACKEND_API_URL}/internal/training/{job_id}/status",
            json={"status": status, "error": error},
            headers={"X-Internal-Auth": os.environ.get("INTERNAL_AUTH_TOKEN")},
            timeout=5
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Failed to update Backend status: {e}")
```

**필요한 Backend API 엔드포인트:**

```python
# mvp/backend/app/api/internal.py (신규 파일)

from fastapi import APIRouter, Header, HTTPException
from app.db.models import TrainingJob, TrainingLog, TrainingMetric
from app.db.database import get_db

router = APIRouter(prefix="/internal/training", tags=["internal"])

# Internal auth middleware
def verify_internal_auth(x_internal_auth: str = Header(...)):
    if x_internal_auth != os.environ.get("INTERNAL_AUTH_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid internal auth token")

@router.post("/{job_id}/logs", dependencies=[Depends(verify_internal_auth)])
async def create_training_log(
    job_id: int,
    log_data: dict,
    db: Session = Depends(get_db)
):
    """Create training log from Training Service."""
    log = TrainingLog(
        job_id=job_id,
        epoch=log_data.get("epoch"),
        message=f"{log_data['metric_name']}: {log_data['value']}",
        timestamp=log_data.get("timestamp")
    )
    db.add(log)
    db.commit()
    return {"status": "ok"}

@router.patch("/{job_id}/status", dependencies=[Depends(verify_internal_auth)])
async def update_training_status(
    job_id: int,
    status_data: dict,
    db: Session = Depends(get_db)
):
    """Update training job status from Training Service."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.status = status_data["status"]
    if status_data.get("error"):
        job.error_message = status_data["error"]

    db.commit()
    return {"status": "ok"}
```

---

### Option B: train.py가 직접 Backend DB에 쓰기

train.py가 Backend PostgreSQL에 직접 연결하여 로그/메트릭을 저장합니다.

**장점:**
- ✅ 가장 직접적인 방법 (중간 레이어 없음)
- ✅ 네트워크 오버헤드 없음

**단점:**
- ❌ Training Service가 Backend DB 접근 권한 필요 (보안 이슈)
- ❌ DB 스키마 변경 시 train.py도 수정 필요 (결합도 높음)
- ❌ Railway에서 PostgreSQL 연결 정보 공유 필요

**비권장:** 의존성 격리 원칙 위배

---

### Option C: MLflow를 단일 진실 공급원(Single Source of Truth)으로 사용

Backend가 MLflow API를 조회하여 학습 진행 상황을 가져옵니다.

**장점:**
- ✅ MLflow가 이미 모든 메트릭을 추적 중
- ✅ Backend DB에 중복 저장 불필요

**단점:**
- ❌ MLflow API 응답 속도가 느림 (대용량 실험 시)
- ❌ 실시간성 떨어짐
- ❌ MLflow와 강한 결합

**비권장:** 현재 아키텍처와 맞지 않음

---

## 권장 구현 순서

### 1단계: MLflow 환경 변수 설정 (즉시)

**Railway → Training Services (timm-service, ultralytics-service) → Variables:**

```bash
# MLflow Tracking Server
MLFLOW_TRACKING_URI=https://mlflow-service-production-xxxx.up.railway.app

# Cloudflare R2 (S3 호환)
AWS_ACCESS_KEY_ID=<your-r2-access-key>
AWS_SECRET_ACCESS_KEY=<your-r2-secret-key>
AWS_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com

# Backend API URL (for log/metric callbacks)
BACKEND_API_URL=https://backend-service-production-xxxx.up.railway.app

# Internal auth token (Backend와 Training Service 간 인증용)
INTERNAL_AUTH_TOKEN=<generate-random-token>
```

**Backend에도 동일한 토큰 설정:**

```bash
INTERNAL_AUTH_TOKEN=<same-random-token>
```

### 2단계: Backend Internal API 구현

1. `mvp/backend/app/api/internal.py` 생성
2. `POST /internal/training/{job_id}/logs` 엔드포인트 구현
3. `PATCH /internal/training/{job_id}/status` 엔드포인트 구현
4. `X-Internal-Auth` 헤더 검증 미들웨어 추가

### 3단계: Training Service 로그 파싱 및 전송 구현

1. `mvp/training/api_server.py`의 `run_training()` 함수 개선
2. `parse_training_logs()` 함수 추가
3. `send_log_to_backend()` 함수 추가
4. `update_backend_status()` 함수 추가

### 4단계: 테스트

1. 로컬에서 테스트 (Backend + Training Service 둘 다 로컬 실행)
2. Railway에 배포
3. 학습 실행 → Frontend에서 실시간 로그 확인
4. MLflow UI에서 메트릭 확인

---

## MLflow 404 에러 원인

**증상:**
```
GET /api/2.0/mlflow/experiments/get-by-name?experiment_name=job_2 HTTP/1.1" 404
```

**원인:**

1. **MLflow Tracking URI 미설정**
   - train.py가 `http://localhost:5000` 사용 (기본값)
   - Railway Training Service는 Railway MLflow 주소 필요
   - 환경 변수 `MLFLOW_TRACKING_URI` 미설정 시 기본값 사용

2. **Experiment 생성 실패**
   - train.py가 MLflow에 연결하지 못해 experiment 생성 안 됨
   - Backend가 존재하지 않는 experiment 조회 → 404

**해결:**
- Training Services에 `MLFLOW_TRACKING_URI` 환경 변수 설정 (1단계 완료 시 해결)

---

## 다음 단계

- [ ] 1단계: MLflow 환경 변수 설정 (Railway)
- [ ] 2단계: Backend Internal API 구현
- [ ] 3단계: Training Service 로그 파싱 구현
- [ ] 4단계: 통합 테스트

---

**작성자**: Claude Code
**최종 수정**: 2025-01-18
