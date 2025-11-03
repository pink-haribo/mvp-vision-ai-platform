# 시나리오 6: 학습(실행) - 학습 버튼 눌렀을 때

## 개요

사용자가 생성된 학습 작업(status=`pending`)의 "학습 시작" 버튼을 클릭하면, **실제 모델 학습이 시작**되는 과정입니다.

**목표:** 학습 프로세스 시작 → 진행 중 상태 업데이트 → 학습 완료 및 결과 저장

**핵심 차이:** 로컬과 배포 환경에서 **학습 실행 방식**이 완전히 다릅니다!

---

## 로컬 환경 (개발) - Subprocess 방식

### 환경 구성
```
Frontend: http://localhost:3000
Backend:  http://localhost:8000
Training: 동일한 컴퓨터에서 subprocess로 실행
```

### 상세 흐름

#### 1단계: 사용자가 "학습 시작" 버튼 클릭

**위치:** 브라우저 (http://localhost:3000/projects/1)

**사용자 동작:**
```
실험 목록에서 pending 상태 실험의 [▶ 학습 시작] 버튼 클릭
```

**Frontend 코드:**
```typescript
// mvp/frontend/components/ExperimentTable.tsx

const handleStartTraining = async (jobId: number) => {
  const token = localStorage.getItem('access_token');

  const response = await fetch(
    `http://localhost:8000/api/v1/training/jobs/${jobId}/start`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );

  if (response.ok) {
    showToast('학습이 시작되었습니다!', 'success');
    refreshExperiments();
  }
};
```

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
POST http://localhost:8000/api/v1/training/jobs/6/start
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

#### 3단계: Backend API 엔드포인트 실행

**위치:** `mvp/backend/app/api/training.py`

```python
@router.post("/jobs/{job_id}/start")
def start_training_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """학습 작업 시작"""

    # 1. 학습 작업 조회
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # 2. 권한 확인
    project = db.query(Project).filter(Project.id == job.project_id).first()
    if project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # 3. 상태 확인
    if job.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start job with status '{job.status}'"
        )

    # 4. TrainingManager로 학습 시작
    from app.utils.training_manager import TrainingManager

    manager = TrainingManager(db)
    success = manager.start_training(job_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to start training")

    return {"message": "Training started", "job_id": job_id}
```

---

#### 4단계: TrainingManager - 로컬 Subprocess 실행 (중요!)

**위치:** `mvp/backend/app/utils/training_manager.py`

```python
class TrainingManager:
    def __init__(self, db: Session):
        self.db = db
        # 실행 모드 감지 (환경변수)
        self.execution_mode = os.getenv("TRAINING_EXECUTION_MODE", "subprocess")

    def start_training(self, job_id: int) -> bool:
        """학습 시작 (실행 모드에 따라 다름)"""

        if self.execution_mode == "subprocess":
            # 로컬 환경: subprocess로 직접 실행
            return self._start_training_subprocess(job_id)
        elif self.execution_mode == "api":
            # 배포 환경: Training Service API 호출 (나중에 설명)
            return self._start_training_api(job_id)
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")

    def _start_training_subprocess(self, job_id: int) -> bool:
        """로컬 subprocess로 학습 실행"""

        # 1. DB에서 학습 작업 조회
        job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return False

        # 2. 상태 업데이트: pending → running
        job.status = "running"
        job.started_at = datetime.now()
        self.db.commit()

        # 3. Python 명령어 구성
        train_script = Path(__file__).parent.parent.parent.parent / "training" / "train.py"

        cmd = [
            "python",  # 또는 가상환경 Python 경로
            str(train_script),
            "--job_id", str(job_id),
            "--framework", job.framework,
            "--model_name", job.model_name,
            "--task_type", job.task_type,
            "--dataset_path", job.dataset_path,
            "--dataset_format", job.dataset_format,
            "--epochs", str(job.epochs),
            "--batch_size", str(job.batch_size),
            "--learning_rate", str(job.learning_rate),
            "--output_dir", job.output_dir,
        ]

        # 4. subprocess 실행 (백그라운드)
        import subprocess

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 5. 프로세스 ID 저장 (나중에 취소 시 사용)
        job.process_id = process.pid
        self.db.commit()

        print(f"[TrainingManager] Started training job {job_id} (PID: {process.pid})")

        return True
```

**동작:**
1. 학습 작업 상태를 `pending` → `running`으로 변경
2. `mvp/training/train.py` 경로 찾기
3. Python subprocess 명령어 구성
4. **`subprocess.Popen()`으로 별도 Python 프로세스 생성**
5. 프로세스 ID (PID) 저장

**핵심:**
- Backend와 Training 코드가 **같은 컴퓨터**에 있음
- Backend가 Training 스크립트를 **직접 실행** 가능
- 별도 Python 프로세스로 실행 (비동기, 백그라운드)

---

#### 5단계: Training 스크립트 실행

**위치:** 새로운 Python 프로세스 (`mvp/training/train.py`)

```python
# mvp/training/train.py

import argparse
import sys
from pathlib import Path

def main():
    # 1. 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--framework", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_format", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"[Train] Starting job {args.job_id}")
    print(f"[Train] Framework: {args.framework}")
    print(f"[Train] Model: {args.model_name}")
    print(f"[Train] Dataset: {args.dataset_path}")

    # 2. Framework별 Adapter 로드
    from adapters import get_adapter

    adapter = get_adapter(args.framework)

    # 3. 학습 실행
    try:
        result = adapter.train(
            model_name=args.model_name,
            task_type=args.task_type,
            dataset_path=args.dataset_path,
            dataset_format=args.dataset_format,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            job_id=args.job_id
        )

        print(f"[Train] Job {args.job_id} completed successfully")
        print(f"[Train] Final accuracy: {result['accuracy']:.4f}")
        print(f"[Train] Final loss: {result['loss']:.4f}")

        # 4. 결과를 DB에 저장 (Backend API 호출 or 직접 DB 업데이트)
        update_job_status(args.job_id, "completed", result)

    except Exception as e:
        print(f"[Train] Job {args.job_id} failed: {e}")
        update_job_status(args.job_id, "failed", {"error": str(e)})

if __name__ == "__main__":
    main()
```

**동작:**
1. 커맨드라인 인자로 학습 설정 받기
2. Framework Adapter 로드 (TimmAdapter or UltralyticsAdapter)
3. **실제 PyTorch 학습 실행**
4. 학습 완료 후 결과를 DB에 업데이트

---

#### 6단계: Adapter - 실제 학습 로직

**위치:** `mvp/training/adapters/ultralytics_adapter.py`

```python
class UltralyticsAdapter(TrainingAdapter):
    def train(self, model_name, task_type, dataset_path, epochs, batch_size, learning_rate, output_dir, job_id, **kwargs):
        """Ultralytics 모델 학습"""

        from ultralytics import YOLO

        # 1. 모델 로드
        if "seg" in model_name:
            model = YOLO(f"{model_name}.pt")  # yolo11n-seg.pt
        else:
            model = YOLO(f"{model_name}.pt")  # yolo11n.pt

        # 2. 데이터셋 YAML 파일 경로
        data_yaml = f"{dataset_path}/data.yaml"

        # 3. 학습 시작 (PyTorch)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            lr0=learning_rate,
            project=output_dir,
            name="run",
            device="cpu",  # 로컬은 CPU (GPU 없음)
            verbose=True
        )

        # 4. 최종 결과 반환
        return {
            "accuracy": results.results_dict.get("metrics/mAP50", 0.0),
            "loss": results.results_dict.get("train/box_loss", 0.0),
            "best_model_path": results.save_dir / "weights" / "best.pt"
        }
```

**동작:**
1. Ultralytics YOLO 라이브러리 import
2. 사전 학습된 모델 가중치 다운로드 (`.pt` 파일)
3. **PyTorch로 실제 학습 실행** (epoch loop, backpropagation, etc.)
4. 학습 중 로그 출력 (stdout)
5. 최종 결과 반환

**학습 출력 예시:**
```
[Train] Starting job 6
[Train] Framework: ultralytics
[Train] Model: yolo11n-seg
[Train] Dataset: /app/datasets/seg-coco8

Ultralytics YOLOv11n-seg 🚀
Epoch   GPU_mem  box_loss  seg_loss  cls_loss  dfl_loss  Instances  Size
1/50    0.00G    1.234     0.567     0.890     1.123     20         640
2/50    0.00G    1.123     0.501     0.834     1.067     20         640
3/50    0.00G    1.067     0.456     0.789     1.012     20         640
...
48/50   0.00G    0.234     0.123     0.156     0.345     20         640
49/50   0.00G    0.223     0.118     0.149     0.338     20         640
50/50   0.00G    0.218     0.115     0.145     0.332     20         640

Training complete (2.3h)
Results saved to outputs/1/20240118_153000_yolo11n-seg/run
```

---

#### 7단계: 학습 완료 후 DB 업데이트

**위치:** `mvp/training/train.py` (학습 완료 후)

```python
def update_job_status(job_id, status, result):
    """학습 완료 후 DB 업데이트"""

    # Backend API 호출 or 직접 DB 업데이트
    # 방법 1: Backend API 호출 (권장)
    import requests

    response = requests.patch(
        f"http://localhost:8000/api/v1/training/jobs/{job_id}",
        json={
            "status": status,
            "accuracy": result.get("accuracy"),
            "loss": result.get("loss"),
            "completed_at": datetime.now().isoformat()
        }
    )

    # 방법 2: 직접 DB 업데이트 (로컬만 가능)
    # from sqlalchemy import create_engine
    # ...
```

**DB 업데이트:**
```sql
UPDATE training_jobs
SET
    status = 'completed',
    accuracy = 0.89,
    loss = 0.218,
    completed_at = '2024-01-18 17:45:00'
WHERE id = 6;
```

---

#### 8단계: Frontend 자동 새로고침 (실시간 업데이트)

**위치:** 브라우저

**방법 1: 폴링 (Polling)**
```typescript
// 5초마다 실험 목록 새로고침
useEffect(() => {
  const interval = setInterval(() => {
    refreshExperiments();
  }, 5000);

  return () => clearInterval(interval);
}, []);
```

**방법 2: WebSocket (실시간)**
```typescript
// WebSocket 연결
const socket = io('ws://localhost:8000');

// 학습 진행률 구독
socket.emit('subscribe', `job:${jobId}`);

// 실시간 업데이트 수신
socket.on('training_progress', (data) => {
  // data: { job_id: 6, epoch: 25, loss: 0.345, accuracy: 0.85 }
  updateJobMetrics(data.job_id, data);
});

// 학습 완료 알림
socket.on('training_complete', (data) => {
  showToast('학습이 완료되었습니다!', 'success');
  refreshExperiments();
});
```

---

## 배포 환경 (Railway) - HTTP API 방식

### 환경 구성
```
Frontend:           https://frontend-production-xxxx.up.railway.app
Backend:            https://backend-production-xxxx.up.railway.app
timm-service:       https://timm-service-production-xxxx.up.railway.app
ultralytics-service: https://ultralytics-service-production-xxxx.up.railway.app
```

**핵심 차이:**
- Backend와 Training 코드가 **별도 컨테이너**
- Backend가 Training 스크립트를 **직접 실행 불가능**
- 대신 **HTTP API**로 Training Service에 요청

### 상세 흐름

#### 1단계: 사용자가 "학습 시작" 버튼 클릭

**동작:** 로컬과 동일

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
POST https://backend-production-xxxx.up.railway.app/api/v1/training/jobs/6/start
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**차이점:** HTTPS, Railway URL

---

#### 3단계: Backend API 엔드포인트 실행

**동작:** 로컬과 동일 (권한 확인, 상태 확인)

```python
@router.post("/jobs/{job_id}/start")
def start_training_job(...):
    # 1. 학습 작업 조회
    # 2. 권한 확인
    # 3. 상태 확인
    # 4. TrainingManager로 학습 시작
    manager = TrainingManager(db)
    success = manager.start_training(job_id)
    ...
```

---

#### 4단계: TrainingManager - HTTP API 방식 (중요!)

**위치:** `mvp/backend/app/utils/training_manager.py`

```python
class TrainingManager:
    def __init__(self, db: Session):
        self.db = db
        # 배포 환경: TRAINING_EXECUTION_MODE=api
        self.execution_mode = os.getenv("TRAINING_EXECUTION_MODE", "api")

    def start_training(self, job_id: int) -> bool:
        """학습 시작"""

        if self.execution_mode == "api":
            # 배포 환경: Training Service API 호출
            return self._start_training_api(job_id)
        elif self.execution_mode == "subprocess":
            # 로컬 환경: subprocess (이미 설명)
            return self._start_training_subprocess(job_id)

    def _start_training_api(self, job_id: int) -> bool:
        """Training Service API로 학습 시작"""

        # 1. DB에서 학습 작업 조회
        job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return False

        # 2. Framework에 따라 Training Service URL 결정
        training_services = {
            "timm": os.getenv("TIMM_SERVICE_URL"),
            "ultralytics": os.getenv("ULTRALYTICS_SERVICE_URL"),
            "huggingface": os.getenv("HUGGINGFACE_SERVICE_URL"),
        }

        service_url = training_services.get(job.framework)
        if not service_url:
            print(f"[ERROR] No Training Service URL for framework '{job.framework}'")
            return False

        # 3. Training Service 헬스체크
        try:
            health_response = requests.get(f"{service_url}/health", timeout=5)
            if health_response.status_code != 200:
                raise Exception(f"Training Service unhealthy: {health_response.status_code}")
        except Exception as e:
            print(f"[ERROR] Training Service not available: {e}")
            job.status = "failed"
            job.error_message = f"Training Service unavailable: {str(e)}"
            self.db.commit()
            return False

        # 4. 학습 요청 페이로드 구성
        payload = {
            "job_id": job_id,
            "framework": job.framework,
            "model_name": job.model_name,
            "task_type": job.task_type,
            "dataset_path": job.dataset_path,
            "dataset_format": job.dataset_format,
            "epochs": job.epochs,
            "batch_size": job.batch_size,
            "learning_rate": job.learning_rate,
            "optimizer": "adam",
            "output_dir": job.output_dir,
            "device": "cpu",  # Railway는 CPU만
            "pretrained": True,
        }

        # 5. Training Service API 호출 (HTTP POST)
        try:
            response = requests.post(
                f"{service_url}/training/start",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                # 6. 상태 업데이트: pending → running
                job.status = "running"
                job.started_at = datetime.now()
                self.db.commit()

                print(f"[TrainingManager] Started training job {job_id} on {service_url}")
                return True
            else:
                raise Exception(f"Training Service error: {response.text}")

        except Exception as e:
            print(f"[ERROR] Failed to start training: {e}")
            job.status = "failed"
            job.error_message = str(e)
            self.db.commit()
            return False
```

**동작:**
1. 학습 작업의 `framework` 확인 (`ultralytics`)
2. 환경변수에서 Training Service URL 가져오기
   - `ULTRALYTICS_SERVICE_URL=https://ultralytics-service-production-xxxx.up.railway.app`
3. Training Service 헬스체크
4. **HTTP POST 요청**으로 학습 시작
5. DB 상태 업데이트 (`pending` → `running`)

**핵심:**
- Backend는 Training 코드가 **없음** (별도 컨테이너)
- **HTTP API**로 Training Service와 통신
- Training Service가 실제 학습 실행

---

#### 5단계: Backend → Training Service HTTP 요청

**요청:**
```http
POST https://ultralytics-service-production-xxxx.up.railway.app/training/start
Content-Type: application/json

{
  "job_id": 6,
  "framework": "ultralytics",
  "model_name": "yolo11n-seg",
  "task_type": "instance_segmentation",
  "dataset_path": "/app/datasets/seg-coco8",
  "dataset_format": "yolo",
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.01,
  "optimizer": "adam",
  "output_dir": "/app/outputs/1/20240118_153000_yolo11n-seg",
  "device": "cpu",
  "pretrained": true
}
```

**네트워크:**
- Railway 내부 네트워크 (프라이빗 URL 사용 가능)
- or 공개 URL (HTTPS)

---

#### 6단계: Training Service API 실행

**위치:** `mvp/training/api_server.py` (ultralytics-service 컨테이너)

```python
# FastAPI 앱
app = FastAPI(title=f"Training Service ({FRAMEWORK})")

# 백그라운드 작업 저장
job_status = {}

@app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """학습 시작 (백그라운드)"""

    job_id = request.job_id

    # 1. 이미 실행 중인지 확인
    if job_id in job_status and job_status[job_id]["status"] == "running":
        raise HTTPException(status_code=409, detail=f"Job {job_id} is already running")

    # 2. 백그라운드 태스크로 학습 실행
    background_tasks.add_task(run_training, request)

    # 3. 즉시 응답 반환 (비동기)
    return {
        "job_id": job_id,
        "status": "started",
        "message": f"Training job {job_id} started in background"
    }


def run_training(request: TrainingRequest):
    """실제 학습 실행 (백그라운드 함수)"""

    job_id = request.job_id

    try:
        # 1. 상태 업데이트
        job_status[job_id] = {"status": "running", "error": None}

        # 2. train.py 실행 (subprocess)
        cmd = [
            "python", "/workspace/training/train.py",
            "--job_id", str(job_id),
            "--framework", request.framework,
            "--model_name", request.model_name,
            "--task_type", request.task_type,
            "--dataset_path", request.dataset_path,
            "--dataset_format", request.dataset_format,
            "--epochs", str(request.epochs),
            "--batch_size", str(request.batch_size),
            "--learning_rate", str(request.learning_rate),
            "--output_dir", request.output_dir,
        ]

        # 3. subprocess 실행
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1시간 타임아웃
        )

        # 4. 결과 확인
        if result.returncode == 0:
            job_status[job_id] = {"status": "completed", "error": None}
            print(f"[TrainingService] Job {job_id} completed successfully")
        else:
            job_status[job_id] = {"status": "failed", "error": result.stderr}
            print(f"[TrainingService] Job {job_id} failed: {result.stderr}")

    except Exception as e:
        job_status[job_id] = {"status": "failed", "error": str(e)}
        print(f"[TrainingService] Job {job_id} exception: {e}")
```

**동작:**
1. FastAPI `BackgroundTasks`로 비동기 학습 실행
2. **즉시 응답 반환** (학습 완료 기다리지 않음)
3. 백그라운드에서 `train.py` subprocess 실행
4. 학습 완료 후 상태 업데이트

**핵심:**
- Training Service가 **자체적으로 학습 실행**
- Backend와 **분리된 컨테이너**에서 실행
- ultralytics 라이브러리가 **이미 설치**되어 있음

---

#### 7단계: train.py 실행 (Adapter 호출)

**동작:** 로컬과 동일 (5-6단계와 동일)

```python
# mvp/training/train.py (ultralytics-service 컨테이너 내부)

# Ultralytics Adapter 사용
from adapters import get_adapter

adapter = get_adapter("ultralytics")  # UltralyticsAdapter

result = adapter.train(
    model_name="yolo11n-seg",
    dataset_path="/app/datasets/seg-coco8",
    ...
)
```

**학습 출력:** 로컬과 동일 (PyTorch 학습)

---

#### 8단계: 학습 완료 후 Backend DB 업데이트

**위치:** `mvp/training/train.py` (학습 완료 후)

```python
def update_job_status(job_id, status, result):
    """학습 완료 후 Backend API 호출"""

    # Backend API URL (환경변수)
    backend_url = os.getenv("BACKEND_URL")

    # PATCH 요청으로 DB 업데이트
    response = requests.patch(
        f"{backend_url}/api/v1/training/jobs/{job_id}",
        json={
            "status": status,
            "accuracy": result.get("accuracy"),
            "loss": result.get("loss"),
            "completed_at": datetime.now().isoformat()
        }
    )

    if response.status_code == 200:
        print(f"[Train] Job {job_id} status updated successfully")
    else:
        print(f"[Train] Failed to update job status: {response.text}")
```

**HTTP 요청:**
```http
PATCH https://backend-production-xxxx.up.railway.app/api/v1/training/jobs/6
Content-Type: application/json

{
  "status": "completed",
  "accuracy": 0.89,
  "loss": 0.218,
  "completed_at": "2024-01-18T17:45:00"
}
```

**Backend API:**
```python
@router.patch("/jobs/{job_id}")
def update_training_job(job_id: int, update: TrainingJobUpdate, db: Session = Depends(get_db)):
    """학습 작업 업데이트"""

    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    if update.status:
        job.status = update.status
    if update.accuracy:
        job.accuracy = update.accuracy
    if update.loss:
        job.loss = update.loss
    if update.completed_at:
        job.completed_at = update.completed_at

    db.commit()
    return job
```

---

## 주요 차이점 요약

| 구분 | 로컬 환경 (Subprocess) | 배포 환경 (HTTP API) |
|------|----------------------|-------------------|
| **Backend와 Training** | 같은 컴퓨터 | 별도 컨테이너 (격리) |
| **학습 시작 방식** | Backend가 subprocess 직접 실행 | Backend가 HTTP POST 요청 |
| **train.py 실행** | Backend가 `subprocess.Popen()` | Training Service가 `subprocess.run()` |
| **네트워크** | localhost (프로세스 간 통신) | HTTP/HTTPS (컨테이너 간 통신) |
| **의존성** | Backend에 PyTorch 불필요 | Training Service에 PyTorch 설치됨 |
| **프레임워크** | 모든 프레임워크 같은 환경 | 프레임워크별 격리 (timm, ultralytics 분리) |
| **스케일링** | 단일 머신 (제한적) | 수평 확장 가능 (컨테이너 복제) |
| **에러 처리** | subprocess 예외 처리 | HTTP timeout, connection error |
| **학습 속도** | 빠름 (로컬) | 네트워크 오버헤드 있음 |

---

## 아키텍처 다이어그램

### 로컬 환경 (Subprocess)

```
┌─────────────────────────────────────────────────────────┐
│ 개발자 컴퓨터                                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Frontend (Next.js)                                    │
│  localhost:3000                                         │
│         │                                               │
│         │ POST /training/jobs/6/start                  │
│         ▼                                               │
│  Backend (FastAPI)                                     │
│  localhost:8000                                         │
│         │                                               │
│         │ subprocess.Popen()                           │
│         ▼                                               │
│  Python Process (train.py)                             │
│  ├─ UltralyticsAdapter                                 │
│  ├─ PyTorch training loop                              │
│  └─ Save results                                        │
│         │                                               │
│         │ PATCH /jobs/6 (status=completed)             │
│         ▼                                               │
│  Backend DB (SQLite)                                   │
│  └─ training_jobs.status = 'completed'                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 배포 환경 (HTTP API)

```
┌──────────────────────────────────────────────────────────────────┐
│ Railway Platform                                                  │
├────────────────┬────────────────┬────────────────────────────────┤
│ Frontend       │ Backend        │ Training Services               │
│ (Next.js)      │ (FastAPI)      │                                │
│                │                │  ┌──────────────────────────┐   │
│ https://...    │ https://...    │  │ ultralytics-service      │   │
│                │                │  │ https://...              │   │
│       │ POST   │       │ HTTP   │  │                          │   │
│       │ /start │       │ POST   │  │  ┌────────────────────┐  │   │
│       └────────┼───────┴────────┼──┼─→│ /training/start    │  │   │
│                │                │  │  │ (BackgroundTasks)  │  │   │
│                │                │  │  └────────────────────┘  │   │
│                │                │  │          │               │   │
│                │                │  │          ▼               │   │
│                │                │  │  ┌────────────────────┐  │   │
│                │                │  │  │ subprocess         │  │   │
│                │                │  │  │ python train.py    │  │   │
│                │                │  │  │                    │  │   │
│                │                │  │  │ UltralyticsAdapter │  │   │
│                │                │  │  │ PyTorch training   │  │   │
│                │                │  │  └────────────────────┘  │   │
│                │                │  │          │               │   │
│                │                │  │          │ PATCH         │   │
│                │     ┌──────────┼──┼──────────┘               │   │
│                │     │          │  │  /jobs/6                 │   │
│                │     ▼          │  │  (status=completed)      │   │
│                │  PostgreSQL    │  └──────────────────────────┘   │
│                │  (Railway DB)  │                                │
│                │                │                                │
└────────────────┴────────────────┴────────────────────────────────┘
```

---

## 환경변수 설정

### 로컬 환경

```bash
# mvp/backend/.env
TRAINING_EXECUTION_MODE=subprocess
```

### 배포 환경 (Railway)

**Backend 서비스 환경변수:**
```bash
TRAINING_EXECUTION_MODE=api
TIMM_SERVICE_URL=https://timm-service-production-xxxx.up.railway.app
ULTRALYTICS_SERVICE_URL=https://ultralytics-service-production-xxxx.up.railway.app
HUGGINGFACE_SERVICE_URL=https://huggingface-service-production-xxxx.up.railway.app
```

**Training Service 환경변수:**
```bash
# ultralytics-service
FRAMEWORK=ultralytics
BACKEND_URL=https://backend-production-xxxx.up.railway.app
```

---

## 관련 파일

### Frontend
- `mvp/frontend/components/ExperimentTable.tsx` - 학습 시작 버튼
- `mvp/frontend/hooks/useTrainingProgress.tsx` - 실시간 진행률 훅

### Backend
- `mvp/backend/app/api/training.py` - 학습 시작 API
- `mvp/backend/app/utils/training_manager.py` - 학습 매니저 (subprocess or API)
- `mvp/backend/app/utils/training_client.py` - Training Service HTTP 클라이언트

### Training
- `mvp/training/train.py` - 학습 스크립트
- `mvp/training/api_server.py` - Training Service API
- `mvp/training/adapters/ultralytics_adapter.py` - Ultralytics Adapter
- `mvp/training/adapters/timm_adapter.py` - timm Adapter

---

## 디버깅 팁

### 로컬: 학습이 시작되지 않을 때

**확인:**
```bash
# Backend 로그 확인
cd mvp/backend
../../mvp/backend/venv/Scripts/python.exe -m uvicorn app.main:app --reload

# 로그 예시:
[TrainingManager] Started training job 6 (PID: 12345)
```

**train.py 직접 실행:**
```bash
cd mvp/training
python train.py --job_id 6 --framework ultralytics --model_name yolo11n-seg ...
```

---

### 배포: 학습이 시작되지 않을 때

**Railway 로그 확인:**

**Backend 로그:**
```
Railway Dashboard → Backend Service → Logs

에러 예시:
[ERROR] Training Service not available: Connection timeout
```

**ultralytics-service 로그:**
```
Railway Dashboard → ultralytics-service → Logs

정상:
[TrainingService] Job 6 started in background
[Train] Starting job 6
[Train] Framework: ultralytics
...

에러:
[TrainingService] Job 6 failed: Dataset not found
```

---

## 성능 최적화

### Railway CPU 제한

**문제:** Railway는 CPU만 제공 (GPU 없음)

**해결:**
```python
# mvp/training/adapters/ultralytics_adapter.py

# CPU 최적화 설정
results = model.train(
    ...
    device="cpu",
    workers=4,  # 데이터로더 워커 수
    amp=False,  # AMP 비활성화 (GPU용)
)
```

### 타임아웃 설정

```python
# Training Service API
subprocess.run(
    cmd,
    timeout=3600  # 1시간
)
```

**Railway 시간 제한:**
- Free tier: 500 hours/month
- Hobby tier: Unlimited

---

## 마무리

이 6개 시나리오를 통해 로컬과 배포 환경의 차이를 이해하셨을 겁니다!

**핵심 요약:**
1. **로그인:** 로컬(SQLite) vs 배포(PostgreSQL)
2. **프로젝트 조회:** DB 연결 방식만 다름
3. **실험 조회:** 쿼리 동일, 네트워크만 다름
4. **모델 조회:** 로컬(Python import) vs 배포(HTTP API)
5. **학습 생성:** DB INSERT, 거의 동일
6. **학습 실행:** 로컬(subprocess) vs 배포(HTTP API to Training Service) ← **가장 큰 차이!**
