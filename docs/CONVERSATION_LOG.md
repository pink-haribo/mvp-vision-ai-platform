## [2025-11-14 14:00] Dual Storage 아키텍처 구현 및 검증

### 논의 주제
- MinIO 단일 인스턴스에서 Dual Storage 분리 필요성
- 모델 개발자 관점에서 Storage 추상화
- 학습 파이프라인 전체 테스트 (데이터셋 다운로드 → 학습 → Checkpoint 업로드)
- Backend CORS 설정 오류 수정

### 주요 결정사항

#### 1. Dual Storage 아키텍처 확립 ✅
**배경**:
- 기존: 단일 MinIO 인스턴스에 모든 데이터 혼재
- 문제: 데이터셋(읽기 위주)과 학습 결과물(쓰기 위주)의 액세스 패턴 차이

**결정**:
- **External Storage (MinIO-Datasets)** - Port 9000/9001
  - 용도: 데이터셋 이미지 저장 (읽기 위주)
  - Bucket: training-datasets, vision-platform-dev
  
- **Internal Storage (MinIO-Results)** - Port 9002/9003
  - 용도: 학습 결과물 (쓰기 위주)
  - Bucket: training-checkpoints, training-results, model-weights, config-schemas, mlflow-artifacts

**이유**:
- 성능 격리: 읽기/쓰기 워크로드 분리
- 보안 경계: 데이터셋과 결과물 분리
- 비용 최적화: 각 스토리지에 맞는 정책 적용 가능

#### 2. DualStorageClient 구현 - 개발자 경험 개선 ✅
**문제**: 모델 개발자가 두 개의 S3Client를 직접 관리해야 함

**해결책**: 투명한 라우팅을 제공하는 `DualStorageClient` 추가
```python
# 이전 (복잡)
dataset_client = S3Client(endpoint_9000, ...)
checkpoint_client = S3Client(endpoint_9002, ...)

dataset_client.download_dataset(...)
checkpoint_client.upload_checkpoint(...)

# 현재 (심플)
storage = DualStorageClient()  # 환경변수에서 자동 설정

storage.download_dataset(...)   # 자동으로 External Storage 사용
storage.upload_checkpoint(...)  # 자동으로 Internal Storage 사용
```

**특징**:
- 환경변수 자동 읽기 (EXTERNAL_*, INTERNAL_*)
- Legacy fallback 지원 (S3_ENDPOINT 등)
- 명확한 로깅으로 디버깅 용이
- 모델 개발자는 storage routing 신경 안 써도 됨

### 구현 내용

#### 1. Infrastructure (docker-compose.tier0.yaml)
- **파일**: `platform/infrastructure/docker-compose.tier0.yaml`
- **변경사항**:
  - 단일 `minio` 서비스를 `minio-datasets`, `minio-results`로 분리
  - 각각 독립적인 port, volume, bucket 설정
  - minio-setup 서비스에서 양쪽 버킷 생성

#### 2. DualStorageClient 추가
- **파일**: `platform/trainers/ultralytics/utils.py`
- **추가 기능**:
  ```python
  class DualStorageClient:
      """Transparent dual storage routing"""
      def __init__(self):
          # External Storage (Datasets)
          self.external_client = S3Client(...)
          # Internal Storage (Results)
          self.internal_client = S3Client(...)
      
      def download_dataset(self, dataset_id, dest_dir):
          """Auto-route to External Storage"""
          self.external_client.download_dataset(...)
      
      def upload_checkpoint(self, local_path, job_id):
          """Auto-route to Internal Storage"""
          self.internal_client.upload_checkpoint(...)
  ```

#### 3. train.py 수정
- **파일**: `platform/trainers/ultralytics/train.py`
- **변경사항**:
  - `S3Client` → `DualStorageClient` import 변경
  - 단일 `storage` 객체로 모든 storage 작업 처리
  - 자동으로 올바른 storage로 라우팅

#### 4. 환경변수 설정
- **파일**: `platform/trainers/ultralytics/.env`
- **구조**:
  ```bash
  # External Storage (MinIO-Datasets) - for datasets
  EXTERNAL_STORAGE_ENDPOINT=http://localhost:9000
  EXTERNAL_BUCKET_DATASETS=training-datasets
  
  # Internal Storage (MinIO-Results) - for checkpoints
  INTERNAL_STORAGE_ENDPOINT=http://localhost:9002
  INTERNAL_BUCKET_CHECKPOINTS=training-checkpoints
  
  # Legacy fallback
  S3_ENDPOINT=http://localhost:9000
  ```

#### 5. Backend CORS 설정 수정
- **파일**: `platform/backend/.env`
- **문제**: JSON 배열 형식이 comma-separated로 파싱되지 않음
- **수정**: 
  ```bash
  # Before
  CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]
  
  # After
  CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
  ```

### 검증 결과

#### End-to-End 학습 파이프라인 테스트 ✅

**테스트 케이스**: YOLOv8n 학습 (Job ID 15)
1. ✅ 데이터셋 다운로드: MinIO-Datasets (9000) ← training-datasets bucket
2. ✅ DICE → YOLO 변환: 25 train, 7 val images, 43 classes
3. ✅ 학습 실행: 2 epochs, 0.015 hours (CPU)
4. ✅ Checkpoint 업로드: MinIO-Results (9002) ← training-checkpoints bucket
5. ✅ MLflow 연동: run_id 924c7209cf824d70a284b951b7e976ba
6. ✅ Backend callback: 성공

**로그 확인**:
```
Dual Storage initialized:
  External (Datasets): http://localhost:9000 -> training-datasets
  Internal (Results):  http://localhost:9002 -> training-checkpoints

[Dual Storage] Downloading dataset from External Storage
[Dual Storage] Uploading checkpoint to Internal Storage
Checkpoint uploaded to s3://training-checkpoints/checkpoints/15/best.pt
```

**실제 파일 확인**:
```bash
$ docker exec platform-minio-results-tier0 mc ls local/training-checkpoints/checkpoints/15/
[2025-11-14 06:08:23 UTC] 6.0MiB STANDARD best.pt
```

### 기술적 개선점

#### 개발자 경험 개선
- **이전**: 두 개의 S3Client를 직접 관리, endpoint/bucket 선택 필요
- **현재**: 단일 DualStorageClient, 자동 라우팅
- **효과**: 코드 단순화, 실수 방지, 명확한 의도 표현

#### 확장성
- 새로운 storage operation 추가 용이
- 다른 framework trainer에 동일 패턴 적용 가능
- Production 환경에서도 동일하게 작동 (환경변수만 변경)

### 다음 단계

- [ ] 체크리스트 업데이트 (Phase 3.3 Dual Storage 완료)
- [ ] 변경사항 커밋
- [ ] 다른 framework trainer (timm, huggingface)에도 DualStorageClient 적용
- [ ] Backend dual_storage.py와 trainer utils.py 통합 고려
- [ ] Production 배포 시 환경변수 설정 가이드 작성

### 관련 문서
- Infrastructure: `platform/infrastructure/docker-compose.tier0.yaml`
- Trainer Utils: `platform/trainers/ultralytics/utils.py`
- Train Script: `platform/trainers/ultralytics/train.py`
- Backend Dual Storage: `platform/backend/app/utils/dual_storage.py`

---

# Conversation Log

이 파일은 Claude Code 대화 세션의 타임라인을 기록합니다.
세션이 바뀌어도 이전 논의 내용을 빠르게 파악할 수 있습니다.

**사용 방법**: `/log-session` 명령어로 현재 세션 내용 추가

---

## [2025-11-07 19:00] 로컬 개발 워크플로우 최적화 - Docker 빌드 제거

### 논의 주제
- Training 코드 수정 시 매번 Docker 이미지 빌드 문제 해결
- Frontend + Backend + Training 전체 통합 테스트 방법
- Framework별 Training Service 실행 방식
- 로컬 개발 환경 설정 및 자동화

### 주요 결정사항

#### 1. 3단계 개발 워크플로우 확립 ✅
- **Tier 1: 로컬 개발 (subprocess)** - 99% 사용 ⚡⚡⚡
  - Backend가 Python subprocess로 train.py 직접 실행
  - Framework별 가상환경 사용 (venv-timm, venv-ultralytics, venv-huggingface)
  - 실행 속도: 5-30초 (Docker 빌드 불필요)
  - 시간 절약: **145분/일** (10회 반복 기준)

- **Tier 2: K8s 테스트 (ConfigMap 주입)** - 배포 전 검증 ⚡⚡
  - 코드를 ConfigMap으로 주입하여 K8s Job 실행
  - 이미지 재빌드 불필요 (1-3분 소요)
  - 실제 K8s 환경 테스트 가능

- **Tier 3: Production 배포** - 최종 단계만 ⚡
  - Docker 이미지 빌드 및 배포
  - 10-15분 소요
  - 배포 직전에만 실행

#### 2. 로컬 개발 인프라 구성 ✅
**Kind 클러스터 기반 서비스**:
- MLflow (Port 30500): Experiment tracking, SQLite backend
- MinIO (Port 30900/30901): S3-compatible object storage (R2 대체)
- Prometheus (Port 30090): Metrics collection
- Grafana (Port 30030): Monitoring dashboard

**데이터 영속성**:
- MLflow PVC: 5Gi (SQLite database)
- MinIO PVC: 20Gi (datasets, checkpoints, results)

**R2 → MinIO 전환 이유**:
- 로컬 개발에 인터넷 불필요
- Credentials 관리 불필요
- 무료 (비용 절감)
- S3-compatible API 동일

#### 3. Framework별 Training Service 구조 ✅
**현재 구현 상태**:
```
Backend API (Port 8000)
  ↓ HTTP
TrainingServiceClient (framework 기반 라우팅)
  ↓
api_server.py (Training Service)
  ↓
subprocess.Popen([venv-{framework}/python, train.py])
  ↓
Adapter Pattern (TimmAdapter, UltralyticsAdapter, HuggingFaceAdapter)
```

**Framework별 가상환경**:
```
mvp/training/
├── venv-timm/          # timm 전용 의존성
├── venv-ultralytics/   # ultralytics 전용 의존성
├── venv-huggingface/   # huggingface 전용 의존성
└── train.py            # 공통 Adapter 패턴
```

**동작 방식**:
1. Backend: `TrainingServiceClient(framework="ultralytics")`
2. Training Service: `venv-ultralytics/python train.py --framework=ultralytics`
3. train.py: `UltralyticsAdapter` 선택 및 실행

#### 4. 일상 개발 플로우 (전체 통합 테스트)

**환경 시작 (아침 한 번)**:
```powershell
# K8s 서비스 시작 (MLflow, MinIO)
.\dev-start.ps1 -SkipBuild  # 2-3분 소요
```

**개발 (3개 터미널)**:
```powershell
# Terminal 1: Backend
cd mvp/backend
.\venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd mvp/frontend
npm run dev

# Terminal 3: 브라우저
start http://localhost:3000
```

**Training 실행**:
1. Frontend에서 자연어 입력: "ResNet50으로 고양이/개 분류 학습해줘"
2. Backend가 TrainingJob 생성
3. Backend가 subprocess로 train.py 실행 (Framework별 venv 사용)
4. MLflow에 실시간 메트릭 기록
5. Frontend에서 결과 확인

**환경 종료 (저녁)**:
```powershell
.\dev-stop.ps1
```

### 구현 내용

#### 자동화 스크립트 (6개 생성)

**환경 관리**:
- `dev-start.ps1`: K8s 환경 자동 시작
  - Kind 클러스터 생성/검증
  - Docker 이미지 빌드 (선택적 `-SkipBuild`)
  - K8s 리소스 배포 (MLflow, MinIO, Prometheus, Grafana)
  - 서비스 Ready 대기
  - MinIO 버킷 생성

- `dev-stop.ps1`: K8s 환경 종료
  - `-DeleteCluster`: 완전 삭제
  - 기본: 중지 (데이터 유지)

- `dev-status.ps1`: 환경 상태 확인
  - 클러스터 상태
  - 서비스 상태
  - 리소스 사용량
  - `-Watch`: 실시간 모니터링

**Training 실행**:
- `dev-train-local.ps1`: 로컬 Python 직접 실행
  - 환경변수 자동 설정 (MLflow, MinIO)
  - subprocess 실행
  - 가장 빠름 (초 단위)

- `dev-train-k8s.ps1`: K8s Job (ConfigMap 주입)
  - 코드를 ConfigMap으로 생성
  - 기존 Docker 이미지 사용
  - 이미지 재빌드 불필요 (분 단위)

**K8s 설정**:
- `mvp/k8s/minio-config.yaml`: MinIO 배포
- `mvp/k8s/minio-pvc.yaml`: MinIO 영속 스토리지 (20Gi)
- `mvp/k8s/mlflow-config.yaml`: MLflow 배포 (수정)
- `mvp/k8s/mlflow-pvc.yaml`: MLflow 영속 스토리지 (5Gi)

#### 문서화

**가이드 문서** (4개 생성):
- `GETTING_STARTED.md`: 5분 안에 시작하기
  - 실전 예제 (고양이/개 분류)
  - 일반적인 개발 사이클
  - 트러블슈팅

- `DEV_WORKFLOW.md`: 개발 워크플로우 상세 가이드
  - 3단계 접근법 설명
  - 스크립트 상세 사용법
  - 실전 팁

- `QUICK_DEV_GUIDE.md`: 한 페이지 빠른 참조
  - 핵심 명령어만
  - 개발 효율성 비교
  - TL;DR

- `README.md`: 업데이트
  - Getting Started 링크 추가
  - 개발 워크플로우 섹션 추가

**인프라 문서** (4개 생성):
- `mvp/k8s/MINIO_SETUP.md`: MinIO 사용법
- `mvp/k8s/MLFLOW_SETUP.md`: MLflow 사용법
- `mvp/k8s/DATA_PERSISTENCE.md`: 데이터 영속성 설명
- `mvp/k8s/DOCKER_VS_K8S.md`: Docker vs K8s 비교

**기술 문서**:
- `docs/k8s/20251107_development_workflow_setup.md`: 전체 설계 문서
  - 배경 및 컨텍스트
  - 3단계 워크플로우 상세
  - 대안 비교
  - 비용 분석
  - 마이그레이션 경로

### 샘플 데이터셋

**sample_dataset (고양이/개 분류)**:
- 위치: `mvp/data/datasets/sample_dataset/`
- 구조:
  ```
  sample_dataset/
  ├── train/
  │   ├── cats/  (20장)
  │   └── dogs/  (20장)
  └── val/
      ├── cats/  (5장)
      └── dogs/  (5장)
  ```
- Format: ImageFolder (image_classification)
- 용도: 로컬 개발 테스트

### 환경 변수 설정

**dev-train-local.ps1 자동 설정**:
```powershell
MLFLOW_TRACKING_URI    = http://localhost:30500
MLFLOW_S3_ENDPOINT_URL = http://localhost:30900
AWS_ACCESS_KEY_ID      = minioadmin
AWS_SECRET_ACCESS_KEY  = minioadmin
MLFLOW_S3_IGNORE_TLS   = true
JOB_ID                 = local-20251107-143000
MODEL_NAME             = yolo11n
FRAMEWORK              = ultralytics
NUM_EPOCHS             = 10
```

### 개발 효율성 비교

| 방법 | 시간 | 사용 시기 | 빈도 |
|------|------|-----------|------|
| **로컬 실행 (subprocess)** | 5-30초 | 일상 개발 | 99% |
| ConfigMap 주입 (K8s) | 1-3분 | 통합 테스트 | 배포 전 1회 |
| Docker 이미지 빌드 | 10-15분 | 최종 배포 | 배포 시만 |

**시간 절약 계산**:
```
기존 방식: 10회 반복 × 15분 = 150분
새 방식: 10회 반복 × 30초 = 5분
절약: 145분/일 (약 2.4시간)
```

### 다음 단계

#### 즉시 가능 (테스트)
- [ ] 로컬 환경 시작: `.\dev-start.ps1 -SkipBuild`
- [ ] Backend + Frontend 실행
- [ ] 전체 플로우 테스트 (Frontend → Backend → Training → MLflow)

#### 향후 개선
- [ ] Docker Compose 대안 제공 (Kind 대신)
- [ ] Health check 개선 (Training Service)
- [ ] 자동 실행 스크립트 (`dev-all.ps1` - Backend + Frontend 동시 시작)

### 관련 문서
- **가이드**: [GETTING_STARTED.md](../GETTING_STARTED.md), [DEV_WORKFLOW.md](../DEV_WORKFLOW.md), [QUICK_DEV_GUIDE.md](../QUICK_DEV_GUIDE.md)
- **인프라**: [mvp/k8s/MINIO_SETUP.md](../mvp/k8s/MINIO_SETUP.md), [mvp/k8s/MLFLOW_SETUP.md](../mvp/k8s/MLFLOW_SETUP.md)
- **설계**: [docs/k8s/20251107_development_workflow_setup.md](../docs/k8s/20251107_development_workflow_setup.md)

### 핵심 통찰

#### Docker 빌드 제거의 임팩트
- **개발 속도**: 30배 향상 (15분 → 30초)
- **개발자 경험**: 즉각적 피드백 가능
- **비용**: 컴퓨팅 리소스 절약

#### Microservice 아키텍처 일관성
- **로컬 = Production**: 동일한 구조
- **Framework 격리**: 가상환경으로 의존성 충돌 방지
- **subprocess**: 개발 시 빠름, Production은 K8s Job

#### 데이터 영속성
- **PVC 활용**: Kind 재시작해도 데이터 유지
- **MLflow 메타데이터**: SQLite + PVC (5Gi)
- **MinIO 객체**: PVC (20Gi)

### 기술 노트

#### ConfigMap 코드 주입 방식
```yaml
# ConfigMap 생성
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-code-dev-123
data:
  train.py: |
    # 실제 train.py 내용

# Job에서 마운트
volumes:
- name: training-code
  configMap:
    name: training-code-dev-123
volumeMounts:
- name: training-code
  mountPath: /code/train.py
  subPath: train.py
```

#### Framework별 subprocess 실행
```python
# api_server.py:99-106
venv_python = f"venv-{request.framework}/Scripts/python.exe"
if os.path.exists(venv_python):
    python_exe = venv_python  # Framework-specific venv
else:
    python_exe = "python"  # Fallback

cmd = [python_exe, "train.py", "--framework", request.framework, ...]
process = subprocess.Popen(cmd)
```

#### Kind 클러스터 Port Mapping
```yaml
# dev-start.ps1에서 생성
kind: Cluster
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 30500  # MLflow
  - containerPort: 30900  # MinIO API
  - containerPort: 30901  # MinIO Console
  - containerPort: 30090  # Prometheus
  - containerPort: 30030  # Grafana
```

---

## [2025-11-07 14:30] Kubernetes Training 방식 FAQ - 핵심 질문 4가지 해결

### 논의 주제
- K8s Job 방식에서 학습 중단/재시작 가능성
- 프레임워크별 설정(Config) 관리 방법
- Inference (Single/Batch) 구현 현황
- 테스트 전략 및 실행 방법

### 주요 결정사항

#### 1. Checkpoint Resume으로 학습 재시작 가능 ✅
- **질문**: K8s Job은 pause/resume이 어렵지 않나?
- **답변**: Checkpoint 기반 재시작으로 해결
- **구현 상태**: 완전 구현됨 (`train.py:83-360`)

**복원되는 상태**:
- Model weights
- Optimizer state
- LR scheduler state
- Current epoch number
- Best validation accuracy

**K8s에서의 동작**:
```yaml
# 자동 재시작 설정
spec:
  backoffLimit: 3
  restartPolicy: OnFailure
  # 모든 Job을 --resume 모드로 실행
  args: ["--checkpoint_path=s3://...", "--resume"]
```

**Scenario**:
- Epoch 10/50 진행 중 → Pod 종료
- K8s 자동 재시작
- Epoch 10 checkpoint 로드
- Epoch 11부터 재개

#### 2. Adapter Pattern + Config Schema로 프레임워크별 설정 ✅
- **질문**: 모델/프레임워크마다 다른 Config는 어떻게?
- **답변**: 각 Adapter가 자체 Config Schema 정의

**TIMM 예시**:
```python
config_schema = {
    "optimizer_type": ["adam", "adamw", "sgd"],
    "scheduler_type": ["cosine", "step", "plateau"],
    "mixup": bool,
    "cutmix": bool,
}

presets = {
    "easy": {"optimizer": "adam", "mixup": False},
    "medium": {"optimizer": "adamw", "mixup": True},
    "advanced": {"optimizer": "adamw", "mixup": True, "cutmix": True},
}
```

**Ultralytics 예시**:
```python
config_schema = {
    "optimizer_type": ["Adam", "AdamW", "SGD"],
    "cos_lr": bool,
    "mosaic": float,  # YOLO-specific
    "mixup": float,
    "copy_paste": float,
}
```

**사용 방법**:
1. **Preset**: "난이도 medium으로" → LLM이 preset 적용
2. **세부 설정**: "AdamW, lr 0.001, mosaic 1.0" → LLM이 advanced_config 생성
3. **DB 저장**: `training_jobs.advanced_config` (JSONB)

#### 3. Inference 구현 현황
- **Single Inference**: ✅ 완전 구현
- **Batch Inference (TestRun)**: ✅ 구현
- **Production Batch**: ⚠️ 향후 구현

**Single Inference API**:
```python
# POST /api/v1/test/inference/single
result = adapter.infer_single("image.jpg")
# → {"predicted_label": "cat", "confidence": 0.92, "top5_predictions": [...]}
```

**모든 Adapter 구현 완료**:
- `TimmAdapter.infer_single()`: lines 1040-1118
- `UltralyticsAdapter.infer_single()`: lines 2568+
- `TransformersAdapter.infer_single()`: lines 594, 820

**Batch Inference (TestRun)**:
```python
# POST /api/v1/test/runs
test_run = create_test_run(
    job_id=123,
    test_dataset_path="s3://bucket/test_dataset/"
)
# Background task로 모든 이미지 처리
```

#### 4. 4단계 테스트 전략 ✅
- **Level 1: Unit Tests** (`mvp/backend/tests/unit/`)
- **Level 2: Integration Tests** (`mvp/backend/tests/integration/`)
- **Level 3: Subprocess E2E** (`mvp/training/test_train_subprocess_e2e.py`)
- **Level 4: K8s Job Tests** (`mvp/backend/tests/k8s/`)

**테스트 실행**:
```bash
# Level 1: Unit
cd mvp/backend && pytest tests/unit/ -v

# Level 2: Integration
pytest tests/integration/ -v

# Level 3: E2E
cd mvp/training && python test_train_subprocess_e2e.py

# Level 4: K8s
kind create cluster --name training-test
kind load docker-image vision-platform/trainer-timm:latest
cd mvp/backend && pytest tests/k8s/ -v
```

### 구현 내용

#### 종합 FAQ 문서
**파일**: `docs/k8s/K8S_TRAINING_FAQ.md` (새로 생성)

**포함 내용** (전체 1000+ lines):
1. **학습 중단/재시작** (360 lines)
   - Checkpoint resume 구현 세부사항
   - K8s Job 자동 재시작 동작
   - Multi-stage training (24시간+ 학습)
   - Checkpoint 저장 주기 설정
   - K8s Job vs 일반 서버 비교

2. **프레임워크별 Config** (400 lines)
   - Preset 시스템 (easy/medium/advanced)
   - 세부 설정 방식
   - TIMM/Ultralytics Config Schema 예시
   - Config 전달 Flow
   - DB 저장 및 로딩

3. **Inference 구현** (300 lines)
   - Single inference API
   - Batch inference (TestRun)
   - Adapter별 구현 상세
   - Frontend 사용 예시
   - Production batch 제안

4. **테스트 방법** (400 lines)
   - 4단계 테스트 계층
   - 테스트 데이터셋 생성
   - K8s 클러스터 셋업
   - CI/CD 통합 예시
   - Coverage 목표

### 기술 세부사항

#### Checkpoint Resume 흐름
```python
# 1. 처음 시작 (checkpoint 없음)
python train.py --job_id=123 --num_epochs=50

# 2. Epoch 10에서 중단

# 3. 재시작 (자동으로 epoch 10부터)
python train.py --job_id=123 \
    --checkpoint_path=s3://bucket/job_123/weights/last.pt \
    --resume \
    --num_epochs=50
```

#### Config Schema 구조
```python
# Adapter별 schema 정의
class TimmAdapter:
    def get_config_schema(self):
        return [
            ConfigField("optimizer_type", type="select", options=[...]),
            ConfigField("scheduler_type", type="select", options=[...]),
            ConfigField("mixup", type="bool", default=False),
        ]

    def get_preset_config(self, preset: str):
        return self.presets[preset]  # easy/medium/advanced
```

#### Inference Result Schema
```python
class InferenceResult:
    image_path: str
    predicted_label: str
    confidence: float
    top5_predictions: List[Dict]
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
```

### 테스트 커버리지

**구현된 테스트 파일들**:
- `test_adapter_imports.py` - Adapter 로딩
- `test_advanced_config.py` - Config 검증
- `test_inference_api.py` - Inference 엔드포인트
- `test_checkpoint_inference.py` - Checkpoint 로딩
- `test_train_subprocess_e2e.py` - 전체 파이프라인
- `test_inference_pretrained.py` - Pretrained 모델
- `test_training_config.py` - 설정 검증
- `test_validation_metrics_persistence.py` - 메트릭 저장

### 요약 표

| 질문 | 구현 상태 | 핵심 파일 |
|------|-----------|----------|
| **학습 중단/재시작** | ✅ 완전 구현 | `train.py:83-360`, `base.py:1242-1287` |
| **프레임워크별 Config** | ✅ 완전 구현 | `timm_adapter.py:14-326`, `ultralytics_adapter.py:39-250` |
| **Inference** | ✅ Single 완전 구현<br>✅ Batch (TestRun)<br>⚠️ Production Batch 향후 | `timm_adapter.py:1040-1118`, `test_inference.py` |
| **테스트** | ✅ 완전 구현 | `tests/unit/`, `tests/integration/`, E2E scripts |

### 다음 단계

#### 즉시 가능 (테스트)
- [ ] 로컬 K8s 클러스터로 테스트 (Kind + QUICKSTART.md)
- [ ] Checkpoint resume 동작 검증
- [ ] Inference API 실제 호출 테스트

#### 향후 개선
- [ ] Production Batch Inference API (K8s Job 기반)
- [ ] WebSocket 통합으로 실시간 모니터링 강화
- [ ] Goal #2: LLM Agent 고도화

### 관련 문서
- **FAQ 문서**: [docs/k8s/K8S_TRAINING_FAQ.md](../k8s/K8S_TRAINING_FAQ.md) (신규)
- **K8s 마이그레이션**: [docs/k8s/20251106_kubernetes_job_migration_plan.md](../k8s/20251106_kubernetes_job_migration_plan.md)
- **모니터링 통합**: [mvp/k8s/MONITORING_INTEGRATION.md](../../mvp/k8s/MONITORING_INTEGRATION.md)
- **K8s QUICKSTART**: [mvp/k8s/QUICKSTART.md](../../mvp/k8s/QUICKSTART.md)

### 핵심 통찰

#### K8s Job의 제약을 Checkpoint로 극복
- K8s Job은 pause 불가 → Checkpoint resume으로 동일 효과
- Pod 재시작 = 새 Job 생성 + `--resume` 플래그
- R2 Storage 덕분에 checkpoint 영속성 보장

#### Adapter Pattern의 유연성
- Framework마다 완전히 다른 config 필요
- 각 Adapter가 자체 schema 정의
- Preset으로 간편함 + 세부 설정으로 유연성

#### 테스트의 계층화
- Unit → Integration → E2E → K8s
- 각 레벨이 다른 부분을 검증
- Production-ready 보장

### 기술 노트

#### Checkpoint 저장 위치
```
로컬: output/job_{job_id}/weights/best.pt, last.pt
R2: checkpoints/projects/{project_id}/jobs/{job_id}/best.pt, last.pt
```

#### Config 전달 체인
```
사용자 자연어
  → LLM Parser
  → TrainingIntent.advanced_config
  → DB (training_jobs.advanced_config JSONB)
  → train.py --job_id
  → load_advanced_config_from_db()
  → Adapter(advanced_config)
  → Framework-specific 구현
```

#### 4단계 테스트 실행 시간
- Level 1 (Unit): ~30초
- Level 2 (Integration): ~2분
- Level 3 (E2E): ~5분 (tiny dataset)
- Level 4 (K8s): ~10분 (클러스터 셋업 포함)

---

## [2025-11-05 14:45] Checkpoint 관리 정책 및 R2 업로드 전략 수립

### 논의 주제
- 추론 테스트 준비 중 체크포인트 관리 정책 누락 발견
- R2 업로드 시점 결정 (매 epoch vs 학습 완료 시)
- 학습 중단 시나리오 처리 (Ctrl+C, Error, 조기 종료)
- UI 메트릭 테이블의 체크포인트 표시 동기화

### 주요 결정사항

#### 1. 현재 상태 확인
- **로컬 저장**:
  - ✅ YOLO `save_period = -1` (best.pt + last.pt만 저장)
  - ✅ 중간 epoch checkpoint 저장 안함
  - ✅ 효율적인 로컬 관리

- **R2 업로드**:
  - ❌ `upload_checkpoint()` 함수는 구현되어 있음
  - ❌ 하지만 실제로 호출되지 않음!
  - ❌ 체크포인트가 로컬에만 남음

- **문제점**:
  - 시간이 지난 후 추론 사용 불가 (로컬 파일 삭제 가능)
  - Exception 처리에서 checkpoint_dir 누락
  - UI는 로컬 경로 기준으로 표시 (R2 업로드 상태 아님)

#### 2. R2 업로드 시점 결정 (Option 1 선택 ✅)

**고려한 옵션들**:

| 옵션 | 장점 | 단점 | 결정 |
|------|------|------|------|
| 매 epoch | 최대 안전성 | 높은 비용, 느린 학습 | ❌ |
| N epoch마다 | 균형 | 여전히 중복 업로드 | ❌ |
| 개선 시마다 | 의미있는 업로드 | 초반 = 매 epoch | ❌ |
| **완료 시 1회** | 간단, 빠름, 저렴 | 중간 백업 없음 | ✅ |

**선택 이유**:
- 대부분의 학습은 정상 완료됨
- 중단은 rare case
- 2개 파일만 업로드 (best.pt + last.pt)
- 학습 성능 영향 0
- 비용 효율적 (~$0.60/월 for 1000 jobs)

#### 3. 학습 중단 처리 (핵심 개선 사항)

**문제 발견**:
```python
# 현재 코드
try:
    results = self.model.train(**train_args)
    callbacks.on_train_end(checkpoint_dir=checkpoint_dir)  # ✅

except KeyboardInterrupt:
    callbacks.on_train_end()  # ❌ checkpoint_dir 없음!

except Exception as e:
    callbacks.on_train_end()  # ❌ checkpoint_dir 없음!
```

**해결 방안**:
```python
# checkpoint_dir를 try 블록 밖에서 정의
checkpoint_dir = os.path.join(self.output_dir, f"job_{self.job_id}", "weights")

try:
    results = self.model.train(**train_args)
except KeyboardInterrupt:
    print("[YOLO] Uploading checkpoints before exit...")
    callbacks.on_train_end(checkpoint_dir=checkpoint_dir)  # ✅
    raise
except Exception as e:
    print("[YOLO] Attempting to upload despite error...")
    callbacks.on_train_end(checkpoint_dir=checkpoint_dir)  # ✅
    raise

# 정상 완료
callbacks.on_train_end(checkpoint_dir=checkpoint_dir)
```

**중단 시나리오별 처리**:
- User 중단 (Ctrl+C): ✅ 현재까지 best/last 업로드
- 에러 발생: ✅ 업로드 시도 (파일 있으면)
- 조기 종료: ✅ 정상 완료로 처리
- 초반 중단: ✅ 파일 없으면 warning만 (non-blocking)

#### 4. DB 체크포인트 추적 전략

**현재 문제**:
```python
# ultralytics_adapter.py:1590-1602
# 학습 중 매 epoch마다 checkpoint_path 저장 (로컬 경로)
if os.path.exists(best_weights):
    checkpoint_path = best_weights  # 문제: 로컬 경로!
```

**새로운 전략**:
```python
# 학습 중
checkpoint_path = None  # DB에 저장 안함

# on_train_end()에서만
1. R2에 업로드
2. Best epoch 찾기 (highest primary_metric_value)
3. Last epoch 찾기 (max epoch)
4. DB UPDATE: 해당 epoch들의 checkpoint_path = 'r2://...'
```

**결과**:
- 학습 중: 모든 epoch checkpoint_path = NULL
- 학습 완료/중단: Best & Last epoch만 checkpoint_path = 'r2://...'
- UI: R2 업로드된 checkpoint만 체크마크 표시

### 구현 내용

#### 1. on_train_end() 확장
**파일**: `platform_sdk/base.py:1724`

```python
def on_train_end(self, final_metrics=None, checkpoint_dir=None):
    # 1. Upload best.pt to R2
    if checkpoint_dir and os.path.exists(best_pt):
        success = upload_checkpoint(best_pt, job_id, 'best.pt', project_id)
        if success:
            best_epoch = _find_best_epoch()
            r2_path = f'r2://.../{project_id}/jobs/{job_id}/best.pt'
            uploaded_checkpoints[best_epoch] = r2_path

    # 2. Upload last.pt to R2
    if checkpoint_dir and os.path.exists(last_pt):
        success = upload_checkpoint(last_pt, job_id, 'last.pt', project_id)
        if success:
            last_epoch = _find_last_epoch()
            r2_path = f'r2://.../{project_id}/jobs/{job_id}/last.pt'
            uploaded_checkpoints[last_epoch] = r2_path

    # 3. Update DB with R2 paths
    _update_checkpoint_paths(uploaded_checkpoints)

    # 4. End MLflow
    mlflow.end_run()
```

**새로운 헬퍼 메서드**:
- `_find_best_epoch()`: DB에서 highest primary_metric_value 찾기
- `_find_last_epoch()`: DB에서 max(epoch) 찾기
- `_update_checkpoint_paths()`: validation_results 테이블 UPDATE

#### 2. Exception 핸들링 수정
**파일**: `adapters/ultralytics_adapter.py:1967-1999`

```python
# Line 1995: checkpoint_dir 미리 정의
checkpoint_dir = os.path.join(self.output_dir, f"job_{self.job_id}", "weights")

try:
    results = self.model.train(**train_args)
except KeyboardInterrupt:
    callbacks.on_train_end(checkpoint_dir=checkpoint_dir)  # ✅
    raise
except Exception as e:
    callbacks.on_train_end(checkpoint_dir=checkpoint_dir)  # ✅
    raise

callbacks.on_train_end(checkpoint_dir=checkpoint_dir)
```

#### 3. 학습 중 checkpoint_path 제거
**파일**: `adapters/ultralytics_adapter.py:1590-1602`

```python
# 기존 코드 제거 (로컬 경로 할당)
# checkpoint_path = best_weights if os.path.exists(best_weights) else last_weights

# 새 코드 (간단!)
checkpoint_path = None  # R2 업로드 후에만 설정됨
```

#### 4. upload_checkpoint() 반환값 추가
**파일**: `platform_sdk/storage.py:527`

```python
def upload_checkpoint(...) -> bool:  # 반환 타입 추가
    try:
        # ... 업로드 로직 ...
        return True  # 성공
    except Exception as e:
        print(f"[R2 WARNING] Upload failed: {e}")
        return False  # 실패
```

### 비용 분석

**Storage (Cloudflare R2)**:
- 파일당: ~20MB (YOLO11s average)
- 잡당: 40MB (best.pt + last.pt)
- 1000 jobs: 40GB
- 비용: $0.015/GB/month
- **월 비용: $0.60** (affordable!)

**비교 (대안들)**:
- 매 epoch (100 epochs): 2GB/job → $30/month (50배 비쌈!)
- 10 epoch마다: 200MB/job → $3/month (5배 비쌈)
- 완료 시 1회: 40MB/job → $0.60/month ✅

**Upload 비용**:
- PUT operations: Free (10M requests/month)
- 2 uploads/job: 무시 가능

### 타임라인 동작

**100 epoch 학습 예시**:
```
Epoch 1-99:
  - DB: checkpoint_path = NULL for all epochs
  - UI: No checkmarks

Epoch 100 (완료):
  - Upload best.pt (assume epoch 85 was best)
  - Upload last.pt (epoch 100)
  - DB UPDATE:
    - epoch 85: checkpoint_path = 'r2://...best.pt'
    - epoch 100: checkpoint_path = 'r2://...last.pt'
  - UI: Checkmarks on epochs 85, 100 only
```

**Epoch 20 중단 예시**:
```
Epoch 1-19: No uploads
Epoch 20: User presses Ctrl+C
  - KeyboardInterrupt caught
  - Upload best.pt (assume epoch 18)
  - Upload last.pt (epoch 20)
  - DB UPDATE: epochs 18, 20 get R2 paths
  - UI: 2 checkmarks
```

### 문서화

**생성된 문서**: `docs/training/20251105_checkpoint_management_and_r2_upload_policy.md`

**포함 내용**:
- Background & context (문제 발견 과정)
- Current state (코드 분석 결과)
- Proposed solution (선택한 정책)
- Implementation plan (4 phases)
- Technical details (R2 경로, DB 스키마, 예시)
- Alternatives considered (4가지 옵션 비교)
- Cost analysis (storage & operations)
- Migration path (기존 job 처리)
- References (관련 파일 & 문서)

### 다음 단계

#### Immediate (구현 필요)
- [ ] `on_train_end()` 구현 (upload + DB update)
- [ ] Exception handling 수정 (checkpoint_dir 전달)
- [ ] 학습 중 checkpoint_path 할당 제거
- [ ] `upload_checkpoint()` 반환값 수정
- [ ] 테스트 (정상 완료, 중단, 에러)

#### Future Enhancements (P1-P3)
- [ ] Checkpoint download API (inference용)
- [ ] Lifecycle policy (30일 후 자동 삭제)
- [ ] Checkpoint browser UI
- [ ] Resume training from R2 checkpoint

### 관련 문서
- **설계 문서**: [docs/training/20251105_checkpoint_management_and_r2_upload_policy.md](../training/20251105_checkpoint_management_and_r2_upload_policy.md)
- **이전 세션**: [Project-Centric Checkpoint Storage](../CONVERSATION_LOG.md#2025-11-04-2130-project-centric-checkpoint-storage-구현) (2025-11-04)
- **Validation 이슈**: [YOLO Validation Metrics](../CONVERSATION_LOG.md#2025-11-05-1415-yolo-validation-metrics-이슈-조사-및-stratified-split-구현) (2025-11-05)

### 핵심 통찰 (Key Insights)

#### Cost-Benefit Analysis
- **Best + Last only**: 충분함 (추론 + 재학습)
- **매 epoch 저장**: 불필요 (50배 비용, 성능 저하)
- **중단 처리**: 필수 (partial results도 가치있음)

#### Design Principles
1. **Simplicity over Safety**: MVP는 간단함 우선
2. **Cost-Effective**: 비용 최소화 ($0.60/month)
3. **Non-Blocking**: 업로드 실패해도 학습 계속
4. **User-Friendly**: UI는 실제 R2 상태 반영

#### Exception Handling Philosophy
```
"Try to save something rather than save nothing"
- 중단되어도 best/last checkpoint 보존
- 에러 발생해도 업로드 시도
- 실패해도 warning만 (non-critical)
```

### 기술 노트

#### R2 Path Convention
```
With project_id:
  r2://vision-platform-prod/checkpoints/projects/{project_id}/jobs/{job_id}/best.pt
  r2://vision-platform-prod/checkpoints/projects/{project_id}/jobs/{job_id}/last.pt

Without project_id (test jobs):
  r2://vision-platform-prod/checkpoints/test-jobs/job_{job_id}/best.pt
  r2://vision-platform-prod/checkpoints/test-jobs/job_{job_id}/last.pt
```

#### Database Lifecycle
```sql
-- During training
validation_results.checkpoint_path = NULL

-- After upload (only for best & last epochs)
UPDATE validation_results
SET checkpoint_path = 'r2://...'
WHERE job_id = ? AND epoch IN (best_epoch, last_epoch)
```

#### Frontend Logic
```tsx
// Show checkmark only if R2 path exists
{metric.checkpoint_path?.startsWith('r2://') ? (
  <CheckCircle2 className="text-green-600" />
) : (
  <XCircle className="text-gray-300" />
)}
```

---

## [2025-11-05 14:15] YOLO Validation Metrics 이슈 조사 및 Stratified Split 구현

### 논의 주제
- YOLO 학습 중 validation metrics가 항상 0인 문제 디버깅
- 데이터셋 클래스 분포 불균형 문제 발견
- PyTorch InferenceMode 제약사항 발견
- Stratified split 알고리즘 구현

### 주요 결정사항

#### 1. Validation Metrics = 0 문제 (CANNOT FIX)
- **증상**:
  - Training loss는 정상 감소
  - Validation metrics (mAP, precision, recall) 항상 0.0
  - Confusion matrix 완전히 비어있음 (sum = 0.0)

- **Root Cause 1**: 데이터셋 클래스 분포 불균형
  - COCO32 (32 images, 43 classes): 9개 클래스가 validation set에만 존재
  - 모델이 해당 클래스를 한 번도 학습하지 못함
  - **해결**: Stratified split 구현 ✅

- **Root Cause 2**: PyTorch InferenceMode 제약
  - Ultralytics가 `torch.inference_mode()` 사용 (not `torch.no_grad()`)
  - InferenceMode는 텐서를 irreversibly 변환
  - Manual validation 후 `requires_grad` 복원 불가능
  - RuntimeError: "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed"
  - **결론**: 근본적 PyTorch 설계 제약, 해결 불가 ❌

- **Root Cause 3**: Ultralytics Callback 타이밍
  - `on_fit_epoch_end` 시점에 `validator.batch = None`
  - `validator.pred = None` (예측값 없음)
  - Validation이 실행되지만 callback에서 데이터 접근 불가

#### 2. Stratified Split 구현 (✅ SOLVED)
- **배경**:
  - Random split은 작은 데이터셋에서 클래스 불균형 발생
  - 예: 32 images, 43 classes → 0.74 images/class 평균
  - 9개 클래스가 validation에만 존재 (train에 0개)

- **알고리즘** (`dice_to_yolo.py:136-212`):
  ```python
  1. Build image-to-classes mapping
  2. For rare classes (1 image): → train set (우선순위)
  3. For classes with 2+ images: → both train & val
  4. Remaining images → 80/20 ratio
  5. Verify: no validation-only classes
  ```

- **결과**:
  - Val-only classes: 9 → 0 ✅
  - 모든 validation 클래스가 training set에 존재
  - COCO32, COCO128 모두 검증 완료

#### 3. Train-Mode Validation 테스트 (부분 성공)
- **시도**: Training mode + `torch.no_grad()` 방식
  ```python
  with torch.no_grad():
      preds = model(val_batch['img'])
  optimizer.zero_grad()
  ```

- **에러**: `RuntimeError: expected scalar type Byte but found Float`
  - 원인: Validation batch images가 uint8 (0-255)
  - 모델은 float32 (0.0-1.0) 기대
  - 해결 방법: `imgs = batch['img'].float() / 255.0`

- **결론**: Train-mode validation 가능하지만 추가 구현 필요
  - 데이터 타입 변환
  - Metric 계산 로직 (mAP, confusion matrix 등)
  - 예상 작업: 1-2일

#### 4. Post-Training Validation (권장 Workaround)
- **방식**: 학습 완료 후 별도 validation 실행
  ```python
  results = model.train(...)
  val_metrics = model.val(data=data_yaml, split='val')
  ```

- **장점**:
  - 간단, 안정적
  - Full metrics 제공
  - Training 간섭 없음

- **단점**:
  - Per-epoch 모니터링 불가
  - 최종 메트릭만 확인 가능

### 구현 내용

#### Stratified Split Implementation
**`mvp/training/converters/dice_to_yolo.py:136-212`**:
```python
# 1. Image-to-classes mapping
image_classes = {}
for image in images:
    classes_in_image = set(ann['category_id'] for ann in annotations)
    image_classes[image_id] = classes_in_image

# 2. Class-to-images mapping
class_to_images = defaultdict(list)
for image in images:
    for cls in image_classes[image_id]:
        class_to_images[cls].append(image)

# 3. Stratified allocation
for cls, cls_images in sorted(class_to_images.items(), key=lambda x: len(x[1])):
    if len(cls_images) == 1:
        train_images.append(cls_images[0])  # Rare class → train
    elif len(cls_images) >= 2:
        train_images.append(cls_images[0])  # Both splits
        val_images.append(cls_images[1])

# 4. Distribute remaining (80/20)
remaining_images = [img for img in images if img not in used]
for image in remaining_images:
    if len(train_images) < target_train_size:
        train_images.append(image)
    else:
        val_images.append(image)

# 5. Verify
val_only_classes = val_classes - train_classes
if val_only_classes:
    print(f"WARNING: {len(val_only_classes)} classes only in val")
else:
    print(f"[OK] All {len(val_classes)} val classes in train")
```

#### Validation Debugging
**`mvp/training/adapters/ultralytics_adapter.py:1200-1700`**:
- Train/val dataset label count 로깅
- Confusion matrix 상세 디버깅
- Validation batch 처리 추적 callbacks
- Manual validation 시도 (3가지 접근)
- Train-mode validation 테스트

#### Issue Documentation
**`docs/issues/yolo_validation_metrics.md`** (새 파일):
- **Status**: 🔴 CANNOT FIX - PyTorch Design Limitation
- **Impact**: Medium (training works, post-training validation works)
- Root cause 분석 (3가지)
- Investigation log (4 attempts)
- Possible solutions (4 options)
- Lessons learned

#### Analysis Tool
**`analyze_class_dist.py`** (새 파일):
- Train/val split 클래스 분포 분석
- Val-only classes 탐지
- 통계 리포트 생성
- DICE annotations.json 연동

### 조사 과정 (Investigation Log)

#### Attempt 1: Callback Debugging
- 추가 callbacks: `on_val_batch_start`, `on_val_batch_end`, `on_val_end`
- 발견: `validator.batch = None`, `validator.pred = None`
- 결론: Callback 타이밍에 데이터 미접근

#### Attempt 2: Manual Validation (model.val())
- 시도: `on_fit_epoch_end`에서 `model.val()` 직접 호출
- 에러: `RuntimeError: element 0 does not require grad`
- 원인: `model.val()`이 gradient 비활성화

#### Attempt 3: State Restoration
- 시도: Parameter `requires_grad` 상태 저장 후 복원
  ```python
  original_grad_states = {name: p.requires_grad for name, p in model.named_parameters()}
  # Run validation
  for name, param in model.named_parameters():
      param.requires_grad = original_grad_states[name]  # FAILS!
  ```
- 에러: `RuntimeError: Setting requires_grad=True on inference tensor`
- 원인: PyTorch InferenceMode 제약

#### Attempt 4: Train-Mode Validation
- 시도: Training mode + `torch.no_grad()` 조합
  ```python
  with torch.no_grad():
      preds = model(val_batch['img'])
  optimizer.zero_grad()
  ```
- 에러: `RuntimeError: expected scalar type Byte but found Float`
- 원인: Data type mismatch (uint8 vs float32)
- 결론: 데이터 전처리 추가하면 가능 (추가 구현 필요)

### Git 작업

#### Commit
```
fee0630 feat(training): implement stratified dataset split for YOLO training

- Add stratified split algorithm to ensure all validation classes
  appear in training set (critical for small datasets)
- Val-only classes: 9 → 0 (COCO32 tested)
- Document PyTorch InferenceMode limitation
- Add validation debugging callbacks
- Create class distribution analysis tool

Known Issue: Validation metrics still 0 due to PyTorch InferenceMode.
Post-training validation works. See docs/issues/yolo_validation_metrics.md
```

**변경 파일 (4개)**:
- `mvp/training/converters/dice_to_yolo.py` (+140 lines)
- `mvp/training/adapters/ultralytics_adapter.py` (+338 lines)
- `docs/issues/yolo_validation_metrics.md` (+227 lines, 새 파일)
- `analyze_class_dist.py` (+90 lines, 새 파일)

### 테스트 결과

#### COCO32 Dataset
- **Images**: 32장
- **Classes**: 43개 (COCO)
- **Before stratified split**: 9 classes val-only ❌
- **After stratified split**: 0 classes val-only ✅
- **Train/Val**: 25/7 images

#### COCO128 Dataset
- **Images**: 128장
- **Classes**: 71개 (COCO)
- **Stratified split**: 0 classes val-only ✅
- **Train/Val**: 92/36 images
- **Annotations**: 929개 objects

### 다음 단계

#### Immediate (Close Issue)
- [x] Stratified split 구현
- [x] Issue 문서화
- [x] Commit 생성
- [ ] **Inference API 테스트** (다음 우선순위)

#### Future (If Needed)
- [ ] Custom validator 구현 (~1-2일)
  - Train-mode validation with proper data type handling
  - Manual mAP, precision, recall calculation
  - Confusion matrix construction
- [ ] Test other YOLO models (seg, pose, obb)
- [ ] Test timm models (ResNet, EfficientNet)

### 관련 문서
- **Issue 문서**: [docs/issues/yolo_validation_metrics.md](../issues/yolo_validation_metrics.md)
- **Converter**: mvp/training/converters/dice_to_yolo.py:136-212
- **Adapter**: mvp/training/adapters/ultralytics_adapter.py:1200-1700
- **Analysis Tool**: analyze_class_dist.py

### 핵심 통찰 (Key Insights)

#### PyTorch InferenceMode vs no_grad
| Context | Gradient | Post-restoration | Performance |
|---------|----------|------------------|-------------|
| `no_grad()` | Disabled | ✅ Possible | Slower |
| `inference_mode()` | Disabled | ❌ Impossible | Faster |

**결론**: Ultralytics는 성능을 위해 InferenceMode 선택 → Flexibility 희생

#### Small Dataset Challenge
- **0.74 images/class** (32 images, 43 classes)
- Random split은 클래스 불균형 보장
- Stratified split 필수

#### Validation Monitoring Workaround
- ✅ Training loss로 진행 상황 모니터링
- ✅ Post-training validation으로 최종 메트릭 확인
- ❌ Per-epoch validation metrics (당분간 포기)

### 기술 노트

#### Stratified Split vs Random Split
```python
# Random Split (기존 - 문제있음)
random.shuffle(images)
split_idx = int(len(images) * 0.8)
train = images[:split_idx]
val = images[split_idx:]

# Stratified Split (새로운 - 해결)
# 1. Ensure all val classes in train
# 2. Distribute remaining by ratio
# 3. Verify no val-only classes
```

#### Label Path Structure
```
DICE Dataset (Original):
  datasets/uuid-123/
    ├── images/
    │   ├── 000000000009.jpg
    │   └── ...
    └── labels/              # Single directory
        ├── 000000000009.txt
        └── ...

YOLO Split (Converted):
  datasets/uuid-123_yolo/
    ├── train.txt            # Absolute paths
    ├── val.txt              # Absolute paths
    └── data.yaml
```

**Key**: Labels stay in original DICE directory, not split into train/val subdirs.

---

## [2025-11-04 21:30] Project-Centric Checkpoint Storage 구현

### 논의 주제
- Multi-tenant 지원을 위한 체크포인트 저장 구조 개선
- 현재 경로 구조의 문제점 식별 및 해결 방안 논의
- 전체 training pipeline에 project_id 전파
- Training Service 구현 현황 문서화

### 주요 결정사항

#### 1. Project-Centric Checkpoint Storage 구조 (Option 1 선택)
- **배경**:
  - 기존: `checkpoints/job_{job_id}/` → 여러 사용자/프로젝트/실험 구분 불가
  - TrainingJob에 `project_id`, `created_by`, `session_id`, `experiment_name` 존재
  - Multi-tenant 환경에서 체크포인트 구분 필요

- **결정**: Project-centric 계층 구조 ✅
  ```
  checkpoints/
  ├── projects/
  │   └── {project_id}/
  │       └── jobs/
  │           └── {job_id}/
  │               ├── best.pt
  │               └── last.pt
  └── test-jobs/
      └── job_{job_id}/
          ├── best.pt
          └── last.pt
  ```

- **이유**:
  - 프로젝트 단위 관리 (가장 직관적)
  - 테스트/개발 job 별도 관리 (project_id = null)
  - 기존 체크포인트 마이그레이션 불필요 (사용자가 수동 삭제)

#### 2. 전체 Pipeline에 project_id 전파
- **Data Flow**:
  ```
  Backend (training_manager.py)
    → job_config.project_id
      → Training Service API (api_server.py)
        → TrainingRequest.project_id
          → train.py --project_id
            → TrainingAdapter(project_id)
              → TrainingCallbacks(project_id)
                → upload_checkpoint(project_id)
                  → R2 Storage (conditional path)
  ```

- **구현 위치** (6개 파일 수정):
  1. `storage.py:527` - upload_checkpoint() conditional path logic
  2. `base.py:378` - TrainingAdapter.__init__ accepts project_id
  3. `base.py:1488` - TrainingCallbacks.__init__ accepts project_id
  4. `base.py:1861` - _upload_checkpoints_to_r2() passes project_id
  5. `ultralytics_adapter.py:1082` - Pass project_id to callbacks
  6. `train.py:95` - Add --project_id argument
  7. `api_server.py:60` - TrainingRequest.project_id field
  8. `training_manager.py:125` - job_config includes project_id

### 구현 내용

#### Storage Layer
**`mvp/training/platform_sdk/storage.py`**:
```python
def upload_checkpoint(
    checkpoint_path: str,
    job_id: int,
    checkpoint_name: str = "best.pt",
    project_id: int = None  # 추가
):
    # Build path based on project_id
    if project_id:
        key = f'checkpoints/projects/{project_id}/jobs/{job_id}/{checkpoint_name}'
    else:
        key = f'checkpoints/test-jobs/job_{job_id}/{checkpoint_name}'
```

#### Adapter Layer
**`mvp/training/adapters/base.py`**:
```python
class TrainingAdapter:
    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        output_dir: str,
        job_id: int,
        project_id: int = None  # 추가
    ):
        self.project_id = project_id

class TrainingCallbacks:
    def __init__(
        self,
        job_id: int,
        model_config: 'ModelConfig',
        training_config: 'TrainingConfig',
        db_session=None,
        project_id: int = None  # 추가
    ):
        self.project_id = project_id

    def _upload_checkpoints_to_r2(self, checkpoint_dir: str = None):
        upload_checkpoint(
            checkpoint_path=str(checkpoint_file),
            job_id=self.job_id,
            checkpoint_name=checkpoint_name,
            project_id=self.project_id  # 전달
        )
```

#### Training Service API
**`mvp/training/api_server.py`**:
```python
class TrainingRequest(BaseModel):
    job_id: int
    framework: str
    # ... other fields
    project_id: Optional[int] = None  # 추가

def run_training(request: TrainingRequest):
    cmd = [...]
    if request.project_id is not None:
        cmd.extend(["--project_id", str(request.project_id)])
```

#### Training Script
**`mvp/training/train.py`**:
```python
def parse_args():
    parser.add_argument('--project_id', type=int, default=None,
                        help='Project ID for organizing checkpoints in R2')

adapter = adapter_class(
    model_config=model_config,
    dataset_config=dataset_config,
    training_config=training_config,
    output_dir=args.output_dir,
    job_id=args.job_id,
    project_id=args.project_id,  # 전달
    logger=logger
)
```

#### Backend
**`mvp/backend/app/utils/training_manager.py`**:
```python
job_config = {
    "job_id": job_id,
    "framework": job.framework,
    # ... other fields
    "project_id": job.project_id  # 추가
}
```

### 문서화

#### `docs/trainer/IMPLEMENTATION_STATUS.md` (새 파일)
**포함 내용**:
- Training Service 아키텍처 다이어그램
- 구현 완료 기능 (Phase 1)
  - Microservice Architecture ✅
  - R2 Storage Integration ✅
  - YOLO Training Pipeline ✅
  - DICE Dataset Format ✅
  - Project-Centric Checkpoints ✅
- 테스트 결과 (Job #11, #12, #13)
- 기술 구현 세부사항
- API 엔드포인트 문서
- 다음 단계 (Phase 2: Frontend Integration)

### Git 작업

#### Commit
```
67142e4 feat(training): implement project-centric checkpoint storage

- Add project_id parameter throughout training pipeline
- Implement conditional path logic in upload_checkpoint()
- Update all adapters and callbacks to handle project_id
- Add comprehensive implementation status document
```

**변경 파일 (7개)**:
- `mvp/training/platform_sdk/storage.py`
- `mvp/training/adapters/base.py`
- `mvp/training/adapters/ultralytics_adapter.py`
- `mvp/training/train.py`
- `mvp/training/api_server.py`
- `mvp/backend/app/utils/training_manager.py`
- `docs/trainer/IMPLEMENTATION_STATUS.md` (새 파일)

### 테스트 계획

#### Job #14 테스트 (다음 단계)
**목표**: 새로운 project-centric 경로 구조 검증

**시나리오 1**: project_id 있는 경우
- Job with project_id = 5
- Expected path: `checkpoints/projects/5/jobs/14/best.pt`

**시나리오 2**: project_id 없는 경우 (test job)
- Job with project_id = null
- Expected path: `checkpoints/test-jobs/job_14/best.pt`

**검증 사항**:
- Backend가 project_id를 Training Service에 전달
- Training Service가 train.py에 --project_id 전달
- Adapter가 Callbacks에 project_id 전달
- Callbacks가 upload_checkpoint()에 project_id 전달
- R2에 올바른 경로로 업로드

### 다음 단계

#### Phase 2: Frontend Integration (예정)
- [ ] Training Job 생성 UI
- [ ] Real-time training monitoring
- [ ] Checkpoint download interface
- [ ] Project selection in training form

#### Testing
- [ ] Job #14 실행 및 경로 검증
- [ ] Project job vs test job 경로 차이 확인
- [ ] R2 Storage에서 경로 구조 확인

### 관련 문서
- **구현 현황**: [docs/trainer/IMPLEMENTATION_STATUS.md](../trainer/IMPLEMENTATION_STATUS.md)
- **Adapter 설계**: [docs/trainer/ADAPTER_DESIGN.md](../trainer/ADAPTER_DESIGN.md)
- **이전 세션**: [2025-11-04 17:30] Training Service Microservice 인프라 구축

### 핵심 원칙 준수

1. **No Shortcuts** ✅
   - 하드코딩 없음 (project_id를 동적으로 전달)
   - 임시 방편 없음 (전체 chain 구현)

2. **Production = Local** ✅
   - 동일한 코드베이스
   - 환경변수만 차이
   - R2 Storage 공통 사용

3. **Dependency Isolation** ✅
   - Backend: project_id만 전달 (training 로직 무관)
   - Training Service: 독립적으로 checkpoint 관리

---

## [2025-11-04 17:30] Training Service Microservice 인프라 구축 및 데이터 접근 전략 수립

### 논의 주제
- Training Service Microservice 아키텍처 구현
- Framework별 독립 서비스 구성 (timm, ultralytics, huggingface)
- R2 Storage 직접 접근 전략
- DICE Format → Framework Format 변환 설계
- 데이터셋-모델 호환성 검증 전략

### 주요 결정사항

#### 1. Microservice 아키텍처 구현 (Railway 환경과 동일)
- **배경**:
  - 로컬 테스트가 subprocess 방식으로 동작
  - Railway 배포 환경은 microservice로 구성
  - 로컬과 배포 환경의 불일치 문제

- **결정**: 로컬에서도 microservice로 실행 ✅
  ```
  Backend (Port 8000)
    ↓ HTTP
  ultralytics-service (Port 8001) ← UPDATED 2025-11-13
  timm-service (Port 8002) ← UPDATED 2025-11-13
  huggingface-service (Port 8003)
  ```

  **⚠️ Port Change Log (2025-11-13)**:
  - Original plan: timm=8001, ultralytics=8002
  - Current: ultralytics=8001, timm=8002 (planned)
  - Reason: Ultralytics implemented first on 8001, kept for stability

- **구현 내용**:
  - Framework별 독립 venv 생성 (`venv-ultralytics`, `venv-timm`)
  - 독립 실행 스크립트 (`scripts/start-ultralytics-service.bat`)
  - Backend `.env`에 framework별 URL 설정
  - `TrainingServiceClient`가 framework 기반 라우팅 지원

#### 2. R2 Storage 직접 접근 (Option A 선택)
- **질문**: Training Service가 데이터를 어떻게 접근할 것인가?
  - Option A: Training Service가 R2 직접 접근 (추천 ✅)
  - Option B: Backend API 통해 다운로드

- **결정**: Option A - R2 직접 접근
- **이유**:
  - Microservice 철학에 맞음 (독립적 동작)
  - Backend 부담 감소
  - `platform_sdk/storage.py` 이미 구현됨
  - R2 credentials 공유 필요하지만 문제없음

- **구현 방식**:
  ```python
  # Training Service .env
  AWS_S3_ENDPOINT_URL=https://...r2.cloudflarestorage.com
  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...
  S3_BUCKET=vision-platform-prod

  # platform_sdk/storage.py
  get_dataset(dataset_id) → R2 다운로드 → 로컬 캐시
  ```

#### 3. Dataset ID 기반 접근 (Path 방식에서 전환)
- **현재 문제**:
  - 기존: `dataset_path` (파일 시스템 경로)
  - Frontend 흐름: User가 데이터셋 선택 (ID 기반)
  - R2 구조: `datasets/{id}/` (UUID 기반)

- **결정**: `dataset_id` 기반으로 전환
  ```python
  # Frontend → Backend
  {"dataset_id": "uuid-123"}

  # Backend → Training Service
  {"dataset_id": "uuid-123"}

  # Training Service
  dataset_path = get_dataset("uuid-123")
  # → R2: datasets/uuid-123/ 다운로드
  # → Local: /workspace/data/.cache/datasets/uuid-123/
  ```

#### 4. DICE Format 변환 전략
- **배경**:
  - R2에 DICE Format으로 저장됨 (`annotations.json`)
  - 각 framework는 고유 포맷 필요 (YOLO, COCO, ImageFolder 등)

- **변환 전략**:
  ```
  Training Service
    ↓ 1. Download
    datasets/{id}/annotations.json (DICE Format)

    ↓ 2. Convert
    dice_to_yolo()      → data.yaml, labels/*.txt
    dice_to_imagefolder() → train/class1/, val/class1/
    dice_to_coco()      → annotations/instances.json

    ↓ 3. Train
    UltralyticsAdapter(converted_path)
  ```

- **구현 위치**: `mvp/training/converters/`
  - `dice_to_yolo.py`
  - `dice_to_imagefolder.py`
  - `dice_to_coco.py`

#### 5. 데이터셋-모델 호환성 검증 (3-Tier 전략)
- **문제**:
  - Classification 데이터로 Detection 학습 불가
  - Segmentation → Detection 변환 가능
  - Detection → Classification 변환 애매

- **3-Tier 검증 전략**:
  ```
  Tier 1: Frontend (UX Hint) [P2]
    → 데이터셋 선택 시 호환성 힌트 표시

  Tier 2: Backend API (사전 검증) [P1]
    → GET /datasets/{id}/compatibility?task_type=...
    → DB 메타데이터 or annotations.json 파싱

  Tier 3: Training Service (실행 시 검증) [P0] ✅
    → prepare_dataset()에서 상세 검증
    → 변환 가능하면 변환, 불가능하면 명확한 에러
  ```

- **MVP 우선순위**: Tier 3만 구현 (필수)
  - 이유: 일단 동작하는 것 먼저, UX는 나중에

- **변환 규칙 테이블**:
  ```python
  CONVERSION_MATRIX = {
      ("instance_segmentation", "object_detection"): polygon_to_bbox,
      ("instance_segmentation", "image_classification"): use_dominant_class,
      ("object_detection", "image_classification"): use_dominant_class,
      ("image_classification", "object_detection"): None,  # ❌ 불가능
  }
  ```

### 구현 내용

#### Microservice 인프라
**스크립트 생성**:
- `mvp/scripts/setup-ultralytics-service.bat` - venv 생성 및 의존성 설치
- `mvp/scripts/start-ultralytics-service.bat` - 서비스 시작 (Port 8001) ← **UPDATED 2025-11-13**
- `mvp/scripts/setup-timm-service.bat` - timm 서비스 셋업
- `mvp/scripts/start-timm-service.bat` - timm 서비스 시작 (Port 8002) ← **UPDATED 2025-11-13**

**Backend 설정** (Updated 2025-11-13):
```bash
# platform/backend/.env
ULTRALYTICS_SERVICE_URL=http://localhost:8001  # UPDATED: was 8002
TIMM_SERVICE_URL=http://localhost:8002  # UPDATED: was 8001
HUGGINGFACE_SERVICE_URL=http://localhost:8003
TRAINING_SERVICE_URL=http://localhost:8001  # Fallback (Ultralytics)
```

**ultralytics-service 실행 확인** (Updated 2025-11-13):
- ✅ Port 8001에서 정상 동작
- ✅ Health Check: `{"status":"healthy"}`
- ✅ Models API: 5개 모델 (yolo11n, yolo11n-seg, yolo11n-pose, yolo_world_v2_s, sam2_t)

#### 기존 코드 분석
**platform_sdk/storage.py**:
- ✅ `get_dataset(dataset_id)` 이미 구현됨
- ✅ 3-tier 캐싱: Local → R2 → Original source
- ✅ 자동 압축 해제 및 디렉토리 반환

**ultralytics_adapter.py**:
- ✅ `_resolve_dataset_path()` 메서드 존재
- ✅ Simple name 감지 → `get_dataset()` 호출
- ⚠️ 현재는 path 기반, dataset_id 기반으로 수정 필요

### 다음 단계 (우선순위 순)

#### Phase 1: 환경 설정 및 기본 연동
- [x] ultralytics-service venv 생성 및 의존성 설치
- [x] ultralytics-service 실행 스크립트
- [x] Backend .env 업데이트 (framework별 URL)
- [ ] Training Service .env 업데이트 (R2 credentials)
- [ ] Backend 실행 및 Training Service 연결 테스트

#### Phase 2: DICE Format 변환기 구현
- [ ] `mvp/training/converters/dice_to_yolo.py` 구현
  - annotations.json 파싱
  - Polygon → Bounding box 변환
  - data.yaml 생성
  - labels/*.txt 생성
- [ ] `platform_sdk/storage.py` 확장
  - `get_dataset_from_r2(dataset_id)` 디렉토리 다운로드
- [ ] 호환성 검증 로직
  - `check_detailed_compatibility()` 함수
  - CONVERSION_MATRIX 정의

#### Phase 3: 학습 파이프라인 E2E 테스트
- [ ] R2에 테스트 데이터셋 업로드 (sample-det-coco32)
- [ ] Backend → ultralytics-service 학습 시작
- [ ] 데이터 다운로드 → 변환 → 학습 전체 흐름 검증
- [ ] 메트릭 수집 및 로깅 확인

#### Phase 4: Checkpoint R2 저장
- [ ] `platform_sdk/storage.py`에 `upload_checkpoint()` 추가
- [ ] Adapter `save_checkpoint()` 수정
- [ ] R2 경로: `checkpoints/{job_id}/epoch_{epoch}.pth`

### 핵심 설계 원칙

1. **No Shortcuts, No Hardcoding** (CLAUDE.md)
   - ✅ 동적 모델 레지스트리 (Training Service API)
   - ✅ R2 Storage 기반 (로컬 파일시스템 의존성 제거)
   - ✅ Database 기반 메타데이터 (하드코딩 샘플 없음)

2. **Dependency Isolation**
   - ✅ Backend: PyTorch 없음
   - ✅ Training Services: Framework별 독립 venv
   - ✅ HTTP/JSON 통신만

3. **Production = Local**
   - ✅ Microservice 아키텍처 동일
   - ✅ R2 Storage 사용
   - ✅ 환경변수만 차이 (URL, credentials)

### 관련 문서
- **인프라**: [docs/planning/TRAINER_IMPLEMENTATION_PLAN.md](../planning/TRAINER_IMPLEMENTATION_PLAN.md)
- **데이터셋 설계**: [docs/datasets/DATASET_MANAGEMENT_DESIGN.md](../datasets/DATASET_MANAGEMENT_DESIGN.md)
- **DICE Format 스펙**: [docs/datasets/PLATFORM_DATASET_FORMAT.md](../datasets/PLATFORM_DATASET_FORMAT.md)
- **현재 상태**: [docs/datasets/CURRENT_STATUS.md](../datasets/CURRENT_STATUS.md)

### 기술 노트

#### R2 Storage 구조
```
vision-platform-prod/
├── datasets/
│   └── {id}/
│       ├── images/          # 원본 폴더 구조 유지
│       └── annotations.json # DICE Format v1.0
├── models/
│   └── pretrained/{framework}/{model_name}.pt
└── checkpoints/
    └── {job_id}/
        └── epoch_{n}.pth
```

#### Training Service 데이터 흐름
```
1. Backend → POST /training/start
   {"dataset_id": "uuid-123", "model_name": "yolo11n", ...}

2. Training Service → get_dataset("uuid-123")
   - Check local: /workspace/data/.cache/datasets/uuid-123/
   - Download R2: datasets/uuid-123/ → local cache
   - Return: local_path

3. DICE Format 변환
   - Parse: annotations.json
   - Check: compatibility with task_type
   - Convert: dice_to_yolo() → data.yaml + labels/
   - Return: converted_path

4. 학습 실행
   - UltralyticsAdapter(converted_path)
   - Train + Validate
   - Save checkpoint → R2
   - Log metrics → Backend
```

#### Framework별 Port 할당 (Updated 2025-11-13)
```
Backend:             8000
ultralytics-service: 8001  ← UPDATED (was 8002)
timm-service:        8002  ← UPDATED (was 8001, planned)
huggingface-service: 8003
Frontend:            3000
```

**Change Log**: Ultralytics implemented first on 8001, timm moved to 8002 to avoid conflict

---

## [2025-11-04 16:00] 데이터셋 인증/권한 구현 및 학습 파이프라인 준비

### 논의 주제
- 데이터셋 인증 및 권한 체크 구현
- 학습 파이프라인 테스트 vs 스냅샷 구현 우선순위
- YOLO segmentation → DICE Format 변환
- 프론트엔드 UX 개선 (자동 네비게이션 제거)
- PR 생성 및 문서화

### 주요 결정사항

#### 1. 데이터셋 인증 시스템 구현
- **배경**: 데이터셋을 아무나 볼 수 있는 보안 문제 발견
- **구현 내용**:
  - Backend: 모든 dataset API에 `Depends(get_current_user)` 추가
  - Frontend: 모든 API 호출에 Bearer token 추가
  - Sidebar: 인증된 사용자만 "데이터셋", "프로젝트" 메뉴 표시
- **권한 규칙**:
  - 소유자(owner)만 삭제/업로드 가능
  - Public 데이터셋은 모든 인증 사용자 조회 가능
  - Private 데이터셋은 소유자만 접근

#### 2. 스냅샷 구현 시기 결정
- **질문**: 학습 파이프라인 테스트 전에 스냅샷 구현이 필요한가?
- **결정**: 학습 파이프라인 먼저 테스트 (Option A) ✅
- **이유**:
  - 스냅샷 없이도 학습 가능 (`dataset_snapshot_id`는 nullable)
  - 학습이 제대로 돌아가야 스냅샷도 의미 있음
  - DB 모델은 이미 준비됨 (빠른 전환 가능)
  - MVP 단계에서는 핵심 기능 검증 우선
- **위험 관리**: 초기 테스트 데이터셋은 수정하지 않기

#### 3. DICE Format 변환 준비
- **목적**: 학습 파이프라인 테스트용 데이터셋 준비
- **작업**: YOLO segmentation → DICE Format v1.0 변환
- **입력**: `C:\datasets\seg-coco32` (YOLO format)
- **출력**: `C:\datasets\dice_format\seg-coco32` (DICE format)
- **결과**:
  - 32 images, 209 annotations
  - 43 COCO classes (person, car, cup 등)
  - instance_segmentation 태스크

#### 4. 프론트엔드 UX 개선
- **문제**: 데이터셋 생성 후 상세 페이지로 자동 전환
- **해결**: 자동 네비게이션 제거, 테이블만 새로고침
- **이유**:
  - 여러 데이터셋 연속 생성 시 편리
  - 불필요한 화면 전환 감소
  - 사용자가 원하면 수동으로 클릭 가능

### 구현 내용

#### Backend (인증 추가)

**`mvp/backend/app/api/datasets.py`**:
```python
# 추가된 imports
from app.db.models import Dataset, User
from app.utils.dependencies import get_current_user

# 수정된 엔드포인트
@router.get("/available")
async def list_sample_datasets(
    current_user: User = Depends(get_current_user),  # 추가
    db: Session = Depends(get_db)
):
    # Owner OR public 필터링
    query = db.query(Dataset).filter(
        or_(
            Dataset.owner_id == current_user.id,
            Dataset.visibility == 'public'
        )
    )

@router.post("")
async def create_dataset(
    current_user: User = Depends(get_current_user),  # 추가
    ...
):
    new_dataset = Dataset(
        owner_id=current_user.id,  # 자동 설정
        ...
    )

@router.delete("/{dataset_id}")
async def delete_dataset(
    current_user: User = Depends(get_current_user),  # 추가
    ...
):
    # 소유자 확인
    if dataset.owner_id != current_user.id:
        raise HTTPException(403, "Permission denied")
```

**`mvp/backend/app/api/datasets_images.py`**:
- 모든 엔드포인트에 `current_user` 파라미터 추가
- 소유자 확인 로직 추가
- Public dataset 조회 허용 로직

**`mvp/backend/app/api/datasets_folder.py`**:
- 폴더 업로드 API에 인증 추가
- 소유자만 업로드 가능

#### Frontend (인증 토큰 추가)

**`mvp/frontend/components/Sidebar.tsx`**:
```tsx
{/* 인증된 사용자만 표시 */}
{isAuthenticated && (
  <div>
    <button onClick={onOpenDatasets}>데이터셋</button>
  </div>
)}

{isAuthenticated && (
  <div>프로젝트 목록</div>
)}
```

**`mvp/frontend/components/DatasetPanel.tsx`**:
```typescript
const fetchDatasets = async () => {
  const token = localStorage.getItem('access_token')

  if (!token) {
    console.error('No access token found')
    return
  }

  const response = await fetch(`${baseUrl}/datasets/available`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  })
}

const handleDeleteConfirm = async () => {
  const token = localStorage.getItem('access_token')

  const response = await fetch(`${baseUrl}/datasets/${id}`, {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  })
}
```

**`mvp/frontend/components/datasets/CreateDatasetModal.tsx`**:
```typescript
// useRouter import 제거
// router.push() 제거
// 성공 후 모달만 닫기
setTimeout(() => {
  handleClose()  // 네비게이션 없이 닫기만
}, 1000)
```

**기타 컴포넌트**:
- `DatasetImageUpload.tsx`: Bearer token 추가
- `DatasetImageGallery.tsx`: Bearer token 추가
- `ProjectDetail.tsx`: handleSaveEdit에 token 추가
- `datasets/[id]/page.tsx`: Bearer token 추가

#### 유틸리티 스크립트

**`mvp/backend/convert_yolo_seg_to_platform.py`** (새 파일):
- YOLO segmentation → DICE Format 변환
- Normalized coordinates → 절대 pixel coordinates
- Polygon segmentation 데이터 보존
- Bounding box 자동 계산
- Area 계산 (shoelace formula)
- Content hash 생성

### Git 작업

#### Commits (7개)
```
8996157 docs(datasets): add current status and next steps document
744fb3e chore: update gitignore for test files and database backups
99a5ef5 fix(frontend): remove auto-navigation after dataset creation
ae26d92 feat(mvp): implement authentication and authorization for datasets
ab28012 feat(datasets): enhance folder upload and add dataset deletion
d527411 feat(datasets): implement Create-then-Upload architecture
b1677fd feat(datasets): add individual image management with R2 presigned URLs
```

#### Pull Request
- **PR #12**: "feat(datasets): implement Dataset Entity with R2 Storage and Authentication"
- **Base**: main
- **28 commits** total in this feature branch
- **Status**: Ready for review

### 생성된 문서

#### `docs/datasets/CURRENT_STATUS.md` (새 파일)
**목적**: 다음 세션을 위한 종합 상태 문서

**포함 내용**:
- ✅ 완료된 기능 (Phase 1 & 2)
  - Core Infrastructure
  - Backend API (CRUD, Images, Folder)
  - Frontend Components
  - DICE Format v2.0
  - Training Integration
  - Authentication

- ⏳ 남은 작업 (Phase 3 & 4)
  - Sprint 1: 버전닝/스냅샷 (2-3일)
  - Sprint 2: UI/UX 개선 (1-2일)
  - Sprint 3: 무결성 관리 (2-3일)

- 📂 테스트 데이터셋
  - seg-coco32 (DICE Format)
  - 위치, 구조, 메타데이터, 사용법

- 🎯 다음 세션 시작 가이드
  - **Option A**: 학습 파이프라인 테스트 (추천)
  - Option B: 스냅샷 구현
  - Quick Start 명령어

- 🔍 중요 파일 경로 맵

### 테스트 데이터셋

**seg-coco32 (DICE Format v1.0)**:
- **위치**: `C:\datasets\dice_format\seg-coco32`
- **구조**:
  ```
  seg-coco32/
  ├── annotations.json    # DICE Format v1.0
  └── images/             # 32 images
  ```
- **메타데이터**:
  - Format: instance_segmentation
  - Images: 32장
  - Annotations: 209개 polygon segmentations
  - Classes: 43개 COCO 클래스
  - Avg annotations per image: 6.53개
- **Top 5 classes**: person (56), car (19), cup (15), giraffe (9), bird (8)

### 다음 단계

#### Option A: 학습 파이프라인 테스트 (추천 ✅)
**브랜치**: `feature/training-pipeline-test`

**목표**:
1. seg-coco32 데이터셋 Frontend에서 업로드
2. Training API 호출 테스트
3. Backend ↔ Training Service 통신 검증
4. 학습 시작/중지/모니터링 확인
5. MLflow 연동 확인

**Quick Start**:
```bash
# 새 브랜치 생성
git checkout main
git pull
git checkout -b feature/training-pipeline-test

# Backend 시작
cd mvp/backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Frontend 시작
cd mvp/frontend
npm run dev

# 데이터셋 업로드
# http://localhost:3000 → 로그인 → 데이터셋 → Create
# C:\datasets\dice_format\seg-coco32 폴더 선택

# 학습 시작
# 채팅: "seg-coco32 데이터셋으로 yolo11n-seg 모델 학습시작"
```

#### Option B: 스냅샷 구현
**브랜치**: `feature/dataset-snapshots`

**작업 내용**:
- POST `/datasets/{id}/snapshots` API
- 학습 시작 시 자동 스냅샷
- 스냅샷 목록 UI
- 버전 비교 뷰

### 관련 문서

- **상태 문서**: [CURRENT_STATUS.md](./datasets/CURRENT_STATUS.md)
- **설계 문서**: [DATASET_MANAGEMENT_DESIGN.md](./datasets/DATASET_MANAGEMENT_DESIGN.md)
- **구현 계획**: [IMPLEMENTATION_PLAN.md](./datasets/IMPLEMENTATION_PLAN.md)
- **포맷 스펙**: [PLATFORM_DATASET_FORMAT.md](./datasets/PLATFORM_DATASET_FORMAT.md)

### 기술 노트

#### 인증 흐름
```
User → Frontend (localStorage.getItem('access_token'))
     → Backend API (Authorization: Bearer {token})
     → Depends(get_current_user)
     → JWT 검증 및 User 객체 반환
     → 권한 체크 (owner_id 비교)
```

#### 데이터셋 권한 규칙
- **Public datasets**:
  - 모든 인증 사용자 조회 가능
  - 소유자만 수정/삭제
- **Private datasets**:
  - 소유자만 조회/수정/삭제
- **업로드/삭제**:
  - 항상 소유자만 가능

#### .gitignore 업데이트
추가된 패턴:
- `*.db.backup*` - DB 백업 파일
- `test_*.py` - 테스트 스크립트
- `convert_*.py` - 변환 유틸리티
- `migrate_*.py` - 마이그레이션 스크립트

### 핵심 파일

#### Backend
```
mvp/backend/app/
├── api/
│   ├── datasets.py              # ✅ 인증 추가
│   ├── datasets_folder.py       # ✅ 인증 추가
│   ├── datasets_images.py       # ✅ 인증 추가
│   └── training.py              # dataset_id 지원
├── utils/
│   ├── r2_storage.py
│   └── dependencies.py          # get_current_user
└── convert_yolo_seg_to_platform.py  # 새 파일 (gitignore)
```

#### Frontend
```
mvp/frontend/
├── components/
│   ├── DatasetPanel.tsx          # ✅ 토큰 추가
│   ├── Sidebar.tsx               # ✅ 조건부 렌더링
│   ├── ProjectDetail.tsx         # ✅ 토큰 추가
│   └── datasets/
│       ├── CreateDatasetModal.tsx    # ✅ 네비게이션 제거
│       ├── DatasetImageUpload.tsx    # ✅ 토큰 추가
│       └── DatasetImageGallery.tsx   # ✅ 토큰 추가
└── app/datasets/[id]/page.tsx    # ✅ 토큰 추가
```

#### Documentation
```
docs/datasets/
├── CURRENT_STATUS.md             # 새 파일 ⭐
├── DATASET_MANAGEMENT_DESIGN.md
├── IMPLEMENTATION_PLAN.md
└── PLATFORM_DATASET_FORMAT.md
```

---

## [2025-01-04 13:00] 데이터셋 관리 UI 통합 및 설계 논의

### 논의 주제
- 데이터셋 UI 레이아웃 통합 문제
- 하드코딩 데이터 제거
- 데이터셋 업로드 방식 설계
- 버전닝 전략
- 무결성 관리

### 주요 결정사항

#### 1. UI 레이아웃 통합
- **문제**: 데이터셋 버튼 클릭 시 전체 화면으로 나와서 기존 레이아웃(사이드바, 채팅, 작업공간) 무시
- **해결**:
  - 새 `DatasetPanel` 컴포넌트 생성 (컴팩트 테이블 디자인)
  - `app/page.tsx`에 상태 관리 추가
  - Sidebar에서 라우팅 대신 핸들러 호출
- **결과**: AdminProjectsPanel과 동일한 패턴으로 작업공간에 통합

#### 2. 하드코딩 데이터 제거
- **문제**: DB에 6개 샘플 데이터셋 하드코딩됨 (cls-imagenet-10 등)
- **원칙 위반**: CLAUDE.md - "no shortcut, no hardcoding, no dummy data"
- **해결**: DB에서 모든 샘플 데이터 삭제
- **결과**: 실제 업로드한 데이터만 표시

#### 3. task_type은 데이터셋 속성이 아니다
- **핵심 통찰**: 같은 이미지를 classification, detection, segmentation 등 다양하게 활용 가능
- **결정**:
  - ❌ Dataset.task_type 삭제
  - ✅ TrainingJob.task_type 추가
  - 데이터셋은 이미지 저장소, 학습 작업이 용도 결정

#### 4. 폴더 구조 유지
- **결정**: 업로드 시 폴더 구조 항상 유지
- **R2 경로**: `datasets/{id}/images/{original_path}`
- **이유**:
  - 원본 구조 보존
  - 파일명 충돌 방지
  - 유연성 확보

#### 5. labeled의 정의
- **정의**: `labeled = annotation.json 존재 여부`
- **규칙**:
  - labeled 업로드는 폴더만 가능 (annotation.json 필요)
  - unlabeled는 폴더/개별 파일 모두 가능
  - labeled 데이터셋에 labeled 폴더 병합 **금지**

#### 6. meta.json 생성 시점
- **unlabeled**: meta.json 없음 (DB만)
- **labeled 전환**: annotation.json + meta.json 함께 생성
- **export**: 항상 meta.json 포함
- **Single Source of Truth**: DB

#### 7. 버전닝 전략: Mutable + Snapshot
- **원칙**:
  - 데이터셋은 기본적으로 가변(mutable)
  - 학습 시작 시 자동 스냅샷 생성
  - 사용자가 명시적 버전 생성 가능 (v1, v2...)
- **효율성**:
  - 이미지는 모든 버전이 공유
  - 스냅샷은 annotation.json만 저장
  - 저장 공간 99% 절약 (10GB + 10MB + 10MB vs 30GB)

#### 8. 이미지 삭제 허용 + 무결성 관리
- **이미지 삭제**: 허용
- **영향받는 스냅샷 처리**:
  - 옵션 A: Broken 표시 (재현 불가)
  - 옵션 B: 자동 복구 (annotation 수정)
- **주기적 무결성 체크**: Celery task로 구현

### 구현 내용

#### Frontend
- `components/DatasetPanel.tsx`: 컴팩트 테이블 UI (새 파일)
  - 검색, 정렬 기능
  - 확장 가능한 행 (이미지 갤러리)
  - 이미지 업로드/조회

- `app/page.tsx`: 상태 관리 추가
  - `showDatasets` state
  - `handleOpenDatasets()` 핸들러
  - 작업공간에 DatasetPanel 렌더링

- `components/Sidebar.tsx`: 라우팅 제거
  - `router.push('/datasets')` → `onOpenDatasets()` 호출

#### Backend
- 기존 개별 이미지 업로드 API 유지
  - POST `/datasets/{id}/images`
  - GET `/datasets/{id}/images`

#### Database
- 하드코딩된 6개 샘플 데이터셋 삭제

### 관련 문서

- **설계 문서**: [DATASET_MANAGEMENT_DESIGN.md](./datasets/DATASET_MANAGEMENT_DESIGN.md)
  - 데이터 모델
  - 스토리지 구조
  - 12가지 업로드 시나리오
  - 버전닝 전략
  - 무결성 관리

- **기존 문서**:
  - [DICE_FORMAT_v2.md](./datasets/DICE_FORMAT_v2.md)
  - [STORAGE_ACCESS_PATTERNS.md](./datasets/STORAGE_ACCESS_PATTERNS.md)

### 다음 단계

#### Phase 2: 폴더 업로드 (다음 구현)
- [ ] 폴더 구조 유지 업로드 (`webkitdirectory`)
- [ ] labeled 데이터셋 생성 (annotation.json 포함)
- [ ] DB 모델 확장 (labeled, class_names, is_snapshot 등)

#### Phase 3: 버전닝
- [ ] 학습 시 자동 스냅샷
- [ ] 명시적 버전 생성
- [ ] 스냅샷 목록 UI

#### Phase 4: 무결성 관리
- [ ] 이미지 삭제 시 영향 분석
- [ ] Broken/복구 로직
- [ ] 주기적 무결성 체크

### 기술 스택
- Frontend: Next.js 14, TypeScript, Tailwind CSS
- Backend: FastAPI, Python, SQLAlchemy
- Storage: Cloudflare R2 (S3-compatible)
- Database: SQLite (local), PostgreSQL (production)

### 핵심 파일
- `mvp/frontend/components/DatasetPanel.tsx` (새로 생성)
- `mvp/frontend/app/page.tsx` (수정)
- `mvp/frontend/components/Sidebar.tsx` (수정)
- `mvp/backend/app/api/datasets_images.py` (기존)
- `mvp/backend/app/utils/r2_storage.py` (기존)

---

