# MVP to Platform Migration Strategy

## 현황 분석

### MVP 현재 상태 (매우 완성도 높음)

**Backend (9000+ lines):**
- ✅ Authentication & Authorization (JWT, Role-based)
- ✅ User Management (Admin API)
- ✅ Project Management
- ✅ Dataset Management (Upload, Validation, Format Conversion)
- ✅ Training Job Management
- ✅ Chat API (LLM Integration)
- ✅ WebSocket (Real-time updates)
- ✅ MLflow Integration
- ✅ Image Tools & Validation
- ✅ K8s Training Manager

**Frontend (51 components):**
- ✅ Authentication UI (Login, Register, Profile)
- ✅ Dashboard & Sidebar
- ✅ Project Management UI
- ✅ Dataset Management UI
- ✅ Training Panel
- ✅ Chat Panel
- ✅ Admin Panels (Users, Projects, Datasets)
- ✅ Real-time Monitoring

**Training (`mvp/training/`):**
- ✅ Adapter Pattern (timm, ultralytics, huggingface)
- ✅ Training Scripts (train.py)
- ✅ API Server (api_server.py)
- ✅ Config Schemas
- ✅ Model Registry
- ✅ Platform SDK

### Platform과의 핵심 차이점

**아키텍처 차이:**

```
MVP Architecture:
┌─────────────────┐
│   Frontend      │
└────────┬────────┘
         │ HTTP
┌────────▼────────┐
│   Backend       │
│  ┌───────────┐  │
│  │ Training  │  │ ← Backend가 직접 K8s Pod 관리
│  │ Manager   │  │
│  └─────┬─────┘  │
└────────┼────────┘
         │ K8s API
┌────────▼────────┐
│  Training Pod   │
│  (train.py)     │
└─────────────────┘
```

```
Platform Architecture:
┌─────────────────┐
│   Frontend      │
└────────┬────────┘
         │ HTTP
┌────────▼────────┐
│   Backend       │ ← HTTP 클라이언트로 변경
└────────┬────────┘
         │ HTTP API
┌────────▼────────┐
│ Training Service│ ← 새로운 독립 서비스
│  (api_server.py)│
└────────┬────────┘
         │ K8s API
┌────────▼────────┐
│  Training Pod   │
│  (train.py)     │
└─────────────────┘
```

**핵심 변경점:**
1. **Training 로직 분리**: `mvp/training/` → `platform/trainers/` (독립 서비스)
2. **Backend 수정**: K8s 직접 관리 → HTTP API 클라이언트
3. **나머지는 그대로 유지**: Auth, Project, Dataset, Frontend 등

## 마이그레이션 전략

### ✅ 권장: MVP 기반 + Training Service 분리

#### 장점:
1. **빠른 개발**: 9000줄 + 51 컴포넌트 재사용
2. **검증된 코드**: 이미 작동하고 테스트된 기능들
3. **집중적 개선**: Training Service 분리에만 집중
4. **점진적 개선**: 나중에 필요한 부분만 리팩토링

#### 작업량 추정:

**1단계: Training Service 분리 (3-5일)**
- `mvp/training/` → `platform/trainers/ultralytics/`, `timm/`, `huggingface/`
- 각 서비스별 Dockerfile 작성
- K8s Deployment 설정
- Health check 추가

**2단계: Backend 수정 (2-3일)**
- `training_manager_k8s.py` 제거
- HTTP 클라이언트 구현 (`training_service_client.py`)
- API 엔드포인트 수정
- 환경 변수 설정

**3단계: 통합 테스트 (2-3일)**
- Local 환경 테스트
- K8s 환경 테스트
- E2E 테스트

**총 예상 기간: 1-2주**

### ❌ 대안: Platform 새로 구축 (현재 진행 중)

#### 단점:
1. **느린 개발**: 모든 기능 재구현 필요
2. **중복 작업**: 이미 작동하는 기능을 다시 만듦
3. **위험 높음**: 새로운 버그 발생 가능
4. **테스트 부족**: 통합 테스트 다시 필요

**총 예상 기간: 2-3개월**

## 실행 계획

### Phase 1: Training Service 분리 (우선순위 높음)

#### 1.1 Ultralytics Training Service

```bash
platform/trainers/ultralytics/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py              # FastAPI app
│   ├── training.py          # Training logic (from mvp/training/train.py)
│   ├── adapters/
│   │   └── ultralytics.py   # Adapter
│   └── schemas/
│       └── config.py        # Config schemas
└── k8s/
    ├── deployment.yaml
    └── service.yaml
```

**API 엔드포인트:**
```python
POST   /train        # Start training job
GET    /jobs/{id}    # Get job status
DELETE /jobs/{id}    # Cancel job
GET    /models/list  # List available models
GET    /health       # Health check
```

#### 1.2 TIMM Training Service

```bash
platform/trainers/timm/
└── (동일한 구조)
```

#### 1.3 HuggingFace Training Service

```bash
platform/trainers/huggingface/
└── (동일한 구조)
```

### Phase 2: MVP Backend 수정

#### 2.1 Training Service Client 생성

```python
# mvp/backend/app/utils/training_service_client.py

import httpx
from typing import Dict, Any

class TrainingServiceClient:
    """HTTP client for Training Services"""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start training job via HTTP API"""
        response = await self.client.post(
            f"{self.service_url}/train",
            json=config
        )
        return response.json()

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status"""
        response = await self.client.get(
            f"{self.service_url}/jobs/{job_id}"
        )
        return response.json()

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from this service"""
        response = await self.client.get(
            f"{self.service_url}/models/list"
        )
        return response.json()
```

#### 2.2 Training API 수정

```python
# mvp/backend/app/api/training.py

from app.utils.training_service_client import TrainingServiceClient

# Service URLs from environment
ULTRALYTICS_SERVICE_URL = os.getenv("ULTRALYTICS_SERVICE_URL", "http://ultralytics-service:8000")
TIMM_SERVICE_URL = os.getenv("TIMM_SERVICE_URL", "http://timm-service:8000")
HUGGINGFACE_SERVICE_URL = os.getenv("HUGGINGFACE_SERVICE_URL", "http://huggingface-service:8000")

# Clients
ultralytics_client = TrainingServiceClient(ULTRALYTICS_SERVICE_URL)
timm_client = TrainingServiceClient(TIMM_SERVICE_URL)
huggingface_client = TrainingServiceClient(HUGGINGFACE_SERVICE_URL)

@router.post("/jobs/{job_id}/start")
async def start_training_job(job_id: int, db: Session = Depends(get_db)):
    """Start training job by calling appropriate Training Service"""

    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Select service based on framework
    if job.framework == "ultralytics":
        client = ultralytics_client
    elif job.framework == "timm":
        client = timm_client
    elif job.framework == "huggingface":
        client = huggingface_client
    else:
        raise HTTPException(status_code=400, detail=f"Unknown framework: {job.framework}")

    # Call Training Service
    result = await client.start_training(job.config)

    # Update job status
    job.status = "running"
    job.external_job_id = result["job_id"]  # Store external job ID
    db.commit()

    return {"message": "Training started", "job_id": job_id}
```

#### 2.3 Models API 수정 (Dynamic Model Registry)

```python
# mvp/backend/app/api/models.py

@router.get("/list")
async def list_models():
    """List all available models from all Training Services"""

    models = []

    # Fetch from Ultralytics Service
    try:
        ultralytics_models = await ultralytics_client.list_models()
        models.extend(ultralytics_models)
    except Exception as e:
        logger.warning(f"Failed to fetch Ultralytics models: {e}")

    # Fetch from TIMM Service
    try:
        timm_models = await timm_client.list_models()
        models.extend(timm_models)
    except Exception as e:
        logger.warning(f"Failed to fetch TIMM models: {e}")

    # Fetch from HuggingFace Service
    try:
        huggingface_models = await huggingface_client.list_models()
        models.extend(huggingface_models)
    except Exception as e:
        logger.warning(f"Failed to fetch HuggingFace models: {e}")

    return models
```

### Phase 3: 환경 설정

#### 3.1 MVP Backend .env 추가

```bash
# Training Service URLs
ULTRALYTICS_SERVICE_URL=http://localhost:8001
TIMM_SERVICE_URL=http://localhost:8002
HUGGINGFACE_SERVICE_URL=http://localhost:8003

# K8s mode (for production)
# ULTRALYTICS_SERVICE_URL=http://ultralytics-service.default.svc.cluster.local:8000
```

#### 3.2 Docker Compose (Local Development)

```yaml
# docker-compose.yml

services:
  # MVP Backend
  backend:
    build: ./mvp/backend
    ports:
      - "8000:8000"
    environment:
      - ULTRALYTICS_SERVICE_URL=http://ultralytics-service:8000
      - TIMM_SERVICE_URL=http://timm-service:8000
    depends_on:
      - ultralytics-service
      - timm-service

  # Training Services
  ultralytics-service:
    build: ./platform/trainers/ultralytics
    ports:
      - "8001:8000"
    volumes:
      - ./datasets:/datasets
      - ./models:/models

  timm-service:
    build: ./platform/trainers/timm
    ports:
      - "8002:8000"
    volumes:
      - ./datasets:/datasets
      - ./models:/models
```

## 제거할 파일들

### MVP Backend에서 제거:
```bash
mvp/backend/app/utils/training_manager_k8s.py  # K8s 직접 관리 로직
mvp/backend/app/utils/training_manager.py      # 로컬 subprocess 관리
```

### MVP Training 폴더 이동:
```bash
mvp/training/ → platform/trainers/ (서비스별 분리)
```

## 비교: 작업량 & 기간

| 항목 | Platform 새로 구축 | MVP 기반 마이그레이션 |
|------|-------------------|---------------------|
| Backend API | 전체 재구현 (2-3주) | 일부 수정 (2-3일) |
| Frontend | 전체 재구현 (3-4주) | 그대로 사용 (0일) |
| Training Service | 새로 구현 (2-3주) | 분리만 (3-5일) |
| 통합 테스트 | 전체 재테스트 (1-2주) | 부분 테스트 (2-3일) |
| **총 기간** | **2-3개월** | **1-2주** |

## 결론 및 권장사항

### ✅ MVP 기반 마이그레이션을 강력 추천

**이유:**
1. **시간 효율**: 10배 빠른 개발 속도
2. **위험 감소**: 검증된 코드 재사용
3. **집중 가능**: Training Service 분리에만 집중
4. **점진적 개선**: 나중에 필요한 부분만 개선

**실행 순서:**
1. ✅ Training Service 분리 (Ultralytics 먼저)
2. ✅ Backend HTTP 클라이언트 구현
3. ✅ Local 환경 테스트
4. ✅ K8s 배포 테스트
5. ✅ 나머지 서비스 (TIMM, HuggingFace) 추가

**Platform 폴더는?**
- `platform/backend/` - 삭제 또는 참고용 보관
- `platform/frontend/` - 삭제 또는 참고용 보관
- `platform/trainers/` - 새로 생성 (Training Services)
- `platform/docs/` - 유지 (설계 문서)

## 다음 단계

1. **즉시 시작**: Ultralytics Training Service 분리
2. **병렬 작업**: MVP Backend HTTP 클라이언트 구현
3. **통합 테스트**: Local Docker Compose 환경 테스트
4. **Production**: K8s 배포

이 접근법으로 **1-2주 내에 완성도 높은 Platform 구현** 가능합니다!
