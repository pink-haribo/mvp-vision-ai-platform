# 시나리오 4: 모델 리스트 조회

## 개요

사용자가 새 학습 실험을 생성하기 위해 사용 가능한 모델 목록을 조회하는 과정입니다.

**목표:** 플랫폼이 지원하는 모든 모델 표시 (이름, 설명, 파라미터 수, 추천 배치 사이즈 등)

**핵심 차이:** 로컬과 배포 환경에서 **모델 데이터를 가져오는 방식**이 완전히 다릅니다!

---

## 로컬 환경 (개발)

### 환경 구성
```
Frontend:        http://localhost:3000
Backend:         http://localhost:8000
Training 코드:   C:\Users\...\mvp\training\ (로컬 파일시스템)
모델 레지스트리: Python 모듈 (직접 import)
```

### 상세 흐름

#### 1단계: 사용자가 "새 실험" 버튼 클릭

**위치:** 브라우저 (http://localhost:3000/projects/1)

**사용자 동작:**
```
프로젝트 상세 페이지에서 [+ 새 실험] 버튼 클릭
→ 모달 창 열림 (모델 선택 화면)
```

**Frontend 코드:**
```typescript
// mvp/frontend/components/NewExperimentModal.tsx
'use client';

export function NewExperimentModal({ projectId, isOpen, onClose }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedFramework, setSelectedFramework] = useState('all');

  useEffect(() => {
    if (isOpen) {
      fetchModels();
    }
  }, [isOpen, selectedFramework]);

  const fetchModels = async () => {
    const token = localStorage.getItem('access_token');

    // 프레임워크 필터가 있으면 쿼리 파라미터 추가
    let url = 'http://localhost:8000/api/v1/models/list';
    if (selectedFramework !== 'all') {
      url += `?framework=${selectedFramework}`;
    }

    const response = await fetch(url, {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    const data = await response.json();
    setModels(data);
    setLoading(false);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <h2>모델 선택</h2>

      {/* 프레임워크 필터 */}
      <FrameworkFilter
        value={selectedFramework}
        onChange={setSelectedFramework}
      />

      {/* 모델 목록 */}
      <ModelGrid models={models} />
    </Modal>
  );
}
```

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
GET http://localhost:8000/api/v1/models/list?framework=ultralytics
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**쿼리 파라미터:**
- `framework` (optional): `timm`, `ultralytics`, `huggingface`
- `task_type` (optional): `image_classification`, `object_detection`, etc.
- `priority` (optional): `0` (P0), `1` (P1), `2` (P2)

---

#### 3단계: Backend API 엔드포인트 실행

**위치:** `mvp/backend/app/api/models.py`

```python
@router.get("/list", response_model=List[ModelInfo])
async def list_models(
    framework: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    priority: Optional[int] = Query(None)
):
    """사용 가능한 모든 모델 목록 조회"""

    # 로컬 환경: model_registry를 직접 import 가능
    if MODEL_REGISTRY_AVAILABLE:
        all_models_data = get_all_models()  # Python 함수 호출
    else:
        # 배포 환경: Training Services에서 가져옴 (나중에 설명)
        all_models_data = fetch_from_training_services()

    # 필터링
    models = []
    for model_data in all_models_data:
        # framework 필터
        if framework and model_data["framework"] != framework:
            continue

        # task_type 필터
        if task_type:
            model_task_types = model_data.get("task_types", [])
            if task_type not in model_task_types:
                continue

        # priority 필터
        if priority is not None and model_data.get("priority") != priority:
            continue

        models.append(ModelInfo(**model_data))

    return models
```

---

#### 4단계: 로컬 모델 레지스트리 로딩 (중요!)

**위치:** `mvp/training/model_registry/__init__.py`

```python
# Backend가 training 디렉토리를 직접 import 가능 (로컬만 가능)

import sys
from pathlib import Path

# training 디렉토리를 Python path에 추가
training_dir = Path(__file__).parent.parent.parent.parent / "training"
if training_dir.exists():
    sys.path.insert(0, str(training_dir))

# 모델 레지스트리 import
from model_registry import (
    TIMM_MODEL_REGISTRY,
    ULTRALYTICS_MODEL_REGISTRY,
    HUGGINGFACE_MODEL_REGISTRY,
    get_all_models
)

MODEL_REGISTRY_AVAILABLE = True
```

**동작:**
```
Backend (mvp/backend/)
    ↓ import
Training Code (mvp/training/)
    ↓ import
Model Registry (mvp/training/model_registry/__init__.py)
    ↓ return
TIMM_MODEL_REGISTRY + ULTRALYTICS_MODEL_REGISTRY + HUGGINGFACE_MODEL_REGISTRY
```

**핵심:**
- 로컬 환경에서는 **파일시스템**을 통해 직접 Python 모듈 import 가능
- `mvp/backend`와 `mvp/training`이 같은 컴퓨터에 있음

---

#### 5단계: 모델 레지스트리 데이터 반환

**위치:** `mvp/training/model_registry/ultralytics_models.py`

```python
ULTRALYTICS_MODEL_REGISTRY = {
    "yolo11n": {
        "display_name": "YOLOv11 Nano (Detection)",
        "description": "Ultra-fast YOLO model for real-time object detection",
        "params": "2.6M",
        "input_size": 640,
        "task_types": ["object_detection"],
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 0.01,
        "tags": ["p0", "fast", "realtime"],
        "priority": 0,
        "benchmark": {
            "coco_map": 37.3,
            "inference_speed_v100": 200,
        },
        "use_cases": ["...", "..."],
        "pros": ["...", "..."],
        "cons": ["...", "..."],
        # ... 더 많은 메타데이터
    },

    "yolo11n-seg": {
        "display_name": "YOLOv11 Nano (Segmentation)",
        "description": "Ultra-fast instance segmentation model",
        "params": "2.9M",
        "task_types": ["instance_segmentation"],
        # ...
    },

    "yolo11n-pose": {
        "display_name": "YOLOv11 Nano (Pose)",
        "description": "Keypoint detection for pose estimation",
        "params": "3.3M",
        "task_types": ["pose_estimation"],
        # ...
    },

    "yolo_world_v2_s": {
        "display_name": "YOLO-World v2 Small",
        "description": "Zero-shot object detection with text prompts",
        "params": "15.2M",
        "task_types": ["object_detection"],
        "special_features": {
            "zero_shot": True,
            "text_prompts": True,
        },
        # ...
    },

    "sam2_t": {
        "display_name": "SAM2 Tiny",
        "description": "Segment Anything Model 2 for zero-shot segmentation",
        "params": "38M",
        "task_types": ["instance_segmentation", "panoptic_segmentation"],
        # ...
    },
}

def get_all_models():
    """모든 프레임워크의 모델 통합"""
    all_models = []

    # timm 모델 추가
    for model_name, info in TIMM_MODEL_REGISTRY.items():
        all_models.append({
            "framework": "timm",
            "model_name": model_name,
            **info
        })

    # ultralytics 모델 추가
    for model_name, info in ULTRALYTICS_MODEL_REGISTRY.items():
        all_models.append({
            "framework": "ultralytics",
            "model_name": model_name,
            **info
        })

    # huggingface 모델 추가
    for model_name, info in HUGGINGFACE_MODEL_REGISTRY.items():
        all_models.append({
            "framework": "huggingface",
            "model_name": model_name,
            **info
        })

    return all_models
```

**결과:**
- timm: 3개 모델
- ultralytics: 5개 모델
- huggingface: (미구현 시 0개)

**총 8개 모델 반환**

---

#### 6단계: Backend → Frontend 응답

**응답:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

[
  {
    "framework": "ultralytics",
    "model_name": "yolo11n",
    "display_name": "YOLOv11 Nano (Detection)",
    "description": "Ultra-fast YOLO model for real-time object detection",
    "params": "2.6M",
    "input_size": 640,
    "task_types": ["object_detection"],
    "pretrained_available": true,
    "recommended_batch_size": 16,
    "recommended_lr": 0.01,
    "tags": ["p0", "fast", "realtime"],
    "priority": 0
  },
  {
    "framework": "ultralytics",
    "model_name": "yolo11n-seg",
    "display_name": "YOLOv11 Nano (Segmentation)",
    ...
  },
  {
    "framework": "ultralytics",
    "model_name": "yolo11n-pose",
    ...
  },
  {
    "framework": "ultralytics",
    "model_name": "yolo_world_v2_s",
    ...
  },
  {
    "framework": "ultralytics",
    "model_name": "sam2_t",
    ...
  }
]
```

---

#### 7단계: Frontend 화면 렌더링

**위치:** 브라우저 모달 창

```typescript
// ModelGrid 렌더링
return (
  <div className="model-grid">
    {models.map(model => (
      <ModelCard
        key={`${model.framework}-${model.model_name}`}
        model={model}
        onSelect={() => handleModelSelect(model)}
      />
    ))}
  </div>
);
```

**화면:**
```
┌─────────────────────────────────────────────────────────────┐
│ 모델 선택                                          [✕ 닫기] │
├─────────────────────────────────────────────────────────────┤
│ 프레임워크: [ultralytics ▼]                                │
├─────────────────────────────────────────────────────────────┤
│ ┌───────────────────┐ ┌───────────────────┐                │
│ │ 🏃 YOLOv11 Nano   │ │ 🎭 YOLOv11n-seg   │                │
│ │ (Detection)       │ │ (Segmentation)    │                │
│ │                   │ │                   │                │
│ │ 2.6M params       │ │ 2.9M params       │                │
│ │ 640x640           │ │ 640x640           │                │
│ │                   │ │                   │                │
│ │ Ultra-fast YOLO.. │ │ Instance segment..│                │
│ │                   │ │                   │                │
│ │ Batch: 16         │ │ Batch: 16         │                │
│ │ LR: 0.01          │ │ LR: 0.01          │                │
│ │                   │ │                   │                │
│ │ [선택] [상세정보] │ │ [선택] [상세정보] │                │
│ └───────────────────┘ └───────────────────┘                │
│                                                             │
│ ┌───────────────────┐ ┌───────────────────┐                │
│ │ 🤸 YOLOv11n-pose  │ │ 🌍 YOLO-World v2 │                │
│ │ ...               │ │ ...               │                │
│ └───────────────────┘ └───────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 배포 환경 (Railway) - 핵심 차이!

### 환경 구성
```
Frontend:           https://frontend-production-xxxx.up.railway.app
Backend:            https://backend-production-xxxx.up.railway.app
timm-service:       https://timm-service-production-xxxx.up.railway.app
ultralytics-service: https://ultralytics-service-production-xxxx.up.railway.app
huggingface-service: https://huggingface-service-production-xxxx.up.railway.app
```

**핵심 차이:**
- Backend와 Training 코드가 **별도의 컨테이너**에서 실행
- Backend는 Training 코드를 **직접 import 불가능**
- 대신 **HTTP API**로 Training Services에서 모델 목록 가져옴

---

### 상세 흐름

#### 1단계: 사용자가 "새 실험" 버튼 클릭

**동작:** 로컬과 동일

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
GET https://backend-production-xxxx.up.railway.app/api/v1/models/list?framework=ultralytics
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**차이점:**
- HTTPS 사용
- Railway URL

---

#### 3단계: Backend API 엔드포인트 실행

**위치:** `mvp/backend/app/api/models.py` (Railway 컨테이너)

```python
@router.get("/list", response_model=List[ModelInfo])
async def list_models(framework: Optional[str] = None, ...):
    """모델 목록 조회"""

    # 배포 환경: model_registry import 불가능
    if MODEL_REGISTRY_AVAILABLE:
        # 로컬: 직접 import (이미 설명)
        all_models_data = get_all_models()
    else:
        # 배포: Training Services에서 HTTP로 가져옴 (중요!)
        all_models_data = get_all_models()  # 이 함수가 다르게 동작

    # 필터링 (동일)
    ...
```

**환경 변수 확인:**
```python
# mvp/backend/app/api/models.py

# model_registry import 실패 시
MODEL_REGISTRY_AVAILABLE = False

# Training Services URL (Railway 환경변수)
TIMM_SERVICE_URL = os.getenv("TIMM_SERVICE_URL")
ULTRALYTICS_SERVICE_URL = os.getenv("ULTRALYTICS_SERVICE_URL")
HUGGINGFACE_SERVICE_URL = os.getenv("HUGGINGFACE_SERVICE_URL")
```

**Railway 환경변수:**
```bash
# Backend 서비스 환경변수
TIMM_SERVICE_URL=https://timm-service-production-xxxx.up.railway.app
ULTRALYTICS_SERVICE_URL=https://ultralytics-service-production-xxxx.up.railway.app
HUGGINGFACE_SERVICE_URL=https://huggingface-service-production-xxxx.up.railway.app
```

---

#### 4단계: Backend → Training Services HTTP 요청 (핵심!)

**Backend가 Training Services에 HTTP 요청**

**코드:**
```python
# mvp/backend/app/api/models.py

def fetch_models_from_service(service_url: str, timeout: int = 5) -> List[Dict]:
    """Training Service에서 모델 목록 가져오기"""
    try:
        response = requests.get(f"{service_url}/models/list", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
    except Exception as e:
        print(f"[WARNING] Failed to fetch models from {service_url}: {e}")

    return []

def get_all_models():
    """모든 Training Services에서 모델 수집"""
    models = []

    # Training Services URL 매핑
    training_services = {
        "timm": os.getenv("TIMM_SERVICE_URL"),
        "ultralytics": os.getenv("ULTRALYTICS_SERVICE_URL"),
        "huggingface": os.getenv("HUGGINGFACE_SERVICE_URL"),
    }

    # 각 서비스에서 모델 가져오기
    for framework, service_url in training_services.items():
        if service_url:
            service_models = fetch_models_from_service(service_url)
            models.extend(service_models)

    return models
```

**HTTP 요청:**
```http
# Backend → timm-service
GET https://timm-service-production-xxxx.up.railway.app/models/list
Timeout: 5초

# Backend → ultralytics-service
GET https://ultralytics-service-production-xxxx.up.railway.app/models/list
Timeout: 5초

# Backend → huggingface-service
GET https://huggingface-service-production-xxxx.up.railway.app/models/list
Timeout: 5초
```

**동작:**
- Backend가 **각 Training Service에 병렬 요청**
- Training Service가 자신의 모델 레지스트리 반환
- Backend가 모든 결과를 **병합**

---

#### 5단계: Training Service API 실행 (중요!)

**위치:** `mvp/training/api_server.py` (ultralytics-service 컨테이너)

```python
# 환경변수로 프레임워크 감지
FRAMEWORK = os.environ.get("FRAMEWORK", "unknown")  # "ultralytics"

# 모델 레지스트리 import (Training Service는 코드가 있음)
from model_registry import get_all_models
from model_registry.ultralytics_models import ULTRALYTICS_MODEL_REGISTRY

@app.get("/models/list")
async def list_models():
    """이 Training Service의 모델 목록 반환"""

    models = []

    if FRAMEWORK == "timm":
        for model_name, info in TIMM_MODEL_REGISTRY.items():
            models.append({
                "framework": "timm",
                "model_name": model_name,
                **info
            })
    elif FRAMEWORK == "ultralytics":
        for model_name, info in ULTRALYTICS_MODEL_REGISTRY.items():
            models.append({
                "framework": "ultralytics",
                "model_name": model_name,
                **info
            })
    # ... huggingface도 동일

    return {
        "framework": FRAMEWORK,
        "model_count": len(models),
        "models": models
    }
```

**동작:**
```
ultralytics-service 컨테이너:
  ├─ Python 3.11 환경
  ├─ ultralytics 라이브러리 설치
  ├─ model_registry 코드 포함
  ├─ ENV FRAMEWORK=ultralytics
  └─ api_server.py 실행 중
```

**응답:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "framework": "ultralytics",
  "model_count": 5,
  "models": [
    {
      "framework": "ultralytics",
      "model_name": "yolo11n",
      "display_name": "YOLOv11 Nano (Detection)",
      ...
    },
    {
      "framework": "ultralytics",
      "model_name": "yolo11n-seg",
      ...
    },
    {
      "framework": "ultralytics",
      "model_name": "yolo11n-pose",
      ...
    },
    {
      "framework": "ultralytics",
      "model_name": "yolo_world_v2_s",
      ...
    },
    {
      "framework": "ultralytics",
      "model_name": "sam2_t",
      ...
    }
  ]
}
```

---

#### 6단계: Backend가 결과 병합

**Backend 코드:**
```python
# 각 Training Service에서 받은 모델 병합
models = []

# timm-service 결과 (3개 모델)
models.extend(timm_models)

# ultralytics-service 결과 (5개 모델)
models.extend(ultralytics_models)

# huggingface-service 결과 (0개 모델, 아직 미구현)
models.extend(huggingface_models)

# 총 8개 모델 반환
return models
```

---

#### 7단계: Backend → Frontend 응답

**응답:** 로컬과 동일 (JSON 형식)

**차이점:**
- 데이터 출처가 다름:
  - 로컬: 직접 Python import
  - 배포: HTTP API로 Training Services에서 가져옴

---

## 주요 차이점 요약

| 구분 | 로컬 환경 | 배포 환경 (Railway) |
|------|----------|-------------------|
| **Frontend URL** | http://localhost:3000 | https://frontend-production-xxxx.up.railway.app |
| **Backend URL** | http://localhost:8000 | https://backend-production-xxxx.up.railway.app |
| **모델 데이터 출처** | Python 직접 import | HTTP API (Training Services) |
| **model_registry 위치** | 로컬 파일시스템 | Training Service 컨테이너 |
| **Backend와 Training** | 같은 컴퓨터 | 별도 컨테이너 (격리) |
| **모델 로딩 방식** | `from model_registry import ...` | `requests.get(...)/models/list` |
| **응답 시간** | ~10-20ms (직접 import) | ~100-200ms (HTTP 요청) |
| **캐싱** | Python module cache | HTTP response cache (선택적) |
| **에러 처리** | ImportError | HTTP timeout, connection error |

---

## 아키텍처 다이어그램

### 로컬 환경

```
┌─────────────────────────────────────────────────────┐
│ 개발자 컴퓨터                                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Frontend (Next.js)                                │
│  localhost:3000                                     │
│         │                                           │
│         │ HTTP GET /api/v1/models/list             │
│         ▼                                           │
│  Backend (FastAPI)                                 │
│  localhost:8000                                     │
│         │                                           │
│         │ Python import                            │
│         ▼                                           │
│  Training Code (model_registry)                    │
│  C:\...\mvp\training\model_registry\               │
│         │                                           │
│         │ return ULTRALYTICS_MODEL_REGISTRY         │
│         ▼                                           │
│  Backend → Frontend                                │
│  (JSON response)                                    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 배포 환경 (Railway)

```
┌─────────────────────────────────────────────────────────────┐
│ Railway Platform                                             │
├───────────────┬────────────────┬────────────────────────────┤
│ Frontend      │ Backend        │ Training Services           │
│ (Next.js)     │ (FastAPI)      │                            │
│               │                │  ┌──────────────────────┐   │
│ https://...   │ https://...    │  │ timm-service         │   │
│               │                │  │ https://...          │   │
│       │       │       │        │  │ ┌──────────────────┐ │   │
│       │ HTTP  │       │ HTTP   │  │ │ api_server.py    │ │   │
│       └───────┼───────┘        │  │ │ FRAMEWORK=timm   │ │   │
│               │                │  │ │ /models/list     │ │   │
│               │        ┌───────┼──┼─┤ → 3 timm models  │ │   │
│               │        │       │  │ └──────────────────┘ │   │
│               │        │       │  └──────────────────────┘   │
│               │        │       │                             │
│               │        │       │  ┌──────────────────────┐   │
│               │        │       │  │ ultralytics-service  │   │
│               │        │       │  │ https://...          │   │
│               │        │       │  │ ┌──────────────────┐ │   │
│               │        │       │  │ │ api_server.py    │ │   │
│               │        │       │  │ │ FRAMEWORK=ultra  │ │   │
│               │        │       │  │ │ /models/list     │ │   │
│               │        └───────┼──┼─┤ → 5 ultra models │ │   │
│               │                │  │ └──────────────────┘ │   │
│               │                │  └──────────────────────┘   │
│               │                │                             │
│               │ Merge results  │                             │
│               │ (8 models)     │                             │
│               │        │       │                             │
│       ┌───────┴────────┘       │                             │
│       │ JSON response          │                             │
│       ▼                        │                             │
│  Frontend renders              │                             │
│                                │                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 성능 비교

### 로컬 환경 (개발)

```
총 응답 시간: ~20ms

1. Frontend → Backend: ~1ms
2. Backend Python import: ~5ms (첫 실행, 이후 캐시됨)
3. 모델 데이터 직렬화: ~5ms
4. Backend → Frontend: ~1ms
```

### 배포 환경 (Railway)

```
총 응답 시간: ~200-400ms

1. Frontend → Backend: ~50-100ms (HTTPS, 인터넷)
2. Backend → timm-service: ~50-100ms (HTTP, Railway 내부 네트워크)
3. Backend → ultralytics-service: ~50-100ms (병렬 요청)
4. Backend → huggingface-service: ~50-100ms (병렬 요청)
5. Backend 결과 병합: ~5ms
6. Backend → Frontend: ~50-100ms (HTTPS, 인터넷)
```

**차이:** 배포 환경이 10-20배 느림 (네트워크 지연)

**최적화 방법:**
- Backend에서 모델 목록 **캐싱** (1시간 TTL)
- Training Services 병렬 요청 (이미 구현됨)

---

## 모델 추가 시 차이점

### 로컬 환경

**새 모델 추가:**
```python
# mvp/training/model_registry/ultralytics_models.py

ULTRALYTICS_MODEL_REGISTRY = {
    # ... 기존 모델들 ...

    "yolo11l": {  # 🆕 새 모델 추가
        "display_name": "YOLOv11 Large",
        "description": "High-accuracy YOLO model",
        "params": "25.3M",
        "task_types": ["object_detection"],
        "priority": 1,
        # ...
    }
}
```

**확인:**
```bash
# Backend 재시작 필요 (Python module reload)
cd mvp/backend
../../mvp/backend/venv/Scripts/python.exe -m uvicorn app.main:app --reload

# Frontend에서 확인 (자동 반영)
curl http://localhost:8000/api/v1/models/list?framework=ultralytics
# → yolo11l 포함됨
```

**소요 시간:** ~1분 (코드 수정 + Backend 재시작)

---

### 배포 환경 (Railway)

**새 모델 추가:**
```python
# mvp/training/model_registry/ultralytics_models.py

ULTRALYTICS_MODEL_REGISTRY = {
    # ... 기존 모델들 ...

    "yolo11l": {  # 🆕 새 모델 추가
        ...
    }
}
```

**배포:**
```bash
git add mvp/training/model_registry/ultralytics_models.py
git commit -m "feat: add YOLO11l model"
git push
```

**Railway 자동 배포:**
```
Railway가 자동으로 감지:
  1. ultralytics-service 재빌드 (~5-7분)
  2. 새 컨테이너 배포
  3. 헬스체크 통과 후 트래픽 전환

Backend는 재배포 불필요!
  → 환경변수만 있으면 자동으로 새 모델 표시
```

**확인:**
```bash
# ultralytics-service 직접 확인
curl https://ultralytics-service-production-xxxx.up.railway.app/models/list
# → yolo11l 포함됨

# Backend 확인 (자동 반영!)
curl https://backend-production-xxxx.up.railway.app/api/v1/models/list?framework=ultralytics
# → yolo11l 포함됨
```

**소요 시간:** ~5-7분 (Railway 빌드 + 배포)

**장점:**
- Backend 수정 불필요
- Backend 재배포 불필요
- 프로덕션에서 안전 (rollback 가능)

---

## 관련 파일

### Frontend
- `mvp/frontend/components/NewExperimentModal.tsx` - 모델 선택 모달
- `mvp/frontend/components/ModelCard.tsx` - 모델 카드 컴포넌트

### Backend
- `mvp/backend/app/api/models.py` - 모델 API
- `mvp/backend/app/api/models.py:fetch_models_from_service()` - Training Service 조회

### Training Services
- `mvp/training/api_server.py` - Training Service API
- `mvp/training/model_registry/__init__.py` - 모델 레지스트리 통합
- `mvp/training/model_registry/timm_models.py` - timm 모델 정의
- `mvp/training/model_registry/ultralytics_models.py` - ultralytics 모델 정의

### Documentation
- `docs/production/DYNAMIC_MODEL_REGISTRATION.md` - 동적 모델 등록 문서

---

## 디버깅 팁

### 로컬: 모델이 안 보일 때

**확인:**
```python
# Backend 로그 확인
[INFO] MODEL_REGISTRY_AVAILABLE: True
[INFO] Loaded 8 models from local registry
```

**문제 해결:**
```bash
# model_registry import 실패 시
cd mvp/backend
python -c "import sys; sys.path.insert(0, '../training'); from model_registry import get_all_models; print(get_all_models())"
```

---

### 배포: 모델이 안 보일 때

**확인:**
```bash
# Railway Backend 로그
[WARNING] Failed to fetch models from https://ultralytics-service-...
[INFO] No Training Services available, using static model definitions
```

**문제:**
- Training Service URL 설정 안 됨
- Training Service가 다운됨

**해결:**
```bash
# 1. 환경변수 확인
railway run env | grep SERVICE_URL

# 2. Training Service 헬스체크
curl https://ultralytics-service-production-xxxx.up.railway.app/health

# 3. 모델 API 직접 확인
curl https://ultralytics-service-production-xxxx.up.railway.app/models/list
```

---

## 캐싱 최적화 (배포 환경)

**문제:** 매번 HTTP 요청하면 느림 (~200-400ms)

**해결:** Backend에서 모델 목록 캐싱

```python
# mvp/backend/app/api/models.py

from functools import lru_cache
from datetime import datetime, timedelta

_model_cache = None
_cache_timestamp = None
CACHE_TTL = timedelta(hours=1)  # 1시간 캐시

def get_all_models():
    """모델 목록 조회 (캐싱 포함)"""
    global _model_cache, _cache_timestamp

    # 캐시 유효성 확인
    if _model_cache and _cache_timestamp:
        if datetime.now() - _cache_timestamp < CACHE_TTL:
            print("[INFO] Returning cached models")
            return _model_cache

    # 캐시 없거나 만료됨 → Training Services에서 가져오기
    print("[INFO] Fetching models from Training Services")
    models = fetch_from_training_services()

    # 캐시 업데이트
    _model_cache = models
    _cache_timestamp = datetime.now()

    return models
```

**효과:**
- 첫 요청: ~200-400ms (Training Services 조회)
- 이후 요청: ~10-20ms (캐시에서 반환)
- 1시간마다 자동 갱신
