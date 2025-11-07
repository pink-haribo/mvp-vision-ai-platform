# Model Plugin Validation Plan

## 목표

Vision AI Platform의 핵심 목표인 "어떤 프레임워크/모델이든 쉽게 플러그인 가능한 구조"를 검증하고, 신규 모델/프레임워크 도입 시 발생하는 문제점을 파악하여 개선한다.

## 검증 전략

실제 모델을 추가하면서 다음 항목들을 평가한다:

1. **Adapter 패턴의 효율성**: 신규 모델 추가 시 얼마나 적은 코드 변경이 필요한가?
2. **프레임워크 독립성**: 플랫폼 코드 수정 없이 새 프레임워크를 지원할 수 있는가?
3. **공통 인터페이스의 적합성**: TaskType, MetricsResult, InferenceResult 등이 모든 모델에 적용 가능한가?
4. **Configuration Schema의 유연성**: 프레임워크별 설정을 동적으로 생성할 수 있는가?
5. **Trainer 의존성 격리**: 플랫폼과 trainer가 독립적으로 동작하는가?

## 현재 상태 분석

### 지원 중인 프레임워크

#### 1. timm (PyTorch Image Models)
- **Adapter**: `mvp/training/adapters/timm_adapter.py`
- **지원 Task**: Image Classification
- **현재 테스트된 모델**:
  - ResNet-50
  - ResNet-18
  - EfficientNet-B0

#### 2. Ultralytics YOLO
- **Adapter**: `mvp/training/adapters/ultralytics_adapter.py`
- **지원 Task**: Object Detection, Instance Segmentation, Pose Estimation, Image Classification
- **현재 테스트된 모델**:
  - YOLOv8n (detection)
  - YOLOv8s (detection)
  - YOLOv8m (detection)
  - YOLOv8n-seg (segmentation)
  - YOLOv8n-pose (pose estimation)

### 아키텍처 강점

✅ **잘 설계된 부분**:
1. **TrainingAdapter 추상 클래스**: 명확한 인터페이스 정의
2. **Task-agnostic 데이터 구조**: InferenceResult, MetricsResult가 모든 task에 대응
3. **ConfigSchema 동적 생성**: 프레임워크별 설정을 UI에서 자동 렌더링
4. **Validation 시스템**: 프레임워크 독립적인 validation 결과 저장
5. **Checkpoint 관리**: 통일된 checkpoint 저장/로드 인터페이스

### 아키텍처 제약

⚠️ **개선이 필요한 부분**:
1. **모델 레지스트리 부재**: 지원 모델 목록이 하드코딩됨
2. **Dataset Format 제약**: ImageFolder, YOLO, COCO만 지원
3. **Custom 전처리 지원 부족**: 프레임워크별 특수 전처리 어려움
4. **의존성 충돌 가능성**: 각 프레임워크의 패키지 버전 충돌
5. **에러 핸들링 표준화 부족**: 프레임워크별 에러 메시지 형식 다름

---

## Phase 1: 기존 프레임워크 모델 추가

**목표**: timm과 Ultralytics의 다양한 모델을 추가하며 Adapter 패턴의 확장성 검증

**기간**: 1주

### 1.1 timm 모델 추가

#### 추가할 모델 (총 10개)

| 카테고리 | 모델명 | 특징 | 난이도 |
|---------|--------|------|--------|
| **Classic CNN** | VGG-16 | 고전적 deep CNN | ⭐ Easy |
| **Classic CNN** | VGG-19 | VGG-16보다 깊은 버전 | ⭐ Easy |
| **ResNet 계열** | ResNet-34 | 경량 ResNet | ⭐ Easy |
| **ResNet 계열** | ResNet-101 | 대형 ResNet | ⭐ Easy |
| **EfficientNet 계열** | EfficientNet-B1 | 중형 EfficientNet | ⭐ Easy |
| **EfficientNet 계열** | EfficientNet-B4 | 대형 EfficientNet | ⭐ Easy |
| **Vision Transformer** | ViT-Base | Transformer 기반 | ⭐⭐ Medium |
| **Vision Transformer** | ViT-Large | 대형 ViT | ⭐⭐ Medium |
| **Mobile** | MobileNetV3-Large | 경량 모바일 모델 | ⭐⭐ Medium |
| **Mobile** | MobileNetV3-Small | 초경량 모바일 모델 | ⭐⭐ Medium |

#### 구현 방법

**Step 1**: 모델 메타데이터 정의

```python
# mvp/training/model_registry/timm_models.py (신규 파일)

from typing import Dict, Any

TIMM_MODEL_REGISTRY = {
    # ResNet 계열
    "resnet18": {
        "display_name": "ResNet-18",
        "description": "18-layer Residual Network",
        "params": "11.7M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["classic", "lightweight", "fast"]
    },
    "resnet34": {
        "display_name": "ResNet-34",
        "description": "34-layer Residual Network",
        "params": "21.8M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["classic", "medium"]
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "description": "50-layer Residual Network with bottleneck blocks",
        "params": "25.6M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.001,
        "tags": ["classic", "popular", "baseline"]
    },
    "resnet101": {
        "display_name": "ResNet-101",
        "description": "101-layer Residual Network",
        "params": "44.5M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 0.0005,
        "tags": ["classic", "heavy", "accurate"]
    },

    # VGG 계열
    "vgg16": {
        "display_name": "VGG-16",
        "description": "16-layer VGG Network (Very Deep CNN)",
        "params": "138M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.0001,
        "tags": ["classic", "heavy"]
    },
    "vgg19": {
        "display_name": "VGG-19",
        "description": "19-layer VGG Network",
        "params": "144M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 0.0001,
        "tags": ["classic", "heavy"]
    },

    # EfficientNet 계열
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "description": "Efficient CNN with compound scaling (baseline)",
        "params": "5.3M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["efficient", "lightweight", "modern"]
    },
    "efficientnet_b1": {
        "display_name": "EfficientNet-B1",
        "description": "EfficientNet scaled up from B0",
        "params": "7.8M",
        "input_size": 240,
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.001,
        "tags": ["efficient", "modern"]
    },
    "efficientnet_b4": {
        "display_name": "EfficientNet-B4",
        "description": "Large EfficientNet variant",
        "params": "19M",
        "input_size": 380,
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 0.0005,
        "tags": ["efficient", "accurate", "modern"]
    },

    # Vision Transformer 계열
    "vit_base_patch16_224": {
        "display_name": "ViT-Base/16",
        "description": "Vision Transformer (Base, 16x16 patches)",
        "params": "86M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.0003,
        "tags": ["transformer", "modern", "attention"]
    },
    "vit_large_patch16_224": {
        "display_name": "ViT-Large/16",
        "description": "Vision Transformer (Large, 16x16 patches)",
        "params": "307M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 8,
        "recommended_lr": 0.0001,
        "tags": ["transformer", "heavy", "accurate"]
    },

    # Mobile 계열
    "mobilenetv3_large_100": {
        "display_name": "MobileNetV3-Large",
        "description": "Efficient mobile CNN (large variant)",
        "params": "5.5M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 128,
        "recommended_lr": 0.001,
        "tags": ["mobile", "efficient", "fast"]
    },
    "mobilenetv3_small_100": {
        "display_name": "MobileNetV3-Small",
        "description": "Ultra-lightweight mobile CNN",
        "params": "2.5M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 256,
        "recommended_lr": 0.001,
        "tags": ["mobile", "ultralight", "fast"]
    },
}

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model metadata."""
    return TIMM_MODEL_REGISTRY.get(model_name, {})

def list_models(tags: list = None) -> list:
    """List models, optionally filtered by tags."""
    if not tags:
        return list(TIMM_MODEL_REGISTRY.keys())

    return [
        name for name, info in TIMM_MODEL_REGISTRY.items()
        if any(tag in info.get("tags", []) for tag in tags)
    ]
```

**Step 2**: TimmAdapter 확장 불필요 확인

기존 TimmAdapter는 이미 모든 timm 모델을 지원하도록 설계됨:

```python
# mvp/training/adapters/timm_adapter.py
def prepare_model(self) -> None:
    """Initialize model."""
    import timm

    # timm.create_model()은 모든 timm 모델을 지원
    self.model = timm.create_model(
        self.model_config.model_name,  # 동적으로 모델명 사용
        pretrained=self.model_config.pretrained,
        num_classes=self.model_config.num_classes
    )
```

**따라서 Adapter 코드 수정 불필요!** ✅

**Step 3**: API에 모델 목록 엔드포인트 추가

```python
# mvp/backend/app/api/models.py (신규 파일)

from fastapi import APIRouter, Query
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/models", tags=["models"])

class ModelInfo(BaseModel):
    """Model information schema."""
    model_name: str
    display_name: str
    description: str
    framework: str
    task_type: str
    params: str
    input_size: int
    pretrained_available: bool
    recommended_batch_size: int
    recommended_lr: float
    tags: List[str]

@router.get("/list", response_model=List[ModelInfo])
async def list_models(
    framework: Optional[str] = Query(None, description="Filter by framework"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    tags: Optional[str] = Query(None, description="Comma-separated tags")
):
    """
    List available models.

    Examples:
    - /models/list?framework=timm
    - /models/list?task_type=image_classification
    - /models/list?tags=lightweight,fast
    """
    from training.model_registry.timm_models import TIMM_MODEL_REGISTRY
    from training.model_registry.ultralytics_models import ULTRALYTICS_MODEL_REGISTRY

    all_models = []

    # Collect timm models
    if not framework or framework == "timm":
        for name, info in TIMM_MODEL_REGISTRY.items():
            all_models.append(ModelInfo(
                model_name=name,
                framework="timm",
                task_type="image_classification",
                **info
            ))

    # Collect ultralytics models
    if not framework or framework == "ultralytics":
        for name, info in ULTRALYTICS_MODEL_REGISTRY.items():
            all_models.append(ModelInfo(
                model_name=name,
                framework="ultralytics",
                **info
            ))

    # Filter by tags if provided
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        all_models = [
            m for m in all_models
            if any(tag in m.tags for tag in tag_list)
        ]

    # Filter by task_type if provided
    if task_type:
        all_models = [m for m in all_models if m.task_type == task_type]

    return all_models

@router.get("/{framework}/{model_name}", response_model=ModelInfo)
async def get_model_info(framework: str, model_name: str):
    """Get detailed model information."""
    from training.model_registry.timm_models import get_model_info as get_timm_info
    from training.model_registry.ultralytics_models import get_model_info as get_yolo_info

    if framework == "timm":
        info = get_timm_info(model_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        return ModelInfo(
            model_name=model_name,
            framework="timm",
            task_type="image_classification",
            **info
        )
    elif framework == "ultralytics":
        info = get_yolo_info(model_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        return ModelInfo(
            model_name=model_name,
            framework="ultralytics",
            **info
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown framework: {framework}")
```

**Step 4**: Frontend에 모델 선택 UI 개선

```tsx
// mvp/frontend/components/training/ModelSelector.tsx

import { useState, useEffect } from 'react';

interface ModelInfo {
  model_name: string;
  display_name: string;
  description: string;
  framework: string;
  task_type: string;
  params: string;
  tags: string[];
  recommended_batch_size: number;
  recommended_lr: number;
}

export default function ModelSelector({ onSelect }: { onSelect: (model: ModelInfo) => void }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [filter, setFilter] = useState({ framework: '', tags: '' });

  useEffect(() => {
    fetchModels();
  }, [filter]);

  const fetchModels = async () => {
    const params = new URLSearchParams();
    if (filter.framework) params.append('framework', filter.framework);
    if (filter.tags) params.append('tags', filter.tags);

    const response = await fetch(`/api/v1/models/list?${params}`);
    const data = await response.json();
    setModels(data);
  };

  return (
    <div className="model-selector">
      {/* Filter Section */}
      <div className="filters">
        <select onChange={(e) => setFilter({...filter, framework: e.target.value})}>
          <option value="">All Frameworks</option>
          <option value="timm">timm (Classification)</option>
          <option value="ultralytics">Ultralytics (Detection/Segmentation)</option>
        </select>

        <input
          type="text"
          placeholder="Filter by tags (e.g., lightweight, fast)"
          onChange={(e) => setFilter({...filter, tags: e.target.value})}
        />
      </div>

      {/* Model Grid */}
      <div className="model-grid">
        {models.map((model) => (
          <div key={model.model_name} className="model-card" onClick={() => onSelect(model)}>
            <h3>{model.display_name}</h3>
            <p>{model.description}</p>
            <div className="model-meta">
              <span>Params: {model.params}</span>
              <span>Batch Size: {model.recommended_batch_size}</span>
            </div>
            <div className="model-tags">
              {model.tags.map(tag => <span key={tag} className="tag">{tag}</span>)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

#### 검증 기준

각 모델에 대해 다음을 확인:

- [ ] 모델이 정상적으로 로드되는가?
- [ ] Training이 에러 없이 완료되는가?
- [ ] Validation metrics가 정상적으로 계산되는가?
- [ ] Checkpoint 저장/로드가 동작하는가?
- [ ] Inference가 정상적으로 수행되는가?
- [ ] 프레임워크 특수 설정(예: ViT의 patch_size)이 지원되는가?

#### 예상 문제점

1. **메모리 부족**: VGG, ViT-Large는 큰 메모리 필요
   - **해결책**: batch_size 자동 조정, gradient checkpointing 옵션 추가

2. **입력 크기 제약**: EfficientNet-B4는 380x380 입력 필요
   - **해결책**: 모델별 input_size를 동적으로 설정

3. **전처리 차이**: ViT는 특수한 normalization 필요
   - **해결책**: ConfigSchema에 preprocessing 설정 추가

---

### 1.2 Ultralytics 모델 추가

#### 추가할 모델 (총 8개)

| 태스크 | 모델명 | 특징 | 난이도 |
|--------|--------|------|--------|
| **Detection** | YOLOv8l | Large detection model | ⭐ Easy |
| **Detection** | YOLOv8x | Extra-large detection | ⭐ Easy |
| **Detection** | YOLOv9c | YOLOv9 compact | ⭐⭐ Medium |
| **Detection** | YOLOv9e | YOLOv9 extended | ⭐⭐ Medium |
| **Segmentation** | YOLOv8m-seg | Medium segmentation | ⭐ Easy |
| **Segmentation** | YOLOv8l-seg | Large segmentation | ⭐ Easy |
| **Pose** | YOLOv8m-pose | Medium pose estimation | ⭐ Easy |
| **OBB** | YOLOv8n-obb | Oriented bounding box | ⭐⭐⭐ Hard |

#### 구현 방법

**Step 1**: 모델 레지스트리 작성

```python
# mvp/training/model_registry/ultralytics_models.py (신규 파일)

ULTRALYTICS_MODEL_REGISTRY = {
    # Detection - YOLOv8
    "yolov8n": {
        "display_name": "YOLOv8n",
        "description": "YOLOv8 Nano - Ultra-fast detection",
        "params": "3.2M",
        "input_size": 640,
        "task_type": "object_detection",
        "pretrained_available": True,
        "recommended_batch_size": 64,
        "recommended_lr": 0.01,
        "tags": ["detection", "ultralight", "realtime"]
    },
    "yolov8s": {
        "display_name": "YOLOv8s",
        "description": "YOLOv8 Small - Fast detection",
        "params": "11.2M",
        "input_size": 640,
        "task_type": "object_detection",
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.01,
        "tags": ["detection", "lightweight", "fast"]
    },
    "yolov8m": {
        "display_name": "YOLOv8m",
        "description": "YOLOv8 Medium - Balanced detection",
        "params": "25.9M",
        "input_size": 640,
        "task_type": "object_detection",
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 0.01,
        "tags": ["detection", "balanced"]
    },
    "yolov8l": {
        "display_name": "YOLOv8l",
        "description": "YOLOv8 Large - High accuracy detection",
        "params": "43.7M",
        "input_size": 640,
        "task_type": "object_detection",
        "pretrained_available": True,
        "recommended_batch_size": 8,
        "recommended_lr": 0.01,
        "tags": ["detection", "accurate"]
    },
    "yolov8x": {
        "display_name": "YOLOv8x",
        "description": "YOLOv8 Extra-Large - Maximum accuracy",
        "params": "68.2M",
        "input_size": 640,
        "task_type": "object_detection",
        "pretrained_available": True,
        "recommended_batch_size": 4,
        "recommended_lr": 0.01,
        "tags": ["detection", "heavy", "accurate"]
    },

    # Detection - YOLOv9
    "yolov9c": {
        "display_name": "YOLOv9c",
        "description": "YOLOv9 Compact - Latest YOLO with programmable gradient",
        "params": "25.5M",
        "input_size": 640,
        "task_type": "object_detection",
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 0.01,
        "tags": ["detection", "latest", "balanced"]
    },
    "yolov9e": {
        "display_name": "YOLOv9e",
        "description": "YOLOv9 Extended - State-of-the-art detection",
        "params": "58.1M",
        "input_size": 640,
        "task_type": "object_detection",
        "pretrained_available": True,
        "recommended_batch_size": 8,
        "recommended_lr": 0.01,
        "tags": ["detection", "latest", "accurate", "sota"]
    },

    # Segmentation
    "yolov8n-seg": {
        "display_name": "YOLOv8n-Seg",
        "description": "YOLOv8 Nano Segmentation",
        "params": "3.4M",
        "input_size": 640,
        "task_type": "instance_segmentation",
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.01,
        "tags": ["segmentation", "lightweight", "fast"]
    },
    "yolov8m-seg": {
        "display_name": "YOLOv8m-Seg",
        "description": "YOLOv8 Medium Segmentation",
        "params": "27.3M",
        "input_size": 640,
        "task_type": "instance_segmentation",
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 0.01,
        "tags": ["segmentation", "balanced"]
    },
    "yolov8l-seg": {
        "display_name": "YOLOv8l-Seg",
        "description": "YOLOv8 Large Segmentation",
        "params": "46.0M",
        "input_size": 640,
        "task_type": "instance_segmentation",
        "pretrained_available": True,
        "recommended_batch_size": 8,
        "recommended_lr": 0.01,
        "tags": ["segmentation", "accurate"]
    },

    # Pose Estimation
    "yolov8n-pose": {
        "display_name": "YOLOv8n-Pose",
        "description": "YOLOv8 Nano Pose - 17 keypoints detection",
        "params": "3.3M",
        "input_size": 640,
        "task_type": "pose_estimation",
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.01,
        "tags": ["pose", "lightweight", "keypoints"]
    },
    "yolov8m-pose": {
        "display_name": "YOLOv8m-Pose",
        "description": "YOLOv8 Medium Pose",
        "params": "26.4M",
        "input_size": 640,
        "task_type": "pose_estimation",
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 0.01,
        "tags": ["pose", "balanced", "keypoints"]
    },

    # OBB (Oriented Bounding Box)
    "yolov8n-obb": {
        "display_name": "YOLOv8n-OBB",
        "description": "YOLOv8 Nano OBB - Rotated object detection",
        "params": "3.1M",
        "input_size": 640,
        "task_type": "obb_detection",
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 0.01,
        "tags": ["obb", "rotated", "aerial"]
    },
}
```

**Step 2**: UltralyticsAdapter 확장 - OBB 지원 추가

OBB(Oriented Bounding Box)는 새로운 task이므로 Adapter 확장 필요:

```python
# mvp/training/adapters/base.py

class TaskType(Enum):
    # ... existing tasks ...
    OBB_DETECTION = "obb_detection"  # 추가
```

```python
# mvp/training/adapters/ultralytics_adapter.py

TASK_SUFFIX_MAP = {
    TaskType.OBJECT_DETECTION: "",
    TaskType.INSTANCE_SEGMENTATION: "-seg",
    TaskType.POSE_ESTIMATION: "-pose",
    TaskType.IMAGE_CLASSIFICATION: "-cls",
    TaskType.OBB_DETECTION: "-obb",  # 추가
}
```

#### 검증 기준

- [ ] YOLOv9 모델이 YOLOv8과 동일한 방식으로 동작하는가?
- [ ] OBB task가 정상적으로 지원되는가?
- [ ] 대형 모델(yolov8x, yolov9e)이 메모리 초과 없이 동작하는가?

#### 예상 문제점

1. **YOLOv9 가중치 포맷 차이**
   - **해결책**: Ultralytics 패키지 최신 버전 사용 (0.0.206+)

2. **OBB task 메트릭 계산**
   - **해결책**: ValidationMetricsCalculator에 OBB 메트릭 추가 (mAP50, rotational error)

3. **OBB 데이터셋 포맷**
   - **해결책**: DOTA, HRSC2016 등 표준 OBB 데이터셋 지원

---

### Phase 1 산출물

- [ ] `mvp/training/model_registry/timm_models.py`
- [ ] `mvp/training/model_registry/ultralytics_models.py`
- [ ] `mvp/backend/app/api/models.py`
- [ ] `mvp/frontend/components/training/ModelSelector.tsx`
- [ ] Phase 1 검증 리포트 (성공/실패 모델 목록, 발견된 문제점)

---

## Phase 2: HuggingFace Transformers 프레임워크 추가

**목표**: 완전히 새로운 프레임워크 추가를 통해 플랫폼의 확장성 검증

**기간**: 2주

### 2.1 HuggingFace 특징 분석

#### timm/Ultralytics와의 차이점

| 항목 | timm/Ultralytics | HuggingFace Transformers |
|------|------------------|--------------------------|
| **모델 로드** | `timm.create_model()`, `YOLO()` | `AutoModel.from_pretrained()` |
| **학습 방식** | 직접 loop 구현 | `Trainer` API 사용 |
| **데이터 로더** | PyTorch DataLoader | HF `Dataset` + `DataCollator` |
| **Tokenization** | 불필요 | Vision 모델도 processor 필요 |
| **Checkpoint** | `state_dict` 저장 | `save_pretrained()` |
| **Metrics** | 직접 계산 | `evaluate()` 메서드 |

#### 지원할 Task와 모델

| Task | 모델 | 난이도 |
|------|------|--------|
| **Image Classification** | ViT (google/vit-base-patch16-224) | ⭐⭐ Medium |
| **Image Classification** | DeiT (facebook/deit-base-distilled-patch16-224) | ⭐⭐ Medium |
| **Object Detection** | DETR (facebook/detr-resnet-50) | ⭐⭐⭐ Hard |
| **Semantic Segmentation** | SegFormer (nvidia/segformer-b0-finetuned-ade-512-512) | ⭐⭐⭐ Hard |
| **Image-to-Text (Captioning)** | BLIP (Salesforce/blip-image-captioning-base) | ⭐⭐⭐⭐ Very Hard |

### 2.2 TransformersAdapter 구현

#### Step 1: Adapter 골격 작성

```python
# mvp/training/adapters/transformers_adapter.py

from .base import TrainingAdapter, MetricsResult, TaskType, InferenceResult
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import torch
from typing import Dict, Any, List

class TransformersAdapter(TrainingAdapter):
    """
    Adapter for HuggingFace Transformers.

    Supported tasks:
    - Image Classification (ViT, DeiT, Swin, BEiT)
    - Object Detection (DETR, YOLOS)
    - Semantic Segmentation (SegFormer, Mask2Former)
    - Image Captioning (BLIP, GIT)
    """

    def __init__(self, model_config, dataset_config, training_config, output_dir, job_id):
        super().__init__(model_config, dataset_config, training_config, output_dir, job_id)
        self.processor = None
        self.trainer = None

    @classmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Return configuration schema for Transformers models."""
        from training.config_schemas import get_transformers_schema
        return get_transformers_schema()

    def prepare_model(self) -> None:
        """Initialize model and processor."""
        model_name = self.model_config.model_name
        task_type = self.model_config.task_type

        # Load processor (handles image preprocessing)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Load model based on task
        if task_type == TaskType.IMAGE_CLASSIFICATION:
            from transformers import AutoModelForImageClassification
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=self.model_config.num_classes,
                ignore_mismatched_sizes=True
            )
        elif task_type == TaskType.OBJECT_DETECTION:
            from transformers import AutoModelForObjectDetection
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            from transformers import AutoModelForSemanticSegmentation
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=self.model_config.num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError(f"Unsupported task: {task_type}")

        # Move to device
        device = torch.device(self.training_config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        print(f"[TransformersAdapter] Loaded {model_name} for {task_type.value}")

    def prepare_dataset(self) -> None:
        """Load and preprocess dataset using HuggingFace datasets."""
        from datasets import load_dataset
        from torch.utils.data import DataLoader

        dataset_path = self.dataset_config.dataset_path
        dataset_format = self.dataset_config.format

        # Load dataset based on format
        if dataset_format == DatasetFormat.IMAGE_FOLDER:
            # ImageFolder → HF Dataset
            dataset = load_dataset("imagefolder", data_dir=dataset_path)

            # Split into train/val
            train_dataset = dataset["train"]
            val_dataset = dataset.get("validation") or dataset.get("val")

            if not val_dataset:
                # Split train into train/val
                split = train_dataset.train_test_split(test_size=0.2, seed=42)
                train_dataset = split["train"]
                val_dataset = split["test"]

            # Apply preprocessing
            def preprocess_function(examples):
                images = [img.convert("RGB") for img in examples["image"]]
                inputs = self.processor(images, return_tensors="pt")
                inputs["labels"] = examples["label"]
                return inputs

            train_dataset = train_dataset.map(preprocess_function, batched=True)
            val_dataset = val_dataset.map(preprocess_function, batched=True)

            # Set format for PyTorch
            train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
            val_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        print(f"[TransformersAdapter] Loaded dataset:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")

    def train_epoch(self, epoch: int) -> MetricsResult:
        """Train one epoch using HuggingFace Trainer."""
        # HuggingFace Trainer handles training loop internally
        # We override train() instead of train_epoch()
        raise NotImplementedError("Use train() method instead")

    def validate(self, epoch: int) -> MetricsResult:
        """Validate using HuggingFace Trainer."""
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call train() first.")

        # Run evaluation
        eval_results = self.trainer.evaluate()

        # Convert to MetricsResult
        metrics = MetricsResult(
            epoch=epoch,
            step=self.trainer.state.global_step,
            train_loss=0.0,  # Not available here
            val_loss=eval_results.get("eval_loss", 0.0),
            metrics={
                "accuracy": eval_results.get("eval_accuracy", 0.0),
                "f1": eval_results.get("eval_f1", 0.0),
            }
        )

        return metrics

    def train(self, start_epoch: int = 0, checkpoint_path: str = None, resume_training: bool = False):
        """Override train() to use HuggingFace Trainer API."""
        # Prepare model and dataset
        self.prepare_model()
        self.prepare_dataset()

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.training_config.epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size * 2,
            learning_rate=self.training_config.learning_rate,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="mlflow",  # MLflow integration
        )

        # Define compute_metrics function
        def compute_metrics(eval_pred: EvalPrediction):
            from sklearn.metrics import accuracy_score, f1_score

            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=-1)

            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted")
            }

        # Create Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
        )

        # Initialize callbacks
        callbacks = TrainingCallbacks(
            job_id=self.job_id,
            model_config=self.model_config,
            training_config=self.training_config,
            db_session=None
        )
        callbacks.on_train_begin()

        try:
            # Start training
            self.trainer.train(resume_from_checkpoint=checkpoint_path if resume_training else None)

            # Get final metrics
            final_metrics = self.trainer.evaluate()
            callbacks.on_train_end(final_metrics)

            # Save final model
            self.model.save_pretrained(f"{self.output_dir}/final_model")
            self.processor.save_pretrained(f"{self.output_dir}/final_model")

        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            callbacks.on_train_end()
            raise

        return []  # Metrics are logged by Trainer

    def save_checkpoint(self, epoch: int, metrics: MetricsResult) -> str:
        """Save checkpoint using HuggingFace format."""
        checkpoint_dir = f"{self.output_dir}/checkpoints/epoch_{epoch}"
        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path: str, inference_mode: bool = True, device: str = None) -> None:
        """Load checkpoint for inference."""
        from transformers import AutoModel

        self.processor = AutoImageProcessor.from_pretrained(checkpoint_path)

        if self.model_config.task_type == TaskType.IMAGE_CLASSIFICATION:
            self.model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
        elif self.model_config.task_type == TaskType.OBJECT_DETECTION:
            self.model = AutoModelForObjectDetection.from_pretrained(checkpoint_path)

        if inference_mode:
            self.model.eval()

        if device:
            self.model.to(device)

        print(f"[TransformersAdapter] Loaded checkpoint from {checkpoint_path}")

    def preprocess_image(self, image_path: str):
        """Preprocess single image."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        return inputs

    def infer_single(self, image_path: str) -> InferenceResult:
        """Run inference on single image."""
        import time
        from pathlib import Path

        start_time = time.time()

        # Preprocess
        inputs = self.preprocess_image(image_path)
        preprocess_time = (time.time() - start_time) * 1000

        # Inference
        infer_start = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        inference_time = (time.time() - infer_start) * 1000

        # Postprocess based on task
        post_start = time.time()

        if self.model_config.task_type == TaskType.IMAGE_CLASSIFICATION:
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Top-1 prediction
            confidence, pred_id = torch.max(probs, dim=-1)
            confidence = confidence.item()
            pred_id = pred_id.item()

            # Top-5 predictions
            top5_probs, top5_ids = torch.topk(probs, min(5, self.model_config.num_classes), dim=-1)
            top5_predictions = [
                {
                    'label_id': int(top5_ids[0, i].item()),
                    'confidence': float(top5_probs[0, i].item())
                }
                for i in range(top5_probs.size(1))
            ]

            postprocess_time = (time.time() - post_start) * 1000

            return InferenceResult(
                image_path=image_path,
                image_name=Path(image_path).name,
                task_type=TaskType.IMAGE_CLASSIFICATION,
                predicted_label_id=pred_id,
                confidence=confidence,
                top5_predictions=top5_predictions,
                inference_time_ms=inference_time,
                preprocessing_time_ms=preprocess_time,
                postprocessing_time_ms=postprocess_time
            )

        else:
            raise NotImplementedError(f"Inference for {self.model_config.task_type} not implemented yet")
```

#### Step 2: 모델 레지스트리 작성

```python
# mvp/training/model_registry/transformers_models.py

TRANSFORMERS_MODEL_REGISTRY = {
    # Vision Transformers (Classification)
    "google/vit-base-patch16-224": {
        "display_name": "ViT-Base/16",
        "description": "Vision Transformer by Google (16x16 patches, 224x224 input)",
        "params": "86M",
        "input_size": 224,
        "task_type": "image_classification",
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 2e-5,
        "tags": ["transformer", "classification", "attention"]
    },
    "facebook/deit-base-distilled-patch16-224": {
        "display_name": "DeiT-Base",
        "description": "Data-efficient Image Transformer by Facebook",
        "params": "87M",
        "input_size": 224,
        "task_type": "image_classification",
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 5e-5,
        "tags": ["transformer", "classification", "distilled"]
    },

    # Object Detection
    "facebook/detr-resnet-50": {
        "display_name": "DETR-ResNet-50",
        "description": "DEtection TRansformer with ResNet-50 backbone",
        "params": "41M",
        "input_size": 800,
        "task_type": "object_detection",
        "pretrained_available": True,
        "recommended_batch_size": 4,
        "recommended_lr": 1e-4,
        "tags": ["transformer", "detection", "end-to-end"]
    },

    # Semantic Segmentation
    "nvidia/segformer-b0-finetuned-ade-512-512": {
        "display_name": "SegFormer-B0",
        "description": "Simple and Efficient Segmentation Transformer",
        "params": "3.8M",
        "input_size": 512,
        "task_type": "semantic_segmentation",
        "pretrained_available": True,
        "recommended_batch_size": 8,
        "recommended_lr": 6e-5,
        "tags": ["transformer", "segmentation", "efficient"]
    },

    # Image Captioning
    "Salesforce/blip-image-captioning-base": {
        "display_name": "BLIP-Base",
        "description": "Bootstrapped Language-Image Pretraining for Image Captioning",
        "params": "223M",
        "input_size": 384,
        "task_type": "image_captioning",
        "pretrained_available": True,
        "recommended_batch_size": 8,
        "recommended_lr": 5e-5,
        "tags": ["multimodal", "captioning", "vision-language"]
    },
}
```

#### Step 3: ConfigSchema 작성

```python
# mvp/training/config_schemas/transformers_schema.py

def get_transformers_schema() -> ConfigSchema:
    """Configuration schema for HuggingFace Transformers models."""
    fields = [
        # ========== Model Settings ==========
        ConfigField(
            name="model_name_or_path",
            type="str",
            default="google/vit-base-patch16-224",
            description="HuggingFace model identifier or local path",
            group="model",
            required=True
        ),

        # ========== Training Arguments ==========
        ConfigField(
            name="warmup_steps",
            type="int",
            default=500,
            min=0,
            max=5000,
            description="Number of warmup steps for learning rate scheduler",
            group="training"
        ),
        ConfigField(
            name="weight_decay",
            type="float",
            default=0.01,
            min=0.0,
            max=0.1,
            description="Weight decay for AdamW optimizer",
            group="optimizer"
        ),
        ConfigField(
            name="adam_beta1",
            type="float",
            default=0.9,
            min=0.0,
            max=1.0,
            description="Beta1 for AdamW",
            group="optimizer",
            advanced=True
        ),
        ConfigField(
            name="adam_beta2",
            type="float",
            default=0.999,
            min=0.0,
            max=1.0,
            description="Beta2 for AdamW",
            group="optimizer",
            advanced=True
        ),
        ConfigField(
            name="gradient_accumulation_steps",
            type="int",
            default=1,
            min=1,
            max=32,
            description="Number of updates steps to accumulate before backward pass",
            group="training"
        ),
        ConfigField(
            name="max_grad_norm",
            type="float",
            default=1.0,
            min=0.0,
            max=10.0,
            description="Maximum gradient norm for clipping",
            group="training",
            advanced=True
        ),

        # ========== Evaluation ==========
        ConfigField(
            name="eval_strategy",
            type="select",
            default="epoch",
            options=["no", "steps", "epoch"],
            description="Evaluation strategy",
            group="evaluation"
        ),
        ConfigField(
            name="save_strategy",
            type="select",
            default="epoch",
            options=["no", "steps", "epoch"],
            description="Checkpoint save strategy",
            group="evaluation"
        ),

        # ========== Data Augmentation ==========
        ConfigField(
            name="random_flip",
            type="bool",
            default=True,
            description="Apply random horizontal flip",
            group="augmentation"
        ),
    ]

    return ConfigSchema(fields=fields)
```

### 2.3 검증 기준

각 HuggingFace 모델에 대해:

- [ ] 모델이 from_pretrained()로 정상 로드되는가?
- [ ] Processor/Tokenizer가 올바르게 동작하는가?
- [ ] Trainer API를 사용한 training이 성공하는가?
- [ ] MLflow 연동이 정상 동작하는가?
- [ ] Checkpoint 저장/로드가 save_pretrained() 방식으로 동작하는가?
- [ ] Inference 결과가 InferenceResult 포맷으로 변환되는가?
- [ ] 기존 timm/Ultralytics와 동일한 UI/API로 사용 가능한가?

### 2.4 예상 문제점과 해결책

#### 문제 1: HF Trainer와 기존 TrainingCallbacks 충돌

**문제**: HuggingFace Trainer는 자체 callback 시스템을 사용함

**해결책**: HF TrainerCallback을 구현하여 우리 TrainingCallbacks와 연동

```python
from transformers import TrainerCallback

class PlatformCallback(TrainerCallback):
    """Bridge between HF Trainer and our TrainingCallbacks."""

    def __init__(self, platform_callbacks: TrainingCallbacks):
        self.callbacks = platform_callbacks

    def on_train_begin(self, args, state, control, **kwargs):
        self.callbacks.on_train_begin()

    def on_epoch_end(self, args, state, control, metrics, **kwargs):
        self.callbacks.on_epoch_end(
            epoch=state.epoch,
            metrics=metrics,
            checkpoint_path=state.best_model_checkpoint
        )

    def on_train_end(self, args, state, control, **kwargs):
        self.callbacks.on_train_end()

# Usage in TransformersAdapter
trainer = Trainer(
    model=self.model,
    args=training_args,
    train_dataset=self.train_dataset,
    eval_dataset=self.val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[PlatformCallback(callbacks)]  # 추가
)
```

#### 문제 2: 데이터셋 포맷 변환

**문제**: HuggingFace는 자체 Dataset 객체를 사용

**해결책**: ImageFolder → HF Dataset 변환 유틸리티 작성

```python
# mvp/training/utils/dataset_converters.py

from datasets import Dataset, Image as HFImage
from pathlib import Path

def imagefolder_to_hf_dataset(root_dir: str) -> Dataset:
    """Convert ImageFolder to HuggingFace Dataset."""
    from PIL import Image

    images = []
    labels = []
    label_names = []

    root = Path(root_dir)
    for label_id, class_dir in enumerate(sorted(root.iterdir())):
        if not class_dir.is_dir():
            continue

        label_names.append(class_dir.name)

        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                images.append(str(img_path))
                labels.append(label_id)

    dataset = Dataset.from_dict({
        "image": images,
        "label": labels
    })

    # Cast image column to HF Image type
    dataset = dataset.cast_column("image", HFImage())

    return dataset, label_names
```

#### 문제 3: 메모리 사용량 증가

**문제**: HF 모델은 일반적으로 더 많은 메모리를 사용

**해결책**: Gradient checkpointing, Mixed precision training 활성화

```python
training_args = TrainingArguments(
    # ... other args ...
    gradient_checkpointing=True,  # 메모리 절약
    fp16=True,  # Mixed precision (CUDA만)
    gradient_accumulation_steps=4,  # Effective batch size 증가
)
```

### Phase 2 산출물

- [ ] `mvp/training/adapters/transformers_adapter.py`
- [ ] `mvp/training/model_registry/transformers_models.py`
- [ ] `mvp/training/config_schemas/transformers_schema.py`
- [ ] `mvp/training/utils/dataset_converters.py`
- [ ] Phase 2 검증 리포트

---

## Phase 3: 커스텀 오픈소스 GitHub 모델 추가

**목표**: 프레임워크가 아닌 개별 GitHub repository의 모델을 지원하여 최대 유연성 검증

**기간**: 2주

### 3.1 대상 모델 선정

커스텀 구현의 다양성을 검증하기 위해 서로 다른 스타일의 모델을 선택:

#### 3.1.1 ConvNeXt (Meta Research)

- **Repository**: https://github.com/facebookresearch/ConvNeXt
- **Task**: Image Classification
- **특징**:
  - Modern CNN (Transformer 스타일 디자인)
  - PyTorch 기반
  - 표준 ImageFolder 데이터셋 지원
- **난이도**: ⭐⭐ Medium
- **선정 이유**: timm과 유사하지만 별도 repository, 커스텀 augmentation pipeline

#### 3.1.2 PP-YOLO (PaddlePaddle)

- **Repository**: https://github.com/PaddlePaddle/PaddleDetection
- **Task**: Object Detection
- **특징**:
  - PaddlePaddle 프레임워크 기반 (PyTorch 아님!)
  - COCO 데이터셋 지원
  - YAML 설정 파일 사용
- **난이도**: ⭐⭐⭐⭐ Very Hard
- **선정 이유**: 완전히 다른 프레임워크, 플랫폼 독립성 극한 테스트

#### 3.1.3 YOLOv7 (WongKinYiu)

- **Repository**: https://github.com/WongKinYiu/yolov7
- **Task**: Object Detection
- **특징**:
  - PyTorch 기반
  - Ultralytics와 다른 구현 스타일
  - 커스텀 데이터 로더, 커스텀 augmentation
- **난이도**: ⭐⭐⭐ Hard
- **선정 이유**: 같은 YOLO 계열이지만 구현이 다름, Adapter 재사용성 테스트

#### 3.1.4 ViTPose (ViTAE-Transformer)

- **Repository**: https://github.com/ViTAE-Transformer/ViTPose
- **Task**: Pose Estimation
- **특징**:
  - MMPose 기반 (OpenMMLab)
  - COCO-Pose 데이터셋
  - Config 파일 기반 학습
- **난이도**: ⭐⭐⭐⭐ Very Hard
- **선정 이유**: MMPose 생태계, 복잡한 설정 관리

### 3.2 CustomAdapter 아키텍처 설계

커스텀 모델은 각각 구현이 다르므로, **Adapter of Adapters** 패턴 사용:

```python
# mvp/training/adapters/custom_adapter.py

from .base import TrainingAdapter
import subprocess
import os
from pathlib import Path

class CustomAdapter(TrainingAdapter):
    """
    Adapter for custom GitHub models.

    Delegates training to external repository's training script via subprocess.
    Acts as a bridge between our platform and external codebases.
    """

    def __init__(self, model_config, dataset_config, training_config, output_dir, job_id):
        super().__init__(model_config, dataset_config, training_config, output_dir, job_id)
        self.repo_dir = None
        self.train_script = None
        self.config_file = None

    @classmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Return configuration schema for custom models."""
        # Dynamic schema based on model's requirements
        return ConfigSchema(fields=[])

    def setup_repository(self) -> None:
        """
        Clone and setup the GitHub repository.

        This is called once before training.
        """
        repo_url = self.model_config.custom_config.get("repo_url")
        repo_dir = Path(self.output_dir) / "repo"

        if not repo_dir.exists():
            print(f"[CustomAdapter] Cloning repository: {repo_url}")
            subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

            # Install requirements if needed
            requirements_file = repo_dir / "requirements.txt"
            if requirements_file.exists():
                print("[CustomAdapter] Installing repository requirements...")
                subprocess.run(
                    ["pip", "install", "-r", str(requirements_file)],
                    check=True
                )

        self.repo_dir = repo_dir

        # Find training script
        self.train_script = self._locate_training_script()

        print(f"[CustomAdapter] Repository setup complete")
        print(f"  Repository: {self.repo_dir}")
        print(f"  Training script: {self.train_script}")

    def _locate_training_script(self) -> Path:
        """Locate the main training script in the repository."""
        # Common patterns for training scripts
        patterns = [
            "train.py",
            "main.py",
            "tools/train.py",
            "scripts/train.py",
        ]

        for pattern in patterns:
            script_path = self.repo_dir / pattern
            if script_path.exists():
                return script_path

        raise FileNotFoundError(f"Training script not found in {self.repo_dir}")

    def prepare_config_file(self) -> str:
        """
        Generate configuration file for the external training script.

        Returns:
            Path to generated config file
        """
        config_format = self.model_config.custom_config.get("config_format", "yaml")

        if config_format == "yaml":
            import yaml
            config_path = self.output_dir / "custom_config.yaml"

            config_dict = self._build_config_dict()

            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            return str(config_path)

        elif config_format == "json":
            import json
            config_path = self.output_dir / "custom_config.json"

            config_dict = self._build_config_dict()

            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            return str(config_path)

        else:
            raise ValueError(f"Unsupported config format: {config_format}")

    def _build_config_dict(self) -> dict:
        """Build configuration dictionary for external script."""
        # Map our TrainingConfig to external format
        # This needs to be customized per model
        return {
            "model": {
                "name": self.model_config.model_name,
                "num_classes": self.model_config.num_classes,
            },
            "dataset": {
                "path": self.dataset_config.dataset_path,
                "format": self.dataset_config.format.value,
            },
            "training": {
                "epochs": self.training_config.epochs,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
            },
            "output": {
                "dir": self.output_dir,
            }
        }

    def prepare_model(self) -> None:
        """Setup repository (model preparation done by external script)."""
        self.setup_repository()

    def prepare_dataset(self) -> None:
        """Dataset preparation done by external script."""
        pass

    def train(self, start_epoch: int = 0, checkpoint_path: str = None, resume_training: bool = False):
        """
        Execute training via external script.

        Runs the repository's training script as a subprocess and monitors output.
        """
        self.prepare_model()

        # Generate config file
        config_path = self.prepare_config_file()

        # Build command
        cmd = self._build_training_command(config_path)

        print(f"[CustomAdapter] Starting external training:")
        print(f"  Command: {' '.join(cmd)}")

        # Execute training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=str(self.repo_dir)
        )

        # Monitor output
        for line in process.stdout:
            print(line, end="")

            # Parse metrics from output (if possible)
            metrics = self._parse_output_line(line)
            if metrics:
                # Log to database
                # ...
                pass

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {process.returncode}")

        print("[CustomAdapter] Training completed")

        return []

    def _build_training_command(self, config_path: str) -> list:
        """Build command to run external training script."""
        # This needs to be customized per model
        cmd = [
            "python",
            str(self.train_script),
            "--config", config_path,
        ]

        # Add GPU selection if needed
        if self.training_config.device == "cuda":
            cmd = ["CUDA_VISIBLE_DEVICES=0"] + cmd

        return cmd

    def _parse_output_line(self, line: str) -> dict:
        """
        Parse metrics from training script output.

        Each model has different output format, so this needs customization.
        """
        import re

        # Example: "Epoch 10/50 - Loss: 0.234, Acc: 89.2%"
        pattern = r"Epoch (\d+)/(\d+).*Loss:\s*([\d.]+).*Acc:\s*([\d.]+)"
        match = re.search(pattern, line)

        if match:
            epoch = int(match.group(1))
            loss = float(match.group(3))
            acc = float(match.group(4))

            return {
                "epoch": epoch,
                "loss": loss,
                "accuracy": acc
            }

        return None

    def train_epoch(self, epoch: int) -> MetricsResult:
        """Not used for custom adapter (external script handles training loop)."""
        raise NotImplementedError()

    def validate(self, epoch: int) -> MetricsResult:
        """Not used for custom adapter (external script handles validation)."""
        raise NotImplementedError()

    def save_checkpoint(self, epoch: int, metrics: MetricsResult) -> str:
        """Checkpoint saved by external script."""
        # Find checkpoint saved by external script
        checkpoint_dir = Path(self.output_dir) / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob(f"*epoch{epoch}*"))
            if checkpoints:
                return str(checkpoints[0])

        return ""

    def load_checkpoint(self, checkpoint_path: str, inference_mode: bool = True, device: str = None) -> None:
        """Load checkpoint (implementation depends on model)."""
        # This needs to be implemented per custom model
        raise NotImplementedError("Custom model checkpoint loading not implemented")

    def preprocess_image(self, image_path: str):
        """Preprocess image (implementation depends on model)."""
        raise NotImplementedError()

    def infer_single(self, image_path: str) -> InferenceResult:
        """Run inference (implementation depends on model)."""
        raise NotImplementedError("Custom model inference not implemented yet")
```

### 3.3 모델별 Wrapper 구현

각 커스텀 모델마다 CustomAdapter를 상속한 구체적인 Adapter 작성:

#### 3.3.1 ConvNeXtAdapter

```python
# mvp/training/adapters/custom/convnext_adapter.py

from ..custom_adapter import CustomAdapter
from ..base import ConfigSchema, ConfigField

class ConvNeXtAdapter(CustomAdapter):
    """Adapter for ConvNeXt (Meta Research)."""

    @classmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Configuration schema for ConvNeXt."""
        fields = [
            ConfigField(
                name="model_variant",
                type="select",
                default="convnext_tiny",
                options=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"],
                description="ConvNeXt model variant",
                group="model"
            ),
            ConfigField(
                name="drop_path_rate",
                type="float",
                default=0.1,
                min=0.0,
                max=0.5,
                description="Drop path rate for stochastic depth",
                group="regularization"
            ),
            # ... more ConvNeXt-specific options
        ]

        return ConfigSchema(fields=fields)

    def _build_config_dict(self) -> dict:
        """Build ConvNeXt configuration."""
        return {
            "MODEL": {
                "TYPE": "convnext",
                "NAME": self.model_config.model_name,
                "NUM_CLASSES": self.model_config.num_classes,
                "DROP_PATH_RATE": self.training_config.advanced_config.get("drop_path_rate", 0.1),
            },
            "DATA": {
                "DATA_PATH": self.dataset_config.dataset_path,
                "BATCH_SIZE": self.training_config.batch_size,
            },
            "TRAIN": {
                "EPOCHS": self.training_config.epochs,
                "BASE_LR": self.training_config.learning_rate,
                "WARMUP_EPOCHS": 5,
                "WEIGHT_DECAY": 0.05,
            },
            "OUTPUT": self.output_dir,
        }

    def _build_training_command(self, config_path: str) -> list:
        """Build ConvNeXt training command."""
        return [
            "python",
            str(self.repo_dir / "main.py"),
            "--cfg", config_path,
            "--data-path", self.dataset_config.dataset_path,
            "--batch-size", str(self.training_config.batch_size),
            "--output", self.output_dir,
        ]
```

#### 3.3.2 YOLOv7Adapter

```python
# mvp/training/adapters/custom/yolov7_adapter.py

class YOLOv7Adapter(CustomAdapter):
    """Adapter for YOLOv7 (WongKinYiu)."""

    def _build_config_dict(self) -> dict:
        """YOLOv7 uses custom data.yaml format."""
        # Generate data.yaml
        data_yaml = {
            "path": self.dataset_config.dataset_path,
            "train": "images/train",
            "val": "images/val",
            "nc": self.model_config.num_classes,
            "names": self._get_class_names(),
        }

        return data_yaml

    def _build_training_command(self, config_path: str) -> list:
        """YOLOv7 training command."""
        return [
            "python",
            str(self.repo_dir / "train.py"),
            "--workers", "8",
            "--device", "0",
            "--batch-size", str(self.training_config.batch_size),
            "--epochs", str(self.training_config.epochs),
            "--data", config_path,
            "--img", "640",
            "--cfg", self._get_model_cfg(),
            "--weights", self._get_pretrained_weights(),
            "--name", f"job_{self.job_id}",
            "--hyp", "data/hyp.scratch.custom.yaml",
        ]

    def _get_model_cfg(self) -> str:
        """Get model configuration file path."""
        model_name = self.model_config.model_name
        return str(self.repo_dir / f"cfg/training/{model_name}.yaml")
```

### 3.4 검증 기준

각 커스텀 모델에 대해:

- [ ] Repository가 자동으로 clone되고 setup되는가?
- [ ] 우리 TrainingConfig가 external config로 올바르게 변환되는가?
- [ ] 외부 스크립트가 성공적으로 실행되는가?
- [ ] 학습 로그가 파싱되어 데이터베이스에 저장되는가?
- [ ] Checkpoint가 우리 플랫폼 포맷으로 변환되는가?
- [ ] 외부 모델도 동일한 UI/API로 사용 가능한가?

### 3.5 예상 문제점과 해결책

#### 문제 1: 의존성 충돌

**문제**: 각 repository마다 다른 패키지 버전 요구

**해결책**: 가상 환경 분리

```python
# mvp/training/utils/venv_manager.py

import subprocess
from pathlib import Path

def create_isolated_env(repo_dir: Path) -> Path:
    """Create isolated virtual environment for custom model."""
    venv_dir = repo_dir / ".venv"

    if not venv_dir.exists():
        subprocess.run(["python", "-m", "venv", str(venv_dir)], check=True)

        # Install requirements in isolated env
        pip = venv_dir / "bin" / "pip"
        requirements = repo_dir / "requirements.txt"

        if requirements.exists():
            subprocess.run([str(pip), "install", "-r", str(requirements)], check=True)

    return venv_dir

# Usage in CustomAdapter
def _build_training_command(self, config_path: str) -> list:
    venv = create_isolated_env(self.repo_dir)
    python = venv / "bin" / "python"

    return [
        str(python),
        str(self.train_script),
        # ... args
    ]
```

#### 문제 2: Config 포맷 다양성

**문제**: YAML, JSON, Python dict, Command-line args 등 다양한 설정 방식

**해결책**: Config Translator 패턴

```python
# mvp/training/config_translators/base.py

class ConfigTranslator(ABC):
    """Translates our TrainingConfig to external format."""

    @abstractmethod
    def translate(self, model_config, dataset_config, training_config) -> str:
        """Returns path to generated config file."""
        pass

# mvp/training/config_translators/yolo_translator.py

class YOLOConfigTranslator(ConfigTranslator):
    def translate(self, model_config, dataset_config, training_config) -> str:
        data_yaml = {
            "path": dataset_config.dataset_path,
            "train": "train",
            "val": "val",
            "nc": model_config.num_classes,
            "names": self._extract_class_names(dataset_config.dataset_path),
        }

        output_path = Path(training_config.output_dir) / "data.yaml"
        with open(output_path, "w") as f:
            yaml.dump(data_yaml, f)

        return str(output_path)
```

#### 문제 3: 출력 파싱의 어려움

**문제**: 각 모델마다 다른 로그 포맷

**해결책**: Regex 패턴 라이브러리 + LLM 기반 파싱

```python
# mvp/training/utils/log_parsers.py

import re
from typing import Optional, Dict

class LogParser:
    """Parse training logs from external scripts."""

    # Pre-defined patterns for common frameworks
    PATTERNS = {
        "yolo": r"Epoch:\s*(\d+)/(\d+).*mAP@0.5:\s*([\d.]+)",
        "mmdet": r"\[Epoch\s+(\d+)\].*loss:\s*([\d.]+)",
        "pytorch": r"Epoch\s*(\d+).*Loss:\s*([\d.]+).*Acc:\s*([\d.]+)",
    }

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.pattern = self.PATTERNS.get(model_type)

    def parse(self, line: str) -> Optional[Dict]:
        """Parse a single log line."""
        if not self.pattern:
            return self._llm_parse(line)

        match = re.search(self.pattern, line)
        if match:
            return self._extract_metrics(match)

        return None

    def _llm_parse(self, line: str) -> Optional[Dict]:
        """Use LLM to parse unknown log format."""
        # Fallback: use Claude to extract metrics
        # This is expensive but handles any format
        pass
```

### Phase 3 산출물

- [ ] `mvp/training/adapters/custom_adapter.py`
- [ ] `mvp/training/adapters/custom/convnext_adapter.py`
- [ ] `mvp/training/adapters/custom/yolov7_adapter.py`
- [ ] `mvp/training/adapters/custom/ppyolo_adapter.py`
- [ ] `mvp/training/adapters/custom/vitpose_adapter.py`
- [ ] `mvp/training/utils/venv_manager.py`
- [ ] `mvp/training/utils/log_parsers.py`
- [ ] `mvp/training/config_translators/`
- [ ] Phase 3 검증 리포트

---

## 최종 검증 리포트 템플릿

각 Phase 완료 후 다음 형식의 리포트 작성:

```markdown
# Phase [N] 검증 리포트

## 요약

- **기간**: YYYY-MM-DD ~ YYYY-MM-DD
- **추가된 모델 수**: X개
- **성공률**: Y% (성공 Z개 / 전체 X개)

## 모델별 검증 결과

### 1. [Model Name]

- **프레임워크**: timm / ultralytics / transformers / custom
- **Task**: image_classification / object_detection / ...
- **상태**: ✅ 성공 / ⚠️ 부분 성공 / ❌ 실패

#### 검증 항목

- [ ] 모델 로드
- [ ] Training 실행
- [ ] Validation
- [ ] Checkpoint 저장/로드
- [ ] Inference

#### 문제점

[발견된 문제점 기술]

#### 해결책

[적용한 해결책 기술]

#### 소요 시간

- Setup: X분
- Configuration: Y분
- First Training: Z분

---

## 아키텍처 개선 사항

### 1. [개선 항목]

**문제**: [발견된 구조적 문제]
**해결**: [적용한 개선 사항]
**영향**: [다른 부분에 미친 영향]

---

## 교훈 (Lessons Learned)

1. **[교훈 1]**: [내용]
2. **[교훈 2]**: [내용]

---

## 다음 Phase 권고사항

[다음 Phase에서 주의할 점, 개선할 점]
```

---

## 전체 타임라인

| Phase | 기간 | 주요 작업 | 산출물 |
|-------|------|----------|--------|
| **Phase 0** | 2일 | 계획 수립, 문서 작성 | 이 문서 |
| **Phase 1** | 1주 | timm + Ultralytics 모델 추가 | 모델 레지스트리, API 개선 |
| **Phase 2** | 2주 | HuggingFace 프레임워크 추가 | TransformersAdapter |
| **Phase 3** | 2주 | 커스텀 GitHub 모델 추가 | CustomAdapter, Wrapper들 |
| **Phase 4** | 1주 | 최종 검증 및 문서화 | 종합 리포트, 개선 제안서 |
| **총 기간** | **6주** | | |

---

## 성공 지표

### 정량적 지표

- **모델 추가 속도**: 신규 모델 1개 추가에 소요되는 평균 시간
  - 목표: timm/Ultralytics < 30분, HuggingFace < 2시간, Custom < 1일

- **코드 재사용률**: 기존 Adapter 코드 대비 신규 코드 비율
  - 목표: timm/Ultralytics > 90%, HuggingFace > 70%, Custom > 50%

- **성공률**: 추가 시도한 모델 중 정상 동작하는 비율
  - 목표: > 85%

### 정성적 지표

- **개발자 경험**: 새 모델 추가가 직관적인가?
- **문서화 품질**: 가이드만 보고 추가 가능한가?
- **에러 핸들링**: 실패 시 명확한 에러 메시지가 제공되는가?

---

## 리스크 및 대응 방안

| 리스크 | 발생 확률 | 영향도 | 대응 방안 |
|--------|----------|--------|----------|
| 의존성 충돌 | 높음 | 중간 | 가상 환경 분리 |
| Config 변환 실패 | 중간 | 높음 | Config Translator 패턴 + LLM fallback |
| 메모리 부족 | 중간 | 중간 | Batch size 자동 조정 |
| 외부 repo 변경 | 낮음 | 높음 | Repo 버전 고정 (git tag/commit hash) |
| 라이선스 문제 | 낮음 | 높음 | 사전에 라이선스 확인, 상업 사용 가능 모델만 |

---

## 참고 자료

### Phase 1 (timm + Ultralytics)

- timm documentation: https://huggingface.co/docs/timm
- Ultralytics documentation: https://docs.ultralytics.com/
- timm available models: https://github.com/huggingface/pytorch-image-models/blob/main/results/

### Phase 2 (HuggingFace)

- Transformers documentation: https://huggingface.co/docs/transformers
- Vision models guide: https://huggingface.co/docs/transformers/tasks/image_classification
- Trainer API: https://huggingface.co/docs/transformers/main_classes/trainer

### Phase 3 (Custom)

- ConvNeXt: https://github.com/facebookresearch/ConvNeXt
- YOLOv7: https://github.com/WongKinYiu/yolov7
- PaddleDetection: https://github.com/PaddlePaddle/PaddleDetection
- ViTPose: https://github.com/ViTAE-Transformer/ViTPose

---

*Document Version: 1.0*
*Created: 2025-10-30*
*Author: Vision AI Platform Team*
