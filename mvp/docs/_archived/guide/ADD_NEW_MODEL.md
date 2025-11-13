# 신규 모델 추가 가이드 (Add New Model Guide)

이 문서는 Vision AI Training Platform에 새로운 딥러닝 모델을 추가하는 전체 과정을 설명합니다.

## 목차

1. [개요](#개요)
2. [모델 추가 체크리스트](#모델-추가-체크리스트)
3. [Step 1: 모델 선정 및 검증](#step-1-모델-선정-및-검증)
4. [Step 2: 모델 레지스트리 추가](#step-2-모델-레지스트리-추가)
5. [Step 3: Adapter 구현 (필요시)](#step-3-adapter-구현-필요시)
6. [Step 4: Config Schema 정의 (필요시)](#step-4-config-schema-정의-필요시)
7. [Step 5: 호환성 테스트](#step-5-호환성-테스트)
8. [Step 6: UI 확인](#step-6-ui-확인)
9. [트러블슈팅](#트러블슈팅)

---

## 개요

Vision AI Training Platform은 **모델 레지스트리 기반 아키텍처**를 사용합니다:

```
Model Registry (메타데이터)
    ↓
Adapter (학습/추론 로직)
    ↓
Config Schema (설정 스키마)
    ↓
API & UI
```

### 지원 프레임워크

현재 지원하는 딥러닝 프레임워크:
- **timm** (PyTorch Image Models) - 이미지 분류
- **ultralytics** (YOLO) - 객체 탐지, 세그멘테이션, 포즈 추정

---

## 모델 추가 체크리스트

새로운 모델을 추가할 때 다음 항목들을 체크하세요:

- [ ] **Step 1**: 모델이 timm 또는 ultralytics 라이브러리에서 지원되는지 확인
- [ ] **Step 2**: 모델 레지스트리에 메타데이터 추가
- [ ] **Step 3**: 새로운 프레임워크라면 Adapter 구현 (기존 프레임워크는 생략)
- [ ] **Step 4**: 새로운 프레임워크라면 Config Schema 정의 (기존 프레임워크는 생략)
- [ ] **Step 5**: 호환성 테스트 실행
- [ ] **Step 6**: UI에서 모델 선택 및 학습 테스트
- [ ] **Step 7**: 코드 커밋 및 PR 생성

---

## Step 1: 모델 선정 및 검증

### 1.1 라이브러리 지원 확인

#### timm 모델 확인
```python
import timm

# 사용 가능한 모든 timm 모델 목록
available_models = timm.list_models()
print(f"Total timm models: {len(available_models)}")

# 특정 모델 검색
model_name = "vgg16"
if model_name in available_models:
    print(f"✓ {model_name} is available in timm")
else:
    print(f"✗ {model_name} is NOT available in timm")

# 모델 생성 테스트
try:
    model = timm.create_model(model_name, pretrained=False, num_classes=10)
    print(f"✓ Model created successfully")
except Exception as e:
    print(f"✗ Error creating model: {e}")
```

#### ultralytics 모델 확인
```python
from ultralytics import YOLO

# YOLO 모델 패턴
# Format: {version}{size}{variant}
# - version: yolov5, yolov8, yolo11, etc.
# - size: n (nano), s (small), m (medium), l (large), x (xlarge)
# - variant: -seg (segmentation), -pose (pose), -cls (classification)

model_name = "yolov8n"
known_patterns = ["yolov5", "yolov8", "yolo11", "yolo_world"]

if any(model_name.startswith(p) for p in known_patterns):
    print(f"✓ {model_name} matches known YOLO pattern")

# 모델 생성 테스트
try:
    model = YOLO(f"{model_name}.pt")  # Will download weights on first use
    print(f"✓ Model weights accessible")
except Exception as e:
    print(f"✗ Error loading model: {e}")
```

### 1.2 모델 정보 수집

추가하려는 모델의 다음 정보를 수집하세요:

1. **기본 정보**
   - Display name (사용자에게 보이는 이름)
   - Model name (라이브러리에서 사용하는 정확한 이름)
   - Description (간단한 설명)
   - Parameter count (예: "25.6M")
   - Input size (예: 224, 640)

2. **성능 벤치마크**
   - ImageNet Top-1/Top-5 accuracy (분류 모델)
   - COCO mAP (탐지/세그멘테이션 모델)
   - Inference speed (예: "120 img/s on V100")

3. **사용 가이드**
   - Use cases (적합한 사용 사례)
   - Pros (장점)
   - Cons (단점)
   - When to use (사용 시점)
   - When not to use (사용하지 말아야 할 시점)
   - Alternatives (대안 모델)

4. **추천 설정**
   - Recommended learning rate
   - Recommended batch size
   - Recommended epochs
   - Recommended optimizer
   - Recommended scheduler

---

## Step 2: 모델 레지스트리 추가

### 2.1 파일 위치

- **timm 모델**: `mvp/training/model_registry/timm_models.py`
- **ultralytics 모델**: `mvp/training/model_registry/ultralytics_models.py`

### 2.2 모델 레지스트리 구조

```python
TIMM_MODEL_REGISTRY = {
    "model_name": {
        # ===== 기본 정보 =====
        "display_name": str,           # UI에 표시될 이름
        "description": str,            # 한 줄 설명
        "params": str,                 # 파라미터 수 (예: "25.6M")
        "input_size": int,             # 입력 이미지 크기
        "pretrained_available": bool,  # Pretrained 가중치 제공 여부
        "recommended_batch_size": int, # 권장 배치 사이즈
        "recommended_lr": float,       # 권장 학습률

        # ===== 태그 및 분류 =====
        "tags": List[str],             # 검색용 태그
        "priority": int,               # 우선순위 (0=P0, 1=P1, 2=P2)
        "task_type": str,              # TaskType enum 값

        # ===== 벤치마크 성능 =====
        "benchmark": {
            "imagenet_top1": float,           # ImageNet Top-1 (%)
            "imagenet_top5": float,           # ImageNet Top-5 (%)
            "inference_speed_v100": float,    # V100 추론 속도
            "inference_speed_unit": str,      # 단위 (예: "img/s")
        },

        # ===== 사용 가이드 =====
        "use_cases": List[str],        # 사용 사례 목록
        "pros": List[str],             # 장점 목록
        "cons": List[str],             # 단점 목록
        "when_to_use": str,            # 사용 시점 (한 문장)
        "when_not_to_use": str,        # 사용 금지 시점 (한 문장)
        "alternatives": List[str],     # 대안 모델 목록

        # ===== 추천 설정 =====
        "recommended_settings": {
            "epochs": int,
            "learning_rate": float,
            "batch_size": int,
            "optimizer": str,
            "scheduler": str,
        },
    }
}
```

### 2.3 실제 추가 예시 (timm - VGG-16)

`mvp/training/model_registry/timm_models.py` 파일을 열고 적절한 우선순위 섹션에 추가:

```python
# ============================================================
# P1: Core Expansion (12 models)
# ============================================================

"vgg16": {
    "display_name": "VGG-16",
    "description": "Classic deep CNN - Simple architecture, excellent for transfer learning",
    "params": "138.4M",
    "input_size": 224,
    "pretrained_available": True,
    "recommended_batch_size": 32,
    "recommended_lr": 0.001,
    "tags": ["p1", "classic", "simple", "transfer-learning"],
    "priority": 1,
    "task_type": "image_classification",
    "benchmark": {
        "imagenet_top1": 71.6,
        "imagenet_top5": 90.6,
        "inference_speed_v100": 120,
        "inference_speed_unit": "img/s",
    },
    "use_cases": [
        "Transfer learning for custom image classification",
        "Educational purposes and research baselines",
        "Feature extraction for computer vision tasks",
        "Simple deployment scenarios",
    ],
    "pros": [
        "Very simple and interpretable architecture",
        "Excellent for transfer learning",
        "Well-documented and widely used",
        "Strong feature extraction capability",
    ],
    "cons": [
        "Large model size (138.4M params)",
        "Lower accuracy than modern architectures",
        "Slow inference compared to efficient models",
        "Not suitable for mobile/edge devices",
    ],
    "when_to_use": "Choose VGG-16 when you need a simple, proven architecture for transfer learning or when interpretability is more important than efficiency.",
    "when_not_to_use": "Avoid VGG-16 for production systems requiring fast inference or deployment on resource-constrained devices.",
    "alternatives": [
        "ResNet-50 (better accuracy/efficiency)",
        "EfficientNet-B0 (much smaller, similar accuracy)",
        "MobileNetV3 (for mobile deployment)",
    ],
    "recommended_settings": {
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "Adam",
        "scheduler": "StepLR",
    },
},
```

### 2.4 실제 추가 예시 (ultralytics - YOLOv5n)

`mvp/training/model_registry/ultralytics_models.py` 파일에 추가:

```python
# ============================================================
# P1: Core Expansion (6 models)
# ============================================================

"yolov5nu": {
    "display_name": "YOLOv5n-Ultralytics",
    "description": "YOLOv5 Nano - Ultra-lightweight detection model",
    "params": "1.9M",
    "input_size": 640,
    "pretrained_available": True,
    "recommended_batch_size": 64,
    "recommended_lr": 0.01,
    "tags": ["p1", "yolov5", "nano", "lightweight", "fast"],
    "priority": 1,
    "task_type": "object_detection",
    "benchmark": {
        "coco_map50": 45.7,
        "coco_map50_95": 28.0,
        "inference_speed": "6.3ms (V100)",
    },
    "use_cases": [
        "Real-time object detection on edge devices",
        "Mobile and embedded vision applications",
        "Rapid prototyping and quick experiments",
        "Resource-constrained deployment scenarios",
    ],
    "pros": [
        "Ultra-lightweight (1.9M params)",
        "Very fast inference (6.3ms)",
        "Good accuracy for its size",
        "Easy to deploy on mobile/edge",
    ],
    "cons": [
        "Lower accuracy than larger models",
        "May struggle with small objects",
        "Limited capacity for complex scenes",
    ],
    "when_to_use": "Choose YOLOv5n when deploying on edge devices or when inference speed is critical and moderate accuracy is acceptable.",
    "when_not_to_use": "Avoid for applications requiring high detection accuracy (>40 mAP) or detecting very small objects.",
    "alternatives": [
        "YOLOv8n (newer, similar performance)",
        "YOLOv5s (more accurate, slightly slower)",
        "MobileNet-SSD (alternative lightweight detector)",
    ],
    "recommended_settings": {
        "epochs": 100,
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "SGD",
        "scheduler": "Cosine",
    },
},
```

### 2.5 TaskType 값

`task_type` 필드에 사용할 수 있는 값:

```python
# timm (분류 모델)
"image_classification"

# ultralytics (탐지, 세그멘테이션, 포즈)
"object_detection"
"instance_segmentation"
"pose_estimation"
"zero_shot_detection"
```

### 2.6 Priority 설정 가이드

모델의 우선순위를 결정하는 기준:

| Priority | 설명 | 대상 모델 |
|----------|------|----------|
| **P0** (0) | Initial Validation - 플랫폼 기본 기능 검증용 | ResNet-50, EfficientNet-B0, YOLOv8n, YOLO11n |
| **P1** (1) | Core Expansion - 핵심 아키텍처 다양성 확보 | VGG-16, MobileNetV3, DenseNet, ViT, YOLOv5 variants |
| **P2** (2) | Full Coverage - 전문화/고급 아키텍처 | MaxViT, BEiT, ConvNeXt, YOLO-World, large models |

**선정 기준:**
- **P0**: 가장 널리 사용되고 안정적인 베이스라인 모델
- **P1**: 다양한 use case를 커버하는 핵심 모델
- **P2**: 특수 목적, 최신 연구, 고성능 모델

---

## Step 3: Adapter 구현 (필요시)

> **Note**: 기존 프레임워크(timm, ultralytics)를 사용하는 경우 이 단계는 **생략**합니다.
> 새로운 프레임워크를 추가할 때만 필요합니다.

### 3.1 Adapter 구조

모든 Adapter는 `mvp/training/adapters/base.py`의 `TrainingAdapter`를 상속해야 합니다.

```python
from adapters.base import TrainingAdapter, MetricsResult, TaskType

class MyNewAdapter(TrainingAdapter):
    """Adapter for new framework."""

    @classmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Return configuration schema."""
        from training.config_schemas import get_my_new_schema
        return get_my_new_schema()

    def train(self, config: Dict[str, Any]) -> MetricsResult:
        """
        Execute training with given config.

        Args:
            config: Training configuration
                - model_name: str
                - dataset_path: str
                - num_epochs: int
                - batch_size: int
                - learning_rate: float
                - ...

        Returns:
            MetricsResult with training metrics
        """
        # 1. Load model
        model = self._load_model(config)

        # 2. Prepare data
        train_loader, val_loader = self._prepare_data(config)

        # 3. Setup optimizer and scheduler
        optimizer = self._setup_optimizer(model, config)
        scheduler = self._setup_scheduler(optimizer, config)

        # 4. Training loop
        for epoch in range(config["num_epochs"]):
            train_metrics = self._train_epoch(model, train_loader, optimizer)
            val_metrics = self._validate(model, val_loader)

            # Save checkpoint
            if val_metrics["accuracy"] > best_acc:
                self._save_checkpoint(model, config["output_dir"])

        # 5. Return final metrics
        return MetricsResult(
            epoch=config["num_epochs"],
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            metrics={
                "accuracy": val_metrics["accuracy"],
                "top5_accuracy": val_metrics["top5_accuracy"],
            }
        )

    def validate(self, config: Dict[str, Any]) -> MetricsResult:
        """Execute validation."""
        # Implementation
        pass
```

### 3.2 Adapter 파일 생성

1. `mvp/training/adapters/my_new_adapter.py` 파일 생성
2. `mvp/training/adapters/__init__.py`에 추가:
   ```python
   from .my_new_adapter import MyNewAdapter

   __all__ = ["TrainingAdapter", "TimmAdapter", "UltralyticsAdapter", "MyNewAdapter"]
   ```

---

## Step 4: Config Schema 정의 (필요시)

> **Note**: 기존 프레임워크를 사용하는 경우 이 단계도 **생략**합니다.

### 4.1 Config Schema 구조

`mvp/training/config_schemas.py`에 새로운 스키마 함수 추가:

```python
def get_my_new_schema() -> ConfigSchema:
    """Return configuration schema for my new framework."""
    fields = [
        # Optimizer Settings
        ConfigField(
            name="optimizer_type",
            type="select",
            default="adam",
            options=["adam", "adamw", "sgd"],
            description="Optimizer algorithm",
            group="optimizer",
            required=False
        ),
        ConfigField(
            name="learning_rate",
            type="float",
            default=0.001,
            min=0.0001,
            max=0.1,
            step=0.0001,
            description="Learning rate",
            group="optimizer",
            required=True
        ),

        # Scheduler Settings
        ConfigField(
            name="scheduler_type",
            type="select",
            default="cosine",
            options=["none", "step", "cosine"],
            description="LR scheduler",
            group="scheduler",
            required=False
        ),

        # Augmentation Settings
        ConfigField(
            name="random_flip",
            type="bool",
            default=True,
            description="Random horizontal flip",
            group="augmentation",
            required=False
        ),
    ]

    presets = {
        "easy": {
            "optimizer_type": "adam",
            "learning_rate": 0.001,
            "scheduler_type": "cosine",
        },
        "medium": {
            "optimizer_type": "adamw",
            "learning_rate": 0.0005,
            "scheduler_type": "cosine",
        },
        "advanced": {
            "optimizer_type": "adamw",
            "learning_rate": 0.0003,
            "scheduler_type": "cosine",
        }
    }

    return ConfigSchema(fields=fields, presets=presets)
```

### 4.2 ConfigField 타입

| Type | 설명 | 추가 파라미터 |
|------|------|--------------|
| `"select"` | 드롭다운 선택 | `options: List[str]` |
| `"int"` | 정수 입력 | `min, max, step` |
| `"float"` | 실수 입력 | `min, max, step` |
| `"bool"` | 체크박스 | - |
| `"text"` | 텍스트 입력 | - |

### 4.3 Group 분류

일관성 있는 group 이름 사용:
- `"optimizer"` - 옵티마이저 설정
- `"scheduler"` - 스케줄러 설정
- `"augmentation"` - 데이터 증강
- `"validation"` - 검증 설정
- `"optimization"` - 학습 최적화

---

## Step 5: 호환성 테스트

### 5.1 테스트 스크립트 실행

```bash
cd mvp/training
venv/Scripts/python.exe test_model_compatibility.py
```

### 5.2 예상 출력

```
============================================================
TESTING TIMM MODEL COMPATIBILITY
============================================================
[OK] timm version: 0.9.12
[OK] Total available timm models: 1017

[OK] [P1] VGG-16                         (vgg16)
...

============================================================
TIMM SUMMARY
============================================================
P0: 4/4 available
P1: 7/7 available  <- 새로운 모델 포함
P2: 8/8 available

============================================================
OVERALL SUMMARY
============================================================
timm: 19/19 models available  <- 총 개수 증가
ultralytics: 19/19 known model patterns
```

### 5.3 실패 시 확인 사항

모델이 `[FAIL]`로 표시되면:

1. **모델명 오타 확인**
   ```python
   import timm
   # 정확한 모델명 검색
   matches = [m for m in timm.list_models() if "vgg" in m.lower()]
   print(matches)  # ['vgg11', 'vgg13', 'vgg16', 'vgg19', ...]
   ```

2. **버전 호환성 확인**
   - timm 버전: 0.9.12 이상
   - ultralytics 버전: 8.0.220 이상

3. **모델명 수정**
   - 레지스트리의 `model_name` (딕셔너리 키)을 정확한 이름으로 수정

---

## Step 6: UI 확인

### 6.1 Backend 서버 시작

```bash
cd mvp/backend
venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

### 6.2 Frontend 서버 시작

```bash
cd mvp/frontend
npm run dev
```

### 6.3 UI에서 확인할 항목

1. **모델 목록 API 테스트**
   ```bash
   curl http://localhost:8000/api/v1/models/list
   ```
   - 새로 추가한 모델이 목록에 표시되는지 확인

2. **우선순위 필터 테스트**
   ```bash
   curl http://localhost:8000/api/v1/models/list?priority=1
   ```
   - P1 모델만 필터링되는지 확인

3. **프레임워크 필터 테스트**
   ```bash
   curl http://localhost:8000/api/v1/models/list?framework=timm
   ```

4. **Model Selector UI 확인**
   - `http://localhost:3000` 접속
   - Training 페이지에서 모델 선택 UI 확인
   - 새로운 모델이 카드로 표시되는지 확인
   - 필터링 (우선순위, 프레임워크, 작업 유형) 동작 확인
   - 모델 카드 클릭 시 상세 정보 표시 확인

5. **Config Schema API 테스트**
   ```bash
   curl http://localhost:8000/api/v1/training/config-schema?framework=timm
   ```
   - Advanced Config 스키마가 올바르게 반환되는지 확인

---

## Step 7: 커밋 및 PR

### 7.1 변경 사항 확인

```bash
git status
git diff mvp/training/model_registry/timm_models.py
```

### 7.2 커밋 메시지 작성

Conventional Commits 형식 사용:

```bash
git add mvp/training/model_registry/timm_models.py
git commit -m "feat(mvp): add VGG-16 model to P1 registry

Add VGG-16 (vgg16) to P1 model registry with comprehensive metadata:
- Display name: VGG-16
- Params: 138.4M
- ImageNet Top-1: 71.6%
- Use cases: Transfer learning, educational purposes
- Recommended settings: Adam optimizer, 0.001 LR

Compatibility: Verified with timm 0.9.12

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 7.3 PR 생성

```bash
git push origin feat/add-vgg16-model
```

PR 제목 및 설명:
```markdown
## feat(mvp): Add VGG-16 to P1 Model Registry

### Summary
Add VGG-16 (classic CNN architecture) to the P1 model registry for core expansion.

### Changes
- Add VGG-16 metadata to `timm_models.py`
- Includes comprehensive documentation (use cases, pros/cons, alternatives)
- Compatibility verified with timm 0.9.12

### Test Results
- ✅ Model available in timm library
- ✅ Compatibility test passed
- ✅ API endpoints working
- ✅ UI displays model correctly

### Model Details
- **Priority**: P1 (Core Expansion)
- **Parameters**: 138.4M
- **ImageNet Top-1**: 71.6%
- **Use Case**: Transfer learning, educational baseline

### Checklist
- [x] Model metadata added to registry
- [x] Compatibility test passed
- [x] UI confirmed working
- [x] Documentation updated
```

---

## 트러블슈팅

### 문제 1: 모델이 API에서 보이지 않음

**증상**: 레지스트리에 추가했지만 API에서 모델이 안 보임

**해결 방법**:
```bash
# Backend 서버 재시작
cd mvp/backend
# Ctrl+C로 서버 중단 후
venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000

# 또는 main.py 파일 touch로 자동 reload
touch app/main.py
```

### 문제 2: 모델명이 틀려서 [FAIL] 표시

**증상**: 호환성 테스트에서 `[FAIL]` 표시

**해결 방법**:
```python
# 1. 정확한 모델명 찾기
import timm
matches = [m for m in timm.list_models() if "vgg" in m]
print(matches)

# 2. 레지스트리의 키를 정확한 이름으로 수정
# 잘못된 예: "vgg-16"
# 올바른 예: "vgg16"
```

### 문제 3: Config Schema가 표시되지 않음

**증상**: Advanced Config UI가 비어있음

**해결 방법**:
```bash
# 1. Config Schema API 테스트
curl http://localhost:8000/api/v1/training/config-schema?framework=timm

# 2. config_schemas.py에서 함수가 올바르게 정의되었는지 확인
# 3. Adapter의 get_config_schema() 메소드 확인
```

### 문제 4: 모델 학습이 실패함

**증상**: 모델을 선택하고 학습 시작 시 에러 발생

**확인 사항**:
1. **모델명이 정확한지** - 레지스트리의 키와 실제 라이브러리 모델명 일치 여부
2. **Pretrained 가중치** - `pretrained_available: True`인데 가중치가 없는 경우
3. **입력 크기** - `input_size`가 모델과 맞는지 확인
4. **배치 사이즈** - GPU 메모리에 맞는 `recommended_batch_size` 설정

```python
# 디버깅: 직접 모델 생성 테스트
import timm
model = timm.create_model("vgg16", pretrained=False, num_classes=10)
print(f"Model created: {model.__class__.__name__}")
```

---

## 부록: 참고 자료

### timm 공식 문서
- GitHub: https://github.com/huggingface/pytorch-image-models
- Docs: https://huggingface.co/docs/timm

### ultralytics 공식 문서
- GitHub: https://github.com/ultralytics/ultralytics
- Docs: https://docs.ultralytics.com/

### 프로젝트 내부 문서
- [Model Registry P0 Implementation](../planning/WEEK1_P0_FINAL.md)
- [Phased Implementation Plan](../planning/WEEK1_PHASED_IMPLEMENTATION.md)
- [Architecture Document](../architecture/ARCHITECTURE.md)

---

## 요약

1. **모델 선정**: timm/ultralytics 라이브러리에서 지원 여부 확인
2. **레지스트리 추가**: 메타데이터 완성 (기본 정보, 벤치마크, 가이드, 추천 설정)
3. **호환성 테스트**: `test_model_compatibility.py` 실행
4. **UI 확인**: Frontend에서 모델 선택 및 표시 확인
5. **커밋 & PR**: Conventional Commits 형식으로 커밋

**소요 시간**: 모델당 약 30-60분 (메타데이터 수집 포함)

새로운 프레임워크 추가 시에만 Adapter와 Config Schema 구현이 필요하며, 기존 프레임워크를 사용하는 경우 **레지스트리 추가만으로 완료**됩니다.
