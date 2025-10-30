# Week 1 P0 Implementation - Final Plan (with YOLO-World)

**Document Version:** 4.0
**Updated:** 2025-10-30
**Status:** Ready to Execute

---

## P0 모델 선정 (6개)

### 전략적 선택 이유

**확장성 조기 검증**:
- ✅ Classic CNN (ResNet-50)
- ✅ Modern Efficient (EfficientNetV2-S)
- ✅ Latest Lightweight (YOLOv11n)
- ✅ Latest Balanced (YOLOv11m)
- ✅ **혁신적 패러다임 (YOLO-World)** 🆕

**YOLO-World 추가 이유**:
1. 🚀 **새로운 Task Type 검증**: Open-vocabulary detection
2. 🎨 **UI 확장성 검증**: 텍스트 프롬프트 입력
3. 📚 **가이드 시스템 확장**: Special Features 섹션
4. 🔧 **Adapter 유연성 검증**: Custom config 처리
5. 💡 **차별화 요소**: Zero-shot detection 실제 구현

---

## P0 모델 상세 (6개)

### timm (2개)

#### 1. ResNet-50 ⭐ **Baseline Standard**
```python
"resnet50": {
    "display_name": "ResNet-50",
    "description": "Most popular baseline CNN - Industry standard for benchmarking",
    "params": "25.6M",
    "input_size": 224,
    "pretrained_available": True,
    "recommended_batch_size": 32,
    "recommended_lr": 0.001,
    "tags": ["baseline", "classic", "standard", "popular"],

    "benchmark": {
        "imagenet_top1": 80.4,
        "imagenet_top5": 95.1,
        "inference_speed_v100": "140 img/s",
        "training_time_epoch": "~2 hours (ImageNet, 8x V100)"
    },

    "use_cases": [
        "Baseline comparison",
        "Transfer learning starting point",
        "Educational purposes",
        "Production-ready classification"
    ],

    "pros": [
        "Well-documented and tested",
        "Excellent transfer learning",
        "Balanced accuracy/speed",
        "Widely supported"
    ],

    "cons": [
        "Not the most efficient",
        "Larger than modern mobile models",
        "Lower accuracy than ViT"
    ],

    "when_to_use": "When you need a reliable, well-understood baseline or starting point for transfer learning",

    "alternatives": [
        "EfficientNetV2-S (more efficient)",
        "ViT-Base (higher accuracy)"
    ]
}
```

#### 2. EfficientNetV2-Small ⭐ **Modern Efficient**
```python
"efficientnetv2_s": {
    "display_name": "EfficientNetV2-Small",
    "description": "Modern efficient CNN - Best accuracy/speed trade-off",
    "params": "21.5M",
    "input_size": 384,
    "pretrained_available": True,
    "recommended_batch_size": 64,
    "recommended_lr": 0.001,
    "tags": ["modern", "efficient", "balanced", "2021"],

    "benchmark": {
        "imagenet_top1": 84.3,
        "imagenet_top5": 97.0,
        "inference_speed_v100": "200 img/s",
        "training_time_epoch": "~1.5 hours (ImageNet, 8x V100)",
        "training_speedup": "11x faster than EfficientNet-B7"
    },

    "use_cases": [
        "Production deployment",
        "Resource-constrained environments",
        "Fast training required",
        "High accuracy needed"
    ],

    "pros": [
        "Training up to 11x faster than EfficientNet-B7",
        "Better accuracy than ResNet-50 with fewer params",
        "Progressive learning for stability",
        "Optimized for modern hardware"
    ],

    "cons": [
        "Larger input size (384) vs ResNet (224)",
        "Slightly more memory during training",
        "Less documentation than ResNet"
    ],

    "when_to_use": "When you want state-of-the-art efficiency and are willing to use modern architectures",

    "alternatives": [
        "ResNet-50 (more stable)",
        "MobileNetV4 (even lighter)"
    ]
}
```

---

### Ultralytics (4개)

#### 3. YOLOv11n ⭐ **Latest Lightweight**
```python
"yolo11n": {
    "display_name": "YOLOv11 Nano",
    "description": "Latest YOLO (Sep 2024) - Ultra-lightweight real-time detection",
    "params": "2.6M",
    "input_size": 640,
    "task_type": "object_detection",
    "pretrained_available": True,
    "recommended_batch_size": 64,
    "recommended_lr": 0.01,
    "tags": ["latest", "2024", "ultralight", "realtime", "edge"],

    "benchmark": {
        "coco_map50": 52.1,
        "coco_map50_95": 39.5,
        "inference_speed_v100": "120 FPS",
        "inference_speed_jetson_nano": "15 FPS",
        "inference_speed_cpu": "25 FPS",
        "model_size_mb": 5.8,
        "vs_yolov8n": "-22% params, +1.2 mAP"
    },

    "use_cases": [
        "Edge devices (Raspberry Pi, Jetson)",
        "Mobile deployment (iOS, Android)",
        "Real-time video processing",
        "Resource-constrained servers"
    ],

    "pros": [
        "22% fewer params than YOLOv8n",
        "Latest architecture (Sep 2024)",
        "Fast inference even on CPU",
        "Very small model size (5.8 MB)"
    ],

    "cons": [
        "Lower accuracy than larger models",
        "May struggle with small objects",
        "Less suitable for high-precision tasks"
    ],

    "when_to_use": "When deployment on edge/mobile devices is critical, or when real-time speed is more important than accuracy",

    "alternatives": [
        "YOLOv11m (better accuracy)",
        "YOLOv8n (more stable)"
    ]
}
```

#### 4. YOLOv11m ⭐ **Latest Balanced**
```python
"yolo11m": {
    "display_name": "YOLOv11 Medium",
    "description": "Latest YOLO (Sep 2024) - Best accuracy/speed balance",
    "params": "20.1M",
    "input_size": 640,
    "task_type": "object_detection",
    "pretrained_available": True,
    "recommended_batch_size": 16,
    "recommended_lr": 0.01,
    "tags": ["latest", "2024", "balanced", "production", "sota"],

    "benchmark": {
        "coco_map50": 67.8,
        "coco_map50_95": 51.5,
        "inference_speed_v100": "60 FPS",
        "inference_speed_t4": "35 FPS",
        "inference_speed_cpu": "5 FPS",
        "model_size_mb": 40.2,
        "vs_yolov8m": "-22% params, +1.3 mAP"
    },

    "use_cases": [
        "Production object detection",
        "Autonomous vehicles",
        "Security/surveillance",
        "Quality inspection",
        "Retail analytics"
    ],

    "pros": [
        "Best accuracy/speed trade-off in YOLO series",
        "22% fewer params than YOLOv8m",
        "Higher mAP than YOLOv8m (+1.3)",
        "Production-ready and battle-tested"
    ],

    "cons": [
        "Requires GPU for real-time",
        "Larger model size than nano (40 MB)",
        "Higher compute requirements"
    ],

    "when_to_use": "When you need the best balance of accuracy and speed for production deployment with GPU available",

    "alternatives": [
        "YOLOv11n (faster, edge)",
        "YOLOv11l (more accurate)"
    ]
}
```

#### 5. YOLO-World-v2-s 🆕 ⭐ **Open-Vocabulary Small**
```python
"yolo_world_v2_s": {
    "display_name": "YOLO-World v2 Small",
    "description": "Open-vocabulary detection (CVPR 2024) - Detect ANY object with text prompts",
    "params": "22M",
    "input_size": 640,
    "task_type": "open_vocabulary_detection",  # 🆕 New Task Type!
    "pretrained_available": True,
    "recommended_batch_size": 16,
    "recommended_lr": 0.01,
    "tags": ["cvpr2024", "open-vocab", "zero-shot", "innovative", "text-prompt"],

    "benchmark": {
        "lvis_map": 26.2,
        "lvis_map_rare": 17.8,  # Rare classes performance
        "coco_map50_95": 44.3,  # Zero-shot on COCO
        "inference_speed_v100": "52 FPS",
        "custom_classes_support": "Unlimited",
        "vs_traditional": "No retraining needed for new classes"
    },

    # 🆕 Special configuration for open-vocabulary
    "special_features": {
        "type": "open_vocabulary",
        "capabilities": [
            "Detect objects without training",
            "Custom text prompts as classes",
            "Zero-shot detection",
            "Dynamic class definition"
        ],
        "example_prompts": [
            "a red apple",
            "damaged product",
            "person wearing a hat",
            "car with license plate"
        ]
    },

    "use_cases": [
        "Retail: Detect new products without retraining",
        "Security: Custom threat detection",
        "Quality control: Find specific defects",
        "Research: Rapid prototyping with new classes",
        "E-commerce: Flexible product detection"
    ],

    "pros": [
        "No retraining for new object classes",
        "Natural language class definition",
        "Fast adaptation to new scenarios",
        "Handles rare/custom objects well"
    ],

    "cons": [
        "Lower accuracy than specialized models",
        "Requires careful prompt engineering",
        "Slower than standard YOLO (text encoding)",
        "Limited to detection (no segmentation yet)"
    ],

    "when_to_use": "When you need flexibility to detect new object types without retraining, or when dealing with long-tail/rare objects",

    "alternatives": [
        "YOLOv11m (higher accuracy, fixed classes)",
        "YOLO-World-v2-m (larger, more accurate)"
    ],

    # 🆕 How to use
    "usage_example": {
        "traditional_yolo": "model.predict('image.jpg')  # Detects 80 COCO classes",
        "yolo_world": "model.set_classes(['cat', 'dog', 'my custom object']).predict('image.jpg')  # Detects custom classes!"
    }
}
```

#### 6. YOLO-World-v2-m 🆕 ⭐ **Open-Vocabulary Medium**
```python
"yolo_world_v2_m": {
    "display_name": "YOLO-World v2 Medium",
    "description": "Open-vocabulary detection (CVPR 2024) - More accurate zero-shot detection",
    "params": "42M",
    "input_size": 640,
    "task_type": "open_vocabulary_detection",
    "pretrained_available": True,
    "recommended_batch_size": 8,
    "recommended_lr": 0.01,
    "tags": ["cvpr2024", "open-vocab", "zero-shot", "accurate"],

    "benchmark": {
        "lvis_map": 35.4,  # State-of-the-art on LVIS
        "lvis_map_rare": 26.8,
        "coco_map50_95": 48.1,
        "inference_speed_v100": "52 FPS",
        "custom_classes_support": "Unlimited",
        "vs_yolo_world_s": "+9.2 mAP on LVIS"
    },

    "special_features": {
        "type": "open_vocabulary",
        "capabilities": [
            "State-of-the-art open-vocab performance",
            "Better rare object detection",
            "Robust to prompt variations",
            "Multi-language support (experimental)"
        ],
        "example_prompts": [
            "vintage car from 1950s",
            "person with blue backpack",
            "damaged packaging box",
            "ripe banana vs unripe banana"
        ]
    },

    "use_cases": [
        "Large-scale retail inventory",
        "Advanced security systems",
        "Medical imaging (custom conditions)",
        "Autonomous vehicles (rare scenarios)",
        "Wildlife monitoring (species detection)"
    ],

    "pros": [
        "Best-in-class open-vocabulary accuracy",
        "Excellent rare object detection",
        "More robust prompt understanding",
        "Still real-time (52 FPS)"
    ],

    "cons": [
        "2x params vs small version",
        "Higher memory usage",
        "Slower than standard YOLO",
        "Requires more compute"
    ],

    "when_to_use": "When you need maximum accuracy for open-vocabulary detection and have sufficient GPU resources",

    "alternatives": [
        "YOLO-World-v2-s (faster, lighter)",
        "YOLOv11l (higher fixed-class accuracy)"
    ]
}
```

---

## 시스템 확장 필요 사항

### 1. TaskType 추가

```python
# mvp/training/adapters/base.py

class TaskType(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    OBB_DETECTION = "obb_detection"
    OPEN_VOCABULARY_DETECTION = "open_vocabulary_detection"  # 🆕 추가!
```

### 2. TrainingConfig 확장

```python
# mvp/backend/app/schemas/training.py

class TrainingConfig(BaseModel):
    """Training configuration schema."""

    framework: str = Field("timm", description="Framework")
    model_name: str = Field(..., description="Model name")
    task_type: str = Field(..., description="Task type")

    # ... existing fields ...

    # 🆕 Open-vocabulary 전용 설정
    custom_prompts: Optional[List[str]] = Field(
        None,
        description="Custom text prompts for open-vocabulary detection (YOLO-World only)"
    )
    prompt_mode: Optional[str] = Field(
        "offline",
        description="Prompt mode: 'offline' (pre-computed) or 'dynamic' (runtime)"
    )
```

### 3. UltralyticsAdapter 확장

```python
# mvp/training/adapters/ultralytics_adapter.py

class UltralyticsAdapter(TrainingAdapter):
    """Adapter for Ultralytics YOLO models."""

    def prepare_model(self) -> None:
        """Initialize model."""
        from ultralytics import YOLO

        model_name = self.model_config.model_name
        task_type = self.model_config.task_type

        # Standard YOLO models
        if task_type != TaskType.OPEN_VOCABULARY_DETECTION:
            model_file = self._get_model_file(model_name, task_type)
            self.model = YOLO(model_file)

        # 🆕 YOLO-World special handling
        else:
            from ultralytics import YOLOWorld

            # Load YOLO-World model
            self.model = YOLOWorld(f"{model_name}.pt")

            # Set custom classes if provided
            if self.model_config.custom_prompts:
                self.model.set_classes(self.model_config.custom_prompts)
                print(f"[YOLOWorld] Custom classes: {self.model_config.custom_prompts}")
            else:
                # Use default COCO classes
                print("[YOLOWorld] Using default COCO classes")

        # Move to device
        device = self.training_config.device if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
```

### 4. Frontend UI 확장

#### ModelSelector.tsx 수정

```typescript
// mvp/frontend/components/training/ModelSelector.tsx

interface ModelSelectorProps {
  onSelect: (model: ModelInfo, config?: ModelConfig) => void;
}

export default function ModelSelector({ onSelect }: ModelSelectorProps) {
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [showPromptModal, setShowPromptModal] = useState(false);

  const handleModelSelect = (model: ModelInfo) => {
    // 🆕 Open-vocabulary model requires prompt input
    if (model.task_type === "open_vocabulary_detection") {
      setSelectedModel(model);
      setShowPromptModal(true);
    } else {
      onSelect(model);
    }
  };

  return (
    <div>
      {/* Model grid */}
      <ModelGrid onSelect={handleModelSelect} />

      {/* 🆕 Prompt input modal for YOLO-World */}
      {showPromptModal && selectedModel && (
        <CustomPromptsModal
          model={selectedModel}
          onConfirm={(prompts) => {
            onSelect(selectedModel, { custom_prompts: prompts });
            setShowPromptModal(false);
          }}
          onCancel={() => setShowPromptModal(false)}
        />
      )}
    </div>
  );
}
```

#### CustomPromptsModal.tsx (신규)

```typescript
// mvp/frontend/components/training/CustomPromptsModal.tsx

interface CustomPromptsModalProps {
  model: ModelInfo;
  onConfirm: (prompts: string[]) => void;
  onCancel: () => void;
}

export default function CustomPromptsModal({ model, onConfirm, onCancel }: CustomPromptsModalProps) {
  const [prompts, setPrompts] = useState<string[]>(['']);

  const addPrompt = () => setPrompts([...prompts, '']);
  const removePrompt = (index: number) => setPrompts(prompts.filter((_, i) => i !== index));
  const updatePrompt = (index: number, value: string) => {
    const newPrompts = [...prompts];
    newPrompts[index] = value;
    setPrompts(newPrompts);
  };

  return (
    <div className="modal">
      <h2>Define Custom Classes for {model.display_name}</h2>

      <div className="prompt-info">
        <p>YOLO-World can detect any objects you describe!</p>
        <p>Enter natural language descriptions of objects to detect:</p>
      </div>

      <div className="prompt-examples">
        <strong>Examples:</strong>
        <ul>
          <li>"red apple"</li>
          <li>"person wearing a hat"</li>
          <li>"damaged product"</li>
          <li>"car with license plate"</li>
        </ul>
      </div>

      <div className="prompt-inputs">
        {prompts.map((prompt, index) => (
          <div key={index} className="prompt-input-row">
            <input
              type="text"
              placeholder={`Class ${index + 1}: e.g., "red apple"`}
              value={prompt}
              onChange={(e) => updatePrompt(index, e.target.value)}
            />
            {prompts.length > 1 && (
              <button onClick={() => removePrompt(index)}>Remove</button>
            )}
          </div>
        ))}
      </div>

      <button onClick={addPrompt}>+ Add Another Class</button>

      <div className="modal-actions">
        <button onClick={onCancel}>Cancel</button>
        <button
          onClick={() => onConfirm(prompts.filter(p => p.trim()))}
          disabled={prompts.filter(p => p.trim()).length === 0}
        >
          Use These Classes
        </button>
      </div>

      <div className="tip">
        💡 Tip: Be specific! "red apple" works better than just "apple"
      </div>
    </div>
  );
}
```

### 5. 가이드 시스템 확장

#### Special Features 섹션 추가

```typescript
// mvp/frontend/components/training/guide/SpecialFeaturesSection.tsx

interface SpecialFeaturesSectionProps {
  features: {
    type: string;
    capabilities: string[];
    example_prompts?: string[];
    usage_example?: {
      traditional_yolo: string;
      yolo_world: string;
    };
  };
}

export default function SpecialFeaturesSection({ features }: SpecialFeaturesSectionProps) {
  return (
    <div className="special-features-section">
      <h3>🌟 Special Features</h3>

      <div className="feature-type">
        <strong>Type:</strong> {features.type}
      </div>

      <div className="capabilities">
        <strong>Capabilities:</strong>
        <ul>
          {features.capabilities.map((cap, i) => (
            <li key={i}>{cap}</li>
          ))}
        </ul>
      </div>

      {features.example_prompts && (
        <div className="example-prompts">
          <strong>Example Prompts:</strong>
          <div className="prompt-chips">
            {features.example_prompts.map((prompt, i) => (
              <span key={i} className="prompt-chip">"{prompt}"</span>
            ))}
          </div>
        </div>
      )}

      {features.usage_example && (
        <div className="usage-comparison">
          <strong>How to Use:</strong>

          <div className="code-comparison">
            <div className="traditional">
              <span className="label">Traditional YOLO:</span>
              <code>{features.usage_example.traditional_yolo}</code>
              <p className="note">Fixed 80 COCO classes</p>
            </div>

            <div className="arrow">→</div>

            <div className="yolo-world">
              <span className="label">YOLO-World:</span>
              <code>{features.usage_example.yolo_world}</code>
              <p className="note">Any custom classes!</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## 구현 스케줄 (Day 1-2 재조정)

### Day 1: Infrastructure + P0 Models (6개)

#### Morning (4h)
```
09:00-10:00  모델 레지스트리 구조 생성
             - mvp/training/model_registry/
             - __init__.py, timm_models.py, ultralytics_models.py

10:00-11:30  P0 6개 모델 메타데이터 작성 (full guide 포함)
             - ResNet-50, EfficientNetV2-S (timm)
             - YOLOv11n, YOLOv11m, YOLO-World s/m (ultralytics)
             - special_features 정보 포함

11:30-13:00  🆕 TaskType 확장 및 Config 수정
             - base.py: OPEN_VOCABULARY_DETECTION 추가
             - training.py: custom_prompts 필드 추가
             - enums.py: TaskType 업데이트
```

#### Afternoon (4h)
```
14:00-15:30  API 엔드포인트 구현
             - GET /models/list
             - GET /models/{framework}/{model_name}
             - GET /models/{framework}/{model_name}/guide
             - Response에 special_features 포함

15:30-17:00  🆕 UltralyticsAdapter 확장
             - YOLO-World 처리 로직
             - set_classes() 호출
             - custom_prompts 전달

17:00-18:00  기본 UI 컴포넌트
             - ModelCard.tsx (기본)
             - ModelSelector.tsx (그리드)
```

### Day 2: Guide System + YOLO-World UI

#### Morning (4h)
```
09:00-10:30  ModelGuideDrawer 컴포넌트 (슬라이드 패널)
             - 6개 섹션 레이아웃
             - 애니메이션, 반응형

10:30-12:00  가이드 섹션 1-3 구현
             - QuickStats.tsx
             - BenchmarkSection.tsx
             - UsageGuidance.tsx

12:00-13:00  🆕 SpecialFeaturesSection.tsx
             - YOLO-World capabilities 표시
             - Example prompts chips
             - Usage comparison
```

#### Afternoon (4h)
```
14:00-15:00  가이드 섹션 4-6 구현
             - ModelComparisonTable.tsx
             - PerformanceChart.tsx (scatter plot)
             - RecommendedSettings.tsx

15:00-16:30  🆕 CustomPromptsModal.tsx
             - 프롬프트 입력 UI
             - Add/Remove prompts
             - Example suggestions

16:30-18:00  P0 통합 테스트
             - 6개 모델 UI 동작 확인
             - YOLO-World 프롬프트 입력 테스트
             - 가이드 정보 표시 확인
```

---

## 검증 기준 (Day 2 종료 시)

### 기능 검증

- [ ] **모델 레지스트리**: 6개 모델 메타데이터 완성
- [ ] **API**: `/models/list`, `/models/{}/guide` 정상 동작
- [ ] **UI - 기본**: 모델 카드 그리드 표시
- [ ] **UI - 가이드**: 슬라이드 패널로 6개 섹션 표시
- [ ] **UI - YOLO-World**: 프롬프트 입력 모달 동작
- [ ] **Adapter**: YOLO-World custom_prompts 처리

### YOLO-World 특수 검증

- [ ] TaskType.OPEN_VOCABULARY_DETECTION 추가됨
- [ ] TrainingConfig.custom_prompts 동작
- [ ] UltralyticsAdapter가 YOLOWorld 로드
- [ ] set_classes() 호출 확인
- [ ] CustomPromptsModal UI 동작
- [ ] SpecialFeaturesSection 표시

### 학습 테스트 (간단)

**Standard Models**:
- [ ] ResNet-50: ImageFolder 데이터로 3 epochs
- [ ] YOLOv11n: COCO subset으로 3 epochs

**YOLO-World**:
- [ ] Custom prompts: ["cat", "dog", "car"]로 inference 테스트
- [ ] Zero-shot detection 동작 확인

---

## P0 완료 시 달성 목표

### 1. 시스템 유효성 검증 ✅

- Adapter 패턴이 다양한 모델에 동작
- Classic CNN (ResNet) ✅
- Modern CNN (EfficientNetV2) ✅
- Latest YOLO (v11) ✅
- Open-vocab (YOLO-World) ✅

### 2. 확장성 검증 ✅

- 새로운 TaskType 추가 가능 (OPEN_VOCABULARY_DETECTION)
- Config 확장 가능 (custom_prompts)
- Adapter 유연성 (특수 모델 처리)
- UI 확장 가능 (CustomPromptsModal)

### 3. 가이드 시스템 검증 ✅

- 6개 섹션 완성
- 특수 기능 표시 (Special Features)
- 인터랙티브 요소 (프롬프트 입력)
- 비교 기능 (Similar Models)

### 4. 차별화 요소 구현 ✅

- **YOLO-World**: 업계 최초 실시간 open-vocabulary
- Zero-shot detection 실제 동작
- 텍스트 프롬프트로 클래스 정의
- 재학습 없이 새 객체 검출

---

## Week 1 나머지 계획 (Day 3-7)

### Day 3-4: P1 (12개 모델)
- timm 6개: Mobile, ViT, Classic
- ultralytics 6개: Detection, Seg, Pose 확장

### Day 5: P2 (15개 모델)
- 모든 변형 포함
- YOLOv10 (NMS-free)
- OBB, MaxViT 등

### Day 6-7: Polish & Docs
- 전체 검증
- 리포트 작성
- Week 2 준비

---

## 기대 효과

### 조기 검증 (Day 2)
- ✅ 시스템 동작 확인
- ✅ YOLO-World로 확장성 입증
- ✅ 가이드 시스템 완성

### 차별화 (Day 2)
- 🚀 Zero-shot detection 제공
- 🎨 텍스트 기반 클래스 정의
- 💡 혁신적 UX (프롬프트 입력)

### 신뢰성 (Day 2)
- 📊 상세한 가이드 정보
- 🔍 모델 비교 기능
- 💬 실제 사용 사례

---

## 다음 단계

**즉시 시작 가능합니다!**

1. **브랜치 생성**
   ```bash
   git checkout -b feat/model-registry-p0-yoloworld
   ```

2. **Day 1 Morning 시작** (09:00)
   ```python
   # 1. TaskType 확장
   # mvp/training/adapters/base.py
   class TaskType(Enum):
       # ... existing
       OPEN_VOCABULARY_DETECTION = "open_vocabulary_detection"

   # 2. 모델 레지스트리 작성
   # mvp/training/model_registry/ultralytics_models.py
   ULTRALYTICS_MODEL_REGISTRY = {
       "yolo11n": { ... },
       "yolo11m": { ... },
       "yolo_world_v2_s": { ... },  # 🆕
       "yolo_world_v2_m": { ... },  # 🆕
   }
   ```

---

*Document Version: 4.0*
*Created: 2025-10-30*
*Ready to Execute!* 🚀
