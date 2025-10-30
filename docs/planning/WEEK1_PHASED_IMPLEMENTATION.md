# Week 1 Phased Implementation Plan

**Document Version:** 3.0
**Updated:** 2025-10-30
**Status:** Implementation Ready

---

## 전략: 점진적 검증 및 확장

### 핵심 아이디어

```
P0 (Quick Win, Day 1-2)
  ↓ 검증: 시스템 동작 확인

P1 (Core Expansion, Day 3-4)
  ↓ 검증: 다양한 모델 타입

P2 (Full Coverage, Day 5-7)
  ↓ 완성: 모든 카테고리 커버
```

**장점**:
1. ✅ 빠른 피드백 (Day 1-2)
2. ✅ 점진적 위험 관리
3. ✅ 각 단계마다 학습 반영
4. ✅ 중단 시점 선택 가능

---

## Priority 0: Quick Win (필수, Day 1-2)

**목표**: 시스템 유효성 검증 + 모델 가이드 UI 구축

**모델 선정** (총 4개):

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

    # 가이드 정보
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
    "alternatives": ["EfficientNetV2-S (more efficient)", "ViT-Base (higher accuracy)"]
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
        "training_time_epoch": "~1.5 hours (ImageNet, 8x V100)"
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
    "alternatives": ["ResNet-50 (more stable)", "MobileNetV4 (even lighter)"]
}
```

### Ultralytics (2개)

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
        "model_size_mb": 5.8
    },
    "use_cases": [
        "Edge devices (Raspberry Pi, Jetson)",
        "Mobile deployment",
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
    "alternatives": ["YOLOv11m (better accuracy)", "YOLOv8n (more stable)"]
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
        "model_size_mb": 40.2
    },
    "use_cases": [
        "Production object detection",
        "Autonomous vehicles",
        "Security/surveillance",
        "Quality inspection"
    ],
    "pros": [
        "Best accuracy/speed trade-off",
        "22% fewer params than YOLOv8m",
        "Higher mAP than YOLOv8m",
        "Production-ready"
    ],
    "cons": [
        "Requires GPU for real-time",
        "Larger model size than nano",
        "Higher compute requirements"
    ],
    "when_to_use": "When you need the best balance of accuracy and speed for production deployment",
    "alternatives": ["YOLOv11n (faster)", "YOLOv11l (more accurate)"]
}
```

---

## Priority 1: Core Expansion (Day 3-4)

**목표**: 주요 카테고리 커버

**모델 선정** (총 12개):

### timm (6개)

#### Mobile 계열 (2개)
- **MobileNetV4-Medium**: 최신 mobile (2024)
- **MobileNetV3-Large**: Baseline mobile

#### ViT 계열 (2개)
- **ViT-Base/16**: Standard transformer
- **SigLIP-2 (ViT-SO150M)**: SOTA ViT (88.1%)

#### Classic CNN (2개)
- **ResNet-18**: Lightweight baseline
- **ConvNeXt-Tiny**: Modern CNN

### Ultralytics (6개)

#### Detection (2개)
- **YOLOv11s**: Small (9.4M params)
- **YOLOv11l**: Large (25.3M params)

#### Segmentation (2개)
- **YOLOv11n-seg**: Lightweight segmentation
- **YOLOv11m-seg**: Balanced segmentation

#### Pose (2개)
- **YOLOv11m-pose**: Balanced pose
- **YOLOv11l-pose**: Accurate pose

---

## Priority 2: Full Coverage (Day 5-7)

**목표**: 모든 특수 기능 및 카테고리

**모델 선정** (총 15개):

### timm (8개)
- **EfficientNet-B0, B4**: Original EfficientNet
- **EfficientNetV2-M**: Larger V2
- **ResNet-101**: Deep ResNet
- **ViT-Large/16**: Large transformer
- **MobileNetV5-Large**: Latest mobile (2024)
- **ConvNeXt-Base**: Larger modern CNN
- **MaxViT-Tiny**: Hybrid CNN+ViT

### Ultralytics (7개)
- **YOLOv11x**: Maximum accuracy detection
- **YOLOv11x-seg**: Maximum accuracy segmentation
- **YOLO-World-v2-s**: Open-vocabulary (혁신!)
- **YOLO-World-v2-m**: Larger open-vocab
- **YOLOv10n, YOLOv10s**: NMS-free
- **YOLOv11n-obb**: Oriented bounding box

---

## 모델 선택 가이드 시스템 설계

### UX 구조

```
┌─────────────────────────────────────────────────────────────┐
│  Model Selection Page                                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Filters: Framework ▼  Task ▼  Tags ▼]  [Search: ___]     │
│                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐│
│  │  ResNet-50     │  │ EfficientNetV2 │  │  YOLOv11n      ││
│  │  25.6M params  │  │ 21.5M params   │  │  2.6M params   ││
│  │  ⭐⭐⭐⭐       │  │ ⭐⭐⭐⭐⭐     │  │  ⭐⭐⭐⭐⭐    ││
│  │                │  │                │  │                ││
│  │  [Select]      │  │  [Select]      │  │  [Select]      ││
│  │  [📖 Guide]    │  │  [📖 Guide]    │  │  [📖 Guide]    ││ ← 가이드 버튼
│  └────────────────┘  └────────────────┘  └────────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
                           ↓ Click [📖 Guide]
┌─────────────────────────────────────────────────────────────┐
│  ← Back                   ResNet-50 Guide               [×]  │ ← Slide Panel
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📊 Quick Stats                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Accuracy:  ████████░░  80.4%                          │ │
│  │  Speed:     ██████░░░░  140 img/s                      │ │
│  │  Size:      █████░░░░░  25.6M params                   │ │
│  │  Difficulty: ███░░░░░░  Easy                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  📈 Benchmark Performance                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ImageNet-1k:     80.4% top-1, 95.1% top-5            │ │
│  │  Inference (V100): 140 img/s                           │ │
│  │  Training Time:    ~2h/epoch (8x V100)                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  🎯 When to Use                                              │
│  Use ResNet-50 when you need a reliable, well-understood    │
│  baseline or starting point for transfer learning.          │
│                                                               │
│  ✅ Pros                          ❌ Cons                    │
│  • Well-documented               • Not most efficient       │
│  • Excellent transfer learning   • Larger than mobile       │
│  • Balanced accuracy/speed       • Lower than ViT           │
│                                                               │
│  💡 Use Cases                                                │
│  • Baseline comparison                                       │
│  • Transfer learning                                         │
│  • Educational purposes                                      │
│  • Production classification                                 │
│                                                               │
│  🔄 Similar Models                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Model          Accuracy  Speed   Size   Best For   │   │
│  │  ResNet-50      80.4%     140     25.6M  Baseline   │ ← Current
│  │  EfficientNetV2 84.3%     200     21.5M  Efficiency │   │
│  │  ViT-Base       84.5%     110     86M    Accuracy   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  📊 Accuracy vs Speed Plot                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 90%│                        ViT-L ●                    │ │
│  │    │                                                   │ │
│  │ 85%│          ViT-B ●   EfficientNetV2 ●              │ │
│  │    │                                                   │ │
│  │ 80%│    ResNet-50 ●                                    │ │
│  │    │                                                   │ │
│  │ 75%│  MobileNetV3 ●                                    │ │
│  │    └────────────────────────────────────────────────  │ │
│  │        50    100   150   200   250 (img/s)            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  🔧 Recommended Settings                                     │
│  Batch Size: 32  |  Learning Rate: 0.001  |  Epochs: 50     │
│                                                               │
│  [Select this Model]  [Compare with Another]                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 가이드 시스템 상세 설계

### 1. 정보 아키텍처 (6 Sections)

#### Section 1: Quick Stats (한눈에 파악)
```typescript
interface QuickStats {
  accuracy: {
    value: number;      // 80.4
    max: number;        // 90 (for bar visualization)
    label: string;      // "ImageNet Top-1"
  };
  speed: {
    value: number;      // 140
    unit: string;       // "img/s"
    device: string;     // "V100"
  };
  size: {
    params: string;     // "25.6M"
    modelSizeMB: number; // 98
  };
  difficulty: "Easy" | "Medium" | "Hard";
}
```

**시각화**:
- Progress bars (색상: 초록 > 80%, 노랑 60-80%, 빨강 < 60%)
- Icons: 🎯 Accuracy, ⚡ Speed, 📦 Size, 🎓 Difficulty

#### Section 2: Benchmark Performance (구체적 수치)
```typescript
interface Benchmark {
  dataset: string;           // "ImageNet-1k"
  metrics: {
    top1: number;            // 80.4
    top5?: number;           // 95.1
    map50?: number;          // For detection
    map50_95?: number;       // For detection
  };
  inference: {
    device: string;          // "V100"
    speed: number;           // 140
    unit: string;            // "img/s"
    batchSize?: number;      // 32
  };
  training: {
    timePerEpoch: string;    // "~2 hours"
    hardware: string;        // "8x V100"
  };
}
```

**표시 형식**:
```
📊 ImageNet-1k:  80.4% top-1  •  95.1% top-5
⚡ Inference:    140 img/s (V100, batch=32)
⏱️ Training:     ~2h/epoch (ImageNet, 8x V100)
```

#### Section 3: When to Use (실용적 조언)
```typescript
interface UsageGuidance {
  summary: string;  // 1-2 문장 요약
  pros: string[];   // 3-5개 장점
  cons: string[];   // 3-5개 단점
  useCases: string[]; // 4-6개 구체적 use case
  whenToUse: string;  // 명확한 사용 시점
  whenNotToUse?: string; // 사용하지 말아야 할 때
}
```

**예시**:
```markdown
🎯 **When to Use**
Use ResNet-50 when you need a reliable, well-understood baseline
or starting point for transfer learning.

✅ **Pros**
• Well-documented and widely tested
• Excellent transfer learning performance
• Balanced accuracy/speed trade-off

❌ **Cons**
• Not the most parameter-efficient
• Lower accuracy than Vision Transformers
```

#### Section 4: Similar Models (비교 테이블)
```typescript
interface ModelComparison {
  models: Array<{
    name: string;
    accuracy: number;
    speed: number;
    size: string;
    bestFor: string;
    isCurrent?: boolean;  // Highlight current model
  }>;
}
```

**테이블 형식**:
```
┌──────────────────┬──────────┬───────┬────────┬─────────────┐
│ Model            │ Accuracy │ Speed │ Size   │ Best For    │
├──────────────────┼──────────┼───────┼────────┼─────────────┤
│ ResNet-50 ⭐     │ 80.4%    │ 140   │ 25.6M  │ Baseline    │
│ EfficientNetV2-S │ 84.3%    │ 200   │ 21.5M  │ Efficiency  │
│ ViT-Base         │ 84.5%    │ 110   │ 86M    │ Accuracy    │
└──────────────────┴──────────┴───────┴────────┴─────────────┘
```

#### Section 5: Visualization (시각적 비교)
```typescript
interface PerformancePlot {
  type: "scatter" | "bar" | "radar";
  xAxis: {
    label: string;     // "Speed (img/s)"
    min: number;
    max: number;
  };
  yAxis: {
    label: string;     // "Accuracy (%)"
    min: number;
    max: number;
  };
  points: Array<{
    name: string;
    x: number;
    y: number;
    isCurrent?: boolean;
    color?: string;
  }>;
}
```

**Scatter Plot** (Accuracy vs Speed):
- X축: Speed (img/s)
- Y축: Accuracy (%)
- 크기: Model size (작은 원 = 작은 모델)
- 색상: Framework (timm=파랑, ultralytics=주황)
- 현재 모델: 테두리 강조

**예시**:
```
Accuracy
   ↑
90%│                    ● ViT-Large
   │
85%│        ● ViT-Base    ● EfficientNetV2
   │
80%│  ◉ ResNet-50
   │
75%│ ● MobileNetV3
   │
   └────────────────────────────────→ Speed
       50    100    150    200   (img/s)

◉ = Current model
● = Other models
```

#### Section 6: Recommended Settings (실용 정보)
```typescript
interface RecommendedSettings {
  batchSize: {
    value: number;
    range: [number, number];  // [min, max]
    note?: string;
  };
  learningRate: {
    value: number;
    range: [number, number];
    note?: string;
  };
  epochs: {
    value: number;
    range: [number, number];
    note?: string;
  };
  imageSize: number;
  optimizer?: string;
  scheduler?: string;
}
```

**표시 형식**:
```
🔧 Recommended Settings

Batch Size:     32    (range: 16-64)
Learning Rate:  0.001 (range: 0.0001-0.01)
Epochs:         50    (range: 20-100)
Image Size:     224×224
Optimizer:      Adam or AdamW
Scheduler:      Cosine annealing

💡 Tip: Start with default values and adjust based on your dataset
```

---

## Frontend 구현 계획

### Component 구조

```typescript
// 1. ModelCard Component (카드)
interface ModelCardProps {
  model: ModelInfo;
  onSelect: (model: ModelInfo) => void;
  onShowGuide: (model: ModelInfo) => void;
}

// 2. ModelGuideDrawer Component (슬라이드 패널)
interface ModelGuideDrawerProps {
  model: ModelInfo;
  isOpen: boolean;
  onClose: () => void;
  onSelect: (model: ModelInfo) => void;
  similarModels: ModelInfo[];
}

// 3. ModelComparisonTable Component
interface ModelComparisonTableProps {
  models: ModelInfo[];
  currentModel: string;
  onModelClick: (model: ModelInfo) => void;
}

// 4. PerformanceScatterPlot Component
interface PerformanceScatterPlotProps {
  models: ModelInfo[];
  currentModel: string;
  xMetric: "speed" | "size";
  yMetric: "accuracy";
}
```

### 파일 구조

```
mvp/frontend/components/training/
├── ModelSelector.tsx              # 메인 모델 선택 페이지
├── ModelCard.tsx                  # 모델 카드
├── ModelGuideDrawer.tsx          # 슬라이드 패널 (가이드)
│
├── guide/                         # 가이드 서브 컴포넌트
│   ├── QuickStats.tsx            # Section 1
│   ├── BenchmarkSection.tsx      # Section 2
│   ├── UsageGuidance.tsx         # Section 3
│   ├── ModelComparisonTable.tsx  # Section 4
│   ├── PerformanceChart.tsx      # Section 5
│   └── RecommendedSettings.tsx   # Section 6
│
└── hooks/
    ├── useModelGuide.ts          # 가이드 데이터 fetch
    └── useModelComparison.ts     # 비교 모델 계산
```

### API 확장

```typescript
// GET /api/v1/models/{framework}/{model_name}/guide
interface ModelGuideResponse {
  model: ModelInfo;
  quickStats: QuickStats;
  benchmark: Benchmark;
  usageGuidance: UsageGuidance;
  similarModels: ModelInfo[];
  recommendedSettings: RecommendedSettings;
  performanceData: {
    allModels: Array<{
      name: string;
      accuracy: number;
      speed: number;
      size: number;
    }>;
  };
}
```

---

## 구현 스케줄 (Day별)

### Day 1: P0 Infrastructure
```
Morning (4h):
- [ ] 모델 레지스트리 구조 생성
- [ ] P0 4개 모델 메타데이터 작성 (full guide 포함)
- [ ] API 엔드포인트: /models/list

Afternoon (4h):
- [ ] ModelCard 컴포넌트 (기본)
- [ ] ModelSelector 페이지 (그리드 레이아웃)
- [ ] 기본 필터링 (framework, tags)
```

### Day 2: P0 Guide System
```
Morning (4h):
- [ ] ModelGuideDrawer 컴포넌트 (슬라이드 패널)
- [ ] QuickStats 섹션
- [ ] BenchmarkSection 섹션
- [ ] UsageGuidance 섹션

Afternoon (4h):
- [ ] ModelComparisonTable 섹션
- [ ] PerformanceChart 섹션 (scatter plot)
- [ ] RecommendedSettings 섹션
- [ ] P0 4개 모델로 통합 테스트
```

### Day 3: P0 Validation + P1 Start
```
Morning (3h):
- [ ] P0 4개 모델 학습 테스트
- [ ] UI/UX 개선 (피드백 반영)
- [ ] 가이드 정보 검증

Afternoon (5h):
- [ ] P1 12개 모델 메타데이터 작성
- [ ] 가이드 정보 작성 (간략화 가능)
- [ ] API에 P1 모델 추가
```

### Day 4: P1 Completion
```
All day (8h):
- [ ] P1 모델 UI 통합
- [ ] 필터링 고도화 (다중 태그, 검색)
- [ ] 정렬 기능 (accuracy, speed, size)
- [ ] P1 주요 모델 학습 테스트
```

### Day 5: P2 Models
```
All day (8h):
- [ ] P2 15개 모델 메타데이터
- [ ] YOLO-World 특수 처리 (open-vocab)
- [ ] YOLOv10, OBB 등 특수 task 지원
- [ ] UI에 P2 모델 통합
```

### Day 6-7: Polish & Documentation
```
Day 6:
- [ ] 모든 모델 카테고리 검증
- [ ] 가이드 정보 완성도 확인
- [ ] UI/UX 최종 개선
- [ ] 성능 최적화 (lazy loading, caching)

Day 7:
- [ ] Week 1 검증 리포트 작성
- [ ] 사용자 가이드 문서 작성
- [ ] 스크린샷 및 데모 영상
- [ ] Week 2 (Docker) 준비
```

---

## 효과적인 모델 선택 가이드 전략

### 1. 사용자 여정 (User Journey)

```
Step 1: 목적 파악
"어떤 작업을 하시나요?"
→ Classification / Detection / Segmentation

Step 2: 제약 조건 확인
"어떤 환경에서 사용하시나요?"
→ Cloud / Edge / Mobile
→ GPU available? Memory limit?

Step 3: 우선순위 설정
"무엇이 가장 중요한가요?"
→ Accuracy / Speed / Size

Step 4: 모델 추천
Top 3 모델 제시 (이유와 함께)

Step 5: 상세 비교
가이드 패널로 심층 정보 확인
```

### 2. 인터랙티브 필터 (Smart Filtering)

```typescript
// 질문 기반 필터
interface SmartFilter {
  questions: [
    {
      id: "purpose",
      question: "What's your use case?",
      options: [
        { label: "Image Classification", filter: { task: "classification" } },
        { label: "Object Detection", filter: { task: "detection" } },
        { label: "Segmentation", filter: { task: "segmentation" } }
      ]
    },
    {
      id: "environment",
      question: "Where will you deploy?",
      options: [
        { label: "Cloud (GPU)", filter: { tags: ["production", "accurate"] } },
        { label: "Edge Device", filter: { tags: ["mobile", "lightweight"] } },
        { label: "Desktop (CPU)", filter: { tags: ["efficient", "balanced"] } }
      ]
    },
    {
      id: "priority",
      question: "What matters most?",
      options: [
        { label: "Accuracy", sort: "accuracy", desc: true },
        { label: "Speed", sort: "speed", desc: true },
        { label: "Small Size", sort: "size", desc: false }
      ]
    }
  ];
}
```

**UI Flow**:
```
┌─────────────────────────────────────────────────┐
│  🎯 Find the Right Model                        │
├─────────────────────────────────────────────────┤
│                                                  │
│  What's your use case?                          │
│  ○ Image Classification                         │
│  ○ Object Detection                             │
│  ○ Segmentation                                 │
│                                                  │
│  Where will you deploy?                         │
│  ○ Cloud (GPU available)                        │
│  ○ Edge Device (limited resources)              │
│  ○ Desktop (CPU only)                           │
│                                                  │
│  What matters most?                             │
│  ○ Accuracy  ○ Speed  ○ Small Size              │
│                                                  │
│  [Show Recommended Models]                      │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│  📊 Recommended for You                         │
├─────────────────────────────────────────────────┤
│                                                  │
│  1️⃣ YOLOv11m (⭐ Best Match)                    │
│     Why: Balanced accuracy/speed for cloud GPU  │
│     [Select]  [Learn More]                      │
│                                                  │
│  2️⃣ YOLOv11n                                    │
│     Why: If you need faster inference           │
│     [Select]  [Learn More]                      │
│                                                  │
│  3️⃣ YOLOv11l                                    │
│     Why: If accuracy is critical                │
│     [Select]  [Learn More]                      │
│                                                  │
│  [See All Models]                               │
└─────────────────────────────────────────────────┘
```

### 3. 시각적 의사결정 도구

#### Decision Tree Visualization
```
Start
  │
  ├─ Classification?
  │   ├─ Mobile? → MobileNetV3/V4
  │   ├─ Accurate? → ViT, EfficientNetV2
  │   └─ Baseline? → ResNet-50
  │
  └─ Detection?
      ├─ Edge? → YOLOv11n
      ├─ Balanced? → YOLOv11m
      ├─ Accurate? → YOLOv11l
      └─ Custom objects? → YOLO-World
```

#### Performance Quadrant
```
        High Accuracy
            ↑
            │
    Slow    │    Fast
    ────────┼────────→
            │
            │
        Low Accuracy
```

모델들을 quadrant에 배치하여 시각화

### 4. 컨텍스트 도움말 (Tooltips)

모든 용어에 hover tooltip:
```typescript
const glossary = {
  "mAP": "Mean Average Precision - Primary metric for object detection",
  "Top-1 Accuracy": "Percentage where the model's #1 prediction is correct",
  "Top-5 Accuracy": "Percentage where correct answer is in top 5 predictions",
  "FPS": "Frames Per Second - How many images processed per second",
  "Params": "Number of trainable parameters - Roughly indicates model size",
  "Transfer Learning": "Using a pre-trained model as starting point",
  // ... more terms
};
```

### 5. Real-world Examples (실제 사례)

각 모델마다 실제 사용 사례:
```typescript
interface RealWorldExample {
  title: string;
  description: string;
  company?: string;
  metrics: {
    before?: string;
    after: string;
  };
  link?: string;
}

// ResNet-50 예시
const resnet50Examples = [
  {
    title: "Medical Image Classification",
    description: "Stanford University used ResNet-50 for pneumonia detection from chest X-rays",
    metrics: {
      after: "93% accuracy, comparable to radiologists"
    },
    link: "https://..."
  },
  {
    title: "E-commerce Product Categorization",
    description: "Major retailer uses ResNet-50 for automatic product tagging",
    metrics: {
      before: "Manual tagging: 100 products/hour",
      after: "Automated: 10,000 products/hour"
    }
  }
];
```

---

## 성공 지표

### Week 1 종료 시

**정량적**:
- [ ] P0 4개 모델 100% 동작
- [ ] P1 12개 모델 100% 동작
- [ ] P2 15개 모델 80% 이상 동작
- [ ] 가이드 시스템 완성도 90%

**정성적**:
- [ ] 모델 선택이 직관적
- [ ] 가이드 정보가 유용함
- [ ] 비교 기능이 효과적
- [ ] 추천 기능이 정확함

---

## 다음 단계 (Week 2)

Week 1 완료 후:
1. **검증 리포트**
   - 각 모델 성능 기록
   - 사용자 피드백 수집
   - 가이드 시스템 효과 분석

2. **Docker 분리 준비**
   - Week 1 경험 반영
   - 의존성 버전 확인
   - Platform SDK 설계 개선

---

*Document Version: 3.0*
*Created: 2025-10-30*
*Author: Vision AI Platform Team*
