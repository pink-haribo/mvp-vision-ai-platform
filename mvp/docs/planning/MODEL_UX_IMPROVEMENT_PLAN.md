# Model & Task UX Improvement Plan (v2)

**작성일**: 2025-10-31 (v2 수정)
**목적**: 모델/태스크 증가에 따른 UX 복잡도 완화 및 사용자 경험 개선

---

## 문제 정의

### 1. 모델 선택의 어려움
- **현상**: 모델이 많아지면서 사용자가 어떤 모델을 선택해야 할지 모름
- **원인**:
  - 같은 태스크에 여러 모델 존재 (ResNet-18/50, YOLOv8n/s/m, ViT 등)
  - 성능/속도 트레이드오프 정보 부족
  - Deprecated 모델과 최신 모델 구분 없음
  - 모델명 표시 불일치 (EoMT vs tue-mps/eomt-vit-large)
- **영향**:
  - 초보 사용자 이탈
  - 비효율적인 모델 선택 (구형 모델 사용)
  - 지원 부담 증가

### 2. 학습 vs 추론 전용 모델 혼재
- **현상**: 추론 전용 모델(Super-Resolution)을 학습 작업에서 선택 가능
- **원인**:
  - 모델 메타데이터에 `inference_only` 속성 없음
  - 추론 전용 도구(SR, Background Removal)가 학습 워크플로에 혼재
- **영향**:
  - 사용자가 SR 모델로 학습 시도 → 혼란
  - 학습 불가 모델인데 학습 UI 표시 (에폭 진행률 등)
  - "즉시 사용 가능한 도구"와 "학습 가능 모델" 구분 불명확

### 3. 테스트/추론 화면 레이아웃 비효율
- **현상**:
  - 이미지 업로드 공간이 낭비됨 (넓은 공간 차지)
  - 이미지 리스트가 너무 좁음 (사용성 저하)
  - 결과 카드에 이미지 리스트 중복 표시
  - 태스크별 코드 중복 (Classification, Detection, Segmentation 등)
- **원인**:
  - 레이아웃 설계 미흡
  - 태스크별로 독립적으로 개발
  - 공통 컴포넌트 부재
- **영향**:
  - 공간 활용도 낮음
  - 유지보수 어려움 (한 곳 수정하면 여러 곳 수정 필요)
  - 새 태스크 추가 시 개발 비용 증가

### 4. 추가 UX 문제점

#### 4.1 Instance/Semantic Segmentation 분리 과도
- 사용자 관점에서 두 개념 구분 불필요
- semantic_segmentation 하나로 통합 필요

#### 4.2 검증 메트릭 슬라이드 패널 불일치
- Confusion Matrix, Per-Class AP 클릭 시 나오는 슬라이드 패널
- 태스크별로 스타일 불일치
- 표준화 필요 (composition pattern 고려)

#### 4.3 학습 상태별 UI 표시 미흡
- `pending` 상태인데 계속 spinner 표시
- `completed` 상태인데 학습 버튼 활성화
- 상태에 맞는 적절한 UI 표시 필요

#### 4.4 학습 화면 핵심 정보 부족
- 제목에 작업 ID만 표시 (예: "Training Job #123")
- 모델명, 태스크명 등 핵심 정보 누락
- 압축적 정보 표시 필요

---

## 해결 방안

## Part 1: 모델 Deprecation & Metadata 시스템

### 1.1 모델 메타데이터 확장

**모델 레지스트리에 추가할 필드**:

```python
# mvp/training/model_registry/timm_models.py 예시
TIMM_MODEL_REGISTRY = {
    "resnet50": {
        # 기존 필드...
        "display_name": "ResNet-50",  # 사용자에게 표시되는 이름
        "model_id": "resnet50",        # 실제 모델 로딩에 사용
        "params": "25.6M",

        # 🆕 새로 추가되는 필드
        "status": "active",  # active | deprecated | experimental
        "inference_only": False,  # 학습 불가 여부
        "recommended": True,  # 추천 모델 (관리자가 수동 설정)
        "deprecation_info": None,

        # 🆕 모델 성능 레이블
        "performance_tier": "balanced",  # fast | balanced | accurate
        "use_case_tags": ["general", "production"],

        # 🆕 대체 모델 추천
        "alternatives": {
            "faster": "efficientnet_b0",
            "more_accurate": "resnet152"
        }
    },

    "resnet18": {
        # Deprecated 예시
        "status": "deprecated",
        "deprecation_info": {
            "reason": "ResNet-50 또는 EfficientNet 사용을 권장합니다.",
            "deprecated_since": "2025-10-01",
            "alternative": "efficientnet_b0"
        },
        "recommended": False
    }
}
```

**HuggingFace 모델 예시** (추론 전용):
```python
"caidas/swin2SR-classical-sr-x2-64": {
    "display_name": "Swin2SR 2x",  # 사용자에게 표시
    "model_id": "caidas/swin2SR-classical-sr-x2-64",  # HF 다운로드 경로
    "status": "experimental",  # ⭐ inference_only는 experimental로 흡수
    "inference_only": True,
    "recommended": True,
    "performance_tier": "accurate",
    "use_case_tags": ["image-tools", "quality-enhancement"]
}
```

**핵심 원칙**:
- `display_name`: 사용자 UI에 표시 (간결함)
- `model_id`: 실제 모델 로딩에 사용 (정확한 경로)
- 모든 화면에서 `display_name` 우선 표시
- `model_id`는 디버깅/로그에서만 사용

### 1.2 모델명 표시 일관성

**문제**:
- 일부 화면: "EoMT" (display_name)
- 일부 화면: "tue-mps/eomt-vit-large" (model_id)

**해결**:
```tsx
// ❌ 잘못된 방식
<div>{job.model_name}</div>  // "tue-mps/eomt-vit-large"

// ✅ 올바른 방식
<div>{getModelDisplayName(job.framework, job.model_name)}</div>  // "EoMT"

// Helper 함수
function getModelDisplayName(framework: string, modelId: string): string {
  const modelInfo = getModelInfo(framework, modelId);
  return modelInfo?.display_name || modelId;  // fallback to model_id
}
```

**적용 위치**:
- 학습 작업 제목
- 모델 선택 드롭다운
- 학습 진행 화면
- 결과 요약

### 1.3 프론트엔드 필터링

**사용자 모델 선택 UI**:

```tsx
// 기본: Recommended 모델만 표시
<ModelSelector
  defaultFilter="recommended"
  showDeprecated={false}
/>

// Advanced 토글
<Toggle>
  [ ] 모든 모델 표시 (deprecated 포함)
</Toggle>
```

**표시 우선순위**:
1. **Recommended + Active** (기본 표시)
2. Active (non-recommended)
3. **Experimental** (경고 뱃지 + inference_only 포함)
4. **Deprecated** (숨김, 토글로 표시 가능)

---

## Part 2: 이미지 도구 & 추론 전용 모델 분리

### 2.1 사이드바 재구성

**기존 문제**:
- SR, Background Removal 같은 "즉시 사용 도구"가 학습 워크플로에 혼재
- "학습"과 "도구" 개념 구분 불명확

**개선안**:

```
현재 사이드바:
┌────────────────────┐
│ 📊 대시보드         │
│ 💬 채팅            │
│ ➕ 프로젝트         │
│ 📂 내 프로젝트      │
│ 👤 유저 정보       │
└────────────────────┘

개선된 사이드바:
┌────────────────────┐
│ 📊 대시보드         │
│ 💬 채팅            │
│                    │
│ 🛠️  이미지 도구     │  ← 🆕 추가 (최상단 배치)
│ ➕ 새 학습 작업     │  ← 기존 "프로젝트" 이름 변경
│ 📂 내 프로젝트      │
│ 👤 유저 정보       │
└────────────────────┘
```

**"이미지 도구" 클릭 시 흐름**:

```
Step 1: 도구 선택 화면
┌────────────────────────────────────────────┐
│ 🛠️  이미지 도구                             │
├────────────────────────────────────────────┤
│                                            │
│  사용 가능한 도구:                          │
│                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ 🔍 2x   │  │ 🔍 4x   │  │ 🎨 배경  │ │
│  │ 화질향상 │  │ 화질향상 │  │ 제거    │ │
│  │         │  │         │  │         │ │
│  │ Swin2SR │  │ Swin2SR │  │ U2-Net  │ │
│  └──────────┘  └──────────┘  └──────────┘ │
│                                            │
│  ┌──────────┐  ┌──────────┐               │
│  │ 🌈 색상  │  │ ✨ 노이즈│               │
│  │ 보정    │  │ 제거    │  [더 추가 예정]│
│  └──────────┘  └──────────┘               │
└────────────────────────────────────────────┘

Step 2: 도구 선택 후 → 즉시 추론 화면
┌────────────────────────────────────────────┐
│ 🔍 2x 화질 향상 (Swin2SR)                   │
├────────────────────────────────────────────┤
│                                            │
│  [이미지 업로드 영역]                       │
│  📷 이미지를 드래그하거나 클릭하여 업로드     │
│                                            │
│  [처리하기]                                │
│                                            │
│  [결과]                                    │
│  Before / After 비교                        │
│  [다운로드]  [다른 이미지로 시도]            │
└────────────────────────────────────────────┘
```

**핵심 특징**:
- 학습 과정 없이 **즉시 사용 가능**
- 모델 선택 불필요 (도구마다 고정 모델)
- 간단한 UX: 업로드 → 처리 → 다운로드

### 2.2 추론 Only 모델의 학습 작업 처리

**시나리오**: 사용자가 "새 학습 작업"에서 추론 only 모델(예: SegFormer) 선택

**현재 문제**:
- 학습 버튼, 에폭 진행률 등 학습 UI 표시
- 학습 실행 시 에러

**개선안**:

```
학습 작업 생성 시 (inference_only=true 모델):
┌────────────────────────────────────────────┐
│ 📋 학습 작업 생성                           │
├────────────────────────────────────────────┤
│ 프레임워크: transformers                    │
│ 모델: SegFormer-B0 (ADE20K) ⚠️ 추론 전용   │
│                                            │
│ ❌ 데이터셋 선택 (비활성화)                 │
│                                            │
│ ✅ Pretrained Weight 선택:                 │
│   ○ ADE20K (150 classes)                   │
│   ○ Cityscapes (19 classes)                │
│   ○ COCO-Stuff (171 classes)               │
│                                            │
│ ❌ 학습 설정 (비활성화)                     │
│                                            │
│ [작업 생성]                                │
└────────────────────────────────────────────┘

학습 작업 상세 화면:
┌────────────────────────────────────────────┐
│ 📊 SegFormer-B0 - Semantic Segmentation    │
│    ⚠️  이 모델은 추론 전용입니다              │
├────────────────────────────────────────────┤
│ 탭 메뉴:                                   │
│ [테스트/추론] (기본 활성)                   │
│ [학습] (비활성화)                          │
│ [검증] (비활성화)                          │
│ [학습 설정] (비활성화)                      │
│ [로그] (비활성화)                          │
├────────────────────────────────────────────┤
│                                            │
│ ℹ️  안내:                                  │
│ 이 모델은 학습이 지원되지 않습니다.          │
│ Pretrained weight(ADE20K)로 즉시 추론 가능합니다. │
│                                            │
│ [테스트/추론 시작하기]                      │
└────────────────────────────────────────────┘
```

**학습 상태 표시 개선**:

```tsx
// 학습 작업 헤더
function TrainingJobHeader({ job }: Props) {
  const isInferenceOnly = job.inference_only;
  const status = job.status;  // pending | running | completed | failed | canceled

  return (
    <div>
      {/* 핵심 정보 압축 표시 */}
      <h1>
        {getModelDisplayName(job.framework, job.model_name)}
        {" - "}
        {getTaskDisplayName(job.task_type)}
      </h1>

      {/* 추론 전용 뱃지 */}
      {isInferenceOnly && (
        <Badge color="blue">추론 전용</Badge>
      )}

      {/* 상태별 UI */}
      {!isInferenceOnly && (
        <>
          {status === 'pending' && (
            <div className="text-gray-500">
              ⏳ 학습 대기 중...
            </div>
          )}
          {status === 'running' && (
            <div className="flex items-center gap-2">
              <Spinner />
              <span>Epoch {job.current_epoch}/{job.num_epochs}</span>
            </div>
          )}
          {status === 'completed' && (
            <div className="text-green-600">
              ✅ 학습 완료
            </div>
          )}
        </>
      )}
    </div>
  );
}
```

---

## Part 3: 통합 추론 패널 재설계

### 3.1 레이아웃 개선

**현재 레이아웃 (문제)**:
```
┌──────────────────────────────────────┐
│ [이미지 업로드 영역 - 넓음]           │  ← 공간 낭비
│ [이미지 리스트 - 좁음]                │  ← 사용성 저하
│ [결과 카드]                          │
│   - 이미지 리스트 (중복)              │  ← 중복
│   - 결과 정보                        │
└──────────────────────────────────────┘
```

**개선된 레이아웃**:
```
┌─────────────────────────────────────────────────┐
│ Row 1: 업로드/리스트 + 설정                      │
├─────────────────────────────────────────────────┤
│                                                 │
│ Col 1 (6/12)                Col 2 (6/12)        │
│ ┌─────────────────────┐    ┌─────────────────┐ │
│ │ 📷 이미지 업로드     │    │ ⚙️  설정         │ │
│ │                     │    │                 │ │
│ │ (업로드 전)         │    │ 모델: ViT-Base  │ │
│ │ 드래그 또는 클릭    │    │ Batch: 32       │ │
│ │ [+] 선택            │    │                 │ │
│ │                     │    │ [추론 시작]      │ │
│ └─────────────────────┘    └─────────────────┘ │
│                                                 │
│ (업로드 후 → 자동으로 리스트로 전환)             │
│ ┌─────────────────────┐    ┌─────────────────┐ │
│ │ 🖼️  이미지 리스트    │    │ ⚙️  설정         │ │
│ │ [🗑️] [➕]          │    │ (동일)          │ │
│ │                     │    │                 │ │
│ │ ┌─┐ ┌─┐ ┌─┐ ┌─┐   │    │                 │ │
│ │ │📷│ │📷│ │📷│ │📷│   │    │                 │ │
│ │ └─┘ └─┘ └─┘ └─┘   │    │                 │ │
│ │ ┌─┐ ┌─┐ ┌─┐       │    │                 │ │
│ │ │📷│ │📷│ │📷│       │    │                 │ │
│ │ └─┘ └─┘ └─┘       │    │                 │ │
│ │                     │    │                 │ │
│ │ (드래그로 추가 가능) │    │                 │ │
│ └─────────────────────┘    └─────────────────┘ │
├─────────────────────────────────────────────────┤
│ Row 2: 결과 카드                                 │
├─────────────────────────────────────────────────┤
│ ┌───────────────────────────────────────────┐   │
│ │ 📊 추론 결과 (7/7 완료)                   │   │
│ ├───────────────────────────────────────────┤   │
│ │                                           │   │
│ │ ❌ 이미지 리스트 제거 (위에 있음)          │   │
│ │                                           │   │
│ │ ✅ 태스크별 뷰어만 표시:                   │   │
│ │                                           │   │
│ │ [Classification]                          │   │
│ │ ┌────┬────┬────┐                         │   │
│ │ │ 🐱  │ 🐶  │ 🐦  │ Top-3 Predictions     │   │
│ │ │ Cat│ Dog│ Bird│                         │   │
│ │ │ 92%│ 5% │ 2% │                         │   │
│ │ └────┴────┴────┘                         │   │
│ │                                           │   │
│ │ [Object Detection]                        │   │
│ │ ┌─────────────────────┐                  │   │
│ │ │ 🖼️  BBox Overlay    │                  │   │
│ │ │ [✓] 박스 표시       │  ← 토글 옵션      │   │
│ │ │                     │                  │   │
│ │ │  🚗  🚶  🚦         │                  │   │
│ │ └─────────────────────┘                  │   │
│ │ Detected: 3 objects                       │   │
│ │                                           │   │
│ │ [Segmentation]                            │   │
│ │ ┌─────────────────────┐                  │   │
│ │ │ 🎨 Mask Overlay     │                  │   │
│ │ │ [✓] 마스크 표시     │  ← 토글 옵션      │   │
│ │ │                     │                  │   │
│ │ │  (colorized mask)   │                  │   │
│ │ └─────────────────────┘                  │   │
│ │ Classes: 5 detected                       │   │
│ │                                           │   │
│ │ ❌ SR은 이미지 도구로 이동 (여기 제거)     │   │
│ │                                           │   │
│ └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

**핵심 개선사항**:
1. **업로드/리스트 통합** (6 columns)
   - 업로드 전: 드래그앤드롭 영역
   - 업로드 후: 자동으로 이미지 리스트 전환
   - 상단 아이콘: 🗑️ (삭제), ➕ (추가)
   - 리스트에 드래그앤드롭으로 추가 가능

2. **설정 영역** (6 columns)
   - 모델 선택, 배치 크기 등
   - 추론 시작 버튼

3. **결과 카드**
   - 이미지 리스트 제거 (위에 있으므로 중복 제거)
   - 태스크별 뷰어만 표시
   - Classification: 기존 유지
   - Detection: BBox overlay (on/off 토글)
   - Segmentation: Mask overlay (on/off 토글)
   - **SR은 이미지 도구로 이동** (이 화면에서 제거)

### 3.2 Composition Pattern 적용

**파일 구조**:
```
frontend/components/inference/
├── InferencePanel.tsx           # 메인 컨테이너
├── InferenceLayout.tsx          # 레이아웃 (업로드+설정, 결과)
├── ImageUploadList.tsx          # 업로드/리스트 통합 컴포넌트
├── InferenceSettings.tsx        # 설정 영역
├── InferenceResults.tsx         # 결과 디스패처
│
├── renderers/                   # 태스크별 렌더러
│   ├── ClassificationRenderer.tsx
│   ├── DetectionRenderer.tsx
│   ├── SegmentationRenderer.tsx
│   └── (SuperResolutionRenderer.tsx → 이미지 도구로 이동)
│
├── shared/                      # 공통 컴포넌트
│   ├── ResultCard.tsx
│   ├── ToggleOverlay.tsx       # BBox/Mask 토글
│   └── MetricsDisplay.tsx
│
└── hooks/
    ├── useInference.ts          # 추론 로직
    └── useTaskConfig.ts         # 태스크별 설정
```

**ImageUploadList.tsx** (핵심 신규 컴포넌트):
```tsx
export function ImageUploadList({ images, onAdd, onRemove }: Props) {
  const hasImages = images.length > 0;

  return (
    <div className="col-span-6">
      {!hasImages ? (
        // 업로드 전: 드래그앤드롭 영역
        <UploadDropzone onUpload={onAdd} />
      ) : (
        // 업로드 후: 이미지 리스트
        <div>
          <div className="flex justify-between mb-2">
            <h3>이미지 리스트 ({images.length})</h3>
            <div className="flex gap-2">
              <IconButton icon="trash" onClick={() => onRemove('all')} />
              <IconButton icon="plus" onClick={triggerFileInput} />
            </div>
          </div>

          <ImageGrid
            images={images}
            onDrop={onAdd}  // 드래그앤드롭으로 추가
            onRemove={onRemove}
          />
        </div>
      )}
    </div>
  );
}
```

**태스크별 렌더러 예시**:

```tsx
// DetectionRenderer.tsx
export function DetectionRenderer({ result }: Props) {
  const [showBBox, setShowBBox] = useState(true);

  return (
    <div>
      <ToggleOverlay
        label="박스 표시"
        checked={showBBox}
        onChange={setShowBBox}
      />

      <ImageViewer
        src={result.image_url}
        overlay={showBBox ? result.bbox_overlay_url : null}
      />

      <MetricsDisplay>
        Detected: {result.num_objects} objects
      </MetricsDisplay>
    </div>
  );
}

// SegmentationRenderer.tsx
export function SegmentationRenderer({ result }: Props) {
  const [showMask, setShowMask] = useState(true);

  return (
    <div>
      <ToggleOverlay
        label="마스크 표시"
        checked={showMask}
        onChange={setShowMask}
      />

      <ImageViewer
        src={result.image_url}
        overlay={showMask ? result.mask_overlay_url : null}
      />

      <MetricsDisplay>
        Classes: {result.unique_classes} detected
      </MetricsDisplay>
    </div>
  );
}
```

### 3.3 태스크 통합

**Instance/Semantic Segmentation 통합**:

```python
# 기존 (과도한 분리)
TaskType.INSTANCE_SEGMENTATION
TaskType.SEMANTIC_SEGMENTATION

# 개선 (통합)
TaskType.SEGMENTATION  # semantic segmentation만 지원

# 이유:
# - 사용자 관점에서 구분 불필요
# - UI/UX 단순화
# - Instance segmentation은 추후 필요 시 추가
```

---

## Part 4: 추가 개선사항

### 4.1 검증 메트릭 슬라이드 패널 표준화

**현재 문제**:
- Confusion Matrix, Per-Class AP 클릭 시 슬라이드 패널
- 태스크별로 구현 방식 다름
- 스타일 불일치

**개선안**:

```tsx
// 공통 슬라이드 패널 컴포넌트
<ValidationSlidePanel
  isOpen={isOpen}
  onClose={onClose}
  title="Confusion Matrix"
  width="xl"
>
  {/* 태스크별 렌더러 */}
  <ValidationRenderer type={metricType} data={data} />
</ValidationSlidePanel>

// 렌더러 레지스트리
const VALIDATION_RENDERERS = {
  confusion_matrix: ConfusionMatrixRenderer,
  per_class_ap: PerClassAPRenderer,
  iou_chart: IoUChartRenderer,
  loss_curve: LossCurveRenderer,
};
```

**적용**:
- Composition pattern 사용 (추론 패널과 동일 패턴)
- 태스크별 스타일 통일
- 코드 재사용

### 4.2 학습 화면 핵심 정보 표시

**현재**:
```
제목: Training Job #123
```

**개선**:
```
제목: ResNet-50 - Image Classification
부제: Job #123 | Created 2025-10-31

또는 더 압축:
ResNet-50 • Image Classification • Job #123
```

**구현**:
```tsx
function TrainingJobTitle({ job }: Props) {
  const modelName = getModelDisplayName(job.framework, job.model_name);
  const taskName = getTaskDisplayName(job.task_type);

  return (
    <div>
      <h1 className="text-2xl font-bold">
        {modelName} - {taskName}
      </h1>
      <div className="text-sm text-gray-500">
        Job #{job.id} | Created {formatDate(job.created_at)}
      </div>
    </div>
  );
}
```

### 4.3 학습 상태별 UI 일관성

**상태별 표시 규칙**:

| 상태 | 학습 버튼 | Spinner | 에폭 표시 | 탭 활성화 |
|------|----------|---------|----------|-----------|
| `pending` | 비활성화 | ❌ (대기 중 텍스트만) | ❌ | 일부 |
| `running` | "중지" 버튼 | ✅ | ✅ | 모두 |
| `completed` | 비활성화 | ❌ | ✅ (최종) | 모두 |
| `failed` | 비활성화 | ❌ | ❌ | 일부 |
| `canceled` | 비활성화 | ❌ | ❌ | 일부 |
| `inference_only` | 비활성화 | ❌ | ❌ | 테스트만 |

**구현**:
```tsx
function TrainingControls({ job }: Props) {
  const { status, inference_only } = job;

  if (inference_only) {
    return <InferenceOnlyBanner />;
  }

  switch (status) {
    case 'pending':
      return (
        <div className="text-gray-500">
          ⏳ 학습 대기 중...
        </div>
      );

    case 'running':
      return (
        <div className="flex items-center gap-4">
          <Spinner />
          <span>Epoch {job.current_epoch}/{job.num_epochs}</span>
          <Button onClick={handleStop}>중지</Button>
        </div>
      );

    case 'completed':
      return (
        <div className="text-green-600">
          ✅ 학습 완료 (Epoch {job.num_epochs})
        </div>
      );

    // ... 기타 상태
  }
}
```

---

## 구현 우선순위 (재조정)

### Phase 1: 모델 메타데이터 & 표시 일관성 (2-3일)
- [ ] 모델 레지스트리에 `status`, `inference_only`, `recommended` 필드 추가
- [ ] `display_name` vs `model_id` 구분 명확화
- [ ] 모든 화면에서 `display_name` 사용 (helper 함수)
- [ ] 학습 화면 제목에 핵심 정보 표시

### Phase 2: 이미지 도구 섹션 (2-3일)
- [ ] 사이드바에 "이미지 도구" 추가 (최상단)
- [ ] 도구 선택 화면 구현
- [ ] SR 도구 페이지 구현 (간단한 업로드-처리-다운로드)
- [ ] `inference_only` 모델 학습 작업 특별 처리

### Phase 3: 통합 추론 패널 레이아웃 (3-4일)
- [ ] `ImageUploadList` 컴포넌트 (업로드/리스트 통합)
- [ ] 레이아웃 변경: [6+6] → [결과]
- [ ] 이미지 리스트에 드래그앤드롭 추가
- [ ] 결과 카드에서 이미지 리스트 제거

### Phase 4: Composition Pattern 적용 (2-3일)
- [ ] 태스크별 렌더러 분리 (Classification, Detection, Segmentation)
- [ ] `useTaskConfig` 훅
- [ ] BBox/Mask 토글 컴포넌트
- [ ] Instance/Semantic segmentation 통합

### Phase 5: 학습 상태 UI 개선 (1-2일)
- [ ] 상태별 UI 표시 로직 통일
- [ ] `pending` 상태 spinner 제거
- [ ] 탭 활성화/비활성화 로직

### Phase 6: 검증 메트릭 패널 표준화 (1-2일)
- [ ] `ValidationSlidePanel` 공통 컴포넌트
- [ ] 메트릭별 렌더러 분리
- [ ] 태스크별 스타일 통일

### Phase 7: 관리자 인터페이스 (2-3일)
- [ ] `/admin/models` 페이지
- [ ] Deprecate/Recommend API
- [ ] 프론트엔드 필터링 (deprecated 숨김/표시 토글)

---

## 예상 효과

### 사용자 경험
- ✅ 명확한 "도구" vs "학습" 구분
- ✅ 모델명 표시 일관성 (혼란 감소)
- ✅ 학습 화면에서 핵심 정보 즉시 파악
- ✅ 추론 화면 공간 활용도 증가
- ✅ 일관된 UI/UX (모든 태스크)

### 개발 효율
- ✅ Composition pattern으로 코드 재사용
- ✅ 새 태스크 추가 용이 (렌더러만 추가)
- ✅ 유지보수 비용 감소
- ✅ 태스크 통합으로 복잡도 감소

### 운영 효율
- ✅ 관리자 모델 큐레이션 가능
- ✅ 추론 전용 도구로 사용자 유입 증가
- ✅ 학습 상태별 적절한 UI로 지원 요청 감소

---

## 주요 의사결정 사항 (확정)

### ✅ Q1: 추론 전용 모델을 별도 섹션으로?
**결정**: 예. "이미지 도구" 섹션 신규 생성
**이유**:
- "도구"와 "학습" 개념 명확히 구분
- 즉시 사용 가능한 기능으로 포지셔닝
- 사용자 진입 장벽 낮춤

### ✅ Q2: experimental에 inference_only 흡수?
**결정**: 예. `status: "experimental"` + `inference_only: true`
**이유**:
- 추론 전용 모델은 대부분 최신 모델
- 별도 상태보다 experimental로 관리가 직관적

### ✅ Q3: Instance/Semantic segmentation 통합?
**결정**: 예. `TaskType.SEGMENTATION` 하나로 통합 (semantic만 지원)
**이유**:
- 사용자 관점에서 구분 불필요
- UI/UX 복잡도 감소

### ✅ Q4: 레이아웃 개선?
**결정**: 예. [업로드/리스트 6] [설정 6] → [결과]
**이유**:
- 공간 활용도 증가
- 이미지 리스트 사용성 개선
- 결과 카드 중복 제거

### ✅ Q5: Composition Pattern 사용?
**결정**: 예. 태스크별 렌더러 완전 분리
**이유**:
- 향후 태스크 증가 대비
- 코드 재사용성 증가
- 유지보수 용이

---

## 다음 단계

1. **Phase 1 즉시 시작** (모델 메타데이터)
   - 영향 범위 작음
   - 빠른 가시적 효과
   - 다른 Phase의 기반

2. **Phase 2-3 병행** (이미지 도구 + 추론 패널)
   - 가장 큰 UX 개선
   - 독립적 작업 가능

3. **Phase 4-6 순차 진행** (Composition, 상태 UI, 검증 패널)
   - Phase 3 완료 후 시작
   - 점진적 개선

4. **Phase 7 최종** (관리자 UI)
   - 급하지 않음
   - 다른 Phase 완료 후

---

**문서 작성**: Claude Code (v2 수정)
**주요 변경**: 이미지 도구 섹션, 레이아웃 재설계, 태스크 통합, 상태 UI 개선
**검토 필요**: 사이드바 순서, 레이아웃 상세, 우선순위
