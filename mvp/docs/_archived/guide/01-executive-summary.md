# 1. Executive Summary

**목적**: Vision AI Training Platform의 전체 비전, 핵심 가치, 설계 원칙을 이해하고 플랫폼의 차별점을 파악합니다.

**대상 독자**: 모든 팀원 (특히 새 팀원, PM, Team Lead)

---

## 1.1 플랫폼 비전과 핵심 가치

### 비전 (Vision)

> **"자연어로 대화하듯 Vision AI 모델을 학습하는 플랫폼"**

Vision AI Training Platform은 개발자와 연구자가 복잡한 설정 파일이나 스크립트 작성 없이, 자연어 대화를 통해 컴퓨터 비전 모델을 학습할 수 있는 플랫폼입니다.

### 해결하는 문제

**기존 문제점**:
1. **높은 진입 장벽**: 각 ML 프레임워크마다 다른 API와 설정 방식
2. **의존성 충돌**: timm, Ultralytics, HuggingFace를 동일 환경에 설치 시 버전 충돌
3. **반복적인 보일러플레이트**: 학습 루프, 메트릭 로깅, 체크포인트 저장 등 반복 작업
4. **모니터링 복잡성**: 학습 현황을 실시간으로 파악하기 어려움
5. **확장성 부족**: 새로운 모델이나 프레임워크 추가 시 전체 코드 수정 필요

**우리의 솔루션**:
1. **자연어 인터페이스**: LLM 기반 의도 파싱으로 직관적인 설정
2. **플러그인 아키텍처**: Adapter Pattern으로 프레임워크 독립성 확보
3. **의존성 격리**: Docker 기반 프레임워크별 독립 실행 환경
4. **통합 모니터링**: MLflow + Prometheus + Grafana 자동 연동
5. **원클릭 추론 API**: 학습 완료 즉시 REST API 생성

### 핵심 가치

```
🗣️ Simplicity (단순성)
   → 자연어로 모델 설정, 복잡한 config 불필요

🔌 Extensibility (확장성)
   → 새 프레임워크/모델을 Adapter 추가로 쉽게 통합

🔒 Isolation (격리성)
   → Docker 기반 의존성 격리로 충돌 없음

📊 Observability (관찰성)
   → 실시간 메트릭, 로그, 시각화 자동 제공

🚀 Productivity (생산성)
   → 학습에서 배포까지 end-to-end 자동화
```

---

## 1.2 주요 설계 원칙

### 원칙 1: 모듈화와 플러그인 아키텍처

**Adapter Pattern을 통한 프레임워크 통합**

플랫폼은 `TrainingAdapter`라는 공통 인터페이스를 정의하고, 각 ML 프레임워크(timm, Ultralytics, HuggingFace)가 이를 구현합니다.

**이점**:
- ✅ 새 프레임워크 추가 시 기존 코드 수정 불필요
- ✅ 프레임워크별 구현을 독립적으로 유지보수
- ✅ 모든 프레임워크가 동일한 방식으로 호출됨

**예시**:
```python
# 공통 인터페이스
class TrainingAdapter(ABC):
    def prepare_model(self) -> None
    def prepare_dataset(self) -> None
    def train_epoch(epoch: int) -> MetricsResult
    def validate(epoch: int) -> MetricsResult
    def save_checkpoint(epoch: int) -> str

# 구현
TimmAdapter(TrainingAdapter)         # timm 프레임워크
UltralyticsAdapter(TrainingAdapter)  # YOLO 프레임워크
TransformersAdapter(TrainingAdapter) # HuggingFace 프레임워크
```

### 원칙 2: 프레임워크 독립성

**문제**: 각 프레임워크마다 다른 API, 데이터 형식, 메트릭 정의

**해결**: 표준화된 데이터 클래스로 추상화

```python
# 표준 메트릭 형식
@dataclass
class MetricsResult:
    epoch: int
    train_loss: float
    val_loss: float
    metrics: Dict[str, float]  # accuracy, mAP, IoU 등

# 표준 추론 결과 형식
@dataclass
class InferenceResult:
    image_path: str
    task_type: TaskType
    predicted_label: Optional[str]         # Classification
    predicted_boxes: Optional[List[Dict]]  # Detection
    predicted_mask: Optional[Any]          # Segmentation
    inference_time_ms: float
```

**이점**:
- ✅ Frontend가 프레임워크를 의식하지 않고 동일한 UI로 표시
- ✅ Backend가 프레임워크별 예외 처리 불필요
- ✅ 메트릭 비교 및 분석 용이

### 원칙 3: 의존성 격리 (Docker Image Separation)

**문제**:
- timm은 PyTorch 2.0 필요
- Ultralytics는 PyTorch 2.1+ 필요
- HuggingFace는 transformers 4.30+ 필요
- 모든 의존성을 한 환경에 설치하면 충돌 발생

**해결**: 프레임워크별 독립 Docker 이미지

```
📦 vision-platform-base (공통 SDK)
   ├─ Platform SDK, MLflow, Database 클라이언트
   │
   ├─▶ 📦 vision-platform-timm
   │     └─ timm 0.9.x + PyTorch 2.0 + torchvision
   │
   ├─▶ 📦 vision-platform-ultralytics
   │     └─ ultralytics 8.1.x + PyTorch 2.1 + YOLO deps
   │
   └─▶ 📦 vision-platform-huggingface
         └─ transformers 4.30+ + accelerate + datasets
```

**이점**:
- ✅ 각 프레임워크가 최적의 의존성 버전 사용
- ✅ 한 프레임워크 업데이트가 다른 프레임워크에 영향 없음
- ✅ 개발자가 새 모델 추가 시 자유롭게 라이브러리 버전 선택
- ✅ CI/CD에서 프레임워크별 개별 테스트 가능

### 원칙 4: 이중 실행 모드 (Dual Execution Mode)

**Subprocess Mode** (로컬 개발):
- 로컬 venv에서 직접 실행
- 빠른 디버깅 및 개발
- Docker 없이도 작동

**Docker Mode** (프로덕션):
- Docker 컨테이너로 실행
- 의존성 격리 보장
- 프로덕션 환경과 동일

**자동 선택**:
```python
class TrainingManager:
    def _detect_execution_mode(self) -> ExecutionMode:
        # Docker 사용 가능 시 → Docker Mode
        # Docker 없으면 → Subprocess Mode
        if docker_available:
            return ExecutionMode.DOCKER
        else:
            return ExecutionMode.SUBPROCESS
```

**이점**:
- ✅ 로컬 개발 시 빠른 iteration
- ✅ 프로덕션 환경에서 안정성 보장
- ✅ 환경 변수로 강제 모드 선택 가능

### 원칙 5: 관찰성 우선 (Observability First)

**자동 통합된 모니터링 스택**:

1. **MLflow**: 실험 추적 및 모델 버전 관리
2. **Prometheus**: 메트릭 수집 (loss, accuracy, GPU 사용량)
3. **Grafana**: 실시간 대시보드
4. **Database**: 구조화된 메트릭 저장 (검색, 비교)
5. **Stdout Logging**: `[METRICS]` 태그로 실시간 출력

**Callbacks 시스템**:
```python
class TrainingCallbacks:
    def on_train_begin(self, config: Dict)
        # MLflow Run 생성

    def on_epoch_end(self, epoch: int, metrics: Dict)
        # MLflow + DB + Prometheus에 자동 로깅

    def on_train_end(self, final_metrics: Dict)
        # MLflow Run 종료, 최종 결과 저장
```

**이점**:
- ✅ Adapter 구현자는 Callbacks만 호출하면 자동 로깅
- ✅ 모든 학습 작업이 동일한 방식으로 추적됨
- ✅ 실험 비교 및 재현성 보장

---

## 1.3 기술 스택 요약

### Frontend

| 영역 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **Framework** | Next.js | 14.x | React 기반 SSR/SSG |
| **Language** | TypeScript | 5.x | 타입 안정성 |
| **Styling** | Tailwind CSS | 3.x | Utility-first CSS |
| **State Management** | Zustand | 4.x | 가벼운 전역 상태 |
| **Server State** | React Query | 5.x | API 캐싱 및 동기화 |
| **UI Components** | shadcn/ui | - | Radix UI 기반 컴포넌트 |

### Backend

| 영역 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **Framework** | FastAPI | 0.109+ | Python 웹 프레임워크 |
| **Language** | Python | 3.11+ | 백엔드 언어 |
| **Database** | SQLite | 3.x | 로컬 개발용 (MVP) |
| **ORM** | SQLAlchemy | 2.0+ | Database 모델링 |
| **Validation** | Pydantic | 2.x | 데이터 검증 |
| **LLM Integration** | LangChain | 0.1+ | LLM 추상화 레이어 |
| **LLM Provider** | Google Gemini | 1.5 | 자연어 의도 파싱 |

### Training Infrastructure

| 영역 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **Classification** | timm | 0.9.x | PyTorch Image Models |
| **Detection/Segmentation** | Ultralytics | 8.1.x | YOLOv8/v9 |
| **Vision-Language** | HuggingFace Transformers | 4.30+ | ViT, DETR, TrOCR 등 |
| **Deep Learning** | PyTorch | 2.0+ | 딥러닝 프레임워크 |
| **Experiment Tracking** | MLflow | 2.10+ | 실험 추적 및 모델 저장 |

### Infrastructure & DevOps

| 영역 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **Containerization** | Docker | 24.x | 의존성 격리 |
| **Monitoring** | Prometheus | 2.x | 메트릭 수집 |
| **Visualization** | Grafana | 10.x | 대시보드 |
| **Orchestration (향후)** | Kubernetes | 1.28+ | 컨테이너 오케스트레이션 |
| **Workflow (향후)** | Temporal | 1.22+ | 학습 워크플로우 관리 |

---

## 1.4 현재 구현 상태 (MVP Phase)

### 지원 프레임워크

| 프레임워크 | 상태 | 지원 태스크 | 대표 모델 |
|------------|------|-------------|-----------|
| **timm** | ✅ 구현 완료 | Image Classification | ResNet-18/50, EfficientNet-B0 |
| **Ultralytics** | ✅ 구현 완료 | Object Detection, Instance Segmentation, Pose Estimation, Semantic Segmentation | YOLOv8n/s/m, YOLOv8-seg, YOLOv8-pose |
| **HuggingFace** | 🚧 부분 구현 | Image Classification, Super-Resolution | ViT, DINOv2, SwinIR, Real-ESRGAN |

### 지원 태스크 (Task Types)

| 태스크 | 지원 프레임워크 | 상태 | 메트릭 |
|--------|----------------|------|--------|
| **Image Classification** | timm, HuggingFace | ✅ 완료 | accuracy, top5_accuracy, loss |
| **Object Detection** | Ultralytics | ✅ 완료 | mAP50, mAP50-95, precision, recall |
| **Instance Segmentation** | Ultralytics | ✅ 완료 | mAP50, mAP50-95, mask IoU |
| **Semantic Segmentation** | Ultralytics | ✅ 완료 | mIoU, pixel accuracy |
| **Pose Estimation** | Ultralytics | ✅ 완료 | PCK (Percentage of Correct Keypoints) |
| **Super-Resolution** | HuggingFace | ✅ 완료 | PSNR, SSIM |
| **Depth Estimation** | HuggingFace | 🚧 개발 중 | - |
| **OCR** | HuggingFace | ⏳ 계획됨 | CER, WER |
| **Image Captioning** | HuggingFace | ⏳ 계획됨 | BLEU, CIDEr |

### 구현된 핵심 기능

#### ✅ Backend
- [x] FastAPI 기반 REST API
- [x] SQLite + SQLAlchemy ORM
- [x] TrainingManager (Subprocess + Docker 이중 모드)
- [x] LLM 기반 자연어 대화 (ConversationManager)
- [x] Dataset 자동 분석 (DatasetAnalyzer)
- [x] 실시간 로그 스트리밍 및 메트릭 파싱
- [x] MLflow Integration (자동 실험 추적)
- [x] Prometheus 메트릭 Export

#### ✅ Frontend
- [x] Next.js 14 + TypeScript
- [x] Chat 인터페이스 (자연어 대화)
- [x] Training Dashboard (실시간 현황)
- [x] Model 선택 및 설정 UI
- [x] 학습 메트릭 시각화 (차트, 테이블)
- [x] MLflow 실험 내장 뷰어
- [x] Grafana 대시보드 임베딩
- [x] Test Inference Panel (추론 테스트)

#### ✅ Training Infrastructure
- [x] Adapter Pattern 기반 아키텍처
- [x] TimmAdapter (ResNet, EfficientNet)
- [x] UltralyticsAdapter (YOLOv8 detection, segmentation, pose)
- [x] HuggingFaceAdapter (ViT, SwinIR, Real-ESRGAN)
- [x] TrainingCallbacks (MLflow, DB 자동 로깅)
- [x] Checkpoint 관리 (best model, periodic save)
- [x] Inference System (단일/배치 추론)
- [x] Validation Metrics (per-class, confusion matrix, PR curves)

#### ✅ Docker Isolation
- [x] Base Image (공통 SDK)
- [x] timm Image
- [x] Ultralytics Image
- [x] HuggingFace Image
- [x] 자동 실행 모드 감지
- [x] Volume Mounts (dataset, output, DB)

#### 🚧 개발 중
- [ ] Kubernetes 배포
- [ ] Temporal 워크플로우
- [ ] 분산 학습 (DDP)
- [ ] Auto-scaling
- [ ] WebSocket 실시간 업데이트

### 지원 데이터셋 형식

| 형식 | 용도 | 상태 |
|------|------|------|
| **ImageFolder** | Classification | ✅ 완료 |
| **COCO** | Detection, Segmentation | ✅ 완료 |
| **YOLO** | Detection | ✅ 완료 |
| **Pascal VOC** | Detection | ⏳ 계획됨 |
| **Custom** | OCR, VQA, Captioning | ⏳ 계획됨 |

### 현재 제약사항

1. **단일 GPU 학습**: 분산 학습 미지원 (향후 추가 예정)
2. **로컬 실행**: Kubernetes 배포 미완성
3. **데이터셋 자동 변환**: COCO ↔ YOLO 변환 미구현
4. **WebSocket**: 실시간 업데이트가 Polling 방식 (WebSocket 전환 예정)
5. **Auto-scaling**: 수동 리소스 관리

### MVP 완성도

```
[████████████████░░] 80% Complete

✅ Core Infrastructure      (100%)
✅ Adapter Pattern          (100%)
✅ Docker Isolation         (100%)
✅ Basic UI/UX              (90%)
✅ Training Execution       (90%)
🚧 Advanced Monitoring      (60%)
🚧 Production Deployment    (30%)
⏳ Auto-scaling             (0%)
```

---

## 다음 단계

이 문서를 읽은 후:

1. **아키텍처 이해**: [High-Level Architecture](./02-architecture/high-level-architecture.md)에서 전체 구조 파악
2. **역할별 문서**: [메인 인덱스](./README.md)에서 자신의 역할에 맞는 컴포넌트 문서 읽기
3. **개발 시작**: [Development Workflow](./05-development-workflow.md)에서 환경 설정 및 작업 프로세스 확인

---

**작성일**: 2025-10-31
**문서 버전**: 1.0
**작성자**: Project Lead

[← 돌아가기](./README.md) | [다음: High-Level Architecture →](./02-architecture/high-level-architecture.md)
