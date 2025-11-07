# Implementation Priority Analysis: Docker vs Model Plugin

**Document Version:** 1.0
**Created:** 2025-10-30
**Status:** Decision Document

---

## Executive Summary

두 가지 주요 계획이 존재합니다:

1. **MODEL_PLUGIN_VALIDATION_PLAN.md**: 다양한 모델/프레임워크 추가로 플랫폼 확장성 검증 (6주)
2. **DOCKER_IMAGE_SEPARATION.md**: 의존성 격리를 위한 Docker 이미지 분리 (3-4주)

**결론**: **Hybrid 접근 (Phase 1 모델 추가 → Docker 분리 → Phase 2-3 모델 추가)**을 권장합니다.

---

## 두 계획 비교 분석

### 1. MODEL_PLUGIN_VALIDATION_PLAN (모델 플러그인 검증)

**목표**: 플랫폼의 확장성과 플러그인 가능성 검증

| Phase | 내용 | 기간 | 의존성 격리 필요도 |
|-------|------|------|--------------------|
| **Phase 1** | timm + Ultralytics 모델 추가 (18개) | 1주 | ⭐ Low (현재 구조 가능) |
| **Phase 2** | HuggingFace 프레임워크 추가 (5개) | 2주 | ⭐⭐⭐ High (충돌 가능) |
| **Phase 3** | Custom GitHub 모델 추가 (4개) | 2주 | ⭐⭐⭐⭐ Critical (필수) |

**핵심 활동**:
- 모델 레지스트리 시스템 구축
- API 엔드포인트 추가 (`/models/list`)
- Frontend 모델 선택 UI 개선
- TransformersAdapter 구현
- CustomAdapter 구현

**의존성 현황**:
- Phase 1: 단일 venv에서 가능 (timm, ultralytics 공존 가능)
- Phase 2: HuggingFace transformers와 기존 프레임워크 버전 충돌 가능
- Phase 3: PaddlePaddle 같은 비-PyTorch 프레임워크는 격리 필수

---

### 2. DOCKER_IMAGE_SEPARATION (Docker 이미지 분리)

**목표**: 의존성 충돌 없는 확장 가능한 구조

| Phase | 내용 | 기간 |
|-------|------|------|
| **Phase 1** | Platform SDK 분리 | 1주 |
| **Phase 2** | Requirements 분리 | 3일 |
| **Phase 3** | Docker 이미지 정의 | 1주 |
| **Phase 4** | TrainingManager Docker 지원 | 1-2주 |
| **Phase 5** | 테스트 및 문서화 | 1주 |

**핵심 활동**:
- 공통 코드를 `platform_sdk/` 패키지로 분리
- 프레임워크별 `requirements-{framework}.txt` 생성
- Base + 프레임워크별 Dockerfile 작성
- TrainingManager에 Docker 실행 모드 추가
- Subprocess 모드 유지 (MVP 호환)

**주요 이점**:
- ✅ 완벽한 의존성 격리
- ✅ 이미지 크기 최적화 (8GB → 2GB per framework)
- ✅ 빌드 시간 단축 (layer caching)
- ✅ 새 프레임워크 추가 시 독립적으로 이미지 생성

---

## 의존성 충돌 분석

### 현재 상황 (MVP)

```python
# mvp/training/requirements.txt (단일 파일)
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
ultralytics==8.0.220
```

- **현재**: timm과 ultralytics는 PyTorch 2.1.0에서 공존 가능 ✅
- **문제 없음**: Phase 1 (timm/Ultralytics 모델 추가)는 현재 구조에서 실행 가능

### Phase 2 (HuggingFace) 시나리오

```python
# HuggingFace transformers 추가 시
torch==2.1.0          # timm 요구
torch>=2.0.0          # ultralytics 요구
torch>=2.1.0          # transformers 권장
transformers>=4.35.0
accelerate>=0.24.0
```

**가능성 평가**:
- ✅ **충돌 가능성**: Low-Medium
- **이유**: 모두 PyTorch 2.1.0 호환
- **단, 주의**: `accelerate` 패키지가 추가 의존성 가져올 수 있음

**결론**: Phase 2는 단일 venv에서도 가능할 수 있지만, 불안정할 가능성 있음

### Phase 3 (Custom) 시나리오

```python
# PaddlePaddle 추가 시
torch==2.1.0           # 기존
paddlepaddle>=2.6.0    # PaddlePaddle
# → paddlepaddle과 torch는 같은 환경에서 충돌!
```

**충돌 확실**:
- ❌ **충돌 가능성**: Critical
- **이유**: PaddlePaddle은 자체 딥러닝 프레임워크 (PyTorch와 공존 불가)
- **YOLOv7**: 특수한 PyTorch 버전 요구 가능
- **MMPose/ViTPose**: mmcv 의존성이 PyTorch 버전에 민감

**결론**: Phase 3는 Docker 격리 필수

---

## 실행 순서 시나리오 비교

### 시나리오 A: 모델 추가 먼저 (전체)

```
1. Phase 1: timm/Ultralytics 모델 추가 (1주)
2. Phase 2: HuggingFace 추가 (2주) → 의존성 문제 발생 가능
3. Phase 3: Custom 모델 추가 (2주) → 의존성 충돌로 막힘
4. Docker 분리 구현 (3-4주)
5. Phase 2-3 재시도

총 소요: 8-10주 + 재작업
```

**장점**:
- ✅ 빠르게 모델 레지스트리 시스템 구축
- ✅ 초기 피드백 빠름 (Phase 1)

**단점**:
- ❌ Phase 2-3에서 막힐 가능성 높음
- ❌ 재작업 필요
- ❌ 시간 낭비

---

### 시나리오 B: Docker 분리 먼저

```
1. Docker 분리 구현 (3-4주)
2. Phase 1: timm/Ultralytics 모델 추가 (1주)
3. Phase 2: HuggingFace 추가 (2주)
4. Phase 3: Custom 모델 추가 (2주)

총 소요: 8-9주
```

**장점**:
- ✅ 의존성 문제 사전 해결
- ✅ 안정적인 개발 환경
- ✅ 재작업 없음

**단점**:
- ❌ 초기 피드백이 늦음 (3-4주 후)
- ❌ Docker 인프라 구축에 시간 투자
- ❌ 모델 추가가 늦어짐

---

### 시나리오 C: Hybrid 접근 ⭐ **권장**

```
1. Phase 1: timm/Ultralytics 모델 추가 (1주)
   → 현재 구조에서 빠르게 검증
   → 모델 레지스트리, API, UI 구축

2. Docker 분리 구현 (3-4주)
   → Phase 1 결과를 바탕으로 설계 개선
   → 의존성 격리 구조 확립

3. Phase 2: HuggingFace 추가 (2주)
   → Docker 이미지로 안정적 구현

4. Phase 3: Custom 모델 추가 (2주)
   → 독립 이미지로 자유롭게 추가

총 소요: 8-9주
```

**장점**:
- ✅ 빠른 초기 피드백 (1주 내)
- ✅ Phase 1으로 Adapter 패턴 검증
- ✅ Docker 설계 시 Phase 1 경험 반영
- ✅ Phase 2-3에서 안정적 개발
- ✅ 재작업 최소화

**단점**:
- ⚠️ Phase 1 코드를 Docker로 마이그레이션 필요 (하지만 간단함)

---

## 상세 실행 계획 (Hybrid 접근)

### Week 1: Quick Win - Phase 1 모델 추가

**목표**: 현재 구조에서 빠르게 모델 추가 및 플랫폼 검증

#### 구현 항목:

**1.1 모델 레지스트리 시스템** (2일)
```python
# mvp/training/model_registry/timm_models.py
TIMM_MODEL_REGISTRY = {
    "resnet18": {...},
    "resnet34": {...},
    "resnet50": {...},  # 기존
    "resnet101": {...},  # 신규
    "vgg16": {...},
    "vgg19": {...},
    "efficientnet_b0": {...},  # 기존
    "efficientnet_b1": {...},
    "efficientnet_b4": {...},
    "vit_base_patch16_224": {...},
    "vit_large_patch16_224": {...},
    "mobilenetv3_large_100": {...},
    "mobilenetv3_small_100": {...},
}
```

**1.2 API 엔드포인트** (1일)
```python
# mvp/backend/app/api/models.py (신규)
@router.get("/list")
async def list_models(framework: str = None, tags: str = None):
    # 모델 목록 반환
    pass

@router.get("/{framework}/{model_name}")
async def get_model_info(framework: str, model_name: str):
    # 모델 상세 정보 반환
    pass
```

**1.3 Frontend UI** (2일)
```tsx
// mvp/frontend/components/training/ModelSelector.tsx
- 모델 카드 그리드
- 필터링 (framework, tags)
- 모델 정보 표시 (params, batch_size, 추천 설정)
```

**1.4 검증** (1일)
- 각 모델로 간단한 학습 실행
- UI에서 모델 선택 및 학습 시작
- 정상 동작 확인

**Deliverables**:
- [ ] 모델 레지스트리 (timm 13개, ultralytics 8개)
- [ ] API 엔드포인트 (`/api/v1/models/list`)
- [ ] Frontend 모델 선택 UI
- [ ] Phase 1 검증 리포트

**주의사항**:
- 단일 venv 사용 (현재 구조 그대로)
- Docker 관련 작업 없음
- 빠른 반복 개발 우선

---

### Week 2-4: 기반 구축 - Docker 분리

**목표**: 의존성 격리 구조 확립

#### Week 2: Platform SDK 분리 + Requirements 분리

**2.1 Platform SDK 패키지 생성** (3일)
```bash
mvp/training/platform_sdk/
├── __init__.py
├── base.py          # TrainingAdapter, ModelConfig 등
├── callbacks.py     # TrainingCallbacks
├── mlflow_utils.py  # MLflow 헬퍼
└── storage.py       # S3, 로컬 파일 처리
```

**이동 및 리팩토링**:
```python
# Before
from .base import TrainingAdapter

# After
from platform_sdk import TrainingAdapter
```

**2.2 Requirements 분리** (2일)
```bash
mvp/training/requirements/
├── requirements-base.txt      # 공통 (MLflow, boto3, numpy)
├── requirements-timm.txt      # base + timm + torch
├── requirements-ultralytics.txt
└── requirements-huggingface.txt  # 미래 대비
```

**검증**:
- 기존 학습 프로세스 정상 동작 확인
- Import 경로 변경 테스트
- 로컬 venv 재구성 및 테스트

#### Week 3: Docker 이미지 구축

**3.1 Dockerfile 작성** (3일)
```dockerfile
# mvp/docker/Dockerfile.base
FROM python:3.11-slim
COPY training/platform_sdk/ /opt/vision-platform/platform_sdk/
COPY training/requirements/requirements-base.txt /tmp/
RUN pip install -r /tmp/requirements-base.txt

# mvp/docker/Dockerfile.timm
FROM vision-platform-base:latest
COPY training/requirements/requirements-timm.txt /tmp/
RUN pip install -r /tmp/requirements-timm.txt
COPY training/adapters/timm_adapter.py /opt/vision-platform/adapters/

# mvp/docker/Dockerfile.ultralytics
FROM vision-platform-base:latest
COPY training/requirements/requirements-ultralytics.txt /tmp/
RUN pip install -r /tmp/requirements-ultralytics.txt
COPY training/adapters/ultralytics_adapter.py /opt/vision-platform/adapters/
```

**3.2 빌드 스크립트** (1일)
```bash
# mvp/docker/build.sh
./build.sh
# → vision-platform-base:latest
# → vision-platform-timm:latest
# → vision-platform-ultralytics:latest
```

**3.3 검증** (1일)
```bash
# Docker로 학습 실행
docker run vision-platform-timm:latest \
  python /opt/vision-platform/train.py \
  --framework timm --model resnet18 ...
```

#### Week 4: TrainingManager Docker 지원

**4.1 ExecutionMode 추가** (2일)
```python
class ExecutionMode(Enum):
    SUBPROCESS = "subprocess"  # 기존 (MVP 호환)
    DOCKER = "docker"          # 신규

class TrainingManager:
    def __init__(self, execution_mode: ExecutionMode = None):
        if execution_mode is None:
            execution_mode = self._detect_execution_mode()
        self.execution_mode = execution_mode

    def start_training(self, job_id: int):
        if self.execution_mode == ExecutionMode.SUBPROCESS:
            return self._start_training_subprocess(job_id)
        elif self.execution_mode == ExecutionMode.DOCKER:
            return self._start_training_docker(job_id)
```

**4.2 Docker 실행 로직** (3일)
```python
IMAGE_MAP = {
    "timm": "vision-platform-timm:latest",
    "ultralytics": "vision-platform-ultralytics:latest",
}

def _start_training_docker(self, job_id: int):
    image = IMAGE_MAP[job.framework]
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{dataset_path}:/workspace/dataset:ro",
        "-v", f"{output_dir}:/workspace/output:rw",
        image,
        "python", "/opt/vision-platform/train.py", ...
    ]
    process = subprocess.Popen(docker_cmd, ...)
```

**4.3 환경 설정** (1일)
```bash
# mvp/backend/.env
TRAINING_EXECUTION_MODE=auto  # subprocess | docker | auto
```

**4.4 검증** (1일)
- Subprocess 모드 테스트 (MVP 호환성)
- Docker 모드 테스트
- Auto-detection 테스트

**Deliverables**:
- [ ] Platform SDK 패키지
- [ ] 프레임워크별 requirements 파일
- [ ] Docker 이미지 (base, timm, ultralytics)
- [ ] TrainingManager Docker 지원
- [ ] 통합 테스트 완료

---

### Week 5-6: Phase 2 - HuggingFace

**목표**: Docker 격리 환경에서 HuggingFace 프레임워크 추가

#### Week 5: TransformersAdapter 구현

**5.1 Docker 이미지 추가** (1일)
```dockerfile
# mvp/docker/Dockerfile.huggingface
FROM vision-platform-base:latest
COPY training/requirements/requirements-huggingface.txt /tmp/
RUN pip install -r /tmp/requirements-huggingface.txt
```

**5.2 TransformersAdapter 작성** (4일)
- HF Trainer API 통합
- Dataset 변환 유틸리티
- Callback 브리지
- Inference 구현

**검증** (1일):
- ViT 모델로 classification 학습
- Docker 컨테이너에서 정상 동작 확인

#### Week 6: 추가 모델 및 테스트

**6.1 나머지 모델 추가** (2일)
- DeiT, DETR, SegFormer, BLIP

**6.2 통합 테스트** (3일)
- 각 모델 학습 실행
- Inference 테스트
- 성능 벤치마크

**Deliverables**:
- [ ] TransformersAdapter
- [ ] HuggingFace Docker 이미지
- [ ] 5개 모델 검증 완료
- [ ] Phase 2 검증 리포트

---

### Week 7-8: Phase 3 - Custom Models

**목표**: 커스텀 GitHub 모델 지원으로 극한 확장성 검증

#### Week 7: CustomAdapter 구현

**7.1 Base CustomAdapter** (3일)
- Repository clone 및 setup
- Config 파일 생성
- 외부 스크립트 실행
- 로그 파싱

**7.2 ConvNeXt + YOLOv7** (2일)
- ConvNeXtAdapter 구현
- YOLOv7Adapter 구현
- Docker 이미지 생성

#### Week 8: 복잡한 모델 및 마무리

**8.1 PP-YOLO + ViTPose** (3일)
- PaddlePaddle 지원 (독립 컨테이너 필수)
- MMPose 지원

**8.2 최종 검증 및 문서화** (2일)
- 전체 시스템 통합 테스트
- 종합 검증 리포트 작성
- 개선 제안서 작성

**Deliverables**:
- [ ] CustomAdapter
- [ ] 4개 커스텀 모델 Wrapper
- [ ] Phase 3 검증 리포트
- [ ] 최종 종합 리포트

---

## 각 시나리오별 리스크 분석

### 시나리오 A (모델 먼저)

| 리스크 | 확률 | 영향 | 완화 방안 |
|--------|------|------|-----------|
| Phase 2에서 의존성 충돌 | 60% | High | 없음 (재작업 필요) |
| Phase 3에서 완전히 막힘 | 90% | Critical | 없음 (Docker 필수) |
| 시간 낭비 | 80% | High | 없음 |

**총 리스크**: ⚠️ **High**

### 시나리오 B (Docker 먼저)

| 리스크 | 확률 | 영향 | 완화 방안 |
|--------|------|------|-----------|
| Docker 학습 곡선 | 40% | Medium | 문서화, 예제 제공 |
| 초기 피드백 지연 | 100% | Low | 단계별 검증 |
| 과도한 인프라 투자 | 30% | Low | MVP 호환 모드 유지 |

**총 리스크**: ✅ **Low**

### 시나리오 C (Hybrid) ⭐

| 리스크 | 확률 | 영향 | 완화 방안 |
|--------|------|------|-----------|
| Phase 1 마이그레이션 | 100% | Low | 모델 레지스트리는 그대로 사용 |
| Docker 학습 곡선 | 40% | Medium | Phase 1 경험 활용 |
| 일정 지연 | 20% | Low | 병렬 작업 가능 |

**총 리스크**: ✅ **Low**

---

## 최종 권장사항

### 권장: 시나리오 C (Hybrid 접근)

**근거**:

1. **빠른 피드백 (Week 1)**:
   - Phase 1으로 모델 레지스트리 시스템 조기 검증
   - Adapter 패턴의 실제 확장성 확인
   - UI/API 설계 검증

2. **안정적 기반 (Week 2-4)**:
   - Phase 1 경험을 바탕으로 Docker 설계 개선
   - 의존성 격리 구조 확립
   - Phase 2-3를 위한 준비

3. **확장 가능한 구조 (Week 5-8)**:
   - Docker 환경에서 HuggingFace 안정적 추가
   - Custom 모델도 독립 컨테이너로 자유롭게 추가
   - 재작업 최소화

4. **총 소요 시간**: 8-9주 (시나리오 B와 동일, 하지만 조기 피드백 확보)

5. **리스크**: Low (의존성 문제 사전 방지 + 빠른 검증)

---

## 실행 계획 요약

```
┌─────────────────────────────────────────────────────────────┐
│  Week 1: Phase 1 모델 추가 (Quick Win)                      │
│  - 모델 레지스트리 (timm 13개, ultralytics 8개)             │
│  - API 엔드포인트 (/models/list)                             │
│  - Frontend 모델 선택 UI                                     │
│  - 검증: 현재 구조에서 정상 동작 확인                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Week 2-4: Docker 분리 (기반 구축)                          │
│  - Platform SDK 분리                                         │
│  - Requirements 분리                                         │
│  - Docker 이미지 구축 (base, timm, ultralytics)             │
│  - TrainingManager Docker 지원                               │
│  - 검증: Docker + Subprocess 모두 정상 동작                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Week 5-6: Phase 2 HuggingFace (안정적 추가)                │
│  - TransformersAdapter 구현                                  │
│  - HuggingFace Docker 이미지                                 │
│  - 5개 모델 추가 (ViT, DeiT, DETR, SegFormer, BLIP)        │
│  - 검증: Docker 격리 환경에서 안정적 동작                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Week 7-8: Phase 3 Custom Models (극한 확장성)              │
│  - CustomAdapter 구현                                        │
│  - 4개 커스텀 모델 (ConvNeXt, YOLOv7, PP-YOLO, ViTPose)    │
│  - 독립 컨테이너로 자유롭게 추가                             │
│  - 최종 검증 및 종합 리포트                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Action Items

### Immediate (이번 주)

- [ ] 본 분석 문서 리뷰 및 승인
- [ ] Week 1 작업 시작 준비
  - [ ] 브랜치 생성: `feat/model-registry-phase1`
  - [ ] 작업 할당
  - [ ] Kickoff 미팅

### Week 1 Deliverables

- [ ] `mvp/training/model_registry/timm_models.py` (13 models)
- [ ] `mvp/training/model_registry/ultralytics_models.py` (8 models)
- [ ] `mvp/backend/app/api/models.py` (API endpoints)
- [ ] `mvp/frontend/components/training/ModelSelector.tsx` (UI)
- [ ] Phase 1 검증 리포트

### Week 2 Preparation

- [ ] Docker 분리 작업 계획 상세화
- [ ] Platform SDK 설계 리뷰
- [ ] 브랜치 생성: `feat/docker-image-separation`

---

## Success Criteria

### Week 1 완료 시

- [ ] 21개 모델 메타데이터 등록 완료
- [ ] API로 모델 목록 조회 가능
- [ ] UI에서 모델 카드로 선택 가능
- [ ] 최소 5개 모델로 학습 성공

### Week 4 완료 시

- [ ] Platform SDK 패키지 생성 완료
- [ ] Docker 이미지 빌드 성공 (base, timm, ultralytics)
- [ ] TrainingManager가 Docker/Subprocess 모두 지원
- [ ] 기존 기능 100% 동작 (MVP 호환)

### Week 6 완료 시

- [ ] HuggingFace 프레임워크 정식 지원
- [ ] 5개 HF 모델 검증 완료
- [ ] Docker 격리 환경에서 안정적 동작

### Week 8 완료 시

- [ ] 4개 커스텀 모델 지원
- [ ] CustomAdapter로 GitHub 모델 자유롭게 추가 가능
- [ ] 종합 검증 리포트 완성
- [ ] 플랫폼 확장성 및 플러그인 가능성 입증

---

## 결론

**권장 실행 순서**: **Hybrid 접근 (시나리오 C)**

1. **Week 1**: Phase 1 모델 추가로 빠른 검증
2. **Week 2-4**: Docker 분리로 안정적 기반 구축
3. **Week 5-6**: Phase 2 HuggingFace 추가
4. **Week 7-8**: Phase 3 Custom 모델 추가

**총 소요**: 8-9주
**리스크**: Low
**예상 효과**: 플랫폼 확장성 입증 + 의존성 격리 완성

---

*Document Version: 1.0*
*Created: 2025-10-30*
*Author: Vision AI Platform Team*
