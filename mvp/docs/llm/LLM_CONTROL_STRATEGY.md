# 자연어 제어 구현 전략

## 문서 개요

**작성일**: 2025-11-01
**목적**: Vision AI Training Platform의 자연어 인터페이스 개선 및 확장 전략 수립
**대상**: Backend/Frontend 개발자, AI Engineer

---

## 목차

1. [현재 상태 분석](#1-현재-상태-분석)
2. [목표 및 요구사항](#2-목표-및-요구사항)
3. [아키텍처 설계](#3-아키텍처-설계)
4. [발화 패턴 → 기능 매핑](#4-발화-패턴--기능-매핑)
5. [MCP 활용 방안](#5-mcp-활용-방안)
6. [구현 로드맵](#6-구현-로드맵)
7. [기술 스택 및 의사결정](#7-기술-스택-및-의사결정)

---

## 1. 현재 상태 분석

### 1.1 구현된 기능

현재 MVP는 다음 기능을 제공합니다:

**학습 관련:**
- ✅ 학습 작업 생성/시작/중지/재시작
- ✅ 체크포인트에서 재개
- ✅ 실시간 메트릭 모니터링 (MLflow 통합)
- ✅ Validation 결과 분석 (Confusion Matrix, PR Curve, 이미지별 결과)
- ✅ 학습 로그 조회
- ✅ 고급 설정 (Optimizer, Scheduler, Augmentation)

**추론 관련:**
- ✅ Quick Inference (단일 이미지)
- ✅ Batch Inference (여러 이미지)
- ✅ Test Runs (레이블된 데이터셋에 대한 추론 + 메트릭)

**데이터셋 관련:**
- ✅ 데이터셋 분석 (포맷 감지, 클래스 분포, 품질 체크)
- ✅ 데이터셋 목록 조회

**모델 관련:**
- ✅ 모델 목록 조회 (framework, task_type, tags 필터링)
- ✅ 모델 가이드 제공 (벤치마크, 장단점, 대안 모델)
- ✅ 모델 비교

**프로젝트 관련:**
- ✅ 프로젝트 생성/수정/삭제
- ✅ 프로젝트 멤버 관리
- ✅ Experiment 조직화

**프레임워크 및 모델:**
- ✅ timm (resnet18, resnet50, efficientnet_b0)
- ✅ ultralytics (yolov8n/s/m, yolo11n/s/m)
- ✅ transformers (ViT, D-FINE, SegFormer, Swin2SR)
- ✅ Task Types: Classification, Detection, Instance Segmentation, Pose Estimation, Semantic Segmentation, Super Resolution

### 1.2 현재 LLM 통합 방식

#### **Approach 1: Intent Parser (`llm.py`)** - 사용되지 않음
- Google Gemini + LangChain
- 자연어 → JSON 구성 파싱
- 대화 컨텍스트 처리

#### **Approach 2: Structured Intent Parser (`llm_structured.py`)** ⭐ **현재 활성**
- Google Gemini (native SDK)
- State Machine 기반 대화 흐름
- Structured Actions 출력

**State Machine:**
```
INITIAL → GATHERING_CONFIG → SELECTING_PROJECT → CREATING_PROJECT → CONFIRMING → TRAINING → COMPLETED/ERROR
```

**Actions:**
- `ASK_CLARIFICATION` - 누락된 정보 요청
- `SHOW_PROJECT_OPTIONS` - 프로젝트 선택 메뉴
- `SHOW_PROJECT_LIST` - 기존 프로젝트 목록
- `CREATE_PROJECT` - 새 프로젝트 생성
- `SELECT_PROJECT` - 기존 프로젝트 선택
- `SKIP_PROJECT` - Uncategorized 사용
- `CONFIRM_TRAINING` - 최종 확인
- `START_TRAINING` - 학습 시작
- `ERROR` - 에러 처리

**현재 구현의 한계:**
1. **학습 설정에만 집중** - 추론, 데이터셋 분석, 모델 검색 등은 자연어로 제어 불가
2. **단방향 흐름** - "학습 설정 → 시작"만 가능, 중간에 설정 변경이나 다른 작업으로 전환 어려움
3. **도구 호출 제한** - LLM이 직접 API를 호출하지 못하고 중간 로직 필요
4. **컨텍스트 제한** - Session temp_data에 저장되지만 장기 기억 부족
5. **확장성 문제** - 새 기능 추가 시 State Machine과 ActionHandler 모두 수정 필요

### 1.3 초기 설계와 현재 구현의 차이

| 항목 | 초기 설계 (ARCHITECTURE.md) | 현재 MVP 구현 |
|------|----------------------------|---------------|
| **아키텍처** | 마이크로서비스 (Intent Parser, Orchestrator, Model Registry 등) | Monolith FastAPI |
| **오케스트레이션** | Temporal Workflows | 로컬 프로세스 관리 |
| **학습 실행** | Kubernetes Pods + Sidecar | subprocess (uvicorn) |
| **LLM** | Claude/GPT-4 + LangChain | Gemini + Native SDK |
| **API Gateway** | Kong | 없음 (직접 노출) |
| **대화 관리** | ConversationContext + Intent Parser | State Machine + Structured Actions |

**변경 이유:**
- MVP 단계에서 복잡도 감소
- 빠른 프로토타이핑
- 로컬 개발 환경에서 실행 가능

---

## 2. 목표 및 요구사항

### 2.1 핵심 목표

> **"사용자가 자연어로 AI 비전 모델의 전체 라이프사이클(데이터셋 분석 → 모델 선택 → 학습 → 검증 → 추론)을 제어할 수 있도록 한다."**

### 2.2 사용자 시나리오

#### **시나리오 1: 초보 사용자 - 전체 가이드**
```
사용자: "불량품 검사 모델을 만들고 싶어요"

AI: "좋습니다! 불량품 검사를 위한 모델을 만들어드리겠습니다.
     먼저 몇 가지 확인할게요:

     1. 어떤 종류의 불량을 검사하시나요? (예: 스크래치, 찍힘, 오염 등)
     2. 데이터셋은 준비되어 있나요?"

사용자: "스크래치, 찍힘, 정상품 세 가지 분류하고 싶어요.
        데이터는 C:/datasets/defect에 있어요."

AI: [데이터셋 분석 도구 호출]
    "데이터셋을 분석했습니다:
     - 총 1,200장 (스크래치 400장, 찍힘 350장, 정상 450장)
     - ImageFolder 포맷
     - 이미지 해상도: 평균 640x480

     이 작업에는 EfficientNet-B0를 추천드립니다.
     - 정확도와 속도의 균형이 좋음
     - 작은 데이터셋에 적합
     - ResNet-50보다 30% 빠름

     학습을 시작하시겠어요?"

사용자: "네, 시작해주세요"

AI: [학습 작업 생성 및 시작]
    "학습을 시작했습니다! (Job ID: 12345)
     - 모델: EfficientNet-B0
     - Epochs: 50
     - 학습률: 0.001

     실시간 진행상황은 Training Panel에서 확인하실 수 있습니다."
```

#### **시나리오 2: 중급 사용자 - 세부 설정**
```
사용자: "YOLO11m으로 객체 검출 모델 학습하려고 하는데,
        learning rate 0.0005로 시작해서 cosine scheduler 쓰고 싶어요"

AI: "YOLO11m 모델로 객체 검출을 설정하겠습니다.
     데이터셋 경로를 알려주시겠어요?"

사용자: "C:/datasets/coco_custom 이고 YOLO 포맷이에요"

AI: [데이터셋 분석]
    "YOLO 포맷 데이터셋 확인했습니다:
     - 클래스: 5개 (person, car, dog, cat, truck)
     - 이미지: 2,500장

     설정 확인:
     - 모델: YOLO11m
     - Learning Rate: 0.0005
     - Scheduler: Cosine Annealing
     - Epochs: 100 (기본값, 변경하시겠어요?)

     이대로 진행하시겠어요?"

사용자: "Epochs 150으로 하고 시작해주세요"

AI: [고급 설정으로 학습 시작]
```

#### **시나리오 3: 고급 사용자 - 실험 관리**
```
사용자: "지난 학습 작업들 중에서 mAP가 가장 높은 모델로 추론 돌려줘"

AI: [학습 작업 조회 + 메트릭 비교]
    "지난 학습 작업 중 가장 성능이 좋은 모델:
     - Job ID: 12340
     - 모델: YOLOv8m
     - mAP50: 0.876
     - Epoch: 142

     추론할 이미지를 업로드하시거나 경로를 알려주세요."

사용자: "C:/test_images 폴더의 모든 이미지에 대해 추론해줘"

AI: [Batch Inference 시작]
    "Inference Job을 생성했습니다. (Job ID: inf_789)
     - 이미지 수: 45장
     - 예상 소요 시간: 약 2분"
```

#### **시나리오 4: 데이터셋 분석 및 모델 추천**
```
사용자: "어떤 모델이 좋을지 모르겠어요. 데이터셋 먼저 분석해주세요"

AI: "데이터셋 경로를 알려주세요."

사용자: "C:/my_dataset"

AI: [데이터셋 분석]
    "데이터셋 분석 결과:
     - 포맷: ImageFolder (분류 작업)
     - 클래스 수: 10개
     - 총 이미지: 3,200장
     - 클래스 불균형: 심각 (최대 450장, 최소 180장)
     - 이미지 품질: 양호 (손상된 파일 없음)

     이 데이터셋에는 다음 모델들을 추천합니다:

     1. ResNet-50 ⭐ 추천
        - 정확도: 높음
        - 훈련 시간: 보통
        - 클래스 불균형 처리: 양호

     2. EfficientNet-B0
        - 정확도: 중상
        - 훈련 시간: 빠름
        - 메모리 효율적

     Class Imbalance가 있으므로 다음 설정을 권장합니다:
     - Weighted Cross Entropy Loss
     - Class Weights 적용
     - Oversampling 고려

     어떤 모델로 진행하시겠어요?"
```

#### **시나리오 5: 학습 중 모니터링 및 조정**
```
사용자: "현재 학습 진행 상황 알려줘"

AI: [현재 학습 작업 조회]
    "진행 중인 학습 작업:
     - Job ID: 12350
     - 모델: ResNet-50
     - 진행률: Epoch 35/100 (35%)
     - 현재 Accuracy: 0.823
     - 예상 남은 시간: 45분
     - 상태: Training

     최근 10 epoch의 loss가 거의 변화가 없습니다. (0.234 → 0.231)
     Early stopping을 고려하시겠어요?"

사용자: "중지하고 체크포인트 저장해줘"

AI: [학습 중지]
    "학습을 중지하고 체크포인트를 저장했습니다.
     - 체크포인트 경로: outputs/job_12350/checkpoints/epoch_35.pt
     - 저장된 Accuracy: 0.823

     나중에 이 체크포인트에서 재개하시려면 '12350번 작업 재개해줘'라고 말씀해주세요."
```

### 2.3 요구사항

#### **기능적 요구사항**

**FR-1: 다양한 인텐트 처리**
- 학습 설정 및 시작
- 추론 실행
- 데이터셋 분석
- 모델 검색 및 비교
- 학습 작업 조회 및 관리
- 검증 결과 분석
- 프로젝트 관리

**FR-2: 컨텍스트 유지**
- 대화 기록 기반 추론
- 사용자 선호도 학습 (모델, 하이퍼파라미터 등)
- 이전 작업 참조 (예: "지난번 학습한 모델로...")

**FR-3: 다국어 지원**
- 한국어 우선
- 영어 지원

**FR-4: 에러 처리 및 복구**
- 명확한 에러 메시지
- 대안 제시
- 부분 실패 시 복구 가능

**FR-5: 유연한 설정 수준**
- 초보: 최소 입력으로 자동 설정
- 중급: 주요 하이퍼파라미터 조정
- 고급: 모든 설정 세밀 제어

#### **비기능적 요구사항**

**NFR-1: 응답 속도**
- LLM 응답: 3초 이내
- 도구 호출: 각 도구별 SLA 준수
- 스트리밍 응답 지원

**NFR-2: 확장성**
- 새 도구 추가 용이
- 새 인텐트 추가 용이
- 프롬프트 업데이트 간편

**NFR-3: 신뢰성**
- LLM 실패 시 Fallback
- 도구 호출 실패 시 재시도
- 부분 실패 허용

**NFR-4: 보안**
- 사용자별 권한 검증
- 민감 정보 마스킹
- API 키 안전 관리

**NFR-5: 모니터링**
- LLM 호출 로그
- 도구 사용 통계
- 사용자 인텐트 분포

---

## 3. 아키텍처 설계

### 3.1 전체 아키텍처 개요

우리는 **듀얼 트랙 접근 방식**을 채택합니다:

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                      │
│  ┌────────────────┐              ┌───────────────────────┐  │
│  │  Web Chat UI   │              │  Claude Code / API    │  │
│  │   (Gemini)     │              │  Client (MCP)         │  │
│  └────────────────┘              └───────────────────────┘  │
└─────────┬───────────────────────────────────┬───────────────┘
          │                                   │
          │ REST API                          │ MCP Protocol
          ▼                                   ▼
┌──────────────────────────────┐  ┌─────────────────────────┐
│   Gemini Conversation API    │  │    MCP Server           │
│   (기존 State Machine)        │  │    (FastAPI)            │
│                              │  │                         │
│  - State Management          │  │  - Tools                │
│  - Action Handlers           │  │  - Resources            │
│  - Session Persistence       │  │  - Prompts              │
└──────────────────────────────┘  └─────────────────────────┘
          │                                   │
          └───────────────┬───────────────────┘
                          ▼
        ┌─────────────────────────────────────────┐
        │         Backend Services                 │
        │  (Training, Inference, Dataset, etc.)    │
        └─────────────────────────────────────────┘
```

### 3.2 Track 1: Gemini State Machine (Web UI)

**목적**: 일반 사용자를 위한 가이드형 대화 인터페이스

**특징:**
- 단계별 안내
- 명확한 선택지 제공
- 초보자 친화적
- 한국어 최적화

**개선 방향:**

1. **State Machine 확장**
   ```python
   class ConversationState(Enum):
       # 기존
       INITIAL = "initial"
       GATHERING_CONFIG = "gathering_config"
       SELECTING_PROJECT = "selecting_project"
       CREATING_PROJECT = "creating_project"
       CONFIRMING = "confirming"
       TRAINING = "training"
       COMPLETED = "completed"
       ERROR = "error"

       # 신규 추가
       ANALYZING_DATASET = "analyzing_dataset"
       SELECTING_MODEL = "selecting_model"
       COMPARING_MODELS = "comparing_models"
       MONITORING_TRAINING = "monitoring_training"
       RUNNING_INFERENCE = "running_inference"
       VIEWING_RESULTS = "viewing_results"
       MANAGING_EXPERIMENTS = "managing_experiments"
   ```

2. **Action 확장**
   ```python
   class ActionType(Enum):
       # 기존
       ASK_CLARIFICATION = "ask_clarification"
       SHOW_PROJECT_OPTIONS = "show_project_options"
       # ... 기존 actions ...

       # 신규 추가
       ANALYZE_DATASET = "analyze_dataset"
       RECOMMEND_MODELS = "recommend_models"
       COMPARE_MODELS = "compare_models"
       SHOW_TRAINING_STATUS = "show_training_status"
       START_INFERENCE = "start_inference"
       SHOW_INFERENCE_RESULTS = "show_inference_results"
       LIST_EXPERIMENTS = "list_experiments"
       EXPORT_RESULTS = "export_results"
   ```

3. **Multi-Intent 지원**
   - 현재: 단일 작업 흐름 (학습 설정 → 시작)
   - 개선: 여러 인텐트 동시 처리

   ```python
   class IntentType(Enum):
       TRAIN_MODEL = "train"
       RUN_INFERENCE = "inference"
       ANALYZE_DATASET = "analyze_dataset"
       SEARCH_MODEL = "search_model"
       MANAGE_PROJECT = "manage_project"
       VIEW_RESULTS = "view_results"
       COMPARE_EXPERIMENTS = "compare_experiments"
   ```

4. **도구 호출 래퍼**
   ```python
   class ToolRegistry:
       """Gemini가 사용할 수 있는 도구 레지스트리"""

       async def call_tool(
           self,
           tool_name: str,
           parameters: dict,
           user_id: int
       ) -> dict:
           """도구 호출 및 결과 반환"""

           # 권한 검증
           if not self._check_permission(user_id, tool_name):
               raise PermissionError

           # 도구 실행
           tool = self.tools[tool_name]
           result = await tool.execute(parameters)

           # 로깅
           await self._log_tool_call(user_id, tool_name, parameters, result)

           return result
   ```

### 3.3 Track 2: MCP Server (Advanced Users / API)

**목적**: 고급 사용자와 개발자를 위한 프로그래매틱 인터페이스

**특징:**
- Claude Code에서 직접 사용
- 도구 기반 작업 실행
- 유연한 워크플로우
- API 클라이언트에서 활용 가능

**MCP 아키텍처:**

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Server                            │
│                  (FastAPI Application)                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Tools (Functions)                   │    │
│  ├─────────────────────────────────────────────────┤    │
│  │  • create_training_job                          │    │
│  │  • start_training                               │    │
│  │  • get_training_status                          │    │
│  │  • stop_training                                │    │
│  │  • run_inference                                │    │
│  │  • analyze_dataset                              │    │
│  │  • search_models                                │    │
│  │  • get_model_guide                              │    │
│  │  • compare_models                               │    │
│  │  • list_experiments                             │    │
│  │  • get_validation_results                       │    │
│  │  • export_model                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │           Resources (Data Sources)               │    │
│  ├─────────────────────────────────────────────────┤    │
│  │  • training_job_details://{job_id}              │    │
│  │  • validation_results://{job_id}/{epoch}        │    │
│  │  • inference_results://{inference_job_id}       │    │
│  │  • dataset_analysis://{dataset_path}            │    │
│  │  • model_catalog://                             │    │
│  │  • experiment_history://{project_id}            │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │               Prompts (Templates)                │    │
│  ├─────────────────────────────────────────────────┤    │
│  │  • model_recommendation                         │    │
│  │  • hyperparameter_tuning                        │    │
│  │  • dataset_quality_check                        │    │
│  │  • training_troubleshooting                     │    │
│  │  • result_interpretation                        │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**MCP Tools 정의 예시:**

```python
# mvp/backend/app/mcp/tools.py

from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("vision-ai-platform")

@server.tool()
async def create_training_job(
    model_name: str,
    task_type: str,
    dataset_path: str,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    project_id: int | None = None
) -> dict:
    """
    Create a new training job with specified configuration.

    Args:
        model_name: Model name (e.g., "resnet50", "yolov8m")
        task_type: Task type (classification, detection, segmentation, etc.)
        dataset_path: Path to dataset directory
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        project_id: Optional project ID to associate with

    Returns:
        dict: Created training job details including job_id
    """
    # Implementation
    pass

@server.tool()
async def analyze_dataset(dataset_path: str) -> dict:
    """
    Analyze dataset structure, quality, and statistics.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        dict: Dataset analysis including format, class distribution, quality metrics
    """
    # Implementation
    pass

@server.tool()
async def search_models(
    task_type: str | None = None,
    framework: str | None = None,
    tags: list[str] | None = None
) -> list[dict]:
    """
    Search available models by filters.

    Args:
        task_type: Filter by task type
        framework: Filter by framework (timm, ultralytics, transformers)
        tags: Filter by tags

    Returns:
        list[dict]: List of matching models with metadata
    """
    # Implementation
    pass

@server.tool()
async def get_training_status(job_id: int) -> dict:
    """
    Get current status and metrics of a training job.

    Args:
        job_id: Training job ID

    Returns:
        dict: Job status, current metrics, progress
    """
    # Implementation
    pass
```

**MCP Resources 정의 예시:**

```python
# mvp/backend/app/mcp/resources.py

@server.resource("training_job_details://{job_id}")
async def get_training_job_resource(uri: str) -> str:
    """
    Get detailed information about a training job.

    Returns formatted text with job configuration, status, and metrics.
    """
    job_id = extract_id_from_uri(uri)
    job = await get_training_job(job_id)

    return f"""
    Training Job #{job.id}

    Configuration:
    - Model: {job.model_name}
    - Task: {job.task_type}
    - Dataset: {job.dataset_path}
    - Epochs: {job.epochs}
    - Learning Rate: {job.learning_rate}

    Status: {job.status}
    Progress: {job.current_epoch}/{job.epochs} epochs

    Latest Metrics:
    - Accuracy: {job.latest_accuracy}
    - Loss: {job.latest_loss}
    """

@server.resource("model_catalog://")
async def get_model_catalog() -> str:
    """
    Get complete model catalog with all available models.
    """
    models = await list_all_models()

    catalog = "Available Models\n\n"
    for model in models:
        catalog += f"- {model.name} ({model.framework})\n"
        catalog += f"  Task: {model.task_type}\n"
        catalog += f"  {model.description}\n\n"

    return catalog
```

**MCP Prompts 정의 예시:**

```python
# mvp/backend/app/mcp/prompts.py

@server.prompt()
async def model_recommendation(
    task_type: str,
    dataset_size: int,
    target_metric: str = "accuracy"
) -> list[Message]:
    """
    Generate model recommendation prompt based on task and dataset.
    """
    models = await search_models(task_type=task_type)
    dataset_info = f"Dataset size: {dataset_size} images"

    return [
        Message(
            role="user",
            content=f"""
            I need to select a model for {task_type} task.
            {dataset_info}
            Target metric: {target_metric}

            Available models:
            {format_models(models)}

            Please recommend the best model and explain why.
            """
        )
    ]
```

### 3.4 통합 아키텍처

두 트랙은 백엔드 서비스를 공유하며, 각각의 강점을 살립니다:

| 측면 | Gemini Track (Web) | MCP Track (API) |
|------|-------------------|-----------------|
| **대상** | 일반 사용자 | 고급 사용자, 개발자 |
| **인터페이스** | Chat UI | Claude Code, API |
| **대화 스타일** | 가이드형, 단계별 | 자유형, 도구 기반 |
| **유연성** | 낮음 (정해진 흐름) | 높음 (자유로운 조합) |
| **학습 곡선** | 낮음 | 높음 |
| **자동화** | 제한적 | 완전 자동화 가능 |
| **커스터마이징** | 제한적 | 높음 |

**공통 백엔드 서비스:**
- Training Manager
- Inference Service
- Dataset Analyzer
- Model Registry
- Project Manager
- Validation Service

---

## 4. 발화 패턴 → 기능 매핑

### 4.1 인텐트 분류

| 인텐트 카테고리 | 예시 발화 | 매핑되는 기능 | API Endpoint | MCP Tool |
|---------------|----------|-------------|-------------|----------|
| **학습 생성** | "ResNet50으로 고양이 분류 모델 만들어줘"<br>"YOLO로 객체 검출 학습하고 싶어요" | 학습 작업 생성 | `POST /training/jobs` | `create_training_job` |
| **학습 제어** | "학습 시작해줘"<br>"학습 중지"<br>"12345번 작업 재개" | 학습 시작/중지/재개 | `POST /training/jobs/{id}/start`<br>`POST /training/jobs/{id}/cancel`<br>`POST /training/jobs/{id}/restart` | `start_training`<br>`stop_training`<br>`resume_training` |
| **학습 모니터링** | "현재 학습 진행 상황 알려줘"<br>"loss 그래프 보여줘" | 학습 상태 조회 | `GET /training/jobs/{id}/status`<br>`GET /training/jobs/{id}/metrics` | `get_training_status`<br>`get_training_metrics` |
| **데이터셋 분석** | "데이터셋 분석해줘"<br>"C:/datasets/my_data 구조 확인" | 데이터셋 분석 | `POST /datasets/analyze` | `analyze_dataset` |
| **모델 검색** | "객체 검출에 좋은 모델 추천해줘"<br>"YOLO 모델들 비교해줘" | 모델 검색/비교 | `GET /models/list`<br>`GET /models/compare` | `search_models`<br>`compare_models` |
| **모델 가이드** | "EfficientNet이 뭐야?"<br>"ResNet50 장단점 알려줘" | 모델 정보 조회 | `GET /models/{framework}/{name}/guide` | `get_model_guide` |
| **추론 실행** | "이 이미지 분류해줘"<br>"test_images 폴더 추론 돌려줘" | Quick/Batch Inference | `POST /inference/quick`<br>`POST /inference/jobs` | `run_quick_inference`<br>`run_batch_inference` |
| **검증 결과** | "validation 결과 보여줘"<br>"confusion matrix 확인" | Validation 결과 조회 | `GET /validation/jobs/{id}/results`<br>`GET /validation/jobs/{id}/summary` | `get_validation_results`<br>`get_confusion_matrix` |
| **프로젝트 관리** | "새 프로젝트 만들어줘"<br>"프로젝트 목록 보여줘" | 프로젝트 CRUD | `POST /projects`<br>`GET /projects` | `create_project`<br>`list_projects` |
| **실험 관리** | "지난 학습들 비교해줘"<br>"가장 성능 좋은 모델 찾아줘" | Experiment 조회/비교 | `GET /projects/{id}/experiments` | `list_experiments`<br>`find_best_model` |
| **결과 해석** | "이 결과가 좋은 건가요?"<br>"정확도 0.85가 괜찮아요?" | 결과 해석 및 조언 | - (LLM 기반) | `interpret_results` (Prompt) |

### 4.2 복합 인텐트 처리

실제 사용자는 여러 인텐트를 한 문장에 담을 수 있습니다:

**예시 1:**
```
사용자: "C:/datasets/defect 데이터셋 분석하고,
        적합한 모델 추천해준 다음에 바로 학습 시작해줘"

분해:
1. analyze_dataset(dataset_path="C:/datasets/defect")
2. recommend_models(based on analysis)
3. create_training_job(model=recommended)
4. start_training(job_id=created_job_id)
```

**처리 방법:**

**Gemini Track (State Machine):**
- 단계별 처리
- 각 단계에서 확인 요청
- 사용자 승인 후 다음 단계

**MCP Track (Tool Chaining):**
- LLM이 자동으로 도구 순서 결정
- 연쇄 호출
- 최종 결과만 사용자에게 보고

### 4.3 컨텍스트 기반 의도 추론

이전 대화를 고려한 의도 파악:

```
사용자: "ResNet50으로 학습 시작했어"
AI: [학습 작업 생성 및 시작]

사용자: "중지해줘"
    ↓
컨텍스트: 방금 시작한 학습 작업
의도: 해당 작업 중지
Action: stop_training(job_id=방금생성한작업)

사용자: "다시 시작"
    ↓
컨텍스트: 방금 중지한 작업
의도: 동일 작업 재개
Action: restart_training(job_id=중지한작업)
```

**컨텍스트 저장 구조:**

```python
class ConversationContext:
    session_id: str
    user_id: int

    # 대화 기록
    message_history: list[Message]

    # 엔티티 추적
    current_training_job: int | None
    current_inference_job: int | None
    current_project: int | None
    last_analyzed_dataset: str | None

    # 사용자 선호도
    preferred_models: dict[str, str]  # task_type -> model_name
    typical_hyperparams: dict

    # 임시 데이터 (State Machine용)
    temp_data: dict
```

---

## 5. MCP 활용 방안

### 5.1 MCP 도입의 이점

1. **표준화된 프로토콜**
   - Anthropic의 공식 표준
   - Claude Code 네이티브 지원
   - 다른 AI 도구와도 호환 가능

2. **도구 중심 설계**
   - LLM이 필요한 도구를 자동 선택
   - 복잡한 State Machine 불필요
   - 확장 용이

3. **풍부한 컨텍스트**
   - Resources를 통해 구조화된 데이터 제공
   - LLM이 필요 시 추가 정보 조회
   - 더 정확한 추론

4. **프로그래매틱 접근**
   - API 클라이언트에서 직접 활용
   - CI/CD 파이프라인 통합
   - 자동화 스크립트 작성

### 5.2 MCP Server 구현

#### **디렉토리 구조**

```
mvp/backend/app/mcp/
├── __init__.py
├── server.py          # MCP Server 초기화 및 실행
├── tools/
│   ├── __init__.py
│   ├── training.py    # 학습 관련 도구
│   ├── inference.py   # 추론 관련 도구
│   ├── dataset.py     # 데이터셋 관련 도구
│   ├── model.py       # 모델 관련 도구
│   └── project.py     # 프로젝트 관련 도구
├── resources/
│   ├── __init__.py
│   ├── training.py    # 학습 작업 리소스
│   ├── validation.py  # 검증 결과 리소스
│   └── model.py       # 모델 카탈로그 리소스
└── prompts/
    ├── __init__.py
    ├── recommendation.py  # 추천 프롬프트
    └── troubleshooting.py # 트러블슈팅 프롬프트
```

#### **MCP Server 초기화**

```python
# mvp/backend/app/mcp/server.py

from mcp.server import Server
from mcp.server.stdio import stdio_server
import asyncio

# MCP Server 인스턴스 생성
mcp_server = Server("vision-ai-training-platform")

# Tools 등록
from .tools import training, inference, dataset, model, project
training.register_tools(mcp_server)
inference.register_tools(mcp_server)
dataset.register_tools(mcp_server)
model.register_tools(mcp_server)
project.register_tools(mcp_server)

# Resources 등록
from .resources import training as training_resources
from .resources import validation, model as model_resources
training_resources.register_resources(mcp_server)
validation.register_resources(mcp_server)
model_resources.register_resources(mcp_server)

# Prompts 등록
from .prompts import recommendation, troubleshooting
recommendation.register_prompts(mcp_server)
troubleshooting.register_prompts(mcp_server)

async def main():
    """MCP Server 실행"""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

#### **Training Tools 구현**

```python
# mvp/backend/app/mcp/tools/training.py

from mcp.server import Server
from mcp.types import Tool, TextContent
from app.services.training_service import TrainingService
from app.db.database import get_db

training_service = TrainingService()

def register_tools(server: Server):
    """학습 관련 도구 등록"""

    @server.tool()
    async def create_training_job(
        model_name: str,
        task_type: str,
        dataset_path: str,
        framework: str = "timm",
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_classes: int | None = None,
        project_id: int | None = None,
        experiment_name: str | None = None
    ) -> dict:
        """
        Create a new training job.

        Args:
            model_name: Model name (e.g., resnet50, yolov8m, vit-base)
            task_type: classification, object_detection, instance_segmentation, etc.
            dataset_path: Absolute path to dataset directory
            framework: timm, ultralytics, or transformers
            epochs: Number of training epochs (default: 100)
            batch_size: Batch size (default: 32)
            learning_rate: Learning rate (default: 0.001)
            num_classes: Number of classes (auto-detected if None)
            project_id: Project ID to associate with (optional)
            experiment_name: Name for this experiment (optional)

        Returns:
            dict: Created job details with job_id, status, and configuration
        """
        async with get_db() as db:
            job = await training_service.create_job(
                db=db,
                model_name=model_name,
                task_type=task_type,
                dataset_path=dataset_path,
                framework=framework,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_classes=num_classes,
                project_id=project_id,
                experiment_name=experiment_name
            )

            return {
                "job_id": job.id,
                "status": job.status,
                "model": job.model_name,
                "framework": job.framework,
                "task_type": job.task_type,
                "dataset": job.dataset_path,
                "epochs": job.epochs,
                "created_at": job.created_at.isoformat()
            }

    @server.tool()
    async def start_training(
        job_id: int,
        resume_from_checkpoint: str | None = None
    ) -> dict:
        """
        Start a training job.

        Args:
            job_id: Training job ID
            resume_from_checkpoint: Path to checkpoint to resume from (optional)

        Returns:
            dict: Job status and estimated completion time
        """
        async with get_db() as db:
            result = await training_service.start_job(
                db=db,
                job_id=job_id,
                checkpoint_path=resume_from_checkpoint
            )

            return {
                "job_id": job_id,
                "status": "running",
                "process_id": result.process_id,
                "started_at": result.started_at.isoformat(),
                "estimated_duration_minutes": result.epochs * 2  # rough estimate
            }

    @server.tool()
    async def get_training_status(job_id: int) -> dict:
        """
        Get current status and progress of a training job.

        Args:
            job_id: Training job ID

        Returns:
            dict: Status, progress, current metrics, and ETA
        """
        async with get_db() as db:
            status = await training_service.get_job_status(db, job_id)

            return {
                "job_id": job_id,
                "status": status.status,
                "current_epoch": status.current_epoch,
                "total_epochs": status.total_epochs,
                "progress_percent": (status.current_epoch / status.total_epochs * 100)
                    if status.total_epochs > 0 else 0,
                "latest_metrics": status.latest_metrics,
                "estimated_remaining_minutes": status.eta_minutes
            }

    @server.tool()
    async def stop_training(job_id: int, save_checkpoint: bool = True) -> dict:
        """
        Stop a running training job.

        Args:
            job_id: Training job ID
            save_checkpoint: Whether to save checkpoint before stopping

        Returns:
            dict: Final status and checkpoint path if saved
        """
        async with get_db() as db:
            result = await training_service.stop_job(
                db=db,
                job_id=job_id,
                save_checkpoint=save_checkpoint
            )

            return {
                "job_id": job_id,
                "status": "stopped",
                "final_epoch": result.final_epoch,
                "checkpoint_path": result.checkpoint_path if save_checkpoint else None
            }

    @server.tool()
    async def list_training_jobs(
        status: str | None = None,
        project_id: int | None = None,
        limit: int = 20
    ) -> list[dict]:
        """
        List training jobs with optional filters.

        Args:
            status: Filter by status (running, completed, failed, etc.)
            project_id: Filter by project ID
            limit: Maximum number of jobs to return

        Returns:
            list[dict]: List of training jobs
        """
        async with get_db() as db:
            jobs = await training_service.list_jobs(
                db=db,
                status=status,
                project_id=project_id,
                limit=limit
            )

            return [
                {
                    "job_id": job.id,
                    "model": job.model_name,
                    "task_type": job.task_type,
                    "status": job.status,
                    "created_at": job.created_at.isoformat(),
                    "final_metric": job.final_accuracy
                }
                for job in jobs
            ]
```

#### **Dataset Tools 구현**

```python
# mvp/backend/app/mcp/tools/dataset.py

def register_tools(server: Server):

    @server.tool()
    async def analyze_dataset(dataset_path: str) -> dict:
        """
        Analyze dataset structure, format, and quality.

        Args:
            dataset_path: Absolute path to dataset directory

        Returns:
            dict: Dataset analysis including format, classes, distribution, quality
        """
        from app.services.dataset_service import DatasetService

        service = DatasetService()
        analysis = await service.analyze(dataset_path)

        return {
            "path": dataset_path,
            "format": analysis.format,
            "task_type": analysis.inferred_task_type,
            "num_classes": len(analysis.classes),
            "classes": analysis.classes,
            "total_images": analysis.total_images,
            "class_distribution": analysis.class_distribution,
            "imbalance_ratio": analysis.imbalance_ratio,
            "corrupted_files": analysis.corrupted_files,
            "quality_score": analysis.quality_score,
            "recommendations": analysis.recommendations
        }

    @server.tool()
    async def list_datasets(base_path: str = "C:/datasets") -> list[dict]:
        """
        List available datasets in a directory.

        Args:
            base_path: Base directory to search for datasets

        Returns:
            list[dict]: List of found datasets with basic info
        """
        # Implementation
        pass

    @server.tool()
    async def validate_dataset(
        dataset_path: str,
        expected_format: str,
        expected_classes: list[str] | None = None
    ) -> dict:
        """
        Validate dataset against expected structure.

        Args:
            dataset_path: Path to dataset
            expected_format: Expected format (imagefolder, yolo, coco)
            expected_classes: Expected class names (optional)

        Returns:
            dict: Validation results with issues and warnings
        """
        # Implementation
        pass
```

#### **Model Tools 구현**

```python
# mvp/backend/app/mcp/tools/model.py

def register_tools(server: Server):

    @server.tool()
    async def search_models(
        task_type: str | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
        min_priority: int = 0
    ) -> list[dict]:
        """
        Search available models by filters.

        Args:
            task_type: Filter by task type
            framework: Filter by framework
            tags: Filter by tags
            min_priority: Minimum priority score

        Returns:
            list[dict]: Matching models with metadata
        """
        from app.services.model_registry import ModelRegistry

        registry = ModelRegistry()
        models = registry.search(
            task_type=task_type,
            framework=framework,
            tags=tags,
            min_priority=min_priority
        )

        return [
            {
                "name": m.name,
                "framework": m.framework,
                "task_type": m.task_type,
                "description": m.description,
                "priority": m.priority,
                "tags": m.tags
            }
            for m in models
        ]

    @server.tool()
    async def get_model_guide(framework: str, model_name: str) -> dict:
        """
        Get comprehensive guide for a specific model.

        Args:
            framework: Framework name (timm, ultralytics, transformers)
            model_name: Model name

        Returns:
            dict: Complete guide including benchmarks, pros/cons, alternatives
        """
        from app.services.model_registry import ModelRegistry

        registry = ModelRegistry()
        guide = registry.get_guide(framework, model_name)

        return {
            "model": guide.name,
            "framework": guide.framework,
            "description": guide.description,
            "benchmarks": guide.benchmarks,
            "pros": guide.pros,
            "cons": guide.cons,
            "best_for": guide.best_for,
            "alternatives": guide.alternatives,
            "typical_hyperparams": guide.typical_hyperparams
        }

    @server.tool()
    async def compare_models(model_specs: list[dict]) -> dict:
        """
        Compare multiple models side by side.

        Args:
            model_specs: List of model specs [{"framework": "timm", "name": "resnet50"}, ...]

        Returns:
            dict: Comparison table with metrics, speed, accuracy, etc.
        """
        # Implementation
        pass

    @server.tool()
    async def recommend_model(
        task_type: str,
        dataset_size: int,
        priority: str = "balanced",  # speed, accuracy, balanced
        constraints: dict | None = None
    ) -> dict:
        """
        Recommend best model based on requirements.

        Args:
            task_type: Task type
            dataset_size: Number of images in dataset
            priority: What to optimize for (speed, accuracy, balanced)
            constraints: Additional constraints (max_params, max_latency, etc.)

        Returns:
            dict: Recommended model with reasoning
        """
        # Implementation using LLM + Model Registry
        pass
```

#### **Resources 구현**

```python
# mvp/backend/app/mcp/resources/training.py

def register_resources(server: Server):

    @server.resource("training://jobs/{job_id}")
    async def training_job_resource(uri: str) -> str:
        """Detailed training job information"""
        job_id = int(uri.split("/")[-1])

        async with get_db() as db:
            job = await training_service.get_job(db, job_id)

        return f"""
# Training Job #{job.id}

## Configuration
- **Model**: {job.model_name} ({job.framework})
- **Task**: {job.task_type}
- **Dataset**: {job.dataset_path}
- **Format**: {job.dataset_format}
- **Classes**: {job.num_classes}

## Hyperparameters
- **Epochs**: {job.epochs}
- **Batch Size**: {job.batch_size}
- **Learning Rate**: {job.learning_rate}
- **Optimizer**: {job.advanced_config.get('optimizer', 'adam')}
- **Scheduler**: {job.advanced_config.get('scheduler', 'step')}

## Status
- **Current Status**: {job.status}
- **Progress**: {job.current_epoch}/{job.epochs} epochs
- **Started**: {job.started_at}
- **Updated**: {job.updated_at}

## Performance
- **Primary Metric**: {job.primary_metric} = {job.primary_metric_value}
- **Final Accuracy**: {job.final_accuracy}
- **Best Checkpoint**: {job.best_checkpoint_path}

## Outputs
- **Output Directory**: {job.output_dir}
- **MLflow Run**: {job.mlflow_run_id}
"""

    @server.resource("training://jobs/{job_id}/metrics")
    async def training_metrics_resource(uri: str) -> str:
        """Training metrics history"""
        job_id = int(uri.split("/")[-2])

        async with get_db() as db:
            metrics = await training_service.get_metrics(db, job_id)

        output = f"# Training Metrics for Job #{job_id}\n\n"
        output += "| Epoch | Loss | Accuracy | Val Loss | Val Acc |\n"
        output += "|-------|------|----------|----------|----------|\n"

        for m in metrics:
            output += f"| {m.epoch} | {m.loss:.4f} | {m.accuracy:.4f} | {m.val_loss:.4f} | {m.val_accuracy:.4f} |\n"

        return output

    @server.resource("models://catalog")
    async def model_catalog_resource(uri: str) -> str:
        """Complete model catalog"""
        from app.services.model_registry import ModelRegistry

        registry = ModelRegistry()
        models = registry.get_all()

        output = "# Model Catalog\n\n"

        for framework in ["timm", "ultralytics", "transformers"]:
            output += f"## {framework.upper()}\n\n"
            framework_models = [m for m in models if m.framework == framework]

            for model in framework_models:
                output += f"### {model.name}\n"
                output += f"- **Task**: {model.task_type}\n"
                output += f"- **Description**: {model.description}\n"
                output += f"- **Tags**: {', '.join(model.tags)}\n\n"

        return output
```

### 5.3 MCP 클라이언트 사용 예시

**Claude Code에서 사용:**

```bash
# .claude/mcp.json 설정
{
  "mcpServers": {
    "vision-ai-platform": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "C:/Users/flyto/Project/Github/mvp-vision-ai-platform/mvp/backend"
    }
  }
}
```

**사용자와의 대화:**
```
User: Analyze the dataset at C:/datasets/defect and recommend a model

Claude: [Uses analyze_dataset tool]
        [Uses recommend_model tool based on analysis]

        I've analyzed your dataset:
        - Format: ImageFolder
        - Classes: 3 (scratch, dent, normal)
        - Total images: 1,200
        - Distribution: Balanced

        I recommend EfficientNet-B0 because:
        1. Good accuracy for small datasets
        2. Fast training (30% faster than ResNet-50)
        3. Efficient memory usage

        Would you like me to create a training job with this model?

User: Yes, please start training

Claude: [Uses create_training_job tool]
        [Uses start_training tool]

        Training started!
        - Job ID: 12345
        - Model: EfficientNet-B0
        - Estimated completion: 45 minutes
```

**Python API 클라이언트에서 사용:**

```python
# scripts/auto_train.py

from mcp import ClientSession
from mcp.client.stdio import stdio_client
import asyncio

async def auto_train_pipeline(dataset_path: str):
    """Automated training pipeline using MCP"""

    async with stdio_client(
        server_params={
            "command": "python",
            "args": ["-m", "app.mcp.server"],
            "cwd": "C:/path/to/backend"
        }
    ) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()

            # Step 1: Analyze dataset
            print("Analyzing dataset...")
            analysis = await session.call_tool(
                "analyze_dataset",
                {"dataset_path": dataset_path}
            )
            print(f"Found {analysis['num_classes']} classes")

            # Step 2: Get model recommendation
            print("Getting model recommendation...")
            recommendation = await session.call_tool(
                "recommend_model",
                {
                    "task_type": analysis['task_type'],
                    "dataset_size": analysis['total_images'],
                    "priority": "balanced"
                }
            )
            model = recommendation['model']
            framework = recommendation['framework']
            print(f"Recommended: {model}")

            # Step 3: Create training job
            print("Creating training job...")
            job = await session.call_tool(
                "create_training_job",
                {
                    "model_name": model,
                    "task_type": analysis['task_type'],
                    "dataset_path": dataset_path,
                    "framework": framework,
                    "epochs": 100
                }
            )
            job_id = job['job_id']
            print(f"Created job #{job_id}")

            # Step 4: Start training
            print("Starting training...")
            await session.call_tool(
                "start_training",
                {"job_id": job_id}
            )

            # Step 5: Monitor progress
            while True:
                await asyncio.sleep(60)  # Check every minute

                status = await session.call_tool(
                    "get_training_status",
                    {"job_id": job_id}
                )

                print(f"Epoch {status['current_epoch']}/{status['total_epochs']} - "
                      f"Accuracy: {status['latest_metrics'].get('accuracy', 'N/A')}")

                if status['status'] in ['completed', 'failed']:
                    break

            print("Training finished!")
            return job_id

if __name__ == "__main__":
    asyncio.run(auto_train_pipeline("C:/datasets/my_data"))
```

### 5.4 MCP vs Gemini: 사용 사례별 선택

| 사용 사례 | 추천 방식 | 이유 |
|---------|---------|------|
| Web UI에서 초보자 학습 설정 | Gemini State Machine | 단계별 가이드, 명확한 UI |
| Web UI에서 빠른 추론 | Gemini | 간단한 작업, UI 통합 |
| 복잡한 실험 워크플로우 | MCP | 유연한 도구 조합 |
| 자동화 스크립트 | MCP | 프로그래매틱 접근 |
| CI/CD 통합 | MCP | API 기반 제어 |
| Claude Code에서 작업 | MCP | 네이티브 지원 |
| 배치 작업 자동화 | MCP | 효율적 도구 호출 |
| 결과 분석 및 해석 | Gemini 또는 MCP Prompts | LLM 추론 능력 활용 |

---

## 6. 구현 로드맵

### Phase 1: Gemini Track 확장 (2주)

**목표**: 현재 State Machine을 확장하여 모든 기능 커버

**작업:**

1. **State 및 Action 확장**
   - [ ] 새로운 State 추가 (ANALYZING_DATASET, RUNNING_INFERENCE 등)
   - [ ] 새로운 Action 추가 (ANALYZE_DATASET, START_INFERENCE 등)
   - [ ] Multi-Intent 지원 로직 구현

2. **Tool Registry 구현**
   - [ ] ToolRegistry 클래스 작성
   - [ ] 각 API 엔드포인트를 도구로 래핑
   - [ ] 권한 검증 로직 추가
   - [ ] 도구 호출 로깅

3. **프롬프트 개선**
   - [ ] System Prompt 업데이트 (모든 기능 포함)
   - [ ] Few-shot Examples 추가
   - [ ] 한국어 응답 품질 향상

4. **컨텍스트 관리 강화**
   - [ ] ConversationContext 확장
   - [ ] 엔티티 추적 (current_training_job 등)
   - [ ] 사용자 선호도 저장

5. **Frontend 업데이트**
   - [ ] ChatPanel에서 새 Action 처리
   - [ ] 데이터셋 분석 결과 표시
   - [ ] 모델 추천 카드 표시
   - [ ] 추론 결과 표시

**산출물:**
- 확장된 State Machine (`llm_structured.py`)
- Tool Registry (`tool_registry.py`)
- 업데이트된 Action Handlers (`action_handlers.py`)
- Frontend 컴포넌트 업데이트

### Phase 2: MCP Server 구현 (3주)

**목표**: MCP 서버 구축 및 핵심 도구 구현

**작업:**

1. **MCP Server 기본 구조**
   - [ ] `mvp/backend/app/mcp/` 디렉토리 생성
   - [ ] MCP Server 초기화 (`server.py`)
   - [ ] FastAPI와 통합
   - [ ] 의존성 추가 (mcp SDK)

2. **Training Tools**
   - [ ] `create_training_job` 도구
   - [ ] `start_training` 도구
   - [ ] `stop_training` 도구
   - [ ] `get_training_status` 도구
   - [ ] `list_training_jobs` 도구

3. **Inference Tools**
   - [ ] `run_quick_inference` 도구
   - [ ] `run_batch_inference` 도구
   - [ ] `get_inference_results` 도구

4. **Dataset Tools**
   - [ ] `analyze_dataset` 도구
   - [ ] `list_datasets` 도구
   - [ ] `validate_dataset` 도구

5. **Model Tools**
   - [ ] `search_models` 도구
   - [ ] `get_model_guide` 도구
   - [ ] `compare_models` 도구
   - [ ] `recommend_model` 도구

6. **Project Tools**
   - [ ] `create_project` 도구
   - [ ] `list_projects` 도구
   - [ ] `list_experiments` 도구

**산출물:**
- 완전한 MCP Server
- 20+ 도구 구현
- API 문서

### Phase 3: MCP Resources & Prompts (1주)

**목표**: Resources와 Prompts 구현

**작업:**

1. **Resources**
   - [ ] `training://jobs/{job_id}` 리소스
   - [ ] `training://jobs/{job_id}/metrics` 리소스
   - [ ] `validation://jobs/{job_id}/results` 리소스
   - [ ] `models://catalog` 리소스
   - [ ] `datasets://analysis/{path}` 리소스

2. **Prompts**
   - [ ] `model_recommendation` 프롬프트
   - [ ] `hyperparameter_tuning` 프롬프트
   - [ ] `dataset_quality_check` 프롬프트
   - [ ] `training_troubleshooting` 프롬프트
   - [ ] `result_interpretation` 프롬프트

**산출물:**
- 5+ Resources
- 5+ Prompts
- 사용 가이드 문서

### Phase 4: 통합 및 테스트 (2주)

**목표**: 두 트랙 통합 및 엔드투엔드 테스트

**작업:**

1. **통합 테스트**
   - [ ] Gemini Track 엔드투엔드 시나리오
   - [ ] MCP Track 엔드투엔드 시나리오
   - [ ] 복합 시나리오 테스트

2. **성능 최적화**
   - [ ] LLM 호출 캐싱
   - [ ] 응답 시간 최적화
   - [ ] 도구 호출 병렬화

3. **에러 처리 강화**
   - [ ] Fallback 메커니즘
   - [ ] 재시도 로직
   - [ ] 명확한 에러 메시지

4. **문서화**
   - [ ] API 문서 업데이트
   - [ ] MCP 사용 가이드 작성
   - [ ] 예제 시나리오 문서
   - [ ] 트러블슈팅 가이드

5. **Claude Code 통합**
   - [ ] `.claude/mcp.json` 설정
   - [ ] Claude Code에서 테스트
   - [ ] 사용 예제 작성

**산출물:**
- 테스트 결과 보고서
- 성능 벤치마크
- 완전한 문서 세트
- Claude Code 통합 가이드

### Phase 5: 고급 기능 (Optional, 2주)

**목표**: 고급 기능 추가

**작업:**

1. **자동화 워크플로우**
   - [ ] AutoML 파이프라인 (데이터셋 → 분석 → 추천 → 학습)
   - [ ] 하이퍼파라미터 자동 튜닝
   - [ ] 앙상블 모델 자동 생성

2. **고급 컨텍스트 관리**
   - [ ] 장기 메모리 (Vector DB)
   - [ ] 사용자 프로파일링
   - [ ] 추천 시스템

3. **멀티모달 입력**
   - [ ] 이미지 업로드 → 유사 데이터셋 추천
   - [ ] 스크린샷 → 에러 해결

4. **협업 기능**
   - [ ] 팀 대화 세션
   - [ ] 실험 공유 및 댓글
   - [ ] 자동 보고서 생성

**산출물:**
- AutoML 파이프라인
- 고급 기능 문서

---

## 7. 기술 스택 및 의사결정

### 7.1 LLM 선택

| LLM | 용도 | 장점 | 단점 | 비용 |
|-----|------|------|------|------|
| **Google Gemini 1.5 Pro** | Gemini Track (Web UI) | • 한국어 우수<br>• 빠른 응답<br>• 저렴 | • Tool calling 제한적<br>• Context window 작음 | $0.00025/1K tokens |
| **Claude 3.5 Sonnet** | MCP Track (API) | • Tool calling 우수<br>• Reasoning 우수<br>• 긴 context | • 한국어 보통<br>• 비교적 비쌈 | $3/1M input, $15/1M output |
| **GPT-4 Turbo** | Fallback | • 범용성 우수<br>• JSON mode | • 비쌈<br>• 느림 | $10/1M input, $30/1M output |

**결정**:
- **Gemini**: Web UI (현재 유지)
- **Claude via MCP**: API/Claude Code (신규)
- **GPT-4**: Fallback (선택적)

### 7.2 MCP SDK

**선택**: `mcp` Python SDK (Anthropic 공식)

**이유**:
- 공식 지원
- Claude Desktop/Code 네이티브 통합
- 활발한 커뮤니티
- 명확한 문서

**대안**:
- 직접 구현 (JSON-RPC): 유연하지만 관리 부담

### 7.3 컨텍스트 저장소

**현재**: PostgreSQL (Session.temp_data JSONB)

**고려 사항**:
- 단기: 현재 방식 유지
- 장기: Vector DB (Chroma, Pinecone) 추가 (장기 메모리, 유사 실험 검색)

### 7.4 캐싱 전략

**LLM 응답 캐싱**:
- Redis에 (prompt_hash, response) 저장
- TTL: 1시간
- Cache hit 시 즉시 응답

**도구 결과 캐싱**:
- 불변 데이터만 캐싱 (모델 카탈로그, 데이터셋 분석)
- 가변 데이터는 캐싱 안 함 (학습 상태)

### 7.5 모니터링 및 로깅

**LLM 호출 로그**:
- 모든 LLM 호출 기록 (prompt, response, latency)
- 비용 추적
- 실패율 모니터링

**도구 사용 통계**:
- 도구별 호출 빈도
- 성공률
- 평균 응답 시간

**인텐트 분포**:
- 사용자 인텐트 카테고리 분포
- 인기 있는 워크플로우 파악

---

## 8. 보안 및 권한

### 8.1 사용자 권한 검증

모든 도구 호출 전 권한 검증:

```python
async def verify_permission(user_id: int, tool_name: str, resource_id: int) -> bool:
    """
    사용자가 리소스에 대해 도구를 실행할 권한이 있는지 확인
    """
    # 예: 다른 사용자의 학습 작업 중지 방지
    if tool_name == "stop_training":
        job = await get_training_job(resource_id)
        if job.creator_id != user_id:
            # 프로젝트 멤버인지 확인
            if not await is_project_member(user_id, job.project_id):
                return False

    return True
```

### 8.2 민감 정보 마스킹

LLM에게 전달되는 데이터에서 민감 정보 제거:

```python
def mask_sensitive_info(data: dict) -> dict:
    """민감한 정보 마스킹"""
    masked = data.copy()

    # 절대 경로 → 상대 경로
    if "dataset_path" in masked:
        masked["dataset_path"] = Path(masked["dataset_path"]).name

    # 사용자 이메일 → 사용자 ID
    if "user_email" in masked:
        del masked["user_email"]

    return masked
```

### 8.3 Rate Limiting

LLM 호출 남용 방지:

```python
# Redis를 이용한 Rate Limiting
async def check_rate_limit(user_id: int) -> bool:
    """
    사용자당 분당 10회 LLM 호출 제한
    """
    key = f"llm_rate_limit:{user_id}"
    count = await redis.incr(key)

    if count == 1:
        await redis.expire(key, 60)  # 1분

    return count <= 10
```

---

## 9. 성공 지표 (KPI)

### 9.1 사용성 지표

- **Task Completion Rate**: 사용자가 자연어로 원하는 작업을 완료한 비율
  - 목표: 85% 이상

- **Average Turns to Completion**: 작업 완료까지 평균 대화 턴 수
  - 목표: 학습 설정 5턴 이내, 추론 3턴 이내

- **Clarification Rate**: LLM이 추가 질문을 하는 비율
  - 목표: 30% 이하 (너무 많은 질문은 UX 저하)

### 9.2 성능 지표

- **LLM Response Time**: LLM 응답 시간
  - 목표: P95 < 3초

- **Tool Execution Time**: 도구 실행 시간
  - 목표: 각 도구별 SLA 준수

- **End-to-End Latency**: 사용자 입력 → 최종 응답
  - 목표: P95 < 5초

### 9.3 품질 지표

- **Intent Recognition Accuracy**: 인텐트 정확도
  - 목표: 90% 이상
  - 측정: Human evaluation

- **Tool Selection Accuracy**: 올바른 도구 선택 비율
  - 목표: 95% 이상

- **Error Recovery Rate**: 에러 발생 후 복구 성공률
  - 목표: 80% 이상

### 9.4 비용 지표

- **Cost per Conversation**: 대화당 LLM 비용
  - 목표: $0.05 이하

- **Cache Hit Rate**: 캐시 적중률
  - 목표: 40% 이상

---

## 10. 리스크 및 완화 방안

| 리스크 | 영향 | 확률 | 완화 방안 |
|--------|------|------|----------|
| LLM이 잘못된 도구 선택 | 높음 | 중간 | • Prompt engineering<br>• Few-shot examples<br>• Tool description 개선<br>• Validation 로직 |
| LLM 응답 지연 | 중간 | 높음 | • 캐싱<br>• 스트리밍 응답<br>• 타임아웃 처리 |
| LLM API 장애 | 높음 | 낮음 | • Fallback LLM<br>• 에러 메시지<br>• 재시도 로직 |
| 보안 취약점 (권한 우회) | 높음 | 낮음 | • 철저한 권한 검증<br>• Security audit<br>• 입력 검증 |
| 비용 폭증 | 중간 | 중간 | • Rate limiting<br>• 예산 알림<br>• 캐싱 강화 |
| 확장성 문제 (동시 사용자) | 중간 | 낮음 | • 로드 밸런싱<br>• 비동기 처리<br>• 큐 시스템 |

---

## 11. 다음 단계

### 즉시 실행

1. **이 문서 검토 및 승인**
   - 팀 리뷰
   - 피드백 반영
   - 최종 승인

2. **Phase 1 시작**
   - State Machine 확장 설계
   - Tool Registry 설계
   - 작업 티켓 생성

3. **MCP 프로토타입**
   - 간단한 MCP 서버 구현
   - 2-3개 도구로 PoC
   - Claude Code에서 테스트

### 향후 고려사항

- **다국어 지원 확대**: 영어, 일본어 등
- **음성 인터페이스**: 음성 → 텍스트 → LLM
- **멀티모달**: 이미지 입력으로 데이터셋 분석
- **AutoML**: 완전 자동화 파이프라인
- **Collaboration**: 팀 대화, 공유 실험

---

## 부록

### A. 참고 문서

- [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) - 시스템 아키텍처
- [API_SPECIFICATION.md](../api/API_SPECIFICATION.md) - API 명세
- [MCP 공식 문서](https://www.anthropic.com/news/model-context-protocol)
- [Gemini API 문서](https://ai.google.dev/docs)

### B. 용어 정의

- **Intent**: 사용자의 의도 (학습, 추론, 분석 등)
- **Action**: LLM이 수행할 작업 (API 호출, 질문, 확인 등)
- **Tool**: MCP에서 LLM이 호출할 수 있는 함수
- **Resource**: MCP에서 LLM이 읽을 수 있는 구조화된 데이터
- **Prompt**: MCP에서 제공하는 프롬프트 템플릿
- **State Machine**: 대화 흐름을 관리하는 상태 기계

### C. 예제 프롬프트

#### System Prompt for Gemini (Expanded)

```
You are an AI assistant for a Vision AI Training Platform. You help users:
1. Configure and train computer vision models
2. Run inference on images
3. Analyze datasets
4. Search and compare models
5. Manage projects and experiments

Available tools:
- analyze_dataset(path): Analyze dataset structure and quality
- search_models(task_type, framework): Search available models
- get_model_guide(framework, model): Get model information
- create_training_job(config): Create a training job
- start_training(job_id): Start training
- get_training_status(job_id): Check training progress
- run_inference(job_id, images): Run inference
- ... (more tools)

When the user asks something:
1. Understand their intent
2. Gather necessary information
3. Call appropriate tools
4. Provide clear, helpful responses in Korean

Always be proactive and helpful. If the user's request is unclear, ask clarifying questions.
```

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 | 작성자 |
|------|------|----------|--------|
| 1.0 | 2025-11-01 | 초안 작성 | Claude Code |

