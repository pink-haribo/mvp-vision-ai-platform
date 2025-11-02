# Gemini Track 확장 가이드

## 개요

현재 구현된 Gemini State Machine을 확장하여 모든 플랫폼 기능을 자연어로 제어할 수 있도록 개선하는 실무 가이드입니다.

**목적**: Phase 1 구현 (Gemini Track 확장)
**대상**: Backend 개발자 (FastAPI, Python)
**예상 소요 시간**: 2주

---

## 목차

1. [현재 구조 분석](#1-현재-구조-분석)
2. [확장 계획](#2-확장-계획)
3. [State 추가 방법](#3-state-추가-방법)
4. [Action 추가 방법](#4-action-추가-방법)
5. [Tool Registry 구현](#5-tool-registry-구현)
6. [Multi-Intent 지원](#6-multi-intent-지원)
7. [Frontend 연동](#7-frontend-연동)
8. [테스트](#8-테스트)

---

## 1. 현재 구조 분석

### 1.1 파일 구조

```
mvp/backend/app/
├── api/chat.py                    # Chat API 엔드포인트
├── utils/
│   ├── llm_structured.py          # Structured Intent Parser (현재 활성)
│   ├── conversation_manager.py    # 대화 흐름 관리
│   └── action_handlers.py         # Action 실행 로직
├── db/models.py                   # Session, Message 모델
└── schemas/chat.py                # Request/Response 스키마
```

### 1.2 현재 State Machine

**States** (`mvp/backend/app/utils/llm_structured.py`):
```python
class ConversationState(Enum):
    INITIAL = "initial"
    GATHERING_CONFIG = "gathering_config"
    SELECTING_PROJECT = "selecting_project"
    CREATING_PROJECT = "creating_project"
    CONFIRMING = "confirming"
    TRAINING = "training"
    COMPLETED = "completed"
    ERROR = "error"
```

**Actions** (`mvp/backend/app/utils/action_handlers.py`):
```python
class ActionType(Enum):
    ASK_CLARIFICATION = "ask_clarification"
    SHOW_PROJECT_OPTIONS = "show_project_options"
    SHOW_PROJECT_LIST = "show_project_list"
    CREATE_PROJECT = "create_project"
    SELECT_PROJECT = "select_project"
    SKIP_PROJECT = "skip_project"
    CONFIRM_TRAINING = "confirm_training"
    START_TRAINING = "start_training"
    ERROR = "error"
```

### 1.3 현재 흐름

```
사용자: "ResNet50으로 학습해줘"
    ↓
LLM Parse (llm_structured.py)
    ↓
Action: ASK_CLARIFICATION
State: GATHERING_CONFIG
    ↓
사용자: "C:/datasets/cats, 3개 클래스"
    ↓
LLM Parse
    ↓
Action: SHOW_PROJECT_OPTIONS
State: SELECTING_PROJECT
    ↓
사용자: "새 프로젝트"
    ↓
Action: CREATE_PROJECT
State: CREATING_PROJECT
    ↓
Action: CONFIRM_TRAINING
State: CONFIRMING
    ↓
사용자: "네"
    ↓
Action: START_TRAINING
State: TRAINING
    ↓
Action: TRAINING_STARTED
State: COMPLETED
```

### 1.4 현재 제약사항

1. **학습 설정에만 집중** - 추론, 데이터셋 분석 등 지원 안 함
2. **단방향 흐름** - 중간에 다른 작업으로 전환 불가
3. **도구 제한** - LLM이 직접 API 호출 못함 (ActionHandler 경유)
4. **컨텍스트 제한** - temp_data에만 저장, 장기 기억 부족

---

## 2. 확장 계획

### 2.1 새로운 States

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
    ANALYZING_DATASET = "analyzing_dataset"          # 데이터셋 분석 중
    SELECTING_MODEL = "selecting_model"              # 모델 선택 중
    COMPARING_MODELS = "comparing_models"            # 모델 비교 중
    MONITORING_TRAINING = "monitoring_training"      # 학습 모니터링 중
    RUNNING_INFERENCE = "running_inference"          # 추론 실행 중
    VIEWING_RESULTS = "viewing_results"              # 결과 조회 중
    MANAGING_EXPERIMENTS = "managing_experiments"    # 실험 관리 중
    IDLE = "idle"                                    # 대기 (다음 작업 대기)
```

### 2.2 새로운 Actions

```python
class ActionType(Enum):
    # 기존
    ASK_CLARIFICATION = "ask_clarification"
    SHOW_PROJECT_OPTIONS = "show_project_options"
    SHOW_PROJECT_LIST = "show_project_list"
    CREATE_PROJECT = "create_project"
    SELECT_PROJECT = "select_project"
    SKIP_PROJECT = "skip_project"
    CONFIRM_TRAINING = "confirm_training"
    START_TRAINING = "start_training"
    ERROR = "error"

    # 데이터셋 관련
    ANALYZE_DATASET = "analyze_dataset"
    SHOW_DATASET_ANALYSIS = "show_dataset_analysis"
    LIST_DATASETS = "list_datasets"

    # 모델 관련
    SEARCH_MODELS = "search_models"
    SHOW_MODEL_INFO = "show_model_info"
    COMPARE_MODELS = "compare_models"
    RECOMMEND_MODELS = "recommend_models"

    # 학습 제어
    STOP_TRAINING = "stop_training"
    RESUME_TRAINING = "resume_training"
    SHOW_TRAINING_STATUS = "show_training_status"
    LIST_TRAINING_JOBS = "list_training_jobs"

    # 추론
    START_QUICK_INFERENCE = "start_quick_inference"
    START_BATCH_INFERENCE = "start_batch_inference"
    SHOW_INFERENCE_RESULTS = "show_inference_results"

    # 결과 조회
    SHOW_VALIDATION_RESULTS = "show_validation_results"
    SHOW_CONFUSION_MATRIX = "show_confusion_matrix"
    EXPORT_RESULTS = "export_results"

    # 실험 관리
    LIST_EXPERIMENTS = "list_experiments"
    COMPARE_EXPERIMENTS = "compare_experiments"
    DELETE_EXPERIMENT = "delete_experiment"

    # 일반
    SHOW_HELP = "show_help"
    RESET_CONVERSATION = "reset_conversation"
```

### 2.3 Intent Categories

```python
class IntentCategory(Enum):
    TRAINING = "training"          # 학습 관련
    INFERENCE = "inference"         # 추론 관련
    DATASET = "dataset"            # 데이터셋 관련
    MODEL = "model"                # 모델 관련
    PROJECT = "project"            # 프로젝트 관련
    RESULTS = "results"            # 결과 관련
    GENERAL = "general"            # 일반
```

---

## 3. State 추가 방법

### 3.1 State Enum 업데이트

**파일**: `mvp/backend/app/utils/llm_structured.py`

```python
class ConversationState(Enum):
    # ... 기존 states ...

    # 새 State 추가
    ANALYZING_DATASET = "analyzing_dataset"
```

### 3.2 State Transition 정의

**State Transition Map**:

```python
# mvp/backend/app/utils/state_machine.py (새 파일)

class StateMachine:
    """State transition logic"""

    # 허용된 전환
    ALLOWED_TRANSITIONS = {
        ConversationState.INITIAL: [
            ConversationState.GATHERING_CONFIG,
            ConversationState.ANALYZING_DATASET,
            ConversationState.SELECTING_MODEL,
            ConversationState.RUNNING_INFERENCE,
            ConversationState.MANAGING_EXPERIMENTS
        ],
        ConversationState.GATHERING_CONFIG: [
            ConversationState.SELECTING_PROJECT,
            ConversationState.ANALYZING_DATASET,  # 중간에 분석 가능
            ConversationState.CONFIRMING,
            ConversationState.INITIAL  # 취소
        ],
        ConversationState.ANALYZING_DATASET: [
            ConversationState.SELECTING_MODEL,
            ConversationState.GATHERING_CONFIG,
            ConversationState.INITIAL
        ],
        # ... 나머지 전환들 ...
    }

    @classmethod
    def can_transition(cls, from_state: ConversationState, to_state: ConversationState) -> bool:
        """Check if transition is allowed"""
        allowed = cls.ALLOWED_TRANSITIONS.get(from_state, [])
        return to_state in allowed

    @classmethod
    def transition(cls, session: Session, new_state: ConversationState) -> bool:
        """Perform state transition"""
        if not cls.can_transition(session.state, new_state):
            raise ValueError(f"Invalid transition from {session.state} to {new_state}")

        session.state = new_state.value
        return True
```

### 3.3 State별 System Prompt

각 State마다 다른 System Prompt를 사용하여 컨텍스트 유지:

```python
# mvp/backend/app/utils/prompts.py (새 파일)

STATE_PROMPTS = {
    ConversationState.INITIAL: """
You are helping a user configure a computer vision training job.
The user has just started. Ask what they want to do.

Available actions:
- Start training configuration
- Analyze dataset
- Search models
- Run inference
- Manage experiments
""",

    ConversationState.GATHERING_CONFIG: """
You are gathering training configuration.

Current config:
{current_config}

Missing fields:
{missing_fields}

Ask for missing information or offer to analyze dataset if not provided.
""",

    ConversationState.ANALYZING_DATASET: """
Dataset analysis is in progress or completed.

Analysis result:
{analysis_result}

Based on this, suggest next steps:
- Recommend models
- Proceed with training
- Adjust configuration
""",

    # ... 다른 states ...
}

def get_system_prompt(state: ConversationState, context: dict) -> str:
    """Get system prompt for current state"""
    template = STATE_PROMPTS[state]
    return template.format(**context)
```

---

## 4. Action 추가 방법

### 4.1 ActionType Enum 업데이트

```python
# mvp/backend/app/utils/action_handlers.py

class ActionType(Enum):
    # ... 기존 actions ...

    # 새 Action 추가
    ANALYZE_DATASET = "analyze_dataset"
```

### 4.2 Action Handler 구현

```python
# mvp/backend/app/utils/action_handlers.py

class ActionHandlers:
    """Execute actions returned by LLM"""

    def __init__(self, db: Session, user_id: int):
        self.db = db
        self.user_id = user_id

    async def execute(self, action: dict, session: Session) -> dict:
        """Execute an action"""
        action_type = ActionType(action["action"])

        # Route to appropriate handler
        handler = getattr(self, f"handle_{action_type.value}", None)
        if not handler:
            raise ValueError(f"No handler for action: {action_type}")

        return await handler(action, session)

    # 기존 handlers...

    async def handle_analyze_dataset(self, action: dict, session: Session) -> dict:
        """
        Handle ANALYZE_DATASET action.

        Action format:
        {
            "action": "analyze_dataset",
            "dataset_path": "C:/datasets/cats"
        }
        """
        from app.services.dataset_service import DatasetService

        dataset_path = action.get("dataset_path")
        if not dataset_path:
            return {
                "error": "dataset_path is required",
                "message": "데이터셋 경로를 제공해주세요."
            }

        # Call Dataset Service
        service = DatasetService()
        try:
            analysis = await service.analyze(dataset_path)

            # Save to session temp_data
            session.temp_data = session.temp_data or {}
            session.temp_data["dataset_analysis"] = {
                "path": dataset_path,
                "format": analysis.format,
                "num_classes": len(analysis.classes),
                "classes": analysis.classes,
                "total_images": analysis.total_images,
                "class_distribution": analysis.class_distribution,
                "imbalance_ratio": analysis.imbalance_ratio,
                "quality_score": analysis.quality_score,
                "recommendations": analysis.recommendations
            }

            # Transition state
            session.state = ConversationState.ANALYZING_DATASET.value

            return {
                "action": "show_dataset_analysis",
                "analysis": session.temp_data["dataset_analysis"],
                "message": self._format_dataset_analysis(analysis)
            }

        except Exception as e:
            return {
                "error": str(e),
                "message": f"데이터셋 분석 중 오류가 발생했습니다: {str(e)}"
            }

    def _format_dataset_analysis(self, analysis) -> str:
        """Format dataset analysis for display"""
        msg = f"""
📊 데이터셋 분석 결과:

**기본 정보:**
- 포맷: {analysis.format}
- 클래스: {len(analysis.classes)}개
- 총 이미지: {analysis.total_images}장

**클래스 분포:**
"""
        for cls, count in analysis.class_distribution.items():
            pct = count / analysis.total_images * 100
            msg += f"- {cls}: {count}장 ({pct:.1f}%)\n"

        msg += f"""
**품질 평가:**
- 품질 점수: {analysis.quality_score}/100
- 불균형 비율: {analysis.imbalance_ratio:.2f}

**권장 사항:**
"""
        for rec in analysis.recommendations:
            msg += f"⚠️ {rec}\n"

        return msg

    async def handle_search_models(self, action: dict, session: Session) -> dict:
        """Handle SEARCH_MODELS action"""
        from app.services.model_registry import ModelRegistry

        task_type = action.get("task_type")
        framework = action.get("framework")
        tags = action.get("tags", [])

        registry = ModelRegistry()
        models = registry.search(
            task_type=task_type,
            framework=framework,
            tags=tags
        )

        # Save to session
        session.temp_data = session.temp_data or {}
        session.temp_data["searched_models"] = [
            {
                "name": m.name,
                "framework": m.framework,
                "task_type": m.task_type,
                "description": m.description,
                "priority": m.priority
            }
            for m in models[:10]  # Top 10
        ]

        return {
            "action": "show_models",
            "models": session.temp_data["searched_models"],
            "message": self._format_model_list(models[:10])
        }

    def _format_model_list(self, models) -> str:
        """Format model list for display"""
        msg = f"검색된 모델 ({len(models)}개):\n\n"
        for i, m in enumerate(models, 1):
            msg += f"{i}. **{m.name}** ({m.framework})\n"
            msg += f"   {m.description}\n"
            msg += f"   우선순위: {m.priority}/10\n\n"
        return msg

    async def handle_start_quick_inference(self, action: dict, session: Session) -> dict:
        """Handle START_QUICK_INFERENCE action"""
        from app.api.test_inference import run_quick_inference_endpoint

        job_id = action.get("job_id")
        image_path = action.get("image_path")

        # Call inference endpoint
        result = await run_quick_inference_endpoint(
            job_id=job_id,
            image_path=image_path,
            db=self.db
        )

        return {
            "action": "show_inference_results",
            "result": result,
            "message": self._format_inference_result(result)
        }

    def _format_inference_result(self, result) -> str:
        """Format inference result"""
        if result.get("task_type") == "classification":
            msg = "분류 결과:\n"
            for i, pred in enumerate(result["predictions"][:3], 1):
                msg += f"{i}. {pred['class']} ({pred['confidence']*100:.1f}%)\n"
        elif result.get("task_type") == "object_detection":
            msg = f"검출 결과:\n"
            msg += f"총 {len(result['detections'])}개 객체 검출\n"
            for det in result['detections'][:5]:
                msg += f"- {det['class']} (신뢰도: {det['confidence']*100:.1f}%)\n"
        else:
            msg = "추론이 완료되었습니다."

        return msg
```

### 4.3 Action → Frontend 매핑

Frontend에서 Action을 받아 UI 업데이트:

```typescript
// mvp/frontend/components/ChatPanel.tsx

const handleActionResponse = (action: ActionResponse) => {
  switch (action.action) {
    case "show_dataset_analysis":
      // 데이터셋 분석 결과 표시
      setDatasetAnalysis(action.analysis);
      break;

    case "show_models":
      // 모델 목록 카드 표시
      setModelList(action.models);
      break;

    case "show_inference_results":
      // 추론 결과 표시
      setInferenceResults(action.result);
      break;

    case "confirm_training":
      // 확인 다이얼로그 표시
      setShowConfirmDialog(true);
      setTrainingConfig(action.config);
      break;

    // ... 기타 actions
  }
};
```

---

## 5. Tool Registry 구현

LLM이 필요한 도구를 호출할 수 있도록 Tool Registry 구현:

### 5.1 Tool Registry 클래스

```python
# mvp/backend/app/utils/tool_registry.py

from typing import Callable, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    DATASET = "dataset"
    MODEL = "model"
    PROJECT = "project"
    RESULTS = "results"

class Tool:
    """Tool definition"""

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        handler: Callable,
        parameters: Dict[str, Any],
        requires_auth: bool = True
    ):
        self.name = name
        self.description = description
        self.category = category
        self.handler = handler
        self.parameters = parameters
        self.requires_auth = requires_auth

    def to_dict(self) -> dict:
        """Convert to LLM-friendly format"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters
        }

class ToolRegistry:
    """Central registry of all available tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def register(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool:
        """Get tool by name"""
        return self.tools.get(name)

    def list_by_category(self, category: ToolCategory) -> list[Tool]:
        """List tools by category"""
        return [t for t in self.tools.values() if t.category == category]

    def get_all_descriptions(self) -> str:
        """Get all tool descriptions for LLM prompt"""
        desc = "Available tools:\n\n"
        for category in ToolCategory:
            tools = self.list_by_category(category)
            if tools:
                desc += f"## {category.value.upper()}\n"
                for tool in tools:
                    desc += f"- **{tool.name}**: {tool.description}\n"
                    desc += f"  Parameters: {tool.parameters}\n\n"
        return desc

    async def call_tool(
        self,
        tool_name: str,
        parameters: dict,
        user_id: int,
        db: Session
    ) -> Any:
        """Call a tool"""
        tool = self.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        # Permission check
        if tool.requires_auth:
            # TODO: Check user permissions
            pass

        # Validate parameters
        # TODO: Parameter validation

        # Execute
        logger.info(f"Executing tool: {tool_name} with params: {parameters}")
        try:
            result = await tool.handler(parameters, user_id, db)
            logger.info(f"Tool {tool_name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {str(e)}")
            raise

    def _register_default_tools(self):
        """Register all default tools"""

        # Training tools
        self.register(Tool(
            name="create_training_job",
            description="Create a new training job with specified configuration",
            category=ToolCategory.TRAINING,
            handler=self._create_training_job,
            parameters={
                "model_name": "str (required)",
                "task_type": "str (required)",
                "dataset_path": "str (required)",
                "epochs": "int (optional, default: 100)",
                "batch_size": "int (optional, default: 32)",
                "learning_rate": "float (optional, default: 0.001)"
            }
        ))

        self.register(Tool(
            name="start_training",
            description="Start a training job",
            category=ToolCategory.TRAINING,
            handler=self._start_training,
            parameters={
                "job_id": "int (required)"
            }
        ))

        self.register(Tool(
            name="get_training_status",
            description="Get current status of a training job",
            category=ToolCategory.TRAINING,
            handler=self._get_training_status,
            parameters={
                "job_id": "int (required)"
            }
        ))

        # Dataset tools
        self.register(Tool(
            name="analyze_dataset",
            description="Analyze dataset structure, format, and quality",
            category=ToolCategory.DATASET,
            handler=self._analyze_dataset,
            parameters={
                "dataset_path": "str (required)"
            }
        ))

        # Model tools
        self.register(Tool(
            name="search_models",
            description="Search available models by filters",
            category=ToolCategory.MODEL,
            handler=self._search_models,
            parameters={
                "task_type": "str (optional)",
                "framework": "str (optional)",
                "tags": "list[str] (optional)"
            }
        ))

        # Inference tools
        self.register(Tool(
            name="run_quick_inference",
            description="Run quick inference on a single image",
            category=ToolCategory.INFERENCE,
            handler=self._run_quick_inference,
            parameters={
                "job_id": "int (required)",
                "image_path": "str (required)"
            }
        ))

        # ... 더 많은 tools ...

    async def _create_training_job(self, params: dict, user_id: int, db: Session):
        """Handler for create_training_job"""
        from app.services.training_service import TrainingService
        service = TrainingService()
        return await service.create_job(db=db, user_id=user_id, **params)

    async def _start_training(self, params: dict, user_id: int, db: Session):
        """Handler for start_training"""
        from app.services.training_service import TrainingService
        service = TrainingService()
        return await service.start_job(db=db, **params)

    async def _get_training_status(self, params: dict, user_id: int, db: Session):
        """Handler for get_training_status"""
        from app.services.training_service import TrainingService
        service = TrainingService()
        return await service.get_job_status(db=db, **params)

    async def _analyze_dataset(self, params: dict, user_id: int, db: Session):
        """Handler for analyze_dataset"""
        from app.services.dataset_service import DatasetService
        service = DatasetService()
        return await service.analyze(params["dataset_path"])

    async def _search_models(self, params: dict, user_id: int, db: Session):
        """Handler for search_models"""
        from app.services.model_registry import ModelRegistry
        registry = ModelRegistry()
        return registry.search(**params)

    async def _run_quick_inference(self, params: dict, user_id: int, db: Session):
        """Handler for run_quick_inference"""
        # Implementation
        pass
```

### 5.2 LLM에 Tool 정보 전달

System Prompt에 Tool 목록 포함:

```python
# mvp/backend/app/utils/llm_structured.py

def get_system_prompt_with_tools(state: ConversationState, tool_registry: ToolRegistry) -> str:
    """Generate system prompt with available tools"""

    base_prompt = f"""
You are an AI assistant for a computer vision training platform.

Current state: {state.value}

{tool_registry.get_all_descriptions()}

When the user requests an action:
1. Identify the intent
2. Select appropriate tool(s)
3. Extract parameters from user message
4. Return structured action with tool call

Response format:
{{
    "intent": "TRAINING.CREATE" | "INFERENCE.QUICK" | ...,
    "action": "create_training_job" | "analyze_dataset" | ...,
    "parameters": {{}},
    "message": "User-friendly message in Korean",
    "next_state": "gathering_config" | "analyzing_dataset" | ...
}}

Always respond in Korean and be helpful.
"""

    return base_prompt
```

---

## 6. Multi-Intent 지원

여러 인텐트를 순차적으로 또는 병렬로 처리:

### 6.1 Intent Queue

```python
# mvp/backend/app/utils/intent_queue.py

from collections import deque
from typing import List, Dict

class IntentQueue:
    """Queue for managing multiple intents"""

    def __init__(self):
        self.queue = deque()
        self.history = []

    def enqueue(self, intent: Dict):
        """Add intent to queue"""
        self.queue.append(intent)

    def dequeue(self) -> Dict:
        """Get next intent"""
        if self.queue:
            intent = self.queue.popleft()
            self.history.append(intent)
            return intent
        return None

    def peek(self) -> Dict:
        """Peek at next intent without removing"""
        return self.queue[0] if self.queue else None

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.queue) == 0

    def clear(self):
        """Clear queue"""
        self.queue.clear()
```

### 6.2 Multi-Intent Parser

```python
# mvp/backend/app/utils/multi_intent_parser.py

class MultiIntentParser:
    """Parse and handle multiple intents in one message"""

    def __init__(self, llm, tool_registry):
        self.llm = llm
        self.tool_registry = tool_registry

    async def parse(self, message: str, context: dict) -> List[Dict]:
        """
        Parse message and extract multiple intents.

        Returns list of intents in execution order.
        """

        prompt = f"""
Parse the following user message and extract all intents.
If there are multiple actions requested, list them in order.

User message: "{message}"

Context: {context}

Return a list of intents with their dependencies:
[
    {{
        "intent": "DATASET.ANALYZE",
        "action": "analyze_dataset",
        "parameters": {{"dataset_path": "..."}},
        "depends_on": []
    }},
    {{
        "intent": "MODEL.RECOMMEND",
        "action": "recommend_model",
        "parameters": {{"task_type": "...", "dataset_size": "..."}},
        "depends_on": ["analyze_dataset"]
    }}
]
"""

        response = await self.llm.generate(prompt)
        intents = self._parse_intents(response)

        return self._order_by_dependency(intents)

    def _parse_intents(self, response: str) -> List[Dict]:
        """Parse LLM response into intent list"""
        # Implementation
        pass

    def _order_by_dependency(self, intents: List[Dict]) -> List[Dict]:
        """Order intents by dependency (topological sort)"""
        # Implementation
        pass
```

### 6.3 Multi-Intent Execution

```python
# mvp/backend/app/utils/conversation_manager.py

class ConversationManager:
    # ... 기존 코드 ...

    async def handle_multi_intent(
        self,
        intents: List[Dict],
        session: Session
    ) -> List[Dict]:
        """Execute multiple intents sequentially"""

        results = []

        for intent in intents:
            # Execute intent
            result = await self._execute_single_intent(intent, session)
            results.append(result)

            # If one fails, stop
            if result.get("error"):
                break

            # Update context for next intent
            self._update_context_from_result(session, result)

        return results

    async def _execute_single_intent(self, intent: Dict, session: Session) -> Dict:
        """Execute single intent"""
        action = intent["action"]
        parameters = intent["parameters"]

        # Call tool
        result = await self.tool_registry.call_tool(
            tool_name=action,
            parameters=parameters,
            user_id=session.user_id,
            db=self.db
        )

        return result

    def _update_context_from_result(self, session: Session, result: Dict):
        """Update session context from result"""
        # Save result to temp_data
        session.temp_data = session.temp_data or {}

        if "dataset_analysis" in result:
            session.temp_data["dataset_analysis"] = result["dataset_analysis"]

        if "job_id" in result:
            session.temp_data["last_created_job"] = result["job_id"]

        # ... 기타 context updates ...
```

---

## 7. Frontend 연동

### 7.1 Action Response 형식

```typescript
// Frontend types

interface ActionResponse {
  action: string;
  message: string;
  data?: any;
  ui_component?: string;  // 표시할 UI 컴포넌트
  next_actions?: string[];  // 다음 가능한 액션들
}

// 예시
{
  "action": "show_dataset_analysis",
  "message": "데이터셋 분석이 완료되었습니다.",
  "data": {
    "format": "ImageFolder",
    "num_classes": 3,
    "classes": ["cat", "dog", "bird"],
    "total_images": 1200
  },
  "ui_component": "DatasetAnalysisCard",
  "next_actions": ["search_models", "create_training_job"]
}
```

### 7.2 Frontend Component

```typescript
// mvp/frontend/components/ChatPanel.tsx

const ChatPanel = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentAction, setCurrentAction] = useState<ActionResponse | null>(null);

  const handleSendMessage = async (text: string) => {
    // Send to backend
    const response = await fetch('/api/v1/chat/message', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        message: text
      })
    });

    const data = await response.json();

    // Add messages
    setMessages([...messages, { role: 'user', content: text }, data]);

    // Handle action
    if (data.action) {
      setCurrentAction(data);
      handleAction(data);
    }
  };

  const handleAction = (action: ActionResponse) => {
    switch (action.action) {
      case 'show_dataset_analysis':
        // Render DatasetAnalysisCard
        break;

      case 'show_models':
        // Render ModelListCards
        break;

      case 'confirm_training':
        // Show confirmation dialog
        break;

      // ... more actions
    }
  };

  return (
    <div>
      <MessageList messages={messages} />
      {currentAction && <ActionComponent action={currentAction} />}
      <MessageInput onSend={handleSendMessage} />
    </div>
  );
};
```

---

## 8. 테스트

### 8.1 Unit Tests

```python
# tests/test_tool_registry.py

import pytest
from app.utils.tool_registry import ToolRegistry, ToolCategory

@pytest.mark.asyncio
async def test_register_tool():
    registry = ToolRegistry()
    initial_count = len(registry.tools)

    # Register custom tool
    registry.register(Tool(
        name="test_tool",
        description="Test",
        category=ToolCategory.TRAINING,
        handler=lambda p, u, db: {"result": "ok"},
        parameters={}
    ))

    assert len(registry.tools) == initial_count + 1

@pytest.mark.asyncio
async def test_call_tool():
    registry = ToolRegistry()

    result = await registry.call_tool(
        tool_name="analyze_dataset",
        parameters={"dataset_path": "C:/datasets/test"},
        user_id=1,
        db=mock_db
    )

    assert "format" in result
    assert "num_classes" in result
```

### 8.2 Integration Tests

```python
# tests/integration/test_multi_intent.py

@pytest.mark.asyncio
async def test_multi_intent_flow():
    """Test: 데이터셋 분석 → 모델 추천 → 학습 생성"""

    message = "C:/datasets/cats 분석하고 모델 추천해서 바로 학습 시작해줘"

    # Parse
    intents = await parser.parse(message)

    assert len(intents) == 3
    assert intents[0]["action"] == "analyze_dataset"
    assert intents[1]["action"] == "recommend_model"
    assert intents[2]["action"] == "create_training_job"

    # Execute
    results = await manager.handle_multi_intent(intents, session)

    assert all(not r.get("error") for r in results)
```

---

## 9. 마이그레이션 계획

### 9.1 기존 코드와의 호환성

Phase 1 구현 시 기존 학습 설정 흐름을 유지하면서 확장:

```python
# 기존 코드 (llm_structured.py)는 그대로 유지
# 새 기능은 tool_registry.py로 추가

# 점진적 마이그레이션
if USE_TOOL_REGISTRY:
    # 새 방식
    result = await tool_registry.call_tool(...)
else:
    # 기존 방식
    result = await action_handlers.execute(...)
```

### 9.2 데이터베이스 마이그레이션

Session 모델에 필드 추가:

```python
# Migration script

# Add intent_queue field to Session
op.add_column('sessions', sa.Column('intent_queue', sa.JSON, nullable=True))

# Add current_intent field
op.add_column('sessions', sa.Column('current_intent', sa.String, nullable=True))
```

---

## 10. 다음 단계

1. ✅ 문서 작성 완료
2. [ ] Tool Registry 구현
3. [ ] 새 State/Action 추가
4. [ ] Multi-Intent Parser 구현
5. [ ] Frontend 연동
6. [ ] 테스트 작성
7. [ ] Phase 1 완료

---

**참고 문서**:
- [LLM_CONTROL_STRATEGY.md](./LLM_CONTROL_STRATEGY.md) - 전체 전략
- [INTENT_MAPPING.md](./INTENT_MAPPING.md) - 인텐트 매핑
- [MCP_IMPLEMENTATION_GUIDE.md](./MCP_IMPLEMENTATION_GUIDE.md) - MCP 구현

