# Conversation State Architecture - 개선 방안

**작성일**: 2025-01-21
**상태**: 제안 (Phase 1 구현 예정)

## 목차
1. [현재 문제점](#현재-문제점)
2. [다른 플랫폼의 접근 방식](#다른-플랫폼의-접근-방식)
3. [3단계 개선 로드맵](#3단계-개선-로드맵)
4. [Phase 1 구현 상세](#phase-1-구현-상세)
5. [Migration Guide](#migration-guide)

---

## 현재 문제점

### 현재 시스템 구조
```
사용자 입력 → LLM 파싱 → 백엔드 텍스트 매칭 → if-else 분기 → 응답
```

### 주요 문제점

#### 1. 암묵적 상태 관리
```python
# ❌ 현재 방식: 이전 메시지 내용으로 상태 추론
if "1️⃣ 신규 프로젝트 생성" in last_assistant_msg.content:
    if user_input == "1":
        # 신규 프로젝트 생성 로직
```

**문제:**
- 대화 상태를 메시지 내용에서 추론
- 이모지/특수문자 인코딩 문제 (cp949 codec error)
- 메시지 내용 변경 시 로직 깨짐
- 디버깅 어려움

#### 2. 취약한 문자열 매칭
```python
# ❌ 다양한 사용자 입력 처리 불가
if user_input == "1":  # "1번", "첫번째", "신규 프로젝트" 등 처리 불가
    ...
```

**문제:**
- 사용자 발화의 다양성 처리 불가
- 새로운 표현 추가마다 if-else 증가
- 유지보수 복잡도 급격히 증가

#### 3. LLM과 백엔드 로직 혼재
```python
# LLM이 이미 의도를 파싱했는데, 백엔드가 또 해석 시도
parsed_result = await llm.parse_intent(message)
# 백엔드가 또 문자열 파싱
if "프로젝트를 지정하지 않았습니다" in last_message:
    ...
```

**문제:**
- 책임 분리 안됨
- LLM의 지능을 제대로 활용하지 못함
- 백엔드가 NLP 역할 중복 수행

#### 4. 확장 불가능
```python
# ❌ 새 시나리오마다 if-else 추가
if scenario_a:
    if case_1: ...
    elif case_2: ...
elif scenario_b:
    if case_1: ...
    # 계속 증가...
```

**미래 시나리오:**
- 데이터셋 업로드 플로우
- 하이퍼파라미터 튜닝 플로우
- 모델 비교 플로우
- 앙상블 학습 플로우
- **→ 현재 방식으로는 관리 불가능**

---

## 다른 플랫폼의 접근 방식

### 1. ChatGPT - Function Calling

```python
# OpenAI Function Calling
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "ResNet으로 학습하고 싶어"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "create_training_job",
            "description": "Create a new training job",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "framework": {"type": "string"},
                    "task_type": {"type": "string"}
                }
            }
        }
    }]
)

# LLM 응답
{
    "tool_calls": [{
        "function": "create_training_job",
        "arguments": {
            "model": "resnet50",
            "framework": "timm",
            "task_type": "image_classification"
        }
    }]
}
```

**장점:**
- LLM이 직접 함수 호출 결정
- 백엔드는 실행만
- 매우 명확한 의도 파악

### 2. Claude - Tool Use

```python
# Anthropic Tool Use
response = client.messages.create(
    model="claude-3-5-sonnet",
    tools=[{
        "name": "create_project",
        "description": "Create a new training project",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "task_type": {"type": "string"}
            }
        }
    }]
)

# Claude 응답
{
    "tool_use": {
        "name": "create_project",
        "input": {"name": "이미지 분류 프로젝트", "task_type": "classification"}
    }
}
```

**장점:**
- 구조화된 출력
- 복잡한 워크플로우 지원

### 3. Gemini - Structured Outputs + JSON Mode

```python
# Google Gemini Structured Output
from google.generativeai.types import content_types

response_schema = content_types.Schema(
    type=content_types.Type.OBJECT,
    properties={
        "action": content_types.Schema(
            type=content_types.Type.STRING,
            enum=["ask_clarification", "create_project", "create_job"]
        ),
        "message": content_types.Schema(type=content_types.Type.STRING),
        "params": content_types.Schema(type=content_types.Type.OBJECT)
    }
)

response = model.generate_content(
    prompt,
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": response_schema
    }
)
```

**장점:**
- 빠른 추론 속도
- JSON Schema 기반 validation
- 비용 효율적

### 비교표

| Platform | Approach | Strengths | Weaknesses |
|----------|----------|-----------|------------|
| **ChatGPT** | Function calling + GPT-4 | 매우 강력한 도구 선택, 복잡한 워크플로우 | 비용 높음, 레이턴시 |
| **Claude** | Tool use + artifacts | 구조화된 출력, 실시간 미리보기 | API 제한적 |
| **Gemini** | Function declarations + JSON mode | Structured output, 빠른 속도, 저렴 | 복잡한 reasoning은 약함 |
| **우리 (현재)** | Text parsing + if-else | 간단한 구현 | ❌ 유지보수 불가능, 확장성 없음 |

---

## 3단계 개선 로드맵

### Phase 1: Conversation State (즉시 구현)
**목표**: 암묵적 상태 → 명시적 상태 관리

**기간**: 1-2일
**우선순위**: 🔥 Critical

```python
# Before
if "1️⃣ 신규 프로젝트 생성" in last_message:
    if user_input == "1": ...

# After
if session.state == "selecting_project":
    if user_input in ["1", "2", "3"]:
        handle_project_selection(user_input, session)
```

**장점:**
- 문자열 매칭 없이 상태로 판단
- 디버깅 쉬움 (state 로그 확인)
- 대화 재개 가능 (세션 복구)
- **현재 문제의 80% 해결**

**구현 범위:**
- DB에 `state`, `temp_data` 컬럼 추가
- State machine 기본 구조
- 상태 기반 라우팅

### Phase 2: Structured Actions (단기)
**목표**: LLM이 action 반환, 백엔드는 실행만

**기간**: 1주
**우선순위**: 🟡 High

```python
# LLM 응답 구조
{
    "action": "create_project",
    "message": "프로젝트를 생성하시겠습니까?",
    "params": {
        "name": "이미지 분류",
        "task_type": "image_classification"
    }
}

# 백엔드 처리
action_handlers = {
    "create_project": handle_create_project,
    "select_project": handle_select_project,
    "create_job": handle_create_job
}

handler = action_handlers[action_result["action"]]
return await handler(action_result["params"])
```

**장점:**
- LLM이 의도를 명확히 표현
- 백엔드는 실행만 (해석 불필요)
- 새 기능 추가 시 action만 추가
- 테스트 용이 (action별 unit test)

**구현 범위:**
- Gemini structured output 적용
- Action handler 구조
- Action별 처리 로직 분리

### Phase 3: Agent Framework (중장기)
**목표**: Multi-step reasoning, tool orchestration

**기간**: 2-3주
**우선순위**: 🟢 Medium

```python
# LangGraph, AutoGen 등 활용
from langgraph.graph import StateGraph

workflow = StateGraph(ConversationState)

workflow.add_node("parse_intent", parse_user_intent)
workflow.add_node("gather_config", gather_training_config)
workflow.add_node("select_project", handle_project_selection)
workflow.add_node("create_job", create_training_job)

workflow.add_conditional_edges(
    "gather_config",
    lambda state: "select_project" if state.config_complete else "gather_config"
)
```

**장점:**
- 복잡한 멀티스텝 워크플로우
- 자동 retry, error handling
- Visual workflow debugging
- Streaming support

**구현 범위:**
- LangGraph/LangChain 통합
- Visual workflow editor
- Advanced error handling

---

## Phase 1 구현 상세

### 1. Database Schema 변경

#### Session 모델 수정

```python
# mvp/backend/app/db/models.py

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 🆕 State 관리
    state = Column(String(50), default="initial", nullable=False, index=True)
    """
    Possible states:
    - initial: 초기 상태 (대화 시작)
    - gathering_config: 학습 설정 수집 중
    - selecting_project: 프로젝트 선택 중
    - confirming: 최종 확인 중
    - complete: 학습 작업 생성 완료
    """

    # 🆕 상태별 임시 데이터
    temp_data = Column(JSON, default={}, nullable=False)
    """
    상태별 임시 데이터 저장:
    {
        "config": {
            "framework": "timm",
            "model_name": "resnet50",
            ...
        },
        "available_projects": [...],
        "selected_project_id": 123,
        "experiment": {...}
    }
    """
```

#### Migration Script

```python
# mvp/backend/alembic/versions/xxxx_add_conversation_state.py

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'xxxx'
down_revision = 'yyyy'  # 이전 migration
branch_labels = None
depends_on = None

def upgrade():
    # Add state column
    op.add_column('sessions',
        sa.Column('state', sa.String(50), nullable=False, server_default='initial')
    )
    op.create_index('ix_sessions_state', 'sessions', ['state'])

    # Add temp_data column
    op.add_column('sessions',
        sa.Column('temp_data', sa.JSON(), nullable=False, server_default='{}')
    )

def downgrade():
    op.drop_index('ix_sessions_state', table_name='sessions')
    op.drop_column('sessions', 'state')
    op.drop_column('sessions', 'temp_data')
```

### 2. State Machine 구조

#### State Enum

```python
# mvp/backend/app/models/conversation.py

from enum import Enum

class ConversationState(str, Enum):
    """대화 상태 정의"""

    INITIAL = "initial"
    """초기 상태 - 새로운 대화 시작"""

    GATHERING_CONFIG = "gathering_config"
    """학습 설정 수집 중 - 모델, 데이터셋, 하이퍼파라미터 등"""

    SELECTING_PROJECT = "selecting_project"
    """프로젝트 선택 중 - 신규/기존 프로젝트 선택"""

    CREATING_PROJECT = "creating_project"
    """프로젝트 생성 중 - 프로젝트 이름/설명 입력"""

    CONFIRMING = "confirming"
    """최종 확인 중 - 학습 시작 전 확인"""

    COMPLETE = "complete"
    """완료 - 학습 작업 생성 완료"""

    ERROR = "error"
    """오류 상태 - 복구 필요"""
```

#### State Transition Logic

```python
# mvp/backend/app/services/conversation_manager.py

from app.models.conversation import ConversationState
from app.db.models import Session as SessionModel

class ConversationManager:
    """대화 상태 관리 및 전환"""

    def __init__(self, db):
        self.db = db

    async def handle_message(
        self,
        session: SessionModel,
        user_message: str
    ) -> dict:
        """
        현재 상태에 따라 메시지 처리

        Args:
            session: DB 세션
            user_message: 사용자 입력

        Returns:
            dict: {
                "message": str,  # 사용자에게 보여줄 메시지
                "state": str,    # 새로운 상태
                "data": dict     # 업데이트된 temp_data
            }
        """
        state = session.state

        # State에 따른 핸들러 라우팅
        handlers = {
            ConversationState.INITIAL: self._handle_initial,
            ConversationState.GATHERING_CONFIG: self._handle_gathering_config,
            ConversationState.SELECTING_PROJECT: self._handle_selecting_project,
            ConversationState.CREATING_PROJECT: self._handle_creating_project,
            ConversationState.CONFIRMING: self._handle_confirming,
        }

        handler = handlers.get(state, self._handle_error)
        result = await handler(session, user_message)

        # State 업데이트
        session.state = result["state"]
        session.temp_data = result["data"]
        self.db.commit()

        return result

    async def _handle_initial(self, session: SessionModel, message: str) -> dict:
        """초기 상태 처리 - LLM으로 의도 파싱"""
        from app.utils.llm import intent_parser

        parsed = await intent_parser.parse_intent(message, context=None)

        if parsed["status"] == "needs_clarification":
            return {
                "message": parsed["clarification"],
                "state": ConversationState.GATHERING_CONFIG,
                "data": {
                    "config": parsed.get("config", {}),
                    "missing_fields": parsed.get("missing_fields", [])
                }
            }
        elif parsed["status"] == "complete":
            # Config 완성 → 프로젝트 선택으로
            return {
                "message": "설정이 완료되었습니다. 프로젝트를 선택해주세요.\n\n1️⃣ 신규 프로젝트 생성\n2️⃣ 기존 프로젝트 선택\n3️⃣ 프로젝트 없이 실험만 진행",
                "state": ConversationState.SELECTING_PROJECT,
                "data": {
                    "config": parsed["config"],
                    "experiment": parsed.get("experiment", {})
                }
            }

    async def _handle_gathering_config(
        self,
        session: SessionModel,
        message: str
    ) -> dict:
        """설정 수집 중 처리"""
        from app.utils.llm import intent_parser

        # 이전 설정 가져오기
        temp_data = session.temp_data or {}
        context = self._build_context(session)

        parsed = await intent_parser.parse_intent(message, context=context)

        # Merge config
        current_config = temp_data.get("config", {})
        new_config = {**current_config, **parsed.get("config", {})}

        if parsed["status"] == "complete":
            # 완성됨 → 프로젝트 선택으로
            return {
                "message": "설정이 완료되었습니다. 프로젝트를 선택해주세요.\n\n1️⃣ 신규 프로젝트 생성\n2️⃣ 기존 프로젝트 선택\n3️⃣ 프로젝트 없이 실험만 진행",
                "state": ConversationState.SELECTING_PROJECT,
                "data": {
                    "config": new_config,
                    "experiment": parsed.get("experiment", {})
                }
            }
        else:
            # 아직 부족 → 계속 수집
            return {
                "message": parsed["clarification"],
                "state": ConversationState.GATHERING_CONFIG,
                "data": {
                    "config": new_config,
                    "missing_fields": parsed.get("missing_fields", [])
                }
            }

    async def _handle_selecting_project(
        self,
        session: SessionModel,
        message: str
    ) -> dict:
        """
        프로젝트 선택 처리

        사용자 입력:
        - "1" or "1번" → 신규 프로젝트 생성
        - "2" or "2번" → 기존 프로젝트 선택
        - "3" or "3번" → 프로젝트 없이 진행
        - 프로젝트 이름 직접 입력 → 해당 프로젝트 검색
        """
        from app.db.models import Project

        # 입력 정규화
        user_input = message.strip().rstrip("번")

        if user_input == "1":
            # 신규 프로젝트 생성
            return {
                "message": "신규 프로젝트 이름을 입력해주세요.\n\n예: 이미지 분류 프로젝트\n(선택사항: 프로젝트 설명도 함께 입력 가능합니다. '-'로 구분)\n예: 동물 분류 프로젝트 - 고양이와 강아지 구분",
                "state": ConversationState.CREATING_PROJECT,
                "data": session.temp_data
            }

        elif user_input == "2":
            # 기존 프로젝트 목록 표시
            projects = self.db.query(Project).filter(
                Project.name != "Uncategorized"
            ).order_by(Project.updated_at.desc()).all()

            if not projects:
                return {
                    "message": "사용 가능한 프로젝트가 없습니다. 신규 프로젝트를 생성하시겠어요?",
                    "state": ConversationState.SELECTING_PROJECT,
                    "data": session.temp_data
                }

            project_list = "다음 프로젝트 중 하나를 선택해주세요:\n\n"
            for idx, project in enumerate(projects, start=1):
                desc = f" - {project.description}" if project.description else ""
                task = f" ({project.task_type})" if project.task_type else ""
                project_list += f"{idx}. **{project.name}**{task}{desc}\n"

            project_list += "\n프로젝트 번호를 입력하거나 프로젝트 이름을 입력해주세요."

            # temp_data에 프로젝트 목록 저장
            temp_data = session.temp_data
            temp_data["available_projects"] = [
                {"id": p.id, "name": p.name} for p in projects
            ]

            return {
                "message": project_list,
                "state": ConversationState.SELECTING_PROJECT,  # 상태 유지
                "data": temp_data
            }

        elif user_input == "3":
            # 프로젝트 없이 진행
            temp_data = session.temp_data
            config = temp_data.get("config", {})

            # Uncategorized 프로젝트 가져오기
            uncategorized = self.db.query(Project).filter(
                Project.name == "Uncategorized"
            ).first()

            if not uncategorized:
                # Uncategorized 프로젝트 생성
                uncategorized = Project(
                    name="Uncategorized",
                    description="프로젝트 없이 진행한 실험들"
                )
                self.db.add(uncategorized)
                self.db.commit()
                self.db.refresh(uncategorized)

            temp_data["selected_project_id"] = uncategorized.id

            return {
                "message": f"학습 설정을 확인해주세요:\n\n{self._format_config(config)}\n\n학습을 시작하시겠습니까? (예/아니오)",
                "state": ConversationState.CONFIRMING,
                "data": temp_data
            }

        elif user_input.isdigit():
            # 프로젝트 번호 선택
            temp_data = session.temp_data
            available_projects = temp_data.get("available_projects", [])

            project_idx = int(user_input) - 1
            if 0 <= project_idx < len(available_projects):
                selected_project = available_projects[project_idx]
                temp_data["selected_project_id"] = selected_project["id"]

                config = temp_data.get("config", {})
                return {
                    "message": f"프로젝트 '{selected_project['name']}'을(를) 선택했습니다.\n\n학습 설정:\n{self._format_config(config)}\n\n학습을 시작하시겠습니까? (예/아니오)",
                    "state": ConversationState.CONFIRMING,
                    "data": temp_data
                }
            else:
                return {
                    "message": "잘못된 번호입니다. 다시 선택해주세요.",
                    "state": ConversationState.SELECTING_PROJECT,
                    "data": temp_data
                }

        else:
            # 프로젝트 이름으로 검색
            project = self.db.query(Project).filter(
                Project.name.ilike(f"%{user_input}%")
            ).first()

            if project:
                temp_data = session.temp_data
                temp_data["selected_project_id"] = project.id

                config = temp_data.get("config", {})
                return {
                    "message": f"프로젝트 '{project.name}'을(를) 선택했습니다.\n\n학습 설정:\n{self._format_config(config)}\n\n학습을 시작하시겠습니까? (예/아니오)",
                    "state": ConversationState.CONFIRMING,
                    "data": temp_data
                }
            else:
                return {
                    "message": f"'{user_input}' 프로젝트를 찾을 수 없습니다. 다시 선택해주세요.",
                    "state": ConversationState.SELECTING_PROJECT,
                    "data": session.temp_data
                }

    async def _handle_creating_project(
        self,
        session: SessionModel,
        message: str
    ) -> dict:
        """신규 프로젝트 생성 처리"""
        from app.db.models import Project

        # 프로젝트 이름 및 설명 파싱
        parts = message.split("-", 1)
        project_name = parts[0].strip()
        project_description = parts[1].strip() if len(parts) > 1 else None

        # 프로젝트 생성
        temp_data = session.temp_data
        config = temp_data.get("config", {})

        new_project = Project(
            name=project_name,
            description=project_description,
            task_type=config.get("task_type")
        )
        self.db.add(new_project)
        self.db.commit()
        self.db.refresh(new_project)

        temp_data["selected_project_id"] = new_project.id

        return {
            "message": f"프로젝트 '{project_name}'이(가) 생성되었습니다.\n\n학습 설정:\n{self._format_config(config)}\n\n학습을 시작하시겠습니까? (예/아니오)",
            "state": ConversationState.CONFIRMING,
            "data": temp_data
        }

    async def _handle_confirming(
        self,
        session: SessionModel,
        message: str
    ) -> dict:
        """최종 확인 처리"""
        user_input = message.strip().lower()

        if user_input in ["예", "yes", "y", "네", "확인", "ok"]:
            # 학습 작업 생성
            temp_data = session.temp_data

            # Training job 생성은 chat.py에서 처리
            return {
                "message": "학습 작업을 생성합니다...",
                "state": ConversationState.COMPLETE,
                "data": temp_data
            }
        else:
            # 취소 - 초기 상태로
            return {
                "message": "취소되었습니다. 다시 시작하시려면 학습 설정을 입력해주세요.",
                "state": ConversationState.INITIAL,
                "data": {}
            }

    def _build_context(self, session: SessionModel) -> str:
        """이전 대화 컨텍스트 생성"""
        from app.db.models import Message as MessageModel

        messages = self.db.query(MessageModel).filter(
            MessageModel.session_id == session.id
        ).order_by(MessageModel.created_at.desc()).limit(10).all()

        context_parts = []
        for msg in reversed(messages):
            context_parts.append(f"[{msg.role.upper()}]: {msg.content}")

        return "\n".join(context_parts)

    def _format_config(self, config: dict) -> str:
        """설정을 보기 좋게 포맷팅"""
        lines = []
        lines.append(f"- 프레임워크: {config.get('framework', 'N/A')}")
        lines.append(f"- 모델: {config.get('model_name', 'N/A')}")
        lines.append(f"- 작업 유형: {config.get('task_type', 'N/A')}")
        lines.append(f"- 데이터셋: {config.get('dataset_path', 'N/A')}")
        lines.append(f"- 에포크: {config.get('epochs', 'N/A')}")
        lines.append(f"- 배치 크기: {config.get('batch_size', 'N/A')}")
        lines.append(f"- 학습률: {config.get('learning_rate', 'N/A')}")
        return "\n".join(lines)

    def _handle_error(self, session: SessionModel, message: str) -> dict:
        """에러 처리"""
        return {
            "message": "죄송합니다. 오류가 발생했습니다. 처음부터 다시 시작해주세요.",
            "state": ConversationState.INITIAL,
            "data": {}
        }
```

### 3. API 엔드포인트 수정

```python
# mvp/backend/app/api/chat.py (수정)

from app.services.conversation_manager import ConversationManager
from app.models.conversation import ConversationState

@router.post("/message", response_model=chat.ChatResponse)
async def chat_message(request: chat.ChatRequest, db: DBSession = Depends(get_db)):
    """
    채팅 메시지 처리 (State-based)
    """
    logger.debug(f"Received chat request: session_id={request.session_id}, message={request.message[:50]}...")

    # Get or create session
    if request.session_id:
        session = db.query(SessionModel).filter(SessionModel.id == request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = SessionModel(state=ConversationState.INITIAL, temp_data={})
        db.add(session)
        db.commit()
        db.refresh(session)

    logger.debug(f"Using session ID: {session.id}, state: {session.state}")

    # Save user message
    user_message = MessageModel(
        session_id=session.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # Process message with state machine
    conversation_manager = ConversationManager(db)
    result = await conversation_manager.handle_message(session, request.message)

    # Save assistant message
    assistant_message = MessageModel(
        session_id=session.id,
        role="assistant",
        content=result["message"],
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    # If state is COMPLETE, create training job
    if session.state == ConversationState.COMPLETE:
        temp_data = session.temp_data
        config = temp_data.get("config", {})
        project_id = temp_data.get("selected_project_id")
        experiment = temp_data.get("experiment", {})

        # Create training job
        from app.db.models import TrainingJob

        training_job = TrainingJob(
            session_id=session.id,
            project_id=project_id,
            framework=config.get("framework"),
            model_name=config.get("model_name"),
            task_type=config.get("task_type"),
            dataset_path=config.get("dataset_path"),
            dataset_format=config.get("dataset_format"),
            num_classes=config.get("num_classes"),
            epochs=config.get("epochs"),
            batch_size=config.get("batch_size"),
            learning_rate=config.get("learning_rate"),
            experiment_name=experiment.get("name"),
            tags=experiment.get("tags"),
            notes=experiment.get("notes"),
            status="pending"
        )
        db.add(training_job)
        db.commit()
        db.refresh(training_job)

        logger.info(f"Created training job ID: {training_job.id}")

        # Reset session state
        session.state = ConversationState.INITIAL
        session.temp_data = {}
        db.commit()

        return chat.ChatResponse(
            session_id=session.id,
            user_message=user_message,
            assistant_message=assistant_message,
            parsed_intent={"status": "complete", "job_id": training_job.id}
        )

    return chat.ChatResponse(
        session_id=session.id,
        user_message=user_message,
        assistant_message=assistant_message,
        parsed_intent={"status": session.state, "data": session.temp_data}
    )
```

---

## Migration Guide

### 단계별 마이그레이션

#### Step 1: 데이터베이스 마이그레이션
```bash
cd mvp/backend

# 마이그레이션 파일 생성
alembic revision -m "add_conversation_state"

# 생성된 파일 수정 (위의 migration script 참고)
# alembic/versions/xxxx_add_conversation_state.py

# 마이그레이션 실행
alembic upgrade head

# 확인
python -c "from app.db.models import Session; print(Session.__table__.columns)"
```

#### Step 2: ConversationManager 구현
```bash
# 새 파일 생성
touch mvp/backend/app/services/conversation_manager.py
touch mvp/backend/app/models/conversation.py

# 위의 코드 복사/붙여넣기
```

#### Step 3: API 엔드포인트 수정
```bash
# 기존 chat.py 백업
cp mvp/backend/app/api/chat.py mvp/backend/app/api/chat.py.backup

# chat.py 수정 (위의 코드 참고)
```

#### Step 4: 테스트
```bash
# 유닛 테스트 실행
pytest tests/unit/test_conversation_manager.py -v

# 통합 테스트
pytest tests/integration/test_chat_flow.py -v

# 수동 테스트
# 1. 새 대화 시작
# 2. "ResNet으로 학습하고 싶어" 입력
# 3. State가 gathering_config로 전환되는지 확인
# 4. 설정 완료 후 selecting_project로 전환되는지 확인
# 5. "2" 입력 시 프로젝트 목록 표시되는지 확인
```

### 기존 세션 마이그레이션

```python
# mvp/backend/scripts/migrate_existing_sessions.py

from app.db.database import SessionLocal
from app.db.models import Session as SessionModel
from app.models.conversation import ConversationState

def migrate_existing_sessions():
    """기존 세션을 initial 상태로 마이그레이션"""
    db = SessionLocal()

    try:
        sessions = db.query(SessionModel).filter(
            SessionModel.state == None  # 기존 세션
        ).all()

        for session in sessions:
            session.state = ConversationState.INITIAL
            session.temp_data = {}

        db.commit()
        print(f"Migrated {len(sessions)} sessions")

    finally:
        db.close()

if __name__ == "__main__":
    migrate_existing_sessions()
```

### Rollback 절차 (문제 발생 시)

```bash
# Step 1: 코드 롤백
git checkout mvp/backend/app/api/chat.py.backup mvp/backend/app/api/chat.py

# Step 2: DB 롤백
cd mvp/backend
alembic downgrade -1

# Step 3: 백엔드 재시작
# (uvicorn이 자동으로 재시작될 것)
```

---

## 테스트 계획

### 유닛 테스트

```python
# tests/unit/test_conversation_manager.py

import pytest
from app.services.conversation_manager import ConversationManager
from app.models.conversation import ConversationState

@pytest.mark.asyncio
async def test_initial_to_gathering_config(db_session):
    """초기 → 설정 수집 전환 테스트"""
    manager = ConversationManager(db_session)
    session = create_test_session(state=ConversationState.INITIAL)

    result = await manager.handle_message(session, "ResNet으로 학습하고 싶어")

    assert result["state"] == ConversationState.GATHERING_CONFIG
    assert "config" in result["data"]

@pytest.mark.asyncio
async def test_selecting_project_option_1(db_session):
    """프로젝트 선택 - 옵션 1 (신규 생성) 테스트"""
    manager = ConversationManager(db_session)
    session = create_test_session(
        state=ConversationState.SELECTING_PROJECT,
        temp_data={"config": {...}}
    )

    result = await manager.handle_message(session, "1")

    assert result["state"] == ConversationState.CREATING_PROJECT
    assert "프로젝트 이름" in result["message"]

@pytest.mark.asyncio
async def test_selecting_project_option_2(db_session):
    """프로젝트 선택 - 옵션 2 (기존 선택) 테스트"""
    manager = ConversationManager(db_session)

    # 기존 프로젝트 생성
    create_test_project(db_session, name="테스트 프로젝트")

    session = create_test_session(
        state=ConversationState.SELECTING_PROJECT,
        temp_data={"config": {...}}
    )

    result = await manager.handle_message(session, "2")

    assert result["state"] == ConversationState.SELECTING_PROJECT
    assert "available_projects" in result["data"]
    assert "테스트 프로젝트" in result["message"]
```

### 통합 테스트

```python
# tests/integration/test_chat_flow.py

@pytest.mark.asyncio
async def test_full_conversation_flow(test_client, db_session):
    """전체 대화 플로우 테스트"""

    # Step 1: 새 대화 시작
    response = await test_client.post("/api/v1/chat/message", json={
        "message": "ResNet50으로 학습하고 싶어"
    })
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    # Step 2: 데이터셋 입력
    response = await test_client.post("/api/v1/chat/message", json={
        "session_id": session_id,
        "message": "C:\\datasets\\cls\\imagenet-10"
    })
    assert response.status_code == 200

    # Step 3: 하이퍼파라미터 입력
    response = await test_client.post("/api/v1/chat/message", json={
        "session_id": session_id,
        "message": "기본값으로 해줘"
    })
    assert response.status_code == 200

    # Step 4: 프로젝트 선택 (신규 생성)
    response = await test_client.post("/api/v1/chat/message", json={
        "session_id": session_id,
        "message": "1"
    })
    assert "프로젝트 이름" in response.json()["assistant_message"]["content"]

    # Step 5: 프로젝트 이름 입력
    response = await test_client.post("/api/v1/chat/message", json={
        "session_id": session_id,
        "message": "이미지 분류 테스트"
    })
    assert response.status_code == 200

    # Step 6: 최종 확인
    response = await test_client.post("/api/v1/chat/message", json={
        "session_id": session_id,
        "message": "예"
    })
    assert response.status_code == 200
    assert response.json()["parsed_intent"]["status"] == "complete"
    assert "job_id" in response.json()["parsed_intent"]
```

---

## 참고 자료

### 외부 문서
- [LangChain State Machine](https://python.langchain.com/docs/langgraph/concepts/low_level)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Gemini Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

### 관련 파일
- `mvp/backend/app/db/models.py` - DB 모델
- `mvp/backend/app/api/chat.py` - Chat API
- `mvp/backend/app/utils/llm.py` - LLM 통합
- `ARCHITECTURE.md` - 전체 아키텍처

---

**변경 이력**
- 2025-01-21: 초안 작성 (Phase 1 설계 완료)
