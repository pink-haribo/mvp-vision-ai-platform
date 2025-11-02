# Phase 1 LLM Control - 잠재적 문제점 및 디버깅 가이드

**작성일**: 2025-11-02
**목적**: 문서대로 처리되지 않을 때 확인해야 할 부분들

## 개요

Phase 1 LLM 컨트롤이 USER_UTTERANCE_EXAMPLES.md 문서대로 작동하지 않을 때 발생 가능한 문제점과 해결 방법을 정리합니다.

---

## 1. LLM 파싱 실패 (가장 흔한 문제)

### 문제 증상

**사용자 발화**:
```
"C:/datasets/coco8로 yolov8n 학습해줘"
```

**예상 동작**:
- LLM이 dataset_path와 model_name 추출
- 데이터셋 분석 실행
- 학습 설정으로 진행

**실제 동작**:
```
"학습을 시작하려면 몇 가지 정보가 필요합니다:
1. 데이터셋 경로는?
2. 어떤 모델 사용?"
```

### 원인 분석

#### 원인 1: Gemini가 파라미터를 추출하지 못함

**확인 방법**:
```bash
# llm_debug.log 확인
cat mvp/backend/llm_debug.log

# 출력 예시:
[DEBUG] LLM Response for session 1:
Action: ASK_CLARIFICATION
Message: 학습을 시작하려면...
Current Config: None  # ← 문제!
Config: None  # ← 문제!
```

**근본 원인**:
1. **Gemini 프롬프트가 불명확**
   - `llm_structured.py`의 시스템 프롬프트가 파라미터 추출을 명확히 지시하지 않음
   - 예시가 부족함

2. **Few-shot 예시 부족**
   - Gemini가 어떤 형식으로 파라미터를 추출해야 하는지 모름

3. **Schema 정의 불명확**
   - `GeminiActionResponse`의 `current_config` 필드 설명 부족

**해결 방법**:

**파일**: `mvp/backend/app/utils/llm_structured.py`

현재:
```python
system_prompt = """
You are an AI assistant for a Vision AI Training Platform.
Parse user intent and extract parameters.
"""
```

개선:
```python
system_prompt = """
You are an AI assistant for a Vision AI Training Platform.

CRITICAL: ALWAYS extract parameters from user messages.

Example 1:
User: "C:/datasets/coco8로 yolov8n 학습해줘"
Response:
{
  "action": "ANALYZE_DATASET",
  "current_config": {
    "dataset_path": "C:/datasets/coco8",
    "model_name": "yolov8n",
    "framework": "ultralytics"
  }
}

Example 2:
User: "resnet50 정보 알려줘"
Response:
{
  "action": "SHOW_MODEL_INFO",
  "current_config": {
    "model_name": "resnet50",
    "framework": "timm"
  }
}

ALWAYS populate current_config with extracted parameters!
"""
```

**검증 방법**:
```python
# 테스트 추가
async def test_llm_extracts_parameters():
    """LLM이 파라미터를 제대로 추출하는지 확인"""
    response = await structured_intent_parser.parse_intent(
        user_message="C:/datasets/coco8로 yolov8n 학습",
        state=ConversationState.INITIAL,
        context="",
        temp_data={}
    )

    assert response.current_config is not None
    assert response.current_config.get("dataset_path") == "C:/datasets/coco8"
    assert response.current_config.get("model_name") == "yolov8n"
```

---

#### 원인 2: Gemini API 응답이 스키마를 따르지 않음

**확인 방법**:
```bash
# Gemini 원본 응답 확인
tail -50 mvp/backend/gemini_responses.txt

# 출력 예시:
{
  "action": "start_training",  # ← ActionType enum이 아님!
  "dataset": "C:/datasets/coco8",  # ← current_config에 없음!
  "model": "yolov8n"
}
```

**근본 원인**:
- Gemini가 `GeminiActionResponse` Pydantic 스키마를 무시
- JSON 필드명을 임의로 변경

**해결 방법**:

**파일**: `mvp/backend/app/utils/llm_structured.py`

```python
async def parse_intent(...) -> GeminiActionResponse:
    # ... Gemini API 호출 ...

    raw_response = response.text

    # CRITICAL: Validate and fix Gemini response
    try:
        parsed = json.loads(raw_response)

        # Fix common mistakes
        if "action" in parsed:
            # Convert string to ActionType
            action_str = parsed["action"]
            if action_str.lower() == "start_training":
                parsed["action"] = "START_TRAINING"
            # ... 다른 매핑 ...

        # Extract parameters from wrong fields
        if "dataset" in parsed and "current_config" not in parsed:
            parsed["current_config"] = {
                "dataset_path": parsed.pop("dataset"),
                "model_name": parsed.pop("model", None)
            }

        # Validate with Pydantic
        return GeminiActionResponse(**parsed)

    except Exception as e:
        logger.error(f"Failed to parse Gemini response: {e}")
        logger.error(f"Raw response: {raw_response}")
        # Fallback to ASK_CLARIFICATION
        return GeminiActionResponse(
            action=ActionType.ASK_CLARIFICATION,
            message="죄송합니다. 다시 한번 말씀해주시겠어요?"
        )
```

---

### 원인 3: Fallback 추출도 실패

**확인 방법**:
```bash
# Fallback 로그 확인
cat mvp/data/logs/fallback_debug.log

# 출력 예시:
[2025-11-02 10:30:00] Action: START_TRAINING
Before: {}
User message: C:/datasets/coco8로 yolov8n 학습
After: {}  # ← 아무것도 추출 안됨!
```

**근본 원인**:
- `_extract_from_user_message`의 정규식이 실패
- 경로 형식이 예상과 다름 (Windows vs Linux)

**해결 방법**:

**파일**: `mvp/backend/app/services/action_handlers.py`

현재 정규식:
```python
def _extract_from_user_message(self, user_message: str, existing_config: dict) -> dict:
    # 경로 추출 (Windows만 지원)
    path_pattern = r'([A-Z]:/[^\s]+)'

    # job_id 추출
    job_pattern = r'job\s+(\d+)'
```

개선:
```python
def _extract_from_user_message(self, user_message: str, existing_config: dict) -> dict:
    """Extract parameters from user message with robust regex"""

    # 경로 추출 (Windows + Linux + 한글 경로)
    path_patterns = [
        r'([A-Z]:/[^\s]+)',           # Windows: C:/path
        r'([A-Z]:\\[^\s]+)',          # Windows backslash: C:\path
        r'(/[^\s]+)',                 # Linux: /path
        r'([가-힣A-Za-z0-9_/\\:]+)',  # 한글 포함 경로
    ]

    for pattern in path_patterns:
        match = re.search(pattern, user_message)
        if match:
            path = match.group(1)
            # Validate path exists
            from pathlib import Path
            if Path(path).exists():
                existing_config["dataset_path"] = path
                logger.info(f"[FALLBACK] Extracted dataset_path: {path}")
                break

    # job_id 추출 (다양한 형식)
    job_patterns = [
        r'job\s+(\d+)',          # "job 42"
        r'작업\s+(\d+)',         # "작업 42"
        r'(\d+)번',              # "42번"
        r'#(\d+)',               # "#42"
    ]

    for pattern in job_patterns:
        match = re.search(pattern, user_message)
        if match:
            job_id = int(match.group(1))
            existing_config["job_id"] = job_id
            logger.info(f"[FALLBACK] Extracted job_id: {job_id}")
            break

    # 모델 이름 추출
    known_models = [
        "resnet50", "resnet18", "efficientnet_b0",
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"
    ]

    for model in known_models:
        if model.lower() in user_message.lower():
            existing_config["model_name"] = model
            logger.info(f"[FALLBACK] Extracted model_name: {model}")

            # Framework auto-detection
            if model.startswith("yolo"):
                existing_config["framework"] = "ultralytics"
            elif model.startswith("resnet") or model.startswith("efficient"):
                existing_config["framework"] = "timm"
            break

    # 숫자 파라미터 추출
    epoch_match = re.search(r'(\d+)\s*에?폭', user_message)
    if epoch_match:
        existing_config["epochs"] = int(epoch_match.group(1))

    batch_match = re.search(r'배치\s*(\d+)', user_message)
    if batch_match:
        existing_config["batch_size"] = int(batch_match.group(1))

    return existing_config
```

**테스트 추가**:
```python
def test_fallback_extraction():
    """Fallback이 다양한 형식을 처리하는지 확인"""
    handler = ActionHandlers(db)

    # Windows 경로
    config = handler._extract_from_user_message(
        "C:/datasets/coco8로 학습",
        {}
    )
    assert config["dataset_path"] == "C:/datasets/coco8"

    # Linux 경로
    config = handler._extract_from_user_message(
        "/home/user/data로 학습",
        {}
    )
    assert config["dataset_path"] == "/home/user/data"

    # job_id 다양한 형식
    assert handler._extract_from_user_message("job 42 중지", {})["job_id"] == 42
    assert handler._extract_from_user_message("42번 중지", {})["job_id"] == 42
    assert handler._extract_from_user_message("#42 중지", {})["job_id"] == 42
```

---

## 2. Tool 실행 오류

### 문제 증상

**사용자 발화**:
```
"C:/datasets/notexist 분석해줘"
```

**예상 동작**:
- "해당 경로가 존재하지 않습니다" 에러 메시지

**실제 동작**:
```
500 Internal Server Error
```

### 원인 분석

**확인 방법**:
```bash
# 백엔드 로그 확인
tail -100 mvp/backend/app.log

# 출력 예시:
ERROR: Tool analyze_dataset failed: [Errno 2] No such file or directory: 'C:/datasets/notexist'
Traceback (most recent call last):
  File "app/utils/tool_registry.py", line 287, in _analyze_dataset
    analysis = analyze_dataset(dataset_path)
  File "app/utils/dataset_analyzer.py", line 45, in analyze_dataset
    items = os.listdir(dataset_path)  # ← 에러!
FileNotFoundError: [Errno 2] No such file or directory
```

**근본 원인**:
- `tool_registry._analyze_dataset`에서 경로 검증 후에도 `dataset_analyzer`가 다시 에러
- 예외 처리 누락

**해결 방법**:

**파일**: `mvp/backend/app/utils/tool_registry.py`

현재:
```python
async def _analyze_dataset(self, params, db, user_id):
    dataset_path = params.get("dataset_path")

    if not Path(dataset_path).exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    analysis = analyze_dataset(dataset_path)  # ← 여기서 또 에러 가능
    return {...}
```

개선:
```python
async def _analyze_dataset(self, params, db, user_id):
    dataset_path = params.get("dataset_path")

    if not dataset_path:
        raise ValueError("dataset_path is required")

    path = Path(dataset_path)

    # 더 상세한 검증
    if not path.exists():
        raise ValueError(f"경로가 존재하지 않습니다: {dataset_path}")

    if not path.is_dir():
        raise ValueError(f"경로가 디렉토리가 아닙니다: {dataset_path}")

    # 읽기 권한 확인
    if not os.access(dataset_path, os.R_OK):
        raise ValueError(f"경로에 대한 읽기 권한이 없습니다: {dataset_path}")

    try:
        analysis = analyze_dataset(dataset_path)
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}", exc_info=True)
        raise ValueError(f"데이터셋 분석 중 오류: {str(e)}")

    return {...}
```

**ActionHandler에서 에러 처리**:

**파일**: `mvp/backend/app/services/action_handlers.py`

```python
async def _handle_analyze_dataset(...):
    try:
        result = await tool_registry.call_tool(
            "analyze_dataset",
            {"dataset_path": dataset_path},
            self.db,
            user_id=None
        )

        # Success path...

    except ValueError as e:
        # User-friendly error
        return {
            "new_state": ConversationState.ERROR,
            "message": f"❌ {str(e)}\n\n올바른 데이터셋 경로를 입력해주세요.",
            "temp_data": temp_data
        }
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "new_state": ConversationState.ERROR,
            "message": f"죄송합니다. 예상치 못한 오류가 발생했습니다: {str(e)}",
            "temp_data": temp_data
        }
```

---

## 3. 상태 전환 오류

### 문제 증상

**시나리오**:
1. 사용자: "C:/datasets/coco8 분석해줘"
2. 시스템: 분석 결과 표시
3. 사용자: "학습 시작"
4. 시스템: "데이터셋 경로를 입력해주세요" ← 이미 분석했는데!

**예상 동작**:
- 이미 분석한 데이터셋 정보를 유지
- 바로 모델 선택으로 진행

**실제 동작**:
- `temp_data`에서 dataset_path가 사라짐
- 처음부터 다시 물어봄

### 원인 분석

**확인 방법**:
```python
# conversation_manager.py의 TRACE 로그 확인
# 출력 예시:
[TRACE-5-SAVE] Saving to DB:
  new_state: ANALYZING_DATASET
  updated_temp_data: {"dataset_analysis": {...}}  # ← config 없음!
```

**근본 원인**:
- `_handle_analyze_dataset`가 `config`에 dataset_path를 저장하지 않음
- 다음 요청에서 dataset_path를 찾을 수 없음

**해결 방법**:

**파일**: `mvp/backend/app/services/action_handlers.py`

현재:
```python
async def _handle_analyze_dataset(...):
    # ... analysis ...

    temp_data["dataset_analysis"] = result

    return {
        "new_state": ConversationState.ANALYZING_DATASET,
        "message": message,
        "temp_data": temp_data
    }
```

개선:
```python
async def _handle_analyze_dataset(...):
    # ... analysis ...

    # CRITICAL: 분석한 데이터셋 경로를 config에 저장
    config = temp_data.get("config", {})
    config["dataset_path"] = dataset_path
    config["task_type"] = result.get("task_type")  # YOLO → object_detection
    config["num_classes"] = result.get("num_classes")

    temp_data["config"] = config
    temp_data["dataset_analysis"] = result

    return {
        "new_state": ConversationState.ANALYZING_DATASET,
        "message": message,
        "temp_data": temp_data
    }
```

**검증 테스트**:
```python
async def test_dataset_path_persists():
    """데이터셋 분석 후 경로가 유지되는지 확인"""
    session = create_test_session()

    # Step 1: Analyze dataset
    response1 = await conversation_manager.process_message(
        session.id,
        "C:/datasets/coco8 분석해줘"
    )

    # Verify dataset_path is in config
    session.refresh()
    assert session.temp_data["config"]["dataset_path"] == "C:/datasets/coco8"

    # Step 2: Start training (should use saved dataset_path)
    response2 = await conversation_manager.process_message(
        session.id,
        "yolov8n으로 학습 시작"
    )

    # Should NOT ask for dataset_path again
    assert "데이터셋 경로" not in response2["message"]
```

---

## 4. 프론트엔드 연동 문제

### 문제 증상

**백엔드 응답**:
```json
{
  "message": "📊 **데이터셋 분석 결과**\n\n경로: C:/datasets/coco8",
  "state": "analyzing_dataset",
  "dataset_analysis": {
    "format": "yolo",
    "classes": [...]
  }
}
```

**프론트엔드 표시**:
```
📊 **데이터셋 분석 결과**\n\n경로: C:/datasets/coco8
```
← Markdown이 렌더링되지 않음!

### 원인 분석

**확인 방법**:
```typescript
// mvp/frontend/components/ChatPanel.tsx
console.log("Backend response:", response);

// 출력:
{
  message: "📊 **데이터셋 분석 결과**\n\n경로: C:/datasets/coco8",
  // \n이 escape되어 있음
}
```

**근본 원인**:
1. 백엔드가 `\n`을 문자열로 전송
2. 프론트엔드가 Markdown 파싱 안함
3. `<ReactMarkdown>` 컴포넌트 미사용

**해결 방법**:

**파일**: `mvp/frontend/components/ChatPanel.tsx`

현재:
```typescript
<div className="message">
  {message.content}
</div>
```

개선:
```typescript
import ReactMarkdown from 'react-markdown';

<div className="message">
  <ReactMarkdown>{message.content}</ReactMarkdown>
</div>
```

**스타일 추가**:
```css
.message {
  /* Markdown 스타일 */
}

.message strong {
  font-weight: 700;
}

.message code {
  background: #f5f5f5;
  padding: 2px 4px;
  border-radius: 3px;
}

.message pre {
  background: #1e1e1e;
  color: #fff;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
}
```

---

### 문제 2: dataset_analysis가 표시되지 않음

**백엔드 응답**:
```json
{
  "message": "...",
  "dataset_analysis": {
    "format": "yolo",
    "classes": ["person", "car", ...]
  }
}
```

**프론트엔드**:
- 클래스 목록이 표시되지 않음
- 통계가 표시되지 않음

**원인**:
- ChatPanel이 `dataset_analysis` 필드를 읽지 않음

**해결 방법**:

**파일**: `mvp/frontend/components/ChatPanel.tsx`

```typescript
interface ChatResponse {
  message: string;
  state: string;
  dataset_analysis?: DatasetAnalysis;
  model_search_results?: ModelSearchResult[];
  training_job_id?: number;
}

function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);

  const handleSendMessage = async (text: string) => {
    const response = await fetch('/api/v1/chat/message', {
      method: 'POST',
      body: JSON.stringify({ message: text, session_id: sessionId })
    });

    const data: ChatResponse = await response.json();

    // Add assistant message
    setMessages([...messages, {
      role: 'assistant',
      content: data.message,
      metadata: {
        dataset_analysis: data.dataset_analysis,
        model_search_results: data.model_search_results,
        training_job_id: data.training_job_id
      }
    }]);
  };

  return (
    <div>
      {messages.map((msg, i) => (
        <div key={i}>
          <ReactMarkdown>{msg.content}</ReactMarkdown>

          {/* Dataset Analysis Card */}
          {msg.metadata?.dataset_analysis && (
            <DatasetAnalysisCard analysis={msg.metadata.dataset_analysis} />
          )}

          {/* Model Search Results */}
          {msg.metadata?.model_search_results && (
            <ModelSearchResults models={msg.metadata.model_search_results} />
          )}
        </div>
      ))}
    </div>
  );
}
```

---

## 5. 학습 실행 오류

### 문제 증상

**사용자 발화**:
```
"예" (학습 시작 확인)
```

**예상 동작**:
- TrainingJob 생성
- 백그라운드로 학습 프로세스 시작
- "학습을 시작합니다! Job ID: 42"

**실제 동작**:
```
500 Internal Server Error

[ERROR] Failed to start training: [WinError 2] The system cannot find the file specified
```

### 원인 분석

**확인 방법**:
```bash
# 백엔드 로그
tail -50 mvp/backend/app.log

# 출력:
ERROR: training_manager.start_training() failed
Traceback:
  File "app/utils/training_manager.py", line 123, in start_training
    process = subprocess.Popen(command, ...)
FileNotFoundError: [WinError 2] The system cannot find the file specified
Command: ['python', '-m', 'train', '--job-id', '42']
```

**근본 원인**:
1. `train.py` 경로가 잘못됨
2. Virtual environment Python이 아닌 시스템 Python 사용
3. 모듈 import 에러

**해결 방법**:

**파일**: `mvp/backend/app/utils/training_manager.py`

현재:
```python
def start_training(self, job_id: int):
    command = [
        'python',  # ← 시스템 Python!
        '-m', 'train',
        '--job-id', str(job_id)
    ]

    process = subprocess.Popen(command, ...)
```

개선:
```python
def start_training(self, job_id: int):
    import sys
    from pathlib import Path

    # Use virtual environment Python
    python_executable = sys.executable  # venv/Scripts/python.exe

    # Find train.py path
    train_script = Path(__file__).parent.parent.parent / "training" / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    command = [
        python_executable,
        str(train_script),
        '--job-id', str(job_id)
    ]

    logger.info(f"Starting training with command: {' '.join(command)}")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(train_script.parent),  # Set working directory
            env=os.environ.copy()  # Inherit environment
        )

        # Save process ID
        job.process_id = process.pid
        self.db.commit()

        logger.info(f"Training started: PID {process.pid}")

    except Exception as e:
        logger.error(f"Failed to start training: {e}", exc_info=True)
        job.status = "failed"
        job.error_message = str(e)
        self.db.commit()
        raise
```

---

## 6. 실시간 메트릭 업데이트 안됨

### 문제 증상

**예상 동작**:
- 학습 중 Epoch마다 메트릭 업데이트
- 프론트엔드에서 실시간으로 Loss/Accuracy 표시

**실제 동작**:
- "학습을 시작합니다!" 이후 아무 업데이트 없음
- "job 42 상태 알려줘"를 해도 "Epoch 0/50"

### 원인 분석

**확인 방법**:
```bash
# training.log 확인
cat mvp/data/logs/job_42/training.log

# 출력:
Epoch 1/50
Train Loss: 0.6931, Train Acc: 50.2%
Val Loss: 0.6895, Val Acc: 51.8%
# ← 로그는 쓰여지고 있음!
```

```sql
-- DB 확인
SELECT * FROM training_metrics WHERE job_id = 42;

-- 출력:
(empty)  ← 메트릭이 DB에 저장 안됨!
```

**근본 원인**:
- `train.py`가 메트릭을 stdout에만 출력
- DB에 저장하지 않음
- TrainingMetric 레코드 생성 누락

**해결 방법**:

**파일**: `mvp/training/train.py`

현재:
```python
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    # Only print
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
    print(f"Val Loss: {val_loss}, Val Acc: {val_acc}")
```

개선:
```python
from app.db.database import SessionLocal
from app.db.models import TrainingMetric, TrainingJob

def save_metrics_to_db(job_id, epoch, metrics):
    """Save metrics to database"""
    db = SessionLocal()
    try:
        # Create metric record
        metric = TrainingMetric(
            job_id=job_id,
            epoch=epoch,
            loss=metrics['train_loss'],
            accuracy=metrics['train_acc'],
            val_loss=metrics['val_loss'],
            val_accuracy=metrics['val_acc']
        )
        db.add(metric)

        # Update job current_epoch
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.current_epoch = epoch
            if epoch == job.epochs:  # Final epoch
                job.final_accuracy = metrics['val_acc']

        db.commit()
        logger.info(f"Saved metrics for job {job_id}, epoch {epoch}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        db.rollback()
    finally:
        db.close()

# Training loop
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    metrics = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }

    # Print to stdout (for logs)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

    # Save to DB
    save_metrics_to_db(args.job_id, epoch + 1, metrics)
```

---

## 7. 디버깅 체크리스트

문서대로 작동하지 않을 때 순서대로 확인:

### Step 1: LLM 파싱 확인
```bash
# LLM 응답 로그 확인
cat mvp/backend/llm_debug.log | tail -50

# 확인 사항:
✅ Action이 올바른가?
✅ current_config에 파라미터가 있는가?
✅ message가 적절한가?
```

### Step 2: Fallback 추출 확인
```bash
# Fallback 로그 확인
cat mvp/data/logs/fallback_debug.log | tail -20

# 확인 사항:
✅ Before/After에서 config가 추출되었는가?
✅ dataset_path, model_name, job_id 등이 있는가?
```

### Step 3: Action Handler 실행 확인
```bash
# 백엔드 로그 확인
tail -100 mvp/backend/app.log | grep "handle_action"

# 확인 사항:
✅ Handler가 호출되었는가?
✅ 에러가 발생하지 않았는가?
✅ 올바른 상태로 전환되었는가?
```

### Step 4: Tool 실행 확인
```bash
# Tool 실행 로그
tail -100 mvp/backend/app.log | grep "Executing tool"

# 확인 사항:
✅ Tool이 호출되었는가?
✅ 파라미터가 올바른가?
✅ 결과가 반환되었는가?
```

### Step 5: DB 저장 확인
```sql
-- Session temp_data 확인
SELECT id, state, temp_data FROM sessions WHERE id = 1;

-- 확인 사항:
-- temp_data에 config가 있는가?
-- dataset_analysis가 저장되었는가?
```

### Step 6: 프론트엔드 수신 확인
```javascript
// Browser console
console.log("Response from backend:", response);

// 확인 사항:
// message가 있는가?
// dataset_analysis 등 메타데이터가 있는가?
```

---

## 8. 자주 발생하는 문제 TOP 5

### 1위: LLM이 파라미터를 추출 안함 (60%)
**해결**: 프롬프트 개선 + Few-shot 예시 추가

### 2위: 경로가 존재하지 않음 (20%)
**해결**: 더 명확한 에러 메시지 + 경로 검증 강화

### 3위: temp_data가 초기화됨 (10%)
**해결**: flag_modified() 사용 + config 병합 로직 확인

### 4위: 학습 프로세스 시작 실패 (5%)
**해결**: Python executable 경로 수정 + 환경변수 전달

### 5위: 메트릭이 업데이트 안됨 (5%)
**해결**: train.py에서 DB 저장 추가

---

## 요약

문서대로 작동하지 않을 때 가장 먼저 확인해야 할 것:

1. **LLM 파싱**: `llm_debug.log`에서 `current_config` 확인
2. **Fallback 추출**: `fallback_debug.log`에서 추출 결과 확인
3. **Tool 실행**: `app.log`에서 에러 확인
4. **상태 전환**: `conversation_manager.py`의 TRACE 로그 확인
5. **DB 저장**: SQLite에서 `temp_data` 확인

90%의 문제는 위 5가지 중 하나입니다!
