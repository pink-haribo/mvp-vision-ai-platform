"""
LLM integration with Dual Provider Support (Phase 1+2)

This module supports both OpenAI-compatible API and Google Gemini API.
Provider is selected via LLM_PROVIDER environment variable.

Supported providers:
- openai: OpenAI, Azure OpenAI, LocalAI, Ollama, vLLM, LiteLLM, etc.
- gemini: Google Gemini API
"""

import json
import logging
from typing import Optional, Dict, Any

from app.core.config import settings
from app.models.conversation import (
    ActionType,
    GeminiActionResponse,  # Keep name for backward compatibility
    ConversationState,
)

logger = logging.getLogger(__name__)


class StructuredIntentParser:
    """
    Parse user intent using LLM with structured output

    Supports dual providers:
    - OpenAI-compatible API (OpenAI, Azure, LocalAI, Ollama, vLLM, etc.)
    - Google Gemini API
    """

    def __init__(self):
        """Initialize the structured intent parser based on LLM_PROVIDER setting"""
        self.provider = settings.LLM_PROVIDER.lower()
        self.model_name = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE

        if self.provider == "gemini":
            self._init_gemini()
        else:
            self._init_openai()

        logger.info(f"LLM Provider initialized: {self.provider}, Model: {self.model_name}")

    def _init_openai(self):
        """Initialize OpenAI-compatible client"""
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    def _init_gemini(self):
        """Initialize Google Gemini client"""
        import google.generativeai as genai
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
            }
        )

    async def _call_openai(self, system_prompt: str, user_content: str) -> str:
        """Call OpenAI-compatible API and return response text"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    async def _call_gemini(self, system_prompt: str, user_content: str) -> str:
        """Call Google Gemini API and return response text"""
        import asyncio
        # Gemini uses a single prompt format
        full_prompt = f"{system_prompt}\n\n{user_content}"
        # Run sync Gemini call in executor for async compatibility
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.gemini_model.generate_content(full_prompt)
        )
        return response.text.strip()

    def _build_system_prompt(self, state: ConversationState) -> str:
        """Build state-specific system prompt"""

        base_prompt = """You are an AI assistant for a computer vision training platform.

LANGUAGE REQUIREMENT:
- You MUST respond in Korean (한국어) at all times
- All messages must be in Korean
- Never respond in English unless explicitly asked

You must respond with structured JSON containing:
- action: one of the supported action types
- message: user-friendly message in Korean
- other fields based on action type

SUPPORTED ACTIONS (Training Setup):
1. ask_clarification: Need more information
2. show_project_options: Show project selection menu (1: new, 2: existing, 3: skip)
3. show_project_list: List available projects
4. create_project: Create new project
5. select_project: Select existing project
6. skip_project: Skip project (use Uncategorized)
7. confirm_training: Ask for final confirmation
8. start_training: Start training (final action)
9. error: Error occurred

PHASE 1 ACTIONS (Dataset/Model/Training Control):
10. analyze_dataset: Analyze dataset structure and quality
    - Use when user provides dataset path and wants analysis
11. show_dataset_analysis: Display dataset analysis results
12. list_datasets: List available datasets
    - Use when user asks: "기본 데이터셋", "사용 가능한 데이터셋", "어떤 데이터셋이 있어", "built-in datasets"
    - Lists datasets from C:\datasets (built-in) and other paths
13. search_models: Search for models by task/framework
14. show_model_info: Show detailed model information
15. recommend_models: Recommend models based on dataset
16. show_training_status: Show training progress and metrics
17. stop_training: Stop running training job
18. list_training_jobs: List training jobs with filters
19. start_quick_inference: Run inference on single image

"""

        if state == ConversationState.INITIAL or state == ConversationState.GATHERING_CONFIG:
            return base_prompt + """
CURRENT STATE: Gathering training configuration

Your task: Extract training configuration from user messages.

SUPPORTED CAPABILITIES:
- Frameworks: timm (classification), ultralytics (detection/segmentation/pose)
- Models:
  * timm: resnet18, resnet50, efficientnet_b0
  * ultralytics: yolov8n, yolov8s, yolov8m, yolo11n, yolo11s, yolo11m
- Task types: image_classification, object_detection, instance_segmentation, pose_estimation
- Dataset formats: imagefolder, coco, yolo

⚠️ **CRITICAL**: Only recommend or mention models listed above!
- DO NOT suggest models not in this list (e.g., yolov5, yolov7, mobilenet, etc.)
- If user asks for unsupported model, suggest closest supported alternative
- Always validate model_name against the supported list before returning

REQUIRED FIELDS:
- framework
- model_name
- task_type
- dataset_path
- epochs
- batch_size
- learning_rate

OPTIONAL FIELDS:
- num_classes (for classification)
- dataset_format (default: imagefolder)

═══════════════════════════════════════════════════════════════
🚨 CRITICAL RULE - READ THIS CAREFULLY 🚨
═══════════════════════════════════════════════════════════════

**RULE #1: NEVER DROP PREVIOUS VALUES**
When you receive current_config in the context, you MUST include EVERY SINGLE field from it in your response.

Example (CORRECT):
Context: current_config = {"framework": "timm", "model_name": "resnet18"}
User: "C:\\datasets\\cls\\imagenet-10"
Your response current_config MUST have:
{
  "framework": "timm",          ← KEEP from context
  "model_name": "resnet18",     ← KEEP from context
  "dataset_path": "C:\\datasets\\cls\\imagenet-10"  ← ADD new
}

Example (WRONG - DO NOT DO THIS):
{
  "dataset_path": "C:\\datasets\\cls\\imagenet-10"  ← Missing framework and model_name!
}

**RULE #2: COPY-PASTE PREVIOUS VALUES**
If you see a field in the context's current_config, COPY IT EXACTLY to your response.
DO NOT try to "simplify" or "optimize" by removing fields.

**RULE #3: VALIDATION CHECKLIST**
Before returning your response, check:
[ ] Did I copy ALL fields from context's current_config?
[ ] Did I add the new information from user's message?
[ ] Is my current_config a SUPERSET of the previous one?

═══════════════════════════════════════════════════════════════

🚨 ACTION SELECTION RULES - CRITICAL 🚨
═══════════════════════════════════════════════════════════════
**BEFORE choosing ask_clarification, CHECK THESE RULES FIRST:**

1. If user asks about "기본 데이터셋", "사용 가능한 데이터셋", "어떤 데이터셋", "built-in dataset", "제공되는 데이터셋"
   → **MUST use action="list_datasets"**
   → Do NOT use ask_clarification for this!

   Example:
   User: "기본으로 제공되는 데이터셋이 있어?"
   ✅ CORRECT: {"action": "list_datasets", "message": "사용 가능한 데이터셋을 확인하고 있습니다..."}
   ❌ WRONG: {"action": "ask_clarification", "message": "기본으로 제공되는 데이터셋은 없습니다..."}

2. If user provides dataset path (e.g., "C:\\datasets\\...") and wants analysis
   → action="analyze_dataset"

3. If user asks about model features/comparison
   → action="search_models" or "show_model_info"
═══════════════════════════════════════════════════════════════

INFERENCE RULES:
1. If user mentions "ResNet" or "EfficientNet" → framework="timm", task_type="image_classification"
2. If user mentions "YOLO" → framework="ultralytics", task_type="object_detection" (or ask which task)
3. If user says "적절히" or "기본값" → use defaults (epochs=50, batch_size=32, learning_rate=0.001)
4. Build config incrementally across messages - PRESERVE all previously collected values

ADVANCED CONFIG PRESETS:
사용자가 프리셋을 언급하면 해당 프리셋을 advanced_config 필드에 설정하세요.
사용 가능한 프리셋:
- "basic": 간단한 학습 설정 (minimal augmentation, Adam optimizer)
- "standard": 균형잡힌 설정 (AdamW optimizer, cosine scheduler, moderate augmentation)
- "aggressive": 강력한 augmentation (작은 데이터셋에 적합)
- "fine_tuning": 사전 학습된 모델 fine-tuning에 최적화

프리셋 사용 예시:
User: "basic 프리셋으로 학습하고 싶어요"
→ Set advanced_config="basic" in config
→ Message: "Basic 프리셋으로 설정합니다. 간단한 augmentation과 Adam optimizer를 사용합니다."

User: "standard 프리셋 사용할게"
→ Set advanced_config="standard" in config
→ Message: "Standard 프리셋으로 설정합니다. AdamW optimizer와 cosine scheduler를 사용합니다."

⚠️ IMPORTANT: 프리셋을 사용할 때는 config에 "advanced_config" 필드를 추가하세요.
예: {"framework": "timm", "model_name": "resnet18", "advanced_config": "standard"}

WHEN USER REQUESTS DATASET ANALYSIS:
If user provides dataset_path AND includes keywords like:
- "분석", "분석해줘", "분석 부탁"
- "확인", "확인해줘", "체크"
- "검증", "살펴봐", "보여줘"
→ Return action="analyze_dataset" with the dataset_path in current_config
→ Message: "데이터셋을 분석하고 있습니다..."

Example:
User: "C:\\datasets\\det-coco8 이게 데이터셋 경로야 분석 부탁해"
```json
{
  "action": "analyze_dataset",
  "message": "데이터셋을 분석하고 있습니다...",
  "current_config": {
    "framework": "ultralytics",
    "task_type": "object_detection",
    "model_name": "yolov8n",
    "dataset_path": "C:\\\\datasets\\\\det-coco8",
    "dataset_format": "yolo"
  }
}
```

WHEN CONFIG IS COMPLETE:
Return action="show_project_options" with the complete config (including ALL previously collected fields).

WHEN INFO IS MISSING:
Return action="ask_clarification" with missing_fields list AND current_config with ALL collected values.

Example conversation flow (CRITICAL - follow this pattern):

User: "ResNet으로 학습하고 싶어"
You return:
```json
{
  "action": "ask_clarification",
  "message": "ResNet 모델을 선택하셨습니다. 어떤 ResNet 모델을 사용하시겠어요? (resnet18, resnet50)",
  "missing_fields": ["model_name", "dataset_path", "epochs", "batch_size", "learning_rate"],
  "current_config": {"framework": "timm", "task_type": "image_classification"}
}
```

User: "resnet18"
You return (NOTE: MUST include previous values!):
```json
{
  "action": "ask_clarification",
  "message": "resnet18 모델을 선택했습니다. 데이터셋 경로를 알려주세요.",
  "missing_fields": ["dataset_path", "epochs", "batch_size", "learning_rate"],
  "current_config": {
    "framework": "timm",
    "task_type": "image_classification",
    "model_name": "resnet18"
  }
}
```

User: "C:\\datasets\\cls\\imagenet-10"
You return (NOTE: MUST include all previous values!):
```json
{
  "action": "ask_clarification",
  "message": "데이터셋 경로를 설정했습니다. 학습 횟수(epochs), 배치 크기(batch_size), 학습률(learning_rate)을 알려주세요.",
  "missing_fields": ["epochs", "batch_size", "learning_rate"],
  "current_config": {
    "framework": "timm",
    "task_type": "image_classification",
    "model_name": "resnet18",
    "dataset_path": "C:\\\\datasets\\\\cls\\\\imagenet-10",
    "dataset_format": "imagefolder"
  }
}
```

User: "기본값으로 해줘"
You return (NOTE: Complete config with ALL fields!):
```json
{
  "action": "show_project_options",
  "message": "설정이 완료되었습니다. 프로젝트를 선택해주세요.\\n\\n1️⃣ 신규 프로젝트 생성\\n2️⃣ 기존 프로젝트 선택\\n3️⃣ 프로젝트 없이 실험만 진행",
  "config": {
    "framework": "timm",
    "model_name": "resnet18",
    "task_type": "image_classification",
    "dataset_path": "C:\\\\datasets\\\\cls\\\\imagenet-10",
    "dataset_format": "imagefolder",
    "num_classes": null,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

═══════════════════════════════════════════════════════════════
🔴 REMEMBER: ALWAYS INCLUDE ALL PREVIOUS CONFIG FIELDS! 🔴
═══════════════════════════════════════════════════════════════
"""

        elif state == ConversationState.SELECTING_PROJECT:
            return base_prompt + """
CURRENT STATE: Selecting project

User is choosing from 3 options. Check the user's message EXACTLY:

**CRITICAL PARSING RULES:**
1. If user message is EXACTLY "1", "1번", or contains "신규":
   → Return action="ask_clarification" with missing_fields=["project_name"]
   → Message should ask for project name

2. If user message is EXACTLY "2", "2번", or contains "기존":
   → Return action="show_project_list"

3. If user message is EXACTLY "3", "3번", or contains "건너뛰기" or "없이":
   → Return action="skip_project"

4. If user provided a project name (not a number):
   → Return action="create_project" with project_name

5. If user provided a project number from a list:
   → Return action="select_project" with project_identifier

**DO NOT** return action="show_project_options" in this state!

Example for "1":
```json
{
  "action": "ask_clarification",
  "message": "신규 프로젝트를 생성합니다. 프로젝트 이름을 입력해주세요. (설명은 선택사항입니다)\\n\\n예시: 이미지 분류 프로젝트 - 고양이와 강아지 분류",
  "missing_fields": ["project_name"]
}
```

Example for "2":
```json
{
  "action": "show_project_list",
  "message": "기존 프로젝트를 조회합니다..."
}
```

Example for "3":
```json
{
  "action": "skip_project",
  "message": "프로젝트 없이 진행합니다."
}
```
"""

        elif state == ConversationState.CREATING_PROJECT:
            return base_prompt + """
CURRENT STATE: Creating new project

User is providing project name and optional description.

Parse formats:
- "프로젝트 이름 - 설명" → Split by " - " to get name and description
- "프로젝트 이름: 설명" → Split by ": " to get name and description
- Just "프로젝트 이름" → Name only, no description

Return action="create_project" with:
- project_name (required)
- project_description (optional, only if user provided)

Examples:
```json
{
  "action": "create_project",
  "message": "'이미지 분류 프로젝트' 프로젝트를 생성했습니다.",
  "project_name": "이미지 분류 프로젝트",
  "project_description": "고양이와 강아지 분류"
}
```

```json
{
  "action": "create_project",
  "message": "'ResNet 실험' 프로젝트를 생성했습니다.",
  "project_name": "ResNet 실험",
  "project_description": null
}
```
"""

        elif state == ConversationState.CONFIRMING:
            return base_prompt + """
CURRENT STATE: Confirming training

User needs to confirm whether to start training.

If user input is:
- "예", "yes", "y", "네", "확인", "ok" → action="start_training"
- "아니오", "no", "취소", "cancel" → action="error" (or back to initial)

Example:
```json
{
  "action": "start_training",
  "message": "학습을 시작합니다..."
}
```
"""

        # ========== Phase 1 New States ==========

        elif state == ConversationState.ANALYZING_DATASET:
            return base_prompt + """
CURRENT STATE: Analyzing dataset

Dataset analysis has been completed or user is asking about dataset.

Available actions:
- show_dataset_analysis: Show analysis results
- recommend_models: Recommend models based on dataset analysis
- gather_config: Continue with training configuration (action="ask_clarification")
- analyze_dataset: Analyze another dataset

User intent examples:
- "이 데이터셋으로 학습해줘" → action="ask_clarification" (gather remaining config)
- "어떤 모델이 좋을까?" → action="recommend_models"
- "데이터셋 분석 결과 다시 보여줘" → action="show_dataset_analysis"
- "다른 데이터셋 분석해줘" → action="analyze_dataset"

Example:
```json
{
  "action": "recommend_models",
  "message": "데이터셋 분석 결과를 바탕으로 적합한 모델을 추천해드리겠습니다."
}
```
"""

        elif state == ConversationState.SELECTING_MODEL:
            return base_prompt + """
CURRENT STATE: Selecting model

User is choosing a model or requesting model information.

Available actions:
- search_models: Search for models by criteria
- show_model_info: Show detailed model information
- recommend_models: Recommend models
- ask_clarification: Continue gathering config (user selected a model)

User intent examples:
- "모델 목록 보여줘" → action="search_models"
- "resnet50 정보 알려줘" → action="show_model_info"
- "추천해줘" → action="recommend_models"
- "resnet50으로 할게" → action="ask_clarification" (update config with model_name="resnet50")

Example:
```json
{
  "action": "ask_clarification",
  "message": "ResNet-50 모델을 선택하셨습니다. 데이터셋 경로를 알려주세요.",
  "missing_fields": ["dataset_path", "epochs", "batch_size", "learning_rate"],
  "current_config": {
    "framework": "timm",
    "model_name": "resnet50",
    "task_type": "image_classification"
  }
}
```
"""

        elif state == ConversationState.MONITORING_TRAINING:
            return base_prompt + """
CURRENT STATE: Monitoring training

User is checking training status or managing training jobs.

Available actions:
- show_training_status: Show current training progress
- list_training_jobs: List all training jobs
- stop_training: Stop a running training job

User intent examples:
- "학습 상태 알려줘" → action="show_training_status"
- "학습 목록 보여줘" → action="list_training_jobs"
- "학습 중지해줘" → action="stop_training"
- "job 123 상태 알려줘" → action="show_training_status"
- "실행중인 학습 보여줘" → action="list_training_jobs"

Example:
```json
{
  "action": "show_training_status",
  "message": "학습 상태를 확인하겠습니다."
}
```
"""

        elif state == ConversationState.RUNNING_INFERENCE:
            return base_prompt + """
CURRENT STATE: Running inference

User wants to run inference on images.

Available actions:
- start_quick_inference: Run inference on a single image

User intent examples:
- "이미지 추론해줘" → action="start_quick_inference"
- "C:/images/test.jpg 예측해줘" → action="start_quick_inference"
- "job 123으로 추론해줘" → action="start_quick_inference"

Note: Extract job_id and image_path from user message. The handler will automatically find the most recent job if not specified.

Example:
```json
{
  "action": "start_quick_inference",
  "message": "추론을 실행하겠습니다."
}
```
"""

        elif state == ConversationState.VIEWING_RESULTS:
            return base_prompt + """
CURRENT STATE: Viewing results

User is viewing training or inference results.

This state is for displaying results. User might want to:
- Start another training
- Run inference
- View different results

Analyze user intent and route to appropriate action.

Example:
```json
{
  "action": "ask_clarification",
  "message": "다른 작업을 도와드릴까요?"
}
```
"""

        elif state == ConversationState.IDLE:
            return base_prompt + """
CURRENT STATE: Idle (waiting for user request)

User can request any action. Analyze their intent and route to:
- Dataset actions (analyze_dataset, list_datasets)
- Model actions (search_models, recommend_models)
- Training setup (ask_clarification to gather config)
- Training monitoring (show_training_status, list_training_jobs)
- Inference (start_quick_inference)

Example for dataset query:
```json
{
  "action": "analyze_dataset",
  "message": "데이터셋을 분석하겠습니다."
}
```

Example for training setup:
```json
{
  "action": "ask_clarification",
  "message": "새로운 학습을 시작하시겠습니까? 어떤 모델을 사용하시겠어요?",
  "missing_fields": ["framework", "model_name", "task_type", "dataset_path", "epochs", "batch_size", "learning_rate"],
  "current_config": {}
}
```
"""

        else:
            return base_prompt

    async def parse_intent(
        self,
        user_message: str,
        state: ConversationState,
        context: Optional[str] = None,
        temp_data: Optional[Dict[str, Any]] = None
    ) -> GeminiActionResponse:
        """
        Parse user intent with current state and context

        Args:
            user_message: User's message
            state: Current conversation state
            context: Previous conversation context
            temp_data: Temporary data from session (config, etc.)

        Returns:
            GeminiActionResponse: Structured action response
        """
        try:
            # Build system prompt based on state
            system_prompt = self._build_system_prompt(state)

            # Build full prompt
            prompt_parts = [system_prompt]

            # Add context if available
            if context:
                prompt_parts.append(f"\n\n=== CONVERSATION HISTORY ===\n{context}\n")

            # Add current config if available
            if temp_data and "config" in temp_data:
                # TRACE: Step 2 - Before calling Gemini
                print(f"\n[TRACE-2-LLM-IN] Passing config to Gemini:")
                print(f"  config: {json.dumps(temp_data['config'], ensure_ascii=False)}")

                config_str = json.dumps(temp_data["config"], ensure_ascii=False, indent=2)
                prompt_parts.append(f"\n\n=== CURRENT CONFIG (YOU MUST INCLUDE ALL OF THESE IN YOUR RESPONSE!) ===\n{config_str}\n")

                # Extra emphasis
                config_fields = list(temp_data["config"].keys())
                prompt_parts.append(f"\n🚨 MANDATORY: Your response MUST include these {len(config_fields)} fields: {', '.join(config_fields)}\n")
            else:
                print(f"\n[TRACE-2-LLM-IN] NO CONFIG to pass to Gemini (temp_data has no 'config' key)")

            # Add user message
            prompt_parts.append(f"\n\n=== USER MESSAGE ===\n{user_message}\n")

            prompt_parts.append("\n\n**IMPORTANT**: Respond ONLY with valid JSON. No markdown, no code blocks, no explanations. Just the JSON object.")

            # Build system prompt and user content
            system_prompt = self._build_system_prompt(state)
            user_content = "\n".join(prompt_parts[1:])  # Everything except system prompt

            logger.debug(f"LLM prompt (state={state}):\n{user_content[:500]}...")

            # Call LLM based on provider
            if self.provider == "gemini":
                response_text = await self._call_gemini(system_prompt, user_content)
            else:
                response_text = await self._call_openai(system_prompt, user_content)

            logger.debug(f"LLM response: {response_text}")

            # DEBUG: Write raw LLM response to file
            try:
                with open("llm_responses.txt", "a", encoding="utf-8") as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"State: {state}, User msg: {user_message}\n")
                    f.write(f"LLM Response:\n{response_text}\n")
                    f.write("="*80 + "\n")
            except Exception:
                pass  # Silently ignore logging errors

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                # Extract JSON from code block
                lines = response_text.split("\n")
                # Skip first line (```json or ```)
                # Take until the closing ```
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        if in_code_block:
                            break  # End of code block
                        else:
                            in_code_block = True  # Start of code block
                            continue
                    if in_code_block:
                        json_lines.append(line)
                response_text = "\n".join(json_lines).strip()

            # Parse JSON
            response_data = json.loads(response_text)

            # Validate with Pydantic
            action_response = GeminiActionResponse(**response_data)

            # TRACE: Step 3 - After Gemini responds
            print(f"\n[TRACE-3-LLM-OUT] Gemini response:")
            print(f"  action: {action_response.action}")
            if action_response.current_config:
                print(f"  current_config: {json.dumps(action_response.current_config, ensure_ascii=False)}")
                print(f"  current_config keys: {list(action_response.current_config.keys())}")
            else:
                print(f"  current_config: NULL/NONE")
            if action_response.config:
                print(f"  config: {json.dumps(action_response.config, ensure_ascii=False)}")

            logger.info(f"Parsed action: {action_response.action}")

            return action_response

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}\nResponse: {response_text}")
            return GeminiActionResponse(
                action=ActionType.ERROR,
                message="죄송합니다. 응답 처리 중 오류가 발생했습니다.",
                error_message=f"JSON parsing failed: {str(e)}"
            )

        except Exception as e:
            logger.error(f"Intent parsing error: {e}", exc_info=True)
            return GeminiActionResponse(
                action=ActionType.ERROR,
                message="죄송합니다. 요청 처리 중 오류가 발생했습니다.",
                error_message=str(e)
            )


# Global instance
structured_intent_parser = StructuredIntentParser()
