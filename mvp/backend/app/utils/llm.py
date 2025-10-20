"""LLM integration for intent parsing."""

import json
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings


class IntentParser:
    """Parse user intent using Gemini API."""

    def __init__(self):
        """Initialize the intent parser."""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=settings.GOOGLE_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )

        self.system_prompt = """You are an AI assistant for a computer vision training platform.

LANGUAGE REQUIREMENT:
- You MUST respond in Korean (한국어) at all times
- All clarification questions must be in Korean
- All confirmations and responses must be in Korean
- Never respond in English unless explicitly asked by the user

CRITICAL CONTEXT HANDLING RULES - FOLLOW THESE STRICTLY:
1. READ THE ENTIRE CONVERSATION HISTORY CAREFULLY - information may have been provided in earlier messages
2. If dataset analysis results are provided, USE THOSE VALUES (e.g., if analysis shows "10 classes", use num_classes=10)
3. If a user says "dataset path is X" or provides a path, REMEMBER IT - do not ask again
4. If a user confirms information (e.g., "yes, 10 classes"), ACCEPT IT - do not ask again
5. When user says "I already told you", CHECK THE CONTEXT for that information
6. NEVER ask for information that was already provided in previous messages or dataset analysis
7. **CRITICAL**: If user mentions task type in FIRST message (e.g., "객체 탐지 모델을 학습하고 싶어"),
   REMEMBER IT for all subsequent messages. DO NOT ask again "what task?"
8. **CRITICAL**: Build up configuration across multiple messages:
   - Message 1: "객체 탐지" → task_type="object_detection"
   - Message 2: "가장 작은 모델" → model_name="yolov8n"
   - Message 3: "데이터셋은 X" → dataset_path="X"
   - Combine all information from ALL previous messages

Your task is to parse user messages and extract training configuration.

SUPPORTED CAPABILITIES:

Frameworks:
- "timm": PyTorch Image Models for image classification (resnet18, resnet50, efficientnet_b0)
- "ultralytics": Ultralytics YOLO for detection, segmentation, pose estimation (yolov8n, yolov8s, yolov8m)

Task Types:
- "image_classification": Image classification (requires num_classes)
- "object_detection": Object detection (bounding boxes)
- "instance_segmentation": Instance segmentation (pixel-level masks)
- "pose_estimation": Human pose estimation (keypoints)

Models by Framework (ordered by size/speed):
- timm (classification):
  * resnet18 (가장 작음/빠름)
  * resnet50 (중간)
  * efficientnet_b0 (효율적)
- ultralytics (detection/segmentation/pose):
  * yolov8n (nano - 가장 작음/빠름/가벼움)
  * yolov8s (small - 중간)
  * yolov8m (medium - 크고 정확함)

Dataset Formats:
- "imagefolder": PyTorch ImageFolder format (for classification)
  * Structure: dataset_path/train/class_name/image.jpg
- "yolo": YOLO format with data.yaml (for detection, segmentation, pose)
  * Has data.yaml file in dataset_path
- "coco": MS COCO JSON format (for detection, segmentation)
  * Has annotations/*.json files

REQUIRED INFORMATION TO EXTRACT:
- framework: "timm" or "ultralytics" (infer from model or task)
- model_name: Model name (e.g., "resnet50", "yolov8n")
- task_type: Task type (e.g., "image_classification", "object_detection")
- dataset_path: Path to dataset **[REQUIRED - NEVER return complete without this]**
  * If mentioned anywhere in conversation, USE IT
  * If dataset analysis shows a path, USE IT
  * If still missing, status="needs_clarification"
- dataset_format: "imagefolder" or "yolo" (infer from task_type or dataset structure)
  * Classification → imagefolder
  * Detection/Segmentation/Pose → yolo
- num_classes: Number of classes (ONLY for classification tasks, integer, minimum 2)
  * If dataset analysis provides this, USE IT
  * If user confirms a number, USE IT
  * SKIP for non-classification tasks (detection, segmentation, pose)
- epochs: Number of training epochs (default: 50)
- batch_size: Batch size (default: 32, or auto-calculate based on dataset size)
- learning_rate: Learning rate (default: 0.001)

**VALIDATION BEFORE RETURNING complete - READ THIS CAREFULLY:**
!!!CRITICAL RULE!!!: dataset_path is ABSOLUTELY REQUIRED for status="complete"
- Check dataset_path value:
  * If dataset_path is null → status="needs_clarification" (NOT complete!)
  * If dataset_path is "None" (string) → status="needs_clarification" (NOT complete!)
  * If dataset_path is empty string → status="needs_clarification" (NOT complete!)
  * If dataset_path is missing → status="needs_clarification" (NOT complete!)
  * ONLY if dataset_path is a valid path → can consider complete

- THEN check other required fields:
  * MUST have: framework, model_name, task_type
  * For classification tasks ONLY: MUST have num_classes
  * If ANY required field is missing: return status="needs_clarification"

**NEVER EVER return status="complete" without a valid dataset_path!**
**If you return complete with dataset_path=null, that is a CRITICAL ERROR!**

FRAMEWORK INFERENCE LOGIC:
- If user mentions "YOLO" or "yolov8" → framework="ultralytics"
- If user mentions "detection", "segmentation", "pose" → framework="ultralytics", model_name="yolov8n" (default)
- If user mentions "ResNet", "EfficientNet", or just "classification" → framework="timm"
- If unclear, default to timm for classification

TASK INFERENCE LOGIC:
- If user says "분류", "classification" → task_type="image_classification"
- If user says "탐지", "detection", "객체 탐지" → task_type="object_detection"
- If user says "분할", "segmentation" → task_type="instance_segmentation"
- If user says "자세", "pose" → task_type="pose_estimation"

MODEL SIZE SELECTION LOGIC:
When user asks for smallest/lightest/fastest model:
- For ultralytics tasks → select "yolov8n" (nano)
- For timm classification → select "resnet18"
Keywords: "가장 작은", "작은", "가벼운", "빠른", "smallest", "lightest", "fastest", "nano"

When user asks for largest/most accurate:
- For ultralytics tasks → select "yolov8m" (medium)
- For timm classification → select "resnet50"
Keywords: "큰", "정확한", "largest", "accurate"

DATASET FORMAT AUTO-DETECTION:
If dataset analysis is provided, use the detected format:
- If analysis shows "format: yolo" → dataset_format="yolo"
- If analysis shows "format: imagefolder" → dataset_format="imagefolder"
- If analysis shows "format: coco" → dataset_format="coco"
- ALWAYS prefer the analyzed format over inference

RESPONSE FORMAT:

If ALL REQUIRED information is available (from ANY source: current message, conversation history, OR dataset analysis):
{
  "status": "complete",
  "config": {
    "framework": "timm" or "ultralytics",
    "model_name": "resnet50" or "yolov8n" etc.,
    "task_type": "image_classification" or "object_detection" etc.,
    "dataset_path": "<valid path - MUST NOT BE null/None/empty>",
    "dataset_format": "imagefolder" or "yolo",
    "num_classes": <int or null>,  // null for non-classification tasks
    "epochs": <int>,
    "batch_size": <int>,
    "learning_rate": <float>
  }
}

**IMPORTANT**: Only return "complete" if:
1. dataset_path is provided and not null/None/empty
2. For classification: num_classes is provided
3. framework, model_name, task_type are all valid

If ANY required information is missing:
{
  "status": "needs_clarification",
  "missing_fields": ["field1", "field2"],
  "clarification": "<natural language question - be specific and ask for MISSING info only>"
}

**CONTEXT USAGE**: When asking for missing info, acknowledge what you already know:
- Good: "데이터셋 경로를 알려주세요. (이미 설정된 것: yolov8n 모델, 객체 탐지 작업)"
- Bad: "어떤 작업을 하시겠어요?" when user already said "객체 탐지" in previous message

EXAMPLES OF GOOD BEHAVIOR:
- User: "Path is /data/cats_dogs" → Remember this path for later
- Dataset analysis: "10 classes detected" → Use num_classes=10
- Dataset analysis: "format: yolo" → Use dataset_format="yolo"
- User: "Train for 200 epochs" → Use epochs=200
- User: "YOLOv8로 객체 탐지" → framework="ultralytics", task_type="object_detection", model_name="yolov8n", dataset_format="yolo", num_classes=null
- User: "가장 작은 모델로 객체 탐지" → framework="ultralytics", task_type="object_detection", model_name="yolov8n"
- User: "가장 가벼운 모델 사용할게" + previous context shows detection → model_name="yolov8n"
- User: "ResNet으로 분류" → framework="timm", task_type="image_classification", model_name="resnet50"
- User: "빠른 모델로 분류" → framework="timm", task_type="image_classification", model_name="resnet18"

EXAMPLES OF BAD BEHAVIOR (NEVER DO THIS):
- Asking "what's the dataset path?" when it was mentioned 2 messages ago
- Asking "how many classes?" when dataset analysis already showed "10 classes"
- Asking "how many classes?" for object detection (it doesn't need num_classes!)
- Asking "confirm the number of classes" when user already said "yes, 10"
- Asking "which model?" when user said "가장 작은" or "smallest" (should auto-select yolov8n or resnet18)
- Asking "dataset format?" when dataset analysis already detected it

Always respond with valid JSON only, no additional text.
IMPORTANT: Do not wrap the JSON in markdown code blocks. Return raw JSON directly without ```json or ``` markers."""

    def _format_dataset_info(self, dataset_info: Dict[str, Any]) -> str:
        """Format dataset analysis result for LLM context."""
        if dataset_info.get("error"):
            return f"Dataset analysis: ERROR - {dataset_info['error']}"

        info_parts = ["Dataset analysis result:"]
        info_parts.append(f"- Format: {dataset_info.get('format', 'Unknown')}")
        info_parts.append(f"- Is labeled: {dataset_info.get('is_labeled', False)}")

        if dataset_info.get('is_labeled'):
            info_parts.append(f"- Number of classes detected: {dataset_info.get('num_classes', 0)}")
            if dataset_info.get('class_names'):
                class_list = ', '.join(dataset_info['class_names'])
                info_parts.append(f"- Class names: [{class_list}]")

        info_parts.append(f"- Total images: {dataset_info.get('total_images', 0)}")
        info_parts.append(f"- Has train/val split: {dataset_info.get('has_train_val_split', False)}")

        if dataset_info.get('image_counts'):
            info_parts.append("- Image distribution:")
            for split, count in dataset_info['image_counts'].items():
                info_parts.append(f"  * {split}: {count} images")

        if dataset_info.get('warnings'):
            info_parts.append("- Warnings:")
            for warning in dataset_info['warnings']:
                info_parts.append(f"  * {warning}")

        info_parts.append("")
        info_parts.append("IMPORTANT: Use the detected values above when generating the config.")
        info_parts.append("If user specifies different values (e.g., says '10 classes' but dataset has 2), warn them about the mismatch.")

        return "\n".join(info_parts)

    async def parse_message(
        self,
        user_message: str,
        context: Optional[str] = None,
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse user message and extract training intent.

        Args:
            user_message: User's natural language message
            context: Optional context from previous messages
            dataset_info: Optional dataset analysis result

        Returns:
            Dictionary with parsing result
        """
        messages = [
            SystemMessage(content=self.system_prompt),
        ]

        if context:
            messages.append(HumanMessage(content=f"Previous context: {context}"))

        # Add dataset analysis information if available
        if dataset_info and dataset_info.get("exists"):
            dataset_context = self._format_dataset_info(dataset_info)
            messages.append(HumanMessage(content=dataset_context))

        messages.append(HumanMessage(content=f"User message: {user_message}"))

        try:
            response = await self.llm.ainvoke(messages)
            print(f"[DEBUG] LLM raw response type: {type(response.content)}")
            print(f"[DEBUG] LLM raw response content: {response.content}")

            # Remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            elif content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()

            print(f"[DEBUG] Cleaned content: {content}")
            result = json.loads(content)
            return result
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON decode error: {str(e)}")
            print(f"[DEBUG] Response content that failed to parse: '{response.content}'")
            return {
                "status": "error",
                "error": "Failed to parse LLM response",
                "detail": str(e),
            }
        except Exception as e:
            print(f"[DEBUG] LLM exception: {str(e)}")
            print(f"[DEBUG] Exception type: {type(e)}")
            return {
                "status": "error",
                "error": "LLM request failed",
                "detail": str(e),
            }

    async def generate_response(self, user_message: str, parsed_result: Dict[str, Any]) -> str:
        """
        Generate natural language response based on parsing result.

        Args:
            user_message: User's original message
            parsed_result: Result from parse_message

        Returns:
            Natural language response
        """
        if parsed_result.get("status") == "complete":
            config = parsed_result.get("config", {})

            # Build response with optional fields
            response_parts = [
                "학습 설정을 확인했습니다:\n",
                f"- 프레임워크: {config.get('framework', 'N/A')}",
                f"- 모델: {config.get('model_name', 'N/A')}",
                f"- 작업 유형: {config.get('task_type', 'N/A')}",
            ]

            # Add num_classes only if it exists (classification tasks only)
            if config.get('num_classes') is not None:
                response_parts.append(f"- 클래스 수: {config.get('num_classes')}")

            response_parts.extend([
                f"- 데이터셋: {config.get('dataset_path', 'N/A')}",
                f"- 데이터셋 형식: {config.get('dataset_format', 'N/A')}",
                f"- 에포크: {config.get('epochs', 50)}",
                f"- 배치 크기: {config.get('batch_size', 32)}",
                f"- 학습률: {config.get('learning_rate', 0.001)}",
                "\n학습을 시작하시겠습니까?"
            ])

            return "\n".join(response_parts)

        elif parsed_result.get("status") == "needs_clarification":
            return parsed_result.get("clarification", "추가 정보가 필요합니다.")

        else:
            return "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."


# Global instance
intent_parser = IntentParser()
