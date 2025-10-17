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

Your task is to parse user messages and extract training configuration.

The user can train ResNet50 models for image classification tasks.

Extract the following information:
- model_name: Must be "resnet50" (only supported model)
- task_type: Must be "classification" (only supported task)
- num_classes: Number of classes (integer, minimum 2)
- dataset_path: Path to dataset (if provided)
- epochs: Number of training epochs (default: 50)
- batch_size: Batch size (default: 32)
- learning_rate: Learning rate (default: 0.001)

If the user's message contains all required information, respond with JSON:
{
  "status": "complete",
  "config": {
    "model_name": "resnet50",
    "task_type": "classification",
    "num_classes": <int>,
    "dataset_path": "<path>",
    "epochs": <int>,
    "batch_size": <int>,
    "learning_rate": <float>
  }
}

If information is missing, respond with JSON:
{
  "status": "needs_clarification",
  "missing_fields": ["field1", "field2"],
  "clarification": "<natural language question to ask user>"
}

Always respond with valid JSON only, no additional text."""

    async def parse_message(self, user_message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse user message and extract training intent.

        Args:
            user_message: User's natural language message
            context: Optional context from previous messages

        Returns:
            Dictionary with parsing result
        """
        messages = [
            SystemMessage(content=self.system_prompt),
        ]

        if context:
            messages.append(HumanMessage(content=f"Previous context: {context}"))

        messages.append(HumanMessage(content=f"User message: {user_message}"))

        try:
            response = await self.llm.ainvoke(messages)
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": "Failed to parse LLM response",
                "detail": str(e),
            }
        except Exception as e:
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
            return f"""학습 설정을 확인했습니다:

- 모델: {config.get('model_name', 'resnet50')}
- 작업: {config.get('task_type', 'classification')}
- 클래스 수: {config.get('num_classes', 'N/A')}
- 데이터셋: {config.get('dataset_path', 'N/A')}
- 에포크: {config.get('epochs', 50)}
- 배치 크기: {config.get('batch_size', 32)}
- 학습률: {config.get('learning_rate', 0.001)}

학습을 시작하시겠습니까?"""

        elif parsed_result.get("status") == "needs_clarification":
            return parsed_result.get("clarification", "추가 정보가 필요합니다.")

        else:
            return "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."


# Global instance
intent_parser = IntentParser()
