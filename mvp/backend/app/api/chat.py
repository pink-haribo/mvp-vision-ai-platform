"""Chat API endpoints."""

import logging
import re
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from app.db.database import get_db
from app.db.models import Session as SessionModel, Message as MessageModel
from app.schemas import chat
from app.utils.llm import intent_parser
from app.services.dataset_analyzer import dataset_analyzer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/capabilities")
async def get_capabilities():
    """
    Get platform capabilities including supported models and parameters.

    Returns information about:
    - Available models
    - Configurable parameters
    - Default values
    """
    return {
        "models": [
            {
                "name": "resnet50",
                "display_name": "ResNet-50",
                "description": "50-layer Residual Network for image classification",
                "task_types": ["classification"],
                "supported": True
            },
            {
                "name": "resnet18",
                "display_name": "ResNet-18",
                "description": "18-layer Residual Network (coming soon)",
                "task_types": ["classification"],
                "supported": False
            },
            {
                "name": "efficientnet_b0",
                "display_name": "EfficientNet-B0",
                "description": "Efficient convolutional network (coming soon)",
                "task_types": ["classification"],
                "supported": False
            }
        ],
        "parameters": [
            {
                "name": "num_classes",
                "display_name": "클래스 수",
                "description": "분류할 클래스의 개수",
                "type": "integer",
                "required": True,
                "min": 2,
                "max": 1000,
                "default": None
            },
            {
                "name": "epochs",
                "display_name": "에포크",
                "description": "학습 반복 횟수",
                "type": "integer",
                "required": False,
                "min": 1,
                "max": 1000,
                "default": 50
            },
            {
                "name": "batch_size",
                "display_name": "배치 크기",
                "description": "한 번에 처리할 이미지 수",
                "type": "integer",
                "required": False,
                "min": 1,
                "max": 256,
                "default": 32
            },
            {
                "name": "learning_rate",
                "display_name": "학습률",
                "description": "모델 가중치 업데이트 비율",
                "type": "float",
                "required": False,
                "min": 0.00001,
                "max": 0.1,
                "default": 0.001
            },
            {
                "name": "dataset_path",
                "display_name": "데이터셋 경로",
                "description": "학습 데이터가 있는 폴더 경로",
                "type": "string",
                "required": True,
                "default": None
            }
        ],
        "task_types": [
            {
                "name": "classification",
                "display_name": "이미지 분류",
                "description": "이미지를 여러 클래스 중 하나로 분류",
                "supported": True
            },
            {
                "name": "object_detection",
                "display_name": "객체 탐지",
                "description": "이미지 내 객체의 위치와 클래스 탐지 (coming soon)",
                "supported": False
            }
        ]
    }


def _extract_dataset_path(message: str, context: str = "") -> str:
    """
    Extract dataset path from user message or context.

    Looks for patterns like:
    - ./data/datasets/sample
    - /absolute/path/to/dataset
    - data/folder
    - "path in quotes"
    """
    combined_text = f"{context}\n{message}"

    # Pattern 1: Paths starting with ./ or / (ASCII only, stop at non-path characters)
    pattern1 = r'\.?/[a-zA-Z0-9_\-./]+'
    matches = re.findall(pattern1, combined_text)
    if matches:
        # Return the last matched path (most recent in conversation)
        return matches[-1]

    # Pattern 2: Quoted paths
    pattern2 = r'["\']([^"\']+?)["\']'
    matches = re.findall(pattern2, combined_text)
    for match in reversed(matches):  # Check from most recent
        # Filter out short strings that are probably not paths
        if '/' in match or '\\' in match or 'data' in match.lower():
            return match

    # Pattern 3: Common dataset path keywords
    if 'dataset' in combined_text.lower() or 'data' in combined_text.lower():
        # Look for word that looks like a path
        words = combined_text.split()
        for word in reversed(words):
            if 'data' in word.lower() and ('/' in word or '\\' in word):
                return word.strip('.,!?')

    return None


@router.post("/sessions", response_model=chat.SessionResponse)
async def create_session(db: DBSession = Depends(get_db)):
    """Create a new chat session."""
    session = SessionModel()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions/{session_id}", response_model=chat.SessionResponse)
async def get_session(session_id: int, db: DBSession = Depends(get_db)):
    """Get a chat session by ID."""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/messages", response_model=list[chat.MessageResponse])
async def get_messages(session_id: int, db: DBSession = Depends(get_db)):
    """Get all messages in a session."""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = (
        db.query(MessageModel)
        .filter(MessageModel.session_id == session_id)
        .order_by(MessageModel.created_at)
        .all()
    )
    return messages


@router.post("/message", response_model=chat.ChatResponse)
async def send_message(request: chat.ChatRequest, db: DBSession = Depends(get_db)):
    """
    Send a message and get AI response.

    This endpoint:
    1. Creates or retrieves a session
    2. Saves the user message
    3. Parses intent using Claude
    4. Generates and saves assistant response
    5. Returns both messages and parsed intent
    """
    logger.debug(f"Received chat request: session_id={request.session_id}, message={request.message[:50]}...")

    try:
        # Create or get session
        if request.session_id:
            session = db.query(SessionModel).filter(SessionModel.id == request.session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            session = SessionModel()
            db.add(session)
            db.commit()
            db.refresh(session)

        logger.debug(f"Using session ID: {session.id}")

        # Save user message
        user_message = MessageModel(
            session_id=session.id,
            role="user",
            content=request.message,
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)

        logger.debug(f"Saved user message ID: {user_message.id}")

        # Get conversation context (last 10 messages for better memory)
        previous_messages = (
            db.query(MessageModel)
            .filter(MessageModel.session_id == session.id)
            .order_by(MessageModel.created_at.desc())
            .limit(10)
            .all()
        )

        # Build rich context with clear formatting
        context_parts = []
        for msg in reversed(previous_messages[:-1]):  # Exclude current message
            context_parts.append(f"[{msg.role.upper()}]: {msg.content}")

        context = "\n".join(context_parts) if context_parts else None

        logger.debug("Calling LLM for intent parsing...")

        # Try to extract dataset path from current message AND full conversation history
        dataset_info = None
        full_conversation = context + f"\n[USER]: {request.message}" if context else request.message
        dataset_path = _extract_dataset_path(request.message, full_conversation)
        logger.debug(f"Extracted dataset path: {dataset_path}")

        if dataset_path:
            logger.debug(f"Dataset path detected: {dataset_path}")
            logger.debug("Analyzing dataset...")
            dataset_info = dataset_analyzer.analyze(dataset_path)
            logger.debug(f"Dataset analysis complete: {dataset_info.get('format')}, "
                        f"{dataset_info.get('num_classes')} classes")

            # Add dataset analysis to context for LLM
            if dataset_info and dataset_info.get('exists'):
                analysis_summary = (
                    f"\n\n[DATASET ANALYSIS - VERY IMPORTANT]:\n"
                    f"- Dataset path: {dataset_info.get('resolved_path', dataset_path)}\n"
                    f"- Format: {dataset_info.get('format')}\n"
                    f"- Number of classes: {dataset_info.get('num_classes')}\n"
                    f"- Class names: {', '.join(dataset_info.get('class_names', []))}\n"
                    f"- Total images: {dataset_info.get('total_images')}\n"
                    f"- Has train/val split: {dataset_info.get('has_train_val_split')}"
                )
                context = (context + analysis_summary) if context else analysis_summary

        # Parse intent (with dataset info if available)
        parsed_result = await intent_parser.parse_message(
            request.message,
            context if context else None,
            dataset_info
        )

        logger.debug(f"Parsed result status: {parsed_result.get('status')}")
        if parsed_result.get("status") == "error":
            logger.error(f"LLM error: {parsed_result.get('error')} - Detail: {parsed_result.get('detail')}")

        # Generate response
        assistant_content = await intent_parser.generate_response(request.message, parsed_result)

        logger.debug(f"Generated response: {assistant_content[:50]}...")

        # Save assistant message
        assistant_message = MessageModel(
            session_id=session.id,
            role="assistant",
            content=assistant_content,
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)

        logger.debug(f"Saved assistant message ID: {assistant_message.id}")

        # Return response
        return chat.ChatResponse(
            session_id=session.id,
            user_message=user_message,
            assistant_message=assistant_message,
            parsed_intent=parsed_result if parsed_result.get("status") == "complete" else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
