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
    - Available frameworks
    - Available models per framework
    - Supported task types
    - Configurable parameters
    - Default values
    """
    return {
        "frameworks": [
            {
                "name": "timm",
                "display_name": "PyTorch Image Models (timm)",
                "description": "Image classification with pretrained models",
                "supported": True
            },
            {
                "name": "ultralytics",
                "display_name": "Ultralytics YOLO",
                "description": "Object detection, segmentation, and pose estimation",
                "supported": True
            },
            {
                "name": "transformers",
                "display_name": "HuggingFace Transformers",
                "description": "Vision-Language models (coming soon)",
                "supported": False
            }
        ],
        "models": [
            # timm models
            {
                "name": "resnet50",
                "display_name": "ResNet-50",
                "description": "50-layer Residual Network",
                "framework": "timm",
                "task_types": ["image_classification"],
                "supported": True
            },
            {
                "name": "resnet18",
                "display_name": "ResNet-18",
                "description": "18-layer Residual Network",
                "framework": "timm",
                "task_types": ["image_classification"],
                "supported": True
            },
            {
                "name": "efficientnet_b0",
                "display_name": "EfficientNet-B0",
                "description": "Efficient convolutional network",
                "framework": "timm",
                "task_types": ["image_classification"],
                "supported": True
            },
            # Ultralytics models
            {
                "name": "yolov8n",
                "display_name": "YOLOv8 Nano",
                "description": "Lightweight object detection model",
                "framework": "ultralytics",
                "task_types": ["object_detection", "instance_segmentation", "pose_estimation", "image_classification"],
                "supported": True
            },
            {
                "name": "yolov8s",
                "display_name": "YOLOv8 Small",
                "description": "Small object detection model",
                "framework": "ultralytics",
                "task_types": ["object_detection", "instance_segmentation", "pose_estimation", "image_classification"],
                "supported": True
            },
            {
                "name": "yolov8m",
                "display_name": "YOLOv8 Medium",
                "description": "Medium object detection model",
                "framework": "ultralytics",
                "task_types": ["object_detection", "instance_segmentation", "pose_estimation", "image_classification"],
                "supported": True
            }
        ],
        "task_types": [
            {
                "name": "image_classification",
                "display_name": "이미지 분류",
                "description": "이미지를 여러 클래스 중 하나로 분류",
                "frameworks": ["timm", "ultralytics"],
                "supported": True
            },
            {
                "name": "object_detection",
                "display_name": "객체 탐지",
                "description": "이미지 내 객체의 위치와 클래스 탐지",
                "frameworks": ["ultralytics"],
                "supported": True
            },
            {
                "name": "instance_segmentation",
                "display_name": "인스턴스 분할",
                "description": "이미지 내 각 객체를 픽셀 단위로 분할",
                "frameworks": ["ultralytics"],
                "supported": True
            },
            {
                "name": "pose_estimation",
                "display_name": "자세 추정",
                "description": "이미지 내 사람의 자세 키포인트 탐지",
                "frameworks": ["ultralytics"],
                "supported": True
            }
        ],
        "dataset_formats": [
            {
                "name": "imagefolder",
                "display_name": "ImageFolder",
                "description": "PyTorch ImageFolder 형식 (class/image.jpg)",
                "task_types": ["image_classification"],
                "supported": True
            },
            {
                "name": "yolo",
                "display_name": "YOLO Format",
                "description": "Ultralytics YOLO 형식 (images/, labels/, data.yaml)",
                "task_types": ["object_detection", "instance_segmentation", "pose_estimation"],
                "supported": True
            },
            {
                "name": "coco",
                "display_name": "COCO Format",
                "description": "MS COCO JSON 형식",
                "task_types": ["object_detection", "instance_segmentation"],
                "supported": False
            }
        ],
        "parameters": [
            {
                "name": "framework",
                "display_name": "프레임워크",
                "description": "사용할 딥러닝 프레임워크",
                "type": "string",
                "required": True,
                "options": ["timm", "ultralytics"],
                "default": "timm"
            },
            {
                "name": "task_type",
                "display_name": "작업 유형",
                "description": "수행할 작업의 종류",
                "type": "string",
                "required": True,
                "options": ["image_classification", "object_detection", "instance_segmentation", "pose_estimation"],
                "default": "image_classification"
            },
            {
                "name": "model_name",
                "display_name": "모델 이름",
                "description": "사용할 모델",
                "type": "string",
                "required": True,
                "default": None
            },
            {
                "name": "num_classes",
                "display_name": "클래스 수",
                "description": "분류할 클래스의 개수 (classification 작업에만 필요)",
                "type": "integer",
                "required": False,
                "min": 2,
                "max": 1000,
                "default": None
            },
            {
                "name": "dataset_format",
                "display_name": "데이터셋 형식",
                "description": "데이터셋의 저장 형식",
                "type": "string",
                "required": False,
                "options": ["imagefolder", "yolo", "coco"],
                "default": "imagefolder"
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

        # Process project and experiment metadata if complete
        if parsed_result.get("status") == "complete":
            project_info = parsed_result.get("project", {})
            experiment_info = parsed_result.get("experiment", {})

            # Handle project creation/retrieval
            project_id = None
            if project_info:
                project_name = project_info.get("name")

                # If project name provided, create or get project
                if project_name:
                    from app.db.models import Project

                    # Check if project exists
                    existing_project = db.query(Project).filter(Project.name == project_name).first()

                    if existing_project:
                        project_id = existing_project.id
                        logger.debug(f"Using existing project: {project_name} (ID: {project_id})")
                    else:
                        # Create new project
                        new_project = Project(
                            name=project_name,
                            description=project_info.get("description"),
                            task_type=project_info.get("task_type") or parsed_result.get("config", {}).get("task_type"),
                        )
                        db.add(new_project)
                        db.commit()
                        db.refresh(new_project)
                        project_id = new_project.id
                        logger.debug(f"Created new project: {project_name} (ID: {project_id})")

            # Add project_id and experiment metadata to parsed_result
            if "metadata" not in parsed_result:
                parsed_result["metadata"] = {}

            parsed_result["metadata"]["project_id"] = project_id
            parsed_result["metadata"]["experiment_name"] = experiment_info.get("name") if experiment_info else None
            parsed_result["metadata"]["tags"] = experiment_info.get("tags", []) if experiment_info else []
            parsed_result["metadata"]["notes"] = experiment_info.get("notes") if experiment_info else None

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
