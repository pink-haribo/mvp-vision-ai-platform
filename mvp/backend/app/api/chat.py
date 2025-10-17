"""Chat API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models
from app.schemas import chat
from app.utils.llm import intent_parser

router = APIRouter()


@router.post("/sessions", response_model=chat.SessionResponse)
async def create_session(db: Session = Depends(get_db)):
    """Create a new chat session."""
    session = models.Session()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions/{session_id}", response_model=chat.SessionResponse)
async def get_session(session_id: int, db: Session = Depends(get_db)):
    """Get a chat session by ID."""
    session = db.query(models.Session).filter(models.Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/messages", response_model=list[chat.MessageResponse])
async def get_messages(session_id: int, db: Session = Depends(get_db)):
    """Get all messages in a session."""
    session = db.query(models.Session).filter(models.Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = (
        db.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .order_by(models.Message.created_at)
        .all()
    )
    return messages


@router.post("/message", response_model=chat.ChatResponse)
async def send_message(request: chat.ChatRequest, db: Session = Depends(get_db)):
    """
    Send a message and get AI response.

    This endpoint:
    1. Creates or retrieves a session
    2. Saves the user message
    3. Parses intent using Claude
    4. Generates and saves assistant response
    5. Returns both messages and parsed intent
    """
    # Create or get session
    if request.session_id:
        session = db.query(models.Session).filter(models.Session.id == request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = models.Session()
        db.add(session)
        db.commit()
        db.refresh(session)

    # Save user message
    user_message = models.Message(
        session_id=session.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # Get conversation context (last 5 messages)
    previous_messages = (
        db.query(models.Message)
        .filter(models.Message.session_id == session.id)
        .order_by(models.Message.created_at.desc())
        .limit(5)
        .all()
    )
    context = "\n".join([f"{msg.role}: {msg.content}" for msg in reversed(previous_messages[:-1])])

    # Parse intent
    parsed_result = await intent_parser.parse_message(request.message, context if context else None)

    # Generate response
    assistant_content = await intent_parser.generate_response(request.message, parsed_result)

    # Save assistant message
    assistant_message = models.Message(
        session_id=session.id,
        role="assistant",
        content=assistant_content,
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    # Return response
    return chat.ChatResponse(
        session_id=session.id,
        user_message=user_message,
        assistant_message=assistant_message,
        parsed_intent=parsed_result if parsed_result.get("status") == "complete" else None,
    )
