"""
Debug API endpoints - Development only
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession
from app.db.database import get_db
from app.db.models import Session as SessionModel, Message
from typing import Dict, Any
import json

router = APIRouter()

@router.get("/session/{session_id}")
async def debug_session(session_id: int, db: DBSession = Depends(get_db)):
    """세션의 temp_data를 직접 조회"""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        return {"error": "Session not found"}

    return {
        "session_id": session_id,
        "state": session.state,
        "temp_data": session.temp_data,
        "config": session.temp_data.get("config") if session.temp_data else None,
        "config_keys": list(session.temp_data.get("config", {}).keys()) if session.temp_data else []
    }

@router.get("/last-session")
async def debug_last_session(db: DBSession = Depends(get_db)):
    """가장 최근 세션 조회"""
    session = db.query(SessionModel).order_by(SessionModel.id.desc()).first()
    if not session:
        return {"error": "No sessions found"}

    return {
        "session_id": session.id,
        "state": session.state,
        "temp_data": session.temp_data,
        "config": session.temp_data.get("config") if session.temp_data else None,
        "config_keys": list(session.temp_data.get("config", {}).keys()) if session.temp_data else []
    }

@router.get("/session/{session_id}/messages")
async def debug_session_messages(session_id: int, db: DBSession = Depends(get_db)):
    """세션의 메시지 내역 조회"""
    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.id).all()

    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None
            }
            for msg in messages
        ]
    }

@router.get("/compare-sessions/{session_id}")
async def compare_session_evolution(session_id: int, db: DBSession = Depends(get_db)):
    """세션의 config 변화 추적 - 메시지 수만큼 config 스냅샷"""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        return {"error": "Session not found"}

    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.id).all()

    # 현재 config
    current_config = session.temp_data.get("config") if session.temp_data else {}

    return {
        "session_id": session_id,
        "current_state": session.state,
        "current_config": current_config,
        "current_config_keys": list(current_config.keys()),
        "message_count": len(messages),
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            }
            for msg in messages
        ],
        "note": "config는 세션 레벨에서만 저장됨. 메시지별 스냅샷은 별도 구현 필요"
    }
