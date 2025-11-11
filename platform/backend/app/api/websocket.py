"""
WebSocket endpoints for real-time training updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional
import logging

from app.services.websocket_manager import get_websocket_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/training")
async def training_updates_websocket(
    websocket: WebSocket,
    job_id: Optional[int] = Query(None),
    session_id: Optional[int] = Query(None),
):
    """
    WebSocket endpoint for real-time training updates.

    Query Parameters:
        job_id: Subscribe to updates for specific training job
        session_id: Subscribe to updates for specific session

    Message Types (Server -> Client):
        - training_status_change: Job status changed
        - training_metrics: New metrics available
        - training_log: New log message
        - training_complete: Training finished
        - training_error: Training failed

    Example usage:
        ws://localhost:8000/api/v1/ws/training?job_id=123
        ws://localhost:8000/api/v1/ws/training?session_id=456
    """
    ws_manager = get_websocket_manager()

    # Connect client
    await ws_manager.connect(websocket, job_id=job_id, session_id=session_id)

    try:
        # Keep connection alive
        while True:
            # Receive messages from client (ping/pong, subscriptions, etc.)
            data = await websocket.receive_text()

            # Handle client messages
            import json
            try:
                message = json.loads(data)
                await handle_client_message(websocket, message, ws_manager)
            except json.JSONDecodeError:
                await ws_manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON"},
                    websocket
                )

    except WebSocketDisconnect:
        logger.info("[WebSocket] Client disconnected")
        ws_manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"[WebSocket] Error: {e}")
        ws_manager.disconnect(websocket)


async def handle_client_message(
    websocket: WebSocket,
    message: dict,
    ws_manager
):
    """
    Handle messages from client.

    Supported message types:
        - ping: Keep-alive ping
        - subscribe: Subscribe to additional job/session
        - unsubscribe: Unsubscribe from job/session
    """
    msg_type = message.get("type")

    if msg_type == "ping":
        # Respond to ping
        await ws_manager.send_personal_message(
            {"type": "pong"},
            websocket
        )

    elif msg_type == "subscribe":
        # Subscribe to additional job/session
        job_id = message.get("job_id")
        session_id = message.get("session_id")

        if job_id:
            if job_id not in ws_manager.job_connections:
                ws_manager.job_connections[job_id] = set()
            ws_manager.job_connections[job_id].add(websocket)
            logger.info(f"[WebSocket] Client subscribed to job {job_id}")

        if session_id:
            if session_id not in ws_manager.session_connections:
                ws_manager.session_connections[session_id] = set()
            ws_manager.session_connections[session_id].add(websocket)
            logger.info(f"[WebSocket] Client subscribed to session {session_id}")

        await ws_manager.send_personal_message(
            {"type": "subscribed", "job_id": job_id, "session_id": session_id},
            websocket
        )

    elif msg_type == "unsubscribe":
        # Unsubscribe from job/session
        job_id = message.get("job_id")
        session_id = message.get("session_id")

        if job_id and job_id in ws_manager.job_connections:
            ws_manager.job_connections[job_id].discard(websocket)

        if session_id and session_id in ws_manager.session_connections:
            ws_manager.session_connections[session_id].discard(websocket)

        await ws_manager.send_personal_message(
            {"type": "unsubscribed", "job_id": job_id, "session_id": session_id},
            websocket
        )


@router.get("/ws/stats")
async def websocket_stats():
    """
    Get WebSocket connection statistics.

    Returns:
        - total_connections: Total active WebSocket connections
        - job_subscriptions: Number of job-specific subscriptions
        - session_subscriptions: Number of session-specific subscriptions
    """
    ws_manager = get_websocket_manager()

    return {
        "total_connections": ws_manager.get_connection_count(),
        "job_subscriptions": len(ws_manager.job_connections),
        "session_subscriptions": len(ws_manager.session_connections),
    }
