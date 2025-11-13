"""
FastAPI application with Monitoring enabled.

This is an example of how to integrate the monitoring components into the main FastAPI app.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("Starting FastAPI application with monitoring...")
    logger.info("=" * 80)

    # Import monitoring components
    from app.services.training_monitor import get_monitor, start_monitoring
    from app.services.websocket_manager import get_websocket_manager
    import asyncio

    # Get instances
    monitor = get_monitor()
    ws_manager = get_websocket_manager()

    # Inject WebSocket manager into monitor
    monitor.set_websocket_manager(ws_manager)

    # Start background monitoring task
    monitoring_task = asyncio.create_task(start_monitoring())
    logger.info("✓ Background training monitor started")

    yield  # Application is running

    # Shutdown
    logger.info("Shutting down...")

    # Stop monitoring
    await monitor.stop()
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass

    logger.info("✓ Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Vision AI Training Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
from app.api import websocket

app.include_router(
    websocket.router,
    prefix="/api/v1",
    tags=["websocket"]
)

# Include other routers (training, chat, etc.)
# from app.api import training, chat
# app.include_router(training.router, prefix="/api/v1", tags=["training"])
# app.include_router(chat.router, prefix="/api/v1", tags=["chat"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vision AI Training Platform API",
        "version": "1.0.0",
        "monitoring": "enabled"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    from app.services.websocket_manager import get_websocket_manager

    ws_manager = get_websocket_manager()

    return {
        "status": "healthy",
        "websocket_connections": ws_manager.get_connection_count(),
    }


# Example: Trigger manual notification (for testing)
@app.post("/api/v1/test/notify")
async def test_notification(job_id: int, message: str):
    """
    Test endpoint to trigger WebSocket notification.

    Args:
        job_id: Job ID to notify
        message: Test message
    """
    from app.services.websocket_manager import get_websocket_manager
    from datetime import datetime

    ws_manager = get_websocket_manager()

    notification = {
        "type": "test_notification",
        "job_id": job_id,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    await ws_manager.broadcast_to_job(job_id, notification)

    return {
        "status": "sent",
        "subscribers": ws_manager.get_job_subscriber_count(job_id)
    }
