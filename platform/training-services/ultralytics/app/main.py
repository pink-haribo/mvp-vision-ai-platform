"""
Ultralytics Training Service

FastAPI service that receives training requests and executes YOLO training.
Reports progress back to Backend via HTTP callbacks.
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(
    title="Ultralytics Training Service",
    description="YOLO model training service with S3 integration",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Training service accepts requests from Backend only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ultralytics-trainer",
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Ultralytics Training Service",
        "framework": "ultralytics",
        "models": ["yolo11n", "yolo11s", "yolo11m", "yolov8n", "yolov8s"],
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/models")
async def list_models():
    """List available YOLO models."""
    return {
        "models": [
            {
                "name": "yolo11n",
                "description": "YOLO11 Nano - Fastest, smallest",
                "tasks": ["detect", "segment", "pose", "classify"],
            },
            {
                "name": "yolo11s",
                "description": "YOLO11 Small - Balanced",
                "tasks": ["detect", "segment", "pose", "classify"],
            },
            {
                "name": "yolo11m",
                "description": "YOLO11 Medium - More accurate",
                "tasks": ["detect", "segment", "pose", "classify"],
            },
            {
                "name": "yolov8n",
                "description": "YOLOv8 Nano",
                "tasks": ["detect", "segment", "pose", "classify"],
            },
            {
                "name": "yolov8s",
                "description": "YOLOv8 Small",
                "tasks": ["detect", "segment", "pose", "classify"],
            },
        ]
    }


# Import router
from app.api import training

app.include_router(training.router, prefix="/training", tags=["training"])
