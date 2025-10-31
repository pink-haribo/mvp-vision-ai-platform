"""FastAPI application entry point."""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file before anything else
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
print(f"[STARTUP] Loaded .env from: {env_path}")

# Add parent directory to sys.path for training module access
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.core.config import settings
from app.api import auth, chat, training, projects, debug, datasets, admin, validation, test_inference, models, image_tools

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
)

# Log CORS origins for debugging
print(f"[CORS] Allowed origins: {settings.BACKEND_CORS_ORIGINS}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix=f"{settings.API_V1_PREFIX}/auth", tags=["auth"])
app.include_router(chat.router, prefix=f"{settings.API_V1_PREFIX}/chat", tags=["chat"])
app.include_router(training.router, prefix=f"{settings.API_V1_PREFIX}/training", tags=["training"])
app.include_router(validation.router, prefix=f"{settings.API_V1_PREFIX}", tags=["validation"])
app.include_router(test_inference.router, prefix=f"{settings.API_V1_PREFIX}", tags=["test_inference"])
app.include_router(image_tools.router, prefix=f"{settings.API_V1_PREFIX}", tags=["image_tools"])  # Image tools
app.include_router(projects.router, prefix=f"{settings.API_V1_PREFIX}/projects", tags=["projects"])
app.include_router(datasets.router, prefix=f"{settings.API_V1_PREFIX}/datasets", tags=["datasets"])
app.include_router(admin.router, prefix=f"{settings.API_V1_PREFIX}/admin", tags=["admin"])
app.include_router(debug.router, prefix=f"{settings.API_V1_PREFIX}/debug", tags=["debug"])
app.include_router(models.router, prefix=f"{settings.API_V1_PREFIX}", tags=["models"])  # Model registry


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "vision-ai-mvp-backend"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Vision AI Training Platform - MVP",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ========== Background Tasks ==========

@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks on application startup."""
    import asyncio
    from datetime import datetime, timedelta
    import shutil

    async def cleanup_old_inference_sessions():
        """
        Periodically clean up old inference session directories.

        Runs every hour and deletes sessions where all files are older than 2 hours.
        Cleans up both inference_temp and image_tools_temp directories.
        """
        while True:
            try:
                await asyncio.sleep(3600)  # Run every 1 hour

                # Cleanup both inference_temp and image_tools_temp
                temp_dirs = [
                    Path(settings.UPLOAD_DIR) / "inference_temp",
                    Path(settings.UPLOAD_DIR) / "image_tools_temp"
                ]

                cutoff_time = datetime.now() - timedelta(hours=2)

                for temp_dir in temp_dirs:
                    if not temp_dir.exists():
                        continue

                    for session_dir in temp_dir.iterdir():
                        if not session_dir.is_dir():
                            continue

                        try:
                            # Check if all files in session are older than cutoff
                            files = list(session_dir.iterdir())

                            if not files:
                                # Empty directory - delete it
                                session_dir.rmdir()
                                print(f"[CLEANUP] Removed empty session: {session_dir.name}")
                                continue

                            all_old = all(
                                datetime.fromtimestamp(f.stat().st_mtime) < cutoff_time
                                for f in files
                            )

                            if all_old:
                                shutil.rmtree(session_dir)
                                print(f"[CLEANUP] Removed old session: {session_dir.name} ({len(files)} files)")

                        except Exception as e:
                            print(f"[CLEANUP] Error processing session {session_dir.name}: {e}")
                            continue

            except Exception as e:
                print(f"[CLEANUP] Background cleanup task error: {e}")
                # Continue running even if there's an error

    # Start the cleanup task
    asyncio.create_task(cleanup_old_inference_sessions())
    print("[STARTUP] Background cleanup task started (runs every 1 hour)")
