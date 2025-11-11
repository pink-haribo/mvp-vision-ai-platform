"""
Platform Backend - Main FastAPI Application

This is the core backend service for the Vision AI Training Platform.
It manages training jobs, communicates with training services via HTTP,
and provides APIs for the frontend.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup: Initialize database
    await init_db()
    yield
    # Shutdown: Clean up resources
    pass


app = FastAPI(
    title="Vision AI Training Platform - Backend",
    description="Backend API for managing training jobs and orchestrating training services",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "platform-backend",
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Vision AI Training Platform - Backend",
        "docs": "/docs",
        "health": "/health",
    }


# Import and include routers
from app.api import auth, projects, training, admin

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
