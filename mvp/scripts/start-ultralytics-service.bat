@echo off
REM Start ultralytics Training Service
REM Port: 8002 (to avoid conflict with backend:8000 and timm:8001)

cd /d "%~dp0..\training"

REM Check if venv exists
if not exist "venv-ultralytics\" (
    echo [ERROR] venv-ultralytics not found!
    echo Please run: scripts\setup-ultralytics-service.bat
    pause
    exit /b 1
)

REM Activate venv
call venv-ultralytics\Scripts\activate.bat

echo ========================================
echo Starting ultralytics Training Service
echo ========================================
echo Framework: ultralytics
echo Port: 8002
echo Models: YOLO family (detection, segmentation, pose)
echo ========================================
echo.
echo Service will be available at:
echo   http://localhost:8002
echo   http://localhost:8002/docs (API documentation)
echo.
echo Press Ctrl+C to stop the service
echo.

REM Set environment variables
set FRAMEWORK=ultralytics
set PORT=8002
set SERVICE_NAME=ultralytics-service

REM Check if .env exists
if exist ".env" (
    echo [INFO] Loading environment variables from .env
) else (
    echo [WARNING] .env file not found, using defaults
)

REM Start API server
python api_server.py

REM If server stops
echo.
echo [INFO] ultralytics Training Service stopped
pause
