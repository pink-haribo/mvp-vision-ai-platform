@echo off
REM Start timm Training Service
REM Port: 8001

cd /d "%~dp0..\training"

REM Check if venv exists
if not exist "venv-timm\" (
    echo [ERROR] venv-timm not found!
    echo Please run: scripts\setup-timm-service.bat
    pause
    exit /b 1
)

REM Activate venv
call venv-timm\Scripts\activate.bat

echo ========================================
echo Starting timm Training Service
echo ========================================
echo Framework: timm
echo Port: 8001
echo Models: ResNet, EfficientNet, ViT, etc.
echo ========================================
echo.
echo Service will be available at:
echo   http://localhost:8001
echo   http://localhost:8001/docs (API documentation)
echo.
echo Press Ctrl+C to stop the service
echo.

REM Set environment variables
set FRAMEWORK=timm
set PORT=8001
set SERVICE_NAME=timm-service

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
echo [INFO] timm Training Service stopped
pause
