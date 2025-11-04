@echo off
REM Setup script for ultralytics Training Service
REM This creates a separate venv for ultralytics-service

echo ========================================
echo Setting up ultralytics Training Service
echo ========================================

cd /d "%~dp0..\training"

REM Check if venv-ultralytics exists
if exist "venv-ultralytics\" (
    echo [INFO] venv-ultralytics already exists. Skipping creation.
    echo [INFO] To recreate, delete venv-ultralytics folder and run again.
) else (
    echo [1/3] Creating Python virtual environment...
    python -m venv venv-ultralytics
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
)

echo.
echo [2/3] Installing dependencies from requirements-ultralytics.txt...
call venv-ultralytics\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-ultralytics.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)

echo.
echo [3/3] Installing FastAPI and uvicorn for API server...
pip install fastapi uvicorn[standard] pydantic
if errorlevel 1 (
    echo [ERROR] Failed to install FastAPI
    exit /b 1
)

echo.
echo ========================================
echo ultralytics Training Service setup complete!
echo ========================================
echo.
echo To start the service:
echo   scripts\start-ultralytics-service.bat
echo.
echo Service will run on: http://localhost:8002
echo.

pause
