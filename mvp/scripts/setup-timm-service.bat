@echo off
REM Setup script for timm Training Service
REM This creates a separate venv for timm-service

echo ========================================
echo Setting up timm Training Service
echo ========================================

cd /d "%~dp0..\training"

REM Check if venv-timm exists
if exist "venv-timm\" (
    echo [INFO] venv-timm already exists. Skipping creation.
    echo [INFO] To recreate, delete venv-timm folder and run again.
) else (
    echo [1/3] Creating Python virtual environment...
    python -m venv venv-timm
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
)

echo.
echo [2/3] Installing dependencies from requirements-timm.txt...
call venv-timm\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-timm.txt
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
echo timm Training Service setup complete!
echo ========================================
echo.
echo To start the service:
echo   scripts\start-timm-service.bat
echo.
echo Service will run on: http://localhost:8001
echo.

pause
