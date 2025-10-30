@echo off
REM ============================================
REM Vision Platform Docker Build Script (Windows)
REM ============================================

echo ======================================
echo Vision Platform Docker Build
echo ======================================

cd /d %~dp0\..

echo [1/4] Building base image...
docker build -f docker/Dockerfile.base -t vision-platform-base:latest .
if %errorlevel% neq 0 (
    echo [ERROR] Base image build failed
    exit /b %errorlevel%
)

echo.
echo [2/4] Building timm image...
docker build -f docker/Dockerfile.timm -t vision-platform-timm:latest .
if %errorlevel% neq 0 (
    echo [ERROR] timm image build failed
    exit /b %errorlevel%
)

echo.
echo [3/4] Building ultralytics image...
docker build -f docker/Dockerfile.ultralytics -t vision-platform-ultralytics:latest .
if %errorlevel% neq 0 (
    echo [ERROR] ultralytics image build failed
    exit /b %errorlevel%
)

echo.
echo [4/4] Building huggingface image...
docker build -f docker/Dockerfile.huggingface -t vision-platform-huggingface:latest .
if %errorlevel% neq 0 (
    echo [ERROR] huggingface image build failed
    exit /b %errorlevel%
)

echo.
echo ======================================
echo All images built successfully!
echo ======================================
echo.

docker images | findstr vision-platform

echo.
echo Ready to use!
echo Test with: docker run --rm vision-platform-ultralytics:latest python -c "from ultralytics import YOLOWorld; print('OK')"
