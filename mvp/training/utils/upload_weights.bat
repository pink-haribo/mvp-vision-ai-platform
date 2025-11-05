@echo off
REM Helper script to upload pretrained weights to R2
REM Usage: upload_weights.bat [ultralytics|timm|all] [--dry-run]

setlocal

set FRAMEWORK=%1
set DRY_RUN=%2

if "%FRAMEWORK%"=="" set FRAMEWORK=all

cd /d "%~dp0\.."

echo ========================================
echo R2 Pretrained Weights Upload
echo ========================================
echo.

REM Check if .env exists
if not exist ".env" (
    echo [ERROR] .env file not found!
    echo Please create .env with R2 credentials:
    echo   AWS_S3_ENDPOINT_URL=https://...
    echo   AWS_ACCESS_KEY_ID=...
    echo   AWS_SECRET_ACCESS_KEY=...
    exit /b 1
)

REM Load environment variables from .env
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    set "%%a=%%b"
)

if "%FRAMEWORK%"=="ultralytics" goto :upload_ultralytics
if "%FRAMEWORK%"=="timm" goto :upload_timm
if "%FRAMEWORK%"=="all" goto :upload_all

echo [ERROR] Invalid framework: %FRAMEWORK%
echo Usage: upload_weights.bat [ultralytics^|timm^|all] [--dry-run]
exit /b 1

:upload_ultralytics
echo.
echo [1/1] Uploading Ultralytics weights...
echo.
call venv-ultralytics\Scripts\activate.bat
python utils/upload_pretrained_weights.py --framework ultralytics %DRY_RUN%
deactivate
goto :end

:upload_timm
echo.
echo [1/1] Uploading timm weights...
echo.
call venv-timm\Scripts\activate.bat
python utils/upload_pretrained_weights.py --framework timm %DRY_RUN%
deactivate
goto :end

:upload_all
echo.
echo [1/2] Uploading Ultralytics weights...
echo.
call venv-ultralytics\Scripts\activate.bat
python utils/upload_pretrained_weights.py --framework ultralytics %DRY_RUN%
deactivate

echo.
echo [2/2] Uploading timm weights...
echo.
call venv-timm\Scripts\activate.bat
python utils/upload_pretrained_weights.py --framework timm %DRY_RUN%
deactivate
goto :end

:end
echo.
echo ========================================
echo Done!
echo ========================================
endlocal
