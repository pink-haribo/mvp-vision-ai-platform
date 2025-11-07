# Run training locally (without rebuilding Docker images)
# Uses K8s services (MLflow, MinIO) but runs code on local machine

param(
    [string]$Script = "mvp/training/train.py",
    [string]$JobId = "local-$(Get-Date -Format 'yyyyMMdd-HHmmss')",
    [string]$ModelName = "yolo11n",
    [string]$Framework = "ultralytics",
    [int]$NumEpochs = 10
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Local Training (Fast Development)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Job ID:       $JobId" -ForegroundColor White
Write-Host "Script:       $Script" -ForegroundColor White
Write-Host "Model:        $ModelName" -ForegroundColor White
Write-Host "Framework:    $Framework" -ForegroundColor White
Write-Host "Epochs:       $NumEpochs" -ForegroundColor White
Write-Host ""

# Check if script exists
if (-not (Test-Path $Script)) {
    Write-Host "✗ Script not found: $Script" -ForegroundColor Red
    exit 1
}

# Check if K8s services are running
Write-Host "Checking K8s services..." -ForegroundColor Yellow

$mlflowRunning = kubectl get pods -n monitoring -l app=mlflow -o jsonpath='{.items[*].status.phase}' 2>$null
$minioRunning = kubectl get pods -n storage -l app=minio -o jsonpath='{.items[*].status.phase}' 2>$null

if ($mlflowRunning -ne "Running" -or $minioRunning -ne "Running") {
    Write-Host "✗ K8s services not running. Start with: .\dev-start.ps1 -SkipBuild" -ForegroundColor Red
    exit 1
}

Write-Host "✓ K8s services are running" -ForegroundColor Green
Write-Host ""

# Setup environment variables for local execution
Write-Host "Setting up environment variables..." -ForegroundColor Yellow

# MLflow
$env:MLFLOW_TRACKING_URI = "http://localhost:30500"
$env:MLFLOW_EXPERIMENT_NAME = "local-development"

# MinIO (S3)
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin"
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:30900"
$env:AWS_S3_ENDPOINT_URL = "http://localhost:30900"
$env:MLFLOW_S3_IGNORE_TLS = "true"

# Training parameters
$env:JOB_ID = $JobId
$env:MODEL_NAME = $ModelName
$env:FRAMEWORK = $Framework
$env:NUM_EPOCHS = $NumEpochs

Write-Host "✓ Environment configured" -ForegroundColor Green
Write-Host ""
Write-Host "Environment Variables:" -ForegroundColor Gray
Write-Host "  MLFLOW_TRACKING_URI:    $env:MLFLOW_TRACKING_URI" -ForegroundColor Gray
Write-Host "  MLFLOW_S3_ENDPOINT_URL: $env:MLFLOW_S3_ENDPOINT_URL" -ForegroundColor Gray
Write-Host "  JOB_ID:                 $env:JOB_ID" -ForegroundColor Gray
Write-Host ""

# Check Python environment
Write-Host "Checking Python environment..." -ForegroundColor Yellow

$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

Write-Host "✓ $pythonVersion" -ForegroundColor Green
Write-Host ""

# Run training
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Training..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Push-Location (Split-Path -Parent $Script)
$scriptName = Split-Path -Leaf $Script

try {
    # Run Python script with environment variables
    python $scriptName

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Training Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "View results:" -ForegroundColor Yellow
    Write-Host "  MLflow UI:      http://localhost:30500" -ForegroundColor White
    Write-Host "  MinIO Console:  http://localhost:30901" -ForegroundColor White
    Write-Host ""

} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Training Failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
