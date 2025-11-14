# Start Tier-0 Development Environment (Docker Compose)
# Run this script after reboot to start all infrastructure services
# Usage: cd platform/infrastructure && .\scripts\start-tier0.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Vision AI Platform - Tier-0 (Docker Compose)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get project root (2 levels up from scripts directory)
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$infraDir = Join-Path $projectRoot "infrastructure"
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"

# Step 1: Check Docker Desktop
Write-Host "[1/6] Checking Docker Desktop..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "  [OK] Docker Desktop is running" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Docker Desktop is not running" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop and run this script again" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check required directories
Write-Host "[2/6] Checking data directories..." -ForegroundColor Yellow
$dataDirs = @(
    "C:\platform-data\postgres",
    "C:\platform-data\redis",
    "C:\platform-data\minio",
    "C:\platform-data\prometheus",
    "C:\platform-data\grafana",
    "C:\platform-data\loki"
)

foreach ($dir in $dataDirs) {
    if (-not (Test-Path $dir)) {
        Write-Host "  Creating: $dir" -ForegroundColor Gray
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "  [OK] All data directories ready" -ForegroundColor Green

# Step 3: Start Docker Compose services
Write-Host "[3/6] Starting infrastructure services..." -ForegroundColor Yellow
Set-Location $infraDir

try {
    docker-compose -f docker-compose.tier0.yaml up -d
    Write-Host "  [OK] Docker Compose started" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Failed to start Docker Compose" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    exit 1
}

# Step 4: Wait for services to be healthy
Write-Host "[4/6] Waiting for services to be ready..." -ForegroundColor Yellow
Write-Host "  This may take 30-60 seconds..." -ForegroundColor Gray
Write-Host ""

$services = @(
    @{Name="PostgreSQL"; Container="platform-postgres-tier0"; Port=5432; CheckCmd="docker exec platform-postgres-tier0 pg_isready -U admin -d platform"},
    @{Name="Redis"; Container="platform-redis-tier0"; Port=6379; CheckCmd="docker exec platform-redis-tier0 redis-cli ping"},
    @{Name="MinIO"; Container="platform-minio-tier0"; Port=9000; CheckCmd="docker exec platform-minio-tier0 curl -f http://localhost:9000/minio/health/live"},
    @{Name="MLflow"; Container="platform-mlflow-tier0"; Port=5000; CheckCmd="docker exec platform-mlflow-tier0 curl -f http://localhost:5000/health"},
    @{Name="Temporal"; Container="platform-temporal-tier0"; Port=7233; CheckCmd="docker exec platform-temporal-tier0 tctl --address localhost:7233 cluster health"},
    @{Name="Prometheus"; Container="platform-prometheus-tier0"; Port=9090; CheckCmd="curl -s http://localhost:9090/-/healthy"},
    @{Name="Grafana"; Container="platform-grafana-tier0"; Port=3200; CheckCmd="curl -s http://localhost:3200/api/health"},
    @{Name="Loki"; Container="platform-loki-tier0"; Port=3100; CheckCmd="curl -s http://localhost:3100/ready"}
)

$maxRetries = 60
$allHealthy = $true

foreach ($svc in $services) {
    Write-Host "  Checking $($svc.Name)..." -NoNewline -ForegroundColor Cyan

    $retries = 0
    $healthy = $false

    while ($retries -lt $maxRetries) {
        try {
            Invoke-Expression $svc.CheckCmd 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $healthy = $true
                break
            }
        } catch {
            # Continue retrying
        }

        $retries++
        Start-Sleep -Seconds 1
        Write-Host "." -NoNewline -ForegroundColor Gray
    }

    if ($healthy) {
        Write-Host " [OK]" -ForegroundColor Green
    } else {
        Write-Host " [ERROR] (timeout)" -ForegroundColor Red
        $allHealthy = $false
    }
}

Write-Host ""

if (-not $allHealthy) {
    Write-Host "  [!] Some services failed health check" -ForegroundColor Yellow
    Write-Host "  You can proceed, but some features may not work" -ForegroundColor Yellow
    Write-Host ""
}

# Step 5: Check and copy .env file
Write-Host "[5/6] Checking environment configuration..." -ForegroundColor Yellow
$envFile = Join-Path $backendDir ".env"
$envTier0 = Join-Path $backendDir ".env.tier0"

if (-not (Test-Path $envFile)) {
    Write-Host "  .env not found, copying from .env.tier0..." -ForegroundColor Gray
    Copy-Item $envTier0 $envFile
    Write-Host "  [OK] .env file created" -ForegroundColor Green
} else {
    # Check if it's Tier-0 config
    $content = Get-Content $envFile -Raw
    if ($content -match "tier0-development") {
        Write-Host "  [OK] .env is configured for Tier-0" -ForegroundColor Green
    } else {
        Write-Host "  [!] .env exists but may not be Tier-0 config" -ForegroundColor Yellow
        Write-Host "    To use Tier-0: cp .env.tier0 .env" -ForegroundColor Gray
    }
}

# Step 6: Start Backend and Frontend
Write-Host "[6/6] Starting Backend and Frontend servers..." -ForegroundColor Yellow

# Start Backend in background
Write-Host "  Starting Backend (port 8000)..." -ForegroundColor Cyan
Set-Location $backendDir
$backendJob = Start-Job -ScriptBlock {
    param($backendDir)
    Set-Location $backendDir
    & .\venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
} -ArgumentList $backendDir
Write-Host "  [OK] Backend started (Job ID: $($backendJob.Id))" -ForegroundColor Green

# Wait a bit for backend to start
Start-Sleep -Seconds 3

# Start Frontend in background
Write-Host "  Starting Frontend (port 3000)..." -ForegroundColor Cyan
Set-Location $frontendDir
$frontendJob = Start-Job -ScriptBlock {
    param($frontendDir)
    Set-Location $frontendDir
    pnpm dev
} -ArgumentList $frontendDir
Write-Host "  [OK] Frontend started (Job ID: $($frontendJob.Id))" -ForegroundColor Green

# Return to infrastructure directory
Set-Location $infraDir

# Display service information
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  [OK] Tier-0 Environment Ready!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Infrastructure Services:" -ForegroundColor Cyan
Write-Host "  PostgreSQL:       localhost:5432 (admin/devpass)" -ForegroundColor White
Write-Host "  Redis:            localhost:6379" -ForegroundColor White
Write-Host "  MinIO API:        http://localhost:9000 (minioadmin/minioadmin)" -ForegroundColor White
Write-Host "  MinIO Console:    http://localhost:9001 (minioadmin/minioadmin)" -ForegroundColor White
Write-Host "  MLflow:           http://localhost:5000" -ForegroundColor White
Write-Host "  Temporal gRPC:    localhost:7233" -ForegroundColor White
Write-Host "  Temporal UI:      http://localhost:8233" -ForegroundColor White
Write-Host ""

Write-Host "Observability Services:" -ForegroundColor Cyan
Write-Host "  Prometheus:       http://localhost:9090" -ForegroundColor White
Write-Host "  Grafana:          http://localhost:3200 (admin/admin)" -ForegroundColor White
Write-Host "  Loki:             localhost:3100" -ForegroundColor White
Write-Host ""

Write-Host "Application Services:" -ForegroundColor Cyan
Write-Host "  Backend API:      http://localhost:8000" -ForegroundColor White
Write-Host "  Backend Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Frontend:         http://localhost:3000" -ForegroundColor White
Write-Host ""

Write-Host "Management Commands:" -ForegroundColor Cyan
Write-Host "  View Backend logs:   Receive-Job $($backendJob.Id) -Keep" -ForegroundColor White
Write-Host "  View Frontend logs:  Receive-Job $($frontendJob.Id) -Keep" -ForegroundColor White
Write-Host "  Stop Backend:        Stop-Job $($backendJob.Id); Remove-Job $($backendJob.Id)" -ForegroundColor White
Write-Host "  Stop Frontend:       Stop-Job $($frontendJob.Id); Remove-Job $($frontendJob.Id)" -ForegroundColor White
Write-Host "  Stop All:            docker-compose -f docker-compose.tier0.yaml down" -ForegroundColor White
Write-Host ""

Write-Host "Resource Usage (Tier-0):" -ForegroundColor Cyan
Write-Host "  Expected RAM: ~1.5-2GB" -ForegroundColor White
Write-Host "  Expected CPU: 10-20%" -ForegroundColor White
Write-Host ""

Write-Host "Tips:" -ForegroundColor Yellow
Write-Host "  - All data is stored in C:\platform-data (shared with Tier-1)" -ForegroundColor Gray
Write-Host "  - To switch to Tier-1: docker-compose down then use Kind" -ForegroundColor Gray
Write-Host "  - Your database and MinIO data will persist" -ForegroundColor Gray
Write-Host ""
