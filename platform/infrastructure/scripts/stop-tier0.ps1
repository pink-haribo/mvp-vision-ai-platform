# Stop Tier-0 Development Environment
# Cleanly shuts down all Tier-0 services
# Usage: cd platform/infrastructure && .\scripts\stop-tier0.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stopping Tier-0 Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get project root
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$infraDir = Join-Path $projectRoot "infrastructure"

# Step 1: Stop PowerShell jobs (Backend/Frontend)
Write-Host "[1/3] Stopping Backend and Frontend servers..." -ForegroundColor Yellow

$jobs = Get-Job | Where-Object { $_.State -eq "Running" }
if ($jobs) {
    foreach ($job in $jobs) {
        Write-Host "  Stopping Job $($job.Id) ($($job.Name))..." -ForegroundColor Gray
        Stop-Job $job.Id
        Remove-Job $job.Id
    }
    Write-Host "  ✓ All jobs stopped" -ForegroundColor Green
} else {
    Write-Host "  No running jobs found" -ForegroundColor Gray
}

# Step 2: Stop Docker Compose services
Write-Host "[2/3] Stopping infrastructure services..." -ForegroundColor Yellow
Set-Location $infraDir

try {
    docker-compose -f docker-compose.tier0.yaml down
    Write-Host "  ✓ All containers stopped" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to stop Docker Compose" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
}

# Step 3: Show data preservation info
Write-Host "[3/3] Data preservation..." -ForegroundColor Yellow
Write-Host "  ✓ All data preserved in C:\platform-data\" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✓ Tier-0 Environment Stopped" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Data Status:" -ForegroundColor Cyan
Write-Host "  • PostgreSQL data:  C:\platform-data\postgres" -ForegroundColor White
Write-Host "  • Redis data:       C:\platform-data\redis" -ForegroundColor White
Write-Host "  • MinIO data:       C:\platform-data\minio" -ForegroundColor White
Write-Host "  • Prometheus data:  C:\platform-data\prometheus" -ForegroundColor White
Write-Host "  • Grafana data:     C:\platform-data\grafana" -ForegroundColor White
Write-Host "  • Loki data:        C:\platform-data\loki" -ForegroundColor White
Write-Host ""

Write-Host "To restart:" -ForegroundColor Yellow
Write-Host "  .\scripts\start-tier0.ps1" -ForegroundColor White
Write-Host ""

Write-Host "To remove all data (WARNING: Destructive!):" -ForegroundColor Yellow
Write-Host "  docker-compose -f docker-compose.tier0.yaml down -v" -ForegroundColor Red
Write-Host "  Remove-Item -Recurse -Force C:\platform-data\*" -ForegroundColor Red
Write-Host ""
