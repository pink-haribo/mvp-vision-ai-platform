# Start Development Environment
# Run this script after reboot to start all infrastructure services

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Vision AI Platform - Dev Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker Desktop
Write-Host "[1/5] Checking Docker Desktop..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "  ✓ Docker Desktop is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker Desktop is not running" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop and run this script again" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check Kind cluster
Write-Host "[2/5] Checking Kind cluster..." -ForegroundColor Yellow
$clusterExists = kind get clusters 2>&1 | Select-String "platform-dev"
if ($clusterExists) {
    Write-Host "  ✓ Kind cluster 'platform-dev' exists" -ForegroundColor Green
} else {
    Write-Host "  ✗ Kind cluster 'platform-dev' not found" -ForegroundColor Red
    Write-Host "  Run: cd platform/infrastructure && kind create cluster --config kind-config.yaml" -ForegroundColor Yellow
    exit 1
}

# Step 3: Wait for cluster to be ready
Write-Host "[3/5] Waiting for cluster to be ready..." -ForegroundColor Yellow
$retries = 0
$maxRetries = 30
while ($retries -lt $maxRetries) {
    try {
        kubectl cluster-info --context kind-platform-dev 2>&1 | Out-Null
        Write-Host "  ✓ Cluster is ready" -ForegroundColor Green
        break
    } catch {
        $retries++
        Write-Host "  Waiting... ($retries/$maxRetries)" -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
}

if ($retries -eq $maxRetries) {
    Write-Host "  ✗ Cluster failed to become ready" -ForegroundColor Red
    exit 1
}

# Step 4: Check pod status
Write-Host "[4/5] Checking infrastructure services..." -ForegroundColor Yellow
Write-Host ""

$namespaces = @("platform", "mlflow", "observability", "temporal")
$allHealthy = $true

foreach ($ns in $namespaces) {
    Write-Host "  Namespace: $ns" -ForegroundColor Cyan
    $pods = kubectl get pods -n $ns --no-headers 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Host "    ⚠ No pods found" -ForegroundColor Yellow
        continue
    }

    $pods | ForEach-Object {
        $fields = $_ -split '\s+'
        $name = $fields[0]
        $ready = $fields[1]
        $status = $fields[2]

        if ($status -eq "Running" -or $status -eq "Completed") {
            Write-Host "    ✓ $name [$status]" -ForegroundColor Green
        } elseif ($status -eq "Pending") {
            Write-Host "    ⏳ $name [$status]" -ForegroundColor Yellow
        } else {
            Write-Host "    ✗ $name [$status]" -ForegroundColor Red
            $allHealthy = $false
        }
    }
    Write-Host ""
}

# Step 5: Service URLs
Write-Host "[5/5] Service Access URLs:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Infrastructure Services:" -ForegroundColor Cyan
Write-Host "    PostgreSQL:       localhost:30543 (admin/devpass)" -ForegroundColor White
Write-Host "    Redis:            localhost:30679" -ForegroundColor White
Write-Host "    MinIO Console:    http://localhost:30901 (minioadmin/minioadmin)" -ForegroundColor White
Write-Host "    MLflow:           http://localhost:30500" -ForegroundColor White
Write-Host "    Prometheus:       http://localhost:30090" -ForegroundColor White
Write-Host "    Grafana:          http://localhost:30030 (admin/prom-operator)" -ForegroundColor White
Write-Host "    Loki:             localhost:30100" -ForegroundColor White
Write-Host "    Temporal UI:      http://localhost:30233" -ForegroundColor White
Write-Host ""

if ($allHealthy) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  ✓ All infrastructure services ready!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "  ⚠ Some services need attention" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Start Backend:   cd platform/backend && poetry run uvicorn app.main:app --reload" -ForegroundColor White
Write-Host "  2. Start Frontend:  cd platform/frontend && pnpm dev" -ForegroundColor White
Write-Host ""
