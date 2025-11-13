# Deploy All Platform Services using Helm
# Tier 1 Development Environment
#
# Prerequisites:
#   - Helm 3.x installed (winget install Helm.Helm)
#   - Kind cluster running (platform-dev)
#   - kubectl configured
#
# Usage:
#   From NEW PowerShell terminal:
#   .\scripts\deploy-helm-all.ps1

# Enable strict error handling
$ErrorActionPreference = "Stop"

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Deploying All Services with Helm Charts" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# Check if Helm is available
Write-Host "[1/10] Checking Helm installation..." -ForegroundColor Yellow
try {
    $helmVersion = helm version --short
    Write-Host "  ✓ Helm is available: $helmVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Helm is not installed or not in PATH" -ForegroundColor Red
    Write-Host "  Run: winget install Helm.Helm" -ForegroundColor Yellow
    Write-Host "  Then restart PowerShell" -ForegroundColor Yellow
    exit 1
}

# Check cluster connection
Write-Host "[2/10] Checking cluster connection..." -ForegroundColor Yellow
try {
    kubectl cluster-info --context kind-platform-dev | Out-Null
    Write-Host "  ✓ Connected to platform-dev cluster" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Cannot connect to platform-dev cluster" -ForegroundColor Red
    exit 1
}

# Add Helm repositories
Write-Host "[3/10] Adding Helm repositories..." -ForegroundColor Yellow
helm repo add bitnami https://charts.bitnami.com/bitnami 2>&1 | Out-Null
Write-Host "  ✓ Added bitnami" -ForegroundColor Green

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>&1 | Out-Null
Write-Host "  ✓ Added prometheus-community" -ForegroundColor Green

helm repo add temporalio https://go.temporal.io/helm-charts 2>&1 | Out-Null
Write-Host "  ✓ Added temporalio" -ForegroundColor Green

helm repo add minio https://charts.min.io/ 2>&1 | Out-Null
Write-Host "  ✓ Added minio" -ForegroundColor Green

Write-Host "[4/10] Updating Helm repositories..." -ForegroundColor Yellow
helm repo update | Out-Null
Write-Host "  ✓ Repositories updated" -ForegroundColor Green

Write-Host ""

# Navigate to helm directory
$helmDir = Join-Path $PSScriptRoot "..\helm"

# Deploy PostgreSQL
Write-Host "[5/10] Deploying PostgreSQL..." -ForegroundColor Yellow
$postgresValues = Join-Path $helmDir "postgres-values.yaml"
helm upgrade --install postgresql bitnami/postgresql `
    --namespace platform `
    --create-namespace `
    --values $postgresValues `
    --wait `
    --timeout 5m
Write-Host "  ✓ PostgreSQL deployed" -ForegroundColor Green

# Deploy Redis
Write-Host "[6/10] Deploying Redis..." -ForegroundColor Yellow
$redisValues = Join-Path $helmDir "redis-values.yaml"
helm upgrade --install redis bitnami/redis `
    --namespace platform `
    --values $redisValues `
    --wait `
    --timeout 5m
Write-Host "  ✓ Redis deployed" -ForegroundColor Green

# Deploy MinIO
Write-Host "[7/10] Deploying MinIO..." -ForegroundColor Yellow
$minioValues = Join-Path $helmDir "minio-values.yaml"
helm upgrade --install minio minio/minio `
    --namespace platform `
    --values $minioValues `
    --wait `
    --timeout 5m
Write-Host "  ✓ MinIO deployed" -ForegroundColor Green

# Deploy kube-prometheus-stack
Write-Host "[8/10] Deploying kube-prometheus-stack (Prometheus + Grafana + Loki)..." -ForegroundColor Yellow
$prometheusValues = Join-Path $helmDir "kube-prometheus-stack-values.yaml"
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack `
    --namespace observability `
    --create-namespace `
    --values $prometheusValues `
    --wait `
    --timeout 10m
Write-Host "  ✓ Observability stack deployed" -ForegroundColor Green

# Deploy Loki
Write-Host "[9/11] Deploying Loki..." -ForegroundColor Yellow
helm repo add grafana https://grafana.github.io/helm-charts 2>&1 | Out-Null
helm repo update | Out-Null
helm upgrade --install loki grafana/loki `
    --namespace observability `
    --set loki.auth_enabled=false `
    --set loki.commonConfig.replication_factor=1 `
    --set singleBinary.replicas=1 `
    --set read.replicas=0 `
    --set backend.replicas=0 `
    --set write.replicas=0 `
    --wait `
    --timeout 5m
Write-Host "  ✓ Loki deployed" -ForegroundColor Green

# Deploy Temporal
Write-Host "[10/11] Deploying Temporal..." -ForegroundColor Yellow
$temporalValues = Join-Path $helmDir "temporal-values.yaml"
helm upgrade --install temporal temporalio/temporal `
    --namespace temporal `
    --create-namespace `
    --values $temporalValues `
    --wait `
    --timeout 10m
Write-Host "  ✓ Temporal deployed" -ForegroundColor Green

# Deploy NodePort services for external access
Write-Host "[11/11] Deploying NodePort services..." -ForegroundColor Yellow
$k8sDir = Join-Path $PSScriptRoot "..\k8s"
kubectl apply -f (Join-Path $k8sDir "platform\nodeports.yaml")
kubectl apply -f (Join-Path $k8sDir "observability\nodeports.yaml")
kubectl apply -f (Join-Path $k8sDir "temporal\nodeports.yaml")
Write-Host "  ✓ NodePort services deployed" -ForegroundColor Green

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Deployment Summary" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# List all Helm releases
Write-Host "Helm Releases:" -ForegroundColor White
helm list --all-namespaces

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Access URLs" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

Write-Host "PostgreSQL:  " -NoNewline -ForegroundColor White
Write-Host "localhost:30543" -ForegroundColor Gray
Write-Host "Redis:       " -NoNewline -ForegroundColor White
Write-Host "localhost:30679" -ForegroundColor Gray
Write-Host "MinIO API:   " -NoNewline -ForegroundColor White
Write-Host "http://localhost:30900" -ForegroundColor Gray
Write-Host "MinIO UI:    " -NoNewline -ForegroundColor White
Write-Host "http://localhost:30901" -ForegroundColor Gray
Write-Host "Prometheus:  " -NoNewline -ForegroundColor White
Write-Host "http://localhost:30090" -ForegroundColor Gray
Write-Host "Grafana:     " -NoNewline -ForegroundColor White
Write-Host "http://localhost:30030 (admin/prom-operator)" -ForegroundColor Gray
Write-Host "Temporal UI: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:30233" -ForegroundColor Gray

Write-Host ""
