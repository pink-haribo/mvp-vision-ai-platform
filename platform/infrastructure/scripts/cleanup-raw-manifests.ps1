# Cleanup Raw Manifests Deployments
# Remove all services deployed via raw YAML manifests
# before deploying with Helm
#
# Usage:
#   .\scripts\cleanup-raw-manifests.ps1

# Enable strict error handling
$ErrorActionPreference = "Stop"

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Cleaning up Raw Manifests Deployments" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

Write-Host "Removing Platform services..." -ForegroundColor Yellow
kubectl delete -f ..\k8s\platform\postgres.yaml --ignore-not-found=true 2>&1 | Out-Null
kubectl delete -f ..\k8s\platform\redis.yaml --ignore-not-found=true 2>&1 | Out-Null
kubectl delete -f ..\k8s\platform\minio.yaml --ignore-not-found=true 2>&1 | Out-Null
Write-Host "  ✓ Platform services removed" -ForegroundColor Green

Write-Host "Removing Observability services..." -ForegroundColor Yellow
kubectl delete -f ..\k8s\observability\prometheus.yaml --ignore-not-found=true 2>&1 | Out-Null
kubectl delete -f ..\k8s\observability\grafana.yaml --ignore-not-found=true 2>&1 | Out-Null
kubectl delete -f ..\k8s\observability\loki.yaml --ignore-not-found=true 2>&1 | Out-Null
Write-Host "  ✓ Observability services removed" -ForegroundColor Green

Write-Host "Removing MLflow..." -ForegroundColor Yellow
kubectl delete -f ..\k8s\mlflow\mlflow.yaml --ignore-not-found=true 2>&1 | Out-Null
kubectl delete -f ..\k8s\mlflow\mlflow-init.yaml --ignore-not-found=true 2>&1 | Out-Null
Write-Host "  ✓ MLflow removed" -ForegroundColor Green

Write-Host "Removing Temporal..." -ForegroundColor Yellow
kubectl delete -f ..\k8s\temporal\temporal-server.yaml --ignore-not-found=true 2>&1 | Out-Null
kubectl delete -f ..\k8s\temporal\temporal-ui.yaml --ignore-not-found=true 2>&1 | Out-Null
kubectl delete -f ..\k8s\temporal\temporal-init.yaml --ignore-not-found=true 2>&1 | Out-Null
Write-Host "  ✓ Temporal removed" -ForegroundColor Green

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Cleanup Complete" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now deploy using Helm:" -ForegroundColor White
Write-Host "  .\deploy-helm-all.ps1" -ForegroundColor Gray
Write-Host ""
