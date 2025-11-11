# Vision AI Training Platform - Development Environment Stop Script
# Stops Kind cluster and cleans up resources

param(
    [switch]$DeleteCluster,   # Delete the entire cluster
    [string]$ClusterName = "training-dev"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Vision AI Training Platform" -ForegroundColor Cyan
Write-Host "Development Environment Shutdown" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if cluster exists
$clusterExists = kind get clusters 2>$null | Select-String -Pattern "^$ClusterName$"

if (-not $clusterExists) {
    Write-Host "✓ Cluster '$ClusterName' does not exist" -ForegroundColor Green
    exit 0
}

if ($DeleteCluster) {
    Write-Host "Deleting Kind cluster '$ClusterName'..." -ForegroundColor Yellow
    Write-Host "(This will delete all data including PVCs)" -ForegroundColor Red
    Write-Host ""

    $confirmation = Read-Host "Are you sure? (yes/no)"
    if ($confirmation -ne "yes") {
        Write-Host "Cancelled." -ForegroundColor Gray
        exit 0
    }

    kind delete cluster --name $ClusterName

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Cluster deleted" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to delete cluster" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Stopping Kind cluster '$ClusterName'..." -ForegroundColor Yellow
    Write-Host ""

    # Kind doesn't have a native "stop" command, but we can stop the Docker container
    $containerName = "$ClusterName-control-plane"

    docker stop $containerName >$null 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Cluster stopped (Docker container paused)" -ForegroundColor Green
        Write-Host ""
        Write-Host "To restart: docker start $containerName" -ForegroundColor Gray
        Write-Host "Or run: .\dev-start.ps1 --SkipBuild" -ForegroundColor Gray
    } else {
        Write-Host "✗ Failed to stop cluster" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Shutdown Complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $DeleteCluster) {
    Write-Host "Note: Cluster is stopped but data is preserved." -ForegroundColor Yellow
    Write-Host "To completely delete the cluster:" -ForegroundColor Yellow
    Write-Host "  .\dev-stop.ps1 -DeleteCluster" -ForegroundColor Gray
    Write-Host ""
}
