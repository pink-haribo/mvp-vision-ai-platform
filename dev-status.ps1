# Vision AI Training Platform - Development Environment Status Script
# Shows current status of all services

param(
    [string]$ClusterName = "training-dev",
    [switch]$Watch  # Continuously watch status
)

$ErrorActionPreference = "SilentlyContinue"

function Show-Status {
    Clear-Host

    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Vision AI Training Platform" -ForegroundColor Cyan
    Write-Host "Development Environment Status" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""

    # Check if cluster exists
    $clusterExists = kind get clusters 2>$null | Select-String -Pattern "^$ClusterName$"

    if (-not $clusterExists) {
        Write-Host "✗ Cluster '$ClusterName' does not exist" -ForegroundColor Red
        Write-Host ""
        Write-Host "To create cluster: .\dev-start.ps1" -ForegroundColor Yellow
        return
    }

    # Check if cluster is running
    $containerStatus = docker inspect -f '{{.State.Status}}' "$ClusterName-control-plane" 2>$null

    if ($containerStatus -ne "running") {
        Write-Host "✗ Cluster '$ClusterName' is stopped" -ForegroundColor Red
        Write-Host ""
        Write-Host "To start cluster: docker start $ClusterName-control-plane" -ForegroundColor Yellow
        Write-Host "Or run: .\dev-start.ps1 --SkipBuild" -ForegroundColor Yellow
        return
    }

    Write-Host "✓ Cluster '$ClusterName' is running" -ForegroundColor Green
    Write-Host ""

    # Get cluster info
    Write-Host "Cluster Information:" -ForegroundColor Yellow
    $kubeContext = kubectl config current-context 2>$null
    Write-Host "  Context:          $kubeContext" -ForegroundColor White

    $kubeVersion = kubectl version 2>$null | Select-String "Server Version" | Select-Object -First 1
    if ($kubeVersion) {
        Write-Host "  Kubernetes:       $($kubeVersion.ToString().Trim())" -ForegroundColor White
    } else {
        Write-Host "  Kubernetes:       (version check unavailable)" -ForegroundColor Gray
    }
    Write-Host ""

    # Storage (MinIO)
    Write-Host "Storage (namespace: storage):" -ForegroundColor Yellow
    $minioPods = kubectl get pods -n storage -l app=minio -o jsonpath='{.items[*].status.phase}' 2>$null
    $minioPVC = kubectl get pvc -n storage minio-pvc -o jsonpath='{.status.phase}' 2>$null

    if ($minioPods -eq "Running") {
        Write-Host "  MinIO:            " -NoNewline -ForegroundColor White
        Write-Host "✓ Running" -ForegroundColor Green
        Write-Host "    Console:        http://localhost:30901 (minioadmin/minioadmin)" -ForegroundColor Gray
        Write-Host "    API:            http://localhost:30900" -ForegroundColor Gray
        Write-Host "    PVC:            $minioPVC (20Gi)" -ForegroundColor Gray
    } else {
        Write-Host "  MinIO:            " -NoNewline -ForegroundColor White
        Write-Host "✗ Not Running" -ForegroundColor Red
    }
    Write-Host ""

    # Monitoring (MLflow, Prometheus, Grafana)
    Write-Host "Monitoring (namespace: monitoring):" -ForegroundColor Yellow

    # MLflow
    $mlflowPods = kubectl get pods -n monitoring -l app=mlflow -o jsonpath='{.items[*].status.phase}' 2>$null
    $mlflowPVC = kubectl get pvc -n monitoring mlflow-pvc -o jsonpath='{.status.phase}' 2>$null

    if ($mlflowPods -eq "Running") {
        Write-Host "  MLflow:           " -NoNewline -ForegroundColor White
        Write-Host "✓ Running" -ForegroundColor Green
        Write-Host "    UI:             http://localhost:30500" -ForegroundColor Gray
        Write-Host "    PVC:            $mlflowPVC (5Gi)" -ForegroundColor Gray
    } else {
        Write-Host "  MLflow:           " -NoNewline -ForegroundColor White
        Write-Host "✗ Not Running" -ForegroundColor Red
    }

    # Prometheus
    $promPods = kubectl get pods -n monitoring -l app=prometheus -o jsonpath='{.items[*].status.phase}' 2>$null

    if ($promPods -eq "Running") {
        Write-Host "  Prometheus:       " -NoNewline -ForegroundColor White
        Write-Host "✓ Running" -ForegroundColor Green
        Write-Host "    UI:             http://localhost:30090" -ForegroundColor Gray
    } else {
        Write-Host "  Prometheus:       " -NoNewline -ForegroundColor White
        Write-Host "✗ Not Running" -ForegroundColor Red
    }

    # Grafana
    $grafanaPods = kubectl get pods -n monitoring -l app=grafana -o jsonpath='{.items[*].status.phase}' 2>$null

    if ($grafanaPods -eq "Running") {
        Write-Host "  Grafana:          " -NoNewline -ForegroundColor White
        Write-Host "✓ Running" -ForegroundColor Green
        Write-Host "    UI:             http://localhost:30030 (admin/admin)" -ForegroundColor Gray
    } else {
        Write-Host "  Grafana:          " -NoNewline -ForegroundColor White
        Write-Host "✗ Not Running" -ForegroundColor Red
    }
    Write-Host ""

    # Training namespace
    Write-Host "Training (namespace: training):" -ForegroundColor Yellow

    $runningJobs = kubectl get jobs -n training --no-headers 2>$null | Measure-Object | Select-Object -ExpandProperty Count
    $completedJobs = kubectl get jobs -n training --field-selector status.successful=1 --no-headers 2>$null | Measure-Object | Select-Object -ExpandProperty Count

    Write-Host "  Active Jobs:      $runningJobs" -ForegroundColor White
    Write-Host "  Completed Jobs:   $completedJobs" -ForegroundColor White

    $trainingPods = kubectl get pods -n training --no-headers 2>$null
    if ($trainingPods) {
        Write-Host "  Running Pods:" -ForegroundColor White
        kubectl get pods -n training --no-headers 2>$null | ForEach-Object {
            Write-Host "    $_" -ForegroundColor Gray
        }
    }
    Write-Host ""

    # Resource usage
    Write-Host "Resource Usage:" -ForegroundColor Yellow

    $nodes = kubectl top nodes --no-headers 2>$null
    if ($nodes) {
        Write-Host "  Nodes:" -ForegroundColor White
        kubectl top nodes 2>$null | ForEach-Object {
            if ($_ -match "NAME" -or $_ -match "training-dev") {
                Write-Host "    $_" -ForegroundColor Gray
            }
        }
    }
    Write-Host ""

    # Docker images
    Write-Host "Docker Images in Cluster:" -ForegroundColor Yellow
    $images = docker exec "$ClusterName-control-plane" crictl images 2>$null | Select-String "trainer"
    if ($images) {
        Write-Host "  $($images.Count) trainer images loaded" -ForegroundColor White
    } else {
        Write-Host "  No trainer images found" -ForegroundColor Red
    }
    Write-Host ""

    # Quick commands
    Write-Host "Quick Commands:" -ForegroundColor Yellow
    Write-Host "  All pods:         kubectl get pods --all-namespaces" -ForegroundColor Gray
    Write-Host "  All services:     kubectl get svc --all-namespaces" -ForegroundColor Gray
    Write-Host "  All PVCs:         kubectl get pvc --all-namespaces" -ForegroundColor Gray
    Write-Host "  MLflow logs:      kubectl logs -n monitoring deployment/mlflow -f" -ForegroundColor Gray
    Write-Host ""

    if ($Watch) {
        Write-Host "Refreshing in 5 seconds... (Ctrl+C to stop)" -ForegroundColor Gray
    }
}

if ($Watch) {
    while ($true) {
        Show-Status
        Start-Sleep -Seconds 5
    }
} else {
    Show-Status
}
