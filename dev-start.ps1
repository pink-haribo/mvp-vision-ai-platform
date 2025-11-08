# Vision AI Training Platform - Development Environment Start Script
# Starts Kind cluster and all required services

param(
    [switch]$SkipBuild,      # Skip Docker image build
    [switch]$SkipLoadImages, # Skip loading Docker images to cluster
    [switch]$Fresh,          # Delete existing cluster and start fresh
    [string]$ClusterName = "training-dev"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Vision AI Training Platform" -ForegroundColor Cyan
Write-Host "Development Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-Command {
    param($CommandName)
    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

# Function to wait for pods to be ready
function Wait-ForPods {
    param(
        [string]$Namespace,
        [string]$LabelSelector,
        [int]$TimeoutSeconds = 120
    )

    Write-Host "Waiting for pods in $Namespace with selector $LabelSelector..." -ForegroundColor Yellow

    $elapsed = 0
    while ($elapsed -lt $TimeoutSeconds) {
        $pods = kubectl get pods -n $Namespace -l $LabelSelector -o jsonpath='{.items[*].status.phase}' 2>$null
        if ($pods -match "Running") {
            $ready = kubectl get pods -n $Namespace -l $LabelSelector -o jsonpath='{.items[*].status.containerStatuses[*].ready}' 2>$null
            if ($ready -notmatch "false") {
                Write-Host "✓ Pods ready in $Namespace" -ForegroundColor Green
                return $true
            }
        }
        Start-Sleep -Seconds 3
        $elapsed += 3
    }

    Write-Host "✗ Timeout waiting for pods in $Namespace" -ForegroundColor Red
    return $false
}

# 1. Check prerequisites
Write-Host "Step 1: Checking prerequisites..." -ForegroundColor Yellow
Write-Host ""

$missing = @()

if (-not (Test-Command "kind")) {
    $missing += "kind"
}

if (-not (Test-Command "kubectl")) {
    $missing += "kubectl"
}

if (-not (Test-Command "docker")) {
    $missing += "docker"
}

if ($missing.Count -gt 0) {
    Write-Host "✗ Missing required tools: $($missing -join ', ')" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install:"
    foreach ($tool in $missing) {
        Write-Host "  - $tool"
    }
    exit 1
}

Write-Host "✓ All prerequisites installed" -ForegroundColor Green
Write-Host ""

# 2. Check Docker is running
Write-Host "Step 2: Checking Docker..." -ForegroundColor Yellow
docker ps >$null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker is running" -ForegroundColor Green
Write-Host ""

# 3. Handle existing cluster
Write-Host "Step 3: Checking Kind cluster..." -ForegroundColor Yellow

try {
    $clusterList = kind get clusters 2>&1
    $clusterExists = ($clusterList -join "`n") -match "^$ClusterName$"
} catch {
    $clusterExists = $false
}

if ($Fresh -and $clusterExists) {
    Write-Host "Deleting existing cluster (--Fresh flag)..." -ForegroundColor Yellow
    kind delete cluster --name $ClusterName
    $clusterExists = $false
}

if (-not $clusterExists) {
    Write-Host "Creating Kind cluster '$ClusterName'..." -ForegroundColor Yellow
    kind create cluster --name $ClusterName --config mvp/k8s/kind-config.yaml

    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to create cluster" -ForegroundColor Red
        exit 1
    }

    Write-Host "✓ Cluster created" -ForegroundColor Green
} else {
    Write-Host "✓ Cluster '$ClusterName' already exists" -ForegroundColor Green

    # Set kubectl context
    kubectl cluster-info --context kind-$ClusterName >$null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to connect to cluster" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# 4. Build and load Docker images
if (-not $SkipLoadImages) {
    if (-not $SkipBuild) {
        Write-Host "Step 4: Building Docker images..." -ForegroundColor Yellow
        Write-Host "(This may take 5-10 minutes on first build)" -ForegroundColor Gray
        Write-Host ""

        # Check if images already exist
        $baseImage = docker images ghcr.io/myorg/trainer-base:v1.0 -q
        $ultralyticsImage = docker images ghcr.io/myorg/trainer-ultralytics:v1.0 -q
        $timmImage = docker images ghcr.io/myorg/trainer-timm:v1.0 -q

        if ($baseImage -and $ultralyticsImage -and $timmImage) {
            Write-Host "✓ Docker images already built" -ForegroundColor Green
            Write-Host "  Use --Fresh to rebuild" -ForegroundColor Gray
        } else {
            Push-Location mvp/training/docker

            # Build images
            Write-Host "Building base image..." -ForegroundColor Yellow
            powershell.exe -ExecutionPolicy Bypass -File build.ps1 -Target base

            Write-Host "Building ultralytics image..." -ForegroundColor Yellow
            powershell.exe -ExecutionPolicy Bypass -File build.ps1 -Target ultralytics

            Write-Host "Building timm image..." -ForegroundColor Yellow
            powershell.exe -ExecutionPolicy Bypass -File build.ps1 -Target timm

            Pop-Location

            Write-Host "✓ Docker images built" -ForegroundColor Green
        }
        Write-Host ""
    } else {
        Write-Host "Step 4: Skipping Docker build (--SkipBuild flag)" -ForegroundColor Gray
        Write-Host ""
    }

    Write-Host "Step 5: Loading images to Kind cluster..." -ForegroundColor Yellow
    Write-Host "(This may take 2-3 minutes)" -ForegroundColor Gray

    # Check if images are already loaded
    $imagesInCluster = docker exec $ClusterName-control-plane crictl images 2>$null

    if ($imagesInCluster -match "trainer-base.*v1.0" -and
        $imagesInCluster -match "trainer-ultralytics.*v1.0" -and
        $imagesInCluster -match "trainer-timm.*v1.0") {
        Write-Host "✓ Images already loaded in cluster" -ForegroundColor Green
    } else {
        kind load docker-image ghcr.io/myorg/trainer-base:v1.0 --name $ClusterName
        kind load docker-image ghcr.io/myorg/trainer-ultralytics:v1.0 --name $ClusterName
        kind load docker-image ghcr.io/myorg/trainer-timm:v1.0 --name $ClusterName
        Write-Host "✓ Images loaded to cluster" -ForegroundColor Green
    }
    Write-Host ""
} else {
    Write-Host "Step 4-5: Skipping Docker image loading (--SkipLoadImages flag)" -ForegroundColor Gray
    Write-Host "  Note: Trainer images will need to be loaded manually before running training jobs" -ForegroundColor Yellow
    Write-Host ""
}

# 6. Deploy Kubernetes resources
Write-Host "Step 6: Deploying Kubernetes resources..." -ForegroundColor Yellow
Write-Host ""

# Create namespaces
Write-Host "  - Creating namespaces..." -ForegroundColor Gray
kubectl create namespace storage --dry-run=client -o yaml | kubectl apply -f - >$null 2>&1
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f - >$null 2>&1

# Training namespace and secrets
Write-Host "  - Training namespace and secrets..." -ForegroundColor Gray
Push-Location mvp/k8s
powershell.exe -ExecutionPolicy Bypass -File setup.ps1
Pop-Location

# MinIO (Storage)
Write-Host "  - MinIO (object storage)..." -ForegroundColor Gray
kubectl apply -f mvp/k8s/minio-pvc.yaml >$null 2>&1
kubectl apply -f mvp/k8s/minio-config.yaml >$null 2>&1

# MLflow
Write-Host "  - MLflow (experiment tracking)..." -ForegroundColor Gray
kubectl apply -f mvp/k8s/mlflow-pvc.yaml >$null 2>&1
kubectl apply -f mvp/k8s/mlflow-config.yaml >$null 2>&1

# Prometheus + Grafana
Write-Host "  - Prometheus (metrics)..." -ForegroundColor Gray
kubectl apply -f mvp/k8s/prometheus/prometheus-config.yaml >$null 2>&1

Write-Host "  - Grafana (dashboards)..." -ForegroundColor Gray
kubectl apply -f mvp/k8s/prometheus/grafana-config.yaml >$null 2>&1

Write-Host "✓ Kubernetes resources deployed" -ForegroundColor Green
Write-Host ""

# 7. Wait for services to be ready
Write-Host "Step 7: Waiting for services to start..." -ForegroundColor Yellow
Write-Host ""

Wait-ForPods -Namespace "storage" -LabelSelector "app=minio" -TimeoutSeconds 60
Wait-ForPods -Namespace "monitoring" -LabelSelector "app=mlflow" -TimeoutSeconds 60
Wait-ForPods -Namespace "monitoring" -LabelSelector "app=prometheus" -TimeoutSeconds 60
Wait-ForPods -Namespace "monitoring" -LabelSelector "app=grafana" -TimeoutSeconds 90

Write-Host ""

# 8. Create MinIO buckets
Write-Host "Step 8: Setting up MinIO buckets..." -ForegroundColor Yellow
$bucketCheck = kubectl exec -n storage deployment/minio -- sh -c "ls -d /data/training-* 2>/dev/null" 2>$null

if (-not $bucketCheck) {
    kubectl exec -n storage deployment/minio -- sh -c "mkdir -p /data/training-datasets /data/training-checkpoints /data/training-results" >$null 2>&1
    Write-Host "✓ MinIO buckets created" -ForegroundColor Green
} else {
    Write-Host "✓ MinIO buckets already exist" -ForegroundColor Green
}
Write-Host ""

# 9. Upload training configuration schemas
Write-Host "Step 9: Uploading training configuration schemas..." -ForegroundColor Yellow

# Create schemas directory in MinIO
kubectl exec -n storage deployment/minio -- mkdir -p /data/training-results/schemas >$null 2>&1

# Check if schema files exist locally
$schemasExist = Test-Path "mvp/training/schemas/ultralytics-schema.json"

if ($schemasExist) {
    # Get MinIO pod name
    $minioPod = kubectl get pods -n storage -l app=minio -o jsonpath='{.items[0].metadata.name}' 2>$null

    if ($minioPod) {
        # Copy schema files to MinIO
        $schemas = @("ultralytics-schema.json", "timm-schema.json")
        $uploadCount = 0

        foreach ($schema in $schemas) {
            $localPath = "mvp/training/schemas/$schema"
            if (Test-Path $localPath) {
                $targetName = $schema -replace "-schema", ""  # ultralytics-schema.json -> ultralytics.json
                kubectl cp $localPath "storage/${minioPod}:/data/training-results/schemas/$targetName" >$null 2>&1
                if ($?) {
                    $uploadCount++
                    Write-Host "  ✓ Uploaded $targetName" -ForegroundColor Gray
                }
            }
        }

        if ($uploadCount -gt 0) {
            Write-Host "✓ Uploaded $uploadCount training configuration schemas" -ForegroundColor Green
        } else {
            Write-Host "⚠ No schemas uploaded (files may not exist)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠ MinIO pod not found, skipping schema upload" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠ Schema files not found in mvp/training/schemas/" -ForegroundColor Yellow
    Write-Host "  Run manually: cd mvp/training && python scripts/upload_schema_to_storage.py --all" -ForegroundColor Gray
}
Write-Host ""

# 10. Display access information
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Development Environment Ready!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Services:" -ForegroundColor Yellow
Write-Host "  MLflow UI:        http://localhost:30500" -ForegroundColor White
Write-Host "  Prometheus:       http://localhost:30090" -ForegroundColor White
Write-Host "  Grafana:          http://localhost:30030 (admin/admin)" -ForegroundColor White
Write-Host "  MinIO Console:    http://localhost:30901 (minioadmin/minioadmin)" -ForegroundColor White
Write-Host "  MinIO API:        http://localhost:30900" -ForegroundColor White
Write-Host ""

Write-Host "Cluster Information:" -ForegroundColor Yellow
Write-Host "  Cluster Name:     $ClusterName" -ForegroundColor White
Write-Host "  Kubernetes:       " -NoNewline -ForegroundColor White
$k8sVersion = kubectl version 2>$null | Select-String "Server Version" | Select-Object -First 1
if ($k8sVersion) {
    Write-Host $k8sVersion.ToString().Trim() -ForegroundColor White
} else {
    Write-Host "(version check unavailable)" -ForegroundColor Gray
}
Write-Host ""

Write-Host "Quick Commands:" -ForegroundColor Yellow
Write-Host "  List all pods:            kubectl get pods --all-namespaces" -ForegroundColor Gray
Write-Host "  Check services:           kubectl get svc --all-namespaces" -ForegroundColor Gray
Write-Host "  View MLflow logs:         kubectl logs -n monitoring deployment/mlflow -f" -ForegroundColor Gray
Write-Host "  Port-forward MLflow:      kubectl port-forward -n monitoring svc/mlflow 5000:5000" -ForegroundColor Gray
Write-Host ""

Write-Host "Stop Environment:" -ForegroundColor Yellow
Write-Host "  .\dev-stop.ps1" -ForegroundColor Gray
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Open MLflow UI: http://localhost:30500" -ForegroundColor White
Write-Host "  2. Run a test training job" -ForegroundColor White
Write-Host "  3. Check metrics in Grafana" -ForegroundColor White
Write-Host ""
