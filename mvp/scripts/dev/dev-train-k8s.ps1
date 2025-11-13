# Run training in K8s without rebuilding images
# Injects code via ConfigMap for fast iteration

param(
    [string]$Script = "mvp/training/train.py",
    [string]$JobId = "dev-$(Get-Date -Format 'yyyyMMdd-HHmmss')",
    [string]$ModelName = "yolo11n",
    [string]$Framework = "ultralytics",
    [string]$Image = "ghcr.io/myorg/trainer-ultralytics:v1.0",
    [int]$NumEpochs = 10,
    [switch]$Watch  # Watch logs in real-time
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "K8s Training (Code Injection)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Job ID:       $JobId" -ForegroundColor White
Write-Host "Script:       $Script" -ForegroundColor White
Write-Host "Model:        $ModelName" -ForegroundColor White
Write-Host "Framework:    $Framework" -ForegroundColor White
Write-Host "Image:        $Image" -ForegroundColor White
Write-Host "Epochs:       $NumEpochs" -ForegroundColor White
Write-Host ""

# Check if script exists
if (-not (Test-Path $Script)) {
    Write-Host "✗ Script not found: $Script" -ForegroundColor Red
    exit 1
}

# Create ConfigMap with training script
Write-Host "Step 1: Creating ConfigMap with training code..." -ForegroundColor Yellow

$configMapName = "training-code-$JobId"
kubectl create configmap $configMapName `
    --from-file=train.py=$Script `
    --namespace=training `
    --dry-run=client -o yaml | kubectl apply -f -

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to create ConfigMap" -ForegroundColor Red
    exit 1
}

Write-Host "✓ ConfigMap created: $configMapName" -ForegroundColor Green
Write-Host ""

# Create Job YAML
Write-Host "Step 2: Creating K8s Job..." -ForegroundColor Yellow

$jobYaml = @"
apiVersion: batch/v1
kind: Job
metadata:
  name: $JobId
  namespace: training
  labels:
    job-type: training
    framework: $Framework
spec:
  ttlSecondsAfterFinished: 3600  # Auto-cleanup after 1 hour
  template:
    metadata:
      labels:
        job-id: $JobId
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: $Image
        command: ["python", "/code/train.py"]
        env:
        # MLflow
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: mlflow-config
              key: mlflow-tracking-uri
        - name: MLFLOW_S3_ENDPOINT_URL
          valueFrom:
            configMapKeyRef:
              name: mlflow-config
              key: mlflow-s3-endpoint-url
        - name: MLFLOW_S3_IGNORE_TLS
          value: "true"

        # MinIO S3 credentials
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: r2-credentials
              key: access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: r2-credentials
              key: secret-key

        # Training parameters
        - name: JOB_ID
          value: "$JobId"
        - name: MODEL_NAME
          value: "$ModelName"
        - name: FRAMEWORK
          value: "$Framework"
        - name: NUM_EPOCHS
          value: "$NumEpochs"

        # Code injection via ConfigMap
        volumeMounts:
        - name: training-code
          mountPath: /code

        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"

      volumes:
      - name: training-code
        configMap:
          name: $configMapName
"@

# Apply Job
$jobYaml | kubectl apply -f -

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to create Job" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Job created: $JobId" -ForegroundColor Green
Write-Host ""

# Wait for Pod to start
Write-Host "Step 3: Waiting for Pod to start..." -ForegroundColor Yellow

$maxWait = 60
$elapsed = 0

while ($elapsed -lt $maxWait) {
    $podName = kubectl get pods -n training -l "job-id=$JobId" -o jsonpath='{.items[0].metadata.name}' 2>$null

    if ($podName) {
        Write-Host "✓ Pod started: $podName" -ForegroundColor Green
        break
    }

    Start-Sleep -Seconds 2
    $elapsed += 2
}

if (-not $podName) {
    Write-Host "✗ Timeout waiting for Pod" -ForegroundColor Red
    Write-Host "Check status: kubectl get pods -n training -l job-id=$JobId" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Show logs
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Started" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Job:          $JobId" -ForegroundColor White
Write-Host "Pod:          $podName" -ForegroundColor White
Write-Host ""

if ($Watch) {
    Write-Host "Streaming logs (Ctrl+C to stop)..." -ForegroundColor Yellow
    Write-Host ""

    kubectl logs -n training -f $podName
} else {
    Write-Host "View logs:" -ForegroundColor Yellow
    Write-Host "  kubectl logs -n training -f $podName" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Monitor job:" -ForegroundColor Yellow
    Write-Host "  kubectl get job -n training $JobId" -ForegroundColor Gray
    Write-Host "  kubectl get pod -n training $podName" -ForegroundColor Gray
    Write-Host ""
    Write-Host "View results:" -ForegroundColor Yellow
    Write-Host "  MLflow UI: http://localhost:30500" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Cleanup:" -ForegroundColor Yellow
Write-Host "  kubectl delete job -n training $JobId" -ForegroundColor Gray
Write-Host "  kubectl delete configmap -n training $configMapName" -ForegroundColor Gray
Write-Host ""
