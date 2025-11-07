# MLflow Tracking Server Setup

MLflow는 ML 실험 추적, 모델 관리, 메트릭 시각화를 위한 플랫폼입니다.

## 배포 상태

```bash
# MLflow Pod 확인
kubectl get pods -n monitoring -l app=mlflow

# MLflow Service 확인
kubectl get svc -n monitoring mlflow
```

## 접근 방법

### MLflow UI (Web Interface)

**NodePort를 통한 직접 접근:**
```
http://localhost:30500
```

**Port-forward를 통한 접근 (권장):**
```bash
kubectl port-forward -n monitoring svc/mlflow 5000:5000

# 브라우저에서 열기
http://localhost:5000
```

### 클러스터 내부 접근 (Training Jobs)

Training namespace의 Pod들은 다음 endpoint를 사용:
```
http://mlflow.monitoring.svc.cluster.local:5000
```

## MLflow와 MinIO 연동

MLflow는 MinIO를 artifact store로 사용하도록 설정되어 있습니다:

**Backend Store:** SQLite (`/mlflow/mlflow.db`)
- 실험 메타데이터, 파라미터, 메트릭 저장

**Artifact Store:** MinIO S3 (`s3://training-results/mlflow-artifacts`)
- 모델 파일, 플롯 이미지, 대용량 아티팩트 저장

## Python에서 MLflow 사용

### 기본 설정

```python
import mlflow
import os

# MLflow Tracking URI 설정
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow.monitoring.svc.cluster.local:5000'

# MinIO (S3) 설정
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio.storage.svc.cluster.local:9000'
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

# 또는 mlflow.set_tracking_uri() 사용
mlflow.set_tracking_uri('http://mlflow.monitoring.svc.cluster.local:5000')
```

### 실험 추적 예제

```python
import mlflow
import mlflow.pytorch

# 실험 이름 설정
mlflow.set_experiment("image-classification")

# Run 시작
with mlflow.start_run(run_name="resnet50-experiment-1"):

    # 파라미터 로깅
    mlflow.log_param("model_name", "resnet50")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("num_epochs", 10)

    # 메트릭 로깅 (각 epoch마다)
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_acc = validate(model, val_loader)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

    # 모델 저장 (artifact로 저장됨)
    mlflow.pytorch.log_model(model, "model")

    # 추가 artifact 저장
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("training_plot.png")

    # 태그 추가
    mlflow.set_tag("framework", "pytorch")
    mlflow.set_tag("task", "classification")
    mlflow.set_tag("dataset", "imagenet-subset")
```

### Training Job에서 사용 (Kubernetes)

Training Pod의 환경변수는 자동으로 설정됩니다:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job-123
  namespace: training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: ghcr.io/myorg/trainer-timm:v1.0
        env:
        # MLflow 설정 (ConfigMap에서 자동 주입)
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: mlflow-config
              key: mlflow-tracking-uri

        # MinIO S3 설정
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
        - name: MLFLOW_S3_ENDPOINT_URL
          valueFrom:
            configMapKeyRef:
              name: mlflow-config
              key: mlflow-s3-endpoint-url
        - name: MLFLOW_S3_IGNORE_TLS
          value: "true"
```

## MLflow UI 기능

### 1. Experiments 페이지
- 모든 실험 목록 확인
- 실험별 Run 목록
- 메트릭 비교

### 2. Run 상세 페이지
- 파라미터, 메트릭, 태그 확인
- 메트릭 시각화 (차트)
- Artifact 다운로드 (모델, 플롯 등)

### 3. Compare Runs
- 여러 Run을 동시에 비교
- 파라미터와 메트릭을 테이블로 비교
- Parallel Coordinates Plot

### 4. Models (Model Registry)
- 모델 등록 및 버전 관리
- 모델 Stage (Staging, Production)
- 모델 설명 및 메타데이터

## 주요 명령어

### 실험 목록 확인

```python
import mlflow

# 모든 실험 목록
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Name: {exp.name}, ID: {exp.experiment_id}")

# 특정 실험의 모든 Run 검색
runs = mlflow.search_runs(experiment_ids=["1"])
print(runs[["run_id", "params.learning_rate", "metrics.val_accuracy"]])
```

### Run 비교

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 특정 실험의 모든 Run 가져오기
runs = client.search_runs(
    experiment_ids=["1"],
    order_by=["metrics.val_accuracy DESC"],
    max_results=5
)

for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {run.data.metrics['val_accuracy']}")
    print(f"Learning Rate: {run.data.params['learning_rate']}")
    print("---")
```

### 모델 로드

```python
import mlflow.pytorch

# Run ID로 모델 로드
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

# 또는 Model Registry에서 로드
model = mlflow.pytorch.load_model("models:/resnet50-classifier/Production")
```

## Artifact 저장 위치

MLflow는 다음과 같이 MinIO에 artifact를 저장합니다:

```
s3://training-results/mlflow-artifacts/
├── 0/                          # Experiment ID 0
│   ├── abc123def456/           # Run ID
│   │   ├── artifacts/
│   │   │   ├── model/          # 모델 파일
│   │   │   ├── plots/          # 플롯 이미지
│   │   │   └── data/           # 기타 데이터
│   │   └── metrics/
│   └── xyz789ghi012/
└── 1/                          # Experiment ID 1
```

MinIO Console에서 확인:
```
http://localhost:30901
Bucket: training-results
Path: mlflow-artifacts/
```

## 데이터 영속성

**현재 설정:** `emptyDir` (Pod 재시작 시 메타데이터 삭제)
- Artifact는 MinIO에 저장되므로 영속적
- 메타데이터(실험, Run 정보)는 SQLite에 저장되므로 Pod 재시작 시 삭제됨

**영속적 저장소로 변경하려면:**

1. PersistentVolumeClaim 생성:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-pvc
  namespace: monitoring
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

2. `mlflow-config.yaml`의 volume 섹션 수정:
```yaml
volumes:
- name: mlflow-data
  persistentVolumeClaim:
    claimName: mlflow-pvc
```

3. 재배포:
```bash
kubectl apply -f mvp/k8s/mlflow-config.yaml
```

## Production 환경 설정

Production에서는 PostgreSQL을 Backend Store로 사용 권장:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://user:password@postgres:5432/mlflow \
  --default-artifact-root s3://production-bucket/mlflow-artifacts \
  --serve-artifacts
```

## 트러블슈팅

### MLflow UI 접근 불가

```bash
# Port-forward로 직접 접근
kubectl port-forward -n monitoring svc/mlflow 5000:5000

# 브라우저: http://localhost:5000
```

### Artifact 업로드 실패

```bash
# MLflow Pod 로그 확인
kubectl logs -n monitoring deployment/mlflow

# MinIO 연결 확인
kubectl exec -n monitoring deployment/mlflow -- \
  curl -f http://minio.storage.svc.cluster.local:9000/minio/health/live
```

### Run이 기록되지 않음

```python
# Python 코드에서 디버그
import mlflow
import os

print("MLFLOW_TRACKING_URI:", os.getenv('MLFLOW_TRACKING_URI'))
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

# 헬스 체크
import requests
response = requests.get('http://mlflow.monitoring.svc.cluster.local:5000/health')
print("MLflow health:", response.text)
```

## 통합된 모니터링 스택

```
┌─────────────────────────────────────┐
│  Monitoring Namespace               │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────┐  ┌──────────┐        │
│  │ MLflow   │  │Prometheus│        │
│  │ :30500   │  │ :30090   │        │
│  └────┬─────┘  └────┬─────┘        │
│       │             │               │
│  ┌────▼─────────────▼─────┐        │
│  │     Grafana :30030     │        │
│  └────────────────────────┘        │
│                                     │
└─────────────────────────────────────┘
           ▲
           │ Metrics & Artifacts
           │
┌──────────┴──────────────────────────┐
│  Training Namespace                 │
├─────────────────────────────────────┤
│  Training Jobs → MLflow + MinIO     │
└─────────────────────────────────────┘
```

## 다음 단계

MLflow 설정이 완료되었으므로:
1. ✅ 실험 추적 및 메트릭 로깅 가능
2. ✅ 모델 버전 관리 가능
3. ✅ Artifact를 MinIO에 저장 가능
4. ✅ UI에서 실험 비교 가능

다음으로 샘플 Training Job을 생성하여 MLflow 통합을 테스트할 수 있습니다.
