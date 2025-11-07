# Docker Desktop vs Kubernetes 실행 비교

MLflow, MinIO 등을 Docker Desktop에서 직접 실행하던 것과 Kubernetes Pod로 실행하는 것의 차이점 정리.

## 실행 방식 비교

### Docker Desktop (기존)

```bash
# MLflow 실행
docker run -d --name mlflow \
  -p 5000:5000 \
  -v ~/mlflow:/mlflow \
  -e AWS_ACCESS_KEY_ID=minioadmin \
  -e AWS_SECRET_ACCESS_KEY=minioadmin \
  -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
  ghcr.io/mlflow/mlflow:v2.9.2 \
  mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow/mlflow.db \
    --default-artifact-root s3://mlflow-artifacts

# MinIO 실행
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -v ~/minio/data:/data \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Training 실행
docker run --rm \
  --network host \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -e AWS_ACCESS_KEY_ID=minioadmin \
  -e AWS_SECRET_ACCESS_KEY=minioadmin \
  -e MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
  trainer-image python train.py
```

**특징:**
- ✅ 간단하고 빠른 시작
- ✅ 로컬 파일 시스템 직접 마운트
- ❌ 수동 컨테이너 관리 (재시작, 헬스체크)
- ❌ 네트워크 설정 복잡 (--network, --link)
- ❌ 리소스 제한 어려움
- ❌ 확장성 제한

### Kubernetes (현재)

```yaml
# MLflow Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: monitoring
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.9.2
        env:
        - name: AWS_ACCESS_KEY_ID
          value: "minioadmin"
        # ... 환경변수
        volumeMounts:
        - name: mlflow-data
          mountPath: /mlflow
      volumes:
      - name: mlflow-data
        persistentVolumeClaim:
          claimName: mlflow-pvc

# Training Job
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
        image: trainer-image
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow.monitoring.svc.cluster.local:5000"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: r2-credentials
              key: access-key
```

**특징:**
- ✅ 자동 재시작 (livenessProbe)
- ✅ 서비스 디스커버리 (DNS 기반)
- ✅ 리소스 제한 (CPU, Memory)
- ✅ 확장성 (replicas)
- ✅ ConfigMap, Secret 관리
- ⚠️ 초기 설정 복잡

## 네트워크 접근 차이

### Docker Desktop

```python
# Training 코드
import mlflow

# 로컬에서 실행 시
mlflow.set_tracking_uri("http://localhost:5000")

# Docker 컨테이너에서 실행 시 (같은 네트워크)
mlflow.set_tracking_uri("http://mlflow:5000")
```

**문제점:**
- 환경마다 URI가 다름
- 하드코딩 필요

### Kubernetes

```python
# Training 코드 (변경 없음!)
import mlflow
import os

# 환경변수에서 자동으로 읽음
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

# 또는 mlflow가 자동으로 환경변수를 읽음
# MLFLOW_TRACKING_URI 환경변수만 설정하면 됨
```

**장점:**
- 코드 변경 없음
- 환경변수로 추상화
- 로컬/K8s 동일한 코드

## 코드 구현 상 차이점

### ❌ 변경 필요 없는 부분

**1. Training 코드 자체:**
```python
# train.py (변경 없음)
import mlflow
import torch

mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_param("lr", 0.001)

    # Training loop
    for epoch in range(10):
        loss = train_epoch(model, dataloader)
        mlflow.log_metric("loss", loss, step=epoch)

    # 모델 저장
    mlflow.pytorch.log_model(model, "model")
```

**2. MLflow API 호출:**
- `mlflow.log_param()` → 동일
- `mlflow.log_metric()` → 동일
- `mlflow.pytorch.log_model()` → 동일

**3. MinIO S3 접근:**
```python
import boto3

# 코드 변경 없음 (환경변수만 다름)
s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)
```

### ✅ 변경되는 부분: 환경변수 값

**Docker Desktop 환경변수:**
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

**Kubernetes 환경변수 (자동 주입):**
```yaml
# Training Job YAML에 정의
env:
- name: MLFLOW_TRACKING_URI
  valueFrom:
    configMapKeyRef:
      name: mlflow-config
      key: mlflow-tracking-uri
  # 값: http://mlflow.monitoring.svc.cluster.local:5000

- name: MLFLOW_S3_ENDPOINT_URL
  valueFrom:
    configMapKeyRef:
      name: mlflow-config
      key: mlflow-s3-endpoint-url
  # 값: http://minio.storage.svc.cluster.local:9000

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
```

## 실제 코드 예제

### 환경에 독립적인 코드 (권장)

```python
# train.py
import mlflow
import os

def main():
    # 환경변수에서 자동으로 읽음
    # Docker: localhost:5000
    # K8s: mlflow.monitoring.svc.cluster.local:5000
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("image-classification")

    with mlflow.start_run(run_name="resnet50-exp1"):
        # Training 코드 (환경 무관)
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        mlflow.log_param("model_name", "resnet50")
        mlflow.log_param("learning_rate", 0.001)

        for epoch in range(num_epochs):
            loss = train_one_epoch(model, train_loader, optimizer)
            mlflow.log_metric("train_loss", loss, step=epoch)

        # 모델 저장 (MinIO에 자동 저장)
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()
```

**이 코드는 Docker와 K8s에서 동일하게 작동합니다!**

### 로컬 개발 vs K8s 실행 비교

**로컬 개발 (Docker Desktop):**
```bash
# 1. 환경변수 설정
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# 2. 직접 실행
python train.py

# 또는 Docker로 실행
docker run --rm \
  --network host \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
  -e AWS_ACCESS_KEY_ID=minioadmin \
  -e AWS_SECRET_ACCESS_KEY=minioadmin \
  trainer-image python train.py
```

**Kubernetes 실행:**
```bash
# 1. Job YAML 작성 (환경변수 자동 주입)
kubectl apply -f training-job.yaml

# Job YAML에 이미 환경변수가 정의되어 있음
# 코드 변경 없이 동일한 train.py 실행
```

## 마이그레이션 가이드

### Docker Desktop → Kubernetes

**변경 필요 없음:**
1. ✅ Training 코드 (`train.py`)
2. ✅ MLflow API 호출
3. ✅ S3/MinIO 접근 코드
4. ✅ 데이터 로딩 코드

**변경 필요:**
1. 🔄 환경변수 값
   - `localhost` → `service-name.namespace.svc.cluster.local`
2. 🔄 실행 방법
   - `docker run` → `kubectl apply -f job.yaml`
3. 🔄 볼륨 마운트
   - `-v ~/data:/data` → PVC + volumeMounts

**추가 이점:**
1. ✅ 자동 재시작 (CrashLoopBackOff)
2. ✅ 리소스 제한 (requests/limits)
3. ✅ ConfigMap으로 설정 관리
4. ✅ Secret으로 자격증명 관리
5. ✅ 로그 중앙 관리 (`kubectl logs`)

## 개발 워크플로우

### Docker Desktop (기존)

```bash
# 1. 서비스 시작
docker-compose up -d mlflow minio

# 2. 코드 수정
vim train.py

# 3. 로컬 실행
export MLFLOW_TRACKING_URI=http://localhost:5000
python train.py

# 4. MLflow UI 확인
open http://localhost:5000
```

### Kubernetes (현재)

```bash
# 1. 서비스는 이미 실행 중 (Deployment)
kubectl get pods -n monitoring

# 2. 코드 수정
vim train.py

# 3. 로컬 테스트 (동일한 방식)
export MLFLOW_TRACKING_URI=http://localhost:30500  # NodePort
python train.py

# 또는 K8s Job으로 실행
kubectl apply -f training-job.yaml

# 4. MLflow UI 확인 (동일)
open http://localhost:30500
```

## 핵심 차이점 요약

| 항목 | Docker Desktop | Kubernetes |
|------|---------------|------------|
| **Training 코드** | ✅ 동일 | ✅ 동일 |
| **MLflow API** | ✅ 동일 | ✅ 동일 |
| **환경변수 값** | localhost:5000 | service-dns:5000 |
| **환경변수 설정** | export 또는 -e | ConfigMap/Secret |
| **실행 방법** | docker run | kubectl apply |
| **서비스 관리** | 수동 (docker start/stop) | 자동 (Deployment) |
| **영속성** | -v 마운트 | PVC |
| **네트워크** | --network 또는 --link | Service DNS |
| **확장성** | 수동 복제 | replicas |

## 결론

**코드 변경이 거의 없습니다!**

- ✅ Training 코드: **변경 없음**
- ✅ MLflow 사용법: **변경 없음**
- ✅ S3/MinIO 접근: **변경 없음**
- 🔄 환경변수 값만 변경 (자동 주입)
- 🔄 실행 방법만 변경 (docker run → kubectl apply)

**장점:**
- 환경변수로 추상화되어 있어 코드 변경 최소화
- 로컬 개발과 K8s 실행의 일관성
- Production 배포 시 코드 변경 불필요

**다음 단계:**
실제 Training Job을 K8s에서 실행하여 전체 플로우를 테스트해보면 됩니다!
