# 데이터 영속성 설정

모든 중요 데이터가 Pod 재시작 후에도 유지되도록 PersistentVolumeClaim(PVC)을 설정했습니다.

## 설정 완료 상태

### ✅ MLflow (Backend Store)

**데이터 위치:** SQLite (`/mlflow/mlflow.db`)
**저장 내용:**
- 실험 메타데이터
- Run 정보 (파라미터, 메트릭)
- 모델 레지스트리

**PVC 설정:**
```yaml
Name: mlflow-pvc
Namespace: monitoring
Storage: 5Gi
```

**확인:**
```bash
kubectl get pvc -n monitoring
kubectl exec -n monitoring deployment/mlflow -- ls -la /mlflow/
```

### ✅ MinIO (Object Storage)

**데이터 위치:** `/data`
**저장 내용:**
- 버킷: training-datasets, training-checkpoints, training-results
- MLflow artifacts (모델 파일, 플롯)
- 학습 체크포인트
- 업로드된 데이터셋

**PVC 설정:**
```yaml
Name: minio-pvc
Namespace: storage
Storage: 20Gi
```

**확인:**
```bash
kubectl get pvc -n storage
kubectl exec -n storage deployment/minio -- ls -la /data/
```

## 데이터 흐름

```
┌─────────────────────────────────────────────┐
│  Training Job                                │
├─────────────────────────────────────────────┤
│                                             │
│  Metrics, Params                            │
│       │                                     │
│       ▼                                     │
│  ┌──────────┐        ┌──────────┐          │
│  │ MLflow   │────────│ SQLite   │          │
│  │          │        │ (PVC)    │          │
│  └────┬─────┘        └──────────┘          │
│       │                                     │
│  Models, Artifacts                          │
│       │                                     │
│       ▼                                     │
│  ┌──────────┐        ┌──────────┐          │
│  │ MinIO    │────────│ /data    │          │
│  │          │        │ (PVC)    │          │
│  └──────────┘        └──────────┘          │
│                                             │
└─────────────────────────────────────────────┘
```

## SQLite vs PostgreSQL

### 로컬 개발 (현재 설정)

**SQLite + PVC**
- ✅ 설정 간단 (추가 서비스 불필요)
- ✅ 리소스 절약
- ✅ 단일 사용자에 충분
- ✅ 빠른 시작
- ⚠️ 동시 쓰기 제한

**언제 사용:**
- 로컬 개발 환경
- Kind/Minikube 클러스터
- 개인 프로젝트
- 프로토타입 단계

### Production 환경

**PostgreSQL 권장**
- ✅ 동시성 지원 (여러 사용자)
- ✅ 확장성
- ✅ 백업/복구 용이
- ✅ 트랜잭션 안정성
- ⚠️ 추가 인프라 필요

**언제 사용:**
- Production 배포
- 팀 협업 환경
- 다수의 동시 실험
- 엔터프라이즈 환경

## Production으로 전환 시

### PostgreSQL로 마이그레이션

1. **PostgreSQL 배포:**
```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: monitoring
spec:
  template:
    spec:
      containers:
      - name: postgres
        image: postgres:16
        env:
        - name: POSTGRES_DB
          value: mlflow
        - name: POSTGRES_USER
          value: mlflow
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
```

2. **MLflow 설정 변경:**
```yaml
# mlflow-config.yaml
containers:
- name: mlflow
  command:
  - mlflow
  - server
  - --backend-store-uri
  - postgresql://mlflow:password@postgres:5432/mlflow  # SQLite 대신
  - --default-artifact-root
  - s3://production-bucket/mlflow-artifacts
```

3. **데이터 마이그레이션:**
```bash
# SQLite → PostgreSQL 마이그레이션 스크립트 필요
# 또는 새로 시작
```

### R2로 전환 (Production Object Storage)

1. **Secret 업데이트:**
```bash
kubectl create secret generic r2-credentials \
  --from-literal=endpoint=https://YOUR_ACCOUNT.r2.cloudflarestorage.com \
  --from-literal=access-key=YOUR_R2_ACCESS_KEY \
  --from-literal=secret-key=YOUR_R2_SECRET_KEY \
  --namespace=training \
  --dry-run=client -o yaml | kubectl apply -f -
```

2. **MLflow Artifact Root 변경:**
```yaml
env:
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
    secretKeyRef:
      name: r2-credentials
      key: endpoint
```

## 스토리지 확장

### PVC 크기 확장

Kind에서는 PVC 크기 변경이 제한적이므로, 처음부터 충분한 크기를 할당하거나 새 PVC를 생성하는 것이 좋습니다.

**Production Kubernetes:**
```bash
# PVC resize 지원 시
kubectl patch pvc mlflow-pvc -n monitoring -p '{"spec":{"resources":{"requests":{"storage":"10Gi"}}}}'
```

### 백업

**MLflow SQLite 백업:**
```bash
# Pod에서 DB 파일 복사
kubectl cp monitoring/mlflow-<pod-id>:/mlflow/mlflow.db ./mlflow-backup.db

# 복원
kubectl cp ./mlflow-backup.db monitoring/mlflow-<pod-id>:/mlflow/mlflow.db
```

**MinIO 데이터 백업:**
```bash
# MinIO Client (mc) 사용
mc mirror local/training-datasets ./backup/datasets/
mc mirror local/training-results ./backup/results/

# 복원
mc mirror ./backup/datasets/ local/training-datasets/
```

## 모니터링

### 스토리지 사용량 확인

```bash
# MLflow PVC 사용량
kubectl exec -n monitoring deployment/mlflow -- df -h /mlflow

# MinIO PVC 사용량
kubectl exec -n storage deployment/minio -- df -h /data

# PVC 상세 정보
kubectl describe pvc mlflow-pvc -n monitoring
kubectl describe pvc minio-pvc -n storage
```

### 로그 확인

```bash
# MLflow 로그
kubectl logs -n monitoring deployment/mlflow -f

# MinIO 로그
kubectl logs -n storage deployment/minio -f
```

## 트러블슈팅

### PVC가 Pending 상태

```bash
# PVC 상태 확인
kubectl describe pvc mlflow-pvc -n monitoring

# StorageClass 확인 (Kind는 standard 사용)
kubectl get storageclass

# PersistentVolume 확인
kubectl get pv
```

### Pod가 PVC를 마운트하지 못함

```bash
# Pod 이벤트 확인
kubectl describe pod -n monitoring -l app=mlflow

# PVC와 Pod가 같은 노드에 있는지 확인 (ReadWriteOnce)
kubectl get pods -n monitoring -o wide
```

### 데이터 손실

PVC를 삭제하면 데이터도 함께 삭제됩니다:

```bash
# ⚠️ 위험: PVC 삭제 = 데이터 삭제
kubectl delete pvc mlflow-pvc -n monitoring  # 절대 하지 말 것!

# 안전한 삭제 순서
1. 백업 생성
2. Deployment 삭제
3. PVC 삭제 (필요 시)
```

## 현재 환경 요약

```
┌─────────────────────────────────────────────┐
│  데이터 영속성 완료 ✅                        │
├─────────────────────────────────────────────┤
│                                             │
│  MLflow (monitoring namespace)              │
│  ├─ Backend: SQLite (mlflow-pvc, 5Gi)      │
│  └─ Artifacts: MinIO S3                    │
│                                             │
│  MinIO (storage namespace)                  │
│  ├─ Storage: minio-pvc (20Gi)              │
│  └─ Buckets: datasets, checkpoints, results│
│                                             │
│  Pod 재시작 시:                              │
│  ✅ MLflow 실험 데이터 유지                  │
│  ✅ MinIO 객체 데이터 유지                   │
│  ✅ 학습 이력 보존                           │
│                                             │
└─────────────────────────────────────────────┘
```

**결론:** 로컬 개발에는 SQLite + PVC가 최적입니다. Production으로 가면 PostgreSQL + R2로 전환하세요.
