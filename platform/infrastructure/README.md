# Platform Infrastructure

Vision AI Training Platform의 Tier 1 개발 환경 인프라 구성.

## 아키텍처

**3-Tier 개발 전략:**
- **Tier 1 (Current)**: Kind 클러스터 (모든 인프라) + 로컬 서버 (Backend, Frontend, Training subprocess)
- **Tier 2**: Kind 클러스터 (모든 인프라 + Training as K8s Job)
- **Tier 3**: Production Kubernetes (모든 서비스)

## 기술 스택

### Kubernetes 환경
- **Kind**: Local Kubernetes cluster
- **Helm**: Package manager for Kubernetes

### 인프라 서비스 (Helm Charts)
- **PostgreSQL** (bitnami/postgresql): 주 데이터베이스
- **Redis** (bitnami/redis): 캐시 및 메시지 큐
- **MinIO** (minio/minio): S3-compatible 객체 스토리지
- **kube-prometheus-stack**: Prometheus + Grafana + AlertManager
- **Loki** (grafana/loki): 로그 수집
- **Temporal** (temporalio/temporal): Workflow 오케스트레이션

## 디렉토리 구조

```
platform/infrastructure/
├── README.md                           # 이 파일
├── kind-config.yaml                    # Kind cluster 설정
├── scripts/
│   ├── setup-kind-cluster.ps1          # Kind cluster 생성
│   ├── create-namespaces.ps1           # Namespace 생성
│   ├── cleanup-raw-manifests.ps1       # 기존 배포 제거
│   └── deploy-helm-all.ps1             # Helm 전체 배포
├── helm/
│   ├── postgres-values.yaml            # PostgreSQL Helm values
│   ├── redis-values.yaml               # Redis Helm values
│   ├── minio-values.yaml               # MinIO Helm values
│   ├── kube-prometheus-stack-values.yaml  # Observability Helm values
│   └── temporal-values.yaml            # Temporal Helm values
└── k8s/
    ├── platform/
    │   ├── nodeports.yaml              # Platform NodePort services
    │   ├── backend-config.yaml         # Backend ConfigMap
    │   ├── backend-secrets.yaml        # Backend Secrets
    │   ├── backend.yaml                # Backend Deployment (for future)
    │   └── frontend.yaml               # Frontend Deployment (for future)
    ├── observability/
    │   └── nodeports.yaml              # Observability NodePort services
    └── temporal/
        └── nodeports.yaml              # Temporal NodePort services
```

## 빠른 시작

### 1. 사전 요구사항

```powershell
# Docker Desktop 실행 (필수)

# Kind 설치
winget install Kubernetes.kind

# kubectl 설치
winget install Kubernetes.kubectl

# Helm 설치
winget install Helm.Helm
```

### 2. Kind 클러스터 생성

```powershell
cd platform/infrastructure
.\scripts\setup-kind-cluster.ps1
```

### 3. Namespace 생성

```powershell
.\scripts\create-namespaces.ps1
```

### 4. (기존 배포 제거 - 필요시)

```powershell
.\scripts\cleanup-raw-manifests.ps1
```

### 5. Helm으로 모든 서비스 배포

**⚠️ 새 PowerShell 터미널에서 실행 (Helm PATH 적용 후):**

```powershell
cd platform/infrastructure
.\scripts\deploy-helm-all.ps1
```

## 수동 배포 (단계별)

Helm repo 추가:
```powershell
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add temporalio https://go.temporal.io/helm-charts
helm repo add minio https://charts.min.io/
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

각 서비스 배포:
```powershell
# PostgreSQL
helm upgrade --install postgresql bitnami/postgresql `
  --namespace platform --create-namespace `
  --values helm/postgres-values.yaml --wait

# Redis
helm upgrade --install redis bitnami/redis `
  --namespace platform `
  --values helm/redis-values.yaml --wait

# MinIO
helm upgrade --install minio minio/minio `
  --namespace platform `
  --values helm/minio-values.yaml --wait

# kube-prometheus-stack
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack `
  --namespace observability --create-namespace `
  --values helm/kube-prometheus-stack-values.yaml --wait --timeout 10m

# Loki
helm upgrade --install loki grafana/loki `
  --namespace observability `
  --set loki.auth_enabled=false `
  --set loki.commonConfig.replication_factor=1 `
  --set singleBinary.replicas=1 --wait

# Temporal
helm upgrade --install temporal temporalio/temporal `
  --namespace temporal --create-namespace `
  --values helm/temporal-values.yaml --wait --timeout 10m

# NodePort Services
kubectl apply -f k8s/platform/nodeports.yaml
kubectl apply -f k8s/observability/nodeports.yaml
kubectl apply -f k8s/temporal/nodeports.yaml
```

## 접속 정보

### 인프라 서비스

| 서비스 | 내부 DNS | 외부 접속 | 인증 |
|--------|---------|----------|------|
| **PostgreSQL** | `postgresql.platform:5432` | `localhost:30543` | admin / devpass |
| **Redis** | `redis-master.platform:6379` | `localhost:30679` | (인증 없음) |
| **MinIO API** | `minio.platform:9000` | `http://localhost:30900` | minioadmin / minioadmin |
| **MinIO Console** | `minio.platform:9001` | `http://localhost:30901` | minioadmin / minioadmin |

### 모니터링

| 서비스 | 내부 DNS | 외부 접속 | 인증 |
|--------|---------|----------|------|
| **Prometheus** | `prometheus-operated.observability:9090` | `http://localhost:30090` | (없음) |
| **Grafana** | `kube-prometheus-stack-grafana.observability:80` | `http://localhost:30030` | admin / prom-operator |
| **Loki** | `loki.observability:3100` | `http://localhost:30100` | (없음) |

### 오케스트레이션

| 서비스 | 내부 DNS | 외부 접속 | 인증 |
|--------|---------|----------|------|
| **Temporal gRPC** | `temporal-frontend.temporal:7233` | `localhost:30700` | (없음) |
| **Temporal UI** | `temporal-web.temporal:8080` | `http://localhost:30233` | (없음) |

## Helm 관리

### 배포 상태 확인

```powershell
# 모든 Helm releases 확인
helm list --all-namespaces

# 특정 namespace의 releases
helm list -n platform
helm list -n observability
helm list -n temporal
```

### 서비스 업그레이드

```powershell
# values 파일 수정 후
helm upgrade postgresql bitnami/postgresql `
  -n platform `
  --values helm/postgres-values.yaml

# 또는 특정 값만 override
helm upgrade postgresql bitnami/postgresql `
  -n platform `
  --reuse-values `
  --set auth.password=newpassword
```

### 서비스 제거

```powershell
# 특정 release 제거
helm uninstall postgresql -n platform

# 모든 platform services 제거
helm uninstall postgresql redis minio -n platform
helm uninstall kube-prometheus-stack loki -n observability
helm uninstall temporal -n temporal
```

### 트러블슈팅

```powershell
# Release 상태 확인
helm status postgresql -n platform

# Release history
helm history postgresql -n platform

# 이전 버전으로 rollback
helm rollback postgresql 1 -n platform

# Values 확인 (실제 적용된 값)
helm get values postgresql -n platform

# Manifest 확인 (실제 배포된 YAML)
helm get manifest postgresql -n platform
```

## 로컬 개발 워크플로우

1. **Kind 클러스터**: 인프라만 실행
2. **로컬 터미널**: Backend, Frontend 실행
   ```powershell
   # Terminal 1: Backend
   cd platform/backend
   venv\Scripts\activate
   uvicorn app.main:app --reload --port 8000

   # Terminal 2: Frontend
   cd platform/frontend
   npm run dev
   ```

3. **Training**: Subprocess 방식으로 host에서 실행

## 업그레이드 경로

### Tier 1 → Tier 2
- Backend, Frontend를 Docker image로 빌드
- Kind 클러스터에 Deployment 추가
- Training을 K8s Job으로 변경

### Tier 2 → Tier 3
- Production Kubernetes 클러스터로 동일한 Helm charts 배포
- Secrets을 Kubernetes Secrets 또는 Vault로 관리
- Ingress, SSL/TLS 설정
- HA (High Availability) 설정

## 참고 자료

- [Kind Documentation](https://kind.sigs.k8s.io/)
- [Helm Documentation](https://helm.sh/docs/)
- [Bitnami Charts](https://github.com/bitnami/charts)
- [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
- [Temporal Helm Charts](https://github.com/temporalio/helm-charts)

## License

Copyright © 2025 Vision AI Platform Team
