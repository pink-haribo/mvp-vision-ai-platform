# Quick Start Guide - Vision AI Training Platform

## 🚀 첫 시작 (Initial Setup)

### 1. Prerequisites 설치

```bash
# Windows
winget install Kubernetes.kind
winget install Kubernetes.kubectl
winget install Helm.Helm

# Verify installations
kind version
kubectl version --client
helm version
```

### 2. Kind Cluster 생성 + Infrastructure 배포

```bash
cd platform/infrastructure

# Create Kind cluster
kind create cluster --config kind-config.yaml

# Deploy all infrastructure with Helm
.\scripts\deploy-helm-all.ps1

# Create MLflow database (one-time)
kubectl exec -n platform postgresql-0 -- env PGPASSWORD=devpass psql -U admin -d postgres -c "CREATE DATABASE mlflow;"

# Deploy MLflow
kubectl apply -f k8s/mlflow/mlflow.yaml
```

### 3. Backend & Frontend 의존성 설치

```bash
# Backend
cd platform/backend
poetry install
poetry run alembic upgrade head  # Initialize database

# Frontend
cd platform/frontend
pnpm install
```

---

## 🔄 재부팅 후 시작 (After Reboot)

재부팅 후에는 **자동 스크립트 하나**로 모든 인프라를 시작할 수 있습니다!

### Windows (PowerShell)

```powershell
cd platform/infrastructure
.\scripts\start-dev-environment.ps1
```

이 스크립트가 자동으로:
1. ✅ Docker Desktop 실행 확인
2. ✅ Kind cluster 상태 확인
3. ✅ 모든 Pod가 Running 상태인지 확인
4. ✅ 서비스 접속 URL 출력

---

## 💻 Backend & Frontend 시작

Infrastructure가 준비되면:

### Terminal 1 - Backend

```bash
cd platform/backend
poetry run uvicorn app.main:app --reload --port 8000
```

**확인**:
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Terminal 2 - Frontend

```bash
cd platform/frontend
pnpm dev
```

**확인**:
- Frontend: http://localhost:3000

---

## 🌐 서비스 접속 URL

| 서비스 | URL | 계정 |
|--------|-----|------|
| **Frontend** | http://localhost:3000 | - |
| **Backend API** | http://localhost:8000 | - |
| **PostgreSQL** | localhost:30543 | admin / devpass |
| **Redis** | localhost:30679 | - |
| **MinIO Console** | http://localhost:30901 | minioadmin / minioadmin |
| **MLflow UI** | http://localhost:30500 | - |
| **Grafana** | http://localhost:30030 | admin / prom-operator |
| **Prometheus** | http://localhost:30090 | - |
| **Temporal UI** | http://localhost:30233 | - |

---

## 🛠️ 유용한 명령어

### 인프라 상태 확인

```bash
# 모든 Pod 상태
kubectl get pods -A

# 특정 namespace
kubectl get pods -n platform
kubectl get pods -n mlflow
kubectl get pods -n observability
kubectl get pods -n temporal

# Helm releases
helm list -A

# 로그 확인
kubectl logs -n platform deployment/postgresql
kubectl logs -n mlflow deployment/mlflow
```

### 서비스 재시작

```bash
# 특정 deployment 재시작
kubectl rollout restart deployment/mlflow -n mlflow

# Pod 강제 재생성
kubectl delete pod <pod-name> -n <namespace>
```

### Database 접속

```bash
# PostgreSQL
kubectl exec -it -n platform postgresql-0 -- env PGPASSWORD=devpass psql -U admin -d platform

# Redis
kubectl exec -it -n platform redis-master-0 -- redis-cli
```

---

## 🔧 문제 해결 (Troubleshooting)

### 1. Kind cluster가 시작되지 않음

```bash
# Cluster 상태 확인
kind get clusters

# Cluster 재생성
kind delete cluster --name platform-dev
kind create cluster --config platform/infrastructure/kind-config.yaml

# Infrastructure 재배포
cd platform/infrastructure
.\scripts\deploy-helm-all.ps1
```

### 2. Pod가 Pending 상태

```bash
# 이벤트 확인
kubectl describe pod <pod-name> -n <namespace>

# 리소스 부족일 경우 - 불필요한 Pod 정리
kubectl delete pod <pod-name> -n <namespace>
```

### 3. MLflow connection error

```bash
# MLflow database 재생성
kubectl exec -n platform postgresql-0 -- env PGPASSWORD=devpass psql -U admin -d postgres -c "DROP DATABASE mlflow;"
kubectl exec -n platform postgresql-0 -- env PGPASSWORD=devpass psql -U admin -d postgres -c "CREATE DATABASE mlflow;"

# MLflow 재시작
kubectl rollout restart deployment/mlflow -n mlflow
```

### 4. Backend database migration 실패

```bash
cd platform/backend

# Migration 상태 확인
poetry run alembic current

# Migration 재실행
poetry run alembic upgrade head

# Migration 초기화 (주의: 데이터 손실)
poetry run alembic downgrade base
poetry run alembic upgrade head
```

### 5. Port already in use

```bash
# Windows에서 포트 사용 확인
netstat -ano | findstr :8000
netstat -ano | findstr :3000

# 프로세스 종료
taskkill /PID <PID> /F
```

---

## 📚 추가 문서

- [Infrastructure README](./infrastructure/README.md) - Helm charts, K8s manifests 상세
- [3-Tier Development Strategy](./docs/development/3_TIER_DEVELOPMENT.md) - 개발 전략
- [CLAUDE.md](../CLAUDE.md) - 전체 프로젝트 가이드

---

## 🎯 일일 개발 루틴

**매일 아침 시작**:
1. Docker Desktop 시작 대기 (자동 시작 설정 권장)
2. `cd platform/infrastructure && .\scripts\start-dev-environment.ps1`
3. Backend 시작: `cd platform/backend && poetry run uvicorn app.main:app --reload`
4. Frontend 시작: `cd platform/frontend && pnpm dev`

**작업 종료**:
- Backend/Frontend는 Ctrl+C로 종료
- Infrastructure는 **그냥 두기** (다음날 재사용)
- PC 종료/재부팅 OK (Docker가 자동으로 처리)

**주말/장기 중단 후**:
- `.\scripts\start-dev-environment.ps1` 실행
- 모든 서비스가 자동으로 재시작됨

---

**Last Updated**: 2025-01-12
