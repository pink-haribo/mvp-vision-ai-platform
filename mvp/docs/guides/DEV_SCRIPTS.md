# Development Environment Scripts

개발 환경을 쉽게 시작/종료/관리하기 위한 스크립트 모음입니다.

## 스크립트 목록

### 1. `dev-start.ps1` - 환경 시작

전체 개발 환경을 자동으로 설정하고 시작합니다.

**기본 사용:**
```powershell
.\dev-start.ps1
```

**옵션:**
```powershell
# Docker 이미지 빌드 스킵 (이미 빌드된 경우)
.\dev-start.ps1 -SkipBuild

# 기존 클러스터 삭제하고 새로 시작
.\dev-start.ps1 -Fresh

# 두 옵션 함께 사용
.\dev-start.ps1 -Fresh -SkipBuild
```

**수행 작업:**
1. ✅ 필수 도구 확인 (kind, kubectl, docker)
2. ✅ Docker 실행 확인
3. ✅ Kind 클러스터 생성 (training-dev)
4. ✅ Docker 이미지 빌드 (base, ultralytics, timm)
5. ✅ 이미지를 Kind 클러스터에 로드
6. ✅ K8s 리소스 배포:
   - Training namespace, secrets, configmaps
   - MinIO (storage namespace)
   - MLflow (monitoring namespace)
   - Prometheus + Grafana (monitoring namespace)
7. ✅ 서비스 준비 대기
8. ✅ MinIO 버킷 생성

**완료 후 접근 가능:**
- MLflow UI: http://localhost:30500
- Prometheus: http://localhost:30090
- Grafana: http://localhost:30030 (admin/admin)
- MinIO Console: http://localhost:30901 (minioadmin/minioadmin)
- MinIO API: http://localhost:30900

**소요 시간:**
- 처음 실행: ~10-15분 (Docker 이미지 빌드 포함)
- 두 번째 실행: ~2-3분 (이미지 빌드 스킵)

---

### 2. `dev-stop.ps1` - 환경 종료

개발 환경을 중지하거나 완전히 삭제합니다.

**기본 사용 (중지):**
```powershell
.\dev-stop.ps1
```
- Kind 클러스터 Docker 컨테이너를 중지합니다
- **데이터는 유지됩니다** (PVC 보존)
- 재시작 가능: `docker start training-dev-control-plane`

**완전 삭제:**
```powershell
.\dev-stop.ps1 -DeleteCluster
```
- Kind 클러스터를 완전히 삭제합니다
- ⚠️ **모든 데이터가 삭제됩니다** (PVC 포함)
- 확인 프롬프트가 표시됩니다

**사용 시나리오:**

**잠시 중단 (나중에 재개):**
```powershell
# 중지
.\dev-stop.ps1

# 나중에 재시작
docker start training-dev-control-plane
# 또는
.\dev-start.ps1 -SkipBuild
```

**완전히 정리하고 처음부터:**
```powershell
.\dev-stop.ps1 -DeleteCluster
.\dev-start.ps1
```

---

### 3. `dev-status.ps1` - 상태 확인

현재 개발 환경의 상태를 확인합니다.

**기본 사용:**
```powershell
.\dev-status.ps1
```

**실시간 모니터링 (5초마다 갱신):**
```powershell
.\dev-status.ps1 -Watch
```

**표시 정보:**
- ✅ Cluster 상태 (실행 중/중지)
- ✅ 각 서비스 상태:
  - MinIO (Storage)
  - MLflow (Monitoring)
  - Prometheus (Monitoring)
  - Grafana (Monitoring)
- ✅ Training Job 현황
- ✅ Resource 사용량
- ✅ Docker 이미지 로드 상태
- ✅ Quick commands

**출력 예시:**
```
========================================
Vision AI Training Platform
Development Environment Status
========================================

Time: 2025-11-07 14:30:00

✓ Cluster 'training-dev' is running

Cluster Information:
  Context:          kind-training-dev
  Kubernetes:       Server Version: v1.34.0

Storage (namespace: storage):
  MinIO:            ✓ Running
    Console:        http://localhost:30901 (minioadmin/minioadmin)
    API:            http://localhost:30900
    PVC:            Bound (20Gi)

Monitoring (namespace: monitoring):
  MLflow:           ✓ Running
    UI:             http://localhost:30500
    PVC:            Bound (5Gi)
  Prometheus:       ✓ Running
    UI:             http://localhost:30090
  Grafana:          ✓ Running
    UI:             http://localhost:30030 (admin/admin)

Training (namespace: training):
  Active Jobs:      0
  Completed Jobs:   0
```

---

## 일반적인 워크플로우

### 첫 번째 실행 (프로젝트 클론 후)

```powershell
# 1. 전체 환경 설정 (처음)
.\dev-start.ps1

# 2. 상태 확인
.\dev-status.ps1

# 3. MLflow UI 열기
start http://localhost:30500

# 4. 작업 완료 후 중지
.\dev-stop.ps1
```

### 매일 작업 시작

```powershell
# 1. 환경 시작 (이미지 빌드 스킵)
.\dev-start.ps1 -SkipBuild

# 2. 상태 확인
.\dev-status.ps1

# 3. 개발 작업...

# 4. 작업 완료 후 중지
.\dev-stop.ps1
```

### 문제 해결 (처음부터 다시)

```powershell
# 1. 기존 환경 완전 삭제
.\dev-stop.ps1 -DeleteCluster

# 2. 새로 시작
.\dev-start.ps1 -Fresh
```

### 이미지 재빌드 필요 시

```powershell
# 1. 클러스터는 유지하고 이미지만 재빌드
.\dev-start.ps1

# 또는 완전히 새로 시작
.\dev-start.ps1 -Fresh
```

---

## 트러블슈팅

### "kind: command not found"

```powershell
# Kind 설치
winget install -e --id Kubernetes.kind
```

### "kubectl: command not found"

```powershell
# kubectl 설치
winget install -e --id Kubernetes.kubectl
```

### "Docker is not running"

1. Docker Desktop 실행
2. 시작될 때까지 대기 (트레이 아이콘 확인)
3. 다시 시도

### 서비스가 시작되지 않음

```powershell
# 1. 상태 확인
.\dev-status.ps1

# 2. 로그 확인
kubectl logs -n monitoring deployment/mlflow
kubectl logs -n storage deployment/minio

# 3. Pod 상태 확인
kubectl get pods --all-namespaces
kubectl describe pod -n monitoring <pod-name>

# 4. 재시작
.\dev-stop.ps1 -DeleteCluster
.\dev-start.ps1
```

### Port가 이미 사용 중

다른 프로그램이 포트를 사용 중일 수 있습니다:

```powershell
# 포트 사용 확인
netstat -ano | findstr :30500
netstat -ano | findstr :30090
netstat -ano | findstr :30030
netstat -ano | findstr :30900

# 프로세스 종료 (PID 확인 후)
taskkill /PID <PID> /F
```

### 이미지 빌드 실패

```powershell
# Docker 메모리 증가 (Docker Desktop 설정)
# Settings > Resources > Memory > 8GB+

# 다시 시도
.\dev-start.ps1 -Fresh
```

---

## 고급 사용법

### 특정 서비스만 재시작

```powershell
# MLflow 재시작
kubectl rollout restart deployment/mlflow -n monitoring

# MinIO 재시작
kubectl rollout restart deployment/minio -n storage

# 상태 확인
kubectl rollout status deployment/mlflow -n monitoring
```

### 로그 실시간 확인

```powershell
# MLflow 로그
kubectl logs -n monitoring deployment/mlflow -f

# MinIO 로그
kubectl logs -n storage deployment/minio -f

# 여러 서비스 동시에 (새 터미널 필요)
kubectl logs -n monitoring deployment/mlflow -f
kubectl logs -n monitoring deployment/prometheus -f
```

### Port-forward 사용

NodePort 대신 port-forward를 사용하려면:

```powershell
# MLflow (5000 포트)
kubectl port-forward -n monitoring svc/mlflow 5000:5000

# MinIO Console (9001 포트)
kubectl port-forward -n storage svc/minio 9001:9001

# 백그라운드로 실행
Start-Job -ScriptBlock {
    kubectl port-forward -n monitoring svc/mlflow 5000:5000
}
```

### 리소스 사용량 모니터링

```powershell
# Node 리소스 사용량
kubectl top nodes

# Pod 리소스 사용량
kubectl top pods --all-namespaces

# 특정 namespace
kubectl top pods -n monitoring
```

---

## 환경 정리

### 디스크 공간 확보

```powershell
# 1. 사용하지 않는 Docker 이미지 삭제
docker image prune -a

# 2. 사용하지 않는 Docker 컨테이너 삭제
docker container prune

# 3. 사용하지 않는 Docker 볼륨 삭제
docker volume prune

# 4. 전체 정리
docker system prune -a --volumes
```

### Kind 클러스터 완전 삭제

```powershell
# 스크립트 사용
.\dev-stop.ps1 -DeleteCluster

# 또는 직접 삭제
kind delete cluster --name training-dev
```

---

## 다음 단계

환경 설정이 완료되었으면:

1. **샘플 Training Job 실행**
   - `mvp/k8s/examples/` 참고

2. **MLflow 실험 추적**
   - http://localhost:30500

3. **Grafana 대시보드**
   - http://localhost:30030

4. **MinIO 데이터 관리**
   - http://localhost:30901

자세한 내용은 각 문서를 참고하세요:
- `mvp/k8s/MLFLOW_SETUP.md`
- `mvp/k8s/MINIO_SETUP.md`
- `mvp/k8s/MONITORING_INTEGRATION.md`
