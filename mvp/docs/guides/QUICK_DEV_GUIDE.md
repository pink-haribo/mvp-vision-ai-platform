# ⚡ Quick Development Guide

Training 코드를 수정하면서 빠르게 개발하는 방법 (한 페이지 요약)

## 🎯 핵심 원칙

**❌ 매번 Docker 이미지를 빌드하지 마세요!**

**✅ 로컬에서 Python으로 직접 실행하세요!**

---

## 📊 개발 단계 선택

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  코드 수정 중?                                           │
│  ├─ YES → 로컬 실행 (초 단위) ⚡⚡⚡                    │
│  └─ NO  → 다음 단계로                                   │
│                                                         │
│  K8s 환경 테스트?                                        │
│  ├─ YES → ConfigMap 주입 (분 단위) ⚡⚡                 │
│  └─ NO  → 다음 단계로                                   │
│                                                         │
│  최종 배포 준비?                                         │
│  ├─ YES → 이미지 빌드 (10분+) ⚡                        │
│  └─ NO  → 로컬 실행으로 돌아가기                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 빠른 명령어

### 로컬 실행 (99% 사용)

```powershell
# 한 줄로 실행
.\dev-train-local.ps1

# 파라미터 지정
.\dev-train-local.ps1 -ModelName yolo11s -NumEpochs 20
```

**특징:**
- ⚡ 가장 빠름 (초 단위)
- 🔧 디버깅 쉬움
- 🔄 즉시 반복 테스트
- 📦 빌드 불필요

---

### K8s 테스트 (가끔 사용)

```powershell
# ConfigMap으로 코드 주입
.\dev-train-k8s.ps1 -Watch
```

**특징:**
- ⚡ 빠름 (분 단위)
- 🎯 K8s 환경 테스트
- 📦 빌드 불필요
- 🔍 리소스 제한 확인

---

### 이미지 빌드 (드물게 사용)

```powershell
# 최종 배포 전에만
cd mvp/training/docker
.\build.ps1 -Target ultralytics
```

**특징:**
- ⚡ 느림 (10분+)
- 🎁 배포 가능한 이미지
- 📦 빌드 필요

---

## 💻 일반적인 하루

```powershell
# 아침: 환경 시작 (한 번만)
.\dev-start.ps1 -SkipBuild

# 개발 반복 (여러 번)
vim mvp/training/train.py       # 코드 수정
.\dev-train-local.ps1            # 테스트
# → MLflow 확인: http://localhost:30500

vim mvp/training/train.py       # 또 수정
.\dev-train-local.ps1            # 다시 테스트

# (10-20회 반복...)

# 저녁: 환경 종료
.\dev-stop.ps1
```

---

## 🎨 개발 환경 구조

```
┌──────────────────────────────────────┐
│  Local Machine (당신의 PC)          │
│                                      │
│  train.py (수정 중...)               │
│      ↓                               │
│  python train.py  ← 로컬 실행        │
│      ↓                               │
│      ├→ MLflow (K8s) ← 메트릭        │
│      └→ MinIO (K8s)  ← 모델 저장     │
│                                      │
└──────────────────────────────────────┘
           ↑
           │ (K8s 서비스만 사용)
           │
┌──────────┴───────────────────────────┐
│  Kind Cluster                        │
│                                      │
│  ✓ MLflow     (http://localhost:30500) │
│  ✓ MinIO      (http://localhost:30901) │
│  ✓ Prometheus (http://localhost:30090) │
│  ✓ Grafana    (http://localhost:30030) │
│                                      │
└──────────────────────────────────────┘
```

**핵심:** 코드는 로컬에서 실행하고, MLflow/MinIO만 K8s를 사용!

---

## ⚙️ 환경 변수 (자동 설정)

`dev-train-local.ps1`이 자동으로 설정:

```bash
MLFLOW_TRACKING_URI    = http://localhost:30500
MLFLOW_S3_ENDPOINT_URL = http://localhost:30900
AWS_ACCESS_KEY_ID      = minioadmin
AWS_SECRET_ACCESS_KEY  = minioadmin
JOB_ID                 = local-20251107-143000
```

**Python 코드에서:**
```python
import os
import mlflow

# 환경변수 자동 읽기
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

# 나머지는 평소처럼 코딩
with mlflow.start_run():
    mlflow.log_param("lr", 0.001)
    # ...
```

---

## 🔧 트러블슈팅

### "K8s services not running"

```powershell
# 환경 시작
.\dev-start.ps1 -SkipBuild
```

### "ModuleNotFoundError"

```powershell
# 가상환경 설정 (한 번만)
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision ultralytics mlflow boto3
```

### MLflow에 데이터가 안 보임

```powershell
# Port-forward 확인
kubectl port-forward -n monitoring svc/mlflow 5000:5000

# 브라우저: http://localhost:5000
```

---

## 📚 더 알아보기

- **[DEV_WORKFLOW.md](DEV_WORKFLOW.md)** - 상세 워크플로우 가이드
- **[DEV_SCRIPTS.md](DEV_SCRIPTS.md)** - 모든 스크립트 설명
- **[mvp/k8s/MLFLOW_SETUP.md](mvp/k8s/MLFLOW_SETUP.md)** - MLflow 사용법

---

## ⚡ TL;DR

```powershell
# 환경 시작 (하루에 한 번)
.\dev-start.ps1 -SkipBuild

# 개발 (여러 번 반복)
.\dev-train-local.ps1

# 환경 종료
.\dev-stop.ps1
```

**끝!** 🎉
