# Getting Started - Fast Training Development

빠르게 Training 코드를 개발하고 테스트하는 실전 가이드입니다.

## 🚀 5분 안에 시작하기

### Step 1: 개발 환경 시작

```powershell
# K8s 클러스터 및 서비스 시작 (이미지 빌드 스킵)
.\dev-start.ps1 -SkipBuild

# 완료 확인 (1-2분 소요)
# ✓ MLflow:     http://localhost:30500
# ✓ MinIO:      http://localhost:30901
# ✓ Prometheus: http://localhost:30090
# ✓ Grafana:    http://localhost:30030
```

### Step 2: 로컬 Python 환경 설정 (한 번만)

```powershell
# 가상환경 생성 및 활성화
cd mvp/training
python -m venv venv
.\venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 완료! (3-5분 소요)
```

### Step 3: 첫 번째 Training 실행

```powershell
# 프로젝트 루트로 이동
cd ..\..

# 로컬에서 Training 실행 (샘플 데이터셋 사용)
.\dev-train-local.ps1

# 결과 확인
# → MLflow UI: http://localhost:30500
```

**축하합니다! 🎉** 첫 Training이 완료되었습니다.

---

## 📝 실전 예제: 고양이/개 분류 모델

### 샘플 데이터셋 구조

```
mvp/data/datasets/sample_dataset/
├── train/
│   ├── cats/    (20장)
│   └── dogs/    (20장)
└── val/
    ├── cats/    (5장)
    └── dogs/    (5장)
```

### 예제 1: 기본 Classification Training

```powershell
# YOLO 모델로 이미지 분류 (기본)
.\dev-train-local.ps1 `
    -ModelName yolo11n `
    -NumEpochs 10

# 실행 과정:
# 1. K8s 서비스 연결 확인 (MLflow, MinIO)
# 2. 환경변수 자동 설정
# 3. Python 스크립트 실행
# 4. MLflow에 메트릭 자동 기록
# 5. 결과 확인: http://localhost:30500
```

### 예제 2: 파라미터 조정

```powershell
# 다른 모델, 더 많은 epoch
.\dev-train-local.ps1 `
    -ModelName yolo11s `
    -NumEpochs 20 `
    -Framework ultralytics

# 또는 TIMM 프레임워크 사용
.\dev-train-local.ps1 `
    -ModelName resnet50 `
    -NumEpochs 15 `
    -Framework timm
```

### 예제 3: K8s에서 테스트 (ConfigMap 주입)

```powershell
# 코드를 수정했고, K8s 환경에서 테스트하고 싶을 때
.\dev-train-k8s.ps1 -Watch

# 동작:
# 1. train.py를 ConfigMap으로 생성
# 2. K8s Job 생성 (기존 Docker 이미지 사용)
# 3. ConfigMap을 /code/train.py로 마운트
# 4. 로그 스트리밍

# 장점: Docker 이미지 재빌드 불필요! (분 단위로 테스트)
```

---

## 🔄 일반적인 개발 사이클

### 시나리오: Training 코드 수정하기

```powershell
# 1. train.py 수정
vim mvp/training/train.py
# (예: learning rate 변경, 새로운 metric 추가, 등)

# 2. 즉시 테스트 (초 단위)
.\dev-train-local.ps1

# 3. MLflow에서 결과 확인
start http://localhost:30500

# 4. 다시 수정
vim mvp/training/train.py

# 5. 다시 테스트
.\dev-train-local.ps1

# (10-20회 반복... 매우 빠름!)

# 6. 안정화되면 K8s 테스트
.\dev-train-k8s.ps1 -Watch

# 7. 최종 확인 후 이미지 빌드 (선택사항)
cd mvp/training/docker
.\build.ps1 -Target ultralytics
```

### 시나리오: 새로운 Adapter 추가

```powershell
# 1. Adapter 파일 생성
vim mvp/training/adapters/my_new_adapter.py

# 2. Registry에 등록
vim mvp/training/adapters/__init__.py

# 3. 로컬에서 즉시 테스트
.\dev-train-local.ps1 -Framework my_new_framework -ModelName my_model

# 4. 반복 개발
# - 코드 수정
# - 로컬 실행
# - 결과 확인
# - 반복...

# 5. K8s에서 통합 테스트
.\dev-train-k8s.ps1 -Watch
```

---

## 🎯 개발 효율성 비교

| 방법 | 소요 시간 | 사용 시기 |
|------|-----------|-----------|
| **로컬 실행** | **5-30초** | **개발 중 (99%)** |
| ConfigMap 주입 | 1-3분 | K8s 환경 테스트 |
| Docker 빌드 | 10-15분 | 최종 배포 전 |

**핵심: 로컬 실행으로 99%의 개발을 완료하세요!**

---

## 🛠️ 유용한 명령어

### 환경 상태 확인

```powershell
# 한 번 확인
.\dev-status.ps1

# 지속적으로 모니터링
.\dev-status.ps1 -Watch
```

### MLflow 결과 확인

```powershell
# MLflow UI 열기
start http://localhost:30500

# 또는 CLI로 확인
kubectl port-forward -n monitoring svc/mlflow 5000:5000
```

### 로그 확인 (K8s Job)

```powershell
# Job 목록
kubectl get jobs -n training

# Pod 로그
kubectl logs -n training -l job-id=<JOB_ID> -f
```

### 데이터 확인 (MinIO)

```powershell
# MinIO Console 열기
start http://localhost:30901

# 로그인: minioadmin / minioadmin
# Buckets:
#   - training-datasets
#   - training-checkpoints
#   - training-results
```

---

## 🐛 트러블슈팅

### "K8s services not running"

```powershell
# 환경 시작
.\dev-start.ps1 -SkipBuild

# 또는 클러스터 재생성 (데이터 초기화)
.\dev-stop.ps1 -DeleteCluster
.\dev-start.ps1 -Fresh
```

### "ModuleNotFoundError: No module named 'ultralytics'"

```powershell
# 가상환경 활성화 확인
cd mvp/training
.\venv\Scripts\activate

# 의존성 재설치
pip install -r requirements.txt
```

### "MLflow connection failed"

```powershell
# Port-forward 설정
kubectl port-forward -n monitoring svc/mlflow 30500:5000

# 또는 환경 재시작
.\dev-stop.ps1
.\dev-start.ps1 -SkipBuild
```

### "Training too slow"

```powershell
# CPU only 모드로 테스트 (빠름)
.\dev-train-local.ps1 -ModelName yolo11n -NumEpochs 2

# 또는 더 작은 모델 사용
.\dev-train-local.ps1 -ModelName yolo11n -NumEpochs 5
```

---

## 📚 다음 단계

### 추천 학습 순서

1. **[QUICK_DEV_GUIDE.md](QUICK_DEV_GUIDE.md)** - 한 페이지 요약 (즉시 참조)
2. **[DEV_WORKFLOW.md](DEV_WORKFLOW.md)** - 워크플로우 상세 가이드
3. **[DEV_SCRIPTS.md](DEV_SCRIPTS.md)** - 모든 스크립트 옵션
4. **[mvp/k8s/MLFLOW_SETUP.md](mvp/k8s/MLFLOW_SETUP.md)** - MLflow 사용법
5. **[mvp/k8s/DOCKER_VS_K8S.md](mvp/k8s/DOCKER_VS_K8S.md)** - 환경 비교

### 더 알아보기

- **Adapter 추가**: `mvp/training/adapters/` 참고
- **새로운 Task Type**: `platform_sdk/config.py` 참고
- **Custom Dataset**: `mvp/data/datasets/` 구조 참고
- **Production 배포**: `mvp/k8s/` 매니페스트 참고

---

## ⚡ TL;DR

```powershell
# 처음 시작 (한 번만)
.\dev-start.ps1 -SkipBuild
cd mvp/training && python -m venv venv && .\venv\Scripts\activate && pip install -r requirements.txt && cd ..\..

# 개발 (매일)
.\dev-train-local.ps1                # 테스트
vim mvp/training/train.py            # 수정
.\dev-train-local.ps1                # 다시 테스트
# (반복...)

# 종료
.\dev-stop.ps1
```

**끝!** 🎉
