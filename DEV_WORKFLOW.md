# 개발 워크플로우 가이드

Training 코드를 수정하면서 개발할 때, **Docker 이미지를 매번 빌드하지 않고** 빠르게 반복 개발하는 방법입니다.

## 🎯 개발 단계별 접근

### 1단계: 로컬 개발 (가장 빠름 ⚡)

**언제 사용:**
- 코드를 자주 수정하면서 테스트
- 빠른 반복 개발
- 디버깅

**방법:**
```powershell
# 로컬 Python으로 직접 실행
# MLflow, MinIO는 K8s 서비스 사용
.\dev-train-local.ps1 -Script mvp/training/train.py
```

**장점:**
- ✅ 가장 빠름 (초 단위)
- ✅ 코드 수정 즉시 테스트
- ✅ 로컬 디버거 사용 가능
- ✅ 이미지 빌드 불필요

**단점:**
- ⚠️ 로컬 환경 필요 (Python, 라이브러리)
- ⚠️ K8s 환경과 약간 다를 수 있음

---

### 2단계: K8s 테스트 (ConfigMap 주입)

**언제 사용:**
- K8s 환경에서 테스트
- 로컬 환경 설정 없이 실행
- 실제 환경과 유사하게 테스트

**방법:**
```powershell
# ConfigMap으로 코드 주입 (빌드 없이 K8s 실행)
.\dev-train-k8s.ps1 -Script mvp/training/train.py -Watch
```

**장점:**
- ✅ 이미지 빌드 불필요
- ✅ 코드 수정 후 바로 테스트 (분 단위)
- ✅ K8s 환경에서 실행
- ✅ 리소스 제한 테스트 가능

**단점:**
- ⚠️ 로컬보다 느림 (Pod 시작 시간)
- ⚠️ 로그 확인이 약간 번거로움

---

### 3단계: 이미지 빌드 (최종 배포)

**언제 사용:**
- 코드가 안정화됨
- 실제 배포 전 최종 테스트
- 다른 사람과 공유

**방법:**
```powershell
# Docker 이미지 빌드
cd mvp/training/docker
.\build.ps1 -Target ultralytics

# Kind 클러스터에 로드
kind load docker-image ghcr.io/myorg/trainer-ultralytics:v1.0 --name training-dev

# K8s Job으로 실행 (빌드된 이미지 사용)
kubectl apply -f training-job.yaml
```

**장점:**
- ✅ 실제 배포와 동일
- ✅ 이미지 공유 가능
- ✅ Production 환경과 동일

**단점:**
- ⚠️ 빌드 시간 소요 (5-10분)
- ⚠️ 수정 시마다 재빌드 필요

---

## 📋 권장 워크플로우

### 일반적인 개발 사이클

```
┌─────────────────────────────────────────────┐
│  1. 로컬 개발 (빠른 반복)                    │
│     ├─ 코드 작성/수정                       │
│     ├─ dev-train-local.ps1 실행             │
│     ├─ 결과 확인 (MLflow)                   │
│     └─ 반복... (초 단위)                    │
│                                             │
│  2. K8s 테스트 (통합 확인)                   │
│     ├─ 코드 안정화됨                        │
│     ├─ dev-train-k8s.ps1 실행               │
│     ├─ K8s 환경에서 동작 확인                │
│     └─ 리소스, 로그 등 검증                  │
│                                             │
│  3. 이미지 빌드 (배포 준비)                  │
│     ├─ 최종 코드 확정                       │
│     ├─ Docker 이미지 빌드                   │
│     ├─ 통합 테스트                          │
│     └─ 배포                                 │
└─────────────────────────────────────────────┘
```

### 구체적인 예시

**기능 개발 시작:**
```powershell
# 1. 환경 시작 (한 번만)
.\dev-start.ps1 -SkipBuild

# 2. 로컬에서 개발
# train.py 수정...
.\dev-train-local.ps1

# 3. 수정, 테스트 반복
# train.py 수정...
.\dev-train-local.ps1

# (반복 10-20회...)
```

**기능 완성 후:**
```powershell
# K8s에서 통합 테스트
.\dev-train-k8s.ps1 -Watch

# 문제 없으면 이미지 빌드 (선택사항)
cd mvp/training/docker
.\build.ps1 -Target ultralytics
```

---

## 🛠️ 스크립트 상세 사용법

### `dev-train-local.ps1` - 로컬 실행

**기본 사용:**
```powershell
.\dev-train-local.ps1
```

**파라미터:**
```powershell
# 다른 스크립트 실행
.\dev-train-local.ps1 -Script mvp/training/custom_train.py

# 모델 지정
.\dev-train-local.ps1 -ModelName yolo11s -NumEpochs 20

# Framework 지정
.\dev-train-local.ps1 -Framework timm -ModelName resnet50
```

**환경변수 자동 설정:**
```
MLFLOW_TRACKING_URI    = http://localhost:30500
MLFLOW_S3_ENDPOINT_URL = http://localhost:30900
AWS_ACCESS_KEY_ID      = minioadmin
AWS_SECRET_ACCESS_KEY  = minioadmin
JOB_ID                 = local-20251107-143000
MODEL_NAME             = yolo11n
FRAMEWORK              = ultralytics
NUM_EPOCHS             = 10
```

**Python 코드에서 사용:**
```python
import os
import mlflow

# 환경변수 자동 읽기
job_id = os.getenv('JOB_ID')
model_name = os.getenv('MODEL_NAME')
num_epochs = int(os.getenv('NUM_EPOCHS', 10))

# MLflow는 MLFLOW_TRACKING_URI를 자동으로 읽음
mlflow.set_experiment("my-experiment")

with mlflow.start_run(run_name=job_id):
    # Training...
    pass
```

---

### `dev-train-k8s.ps1` - K8s 실행 (ConfigMap)

**기본 사용:**
```powershell
# 로그 스트리밍
.\dev-train-k8s.ps1 -Watch

# 백그라운드 실행
.\dev-train-k8s.ps1
```

**파라미터:**
```powershell
# 다른 이미지 사용
.\dev-train-k8s.ps1 -Image ghcr.io/myorg/trainer-timm:v1.0

# 파라미터 지정
.\dev-train-k8s.ps1 -ModelName yolo11m -NumEpochs 50
```

**동작 원리:**
1. `train.py`를 ConfigMap으로 생성
2. K8s Job 생성 (기존 이미지 사용)
3. ConfigMap을 `/code/train.py`로 마운트
4. `python /code/train.py` 실행

**장점:**
- 이미지 내부의 `train.py`를 덮어씀
- 이미지 재빌드 불필요
- 코드만 주입

---

## 💡 실전 팁

### Tip 1: 로컬 개발 시 가상환경 사용

```powershell
# Python 가상환경 생성 (한 번만)
python -m venv venv

# 활성화
.\venv\Scripts\activate

# 의존성 설치 (Docker 이미지와 동일하게)
pip install torch torchvision ultralytics mlflow boto3

# 이후 개발
.\dev-train-local.ps1
```

### Tip 2: 빠른 디버깅

```python
# train.py
import os

# 로컬 개발 시에만 디버그 모드
if os.getenv('JOB_ID', '').startswith('local-'):
    import pdb
    pdb.set_trace()  # 브레이크포인트

# 또는
if 'local' in os.getenv('JOB_ID', ''):
    NUM_EPOCHS = 2  # 빠른 테스트
else:
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 100))
```

### Tip 3: MLflow Autolog 활용

```python
import mlflow.pytorch

# 자동 로깅 (파라미터, 메트릭 자동 기록)
mlflow.pytorch.autolog()

# Training...
# (MLflow가 자동으로 모든 것을 기록)
```

### Tip 4: 코드 수정 후 빠른 확인

```powershell
# 터미널 1: 로컬 실행
.\dev-train-local.ps1

# 터미널 2: MLflow UI 바로 열기
start http://localhost:30500

# 터미널 3: 상태 모니터링
.\dev-status.ps1 -Watch
```

---

## 🔍 비교표

| 방법 | 속도 | K8s 환경 | 디버깅 | 빌드 필요 | 사용 시기 |
|------|------|----------|--------|-----------|-----------|
| **로컬 실행** | ⚡⚡⚡ (초) | ❌ | ✅ 쉬움 | ❌ | 개발 중 |
| **ConfigMap 주입** | ⚡⚡ (분) | ✅ | ⚠️ 보통 | ❌ | 통합 테스트 |
| **이미지 빌드** | ⚡ (10분+) | ✅ | ❌ 어려움 | ✅ | 최종 배포 |

---

## 🚫 피해야 할 것

**❌ 매번 이미지 빌드하지 말 것:**
```powershell
# 잘못된 워크플로우
# 코드 수정
vim train.py

# 이미지 빌드 (5-10분 소요) - 비효율!
.\build.ps1

# 테스트
kubectl apply -f job.yaml

# 다시 수정, 다시 빌드... (시간 낭비)
```

**✅ 올바른 워크플로우:**
```powershell
# 코드 수정 (반복)
vim train.py
.\dev-train-local.ps1  # 초 단위 테스트

vim train.py
.\dev-train-local.ps1  # 즉시 확인

# 안정화됨 → K8s 테스트
.\dev-train-k8s.ps1

# 완성 → 이미지 빌드 (최종 1회)
.\build.ps1
```

---

## 📚 관련 문서

- **`DEV_SCRIPTS.md`** - 개발 환경 스크립트
- **`mvp/k8s/DOCKER_VS_K8S.md`** - Docker vs K8s 비교
- **`mvp/k8s/MLFLOW_SETUP.md`** - MLflow 사용법

---

## 다음 단계

개발 워크플로우를 이해했다면:

1. **로컬 개발 환경 설정**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r mvp/training/requirements.txt
   ```

2. **첫 번째 Training 실행**
   ```powershell
   .\dev-train-local.ps1
   ```

3. **MLflow에서 결과 확인**
   ```
   http://localhost:30500
   ```

4. **코드 수정 및 반복**
   - `train.py` 수정
   - `.\dev-train-local.ps1` 실행
   - 결과 확인
   - 반복...
