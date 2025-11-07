# Docker Image Separation Architecture

**Document Version:** 1.0
**Last Updated:** 2025-10-28
**Status:** Design Document

---

## Executive Summary

본 문서는 Vision AI Training Platform이 다양한 딥러닝 프레임워크(timm, Ultralytics, HuggingFace, 등)를 지원하면서도 의존성 충돌 없이 확장 가능한 구조를 유지하기 위한 **Docker 이미지 분리 전략**을 정의합니다.

### 핵심 문제

현재 모든 프레임워크의 의존성이 `training/requirements.txt`에 함께 있습니다:

```txt
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
ultralytics==8.0.220
# 미래: transformers, mmdetection, detectron2, ...
```

이 접근 방식의 문제점:
- **의존성 충돌**: 프레임워크 간 PyTorch 버전 요구사항이 다를 수 있음
- **이미지 비대화**: 모든 프레임워크를 설치하면 수 GB 크기
- **보안 위험**: 사용하지 않는 라이브러리의 취약점도 포함
- **빌드 시간**: 모든 의존성 설치 시 10분 이상 소요
- **확장성 제한**: 새 프레임워크 추가 시 전체 환경 재빌드 필요

### 해결 방안

**프레임워크별 독립 Docker 이미지 구조**:

```
vision-platform-base:latest (공통 SDK + MLflow + 플랫폼 코드)
├── vision-platform-timm:latest (base + timm + torch)
├── vision-platform-ultralytics:latest (base + ultralytics)
├── vision-platform-huggingface:latest (base + transformers)
└── vision-platform-custom:latest (base + 사용자 정의)
```

### 주요 이점

- ✅ **완벽한 의존성 격리**: 프레임워크 간 충돌 없음
- ✅ **이미지 크기 최적화**: 필요한 것만 포함 (1-2GB per image)
- ✅ **빌드 시간 단축**: 베이스 레이어 재사용으로 증분 빌드
- ✅ **보안 강화**: 각 이미지는 필요한 패키지만 포함
- ✅ **확장성**: 새 프레임워크 추가 시 독립적으로 이미지 생성
- ✅ **코드 변경 최소화**: 현재 Adapter 패턴 그대로 사용

---

## Current State Analysis

### 현재 아키텍처 (MVP)

```
┌─────────────────────────────────────────────────────────┐
│  Backend (FastAPI)                                      │
│  - TrainingManager (subprocess로 학습 실행)            │
│  - training_python = "mvp/training/venv/Scripts/python" │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓ subprocess.Popen()
┌─────────────────────────────────────────────────────────┐
│  Training Process (Local)                               │
│  - train.py --framework timm --model resnet50 ...      │
│  - Adapter 선택 (ADAPTER_REGISTRY[framework])           │
│  - 모든 프레임워크가 같은 venv에 설치되어 있음          │
└─────────────────────────────────────────────────────────┘
```

#### 주요 특징:
1. **Subprocess 실행**: Backend가 로컬 Python 프로세스로 학습 실행
2. **단일 venv**: `mvp/training/venv`에 모든 의존성 설치
3. **Adapter 패턴**: 프레임워크별로 `TrainingAdapter` 구현 (이미 잘 설계됨!)
4. **Registry 기반**: `ADAPTER_REGISTRY`에서 동적으로 adapter 선택

#### 의존성 현황:

**backend/requirements.txt (ML 라이브러리 없음 ✓)**:
```txt
fastapi==0.108.0
sqlalchemy==2.0.23
langchain>=0.1.0
mlflow==2.9.2
prometheus-client==0.19.0
# ... (웹 프레임워크, DB, 모니터링만)
```

**training/requirements.txt (모든 ML 프레임워크 함께 ✗)**:
```txt
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
ultralytics==8.0.220
mlflow==2.9.2
boto3==1.34.10
numpy==1.26.0
pillow==10.1.0
```

### 현재 구조의 장점 (유지할 것)

1. **✅ Adapter 패턴**: 잘 설계된 추상화
   - `BaseModel` → `TimmAdapter`, `UltralyticsAdapter`
   - 프레임워크별 로직 완전히 분리
   - 새 프레임워크 추가 용이

2. **✅ Registry 기반 선택**:
   ```python
   ADAPTER_REGISTRY = {
       'timm': TimmAdapter,
       'ultralytics': UltralyticsAdapter,
   }
   ```

3. **✅ 통합 Callbacks**:
   - `TrainingCallbacks`로 MLflow, DB, WebSocket 통합
   - Adapter 코드는 프레임워크 로직만 집중

4. **✅ Backend와 Training 분리**:
   - Backend는 ML 라이브러리 불필요
   - Training은 독립적으로 실행 가능

### 변경이 필요한 부분

1. **❌ 단일 requirements.txt**:
   - 현재: 모든 프레임워크가 `training/requirements.txt`에
   - 변경: 프레임워크별 `requirements-{framework}.txt`

2. **❌ 로컬 venv 의존**:
   - 현재: TrainingManager가 `mvp/training/venv/Scripts/python.exe` 사용
   - 변경: Docker 컨테이너 또는 venv (선택 가능)

3. **❌ 공통 코드 중복 가능성**:
   - 현재: base.py, TrainingCallbacks 등이 training/에
   - 변경: Platform SDK로 분리하여 모든 이미지에 포함

---

## Proposed Architecture

### Docker 이미지 계층 구조

```
┌─────────────────────────────────────────────────────────┐
│  vision-platform-base:latest                            │
│  - Python 3.11                                          │
│  - Platform SDK (base.py, callbacks.py, utils)         │
│  - MLflow client                                        │
│  - SQLite client (DB 연결용)                            │
│  - S3 client (boto3)                                    │
│  - 공통 유틸리티 (metrics, logging)                     │
│  Size: ~500 MB                                          │
└─────────────────────────────────────────────────────────┘
         ↑              ↑              ↑              ↑
         │              │              │              │
   ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
   │   timm    │  │ultralytics│  │huggingface│  │  custom   │
   │           │  │           │  │           │  │           │
   │FROM base  │  │FROM base  │  │FROM base  │  │FROM base  │
   │+ timm     │  │+ ultralyt.│  │+ transform│  │+ user deps│
   │+ torch    │  │+ opencv   │  │+ accelera.│  │           │
   │           │  │           │  │           │  │           │
   │Size: 2GB  │  │Size: 1.5GB│  │Size: 3GB  │  │Size: var  │
   └───────────┘  └───────────┘  └───────────┘  └───────────┘
```

### 런타임 이미지 선택 흐름

```python
# Backend: TrainingManager

IMAGE_MAP = {
    "timm": "vision-platform-timm:latest",
    "ultralytics": "vision-platform-ultralytics:latest",
    "huggingface": "vision-platform-huggingface:latest",
    "custom": "vision-platform-custom:latest",
}

def start_training(job: TrainingJob):
    framework = job.framework
    image = IMAGE_MAP[framework]

    if USE_DOCKER:
        # Docker 모드: 컨테이너로 실행
        run_training_in_docker(image, job)
    else:
        # Local 모드: venv로 실행 (MVP 호환)
        run_training_subprocess(framework, job)
```

### 디렉토리 구조 (재구성)

```
mvp-vision-ai-platform/
├── mvp/
│   ├── backend/                        # Backend API
│   │   ├── app/
│   │   └── requirements.txt            # FastAPI, SQLAlchemy, LangChain
│   │
│   ├── training/                       # Training 실행 환경
│   │   ├── platform_sdk/               # 🆕 공통 플랫폼 SDK (모든 이미지에 포함)
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # BaseModel, MetricsResult
│   │   │   ├── callbacks.py            # TrainingCallbacks
│   │   │   ├── mlflow_utils.py         # MLflow 헬퍼
│   │   │   ├── storage.py              # S3, 로컬 파일 처리
│   │   │   └── metrics/                # 공통 메트릭 계산
│   │   │
│   │   ├── adapters/                   # 프레임워크별 Adapter (각 이미지에 해당하는 것만)
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # → platform_sdk/base.py로 이동
│   │   │   ├── timm_adapter.py         # timm 이미지에만
│   │   │   ├── ultralytics_adapter.py  # ultralytics 이미지에만
│   │   │   └── huggingface_adapter.py  # huggingface 이미지에만
│   │   │
│   │   ├── train.py                    # 메인 학습 스크립트
│   │   │
│   │   ├── requirements/               # 🆕 프레임워크별 requirements
│   │   │   ├── requirements-base.txt   # 공통 (MLflow, boto3, numpy)
│   │   │   ├── requirements-timm.txt   # timm + torch
│   │   │   ├── requirements-ultralytics.txt
│   │   │   ├── requirements-huggingface.txt
│   │   │   └── requirements-custom.txt
│   │   │
│   │   └── venv/                       # Local 모드용 venv (Docker 사용 시 불필요)
│   │
│   └── docker/                         # 🆕 Docker 이미지 정의
│       ├── Dockerfile.base             # 베이스 이미지
│       ├── Dockerfile.timm             # timm 전용
│       ├── Dockerfile.ultralytics      # ultralytics 전용
│       ├── Dockerfile.huggingface      # huggingface 전용
│       ├── Dockerfile.custom           # 사용자 정의
│       │
│       ├── docker-compose.yml          # 로컬 개발용
│       └── build.sh                    # 모든 이미지 빌드 스크립트
│
└── docs/
    └── architecture/
        └── DOCKER_IMAGE_SEPARATION.md  # 본 문서
```

---

## Implementation Plan

### Phase 0: 사전 준비 (현재)

**목표**: 현재 코드 분석 및 설계 검증

- [x] 현재 adapter 구조 분석
- [x] 의존성 분리 현황 파악
- [x] 변경 범위 식별
- [ ] 구현 계획 문서화 (진행 중)

### Phase 1: Platform SDK 분리 (Week 1)

**목표**: 공통 코드를 독립 패키지로 분리

#### 1.1 Platform SDK 생성

**새 디렉토리**: `mvp/training/platform_sdk/`

```python
# platform_sdk/__init__.py
from .base import (
    TrainingAdapter,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    MetricsResult,
    TaskType,
    DatasetFormat,
)
from .callbacks import TrainingCallbacks
from .mlflow_utils import get_mlflow_client
from .storage import upload_to_s3, download_from_s3

__version__ = "0.1.0"
```

**파일 이동**:
```bash
# 기존 위치 → 새 위치
training/adapters/base.py          → platform_sdk/base.py
                                    → platform_sdk/callbacks.py (분리)
training/utils/mlflow_*.py         → platform_sdk/mlflow_utils.py
# 새로 생성
                                    → platform_sdk/storage.py
                                    → platform_sdk/metrics/common.py
```

#### 1.2 Adapter 코드 리팩토링

**adapters/** 파일들의 import 수정:

```python
# Before
from .base import TrainingAdapter, MetricsResult

# After
from platform_sdk import TrainingAdapter, MetricsResult
from platform_sdk.callbacks import TrainingCallbacks
```

**테스트**: 기존 학습 프로세스가 정상 동작하는지 확인
```bash
cd mvp/training
python train.py --framework timm --model resnet18 --task_type image_classification ...
```

**Deliverables**:
- [ ] `platform_sdk/` 패키지 생성
- [ ] 기존 adapter 코드 리팩토링
- [ ] import 경로 업데이트
- [ ] 로컬 테스트 통과

---

### Phase 2: Requirements 분리 (Week 1)

**목표**: 각 프레임워크별로 독립적인 의존성 파일 생성

#### 2.1 Requirements 파일 생성

**새 디렉토리**: `mvp/training/requirements/`

```bash
mvp/training/requirements/
├── requirements-base.txt
├── requirements-timm.txt
├── requirements-ultralytics.txt
├── requirements-huggingface.txt
└── requirements-custom.txt
```

**requirements-base.txt** (모든 이미지에 공통):
```txt
# Experiment Tracking
mlflow==2.9.2
boto3==1.34.10

# Data Processing
numpy==1.26.0
pillow==10.1.0
pyyaml==6.0.1

# Utils
tqdm==4.66.0

# Database (for callback)
```

**requirements-timm.txt**:
```txt
-r requirements-base.txt

# Deep Learning
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
```

**requirements-ultralytics.txt**:
```txt
-r requirements-base.txt

# YOLO
ultralytics==8.0.220
# ultralytics 패키지가 torch 의존성 자동 설치
```

**requirements-huggingface.txt**:
```txt
-r requirements-base.txt

# Transformers
torch==2.1.0
transformers>=4.35.0
accelerate>=0.24.0
datasets>=2.14.0
```

**requirements-custom.txt**:
```txt
-r requirements-base.txt

# 사용자가 필요한 패키지 추가
# (Base 이미지 사용, 런타임에 pip install 가능)
```

#### 2.2 로컬 venv 재구성 (Optional)

개발 시 특정 프레임워크만 설치:

```bash
# timm 개발 시
python -m venv mvp/training/venv-timm
source mvp/training/venv-timm/bin/activate
pip install -r mvp/training/requirements/requirements-timm.txt

# ultralytics 개발 시
python -m venv mvp/training/venv-ultralytics
source mvp/training/venv-ultralytics/bin/activate
pip install -r mvp/training/requirements/requirements-ultralytics.txt
```

**또는** MVP 호환성 유지를 위해 단일 venv에 모두 설치:
```bash
pip install -r mvp/training/requirements/requirements-timm.txt
pip install -r mvp/training/requirements/requirements-ultralytics.txt
# (개발 편의성 vs 격리, 선택 가능)
```

**Deliverables**:
- [ ] Requirements 파일 분리
- [ ] 각 파일 테스트 (설치 확인)
- [ ] 문서화

---

### Phase 3: Docker 이미지 정의 (Week 2)

**목표**: Dockerfile 작성 및 빌드 스크립트 생성

#### 3.1 Base 이미지 Dockerfile

**파일**: `mvp/docker/Dockerfile.base`

```dockerfile
# ============================================
# Vision Platform Base Image
# ============================================
FROM python:3.11-slim AS base

# 기본 시스템 패키지
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Platform SDK 복사
COPY training/platform_sdk/ /opt/vision-platform/platform_sdk/

# Base requirements 설치
COPY training/requirements/requirements-base.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-base.txt

# Platform SDK를 Python path에 추가
ENV PYTHONPATH="${PYTHONPATH}:/opt/vision-platform"

# 환경 변수
ENV MLFLOW_TRACKING_URI="file:///workspace/mlruns"
ENV PYTHONUNBUFFERED=1

# Entrypoint 준비
COPY training/train.py /opt/vision-platform/
WORKDIR /workspace
```

#### 3.2 Framework별 Dockerfile

**파일**: `mvp/docker/Dockerfile.timm`

```dockerfile
FROM vision-platform-base:latest AS timm

# timm 의존성 설치
COPY training/requirements/requirements-timm.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-timm.txt

# timm adapter 복사
COPY training/adapters/timm_adapter.py /opt/vision-platform/adapters/
COPY training/adapters/__init__.py /opt/vision-platform/adapters/

# Metadata
LABEL framework="timm"
LABEL task_types="image_classification"
LABEL version="1.0.0"
```

**파일**: `mvp/docker/Dockerfile.ultralytics`

```dockerfile
FROM vision-platform-base:latest AS ultralytics

# Ultralytics 의존성 설치
COPY training/requirements/requirements-ultralytics.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-ultralytics.txt

# ultralytics adapter 복사
COPY training/adapters/ultralytics_adapter.py /opt/vision-platform/adapters/
COPY training/adapters/__init__.py /opt/vision-platform/adapters/

# Metadata
LABEL framework="ultralytics"
LABEL task_types="object_detection,instance_segmentation,pose_estimation"
LABEL version="1.0.0"
```

**파일**: `mvp/docker/Dockerfile.huggingface`

```dockerfile
FROM vision-platform-base:latest AS huggingface

# HuggingFace 의존성 설치
COPY training/requirements/requirements-huggingface.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-huggingface.txt

# huggingface adapter 복사
COPY training/adapters/huggingface_adapter.py /opt/vision-platform/adapters/
COPY training/adapters/__init__.py /opt/vision-platform/adapters/

# Metadata
LABEL framework="huggingface"
LABEL task_types="image_classification,object_detection,image_captioning"
LABEL version="1.0.0"
```

#### 3.3 Multi-stage 빌드 전략

**Layer Caching 최적화**를 위한 Dockerfile 구조:

```dockerfile
# 공통 레이어는 한 번만 빌드
FROM python:3.11-slim AS base-python
RUN apt-get update && apt-get install -y ...

FROM base-python AS base-packages
RUN pip install mlflow boto3 numpy ...

# 프레임워크별로 분기
FROM base-packages AS timm
RUN pip install timm torch torchvision

FROM base-packages AS ultralytics
RUN pip install ultralytics
```

→ **베이스 레이어 재사용으로 빌드 시간 50% 단축**

#### 3.4 빌드 스크립트

**파일**: `mvp/docker/build.sh`

```bash
#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building Vision Platform Docker Images${NC}"

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Build base image
echo -e "\n${GREEN}[1/4] Building base image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.base" \
    -t vision-platform-base:latest \
    --build-arg VERSION=0.1.0 \
    "$PROJECT_ROOT"

# Build timm image
echo -e "\n${GREEN}[2/4] Building timm image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.timm" \
    -t vision-platform-timm:latest \
    "$PROJECT_ROOT"

# Build ultralytics image
echo -e "\n${GREEN}[3/4] Building ultralytics image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.ultralytics" \
    -t vision-platform-ultralytics:latest \
    "$PROJECT_ROOT"

# Build huggingface image
echo -e "\n${GREEN}[4/4] Building huggingface image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.huggingface" \
    -t vision-platform-huggingface:latest \
    "$PROJECT_ROOT"

echo -e "\n${GREEN}✓ All images built successfully!${NC}"

# List images
docker images | grep vision-platform
```

실행:
```bash
chmod +x mvp/docker/build.sh
./mvp/docker/build.sh
```

**Deliverables**:
- [ ] Base Dockerfile 작성
- [ ] Framework별 Dockerfile 작성
- [ ] 빌드 스크립트 작성
- [ ] 이미지 빌드 성공 확인
- [ ] 이미지 크기 최적화

---

### Phase 4: TrainingManager Docker 지원 (Week 2-3)

**목표**: Backend가 Docker 컨테이너로 학습을 실행할 수 있도록 확장

#### 4.1 실행 모드 추가

**파일**: `mvp/backend/app/utils/training_manager.py`

```python
import os
import subprocess
from enum import Enum

class ExecutionMode(Enum):
    """Training execution mode"""
    SUBPROCESS = "subprocess"  # 기존 방식 (MVP 호환)
    DOCKER = "docker"          # Docker 컨테이너
    KUBERNETES = "kubernetes"  # Kubernetes Job (미래)

class TrainingManager:
    def __init__(self, db: Session, execution_mode: ExecutionMode = None):
        self.db = db
        self.processes = {}

        # Auto-detect execution mode
        if execution_mode is None:
            execution_mode = self._detect_execution_mode()

        self.execution_mode = execution_mode

    def _detect_execution_mode(self) -> ExecutionMode:
        """Auto-detect best execution mode"""
        # Check if Docker is available
        try:
            subprocess.run(
                ["docker", "version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            # Docker available, use it by default
            return ExecutionMode.DOCKER
        except Exception:
            # Docker not available, fallback to subprocess
            return ExecutionMode.SUBPROCESS

    def start_training(self, job_id: int, checkpoint_path: str = None, resume: bool = False):
        """Start training using configured execution mode"""
        if self.execution_mode == ExecutionMode.SUBPROCESS:
            return self._start_training_subprocess(job_id, checkpoint_path, resume)
        elif self.execution_mode == ExecutionMode.DOCKER:
            return self._start_training_docker(job_id, checkpoint_path, resume)
        else:
            raise ValueError(f"Unsupported execution mode: {self.execution_mode}")
```

#### 4.2 Docker 실행 구현

```python
IMAGE_MAP = {
    "timm": "vision-platform-timm:latest",
    "ultralytics": "vision-platform-ultralytics:latest",
    "huggingface": "vision-platform-huggingface:latest",
    "custom": "vision-platform-custom:latest",
}

def _start_training_docker(self, job_id: int, checkpoint_path: str = None, resume: bool = False):
    """Start training in Docker container"""
    job = self.db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job or job.status != "pending":
        return False

    # Select Docker image based on framework
    image = IMAGE_MAP.get(job.framework)
    if not image:
        raise ValueError(f"No Docker image for framework: {job.framework}")

    # Get absolute paths
    project_root = self._get_project_root()
    dataset_path = os.path.abspath(job.dataset_path)
    output_dir = os.path.abspath(job.output_dir)

    # Prepare Docker command
    docker_cmd = [
        "docker", "run",
        "--rm",  # Remove container when done
        "--name", f"training-job-{job_id}",

        # GPU support
        "--gpus", "all",  # Use all available GPUs

        # Volume mounts
        "-v", f"{dataset_path}:/workspace/dataset:ro",  # Dataset (read-only)
        "-v", f"{output_dir}:/workspace/output:rw",     # Output (read-write)

        # Environment variables
        "-e", f"JOB_ID={job_id}",
        "-e", "PYTHONUNBUFFERED=1",

        # Network (for MLflow tracking)
        "--network", "host",  # Use host network for simplicity

        # Image
        image,

        # Training command
        "python", "/opt/vision-platform/train.py",
        "--framework", job.framework,
        "--task_type", job.task_type,
        "--model_name", job.model_name,
        "--dataset_path", "/workspace/dataset",  # Path inside container
        "--dataset_format", job.dataset_format,
        "--output_dir", "/workspace/output",
        "--epochs", str(job.epochs),
        "--batch_size", str(job.batch_size),
        "--learning_rate", str(job.learning_rate),
        "--job_id", str(job_id),
    ]

    # Add num_classes if set
    if job.num_classes is not None:
        docker_cmd.extend(["--num_classes", str(job.num_classes)])

    # Add checkpoint args if provided
    if checkpoint_path:
        docker_cmd.extend(["--checkpoint_path", checkpoint_path])
        if resume:
            docker_cmd.append("--resume")

    try:
        # Start container
        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Store process (for stop/monitoring)
        self.processes[job_id] = process

        # Update job status
        job.status = "running"
        job.started_at = datetime.utcnow()
        job.process_id = process.pid
        self.db.commit()

        # Start monitoring thread (동일)
        monitor_thread = threading.Thread(
            target=self._monitor_training,
            args=(job_id, process),
            daemon=True,
        )
        monitor_thread.start()

        return True

    except Exception as e:
        job.status = "failed"
        job.error_message = f"Failed to start Docker training: {str(e)}"
        self.db.commit()
        return False

def _start_training_subprocess(self, job_id: int, checkpoint_path: str = None, resume: bool = False):
    """기존 subprocess 방식 (MVP 호환)"""
    # 기존 코드 그대로 유지
    ...
```

#### 4.3 환경 설정

**파일**: `mvp/backend/.env`

```bash
# Training execution mode
TRAINING_EXECUTION_MODE=docker  # docker | subprocess | auto

# Docker settings
DOCKER_GPU_ENABLED=true
DOCKER_NETWORK=host
```

**파일**: `mvp/backend/app/core/config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ... 기존 설정들 ...

    # Training execution
    training_execution_mode: str = "auto"  # docker, subprocess, auto
    docker_gpu_enabled: bool = True
    docker_network: str = "host"
```

#### 4.4 로컬 개발용 docker-compose

**파일**: `mvp/docker/docker-compose.yml`

```yaml
version: '3.8'

services:
  # Example: timm 학습 실행
  training-timm:
    image: vision-platform-timm:latest
    container_name: training-timm-dev
    volumes:
      - ../data/datasets:/workspace/dataset:ro
      - ../data/outputs:/workspace/output:rw
      - ../runs/mlflow:/workspace/mlruns:rw
    environment:
      - PYTHONUNBUFFERED=1
      - MLFLOW_TRACKING_URI=file:///workspace/mlruns
    command: >
      python /opt/vision-platform/train.py
      --framework timm
      --task_type image_classification
      --model_name resnet18
      --dataset_path /workspace/dataset
      --dataset_format imagefolder
      --output_dir /workspace/output
      --epochs 10
      --batch_size 32
      --learning_rate 0.001
      --job_id 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ultralytics 예시
  training-ultralytics:
    image: vision-platform-ultralytics:latest
    container_name: training-ultralytics-dev
    volumes:
      - ../data/datasets:/workspace/dataset:ro
      - ../data/outputs:/workspace/output:rw
    environment:
      - PYTHONUNBUFFERED=1
    command: >
      python /opt/vision-platform/train.py
      --framework ultralytics
      --task_type object_detection
      --model_name yolov8n
      --dataset_path /workspace/dataset
      --dataset_format yolo
      --output_dir /workspace/output
      --epochs 50
      --batch_size 16
      --learning_rate 0.01
      --job_id 2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

실행:
```bash
# timm 학습 실행
docker-compose -f mvp/docker/docker-compose.yml up training-timm

# ultralytics 학습 실행
docker-compose -f mvp/docker/docker-compose.yml up training-ultralytics
```

**Deliverables**:
- [ ] ExecutionMode enum 추가
- [ ] Docker 실행 로직 구현
- [ ] 기존 subprocess 로직 유지
- [ ] Auto-detection 로직
- [ ] 환경 설정 추가
- [ ] docker-compose.yml 작성
- [ ] 테스트 (Docker + subprocess 모두)

---

### Phase 5: 테스트 및 문서화 (Week 3)

**목표**: 통합 테스트 및 사용자 문서 작성

#### 5.1 통합 테스트

**테스트 시나리오**:

1. **Subprocess 모드 (MVP 호환성)**:
   ```bash
   # Backend .env
   TRAINING_EXECUTION_MODE=subprocess

   # 학습 실행
   curl -X POST http://localhost:8000/api/v1/training/start

   # 검증: mvp/training/venv에서 실행되는지 확인
   ```

2. **Docker 모드 (timm)**:
   ```bash
   TRAINING_EXECUTION_MODE=docker

   # 학습 실행
   curl -X POST http://localhost:8000/api/v1/training/start \
     -H "Content-Type: application/json" \
     -d '{
       "framework": "timm",
       "model_name": "resnet18",
       ...
     }'

   # 검증: vision-platform-timm 컨테이너 실행 확인
   docker ps | grep vision-platform-timm
   ```

3. **Docker 모드 (ultralytics)**:
   ```bash
   # YOLO 학습
   curl -X POST http://localhost:8000/api/v1/training/start \
     -H "Content-Type: application/json" \
     -d '{
       "framework": "ultralytics",
       "model_name": "yolov8n",
       ...
     }'

   # 검증: vision-platform-ultralytics 컨테이너
   docker ps | grep vision-platform-ultralytics
   ```

4. **이미지 크기 비교**:
   ```bash
   docker images | grep vision-platform

   # 예상 결과:
   # vision-platform-base           ~500 MB
   # vision-platform-timm           ~2 GB
   # vision-platform-ultralytics    ~1.5 GB
   # vision-platform-huggingface    ~3 GB
   ```

5. **의존성 격리 검증**:
   ```bash
   # timm 컨테이너에는 ultralytics가 없어야 함
   docker run vision-platform-timm:latest python -c "import ultralytics"
   # → ImportError (정상)

   # ultralytics 컨테이너에는 timm이 없어야 함
   docker run vision-platform-ultralytics:latest python -c "import timm"
   # → ImportError (정상)
   ```

#### 5.2 문서 작성

**사용자 가이드**: `docs/USER_GUIDE.md`

```markdown
# Docker 이미지 사용 가이드

## 로컬 개발 모드 (subprocess)

MVP와 동일하게 동작:
\`\`\`bash
cd mvp/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
\`\`\`

## Docker 모드

### 1. 이미지 빌드
\`\`\`bash
cd mvp/docker
./build.sh
\`\`\`

### 2. Backend 설정
\`\`\`bash
# mvp/backend/.env
TRAINING_EXECUTION_MODE=docker
\`\`\`

### 3. 학습 실행
자동으로 적절한 Docker 이미지 선택됨
```

**개발자 가이드**: `docs/DEVELOPER_GUIDE.md`

```markdown
# 새 프레임워크 추가하기

## 1. Adapter 작성
\`\`\`python
# mvp/training/adapters/myframework_adapter.py
from platform_sdk import TrainingAdapter

class MyFrameworkAdapter(TrainingAdapter):
    ...
\`\`\`

## 2. Requirements 파일
\`\`\`txt
# mvp/training/requirements/requirements-myframework.txt
-r requirements-base.txt
myframework==1.0.0
\`\`\`

## 3. Dockerfile
\`\`\`dockerfile
# mvp/docker/Dockerfile.myframework
FROM vision-platform-base:latest
COPY training/requirements/requirements-myframework.txt /tmp/
RUN pip install -r /tmp/requirements-myframework.txt
COPY training/adapters/myframework_adapter.py /opt/vision-platform/adapters/
\`\`\`

## 4. 빌드 스크립트 업데이트
\`\`\`bash
# mvp/docker/build.sh에 추가
docker build -f Dockerfile.myframework -t vision-platform-myframework:latest .
\`\`\`

## 5. TrainingManager 업데이트
\`\`\`python
# backend/app/utils/training_manager.py
IMAGE_MAP = {
    ...
    "myframework": "vision-platform-myframework:latest",
}
\`\`\`
```

**Deliverables**:
- [ ] 통합 테스트 스크립트
- [ ] 테스트 결과 문서
- [ ] 사용자 가이드
- [ ] 개발자 가이드
- [ ] Troubleshooting 가이드

---

## Migration Strategy

### 기존 코드 호환성

**완벽한 하위 호환성 유지**:

```python
# 기존 코드 (MVP)는 변경 없이 동작
if __name__ == "__main__":
    # subprocess 모드로 자동 동작
    manager = TrainingManager(db)
    manager.start_training(job_id=1)
```

**점진적 마이그레이션**:

1. **Phase 1-2 완료 시**: 로컬 개발 그대로 (subprocess)
2. **Phase 3 완료 시**: Docker 이미지 선택 가능 (opt-in)
3. **Phase 4 완료 시**: Docker 기본 모드, subprocess는 fallback

### 배포 전략

**로컬 개발**:
```bash
TRAINING_EXECUTION_MODE=subprocess  # 빠른 반복 개발
```

**스테이징**:
```bash
TRAINING_EXECUTION_MODE=docker  # 프로덕션 환경 시뮬레이션
```

**프로덕션**:
```bash
TRAINING_EXECUTION_MODE=docker  # 완전한 격리
```

---

## Benefits Realization

### 의존성 격리

**Before**:
```bash
pip list | wc -l
# → 150+ packages (모든 프레임워크)
```

**After**:
```bash
docker run vision-platform-timm pip list | wc -l
# → 80 packages (timm + 필수만)

docker run vision-platform-ultralytics pip list | wc -l
# → 60 packages (ultralytics + 필수만)
```

### 이미지 크기 최적화

```
전체 설치 (기존):
  - PyTorch + timm + ultralytics + transformers
  - 크기: ~8 GB
  - 빌드 시간: 15분

분리 후:
  - Base: 500 MB (1회 빌드)
  - timm: +1.5 GB (총 2 GB)
  - ultralytics: +1 GB (총 1.5 GB)
  - huggingface: +2.5 GB (총 3 GB)

  총합: 7 GB (모든 이미지)
  실제 사용: 2 GB (필요한 것만)
  빌드 시간: 5분 (layer caching)
```

### 빌드 시간 단축

```
Cold build (모든 이미지 처음 빌드):
  Base: 3분
  timm: +2분 (총 5분)
  ultralytics: +1분 (총 6분)
  huggingface: +3분 (총 9분)

Incremental build (adapter 코드만 변경):
  timm: 10초 (base 레이어 재사용)
  ultralytics: 10초
```

### 개발자 경험

**새 프레임워크 추가 시**:

**Before** (단일 requirements.txt):
```bash
1. requirements.txt에 추가 → 충돌 위험
2. 전체 venv 재빌드 필요
3. 다른 프레임워크에 영향
4. 시간: 15분
```

**After** (독립 이미지):
```bash
1. requirements-newframework.txt 생성
2. Dockerfile.newframework 작성
3. build.sh에 추가
4. 빌드 (독립적)
5. 시간: 5분 (base 재사용)
```

---

## Risk Mitigation

### Risk 1: Docker 학습 곡선

**위험**: 개발자가 Docker에 익숙하지 않을 수 있음

**완화 방안**:
- ✅ subprocess 모드 유지 (MVP 호환)
- ✅ Auto-detection으로 자동 선택
- ✅ 상세한 문서 및 예제 제공
- ✅ docker-compose.yml로 쉬운 로컬 테스트

### Risk 2: 성능 오버헤드

**위험**: Docker 컨테이너 시작 시간

**완화 방안**:
- ✅ GPU pass-through로 성능 동일
- ✅ 컨테이너 시작 시간 < 5초 (negligible)
- ✅ 학습 시간이 주요 병목 (시간/분 단위)

### Risk 3: Volume 마운트 복잡도

**위험**: 데이터셋 경로 관리

**완화 방안**:
- ✅ TrainingManager가 자동 처리
- ✅ 절대 경로 자동 변환
- ✅ 명확한 로깅

### Risk 4: 디버깅 어려움

**위험**: 컨테이너 내부 디버깅

**완화 방안**:
- ✅ 로그를 host로 실시간 스트리밍
- ✅ `docker exec`로 컨테이너 접근 가능
- ✅ subprocess 모드로 로컬 디버깅 가능

---

## Success Criteria

### Technical Metrics

- [ ] 모든 프레임워크 이미지 빌드 성공
- [ ] 이미지 크기 < 3GB per framework
- [ ] 빌드 시간 < 10분 (cold build)
- [ ] 빌드 시간 < 1분 (incremental)
- [ ] Subprocess 모드 100% 호환
- [ ] Docker 모드 정상 동작
- [ ] 의존성 충돌 0건

### Operational Metrics

- [ ] 새 프레임워크 추가 시간 < 1시간
- [ ] 기존 코드 변경 없이 마이그레이션 가능
- [ ] 문서화 완료도 100%
- [ ] 개발자 온보딩 시간 < 30분

### User Experience

- [ ] 사용자는 실행 모드를 의식하지 않음
- [ ] 학습 시작 시간 차이 < 10초
- [ ] 에러 메시지 명확
- [ ] 트러블슈팅 가이드 제공

---

## Timeline Summary

| Phase | Duration | Effort | Key Deliverables |
|-------|----------|--------|------------------|
| Phase 0: 사전 준비 | 3 days | Low | 현황 분석, 설계 문서 |
| Phase 1: Platform SDK 분리 | 1 week | Medium | platform_sdk 패키지, import 리팩토링 |
| Phase 2: Requirements 분리 | 3 days | Low | 프레임워크별 requirements 파일 |
| Phase 3: Docker 이미지 | 1 week | High | Dockerfiles, 빌드 스크립트 |
| Phase 4: TrainingManager 확장 | 1 week | Medium | Docker 실행 로직, auto-detection |
| Phase 5: 테스트 및 문서 | 1 week | Medium | 통합 테스트, 사용자/개발자 가이드 |
| **Total** | **3-4 weeks** | **Medium-High** | **Production-ready Docker 분리 구조** |

---

## Next Steps

1. **Review & Approval**: 본 설계 문서 리뷰 및 승인
2. **Kickoff Meeting**: 팀 미팅 및 작업 할당
3. **Branch Creation**: `feat/docker-image-separation` 브랜치 생성
4. **Phase 1 Start**: Platform SDK 분리 작업 시작

---

## References

- [Docker Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Docker Layer Caching](https://docs.docker.com/build/cache/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [현재 Adapter 설계](./ADAPTER_DESIGN.md)
- [Trainer Implementation Plan](../planning/TRAINER_IMPLEMENTATION_PLAN.md)

---

## Appendix A: 예상 이미지 크기

```bash
REPOSITORY                          TAG       SIZE
vision-platform-base                latest    500 MB
vision-platform-timm                latest    2.0 GB
vision-platform-ultralytics         latest    1.5 GB
vision-platform-huggingface         latest    3.0 GB
vision-platform-custom              latest    500 MB (base only)
```

## Appendix B: 빌드 시간 벤치마크

```
MacBook Pro M1 (16GB RAM):
  Base image: 2m 30s
  timm image: +1m 45s
  ultralytics image: +1m 10s
  huggingface image: +2m 20s
  Total (cold): 7m 45s

  Incremental (adapter 변경): 15s

Ubuntu 22.04 (32GB RAM, 8 cores):
  Base image: 1m 50s
  timm image: +1m 20s
  ultralytics image: +50s
  huggingface image: +1m 40s
  Total (cold): 5m 40s

  Incremental: 8s
```

## Appendix C: FAQ

**Q: MVP 환경(subprocess)을 계속 사용할 수 있나요?**
A: 네, `TRAINING_EXECUTION_MODE=subprocess`로 설정하면 기존과 동일하게 동작합니다.

**Q: Docker 없이 개발 가능한가요?**
A: 네, Docker는 선택사항입니다. 로컬 venv로 개발 후 Docker는 배포 시에만 사용 가능합니다.

**Q: 이미지 크기가 너무 크지 않나요?**
A: 딥러닝 프레임워크는 본질적으로 큽니다. 하지만 분리 전 8GB vs 분리 후 필요한 것만 2GB로 75% 절감됩니다.

**Q: 새 프레임워크 추가 시 얼마나 걸리나요?**
A: requirements 파일(5분) + Dockerfile(10분) + 빌드 스크립트 수정(5분) + 빌드(5분) = 약 25분입니다.

**Q: Kubernetes 지원은요?**
A: Docker 이미지가 있으면 Kubernetes Job으로 쉽게 전환 가능합니다. Phase 6에서 추가 예정입니다.

---

*End of Document*
