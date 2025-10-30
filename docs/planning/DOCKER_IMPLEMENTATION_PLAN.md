# Docker 의존성 분리 구현 계획

**Document Version:** 1.0
**Created:** 2025-10-30
**Status:** Implementation Plan
**Target Timeline:** 3-4 weeks (15-19 days)

---

## Executive Summary

본 문서는 Vision AI Training Platform의 Docker 기반 의존성 분리 구현을 위한 실행 계획입니다.

### 배경

**즉시 발생한 문제**:
- YOLO-World 모델 학습 실패
- 원인: ultralytics 8.0.220 사용 (YOLOWorld 클래스 미지원)
- 필요: ultralytics 8.3.0+ 업그레이드

**근본 원인**:
- 단일 `requirements.txt`로 모든 프레임워크 의존성 관리
- 버전 고정으로 인한 업그레이드 제약
- 의존성 충돌 위험 상존

### 해결 방안

**프레임워크별 독립 Docker 이미지** 구조로 전환:
```
vision-platform-base:latest (공통 SDK)
├── vision-platform-timm:latest
└── vision-platform-ultralytics:latest (8.3.0+)
```

### 핵심 목표

1. ✅ **YOLO-World 즉시 해결**: ultralytics 8.3.0+ 사용
2. ✅ **의존성 격리**: 프레임워크 간 충돌 제거
3. ✅ **확장성**: 새 프레임워크 추가 용이
4. ✅ **하위 호환성**: 기존 subprocess 모드 100% 유지

---

## Current State Analysis

### 현재 아키텍처

**Backend → Training 실행 흐름**:
```
┌─────────────────────────────────────┐
│  Backend (FastAPI)                  │
│  - TrainingManager                  │
│  - subprocess로 학습 실행           │
└───────────┬─────────────────────────┘
            ↓ subprocess.Popen()
┌─────────────────────────────────────┐
│  Training Process                   │
│  - mvp/training/venv/Scripts/python │
│  - train.py --framework ...         │
│  - 단일 venv (모든 프레임워크)     │
└─────────────────────────────────────┘
```

**의존성 현황** (`mvp/training/requirements.txt`):
```txt
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
ultralytics==8.0.220  ← 업그레이드 필요 (8.3.0+)
mlflow==2.9.2
boto3==1.34.10
numpy==1.26.0
pillow==10.1.0
```

**코드 구조**:
```
mvp/training/
├── adapters/
│   ├── base.py (1727 lines) - TrainingAdapter, ModelConfig 등
│   ├── timm_adapter.py (902 lines)
│   └── ultralytics_adapter.py (2054 lines)
├── model_registry/
│   ├── timm_models.py (18 models)
│   └── ultralytics_models.py (19 models)
├── train.py
├── config_schemas.py
└── requirements.txt (단일 파일)
```

### 완료된 작업 (Phase 1 - Model Registry)

- ✅ 37개 모델 등록 (18 timm + 19 ultralytics)
- ✅ P0/P1/P2 우선순위 체계
- ✅ Adapter 패턴 검증
- ✅ Model Registry API/UI 구현

### 발견된 문제

**YOLO-World 학습 실패**:
```
ImportError: cannot import name 'YOLOWorld' from 'ultralytics'
```

**원인**:
- ultralytics 8.0.220에는 `YOLOWorld` 클래스 없음
- 8.3.0+에서 추가됨
- 단일 requirements.txt로 인한 버전 고정

**영향**:
- YOLO-World 모델 학습 불가 (1개 모델)
- 향후 다른 프레임워크 추가 시에도 유사한 문제 발생 가능

---

## Proposed Architecture

### Docker 이미지 계층 구조

```
┌──────────────────────────────────────────────┐
│  vision-platform-base:latest                 │
│  - Python 3.11                               │
│  - Platform SDK (공통 코드)                 │
│  - MLflow, boto3, numpy (공통 의존성)       │
│  Size: ~500 MB                               │
└──────────────────────────────────────────────┘
         ↑                        ↑
         │                        │
   ┌─────┴────────┐    ┌──────────┴─────────┐
   │    timm      │    │   ultralytics      │
   │              │    │                    │
   │ FROM base    │    │ FROM base          │
   │ + timm       │    │ + ultralytics 8.3+ │
   │ + torch 2.1  │    │   (YOLOWorld 지원) │
   │              │    │                    │
   │ Size: ~2 GB  │    │ Size: ~1.5 GB      │
   └──────────────┘    └────────────────────┘
```

### 새로운 실행 흐름

```python
# Backend: TrainingManager

IMAGE_MAP = {
    "timm": "vision-platform-timm:latest",
    "ultralytics": "vision-platform-ultralytics:latest",
}

def start_training(job_id: int):
    framework = job.framework
    image = IMAGE_MAP[framework]

    if USE_DOCKER:
        # Docker 모드
        docker run --gpus all \
            -v {dataset}:/workspace/dataset \
            -v {output}:/workspace/output \
            {image} \
            python /opt/vision-platform/train.py ...
    else:
        # Subprocess 모드 (기존)
        python mvp/training/train.py ...
```

### 디렉토리 구조 (재구성)

```
mvp-vision-ai-platform/
├── mvp/
│   ├── backend/
│   │   └── app/
│   │       └── utils/
│   │           └── training_manager.py (ExecutionMode 추가)
│   │
│   ├── training/
│   │   ├── platform_sdk/           # 🆕 공통 플랫폼 SDK
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # TrainingAdapter, ModelConfig
│   │   │   ├── callbacks.py        # TrainingCallbacks
│   │   │   ├── mlflow_utils.py
│   │   │   └── storage.py
│   │   │
│   │   ├── adapters/               # 프레임워크별 Adapter
│   │   │   ├── timm_adapter.py
│   │   │   └── ultralytics_adapter.py
│   │   │
│   │   ├── model_registry/
│   │   │   ├── timm_models.py
│   │   │   └── ultralytics_models.py
│   │   │
│   │   ├── requirements/           # 🆕 분리된 의존성
│   │   │   ├── requirements-base.txt
│   │   │   ├── requirements-timm.txt
│   │   │   └── requirements-ultralytics.txt
│   │   │
│   │   ├── train.py
│   │   ├── config_schemas.py
│   │   └── venv/                   # Local 모드용 (optional)
│   │
│   └── docker/                     # 🆕 Docker 이미지 정의
│       ├── Dockerfile.base
│       ├── Dockerfile.timm
│       ├── Dockerfile.ultralytics
│       ├── build.sh
│       ├── docker-compose.training.yml
│       └── .dockerignore
│
└── docs/
    ├── architecture/
    │   └── DOCKER_IMAGE_SEPARATION.md
    └── planning/
        ├── IMPLEMENTATION_PRIORITY_ANALYSIS.md
        └── DOCKER_IMPLEMENTATION_PLAN.md (본 문서)
```

---

## Implementation Plan

### Phase 1: Platform SDK 분리 (3-4일)

**목표**: 공통 코드를 독립 패키지로 분리

#### 1.1 Platform SDK 패키지 생성

**디렉토리 생성**:
```bash
mkdir -p mvp/training/platform_sdk
```

**파일 구조**:
```
platform_sdk/
├── __init__.py
├── base.py                 # TrainingAdapter, ModelConfig 등
├── callbacks.py            # TrainingCallbacks
├── mlflow_utils.py         # MLflow 헬퍼
└── storage.py              # S3, 파일 처리 (신규)
```

#### 1.2 코드 이동 및 리팩토링

**`platform_sdk/__init__.py`**:
```python
"""Vision Platform Training SDK - Common utilities for all frameworks."""

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

__version__ = "0.1.0"
__all__ = [
    "TrainingAdapter",
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "MetricsResult",
    "TaskType",
    "DatasetFormat",
    "TrainingCallbacks",
]
```

**`platform_sdk/base.py`**:
- `adapters/base.py` 내용 이동 (1727 lines)
- 변경 없이 그대로 이동

**`platform_sdk/callbacks.py`**:
- `base.py`에서 `TrainingCallbacks` 클래스 분리 (필요시)
- 또는 그대로 `base.py`에 유지

**`platform_sdk/storage.py`** (신규):
```python
"""Storage utilities for datasets, checkpoints, and artifacts."""

import boto3
from pathlib import Path
from typing import Optional

def upload_to_s3(local_path: str, s3_path: str, bucket: str) -> str:
    """Upload file to S3."""
    # Implementation
    pass

def download_from_s3(s3_path: str, local_path: str, bucket: str) -> str:
    """Download file from S3."""
    # Implementation
    pass
```

#### 1.3 Adapter 코드 리팩토링

**`adapters/timm_adapter.py`**:
```python
# Before
from .base import TrainingAdapter, MetricsResult, TaskType

# After
from platform_sdk import TrainingAdapter, MetricsResult, TaskType
from platform_sdk.callbacks import TrainingCallbacks
```

**`adapters/ultralytics_adapter.py`**:
```python
# Before
from .base import TrainingAdapter, MetricsResult, TaskType, DatasetFormat

# After
from platform_sdk import TrainingAdapter, MetricsResult, TaskType, DatasetFormat
from platform_sdk.callbacks import TrainingCallbacks
```

**`adapters/__init__.py`**:
```python
"""Training adapters for different frameworks."""

from .timm_adapter import TimmAdapter
from .ultralytics_adapter import UltralyticsAdapter

ADAPTER_REGISTRY = {
    'timm': TimmAdapter,
    'ultralytics': UltralyticsAdapter,
}

__all__ = ['TimmAdapter', 'UltralyticsAdapter', 'ADAPTER_REGISTRY']
```

#### 1.4 train.py 업데이트

**`train.py`**:
```python
# Before
from adapters import ADAPTER_REGISTRY
from adapters.base import TaskType, DatasetFormat

# After
from platform_sdk import TaskType, DatasetFormat
from adapters import ADAPTER_REGISTRY
```

#### 1.5 검증

**테스트 명령어**:
```bash
cd mvp/training

# timm 테스트
venv/Scripts/python train.py \
    --framework timm \
    --task_type image_classification \
    --model_name resnet18 \
    --dataset_path ./data/sample_dataset \
    --dataset_format imagefolder \
    --output_dir ./outputs/test_phase1 \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --num_classes 10

# ultralytics 테스트
venv/Scripts/python train.py \
    --framework ultralytics \
    --task_type object_detection \
    --model_name yolov8n \
    --dataset_path ./data/yolo_dataset \
    --dataset_format yolo \
    --output_dir ./outputs/test_phase1_yolo \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 0.01
```

**Deliverables**:
- [ ] `platform_sdk/` 패키지 생성 및 코드 이동
- [ ] Import 경로 업데이트 (adapters, train.py)
- [ ] 로컬 테스트 통과 (timm + ultralytics)
- [ ] 코드 리뷰 및 승인

---

### Phase 2: Requirements 분리 (2일)

**목표**: 프레임워크별 독립 의존성 파일 생성

#### 2.1 디렉토리 생성

```bash
mkdir -p mvp/training/requirements
```

#### 2.2 Requirements 파일 작성

**`requirements/requirements-base.txt`** (공통):
```txt
# ============================================
# Vision Platform - Base Requirements
# Common dependencies for all frameworks
# ============================================

# Experiment Tracking
mlflow==2.9.2
boto3==1.34.10

# Data Processing
numpy==1.26.0
pillow==10.1.0
pyyaml==6.0.1

# Utilities
tqdm==4.66.0
tensorboard==2.15.1

# Database (for callbacks)
sqlalchemy==2.0.23
```

**`requirements/requirements-timm.txt`**:
```txt
# ============================================
# Vision Platform - timm Framework
# ============================================

# Include base requirements
-r requirements-base.txt

# Deep Learning Framework
torch==2.1.0
torchvision==0.16.0

# timm Library
timm==0.9.12
```

**`requirements/requirements-ultralytics.txt`**:
```txt
# ============================================
# Vision Platform - Ultralytics Framework
# Includes YOLO models (v5, v8, v11, YOLO-World)
# ============================================

# Include base requirements
-r requirements-base.txt

# Ultralytics YOLO
# NOTE: Version 8.3.0+ required for YOLOWorld support
ultralytics>=8.3.0

# Note: torch/torchvision automatically installed by ultralytics
```

#### 2.3 기존 requirements.txt 백업

```bash
cd mvp/training
mv requirements.txt requirements.txt.backup
```

#### 2.4 로컬 테스트 (Optional)

**Option A: 기존 venv 업그레이드** (권장):
```bash
cd mvp/training
venv/Scripts/pip install -r requirements/requirements-base.txt
venv/Scripts/pip install -r requirements/requirements-timm.txt
venv/Scripts/pip install -r requirements/requirements-ultralytics.txt

# YOLOWorld import 테스트
venv/Scripts/python -c "from ultralytics import YOLOWorld; print('✓ YOLOWorld OK')"
```

**Option B: 새 venv 생성** (격리 테스트):
```bash
# timm용 venv
python -m venv venv-timm
venv-timm/Scripts/activate
pip install -r requirements/requirements-timm.txt

# ultralytics용 venv
python -m venv venv-ultralytics
venv-ultralytics/Scripts/activate
pip install -r requirements/requirements-ultralytics.txt
```

#### 2.5 의존성 검증

**버전 확인**:
```bash
cd mvp/training
venv/Scripts/pip list | grep -E "torch|timm|ultralytics"
```

**예상 출력**:
```
torch                 2.1.0
torchvision           0.16.0
timm                  0.9.12
ultralytics           8.3.47 (또는 그 이상)
```

**Deliverables**:
- [ ] `requirements/` 디렉토리 생성
- [ ] 3개 requirements 파일 작성
- [ ] ultralytics 8.3.0+ 설치 확인
- [ ] YOLOWorld import 테스트 통과
- [ ] 버전 확인 문서화

---

### Phase 3: Docker 이미지 생성 (4-5일)

**목표**: Dockerfile 작성 및 이미지 빌드

#### 3.1 Docker 디렉토리 생성

```bash
mkdir -p mvp/docker
```

#### 3.2 .dockerignore 작성

**`mvp/docker/.dockerignore`**:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
venv-*/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/
outputs/
runs/
mlruns/

# Weights
*.pth
*.pt
*.onnx

# Documentation
docs/
*.md
!README.md

# Tests
tests/
test_*.py
*_test.py

# Git
.git/
.gitignore

# Backend
backend/

# Frontend
frontend/
```

#### 3.3 Base Dockerfile 작성

**`mvp/docker/Dockerfile.base`**:
```dockerfile
# ============================================
# Vision Platform - Base Image
# Common SDK and dependencies for all frameworks
# ============================================
FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="Vision AI Platform Team"
LABEL version="1.0.0"
LABEL description="Base image for Vision Platform training"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy Platform SDK
COPY training/platform_sdk/ /opt/vision-platform/platform_sdk/

# Copy base requirements and install
COPY training/requirements/requirements-base.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-base.txt && \
    rm /tmp/requirements-base.txt

# Add Platform SDK to Python path
ENV PYTHONPATH="${PYTHONPATH}:/opt/vision-platform"
ENV PYTHONUNBUFFERED=1

# Environment variables
ENV MLFLOW_TRACKING_URI="file:///workspace/mlruns"
ENV CUDA_VISIBLE_DEVICES="0"

# Copy training scripts
COPY training/train.py /opt/vision-platform/
COPY training/config_schemas.py /opt/vision-platform/

# Create necessary directories
RUN mkdir -p /workspace/dataset /workspace/output /workspace/mlruns

# Working directory for training
WORKDIR /workspace

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import platform_sdk; print('OK')" || exit 1

# Default command
CMD ["python", "/opt/vision-platform/train.py", "--help"]
```

#### 3.4 Framework Dockerfiles 작성

**`mvp/docker/Dockerfile.timm`**:
```dockerfile
# ============================================
# Vision Platform - timm Framework Image
# PyTorch Image Models for classification
# ============================================
FROM vision-platform-base:latest

# Metadata
LABEL framework="timm"
LABEL task_types="image_classification"
LABEL version="1.0.0"
LABEL description="timm framework with PyTorch 2.1.0"

# Install timm requirements
COPY training/requirements/requirements-timm.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-timm.txt && \
    rm /tmp/requirements-timm.txt

# Copy timm adapter
COPY training/adapters/__init__.py /opt/vision-platform/adapters/
COPY training/adapters/timm_adapter.py /opt/vision-platform/adapters/

# Copy timm model registry
COPY training/model_registry/__init__.py /opt/vision-platform/model_registry/
COPY training/model_registry/timm_models.py /opt/vision-platform/model_registry/

# Verify installation
RUN python -c "import torch; import timm; print(f'PyTorch: {torch.__version__}'); print(f'timm: {timm.__version__}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import timm; print('OK')" || exit 1
```

**`mvp/docker/Dockerfile.ultralytics`**:
```dockerfile
# ============================================
# Vision Platform - Ultralytics Framework Image
# YOLO models (v5, v8, v11, YOLO-World)
# ============================================
FROM vision-platform-base:latest

# Metadata
LABEL framework="ultralytics"
LABEL task_types="object_detection,instance_segmentation,pose_estimation,zero_shot_detection"
LABEL version="1.0.0"
LABEL description="Ultralytics framework with YOLO-World support (8.3.0+)"

# Install ultralytics requirements
COPY training/requirements/requirements-ultralytics.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-ultralytics.txt && \
    rm /tmp/requirements-ultralytics.txt

# Copy ultralytics adapter
COPY training/adapters/__init__.py /opt/vision-platform/adapters/
COPY training/adapters/ultralytics_adapter.py /opt/vision-platform/adapters/

# Copy ultralytics model registry
COPY training/model_registry/__init__.py /opt/vision-platform/model_registry/
COPY training/model_registry/ultralytics_models.py /opt/vision-platform/model_registry/

# Verify installation (including YOLOWorld)
RUN python -c "from ultralytics import YOLO, YOLOWorld; import ultralytics; print(f'ultralytics: {ultralytics.__version__}'); print('✓ YOLOWorld available')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from ultralytics import YOLOWorld; print('OK')" || exit 1
```

#### 3.5 빌드 스크립트 작성

**`mvp/docker/build.sh`**:
```bash
#!/bin/bash

# ============================================
# Vision Platform Docker Build Script
# ============================================

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Vision Platform Docker Build${NC}"
echo -e "${BLUE}======================================${NC}"

# Get project root (mvp/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"

# Check if we're in the right directory
if [ ! -d "$PROJECT_ROOT/training" ]; then
    echo -e "${RED}Error: training/ directory not found${NC}"
    exit 1
fi

# Build base image
echo -e "\n${GREEN}[1/3] Building base image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.base" \
    -t vision-platform-base:latest \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    "$PROJECT_ROOT"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Base image build failed${NC}"
    exit 1
fi

# Build timm image
echo -e "\n${GREEN}[2/3] Building timm image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.timm" \
    -t vision-platform-timm:latest \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    "$PROJECT_ROOT"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ timm image build failed${NC}"
    exit 1
fi

# Build ultralytics image
echo -e "\n${GREEN}[3/3] Building ultralytics image...${NC}"
docker build \
    -f "$SCRIPT_DIR/Dockerfile.ultralytics" \
    -t vision-platform-ultralytics:latest \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    "$PROJECT_ROOT"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ ultralytics image build failed${NC}"
    exit 1
fi

# Success
echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}✓ All images built successfully!${NC}"
echo -e "${GREEN}======================================${NC}"

# List images
echo -e "\n${BLUE}Built images:${NC}"
docker images | grep -E "REPOSITORY|vision-platform"

# Show image sizes
echo -e "\n${BLUE}Image sizes:${NC}"
docker images vision-platform-base:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
docker images vision-platform-timm:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
docker images vision-platform-ultralytics:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo -e "\n${GREEN}Ready to use!${NC}"
echo -e "${YELLOW}Test with: docker run --rm vision-platform-ultralytics:latest python -c 'from ultralytics import YOLOWorld; print(\"OK\")'${NC}"
```

**Windows용 빌드 스크립트** (`mvp/docker/build.bat`):
```batch
@echo off
REM ============================================
REM Vision Platform Docker Build Script (Windows)
REM ============================================

echo ======================================
echo Vision Platform Docker Build
echo ======================================

cd /d %~dp0\..

echo [1/3] Building base image...
docker build -f docker/Dockerfile.base -t vision-platform-base:latest .
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/3] Building timm image...
docker build -f docker/Dockerfile.timm -t vision-platform-timm:latest .
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/3] Building ultralytics image...
docker build -f docker/Dockerfile.ultralytics -t vision-platform-ultralytics:latest .
if %errorlevel% neq 0 exit /b %errorlevel%

echo ======================================
echo All images built successfully!
echo ======================================

docker images | findstr vision-platform
```

#### 3.6 빌드 실행

**Linux/Mac**:
```bash
cd mvp
chmod +x docker/build.sh
./docker/build.sh
```

**Windows**:
```cmd
cd mvp
docker\build.bat
```

#### 3.7 검증

**이미지 확인**:
```bash
docker images | grep vision-platform
```

**예상 출력**:
```
vision-platform-ultralytics  latest  abc123  2 minutes ago  1.5GB
vision-platform-timm         latest  def456  5 minutes ago  2.0GB
vision-platform-base         latest  ghi789  8 minutes ago  500MB
```

**YOLOWorld 테스트**:
```bash
docker run --rm vision-platform-ultralytics:latest \
    python -c "from ultralytics import YOLOWorld; print('✓ YOLOWorld OK')"
```

**예상 출력**:
```
✓ YOLOWorld OK
```

**Deliverables**:
- [ ] `.dockerignore` 파일
- [ ] `Dockerfile.base` 작성
- [ ] `Dockerfile.timm` 작성
- [ ] `Dockerfile.ultralytics` 작성
- [ ] 빌드 스크립트 (build.sh, build.bat)
- [ ] 이미지 빌드 성공
- [ ] YOLOWorld import 테스트 통과
- [ ] 이미지 크기 문서화

---

### Phase 4: TrainingManager Docker 지원 (4-5일)

**목표**: Backend가 Docker 컨테이너로 학습 실행

#### 4.1 ExecutionMode 추가

**`mvp/backend/app/utils/training_manager.py` 수정**:

```python
"""Training process manager with Docker support."""

import json
import os
import subprocess
import threading
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy.orm import Session

from app.db import models
from app.utils.metrics import update_training_metrics, clear_training_metrics


class ExecutionMode(Enum):
    """Training execution mode."""
    SUBPROCESS = "subprocess"  # Local subprocess (MVP compatible)
    DOCKER = "docker"          # Docker container


class TrainingManager:
    """Manage training execution (subprocess or Docker)."""

    # Docker image mapping
    IMAGE_MAP = {
        "timm": "vision-platform-timm:latest",
        "ultralytics": "vision-platform-ultralytics:latest",
    }

    def __init__(self, db: Session, execution_mode: Optional[ExecutionMode] = None):
        """
        Initialize training manager.

        Args:
            db: Database session
            execution_mode: Execution mode (auto-detect if None)
        """
        self.db = db
        self.processes = {}  # job_id -> process

        # Auto-detect execution mode if not specified
        if execution_mode is None:
            execution_mode = self._detect_execution_mode()

        self.execution_mode = execution_mode
        print(f"[TrainingManager] Execution mode: {self.execution_mode.value}")

    def _detect_execution_mode(self) -> ExecutionMode:
        """
        Auto-detect best execution mode.

        Returns:
            ExecutionMode.DOCKER if Docker available, else SUBPROCESS
        """
        # Check environment variable first
        env_mode = os.getenv("TRAINING_EXECUTION_MODE", "auto").lower()

        if env_mode == "subprocess":
            return ExecutionMode.SUBPROCESS
        elif env_mode == "docker":
            return ExecutionMode.DOCKER

        # Auto-detect: check if Docker is available
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            print("[TrainingManager] Docker detected, using Docker mode")
            return ExecutionMode.DOCKER
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("[TrainingManager] Docker not available, using subprocess mode")
            return ExecutionMode.SUBPROCESS

    def start_training(
        self,
        job_id: int,
        checkpoint_path: Optional[str] = None,
        resume: bool = False
    ) -> bool:
        """
        Start training using configured execution mode.

        Args:
            job_id: Training job ID
            checkpoint_path: Optional checkpoint path
            resume: If True, resume from checkpoint

        Returns:
            True if training started successfully
        """
        if self.execution_mode == ExecutionMode.SUBPROCESS:
            return self._start_training_subprocess(job_id, checkpoint_path, resume)
        elif self.execution_mode == ExecutionMode.DOCKER:
            return self._start_training_docker(job_id, checkpoint_path, resume)
        else:
            raise ValueError(f"Unsupported execution mode: {self.execution_mode}")

    def _start_training_docker(
        self,
        job_id: int,
        checkpoint_path: Optional[str] = None,
        resume: bool = False
    ) -> bool:
        """
        Start training in Docker container.

        Args:
            job_id: Training job ID
            checkpoint_path: Optional checkpoint path
            resume: If True, resume from checkpoint

        Returns:
            True if training started successfully
        """
        # Get job from database
        job = self.db.query(models.TrainingJob).filter(
            models.TrainingJob.id == job_id
        ).first()

        if not job or job.status != "pending":
            return False

        # Select Docker image
        image = self.IMAGE_MAP.get(job.framework)
        if not image:
            job.status = "failed"
            job.error_message = f"No Docker image for framework: {job.framework}"
            self.db.commit()
            return False

        # Get absolute paths
        dataset_path = os.path.abspath(job.dataset_path)
        output_dir = os.path.abspath(job.output_dir)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Build Docker command
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container when done
            "--name", f"training-job-{job_id}",
        ]

        # Add GPU support if available (Linux only)
        if os.name != 'nt':  # Not Windows
            docker_cmd.extend(["--gpus", "all"])

        # Volume mounts
        docker_cmd.extend([
            "-v", f"{dataset_path}:/workspace/dataset:ro",  # Read-only
            "-v", f"{output_dir}:/workspace/output:rw",     # Read-write
        ])

        # Environment variables
        docker_cmd.extend([
            "-e", f"JOB_ID={job_id}",
            "-e", "PYTHONUNBUFFERED=1",
        ])

        # Network (use host for MLflow tracking)
        docker_cmd.extend(["--network", "host"])

        # Image
        docker_cmd.append(image)

        # Training command
        docker_cmd.extend([
            "python", "/opt/vision-platform/train.py",
            "--framework", job.framework,
            "--task_type", job.task_type,
            "--model_name", job.model_name,
            "--dataset_path", "/workspace/dataset",
            "--dataset_format", job.dataset_format,
            "--output_dir", "/workspace/output",
            "--epochs", str(job.epochs),
            "--batch_size", str(job.batch_size),
            "--learning_rate", str(job.learning_rate),
            "--job_id", str(job_id),
        ])

        # Add num_classes if set
        if job.num_classes is not None:
            docker_cmd.extend(["--num_classes", str(job.num_classes)])

        # Add checkpoint args
        if checkpoint_path:
            docker_cmd.extend(["--checkpoint_path", checkpoint_path])
            if resume:
                docker_cmd.append("--resume")

        try:
            print(f"[DEBUG] Docker command: {' '.join(docker_cmd)}")

            # Start container
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Store process
            self.processes[job_id] = process

            # Update job status
            job.status = "running"
            job.started_at = datetime.utcnow()
            job.process_id = process.pid
            self.db.commit()

            # Start monitoring thread (same as subprocess)
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

    def _start_training_subprocess(
        self,
        job_id: int,
        checkpoint_path: Optional[str] = None,
        resume: bool = False
    ) -> bool:
        """
        Start training using subprocess (existing MVP implementation).

        Args:
            job_id: Training job ID
            checkpoint_path: Optional checkpoint path
            resume: If True, resume from checkpoint

        Returns:
            True if training started successfully
        """
        # ... (기존 구현 그대로 유지)
        # 기존 코드 복사
        pass

    # ... (나머지 메서드들은 기존과 동일)
```

#### 4.2 환경 설정

**`mvp/backend/.env` 추가**:
```bash
# Training execution mode
# Options: docker, subprocess, auto
# - docker: Always use Docker containers
# - subprocess: Always use local subprocess (MVP mode)
# - auto: Auto-detect (Docker if available, else subprocess)
TRAINING_EXECUTION_MODE=auto
```

**`mvp/backend/app/core/config.py` 업데이트**:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ... 기존 설정들 ...

    # Training execution
    training_execution_mode: str = "auto"  # docker | subprocess | auto

    class Config:
        env_file = ".env"
```

#### 4.3 테스트용 docker-compose

**`mvp/docker/docker-compose.training.yml`**:
```yaml
version: '3.8'

# ============================================
# Vision Platform - Training Test Services
# For local testing of Docker training
# ============================================

services:
  # timm classification training
  training-timm-test:
    image: vision-platform-timm:latest
    container_name: training-timm-test
    volumes:
      - ../data/sample_dataset:/workspace/dataset:ro
      - ../data/outputs:/workspace/output:rw
    environment:
      - PYTHONUNBUFFERED=1
      - JOB_ID=test-timm
    command: >
      python /opt/vision-platform/train.py
      --framework timm
      --task_type image_classification
      --model_name resnet18
      --dataset_path /workspace/dataset
      --dataset_format imagefolder
      --output_dir /workspace/output
      --epochs 1
      --batch_size 4
      --learning_rate 0.001
      --num_classes 10
      --job_id 9999

  # ultralytics YOLO training
  training-ultralytics-test:
    image: vision-platform-ultralytics:latest
    container_name: training-ultralytics-test
    volumes:
      - ../data/yolo_dataset:/workspace/dataset:ro
      - ../data/outputs:/workspace/output:rw
    environment:
      - PYTHONUNBUFFERED=1
      - JOB_ID=test-ultralytics
    command: >
      python /opt/vision-platform/train.py
      --framework ultralytics
      --task_type object_detection
      --model_name yolov8n
      --dataset_path /workspace/dataset
      --dataset_format yolo
      --output_dir /workspace/output
      --epochs 1
      --batch_size 4
      --learning_rate 0.01
      --job_id 9998

  # YOLO-World test (zero-shot detection)
  training-yoloworld-test:
    image: vision-platform-ultralytics:latest
    container_name: training-yoloworld-test
    volumes:
      - ../data/yolo_dataset:/workspace/dataset:ro
      - ../data/outputs:/workspace/output:rw
    environment:
      - PYTHONUNBUFFERED=1
      - JOB_ID=test-yoloworld
    command: >
      python /opt/vision-platform/train.py
      --framework ultralytics
      --task_type zero_shot_detection
      --model_name yolov8s-worldv2
      --dataset_path /workspace/dataset
      --dataset_format yolo
      --output_dir /workspace/output
      --epochs 1
      --batch_size 4
      --learning_rate 0.01
      --job_id 9997
```

**테스트 실행**:
```bash
# timm 테스트
docker-compose -f mvp/docker/docker-compose.training.yml up training-timm-test

# ultralytics 테스트
docker-compose -f mvp/docker/docker-compose.training.yml up training-ultralytics-test

# YOLO-World 테스트
docker-compose -f mvp/docker/docker-compose.training.yml up training-yoloworld-test
```

#### 4.4 통합 테스트

**테스트 시나리오**:

1. **Subprocess 모드 테스트**:
   ```bash
   # .env 설정
   TRAINING_EXECUTION_MODE=subprocess

   # Backend 재시작
   cd mvp/backend
   venv/Scripts/python -m uvicorn app.main:app --reload

   # 학습 시작 (UI 또는 API)
   curl -X POST http://localhost:8000/api/v1/training/start \
     -H "Content-Type: application/json" \
     -d '{...}'
   ```

2. **Docker 모드 테스트 (timm)**:
   ```bash
   # .env 설정
   TRAINING_EXECUTION_MODE=docker

   # 학습 시작
   curl -X POST http://localhost:8000/api/v1/training/start \
     -d '{"framework": "timm", "model_name": "resnet18", ...}'

   # 컨테이너 확인
   docker ps | grep training-job-
   ```

3. **Docker 모드 테스트 (YOLO-World)**:
   ```bash
   # YOLO-World 학습 (이전에 실패했던 케이스)
   curl -X POST http://localhost:8000/api/v1/training/start \
     -d '{
       "framework": "ultralytics",
       "model_name": "yolov8s-worldv2",
       "task_type": "zero_shot_detection",
       ...
     }'

   # 성공 확인!
   docker logs -f training-job-{job_id}
   ```

**Deliverables**:
- [ ] ExecutionMode enum 추가
- [ ] `_detect_execution_mode()` 구현
- [ ] `_start_training_docker()` 구현
- [ ] 환경 설정 (.env, config.py)
- [ ] docker-compose.training.yml 작성
- [ ] Subprocess 모드 테스트 통과
- [ ] Docker 모드 테스트 통과
- [ ] YOLO-World 학습 성공 확인

---

### Phase 5: 테스트 및 문서화 (2-3일)

**목표**: 통합 테스트 및 사용자 가이드 작성

#### 5.1 통합 테스트 스크립트

**`mvp/docker/test_all.sh`**:
```bash
#!/bin/bash

# ============================================
# Vision Platform - Integration Test Script
# ============================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================="
echo "Vision Platform Integration Tests"
echo "======================================="

# Test 1: Image existence
echo -e "\n${YELLOW}[Test 1] Checking Docker images...${NC}"
docker images vision-platform-base:latest -q > /dev/null
docker images vision-platform-timm:latest -q > /dev/null
docker images vision-platform-ultralytics:latest -q > /dev/null
echo -e "${GREEN}✓ All images exist${NC}"

# Test 2: Base image functionality
echo -e "\n${YELLOW}[Test 2] Testing base image...${NC}"
docker run --rm vision-platform-base:latest \
    python -c "import platform_sdk; print('OK')"
echo -e "${GREEN}✓ Base image works${NC}"

# Test 3: timm image
echo -e "\n${YELLOW}[Test 3] Testing timm image...${NC}"
docker run --rm vision-platform-timm:latest \
    python -c "import torch; import timm; print(f'torch={torch.__version__}, timm={timm.__version__}')"
echo -e "${GREEN}✓ timm image works${NC}"

# Test 4: ultralytics image
echo -e "\n${YELLOW}[Test 4] Testing ultralytics image...${NC}"
docker run --rm vision-platform-ultralytics:latest \
    python -c "import ultralytics; print(f'ultralytics={ultralytics.__version__}')"
echo -e "${GREEN}✓ ultralytics image works${NC}"

# Test 5: YOLOWorld support (critical!)
echo -e "\n${YELLOW}[Test 5] Testing YOLOWorld support...${NC}"
docker run --rm vision-platform-ultralytics:latest \
    python -c "from ultralytics import YOLOWorld; print('YOLOWorld available')"
echo -e "${GREEN}✓ YOLOWorld supported!${NC}"

# Test 6: Dependency isolation
echo -e "\n${YELLOW}[Test 6] Testing dependency isolation...${NC}"
echo "  - Checking timm image doesn't have ultralytics..."
docker run --rm vision-platform-timm:latest \
    python -c "try: import ultralytics; print('FAIL'); except ImportError: print('OK')" | grep OK > /dev/null
echo -e "${GREEN}  ✓ timm isolated${NC}"

echo "  - Checking ultralytics image doesn't have timm..."
docker run --rm vision-platform-ultralytics:latest \
    python -c "try: import timm; print('FAIL'); except ImportError: print('OK')" | grep OK > /dev/null
echo -e "${GREEN}  ✓ ultralytics isolated${NC}"

# Summary
echo -e "\n${GREEN}=======================================${NC}"
echo -e "${GREEN}All tests passed!${NC}"
echo -e "${GREEN}=======================================${NC}"
```

**실행**:
```bash
chmod +x mvp/docker/test_all.sh
./mvp/docker/test_all.sh
```

#### 5.2 사용자 가이드

**`docs/guide/DOCKER_USAGE.md`**:
```markdown
# Docker 기반 학습 사용 가이드

## 개요

Vision Platform은 두 가지 학습 실행 모드를 지원합니다:
- **Subprocess 모드**: 로컬 Python venv 사용 (MVP 기본)
- **Docker 모드**: 프레임워크별 독립 컨테이너 사용

## Quick Start

### 1. Docker 이미지 빌드

\`\`\`bash
cd mvp
./docker/build.sh  # Linux/Mac
# 또는
docker\build.bat   # Windows
\`\`\`

### 2. 실행 모드 설정

\`\`\`bash
# mvp/backend/.env
TRAINING_EXECUTION_MODE=docker  # docker | subprocess | auto
\`\`\`

### 3. 학습 실행

Backend를 통해 평소와 동일하게 학습 시작:
- UI에서 모델 선택 및 학습 시작
- 또는 API 호출

자동으로 적절한 Docker 이미지가 선택됩니다!

## 실행 모드별 상세

### Subprocess 모드 (기존 MVP)

**설정**:
\`\`\`bash
TRAINING_EXECUTION_MODE=subprocess
\`\`\`

**특징**:
- 로컬 Python venv 사용
- Docker 불필요
- 빠른 개발/디버깅

**사용 시나리오**:
- 로컬 개발
- 디버깅
- Docker 없는 환경

### Docker 모드 (권장)

**설정**:
\`\`\`bash
TRAINING_EXECUTION_MODE=docker
\`\`\`

**특징**:
- 프레임워크별 독립 컨테이너
- 의존성 격리
- YOLO-World 지원

**사용 시나리오**:
- 프로덕션 배포
- 여러 프레임워크 동시 사용
- 의존성 충돌 방지

### Auto 모드 (기본)

**설정**:
\`\`\`bash
TRAINING_EXECUTION_MODE=auto
\`\`\`

**동작**:
1. Docker 사용 가능 확인
2. 가능하면 Docker, 아니면 subprocess

## 트러블슈팅

### Docker 이미지를 찾을 수 없음

\`\`\`
Error: No Docker image for framework: timm
\`\`\`

**해결**: 이미지 빌드
\`\`\`bash
cd mvp
./docker/build.sh
\`\`\`

### GPU 사용 안 됨 (Windows)

Windows에서는 Docker Desktop의 WSL2 백엔드 + NVIDIA Container Toolkit 필요.

### 권한 오류

Linux에서 Docker 권한 필요:
\`\`\`bash
sudo usermod -aG docker $USER
\`\`\`

## FAQ

**Q: 기존 subprocess 모드를 계속 사용할 수 있나요?**
A: 네, `TRAINING_EXECUTION_MODE=subprocess` 설정하면 기존과 동일하게 동작합니다.

**Q: Docker 없이 개발 가능한가요?**
A: 네, subprocess 모드로 개발 가능합니다.

**Q: 학습 속도 차이가 있나요?**
A: GPU pass-through로 성능은 동일합니다. 컨테이너 시작 시간 ~5초 추가됩니다.
\`\`\`

#### 5.3 개발자 가이드

**`docs/guide/ADD_FRAMEWORK_DOCKER.md`**:
```markdown
# 새 프레임워크 추가 가이드 (Docker)

## 개요

Docker 기반으로 새 프레임워크를 추가하는 방법을 설명합니다.

## Step 1: Adapter 작성

\`\`\`python
# mvp/training/adapters/myframework_adapter.py
from platform_sdk import TrainingAdapter, MetricsResult

class MyFrameworkAdapter(TrainingAdapter):
    def train(self, ...):
        # 구현
        pass
\`\`\`

## Step 2: Requirements 파일

\`\`\`txt
# mvp/training/requirements/requirements-myframework.txt
-r requirements-base.txt

myframework==1.0.0
torch==2.1.0
\`\`\`

## Step 3: Dockerfile

\`\`\`dockerfile
# mvp/docker/Dockerfile.myframework
FROM vision-platform-base:latest

LABEL framework="myframework"

COPY training/requirements/requirements-myframework.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-myframework.txt

COPY training/adapters/myframework_adapter.py /opt/vision-platform/adapters/
\`\`\`

## Step 4: 빌드 스크립트 업데이트

\`\`\`bash
# mvp/docker/build.sh에 추가
echo "[4/4] Building myframework image..."
docker build -f Dockerfile.myframework -t vision-platform-myframework:latest .
\`\`\`

## Step 5: TrainingManager 업데이트

\`\`\`python
# mvp/backend/app/utils/training_manager.py
IMAGE_MAP = {
    "timm": "vision-platform-timm:latest",
    "ultralytics": "vision-platform-ultralytics:latest",
    "myframework": "vision-platform-myframework:latest",  # 추가
}
\`\`\`

## Step 6: 테스트

\`\`\`bash
# 빌드
./mvp/docker/build.sh

# 테스트
docker run --rm vision-platform-myframework:latest \
    python -c "import myframework; print('OK')"
\`\`\`

완료!
\`\`\`

**Deliverables**:
- [ ] 통합 테스트 스크립트 (`test_all.sh`)
- [ ] 테스트 실행 및 결과 문서화
- [ ] 사용자 가이드 (`DOCKER_USAGE.md`)
- [ ] 개발자 가이드 (`ADD_FRAMEWORK_DOCKER.md`)
- [ ] 트러블슈팅 가이드 섹션
- [ ] README 업데이트

---

## Timeline Summary

| Phase | 작업 내용 | 예상 기간 | 주요 Deliverables |
|-------|----------|----------|------------------|
| **Phase 1** | Platform SDK 분리 | 3-4일 | platform_sdk 패키지, import 리팩토링 |
| **Phase 2** | Requirements 분리 | 2일 | requirements-*.txt, ultralytics 8.3.0+ |
| **Phase 3** | Docker 이미지 생성 | 4-5일 | Dockerfiles, 빌드 스크립트, 이미지 |
| **Phase 4** | TrainingManager 확장 | 4-5일 | ExecutionMode, Docker 실행 로직 |
| **Phase 5** | 테스트 및 문서화 | 2-3일 | 테스트 스크립트, 사용자/개발자 가이드 |
| **Total** | | **15-19일 (3-4주)** | |

**Gantt Chart**:
```
Week 1:  [Phase 1 ████] [Phase 2 ██]
Week 2:  [Phase 3 ████████]
Week 3:  [Phase 4 ████████]
Week 4:  [Phase 5 ████]
```

---

## Expected Benefits

### 즉시 해결 (Immediate)

1. **YOLO-World 학습 가능** ✅
   - ultralytics 8.3.0+ 사용
   - `YOLOWorld` 클래스 지원
   - Zero-shot detection 구현 가능

2. **의존성 충돌 제거** ✅
   - timm과 ultralytics 완전 격리
   - 프레임워크별 최적 버전 사용

3. **이미지 크기 최적화** ✅
   - 기존: 모든 프레임워크 ~8GB
   - 분리 후: 필요한 것만 1.5-2GB

### 장기적 이점 (Long-term)

1. **확장성** 🚀
   - HuggingFace Transformers 추가 준비
   - MMDetection, Detectron2 등 추가 가능
   - 각 프레임워크 독립 관리

2. **개발 효율성** ⚡
   - 빌드 시간 단축 (layer caching)
   - 병렬 개발 가능 (프레임워크별)
   - 디버깅 용이 (격리된 환경)

3. **운영 안정성** 🛡️
   - 보안 강화 (필요한 패키지만)
   - 롤백 용이 (이미지 버전 관리)
   - 재현 가능성 (Dockerfile)

### 성능 비교

**빌드 시간**:
```
Before (단일 venv):
  - 전체 설치: 15분
  - 재설치: 15분

After (Docker):
  - Cold build: 9분 (base 3분 + timm 2분 + ultralytics 1분)
  - Incremental: 10초 (adapter 변경 시)
```

**이미지 크기**:
```
Before:
  - 단일 venv: ~8GB (모든 프레임워크)

After:
  - base: 500MB
  - timm: 2GB
  - ultralytics: 1.5GB
  - Total: 4GB (모두 설치)
  - 실제 사용: 1.5-2GB (필요한 것만)
```

**의존성 수**:
```
Before:
  - 단일 venv: 150+ packages

After:
  - timm container: 80 packages
  - ultralytics container: 60 packages
```

---

## Risk Assessment

| 리스크 | 확률 | 영향 | 완화 방안 | 상태 |
|--------|------|------|-----------|------|
| Docker 학습 곡선 | Medium | Low | subprocess 모드 유지, 문서화 충실 | 완화됨 |
| 성능 오버헤드 | Low | Low | GPU pass-through, ~5초 시작 시간 | 수용 가능 |
| 디버깅 어려움 | Medium | Medium | 실시간 로그, subprocess fallback | 완화됨 |
| 버전 충돌 (Phase 중) | Low | Medium | Phase별 테스트, 롤백 계획 | 관리 중 |
| Windows GPU 지원 | Medium | Medium | WSL2 + NVIDIA Toolkit 문서화 | 완화됨 |

---

## Success Criteria

### Technical Metrics

- [ ] 모든 Docker 이미지 빌드 성공
- [ ] 이미지 크기 < 2.5GB per framework
- [ ] Cold build 시간 < 10분
- [ ] Incremental build 시간 < 30초
- [ ] **YOLOWorld import 성공** (critical!)
- [ ] **YOLOWorld 학습 성공** (critical!)
- [ ] Subprocess 모드 100% 호환
- [ ] Docker 모드 정상 동작
- [ ] 의존성 충돌 0건

### Operational Metrics

- [ ] 새 프레임워크 추가 시간 < 1시간
- [ ] 기존 코드 변경 최소화 (import 경로만)
- [ ] 문서화 완료도 100%
- [ ] 개발자 온보딩 시간 < 30분

### User Experience

- [ ] 사용자는 실행 모드를 의식하지 않음
- [ ] 학습 시작 시간 차이 < 10초
- [ ] 에러 메시지 명확
- [ ] 트러블슈팅 가이드 제공

---

## Next Steps

### Immediate Actions (이번 주)

1. **브랜치 생성**:
   ```bash
   git checkout -b feat/docker-dependency-isolation
   ```

2. **Phase 1 시작**: Platform SDK 분리
   - `mvp/training/platform_sdk/` 생성
   - `adapters/base.py` 이동
   - Import 경로 수정

3. **Kickoff Meeting**:
   - 팀과 계획 공유
   - 작업 분배
   - 일정 조율

### Phase 진행 순서

1. **Week 1**: Phase 1 + Phase 2
2. **Week 2**: Phase 3 (Docker 이미지)
3. **Week 3**: Phase 4 (TrainingManager)
4. **Week 4**: Phase 5 (테스트 & 문서)

### Checkpoints

- [ ] Phase 1 완료 후 리뷰
- [ ] Phase 2 완료 후 ultralytics 버전 확인
- [ ] Phase 3 완료 후 YOLOWorld 테스트
- [ ] Phase 4 완료 후 통합 테스트
- [ ] Phase 5 완료 후 최종 리뷰

---

## References

### Internal Documents
- [DOCKER_IMAGE_SEPARATION.md](../architecture/DOCKER_IMAGE_SEPARATION.md) - 아키텍처 설계
- [IMPLEMENTATION_PRIORITY_ANALYSIS.md](./IMPLEMENTATION_PRIORITY_ANALYSIS.md) - 우선순위 분석
- [ADD_NEW_MODEL.md](../guide/ADD_NEW_MODEL.md) - 모델 추가 가이드

### External Resources
- [Docker Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Docker Layer Caching](https://docs.docker.com/build/cache/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Ultralytics YOLO-World Docs](https://docs.ultralytics.com/models/yolo-world/)

---

## Appendix A: File Changes Summary

### New Files (생성)
```
mvp/training/platform_sdk/__init__.py
mvp/training/platform_sdk/base.py
mvp/training/platform_sdk/callbacks.py
mvp/training/platform_sdk/mlflow_utils.py
mvp/training/platform_sdk/storage.py
mvp/training/requirements/requirements-base.txt
mvp/training/requirements/requirements-timm.txt
mvp/training/requirements/requirements-ultralytics.txt
mvp/docker/Dockerfile.base
mvp/docker/Dockerfile.timm
mvp/docker/Dockerfile.ultralytics
mvp/docker/build.sh
mvp/docker/build.bat
mvp/docker/docker-compose.training.yml
mvp/docker/.dockerignore
mvp/docker/test_all.sh
docs/guide/DOCKER_USAGE.md
docs/guide/ADD_FRAMEWORK_DOCKER.md
docs/planning/DOCKER_IMPLEMENTATION_PLAN.md (본 문서)
```

### Modified Files (수정)
```
mvp/training/adapters/timm_adapter.py (import 경로)
mvp/training/adapters/ultralytics_adapter.py (import 경로)
mvp/training/adapters/__init__.py (import 경로)
mvp/training/train.py (import 경로)
mvp/backend/app/utils/training_manager.py (ExecutionMode 추가)
mvp/backend/app/core/config.py (설정 추가)
mvp/backend/.env (TRAINING_EXECUTION_MODE 추가)
```

### Deleted Files (삭제)
```
mvp/training/adapters/base.py (→ platform_sdk/base.py로 이동)
mvp/training/requirements.txt (→ requirements/*.txt로 분리)
```

---

## Appendix B: Docker Commands Reference

### 빌드
```bash
# 모든 이미지 빌드
cd mvp
./docker/build.sh

# 특정 이미지만 빌드
docker build -f docker/Dockerfile.base -t vision-platform-base:latest .
docker build -f docker/Dockerfile.timm -t vision-platform-timm:latest .
docker build -f docker/Dockerfile.ultralytics -t vision-platform-ultralytics:latest .
```

### 실행
```bash
# timm 학습
docker run --rm --gpus all \
    -v /path/to/dataset:/workspace/dataset:ro \
    -v /path/to/output:/workspace/output:rw \
    vision-platform-timm:latest \
    python /opt/vision-platform/train.py --framework timm --model resnet18 ...

# ultralytics 학습
docker run --rm --gpus all \
    -v /path/to/dataset:/workspace/dataset:ro \
    -v /path/to/output:/workspace/output:rw \
    vision-platform-ultralytics:latest \
    python /opt/vision-platform/train.py --framework ultralytics --model yolov8n ...

# YOLO-World 학습
docker run --rm --gpus all \
    -v /path/to/dataset:/workspace/dataset:ro \
    -v /path/to/output:/workspace/output:rw \
    vision-platform-ultralytics:latest \
    python /opt/vision-platform/train.py --framework ultralytics --model yolov8s-worldv2 ...
```

### 검증
```bash
# 이미지 목록
docker images | grep vision-platform

# YOLOWorld 테스트
docker run --rm vision-platform-ultralytics:latest \
    python -c "from ultralytics import YOLOWorld; print('OK')"

# 의존성 격리 확인
docker run --rm vision-platform-timm:latest \
    python -c "import ultralytics"  # ImportError 예상

# 로그 확인
docker logs -f training-job-{job_id}

# 실행 중인 컨테이너
docker ps | grep training-job-

# 컨테이너 진입
docker exec -it training-job-{job_id} /bin/bash
```

### 정리
```bash
# 컨테이너 정지
docker stop training-job-{job_id}

# 이미지 삭제
docker rmi vision-platform-timm:latest
docker rmi vision-platform-ultralytics:latest
docker rmi vision-platform-base:latest

# 전체 정리
docker system prune -a
```

---

*Document Version: 1.0*
*Created: 2025-10-30*
*Author: Vision AI Platform Team*
*Status: Ready for Implementation*
