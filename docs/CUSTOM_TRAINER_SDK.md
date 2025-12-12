# Custom Trainer SDK Guide

Vision AI Training Platform에서 커스텀 학습 이미지를 개발하기 위한 가이드입니다.

## 목차

1. [Overview](#overview)
2. [필수 요구사항](#필수-요구사항)
3. [TrainerSDK 사용법](#trainersdk-사용법)
4. [환경 변수](#환경-변수)
5. [Volume Mount Paths (K8s Mode)](#volume-mount-paths-k8s-mode)
6. [Lifecycle Callbacks](#lifecycle-callbacks)
7. [Storage 연동](#storage-연동)
8. [Exit Codes](#exit-codes)
9. [Dockerfile 템플릿](#dockerfile-템플릿)
10. [전체 예제](#전체-예제)
11. [Checklist](#checklist)

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (Platform)                            │
├─────────────────────────────────────────────────────────────────┤
│  1. TrainingJob 생성                                             │
│  2. Kubernetes Job 생성 (환경변수 주입)                           │
│  3. Callback 수신 (progress, completion, failure)                │
│  4. Checkpoint 관리 (S3)                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Environment Variables
                            │ (JOB_ID, CALLBACK_URL, ...)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                Custom Trainer Container                          │
├─────────────────────────────────────────────────────────────────┤
│  trainer_sdk.py (복사해서 사용)                                   │
│      ↓                                                          │
│  train.py (개발자 구현)                                          │
│      - SDK 초기화                                                │
│      - report_started()                                         │
│      - 데이터셋 다운로드                                          │
│      - 학습 루프 (매 epoch report_progress())                    │
│      - 체크포인트 업로드                                          │
│      - report_completed() 또는 report_failed()                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 필수 요구사항

### 1. Entry Point

컨테이너는 반드시 `train.py`를 entry point로 가져야 합니다:

```dockerfile
# Dockerfile
CMD ["python", "train.py"]
```

### 2. trainer_sdk.py 포함

`platform/trainers/ultralytics/trainer_sdk.py`를 복사하여 사용합니다.
이 파일은 Backend와의 통신을 담당하는 단일 파일 SDK입니다.

```dockerfile
COPY trainer_sdk.py ./
COPY train.py ./
```

### 3. 필수 의존성

```
httpx>=0.24.0      # HTTP 클라이언트 (callback 전송)
boto3>=1.28.0      # S3 스토리지 (데이터셋/체크포인트)
PyYAML>=6.0        # 설정 파일 파싱
python-dotenv      # 환경 변수 로딩
```

---

## TrainerSDK 사용법

### 기본 사용 패턴

```python
from trainer_sdk import TrainerSDK, ErrorType

def main():
    # 1. SDK 초기화 (환경변수에서 자동으로 설정 로드)
    sdk = TrainerSDK()

    try:
        # 2. 학습 시작 보고
        sdk.report_started('training')

        # 3. 데이터셋 다운로드
        dataset_dir = sdk.download_dataset(dataset_id, '/tmp/dataset')

        # 4. 학습 루프
        for epoch in range(1, epochs + 1):
            metrics = train_one_epoch(...)

            # 매 epoch 진행 상황 보고
            sdk.report_progress(
                epoch=epoch,
                total_epochs=epochs,
                metrics={
                    'loss': metrics['loss'],
                    'accuracy': metrics['accuracy'],
                    'learning_rate': metrics['lr']
                }
            )

        # 5. 체크포인트 업로드
        best_uri = sdk.upload_checkpoint('/tmp/best.pt', 'best')
        last_uri = sdk.upload_checkpoint('/tmp/last.pt', 'last')

        # 6. 학습 완료 보고
        sdk.report_completed(
            final_metrics={'accuracy': 0.95, 'loss': 0.05},
            checkpoints={'best': best_uri, 'last': last_uri},
            total_epochs=epochs
        )

        return 0  # 성공

    except Exception as e:
        # 7. 에러 보고
        sdk.report_failed(
            error_type=ErrorType.TRAINING_ERROR,
            message=str(e),
            traceback=traceback.format_exc()
        )
        return 1  # 실패

    finally:
        sdk.close()
```

---

## 환경 변수

Backend가 컨테이너 실행 시 주입하는 환경 변수들입니다.

### 필수 환경 변수

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `JOB_ID` | 학습 작업 ID | `123` |
| `CALLBACK_URL` | Backend API URL | `http://backend:8000/api/v1` |
| `MODEL_NAME` | 모델 이름 | `resnet50`, `yolo11n` |
| `DATASET_S3_URI` | 데이터셋 S3 URI | `s3://datasets/ds_abc123/` |

### 설정 관련 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `CONFIG` | 학습 설정 JSON | `{}` |
| `EPOCHS` | 학습 에폭 수 | `100` |
| `BATCH_SIZE` | 배치 크기 | `16` |
| `LEARNING_RATE` | 학습률 | `0.01` |
| `IMGSZ` | 이미지 크기 | `640` |
| `DEVICE` | 학습 디바이스 | `0` (GPU) |

### 스토리지 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `EXTERNAL_STORAGE_ENDPOINT` | 데이터셋 S3 endpoint | `http://localhost:9000` |
| `EXTERNAL_STORAGE_ACCESS_KEY` | S3 Access Key | `minioadmin` |
| `EXTERNAL_STORAGE_SECRET_KEY` | S3 Secret Key | `minioadmin` |
| `EXTERNAL_BUCKET_DATASETS` | 데이터셋 버킷 | `training-datasets` |
| `INTERNAL_STORAGE_ENDPOINT` | 체크포인트 S3 endpoint | `http://localhost:9002` |
| `INTERNAL_BUCKET_CHECKPOINTS` | 체크포인트 버킷 | `training-checkpoints` |

### CONFIG JSON 구조

```json
{
  "epochs": 100,
  "batch": 16,
  "imgsz": 640,
  "learning_rate": 0.01,
  "optimizer": "adamw",
  "device": "0",
  "advanced_config": {
    "mosaic": 1.0,
    "mixup": 0.0,
    "warmup_epochs": 3
  }
}
```

SDK에서 설정 로드:

```python
sdk = TrainerSDK()

# 기본 설정 로드
basic_config = sdk.get_basic_config()
# {'epochs': 100, 'batch': 16, 'imgsz': 640, ...}

# 프레임워크별 고급 설정
advanced_config = sdk.get_advanced_config()
# {'mosaic': 1.0, 'mixup': 0.0, ...}

# 전체 설정
full_config = sdk.get_full_config()
# {'basic': {...}, 'advanced': {...}}
```

---

## Volume Mount Paths (K8s Mode)

K8s 모드에서 Backend가 컨테이너에 마운트하는 볼륨입니다.

### 표준 마운트 경로

| 마운트 경로 | 타입 | 용도 | 권장 사용법 |
|-------------|------|------|-------------|
| `/tmp` | emptyDir | 임시 파일 | 데이터셋 다운로드, 임시 파일 |
| `/workspace` | emptyDir | 작업 디렉토리 | 학습 결과, 로그, 체크포인트 |

### 사용 예시

```python
# 데이터셋 다운로드 (권장: /tmp 사용)
dataset_dir = sdk.download_dataset(dataset_id, '/tmp/dataset')

# 체크포인트 저장 (권장: /workspace 사용)
checkpoint_path = '/workspace/checkpoints/best.pt'
torch.save(model.state_dict(), checkpoint_path)

# 체크포인트 업로드
best_uri = sdk.upload_checkpoint(checkpoint_path, 'best')
```

### 주의사항

- **emptyDir**: Pod 종료 시 삭제됨. 영구 저장이 필요한 파일은 반드시 S3에 업로드
- **용량 제한**: 기본 제한 없음 (노드 디스크 공간에 의존)
- **경로 하드코딩 금지**: 환경변수나 설정으로 경로 관리 권장

```python
# ❌ 하드코딩 (비권장)
dataset_dir = '/tmp/dataset'

# ✅ 환경변수 활용 (권장)
import os
WORK_DIR = os.getenv('WORK_DIR', '/workspace')
TMP_DIR = os.getenv('TMP_DIR', '/tmp')
```

### PVC vs emptyDir

현재 기본 설정은 **emptyDir**을 사용합니다:

| 특성 | emptyDir (현재) | PVC |
|------|-----------------|-----|
| 데이터 영속성 | Pod 종료 시 삭제 | 유지 |
| 프로비저닝 | 자동 | 사전 생성 필요 |
| 성능 | 노드 디스크 (SSD/HDD) | 스토리지 클래스에 따름 |
| 용량 관리 | 노드에 의존 | 명시적 할당 |
| 사용 사례 | 대부분의 학습 작업 | 대용량 캐시, 재사용 필요 시 |

> **설계 철학**: S3를 primary storage로 사용하므로 emptyDir로 충분합니다.
> 데이터셋은 S3에서 다운로드, 체크포인트는 S3로 업로드하는 패턴입니다.

---

## Lifecycle Callbacks

Backend에 학습 상태를 보고하는 메서드들입니다.

### 1. report_started()

학습 시작을 알립니다. **반드시 학습 시작 전에 호출해야 합니다.**

```python
sdk.report_started(
    operation_type='training',  # 'training', 'inference', 'export'
    total_epochs=100            # 선택적, 미지정시 CONFIG에서 읽음
)
```

### 2. report_progress()

매 epoch 또는 일정 간격으로 진행 상황을 보고합니다.

```python
sdk.report_progress(
    epoch=5,
    total_epochs=100,
    metrics={
        'loss': 0.234,           # 필수
        'accuracy': 0.85,        # 선택적 (또는 mAP, F1 등)
        'learning_rate': 0.001,  # 선택적
        # 프레임워크별 추가 메트릭
        'mAP50': 0.75,
        'mAP50-95': 0.52,
        'precision': 0.80,
        'recall': 0.78
    },
    extra_data={                 # 선택적: 프레임워크별 raw 데이터
        'raw_metrics': {...}
    }
)
```

### 3. report_validation()

검증 결과를 별도로 보고합니다 (선택적).

```python
sdk.report_validation(
    epoch=50,
    task_type='detection',       # 'detection', 'classification', 'segmentation'
    primary_metric=('mAP50-95', 0.52),
    all_metrics={...},
    class_names=['cat', 'dog'],
    visualization_urls={         # 선택적: 업로드된 시각화 이미지
        'confusion_matrix': 's3://...',
        'pr_curve': 's3://...'
    }
)
```

### 4. report_completed()

학습 성공 완료를 알립니다.

```python
sdk.report_completed(
    final_metrics={
        'loss': 0.05,
        'accuracy': 0.95,
        'mAP50-95': 0.72
    },
    checkpoints={
        'best': 's3://checkpoints/123/best.pt',
        'last': 's3://checkpoints/123/last.pt'
    },
    total_epochs=100
)
```

### 5. report_failed()

학습 실패를 알립니다.

```python
from trainer_sdk import ErrorType

sdk.report_failed(
    error_type=ErrorType.RESOURCE_ERROR,  # 아래 ErrorType 참조
    message='CUDA out of memory',
    traceback=traceback.format_exc(),
    epochs_completed=35
)
```

**ErrorType 종류:**

| ErrorType | 설명 |
|-----------|------|
| `DATASET_ERROR` | 데이터셋 관련 오류 (다운로드 실패, 포맷 오류) |
| `CHECKPOINT_ERROR` | 체크포인트 관련 오류 |
| `CONFIG_ERROR` | 설정 오류 (잘못된 파라미터) |
| `RESOURCE_ERROR` | 리소스 오류 (GPU OOM, 디스크 부족) |
| `NETWORK_ERROR` | 네트워크 오류 (S3 연결 실패) |
| `FRAMEWORK_ERROR` | 프레임워크별 오류 (YOLO, timm 등) |
| `VALIDATION_ERROR` | 검증 오류 (NaN loss 등) |
| `UNKNOWN_ERROR` | 기타 오류 |

---

## Storage 연동

### 데이터셋 다운로드

```python
# 기본 다운로드
dataset_dir = sdk.download_dataset(dataset_id, '/tmp/dataset')

# 선택적 다운로드 (라벨된 이미지만 - 6배 빠름)
dataset_dir = sdk.download_dataset_selective(dataset_id, '/tmp/dataset')

# 캐싱을 활용한 다운로드 (동일 스냅샷 재사용)
dataset_dir = sdk.download_dataset_with_cache(
    snapshot_id='snap_abc123',
    dataset_id='ds_xyz789',
    dataset_version_hash='abc123def456...',
    dest_dir='/tmp/training/123'
)
```

### 데이터셋 포맷 변환

플랫폼은 DICE 포맷(COCO 기반)을 사용합니다. 프레임워크에 맞게 변환하세요:

```python
# DICE → YOLO 변환
sdk.convert_dataset(
    dataset_dir='/tmp/dataset',
    source_format='dice',
    target_format='yolo',
    split_config={
        'train_ratio': 0.8,
        'val_ratio': 0.2,
        'seed': 42
    },
    task_type='detection'  # 'detection', 'classification', 'segmentation'
)
```

### 체크포인트 업로드

```python
# 체크포인트 업로드 (Internal Storage에 저장)
best_uri = sdk.upload_checkpoint('/tmp/weights/best.pt', 'best')
last_uri = sdk.upload_checkpoint('/tmp/weights/last.pt', 'last')
epoch_uri = sdk.upload_checkpoint('/tmp/weights/epoch_50.pt', 'epoch_50')

# 일반 파일 업로드 (결과 이미지 등)
sdk.upload_file(
    '/tmp/confusion_matrix.png',
    f'job{job_id}/validation/confusion_matrix.png',
    content_type='image/png',
    storage_type='internal'
)
```

### 체크포인트 다운로드 (Fine-tuning용)

```python
# 기존 체크포인트에서 시작
checkpoint_path = sdk.download_checkpoint(
    's3://checkpoints/previous_job/best.pt',
    '/tmp/pretrained.pt'
)
model.load_state_dict(torch.load(checkpoint_path))
```

---

## Exit Codes

컨테이너 종료 코드는 Backend에서 상태 판단에 사용됩니다.

| Exit Code | 의미 | 설명 |
|-----------|------|------|
| `0` | Success | 학습 성공 완료 |
| `1` | Training Failure | 학습 중 오류 발생 |
| `2` | Callback Failure | Backend 통신 실패 |

```python
if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
```

---

## Dockerfile 템플릿

```dockerfile
# =========================================
# Custom Trainer Dockerfile Template
# =========================================

# 1. Base Image (프레임워크에 맞게 선택)
FROM python:3.11-slim AS builder

# 2. UV Package Manager 설치 (빠른 의존성 설치)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 3. 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 작업 디렉토리
WORKDIR /app

# 5. 의존성 설치
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

# =========================================
# Production Stage
# =========================================
FROM python:3.11-slim AS production

# 시스템 의존성 (런타임용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 패키지 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 6. SDK 및 코드 복사 (반드시 trainer_sdk.py 포함)
COPY trainer_sdk.py ./
COPY train.py ./
COPY utils.py ./           # 있는 경우
COPY requirements.txt ./   # 있는 경우

# 7. 비특권 사용자 생성
RUN useradd -m -u 1000 appuser
USER appuser

# 8. Entry Point
CMD ["python", "train.py"]
```

### pyproject.toml 예시

```toml
[project]
name = "my-custom-trainer"
version = "1.0.0"
requires-python = ">=3.11"

dependencies = [
    # SDK 필수 의존성
    "httpx>=0.24.0",
    "boto3>=1.28.0",
    "PyYAML>=6.0",
    "python-dotenv>=1.0.0",

    # 프레임워크 의존성 (예시: timm)
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "timm>=0.9.0",
    "tqdm>=4.65.0",
]
```

---

## 전체 예제

### 최소 구현 예제

```python
#!/usr/bin/env python3
"""
Minimal Custom Trainer Example
"""
import os
import sys
import traceback
from trainer_sdk import TrainerSDK, ErrorType


def main():
    sdk = TrainerSDK()

    try:
        # 설정 로드
        config = sdk.get_basic_config()
        epochs = config['epochs']

        # 학습 시작 보고
        sdk.report_started('training', total_epochs=epochs)

        # 데이터셋 다운로드
        dataset_id = os.getenv('DATASET_S3_URI', '').split('/')[-2]
        dataset_dir = sdk.download_dataset(dataset_id, '/tmp/dataset')

        # 데이터셋 변환 (DICE → 사용할 포맷)
        sdk.convert_dataset(dataset_dir, 'dice', 'yolo')

        # =========================================
        # 여기에 실제 학습 코드 구현
        # =========================================

        for epoch in range(1, epochs + 1):
            # 학습 진행...
            loss = 1.0 / epoch  # 예시
            accuracy = epoch / epochs

            # 진행 상황 보고
            sdk.report_progress(
                epoch=epoch,
                total_epochs=epochs,
                metrics={'loss': loss, 'accuracy': accuracy}
            )

        # 체크포인트 저장 (예시)
        # torch.save(model.state_dict(), '/tmp/best.pt')
        # best_uri = sdk.upload_checkpoint('/tmp/best.pt', 'best')

        # 완료 보고
        sdk.report_completed(
            final_metrics={'loss': 0.01, 'accuracy': 0.99},
            checkpoints={},  # {'best': best_uri}
            total_epochs=epochs
        )

        return 0

    except Exception as e:
        sdk.report_failed(
            error_type=ErrorType.UNKNOWN_ERROR,
            message=str(e),
            traceback=traceback.format_exc()
        )
        return 1

    finally:
        sdk.close()


if __name__ == "__main__":
    sys.exit(main())
```

---

## Checklist

커스텀 트레이너 이미지 배포 전 확인사항:

### 필수 항목

- [ ] `trainer_sdk.py` 파일이 컨테이너에 포함되어 있는가?
- [ ] `train.py`가 entry point로 설정되어 있는가?
- [ ] `report_started()`를 학습 시작 전에 호출하는가?
- [ ] `report_progress()`를 매 epoch 또는 일정 간격으로 호출하는가?
- [ ] `report_completed()`를 학습 성공 시 호출하는가?
- [ ] `report_failed()`를 예외 발생 시 호출하는가?
- [ ] Exit code가 올바르게 반환되는가? (0=성공, 1=실패, 2=콜백실패)

### 볼륨 마운트 항목 (K8s Mode)

- [ ] 데이터셋을 `/tmp` 경로에 다운로드하는가?
- [ ] 체크포인트/결과물을 `/workspace` 경로에 저장하는가?
- [ ] 영구 저장이 필요한 파일은 S3에 업로드하는가?
- [ ] 경로를 하드코딩하지 않고 환경변수로 관리하는가?

### 권장 항목

- [ ] 체크포인트를 `sdk.upload_checkpoint()`로 업로드하는가?
- [ ] 데이터셋을 `sdk.download_dataset()`로 다운로드하는가?
- [ ] ErrorType을 적절하게 구분하여 사용하는가?
- [ ] GPU 메모리 오류 시 `ErrorType.RESOURCE_ERROR`를 사용하는가?
- [ ] 로깅이 충분히 되어 있는가? (디버깅용)

### 테스트 항목

- [ ] 로컬에서 환경변수를 설정하고 실행이 되는가?
- [ ] Backend 없이 dry-run 테스트가 가능한가?
- [ ] GPU 없는 환경에서도 CPU로 실행 가능한가?

---

## 문의 및 지원

- 기존 트레이너 참조: `platform/trainers/ultralytics/train.py`
- TrainerSDK 소스: `platform/trainers/ultralytics/trainer_sdk.py`
- 이슈 리포트: GitHub Issues

