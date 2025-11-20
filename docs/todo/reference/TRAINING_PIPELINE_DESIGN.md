# Training Pipeline Design Document

**Date**: 2025-11-20
**Status**: Draft
**Author**: Claude Code

---

## Executive Summary

이 문서는 Vision AI Training Platform의 Training 파이프라인 전체 흐름을 상세하게 정의합니다. Inference와 Export E2E 테스트 완료 후, Training 파이프라인 구현 및 테스트를 위한 기준 문서입니다.

---

## 1. Training Pipeline Overview

### 1.1 High-Level Flow (6 Stages)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. Model   │ →  │ 2. Dataset  │ →  │  3. Config  │
│  Selection  │    │  Selection  │    │   Setting   │
└─────────────┘    └─────────────┘    └─────────────┘
       ↓                                      ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  6. Result  │ ←  │ 5. Monitor  │ ←  │  4. Start   │
│   Review    │    │  Training   │    │  Training   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 1.2 Key Actors

| Actor | Role |
|-------|------|
| **User** | Frontend를 통해 학습 설정 및 모니터링 |
| **Frontend** | UI 제공, API 호출, WebSocket 수신 |
| **Backend** | API 처리, Job 관리, 모니터링 통합 |
| **Trainer** | 실제 학습 실행, SDK를 통해 Backend와 통신 |
| **SDK** | Trainer와 Backend 간 표준 통신 인터페이스 |

---

## 2. Stage 1: Model Selection

> **Scope Note**: Model 조회 기능은 이미 별도 세션에서 구현 완료됨.
> 이 문서는 Training SDK 파이프라인에 집중하므로 상세 내용은 생략합니다.

### 2.1 Summary

- 사용자가 모델 선택 (Frontend에서 model list API 호출)
- Model capability는 S3에 미리 업로드된 JSON 파일로 관리
- 이 단계는 SDK와 무관하며 Backend API만 관여

---

## 3. Stage 2: Dataset Selection

### 3.1 Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 2.1 사용자가 등록된 데이터셋 리스트 조회                              │
│     └─ GET /api/v1/datasets?task_type=detection                   │
│     └─ 데이터셋 업로드는 별도 데이터 서비스에서 처리                  │
│                                                                   │
│ 2.2 백엔드가 데이터셋 정보 제공 (DICE format)                        │
│     └─ Response: { id, name, task_types, image_count, ... }       │
│     └─ DICE format만 취급 (표준 포맷)                               │
│                                                                   │
│ 2.3 사용자가 데이터셋 상세 정보 조회                                 │
│     └─ GET /api/v1/datasets/{id}                                  │
│     └─ Task별 annotation 파일 정보 확인                             │
│                                                                   │
│ 2.4 데이터셋 선택                                                   │
│     └─ User selects dataset_id + task_type in training config     │
│                                                                   │
│ 2.5 Trainer가 DICE format을 자체 format으로 변환                    │
│     └─ DICE → YOLO format (Ultralytics)                           │
│     └─ DICE → ImageFolder (timm)                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 DICE Format Standard

> **Note**: 우리 플랫폼은 **DICE format만** 취급합니다.
> 데이터셋 업로드/관리는 별도 데이터 서비스에서 처리하며, 우리는 등록된 데이터셋을 조회만 합니다.

**DICE Format Structure**:
```
dataset/
├── images/
│   ├── train/
│   │   ├── image_001.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── annotations_classification.json    # Classification task용
├── annotations_detection.json         # Detection task용
├── annotations_segmentation.json      # Segmentation task용
├── annotations_pose.json              # Pose estimation task용
└── metadata.json                      # 데이터셋 메타정보
```

**Task별 Annotation 파일** (신규 사양):
- 기존: `annotations.json` 하나로 모든 task 포함
- 변경: task별로 분리된 annotation 파일
  - `annotations_classification.json`
  - `annotations_detection.json`
  - `annotations_segmentation.json`
  - `annotations_pose.json`

### 3.2.1 Classification Annotation Sample

```json
{
  "format_version": "1.0",
  "dataset_id": "8f97389d-aa20-4de3-9872-d3cf8909a53c",
  "dataset_name": "det-mvtec-ad",
  "task_type": "classification",
  "created_at": "2025-11-18T22:04:10.784593+09:00",
  "last_modified_at": "2025-11-20T07:56:17.818801+09:00",
  "version": 1,
  "classes": [
    {
      "id": 0,
      "name": "bottle_broken_large",
      "color": "#d92653",
      "supercategory": "object"
    },
    {
      "id": 1,
      "name": "bottle_contamination",
      "color": "#d926a6",
      "supercategory": "object"
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "bottle/broken_large/000.png",
      "width": 900,
      "height": 900,
      "depth": 3,
      "split": "train",
      "annotations": [
        {
          "id": 1777,
          "image_id": 1,
          "class_id": 0,
          "class_name": "bottle_broken_large",
          "attributes": {}
        }
      ],
      "metadata": {
        "labeled_by": "admin@example.com",
        "labeled_at": "2025-11-20T07:55:27.892368+09:00",
        "reviewed_by": "admin@example.com",
        "reviewed_at": "2025-11-20T07:56:04.123554+09:00",
        "source": "platform_labeler_v1.0"
      }
    }
  ],
  "statistics": {
    "total_images": 63,
    "total_annotations": 63,
    "avg_annotations_per_image": 1.0,
    "class_distribution": {
      "bottle_broken_large": 20,
      "bottle_broken_small": 22,
      "bottle_contamination": 21
    },
    "split_distribution": {
      "train": 63
    }
  }
}
```

### 3.2.2 Detection Annotation Sample

```json
{
  "format_version": "1.0",
  "dataset_id": "8f97389d-aa20-4de3-9872-d3cf8909a53c",
  "dataset_name": "det-mvtec-ad",
  "task_type": "object_detection",
  "created_at": "2025-11-18T22:04:10.784593+09:00",
  "last_modified_at": "2025-11-20T07:53:59.911606+09:00",
  "version": 1,
  "classes": [
    {
      "id": 0,
      "name": "defect",
      "color": "#d96826",
      "supercategory": "object"
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "bottle/broken_large/000.png",
      "width": 900,
      "height": 900,
      "depth": 3,
      "split": "train",
      "annotations": [
        {
          "id": 1685,
          "image_id": 1,
          "class_id": 0,
          "class_name": "defect",
          "bbox": [351.29, 286.94, 430.0, 512.22],
          "bbox_format": "xywh",
          "area": 220255.55,
          "iscrowd": 0,
          "attributes": {}
        }
      ],
      "metadata": {
        "labeled_by": "admin@example.com",
        "labeled_at": "2025-11-20T05:57:38.582405+09:00",
        "reviewed_by": "admin@example.com",
        "reviewed_at": "2025-11-20T06:02:40.419467+09:00",
        "source": "platform_labeler_v1.0"
      }
    }
  ],
  "statistics": {
    "total_images": 63,
    "total_annotations": 85,
    "avg_annotations_per_image": 1.35,
    "class_distribution": {
      "defect": 85
    },
    "split_distribution": {
      "train": 63
    }
  }
}
```

### 3.2.3 Key Differences by Task Type

| Field | Classification | Detection |
|-------|---------------|-----------|
| `task_type` | `"classification"` | `"object_detection"` |
| `bbox` | ❌ | ✅ `[x, y, w, h]` |
| `bbox_format` | ❌ | ✅ `"xywh"` |
| `area` | ❌ | ✅ bbox 면적 |
| `iscrowd` | ❌ | ✅ crowd annotation 여부 |

### 3.3 Dataset Query API

**API**: `GET /api/v1/datasets`

**Query Parameters**:
- `task_type`: Filter by task type (detection, segmentation, classification, pose)
- `search`: Search by name
- `limit`, `offset`: Pagination

**Response**:
```json
{
  "datasets": [
    {
      "id": 42,
      "name": "vehicles_v1",
      "description": "Vehicle detection dataset",
      "task_types": ["detection", "classification"],
      "total_images": 5000,
      "splits": {
        "train": 4000,
        "val": 800,
        "test": 200
      },
      "annotations": {
        "detection": {
          "file": "annotations_detection.json",
          "classes": ["car", "truck", "bus", "motorcycle"],
          "total_annotations": 15000
        },
        "classification": {
          "file": "annotations_classification.json",
          "classes": ["vehicle", "non-vehicle"],
          "total_annotations": 5000
        }
      },
      "storage_path": "s3://training-datasets/42/",
      "created_at": "2025-11-20T10:00:00Z"
    }
  ],
  "total": 10
}
```

### 3.4 Trainer Format Conversion

각 Trainer는 DICE format을 자체 format으로 변환하는 책임을 가집니다.

**SDK Method**:
```python
class TrainerSDK:
    def download_dataset(self, dataset_id: int, task_type: str) -> str:
        """
        1. S3에서 DICE format 데이터셋 다운로드
        2. task_type에 맞는 annotation 파일 선택
           - detection → annotations_detection.json
           - classification → annotations_classification.json
        3. 로컬 경로 반환
        """
        pass

# Ultralytics trainer에서 사용
def convert_dice_to_yolo(dice_path: str, task_type: str) -> str:
    """
    DICE format → YOLO format 변환

    Input (DICE):
    - images/train/*.jpg
    - annotations_detection.json

    Output (YOLO):
    - images/train/*.jpg
    - labels/train/*.txt
    - data.yaml
    """
    pass
```

### 3.5 Annotation File Changes (from Labeler Team)

> **Important**: 레이블러 팀의 새로운 annotation 사양을 반영해야 합니다.

**기존 방식**:
```
dataset/
└── annotations.json    # 모든 task의 annotation 포함
```

**변경된 방식**:
```
dataset/
├── annotations_classification.json
├── annotations_detection.json
├── annotations_segmentation.json
└── annotations_pose.json
```

**영향**:
1. **Backend**: 데이터셋 조회 시 task별 annotation 파일 정보 제공
2. **SDK**: task_type에 따라 적절한 annotation 파일 다운로드
3. **Trainer**: 해당 task의 annotation만 사용하여 format 변환

### 3.4 Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Dataset upload API | ✅ Implemented | Works with Dual Storage |
| Dataset analysis | ✅ Implemented | Auto-detect format |
| Dataset list API | ✅ Implemented | With filtering |
| Annotation integration | ❌ Not implemented | Need labeler team input |

---

## 4. Stage 3: Configuration Setting

> **Scope Note**: ConfigSchema 조회 및 Dynamic UI 표시 기능은 이미 별도 세션에서 구현 완료됨.
> 이 문서는 Training SDK 파이프라인에 집중하므로, **Config → Backend → Trainer 환경변수 주입** 부분만 상세히 다룹니다.

### 4.1 Summary (Already Implemented)

- 사용자가 모델 선택 시 ConfigSchema 조회 (GET `/api/v1/models/{model_name}/config-schema`)
- Frontend가 ConfigSchema 기반으로 Dynamic UI 생성
- 사용자가 UI에서 config 설정 (imgsz, epochs, batch, lr, etc.)
- 이 부분은 SDK와 무관하며 Backend API만 관여

### 4.2 Config Flow: UI → Backend → Trainer (SDK 관련)

Training Job 생성 시 config가 어떻게 Trainer에 전달되는지 정의합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│ 4.2.1 Frontend가 Training Job 생성 요청                           │
│       └─ POST /api/v1/training/jobs                               │
│       └─ Request body에 config 포함                               │
│                                                                   │
│ 4.2.2 Backend가 TrainingJob 레코드 생성                            │
│       └─ DB에 config JSON 저장                                    │
│       └─ job_id 생성                                              │
│                                                                   │
│ 4.2.3 Backend가 Trainer subprocess 시작                           │
│       └─ 환경변수로 config 전달                                    │
│       └─ 또는 config JSON 파일로 전달                              │
│                                                                   │
│ 4.2.4 Trainer가 환경변수/파일에서 config 파싱                       │
│       └─ SDK가 자동으로 config 로드                                │
│       └─ Framework 네이티브 config로 변환                          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Config Types: Basic vs Advanced

Config는 두 가지 타입으로 구분됩니다:

| Type | Description | Examples | 특징 |
|------|-------------|----------|------|
| **Basic Config** | 모든 trainer에 공통으로 적용되는 기본 파라미터 | `imgsz`, `epochs`, `batch`, `lr0`, `optimizer` | SDK가 표준 인터페이스로 제공 |
| **Advanced Config** | Trainer별로 다르게 정의된 framework-specific 파라미터 | Ultralytics: `mosaic`, `mixup`, `cos_lr`, `nms` | Trainer가 직접 파싱 |

**Frontend에서의 구분**:
```json
// Training Job 생성 요청
{
  "model_name": "yolo11n",
  "dataset_id": 42,
  "task_type": "detection",
  "config": {
    // Basic Config (모든 trainer 공통)
    "imgsz": 640,
    "epochs": 100,
    "batch": 16,
    "lr0": 0.01,
    "optimizer": "SGD"
  },
  "advanced_config": {
    // Advanced Config (Ultralytics-specific)
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cos_lr": false,
    "nms": 0.7,
    "conf": 0.001,
    "iou": 0.7,
    "multi_scale": false
  }
}
```

### 4.4 Config Injection: Backend → Trainer

#### 4.4.1 Updated Design: Environment Variables vs CONFIG JSON (2025-11-20)

**Design Principle**:
- **환경변수**: 단순 값 (string, int, float), 모든 Trainer 공통, 변경 빈도 낮음
- **CONFIG JSON**: 중첩 구조, Trainer별 다름, 선택적 필드 많음

**환경변수 전달 구조**:
```python
# Backend에서 subprocess 시작 시
env = {
    # ===== 환경변수: 단순 공통 값 =====
    # Job 메타정보
    "JOB_ID": str(job.id),
    "CALLBACK_URL": settings.backend_url,
    "MODEL_NAME": job.model_name,
    "TASK_TYPE": job.task_type,
    "FRAMEWORK": job.framework,
    "DATASET_ID": str(job.dataset_id),
    "DATASET_S3_URI": dataset_s3_uri,

    # Basic Config (공통 파라미터)
    "EPOCHS": str(config.epochs),
    "BATCH_SIZE": str(config.batch_size),
    "LEARNING_RATE": str(config.learning_rate),
    "IMGSZ": str(config.get("imgsz", 640)),
    "DEVICE": str(config.get("device", "0")),

    # ===== CONFIG JSON: 복잡한 Trainer별 설정 =====
    "CONFIG": json.dumps({
        "advanced_config": config.get("advanced_config", {}),
        "primary_metric": config.get("primary_metric"),
        "primary_metric_mode": config.get("primary_metric_mode", "max"),
        "custom_prompts": config.get("custom_prompts"),
        "split_strategy": config.get("split_strategy"),
        # ... 기타 복잡한 설정
    }),
    # 예: '{"advanced_config": {"mosaic": 1.0, "mixup": 0.15, ...}, "primary_metric": "mAP50-95"}'

    # Storage 설정
    "EXTERNAL_STORAGE_ENDPOINT": settings.external_storage_endpoint,
    "EXTERNAL_STORAGE_ACCESS_KEY": settings.external_storage_access_key,
    # ... 기타 Storage 환경변수
}

subprocess.run(["python", "train.py"], env=env)
```

**분류 기준**:

| 환경변수 (단순) | CONFIG JSON (복잡) |
|----------------|-------------------|
| `JOB_ID` | `advanced_config` dict |
| `TASK_TYPE` | `augmentation` dict |
| `FRAMEWORK` | `early_stopping` dict |
| `MODEL_NAME` | `primary_metric` |
| `DATASET_ID` | `custom_prompts` list |
| `EPOCHS` | `split_strategy` dict |
| `BATCH_SIZE` | Trainer-specific params |
| `LEARNING_RATE` | |
| `IMGSZ` | |
| `DEVICE` | |

### 4.5 SDK Config Loading

**Trainer에서 Config 로드**:
```python
# train.py (Ultralytics trainer)
import os
import json
from training_sdk import TrainerSDK

# SDK 초기화
sdk = TrainerSDK.from_env()

# Basic Config 로드 (SDK 표준 메서드)
basic_config = sdk.get_basic_config()
# Returns: {"imgsz": 640, "epochs": 100, "batch": 16, "lr0": 0.01, ...}

# Advanced Config 로드 (SDK 메서드)
advanced_config = sdk.get_advanced_config()
# Returns: {"mosaic": 1.0, "mixup": 0.0, "cos_lr": false, ...}

# 또는 전체 Config 한번에 로드
full_config = sdk.get_full_config()
# Returns: {
#   "basic": {"imgsz": 640, ...},
#   "advanced": {"mosaic": 1.0, ...}
# }

# Ultralytics 학습에 적용
model = YOLO(f"{sdk.model_name}.pt")
results = model.train(
    data=data_yaml_path,
    # Basic config
    imgsz=basic_config["imgsz"],
    epochs=basic_config["epochs"],
    batch=basic_config["batch"],
    lr0=basic_config["lr0"],
    optimizer=basic_config["optimizer"],
    augment=basic_config["augment"],
    # Advanced config (Ultralytics-specific)
    mosaic=advanced_config.get("mosaic", 1.0),
    mixup=advanced_config.get("mixup", 0.0),
    copy_paste=advanced_config.get("copy_paste", 0.0),
    cos_lr=advanced_config.get("cos_lr", False),
    nms=advanced_config.get("nms", 0.7),
)
```

### 4.6 Advanced Config Examples by Framework

#### Ultralytics (YOLO)
```json
{
  "mosaic": 1.0,           // Mosaic augmentation
  "mixup": 0.0,            // Mixup augmentation
  "copy_paste": 0.0,       // Copy-paste augmentation
  "degrees": 0.0,          // Rotation degrees
  "translate": 0.1,        // Translation
  "scale": 0.5,            // Scale
  "shear": 0.0,            // Shear
  "perspective": 0.0,      // Perspective
  "flipud": 0.0,           // Vertical flip
  "fliplr": 0.5,           // Horizontal flip
  "cos_lr": false,         // Cosine learning rate scheduler
  "nms": 0.7,              // NMS IoU threshold
  "multi_scale": false     // Multi-scale training
}
```

#### timm (Classification)
```json
{
  "drop_rate": 0.0,        // Dropout rate
  "drop_path_rate": 0.1,   // Stochastic depth rate
  "mixup_alpha": 0.8,      // Mixup alpha
  "cutmix_alpha": 1.0,     // CutMix alpha
  "label_smoothing": 0.1,  // Label smoothing
  "aa": "rand-m9-mstd0.5", // AutoAugment policy
  "reprob": 0.25,          // Random erase probability
  "remode": "pixel",       // Random erase mode
  "model_ema": true,       // Exponential moving average
  "model_ema_decay": 0.9999
}
```

#### HuggingFace Transformers
```json
{
  "gradient_checkpointing": true,
  "fp16": true,
  "gradient_accumulation_steps": 4,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "logging_steps": 100,
  "eval_steps": 500,
  "save_steps": 1000
}
```

### 4.7 Alternative: Config File Method

대규모 config이거나 복잡한 nested 구조인 경우 파일로 전달:

**Backend가 Config 파일 생성**:
```python
# Backend에서
config_data = {
    "basic": config,
    "advanced": advanced_config
}
config_path = f"/tmp/training_configs/{job.id}/config.json"
with open(config_path, "w") as f:
    json.dump(config_data, f)

env = {
    "JOB_ID": str(job.id),
    "CONFIG_PATH": config_path,
    # ... 기타 환경변수
}
```

**Trainer에서 Config 파일 읽기**:
```python
# train.py
config = sdk.load_config_from_file(os.environ.get("CONFIG_PATH"))
basic = config["basic"]
advanced = config["advanced"]
```

### 4.8 SDK Config Interface (Updated)

**TrainerSDK 메서드**:
```python
class TrainerSDK:
    def get_basic_config(self) -> dict:
        """
        Basic Config 로드 (환경변수 CONFIG_* 에서)

        Returns:
            {
                "imgsz": 640,
                "epochs": 100,
                "batch": 16,
                "lr0": 0.01,
                "optimizer": "SGD",
                "augment": True
            }
        """
        return {
            "imgsz": int(os.environ.get("CONFIG_IMGSZ", 640)),
            "epochs": int(os.environ.get("CONFIG_EPOCHS", 100)),
            "batch": int(os.environ.get("CONFIG_BATCH", 16)),
            "lr0": float(os.environ.get("CONFIG_LR0", 0.01)),
            "optimizer": os.environ.get("CONFIG_OPTIMIZER", "SGD"),
            "augment": os.environ.get("CONFIG_AUGMENT", "True") == "True",
        }

    def get_advanced_config(self) -> dict:
        """
        Advanced Config 로드 (환경변수 ADVANCED_CONFIG JSON에서)

        Returns:
            Trainer-specific config dict
            예: {"mosaic": 1.0, "mixup": 0.0, ...}
        """
        advanced_json = os.environ.get("ADVANCED_CONFIG", "{}")
        return json.loads(advanced_json)

    def get_full_config(self) -> dict:
        """
        Basic + Advanced Config 모두 로드

        Returns:
            {
                "basic": {...},
                "advanced": {...}
            }
        """
        return {
            "basic": self.get_basic_config(),
            "advanced": self.get_advanced_config()
        }

    def load_config_from_file(self, path: str) -> dict:
        """Config JSON 파일에서 로드 (basic + advanced 포함)"""
        with open(path, "r") as f:
            return json.load(f)

    @property
    def model_name(self) -> str:
        """환경변수에서 MODEL_NAME 반환"""
        return os.environ.get("MODEL_NAME")

    @property
    def dataset_id(self) -> int:
        """환경변수에서 DATASET_ID 반환"""
        return int(os.environ.get("DATASET_ID"))

    @property
    def task_type(self) -> str:
        """환경변수에서 TASK_TYPE 반환"""
        return os.environ.get("TASK_TYPE")
```

### 4.9 Config Parameter Categories

**Basic Config (공통)**:

| Category | Parameters | Description |
|----------|------------|-------------|
| **Model** | `imgsz`, `pretrained` | 모델 입력 관련 |
| **Training** | `epochs`, `batch`, `lr0`, `optimizer` | 학습 하이퍼파라미터 |
| **Data** | `augment`, `cache`, `workers` | 데이터 로딩 관련 |
| **Output** | `project`, `name`, `save_period` | 결과 저장 관련 |

**Advanced Config (Framework별)**:

| Framework | Categories | Examples |
|-----------|------------|----------|
| **Ultralytics** | Augmentation, Scheduler, NMS | `mosaic`, `mixup`, `cos_lr`, `nms` |
| **timm** | Regularization, Augmentation | `drop_rate`, `mixup_alpha`, `label_smoothing` |
| **HuggingFace** | Training, Optimization | `gradient_checkpointing`, `fp16`, `warmup_ratio` |

### 4.10 Config Validation (Trainer Side)

Trainer는 config을 받아서 학습 시작 전 검증할 수 있습니다:

```python
# train.py
basic = sdk.get_basic_config()
advanced = sdk.get_advanced_config()

# Basic config 필수 파라미터 검증
required = ["imgsz", "epochs", "batch"]
missing = [k for k in required if k not in basic]
if missing:
    sdk.callback_failed(f"Missing required basic config: {missing}")
    sys.exit(1)

# Basic config 값 범위 검증
if basic["imgsz"] % 32 != 0:
    sdk.callback_failed("imgsz must be multiple of 32")
    sys.exit(1)

# Advanced config는 선택적이므로 default 값 사용
mosaic = advanced.get("mosaic", 1.0)
mixup = advanced.get("mixup", 0.0)
```

### 4.11 Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Config schema API | ✅ Implemented | Basic + Advanced Dynamic UI 지원 |
| Frontend config UI | ✅ Implemented | ConfigSchema 기반 |
| Basic Config → 환경변수 주입 | ⚠️ Partial | 기본 구현 있음 |
| Advanced Config → JSON 환경변수 | ❌ Not implemented | ADVANCED_CONFIG 추가 필요 |
| SDK get_basic_config() | ❌ Not implemented | 구현 필요 |
| SDK get_advanced_config() | ❌ Not implemented | 구현 필요 |

---

## 5. Stage 4: Start Training

### 5.1 Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 4.1 Training Job 생성 요청                                        │
│     └─ POST /api/v1/training/jobs                                 │
│                                                                   │
│ 4.2 Backend가 Job 생성 및 상태를 'pending'으로 설정                  │
│     └─ DB에 TrainingJob 레코드 생성                                │
│     └─ WebSocket으로 job_created 이벤트 broadcast                  │
│                                                                   │
│ 4.3 Backend가 Trainer subprocess 시작                              │
│     └─ Framework별 trainer 실행 (ultralytics/train.py)            │
│     └─ 환경변수로 job 정보 전달                                     │
│                                                                   │
│ 4.4 Trainer가 SDK 초기화 및 started callback 호출                   │
│     └─ POST /api/v1/training/jobs/{id}/callback/started           │
│     └─ 상태 'pending' → 'running'                                  │
│                                                                   │
│ 4.5 Trainer가 데이터셋 다운로드                                     │
│     └─ SDK.download_dataset()                                     │
│     └─ S3에서 로컬로 다운로드 및 압축 해제                          │
│                                                                   │
│ 4.6 Trainer가 학습 루프 시작                                        │
│     └─ Model load → Train loop → Validation                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Create Training Job

**API**: `POST /api/v1/training/jobs`

```json
// Request
{
  "model_name": "yolo11n",
  "framework": "ultralytics",
  "task_type": "detection",
  "dataset_id": 42,
  "config": {
    "imgsz": 640,
    "epochs": 100,
    "batch": 16,
    "lr0": 0.01,
    "optimizer": "SGD",
    "augment": true
  },
  "name": "Vehicle Detection v1",
  "description": "First training run for vehicle detection"
}

// Response
{
  "id": 25,
  "name": "Vehicle Detection v1",
  "model_name": "yolo11n",
  "framework": "ultralytics",
  "task_type": "detection",
  "dataset_id": 42,
  "status": "pending",
  "config": { ... },
  "created_at": "2025-11-20T10:30:00Z"
}
```

### 5.3 Trainer Subprocess Environment

Backend가 Trainer를 시작할 때 전달하는 환경변수:

```bash
TRAINING_JOB_ID=25
BACKEND_URL=http://localhost:8000/api/v1
S3_ENDPOINT=http://localhost:9002
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
DATASET_PATH=s3://training-datasets/42/
OUTPUT_PATH=/workspace/output
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 5.4 Started Callback (SDK → Backend)

**API**: `POST /api/v1/training/jobs/{id}/callback/started`

```json
// Request
{
  "pid": 12345,
  "hostname": "trainer-pod-xyz",
  "gpu_info": {
    "name": "NVIDIA RTX 4090",
    "memory_total_gb": 24
  }
}

// Response
{
  "status": "ok",
  "job_id": 25
}
```

### 5.5 Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Training job creation | ✅ Implemented | In training.py |
| Subprocess launcher | ✅ Implemented | In training_subprocess.py |
| Started callback | ⚠️ Partial | Need to verify |
| Dataset download | ✅ Implemented | In SDK |

---

## 6. Stage 5: Monitor Training

### 6.1 Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 5.1 Trainer가 Epoch 시작 시 progress callback 호출                 │
│     └─ POST /api/v1/training/jobs/{id}/callback/progress          │
│     └─ { epoch, total_epochs, phase: "train" }                    │
│                                                                   │
│ 5.2 Trainer가 Step마다 metrics callback 호출                       │
│     └─ POST /api/v1/training/jobs/{id}/callback/metrics           │
│     └─ { step, loss, lr, ... }                                    │
│                                                                   │
│ 5.3 Backend가 WebSocket으로 실시간 broadcast                       │
│     └─ ws://api/v1/ws/training?job_id=25                          │
│     └─ Frontend가 차트 업데이트                                    │
│                                                                   │
│ 5.4 Trainer가 Epoch 종료 시 validation metrics 전송                │
│     └─ { epoch, val_loss, mAP50, mAP50-95, ... }                  │
│                                                                   │
│ 5.5 Trainer가 Checkpoint 저장 및 업로드                            │
│     └─ SDK.upload_checkpoint(path, is_best)                       │
│     └─ POST /api/v1/training/jobs/{id}/callback/checkpoint        │
│                                                                   │
│ 5.6 Backend가 MLflow에 metrics 기록                                │
│     └─ Proxied logging via SDK callbacks                          │
│                                                                   │
│ 5.7 사용자가 실시간 모니터링                                        │
│     └─ Loss chart, mAP chart, GPU usage, ETA                      │
│     └─ Early stopping 버튼                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Progress Callback

**API**: `POST /api/v1/training/jobs/{id}/callback/progress`

```json
// Request
{
  "epoch": 5,
  "total_epochs": 100,
  "phase": "train",  // "train" | "val"
  "eta_seconds": 7200
}

// Backend Action
1. Update job.current_epoch = 5
2. Broadcast via WebSocket
3. Log to MLflow (epoch marker)
```

### 6.3 Metrics Callback

**API**: `POST /api/v1/training/jobs/{id}/callback/metrics`

```json
// Request (Training metrics)
{
  "epoch": 5,
  "step": 250,
  "phase": "train",
  "metrics": {
    "loss": 0.0234,
    "box_loss": 0.0156,
    "cls_loss": 0.0078,
    "lr": 0.00987
  }
}

// Request (Validation metrics)
{
  "epoch": 5,
  "phase": "val",
  "metrics": {
    "val_loss": 0.0312,
    "mAP50": 0.756,
    "mAP50-95": 0.523,
    "precision": 0.812,
    "recall": 0.734
  }
}

// Backend Action
1. Store in metrics table
2. Broadcast via WebSocket
3. Log to MLflow
4. Check early stopping conditions
```

### 6.4 Checkpoint Callback

**API**: `POST /api/v1/training/jobs/{id}/callback/checkpoint`

```json
// Request
{
  "epoch": 5,
  "checkpoint_path": "s3://training-checkpoints/checkpoints/25/epoch_5.pt",
  "is_best": false,
  "metrics": {
    "mAP50": 0.756,
    "mAP50-95": 0.523
  },
  "file_size_bytes": 52428800
}

// For best model
{
  "epoch": 5,
  "checkpoint_path": "s3://training-checkpoints/checkpoints/25/best.pt",
  "is_best": true,
  "metrics": { ... }
}

// Backend Action
1. Update job.latest_checkpoint_path
2. If is_best: Update job.best_checkpoint_path
3. Broadcast via WebSocket
4. Log artifact to MLflow
```

### 6.5 WebSocket Message Types

**Connection**: `ws://api/v1/ws/training?job_id=25`

```typescript
// Message types
type WebSocketMessage =
  | { type: "job_status", data: { status: JobStatus } }
  | { type: "progress", data: { epoch: number, total: number, phase: string } }
  | { type: "metrics", data: { epoch: number, metrics: Record<string, number> } }
  | { type: "checkpoint", data: { path: string, is_best: boolean } }
  | { type: "log", data: { level: string, message: string, timestamp: string } }
  | { type: "error", data: { code: string, message: string } }
```

### 6.6 Log Callback

Trainer의 로그를 Backend로 전송하고 실시간으로 사용자에게 표시합니다.

#### 6.6.1 Log Callback API

**API**: `POST /api/v1/training/jobs/{id}/callback/log`

```json
// Request
{
  "level": "INFO",      // "DEBUG" | "INFO" | "WARNING" | "ERROR"
  "message": "Starting epoch 5/100",
  "timestamp": "2025-11-20T10:35:00.123Z",
  "source": "trainer",  // "trainer" | "sdk" | "system"
  "metadata": {         // Optional
    "epoch": 5,
    "step": 0
  }
}

// Batch request (multiple logs at once)
{
  "logs": [
    {
      "level": "INFO",
      "message": "Loading model weights...",
      "timestamp": "2025-11-20T10:30:00.000Z"
    },
    {
      "level": "INFO",
      "message": "Model loaded successfully",
      "timestamp": "2025-11-20T10:30:05.123Z"
    }
  ]
}

// Response
{
  "status": "ok",
  "received": 2
}
```

#### 6.6.2 SDK Log Methods

```python
class TrainerSDK:
    def log(self, message: str, level: str = "INFO", **metadata):
        """
        단일 로그 메시지 전송

        Args:
            message: 로그 메시지
            level: DEBUG, INFO, WARNING, ERROR
            **metadata: 추가 메타데이터 (epoch, step 등)
        """
        pass

    def log_info(self, message: str, **metadata):
        """INFO 레벨 로그"""
        self.log(message, "INFO", **metadata)

    def log_warning(self, message: str, **metadata):
        """WARNING 레벨 로그"""
        self.log(message, "WARNING", **metadata)

    def log_error(self, message: str, **metadata):
        """ERROR 레벨 로그"""
        self.log(message, "ERROR", **metadata)

    def log_debug(self, message: str, **metadata):
        """DEBUG 레벨 로그"""
        self.log(message, "DEBUG", **metadata)

    def flush_logs(self):
        """
        버퍼된 로그를 일괄 전송
        (성능을 위해 로그를 버퍼링하고 주기적으로 flush)
        """
        pass
```

#### 6.6.3 Trainer에서 Log 사용 예시

```python
# train.py
from training_sdk import TrainerSDK

sdk = TrainerSDK.from_env()

# 학습 시작 로그
sdk.log_info("Training started", model=sdk.model_name, epochs=100)

# 데이터셋 로드
sdk.log_info(f"Loading dataset from {dataset_path}")
try:
    dataset = load_dataset(dataset_path)
    sdk.log_info(f"Dataset loaded: {len(dataset)} images")
except Exception as e:
    sdk.log_error(f"Failed to load dataset: {e}")
    sdk.callback_failed(str(e))
    sys.exit(1)

# 학습 루프
for epoch in range(epochs):
    sdk.log_info(f"Starting epoch {epoch+1}/{epochs}", epoch=epoch+1)

    for step, batch in enumerate(dataloader):
        # ... training code ...

        if step % 100 == 0:
            sdk.log_debug(f"Step {step}: loss={loss:.4f}", epoch=epoch+1, step=step)

    # Epoch 완료
    sdk.log_info(f"Epoch {epoch+1} completed: mAP50={metrics['mAP50']:.3f}")

# 학습 완료
sdk.log_info("Training completed successfully")
sdk.flush_logs()  # 남은 로그 모두 전송
```

#### 6.6.4 Log Storage

**Database Table**: `training_logs`

```sql
CREATE TABLE training_logs (
    id SERIAL PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES training_jobs(id),
    level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    source VARCHAR(20) DEFAULT 'trainer',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_training_logs_job_id ON training_logs(job_id);
CREATE INDEX idx_training_logs_timestamp ON training_logs(timestamp);
CREATE INDEX idx_training_logs_level ON training_logs(level);
```

#### 6.6.5 Log Query API

**API**: `GET /api/v1/training/jobs/{id}/logs`

```json
// Request
GET /api/v1/training/jobs/25/logs?level=INFO&limit=100&offset=0

// Query Parameters
- level: Filter by log level (DEBUG, INFO, WARNING, ERROR)
- limit: Number of logs to return (default: 100)
- offset: Pagination offset
- since: Logs after this timestamp (ISO 8601)
- until: Logs before this timestamp (ISO 8601)

// Response
{
  "logs": [
    {
      "id": 1234,
      "level": "INFO",
      "message": "Starting epoch 5/100",
      "timestamp": "2025-11-20T10:35:00.123Z",
      "source": "trainer",
      "metadata": {
        "epoch": 5
      }
    },
    // ... more logs
  ],
  "total": 542,
  "has_more": true
}
```

#### 6.6.6 Real-time Log Streaming

WebSocket을 통해 실시간 로그 수신:

```typescript
// Frontend WebSocket handler
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === "log") {
    const { level, message: logMessage, timestamp } = message.data;

    // 로그 레벨에 따른 색상
    const colorMap = {
      DEBUG: "gray",
      INFO: "blue",
      WARNING: "orange",
      ERROR: "red"
    };

    appendToLogPanel({
      timestamp,
      level,
      message: logMessage,
      color: colorMap[level]
    });
  }
};
```

#### 6.6.7 Log Buffering Strategy

성능을 위해 SDK에서 로그를 버퍼링:

```python
class TrainerSDK:
    def __init__(self):
        self._log_buffer = []
        self._log_buffer_size = 50       # 버퍼 크기
        self._log_flush_interval = 5.0   # 초
        self._last_flush = time.time()

    def log(self, message: str, level: str = "INFO", **metadata):
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "trainer",
            "metadata": metadata if metadata else None
        }

        self._log_buffer.append(log_entry)

        # Auto flush conditions
        if (len(self._log_buffer) >= self._log_buffer_size or
            time.time() - self._last_flush >= self._log_flush_interval or
            level == "ERROR"):
            self.flush_logs()

    def flush_logs(self):
        if not self._log_buffer:
            return

        try:
            requests.post(
                f"{self.backend_url}/training/jobs/{self.job_id}/callback/log",
                json={"logs": self._log_buffer}
            )
            self._log_buffer = []
            self._last_flush = time.time()
        except Exception as e:
            # 로그 전송 실패해도 학습은 계속
            print(f"Failed to flush logs: {e}")
```

### 6.7 Frontend Monitoring Components

```
┌──────────────────────────────────────────────────┐
│  Training Job #25: Vehicle Detection v1          │
│  Status: Running | Epoch 5/100 | ETA: 2h 30m     │
├──────────────────────────────────────────────────┤
│                                                  │
│  [Loss Chart]          [mAP Chart]               │
│  ┌─────────────┐      ┌─────────────┐            │
│  │  ╲          │      │          ╱  │            │
│  │   ╲_____    │      │      ___╱   │            │
│  └─────────────┘      └─────────────┘            │
│                                                  │
│  Current Metrics:                                │
│  • Loss: 0.0234    • mAP50: 0.756                │
│  • LR: 0.00987     • mAP50-95: 0.523             │
│                                                  │
│  Checkpoints:                                    │
│  • Best: epoch_3 (mAP50: 0.768)                  │
│  • Latest: epoch_5                               │
│                                                  │
│  [Stop Training] [Pause] [View Logs]             │
└──────────────────────────────────────────────────┘
```

### 6.8 Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Progress callback | ⚠️ Partial | Need SDK integration |
| Metrics callback | ⚠️ Partial | Need SDK integration |
| Checkpoint callback | ⚠️ Partial | Need SDK integration |
| Log callback API | ❌ Not implemented | Need to add endpoint |
| Log storage (DB table) | ❌ Not implemented | training_logs 테이블 생성 필요 |
| SDK log methods | ❌ Not implemented | log_info, log_error 등 구현 필요 |
| Log query API | ❌ Not implemented | GET /jobs/{id}/logs 필요 |
| Log buffering | ❌ Not implemented | SDK에 버퍼링 로직 추가 필요 |
| WebSocket broadcast | ✅ Implemented | WebSocketManager exists |
| MLflow integration | ⚠️ Partial | Need to verify proxy flow |
| Frontend charts | ⚠️ Partial | Basic charts exist |
| Frontend log panel | ❌ Not implemented | Log viewer UI 필요 |

---

## 7. Stage 6: Result Review

### 7.1 Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 6.1 Trainer가 학습 완료 후 completion callback 호출                │
│     └─ POST /api/v1/training/jobs/{id}/callback/completion        │
│     └─ Final metrics, best checkpoint path                        │
│                                                                   │
│ 6.2 Backend가 Job 상태를 'completed'로 업데이트                     │
│     └─ Update training_results                                    │
│     └─ WebSocket broadcast                                        │
│                                                                   │
│ 6.3 사용자가 최종 결과 확인                                         │
│     └─ GET /api/v1/training/jobs/{id}                             │
│     └─ Final metrics, training time, best checkpoint              │
│                                                                   │
│ 6.4 사용자가 학습 히스토리 조회                                     │
│     └─ GET /api/v1/training/jobs/{id}/metrics                     │
│     └─ All epoch metrics for analysis                             │
│                                                                   │
│ 6.5 사용자가 Export 또는 Inference 진행                            │
│     └─ Create export job with best checkpoint                     │
│     └─ Test inference with sample images                          │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Completion Callback

**API**: `POST /api/v1/training/jobs/{id}/callback/completion`

```json
// Request (Success)
{
  "status": "completed",
  "training_results": {
    "total_epochs": 100,
    "best_epoch": 87,
    "best_metrics": {
      "mAP50": 0.856,
      "mAP50-95": 0.623,
      "precision": 0.892,
      "recall": 0.834
    },
    "final_metrics": {
      "mAP50": 0.845,
      "mAP50-95": 0.612
    },
    "training_time_seconds": 9000
  },
  "best_checkpoint_path": "s3://training-checkpoints/checkpoints/25/best.pt",
  "final_checkpoint_path": "s3://training-checkpoints/checkpoints/25/last.pt"
}

// Request (Failed)
{
  "status": "failed",
  "error_code": "OOM",
  "error_message": "CUDA out of memory. Try reducing batch size.",
  "last_successful_epoch": 42
}

// Request (Stopped by user)
{
  "status": "stopped",
  "stopped_at_epoch": 50,
  "best_checkpoint_path": "s3://training-checkpoints/checkpoints/25/best.pt"
}
```

### 7.3 Training Results Response

**API**: `GET /api/v1/training/jobs/{id}`

```json
{
  "id": 25,
  "name": "Vehicle Detection v1",
  "status": "completed",
  "model_name": "yolo11n",
  "framework": "ultralytics",
  "task_type": "detection",
  "dataset_id": 42,
  "config": { ... },

  "training_results": {
    "total_epochs": 100,
    "best_epoch": 87,
    "best_metrics": {
      "mAP50": 0.856,
      "mAP50-95": 0.623
    },
    "training_time_seconds": 9000,
    "gpu_hours": 2.5
  },

  "checkpoints": {
    "best": "s3://training-checkpoints/checkpoints/25/best.pt",
    "last": "s3://training-checkpoints/checkpoints/25/last.pt"
  },

  "mlflow_run_id": "abc123def456",
  "mlflow_experiment_id": "1",

  "created_at": "2025-11-20T10:30:00Z",
  "started_at": "2025-11-20T10:30:05Z",
  "completed_at": "2025-11-20T13:00:05Z"
}
```

### 7.4 Metrics History

**API**: `GET /api/v1/training/jobs/{id}/metrics`

```json
{
  "job_id": 25,
  "epochs": [
    {
      "epoch": 1,
      "train_metrics": { "loss": 0.5234, "box_loss": 0.3456, "cls_loss": 0.1778 },
      "val_metrics": { "mAP50": 0.234, "mAP50-95": 0.123 }
    },
    {
      "epoch": 2,
      "train_metrics": { "loss": 0.3456, "box_loss": 0.2234, "cls_loss": 0.1222 },
      "val_metrics": { "mAP50": 0.456, "mAP50-95": 0.234 }
    }
    // ... all epochs
  ]
}
```

### 7.5 Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Completion callback | ⚠️ Partial | Need SDK integration |
| Training results | ✅ Implemented | In TrainingJob model |
| Metrics history API | ❌ Not implemented | Need to add |
| Frontend results view | ⚠️ Partial | Basic view exists |

---

## 8. SDK Callback Summary

### 8.1 All Callback Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/training/jobs/{id}/callback/started` | POST | Training started |
| `/training/jobs/{id}/callback/progress` | POST | Epoch progress |
| `/training/jobs/{id}/callback/metrics` | POST | Metrics update |
| `/training/jobs/{id}/callback/checkpoint` | POST | Checkpoint saved |
| `/training/jobs/{id}/callback/completion` | POST | Training finished |
| `/training/jobs/{id}/callback/log` | POST | Log message |

### 8.2 SDK Method Mapping

```python
class TrainerSDK:
    def report_started(self, gpu_info: dict) -> None:
        """POST /callback/started"""

    def report_progress(self, epoch: int, total: int, phase: str) -> None:
        """POST /callback/progress"""

    def report_metrics(self, epoch: int, phase: str, metrics: dict) -> None:
        """POST /callback/metrics"""

    def upload_checkpoint(self, local_path: str, is_best: bool) -> str:
        """Upload to S3 + POST /callback/checkpoint"""

    def report_completed(self, results: dict) -> None:
        """POST /callback/completion with status='completed'"""

    def report_failed(self, error_code: str, message: str) -> None:
        """POST /callback/completion with status='failed'"""

    def log(self, level: str, message: str) -> None:
        """POST /callback/log"""
```

---

## 9. Error Handling

### 9.1 Error Types

| Error Code | Description | Action |
|------------|-------------|--------|
| `OOM` | CUDA out of memory | Suggest reducing batch size |
| `DATASET_NOT_FOUND` | Dataset download failed | Check S3 connection |
| `INVALID_CONFIG` | Configuration error | Show validation errors |
| `CHECKPOINT_UPLOAD_FAILED` | S3 upload failed | Retry or alert |
| `TIMEOUT` | Training timeout | Check if stuck |
| `USER_STOPPED` | User requested stop | Graceful shutdown |

### 9.2 Recovery Mechanisms

1. **Checkpoint Resume**: 마지막 체크포인트에서 학습 재개
2. **Auto-retry**: 일시적 오류 시 자동 재시도 (max 3회)
3. **Graceful Shutdown**: SIGTERM 수신 시 현재 epoch 완료 후 종료

---

## 10. Integration with Annotation Team

### 10.1 Expected Changes from Labeler Team

> **TODO**: 레이블러 팀과 협의 필요

**Possible Integration Points**:

1. **Dataset Export Format**
   - YOLO format support
   - COCO format support
   - Class mapping file

2. **Webhook Events**
   - `labeling_started`
   - `labeling_progress`
   - `labeling_complete`
   - `export_ready`

3. **Dataset Versioning**
   - Version tracking
   - Diff between versions
   - Rollback support

4. **Quality Metrics**
   - Annotation quality score
   - Inter-annotator agreement
   - Edge case flagging

### 10.2 Required API Changes

```python
# New endpoints for labeler integration
POST /api/v1/datasets/import-from-labeler
GET /api/v1/labeler/projects
GET /api/v1/labeler/projects/{id}/status
```

---

## 11. Implementation Priorities

### 11.1 Phase 1: Core Training Flow (P0)

1. **Training job creation** - Verify existing implementation
2. **Started callback** - Implement/verify in SDK
3. **Progress & metrics callbacks** - Implement in SDK
4. **Checkpoint callback** - Implement in SDK
5. **Completion callback** - Implement/verify in SDK
6. **WebSocket broadcast** - Verify existing implementation

### 11.2 Phase 2: Model & Dataset Management (P1)

1. **Model registration API** - New endpoint
2. **Model list with filtering** - Enhance existing
3. **Config schema API** - New endpoint
4. **Config validation** - New endpoint

### 11.3 Phase 3: Monitoring & Results (P1)

1. **Metrics history API** - New endpoint
2. **Frontend charts** - Enhance existing
3. **MLflow integration** - Verify proxy flow

### 11.4 Phase 4: Integration (P2)

1. **Annotation team webhook** - New feature
2. **Dataset versioning** - New feature
3. **Training resume** - New feature

---

## 12. E2E Test Plan

### 12.1 Test Scenario

```
1. Register models from Ultralytics trainer
2. Upload test dataset (small, ~100 images)
3. Create training job with minimal config (epochs=5)
4. Verify all callbacks:
   - started
   - progress (per epoch)
   - metrics (train + val)
   - checkpoint (best + last)
   - completion
5. Verify WebSocket messages received
6. Verify MLflow experiment created
7. Verify final results in API response
8. Create export job from best checkpoint
9. Run inference on test image
```

### 12.2 Test Commands

```bash
# 1. Create training job
curl -X POST http://localhost:8000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "yolo11n",
    "framework": "ultralytics",
    "task_type": "detection",
    "dataset_id": 1,
    "config": {
      "imgsz": 640,
      "epochs": 5,
      "batch": 8
    },
    "name": "E2E Test Training"
  }'

# 2. Monitor via WebSocket (use wscat or similar)
wscat -c ws://localhost:8000/api/v1/ws/training?job_id=26

# 3. Check final results
curl http://localhost:8000/api/v1/training/jobs/26

# 4. Check metrics history
curl http://localhost:8000/api/v1/training/jobs/26/metrics
```

---

## 13. Open Questions

### 13.1 For Discussion

1. **Early Stopping**: 자동 early stopping 구현할 것인가?
   - Patience 설정
   - Metric threshold

2. **Multi-GPU Training**: 다중 GPU 지원 범위는?
   - DDP (Distributed Data Parallel)
   - 단일 노드 vs 다중 노드

3. **Resource Limits**: Job당 리소스 제한은?
   - Max training time
   - Max GPU memory
   - Max storage

4. **Annotation Integration**: 레이블러 팀 연동 방식은?
   - Webhook vs Polling
   - Auth mechanism

### 13.2 For Labeler Team

1. Export format 지원 목록?
2. Webhook payload schema?
3. Authentication 방식?
4. Dataset versioning 전략?

---

## 14. Related Documents

- [E2E_TEST_REPORT_20251120.md](./E2E_TEST_REPORT_20251120.md) - Inference & Export E2E test results
- [IMPLEMENTATION_TO_DO_LIST.md](../IMPLEMENTATION_TO_DO_LIST.md) - Overall implementation status
- [THIN_SDK_DESIGN.md](../../docs/sdk/THIN_SDK_DESIGN.md) - SDK architecture
- [EXPORT_CONVENTION.md](../../docs/EXPORT_CONVENTION.md) - Export pipeline design

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-20 | 0.1 | Initial draft |

