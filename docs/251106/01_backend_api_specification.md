# Backend API 명세서

> **작성일**: 2025-11-06
> **Version**: 1.0
> **Status**: Production

## 목차

1. [개요](#개요)
2. [API 엔드포인트 목록](#api-엔드포인트-목록)
3. [인증](#인증)
4. [Training API](#training-api)
5. [Dataset API](#dataset-api)
6. [Model API](#model-api)
7. [Validation API](#validation-api)
8. [Inference API](#inference-api)
9. [Internal API](#internal-api)
10. [에러 코드](#에러-코드)

---

## 개요

### Base URL

```
Local:      http://localhost:8000
Production: https://mvp-vision-ai-platform-production.up.railway.app
```

### API Prefix

```
/api/v1
```

### Content-Type

- Request: `application/json`
- Response: `application/json`
- File Upload: `multipart/form-data`

---

## API 엔드포인트 목록

### 인증 (Authentication)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/login` | 로그인 및 JWT 토큰 발급 |

### Training API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/training/jobs` | 새로운 학습 작업 생성 |
| POST | `/api/v1/training/jobs/{job_id}/start` | 학습 작업 시작 |
| GET | `/api/v1/training/jobs/{job_id}` | 학습 작업 상세 조회 |
| GET | `/api/v1/training/jobs/{job_id}/metrics` | 학습 메트릭 조회 |
| GET | `/api/v1/training/jobs/{job_id}/logs` | 학습 로그 조회 |
| POST | `/api/v1/training/stop/{job_id}` | 학습 작업 중지 |
| GET | `/api/v1/training/config-schema` | 고급 설정 스키마 조회 |

### Dataset API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/datasets` | 데이터셋 생성 (multipart/form-data) |
| GET | `/api/v1/datasets` | 데이터셋 목록 조회 |
| GET | `/api/v1/datasets/{dataset_id}` | 데이터셋 상세 조회 |

### Model API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/models/list` | 사용 가능한 모델 목록 조회 |

### Validation API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/validation/results/{job_id}` | Validation 결과 조회 |
| GET | `/api/v1/validation/images/{validation_result_id}` | 이미지별 validation 결과 조회 |

### Inference API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/test-inference` | 체크포인트 기반 추론 (multipart/form-data) |

### Internal API

> Training Services와 Backend 간 통신용 (Internal use only)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/internal/training/callback` | Training Service → Backend 콜백 | X-Internal-Auth |

---

## 인증

### JWT Token 기반 인증

**Header:**
```http
Authorization: Bearer <token>
```

### POST /api/v1/auth/login

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "full_name": "John Doe"
  }
}
```

---

## Training API

### POST /api/v1/training/jobs

**Description:** 새로운 학습 작업 생성

**Request:**
```json
{
  "session_id": 1,
  "project_id": 2,
  "experiment_name": "ResNet-50 Experiment",
  "tags": ["baseline", "classification"],
  "notes": "Initial baseline training",
  "config": {
    "framework": "timm",
    "model_name": "resnet50",
    "task_type": "image_classification",
    "dataset_id": "8ab5c6e4-0f92-4fff-8f4e-441c08d94cef",
    "dataset_format": "dice",
    "num_classes": 10,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "advanced_config": {
      "optimizer": {
        "type": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999]
      },
      "scheduler": {
        "type": "cosine",
        "T_max": 50,
        "eta_min": 1e-6,
        "warmup_epochs": 5,
        "warmup_lr": 1e-6
      },
      "augmentation": {
        "enabled": true,
        "random_flip": true,
        "random_flip_prob": 0.5,
        "color_jitter": true,
        "brightness": 0.2,
        "contrast": 0.2
      },
      "mixed_precision": true,
      "gradient_clip_value": 1.0
    }
  }
}
```

**Response:**
```json
{
  "id": 5,
  "session_id": 1,
  "project_id": 2,
  "experiment_name": "ResNet-50 Experiment",
  "tags": ["baseline", "classification"],
  "notes": "Initial baseline training",
  "framework": "timm",
  "model_name": "resnet50",
  "task_type": "image_classification",
  "dataset_id": "8ab5c6e4-0f92-4fff-8f4e-441c08d94cef",
  "dataset_path": "8ab5c6e4-0f92-4fff-8f4e-441c08d94cef",
  "dataset_format": "dice",
  "num_classes": 10,
  "output_dir": "/app/data/outputs/job_20251106_120000",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "status": "pending",
  "primary_metric": "accuracy",
  "primary_metric_mode": "max",
  "created_at": "2025-11-06T12:00:00Z",
  "started_at": null,
  "completed_at": null
}
```

### POST /api/v1/training/jobs/{job_id}/start

**Description:** 학습 작업 시작

**Path Parameters:**
- `job_id` (integer): 학습 작업 ID

**Query Parameters:**
- `checkpoint_path` (string, optional): 체크포인트 경로
- `resume` (boolean, optional): 재개 여부 (default: false)

**Response:**
```json
{
  "id": 5,
  "status": "running",
  "started_at": "2025-11-06T12:05:00Z"
}
```

### GET /api/v1/training/jobs/{job_id}

**Description:** 학습 작업 상세 조회

**Response:**
```json
{
  "id": 5,
  "session_id": 1,
  "project_id": 2,
  "project_name": "My Project",
  "experiment_name": "ResNet-50 Experiment",
  "framework": "timm",
  "model_name": "resnet50",
  "task_type": "image_classification",
  "dataset_id": "8ab5c6e4-0f92-4fff-8f4e-441c08d94cef",
  "status": "running",
  "primary_metric": "accuracy",
  "primary_metric_mode": "max",
  "current_epoch": 10,
  "total_epochs": 50,
  "mlflow_run_id": "abc123def456",
  "created_at": "2025-11-06T12:00:00Z",
  "started_at": "2025-11-06T12:05:00Z",
  "completed_at": null
}
```

### GET /api/v1/training/jobs/{job_id}/metrics

**Description:** 학습 메트릭 조회

**Query Parameters:**
- `limit` (integer, optional): 최대 개수 (default: 100)

**Response:**
```json
[
  {
    "id": 1,
    "job_id": 5,
    "epoch": 1,
    "loss": 2.3025,
    "accuracy": 0.125,
    "learning_rate": 0.001,
    "extra_metrics": {
      "precision": 0.13,
      "recall": 0.12,
      "f1": 0.125
    },
    "created_at": "2025-11-06T12:10:00Z"
  }
]
```

### GET /api/v1/training/jobs/{job_id}/logs

**Description:** 학습 로그 조회

**Query Parameters:**
- `limit` (integer, optional): 최대 개수 (default: 500)
- `log_type` (string, optional): 로그 타입 (stdout, stderr)

**Response:**
```json
[
  {
    "id": 1,
    "job_id": 5,
    "log_type": "stdout",
    "message": "[INFO] Starting training...",
    "created_at": "2025-11-06T12:05:00Z"
  }
]
```

### POST /api/v1/training/stop/{job_id}

**Description:** 학습 작업 중지

**Response:**
```json
{
  "job_id": 5,
  "status": "stopped",
  "message": "Training job stopped successfully"
}
```

### GET /api/v1/training/config-schema

**Description:** 고급 설정 스키마 조회

**Query Parameters:**
- `framework` (string, required): 프레임워크 (timm, ultralytics)
- `task_type` (string, optional): 태스크 타입

**Response:**
```json
{
  "framework": "timm",
  "task_type": "image_classification",
  "schema": {
    "fields": [
      {
        "name": "optimizer.type",
        "type": "select",
        "label": "Optimizer",
        "description": "Optimization algorithm",
        "default": "adamw",
        "options": [
          {
            "value": "adam",
            "label": "Adam",
            "description": "Adaptive Moment Estimation"
          },
          {
            "value": "adamw",
            "label": "AdamW",
            "description": "Adam with weight decay"
          },
          {
            "value": "sgd",
            "label": "SGD",
            "description": "Stochastic Gradient Descent"
          }
        ]
      },
      {
        "name": "optimizer.learning_rate",
        "type": "number",
        "label": "Learning Rate",
        "description": "Initial learning rate",
        "default": 0.001,
        "min": 1e-6,
        "max": 1.0,
        "step": 1e-5
      }
    ]
  },
  "presets": {
    "basic": { /* ... */ },
    "standard": { /* ... */ },
    "aggressive": { /* ... */ }
  }
}
```

---

## Dataset API

### POST /api/v1/datasets

**Description:** 데이터셋 생성

**Request (multipart/form-data):**
```
name: "My Dataset"
description: "Description here"
format: "dice"
visibility: "private"
project_id: 2
files: [File, File, ...]
```

**Response:**
```json
{
  "id": "8ab5c6e4-0f92-4fff-8f4e-441c08d94cef",
  "name": "My Dataset",
  "description": "Description here",
  "format": "dice",
  "visibility": "private",
  "project_id": 2,
  "file_count": 25,
  "total_size_bytes": 1048576,
  "num_classes": 10,
  "created_at": "2025-11-06T12:00:00Z"
}
```

### GET /api/v1/datasets

**Description:** 데이터셋 목록 조회

**Query Parameters:**
- `project_id` (integer, optional): 프로젝트 ID
- `format` (string, optional): 포맷 필터
- `skip` (integer, optional): 오프셋
- `limit` (integer, optional): 최대 개수

**Response:**
```json
{
  "datasets": [
    {
      "id": "8ab5c6e4-0f92-4fff-8f4e-441c08d94cef",
      "name": "My Dataset",
      "format": "dice",
      "file_count": 25,
      "num_classes": 10,
      "created_at": "2025-11-06T12:00:00Z"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 50
}
```

### GET /api/v1/datasets/{dataset_id}

**Description:** 데이터셋 상세 조회

**Response:**
```json
{
  "id": "8ab5c6e4-0f92-4fff-8f4e-441c08d94cef",
  "name": "My Dataset",
  "description": "Description here",
  "format": "dice",
  "visibility": "private",
  "project_id": 2,
  "file_count": 25,
  "total_size_bytes": 1048576,
  "num_classes": 10,
  "class_distribution": {
    "cat": 10,
    "dog": 15
  },
  "created_at": "2025-11-06T12:00:00Z",
  "updated_at": "2025-11-06T12:00:00Z"
}
```

---

## Model API

### GET /api/v1/models/list

**Description:** 사용 가능한 모델 목록 조회

**Query Parameters:**
- `framework` (string, optional): 프레임워크 필터
- `task_type` (string, optional): 태스크 타입 필터

**Response:**
```json
{
  "models": [
    {
      "model_name": "resnet50",
      "framework": "timm",
      "task_types": ["image_classification"],
      "default_image_size": 224,
      "parameters": "25.6M",
      "pretrained": true,
      "description": "ResNet-50 architecture"
    }
  ],
  "total": 1
}
```

---

## Validation API

### GET /api/v1/validation/results/{job_id}

**Description:** Validation 결과 조회

**Response:**
```json
{
  "job_id": 5,
  "results": [
    {
      "id": 1,
      "epoch": 10,
      "task_type": "image_classification",
      "primary_metric_name": "accuracy",
      "primary_metric_value": 0.85,
      "overall_loss": 0.45,
      "metrics": {
        "accuracy": 0.85,
        "precision": 0.84,
        "recall": 0.83,
        "f1": 0.835
      },
      "per_class_metrics": {
        "cat": {
          "precision": 0.90,
          "recall": 0.85,
          "f1": 0.875,
          "support": 100
        },
        "dog": {
          "precision": 0.80,
          "recall": 0.82,
          "f1": 0.81,
          "support": 150
        }
      },
      "confusion_matrix": [[85, 15], [27, 123]],
      "checkpoint_path": "/app/data/outputs/job_5/checkpoints/epoch_10.pth",
      "created_at": "2025-11-06T12:15:00Z"
    }
  ]
}
```

### GET /api/v1/validation/images/{validation_result_id}

**Description:** 이미지별 validation 결과 조회

**Query Parameters:**
- `skip` (integer, optional): 오프셋
- `limit` (integer, optional): 최대 개수
- `is_correct` (boolean, optional): 정답 여부 필터

**Response:**
```json
{
  "validation_result_id": 1,
  "images": [
    {
      "id": 1,
      "image_path": "/path/to/image.jpg",
      "image_name": "cat_001.jpg",
      "true_label": "cat",
      "true_label_id": 0,
      "predicted_label": "cat",
      "predicted_label_id": 0,
      "confidence": 0.95,
      "is_correct": true,
      "top5_predictions": [
        {"label": "cat", "confidence": 0.95},
        {"label": "dog", "confidence": 0.04}
      ]
    }
  ],
  "total": 250,
  "skip": 0,
  "limit": 50
}
```

---

## Inference API

### POST /api/v1/test-inference

**Description:** 체크포인트 기반 추론

**Request (multipart/form-data):**
```
training_job_id: 5
file: <image file>
confidence_threshold: 0.25
top_k: 5
```

**Response:**
```json
{
  "predictions": [
    {
      "label": "cat",
      "label_id": 0,
      "confidence": 0.95
    },
    {
      "label": "dog",
      "label_id": 1,
      "confidence": 0.04
    }
  ],
  "task_type": "image_classification",
  "model_name": "resnet50",
  "inference_time_ms": 45.2
}
```

---

## Internal API

> Training Services와 Backend 간 통신용 API (Internal use only)

### POST /api/v1/internal/training/callback

**Description:** Training Service에서 Backend로 콜백

**Headers:**
```
X-Internal-Auth: <internal_secret>
```

**Request:**
```json
{
  "job_id": 5,
  "event": "epoch_complete",
  "data": {
    "epoch": 10,
    "loss": 0.45,
    "accuracy": 0.85,
    "learning_rate": 0.0008
  }
}
```

**Response:**
```json
{
  "status": "received",
  "job_id": 5
}
```

---

## 에러 코드

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | OK - 성공 |
| 201 | Created - 리소스 생성 성공 |
| 400 | Bad Request - 잘못된 요청 |
| 401 | Unauthorized - 인증 실패 |
| 403 | Forbidden - 권한 없음 |
| 404 | Not Found - 리소스 없음 |
| 409 | Conflict - 리소스 충돌 |
| 422 | Unprocessable Entity - 검증 실패 |
| 500 | Internal Server Error - 서버 오류 |
| 503 | Service Unavailable - 서비스 이용 불가 |

### Error Response Format

```json
{
  "detail": "Error message here",
  "code": "ERROR_CODE",
  "field": "field_name"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | 입력 검증 실패 |
| `DATASET_NOT_FOUND` | 데이터셋 없음 |
| `JOB_NOT_FOUND` | 학습 작업 없음 |
| `INVALID_STATUS` | 잘못된 상태 전환 |
| `TRAINING_SERVICE_UNAVAILABLE` | Training Service 연결 실패 |
| `CHECKPOINT_NOT_FOUND` | 체크포인트 없음 |

---

## 참고 문서

- [API 시나리오 플로우](./04_user_flow_scenarios.md)
- [Config Schema 가이드](./03_config_schema_guide.md)
- [기존 API 명세](../api/API_SPECIFICATION.md)
