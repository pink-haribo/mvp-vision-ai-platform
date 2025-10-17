# API 명세서

## 목차
- [개요](#개요)
- [인증](#인증)
- [Public APIs](#public-apis)
  - [Chat & Intent](#1-chat--intent)
  - [Workflow Management](#2-workflow-management)
  - [Dataset Management](#3-dataset-management)
  - [Models](#4-models)
  - [Inference](#5-inference)
  - [User & Projects](#6-user--projects)
- [Internal APIs](#internal-apis)
- [WebSocket](#websocket)
- [에러 처리](#에러-처리)
- [Rate Limiting](#rate-limiting)

## 개요

### 베이스 URL

| 환경 | URL |
|------|-----|
| Development | `http://localhost:8000/api/v1` |
| Staging | `https://staging-api.vision-platform.com/api/v1` |
| Production | `https://api.vision-platform.com/api/v1` |

### API 버전

현재 버전: **v1**

버전은 URL 경로에 포함됩니다 (`/api/v1`).

### Content Type

모든 요청과 응답은 JSON 형식입니다.

```
Content-Type: application/json
```

---

## 인증

### JWT 토큰 기반 인증

모든 보호된 엔드포인트는 JWT Bearer Token을 요구합니다.

```http
Authorization: Bearer <access_token>
```

### 토큰 발급

#### POST /auth/token

**Request:**
```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 토큰 갱신

#### POST /auth/refresh

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## Public APIs

### 1. Chat & Intent

#### POST /chat/message

자연어 메시지를 전송하고 의도를 파싱합니다.

**Request:**
```http
POST /api/v1/chat/message
Authorization: Bearer <token>
Content-Type: application/json

{
  "session_id": "sess_abc123",
  "message": "ResNet50으로 고양이 3종류 분류하는 모델 만들어줘",
  "context": {}
}
```

**Response (200 OK) - 추가 정보 필요:**
```json
{
  "type": "clarification_needed",
  "parsed_intent": {
    "task_type": "classification",
    "model": "resnet50",
    "num_classes": 3,
    "confidence": 0.95
  },
  "questions": [
    {
      "id": "q1",
      "field": "class_names",
      "question": "3종류가 구체적으로 어떤 클래스인가요? (예: 페르시안, 코숏, 러시안블루)",
      "required": true,
      "suggestions": []
    },
    {
      "id": "q2",
      "field": "dataset",
      "question": "데이터셋은 어디에 있나요?",
      "required": true,
      "suggestions": ["my-drive/cats", "uploaded-datasets"]
    }
  ],
  "message": "좋아요! ResNet50으로 3종류 분류 모델을 만들어드릴게요. 몇 가지만 확인할게요."
}
```

**Response (200 OK) - 의도 완성:**
```json
{
  "type": "intent_complete",
  "training_config": {
    "task_type": "classification",
    "model_name": "resnet50",
    "model_source": "timm",
    "num_classes": 3,
    "class_names": ["치즈", "나비", "츄르"],
    "dataset": {
      "source": "uploaded",
      "dataset_id": "ds_123"
    },
    "hyperparameters": {
      "epochs": 100,
      "batch_size": 32,
      "learning_rate": 0.001,
      "optimizer": "adam"
    }
  },
  "message": "모든 설정이 완료되었습니다! 학습을 시작하시겠어요?",
  "actions": [
    {
      "type": "start_training",
      "label": "학습 시작"
    },
    {
      "type": "edit_config",
      "label": "설정 수정"
    }
  ]
}
```

#### GET /chat/sessions/{session_id}

대화 세션 조회

**Request:**
```http
GET /api/v1/chat/sessions/sess_abc123
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "session_id": "sess_abc123",
  "created_at": "2025-10-17T10:00:00Z",
  "updated_at": "2025-10-17T10:30:00Z",
  "messages": [
    {
      "role": "user",
      "content": "ResNet50으로 고양이 분류 모델 만들어줘",
      "timestamp": "2025-10-17T10:00:00Z"
    },
    {
      "role": "assistant",
      "content": "좋아요! 몇 가지 확인할게요...",
      "timestamp": "2025-10-17T10:00:05Z"
    }
  ],
  "status": "active"
}
```

#### DELETE /chat/sessions/{session_id}

대화 세션 삭제

**Request:**
```http
DELETE /api/v1/chat/sessions/sess_abc123
Authorization: Bearer <token>
```

**Response (204 No Content)**

---

### 2. Workflow Management

#### POST /workflows

새 학습 워크플로우 생성

**Request:**
```http
POST /api/v1/workflows
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "고양이 분류 모델",
  "description": "치즈, 나비, 츄르 3종류 분류",
  "config": {
    "task_type": "classification",
    "model": "resnet50",
    "dataset_id": "ds_123",
    "hyperparameters": {
      "epochs": 100,
      "batch_size": 32,
      "learning_rate": 0.001
    }
  }
}
```

**Response (201 Created):**
```json
{
  "workflow_id": "wf_789xyz",
  "name": "고양이 분류 모델",
  "status": "pending",
  "created_at": "2025-10-17T10:00:00Z",
  "estimated_duration_minutes": 30,
  "queue_position": 2
}
```

#### GET /workflows

워크플로우 목록 조회

**Request:**
```http
GET /api/v1/workflows?status=training&page=1&limit=20
Authorization: Bearer <token>
```

**Query Parameters:**
- `status` (optional): `pending`, `training`, `completed`, `failed`, `cancelled`
- `page` (optional): 페이지 번호 (default: 1)
- `limit` (optional): 페이지 크기 (default: 20, max: 100)
- `sort` (optional): `created_at`, `-created_at` (default: `-created_at`)

**Response (200 OK):**
```json
{
  "workflows": [
    {
      "workflow_id": "wf_789xyz",
      "name": "고양이 분류 모델",
      "status": "training",
      "progress": 45,
      "created_at": "2025-10-17T10:00:00Z",
      "started_at": "2025-10-17T10:05:00Z"
    }
  ],
  "total": 15,
  "page": 1,
  "limit": 20,
  "has_next": false
}
```

#### GET /workflows/{workflow_id}

워크플로우 상세 조회

**Request:**
```http
GET /api/v1/workflows/wf_789xyz
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "workflow_id": "wf_789xyz",
  "name": "고양이 분류 모델",
  "description": "치즈, 나비, 츄르 3종류 분류",
  "status": "training",
  "progress": {
    "current_epoch": 45,
    "total_epochs": 100,
    "percentage": 45
  },
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.892,
    "val_loss": 0.312,
    "val_accuracy": 0.856
  },
  "resources": {
    "gpu_type": "T4",
    "gpu_count": 1,
    "memory_gb": 16
  },
  "created_at": "2025-10-17T10:00:00Z",
  "started_at": "2025-10-17T10:05:00Z",
  "completed_at": null,
  "estimated_completion_at": "2025-10-17T10:35:00Z"
}
```

#### GET /workflows/{workflow_id}/metrics

워크플로우 메트릭 히스토리 조회

**Request:**
```http
GET /api/v1/workflows/wf_789xyz/metrics?metric=loss,accuracy&start_time=2025-10-17T10:00:00Z
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "workflow_id": "wf_789xyz",
  "metrics": [
    {
      "timestamp": "2025-10-17T10:05:00Z",
      "epoch": 1,
      "step": 100,
      "loss": 2.456,
      "accuracy": 0.234
    },
    {
      "timestamp": "2025-10-17T10:10:00Z",
      "epoch": 10,
      "step": 1000,
      "loss": 1.123,
      "accuracy": 0.678
    }
  ]
}
```

#### DELETE /workflows/{workflow_id}

워크플로우 중단 및 삭제

**Request:**
```http
DELETE /api/v1/workflows/wf_789xyz
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "workflow_id": "wf_789xyz",
  "status": "cancelled",
  "cancelled_at": "2025-10-17T10:15:00Z"
}
```

#### GET /workflows/{workflow_id}/logs

워크플로우 로그 조회

**Request:**
```http
GET /api/v1/workflows/wf_789xyz/logs?tail=100
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "workflow_id": "wf_789xyz",
  "logs": [
    {
      "timestamp": "2025-10-17T10:05:01Z",
      "level": "INFO",
      "message": "Starting training..."
    },
    {
      "timestamp": "2025-10-17T10:05:05Z",
      "level": "INFO",
      "message": "Epoch 1/100 - Loss: 2.456"
    }
  ],
  "total_lines": 5432
}
```

---

### 3. Dataset Management

#### POST /datasets/upload

데이터셋 업로드

**Request:**
```http
POST /api/v1/datasets/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <binary>
name: "My Cat Dataset"
format: "coco"
description: "고양이 품종 데이터셋"
```

**Response (201 Created):**
```json
{
  "dataset_id": "ds_123",
  "name": "My Cat Dataset",
  "format": "coco",
  "size_mb": 450,
  "file_count": 1000,
  "status": "processing",
  "created_at": "2025-10-17T10:00:00Z"
}
```

#### GET /datasets

데이터셋 목록 조회

**Request:**
```http
GET /api/v1/datasets?page=1&limit=20
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "datasets": [
    {
      "dataset_id": "ds_123",
      "name": "My Cat Dataset",
      "format": "coco",
      "size_mb": 450,
      "file_count": 1000,
      "status": "ready",
      "created_at": "2025-10-17T10:00:00Z"
    }
  ],
  "total": 10,
  "page": 1,
  "limit": 20
}
```

#### GET /datasets/{dataset_id}

데이터셋 상세 조회

**Request:**
```http
GET /api/v1/datasets/ds_123
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "dataset_id": "ds_123",
  "name": "My Cat Dataset",
  "description": "고양이 품종 데이터셋",
  "format": "coco",
  "size_mb": 450,
  "file_count": 1000,
  "status": "ready",
  "statistics": {
    "total_images": 1000,
    "train_images": 800,
    "val_images": 150,
    "test_images": 50,
    "classes": [
      {"name": "치즈", "count": 350},
      {"name": "나비", "count": 320},
      {"name": "츄르", "count": 330}
    ]
  },
  "created_at": "2025-10-17T10:00:00Z",
  "updated_at": "2025-10-17T10:05:00Z"
}
```

#### POST /datasets/{dataset_id}/validate

데이터셋 검증

**Request:**
```http
POST /api/v1/datasets/ds_123/validate
Authorization: Bearer <token>
Content-Type: application/json

{
  "expected_format": "coco",
  "expected_classes": 3
}
```

**Response (200 OK):**
```json
{
  "dataset_id": "ds_123",
  "valid": true,
  "issues": [
    {
      "severity": "warning",
      "message": "클래스 불균형 감지 (최대 10% 차이)",
      "details": {
        "치즈": 350,
        "나비": 320,
        "츄르": 330
      }
    }
  ],
  "recommendations": [
    "데이터 증강(augmentation)을 사용하여 클래스 균형을 맞추는 것을 권장합니다."
  ]
}
```

---

### 4. Models

#### GET /models

사용 가능한 모델 목록 조회

**Request:**
```http
GET /api/v1/models?task_type=classification&framework=timm&page=1&limit=20
Authorization: Bearer <token>
```

**Query Parameters:**
- `task_type` (optional): `classification`, `detection`, `segmentation`
- `framework` (optional): `timm`, `huggingface`, `ultralytics`, etc.
- `search` (optional): 모델 이름 검색
- `page`, `limit`: 페이지네이션

**Response (200 OK):**
```json
{
  "models": [
    {
      "model_id": "resnet50",
      "name": "ResNet50",
      "framework": "timm",
      "task_types": ["classification"],
      "parameters": "25.6M",
      "pretrained_weights": ["imagenet", "imagenet21k"],
      "description": "Deep residual network with 50 layers",
      "metrics": {
        "imagenet_top1": 0.7613,
        "imagenet_top5": 0.9291
      }
    },
    {
      "model_id": "yolov8n",
      "name": "YOLOv8 Nano",
      "framework": "ultralytics",
      "task_types": ["detection"],
      "parameters": "3.2M",
      "pretrained_weights": ["coco"],
      "description": "Ultralytics YOLOv8 nano model for object detection"
    }
  ],
  "total": 150,
  "page": 1,
  "limit": 20
}
```

#### GET /models/{model_id}

모델 상세 정보

**Request:**
```http
GET /api/v1/models/resnet50
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "model_id": "resnet50",
  "name": "ResNet50",
  "framework": "timm",
  "task_types": ["classification"],
  "parameters": "25.6M",
  "pretrained_weights": ["imagenet", "imagenet21k"],
  "description": "Deep residual network with 50 layers",
  "input_size": [224, 224, 3],
  "supported_backends": ["pytorch", "onnx"],
  "metrics": {
    "imagenet_top1": 0.7613,
    "imagenet_top5": 0.9291
  },
  "example_usage": "import timm\nmodel = timm.create_model('resnet50', pretrained=True)",
  "documentation_url": "https://pytorch.org/vision/stable/models.html#resnet"
}
```

---

### 5. Inference

#### POST /inference/{workflow_id}/predict

단일 이미지 추론

**Request:**
```http
POST /api/v1/inference/wf_789xyz/predict
Authorization: Bearer <token>
Content-Type: application/json

{
  "image_url": "https://example.com/cat.jpg"
}
```

또는

```http
POST /api/v1/inference/wf_789xyz/predict
Authorization: Bearer <token>
Content-Type: application/json

{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
}
```

**Response (200 OK):**
```json
{
  "predictions": [
    {
      "class": "치즈",
      "confidence": 0.956,
      "class_id": 0
    },
    {
      "class": "나비",
      "confidence": 0.032,
      "class_id": 1
    },
    {
      "class": "츄르",
      "confidence": 0.012,
      "class_id": 2
    }
  ],
  "inference_time_ms": 45,
  "model_version": "wf_789xyz_epoch_100"
}
```

#### POST /inference/{workflow_id}/batch

배치 추론

**Request:**
```http
POST /api/v1/inference/wf_789xyz/batch
Authorization: Bearer <token>
Content-Type: application/json

{
  "images": [
    {"image_url": "https://example.com/cat1.jpg"},
    {"image_url": "https://example.com/cat2.jpg"}
  ]
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "batch_job_456",
  "status": "processing",
  "total_images": 2,
  "estimated_completion_time": "2025-10-17T10:05:00Z"
}
```

#### GET /inference/{workflow_id}/batch/{job_id}

배치 추론 결과 조회

**Request:**
```http
GET /api/v1/inference/wf_789xyz/batch/batch_job_456
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "job_id": "batch_job_456",
  "status": "completed",
  "total_images": 2,
  "completed_images": 2,
  "results": [
    {
      "image_url": "https://example.com/cat1.jpg",
      "predictions": [...]
    },
    {
      "image_url": "https://example.com/cat2.jpg",
      "predictions": [...]
    }
  ],
  "completed_at": "2025-10-17T10:05:00Z"
}
```

---

### 6. User & Projects

#### GET /user/profile

사용자 프로필 조회

**Request:**
```http
GET /api/v1/user/profile
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "user_id": "user_123",
  "email": "user@example.com",
  "name": "홍길동",
  "plan": "pro",
  "created_at": "2025-01-01T00:00:00Z",
  "usage": {
    "gpu_hours_used": 45.5,
    "gpu_hours_limit": 100,
    "storage_gb_used": 12.3,
    "storage_gb_limit": 50
  }
}
```

#### GET /projects

프로젝트 목록 조회

**Request:**
```http
GET /api/v1/projects
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "projects": [
    {
      "project_id": "proj_789",
      "name": "고양이 분류 프로젝트",
      "description": "여러 품종 분류",
      "workflow_count": 5,
      "dataset_count": 2,
      "created_at": "2025-10-01T00:00:00Z"
    }
  ]
}
```

---

## Internal APIs

서비스 간 통신용 API (외부 노출하지 않음)

### POST /internal/orchestrator/create-workflow

```http
POST /internal/orchestrator/create-workflow
X-Internal-Auth: <service_token>

{
  "experiment_name": "...",
  "user_id": "...",
  "training_config": {}
}
```

### POST /internal/models/prepare

```http
POST /internal/models/prepare
X-Internal-Auth: <service_token>

{
  "model_name": "resnet50",
  "task_type": "classification",
  "config": {}
}
```

---

## WebSocket

### ws://api/v1/ws/workflows/{workflow_id}

실시간 학습 진행상황 구독

**Connect:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/workflows/wf_789xyz?token=<access_token>');

ws.onopen = () => {
  console.log('Connected to workflow updates');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

**Message Format:**
```json
{
  "type": "training_progress",
  "workflow_id": "wf_789xyz",
  "timestamp": "2025-10-17T10:05:00Z",
  "data": {
    "epoch": 45,
    "step": 4500,
    "metrics": {
      "loss": 0.234,
      "accuracy": 0.892
    },
    "eta_minutes": 12,
    "gpu_utilization": 85,
    "memory_usage_gb": 12.5
  }
}
```

**Message Types:**
- `training_progress`: 학습 진행상황
- `training_complete`: 학습 완료
- `training_error`: 학습 오류
- `checkpoint_saved`: 체크포인트 저장됨

---

## 에러 처리

### 표준 에러 응답 형식

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "specific_field",
      "reason": "Detailed explanation"
    },
    "request_id": "req_abc123"
  }
}
```

### HTTP 상태 코드

| Status | Code | 설명 |
|--------|------|------|
| 400 | `INVALID_REQUEST` | 잘못된 요청 파라미터 |
| 401 | `UNAUTHORIZED` | 인증 실패 (토큰 없음/만료) |
| 403 | `FORBIDDEN` | 권한 없음 |
| 404 | `NOT_FOUND` | 리소스를 찾을 수 없음 |
| 409 | `CONFLICT` | 리소스 충돌 |
| 422 | `VALIDATION_ERROR` | 데이터 검증 실패 |
| 429 | `RATE_LIMIT_EXCEEDED` | 요청 제한 초과 |
| 500 | `INTERNAL_ERROR` | 서버 내부 오류 |
| 503 | `SERVICE_UNAVAILABLE` | 서비스 이용 불가 |

### 에러 응답 예시

**400 Bad Request:**
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid model name",
    "details": {
      "field": "model",
      "reason": "Model 'resnet999' not found in registry",
      "available_models": ["resnet50", "resnet101"]
    },
    "request_id": "req_abc123"
  }
}
```

**401 Unauthorized:**
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Access token has expired",
    "details": {
      "expired_at": "2025-10-17T09:00:00Z"
    }
  }
}
```

**422 Validation Error:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "errors": [
        {
          "field": "epochs",
          "message": "Must be between 1 and 1000",
          "value": 5000
        },
        {
          "field": "batch_size",
          "message": "Must be a positive integer",
          "value": -32
        }
      ]
    }
  }
}
```

---

## Rate Limiting

### 제한 정책

| Plan | Requests/minute | Requests/hour | Concurrent Workflows |
|------|----------------|---------------|---------------------|
| Free | 60 | 1000 | 1 |
| Pro | 300 | 10000 | 5 |
| Enterprise | 1000 | 50000 | 20 |

### Rate Limit 헤더

모든 API 응답에 포함:

```http
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 245
X-RateLimit-Reset: 1634472000
```

### Rate Limit 초과 시

**Response (429 Too Many Requests):**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 300,
      "reset_at": "2025-10-17T10:05:00Z"
    }
  }
}
```

---

## 다음 단계

- [아키텍처 이해하기](ARCHITECTURE.md)
- [개발 환경 설정](DEVELOPMENT.md)
- [WebSocket 통합 가이드](guides/WEBSOCKET.md)
- [인증 구현 예시](guides/AUTHENTICATION.md)
