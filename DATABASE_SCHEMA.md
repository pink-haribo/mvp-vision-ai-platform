# 데이터베이스 스키마

## 목차
- [개요](#개요)
- [PostgreSQL (Primary DB)](#postgresql-primary-db)
- [MongoDB (Document Store)](#mongodb-document-store)
- [Redis (Cache & Queue)](#redis-cache--queue)
- [마이그레이션](#마이그레이션)
- [인덱스 전략](#인덱스-전략)

## 개요

Vision Platform은 다중 데이터베이스 아키텍처를 사용합니다:

- **PostgreSQL**: 관계형 데이터 (사용자, 프로젝트, 워크플로우 메타데이터)
- **MongoDB**: 유연한 스키마가 필요한 데이터 (Config, 로그)
- **Redis**: 캐시, 세션, 실시간 상태
- **S3/MinIO**: 대용량 파일 (데이터셋, 모델 가중치)

---

## PostgreSQL (Primary DB)

### ERD 개요

```
users
  ↓ 1:N
projects
  ↓ 1:N
workflows
  ↓ 1:N
workflow_runs
  ↓ 1:N
metrics

datasets
  ↓ M:N (project_datasets)
projects
```

### 테이블 정의

#### users

사용자 계정 정보

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    avatar_url TEXT,

    -- 플랜 정보
    plan VARCHAR(20) DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'enterprise')),

    -- 사용량 제한
    gpu_hours_limit DECIMAL(10,2) DEFAULT 10.0,
    storage_gb_limit DECIMAL(10,2) DEFAULT 5.0,

    -- 타임스탬프
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE,

    -- 소프트 삭제
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_created_at ON users(created_at DESC);
```

#### projects

사용자의 프로젝트 (워크플로우의 논리적 그룹)

```sql
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- 메타데이터
    tags TEXT[] DEFAULT '{}',
    is_public BOOLEAN DEFAULT FALSE,

    -- 통계 (캐시)
    workflow_count INTEGER DEFAULT 0,
    dataset_count INTEGER DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_projects_user_id ON projects(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_projects_created_at ON projects(created_at DESC);
CREATE INDEX idx_projects_tags ON projects USING GIN(tags);
```

#### datasets

데이터셋 메타데이터

```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- 저장소 정보
    storage_path TEXT NOT NULL, -- S3 경로
    format VARCHAR(50) NOT NULL, -- 'coco', 'yolo', 'imagenet', 'pascal_voc'

    -- 통계
    size_bytes BIGINT NOT NULL,
    file_count INTEGER NOT NULL,

    -- 상태
    status VARCHAR(20) DEFAULT 'processing' CHECK (
        status IN ('processing', 'ready', 'error', 'archived')
    ),

    -- 검증 결과 (JSONB)
    validation_result JSONB,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_datasets_user_id ON datasets(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_datasets_status ON datasets(status);
CREATE INDEX idx_datasets_format ON datasets(format);
```

#### project_datasets

프로젝트와 데이터셋의 다대다 관계

```sql
CREATE TABLE project_datasets (
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,

    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (project_id, dataset_id)
);

CREATE INDEX idx_project_datasets_project ON project_datasets(project_id);
CREATE INDEX idx_project_datasets_dataset ON project_datasets(dataset_id);
```

#### workflows

학습 워크플로우 정의

```sql
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Temporal workflow ID
    temporal_workflow_id VARCHAR(255) UNIQUE,

    -- 상태
    status VARCHAR(20) DEFAULT 'pending' CHECK (
        status IN ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled')
    ),

    -- 진행률
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    current_epoch INTEGER,
    total_epochs INTEGER,

    -- 리소스 정보
    allocated_gpu_type VARCHAR(50),
    allocated_gpu_count INTEGER,

    -- 결과
    final_metrics JSONB,
    error_message TEXT,

    -- 타임스탬프
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_workflows_project_id ON workflows(project_id);
CREATE INDEX idx_workflows_user_id ON workflows(user_id);
CREATE INDEX idx_workflows_status ON workflows(status);
CREATE INDEX idx_workflows_temporal_id ON workflows(temporal_workflow_id) WHERE temporal_workflow_id IS NOT NULL;
CREATE INDEX idx_workflows_created_at ON workflows(created_at DESC);
```

#### workflow_configs

워크플로우 설정 (별도 테이블로 정규화)

```sql
CREATE TABLE workflow_configs (
    workflow_id UUID PRIMARY KEY REFERENCES workflows(id) ON DELETE CASCADE,

    -- 모델 설정
    task_type VARCHAR(50) NOT NULL, -- 'classification', 'detection', 'segmentation'
    model_name VARCHAR(255) NOT NULL,
    model_source VARCHAR(50) NOT NULL, -- 'timm', 'huggingface', 'ultralytics'

    -- 데이터 설정
    dataset_id UUID NOT NULL REFERENCES datasets(id),
    num_classes INTEGER,
    class_names TEXT[],

    -- 하이퍼파라미터 (JSONB로 유연하게 저장)
    hyperparameters JSONB NOT NULL DEFAULT '{}',

    -- 리소스 요구사항
    required_gpu_count INTEGER DEFAULT 1,
    required_memory_gb INTEGER DEFAULT 16,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_workflow_configs_dataset ON workflow_configs(dataset_id);
CREATE INDEX idx_workflow_configs_task_type ON workflow_configs(task_type);
CREATE INDEX idx_workflow_configs_model_source ON workflow_configs(model_source);
```

#### metrics

학습 메트릭 (시계열 데이터)

```sql
CREATE TABLE metrics (
    id BIGSERIAL PRIMARY KEY,
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,

    -- 메트릭 정보
    epoch INTEGER NOT NULL,
    step INTEGER NOT NULL,

    -- 성능 메트릭
    train_loss DECIMAL(10,6),
    train_accuracy DECIMAL(10,6),
    val_loss DECIMAL(10,6),
    val_accuracy DECIMAL(10,6),

    -- 시스템 메트릭
    gpu_utilization DECIMAL(5,2),
    memory_usage_gb DECIMAL(10,2),

    -- 추가 메트릭 (JSONB로 확장 가능)
    extra_metrics JSONB,

    -- 타임스탬프
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_metrics_workflow_id ON metrics(workflow_id, recorded_at DESC);
CREATE INDEX idx_metrics_epoch ON metrics(workflow_id, epoch);

-- Partitioning 고려 (대용량 데이터 시)
-- CREATE TABLE metrics_partition_YYYY_MM PARTITION OF metrics
-- FOR VALUES FROM ('YYYY-MM-01') TO ('YYYY-MM+1-01');
```

#### inference_endpoints

배포된 추론 엔드포인트

```sql
CREATE TABLE inference_endpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    endpoint_url TEXT NOT NULL UNIQUE,

    -- 상태
    status VARCHAR(20) DEFAULT 'deploying' CHECK (
        status IN ('deploying', 'active', 'inactive', 'error')
    ),

    -- 통계
    total_requests BIGINT DEFAULT 0,
    last_request_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_inference_endpoints_workflow ON inference_endpoints(workflow_id);
CREATE INDEX idx_inference_endpoints_user ON inference_endpoints(user_id);
CREATE INDEX idx_inference_endpoints_status ON inference_endpoints(status);
```

#### usage_tracking

사용량 추적 (GPU 시간, 스토리지 등)

```sql
CREATE TABLE usage_tracking (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- 리소스 타입
    resource_type VARCHAR(50) NOT NULL, -- 'gpu_hours', 'storage_gb', 'api_requests'

    -- 사용량
    amount DECIMAL(10,4) NOT NULL,
    unit VARCHAR(20) NOT NULL,

    -- 연관 엔티티
    workflow_id UUID REFERENCES workflows(id) ON DELETE SET NULL,

    -- 비용 (선택)
    cost_usd DECIMAL(10,4),

    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_usage_tracking_user_id ON usage_tracking(user_id, recorded_at DESC);
CREATE INDEX idx_usage_tracking_resource_type ON usage_tracking(resource_type);
CREATE INDEX idx_usage_tracking_workflow_id ON usage_tracking(workflow_id) WHERE workflow_id IS NOT NULL;
```

#### api_keys

API 키 관리

```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL UNIQUE, -- Hashed API key

    -- 권한
    scopes TEXT[] DEFAULT '{}', -- ['read', 'write', 'admin']

    -- 사용량
    last_used_at TIMESTAMP WITH TIME ZONE,
    total_requests BIGINT DEFAULT 0,

    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    revoked_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash) WHERE revoked_at IS NULL;
```

---

## MongoDB (Document Store)

MongoDB는 유연한 스키마가 필요한 데이터를 저장합니다.

### Collections

#### chat_sessions

대화 세션

```javascript
{
  _id: ObjectId("..."),
  sessionId: "sess_abc123",
  userId: "uuid-...",

  // 대화 메시지
  messages: [
    {
      role: "user",
      content: "ResNet50으로 고양이 분류 모델 만들어줘",
      timestamp: ISODate("2025-10-17T10:00:00Z")
    },
    {
      role: "assistant",
      content: "좋아요! 몇 가지 확인할게요...",
      timestamp: ISODate("2025-10-17T10:00:05Z"),
      parsedIntent: {
        taskType: "classification",
        model: "resnet50",
        confidence: 0.95
      }
    }
  ],

  // 파싱된 의도 (누적)
  currentIntent: {
    taskType: "classification",
    model: "resnet50",
    numClasses: 3,
    classNames: ["치즈", "나비", "츄르"],
    completeness: 85
  },

  status: "active", // "active", "completed", "abandoned"

  createdAt: ISODate("2025-10-17T10:00:00Z"),
  updatedAt: ISODate("2025-10-17T10:15:00Z")
}
```

**Indexes:**
```javascript
db.chat_sessions.createIndex({ sessionId: 1 }, { unique: true });
db.chat_sessions.createIndex({ userId: 1, createdAt: -1 });
db.chat_sessions.createIndex({ status: 1 });
```

#### training_configs

학습 설정 스냅샷 (완전한 설정을 JSON으로)

```javascript
{
  _id: ObjectId("..."),
  workflowId: "uuid-...",
  version: 1,

  // 전체 설정
  config: {
    task: {
      type: "classification",
      numClasses: 3,
      classNames: ["치즈", "나비", "츄르"]
    },
    model: {
      name: "resnet50",
      source: "timm",
      pretrained: true,
      pretrainedWeights: "imagenet"
    },
    dataset: {
      datasetId: "uuid-...",
      trainSplit: 0.8,
      valSplit: 0.15,
      testSplit: 0.05,
      augmentation: {
        randomFlip: true,
        randomRotation: 15,
        colorJitter: true
      }
    },
    training: {
      epochs: 100,
      batchSize: 32,
      learningRate: 0.001,
      optimizer: "adam",
      scheduler: {
        type: "cosine",
        warmupEpochs: 5
      },
      earlyStop: {
        enabled: true,
        patience: 10,
        metric: "val_loss"
      }
    },
    resources: {
      gpuCount: 1,
      gpuType: "T4",
      memoryGb: 16
    }
  },

  // 생성 방식
  source: "llm_parsed", // "llm_parsed", "manual", "template"

  createdAt: ISODate("2025-10-17T10:00:00Z")
}
```

**Indexes:**
```javascript
db.training_configs.createIndex({ workflowId: 1 });
db.training_configs.createIndex({ createdAt: -1 });
```

#### workflow_logs

워크플로우 로그 (stdout/stderr)

```javascript
{
  _id: ObjectId("..."),
  workflowId: "uuid-...",

  logs: [
    {
      timestamp: ISODate("2025-10-17T10:05:00Z"),
      level: "INFO",
      source: "trainer", // "trainer", "sidecar", "platform"
      message: "Starting training...",
      context: {
        epoch: 1,
        step: 0
      }
    },
    {
      timestamp: ISODate("2025-10-17T10:05:30Z"),
      level: "INFO",
      source: "trainer",
      message: "Epoch 1/100 - Loss: 2.456, Accuracy: 0.234"
    }
  ],

  // TTL (30일 후 자동 삭제)
  expiresAt: ISODate("2025-11-16T10:00:00Z")
}
```

**Indexes:**
```javascript
db.workflow_logs.createIndex({ workflowId: 1 });
db.workflow_logs.createIndex({ "logs.timestamp": 1 });
db.workflow_logs.createIndex({ expiresAt: 1 }, { expireAfterSeconds: 0 }); // TTL index
```

#### model_registry

모델 메타데이터 (timm, HuggingFace 등)

```javascript
{
  _id: ObjectId("..."),
  modelId: "resnet50",
  name: "ResNet50",
  framework: "timm",

  // 메타데이터
  taskTypes: ["classification"],
  parameters: "25.6M",
  pretrainedWeights: ["imagenet", "imagenet21k"],

  // 성능
  metrics: {
    imagenetTop1: 0.7613,
    imagenetTop5: 0.9291
  },

  // 입력 스펙
  inputSize: [224, 224, 3],
  supportedBackends: ["pytorch", "onnx"],

  // 문서
  description: "Deep residual network with 50 layers",
  documentationUrl: "https://pytorch.org/vision/stable/models.html#resnet",

  // 사용 통계
  usageCount: 1234,
  lastUsedAt: ISODate("2025-10-17T10:00:00Z"),

  createdAt: ISODate("2025-01-01T00:00:00Z"),
  updatedAt: ISODate("2025-10-17T10:00:00Z")
}
```

**Indexes:**
```javascript
db.model_registry.createIndex({ modelId: 1 }, { unique: true });
db.model_registry.createIndex({ framework: 1, taskTypes: 1 });
db.model_registry.createIndex({ name: "text" }); // Text search
```

#### audit_logs

감사 로그 (보안, 규정 준수)

```javascript
{
  _id: ObjectId("..."),
  userId: "uuid-...",
  action: "workflow.create", // "workflow.create", "dataset.upload", "user.login"

  resource: {
    type: "workflow",
    id: "uuid-..."
  },

  metadata: {
    ipAddress: "192.168.1.1",
    userAgent: "Mozilla/5.0...",
    changes: {
      before: {},
      after: {}
    }
  },

  timestamp: ISODate("2025-10-17T10:00:00Z")
}
```

**Indexes:**
```javascript
db.audit_logs.createIndex({ userId: 1, timestamp: -1 });
db.audit_logs.createIndex({ action: 1 });
db.audit_logs.createIndex({ "resource.type": 1, "resource.id": 1 });
```

---

## Redis (Cache & Queue)

Redis는 캐시, 세션, 실시간 상태를 저장합니다.

### Key Patterns

#### 세션 (Session)

```
session:{session_id}
  TTL: 7 days
  Value: JSON {userId, email, expiresAt}
```

#### JWT Refresh Token Blacklist

```
jwt:blacklist:{token_hash}
  TTL: token expiry time
  Value: "1"
```

#### 실시간 워크플로우 상태

```
workflow:status:{workflow_id}
  TTL: 24 hours
  Value: JSON {status, progress, currentEpoch, metrics}
```

#### 실시간 메트릭 (최신 값만 캐시)

```
workflow:metrics:{workflow_id}
  TTL: 1 hour
  Value: JSON {loss, accuracy, gpuUtilization, ...}
```

#### Rate Limiting

```
ratelimit:{user_id}:{endpoint}:{minute}
  TTL: 60 seconds
  Value: request count
```

#### Celery Queue (비동기 작업)

```
celery:queue:default
  List of task messages
```

#### WebSocket Pub/Sub

```
Channel: workflow:{workflow_id}:updates
  Messages: {type: "training_progress", data: {...}}
```

---

## 마이그레이션

### Alembic (PostgreSQL)

```bash
# 새 마이그레이션 생성
cd backend/orchestrator
poetry run alembic revision -m "Add workflow_configs table"

# 마이그레이션 적용
poetry run alembic upgrade head

# 롤백
poetry run alembic downgrade -1

# 히스토리 확인
poetry run alembic history
```

**마이그레이션 예시:**

```python
# alembic/versions/001_initial.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('plan', sa.String(20), server_default='free'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_index('idx_users_email', 'users', ['email'])

def downgrade():
    op.drop_table('users')
```

### MongoDB Indexes

```python
# scripts/init_mongodb.py
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT

client = MongoClient(MONGODB_URL)
db = client.vision_platform

# chat_sessions indexes
db.chat_sessions.create_index([("sessionId", ASCENDING)], unique=True)
db.chat_sessions.create_index([("userId", ASCENDING), ("createdAt", DESCENDING)])
db.chat_sessions.create_index([("status", ASCENDING)])

# training_configs indexes
db.training_configs.create_index([("workflowId", ASCENDING)])

# workflow_logs indexes
db.workflow_logs.create_index([("workflowId", ASCENDING)])
db.workflow_logs.create_index([("expiresAt", ASCENDING)], expireAfterSeconds=0)

# model_registry indexes
db.model_registry.create_index([("modelId", ASCENDING)], unique=True)
db.model_registry.create_index([("framework", ASCENDING), ("taskTypes", ASCENDING)])
db.model_registry.create_index([("name", TEXT)])

print("MongoDB indexes created successfully!")
```

---

## 인덱스 전략

### PostgreSQL

1. **Primary Keys**: 자동 인덱스
2. **Foreign Keys**: 명시적 인덱스 생성 (JOIN 성능)
3. **자주 검색하는 컬럼**: `status`, `created_at`, `user_id`
4. **복합 인덱스**: `(user_id, created_at DESC)` - 사용자별 최근 항목 조회
5. **Partial Index**: `WHERE deleted_at IS NULL` - 소프트 삭제 무시
6. **GIN Index**: JSONB, Array 컬럼

### MongoDB

1. **Unique Indexes**: `sessionId`, `modelId`
2. **Compound Indexes**: `(userId, createdAt)` - 사용자별 시계열 조회
3. **Text Search**: 모델 이름 검색
4. **TTL Indexes**: 로그 자동 삭제

### Redis

- Key Expiration으로 자동 정리
- Sorted Sets for leaderboards/rankings (필요시)

---

## 백업 전략

### PostgreSQL

```bash
# 일일 백업
pg_dump -U admin vision_platform > backup_$(date +%Y%m%d).sql

# 특정 테이블만
pg_dump -U admin -t workflows vision_platform > workflows_backup.sql

# 복원
psql -U admin vision_platform < backup_20251017.sql
```

### MongoDB

```bash
# 백업
mongodump --uri="mongodb://localhost:27017/vision_platform" --out=/backup

# 복원
mongorestore --uri="mongodb://localhost:27017/vision_platform" /backup/vision_platform
```

### Redis

```bash
# RDB 스냅샷 (자동)
save 900 1      # 900초 동안 1개 이상 변경 시 저장
save 300 10     # 300초 동안 10개 이상 변경 시 저장

# AOF (Append Only File) - 더 안전
appendonly yes
```

---

## 다음 단계

- [아키텍처 문서](ARCHITECTURE.md)
- [API 명세](API_SPECIFICATION.md)
- [개발 환경 설정](DEVELOPMENT.md)
