# Phase 12.2 ClearML Migration - Execution Plan

**작성일**: 2025-11-27
**브랜치**: `feature/phase-12.2-clearml-migration`
**예상 기간**: 6일 (Day 1-6)
**선행 작업**: Phase 12.0-12.1 완료 (PR #39)

---

## 개요

MLflow를 ClearML로 완전 전환하여 더 강력한 MLOps 플랫폼 구축. Temporal Workflow와 통합하여 실험 추적을 orchestration layer에 통합.

**핵심 변경사항**:
- MLflow → ClearML (실험 추적, 모델 레지스트리, 메트릭 시각화)
- Temporal Activity에서 ClearML Task 생성
- Training SDK에서 ClearML 자동 로깅
- Frontend에서 ClearML Web UI 임베드

---

## Day 1-2: ClearML 인프라 구축

### Day 1 오전: Docker Compose 설정

**목표**: Local dev 환경에서 ClearML Server 실행

#### 1.1 Docker Compose 파일 생성

```bash
# 새 파일 생성
touch infrastructure/docker-compose.clearml.yaml
```

**파일 내용**:
```yaml
# infrastructure/docker-compose.clearml.yaml
version: '3.8'

services:
  # Elasticsearch (ClearML dependency)
  clearml-elastic:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.9
    container_name: clearml-elastic
    restart: unless-stopped
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - cluster.name=clearml
      - bootstrap.memory_lock=true
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - clearml_elastic_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - clearml

  # MongoDB (ClearML dependency)
  clearml-mongo:
    image: mongo:7.0
    container_name: clearml-mongo
    restart: unless-stopped
    volumes:
      - clearml_mongo_data:/data/db
    ports:
      - "27018:27017"  # Avoid conflict with existing MongoDB
    networks:
      - clearml

  # Redis (ClearML dependency)
  clearml-redis:
    image: redis:7.2-alpine
    container_name: clearml-redis
    restart: unless-stopped
    volumes:
      - clearml_redis_data:/data
    ports:
      - "6380:6379"  # Avoid conflict with existing Redis
    networks:
      - clearml

  # ClearML API Server
  clearml-apiserver:
    image: allegroai/clearml:latest
    container_name: clearml-apiserver
    restart: unless-stopped
    command: apiserver
    depends_on:
      - clearml-elastic
      - clearml-mongo
      - clearml-redis
    environment:
      CLEARML_ELASTIC_SERVICE_HOST: clearml-elastic
      CLEARML_ELASTIC_SERVICE_PORT: 9200
      CLEARML_MONGODB_SERVICE_HOST: clearml-mongo
      CLEARML_MONGODB_SERVICE_PORT: 27017
      CLEARML_REDIS_SERVICE_HOST: clearml-redis
      CLEARML_REDIS_SERVICE_PORT: 6379
      CLEARML_SERVER_DEPLOYMENT_TYPE: standalone
    volumes:
      - clearml_api_logs:/var/log/clearml
      - clearml_api_config:/opt/clearml/config
    ports:
      - "8008:8008"
    networks:
      - clearml

  # ClearML Web Server
  clearml-webserver:
    image: allegroai/clearml:latest
    container_name: clearml-webserver
    restart: unless-stopped
    command: webserver
    depends_on:
      - clearml-apiserver
    environment:
      CLEARML_SERVER_API_HOST: http://clearml-apiserver:8008
    ports:
      - "8080:80"
    networks:
      - clearml

  # ClearML File Server
  clearml-fileserver:
    image: allegroai/clearml:latest
    container_name: clearml-fileserver
    restart: unless-stopped
    command: fileserver
    volumes:
      - clearml_fileserver_data:/mnt/fileserver
    ports:
      - "8081:8081"
    networks:
      - clearml

volumes:
  clearml_elastic_data:
  clearml_mongo_data:
  clearml_redis_data:
  clearml_api_logs:
  clearml_api_config:
  clearml_fileserver_data:

networks:
  clearml:
    driver: bridge
```

#### 1.2 실행 및 검증

```bash
# 실행
docker-compose -f infrastructure/docker-compose.clearml.yaml up -d

# 로그 확인
docker-compose -f infrastructure/docker-compose.clearml.yaml logs -f

# 상태 확인
docker-compose -f infrastructure/docker-compose.clearml.yaml ps

# Health check
curl http://localhost:8008/api/v1.5/debug.ping
curl http://localhost:8080  # Web UI
```

**예상 결과**:
```json
{
  "meta": {
    "status": "ok"
  },
  "data": "pong"
}
```

**Checklist**:
- [ ] docker-compose.clearml.yaml 작성
- [ ] Docker Compose 실행 (6개 컨테이너)
- [ ] Elasticsearch health check: http://localhost:9200
- [ ] ClearML API health check: http://localhost:8008/api/v1.5/debug.ping
- [ ] ClearML Web UI 접속: http://localhost:8080

---

### Day 1 오후: ClearML 계정 및 API 키 생성

#### 1.3 Web UI 초기 설정

```bash
# 1. Web UI 접속
open http://localhost:8080

# 2. 초기 계정 생성
#    Username: admin
#    Password: admin (기본값)

# 3. Settings → Workspace → Create new credentials
#    - Name: Platform Backend
#    - Description: Vision AI Training Platform
#    - Copy Access Key + Secret Key
```

#### 1.4 환경변수 설정

```bash
# platform/backend/.env에 추가
echo "
# ClearML Configuration
CLEARML_API_HOST=http://localhost:8008
CLEARML_WEB_HOST=http://localhost:8080
CLEARML_FILES_HOST=http://localhost:8081
CLEARML_API_ACCESS_KEY=<YOUR_ACCESS_KEY>
CLEARML_API_SECRET_KEY=<YOUR_SECRET_KEY>
" >> .env
```

#### 1.5 연결 테스트

```python
# test_clearml_connection.py
from clearml import Task
import os

# Test connection
try:
    task = Task.init(
        project_name="Test Project",
        task_name="Connection Test",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False
    )

    # Log test metric
    task.logger.report_scalar(
        title="test",
        series="connection",
        value=1.0,
        iteration=0
    )

    task.mark_completed()

    print(f"✓ ClearML connection successful!")
    print(f"  Task ID: {task.id}")
    print(f"  Web UI: http://localhost:8080/projects/*/experiments/{task.id}")

except Exception as e:
    print(f"✗ ClearML connection failed: {e}")
```

```bash
# 실행
cd platform/backend
python test_clearml_connection.py
```

**Checklist**:
- [ ] Web UI 계정 생성
- [ ] API 키 생성
- [ ] .env 업데이트
- [ ] 연결 테스트 성공
- [ ] Web UI에서 Test Task 확인

---

### Day 2: Backend 설정 업데이트

#### 2.1 Config 업데이트

```python
# platform/backend/app/core/config.py
class Settings(BaseSettings):
    # ... existing fields ...

    # ClearML Configuration
    CLEARML_API_HOST: str = "http://localhost:8008"
    CLEARML_WEB_HOST: str = "http://localhost:8080"
    CLEARML_FILES_HOST: str = "http://localhost:8081"
    CLEARML_API_ACCESS_KEY: str
    CLEARML_API_SECRET_KEY: str

    # ClearML Options
    CLEARML_DEFAULT_PROJECT: str = "Platform Training"
    CLEARML_AUTO_CONNECT_FRAMEWORKS: bool = False  # Manual control
```

#### 2.2 Requirements 업데이트

```bash
# platform/backend/requirements.txt에 추가
clearml==1.16.2
```

```bash
# 설치
cd platform/backend
pip install clearml==1.16.2

# 또는 poetry 사용 시
poetry add clearml==1.16.2
```

#### 2.3 ClearML 초기화

```python
# platform/backend/app/main.py
from clearml import Task
from app.core.config import settings

@app.on_event("startup")
async def startup_event():
    # ... existing startup code ...

    # Initialize ClearML SDK
    logger.info("[STARTUP] Initializing ClearML SDK...")
    os.environ['CLEARML_API_HOST'] = settings.CLEARML_API_HOST
    os.environ['CLEARML_WEB_HOST'] = settings.CLEARML_WEB_HOST
    os.environ['CLEARML_FILES_HOST'] = settings.CLEARML_FILES_HOST
    os.environ['CLEARML_API_ACCESS_KEY'] = settings.CLEARML_API_ACCESS_KEY
    os.environ['CLEARML_API_SECRET_KEY'] = settings.CLEARML_API_SECRET_KEY

    logger.info(f"[CLEARML] API Host: {settings.CLEARML_API_HOST}")
    logger.info(f"[CLEARML] Web UI: {settings.CLEARML_WEB_HOST}")
```

**Checklist**:
- [ ] Config 클래스 업데이트
- [ ] requirements.txt 업데이트
- [ ] clearml 패키지 설치
- [ ] main.py startup hook 추가
- [ ] Backend 재시작 및 로그 확인

---

## Day 3-4: ClearMLService & Database

### Day 3 오전: ClearMLService 구현

#### 3.1 Service 클래스 생성

```bash
# 새 파일 생성
touch platform/backend/app/services/clearml_service.py
```

**파일 구조**:
```python
# platform/backend/app/services/clearml_service.py
from typing import Optional, Dict, List, Any
from clearml import Task, Model
from sqlalchemy.orm import Session
from app.db import models
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class ClearMLService:
    """ClearML integration service for experiment tracking"""

    def __init__(self, db: Session):
        self.db = db

    # 1. Task Management
    def create_task(
        self,
        job_id: int,
        task_name: str,
        task_type: str,
        project_name: Optional[str] = None
    ) -> Optional[str]:
        """Create ClearML task for training job"""
        # Implementation
        pass

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get ClearML task by ID"""
        # Implementation
        pass

    # 2. Metrics Logging
    def log_metrics(
        self,
        task_id: str,
        metrics: Dict[str, float],
        iteration: int
    ):
        """Log metrics to ClearML task"""
        # Implementation
        pass

    # 3. Artifact Management
    def upload_artifact(
        self,
        task_id: str,
        artifact_name: str,
        artifact_object: Any,
        metadata: Optional[Dict] = None
    ):
        """Upload artifact (checkpoint, model, etc.)"""
        # Implementation
        pass

    # 4. Task Status
    def mark_completed(self, task_id: str):
        """Mark task as completed"""
        # Implementation
        pass

    def mark_failed(self, task_id: str, status_reason: str):
        """Mark task as failed"""
        # Implementation
        pass

    # 5. Model Registry
    def register_model(
        self,
        task_id: str,
        model_path: str,
        model_name: str,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """Register model in ClearML Model Repository"""
        # Implementation
        pass
```

**전체 구현**은 CLEARML_MIGRATION_PLAN.md의 Phase 2.1 참조.

#### 3.2 Unit Tests

```python
# platform/backend/tests/services/test_clearml_service.py
import pytest
from app.services.clearml_service import ClearMLService
from app.db.database import SessionLocal


class TestClearMLService:
    def test_create_task(self):
        """Test ClearML task creation"""
        db = SessionLocal()
        service = ClearMLService(db)

        task_id = service.create_task(
            job_id=1,
            task_name="Test Task",
            task_type="training"
        )

        assert task_id is not None
        assert len(task_id) > 0

        # Cleanup
        task = service.get_task(task_id)
        task.delete()
        db.close()

    def test_log_metrics(self):
        """Test metrics logging"""
        # Implementation
        pass

    def test_upload_artifact(self):
        """Test artifact upload"""
        # Implementation
        pass
```

**Checklist**:
- [ ] ClearMLService 클래스 생성
- [ ] create_task() 구현
- [ ] get_task() 구현
- [ ] log_metrics() 구현
- [ ] upload_artifact() 구현
- [ ] mark_completed/failed() 구현
- [ ] register_model() 구현
- [ ] Unit tests 작성 및 통과

---

### Day 3 오후 - Day 4 오전: Database Migration

#### 4.1 Migration Script 작성

```bash
# 새 파일 생성
touch platform/backend/migrate_add_clearml_fields.py
```

```python
# platform/backend/migrate_add_clearml_fields.py
"""
Add clearml_task_id column to training/inference/export jobs.

Phase 12.2: ClearML Migration
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"[MIGRATION] Loaded .env from: {env_path}")

sys.path.insert(0, str(backend_dir))

from sqlalchemy import text
from app.db.database import SessionLocal


def migrate_add_clearml_fields():
    """Add clearml_task_id columns to job tables."""
    db = SessionLocal()

    try:
        print("[MIGRATION] Adding clearml_task_id columns...")

        # 1. TrainingJob
        print("  Adding to training_jobs...")
        db.execute(text("""
            ALTER TABLE training_jobs
            ADD COLUMN IF NOT EXISTS clearml_task_id VARCHAR(255)
        """))
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_training_jobs_clearml_task_id
            ON training_jobs(clearml_task_id)
        """))

        # 2. InferenceJob
        print("  Adding to inference_jobs...")
        db.execute(text("""
            ALTER TABLE inference_jobs
            ADD COLUMN IF NOT EXISTS clearml_task_id VARCHAR(255)
        """))
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_inference_jobs_clearml_task_id
            ON inference_jobs(clearml_task_id)
        """))

        # 3. ExportJob
        print("  Adding to export_jobs...")
        db.execute(text("""
            ALTER TABLE export_jobs
            ADD COLUMN IF NOT EXISTS clearml_task_id VARCHAR(255)
        """))
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_export_jobs_clearml_task_id
            ON export_jobs(clearml_task_id)
        """))

        db.commit()
        print("[MIGRATION] Successfully added clearml_task_id columns")

    except Exception as e:
        print(f"[MIGRATION] Error during migration: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 80)
    print("Phase 12.2: Add clearml_task_id columns")
    print("=" * 80)
    migrate_add_clearml_fields()
    print("=" * 80)
    print("Migration completed successfully!")
    print("=" * 80)
```

#### 4.2 Model 업데이트

```python
# platform/backend/app/db/models.py
class TrainingJob(Base):
    __tablename__ = "training_jobs"

    # ... existing fields ...

    # ClearML Task ID (Phase 12.2: ClearML Migration)
    clearml_task_id = Column(String(255), nullable=True, index=True)

    # MLflow Run ID (deprecated - will be removed after migration)
    mlflow_run_id = Column(String(255), nullable=True, index=True)


class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    # ... existing fields ...

    # ClearML Task ID (Phase 12.2: ClearML Migration)
    clearml_task_id = Column(String(255), nullable=True, index=True)


class ExportJob(Base):
    __tablename__ = "export_jobs"

    # ... existing fields ...

    # ClearML Task ID (Phase 12.2: ClearML Migration)
    clearml_task_id = Column(String(255), nullable=True, index=True)
```

#### 4.3 Migration 실행

```bash
cd platform/backend
python migrate_add_clearml_fields.py
```

**Checklist**:
- [ ] Migration script 작성
- [ ] models.py 업데이트
- [ ] Migration 실행
- [ ] Database schema 확인
- [ ] Rollback 테스트

---

### Day 4 오후: Temporal Activity 통합

#### 4.4 create_clearml_task Activity 구현

```python
# platform/backend/app/workflows/training_workflow.py
from app.services.clearml_service import ClearMLService

@activity.defn(name="create_clearml_task")
async def create_clearml_task(job_id: int) -> str:
    """
    Create ClearML Task for experiment tracking.

    Args:
        job_id: TrainingJob ID

    Returns:
        ClearML Task ID
    """
    logger.info(f"[Activity] create_clearml_task - job_id={job_id}")

    db = SessionLocal()
    try:
        # 1. Load TrainingJob
        job = db.query(models.TrainingJob).filter(
            models.TrainingJob.id == job_id
        ).first()

        if not job:
            raise ValueError(f"TrainingJob {job_id} not found")

        # 2. Create ClearML Task
        clearml_service = ClearMLService(db)

        task_id = clearml_service.create_task(
            job_id=job_id,
            task_name=f"Training Job {job_id} - {job.model_name}",
            task_type="training",
            project_name=f"Project {job.project_id}" if job.project_id else "Platform Training"
        )

        if not task_id:
            raise RuntimeError("Failed to create ClearML task")

        logger.info(f"[create_clearml_task] Created task: {task_id}")
        logger.info(f"[create_clearml_task] Web UI: {settings.CLEARML_WEB_HOST}/projects/*/experiments/{task_id}")

        return task_id

    finally:
        db.close()
```

**Checklist**:
- [ ] create_clearml_task activity 구현
- [ ] Temporal Worker 재시작
- [ ] E2E 테스트 (test_workflow.py)
- [ ] ClearML Web UI에서 Task 확인

---

## Day 5: Training SDK 통합

### Day 5 오전: SDK 업데이트

#### 5.1 TrainerSDK ClearML 통합

```python
# platform/trainers/ultralytics/trainer_sdk.py
from clearml import Task
from typing import Optional


class TrainerSDK:
    def __init__(self):
        # ... existing initialization ...

        # ClearML Task (automatically detected if running in context)
        self.clearml_task: Optional[Task] = None
        self._init_clearml_task()

    def _init_clearml_task(self):
        """Initialize ClearML Task if running in Temporal Workflow"""
        try:
            # Try to get current task (created by Temporal Activity)
            self.clearml_task = Task.current_task()

            if self.clearml_task:
                logger.info(f"[TrainerSDK] ClearML Task detected: {self.clearml_task.id}")
                logger.info(f"[TrainerSDK] Web UI: {self.clearml_task.get_output_log_web_page()}")

                # Connect configuration
                config = self.get_full_config()
                self.clearml_task.connect_configuration(config)
            else:
                logger.warning("[TrainerSDK] No ClearML Task context (running standalone)")

        except Exception as e:
            logger.warning(f"[TrainerSDK] ClearML initialization failed: {e}")
            self.clearml_task = None

    def report_progress(
        self,
        epoch: int,
        total_epochs: int,
        metrics: TrainingCallbackMetrics
    ):
        """Report training progress to Backend and ClearML"""
        # 1. Backend callback (existing)
        callback_data = {
            "operation_type": "training",
            "epoch": epoch,
            "total_epochs": total_epochs,
            "metrics": metrics.dict()
        }
        self.http_client.post(
            f"{self.callback_url}/progress",
            json=callback_data
        )

        # 2. ClearML metrics logging (new)
        if self.clearml_task:
            self._log_metrics_to_clearml(epoch, metrics)

    def _log_metrics_to_clearml(self, iteration: int, metrics: TrainingCallbackMetrics):
        """Log metrics to ClearML Task"""
        if not self.clearml_task:
            return

        metrics_dict = metrics.dict()

        for metric_name, value in metrics_dict.items():
            if value is None:
                continue

            # Parse metric name (e.g., "train_loss" -> series="train", title="loss")
            if "_" in metric_name:
                series, title = metric_name.split("_", 1)
            else:
                series = "metrics"
                title = metric_name

            self.clearml_task.logger.report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=iteration
            )

    def upload_checkpoint(
        self,
        local_path: str,
        checkpoint_type: str,
        is_best: bool = False,
        metadata: Optional[Dict] = None
    ):
        """Upload checkpoint to S3 and register in ClearML"""
        # 1. Upload to S3 (existing)
        s3_key = f"checkpoints/{self.job_id}/{checkpoint_type}.pt"
        self.internal_storage.upload_file(local_path, s3_key)

        # 2. Register in ClearML (new)
        if self.clearml_task:
            self.clearml_task.upload_artifact(
                name=f"checkpoint_{checkpoint_type}",
                artifact_object=local_path,
                metadata={
                    "checkpoint_type": checkpoint_type,
                    "is_best": is_best,
                    "s3_key": s3_key,
                    **(metadata or {})
                }
            )

            # If best checkpoint, register as output model
            if is_best:
                from clearml import OutputModel
                output_model = OutputModel(task=self.clearml_task)
                output_model.update_weights(weights_filename=local_path)

        # 3. Backend callback (existing)
        # ...

    def report_completed(self, final_metrics: Dict[str, Any]):
        """Report training completion"""
        # 1. Backend callback (existing)
        # ...

        # 2. Mark ClearML Task as completed (new)
        if self.clearml_task:
            self.clearml_task.mark_completed()
            logger.info(f"[TrainerSDK] ClearML Task marked as completed")
```

**Checklist**:
- [ ] TrainerSDK에 ClearML 통합
- [ ] _init_clearml_task() 구현
- [ ] _log_metrics_to_clearml() 구현
- [ ] upload_checkpoint() ClearML 연동
- [ ] report_completed() ClearML 연동

---

### Day 5 오후: E2E 테스트

#### 5.2 통합 테스트

```bash
# 1. Temporal Worker 실행
cd platform/backend
python -m app.workflows.worker

# 2. 테스트 실행
python test_workflow.py

# 예상 출력:
# [OK] Found dataset: 1 (COCO-2017-subset)
# [OK] Created training job: 10
# [OK] Workflow started: training-job-10
# [OK] ClearML Task created: abc123def456
# [OK] Web UI: http://localhost:8080/projects/*/experiments/abc123def456
# [OK] Training subprocess started (PID: 12345)
# ...
```

#### 5.3 Web UI 확인

```bash
# ClearML Web UI 접속
open http://localhost:8080

# 확인 사항:
# - Project "Project 1" 존재
# - Task "Training Job 10 - yolo11n" 존재
# - Metrics 실시간 업데이트 (train_loss, val_accuracy, etc.)
# - Artifacts (checkpoints) 업로드 확인
# - Output Model 등록 확인
```

**Checklist**:
- [ ] E2E 테스트 성공
- [ ] ClearML Task 자동 생성
- [ ] Metrics 실시간 로깅
- [ ] Checkpoint 업로드
- [ ] Output Model 등록
- [ ] Web UI에서 모든 정보 확인 가능

---

## Day 6: Frontend 통합 & 문서화

### Day 6 오전: Frontend ClearML UI

#### 6.1 ClearMLPanel 컴포넌트

```bash
# 새 파일 생성
touch platform/frontend/components/training/ClearMLPanel.tsx
```

```typescript
// platform/frontend/components/training/ClearMLPanel.tsx
import { useState, useEffect } from 'react';

interface ClearMLPanelProps {
  taskId: string | null;
}

export function ClearMLPanel({ taskId }: ClearMLPanelProps) {
  const clearmlWebHost = process.env.NEXT_PUBLIC_CLEARML_WEB_HOST || 'http://localhost:8080';

  if (!taskId) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        No ClearML Task associated with this job
      </div>
    );
  }

  const iframeUrl = `${clearmlWebHost}/projects/*/experiments/${taskId}`;

  return (
    <div className="clearml-panel h-full">
      <div className="mb-2 text-sm text-gray-600">
        ClearML Task: <a href={iframeUrl} target="_blank" className="text-blue-600 hover:underline">{taskId}</a>
      </div>
      <iframe
        src={iframeUrl}
        className="w-full h-[600px] border border-gray-300 rounded"
        title="ClearML Task View"
      />
    </div>
  );
}
```

#### 6.2 TrainingPanel 업데이트

```typescript
// platform/frontend/components/training/TrainingPanel.tsx
import { ClearMLPanel } from './ClearMLPanel';

export function TrainingPanel({ jobId }: { jobId: number }) {
  const { data: job } = useQuery(['trainingJob', jobId], () => fetchJob(jobId));

  return (
    <Tabs defaultValue="overview">
      <TabsList>
        <TabsTrigger value="overview">Overview</TabsTrigger>
        <TabsTrigger value="metrics">Metrics</TabsTrigger>
        <TabsTrigger value="logs">Logs</TabsTrigger>
        <TabsTrigger value="clearml">ClearML</TabsTrigger>  {/* NEW */}
      </TabsList>

      {/* ... other tabs ... */}

      <TabsContent value="clearml">
        <ClearMLPanel taskId={job?.clearml_task_id} />
      </TabsContent>
    </Tabs>
  );
}
```

**Checklist**:
- [ ] ClearMLPanel 컴포넌트 생성
- [ ] TrainingPanel에 ClearML 탭 추가
- [ ] 환경변수 추가 (NEXT_PUBLIC_CLEARML_WEB_HOST)
- [ ] iframe 렌더링 확인
- [ ] Task 링크 동작 확인

---

### Day 6 오후: 문서화 및 정리

#### 6.3 문서 업데이트

```bash
# 1. IMPLEMENTATION_TO_DO_LIST.md 업데이트
# Phase 12.2 체크리스트 완료 표시

# 2. ARCHITECTURE.md 업데이트
# ClearML 통합 아키텍처 다이어그램 추가

# 3. DEVELOPMENT.md 업데이트
# ClearML 설정 가이드 추가
```

#### 6.4 Migration Guide 작성

```bash
# 새 문서 생성
touch docs/guides/CLEARML_MIGRATION_GUIDE.md
```

**내용**:
- ClearML 인프라 설정 방법
- 환경변수 설정 가이드
- Database migration 실행 방법
- 기존 MLflow 데이터 마이그레이션
- Rollback 절차

**Checklist**:
- [ ] IMPLEMENTATION_TO_DO_LIST.md 업데이트
- [ ] ARCHITECTURE.md ClearML 섹션 추가
- [ ] DEVELOPMENT.md ClearML 설정 가이드
- [ ] CLEARML_MIGRATION_GUIDE.md 작성
- [ ] README.md 업데이트

---

## Commit Strategy

### Commit Sequence

```bash
# Day 1-2: Infrastructure
git add infrastructure/docker-compose.clearml.yaml
git add platform/backend/.env.example
git commit -m "feat(phase12.2): add ClearML Docker Compose setup"

git add platform/backend/app/core/config.py
git add platform/backend/requirements.txt
git commit -m "feat(phase12.2): add ClearML configuration and dependencies"

# Day 3: Service
git add platform/backend/app/services/clearml_service.py
git add platform/backend/tests/services/test_clearml_service.py
git commit -m "feat(phase12.2): implement ClearMLService"

# Day 4: Database & Temporal
git add platform/backend/migrate_add_clearml_fields.py
git add platform/backend/app/db/models.py
git commit -m "feat(phase12.2): add clearml_task_id database fields"

git add platform/backend/app/workflows/training_workflow.py
git commit -m "feat(phase12.2): integrate ClearML with Temporal Workflow"

# Day 5: SDK
git add platform/trainers/ultralytics/trainer_sdk.py
git commit -m "feat(phase12.2): integrate ClearML into TrainerSDK"

# Day 6: Frontend & Docs
git add platform/frontend/components/training/ClearMLPanel.tsx
git add platform/frontend/components/training/TrainingPanel.tsx
git commit -m "feat(phase12.2): add ClearML UI integration"

git add docs/
git commit -m "docs(phase12.2): update documentation for ClearML migration"
```

---

## Testing Checklist

### Unit Tests
- [ ] ClearMLService.create_task()
- [ ] ClearMLService.log_metrics()
- [ ] ClearMLService.upload_artifact()
- [ ] ClearMLService.register_model()

### Integration Tests
- [ ] Temporal Activity: create_clearml_task
- [ ] TrainerSDK ClearML 통합
- [ ] Frontend ClearMLPanel 렌더링

### E2E Tests
- [ ] Complete training flow with ClearML
- [ ] Metrics 실시간 업데이트
- [ ] Checkpoint 업로드 및 등록
- [ ] Web UI에서 모든 정보 확인

### Performance Tests
- [ ] ClearML API 응답 시간 (<100ms)
- [ ] Metrics logging 지연 (<50ms)
- [ ] Artifact upload 속도 (>10MB/s)

---

## Rollback Plan

### If Migration Fails

```bash
# 1. Revert code
git revert HEAD~7..HEAD  # Revert last 7 commits

# 2. Stop ClearML
docker-compose -f infrastructure/docker-compose.clearml.yaml down -v

# 3. Database rollback
cd platform/backend
python -c "
from sqlalchemy import text
from app.db.database import SessionLocal

db = SessionLocal()
db.execute(text('ALTER TABLE training_jobs DROP COLUMN clearml_task_id'))
db.execute(text('ALTER TABLE inference_jobs DROP COLUMN clearml_task_id'))
db.execute(text('ALTER TABLE export_jobs DROP COLUMN clearml_task_id'))
db.commit()
db.close()
"

# 4. Restart backend
docker-compose restart backend
```

---

## Success Criteria

### Infrastructure
- [x] ClearML Server 실행 (6개 컨테이너)
- [x] Web UI 접속 가능
- [x] API Health check 성공

### Backend
- [ ] ClearMLService 완전 구현
- [ ] Temporal Activity 통합
- [ ] Database migration 성공

### SDK
- [ ] ClearML 자동 초기화
- [ ] Metrics 실시간 로깅
- [ ] Checkpoint 자동 업로드

### Frontend
- [ ] ClearML UI 임베드
- [ ] Task 정보 표시

### Documentation
- [ ] 모든 문서 업데이트
- [ ] Migration guide 작성

---

## Next Steps (After Phase 12.2)

### Phase 12.3: Storage Pattern Unification (1 day)
- DualStorageClient 패턴 정리
- S3 경로 일관성 확보

### Phase 12.4: Callback Logic Refactoring (1 day)
- Callback endpoint 정리
- WebSocket 통합

### Phase 12.5: Testing & Documentation (2 days)
- 전체 Phase 12 통합 테스트
- 최종 문서 정리

---

**작성자**: Claude
**검토 필요**: Day 3 ClearMLService 구현 검증
**다음 단계**: Day 1 시작 승인 대기
