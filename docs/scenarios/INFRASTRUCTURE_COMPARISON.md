# 인프라 비교: 로컬 vs Railway vs 필요 여부

## 현재 상태 요약

| 서비스 | docker-compose | Railway | MVP에서 실제 사용? | 상태 |
|--------|----------------|---------|-------------------|------|
| **PostgreSQL** | ✅ Container | ✅ Managed | ✅ 사용 중 | 필수 |
| **MLflow** | ✅ Container | ❌ 없음 | ✅ 사용 중 | **누락!** |
| **MinIO** | ✅ Container | ❌ 없음 | ✅ 사용 중 | **누락!** |
| **Prometheus** | ✅ Container | ❌ 없음 | ✅ 사용 중 | 선택 |
| **Grafana** | ✅ Container | ❌ 없음 | ✅ 사용 중 | 선택 |
| **MongoDB** | ✅ Container | ❌ 없음 | ❌ 미사용 | 불필요 |
| **Redis** | ✅ Container | ❌ 없음 | ❌ 미사용 | 불필요 |
| **Temporal** | ✅ Container | ❌ 없음 | ❌ 미사용 | 불필요 |
| **Mailhog** | ✅ Container | ❌ 없음 | ❌ 미사용 | 불필요 |
| **pgAdmin** | ⚙️ Optional | ❌ 없음 | ❌ 미사용 | 개발 도구 |
| **mongo-express** | ⚙️ Optional | ❌ 없음 | ❌ 미사용 | 개발 도구 |

---

## 1. PostgreSQL ✅ (필수)

### 로컬
```yaml
# docker-compose.yml
postgres:
  image: postgres:16-alpine
  ports: ["5432:5432"]
  environment:
    POSTGRES_DB: vision_platform
    POSTGRES_USER: admin
    POSTGRES_PASSWORD: devpass
```

### Railway
- **Managed PostgreSQL Service** 사용
- URL 형식: `postgresql://user:pass@region.railway.app:5432/railway`
- 자동 백업, 스케일링 지원

### MVP에서 사용 여부
✅ **필수 사용 중**
- `mvp/backend/app/db/models.py` - User, Project, TrainingJob 등 모든 데이터
- `mvp/training/train.py` - advanced_config 로딩

### 결론
- **로컬**: Docker container 계속 사용 ✅
- **Railway**: Managed service 계속 사용 ✅
- **상태**: 정상 ✅

---

## 2. MLflow ⚠️ (필수인데 Railway에 없음!)

### 로컬
```yaml
# docker-compose.yml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.9.2
  ports: ["5000:5000"]
  command: mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db
  environment:
    - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
```

### Railway
❌ **배포되지 않음!**

### MVP에서 사용 여부
✅ **실제로 사용 중!**

**증거:**
```python
# mvp/training/train.py:27-30
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
```

**사용 위치:**
- `mvp/training/adapters/base.py` - `mlflow.start_run()`, `mlflow.log_metric()`
- `mvp/training/platform_sdk/base.py` - MLflow 초기화
- 모든 Adapter에서 실험 추적

### 결론
- **로컬**: 계속 사용 ✅
- **Railway**: **MLflow 서비스 추가 필요!** ⚠️
- **대안** (단기):
  - Railway에 별도 MLflow 서비스 배포
  - 또는 MLflow 없이 DB만 사용 (기능 제한됨)

---

## 3. MinIO (S3-compatible storage) ⚠️ (필수인데 Railway에 없음!)

### 로컬
```yaml
# docker-compose.yml
minio:
  image: minio/minio:latest
  ports: ["9000:9000", "9001:9001"]
  command: server /data --console-address ":9001"
  environment:
    MINIO_ROOT_USER: minioadmin
    MINIO_ROOT_PASSWORD: minioadmin
```

### Railway
❌ **배포되지 않음!**

### MVP에서 사용 여부
✅ **실제로 사용 중!**

**증거:**
```python
# mvp/training/train.py:28-30
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
```

**사용 목적:**
- MLflow artifacts 저장 (모델 체크포인트, 실험 결과)
- `--default-artifact-root s3://vision-platform-dev/mlflow-artifacts`

### 결론
- **로컬**: 계속 사용 ✅
- **Railway**: **S3 storage 추가 필요!** ⚠️
- **대안**:
  1. **외부 S3** (AWS S3, Cloudflare R2, Backblaze B2)
  2. **Railway Volumes** (ephemeral, 재시작 시 삭제됨 - 비추천)
  3. **MinIO 직접 배포** (별도 Railway 서비스)

---

## 4. Prometheus + Grafana 🤔 (선택적)

### 로컬
```yaml
# docker-compose.yml
prometheus:
  image: prom/prometheus:latest
  ports: ["9090:9090"]

grafana:
  image: grafana/grafana:latest
  ports: ["3001:3000"]
```

### Railway
❌ **배포되지 않음**

### MVP에서 사용 여부
✅ **사용 중이지만 필수는 아님**

**증거:**
- `mvp/backend/app/utils/metrics.py` - Prometheus 메트릭 export
- `mvp/frontend/components/GrafanaEmbed.tsx` - Grafana iframe 임베딩

### 결론
- **로컬**: 계속 사용 (개발 편의성) ✅
- **Railway**: 생략 가능 ✅
  - **이유**: 프론트엔드에서 `/api/v1/training/{id}/metrics` API로 직접 차트 표시 가능
  - Grafana 임베딩은 "있으면 좋지만" 필수 아님

---

## 5. MongoDB ❌ (계획에만 있고 미사용)

### 로컬
```yaml
# docker-compose.yml
mongodb:
  image: mongo:7
  ports: ["27017:27017"]
```

### Railway
❌ 없음

### MVP에서 사용 여부
❌ **전혀 사용하지 않음**

**증거:**
```bash
grep -r "mongodb" mvp/backend mvp/training
# 결과: 0개
```

**원래 계획** (CLAUDE.md):
- MongoDB 7 (configs, workflow definitions)
- → Temporal 워크플로우용

### 결론
- **로컬**: **불필요! 중지 가능** ✅
- **Railway**: 추가 불필요 ✅
- **docker-compose**: 제거 또는 `profiles: [future]`로 이동

---

## 6. Redis ❌ (계획에만 있고 미사용)

### 로컬
```yaml
# docker-compose.yml
redis:
  image: redis:7.2-alpine
  ports: ["6379:6379"]
```

### Railway
❌ 없음

### MVP에서 사용 여부
❌ **전혀 사용하지 않음**

**증거:**
```bash
grep -r "redis" mvp/backend mvp/training
# 결과: 0개 (import redis, RedisClient 등 없음)
```

**원래 계획** (CLAUDE.md):
- Redis 7.2 (cache, Celery queue, real-time state)

### 결론
- **로컬**: **불필요! 중지 가능** ✅
- **Railway**: 추가 불필요 ✅
- **docker-compose**: 제거 또는 `profiles: [future]`로 이동

---

## 7. Temporal ❌ (계획에만 있고 미사용)

### 로컬
```yaml
# docker-compose.yml
temporal:
  image: temporalio/auto-setup:latest
  ports: ["7233:7233", "8233:8233"]
```

### Railway
❌ 없음

### MVP에서 사용 여부
❌ **전혀 사용하지 않음**

**증거:**
```bash
grep -r "temporal" mvp/backend mvp/training
# 결과: 0개
```

**원래 계획** (CLAUDE.md):
- Temporal 1.22.x for workflow orchestration

### 결론
- **로컬**: **불필요! 중지 가능** ✅
- **Railway**: 추가 불필요 ✅
- **docker-compose**: 제거 또는 `profiles: [future]`로 이동

---

## 8. Mailhog ❌ (개발 도구)

### 로컬
```yaml
# docker-compose.yml
mailhog:
  image: mailhog/mailhog:latest
  ports: ["1025:1025", "8025:8025"]
```

### MVP에서 사용 여부
❌ **전혀 사용하지 않음**

**원래 목적:**
- 개발 환경에서 이메일 전송 테스트
- 회원가입, 비밀번호 재설정 등

### 결론
- **로컬**: 불필요, 중지 가능 ✅
- **Railway**: 추가 불필요 ✅

---

## 9. pgAdmin / mongo-express ⚙️ (개발 도구)

### 로컬
```yaml
# docker-compose.yml
pgadmin:
  profiles: [tools]  # --profile tools 필요

mongo-express:
  profiles: [tools]
```

### 결론
- **로컬**: profiles로 분리되어 있어 기본 실행 안 됨 ✅
- 필요할 때만 `docker-compose --profile tools up` 사용
- **Railway**: 추가 불필요 ✅

---

## 권장 사항

### 즉시 조치 필요 (Production Branch)

#### 1. Railway에 MLflow 추가
```yaml
# railway.toml (새로 생성)
[[services]]
name = "mlflow-service"
dockerfile = "docker/mlflow.Dockerfile"

[services.env]
MLFLOW_BACKEND_STORE_URI = "postgresql://..."
MLFLOW_DEFAULT_ARTIFACT_ROOT = "s3://..."
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."
```

#### 2. S3 Storage 설정
**옵션 A: AWS S3 사용 (권장)**
```bash
# Railway 환경 변수
AWS_S3_BUCKET=vision-platform-prod
AWS_REGION=ap-northeast-2
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

**옵션 B: Cloudflare R2 사용 (저렴)**
```bash
# Railway 환경 변수
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
S3_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
```

#### 3. Training Services 환경 변수 업데이트
```bash
# timm-service, ultralytics-service, huggingface-service
MLFLOW_TRACKING_URI=https://mlflow-service-production-xxxx.up.railway.app
AWS_S3_ENDPOINT_URL=https://...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 로컬 docker-compose 정리

#### `docker-compose.mvp.yml` (MVP에서 실제 사용)
```yaml
version: '3.8'

services:
  postgres:
    # 현재 설정 유지

  mlflow:
    # 현재 설정 유지

  minio:
    # 현재 설정 유지

  minio-init:
    # 현재 설정 유지

  prometheus:
    # 선택적 (프론트엔드 차트로 대체 가능)
    profiles: [monitoring]

  grafana:
    # 선택적 (프론트엔드 차트로 대체 가능)
    profiles: [monitoring]

# MongoDB, Redis, Temporal은 제거
```

#### `docker-compose.full.yml` (전체 아키텍처, 미래용)
```yaml
# 모든 서비스 포함 (MongoDB, Redis, Temporal 등)
# 전체 아키텍처 구현 시 사용
```

---

## 최종 비교표

### MVP 필수 인프라

| 서비스 | 로컬 | Railway | 비고 |
|--------|------|---------|------|
| **PostgreSQL** | Docker | Managed | 둘 다 있음 ✅ |
| **MLflow** | Docker | ❌ 필요! | **추가 필요** ⚠️ |
| **S3 Storage** | MinIO | ❌ 필요! | AWS S3/R2 사용 ⚠️ |
| **Backend** | Local | Railway | ✅ |
| **Frontend** | Local | Railway | ✅ |
| **Training Services** | Local | Railway | ✅ |

### 선택적 인프라

| 서비스 | 로컬 | Railway | 비고 |
|--------|------|---------|------|
| **Prometheus** | Docker | 생략 가능 | 프론트엔드 차트로 대체 |
| **Grafana** | Docker | 생략 가능 | 프론트엔드 차트로 대체 |

### 미사용 (제거 가능)

| 서비스 | 로컬 | Railway | 비고 |
|--------|------|---------|------|
| **MongoDB** | 중지 가능 | 불필요 | Temporal용, MVP 미사용 |
| **Redis** | 중지 가능 | 불필요 | 캐시/큐용, MVP 미사용 |
| **Temporal** | 중지 가능 | 불필요 | 워크플로우용, MVP 미사용 |
| **Mailhog** | 중지 가능 | 불필요 | 이메일 테스트, MVP 미사용 |

---

## 다음 단계

1. **즉시**: Railway에 MLflow + S3 추가
2. **선택**: Prometheus/Grafana 추가 또는 프론트엔드 차트만 사용
3. **정리**: 로컬 docker-compose를 `mvp.yml`과 `full.yml`로 분리
4. **테스트**: 모든 Training 기능이 Railway에서 정상 동작하는지 확인

---

**작성일**: 2025-01-18
**작성자**: Claude Code
**버전**: 1.0
