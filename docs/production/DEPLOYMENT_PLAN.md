# Production Deployment Plan

**목적**: 현재 MVP 구조가 프로덕션 환경에서 정상 동작하는지 검증
**범위**: Frontend + Backend + Database 최소 구성
**이후**: 검증 완료 후 로컬 개발 환경으로 복귀

---

## 배포 전략 비교

### 옵션 1: Railway (All-in-One) ⭐ **추천**

**장점:**
- ✅ 모든 서비스를 한 곳에서 관리
- ✅ Docker container 지원 (FastAPI 최적)
- ✅ PostgreSQL, Redis 자동 프로비저닝
- ✅ 환경 변수 관리 간편
- ✅ GitHub 연동 자동 배포
- ✅ 무료 tier로 테스트 가능 ($5 credit/month)
- ✅ 간단한 설정

**단점:**
- ⚠️ 무료 tier 제한적 (sleep mode 있음)

**비용 (예상):**
- Starter Plan: $5/month (개발/테스트용)
- 또는 무료 tier로 시작

---

### 옵션 2: 분산 배포 (Vercel + Render + Supabase)

**구성:**
- Frontend: Vercel (Next.js)
- Backend: Render (Docker)
- Database: Supabase (PostgreSQL)
- Redis: Upstash

**장점:**
- ✅ 각 서비스별 최적화
- ✅ Vercel의 강력한 CDN (frontend)
- ✅ Supabase의 풍부한 DB 기능
- ✅ 확장성 좋음

**단점:**
- ⚠️ 설정 복잡 (4개 서비스 관리)
- ⚠️ CORS 설정 필요
- ⚠️ 서비스 간 네트워크 latency

**비용 (예상):**
- Vercel: 무료 (Hobby)
- Render: $7/month (Web Service)
- Supabase: 무료 tier
- Upstash: 무료 tier
- **Total**: ~$7/month

---

## 추천 방안: Railway (옵션 1)

### 아키텍처

```
┌─────────────────────────────────────────────────┐
│                   Railway                       │
│                                                 │
│  ┌──────────────┐  ┌──────────────┐            │
│  │   Frontend   │  │   Backend    │            │
│  │   (Next.js)  │◄─┤   (FastAPI)  │            │
│  └──────────────┘  └───────┬──────┘            │
│                            │                    │
│                    ┌───────▼──────┐             │
│                    │  PostgreSQL  │             │
│                    └──────────────┘             │
│                                                 │
│                    ┌──────────────┐             │
│                    │    Redis     │             │
│                    └──────────────┘             │
└─────────────────────────────────────────────────┘
```

### 배포 구성

#### 1. Frontend (Next.js)
- **플랫폼**: Railway
- **빌드 방식**: Nixpacks 자동 감지
- **도메인**: `your-app.up.railway.app` (자동 제공)
- **환경 변수**:
  ```
  NEXT_PUBLIC_API_URL=https://backend-service.up.railway.app/api/v1
  NEXT_PUBLIC_WS_URL=wss://backend-service.up.railway.app/ws
  ```

#### 2. Backend (FastAPI)
- **플랫폼**: Railway
- **빌드 방식**: Dockerfile
- **포트**: 8000
- **환경 변수**:
  ```
  # Database (Railway 자동 제공)
  DATABASE_URL=${DATABASE_URL}
  REDIS_URL=${REDIS_URL}

  # LLM
  GOOGLE_API_KEY=your-key-here

  # Auth
  JWT_SECRET=your-secret-key
  JWT_ALGORITHM=HS256
  ACCESS_TOKEN_EXPIRE_MINUTES=60

  # CORS
  CORS_ORIGINS=https://your-frontend.up.railway.app
  ```

#### 3. Database (PostgreSQL)
- **플랫폼**: Railway Postgres Plugin
- **버전**: PostgreSQL 16
- **프로비저닝**: 자동 (플러그인 추가)
- **연결 문자열**: 자동으로 `DATABASE_URL` 환경 변수 주입

#### 4. Redis
- **플랫폼**: Railway Redis Plugin
- **프로비저닝**: 자동 (플러그인 추가)
- **연결 문자열**: 자동으로 `REDIS_URL` 환경 변수 주입

---

## 배포 준비 작업

### 1. Dockerfile 준비 (Backend)

`mvp/backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run migrations and start server
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### 2. Railway 설정 파일

`mvp/backend/railway.toml`:
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

`mvp/frontend/railway.toml`:
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "npm run start"
```

### 3. 환경 변수 체크리스트

**Backend 필수 환경 변수:**
- [x] `DATABASE_URL` - Railway 자동 주입
- [x] `REDIS_URL` - Railway 자동 주입
- [ ] `GOOGLE_API_KEY` - 수동 설정 필요 ⚠️
- [ ] `JWT_SECRET` - 수동 설정 필요 ⚠️
- [ ] `CORS_ORIGINS` - Frontend URL로 설정

**Frontend 필수 환경 변수:**
- [ ] `NEXT_PUBLIC_API_URL` - Backend URL로 설정
- [ ] `NEXT_PUBLIC_WS_URL` - Backend WebSocket URL로 설정

### 4. 코드 수정사항

#### A. CORS 설정 수정 (`mvp/backend/app/main.py`)

```python
# 환경 변수에서 CORS origins 읽기
import os

allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### B. 데이터셋 경로 수정 (`mvp/backend/app/utils/tool_registry.py`)

프로덕션에서는 `C:\datasets` 경로가 없으므로:

```python
# Linux 경로로 변경 또는 환경 변수 사용
builtin_path = Path(os.getenv("DATASETS_PATH", "/app/datasets"))
```

**옵션:**
1. 배포 시 `/app/datasets` 경로에 샘플 데이터셋 포함
2. 또는 데이터셋 없이 배포 (사용자 업로드만 지원)

#### C. Static 파일 경로

`outputs/`, `uploads/` 등의 경로를 환경 변수로 관리:

```python
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
```

---

## 배포 단계

### Phase 1: Railway 프로젝트 생성

1. **Railway 계정 생성**
   - https://railway.app 접속
   - GitHub 계정으로 로그인

2. **새 프로젝트 생성**
   - "New Project" 클릭
   - "Deploy from GitHub repo" 선택
   - `mvp-vision-ai-platform` 저장소 선택

3. **서비스 추가**
   ```
   ① Frontend (mvp/frontend)
   ② Backend (mvp/backend)
   ③ PostgreSQL Plugin
   ④ Redis Plugin
   ```

### Phase 2: 환경 변수 설정

**Backend Service:**
```bash
GOOGLE_API_KEY=<your-gemini-api-key>
JWT_SECRET=<generate-random-secret>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
CORS_ORIGINS=https://<frontend-url>.up.railway.app
DATASETS_PATH=/app/datasets
```

**Frontend Service:**
```bash
NEXT_PUBLIC_API_URL=https://<backend-url>.up.railway.app/api/v1
NEXT_PUBLIC_WS_URL=wss://<backend-url>.up.railway.app/ws
```

### Phase 3: 빌드 및 배포

1. **Backend 먼저 배포**
   - Railway가 자동으로 Dockerfile 감지
   - Build logs 확인
   - Health check 통과 확인

2. **Frontend 배포**
   - 환경 변수에서 Backend URL 확인
   - Build 완료 대기
   - 도메인 접속 확인

### Phase 4: 데이터베이스 초기화

Railway CLI 또는 웹 콘솔에서:

```bash
# Railway CLI 설치
npm install -g @railway/cli

# 프로젝트 연결
railway login
railway link

# Backend 서비스 선택 후 마이그레이션 실행
railway run alembic upgrade head

# 또는 Railway 웹 콘솔의 "Run Command" 기능 사용
```

### Phase 5: 초기 사용자 생성

```bash
# Railway 콘솔에서 Python 실행
railway run python

# 아래 스크립트 실행
from app.db.database import SessionLocal
from app.db.models import User
from app.core.security import get_password_hash

db = SessionLocal()
user = User(
    email="admin@example.com",
    username="admin",
    hashed_password=get_password_hash("admin123"),
    is_active=True,
    is_superuser=True
)
db.add(user)
db.commit()
```

---

## 배포 후 검증 체크리스트

### Backend 확인
- [ ] Health check: `GET https://backend-url/health`
- [ ] API docs: `GET https://backend-url/docs`
- [ ] Database 연결: Migration 실행 확인
- [ ] Redis 연결: 로그 확인

### Frontend 확인
- [ ] 페이지 로딩: `https://frontend-url`
- [ ] API 연결: Network 탭에서 확인
- [ ] 로그인: 사용자 생성 후 로그인 테스트

### 기능 검증
- [ ] 회원가입 / 로그인
- [ ] 채팅 메시지 전송
- [ ] 기본 데이터셋 조회 (없으면 skip)
- [ ] 프로젝트 생성
- [ ] 학습 설정 입력
- [ ] 학습 시작 (실제 실행은 skip 가능)

---

## Docker Container 관리

### Local Development
개발 환경에서는 Docker Compose 계속 사용:

```bash
# Infrastructure만 Docker로 실행
docker-compose up -d postgres redis

# Backend/Frontend는 로컬에서 직접 실행
cd mvp/backend && uvicorn app.main:app --reload
cd mvp/frontend && npm run dev
```

### Production (Railway)
Railway가 자동으로 Docker container 관리:
- Dockerfile 기반 자동 빌드
- Health check 기반 재시작
- Zero-downtime 배포

---

## 비용 관리

### Railway 무료 Tier
- $5 credit/month 제공
- 예상 사용량:
  - Frontend: ~$2-3/month
  - Backend: ~$3-5/month
  - PostgreSQL: ~$2/month
  - Redis: ~$1/month
  - **Total**: ~$8-11/month

### 비용 절감 팁
1. **Sleep Mode 활성화**: 1시간 비활성 시 자동 sleep
2. **Hobby Plan**: 테스트 후 불필요한 서비스 삭제
3. **데이터셋**: 프로덕션에 대용량 데이터 올리지 않기

---

## 롤백 계획

문제 발생 시:

1. **Railway에서 이전 배포로 롤백**
   - Deployments 탭에서 이전 버전 선택
   - "Rollback" 클릭

2. **환경 변수 복구**
   - 변경 전 값 백업
   - Variables 탭에서 수정

3. **데이터베이스 복구**
   - Railway PostgreSQL 자동 백업 사용
   - 또는 migration downgrade

---

## 대안: Vercel + Render + Supabase (옵션 2)

간단 요약만 제공 (상세 문서는 별도 작성 가능):

### Frontend (Vercel)
```bash
vercel login
cd mvp/frontend
vercel --prod
```

### Backend (Render)
- Dashboard에서 "New Web Service"
- GitHub 연결
- Docker 배포
- 환경 변수 설정

### Database (Supabase)
- 새 프로젝트 생성
- Connection string 복사
- Backend에 `DATABASE_URL` 설정

### Redis (Upstash)
- 새 database 생성
- Connection string 복사
- Backend에 `REDIS_URL` 설정

---

## 다음 단계

1. **이 문서 리뷰** ← 현재 단계
2. **Railway 계정 생성 및 테스트**
3. **코드 수정 (CORS, 경로 등)**
4. **배포 실행**
5. **검증 및 문제 해결**
6. **로컬 개발 환경으로 복귀**

---

## 질문 사항

- [ ] 데이터셋을 프로덕션에 포함할 것인가?
  - Yes → 샘플 데이터 준비
  - No → 사용자 업로드만 지원

- [ ] 도메인 설정 필요한가?
  - Yes → Railway에서 커스텀 도메인 설정
  - No → Railway 자동 도메인 사용

- [ ] 학습 실행을 프로덕션에서 테스트할 것인가?
  - Yes → GPU 인스턴스 고려 (비용 증가)
  - No → 학습 시작만 확인

---

**작성일**: 2025-11-02
**작성자**: Claude Code
**버전**: 1.0
