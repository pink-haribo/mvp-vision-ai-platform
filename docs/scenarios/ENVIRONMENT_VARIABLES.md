# 환경변수 설정 가이드

## 개요

Vision AI Training Platform은 **같은 소스코드**로 로컬과 배포 환경에서 동작합니다.

환경별 차이는 **환경변수(Environment Variables)**로 제어합니다.

---

## 핵심 원리

### 환경변수란?

운영체제 수준에서 설정하는 **키-값 쌍**입니다.

```bash
# 예시
DATABASE_URL=sqlite:///./vision_platform.db
TRAINING_EXECUTION_MODE=subprocess
```

### 코드에서 사용

```python
import os

# 환경변수 읽기
database_url = os.getenv("DATABASE_URL")
execution_mode = os.getenv("TRAINING_EXECUTION_MODE", "subprocess")  # 기본값 설정

# 환경에 따라 다르게 동작
if execution_mode == "subprocess":
    # 로컬: subprocess로 실행
    run_training_subprocess()
elif execution_mode == "api":
    # 배포: HTTP API로 실행
    run_training_api()
```

**장점:**
- ✅ 소스코드 변경 없이 환경별 설정 가능
- ✅ 민감 정보(API Key, DB 비밀번호) 숨김
- ✅ 배포 환경에서 쉽게 설정 변경

---

## 주요 환경변수

### Backend (FastAPI)

#### 1. 데이터베이스 연결

**`DATABASE_URL`** (필수)

**설명:** 데이터베이스 연결 문자열

**로컬:**
```bash
DATABASE_URL=sqlite:///./vision_platform.db
```

**배포:**
```bash
DATABASE_URL=postgresql://postgres:password@containers-us-west-xxx.railway.app:5432/railway
```

**코드:**
```python
# mvp/backend/app/db/database.py

DATABASE_URL = settings.DATABASE_URL

# SQLite vs PostgreSQL 자동 감지
is_sqlite = DATABASE_URL.startswith("sqlite")

if is_sqlite:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)
```

---

#### 2. 학습 실행 모드

**`TRAINING_EXECUTION_MODE`** (필수)

**설명:** 학습을 어떻게 실행할지 결정

**로컬:**
```bash
TRAINING_EXECUTION_MODE=subprocess
```

**배포:**
```bash
TRAINING_EXECUTION_MODE=api
```

**코드:**
```python
# mvp/backend/app/utils/training_manager.py

execution_mode = os.getenv("TRAINING_EXECUTION_MODE", "subprocess")

if execution_mode == "subprocess":
    # 로컬: Backend가 train.py를 subprocess로 직접 실행
    return self._start_training_subprocess(job_id)
elif execution_mode == "api":
    # 배포: Backend가 Training Service API에 HTTP 요청
    return self._start_training_api(job_id)
```

---

#### 3. Training Service URL (배포 환경만)

**`TIMM_SERVICE_URL`** (배포 환경 필수)
**`ULTRALYTICS_SERVICE_URL`** (배포 환경 필수)
**`HUGGINGFACE_SERVICE_URL`** (선택)

**설명:** 각 프레임워크 Training Service의 URL

**로컬:**
```bash
# 설정 안 함 (subprocess 모드라서 불필요)
```

**배포:**
```bash
TIMM_SERVICE_URL=https://timm-service-production-xxxx.up.railway.app
ULTRALYTICS_SERVICE_URL=https://ultralytics-service-production-xxxx.up.railway.app
HUGGINGFACE_SERVICE_URL=https://huggingface-service-production-xxxx.up.railway.app
```

**코드:**
```python
# mvp/backend/app/utils/training_client.py

service_url = os.getenv("TIMM_SERVICE_URL")  # framework에 따라 다름

response = requests.post(f"{service_url}/training/start", json=payload)
```

---

#### 4. 인증 설정

**`JWT_SECRET`** (필수)

**설명:** JWT 토큰 서명에 사용하는 비밀키

**로컬:**
```bash
JWT_SECRET=your-super-secret-key-for-development
```

**배포:**
```bash
JWT_SECRET=prod-super-secret-key-change-this-in-production-xxxx
```

**주의:** 배포 환경에서는 **반드시 변경**해야 합니다!

**코드:**
```python
# mvp/backend/app/core/security.py

SECRET_KEY = settings.JWT_SECRET

def create_access_token(data: dict):
    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return encoded_jwt
```

---

**`JWT_ALGORITHM`** (선택, 기본값: `HS256`)

```bash
JWT_ALGORITHM=HS256
```

---

#### 5. LLM API Keys (Gemini)

**`GOOGLE_API_KEY`** (필수)

**설명:** Google Gemini API 키

**로컬:**
```bash
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**배포:**
```bash
GOOGLE_API_KEY=AIzaSyYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
```

**코드:**
```python
# mvp/backend/app/utils/llm.py

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
```

---

### Frontend (Next.js)

#### 1. Backend API URL

**`NEXT_PUBLIC_API_URL`** (필수)

**설명:** Backend API 엔드포인트

**로컬:**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

**배포:**
```bash
NEXT_PUBLIC_API_URL=https://backend-production-xxxx.up.railway.app/api/v1
```

**코드:**
```typescript
// mvp/frontend/app/login/page.tsx

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

const response = await fetch(`${API_URL}/auth/login`, {
  method: 'POST',
  body: JSON.stringify({ email, password })
});
```

**주의:** Next.js에서 브라우저에서 접근 가능한 환경변수는 `NEXT_PUBLIC_` 접두사 필요!

---

#### 2. WebSocket URL (선택)

**`NEXT_PUBLIC_WS_URL`** (선택)

**설명:** 실시간 업데이트용 WebSocket URL

**로컬:**
```bash
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

**배포:**
```bash
NEXT_PUBLIC_WS_URL=wss://backend-production-xxxx.up.railway.app/ws
```

---

### Training Services

#### 1. 프레임워크 식별

**`FRAMEWORK`** (필수)

**설명:** 이 Training Service가 어떤 프레임워크인지 식별

**timm-service:**
```bash
FRAMEWORK=timm
```

**ultralytics-service:**
```bash
FRAMEWORK=ultralytics
```

**huggingface-service:**
```bash
FRAMEWORK=huggingface
```

**코드:**
```python
# mvp/training/api_server.py

FRAMEWORK = os.environ.get("FRAMEWORK", "unknown")

@app.get("/models/list")
async def list_models():
    if FRAMEWORK == "timm":
        models = TIMM_MODEL_REGISTRY
    elif FRAMEWORK == "ultralytics":
        models = ULTRALYTICS_MODEL_REGISTRY
    # ...
```

**설정 위치:** Dockerfile에서 `ENV` 명령어로 설정
```dockerfile
# mvp/training/Dockerfile.timm
ENV FRAMEWORK=timm
```

---

#### 2. Backend URL (학습 완료 후 DB 업데이트용)

**`BACKEND_URL`** (배포 환경 필수)

**설명:** 학습 완료 후 Backend API 호출하여 DB 업데이트

**로컬:**
```bash
BACKEND_URL=http://localhost:8000
```

**배포:**
```bash
BACKEND_URL=https://backend-production-xxxx.up.railway.app
```

**코드:**
```python
# mvp/training/train.py

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

# 학습 완료 후
requests.patch(
    f"{backend_url}/api/v1/training/jobs/{job_id}",
    json={"status": "completed", "accuracy": 0.89}
)
```

---

## 환경변수 설정 방법

### 로컬 환경

#### 1. `.env` 파일 생성

**Backend:**
```bash
# mvp/backend/.env

DATABASE_URL=sqlite:///./vision_platform.db
TRAINING_EXECUTION_MODE=subprocess
JWT_SECRET=your-super-secret-key-for-development
JWT_ALGORITHM=HS256
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**Frontend:**
```bash
# mvp/frontend/.env.local

NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

#### 2. 코드에서 자동 로드

**Backend (Python):**
```python
# mvp/backend/app/core/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    TRAINING_EXECUTION_MODE: str = "subprocess"
    JWT_SECRET: str
    GOOGLE_API_KEY: str

    class Config:
        env_file = ".env"  # .env 파일 자동 로드

settings = Settings()
```

**Frontend (Next.js):**
```typescript
// Next.js가 자동으로 .env.local 파일 로드
const apiUrl = process.env.NEXT_PUBLIC_API_URL;
```

---

### 배포 환경 (Railway)

#### 1. Railway 대시보드에서 설정

```
Railway Dashboard
  → 서비스 선택 (예: Backend)
  → Variables 탭
  → Add Variable
```

**예시: Backend 서비스**
```
DATABASE_URL = postgresql://postgres:xxx@railway.app:5432/railway (Railway 자동 제공)
TRAINING_EXECUTION_MODE = api
JWT_SECRET = prod-super-secret-key-xxxx
GOOGLE_API_KEY = AIzaSyYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
TIMM_SERVICE_URL = https://timm-service-production-xxxx.up.railway.app
ULTRALYTICS_SERVICE_URL = https://ultralytics-service-production-xxxx.up.railway.app
```

#### 2. 배포 시 자동 주입

Railway가 컨테이너 시작 시 환경변수를 자동으로 주입합니다.

```bash
# Railway 컨테이너 내부에서
echo $DATABASE_URL
# → postgresql://postgres:xxx@railway.app:5432/railway
```

---

## 환경변수 우선순위

환경변수는 다음 순서로 적용됩니다:

```
1. 시스템 환경변수 (Railway 설정)
   ↓
2. .env 파일
   ↓
3. 코드 내 기본값 (os.getenv("KEY", "default"))
```

**예시:**
```python
execution_mode = os.getenv("TRAINING_EXECUTION_MODE", "subprocess")

# Railway에 TRAINING_EXECUTION_MODE=api 설정되어 있으면 → "api"
# 설정 안 되어 있으면 → "subprocess" (기본값)
```

---

## 환경별 설정 예시

### 로컬 개발 환경

**mvp/backend/.env**
```bash
DATABASE_URL=sqlite:///./vision_platform.db
TRAINING_EXECUTION_MODE=subprocess
JWT_SECRET=dev-secret-key
JWT_ALGORITHM=HS256
GOOGLE_API_KEY=AIzaSy...
```

**mvp/frontend/.env.local**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_ENVIRONMENT=development
```

**특징:**
- SQLite 사용 (파일 기반)
- subprocess로 학습 실행
- HTTP 사용 (HTTPS 아님)
- 간단한 JWT 시크릿

---

### Railway 배포 환경

**Backend 서비스 환경변수**
```bash
# Railway 자동 제공
DATABASE_URL=postgresql://postgres:xxx@containers-us-west-xxx.railway.app:5432/railway
PORT=8000

# 수동 설정
TRAINING_EXECUTION_MODE=api
JWT_SECRET=prod-super-secret-key-change-this-in-production-xxxx
JWT_ALGORITHM=HS256
GOOGLE_API_KEY=AIzaSy...
TIMM_SERVICE_URL=https://timm-service-production-xxxx.up.railway.app
ULTRALYTICS_SERVICE_URL=https://ultralytics-service-production-xxxx.up.railway.app
HUGGINGFACE_SERVICE_URL=https://huggingface-service-production-xxxx.up.railway.app
```

**Frontend 서비스 환경변수**
```bash
NEXT_PUBLIC_API_URL=https://backend-production-xxxx.up.railway.app/api/v1
NEXT_PUBLIC_WS_URL=wss://backend-production-xxxx.up.railway.app/ws
NEXT_PUBLIC_ENVIRONMENT=production
```

**timm-service 환경변수**
```bash
FRAMEWORK=timm
BACKEND_URL=https://backend-production-xxxx.up.railway.app
PORT=8001
```

**ultralytics-service 환경변수**
```bash
FRAMEWORK=ultralytics
BACKEND_URL=https://backend-production-xxxx.up.railway.app
PORT=8001
```

**특징:**
- PostgreSQL 사용 (Railway 관리형)
- HTTP API로 Training Service 호출
- HTTPS 사용
- 강력한 JWT 시크릿

---

## 민감 정보 관리

### ⚠️ 절대 Git에 커밋하면 안 되는 것

```bash
# .gitignore에 추가 필수!
.env
.env.local
.env.production

# 민감 정보
JWT_SECRET=xxx
GOOGLE_API_KEY=xxx
DATABASE_URL=postgresql://user:password@...
```

### ✅ Git에 커밋해도 되는 것

```bash
# .env.example (템플릿, 실제 값은 제거)
DATABASE_URL=sqlite:///./vision_platform.db
TRAINING_EXECUTION_MODE=subprocess
JWT_SECRET=your-secret-key-here
GOOGLE_API_KEY=your-api-key-here
```

**사용 방법:**
```bash
# 로컬 개발 시작 시
cp .env.example .env
# .env 파일을 열어서 실제 값 입력
```

---

## 환경변수 검증

### 시작 시 검증

```python
# mvp/backend/app/main.py

@app.on_event("startup")
async def validate_env():
    """환경변수 검증"""
    required_vars = [
        "DATABASE_URL",
        "JWT_SECRET",
        "GOOGLE_API_KEY"
    ]

    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    print("[STARTUP] All required environment variables are set")
```

---

## 디버깅

### 환경변수 확인

**로컬:**
```bash
# Windows (PowerShell)
echo $env:DATABASE_URL

# Mac/Linux
echo $DATABASE_URL
```

**배포 (Railway):**
```bash
# Railway CLI
railway run env

# 또는 Railway 대시보드
# Service → Variables 탭에서 확인
```

### 코드에서 출력 (디버깅용)

```python
# mvp/backend/app/main.py

print(f"[DEBUG] DATABASE_URL: {os.getenv('DATABASE_URL')}")
print(f"[DEBUG] TRAINING_EXECUTION_MODE: {os.getenv('TRAINING_EXECUTION_MODE')}")
print(f"[DEBUG] TIMM_SERVICE_URL: {os.getenv('TIMM_SERVICE_URL')}")
```

**주의:** 민감 정보는 출력하지 마세요!

---

## 요약

### 핵심 개념

1. **같은 코드, 다른 환경변수**
   - 소스코드는 변경하지 않음
   - 환경변수로 동작 변경

2. **로컬 = .env 파일**
   - `mvp/backend/.env`
   - `mvp/frontend/.env.local`

3. **배포 = Railway 대시보드 설정**
   - Service → Variables 탭

4. **민감 정보는 Git에 절대 커밋 안 함**
   - `.gitignore`에 `.env` 추가

### 주요 환경변수

| 환경변수 | 로컬 | 배포 | 설명 |
|---------|------|------|------|
| `DATABASE_URL` | sqlite:///./db | postgresql://... | DB 연결 |
| `TRAINING_EXECUTION_MODE` | subprocess | api | 학습 실행 방식 |
| `TIMM_SERVICE_URL` | (없음) | https://... | Training Service URL |
| `NEXT_PUBLIC_API_URL` | http://localhost:8000 | https://... | Backend URL |
| `FRAMEWORK` | (없음) | timm/ultralytics | Training Service 식별 |

---

## 관련 문서

- `docs/scenarios/README.md` - 로컬/배포 차이 시나리오
- `docs/scenarios/04-model-list-flow.md` - 환경변수로 모델 조회 방식 변경
- `docs/scenarios/06-training-execution-flow.md` - 환경변수로 학습 실행 방식 변경
- `docs/production/FRAMEWORK_ISOLATION_DEPLOYMENT.md` - 배포 가이드
