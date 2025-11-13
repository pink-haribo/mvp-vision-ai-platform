# 배포 전 체크리스트

Railway 배포 전에 필수로 수정해야 하는 코드 변경사항입니다.

---

## 1. CORS 설정 수정 ✅ 필수

**파일**: `mvp/backend/app/main.py`

**현재 상태 확인 필요**:
```python
# 하드코딩된 CORS origins가 있는지 확인
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ← 이렇게 되어 있으면 수정 필요
    ...
)
```

**수정 필요**:
```python
import os

# 환경 변수에서 CORS origins 읽기
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
cors_origins = [origin.strip() for origin in cors_origins]  # 공백 제거

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 2. 데이터셋 경로 수정 ✅ 필수

**파일**: `mvp/backend/app/utils/tool_registry.py`

**현재 코드** (line ~311):
```python
# Scan C:\datasets for built-in datasets
builtin_path = Path("C:/datasets")
```

**수정 필요**:
```python
import os

# 환경 변수에서 경로 읽기 (Windows/Linux 호환)
builtin_path = Path(os.getenv("DATASETS_PATH", "C:/datasets"))
```

**이유**:
- 프로덕션 Linux 환경에서 `C:/datasets` 경로 없음
- Railway에서 `/app/datasets` 사용

---

## 3. Health Check 엔드포인트 확인 ✅ 확인 필요

**파일**: `mvp/backend/app/main.py` 또는 `app/api/health.py`

**필요한 엔드포인트**:
```python
@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "ok"}
```

**확인 사항**:
- [ ] `/health` 엔드포인트가 존재하는가?
- [ ] Database 연결 확인 필요한가? (선택)
- [ ] Redis 연결 확인 필요한가? (선택)

**고급 Health Check (선택)**:
```python
from app.db.database import engine
from sqlalchemy import text

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health = {
        "status": "ok",
        "database": "unknown",
        "redis": "unknown"
    }

    # Database check
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health["database"] = "ok"
    except Exception as e:
        health["database"] = "error"
        health["status"] = "degraded"

    # Redis check (optional)
    # try:
    #     redis_client.ping()
    #     health["redis"] = "ok"
    # except:
    #     health["redis"] = "error"

    return health
```

---

## 4. Static Files 경로 확인 ⚠️ 권장

**파일들**:
- `mvp/backend/app/api/training.py`
- 기타 파일 업로드/다운로드 관련 코드

**확인 사항**:
```python
# 하드코딩된 경로가 있는지 확인
output_dir = "./outputs"  # ← 환경 변수로 변경 권장
upload_dir = "./uploads"  # ← 환경 변수로 변경 권장
```

**권장 수정**:
```python
import os

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
```

---

## 5. Database URL 설정 확인 ✅ 자동 처리

**파일**: `mvp/backend/app/core/config.py`

**확인 사항**:
```python
class Settings(BaseSettings):
    DATABASE_URL: str
    # Railway가 자동으로 주입하므로 default 값 불필요
```

Railway PostgreSQL plugin이 자동으로 `DATABASE_URL` 주입 → **수정 불필요**

---

## 6. Requirements.txt 확인 ✅ 필수

**파일**: `mvp/backend/requirements.txt`

**확인 사항**:
- [ ] 모든 의존성이 포함되어 있는가?
- [ ] 버전이 명시되어 있는가?
- [ ] `google-generativeai` 포함 확인

**테스트 방법**:
```bash
cd mvp/backend
python -m venv test_venv
source test_venv/bin/activate  # Windows: test_venv\Scripts\activate
pip install -r requirements.txt
# 에러 없이 설치되는지 확인
```

---

## 7. Alembic Migration 확인 ✅ 필수

**파일**: `mvp/backend/alembic.ini`, `mvp/backend/alembic/env.py`

**확인 사항**:
- [ ] `alembic.ini`에서 `sqlalchemy.url` 설정 확인
- [ ] 환경 변수 `DATABASE_URL` 사용하는지 확인

**alembic/env.py에서**:
```python
from app.core.config import settings

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
```

---

## 8. 프론트엔드 API URL 설정 확인 ✅ 필수

**파일**: `mvp/frontend/next.config.js` 또는 코드에서 직접 사용

**확인 사항**:
```typescript
// 환경 변수 사용 확인
const apiUrl = process.env.NEXT_PUBLIC_API_URL
```

Next.js는 `NEXT_PUBLIC_` prefix가 있는 환경 변수만 클라이언트에 노출 → **이미 올바름**

---

## 9. 로깅 설정 ⚠️ 권장

**파일**: `mvp/backend/app/main.py`

**프로덕션 로깅 설정**:
```python
import logging
import os

# 환경에 따라 로그 레벨 설정
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

---

## 10. 환경 확인 엔드포인트 (디버깅용) ⚠️ 선택

**개발/디버깅 시에만 추가 권장**:

```python
@app.get("/debug/env")
async def debug_env():
    """Show environment info (REMOVE IN PRODUCTION!)"""
    import os
    return {
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "has_database": bool(os.getenv("DATABASE_URL")),
        "has_redis": bool(os.getenv("REDIS_URL")),
        "has_google_api_key": bool(os.getenv("GOOGLE_API_KEY")),
        "cors_origins": os.getenv("CORS_ORIGINS", "not set"),
    }
```

**주의**: 배포 후 확인용으로만 사용, 민감한 정보 노출 주의!

---

## 체크리스트 요약

### 필수 수정 (배포 전)
- [ ] 1. CORS 설정을 환경 변수로 변경
- [ ] 2. 데이터셋 경로를 환경 변수로 변경
- [ ] 3. Health check 엔드포인트 확인
- [ ] 6. Requirements.txt 확인
- [ ] 7. Alembic migration 설정 확인

### 권장 수정
- [ ] 4. Static files 경로를 환경 변수로 변경
- [ ] 9. 프로덕션 로깅 설정

### 확인 사항
- [ ] 5. Database URL 자동 주입 (Railway) - 코드 확인
- [ ] 8. Frontend API URL 환경 변수 사용 - 코드 확인

### 디버깅 (선택)
- [ ] 10. 환경 확인 엔드포인트 추가 (배포 후 제거)

---

## 다음 단계

1. **코드 수정 완료**
2. **로컬에서 테스트**
   ```bash
   # .env 파일에 프로덕션 설정 시뮬레이션
   CORS_ORIGINS=http://localhost:3000
   DATASETS_PATH=/tmp/datasets

   # Backend 실행
   uvicorn app.main:app --reload

   # Health check 확인
   curl http://localhost:8000/health
   ```
3. **커밋 & Push**
4. **Railway 배포 시작**

---

**작성일**: 2025-11-02
**중요도**: ⭐⭐⭐⭐⭐ (배포 전 필수)
