# Railway User DB - 접속 정보 (Labeler 팀 공유용)

**작성일**: 2025-11-25
**Phase 11 Stage 1 완료**: Shared User DB Railway 배포

---

## 1. Railway User DB 개요

**목적**: Platform과 Labeler 서비스가 공유하는 사용자 인증/권한 데이터베이스

**배포 환경**: Railway PostgreSQL (Production)

**공유 테이블**:
- `organizations`: 조직 정보
- `users`: 사용자 정보 (인증, 권한)

**참고**: `invitations`, `project_members` 테이블은 Platform DB에 대한 FK 제약으로 인해 현재 제외됨

---

## 2. 데이터베이스 연결 정보

### 🔐 Railway PostgreSQL Connection String

```bash
# DATABASE_URL (외부 접속용)
postgresql://postgres:hNBDsIoezlnZSoGNKmGsxYcLiZekJiSj@gondola.proxy.rlwy.net:10185/railway

# 또는 내부 URL (Railway 내 서비스용)
postgresql://postgres:hNBDsIoezlnZSoGNKmGsxYcLiZekJiSj@postgres.railway.internal:5432/railway
```

### 연결 정보 분리

| 항목 | 값 |
|------|-----|
| **Host** | `gondola.proxy.rlwy.net` (외부) / `postgres.railway.internal` (내부) |
| **Port** | `10185` (외부) / `5432` (내부) |
| **Database** | `railway` |
| **User** | `postgres` |
| **Password** | `hNBDsIoezlnZSoGNKmGsxYcLiZekJiSj` |

---

## 3. 현재 데이터 현황

**마이그레이션 완료일**: 2025-11-25

**데이터 현황**:
- ✅ organizations: 2 rows
- ✅ users: 5 rows

**테스트 계정**:
- Email: `admin@example.com`
- Password: `admin123`
- Role: `admin`

---

## 4. Labeler Backend 연동 가이드

### 4.1 환경변수 설정

Labeler Backend `.env` 파일에 추가:

```bash
# User Database (Shared with Platform)
USER_DATABASE_URL=postgresql://postgres:hNBDsIoezlnZSoGNKmGsxYcLiZekJiSj@gondola.proxy.rlwy.net:10185/railway
```

### 4.2 SQLAlchemy 연결 설정

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# User DB Engine (Shared)
user_engine = create_engine(
    os.getenv('USER_DATABASE_URL'),
    echo=False
)

UserSessionLocal = sessionmaker(bind=user_engine)

def get_user_db():
    """Dependency for User DB sessions."""
    db = UserSessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 4.3 User 모델 정의

```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import ENUM as SQLEnum
from enum import Enum

class UserRole(str, Enum):
    """User role enum (lowercase values for PostgreSQL)"""
    GUEST = "guest"
    BASIC_ENGINEER = "basic_engineer"
    STANDARD_ENGINEER = "standard_engineer"
    ADVANCED_ENGINEER = "advanced_engineer"
    MANAGER = "manager"
    ADMIN = "admin"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    system_role = Column(SQLEnum(UserRole, values_callable=lambda x: [e.value for e in x]), nullable=False, default=UserRole.GUEST)
    is_active = Column(Boolean, nullable=False, default=True)
    # ... 기타 필드
```

**중요**: `SQLEnum`에 `values_callable` 파라미터를 반드시 추가해야 합니다. (lowercase enum 값 처리)

### 4.4 인증 엔드포인트 구현

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/auth/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_db: Session = Depends(get_user_db)
):
    """Login with shared User DB"""
    # Query from Railway User DB
    user = user_db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect credentials")

    # Generate JWT token
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me")
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    user_db: Session = Depends(get_user_db)
):
    """Get current user from shared User DB"""
    # Decode JWT and fetch user from Railway DB
    payload = decode_access_token(token)
    user_id = int(payload["sub"])

    user = user_db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user
```

---

## 5. 테스트 방법

### 5.1 로그인 테스트

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@example.com&password=admin123"
```

**Expected Response**:
```json
{
  "access_token": "eyJhbGc...",
  "refresh_token": "eyJhbGc...",
  "token_type": "bearer"
}
```

### 5.2 사용자 정보 조회

```bash
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer <access_token>"
```

**Expected Response**:
```json
{
  "id": 1,
  "email": "admin@example.com",
  "full_name": "Admin User",
  "system_role": "admin",
  "is_active": true
}
```

---

## 6. 보안 주의사항

### 6.1 연결 문자열 관리

**절대 금지**:
- ❌ 연결 문자열을 코드에 하드코딩
- ❌ Git에 연결 문자열 커밋
- ❌ 공개 채널에 연결 정보 공유

**권장 방법**:
- ✅ 환경변수 (.env 파일) 사용
- ✅ Secret 관리 도구 사용 (Railway Variables, K8s Secrets)
- ✅ 비공개 채널로 공유 (Slack DM, 암호화된 이메일)

### 6.2 JWT Secret 통일

Platform과 Labeler가 **동일한 JWT_SECRET**을 사용해야 토큰을 상호 검증할 수 있습니다.

```bash
# .env (Platform & Labeler 동일)
JWT_SECRET=your-super-secret-key-change-this-in-production-tier0
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

## 7. Railway Dashboard 접속

**프로젝트 URL**: https://railway.com/project/9d57f05c-cbcc-4769-bc8d-7104636f76c1

**서비스**: user-db (PostgreSQL)

**모니터링**:
- Metrics: CPU, Memory, Network
- Logs: Real-time query logs
- Variables: 환경변수 관리

**참고**: Labeler 팀은 Railway 프로젝트에 대한 직접 접근 권한이 없으므로, 변경이 필요하면 Platform 팀에 요청해주세요.

---

## 8. 문제 해결 (Troubleshooting)

### 8.1 연결 실패 (Connection Timeout)

**원인**: Railway 외부 접속 URL 사용 문제

**해결**:
1. Railway 대시보드에서 최신 연결 URL 확인
2. 방화벽/네트워크 설정 확인
3. Railway 서비스 상태 확인

### 8.2 Enum 값 에러 (`invalid input value for enum userrole: "admin"`)

**원인**: PostgreSQL enum 타입이 uppercase로 생성됨

**해결**: SQLAlchemy 모델에 `values_callable` 추가
```python
system_role = Column(
    SQLEnum(UserRole, values_callable=lambda x: [e.value for e in x]),
    nullable=False
)
```

### 8.3 인증 실패 (Invalid Token)

**원인**: JWT_SECRET이 Platform과 Labeler 간 불일치

**해결**: 동일한 JWT_SECRET 사용 확인

---

## 9. 다음 단계 (Stage 2: Cloudflare R2)

**예정 일정**: Week 1, Day 4-5

**작업 내용**:
- Cloudflare R2 버킷 생성 (Labeler 팀)
- External Storage (MinIO → R2) 마이그레이션
- Platform/Labeler 연동 테스트

**Labeler 팀 준비사항**:
1. Cloudflare R2 계정 생성
2. R2 버킷 생성 (`labeler-datasets`)
3. API 토큰 생성 (Read/Write 권한)
4. CORS 설정 (Platform 도메인 허용)

---

## 10. 연락처

**질문/문제 발생 시**:
- Platform 팀: [연락처]
- Railway Support: https://railway.app/help

**문서 업데이트**: 2025-11-25
