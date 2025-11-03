# 시나리오 1: 유저 로그인

## 개요

사용자가 이메일과 비밀번호로 로그인하는 과정입니다.

**목표:** 사용자 인증 후 JWT 토큰 발급

---

## 로컬 환경 (개발)

### 환경 구성
```
Frontend: http://localhost:3000 (Next.js Dev Server)
Backend:  http://localhost:8000 (FastAPI Uvicorn)
Database: SQLite (mvp/backend/vision_platform.db)
```

### 상세 흐름

#### 1단계: 사용자가 로그인 폼 입력

**위치:** 브라우저 (http://localhost:3000/login)

```
사용자 입력:
- Email: admin@example.com
- Password: admin123
```

**Frontend 코드:**
```typescript
// mvp/frontend/app/login/page.tsx
const handleSubmit = async (e: FormEvent) => {
  e.preventDefault();

  // 로컬 Backend API 호출
  const response = await fetch('http://localhost:8000/api/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });

  const data = await response.json();
  // ...
};
```

**동작:**
- Next.js가 `localhost:3000`에서 실행 중
- 사용자가 입력 → "로그인" 버튼 클릭
- JavaScript `fetch()` 실행

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
POST http://localhost:8000/api/v1/auth/login
Content-Type: application/json

{
  "email": "admin@example.com",
  "password": "admin123"
}
```

**네트워크:**
- 같은 컴퓨터 내에서 `localhost:3000` → `localhost:8000`
- CORS 허용 필요 (Backend가 `localhost:3000` origin 허용)

---

#### 3단계: Backend API 엔드포인트 실행

**위치:** `mvp/backend/app/api/auth.py`

```python
@router.post("/login")
def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    # 1. 데이터베이스에서 사용자 조회
    user = db.query(User).filter(User.email == credentials.email).first()

    # 2. 비밀번호 검증
    if not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # 3. JWT 토큰 생성
    access_token = create_access_token(data={"sub": user.email})

    # 4. 응답 반환
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name
        }
    }
```

**동작:**
1. FastAPI 라우터가 요청 수신
2. SQLAlchemy로 SQLite DB 쿼리
3. bcrypt로 비밀번호 검증
4. python-jose로 JWT 생성

---

#### 4단계: Database 조회

**위치:** `mvp/backend/vision_platform.db` (SQLite 파일)

```sql
-- SQLAlchemy가 실행하는 쿼리
SELECT id, email, hashed_password, full_name
FROM users
WHERE email = 'admin@example.com'
LIMIT 1;
```

**결과:**
```
id: 1
email: admin@example.com
hashed_password: $2b$12$...
full_name: Admin User
```

**특징:**
- 로컬에서는 **SQLite 파일** 사용
- 파일 경로: `C:\Users\flyto\Project\Github\mvp-vision-ai-platform\mvp\backend\vision_platform.db`
- 동시 접속 제한 있음 (단일 사용자 개발용)

---

#### 5단계: Backend → Frontend 응답

**응답:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "admin@example.com",
    "full_name": "Admin User"
  }
}
```

---

#### 6단계: Frontend 토큰 저장

**위치:** 브라우저 LocalStorage

```typescript
// mvp/frontend/app/login/page.tsx
if (response.ok) {
  const data = await response.json();

  // LocalStorage에 저장
  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('user', JSON.stringify(data.user));

  // 대시보드로 이동
  router.push('/dashboard');
}
```

**브라우저 상태:**
```
LocalStorage:
  access_token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  user: '{"id":1,"email":"admin@example.com","full_name":"Admin User"}'
```

---

## 배포 환경 (Railway)

### 환경 구성
```
Frontend: https://frontend-production-xxxx.up.railway.app (Vercel/Railway)
Backend:  https://backend-production-xxxx.up.railway.app (Railway)
Database: PostgreSQL (Railway 관리형 DB)
```

### 상세 흐름

#### 1단계: 사용자가 로그인 폼 입력

**위치:** 브라우저 (https://frontend-production-xxxx.up.railway.app/login)

```
사용자 입력:
- Email: admin@example.com
- Password: admin123
```

**Frontend 코드:**
```typescript
// mvp/frontend/app/login/page.tsx
const handleSubmit = async (e: FormEvent) => {
  e.preventDefault();

  // 배포 Backend API 호출 (환경변수)
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });

  const data = await response.json();
  // ...
};
```

**환경변수:**
```bash
# Railway 설정
NEXT_PUBLIC_API_URL=https://backend-production-xxxx.up.railway.app/api/v1
```

**동작:**
- Next.js가 Railway/Vercel에서 실행 중
- 프로덕션 빌드된 JavaScript 실행
- `process.env.NEXT_PUBLIC_API_URL` 사용

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
POST https://backend-production-xxxx.up.railway.app/api/v1/auth/login
Content-Type: application/json

{
  "email": "admin@example.com",
  "password": "admin123"
}
```

**네트워크:**
- 인터넷을 통한 요청 (HTTPS)
- Frontend 서버 → Backend 서버 (다른 Railway 컨테이너)
- Railway 내부 네트워크 or 공개 URL 사용

---

#### 3단계: Backend API 엔드포인트 실행

**위치:** Railway 컨테이너 (`mvp/backend/app/api/auth.py`)

```python
@router.post("/login")
def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    # 로컬과 동일한 코드
    user = db.query(User).filter(User.email == credentials.email).first()
    # ...
```

**차이점:**
- Docker 컨테이너에서 실행 (Python 3.11-slim 이미지)
- 환경변수로 설정 주입
- 로그는 Railway 대시보드에서 확인

---

#### 4단계: Database 조회

**위치:** Railway PostgreSQL (관리형 서비스)

```sql
-- SQLAlchemy가 실행하는 쿼리 (동일)
SELECT id, email, hashed_password, full_name
FROM users
WHERE email = 'admin@example.com'
LIMIT 1;
```

**연결 정보:**
```python
# mvp/backend/app/core/config.py
DATABASE_URL = os.getenv("DATABASE_URL")
# Railway 설정: postgresql://postgres:xxx@containers-us-west-xxx.railway.app:5432/railway
```

**특징:**
- 로컬: SQLite (파일)
- 배포: **PostgreSQL** (서버)
- Railway가 자동으로 DATABASE_URL 제공
- 동시 접속 제한 없음
- 백업, 복제 자동 관리

---

#### 5단계: Backend → Frontend 응답

**응답:** (로컬과 동일)
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": { ... }
}
```

**차이점:**
- HTTPS 암호화 (TLS/SSL)
- Railway가 자동으로 SSL 인증서 제공

---

#### 6단계: Frontend 토큰 저장

**위치:** 브라우저 LocalStorage (로컬과 동일)

```typescript
localStorage.setItem('access_token', data.access_token);
localStorage.setItem('user', JSON.stringify(data.user));
router.push('/dashboard');
```

**브라우저 상태:** (로컬과 동일)

---

## 주요 차이점 요약

| 구분 | 로컬 환경 | 배포 환경 (Railway) |
|------|----------|-------------------|
| **Frontend URL** | http://localhost:3000 | https://frontend-production-xxxx.up.railway.app |
| **Backend URL** | http://localhost:8000 | https://backend-production-xxxx.up.railway.app |
| **프로토콜** | HTTP (암호화 없음) | HTTPS (TLS/SSL 암호화) |
| **데이터베이스** | SQLite (파일) | PostgreSQL (관리형 서버) |
| **DB 연결** | 로컬 파일 읽기 | TCP/IP 네트워크 연결 |
| **환경변수** | `.env` 파일 | Railway 대시보드 설정 |
| **로그 확인** | 터미널 출력 | Railway 대시보드 |
| **성능** | 빠름 (로컬) | 네트워크 지연 있음 |
| **보안** | 개발용 (낮음) | 프로덕션용 (높음) |
| **동시 접속** | 1명 (개발자) | 다수 (실제 사용자) |

---

## 코드 차이 (환경별 동작)

### Frontend: 환경변수로 API URL 결정

```typescript
// mvp/frontend/app/login/page.tsx

// 로컬: http://localhost:8000/api/v1
// 배포: https://backend-production-xxxx.up.railway.app/api/v1
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
```

### Backend: 데이터베이스 자동 감지

```python
# mvp/backend/app/db/database.py

# DATABASE_URL 형식에 따라 자동 판단
is_sqlite = settings.DATABASE_URL.startswith("sqlite")

if is_sqlite:
    # 로컬: SQLite 설정
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    # 배포: PostgreSQL 설정
    engine = create_engine(
        settings.DATABASE_URL,
        pool_size=5,
        max_overflow=10
    )
```

---

## 관련 파일

### Frontend
- `mvp/frontend/app/login/page.tsx` - 로그인 UI
- `mvp/frontend/.env.local` - 로컬 환경변수
- Railway 환경변수: `NEXT_PUBLIC_API_URL`

### Backend
- `mvp/backend/app/api/auth.py` - 로그인 API
- `mvp/backend/app/core/security.py` - 비밀번호 검증, JWT 생성
- `mvp/backend/app/db/models.py` - User 모델
- `mvp/backend/app/db/database.py` - DB 연결 설정
- `mvp/backend/.env` - 로컬 환경변수
- Railway 환경변수: `DATABASE_URL`, `JWT_SECRET`

### Database
- 로컬: `mvp/backend/vision_platform.db`
- 배포: Railway PostgreSQL (자동 프로비저닝)

---

## 디버깅 팁

### 로컬 환경

**Backend 로그 보기:**
```bash
cd mvp/backend
../../mvp/backend/venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

**Database 확인:**
```bash
sqlite3 mvp/backend/vision_platform.db
> SELECT * FROM users;
```

**Frontend 로그:**
- 브라우저 개발자 도구 (F12) → Console

---

### 배포 환경

**Backend 로그 보기:**
- Railway 대시보드 → Backend 서비스 → Deployments → Logs

**Database 확인:**
```bash
# Railway CLI
railway run psql $DATABASE_URL
> SELECT * FROM users;
```

**Frontend 로그:**
- Railway/Vercel 대시보드 → Logs
- 브라우저 개발자 도구 (동일)

---

## 문제 해결

### 로컬: "CORS error"
**원인:** Backend가 Frontend origin 허용하지 않음

**해결:**
```python
# mvp/backend/app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 확인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### 배포: "Invalid credentials"
**원인:** DB에 사용자 없음 (첫 배포 시)

**해결:**
- Backend startup 이벤트가 admin 계정 자동 생성
- Railway 로그 확인: `Created default admin user: admin@example.com / admin123`
