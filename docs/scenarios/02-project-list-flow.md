# 시나리오 2: 프로젝트 조회

## 개요

로그인한 사용자가 자신의 프로젝트 목록을 조회하는 과정입니다.

**목표:** 사용자가 생성한 모든 프로젝트 표시 (이름, 설명, 생성일, 실험 수)

---

## 로컬 환경 (개발)

### 환경 구성
```
Frontend: http://localhost:3000/dashboard
Backend:  http://localhost:8000
Database: SQLite (vision_platform.db)
```

### 상세 흐름

#### 1단계: 대시보드 페이지 로드

**위치:** 브라우저 (http://localhost:3000/dashboard)

**Frontend 코드:**
```typescript
// mvp/frontend/app/dashboard/page.tsx
'use client';

import { useEffect, useState } from 'react';

export default function DashboardPage() {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    // LocalStorage에서 토큰 가져오기
    const token = localStorage.getItem('access_token');

    // Backend API 호출
    const response = await fetch('http://localhost:8000/api/v1/projects', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    const data = await response.json();
    setProjects(data);
    setLoading(false);
  };

  return (
    <div>
      <h1>내 프로젝트</h1>
      {projects.map(project => (
        <ProjectCard key={project.id} project={project} />
      ))}
    </div>
  );
}
```

**동작:**
1. React 컴포넌트 마운트
2. `useEffect` 훅 실행 → `fetchProjects()` 호출
3. LocalStorage에서 `access_token` 읽기
4. HTTP 요청 준비

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
GET http://localhost:8000/api/v1/projects
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**헤더:**
- `Authorization`: JWT 토큰 포함
- Backend가 토큰 검증 → 사용자 식별

**네트워크:**
- `localhost:3000` (Next.js) → `localhost:8000` (FastAPI)
- 같은 컴퓨터 내부 통신

---

#### 3단계: Backend 인증 미들웨어

**위치:** `mvp/backend/app/api/dependencies.py`

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """JWT 토큰에서 현재 사용자 추출"""

    token = credentials.credentials

    try:
        # 1. JWT 토큰 디코딩
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )

        # 2. 토큰에서 이메일 추출
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        # 3. 데이터베이스에서 사용자 조회
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")

        return user

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**동작:**
1. `Authorization` 헤더에서 토큰 추출
2. JWT 서명 검증 (`JWT_SECRET` 사용)
3. 토큰에서 이메일 추출
4. DB에서 User 객체 조회
5. User 객체를 API 엔드포인트에 전달

---

#### 4단계: Backend API 엔드포인트 실행

**위치:** `mvp/backend/app/api/projects.py`

```python
@router.get("/", response_model=List[ProjectResponse])
def get_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """현재 사용자의 모든 프로젝트 조회"""

    # 1. 사용자의 프로젝트 조회 (with 관계 로딩)
    projects = db.query(Project)\
        .filter(Project.user_id == current_user.id)\
        .options(joinedload(Project.training_jobs))\
        .order_by(Project.created_at.desc())\
        .all()

    # 2. 각 프로젝트의 실험 수 계산
    result = []
    for project in projects:
        result.append({
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "experiment_count": len(project.training_jobs)
        })

    return result
```

**동작:**
1. `current_user`는 미들웨어가 자동 주입
2. SQLAlchemy로 프로젝트 조회
3. `user_id == current_user.id` 필터 → 본인 프로젝트만
4. `joinedload`로 관련 학습 작업도 함께 로드 (N+1 쿼리 방지)
5. 생성일 역순 정렬

---

#### 5단계: Database 쿼리 실행

**위치:** `mvp/backend/vision_platform.db` (SQLite 파일)

```sql
-- SQLAlchemy가 생성하는 쿼리
SELECT
    projects.id,
    projects.name,
    projects.description,
    projects.created_at,
    projects.updated_at,
    projects.user_id
FROM projects
WHERE projects.user_id = 1
ORDER BY projects.created_at DESC;

-- 각 프로젝트의 학습 작업 조회
SELECT
    training_jobs.id,
    training_jobs.project_id,
    training_jobs.status
FROM training_jobs
WHERE training_jobs.project_id IN (1, 2, 3);
```

**결과 예시:**
```
projects:
  id | name              | description           | user_id | created_at
  1  | Dog vs Cat        | 개/고양이 분류 프로젝트  | 1       | 2024-01-15
  2  | Face Detection    | 얼굴 인식 모델         | 1       | 2024-01-20
  3  | Car Segmentation  | 자동차 세그멘테이션     | 1       | 2024-01-25

training_jobs:
  id | project_id | status
  1  | 1          | completed
  2  | 1          | failed
  3  | 2          | running
  4  | 2          | completed
  5  | 2          | completed
```

**데이터베이스 동작:**
- SQLite 파일을 직접 읽기 (파일 I/O)
- 인덱스 사용: `projects.user_id` (WHERE 절), `created_at` (ORDER BY)
- 트랜잭션: READ COMMITTED (기본)

---

#### 6단계: Backend → Frontend 응답

**응답:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

[
  {
    "id": 3,
    "name": "Car Segmentation",
    "description": "자동차 세그멘테이션",
    "created_at": "2024-01-25T10:30:00",
    "updated_at": "2024-01-25T10:30:00",
    "experiment_count": 0
  },
  {
    "id": 2,
    "name": "Face Detection",
    "description": "얼굴 인식 모델",
    "created_at": "2024-01-20T15:20:00",
    "updated_at": "2024-01-22T09:15:00",
    "experiment_count": 3
  },
  {
    "id": 1,
    "name": "Dog vs Cat",
    "description": "개/고양이 분류 프로젝트",
    "created_at": "2024-01-15T14:00:00",
    "updated_at": "2024-01-18T11:30:00",
    "experiment_count": 2
  }
]
```

---

#### 7단계: Frontend 화면 렌더링

**위치:** 브라우저 DOM 업데이트

```typescript
// React State 업데이트
setProjects(data);  // 위 JSON 데이터
setLoading(false);

// 화면 렌더링
return (
  <div>
    <h1>내 프로젝트</h1>
    {projects.map(project => (
      <ProjectCard
        key={project.id}
        project={project}
      />
    ))}
  </div>
);
```

**화면:**
```
┌─────────────────────────────────────────┐
│ 내 프로젝트                              │
├─────────────────────────────────────────┤
│ 📁 Car Segmentation                     │
│    자동차 세그멘테이션                   │
│    실험: 0개 | 2024-01-25               │
├─────────────────────────────────────────┤
│ 📁 Face Detection                       │
│    얼굴 인식 모델                        │
│    실험: 3개 | 2024-01-20               │
├─────────────────────────────────────────┤
│ 📁 Dog vs Cat                           │
│    개/고양이 분류 프로젝트               │
│    실험: 2개 | 2024-01-15               │
└─────────────────────────────────────────┘
```

---

## 배포 환경 (Railway)

### 환경 구성
```
Frontend: https://frontend-production-xxxx.up.railway.app/dashboard
Backend:  https://backend-production-xxxx.up.railway.app
Database: PostgreSQL (Railway 관리형)
```

### 상세 흐름

#### 1단계: 대시보드 페이지 로드

**위치:** 브라우저 (https://frontend-production-xxxx.up.railway.app/dashboard)

**Frontend 코드:**
```typescript
// mvp/frontend/app/dashboard/page.tsx
const fetchProjects = async () => {
  const token = localStorage.getItem('access_token');

  // 환경변수 사용
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });

  const data = await response.json();
  setProjects(data);
};
```

**환경변수:**
```bash
# Railway 설정
NEXT_PUBLIC_API_URL=https://backend-production-xxxx.up.railway.app/api/v1
```

**차이점:**
- 로컬: `http://localhost:8000`
- 배포: `https://backend-production-xxxx.up.railway.app`
- HTTPS 사용

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
GET https://backend-production-xxxx.up.railway.app/api/v1/projects
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**네트워크:**
- 인터넷을 통한 HTTPS 요청
- Railway 내부 네트워크 or 공개 URL
- TLS/SSL 암호화

**차이점:**
- 로컬: HTTP, localhost
- 배포: HTTPS, 인터넷 (지연 시간 추가)

---

#### 3단계: Backend 인증 미들웨어

**동작:** 로컬과 동일

```python
def get_current_user(...):
    # JWT 토큰 검증
    payload = jwt.decode(token, settings.JWT_SECRET, ...)
    # DB 조회
    user = db.query(User).filter(User.email == email).first()
    return user
```

**환경변수:**
```bash
# Railway Backend 서비스 환경변수
JWT_SECRET=your-production-secret-key-change-this
JWT_ALGORITHM=HS256
```

**차이점:**
- 로컬: `.env` 파일
- 배포: Railway 대시보드 환경변수 (암호화 저장)

---

#### 4단계: Backend API 엔드포인트 실행

**동작:** 로컬과 완전 동일

```python
@router.get("/", response_model=List[ProjectResponse])
def get_projects(current_user: User = Depends(get_current_user), ...):
    projects = db.query(Project)\
        .filter(Project.user_id == current_user.id)\
        .all()
    return result
```

**차이점:** 없음 (코드 동일)

---

#### 5단계: Database 쿼리 실행

**위치:** Railway PostgreSQL

```sql
-- SQLAlchemy가 생성하는 쿼리 (동일)
SELECT
    projects.id,
    projects.name,
    projects.description,
    projects.created_at,
    projects.updated_at,
    projects.user_id
FROM projects
WHERE projects.user_id = 1
ORDER BY projects.created_at DESC;
```

**연결:**
```python
# DATABASE_URL로 PostgreSQL 연결
DATABASE_URL = "postgresql://postgres:xxx@containers-us-west-xxx.railway.app:5432/railway"

engine = create_engine(
    DATABASE_URL,
    pool_size=5,        # 연결 풀 (동시 요청 처리)
    max_overflow=10,    # 최대 추가 연결
    pool_pre_ping=True  # 연결 유효성 확인
)
```

**차이점:**

| 항목 | 로컬 (SQLite) | 배포 (PostgreSQL) |
|------|--------------|------------------|
| **연결 방식** | 파일 I/O | TCP/IP 네트워크 |
| **연결 주소** | `vision_platform.db` | `containers-us-west-xxx.railway.app:5432` |
| **동시 접속** | 제한적 (파일 락) | 다수 (연결 풀) |
| **트랜잭션** | 파일 수준 락 | MVCC (Multi-Version Concurrency Control) |
| **쿼리 속도** | 매우 빠름 (로컬) | 네트워크 지연 (~5-50ms) |
| **백업** | 수동 (파일 복사) | 자동 (Railway 관리) |

---

#### 6단계: Backend → Frontend 응답

**응답:** 로컬과 동일 (JSON 형식)

```json
[
  {
    "id": 3,
    "name": "Car Segmentation",
    ...
  }
]
```

**차이점:**
- HTTPS 암호화 (TLS 1.3)
- Response Header에 SSL 정보 포함

---

#### 7단계: Frontend 화면 렌더링

**동작:** 로컬과 완전 동일

```typescript
setProjects(data);
setLoading(false);
```

**화면:** 로컬과 동일

---

## 주요 차이점 요약

| 구분 | 로컬 환경 | 배포 환경 (Railway) |
|------|----------|-------------------|
| **Frontend URL** | http://localhost:3000 | https://frontend-production-xxxx.up.railway.app |
| **Backend URL** | http://localhost:8000 | https://backend-production-xxxx.up.railway.app |
| **프로토콜** | HTTP | HTTPS (암호화) |
| **네트워크** | localhost (loopback) | 인터넷 (TCP/IP) |
| **데이터베이스** | SQLite (파일) | PostgreSQL (서버) |
| **DB 연결** | 파일 I/O (직접 읽기) | TCP/IP 네트워크 (5432 포트) |
| **연결 풀** | 없음 (단일 파일) | pool_size=5 (동시 요청) |
| **쿼리 속도** | ~1ms (로컬) | ~10-50ms (네트워크) |
| **인증 토큰** | LocalStorage (동일) | LocalStorage (동일) |
| **환경변수** | `.env` 파일 | Railway 대시보드 |
| **에러 로그** | 터미널 | Railway Logs |

---

## 성능 비교

### 로컬 환경 (개발)

```
총 응답 시간: ~50ms

1. Frontend → Backend 요청: ~1ms (localhost)
2. JWT 토큰 검증: ~2ms
3. DB 쿼리 (SQLite): ~5ms (파일 I/O)
4. JSON 직렬화: ~2ms
5. Backend → Frontend 응답: ~1ms
```

### 배포 환경 (Railway)

```
총 응답 시간: ~200-500ms

1. Frontend → Backend 요청: ~50-100ms (인터넷, HTTPS)
2. JWT 토큰 검증: ~2ms
3. DB 쿼리 (PostgreSQL): ~10-50ms (네트워크 + 쿼리)
4. JSON 직렬화: ~2ms
5. Backend → Frontend 응답: ~50-100ms (인터넷, HTTPS)
```

**차이:** 배포 환경이 4-10배 느림 (네트워크 지연)

---

## 코드 차이 (환경별 동작)

### Frontend: API URL만 다름

```typescript
// 로컬
const API_URL = 'http://localhost:8000/api/v1';

// 배포
const API_URL = process.env.NEXT_PUBLIC_API_URL;
// = 'https://backend-production-xxxx.up.railway.app/api/v1'
```

### Backend: 데이터베이스 설정만 다름

```python
# 로컬
DATABASE_URL = "sqlite:///./vision_platform.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# 배포
DATABASE_URL = "postgresql://postgres:xxx@railway.app:5432/railway"
engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)
```

**비즈니스 로직은 완전히 동일!**

---

## 관련 파일

### Frontend
- `mvp/frontend/app/dashboard/page.tsx` - 대시보드 페이지
- `mvp/frontend/components/ProjectCard.tsx` - 프로젝트 카드 컴포넌트

### Backend
- `mvp/backend/app/api/projects.py` - 프로젝트 API
- `mvp/backend/app/api/dependencies.py` - 인증 미들웨어
- `mvp/backend/app/db/models.py` - Project, User 모델
- `mvp/backend/app/core/security.py` - JWT 검증

### Database
- 로컬: `mvp/backend/vision_platform.db`
- 배포: Railway PostgreSQL

---

## 디버깅 팁

### 로컬: 프로젝트가 안 보일 때

**확인 사항:**
1. JWT 토큰이 유효한가?
   ```typescript
   const token = localStorage.getItem('access_token');
   console.log('Token:', token);
   ```

2. Backend API 응답 확인
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/v1/projects
   ```

3. Database에 프로젝트가 있는가?
   ```bash
   sqlite3 mvp/backend/vision_platform.db
   > SELECT * FROM projects WHERE user_id = 1;
   ```

---

### 배포: 프로젝트가 안 보일 때

**확인 사항:**
1. Railway Backend 로그 확인
   - Railway 대시보드 → Backend → Logs
   - 에러 메시지 검색

2. PostgreSQL 데이터 확인
   ```bash
   railway run psql $DATABASE_URL
   > SELECT * FROM projects WHERE user_id = 1;
   ```

3. 네트워크 확인 (브라우저 개발자 도구)
   - F12 → Network 탭
   - `/projects` 요청 클릭
   - Status 200인지 확인
   - Response 데이터 확인

---

## 최적화 팁

### N+1 쿼리 문제 해결

**문제:**
```python
# 각 프로젝트마다 별도 쿼리 실행 (느림)
for project in projects:
    experiment_count = db.query(TrainingJob)\
        .filter(TrainingJob.project_id == project.id)\
        .count()
```

**해결:**
```python
# 한 번에 조인해서 가져오기 (빠름)
projects = db.query(Project)\
    .options(joinedload(Project.training_jobs))\
    .all()

for project in projects:
    experiment_count = len(project.training_jobs)  # 메모리에서 계산
```

### 페이지네이션 (프로젝트 많을 때)

```python
@router.get("/")
def get_projects(
    page: int = 1,
    per_page: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * per_page

    projects = db.query(Project)\
        .filter(Project.user_id == current_user.id)\
        .offset(offset)\
        .limit(per_page)\
        .all()

    return projects
```
