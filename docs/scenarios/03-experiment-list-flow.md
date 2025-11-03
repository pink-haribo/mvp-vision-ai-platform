# 시나리오 3: 프로젝트 내 실험 조회

## 개요

사용자가 특정 프로젝트를 선택하고, 해당 프로젝트의 모든 학습 실험(Training Jobs)을 조회하는 과정입니다.

**목표:** 선택한 프로젝트의 실험 목록 표시 (모델명, 상태, 정확도, 생성일)

---

## 로컬 환경 (개발)

### 환경 구성
```
Frontend: http://localhost:3000/projects/[project_id]
Backend:  http://localhost:8000
Database: SQLite (vision_platform.db)
```

### 상세 흐름

#### 1단계: 사용자가 프로젝트 선택

**위치:** 브라우저 (http://localhost:3000/dashboard)

**사용자 동작:**
```
대시보드에서 "Dog vs Cat" 프로젝트 클릭
→ http://localhost:3000/projects/1 페이지로 이동
```

**Frontend 라우팅:**
```typescript
// mvp/frontend/app/projects/[id]/page.tsx
'use client';

import { useParams } from 'next/navigation';

export default function ProjectDetailPage() {
  const params = useParams();
  const projectId = params.id;  // "1"

  const [project, setProject] = useState(null);
  const [experiments, setExperiments] = useState([]);

  useEffect(() => {
    fetchProjectDetail(projectId);
    fetchExperiments(projectId);
  }, [projectId]);

  // ...
}
```

**동작:**
- Next.js 동적 라우트: `[id]` → URL 파라미터
- `useParams()` 훅으로 `id` 추출
- 두 개의 API 호출 준비

---

#### 2단계-A: 프로젝트 정보 조회

**요청:**
```http
GET http://localhost:8000/api/v1/projects/1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Backend API:**
```python
# mvp/backend/app/api/projects.py

@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """프로젝트 상세 정보 조회"""

    project = db.query(Project)\
        .filter(Project.id == project_id)\
        .filter(Project.user_id == current_user.id)\
        .first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return project
```

**Database 쿼리:**
```sql
SELECT id, name, description, created_at, updated_at, user_id
FROM projects
WHERE id = 1 AND user_id = 1
LIMIT 1;
```

**응답:**
```json
{
  "id": 1,
  "name": "Dog vs Cat",
  "description": "개/고양이 분류 프로젝트",
  "created_at": "2024-01-15T14:00:00",
  "updated_at": "2024-01-18T11:30:00"
}
```

---

#### 2단계-B: 실험 목록 조회 (핵심)

**요청:**
```http
GET http://localhost:8000/api/v1/projects/1/experiments
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Frontend 코드:**
```typescript
const fetchExperiments = async (projectId: string) => {
  const token = localStorage.getItem('access_token');

  const response = await fetch(
    `http://localhost:8000/api/v1/projects/${projectId}/experiments`,
    {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );

  const data = await response.json();
  setExperiments(data);
};
```

---

#### 3단계: Backend API 엔드포인트 실행

**위치:** `mvp/backend/app/api/projects.py`

```python
@router.get("/{project_id}/experiments", response_model=List[TrainingJobResponse])
def get_project_experiments(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """프로젝트의 모든 실험 조회"""

    # 1. 프로젝트 권한 확인
    project = db.query(Project)\
        .filter(Project.id == project_id)\
        .filter(Project.user_id == current_user.id)\
        .first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # 2. 학습 작업 조회
    experiments = db.query(TrainingJob)\
        .filter(TrainingJob.project_id == project_id)\
        .order_by(TrainingJob.created_at.desc())\
        .all()

    return experiments
```

**동작:**
1. 프로젝트 소유권 확인 (`user_id == current_user.id`)
2. 해당 프로젝트의 모든 학습 작업 조회
3. 최신순 정렬

---

#### 4단계: Database 쿼리 실행

**위치:** `mvp/backend/vision_platform.db` (SQLite 파일)

```sql
-- 1. 프로젝트 권한 확인
SELECT id, user_id
FROM projects
WHERE id = 1 AND user_id = 1
LIMIT 1;

-- 2. 학습 작업 조회
SELECT
    id,
    project_id,
    model_name,
    framework,
    task_type,
    status,
    dataset_path,
    epochs,
    batch_size,
    learning_rate,
    accuracy,
    loss,
    created_at,
    updated_at,
    started_at,
    completed_at,
    error_message
FROM training_jobs
WHERE project_id = 1
ORDER BY created_at DESC;
```

**결과 예시:**
```
training_jobs:
  id | model_name        | framework | task_type            | status    | accuracy | loss  | created_at
  5  | yolo11n-seg       | ultralytics | instance_segmentation | running  | NULL     | NULL  | 2024-01-18 15:30:00
  3  | yolo11n           | ultralytics | object_detection     | completed | 0.89     | 0.234 | 2024-01-17 10:00:00
  2  | resnet50          | timm      | image_classification | failed    | NULL     | NULL  | 2024-01-16 14:20:00
  1  | efficientnetv2_s  | timm      | image_classification | completed | 0.92     | 0.156 | 2024-01-15 16:45:00
```

**데이터베이스 동작:**
- SQLite 파일 읽기
- 인덱스 사용: `project_id` (WHERE 절)
- 정렬: `created_at DESC` (메모리 소트)

---

#### 5단계: Backend → Frontend 응답

**응답:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

[
  {
    "id": 5,
    "project_id": 1,
    "model_name": "yolo11n-seg",
    "framework": "ultralytics",
    "task_type": "instance_segmentation",
    "status": "running",
    "dataset_path": "/app/datasets/seg-coco8",
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.01,
    "accuracy": null,
    "loss": null,
    "created_at": "2024-01-18T15:30:00",
    "started_at": "2024-01-18T15:31:00",
    "completed_at": null,
    "error_message": null
  },
  {
    "id": 3,
    "model_name": "yolo11n",
    "framework": "ultralytics",
    "task_type": "object_detection",
    "status": "completed",
    "accuracy": 0.89,
    "loss": 0.234,
    "created_at": "2024-01-17T10:00:00",
    "completed_at": "2024-01-17T11:45:00",
    ...
  },
  {
    "id": 2,
    "model_name": "resnet50",
    "framework": "timm",
    "status": "failed",
    "error_message": "CUDA out of memory",
    ...
  },
  {
    "id": 1,
    "model_name": "efficientnetv2_s",
    "framework": "timm",
    "status": "completed",
    "accuracy": 0.92,
    "loss": 0.156,
    ...
  }
]
```

---

#### 6단계: Frontend 화면 렌더링

**위치:** 브라우저 DOM 업데이트

```typescript
// React State 업데이트
setProject(projectData);
setExperiments(experimentsData);

// 화면 렌더링
return (
  <div>
    <h1>{project.name}</h1>
    <p>{project.description}</p>

    <h2>실험 목록</h2>
    <table>
      <thead>
        <tr>
          <th>모델</th>
          <th>태스크</th>
          <th>상태</th>
          <th>정확도</th>
          <th>생성일</th>
        </tr>
      </thead>
      <tbody>
        {experiments.map(exp => (
          <tr key={exp.id}>
            <td>{exp.model_name}</td>
            <td>{exp.task_type}</td>
            <td><StatusBadge status={exp.status} /></td>
            <td>{exp.accuracy ? `${(exp.accuracy * 100).toFixed(1)}%` : '-'}</td>
            <td>{formatDate(exp.created_at)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);
```

**화면:**
```
┌───────────────────────────────────────────────────────────────┐
│ Dog vs Cat                                                     │
│ 개/고양이 분류 프로젝트                                         │
├───────────────────────────────────────────────────────────────┤
│ 실험 목록                                        [+ 새 실험]   │
├─────────────┬──────────────┬──────────┬─────────┬────────────┤
│ 모델        │ 태스크       │ 상태     │ 정확도  │ 생성일     │
├─────────────┼──────────────┼──────────┼─────────┼────────────┤
│ yolo11n-seg │ segmentation │ 🟡 실행중 │ -       │ 2024-01-18│
│ yolo11n     │ detection    │ ✅ 완료   │ 89.0%   │ 2024-01-17│
│ resnet50    │ classification│ ❌ 실패  │ -       │ 2024-01-16│
│ efficientnetv2_s │ classification│ ✅ 완료 │ 92.0% │ 2024-01-15│
└─────────────┴──────────────┴──────────┴─────────┴────────────┘
```

---

## 배포 환경 (Railway)

### 환경 구성
```
Frontend: https://frontend-production-xxxx.up.railway.app/projects/[id]
Backend:  https://backend-production-xxxx.up.railway.app
Database: PostgreSQL (Railway)
```

### 상세 흐름

#### 1단계: 사용자가 프로젝트 선택

**위치:** https://frontend-production-xxxx.up.railway.app/projects/1

**동작:** 로컬과 동일

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
GET https://backend-production-xxxx.up.railway.app/api/v1/projects/1/experiments
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Frontend 코드:**
```typescript
// 환경변수 사용
const API_URL = process.env.NEXT_PUBLIC_API_URL;

const response = await fetch(
  `${API_URL}/projects/${projectId}/experiments`,
  { headers: { 'Authorization': `Bearer ${token}` } }
);
```

**차이점:**
- 로컬: `http://localhost:8000`
- 배포: `https://backend-production-xxxx.up.railway.app`
- HTTPS, 인터넷 통신

---

#### 3단계: Backend API 엔드포인트 실행

**동작:** 로컬과 완전 동일

```python
@router.get("/{project_id}/experiments", ...)
def get_project_experiments(...):
    experiments = db.query(TrainingJob)\
        .filter(TrainingJob.project_id == project_id)\
        .all()
    return experiments
```

**차이점:** 없음 (코드 동일)

---

#### 4단계: Database 쿼리 실행

**위치:** Railway PostgreSQL

```sql
-- PostgreSQL 쿼리 (SQL 구문은 동일)
SELECT
    id,
    project_id,
    model_name,
    ...
FROM training_jobs
WHERE project_id = 1
ORDER BY created_at DESC;
```

**연결:**
```python
# PostgreSQL 연결 풀
engine = create_engine(
    "postgresql://postgres:xxx@railway.app:5432/railway",
    pool_size=5,        # 5개 연결 유지
    max_overflow=10,    # 최대 15개 연결
    pool_pre_ping=True  # 연결 유효성 확인
)
```

**차이점:**

| 항목 | 로컬 (SQLite) | 배포 (PostgreSQL) |
|------|--------------|------------------|
| **연결** | 파일 직접 읽기 | TCP/IP 네트워크 |
| **쿼리 속도** | ~2-5ms | ~10-30ms (네트워크) |
| **동시 쿼리** | 순차 처리 (파일 락) | 병렬 처리 (MVCC) |
| **인덱스** | B-tree (동일) | B-tree (동일) |
| **트랜잭션** | 파일 수준 락 | Row-level locking |

---

#### 5단계: Backend → Frontend 응답

**응답:** 로컬과 동일 (JSON 형식)

**차이점:**
- HTTPS 암호화
- 네트워크 지연 (~50-100ms)

---

#### 6단계: Frontend 화면 렌더링

**동작:** 로컬과 완전 동일

---

## 주요 차이점 요약

| 구분 | 로컬 환경 | 배포 환경 (Railway) |
|------|----------|-------------------|
| **URL** | http://localhost:3000/projects/1 | https://frontend-production-xxxx.up.railway.app/projects/1 |
| **API 엔드포인트** | http://localhost:8000/api/v1/projects/1/experiments | https://backend-production-xxxx.up.railway.app/api/v1/projects/1/experiments |
| **프로토콜** | HTTP | HTTPS |
| **데이터베이스** | SQLite | PostgreSQL |
| **쿼리 속도** | ~2-5ms | ~10-30ms |
| **응답 시간** | ~20-50ms | ~100-300ms |
| **동시 사용자** | 1명 | 다수 |

---

## 실험 상태(Status) 이해

### 상태 종류

```python
# mvp/backend/app/db/models.py

class TrainingJob(Base):
    # ...
    status = Column(String)  # pending, running, completed, failed, cancelled
```

**상태 전환:**
```
pending → running → completed
                 └→ failed
                 └→ cancelled
```

### 각 상태별 의미

| 상태 | 의미 | accuracy | loss | error_message |
|------|-----|----------|------|---------------|
| **pending** | 대기 중 (학습 시작 전) | NULL | NULL | NULL |
| **running** | 실행 중 | NULL (or 중간값) | NULL (or 중간값) | NULL |
| **completed** | 정상 완료 | 최종값 (0.92) | 최종값 (0.156) | NULL |
| **failed** | 실패 | NULL | NULL | "CUDA out of memory" |
| **cancelled** | 사용자가 취소 | NULL (or 중간값) | NULL (or 중간값) | "Cancelled by user" |

### Frontend 표시

```typescript
const StatusBadge = ({ status }) => {
  const config = {
    pending: { emoji: '⏳', text: '대기 중', color: 'gray' },
    running: { emoji: '🟡', text: '실행 중', color: 'yellow' },
    completed: { emoji: '✅', text: '완료', color: 'green' },
    failed: { emoji: '❌', text: '실패', color: 'red' },
    cancelled: { emoji: '🚫', text: '취소', color: 'orange' },
  };

  const { emoji, text, color } = config[status];

  return (
    <span className={`badge-${color}`}>
      {emoji} {text}
    </span>
  );
};
```

---

## 성능 최적화

### 문제: 실험이 많을 때 (100개 이상)

**해결 1: 페이지네이션**

```python
@router.get("/{project_id}/experiments")
def get_project_experiments(
    project_id: int,
    page: int = 1,
    per_page: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * per_page

    experiments = db.query(TrainingJob)\
        .filter(TrainingJob.project_id == project_id)\
        .offset(offset)\
        .limit(per_page)\
        .order_by(TrainingJob.created_at.desc())\
        .all()

    return experiments
```

**해결 2: 상태별 필터링**

```typescript
// Frontend
const [statusFilter, setStatusFilter] = useState('all');

const fetchExperiments = async (projectId, status = 'all') => {
  let url = `${API_URL}/projects/${projectId}/experiments`;

  if (status !== 'all') {
    url += `?status=${status}`;
  }

  const response = await fetch(url, ...);
  // ...
};

// Backend
@router.get("/{project_id}/experiments")
def get_project_experiments(
    project_id: int,
    status: Optional[str] = None,
    ...
):
    query = db.query(TrainingJob)\
        .filter(TrainingJob.project_id == project_id)

    if status:
        query = query.filter(TrainingJob.status == status)

    return query.order_by(TrainingJob.created_at.desc()).all()
```

---

## 관련 파일

### Frontend
- `mvp/frontend/app/projects/[id]/page.tsx` - 프로젝트 상세 페이지
- `mvp/frontend/components/ExperimentTable.tsx` - 실험 목록 테이블
- `mvp/frontend/components/StatusBadge.tsx` - 상태 뱃지

### Backend
- `mvp/backend/app/api/projects.py` - 프로젝트 & 실험 API
- `mvp/backend/app/db/models.py` - TrainingJob 모델

### Database
- 로컬: `mvp/backend/vision_platform.db`
- 배포: Railway PostgreSQL

---

## 디버깅 팁

### 로컬: 실험 목록이 비어있을 때

**확인:**
```bash
# Database 확인
sqlite3 mvp/backend/vision_platform.db
> SELECT * FROM training_jobs WHERE project_id = 1;
```

**데이터 없으면:**
```sql
-- 테스트 데이터 삽입
INSERT INTO training_jobs (
    project_id,
    model_name,
    framework,
    task_type,
    status,
    dataset_path,
    epochs,
    batch_size,
    learning_rate,
    created_at
) VALUES (
    1,
    'resnet50',
    'timm',
    'image_classification',
    'completed',
    '/datasets/imagenet',
    50,
    32,
    0.001,
    datetime('now')
);
```

---

### 배포: 실험 목록이 안 불러와질 때

**Railway 로그 확인:**
```
Railway Dashboard → Backend Service → Logs

에러 예시:
[ERROR] Project not found
→ 권한 문제: user_id 불일치

[ERROR] Database connection timeout
→ PostgreSQL 연결 문제
```

**해결:**
```bash
# Railway shell로 DB 확인
railway run psql $DATABASE_URL
> SELECT * FROM training_jobs WHERE project_id = 1;
```

---

## 추가 기능 예시

### 실시간 업데이트 (WebSocket)

**running 상태 실험의 실시간 진행률 표시**

```typescript
// Frontend
useEffect(() => {
  const socket = io(process.env.NEXT_PUBLIC_WS_URL);

  // 실험별 WebSocket 구독
  experiments.forEach(exp => {
    if (exp.status === 'running') {
      socket.emit('subscribe', `experiment:${exp.id}`);
    }
  });

  // 실시간 업데이트 수신
  socket.on('training_progress', (data) => {
    // data: { experiment_id: 5, epoch: 10, loss: 0.234, accuracy: 0.85 }
    updateExperimentMetrics(data.experiment_id, data);
  });

  return () => socket.disconnect();
}, [experiments]);
```

**화면:**
```
┌─────────────┬──────────────┬──────────────────────┬─────────┐
│ 모델        │ 태스크       │ 상태                 │ 정확도  │
├─────────────┼──────────────┼──────────────────────┼─────────┤
│ yolo11n-seg │ segmentation │ 🟡 실행 중 (Epoch 10/50) │ 85.2%│
│             │              │ ████████░░░░░░░░░░ 40% │       │
└─────────────┴──────────────┴──────────────────────┴─────────┘
```
