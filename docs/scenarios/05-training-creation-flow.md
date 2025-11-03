# 시나리오 5: 학습(실험) 생성

## 개요

사용자가 모델을 선택하고, 데이터셋과 하이퍼파라미터를 설정해서 새 학습 작업을 생성하는 과정입니다.

**목표:** 학습 작업을 데이터베이스에 생성하고, 학습 준비 완료 상태로 만들기

**주의:** 이 단계에서는 **실제 학습은 시작되지 않습니다!** (다음 시나리오에서 설명)

---

## 로컬 환경 (개발)

### 환경 구성
```
Frontend: http://localhost:3000
Backend:  http://localhost:8000
Database: SQLite (vision_platform.db)
```

### 상세 흐름

#### 1단계: 사용자가 모델과 설정 입력

**위치:** 브라우저 (http://localhost:3000/projects/1)

**사용자 동작:**
```
1. [+ 새 실험] 클릭 → 모달 열림
2. 모델 선택: "yolo11n-seg" (Segmentation)
3. 데이터셋 선택: "/app/datasets/seg-coco8"
4. 하이퍼파라미터 설정:
   - Epochs: 50
   - Batch Size: 16
   - Learning Rate: 0.01
   - Optimizer: Adam
5. [학습 생성] 버튼 클릭
```

**Frontend 코드:**
```typescript
// mvp/frontend/components/NewExperimentModal.tsx

const [formData, setFormData] = useState({
  model_name: '',
  framework: '',
  task_type: '',
  dataset_path: '',
  dataset_format: 'yolo',  // imagefolder, yolo, coco
  epochs: 50,
  batch_size: 16,
  learning_rate: 0.01,
  optimizer: 'adam',
  pretrained: true,
});

const handleSubmit = async (e) => {
  e.preventDefault();

  const token = localStorage.getItem('access_token');

  const response = await fetch(
    `http://localhost:8000/api/v1/training/jobs`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        project_id: projectId,  // 1
        model_name: formData.model_name,  // "yolo11n-seg"
        framework: formData.framework,  // "ultralytics"
        task_type: formData.task_type,  // "instance_segmentation"
        dataset_path: formData.dataset_path,  // "/app/datasets/seg-coco8"
        dataset_format: formData.dataset_format,  // "yolo"
        epochs: formData.epochs,  // 50
        batch_size: formData.batch_size,  // 16
        learning_rate: formData.learning_rate,  // 0.01
        optimizer: formData.optimizer,  // "adam"
        pretrained: formData.pretrained,  // true
      })
    }
  );

  if (response.ok) {
    const newJob = await response.json();
    console.log('학습 작업 생성:', newJob);

    // 모달 닫고 목록 새로고침
    onClose();
    refreshExperiments();
  }
};
```

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
POST http://localhost:8000/api/v1/training/jobs
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "project_id": 1,
  "model_name": "yolo11n-seg",
  "framework": "ultralytics",
  "task_type": "instance_segmentation",
  "dataset_path": "/app/datasets/seg-coco8",
  "dataset_format": "yolo",
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.01,
  "optimizer": "adam",
  "pretrained": true
}
```

---

#### 3단계: Backend API 엔드포인트 실행

**위치:** `mvp/backend/app/api/training.py`

```python
@router.post("/jobs", response_model=TrainingJobResponse, status_code=201)
def create_training_job(
    job_request: TrainingJobCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """새 학습 작업 생성"""

    # 1. 프로젝트 권한 확인
    project = db.query(Project)\
        .filter(Project.id == job_request.project_id)\
        .filter(Project.user_id == current_user.id)\
        .first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # 2. 데이터셋 경로 검증 (선택적)
    if not Path(job_request.dataset_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset not found: {job_request.dataset_path}"
        )

    # 3. 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{project.id}/{timestamp}_{job_request.model_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 4. TrainingJob 생성 (DB에 저장)
    training_job = TrainingJob(
        project_id=job_request.project_id,
        model_name=job_request.model_name,
        framework=job_request.framework,
        task_type=job_request.task_type,
        dataset_path=job_request.dataset_path,
        dataset_format=job_request.dataset_format,
        num_classes=None,  # dataset_analyzer가 나중에 자동 감지
        epochs=job_request.epochs,
        batch_size=job_request.batch_size,
        learning_rate=job_request.learning_rate,
        optimizer=job_request.optimizer,
        output_dir=output_dir,
        status="pending",  # 대기 중 상태
        created_at=datetime.now(),
    )

    db.add(training_job)
    db.commit()
    db.refresh(training_job)

    return training_job
```

**동작:**
1. JWT 토큰으로 사용자 인증
2. 프로젝트 소유권 확인 (`user_id` 매칭)
3. 데이터셋 경로 검증
4. 출력 디렉토리 생성 (로컬 파일시스템)
5. TrainingJob 객체 생성 → DB INSERT
6. 상태: `pending` (학습 시작 전)

---

#### 4단계: Database INSERT

**위치:** `mvp/backend/vision_platform.db` (SQLite 파일)

```sql
INSERT INTO training_jobs (
    project_id,
    model_name,
    framework,
    task_type,
    dataset_path,
    dataset_format,
    num_classes,
    epochs,
    batch_size,
    learning_rate,
    optimizer,
    output_dir,
    status,
    created_at,
    updated_at
) VALUES (
    1,                                    -- project_id
    'yolo11n-seg',                        -- model_name
    'ultralytics',                        -- framework
    'instance_segmentation',              -- task_type
    '/app/datasets/seg-coco8',            -- dataset_path
    'yolo',                               -- dataset_format
    NULL,                                 -- num_classes (나중에 감지)
    50,                                   -- epochs
    16,                                   -- batch_size
    0.01,                                 -- learning_rate
    'adam',                               -- optimizer
    'outputs/1/20240118_153000_yolo11n-seg',  -- output_dir
    'pending',                            -- status
    '2024-01-18 15:30:00',                -- created_at
    '2024-01-18 15:30:00'                 -- updated_at
);
```

**생성된 레코드:**
```
id: 6 (자동 증가)
project_id: 1
model_name: yolo11n-seg
framework: ultralytics
status: pending
created_at: 2024-01-18 15:30:00
```

**데이터베이스 상태:**
- 새 행이 `training_jobs` 테이블에 추가됨
- Foreign Key: `project_id` → `projects.id`
- 인덱스 업데이트: `project_id`, `status`

---

#### 5단계: 파일시스템 출력 디렉토리 생성

**위치:** `C:\Users\flyto\Project\Github\mvp-vision-ai-platform\mvp\backend\outputs\`

```
mvp/backend/outputs/
  └── 1/  (project_id)
      └── 20240118_153000_yolo11n-seg/  (새로 생성됨)
          ├── checkpoints/  (학습 시 가중치 저장)
          ├── logs/         (학습 로그)
          └── results/      (학습 결과, 그래프 등)
```

**동작:**
```python
# Python pathlib
Path("outputs/1/20240118_153000_yolo11n-seg").mkdir(parents=True, exist_ok=True)
```

**파일시스템 변경:**
- 디렉토리 3개 생성 (outputs, 1, 20240118_153000_yolo11n-seg)
- 권한: 현재 사용자 (개발자)

---

#### 6단계: Backend → Frontend 응답

**응답:**
```http
HTTP/1.1 201 Created
Content-Type: application/json

{
  "id": 6,
  "project_id": 1,
  "model_name": "yolo11n-seg",
  "framework": "ultralytics",
  "task_type": "instance_segmentation",
  "dataset_path": "/app/datasets/seg-coco8",
  "dataset_format": "yolo",
  "num_classes": null,
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.01,
  "optimizer": "adam",
  "output_dir": "outputs/1/20240118_153000_yolo11n-seg",
  "status": "pending",
  "accuracy": null,
  "loss": null,
  "created_at": "2024-01-18T15:30:00",
  "updated_at": "2024-01-18T15:30:00",
  "started_at": null,
  "completed_at": null,
  "error_message": null
}
```

**HTTP 상태 코드:**
- `201 Created`: 리소스가 성공적으로 생성됨

---

#### 7단계: Frontend 화면 업데이트

**위치:** 브라우저

```typescript
// 응답 처리
if (response.ok) {
  const newJob = await response.json();

  // Toast 알림
  showToast('학습 작업이 생성되었습니다!', 'success');

  // 모달 닫기
  onClose();

  // 실험 목록 새로고침
  refreshExperiments();  // GET /projects/1/experiments 재호출
}
```

**화면 변화:**
```
┌───────────────────────────────────────────────────────────────┐
│ Dog vs Cat                                                     │
├───────────────────────────────────────────────────────────────┤
│ 실험 목록                                        [+ 새 실험]   │
├─────────────┬──────────────┬──────────┬─────────┬────────────┤
│ 모델        │ 태스크       │ 상태     │ 정확도  │ 생성일     │
├─────────────┼──────────────┼──────────┼─────────┼────────────┤
│ yolo11n-seg │ segmentation │ ⏳ 대기중 │ -       │ 방금 전 🆕│ ← 새로 추가됨!
│ yolo11n     │ detection    │ ✅ 완료   │ 89.0%   │ 2024-01-17│
│ resnet50    │ classification│ ❌ 실패  │ -       │ 2024-01-16│
└─────────────┴──────────────┴──────────┴─────────┴────────────┘
```

---

## 배포 환경 (Railway)

### 환경 구성
```
Frontend: https://frontend-production-xxxx.up.railway.app
Backend:  https://backend-production-xxxx.up.railway.app
Database: PostgreSQL (Railway)
```

### 상세 흐름

#### 1단계: 사용자가 모델과 설정 입력

**동작:** 로컬과 동일

---

#### 2단계: Frontend → Backend HTTP 요청

**요청:**
```http
POST https://backend-production-xxxx.up.railway.app/api/v1/training/jobs
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "project_id": 1,
  "model_name": "yolo11n-seg",
  ...
}
```

**차이점:**
- HTTPS 프로토콜
- Railway URL

---

#### 3단계: Backend API 엔드포인트 실행

**동작:** 로컬과 거의 동일

```python
@router.post("/jobs", ...)
def create_training_job(...):
    # 1. 프로젝트 권한 확인 (동일)
    # 2. 데이터셋 검증 (동일)
    # 3. 출력 디렉토리 생성 (동일)
    # 4. TrainingJob 생성 (동일)
    ...
```

**차이점:**
- Docker 컨테이너에서 실행
- 환경변수로 설정 주입

---

#### 4단계: Database INSERT

**위치:** Railway PostgreSQL

```sql
-- PostgreSQL 쿼리 (SQL 구문은 동일)
INSERT INTO training_jobs (
    project_id,
    model_name,
    framework,
    ...
) VALUES (
    1,
    'yolo11n-seg',
    'ultralytics',
    ...
) RETURNING id;  -- PostgreSQL은 RETURNING 사용
```

**연결:**
```python
# PostgreSQL 연결
DATABASE_URL = "postgresql://postgres:xxx@railway.app:5432/railway"
engine = create_engine(DATABASE_URL, pool_size=5)
```

**차이점:**

| 항목 | 로컬 (SQLite) | 배포 (PostgreSQL) |
|------|--------------|------------------|
| **연결** | 파일 직접 쓰기 | TCP/IP 네트워크 |
| **INSERT 속도** | ~1-2ms | ~5-10ms (네트워크) |
| **트랜잭션** | 파일 락 | Row-level locking |
| **AUTOINCREMENT** | SQLite `AUTOINCREMENT` | PostgreSQL `SERIAL` |
| **RETURNING** | 지원 안 함 | `RETURNING id` 사용 |

---

#### 5단계: 파일시스템 출력 디렉토리 생성

**위치:** Docker 컨테이너 내부 (`/app/outputs/`)

```
Docker 컨테이너:
  /app/
    ├── outputs/  (볼륨 마운트 or 컨테이너 내부)
    │   └── 1/
    │       └── 20240118_153000_yolo11n-seg/
    └── datasets/  (샘플 데이터셋)
```

**Railway 볼륨 설정:**
```yaml
# railway.toml (선택적)
[volumes]
outputs:
  mount_path: /app/outputs
```

**차이점:**

| 항목 | 로컬 | 배포 (Railway) |
|------|------|---------------|
| **경로** | `C:\Users\...\mvp\backend\outputs\` | `/app/outputs/` (컨테이너) |
| **파일시스템** | Windows NTFS | Linux ext4 |
| **영속성** | 영구 저장 | 볼륨 사용 시 영구, 아니면 재배포 시 삭제 |
| **권한** | 현재 사용자 | Docker 사용자 (uid 1000) |

**주의:**
- Railway는 기본적으로 **ephemeral storage** (임시 저장)
- 재배포 시 `/app/outputs/` 디렉토리 내용 삭제됨
- **해결:** Railway Volume 설정 or S3 사용

---

#### 6단계: Backend → Frontend 응답

**응답:** 로컬과 동일 (JSON 형식)

**차이점:**
- HTTPS 암호화

---

#### 7단계: Frontend 화면 업데이트

**동작:** 로컬과 동일

---

## 주요 차이점 요약

| 구분 | 로컬 환경 | 배포 환경 (Railway) |
|------|----------|-------------------|
| **API URL** | http://localhost:8000 | https://backend-production-xxxx.up.railway.app |
| **프로토콜** | HTTP | HTTPS |
| **데이터베이스** | SQLite | PostgreSQL |
| **INSERT 속도** | ~1-2ms | ~5-10ms |
| **출력 디렉토리** | Windows 로컬 드라이브 | Docker 컨테이너 내부 |
| **디렉토리 경로** | `C:\...\outputs\` | `/app/outputs/` |
| **영속성** | 영구 저장 | Ephemeral (재배포 시 삭제) |
| **응답 시간** | ~20-50ms | ~100-200ms |

---

## 학습 작업 상태(Status) 전환

### 생성 시 초기 상태

```python
training_job.status = "pending"  # 대기 중
```

### 상태 전환 흐름

```
pending  → (사용자가 "학습 시작" 버튼 클릭)
running  → (학습 진행 중)
completed → (정상 완료)
         or
failed   → (에러 발생)
         or
cancelled → (사용자 취소)
```

**이 시나리오에서의 상태:** `pending` (다음 시나리오에서 `running`으로 변경)

---

## 데이터베이스 스키마

### TrainingJob 모델

```python
# mvp/backend/app/db/models.py

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    # 모델 정보
    model_name = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    task_type = Column(String, nullable=False)

    # 데이터셋 정보
    dataset_path = Column(String, nullable=False)
    dataset_format = Column(String, default="imagefolder")
    num_classes = Column(Integer, nullable=True)

    # 하이퍼파라미터
    epochs = Column(Integer, default=50)
    batch_size = Column(Integer, default=32)
    learning_rate = Column(Float, default=0.001)
    optimizer = Column(String, default="adam")

    # 출력
    output_dir = Column(String, nullable=False)

    # 상태
    status = Column(String, default="pending")  # pending, running, completed, failed, cancelled

    # 결과 (학습 완료 후 저장)
    accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)

    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 에러 정보
    error_message = Column(Text, nullable=True)

    # 관계
    project = relationship("Project", back_populates="training_jobs")
```

---

## 검증 로직

### 1. 프로젝트 권한 확인

```python
project = db.query(Project)\
    .filter(Project.id == job_request.project_id)\
    .filter(Project.user_id == current_user.id)\
    .first()

if not project:
    raise HTTPException(status_code=404, detail="Project not found")
```

**목적:**
- 다른 사용자의 프로젝트에 학습 작업 생성 방지
- JWT 토큰의 사용자와 프로젝트 소유자 일치 확인

---

### 2. 데이터셋 경로 검증

```python
from pathlib import Path

if not Path(job_request.dataset_path).exists():
    raise HTTPException(
        status_code=400,
        detail=f"Dataset not found: {job_request.dataset_path}"
    )
```

**목적:**
- 존재하지 않는 데이터셋 경로 방지
- 학습 시작 전 미리 에러 감지

**로컬 vs 배포:**
- 로컬: `C:\datasets\seg-coco8` (절대 경로)
- 배포: `/app/datasets/seg-coco8` (Docker 컨테이너 경로)

---

### 3. 하이퍼파라미터 범위 확인 (선택적)

```python
if job_request.epochs < 1 or job_request.epochs > 1000:
    raise HTTPException(status_code=400, detail="Epochs must be between 1 and 1000")

if job_request.batch_size < 1 or job_request.batch_size > 512:
    raise HTTPException(status_code=400, detail="Batch size must be between 1 and 512")

if job_request.learning_rate <= 0 or job_request.learning_rate > 1:
    raise HTTPException(status_code=400, detail="Learning rate must be between 0 and 1")
```

**목적:**
- 잘못된 하이퍼파라미터로 학습 시작 방지
- GPU OOM 방지 (batch_size 너무 큼)

---

## 출력 디렉토리 명명 규칙

```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs/{project_id}/{timestamp}_{model_name}"

# 예시:
# outputs/1/20240118_153000_yolo11n-seg
```

**형식:**
- `outputs/`: 고정 경로
- `{project_id}/`: 프로젝트별 분리
- `{timestamp}_{model_name}`: 실험별 고유 디렉토리

**장점:**
- 충돌 방지 (timestamp로 고유성 보장)
- 정렬 편리 (timestamp 순)
- 직관적 (model_name 포함)

---

## 관련 파일

### Frontend
- `mvp/frontend/components/NewExperimentModal.tsx` - 실험 생성 모달
- `mvp/frontend/components/ModelSelector.tsx` - 모델 선택 컴포넌트
- `mvp/frontend/components/HyperparameterForm.tsx` - 하이퍼파라미터 입력 폼

### Backend
- `mvp/backend/app/api/training.py` - 학습 API
- `mvp/backend/app/db/models.py` - TrainingJob 모델
- `mvp/backend/app/schemas/training.py` - Pydantic 스키마

### Database
- 로컬: `mvp/backend/vision_platform.db`
- 배포: Railway PostgreSQL

---

## 디버깅 팁

### 로컬: 학습 작업 생성 실패

**에러: "Project not found"**
```
원인: project_id와 user_id 불일치
해결: 프로젝트 소유자 확인
```

```bash
sqlite3 mvp/backend/vision_platform.db
> SELECT id, user_id FROM projects WHERE id = 1;
> SELECT id, email FROM users;
```

---

**에러: "Dataset not found"**
```
원인: 데이터셋 경로 오류
해결: 경로 확인
```

```bash
# Windows
dir C:\datasets\seg-coco8

# Linux/Mac
ls /app/datasets/seg-coco8
```

---

### 배포: 학습 작업 생성 실패

**Railway 로그 확인:**
```
Railway Dashboard → Backend Service → Logs

에러 예시:
[ERROR] Dataset not found: /app/datasets/seg-coco8
→ 샘플 데이터셋이 Docker 이미지에 포함되지 않음
```

**해결:**
```dockerfile
# mvp/backend/Dockerfile
COPY mvp/backend/sample_datasets/ /app/datasets/
```

---

## 다음 단계

이 시나리오에서는 **학습 작업을 생성**했습니다:
- Database에 레코드 생성
- 상태: `pending`
- 출력 디렉토리 준비

**다음 시나리오 (6번):**
- 사용자가 "학습 시작" 버튼 클릭
- 상태: `pending` → `running`
- **실제 학습 실행** (subprocess or Training Service API)
- 실시간 진행률 업데이트
- 학습 완료 처리
