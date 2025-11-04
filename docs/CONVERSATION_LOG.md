# Conversation Log

이 파일은 Claude Code 대화 세션의 타임라인을 기록합니다.
세션이 바뀌어도 이전 논의 내용을 빠르게 파악할 수 있습니다.

**사용 방법**: `/log-session` 명령어로 현재 세션 내용 추가

---

## [2025-11-04 17:30] Training Service Microservice 인프라 구축 및 데이터 접근 전략 수립

### 논의 주제
- Training Service Microservice 아키텍처 구현
- Framework별 독립 서비스 구성 (timm, ultralytics, huggingface)
- R2 Storage 직접 접근 전략
- DICE Format → Framework Format 변환 설계
- 데이터셋-모델 호환성 검증 전략

### 주요 결정사항

#### 1. Microservice 아키텍처 구현 (Railway 환경과 동일)
- **배경**:
  - 로컬 테스트가 subprocess 방식으로 동작
  - Railway 배포 환경은 microservice로 구성
  - 로컬과 배포 환경의 불일치 문제

- **결정**: 로컬에서도 microservice로 실행 ✅
  ```
  Backend (Port 8000)
    ↓ HTTP
  ultralytics-service (Port 8002)
  timm-service (Port 8001)
  huggingface-service (Port 8003)
  ```

- **구현 내용**:
  - Framework별 독립 venv 생성 (`venv-ultralytics`, `venv-timm`)
  - 독립 실행 스크립트 (`scripts/start-ultralytics-service.bat`)
  - Backend `.env`에 framework별 URL 설정
  - `TrainingServiceClient`가 framework 기반 라우팅 지원

#### 2. R2 Storage 직접 접근 (Option A 선택)
- **질문**: Training Service가 데이터를 어떻게 접근할 것인가?
  - Option A: Training Service가 R2 직접 접근 (추천 ✅)
  - Option B: Backend API 통해 다운로드

- **결정**: Option A - R2 직접 접근
- **이유**:
  - Microservice 철학에 맞음 (독립적 동작)
  - Backend 부담 감소
  - `platform_sdk/storage.py` 이미 구현됨
  - R2 credentials 공유 필요하지만 문제없음

- **구현 방식**:
  ```python
  # Training Service .env
  AWS_S3_ENDPOINT_URL=https://...r2.cloudflarestorage.com
  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...
  S3_BUCKET=vision-platform-prod

  # platform_sdk/storage.py
  get_dataset(dataset_id) → R2 다운로드 → 로컬 캐시
  ```

#### 3. Dataset ID 기반 접근 (Path 방식에서 전환)
- **현재 문제**:
  - 기존: `dataset_path` (파일 시스템 경로)
  - Frontend 흐름: User가 데이터셋 선택 (ID 기반)
  - R2 구조: `datasets/{id}/` (UUID 기반)

- **결정**: `dataset_id` 기반으로 전환
  ```python
  # Frontend → Backend
  {"dataset_id": "uuid-123"}

  # Backend → Training Service
  {"dataset_id": "uuid-123"}

  # Training Service
  dataset_path = get_dataset("uuid-123")
  # → R2: datasets/uuid-123/ 다운로드
  # → Local: /workspace/data/.cache/datasets/uuid-123/
  ```

#### 4. DICE Format 변환 전략
- **배경**:
  - R2에 DICE Format으로 저장됨 (`annotations.json`)
  - 각 framework는 고유 포맷 필요 (YOLO, COCO, ImageFolder 등)

- **변환 전략**:
  ```
  Training Service
    ↓ 1. Download
    datasets/{id}/annotations.json (DICE Format)

    ↓ 2. Convert
    dice_to_yolo()      → data.yaml, labels/*.txt
    dice_to_imagefolder() → train/class1/, val/class1/
    dice_to_coco()      → annotations/instances.json

    ↓ 3. Train
    UltralyticsAdapter(converted_path)
  ```

- **구현 위치**: `mvp/training/converters/`
  - `dice_to_yolo.py`
  - `dice_to_imagefolder.py`
  - `dice_to_coco.py`

#### 5. 데이터셋-모델 호환성 검증 (3-Tier 전략)
- **문제**:
  - Classification 데이터로 Detection 학습 불가
  - Segmentation → Detection 변환 가능
  - Detection → Classification 변환 애매

- **3-Tier 검증 전략**:
  ```
  Tier 1: Frontend (UX Hint) [P2]
    → 데이터셋 선택 시 호환성 힌트 표시

  Tier 2: Backend API (사전 검증) [P1]
    → GET /datasets/{id}/compatibility?task_type=...
    → DB 메타데이터 or annotations.json 파싱

  Tier 3: Training Service (실행 시 검증) [P0] ✅
    → prepare_dataset()에서 상세 검증
    → 변환 가능하면 변환, 불가능하면 명확한 에러
  ```

- **MVP 우선순위**: Tier 3만 구현 (필수)
  - 이유: 일단 동작하는 것 먼저, UX는 나중에

- **변환 규칙 테이블**:
  ```python
  CONVERSION_MATRIX = {
      ("instance_segmentation", "object_detection"): polygon_to_bbox,
      ("instance_segmentation", "image_classification"): use_dominant_class,
      ("object_detection", "image_classification"): use_dominant_class,
      ("image_classification", "object_detection"): None,  # ❌ 불가능
  }
  ```

### 구현 내용

#### Microservice 인프라
**스크립트 생성**:
- `mvp/scripts/setup-ultralytics-service.bat` - venv 생성 및 의존성 설치
- `mvp/scripts/start-ultralytics-service.bat` - 서비스 시작 (Port 8002)
- `mvp/scripts/setup-timm-service.bat` - timm 서비스 셋업
- `mvp/scripts/start-timm-service.bat` - timm 서비스 시작 (Port 8001)

**Backend 설정**:
```bash
# mvp/backend/.env
TIMM_SERVICE_URL=http://localhost:8001
ULTRALYTICS_SERVICE_URL=http://localhost:8002
HUGGINGFACE_SERVICE_URL=http://localhost:8003
TRAINING_SERVICE_URL=http://localhost:8001  # Fallback
```

**ultralytics-service 실행 확인**:
- ✅ Port 8002에서 정상 동작
- ✅ Health Check: `{"status":"healthy"}`
- ✅ Models API: 5개 모델 (yolo11n, yolo11n-seg, yolo11n-pose, yolo_world_v2_s, sam2_t)

#### 기존 코드 분석
**platform_sdk/storage.py**:
- ✅ `get_dataset(dataset_id)` 이미 구현됨
- ✅ 3-tier 캐싱: Local → R2 → Original source
- ✅ 자동 압축 해제 및 디렉토리 반환

**ultralytics_adapter.py**:
- ✅ `_resolve_dataset_path()` 메서드 존재
- ✅ Simple name 감지 → `get_dataset()` 호출
- ⚠️ 현재는 path 기반, dataset_id 기반으로 수정 필요

### 다음 단계 (우선순위 순)

#### Phase 1: 환경 설정 및 기본 연동
- [x] ultralytics-service venv 생성 및 의존성 설치
- [x] ultralytics-service 실행 스크립트
- [x] Backend .env 업데이트 (framework별 URL)
- [ ] Training Service .env 업데이트 (R2 credentials)
- [ ] Backend 실행 및 Training Service 연결 테스트

#### Phase 2: DICE Format 변환기 구현
- [ ] `mvp/training/converters/dice_to_yolo.py` 구현
  - annotations.json 파싱
  - Polygon → Bounding box 변환
  - data.yaml 생성
  - labels/*.txt 생성
- [ ] `platform_sdk/storage.py` 확장
  - `get_dataset_from_r2(dataset_id)` 디렉토리 다운로드
- [ ] 호환성 검증 로직
  - `check_detailed_compatibility()` 함수
  - CONVERSION_MATRIX 정의

#### Phase 3: 학습 파이프라인 E2E 테스트
- [ ] R2에 테스트 데이터셋 업로드 (sample-det-coco32)
- [ ] Backend → ultralytics-service 학습 시작
- [ ] 데이터 다운로드 → 변환 → 학습 전체 흐름 검증
- [ ] 메트릭 수집 및 로깅 확인

#### Phase 4: Checkpoint R2 저장
- [ ] `platform_sdk/storage.py`에 `upload_checkpoint()` 추가
- [ ] Adapter `save_checkpoint()` 수정
- [ ] R2 경로: `checkpoints/{job_id}/epoch_{epoch}.pth`

### 핵심 설계 원칙

1. **No Shortcuts, No Hardcoding** (CLAUDE.md)
   - ✅ 동적 모델 레지스트리 (Training Service API)
   - ✅ R2 Storage 기반 (로컬 파일시스템 의존성 제거)
   - ✅ Database 기반 메타데이터 (하드코딩 샘플 없음)

2. **Dependency Isolation**
   - ✅ Backend: PyTorch 없음
   - ✅ Training Services: Framework별 독립 venv
   - ✅ HTTP/JSON 통신만

3. **Production = Local**
   - ✅ Microservice 아키텍처 동일
   - ✅ R2 Storage 사용
   - ✅ 환경변수만 차이 (URL, credentials)

### 관련 문서
- **인프라**: [docs/planning/TRAINER_IMPLEMENTATION_PLAN.md](../planning/TRAINER_IMPLEMENTATION_PLAN.md)
- **데이터셋 설계**: [docs/datasets/DATASET_MANAGEMENT_DESIGN.md](../datasets/DATASET_MANAGEMENT_DESIGN.md)
- **DICE Format 스펙**: [docs/datasets/PLATFORM_DATASET_FORMAT.md](../datasets/PLATFORM_DATASET_FORMAT.md)
- **현재 상태**: [docs/datasets/CURRENT_STATUS.md](../datasets/CURRENT_STATUS.md)

### 기술 노트

#### R2 Storage 구조
```
vision-platform-prod/
├── datasets/
│   └── {id}/
│       ├── images/          # 원본 폴더 구조 유지
│       └── annotations.json # DICE Format v1.0
├── models/
│   └── pretrained/{framework}/{model_name}.pt
└── checkpoints/
    └── {job_id}/
        └── epoch_{n}.pth
```

#### Training Service 데이터 흐름
```
1. Backend → POST /training/start
   {"dataset_id": "uuid-123", "model_name": "yolo11n", ...}

2. Training Service → get_dataset("uuid-123")
   - Check local: /workspace/data/.cache/datasets/uuid-123/
   - Download R2: datasets/uuid-123/ → local cache
   - Return: local_path

3. DICE Format 변환
   - Parse: annotations.json
   - Check: compatibility with task_type
   - Convert: dice_to_yolo() → data.yaml + labels/
   - Return: converted_path

4. 학습 실행
   - UltralyticsAdapter(converted_path)
   - Train + Validate
   - Save checkpoint → R2
   - Log metrics → Backend
```

#### Framework별 Port 할당
```
Backend:           8000
timm-service:      8001
ultralytics-service: 8002
huggingface-service: 8003
Frontend:          3000
```

---

## [2025-11-04 16:00] 데이터셋 인증/권한 구현 및 학습 파이프라인 준비

### 논의 주제
- 데이터셋 인증 및 권한 체크 구현
- 학습 파이프라인 테스트 vs 스냅샷 구현 우선순위
- YOLO segmentation → DICE Format 변환
- 프론트엔드 UX 개선 (자동 네비게이션 제거)
- PR 생성 및 문서화

### 주요 결정사항

#### 1. 데이터셋 인증 시스템 구현
- **배경**: 데이터셋을 아무나 볼 수 있는 보안 문제 발견
- **구현 내용**:
  - Backend: 모든 dataset API에 `Depends(get_current_user)` 추가
  - Frontend: 모든 API 호출에 Bearer token 추가
  - Sidebar: 인증된 사용자만 "데이터셋", "프로젝트" 메뉴 표시
- **권한 규칙**:
  - 소유자(owner)만 삭제/업로드 가능
  - Public 데이터셋은 모든 인증 사용자 조회 가능
  - Private 데이터셋은 소유자만 접근

#### 2. 스냅샷 구현 시기 결정
- **질문**: 학습 파이프라인 테스트 전에 스냅샷 구현이 필요한가?
- **결정**: 학습 파이프라인 먼저 테스트 (Option A) ✅
- **이유**:
  - 스냅샷 없이도 학습 가능 (`dataset_snapshot_id`는 nullable)
  - 학습이 제대로 돌아가야 스냅샷도 의미 있음
  - DB 모델은 이미 준비됨 (빠른 전환 가능)
  - MVP 단계에서는 핵심 기능 검증 우선
- **위험 관리**: 초기 테스트 데이터셋은 수정하지 않기

#### 3. DICE Format 변환 준비
- **목적**: 학습 파이프라인 테스트용 데이터셋 준비
- **작업**: YOLO segmentation → DICE Format v1.0 변환
- **입력**: `C:\datasets\seg-coco32` (YOLO format)
- **출력**: `C:\datasets\dice_format\seg-coco32` (DICE format)
- **결과**:
  - 32 images, 209 annotations
  - 43 COCO classes (person, car, cup 등)
  - instance_segmentation 태스크

#### 4. 프론트엔드 UX 개선
- **문제**: 데이터셋 생성 후 상세 페이지로 자동 전환
- **해결**: 자동 네비게이션 제거, 테이블만 새로고침
- **이유**:
  - 여러 데이터셋 연속 생성 시 편리
  - 불필요한 화면 전환 감소
  - 사용자가 원하면 수동으로 클릭 가능

### 구현 내용

#### Backend (인증 추가)

**`mvp/backend/app/api/datasets.py`**:
```python
# 추가된 imports
from app.db.models import Dataset, User
from app.utils.dependencies import get_current_user

# 수정된 엔드포인트
@router.get("/available")
async def list_sample_datasets(
    current_user: User = Depends(get_current_user),  # 추가
    db: Session = Depends(get_db)
):
    # Owner OR public 필터링
    query = db.query(Dataset).filter(
        or_(
            Dataset.owner_id == current_user.id,
            Dataset.visibility == 'public'
        )
    )

@router.post("")
async def create_dataset(
    current_user: User = Depends(get_current_user),  # 추가
    ...
):
    new_dataset = Dataset(
        owner_id=current_user.id,  # 자동 설정
        ...
    )

@router.delete("/{dataset_id}")
async def delete_dataset(
    current_user: User = Depends(get_current_user),  # 추가
    ...
):
    # 소유자 확인
    if dataset.owner_id != current_user.id:
        raise HTTPException(403, "Permission denied")
```

**`mvp/backend/app/api/datasets_images.py`**:
- 모든 엔드포인트에 `current_user` 파라미터 추가
- 소유자 확인 로직 추가
- Public dataset 조회 허용 로직

**`mvp/backend/app/api/datasets_folder.py`**:
- 폴더 업로드 API에 인증 추가
- 소유자만 업로드 가능

#### Frontend (인증 토큰 추가)

**`mvp/frontend/components/Sidebar.tsx`**:
```tsx
{/* 인증된 사용자만 표시 */}
{isAuthenticated && (
  <div>
    <button onClick={onOpenDatasets}>데이터셋</button>
  </div>
)}

{isAuthenticated && (
  <div>프로젝트 목록</div>
)}
```

**`mvp/frontend/components/DatasetPanel.tsx`**:
```typescript
const fetchDatasets = async () => {
  const token = localStorage.getItem('access_token')

  if (!token) {
    console.error('No access token found')
    return
  }

  const response = await fetch(`${baseUrl}/datasets/available`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  })
}

const handleDeleteConfirm = async () => {
  const token = localStorage.getItem('access_token')

  const response = await fetch(`${baseUrl}/datasets/${id}`, {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  })
}
```

**`mvp/frontend/components/datasets/CreateDatasetModal.tsx`**:
```typescript
// useRouter import 제거
// router.push() 제거
// 성공 후 모달만 닫기
setTimeout(() => {
  handleClose()  // 네비게이션 없이 닫기만
}, 1000)
```

**기타 컴포넌트**:
- `DatasetImageUpload.tsx`: Bearer token 추가
- `DatasetImageGallery.tsx`: Bearer token 추가
- `ProjectDetail.tsx`: handleSaveEdit에 token 추가
- `datasets/[id]/page.tsx`: Bearer token 추가

#### 유틸리티 스크립트

**`mvp/backend/convert_yolo_seg_to_platform.py`** (새 파일):
- YOLO segmentation → DICE Format 변환
- Normalized coordinates → 절대 pixel coordinates
- Polygon segmentation 데이터 보존
- Bounding box 자동 계산
- Area 계산 (shoelace formula)
- Content hash 생성

### Git 작업

#### Commits (7개)
```
8996157 docs(datasets): add current status and next steps document
744fb3e chore: update gitignore for test files and database backups
99a5ef5 fix(frontend): remove auto-navigation after dataset creation
ae26d92 feat(mvp): implement authentication and authorization for datasets
ab28012 feat(datasets): enhance folder upload and add dataset deletion
d527411 feat(datasets): implement Create-then-Upload architecture
b1677fd feat(datasets): add individual image management with R2 presigned URLs
```

#### Pull Request
- **PR #12**: "feat(datasets): implement Dataset Entity with R2 Storage and Authentication"
- **Base**: main
- **28 commits** total in this feature branch
- **Status**: Ready for review

### 생성된 문서

#### `docs/datasets/CURRENT_STATUS.md` (새 파일)
**목적**: 다음 세션을 위한 종합 상태 문서

**포함 내용**:
- ✅ 완료된 기능 (Phase 1 & 2)
  - Core Infrastructure
  - Backend API (CRUD, Images, Folder)
  - Frontend Components
  - DICE Format v2.0
  - Training Integration
  - Authentication

- ⏳ 남은 작업 (Phase 3 & 4)
  - Sprint 1: 버전닝/스냅샷 (2-3일)
  - Sprint 2: UI/UX 개선 (1-2일)
  - Sprint 3: 무결성 관리 (2-3일)

- 📂 테스트 데이터셋
  - seg-coco32 (DICE Format)
  - 위치, 구조, 메타데이터, 사용법

- 🎯 다음 세션 시작 가이드
  - **Option A**: 학습 파이프라인 테스트 (추천)
  - Option B: 스냅샷 구현
  - Quick Start 명령어

- 🔍 중요 파일 경로 맵

### 테스트 데이터셋

**seg-coco32 (DICE Format v1.0)**:
- **위치**: `C:\datasets\dice_format\seg-coco32`
- **구조**:
  ```
  seg-coco32/
  ├── annotations.json    # DICE Format v1.0
  └── images/             # 32 images
  ```
- **메타데이터**:
  - Format: instance_segmentation
  - Images: 32장
  - Annotations: 209개 polygon segmentations
  - Classes: 43개 COCO 클래스
  - Avg annotations per image: 6.53개
- **Top 5 classes**: person (56), car (19), cup (15), giraffe (9), bird (8)

### 다음 단계

#### Option A: 학습 파이프라인 테스트 (추천 ✅)
**브랜치**: `feature/training-pipeline-test`

**목표**:
1. seg-coco32 데이터셋 Frontend에서 업로드
2. Training API 호출 테스트
3. Backend ↔ Training Service 통신 검증
4. 학습 시작/중지/모니터링 확인
5. MLflow 연동 확인

**Quick Start**:
```bash
# 새 브랜치 생성
git checkout main
git pull
git checkout -b feature/training-pipeline-test

# Backend 시작
cd mvp/backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Frontend 시작
cd mvp/frontend
npm run dev

# 데이터셋 업로드
# http://localhost:3000 → 로그인 → 데이터셋 → Create
# C:\datasets\dice_format\seg-coco32 폴더 선택

# 학습 시작
# 채팅: "seg-coco32 데이터셋으로 yolo11n-seg 모델 학습시작"
```

#### Option B: 스냅샷 구현
**브랜치**: `feature/dataset-snapshots`

**작업 내용**:
- POST `/datasets/{id}/snapshots` API
- 학습 시작 시 자동 스냅샷
- 스냅샷 목록 UI
- 버전 비교 뷰

### 관련 문서

- **상태 문서**: [CURRENT_STATUS.md](./datasets/CURRENT_STATUS.md)
- **설계 문서**: [DATASET_MANAGEMENT_DESIGN.md](./datasets/DATASET_MANAGEMENT_DESIGN.md)
- **구현 계획**: [IMPLEMENTATION_PLAN.md](./datasets/IMPLEMENTATION_PLAN.md)
- **포맷 스펙**: [PLATFORM_DATASET_FORMAT.md](./datasets/PLATFORM_DATASET_FORMAT.md)

### 기술 노트

#### 인증 흐름
```
User → Frontend (localStorage.getItem('access_token'))
     → Backend API (Authorization: Bearer {token})
     → Depends(get_current_user)
     → JWT 검증 및 User 객체 반환
     → 권한 체크 (owner_id 비교)
```

#### 데이터셋 권한 규칙
- **Public datasets**:
  - 모든 인증 사용자 조회 가능
  - 소유자만 수정/삭제
- **Private datasets**:
  - 소유자만 조회/수정/삭제
- **업로드/삭제**:
  - 항상 소유자만 가능

#### .gitignore 업데이트
추가된 패턴:
- `*.db.backup*` - DB 백업 파일
- `test_*.py` - 테스트 스크립트
- `convert_*.py` - 변환 유틸리티
- `migrate_*.py` - 마이그레이션 스크립트

### 핵심 파일

#### Backend
```
mvp/backend/app/
├── api/
│   ├── datasets.py              # ✅ 인증 추가
│   ├── datasets_folder.py       # ✅ 인증 추가
│   ├── datasets_images.py       # ✅ 인증 추가
│   └── training.py              # dataset_id 지원
├── utils/
│   ├── r2_storage.py
│   └── dependencies.py          # get_current_user
└── convert_yolo_seg_to_platform.py  # 새 파일 (gitignore)
```

#### Frontend
```
mvp/frontend/
├── components/
│   ├── DatasetPanel.tsx          # ✅ 토큰 추가
│   ├── Sidebar.tsx               # ✅ 조건부 렌더링
│   ├── ProjectDetail.tsx         # ✅ 토큰 추가
│   └── datasets/
│       ├── CreateDatasetModal.tsx    # ✅ 네비게이션 제거
│       ├── DatasetImageUpload.tsx    # ✅ 토큰 추가
│       └── DatasetImageGallery.tsx   # ✅ 토큰 추가
└── app/datasets/[id]/page.tsx    # ✅ 토큰 추가
```

#### Documentation
```
docs/datasets/
├── CURRENT_STATUS.md             # 새 파일 ⭐
├── DATASET_MANAGEMENT_DESIGN.md
├── IMPLEMENTATION_PLAN.md
└── PLATFORM_DATASET_FORMAT.md
```

---

## [2025-01-04 13:00] 데이터셋 관리 UI 통합 및 설계 논의

### 논의 주제
- 데이터셋 UI 레이아웃 통합 문제
- 하드코딩 데이터 제거
- 데이터셋 업로드 방식 설계
- 버전닝 전략
- 무결성 관리

### 주요 결정사항

#### 1. UI 레이아웃 통합
- **문제**: 데이터셋 버튼 클릭 시 전체 화면으로 나와서 기존 레이아웃(사이드바, 채팅, 작업공간) 무시
- **해결**:
  - 새 `DatasetPanel` 컴포넌트 생성 (컴팩트 테이블 디자인)
  - `app/page.tsx`에 상태 관리 추가
  - Sidebar에서 라우팅 대신 핸들러 호출
- **결과**: AdminProjectsPanel과 동일한 패턴으로 작업공간에 통합

#### 2. 하드코딩 데이터 제거
- **문제**: DB에 6개 샘플 데이터셋 하드코딩됨 (cls-imagenet-10 등)
- **원칙 위반**: CLAUDE.md - "no shortcut, no hardcoding, no dummy data"
- **해결**: DB에서 모든 샘플 데이터 삭제
- **결과**: 실제 업로드한 데이터만 표시

#### 3. task_type은 데이터셋 속성이 아니다
- **핵심 통찰**: 같은 이미지를 classification, detection, segmentation 등 다양하게 활용 가능
- **결정**:
  - ❌ Dataset.task_type 삭제
  - ✅ TrainingJob.task_type 추가
  - 데이터셋은 이미지 저장소, 학습 작업이 용도 결정

#### 4. 폴더 구조 유지
- **결정**: 업로드 시 폴더 구조 항상 유지
- **R2 경로**: `datasets/{id}/images/{original_path}`
- **이유**:
  - 원본 구조 보존
  - 파일명 충돌 방지
  - 유연성 확보

#### 5. labeled의 정의
- **정의**: `labeled = annotation.json 존재 여부`
- **규칙**:
  - labeled 업로드는 폴더만 가능 (annotation.json 필요)
  - unlabeled는 폴더/개별 파일 모두 가능
  - labeled 데이터셋에 labeled 폴더 병합 **금지**

#### 6. meta.json 생성 시점
- **unlabeled**: meta.json 없음 (DB만)
- **labeled 전환**: annotation.json + meta.json 함께 생성
- **export**: 항상 meta.json 포함
- **Single Source of Truth**: DB

#### 7. 버전닝 전략: Mutable + Snapshot
- **원칙**:
  - 데이터셋은 기본적으로 가변(mutable)
  - 학습 시작 시 자동 스냅샷 생성
  - 사용자가 명시적 버전 생성 가능 (v1, v2...)
- **효율성**:
  - 이미지는 모든 버전이 공유
  - 스냅샷은 annotation.json만 저장
  - 저장 공간 99% 절약 (10GB + 10MB + 10MB vs 30GB)

#### 8. 이미지 삭제 허용 + 무결성 관리
- **이미지 삭제**: 허용
- **영향받는 스냅샷 처리**:
  - 옵션 A: Broken 표시 (재현 불가)
  - 옵션 B: 자동 복구 (annotation 수정)
- **주기적 무결성 체크**: Celery task로 구현

### 구현 내용

#### Frontend
- `components/DatasetPanel.tsx`: 컴팩트 테이블 UI (새 파일)
  - 검색, 정렬 기능
  - 확장 가능한 행 (이미지 갤러리)
  - 이미지 업로드/조회

- `app/page.tsx`: 상태 관리 추가
  - `showDatasets` state
  - `handleOpenDatasets()` 핸들러
  - 작업공간에 DatasetPanel 렌더링

- `components/Sidebar.tsx`: 라우팅 제거
  - `router.push('/datasets')` → `onOpenDatasets()` 호출

#### Backend
- 기존 개별 이미지 업로드 API 유지
  - POST `/datasets/{id}/images`
  - GET `/datasets/{id}/images`

#### Database
- 하드코딩된 6개 샘플 데이터셋 삭제

### 관련 문서

- **설계 문서**: [DATASET_MANAGEMENT_DESIGN.md](./datasets/DATASET_MANAGEMENT_DESIGN.md)
  - 데이터 모델
  - 스토리지 구조
  - 12가지 업로드 시나리오
  - 버전닝 전략
  - 무결성 관리

- **기존 문서**:
  - [DICE_FORMAT_v2.md](./datasets/DICE_FORMAT_v2.md)
  - [STORAGE_ACCESS_PATTERNS.md](./datasets/STORAGE_ACCESS_PATTERNS.md)

### 다음 단계

#### Phase 2: 폴더 업로드 (다음 구현)
- [ ] 폴더 구조 유지 업로드 (`webkitdirectory`)
- [ ] labeled 데이터셋 생성 (annotation.json 포함)
- [ ] DB 모델 확장 (labeled, class_names, is_snapshot 등)

#### Phase 3: 버전닝
- [ ] 학습 시 자동 스냅샷
- [ ] 명시적 버전 생성
- [ ] 스냅샷 목록 UI

#### Phase 4: 무결성 관리
- [ ] 이미지 삭제 시 영향 분석
- [ ] Broken/복구 로직
- [ ] 주기적 무결성 체크

### 기술 스택
- Frontend: Next.js 14, TypeScript, Tailwind CSS
- Backend: FastAPI, Python, SQLAlchemy
- Storage: Cloudflare R2 (S3-compatible)
- Database: SQLite (local), PostgreSQL (production)

### 핵심 파일
- `mvp/frontend/components/DatasetPanel.tsx` (새로 생성)
- `mvp/frontend/app/page.tsx` (수정)
- `mvp/frontend/components/Sidebar.tsx` (수정)
- `mvp/backend/app/api/datasets_images.py` (기존)
- `mvp/backend/app/utils/r2_storage.py` (기존)

---

