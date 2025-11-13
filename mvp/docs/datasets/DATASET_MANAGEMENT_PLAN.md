# R2 기반 데이터셋 관리 시스템

**작성일**: 2025-11-03
**브랜치**: feature/dataset-entity
**상태**: Phase 1 구현 완료, Phase 2 진행 중

---

## 1. 배경 및 문제 인식

### 1.1 초기 상황: 로컬 파일 시스템의 한계

**초기 아키텍처** (MVP Phase 1-2):
```
사용자 → 로컬 파일 경로 입력 (C:\datasets\imagenet-10)
         ↓
Backend → 로컬 파일 시스템에서 직접 읽기
         ↓
Training Service → 로컬에서 학습
```

**문제점**:
1. **배포 불가능**: Railway/Cloud 환경에서 로컬 파일 시스템 접근 불가
2. **협업 불가**: 팀원이 같은 데이터셋 사용 불가
3. **재현성 부족**: 데이터셋 버전 관리 안 됨
4. **확장성 제한**: 대용량 데이터 처리 어려움
5. **권한 관리 없음**: 누가 어떤 데이터에 접근할 수 있는지 불명확

### 1.2 전환점: Railway 배포 준비

**2025-10-29 ~ 11-03 주요 결정**:
- Railway 프로덕션 배포를 위해 클라우드 스토리지 필수
- Cloudflare R2 선택 (S3 호환, 무료 egress)
- 데이터셋을 First-Class Entity로 승격

**핵심 질문들**:
1. 사용자가 데이터를 어떻게 업로드하고 관리할 것인가?
2. 플랫폼 제공 샘플 데이터셋은 어떻게 취급할 것인가?
3. 팀원 간 데이터셋 공유는 어떻게 구현할 것인가?
4. 데이터셋 버전 관리는 필요한가?

---

## 2. 핵심 설계 원칙

### 2.1 데이터셋 = R2 Storage + DB Metadata

**기본 아이디어**:
```
Dataset {
  id: UUID (e.g., "cls-imagenet-10", "550e8400-e29b-41d4-a716-446655440000")
  R2 Storage: datasets/{id}/ (실제 이미지/레이블 파일)
  DB Record: 메타데이터 (이름, 형식, 클래스 수, 소유자, 권한)
}
```

**왜 UUID?**
- 전역적으로 고유 (multi-tenant 지원)
- 충돌 없음
- 예측 불가능 (보안)

### 2.2 Mutable Dataset (점진적 데이터 구축)

**사용자 시나리오**:
```
Day 1: 사용자가 100장 업로드 → 학습 시작
Day 2: 추가로 50장 업로드 → 동일 데이터셋, num_images 업데이트
Day 3: 150장 레이블링 완료 → 메타데이터 갱신, 학습 재시작
```

**Immutable vs Mutable 비교**:

| 방식 | 장점 | 단점 | 선택 |
|------|------|------|------|
| **Immutable** | 재현성 보장, 버전 관리 쉬움 | 작은 변경에도 새 버전 생성, 스토리지 낭비 | ❌ |
| **Mutable** | 유연성, 스토리지 효율적 | 재현성 어려움 (해결: content_hash) | ✅ |

**우리의 선택: Mutable + content_hash**
- 데이터는 mutable (덮어쓰기 가능)
- 재현성은 `content_hash` (SHA256)로 보장
- 중요한 시점은 `version` 필드로 스냅샷

### 2.3 Public/Private Visibility

**3가지 접근 레벨**:

```sql
visibility ENUM('public', 'private', 'organization')
```

| Visibility | 접근 가능 대상 | 사용 사례 |
|------------|----------------|-----------|
| `public` | 모든 사용자 | 플랫폼 샘플 데이터셋 (ImageNet, COCO128) |
| `private` | 소유자만 | 개인 프로젝트 데이터 |
| `organization` | 같은 조직 멤버 | 팀 협업 데이터 |

**플랫폼 샘플 데이터셋 = 단순히 `visibility='public'` + `tags=['platform-sample']`**

### 2.4 No Special Treatment for Public Datasets

**중요한 철학**:
> ImageNet, COCO 같은 공개 데이터셋을 특별 취급하지 않는다.
> 그냥 우리 데이터 관리 방식에 따른 "one of them"이다.

**이유**:
- 우리 타겟 사용자 = **커스텀 데이터를 다루는 사람들**
- HuggingFace, TFDS 통합은 우선순위 낮음
- 플랫폼 샘플 = 튜토리얼/테스트용, 특별한 API 불필요

**결과**:
```python
# ❌ Wrong: 샘플 데이터셋과 사용자 데이터셋을 다르게 취급
if dataset.is_sample:
    return SampleDatasetResponse(...)
else:
    return UserDatasetResponse(...)

# ✅ Correct: 모두 동일한 Dataset 모델
dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
return DatasetResponse(dataset)  # 샘플이든 사용자 데이터든 동일
```

---

## 3. 데이터베이스 스키마

### 3.1 Dataset 테이블

```sql
CREATE TABLE datasets (
    -- Identity
    id VARCHAR(100) PRIMARY KEY,  -- UUID or human-readable (e.g., "cls-imagenet-10")
    name VARCHAR(200) NOT NULL,
    description TEXT,

    -- Ownership (nullable for public datasets)
    owner_id INTEGER REFERENCES users(id) ON DELETE SET NULL,

    -- Access Control
    visibility VARCHAR(20) DEFAULT 'private',  -- 'public', 'private', 'organization'
    tags JSON,  -- ['platform-sample', 'object-detection', 'coco']

    -- Storage (R2/S3/GCS)
    storage_path VARCHAR(500) NOT NULL,  -- "datasets/det-coco8/" or "datasets/{uuid}/"
    storage_type VARCHAR(20) DEFAULT 'r2',  -- 'r2', 's3', 'gcs'

    -- Dataset Metadata (auto-detected or user-provided)
    format VARCHAR(50) NOT NULL,  -- 'yolo', 'imagefolder', 'coco', 'pascal_voc'
    task_type VARCHAR(50) NOT NULL,  -- 'image_classification', 'object_detection'
    num_classes INTEGER,
    num_images INTEGER DEFAULT 0,
    class_names JSON,  -- ['cat', 'dog', 'bird', ...]

    -- Versioning (for reproducibility)
    version INTEGER DEFAULT 1,
    content_hash VARCHAR(64),  -- SHA256 of dataset content
    last_modified_at TIMESTAMP,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_datasets_visibility ON datasets(visibility);
CREATE INDEX idx_datasets_owner ON datasets(owner_id);
CREATE INDEX idx_datasets_task_type ON datasets(task_type);
```

**필드 설명**:

| 필드 | 설명 | 예시 |
|------|------|------|
| `id` | UUID 또는 사람이 읽을 수 있는 ID | `cls-imagenet-10`, `550e8400-...` |
| `storage_path` | R2/S3 내 경로 | `datasets/det-coco8/` |
| `storage_type` | 스토리지 제공자 | `r2`, `s3`, `gcs` |
| `format` | 데이터셋 형식 | `imagefolder`, `yolo`, `coco` |
| `task_type` | 작업 유형 | `image_classification`, `object_detection` |
| `num_images` | 총 이미지 수 | `13000`, `128` |
| `content_hash` | 데이터 무결성 검증 | `a3f2b8c9d1e5...` (SHA256) |
| `tags` | 유연한 분류 | `['platform-sample', 'coco', 'detection']` |

### 3.2 DatasetPermission 테이블

```sql
CREATE TABLE dataset_permissions (
    id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(100) REFERENCES datasets(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,

    -- Role-based access
    role VARCHAR(20) DEFAULT 'viewer',  -- 'owner', 'editor', 'viewer'

    -- Audit
    granted_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    granted_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(dataset_id, user_id)
);

CREATE INDEX idx_dataset_permissions_user ON dataset_permissions(user_id);
CREATE INDEX idx_dataset_permissions_dataset ON dataset_permissions(dataset_id);
```

**역할 정의**:

| Role | 권한 |
|------|------|
| `owner` | 읽기, 쓰기, 삭제, 권한 부여 |
| `editor` | 읽기, 쓰기 (파일 업로드/수정) |
| `viewer` | 읽기만 가능 (학습에 사용) |

**접근 제어 로직**:
```python
def can_access_dataset(user_id: int, dataset_id: str, db: Session) -> bool:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    # 1. Public dataset → 모든 사용자 접근 가능
    if dataset.visibility == 'public':
        return True

    # 2. Owner → 항상 접근 가능
    if dataset.owner_id == user_id:
        return True

    # 3. DatasetPermission 확인
    permission = db.query(DatasetPermission).filter(
        DatasetPermission.dataset_id == dataset_id,
        DatasetPermission.user_id == user_id
    ).first()

    return permission is not None
```

---

## 4. R2 스토리지 구조

### 4.1 R2 Bucket 구조

```
vision-platform-dev/  (또는 vision-platform-prod/)
├── datasets/
│   ├── cls-imagenet-10/
│   │   ├── train/
│   │   │   ├── cat/
│   │   │   │   ├── img_001.jpg
│   │   │   │   └── ...
│   │   │   └── dog/
│   │   └── val/
│   │   └── meta.json  ← 메타데이터 (자동 생성)
│   │
│   ├── det-coco128/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── data.yaml
│   │   └── meta.json
│   │
│   └── {user-uuid}/
│       ├── images/
│       ├── labels/
│       └── meta.json
│
├── models/
│   └── pretrained/  (학습된 모델 weights)
│
└── checkpoints/
    └── {job_id}/  (학습 중 checkpoints)
```

### 4.2 meta.json 구조

```json
{
  "id": "cls-imagenet-10",
  "name": "ImageNet-10",
  "format": "imagefolder",
  "task_type": "image_classification",
  "num_classes": 10,
  "num_images": 13000,
  "class_names": ["cat", "dog", "bird", "car", "plane", ...],
  "splits": {
    "train": 10400,
    "val": 2600
  },
  "statistics": {
    "avg_width": 224,
    "avg_height": 224,
    "total_size_mb": 2048.5
  },
  "content_hash": "a3f2b8c9d1e5f7a2c4b6d8e0...",  // SHA256
  "last_modified": "2025-11-03T10:30:00Z",
  "version": 1
}
```

**meta.json 역할**:
1. R2에서 다운로드 없이 메타데이터 확인
2. DB 레코드와 동기화 검증
3. 버전 충돌 감지

---

## 5. 데이터 플로우

### 5.1 플랫폼 샘플 데이터셋 (Admin 업로드)

```
[Admin 로컬]
  ↓
scripts/upload_sample_datasets.py
  1. DatasetAnalyzer로 자동 분석 (형식, 클래스, 이미지 수)
  2. meta.json 생성
  3. R2에 업로드 (datasets/cls-imagenet-10/)
  4. DB에 Dataset 레코드 생성
     - visibility: 'public'
     - tags: ['platform-sample']
  ↓
[R2: datasets/cls-imagenet-10/]
[DB: Dataset(id='cls-imagenet-10', visibility='public')]
```

**코드 예시**:
```python
# scripts/upload_sample_datasets.py
dataset = Dataset(
    id="cls-imagenet-10",
    name="ImageNet-10",
    description="Platform sample dataset for image_classification",
    owner_id=None,  # 소유자 없음 (플랫폼 제공)
    visibility="public",
    tags=["platform-sample", "imagenet"],
    storage_path="datasets/cls-imagenet-10/",
    storage_type="r2",
    format="imagefolder",
    task_type="image_classification",
    num_classes=10,
    num_images=13000,
    class_names=["cat", "dog", ...],
    content_hash="a3f2b8c9...",
    version=1
)
db.add(dataset)
db.commit()
```

### 5.2 사용자 데이터셋 업로드

**시나리오 1: 초기 업로드 (Day 1)**
```
[Frontend: DatasetUploadModal]
  → 파일 선택 (zip/folder)
  ↓
POST /api/v1/datasets/upload
  1. UUID 생성 (e.g., "550e8400-...")
  2. 임시 저장 및 자동 분석 (DatasetAnalyzer)
  3. R2 업로드 (datasets/{uuid}/)
  4. DB 레코드 생성
     - owner_id: 현재 사용자
     - visibility: 'private'
     - num_images: 100
  5. 응답: dataset_id
  ↓
[R2: datasets/550e8400-.../]
[DB: Dataset(id='550e8400-...', owner_id=123, num_images=100)]
```

**시나리오 2: 점진적 업로드 (Day 2, +50장)**
```
[Frontend]
  → 기존 dataset_id 선택
  → 파일 추가 업로드
  ↓
POST /api/v1/datasets/{id}/upload
  1. 권한 확인 (owner or editor)
  2. R2에 추가 업로드 (덮어쓰기)
  3. 메타데이터 재분석
  4. DB 업데이트
     - num_images: 150 (100 → 150)
     - content_hash: 새 해시
     - last_modified_at: NOW()
  ↓
[R2: datasets/550e8400-.../ (150장)]
[DB: Dataset.num_images = 150, updated]
```

**시나리오 3: 레이블링 완료 (Day 3)**
```
[User]
  → 로컬에서 레이블링
  → 레이블 파일만 업로드
  ↓
POST /api/v1/datasets/{id}/upload (labels_only=True)
  1. R2의 images는 그대로, labels만 업데이트
  2. 메타데이터 재분석
  3. DB 업데이트
     - last_modified_at: NOW()
     - content_hash: 새 해시
  ↓
[학습 시작 가능]
```

### 5.3 학습 시작 (Dataset 사용)

```
[Frontend: TrainingConfigPanel]
  → dataset_id 선택 (e.g., "cls-imagenet-10")
  ↓
POST /api/v1/training/start
  {
    "dataset_id": "cls-imagenet-10",
    "model_name": "resnet18",
    ...
  }
  ↓
[Backend: TrainingManager]
  1. Dataset 조회 (DB)
  2. 권한 확인 (can_access_dataset)
  3. TrainingJob 생성
     - dataset_id: "cls-imagenet-10"
     - dataset_path: null (불필요, R2에서 lazy download)
  ↓
[Training Service]
  1. R2에서 lazy download
     - 로컬 캐시 확인 (/datasets/cls-imagenet-10/)
     - 없으면 R2에서 다운로드
  2. 학습 시작
  ↓
[학습 완료]
  → 결과 저장 (R2: models/{job_id}/)
```

**Lazy Download 최적화**:
```python
# mvp/training/adapters/dataset_handler.py
def prepare_dataset(dataset_id: str) -> Path:
    local_path = Path(f"/datasets/{dataset_id}")

    # 1. 로컬 캐시 확인
    if local_path.exists():
        # content_hash 검증
        if verify_hash(local_path, expected_hash):
            return local_path  # 캐시 히트

    # 2. R2에서 다운로드
    download_from_r2(f"datasets/{dataset_id}/", local_path)

    return local_path
```

---

## 6. API 엔드포인트

### 6.1 구현 완료 (✅)

#### `GET /api/v1/datasets/available`
**설명**: 사용자가 접근 가능한 데이터셋 목록 조회

**권한**: 로그인 사용자 (옵션)

**응답**:
```json
[
  {
    "id": "cls-imagenet-10",
    "name": "ImageNet-10",
    "description": "Platform sample dataset",
    "format": "imagefolder",
    "task_type": "image_classification",
    "num_items": 13000,
    "size_mb": null,
    "source": "r2",
    "path": "cls-imagenet-10"
  },
  ...
]
```

**필터링**:
- `?task_type=image_classification` - Task type 필터
- `?tags=platform-sample` - 태그 필터

#### `POST /api/v1/datasets/analyze`
**설명**: 로컬 폴더 자동 분석

**Request**:
```json
{
  "path": "C:\\datasets\\imagenet-10",
  "format_hint": null  // null이면 auto-detect
}
```

**Response**:
```json
{
  "status": "success",
  "dataset_info": {
    "format": "imagefolder",
    "confidence": 0.98,
    "task_type": "image_classification",
    "structure": {
      "num_classes": 10,
      "num_images": 13000,
      "classes": ["cat", "dog", ...]
    },
    "statistics": {
      "total_images": 13000,
      "source": "local",
      "validated": true
    }
  }
}
```

#### `POST /api/v1/datasets/upload`
**설명**: 데이터셋 업로드 (구현됨, UI 미연동)

**Request**: `multipart/form-data`
```
file: dataset.zip
name: "My Custom Dataset"
task_type: "image_classification"
```

**Response**:
```json
{
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploaded",
  "num_images": 1000,
  "format": "imagefolder"
}
```

### 6.2 필요하지만 미구현 (⬜)

#### `GET /api/v1/datasets/{id}`
**설명**: 특정 데이터셋 상세 조회

**권한**: Owner, Editor, Viewer (또는 public)

**응답**:
```json
{
  "id": "cls-imagenet-10",
  "name": "ImageNet-10",
  "description": "...",
  "owner": {
    "id": 1,
    "email": "admin@example.com"
  },
  "visibility": "public",
  "tags": ["platform-sample", "imagenet"],
  "storage_path": "datasets/cls-imagenet-10/",
  "format": "imagefolder",
  "task_type": "image_classification",
  "num_classes": 10,
  "num_images": 13000,
  "class_names": ["cat", "dog", ...],
  "version": 1,
  "content_hash": "a3f2b8c9...",
  "created_at": "2025-11-01T10:00:00Z",
  "updated_at": "2025-11-03T15:30:00Z"
}
```

#### `PUT /api/v1/datasets/{id}`
**설명**: 메타데이터 수정 (이름, 설명, tags 등)

**권한**: Owner, Editor

**Request**:
```json
{
  "name": "ImageNet-10 (Updated)",
  "description": "New description",
  "tags": ["platform-sample", "imagenet", "tutorial"]
}
```

#### `POST /api/v1/datasets/{id}/upload`
**설명**: 기존 데이터셋에 파일 추가 (점진적 업로드)

**권한**: Owner, Editor

**Request**: `multipart/form-data`
```
files: [img_101.jpg, img_102.jpg, ...]
labels_only: false  // true면 이미지 제외, 레이블만
```

#### `DELETE /api/v1/datasets/{id}`
**설명**: 데이터셋 삭제 (DB + R2)

**권한**: Owner only

**옵션**:
- `?storage=true` - R2 스토리지도 함께 삭제 (기본: false, DB만 삭제)

#### `GET /api/v1/datasets/{id}/permissions`
**설명**: 권한 목록 조회

**권한**: Owner

**응답**:
```json
[
  {
    "user_id": 123,
    "email": "teammate@example.com",
    "role": "editor",
    "granted_by": 1,
    "granted_at": "2025-11-02T10:00:00Z"
  }
]
```

#### `POST /api/v1/datasets/{id}/permissions`
**설명**: 권한 부여

**권한**: Owner

**Request**:
```json
{
  "user_id": 123,
  "role": "editor"  // viewer, editor
}
```

---

## 7. 현재 구현 상태

### 7.1 ✅ 완료된 기능

#### Backend
- ✅ **Dataset DB 모델** (`models.py:60-105`)
  - UUID-based ID
  - visibility (public/private/organization)
  - storage_path, storage_type (r2/s3/gcs)
  - format, task_type, num_classes, num_images, class_names
  - version, content_hash, last_modified_at
  - Relationships: owner, permissions, training_jobs

- ✅ **DatasetPermission 모델** (`models.py:107-126`)
  - Role-based access (owner/editor/viewer)
  - Audit trail (granted_by, granted_at)

- ✅ **R2 통합**
  - Lazy download (training/adapters/dataset_handler.py)
  - 플랫폼 샘플 업로드 스크립트 (scripts/upload_sample_datasets.py)

- ✅ **DatasetAnalyzer** (`app/utils/dataset_analyzer.py`)
  - 형식 자동 감지 (ImageFolder, YOLO, COCO)
  - 클래스 및 샘플 수 카운트
  - meta.json 생성

- ✅ **API 엔드포인트**
  - `GET /api/v1/datasets/available` - 목록 조회
  - `POST /api/v1/datasets/analyze` - 자동 분석
  - `POST /api/v1/datasets/upload` - 업로드

- ✅ **TrainingJob 통합**
  - `dataset_id` 외래키
  - Training API에서 dataset_id로 학습 시작

#### Frontend
- ✅ **AdminDatasetsPanel** (테이블 형태)
  - 데이터셋 목록 조회
  - 검색, 필터링, 정렬
  - Task type별 색상 코딩
  - 통계 표시 (총 데이터셋 수, 이미지 수)

- ✅ **Dataset Types** (`types/dataset.ts`)
  - TypeScript 타입 정의

- ✅ **Sidebar 통합**
  - Admin 계정에 "데이터셋 관리" 버튼

### 7.2 ⚠️ 미완성 기능

#### Priority 1 (긴급 - UX 문제)
- ⬜ **TrainingConfigPanel 데이터셋 선택 UI**
  - 현재: 경로 입력 텍스트 박스
  - 필요: DatasetSourceSelector (플랫폼 샘플 드롭다운 + 로컬 폴더)

#### Priority 2 (단기 - 관리 기능)
- ⬜ `GET /api/v1/datasets/{id}` - 상세 조회
- ⬜ `PUT /api/v1/datasets/{id}` - 메타데이터 수정
- ⬜ `DELETE /api/v1/datasets/{id}` - 삭제
- ⬜ AdminDatasetsPanel 상세 모달

#### Priority 3 (단기 - 업로드 UI)
- ⬜ DatasetUploadModal 컴포넌트
- ⬜ 드래그 앤 드롭 업로드
- ⬜ 진행률 표시
- ⬜ 업로드 후 자동 분석

#### Priority 4 (중기 - 협업)
- ⬜ `POST /api/v1/datasets/{id}/upload` - 점진적 업로드
- ⬜ `GET /api/v1/datasets/{id}/permissions` - 권한 조회
- ⬜ `POST /api/v1/datasets/{id}/permissions` - 권한 부여
- ⬜ 권한 관리 UI

#### Priority 5 (중기 - 버전 관리)
- ⬜ 버전 히스토리 API
- ⬜ 버전 간 diff 비교
- ⬜ 특정 버전으로 롤백

---

## 8. 구현 로드맵

### Phase 1A: UX 긴급 개선 (2일)
**기간**: 2025-11-03 ~ 2025-11-05

**목표**: TrainingConfigPanel에서 데이터셋 선택 직관적으로 만들기

**Tasks**:
1. ✅ AdminDatasetsPanel 테이블 형태로 변경
2. ⬜ DatasetSourceSelector 컴포넌트 생성
3. ⬜ PlatformDatasetTab 구현 (플랫폼 샘플 드롭다운)
4. ⬜ LocalDatasetTab 구현 (경로 입력 + 분석 버튼)
5. ⬜ TrainingConfigPanel에 통합

**Success Criteria**:
- 사용자가 플랫폼 샘플을 드롭다운에서 선택 가능
- 로컬 폴더 선택 시 자동 분석 결과 표시
- 형식/클래스 수/이미지 수 자동 표시

---

### Phase 1B: 관리 기능 보완 (3일)
**기간**: 2025-11-06 ~ 2025-11-08

**Tasks**:
1. ⬜ `GET /api/v1/datasets/{id}` 구현
2. ⬜ `PUT /api/v1/datasets/{id}` 구현
3. ⬜ `DELETE /api/v1/datasets/{id}` 구현
4. ⬜ AdminDatasetsPanel 상세 모달 추가
5. ⬜ 수정/삭제 기능 UI

**Success Criteria**:
- Admin이 데이터셋 메타데이터 수정 가능
- Admin이 데이터셋 삭제 가능 (DB + R2 옵션)
- 상세 정보 모달에서 모든 메타데이터 확인

---

### Phase 2: 업로드 및 점진적 데이터 구축 (7일)
**기간**: 2025-11-09 ~ 2025-11-15

**Tasks**:
1. ⬜ DatasetUploadModal 컴포넌트
2. ⬜ 드래그 앤 드롭 업로드
3. ⬜ 진행률 표시 (WebSocket)
4. ⬜ 압축 파일 지원 (.zip, .tar.gz)
5. ⬜ `POST /api/v1/datasets/{id}/upload` (점진적 업로드)
6. ⬜ 업로드 후 자동 분석 및 메타데이터 갱신

**Success Criteria**:
- 사용자가 zip 파일 업로드 가능
- 업로드된 데이터셋이 자동 분석되어 DB에 저장
- 기존 데이터셋에 파일 추가 가능 (Day 1: 100장, Day 2: +50장)

---

### Phase 3: 협업 및 권한 관리 (5일)
**기간**: 2025-11-16 ~ 2025-11-20

**Tasks**:
1. ⬜ `GET /api/v1/datasets/{id}/permissions` 구현
2. ⬜ `POST /api/v1/datasets/{id}/permissions` 구현
3. ⬜ 권한 관리 UI (모달)
4. ⬜ 팀원 초대 기능
5. ⬜ 권한별 UI 조건부 렌더링

**Success Criteria**:
- Dataset owner가 팀원에게 viewer/editor 권한 부여 가능
- 팀원이 공유된 데이터셋으로 학습 가능
- Editor는 데이터 추가 가능, Viewer는 읽기만 가능

---

### Phase 4: 버전 관리 및 고급 기능 (7일)
**기간**: 2025-11-21 ~ 2025-11-27

**Tasks**:
1. ⬜ 버전 히스토리 API
2. ⬜ 버전 간 diff 비교
3. ⬜ 특정 버전으로 롤백
4. ⬜ content_hash 자동 검증
5. ⬜ 중복 이미지 감지 (perceptual hashing)

**Success Criteria**:
- 데이터셋 변경 히스토리 추적 가능
- 이전 버전으로 롤백 가능
- 재현성 보장 (content_hash)

---

## 9. 기술 부채 및 개선 사항

### 9.1 현재 문제점

1. **TrainingConfigPanel의 dataset_path 입력**
   - 문제: 사용자가 절대 경로를 직접 입력해야 함
   - 영향: UX 저하, R2 기반 시스템과 불일치
   - 해결: DatasetSourceSelector 구현 (Priority 1)

2. **업로드 API는 있지만 UI 미연동**
   - 문제: Backend는 구현되었지만 Frontend에서 사용 불가
   - 해결: DatasetUploadModal 구현 (Priority 3)

3. **점진적 업로드 미구현**
   - 문제: 사용자가 데이터를 나눠서 업로드할 수 없음
   - 영향: Day 1: 100장, Day 2: +50장 시나리오 불가능
   - 해결: `POST /datasets/{id}/upload` 구현

4. **권한 관리 UI 없음**
   - 문제: DB에 DatasetPermission이 있지만 UI 없음
   - 영향: 팀 협업 불가능
   - 해결: 권한 관리 UI 구현 (Priority 4)

5. **버전 관리 미완성**
   - 문제: content_hash는 있지만 히스토리 추적 안 됨
   - 영향: 재현성 검증 어려움
   - 해결: 버전 히스토리 API (Priority 5)

### 9.2 아키텍처 개선 필요 사항

1. **Dataset 조회 성능 최적화**
   - 현재: 매번 DB 쿼리
   - 개선: Redis 캐싱 (특히 `GET /datasets/available`)

2. **R2 lazy download 최적화**
   - 현재: 학습 시작 시마다 다운로드 체크
   - 개선: 로컬 캐시 확인, 병렬 다운로드

3. **content_hash 자동 계산**
   - 현재: 업로드 시 한 번만 계산
   - 개선: 정기적으로 재계산 (무결성 검증)

4. **Dataset 삭제 시 스토리지 정리**
   - 현재: DB에서만 삭제
   - 개선: R2 객체도 함께 삭제 (옵션)

---

## 10. 참고 자료

### 10.1 주요 커밋
- `c175512` - Dataset as first-class entity (핵심 설계)
- `eb4e5a9` - R2-based dataset lazy download system
- `23ffde2` - R2 sample dataset analysis support
- `1bcf504` - Migrate API from hardcoded to DB-based
- `09e9379` - Training API integration with dataset_id

### 10.2 구현 파일
**Backend**:
- `mvp/backend/app/db/models.py:60-126` - Dataset, DatasetPermission 모델
- `mvp/backend/app/api/datasets.py` - Dataset API
- `mvp/backend/app/utils/dataset_analyzer.py` - 자동 분석
- `mvp/training/adapters/dataset_handler.py` - R2 lazy download
- `scripts/upload_sample_datasets.py` - 플랫폼 샘플 업로드

**Frontend**:
- `mvp/frontend/types/dataset.ts` - TypeScript 타입
- `mvp/frontend/components/AdminDatasetsPanel.tsx` - Admin 테이블
- `mvp/frontend/components/TrainingConfigPanel.tsx` - (개선 필요)

---

## 11. 결론

### 11.1 핵심 철학

> **"플랫폼 샘플이든 사용자 커스텀이든, 모두 동일한 Dataset 엔티티로 취급한다."**

- ImageNet, COCO는 특별하지 않음
- 단지 `visibility='public'` + `tags=['platform-sample']`일 뿐
- 모든 데이터셋은 R2 + DB 메타데이터로 관리

### 11.2 현재 상황

**완료**:
- R2 기반 스토리지 인프라 구축
- Dataset/DatasetPermission DB 모델
- 플랫폼 샘플 업로드 및 조회
- 자동 분석 API

**미완성 (긴급)**:
- TrainingConfigPanel UI 개선
- 업로드 UI
- 점진적 업로드 (Day 1, 2, 3 시나리오)
- 권한 관리 UI

### 11.3 다음 단계

**즉시 착수** (Priority 1):
1. DatasetSourceSelector 구현
2. 플랫폼 샘플 드롭다운
3. TrainingConfigPanel 통합

**이번 주 내** (Priority 2-3):
4. 관리 API (GET/PUT/DELETE)
5. AdminDatasetsPanel 상세 모달
6. DatasetUploadModal

**다음 주** (Priority 4-5):
7. 점진적 업로드
8. 권한 관리
9. 버전 관리

---

**Last Updated**: 2025-11-03
**Author**: Development Team
**Status**: Living Document (계속 업데이트)
