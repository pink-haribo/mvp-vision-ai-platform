# Dataset Management - Current Status

**Last Updated**: 2025-11-04
**Branch**: `feature/dataset-entity` (Merged to `main` via PR #12)
**Next Branch**: TBD (학습 파이프라인 테스트 or 스냅샷 구현)

---

## ✅ 완료된 기능 (Phase 1 & 2)

### 1. Core Infrastructure
- [x] **Dataset Entity** - DB 기반 데이터셋 관리
  - `mvp/backend/app/db/models.py` - Dataset, DatasetPermission 모델
  - UUID 기반 ID, visibility, owner_id, labeled, format 등
  - 스냅샷 관련 필드 준비됨 (is_snapshot, parent_dataset_id, version_tag)

- [x] **R2 Cloud Storage**
  - `mvp/backend/app/utils/r2_storage.py` - R2Storage 유틸리티
  - Presigned URLs 생성 (이미지 다운로드용)
  - 폴더 구조 유지 업로드
  - 업로드/다운로드/삭제 기능

### 2. Backend API
- [x] **Dataset CRUD**
  - `POST /datasets` - 빈 데이터셋 생성
  - `GET /datasets/available` - 데이터셋 목록 (소유자 + public)
  - `DELETE /datasets/{id}` - 데이터셋 삭제 (owner only)

- [x] **Image Management**
  - `POST /datasets/{id}/images` - 개별 이미지 업로드
  - `GET /datasets/{id}/images` - 이미지 목록 조회
  - `GET /datasets/{id}/images/{filename}/url` - Presigned URL 생성

- [x] **Folder Upload**
  - `POST /datasets/{id}/upload-images` - 폴더 업로드
  - webkitRelativePath 지원
  - annotation.json 자동 감지 및 처리
  - 폴더 구조 유지 (R2에 상대 경로로 저장)

- [x] **Authentication & Authorization**
  - JWT 기반 인증 (get_current_user)
  - Owner-based permissions
  - Public datasets 조회 가능
  - Private datasets owner만 접근

### 3. Frontend Components
- [x] **DatasetPanel** (`mvp/frontend/components/DatasetPanel.tsx`)
  - 테이블 형식 데이터셋 목록
  - 검색, 정렬 기능
  - 이미지 보기 (확장 행)
  - 삭제 기능

- [x] **CreateDatasetModal** (`mvp/frontend/components/datasets/CreateDatasetModal.tsx`)
  - 데이터셋 메타데이터 생성
  - Visibility 선택 (private/public)
  - 생성 후 모달 닫기 (페이지 전환 없음)

- [x] **DatasetImageUpload** (`mvp/frontend/components/datasets/DatasetImageUpload.tsx`)
  - 폴더 업로드 UI
  - 진행 상태 표시
  - annotation.json 자동 감지

- [x] **DatasetImageGallery** (`mvp/frontend/components/datasets/DatasetImageGallery.tsx`)
  - 이미지 그리드 표시
  - Presigned URL 기반 이미지 로딩

- [x] **Sidebar Integration**
  - "데이터셋" 메뉴 (인증된 사용자만)
  - 인증 상태에 따라 표시/숨김

### 4. DICE Format v2.0
- [x] **Format Specification**
  - `docs/datasets/PLATFORM_DATASET_FORMAT.md`
  - annotations.json 기반 단일 파일 포맷
  - 상대 경로 사용 (Cloud 호환)
  - Task별 스키마 정의
  - Migration info 포함

- [x] **Conversion Tools**
  - `mvp/backend/convert_yolo_seg_to_platform.py`
  - YOLO segmentation → DICE Format 변환
  - 테스트 데이터셋 준비: `C:\datasets\dice_format\seg-coco32`

### 5. Training Integration
- [x] **dataset_id Support**
  - TrainingJob.dataset_id (nullable)
  - TrainingJob.dataset_snapshot_id (nullable, 향후 사용)
  - Action handlers 업데이트 (start_training)
  - Backend → Training Service API 연동

---

## ⏳ 남은 작업 (Phase 3 & 4)

### Sprint 1: 버전닝 및 스냅샷 (중요도: 🔥 High)
**예상 소요**: 2-3일

#### Backend API
- [ ] **스냅샷 생성 API**
  ```python
  POST /datasets/{dataset_id}/snapshots
  - annotation.json과 meta.json만 복사
  - 이미지는 공유 (storage efficiency)
  - 자동 snapshot ID 생성: {dataset_id}@snapshot-{timestamp}
  ```

- [ ] **스냅샷 목록 API**
  ```python
  GET /datasets/{dataset_id}/snapshots
  - 모든 스냅샷 반환 (version_tag, created_at 등)
  ```

- [ ] **학습 시작 시 자동 스냅샷**
  ```python
  # mvp/backend/app/api/training.py
  @router.post("/start")
  async def start_training(...):
      # 1. 스냅샷 생성
      snapshot_id = await create_snapshot(dataset_id)

      # 2. TrainingJob에 snapshot_id 기록
      job.dataset_snapshot_id = snapshot_id
  ```

#### Frontend UI
- [ ] **버전 관리 섹션**
  - 데이터셋 상세 페이지에 "Versions" 탭
  - 버전 목록 (HEAD, v1, v2, snapshot-xxx)
  - "Create Version" 버튼

- [ ] **스냅샷 비교 뷰**
  - 버전 간 차이 표시
  - 클래스 수, 이미지 수 변화

#### Files to Modify
```
mvp/backend/app/api/datasets.py         # 스냅샷 API 추가
mvp/backend/app/api/training.py         # 자동 스냅샷 로직
mvp/frontend/app/datasets/[id]/page.tsx # 버전 탭 추가
mvp/frontend/components/datasets/       # 버전 관리 컴포넌트
```

### Sprint 2: UI/UX 개선 (중요도: Medium)
**예상 소요**: 1-2일

- [ ] **폴더 구조 미리보기**
  - 업로드 전 폴더 트리 표시
  - 이미지 파일 개수 표시
  - annotation.json 감지 표시

- [ ] **업로드 진행률 개선**
  - 실시간 진행률 (50/100 files)
  - 업로드 속도 표시
  - 실패한 파일 목록

- [ ] **에러 핸들링 강화**
  - 네트워크 오류 재시도
  - 부분 업로드 복구
  - 상세한 에러 메시지

### Sprint 3: 무결성 관리 (중요도: Low)
**예상 소요**: 2-3일

- [ ] **이미지 삭제 시 영향 분석**
  - 어떤 스냅샷이 영향받는지 확인
  - 사용자에게 확인 요청

- [ ] **스냅샷 무결성 체크**
  - Celery task로 주기적 검증
  - 누락된 이미지 감지
  - Broken 스냅샷 표시

- [ ] **복구 기능**
  - Broken 스냅샷 복구 API
  - annotation.json에서 누락 이미지 제거

---

## 📂 테스트 데이터셋

### 1. seg-coco32 (Instance Segmentation)
**위치**: `C:\datasets\dice_format\seg-coco32`

**구조**:
```
seg-coco32/
├── annotations.json    # DICE Format v1.0
└── images/             # 32 images
    ├── 000000000009.jpg
    ├── 000000000025.jpg
    └── ...
```

**메타데이터**:
- Format: DICE v1.0 (instance_segmentation)
- Images: 32장
- Annotations: 209개 polygon segmentations
- Classes: 43개 COCO 클래스
- Top classes: person (56), car (19), cup (15)

**사용법**:
```python
# 1. Frontend에서 데이터셋 생성
POST /datasets
{
  "name": "COCO Seg 32",
  "description": "Test dataset for segmentation",
  "visibility": "private"
}

# 2. 폴더 업로드
POST /datasets/{id}/upload-images
# C:\datasets\dice_format\seg-coco32 폴더 선택

# 3. 학습 시작
POST /training/start
{
  "dataset_id": "{id}",
  "task_type": "instance_segmentation",
  "model_name": "yolo11n-seg",
  ...
}
```

---

## 🎯 다음 세션 시작 가이드

### Option A: 학습 파이프라인 테스트 (추천 ✅)
**목표**: Backend ↔ Training Service 통신 검증

**브랜치**: `feature/training-pipeline-test`

**작업 순서**:
1. seg-coco32 데이터셋 업로드
2. Training API 호출 테스트
3. 학습 시작/중지/모니터링 확인
4. MLflow 연동 확인
5. 에러 핸들링 검증

**이유**:
- 학습이 제대로 돌아가는지 먼저 확인
- 스냅샷은 학습 파이프라인이 검증된 후 추가해도 늦지 않음
- 데이터셋 버전 없이도 학습 가능 (dataset_snapshot_id는 nullable)

### Option B: 스냅샷 구현
**목표**: 데이터셋 버전 관리 완성

**브랜치**: `feature/dataset-snapshots`

**작업 순서**:
1. 스냅샷 생성 API 구현
2. 학습 시작 시 자동 스냅샷
3. 스냅샷 목록 UI
4. 버전 비교 뷰

---

## 🔍 중요 파일 경로

### Backend
```
mvp/backend/app/
├── db/
│   └── models.py                    # Dataset, DatasetPermission 모델
├── api/
│   ├── datasets.py                  # Dataset CRUD API
│   ├── datasets_folder.py           # Folder upload API
│   ├── datasets_images.py           # Image management API
│   └── training.py                  # Training API (dataset_id 지원)
├── utils/
│   ├── r2_storage.py                # R2Storage 유틸리티
│   └── dependencies.py              # get_current_user
└── schemas/
    └── training.py                  # TrainingConfig 스키마
```

### Frontend
```
mvp/frontend/
├── app/
│   └── datasets/
│       ├── page.tsx                 # 데이터셋 목록 페이지
│       └── [id]/
│           └── page.tsx             # 데이터셋 상세 페이지
└── components/
    ├── DatasetPanel.tsx             # 메인 데이터셋 패널
    ├── Sidebar.tsx                  # 사이드바 (데이터셋 메뉴)
    └── datasets/
        ├── CreateDatasetModal.tsx   # 생성 모달
        ├── DatasetImageUpload.tsx   # 업로드
        └── DatasetImageGallery.tsx  # 갤러리
```

### Documentation
```
docs/datasets/
├── CURRENT_STATUS.md                # 이 문서
├── DATASET_MANAGEMENT_DESIGN.md     # 설계 문서
├── IMPLEMENTATION_PLAN.md           # 구현 계획
├── PLATFORM_DATASET_FORMAT.md       # DICE Format 스펙
└── README.md                        # 개요
```

---

## 📝 메모 및 주의사항

### 1. 스냅샷 vs 학습 파이프라인
- **현재 결정**: 학습 파이프라인 먼저 테스트
- **이유**:
  - 스냅샷 없이도 학습 가능 (dataset_id만으로 충분)
  - 학습이 제대로 돌아가야 스냅샷 기능도 의미 있음
  - DB 모델은 이미 준비되어 있음 (dataset_snapshot_id)

### 2. DICE Format 변환
- YOLO → DICE 변환 스크립트: `mvp/backend/convert_yolo_seg_to_platform.py`
- 다른 포맷 변환 시 참고 가능
- 변환 스크립트는 .gitignore에 포함 (일회성 도구)

### 3. R2 Storage
- Presigned URL 만료 시간: 기본 1시간 (조정 가능)
- 이미지는 `datasets/{id}/images/` 경로에 저장
- annotation.json은 `datasets/{id}/annotation.json`

### 4. Authentication
- 모든 dataset API는 인증 필요
- Public dataset은 조회만 가능 (수정/삭제 불가)
- Owner만 업로드/삭제 가능

### 5. 다음 PR 준비
- 브랜치: main에서 새 브랜치 생성
- PR 제목: `feat(training): implement training pipeline test` or `feat(datasets): implement snapshot versioning`
- 작은 단위로 커밋 (기능별로 나누기)

---

## 🚀 Quick Start (다음 세션)

### 학습 파이프라인 테스트 시작하기

```bash
# 1. 새 브랜치 생성
git checkout main
git pull
git checkout -b feature/training-pipeline-test

# 2. Backend 서버 시작
cd mvp/backend
source venv/bin/activate  # or .\venv\Scripts\activate (Windows)
uvicorn app.main:app --reload --port 8000

# 3. Frontend 서버 시작 (다른 터미널)
cd mvp/frontend
npm run dev

# 4. 데이터셋 업로드
# - 웹 UI에서 http://localhost:3000
# - 로그인
# - "데이터셋" 메뉴 → "Create Dataset"
# - 폴더 업로드: C:\datasets\dice_format\seg-coco32

# 5. 학습 시작 테스트
# - 채팅에서 "seg-coco32 데이터셋으로 yolo11n-seg 모델 학습시작"
# - 학습 상태 모니터링
# - MLflow 확인: http://localhost:5001
```

---

**End of Document**
