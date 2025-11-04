# Dataset Management Implementation Plan

**Date**: 2025-01-04
**Status**: Ready to Implement
**Based on**: [DATASET_MANAGEMENT_DESIGN.md](./DATASET_MANAGEMENT_DESIGN.md)

이 문서는 데이터셋 관리 시스템의 구체적인 구현 계획입니다.

---

## 현재 상태 분석

### ✅ 이미 구현된 기능
- 개별 이미지 업로드 API (POST `/datasets/{id}/images`)
- 이미지 목록 조회 API (GET `/datasets/{id}/images`)
- Presigned URL 생성
- R2 스토리지 연동
- 기본 Dataset DB 모델
- DatasetPanel UI (컴팩트 테이블)

### ⚠️ 설계와 불일치
- Dataset.task_type 존재 (제거 필요)
- labeled 필드 없음
- 스냅샷 관련 필드 없음
- TrainingJob.dataset_snapshot_id 타입 불일치

---

## Phase 1: DB 마이그레이션

### 목표
설계 문서에 맞게 DB 모델 업데이트

### Dataset 모델 변경사항

#### 제거할 필드
```python
task_type = Column(String(50), nullable=False)  # ❌ 제거
```

#### 추가할 필드
```python
# Labeled 상태
labeled = Column(Boolean, nullable=False, default=False, index=True)
annotation_path = Column(String(500), nullable=True)  # "datasets/{id}/annotation.json"

# 스냅샷 관련
is_snapshot = Column(Boolean, nullable=False, default=False, index=True)
parent_dataset_id = Column(String(100), ForeignKey('datasets.id'), nullable=True, index=True)
snapshot_created_at = Column(DateTime, nullable=True)
version_tag = Column(String(50), nullable=True)  # "v1", "v2", etc.

# 무결성 관리
status = Column(String(20), nullable=False, default='valid', index=True)  # valid, broken, repairing
integrity_status = Column(JSON, nullable=True)  # 무결성 체크 결과
```

#### 수정할 필드
```python
# 기존
num_classes = Column(Integer, nullable=True)

# 변경: labeled인 경우에만 값 존재
num_classes = Column(Integer, nullable=True)  # labeled=True인 경우만
```

### TrainingJob 모델 변경사항

#### 수정할 필드
```python
# 기존
dataset_version = Column(Integer, nullable=True)

# 변경
dataset_snapshot_id = Column(String(150), nullable=True, index=True)  # "dataset-A@snapshot-20250104-123045"
snapshot_status_at_start = Column(String(20), nullable=True)  # 학습 시작 시 스냅샷 상태
```

### 마이그레이션 스크립트

```python
# mvp/backend/migrate_dataset_redesign.py
"""
Dataset 모델 재설계 마이그레이션

변경사항:
1. Dataset.task_type 제거
2. labeled, annotation_path 추가
3. 스냅샷 관련 필드 추가
4. 무결성 관리 필드 추가
5. TrainingJob.dataset_snapshot_id 타입 변경
"""

from sqlalchemy import create_engine, text
from app.core.config import settings

def migrate():
    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        # 1. Dataset 테이블 변경
        conn.execute(text("""
            ALTER TABLE datasets
            DROP COLUMN task_type
        """))

        conn.execute(text("""
            ALTER TABLE datasets
            ADD COLUMN labeled BOOLEAN NOT NULL DEFAULT FALSE,
            ADD COLUMN annotation_path VARCHAR(500),
            ADD COLUMN is_snapshot BOOLEAN NOT NULL DEFAULT FALSE,
            ADD COLUMN parent_dataset_id VARCHAR(100),
            ADD COLUMN snapshot_created_at TIMESTAMP,
            ADD COLUMN version_tag VARCHAR(50),
            ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'valid',
            ADD COLUMN integrity_status JSON
        """))

        # 인덱스 추가
        conn.execute(text("""
            CREATE INDEX idx_datasets_labeled ON datasets(labeled);
            CREATE INDEX idx_datasets_is_snapshot ON datasets(is_snapshot);
            CREATE INDEX idx_datasets_parent_id ON datasets(parent_dataset_id);
            CREATE INDEX idx_datasets_status ON datasets(status);
        """))

        # 외래 키 추가
        conn.execute(text("""
            ALTER TABLE datasets
            ADD CONSTRAINT fk_datasets_parent
            FOREIGN KEY (parent_dataset_id) REFERENCES datasets(id);
        """))

        # 2. TrainingJob 테이블 변경
        conn.execute(text("""
            ALTER TABLE training_jobs
            DROP COLUMN dataset_version
        """))

        conn.execute(text("""
            ALTER TABLE training_jobs
            ADD COLUMN dataset_snapshot_id VARCHAR(150),
            ADD COLUMN snapshot_status_at_start VARCHAR(20)
        """))

        conn.execute(text("""
            CREATE INDEX idx_training_jobs_snapshot ON training_jobs(dataset_snapshot_id);
        """))

        conn.commit()
        print("✅ Migration completed successfully")

if __name__ == "__main__":
    migrate()
```

---

## Phase 2: 폴더 업로드 (Backend)

### 2.1. API 엔드포인트

#### POST `/datasets/upload-folder`
```python
# mvp/backend/app/api/datasets.py

@router.post("/upload-folder", response_model=DatasetUploadResponse)
async def upload_dataset_folder(
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(...),
    description: str = Form(""),
    visibility: str = Form("private"),
    has_annotation: bool = Form(False),
    db: Session = Depends(get_db)
):
    """
    폴더 구조 유지하며 데이터셋 업로드

    Parameters:
    - files: 업로드할 파일 리스트 (폴더 구조 포함)
    - dataset_name: 데이터셋 이름
    - description: 설명
    - visibility: 공개 범위
    - has_annotation: annotation.json 포함 여부
    """

    # 1. 데이터셋 ID 생성
    dataset_id = str(uuid.uuid4())

    # 2. 파일 분류
    image_files = []
    annotation_file = None
    meta_file = None

    for file in files:
        if file.filename.endswith('annotation.json'):
            annotation_file = file
        elif file.filename.endswith('meta.json'):
            meta_file = file
        elif is_image_file(file.filename):
            image_files.append(file)

    # 3. 폴더 구조 분석
    folder_structure = analyze_folder_structure([f.filename for f in image_files])

    # 4. R2에 이미지 업로드 (구조 유지)
    for img_file in image_files:
        # 상대 경로 추출
        relative_path = extract_relative_path(img_file.filename)

        # R2: datasets/{id}/images/{relative_path}
        r2_path = f"datasets/{dataset_id}/images/{relative_path}"

        content = await img_file.read()
        r2_storage.upload_fileobj(
            BytesIO(content),
            r2_path,
            content_type=img_file.content_type
        )

    # 5. annotation.json 처리 (있는 경우)
    labeled = False
    num_classes = None
    class_names = []

    if has_annotation and annotation_file:
        annotation_content = await annotation_file.read()
        annotation_data = json.loads(annotation_content)

        # R2에 저장
        r2_storage.upload_fileobj(
            BytesIO(annotation_content),
            f"datasets/{dataset_id}/annotation.json",
            content_type="application/json"
        )

        # meta.json 생성
        meta_data = generate_meta_json(dataset_name, annotation_data)
        r2_storage.upload_fileobj(
            BytesIO(json.dumps(meta_data).encode()),
            f"datasets/{dataset_id}/meta.json",
            content_type="application/json"
        )

        labeled = True
        num_classes = len(annotation_data.get('categories', []))
        class_names = [c['name'] for c in annotation_data.get('categories', [])]

    # 6. DB 저장
    dataset = Dataset(
        id=dataset_id,
        name=dataset_name,
        description=description,
        owner_id=None,  # TODO: 인증 추가
        visibility=visibility,
        storage_path=f"datasets/{dataset_id}/",
        storage_type="r2",
        format="dice" if labeled else "unlabeled",
        labeled=labeled,
        annotation_path=f"datasets/{dataset_id}/annotation.json" if labeled else None,
        num_images=len(image_files),
        num_classes=num_classes,
        class_names=class_names,
        is_snapshot=False,
        status="valid"
    )

    db.add(dataset)
    db.commit()

    return DatasetUploadResponse(
        status="success",
        dataset_id=dataset_id,
        message=f"Dataset '{dataset_name}' created with {len(image_files)} images",
        metadata={
            "labeled": labeled,
            "num_images": len(image_files),
            "num_classes": num_classes
        }
    )
```

### 2.2. 헬퍼 함수

```python
def analyze_folder_structure(filenames: List[str]) -> dict:
    """폴더 구조 분석"""
    has_subdirs = any('/' in f for f in filenames)

    if has_subdirs:
        # 서브디렉토리 존재 → 구조 분석
        dirs = set()
        for f in filenames:
            if '/' in f:
                dir_name = f.split('/')[0]
                dirs.add(dir_name)

        return {
            "has_structure": True,
            "directories": list(dirs),
            "depth": max(f.count('/') for f in filenames) + 1
        }
    else:
        return {
            "has_structure": False,
            "directories": [],
            "depth": 1
        }

def extract_relative_path(filename: str) -> str:
    """파일명에서 상대 경로 추출"""
    # 브라우저에서 전송된 경로 처리
    # 예: "dataset/cats/cat1.jpg" → "cats/cat1.jpg"
    parts = filename.split('/')
    if len(parts) > 1:
        # 첫 번째 폴더(dataset)는 제외
        return '/'.join(parts[1:])
    return filename

def is_image_file(filename: str) -> bool:
    """이미지 파일 여부 확인"""
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))

def generate_meta_json(dataset_name: str, annotation_data: dict) -> dict:
    """meta.json 생성"""
    return {
        "dataset_id": annotation_data.get('dataset_id'),
        "dataset_name": dataset_name,
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "num_images": len(annotation_data.get('images', [])),
        "num_classes": len(annotation_data.get('categories', [])),
        "labeled": True,
        "platform": "vision-ai-platform",
        "format_version": "dice-2.0"
    }
```

---

## Phase 3: 폴더 업로드 (Frontend)

### 3.1. DatasetUploadModal 수정

```tsx
// mvp/frontend/components/datasets/DatasetUploadModal.tsx

export default function DatasetUploadModal({ isOpen, onClose, onUploadSuccess }: Props) {
  const [uploadMode, setUploadMode] = useState<'zip' | 'folder'>('folder')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [datasetName, setDatasetName] = useState('')
  const [description, setDescription] = useState('')
  const [hasAnnotation, setHasAnnotation] = useState(false)

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    setSelectedFiles(files)

    // annotation.json 자동 감지
    const hasAnnotationFile = files.some(f => f.name === 'annotation.json')
    setHasAnnotation(hasAnnotationFile)
  }

  const handleUpload = async () => {
    const formData = new FormData()

    // 파일 추가
    selectedFiles.forEach(file => {
      formData.append('files', file, file.webkitRelativePath || file.name)
    })

    // 메타데이터
    formData.append('dataset_name', datasetName)
    formData.append('description', description)
    formData.append('visibility', 'private')
    formData.append('has_annotation', hasAnnotation.toString())

    const response = await fetch(`${API_URL}/datasets/upload-folder`, {
      method: 'POST',
      body: formData
    })

    // 처리...
  }

  return (
    <Dialog open={isOpen} onClose={onClose}>
      {/* 업로드 모드 선택 */}
      <RadioGroup value={uploadMode} onChange={setUploadMode}>
        <Radio value="folder">폴더 업로드 (권장)</Radio>
        <Radio value="zip">ZIP 파일 (레거시)</Radio>
      </RadioGroup>

      {uploadMode === 'folder' && (
        <>
          {/* 데이터셋 정보 */}
          <Input
            label="데이터셋 이름"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
          />

          <Textarea
            label="설명"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />

          {/* 폴더 선택 */}
          <input
            type="file"
            webkitdirectory=""
            directory=""
            multiple
            onChange={handleFolderSelect}
          />

          {/* 선택된 파일 미리보기 */}
          {selectedFiles.length > 0 && (
            <div>
              <p>선택된 파일: {selectedFiles.length}개</p>

              {hasAnnotation && (
                <Alert variant="info">
                  annotation.json 감지됨 - labeled 데이터셋으로 생성됩니다
                </Alert>
              )}

              {/* 폴더 구조 미리보기 */}
              <FolderPreview files={selectedFiles} />
            </div>
          )}
        </>
      )}

      <Button onClick={handleUpload} disabled={!datasetName || selectedFiles.length === 0}>
        업로드
      </Button>
    </Dialog>
  )
}
```

### 3.2. 폴더 구조 미리보기

```tsx
function FolderPreview({ files }: { files: File[] }) {
  const structure = useMemo(() => {
    const tree: any = {}

    files.forEach(file => {
      const path = file.webkitRelativePath || file.name
      const parts = path.split('/')

      let current = tree
      parts.forEach((part, i) => {
        if (i === parts.length - 1) {
          // 파일
          if (!current.files) current.files = []
          current.files.push(part)
        } else {
          // 폴더
          if (!current[part]) current[part] = {}
          current = current[part]
        }
      })
    })

    return tree
  }, [files])

  return (
    <div className="bg-gray-50 p-4 rounded text-xs font-mono">
      <p className="font-bold mb-2">폴더 구조:</p>
      <TreeView tree={structure} />
    </div>
  )
}
```

---

## Phase 4: 버전닝 (스냅샷)

### 4.1. 스냅샷 생성 API

```python
# mvp/backend/app/api/datasets.py

@router.post("/{dataset_id}/snapshots", response_model=SnapshotResponse)
async def create_snapshot(
    dataset_id: str,
    version_tag: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """데이터셋 스냅샷 생성"""

    # 1. 원본 데이터셋 확인
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    if not dataset.labeled:
        raise HTTPException(400, "Can only snapshot labeled datasets")

    # 2. 스냅샷 ID 생성
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_id = f"{dataset_id}@snapshot-{timestamp}"

    if version_tag:
        snapshot_id = f"{dataset_id}@{version_tag}"

    # 3. annotation.json 복사
    source_annotation = f"datasets/{dataset_id}/annotation.json"
    target_annotation = f"datasets/{dataset_id}/snapshots/{snapshot_id}/annotation.json"
    r2_storage.copy(source_annotation, target_annotation)

    # 4. meta.json 복사
    source_meta = f"datasets/{dataset_id}/meta.json"
    target_meta = f"datasets/{dataset_id}/snapshots/{snapshot_id}/meta.json"
    r2_storage.copy(source_meta, target_meta)

    # 5. DB에 스냅샷 기록
    snapshot = Dataset(
        id=snapshot_id,
        name=f"{dataset.name} ({version_tag or 'snapshot'})",
        description=dataset.description,
        owner_id=dataset.owner_id,
        visibility=dataset.visibility,
        storage_path=dataset.storage_path,  # 이미지는 공유
        storage_type=dataset.storage_type,
        format=dataset.format,
        labeled=True,
        annotation_path=target_annotation,
        num_images=dataset.num_images,
        num_classes=dataset.num_classes,
        class_names=dataset.class_names,
        is_snapshot=True,
        parent_dataset_id=dataset_id,
        snapshot_created_at=datetime.utcnow(),
        version_tag=version_tag,
        status="valid"
    )

    db.add(snapshot)
    db.commit()

    return SnapshotResponse(
        snapshot_id=snapshot_id,
        created_at=snapshot.snapshot_created_at,
        version_tag=version_tag
    )
```

### 4.2. 학습 시작 시 자동 스냅샷

```python
# mvp/backend/app/api/training.py

@router.post("/start", response_model=TrainingJobResponse)
async def start_training(
    config: TrainingConfig,
    db: Session = Depends(get_db)
):
    # 1. 데이터셋 스냅샷 생성
    snapshot_response = await create_snapshot(
        dataset_id=config.dataset_id,
        version_tag=None,  # 자동 생성
        db=db
    )

    # 2. 학습 작업 생성
    job = TrainingJob(
        dataset_id=config.dataset_id,
        dataset_snapshot_id=snapshot_response.snapshot_id,
        snapshot_status_at_start="valid",
        task_type=config.task_type,
        # ... 기타 설정
    )

    db.add(job)
    db.commit()

    # 3. 학습 시작
    # ...
```

---

## 구현 우선순위

### Sprint 1 (1-2일)
- [ ] DB 마이그레이션 스크립트 작성
- [ ] 마이그레이션 실행 및 검증
- [ ] models.py 업데이트

### Sprint 2 (2-3일)
- [ ] 폴더 업로드 API 구현
- [ ] 폴더 구조 분석 로직
- [ ] annotation.json 처리

### Sprint 3 (2-3일)
- [ ] 폴더 업로드 UI 구현
- [ ] 폴더 구조 미리보기
- [ ] 테스트 및 버그 수정

### Sprint 4 (1-2일)
- [ ] 스냅샷 생성 API
- [ ] 학습 시 자동 스냅샷
- [ ] 스냅샷 목록 UI

---

## 다음 단계

1. Phase 1 (DB 마이그레이션) 시작
2. 마이그레이션 완료 후 Phase 2 진행
3. 각 Phase 완료 시 `/log-session` 실행하여 진행 상황 기록
