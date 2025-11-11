# Dataset Management Design

**Date**: 2025-01-04
**Status**: Design Approved

이 문서는 Vision AI Platform의 데이터셋 관리 시스템 설계를 정리합니다.

## 목차
- [핵심 원칙](#핵심-원칙)
- [데이터 모델](#데이터-모델)
- [스토리지 구조](#스토리지-구조)
- [업로드 시나리오](#업로드-시나리오)
- [버전닝 전략](#버전닝-전략)
- [무결성 관리](#무결성-관리)

---

## 핵심 원칙

### 1. task_type은 데이터셋 속성이 아니다

**잘못된 개념**:
```python
# ❌ 데이터셋이 task_type을 가짐
class Dataset:
    task_type: str  # classification, detection, segmentation
```

**올바른 개념**:
```python
# ✅ 데이터셋: 이미지 저장소
class Dataset:
    id: str
    name: str
    num_images: int
    labeled: bool

# ✅ 학습 작업: task_type 지정
class TrainingJob:
    dataset_id: str
    task_type: str  # classification, detection, segmentation
```

**이유**: 같은 이미지들을 가지고 classification, detection, segmentation 등 다양하게 활용 가능

---

### 2. 폴더 구조는 항상 유지

**원본**:
```
my_dataset/
  ├── cats/cat1.jpg
  ├── dogs/dog1.jpg
  └── subdir/bird1.jpg
```

**R2 저장**:
```
datasets/{id}/images/cats/cat1.jpg
datasets/{id}/images/dogs/dog1.jpg
datasets/{id}/images/subdir/bird1.jpg
```

**이유**:
- ✅ 원본 구조 보존
- ✅ 파일명 충돌 방지
- ✅ 유연성 (필요시 평면화는 언제든 가능)

---

### 3. labeled의 정의

```
labeled = 우리 플랫폼의 annotation.json 존재 여부
```

**labeled 업로드**:
- ✅ 폴더 업로드 + annotation.json 포함
- ❌ 개별 파일 업로드 (논리적으로 불가능)

**unlabeled 업로드**:
- ✅ 폴더 업로드
- ✅ 개별 파일 업로드

---

### 4. annotation 병합 금지

**규칙**: labeled 데이터셋에 labeled 폴더 추가 **불가**

**이유**:
- ID 충돌 해결 복잡
- 클래스 매핑 복잡
- 데이터 무결성 위험

**대안**:
- 레이블러 툴 사용
- 새 데이터셋 생성

---

### 5. meta.json 생성 시점

**규칙**:
- unlabeled: meta.json 없음 (DB만)
- labeled 전환 시: annotation.json + meta.json 함께 생성
- export 시: 항상 meta.json 포함

**Single Source of Truth**: DB (meta.json은 DB에서 생성)

---

## 데이터 모델

### Dataset (DB)

```python
class Dataset(Base):
    __tablename__ = "datasets"

    # 기본 정보
    id: str  # UUID
    name: str
    description: str

    # 상태
    num_images: int
    labeled: bool  # annotation.json 존재 여부

    # labeled=True인 경우에만
    num_classes: Optional[int]
    class_names: Optional[List[str]]

    # 스토리지
    storage_type: str  # "r2"
    storage_path: str  # "datasets/{id}/"
    annotation_path: Optional[str]  # "datasets/{id}/annotation.json"

    # 스냅샷 관리
    is_snapshot: bool = False
    parent_dataset_id: Optional[str]  # 스냅샷인 경우 원본 ID
    snapshot_created_at: Optional[datetime]
    version_tag: Optional[str]  # "v1", "v2" 등

    # 무결성
    status: str  # "valid", "broken", "repairing"
    integrity_status: Optional[dict]  # JSON

    # 메타데이터
    visibility: str  # "public", "private", "organization"
    owner_id: Optional[int]
    created_at: datetime
    updated_at: datetime
```

### TrainingJob (DB)

```python
class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: int
    dataset_id: str  # HEAD 데이터셋 ID
    dataset_snapshot_id: str  # 실제 사용한 스냅샷 ID
    snapshot_status_at_start: str  # 학습 시작 시 스냅샷 상태

    task_type: str  # classification, detection, segmentation
    model_name: str
    # ... 기타 학습 설정
```

---

## 스토리지 구조

### R2 구조

#### unlabeled 데이터셋:
```
bucket/datasets/{dataset-id}/
  └── images/
      ├── cats/cat1.jpg
      ├── dogs/dog1.jpg
      └── ...

(meta.json 없음 - DB에만 존재)
```

#### labeled 데이터셋 (HEAD):
```
bucket/datasets/{dataset-id}/
  ├── meta.json           ← labeled 전환 시 생성
  ├── annotation.json     ← labeled 전환 시 생성
  └── images/             ← 모든 버전이 공유
      ├── cats/cat1.jpg
      ├── dogs/dog1.jpg
      └── ...
```

#### 스냅샷:
```
bucket/datasets/{dataset-id}/
  ├── meta.json
  ├── annotation.json     ← HEAD (최신)
  ├── images/             ← 모든 버전 공유!
  │   └── *.jpg
  └── snapshots/
      ├── snapshot-20250104-123045/
      │   ├── annotation.json  ← 과거 시점 annotation만
      │   └── meta.json
      └── snapshot-20250105-083020/
          ├── annotation.json
          └── meta.json
```

**크기 비교**:
```
기존 방식 (이미지 복사): 10GB + 10GB + 10GB = 30GB
새 방식 (annotation만): 10GB + 10MB + 10MB = 10.02GB
절약률: 99%
```

---

## 업로드 시나리오

### 허용되는 조합

| Case | 프로젝트 | 방식 | Annotation | 허용? | 설명 |
|------|---------|------|------------|-------|------|
| 1 | 새 | 폴더 | ❌ | ✅ | 빈 데이터셋 생성 |
| 2 | 새 | 폴더 | ✅ | ✅ | labeled 데이터셋 생성 |
| 3 | 새 | 개별 | ❌ | ✅ | 소량 테스트 데이터 |
| 4 | 새 | 개별 | ✅ | ❌ | **불가능** |
| 5 | 기존(unlabeled) | 폴더 | ❌ | ✅ | 이미지 추가 |
| 6 | 기존(unlabeled) | 폴더 | ✅ | ⚠️ | **첫 annotation 추가** (labeled로 전환) |
| 7 | 기존(unlabeled) | 개별 | ❌ | ✅ | 이미지 추가 (현재 구현) |
| 8 | 기존(unlabeled) | 개별 | ✅ | ❌ | **불가능** |
| 9 | 기존(labeled) | 폴더 | ❌ | ✅ | 이미지만 추가 (레이블러로 작업) |
| 10 | 기존(labeled) | 폴더 | ✅ | ❌ | **금지** (병합 복잡도) |
| 11 | 기존(labeled) | 개별 | ❌ | ✅ | 이미지만 추가 |
| 12 | 기존(labeled) | 개별 | ✅ | ❌ | **불가능** |

### 핵심 규칙

1. **labeled 전환은 한 번만**
   ```
   unlabeled → labeled (최초 1회)
   labeled → unlabeled (불가능)
   ```

2. **annotation.json은 전체 교체만 가능**
   ```
   ✅ 전체 삭제 후 새로 업로드
   ❌ 기존과 병합
   ```

3. **labeled 데이터셋 수정은 레이블러에서만**
   ```
   labeled 데이터셋 → 레이블러 툴로만 편집
   annotation 직접 업로드는 새 데이터셋에만
   ```

---

## 버전닝 전략

### 개요: Mutable + Snapshot

**원칙**:
1. 데이터셋은 기본적으로 **가변(mutable)**
2. 학습 시작 시 **자동 스냅샷** 생성
3. 사용자가 원하면 **명시적 버전** 생성 가능 (v1, v2...)
4. TrainingJob에 `dataset_snapshot_id` 기록 → 재현성 보장

### 이미지 관리

**규칙**:
```
✅ 이미지 추가: 언제든지 가능
✅ 이미지 삭제: 가능 (스냅샷 무결성 관리 필요)
✅ 이미지 수정: 가능 (삭제 후 추가)
```

**이유**: annotation.json만 버전 관리하므로 이미지는 공유

---

### 스냅샷 생성

```python
def start_training(dataset_id):
    # 1. annotation.json 스냅샷 생성
    snapshot_id = f"{dataset_id}@snapshot-{datetime.now().isoformat()}"

    # 현재 annotation 복사
    source = f"datasets/{dataset_id}/annotation.json"
    target = f"datasets/{dataset_id}/snapshots/{snapshot_id}/annotation.json"
    r2.copy(source, target)

    # meta.json도 복사
    source_meta = f"datasets/{dataset_id}/meta.json"
    target_meta = f"datasets/{dataset_id}/snapshots/{snapshot_id}/meta.json"
    r2.copy(source_meta, target_meta)

    # 2. DB 기록
    snapshot = Dataset(
        id=snapshot_id,
        name=f"{dataset.name} (snapshot)",
        is_snapshot=True,
        parent_dataset_id=dataset_id,
        snapshot_created_at=datetime.now(),
        status="valid"
    )
    db.add(snapshot)

    # 3. 학습 작업에 스냅샷 ID 기록
    job = TrainingJob(
        dataset_id=dataset_id,
        dataset_snapshot_id=snapshot_id,
        snapshot_status_at_start="valid"
    )
    db.add(job)

    # 4. 학습은 스냅샷 사용
    return {
        "images_path": f"datasets/{dataset_id}/images/",  # 공유
        "annotation_path": f"datasets/{dataset_id}/snapshots/{snapshot_id}/annotation.json"
    }
```

---

### 명시적 버전 생성

```python
def create_version(dataset_id, version_tag):
    """사용자가 명시적으로 버전 생성"""

    # 스냅샷 생성 (동일한 로직)
    snapshot_id = f"{dataset_id}@{version_tag}"
    create_snapshot(dataset_id, snapshot_id)

    # 버전 태그 추가
    snapshot = db.query(Dataset).filter(Dataset.id == snapshot_id).first()
    snapshot.version_tag = version_tag
    db.commit()
```

**UI**:
```
┌─────────────────────────────────────┐
│ 데이터셋: 반려동물 분류                │
│                                     │
│ [새 버전 생성]                        │
│                                     │
│ 버전 기록:                           │
│ ├─ HEAD (현재)        150 images    │
│ ├─ v1                 100 images    │  ← 사용자 생성
│ └─ @snapshot-001      95 images     │  ← 학습 시 자동
│    (학습 Job #123)                  │
└─────────────────────────────────────┘
```

---

## 무결성 관리

### 이미지 삭제 프로세스

#### Step 1: 영향 분석
```python
def delete_image(dataset_id, filename):
    # 영향받는 스냅샷 찾기
    affected_snapshots = []

    snapshots = get_snapshots(dataset_id)
    for snapshot in snapshots:
        annotation = load_annotation(snapshot)
        images = [img['file_name'] for img in annotation['images']]

        if filename in images:
            affected_snapshots.append(snapshot)

    # 사용자에게 확인 요청
    return {
        "status": "confirmation_required",
        "affected_snapshots": affected_snapshots,
        "options": {
            "mark_broken": "스냅샷을 broken으로 표시",
            "auto_repair": "annotation에서 자동 제거",
            "cancel": "삭제 취소"
        }
    }
```

#### Step 2A: Broken 표시
```python
def delete_image_mark_broken(dataset_id, filename):
    # 이미지 삭제
    r2.delete(f"datasets/{dataset_id}/images/{filename}")

    # 영향받는 스냅샷 broken 표시
    for snapshot in affected_snapshots:
        snapshot.status = "broken"
        snapshot.integrity_status = {
            "total_images": get_image_count(snapshot),
            "missing_images": [filename],
            "missing_count": 1,
            "broken_at": datetime.now()
        }
        db.commit()
```

**결과**: 해당 스냅샷 사용한 학습은 재현 불가능 (기록은 유지)

#### Step 2B: 자동 복구
```python
def delete_image_auto_repair(dataset_id, filename):
    # 이미지 삭제
    r2.delete(f"datasets/{dataset_id}/images/{filename}")

    # 각 스냅샷 annotation 수정
    for snapshot in affected_snapshots:
        annotation = load_annotation(snapshot)

        # 이미지 제거
        annotation['images'] = [
            img for img in annotation['images']
            if img['file_name'] != filename
        ]

        # 해당 이미지의 annotation 제거
        image_id = get_image_id(annotation, filename)
        annotation['annotations'] = [
            ann for ann in annotation['annotations']
            if ann['image_id'] != image_id
        ]

        # 저장
        save_annotation(snapshot, annotation)

        snapshot.status = "valid"
        snapshot.integrity_status = {
            "repaired_at": datetime.now(),
            "removed_images": [filename]
        }
```

**결과**: 스냅샷은 valid 유지, 수정된 버전으로 재현 가능

---

### 주기적 무결성 체크

```python
@celery_task(schedule="daily")
def check_snapshot_integrity():
    """모든 스냅샷의 무결성 확인"""

    snapshots = db.query(Dataset).filter(
        Dataset.is_snapshot == True,
        Dataset.status == "valid"
    ).all()

    for snapshot in snapshots:
        annotation = load_annotation(snapshot)
        missing = []

        for img in annotation['images']:
            path = f"datasets/{snapshot.parent_dataset_id}/images/{img['file_name']}"
            if not r2.exists(path):
                missing.append(img['file_name'])

        if missing:
            snapshot.status = "broken"
            snapshot.integrity_status = {
                "total_images": len(annotation['images']),
                "missing_images": missing,
                "missing_count": len(missing),
                "detected_at": datetime.now()
            }
            db.commit()

            notify_admin(f"Snapshot {snapshot.id} broken: {len(missing)} images missing")
```

---

### 스냅샷 복구

```python
def repair_snapshot(snapshot_id):
    """Broken 스냅샷 복구"""
    snapshot = db.query(Dataset).get(snapshot_id)
    annotation = load_annotation(snapshot)
    missing = snapshot.integrity_status['missing_images']

    # 누락된 이미지를 annotation에서 제거
    annotation['images'] = [
        img for img in annotation['images']
        if img['file_name'] not in missing
    ]

    # 해당 이미지의 annotation 제거
    missing_ids = [get_image_id(annotation, f) for f in missing]
    annotation['annotations'] = [
        ann for ann in annotation['annotations']
        if ann['image_id'] not in missing_ids
    ]

    # 저장
    save_annotation(snapshot, annotation)
    snapshot.status = "valid"
    snapshot.integrity_status = {
        "repaired_at": datetime.now(),
        "removed_images": missing
    }
    db.commit()
```

---

### 스냅샷 상태

- **valid**: 모든 이미지 존재, 학습 가능
- **broken**: 일부 이미지 누락, 학습 불가
- **repairing**: 복구 중

---

## 스냅샷 정리 정책

### 자동 정리

```python
@celery_task(schedule="weekly")
def cleanup_old_snapshots():
    """오래된 스냅샷 자동 정리"""

    # 옵션 A: 시간 기반 (30일)
    cutoff = datetime.now() - timedelta(days=30)
    old_snapshots = db.query(Dataset).filter(
        Dataset.is_snapshot == True,
        Dataset.snapshot_created_at < cutoff,
        Dataset.version_tag == None  # 버전 태그 없는 것만
    ).all()

    # 옵션 B: 개수 기반 (최근 10개만 유지)
    for dataset in db.query(Dataset).filter(Dataset.is_snapshot == False).all():
        snapshots = db.query(Dataset).filter(
            Dataset.parent_dataset_id == dataset.id
        ).order_by(Dataset.snapshot_created_at.desc()).all()

        # 11번째부터 삭제
        for snapshot in snapshots[10:]:
            if not snapshot.version_tag:  # 버전 태그 있으면 보존
                delete_snapshot(snapshot.id)
```

---

## 구현 우선순위

### Phase 1: 기본 기능 (현재)
- [x] 개별 이미지 업로드 (unlabeled)
- [x] 이미지 조회 (Presigned URL)
- [x] DB 모델 (기본)

### Phase 2: 폴더 업로드
- [ ] 폴더 구조 유지 업로드
- [ ] labeled 데이터셋 생성 (annotation.json 포함)
- [ ] DB 모델 확장 (labeled, class_names 등)

### Phase 3: 버전닝
- [ ] 학습 시 자동 스냅샷
- [ ] 명시적 버전 생성
- [ ] 스냅샷 목록 UI

### Phase 4: 무결성 관리
- [ ] 이미지 삭제 시 영향 분석
- [ ] Broken/복구 로직
- [ ] 주기적 무결성 체크

### Phase 5: 최적화
- [ ] 스냅샷 자동 정리
- [ ] 스냅샷 압축 (백업용)

---

## 참고 문서

- [DICE Format v2.0](./DICE_FORMAT_v2.md)
- [Storage Access Patterns](./STORAGE_ACCESS_PATTERNS.md)
- [Dataset API Specification](../api/DATASET_API.md) (TBD)

---

## 변경 이력

- 2025-01-04: 초안 작성 (설계 논의 기반)
