# Unlabeled Dataset 처리 및 meta.json vs annotations.json

**Date:** 2025-01-04
**Topic:** 레이블링 전 데이터셋 처리 및 파일 역할 구분

---

## 📋 목차

1. [핵심 질문](#핵심-질문)
2. [meta.json vs annotations.json 역할 구분](#metajson-vs-annotationsjson-역할-구분)
3. [레이블링 전 데이터셋 처리](#레이블링-전-데이터셋-처리)
4. [레이블링 워크플로우](#레이블링-워크플로우)
5. [실제 예시](#실제-예시)

---

## 핵심 질문

### Q1: 레이블이 없는 파일은 어떻게 관리되나?

**시나리오:**
```
유저가 100장 업로드 (이미지만)
→ 레이블링 전 상태
→ annotations.json은 어떻게 되는가?
```

**답변:** annotations.json은 생성되며, annotation 필드가 `null` 또는 `"status": "unlabeled"` 상태로 저장됩니다.

---

### Q2: meta.json과 annotations.json의 차이는?

**혼란의 원인:**
문서에 두 파일이 모두 등장하지만 역할 구분이 명확하지 않았습니다.

**답변:**
- **meta.json**: DB 동기화용 경량 메타데이터 (버전, 해시, 통계)
- **annotations.json**: 실제 레이블 데이터 (전체 이미지 + 레이블 정보)

---

## meta.json vs annotations.json 역할 구분

### 비교표

| 항목 | meta.json | annotations.json |
|------|-----------|------------------|
| **목적** | DB 동기화, 캐시 invalidation | 실제 레이블 데이터 저장 |
| **크기** | ~1KB (고정) | ~800KB - 200MB (이미지 수에 비례) |
| **읽기 빈도** | 매우 자주 (변경 감지) | 필요 시만 (학습 시작, 레이블링) |
| **포함 정보** | 버전, 해시, 기본 통계 | 전체 이미지 목록, 레이블, 메타데이터 |
| **필수 여부** | 선택 (없으면 DB에서 조회) | 필수 (레이블 데이터 소스) |

---

### meta.json 상세

**역할:** "이 데이터셋이 변경되었는가?" 빠르게 확인

**구조:**
```json
{
  "dataset_id": "user123-cats-dogs",
  "version": 3,
  "content_hash": "sha256:abc123def456...",
  "last_modified_at": "2025-01-20T15:30:00Z",

  "statistics": {
    "total_images": 1000,
    "labeled_images": 850,
    "unlabeled_images": 150,
    "num_classes": 2
  },

  "format_version": "1.0",
  "task_type": "image_classification"
}
```

**사용 예시:**
```python
# Training 시작 전: 캐시가 유효한지 확인
cached_hash = local_cache.get_hash("user123-cats-dogs")
meta = download_from_r2("datasets/user123-cats-dogs/meta.json")

if cached_hash == meta['content_hash']:
    # 캐시 재사용 (annotations.json 다운로드 불필요!)
    use_local_cache()
else:
    # 변경 감지: annotations.json 다운로드 필요
    download_annotations()
```

---

### annotations.json 상세

**역할:** 실제 레이블 데이터 저장 및 학습 소스

**구조:**
```json
{
  "format_version": "1.0",
  "dataset_id": "user123-cats-dogs",
  "task_type": "image_classification",

  "created_at": "2025-01-15T10:00:00Z",
  "last_modified_at": "2025-01-20T15:30:00Z",
  "version": 3,
  "content_hash": "sha256:abc123def456...",

  "classes": [...],
  "splits": {...},

  "images": [
    {
      "id": 1,
      "file_name": "img001.jpg",
      "annotation": {...},  // ← 실제 레이블
      "metadata": {...}
    },
    // ... 1000개 이미지
  ],

  "statistics": {...}
}
```

**사용 예시:**
```python
# Training 시작: 전체 데이터 로딩
annotations = load_annotations("datasets/user123-cats-dogs/annotations.json")

for img in annotations['images']:
    image_path = img['file_name']
    label = img['annotation']['class_id']
    train(image_path, label)
```

---

### R2 디렉토리 구조

```
s3://bucket/datasets/user123-cats-dogs/
├── meta.json                 ← 경량 메타데이터 (1KB)
├── annotations.json          ← 실제 레이블 (800KB - 200MB)
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── masks/                    ← Optional (segmentation)
```

---

## 레이블링 전 데이터셋 처리

### 시나리오: 이미지 100장 업로드 (레이블 없음)

#### Step 1: 업로드

**사용자 행동:**
```
1. 플랫폼 UI에서 "데이터셋 생성" 클릭
2. 이미지 100장 선택 (ZIP 또는 개별 업로드)
3. Task Type 선택: "Image Classification"
4. 업로드 시작
```

**Backend 처리:**
```python
# 1. 이미지 파일만 R2 업로드
for img in uploaded_images:
    r2.upload(f"datasets/{dataset_id}/images/{img.name}", img)

# 2. annotations.json 생성 (레이블 없는 상태)
annotations = {
    "format_version": "1.0",
    "dataset_id": dataset_id,
    "task_type": "image_classification",
    "created_at": now(),
    "version": 1,

    "classes": [],  # ← 아직 클래스 정의 없음

    "images": [
        {
            "id": 1,
            "file_name": "img001.jpg",
            "width": 1920,
            "height": 1080,
            "split": "train",

            "annotation": null,  # ← 레이블 없음!

            "metadata": {
                "labeled_by": null,
                "labeled_at": null,
                "status": "unlabeled"  # ← 명시적 상태
            }
        },
        {
            "id": 2,
            "file_name": "img002.jpg",
            "annotation": null,
            "metadata": {"status": "unlabeled"}
        }
        // ... 100개
    ],

    "statistics": {
        "total_images": 100,
        "labeled_images": 0,      // ← 0개
        "unlabeled_images": 100,  // ← 100개
        "labeling_progress": 0.0  // ← 0%
    }
}

r2.upload(f"datasets/{dataset_id}/annotations.json", annotations)

# 3. meta.json 생성
meta = {
    "dataset_id": dataset_id,
    "version": 1,
    "content_hash": calc_hash(annotations),
    "last_modified_at": now(),
    "statistics": {
        "total_images": 100,
        "labeled_images": 0,
        "unlabeled_images": 100
    },
    "format_version": "1.0",
    "task_type": "image_classification"
}

r2.upload(f"datasets/{dataset_id}/meta.json", meta)

# 4. DB 업데이트
db.datasets.create(
    id=dataset_id,
    name="My Unlabeled Dataset",
    owner_id=user_id,
    task_type="image_classification",
    format="platform",
    num_images=100,
    num_classes=0,          # ← 클래스 없음
    labeling_progress=0.0,  # ← 0%
    content_hash=meta['content_hash']
)
```

---

#### Step 2: 레이블링 시작

**UI 표시:**
```
데이터셋 카드:
┌─────────────────────────────────┐
│ 📦 My Unlabeled Dataset         │
│                                 │
│ 상태: ⚠️ 레이블링 필요 (0%)      │
│ 이미지: 100장                    │
│ 클래스: 정의되지 않음            │
│                                 │
│ [레이블링 시작]  [설정]         │
└─────────────────────────────────┘
```

**사용자 클릭: "레이블링 시작"**

1. **클래스 정의:**
   ```
   클래스 추가:
   ├─ "cat" [색상: #FF6B6B]
   └─ "dog" [색상: #4ECDC4]
   ```

2. **레이블링 툴 오픈:**
   ```
   이미지: img001.jpg
   ┌──────────────────┐
   │                  │
   │  [고양이 이미지]  │
   │                  │
   └──────────────────┘

   클래스 선택:
   ○ cat
   ○ dog

   [다음 이미지]
   ```

---

#### Step 3: 레이블 저장 (증분 업데이트)

**레이블링 세션:**
```
Day 1: 20장 레이블링 (cat: 12, dog: 8)
  ↓
annotations.json 업데이트
  ↓
meta.json 업데이트 (version: 2, content_hash 변경)
  ↓
DB 업데이트 (labeling_progress: 20%)
```

**업데이트된 annotations.json:**
```json
{
  "version": 2,  // ← 증가
  "content_hash": "sha256:new_hash...",  // ← 변경

  "classes": [  // ← 새로 추가
    {"id": 0, "name": "cat", "color": "#FF6B6B"},
    {"id": 1, "name": "dog", "color": "#4ECDC4"}
  ],

  "images": [
    {
      "id": 1,
      "file_name": "img001.jpg",

      "annotation": {  // ← 레이블 추가됨!
        "class_id": 0,
        "class_name": "cat",
        "confidence": 1.0
      },

      "metadata": {
        "labeled_by": "user123",
        "labeled_at": "2025-01-20T10:15:00Z",
        "status": "labeled"  // ← 변경
      }
    },
    {
      "id": 2,
      "file_name": "img002.jpg",

      "annotation": null,  // ← 아직 레이블 없음

      "metadata": {
        "status": "unlabeled"
      }
    }
    // ... 나머지 이미지
  ],

  "statistics": {
    "total_images": 100,
    "labeled_images": 20,      // ← 20개로 증가
    "unlabeled_images": 80,    // ← 80개로 감소
    "labeling_progress": 0.2   // ← 20%
  }
}
```

---

## 레이블링 워크플로우

### 전체 흐름

```
┌──────────────────────────────────────────────────────┐
│ Phase 1: 이미지 업로드                                │
│ - 이미지만 업로드 (레이블 없음)                       │
│ - annotations.json 생성 (annotation: null)           │
│ - meta.json 생성 (labeled: 0, unlabeled: 100)       │
└──────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│ Phase 2: 클래스 정의                                  │
│ - 사용자가 클래스 추가 (cat, dog)                     │
│ - annotations.json 업데이트 (classes 추가)           │
└──────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│ Phase 3: 레이블링 (증분)                              │
│                                                      │
│ Day 1: 20장 레이블링                                 │
│   → annotations.json 업데이트 (20개 annotation 추가) │
│   → meta.json 업데이트 (version: 2, hash 변경)      │
│   → labeling_progress: 20%                          │
│                                                      │
│ Day 2: 30장 추가 레이블링                            │
│   → annotations.json 업데이트 (50개 annotation)      │
│   → meta.json 업데이트 (version: 3)                 │
│   → labeling_progress: 50%                          │
│                                                      │
│ Day 3: 50장 추가 레이블링 (완료!)                    │
│   → annotations.json 업데이트 (100개 모두 labeled)   │
│   → meta.json 업데이트 (version: 4)                 │
│   → labeling_progress: 100%                         │
└──────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│ Phase 4: 학습 가능                                    │
│ - 100% 레이블링 완료                                  │
│ - 학습 시작 가능                                      │
└──────────────────────────────────────────────────────┘
```

---

### 부분 레이블링 데이터셋으로 학습 가능?

**질문:** 50% 레이블링된 상태에서 학습 가능한가?

**답변:** 가능! (필터링 사용)

```python
# Training Runner
annotations = load_annotations("datasets/user123/annotations.json")

# 레이블된 이미지만 필터링
labeled_images = [
    img for img in annotations['images']
    if img['annotation'] is not None  # ← 레이블 있는 것만
]

print(f"Total: {len(annotations['images'])}")
print(f"Labeled: {len(labeled_images)}")
print(f"Will train on: {len(labeled_images)} images")

# 학습 진행
for img in labeled_images:
    train(img['file_name'], img['annotation'])
```

**UI 경고:**
```
⚠️ 이 데이터셋은 50% 레이블링되었습니다.
   - 전체: 100장
   - 레이블됨: 50장
   - 레이블 안됨: 50장

학습은 레이블된 50장으로만 진행됩니다.
계속하시겠습니까?

[취소]  [계속 학습]
```

---

## 실제 예시

### 예시 1: 완전히 Unlabeled Dataset

**R2 구조:**
```
s3://bucket/datasets/user123-unlabeled/
├── meta.json
├── annotations.json
└── images/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

**annotations.json:**
```json
{
  "format_version": "1.0",
  "dataset_id": "user123-unlabeled",
  "task_type": "image_classification",
  "version": 1,
  "created_at": "2025-01-20T10:00:00Z",

  "classes": [],  // ← 클래스 없음

  "images": [
    {
      "id": 1,
      "file_name": "img001.jpg",
      "width": 1920,
      "height": 1080,
      "split": "train",
      "annotation": null,  // ← 레이블 없음
      "metadata": {
        "status": "unlabeled",
        "uploaded_at": "2025-01-20T10:00:00Z"
      }
    },
    {
      "id": 2,
      "file_name": "img002.jpg",
      "annotation": null,
      "metadata": {"status": "unlabeled"}
    }
  ],

  "statistics": {
    "total_images": 100,
    "labeled_images": 0,
    "unlabeled_images": 100,
    "labeling_progress": 0.0
  }
}
```

**meta.json:**
```json
{
  "dataset_id": "user123-unlabeled",
  "version": 1,
  "content_hash": "sha256:abc123...",
  "last_modified_at": "2025-01-20T10:00:00Z",
  "statistics": {
    "total_images": 100,
    "labeled_images": 0,
    "unlabeled_images": 100
  },
  "format_version": "1.0",
  "task_type": "image_classification"
}
```

---

### 예시 2: 부분 Labeled Dataset (50%)

**annotations.json (일부만):**
```json
{
  "version": 3,
  "classes": [
    {"id": 0, "name": "cat"},
    {"id": 1, "name": "dog"}
  ],

  "images": [
    // Labeled (50개)
    {
      "id": 1,
      "file_name": "img001.jpg",
      "annotation": {
        "class_id": 0,
        "class_name": "cat",
        "confidence": 1.0
      },
      "metadata": {
        "status": "labeled",
        "labeled_by": "user123",
        "labeled_at": "2025-01-20T10:15:00Z"
      }
    },

    // Unlabeled (50개)
    {
      "id": 51,
      "file_name": "img051.jpg",
      "annotation": null,
      "metadata": {
        "status": "unlabeled"
      }
    }
  ],

  "statistics": {
    "total_images": 100,
    "labeled_images": 50,
    "unlabeled_images": 50,
    "labeling_progress": 0.5
  }
}
```

---

### 예시 3: Detection Task (Unlabeled)

**annotations.json:**
```json
{
  "task_type": "object_detection",
  "classes": [],  // ← 클래스 없음

  "images": [
    {
      "id": 1,
      "file_name": "street001.jpg",
      "annotations": [],  // ← 빈 배열 (박스 없음)
      "metadata": {
        "status": "unlabeled"
      }
    }
  ],

  "statistics": {
    "total_images": 100,
    "labeled_images": 0,
    "total_annotations": 0
  }
}
```

---

## 상태 전이 다이어그램

```
┌──────────────┐
│   Created    │  이미지 업로드 완료
│  (Unlabeled) │  annotation: null
└──────┬───────┘
       │
       │ 클래스 정의
       │
       ↓
┌──────────────┐
│  Ready for   │  클래스 정의됨
│  Labeling    │  classes: [cat, dog]
└──────┬───────┘
       │
       │ 레이블링 시작
       │
       ↓
┌──────────────┐
│  Partially   │  50% 레이블링됨
│  Labeled     │  labeled: 50, unlabeled: 50
└──────┬───────┘
       │
       │ 계속 레이블링
       │
       ↓
┌──────────────┐
│   Fully      │  100% 레이블링 완료
│   Labeled    │  labeled: 100, unlabeled: 0
└──────┬───────┘
       │
       │ 학습 시작
       │
       ↓
┌──────────────┐
│  Training    │  학습 중
│  In Progress │
└──────────────┘
```

---

## API 엔드포인트

### GET /datasets/{id}/labeling-status

**Response:**
```json
{
  "dataset_id": "user123-cats-dogs",
  "total_images": 100,
  "labeled_images": 50,
  "unlabeled_images": 50,
  "labeling_progress": 0.5,
  "can_start_training": true,  // ← 부분 학습 가능

  "classes_defined": true,
  "num_classes": 2,

  "last_labeled_at": "2025-01-20T15:30:00Z",
  "labeling_speed": "10 images/hour"
}
```

---

### POST /datasets/{id}/images/{image_id}/label

**Request:**
```json
{
  "annotation": {
    "class_id": 0,
    "class_name": "cat",
    "confidence": 1.0
  }
}
```

**Response:**
```json
{
  "success": true,
  "image_id": 1,
  "previous_status": "unlabeled",
  "current_status": "labeled",
  "dataset_progress": 0.51  // ← 업데이트됨
}
```

**Backend 처리:**
```python
# 1. annotations.json 다운로드
annotations = r2.download("datasets/{id}/annotations.json")

# 2. 해당 이미지 업데이트
for img in annotations['images']:
    if img['id'] == image_id:
        img['annotation'] = request.annotation
        img['metadata']['status'] = 'labeled'
        img['metadata']['labeled_by'] = current_user.id
        img['metadata']['labeled_at'] = now()

# 3. 통계 재계산
annotations['statistics']['labeled_images'] += 1
annotations['statistics']['unlabeled_images'] -= 1
annotations['statistics']['labeling_progress'] = (
    annotations['statistics']['labeled_images'] /
    annotations['statistics']['total_images']
)

# 4. 버전 업데이트
annotations['version'] += 1
annotations['last_modified_at'] = now()
annotations['content_hash'] = calc_hash(annotations)

# 5. R2 재업로드
r2.upload("datasets/{id}/annotations.json", annotations)

# 6. meta.json 업데이트
meta['version'] = annotations['version']
meta['content_hash'] = annotations['content_hash']
meta['statistics'] = annotations['statistics']
r2.upload("datasets/{id}/meta.json", meta)

# 7. DB 업데이트
db.update(dataset_id, labeling_progress=annotations['statistics']['labeling_progress'])
```

---

## 요약

### meta.json vs annotations.json

| | meta.json | annotations.json |
|---|-----------|------------------|
| **크기** | ~1KB | ~800KB - 200MB |
| **목적** | 빠른 변경 감지 | 실제 레이블 데이터 |
| **읽기** | 매우 자주 | 필요 시만 |
| **내용** | 버전, 해시, 간단한 통계 | 전체 이미지 + 레이블 |

### Unlabeled Dataset 처리

- ✅ 이미지만 업로드 가능 (레이블 없어도 OK)
- ✅ `annotation: null` 상태로 저장
- ✅ `status: "unlabeled"` 명시적 표시
- ✅ 증분 레이블링 지원 (Day 1: 20장, Day 2: 30장...)
- ✅ 부분 레이블링 상태에서도 학습 가능 (labeled만 사용)
- ✅ `labeling_progress` 자동 계산 (0% → 100%)

---

**Last Updated:** 2025-01-04
**Status:** Complete
