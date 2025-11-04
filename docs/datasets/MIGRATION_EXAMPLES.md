# Dataset Migration Examples

v0.9 → v1.0 마이그레이션 실제 예시

---

## Example 1: Classification Dataset

### Before (v0.9)

**디렉토리 구조:**
```
E:/my-cats-dogs/
├── images/
│   ├── cat_001.jpg
│   └── dog_001.jpg
├── labels/
│   ├── cat_001.json
│   └── dog_001.json
└── label_map.json
```

**label_map.json:**
```json
{
  "task_type": "cls",
  "class_summary": {
    "num_classes": 2,
    "classes": [
      {"name": "cat", "idx": 0, "color": "#FF6B6B"},
      {"name": "dog", "idx": 1, "color": "#4ECDC4"}
    ]
  },
  "data_summary": [
    {
      "img_path": "E:/my-cats-dogs/images/cat_001.jpg",
      "label_path": "E:/my-cats-dogs/labels/cat_001.json"
    },
    {
      "img_path": "E:/my-cats-dogs/images/dog_001.jpg",
      "label_path": "E:/my-cats-dogs/labels/dog_001.json"
    }
  ]
}
```

**labels/cat_001.json:**
```json
{
  "version": "0.9",
  "task_type": "cls",
  "shapes": [
    {
      "label": "cat",
      "points": [[0, 0]],
      "group_id": null,
      "shape_type": "point",
      "flags": {}
    }
  ],
  "split": "train",
  "imageHeight": 1080,
  "imageWidth": 1920,
  "imageDepth": 3
}
```

### After (v1.0)

**디렉토리 구조 (R2):**
```
s3://bucket/datasets/user123-cats-dogs/
├── annotations.json          ← NEW: 통합 파일
├── images/
│   ├── cat_001.jpg
│   └── dog_001.jpg
├── legacy/                   ← NEW: v0.9 백업
│   ├── labels/
│   │   ├── cat_001.json
│   │   └── dog_001.json
│   └── label_map.json
└── meta.json
```

**annotations.json:**
```json
{
  "format_version": "1.0",
  "dataset_id": "user123-cats-dogs",
  "task_type": "image_classification",

  "migration_info": {
    "migrated_from": "v0.9",
    "migration_date": "2025-01-15T10:00:00Z",
    "original_paths": {
      "images": "E:/my-cats-dogs/images/",
      "labels": "E:/my-cats-dogs/labels/"
    }
  },

  "classes": [
    {"id": 0, "name": "cat", "color": "#FF6B6B"},
    {"id": 1, "name": "dog", "color": "#4ECDC4"}
  ],

  "images": [
    {
      "id": 1,
      "file_name": "cat_001.jpg",
      "width": 1920,
      "height": 1080,
      "split": "train",

      "annotation": {
        "class_id": 0,
        "class_name": "cat",
        "confidence": 1.0
      },

      "legacy_v09": {
        "shape_type": "point",
        "points": [[0, 0]]
      }
    },
    {
      "id": 2,
      "file_name": "dog_001.jpg",
      "width": 1920,
      "height": 1080,
      "split": "train",

      "annotation": {
        "class_id": 1,
        "class_name": "dog",
        "confidence": 1.0
      },

      "legacy_v09": {
        "shape_type": "point",
        "points": [[0, 0]]
      }
    }
  ]
}
```

---

## Example 2: Detection Dataset

### Before (v0.9)

**labels/street_001.json:**
```json
{
  "version": "0.9",
  "task_type": "det",
  "shapes": [
    {
      "label": "car",
      "points": [[100, 150], [400, 350]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "person",
      "points": [[500, 200], [600, 400]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    }
  ],
  "split": "train",
  "imageHeight": 600,
  "imageWidth": 800,
  "imageDepth": 4
}
```

### After (v1.0)

**annotations.json (해당 이미지 부분):**
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "street_001.jpg",
      "width": 800,
      "height": 600,
      "split": "train",

      "annotations": [
        {
          "id": 1001,
          "class_id": 0,
          "class_name": "car",
          "bbox": [100, 150, 300, 200],  // [x, y, w, h]
          "bbox_format": "xywh",
          "area": 60000
        },
        {
          "id": 1002,
          "class_id": 1,
          "class_name": "person",
          "bbox": [500, 200, 100, 200],
          "bbox_format": "xywh",
          "area": 20000
        }
      ],

      "legacy_v09": {
        "shapes": [
          {
            "label": "car",
            "points": [[100, 150], [400, 350]],
            "shape_type": "rectangle"
          },
          {
            "label": "person",
            "points": [[500, 200], [600, 400]],
            "shape_type": "rectangle"
          }
        ]
      }
    }
  ]
}
```

**변환 로직:**
```python
# v0.9 points: [[top-left], [bottom-right]]
points = [[100, 150], [400, 350]]

x1, y1 = points[0]  # 100, 150
x2, y2 = points[1]  # 400, 350

# v1.0 bbox: [x, y, w, h]
x = x1                # 100
y = y1                # 150
w = abs(x2 - x1)      # 300
h = abs(y2 - y1)      # 200

bbox = [100, 150, 300, 200]
```

---

## Example 3: Instance Segmentation

### Before (v0.9)

**labels/cat_seg.json:**
```json
{
  "version": "0.9",
  "task_type": "det",
  "shapes": [
    {
      "label": "cat",
      "points": [
        [2818.5, 373.48],
        [2887.0, 360.5],
        [2900.0, 426.5],
        [2831.5, 439.48]
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "split": "train",
  "imageHeight": 600,
  "imageWidth": 800,
  "imageDepth": 4
}
```

### After (v1.0)

**annotations.json (해당 이미지 부분):**
```json
{
  "task_type": "instance_segmentation",

  "images": [
    {
      "id": 1,
      "file_name": "cat_seg.jpg",
      "width": 800,
      "height": 600,
      "split": "train",

      "annotations": [
        {
          "id": 2001,
          "class_id": 0,
          "class_name": "cat",
          "bbox": [2818.5, 360.5, 81.5, 78.98],
          "segmentation": [
            [2818.5, 373.48, 2887.0, 360.5, 2900.0, 426.5, 2831.5, 439.48]
          ],
          "area": 3200
        }
      ],

      "legacy_v09": {
        "shapes": [
          {
            "label": "cat",
            "points": [
              [2818.5, 373.48],
              [2887.0, 360.5],
              [2900.0, 426.5],
              [2831.5, 439.48]
            ],
            "shape_type": "polygon"
          }
        ]
      }
    }
  ]
}
```

**변환 로직:**
```python
# v0.9 points: [[x1, y1], [x2, y2], ...]
points = [
    [2818.5, 373.48],
    [2887.0, 360.5],
    [2900.0, 426.5],
    [2831.5, 439.48]
]

# v1.0 segmentation: flattened array
segmentation = [2818.5, 373.48, 2887.0, 360.5, 2900.0, 426.5, 2831.5, 439.48]

# Calculate bbox from polygon
xs = [p[0] for p in points]  # [2818.5, 2887.0, 2900.0, 2831.5]
ys = [p[1] for p in points]  # [373.48, 360.5, 426.5, 439.48]

x_min = min(xs)  # 2818.5
y_min = min(ys)  # 360.5
x_max = max(xs)  # 2900.0
y_max = max(ys)  # 439.48

bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
# [2818.5, 360.5, 81.5, 78.98]
```

---

## Example 4: Semantic Segmentation

### Before (v0.9)

**label_map.json:**
```json
{
  "task_type": "seg",
  "class_summary": {
    "num_classes": 3,
    "classes": [
      {"name": "_background_", "idx": 0, "color": "#000000"},
      {"name": "cat", "idx": 1, "color": "#FF0000"},
      {"name": "dog", "idx": 2, "color": "#00FF00"}
    ]
  },
  "data_summary": [
    {
      "img_path": "E:/data/images/image1.jpg",
      "label_path": "E:/data/labels/image1.json",
      "mask_path": "E:/data/masks/image1_mask.png"
    }
  ]
}
```

### After (v1.0)

**annotations.json:**
```json
{
  "task_type": "semantic_segmentation",

  "classes": [
    {"id": 1, "name": "cat", "color": "#FF0000"},
    {"id": 2, "name": "dog", "color": "#00FF00"}
  ],

  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 1920,
      "height": 1080,
      "split": "train",

      "annotation": {
        "mask_file": "masks/image1_mask.png",
        "mask_format": "indexed_png",
        "num_classes": 3
      },

      "legacy_v09": {
        "mask_path": "E:/data/masks/image1_mask.png"
      }
    }
  ]
}
```

---

## 변환 결과 비교

| Feature | v0.9 | v1.0 | 변화 |
|---------|------|------|------|
| **파일 수** | 1 + N (summary + 개별) | 1 (통합) | ✅ 단순화 |
| **경로** | 절대 경로 | 상대 경로 | ✅ Cloud 호환 |
| **task_type** | cls/det/seg | image_classification/object_detection/... | ✅ 명확화 |
| **bbox 포맷** | [[x1,y1], [x2,y2]] | [x, y, w, h] | ✅ 표준화 (COCO) |
| **segmentation** | [[x1,y1], [x2,y2]] | [x1, y1, x2, y2, ...] | ✅ 표준화 (COCO) |
| **메타데이터** | ❌ | labeled_by, reviewed_by, timestamps | ✅ 추가 |
| **통계** | ❌ | 자동 계산 | ✅ 추가 |
| **하위 호환** | N/A | legacy_v09 필드 | ✅ 보존 |

---

## 자동 변환 프로세스

### 1. Upload & Detect

```
사용자: my-dataset.zip 업로드
  ↓
Backend: 압축 해제
  ↓
Backend: 포맷 감지
  - label_map.json 발견 → v0.9 감지
  ↓
Backend: "v0.9 포맷이 감지되었습니다. v1.0으로 변환하시겠습니까?"
  ↓
사용자: "예"
```

### 2. Migration

```
Migrator: label_map.json 파싱
  - task_type: "det" → "object_detection"
  - classes 추출
  ↓
Migrator: 개별 레이블 파일 처리
  - labels/img001.json 파싱
  - shapes → annotations 변환
  - 경로 정규화 (E:/ → 상대 경로)
  ↓
Migrator: annotations.json 생성
  - migration_info 추가
  - legacy_v09 필드 보존
  ↓
Migrator: 통계 계산
  - splits, class_distribution 등
```

### 3. Upload to R2

```
Backend: R2 업로드
  s3://bucket/datasets/user123-my-dataset/
  ├── annotations.json       ← v1.0
  ├── images/
  ├── legacy/               ← v0.9 백업
  │   ├── labels/
  │   └── label_map.json
  └── meta.json
  ↓
Backend: DB 업데이트
  Dataset(
    id='user123-my-dataset',
    format='platform',
    task_type='object_detection',
    content_hash='sha256:...'
  )
  ↓
Backend: "✅ 마이그레이션 완료! 1000개 이미지 변환됨."
```

### 4. Training

```
사용자: 학습 시작
  ↓
Trainer: annotations.json 다운로드
  ↓
Trainer: 포맷 확인
  - format_version: "1.0" → Platform Format
  - task_type: "object_detection"
  ↓
Converter: YOLO 포맷 변환 (ultralytics용)
  - annotations.json → images/, labels/, data.yaml
  ↓
Ultralytics: 학습 시작
  ✅ 성공!
```

---

**Last Updated:** 2025-01-04
