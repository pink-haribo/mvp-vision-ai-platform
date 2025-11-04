# Platform Dataset Format Specification

**Version:** 1.0
**Date:** 2025-01-04
**Status:** Draft

---

## 📋 목차

1. [개요](#개요)
2. [기존 포맷 분석](#기존-포맷-분석)
3. [새로운 포맷 설계](#새로운-포맷-설계)
4. [하위 호환성 전략](#하위-호환성-전략)
5. [마이그레이션 가이드](#마이그레이션-가이드)
6. [구현 계획](#구현-계획)

---

## 개요

### 배경

**기존 시스템 (AI 검사 툴 v0.9)**
- 오프라인 로컬 기반 레이블링 툴
- Classification, Detection, Segmentation 지원
- **포맷 특징:**
  - 이미지 1개당 레이블 파일 1개 (image.jpg + image.json)
  - summary.json으로 전체 데이터셋 관리
  - 로컬 절대 경로 사용 (E:/, C:/ 등)

**현재 상황:**
- ✅ 많은 기존 사용자가 이 포맷으로 데이터셋 구축
- ⚠️ 새로운 클라우드 기반 플랫폼으로 전환 필요
- ⚠️ 더 다양한 태스크 지원 필요 (Pose, Super-Resolution 등)
- ⚠️ R2/Cloud storage에 적합한 구조 필요

### 설계 목표

1. **하위 호환성 100%**: 기존 v0.9 포맷 완전 지원
2. **Cloud 최적화**: 상대 경로, 계층적 구조
3. **확장 가능**: 새 태스크 추가 용이
4. **효율성**: 중복 제거, 빠른 인덱싱
5. **리치 메타데이터**: 버전 관리, 수정 이력, 협업 정보

---

## 기존 포맷 분석

### v0.9 포맷 구조

#### 디렉토리 구조 (로컬)
```
E:/my-dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── img003.jpg
├── labels/
│   ├── img001.json
│   ├── img002.json
│   └── img003.json
├── masks/                    # segmentation only
│   ├── img001_mask.png
│   └── img002_mask.png
└── label_map.json            # summary file
```

#### 개별 레이블 파일 (img001.json)

**Classification:**
```json
{
  "version": "0.9",
  "task_type": "cls",
  "shapes": [
    {
      "label": "Cat",
      "points": [[0, 0]],          // dummy point
      "group_id": null,
      "shape_type": "point",
      "flags": {}
    }
  ],
  "split": "train",
  "imageHeight": 600,
  "imageWidth": 800,
  "imageDepth": 4
}
```

**Detection:**
```json
{
  "version": "0.9",
  "task_type": "det",
  "shapes": [
    {
      "label": "cat",
      "points": [[100, 150], [400, 350]],  // [top-left, bottom-right]
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "dog",
      "points": [[500, 200], [700, 400]],
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

**Segmentation:**
```json
{
  "version": "0.9",
  "task_type": "det",                      // Note: task_type is "det"
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

#### Summary 파일 (label_map.json)

```json
{
  "task_type": "seg",
  "class_summary": {
    "num_classes": 3,
    "classes": [
      {
        "name": "_background_",
        "idx": 0,
        "color": "#000000"
      },
      {
        "name": "cat",
        "idx": 1,
        "color": "#FF0000"
      },
      {
        "name": "dog",
        "idx": 2,
        "color": "#00FF00"
      }
    ]
  },
  "data_summary": [
    {
      "img_path": "E:/data/images/image1.jpg",
      "label_path": "E:/data/labels/image1.json",
      "mask_path": "E:/data/masks/image1_mask.png"
    },
    {
      "img_path": "E:/data/images/image2.jpg",
      "label_path": "E:/data/labels/image2.json",
      "mask_path": "E:/data/masks/image2_mask.png"
    }
  ]
}
```

### v0.9 포맷의 한계

| 문제 | 설명 | 영향 |
|------|------|------|
| **로컬 경로 의존** | 절대 경로 (E:/, C:/) 사용 | R2/Cloud 업로드 시 경로 깨짐 |
| **제한된 메타데이터** | 버전, 수정 이력, 레이블러 정보 없음 | 협업, 추적 불가 |
| **확장성 부족** | Pose, Super-Resolution 등 미지원 | 신규 태스크 추가 어려움 |
| **중복 정보** | 각 json에 imageHeight/Width 반복 | 스토리지 낭비 |
| **분산된 정보** | summary.json + 개별 json 분리 | 데이터 일관성 관리 어려움 |
| **task_type 불일치** | segmentation도 task_type="det" | 혼란 야기 |

---

## 새로운 포맷 설계

### v1.0 Platform Format

#### 핵심 원칙

1. **Single Source of Truth**: 하나의 `annotations.json` 파일
2. **상대 경로**: Cloud storage 호환
3. **확장 가능한 스키마**: 새 태스크 추가 용이
4. **하위 호환 모드**: v0.9 포맷 자동 감지 및 변환

#### 디렉토리 구조 (R2/Cloud)

```
s3://bucket/datasets/user123-cats-dogs/
├── annotations.json          ← Single source of truth
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── img003.jpg
├── masks/                    ← Segmentation only
│   ├── img001_mask.png
│   └── img002_mask.png
├── meta.json                 ← DB 동기화용 메타데이터
└── legacy/                   ← Optional: 기존 v0.9 백업
    ├── labels/
    │   ├── img001.json
    │   └── img002.json
    └── label_map.json
```

#### annotations.json 전체 구조

```json
{
  "format_version": "1.0",
  "dataset_id": "user123-cats-dogs",
  "dataset_name": "My Cats and Dogs Dataset",

  "task_type": "image_classification",

  "created_at": "2025-01-15T10:00:00Z",
  "last_modified_at": "2025-01-20T15:30:00Z",
  "version": 3,
  "content_hash": "sha256:abc123...",

  "migration_info": {
    "migrated_from": "v0.9",
    "migration_date": "2025-01-15T10:00:00Z",
    "original_paths": {
      "images": "E:/my-dataset/images/",
      "labels": "E:/my-dataset/labels/"
    }
  },

  "classes": [
    {
      "id": 0,
      "name": "cat",
      "color": "#FF6B6B",
      "supercategory": "animal"
    },
    {
      "id": 1,
      "name": "dog",
      "color": "#4ECDC4",
      "supercategory": "animal"
    }
  ],

  "splits": {
    "train": 800,
    "val": 150,
    "test": 50
  },

  "images": [
    {
      "id": 1,
      "file_name": "img001.jpg",
      "width": 1920,
      "height": 1080,
      "depth": 3,
      "split": "train",

      "annotation": { /* task-specific */ },

      "metadata": {
        "labeled_by": "user123",
        "labeled_at": "2025-01-15T10:05:00Z",
        "reviewed_by": "admin",
        "reviewed_at": "2025-01-16T09:00:00Z",
        "source": "platform_labeler_v1.0"
      }
    }
  ],

  "statistics": {
    "total_images": 1000,
    "total_annotations": 1000,
    "avg_annotations_per_image": 1.0,
    "class_distribution": {
      "cat": 600,
      "dog": 400
    }
  }
}
```

### Task별 Annotation 스키마

#### 1. Classification

```json
{
  "id": 1,
  "file_name": "cat_001.jpg",
  "width": 1920,
  "height": 1080,
  "depth": 3,
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
}
```

#### 2. Object Detection

```json
{
  "id": 2,
  "file_name": "street_001.jpg",
  "width": 1920,
  "height": 1080,
  "split": "train",

  "annotations": [
    {
      "id": 1001,
      "class_id": 0,
      "class_name": "car",
      "bbox": [100, 200, 300, 400],
      "bbox_format": "xywh",
      "area": 120000,
      "iscrowd": 0
    },
    {
      "id": 1002,
      "class_id": 1,
      "class_name": "person",
      "bbox": [500, 300, 100, 200],
      "bbox_format": "xywh",
      "area": 20000,
      "iscrowd": 0
    }
  ],

  "legacy_v09": {
    "shapes": [
      {
        "label": "car",
        "points": [[100, 200], [400, 600]],
        "shape_type": "rectangle"
      }
    ]
  }
}
```

#### 3. Instance Segmentation

```json
{
  "id": 3,
  "file_name": "cat_segmentation.jpg",
  "width": 3000,
  "height": 2000,
  "split": "train",

  "annotations": [
    {
      "id": 2001,
      "class_id": 0,
      "class_name": "cat",
      "bbox": [2800, 350, 100, 100],
      "segmentation": [
        [2818.5, 373.48, 2887.0, 360.5, 2900.0, 426.5, 2831.5, 439.48]
      ],
      "area": 2500,
      "iscrowd": 0
    }
  ],

  "legacy_v09": {
    "shapes": [
      {
        "label": "cat",
        "points": [[2818.5, 373.48], [2887.0, 360.5], [2900.0, 426.5], [2831.5, 439.48]],
        "shape_type": "polygon"
      }
    ]
  }
}
```

#### 4. Semantic Segmentation

```json
{
  "id": 4,
  "file_name": "scene_001.jpg",
  "width": 1920,
  "height": 1080,
  "split": "train",

  "annotation": {
    "mask_file": "masks/scene_001_mask.png",
    "mask_format": "indexed_png",
    "num_classes": 3
  },

  "legacy_v09": {
    "mask_path": "masks/scene_001_mask.png"
  }
}
```

#### 5. Pose Estimation (NEW)

```json
{
  "id": 5,
  "file_name": "person_pose.jpg",
  "width": 1920,
  "height": 1080,
  "split": "train",

  "annotations": [
    {
      "id": 3001,
      "class_id": 0,
      "class_name": "person",
      "bbox": [500, 200, 300, 600],
      "keypoints": [
        [520, 220, 2],  // [x, y, visibility] - nose
        [510, 240, 2],  // left_eye
        [530, 240, 2],  // right_eye
        [500, 260, 2],  // left_ear
        [540, 260, 2],  // right_ear
        // ... 17 keypoints total (COCO format)
      ],
      "num_keypoints": 17
    }
  ]
}
```

#### 6. Super-Resolution (NEW)

```json
{
  "id": 6,
  "file_name": "low_res_001.jpg",
  "width": 480,
  "height": 270,
  "split": "train",

  "annotation": {
    "hr_image": "hr_images/high_res_001.jpg",
    "upscale_factor": 4,
    "hr_width": 1920,
    "hr_height": 1080
  }
}
```

---

## 하위 호환성 전략

### 3-Tier 호환성 모델

```
┌─────────────────────────────────────────────┐
│  Tier 1: Native v1.0 (신규 사용자)          │
│  - 플랫폼 레이블러 사용                      │
│  - annotations.json 직접 생성                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Tier 2: Auto-Migration (기존 사용자)       │
│  - v0.9 포맷 업로드                          │
│  - 자동 변환 → v1.0                          │
│  - legacy_v09 필드 보존                      │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Tier 3: Dual-Format Support (하이브리드)   │
│  - v1.0 annotations.json 생성                │
│  - legacy/ 폴더에 v0.9 백업 유지             │
│  - 양방향 export 지원                        │
└─────────────────────────────────────────────┘
```

### 자동 감지 및 변환

#### Backend API: Dataset Upload

```python
# mvp/backend/app/api/datasets.py

@router.post("/datasets/upload")
async def upload_dataset(files: List[UploadFile], ...):
    """
    데이터셋 업로드 시 포맷 자동 감지 및 변환.

    지원 포맷:
    - v1.0: annotations.json
    - v0.9: label_map.json + individual labels/
    - YOLO: data.yaml
    - COCO: instances.json
    - ImageFolder: directory structure
    """
    # 1. 업로드된 파일 분석
    detected_format = detect_dataset_format(files)

    # 2. 포맷별 처리
    if detected_format == "platform_v1.0":
        # annotations.json 직접 사용
        annotations = parse_v1_annotations(files)

    elif detected_format == "platform_v0.9":
        # v0.9 → v1.0 자동 변환
        print("[Migration] Detected v0.9 format, converting to v1.0...")
        annotations = migrate_v09_to_v10(files)

    elif detected_format == "yolo":
        annotations = convert_yolo_to_v10(files)

    # ... 기타 포맷

    # 3. R2 업로드 (v1.0 포맷)
    await upload_to_r2(dataset_id, annotations)
```

#### Format Detector

```python
# mvp/backend/app/utils/format_detector.py

def detect_dataset_format(files: List[UploadFile]) -> str:
    """
    업로드된 파일 구조를 분석하여 포맷 감지.

    Returns:
        "platform_v1.0" | "platform_v0.9" | "yolo" | "coco" | "imagefolder"
    """
    file_names = [f.filename for f in files]

    # v1.0: annotations.json 존재
    if "annotations.json" in file_names:
        # 파일 내용 확인
        annotations = json.loads(find_file(files, "annotations.json").read())
        if annotations.get("format_version") == "1.0":
            return "platform_v1.0"

    # v0.9: label_map.json + labels/ 디렉토리
    if "label_map.json" in file_names:
        label_dirs = [f for f in file_names if f.startswith("labels/")]
        if label_dirs:
            return "platform_v0.9"

    # YOLO: data.yaml
    if "data.yaml" in file_names:
        return "yolo"

    # COCO: annotations/instances_*.json
    coco_files = [f for f in file_names if "instances_" in f and f.endswith(".json")]
    if coco_files:
        return "coco"

    # ImageFolder: 디렉토리 구조만 존재
    if any(f.startswith("train/") for f in file_names):
        return "imagefolder"

    raise ValueError("Unknown dataset format")
```

#### v0.9 → v1.0 Migrator

```python
# mvp/backend/app/utils/dataset_migrator.py

class V09ToV10Migrator:
    """
    v0.9 포맷을 v1.0으로 변환.

    Input:
        - label_map.json (summary)
        - labels/*.json (individual annotations)
        - images/*.jpg
        - masks/*.png (optional)

    Output:
        - annotations.json (v1.0 format)
        - images/ (unchanged)
        - masks/ (unchanged)
        - legacy/ (v0.9 백업)
    """

    def migrate(self, v09_files: List[UploadFile]) -> dict:
        """
        v0.9 데이터셋을 v1.0 포맷으로 변환.

        Returns:
            v1.0 annotations.json dict
        """
        # 1. label_map.json 파싱
        label_map = self._parse_label_map(v09_files)

        task_type = label_map['task_type']
        classes = label_map['class_summary']['classes']
        data_summary = label_map['data_summary']

        # 2. v1.0 annotations.json 생성
        annotations = {
            "format_version": "1.0",
            "dataset_id": generate_dataset_id(),
            "dataset_name": "Migrated from v0.9",

            "task_type": self._normalize_task_type(task_type),

            "created_at": datetime.utcnow().isoformat() + "Z",
            "last_modified_at": datetime.utcnow().isoformat() + "Z",
            "version": 1,
            "content_hash": None,  # 나중에 계산

            "migration_info": {
                "migrated_from": "v0.9",
                "migration_date": datetime.utcnow().isoformat() + "Z",
                "original_paths": self._extract_original_paths(data_summary)
            },

            "classes": self._convert_classes(classes),
            "splits": {},  # 나중에 계산
            "images": [],
            "statistics": {}
        }

        # 3. 개별 레이블 파일 처리
        for entry in data_summary:
            img_filename = os.path.basename(entry['img_path'])
            label_filename = os.path.basename(entry['label_path'])

            # labels/xxx.json 파싱
            label_data = self._parse_label_file(v09_files, label_filename)

            # v1.0 image entry 생성
            image_entry = self._convert_image_entry(
                img_filename=img_filename,
                label_data=label_data,
                task_type=task_type,
                entry=entry
            )

            annotations['images'].append(image_entry)

        # 4. Split 통계 계산
        annotations['splits'] = self._calculate_splits(annotations['images'])

        # 5. Content hash 계산
        annotations['content_hash'] = self._calculate_content_hash(annotations)

        return annotations

    def _normalize_task_type(self, v09_task_type: str) -> str:
        """
        v0.9 task_type → v1.0 표준 task_type.

        v0.9:
        - cls → image_classification
        - det → object_detection (or instance_segmentation)
        - seg → semantic_segmentation
        """
        mapping = {
            "cls": "image_classification",
            "det": "object_detection",  # default
            "seg": "semantic_segmentation"
        }
        return mapping.get(v09_task_type, v09_task_type)

    def _convert_classes(self, v09_classes: List[dict]) -> List[dict]:
        """
        v0.9 classes → v1.0 classes.

        v0.9: {"name": "cat", "idx": 1, "color": "#FF0000"}
        v1.0: {"id": 1, "name": "cat", "color": "#FF0000"}
        """
        return [
            {
                "id": cls['idx'],
                "name": cls['name'],
                "color": cls.get('color', '#000000')
            }
            for cls in v09_classes
            if cls['name'] != '_background_'  # 배경 클래스 제외
        ]

    def _convert_image_entry(
        self,
        img_filename: str,
        label_data: dict,
        task_type: str,
        entry: dict
    ) -> dict:
        """
        개별 이미지 + 레이블 → v1.0 image entry.
        """
        image_id = int(os.path.splitext(img_filename)[0].replace('img', ''))

        base_entry = {
            "id": image_id,
            "file_name": img_filename,
            "width": label_data.get('imageWidth', 0),
            "height": label_data.get('imageHeight', 0),
            "depth": label_data.get('imageDepth', 3),
            "split": label_data.get('split', 'train'),

            "metadata": {
                "labeled_by": "unknown",
                "labeled_at": datetime.utcnow().isoformat() + "Z",
                "source": "migrated_from_v0.9"
            }
        }

        # Task별 annotation 변환
        if task_type == "cls":
            base_entry['annotation'] = self._convert_cls_annotation(label_data)
        elif task_type == "det":
            # shape_type으로 det vs seg 구분
            if self._is_segmentation(label_data):
                base_entry['annotations'] = self._convert_seg_annotations(label_data)
            else:
                base_entry['annotations'] = self._convert_det_annotations(label_data)
        elif task_type == "seg":
            base_entry['annotation'] = self._convert_semantic_seg_annotation(label_data, entry)

        # legacy v0.9 정보 보존
        base_entry['legacy_v09'] = {
            "shapes": label_data.get('shapes', [])
        }

        return base_entry

    def _is_segmentation(self, label_data: dict) -> bool:
        """
        Detection vs Segmentation 구분.

        v0.9에서는 둘 다 task_type="det"이므로 shape_type으로 구분.
        - rectangle → object_detection
        - polygon → instance_segmentation
        """
        shapes = label_data.get('shapes', [])
        if not shapes:
            return False

        return any(s['shape_type'] == 'polygon' for s in shapes)

    def _convert_cls_annotation(self, label_data: dict) -> dict:
        """
        v0.9 classification → v1.0.

        v0.9:
        {
          "shapes": [{"label": "Cat", "points": [[0, 0]], "shape_type": "point"}]
        }

        v1.0:
        {
          "class_id": 0,
          "class_name": "Cat",
          "confidence": 1.0
        }
        """
        shapes = label_data.get('shapes', [])
        if not shapes:
            return None

        label = shapes[0]['label']

        return {
            "class_id": self._get_class_id(label),
            "class_name": label,
            "confidence": 1.0
        }

    def _convert_det_annotations(self, label_data: dict) -> List[dict]:
        """
        v0.9 detection → v1.0.

        v0.9:
        {
          "shapes": [
            {
              "label": "cat",
              "points": [[100, 150], [400, 350]],  // [top-left, bottom-right]
              "shape_type": "rectangle"
            }
          ]
        }

        v1.0:
        {
          "id": 1001,
          "class_id": 0,
          "class_name": "cat",
          "bbox": [100, 150, 300, 200],  // [x, y, w, h]
          "bbox_format": "xywh",
          "area": 60000
        }
        """
        shapes = label_data.get('shapes', [])
        annotations = []

        for i, shape in enumerate(shapes):
            if shape['shape_type'] != 'rectangle':
                continue

            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]

            w = abs(x2 - x1)
            h = abs(y2 - y1)
            x = min(x1, x2)
            y = min(y1, y2)

            annotations.append({
                "id": i + 1001,
                "class_id": self._get_class_id(shape['label']),
                "class_name": shape['label'],
                "bbox": [x, y, w, h],
                "bbox_format": "xywh",
                "area": w * h,
                "iscrowd": 0
            })

        return annotations

    def _convert_seg_annotations(self, label_data: dict) -> List[dict]:
        """
        v0.9 instance segmentation → v1.0.

        v0.9:
        {
          "shapes": [
            {
              "label": "cat",
              "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
              "shape_type": "polygon"
            }
          ]
        }

        v1.0:
        {
          "id": 2001,
          "class_id": 0,
          "class_name": "cat",
          "bbox": [x_min, y_min, w, h],
          "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],
          "area": polygon_area
        }
        """
        shapes = label_data.get('shapes', [])
        annotations = []

        for i, shape in enumerate(shapes):
            if shape['shape_type'] != 'polygon':
                continue

            points = shape['points']

            # Flatten points: [[x1, y1], [x2, y2]] → [x1, y1, x2, y2]
            flat_points = [coord for point in points for coord in point]

            # Calculate bbox from polygon
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            annotations.append({
                "id": i + 2001,
                "class_id": self._get_class_id(shape['label']),
                "class_name": shape['label'],
                "bbox": bbox,
                "segmentation": [flat_points],
                "area": self._calculate_polygon_area(points),
                "iscrowd": 0
            })

        return annotations

    def _convert_semantic_seg_annotation(self, label_data: dict, entry: dict) -> dict:
        """
        v0.9 semantic segmentation → v1.0.

        v0.9:
        {
          "mask_path": "E:/data/masks/image1_mask.png"
        }

        v1.0:
        {
          "mask_file": "masks/image1_mask.png",
          "mask_format": "indexed_png"
        }
        """
        mask_path = entry.get('mask_path', '')
        mask_filename = os.path.basename(mask_path)

        return {
            "mask_file": f"masks/{mask_filename}",
            "mask_format": "indexed_png",
            "num_classes": len(self.classes)
        }
```

---

## 마이그레이션 가이드

### 기존 사용자 시나리오

#### 시나리오 1: 로컬 v0.9 데이터셋 → 플랫폼 업로드

**사용자 상황:**
- E:/my-dataset/ 에 v0.9 포맷 데이터셋 보유
- images/, labels/, label_map.json 구조

**마이그레이션 절차:**

1. **데이터셋 압축**
   ```bash
   # 로컬에서 압축 (경로 구조 유지)
   cd E:/my-dataset
   zip -r my-dataset.zip images/ labels/ label_map.json
   ```

2. **플랫폼 업로드**
   - 웹 UI에서 "데이터셋 업로드" 클릭
   - my-dataset.zip 드래그 앤 드롭
   - ✅ 자동 감지: "v0.9 포맷이 감지되었습니다. v1.0으로 자동 변환됩니다."

3. **자동 변환 실행**
   ```
   [Backend] Detecting format... v0.9
   [Backend] Migrating v0.9 → v1.0...
   [Backend] Converting 1000 images...
   [Backend] Generating annotations.json...
   [Backend] Uploading to R2...
   [Backend] ✅ Complete!
   ```

4. **결과 확인**
   - R2에 저장된 구조:
     ```
     s3://bucket/datasets/user123-my-dataset/
     ├── annotations.json        ← v1.0 포맷
     ├── images/
     │   ├── img001.jpg
     │   └── ...
     ├── legacy/                 ← v0.9 백업
     │   ├── labels/
     │   └── label_map.json
     └── meta.json
     ```

5. **학습 시작**
   - 플랫폼에서 바로 학습 가능
   - 변환된 v1.0 포맷 사용
   - 기존 v0.9 레이블 정보는 legacy_v09 필드에 보존

#### 시나리오 2: 기존 툴 계속 사용 + 플랫폼 통합

**사용자 상황:**
- 기존 AI 검사 툴 v0.9 계속 사용 중
- 주기적으로 플랫폼 업로드 필요

**권장 워크플로우:**

1. **로컬에서 레이블링 (v0.9 툴)**
   - 기존 툴로 작업 계속
   - images/, labels/ 생성

2. **Export to Platform 기능 추가**
   - 기존 툴에 "플랫폼으로 내보내기" 버튼 추가
   - 클릭 시 자동으로:
     - v1.0 annotations.json 생성
     - API로 플랫폼 업로드
     - 로컬에는 v0.9 유지

3. **플랫폼에서 학습**
   - 업로드된 v1.0 포맷으로 학습

#### 시나리오 3: 양방향 동기화

**사용자 상황:**
- 여러 명이 협업
- 일부는 로컬 툴, 일부는 플랫폼 레이블러 사용

**해결책: Dual-Format Sync**

```python
# Export v1.0 → v0.9 (플랫폼 → 로컬)
class V10ToV09Exporter:
    """
    v1.0 annotations.json → v0.9 format.

    Use case:
    - 플랫폼에서 레이블링한 데이터를 로컬 툴로 다운로드
    """

    def export(self, annotations_v10: dict) -> dict:
        """
        v1.0 → v0.9 변환.

        Returns:
            {
                "label_map.json": {...},
                "labels/": {
                    "img001.json": {...},
                    "img002.json": {...}
                }
            }
        """
        # ... 구현
```

---

## 구현 계획

### Phase 1: Core Migration (1주)

**목표:** v0.9 포맷 자동 변환 지원

- [ ] Format Detector 구현
  - `detect_dataset_format()`
  - 지원: v1.0, v0.9, YOLO, COCO, ImageFolder

- [ ] V09ToV10Migrator 구현
  - `migrate()` 메서드
  - Classification, Detection, Segmentation 지원
  - legacy_v09 필드 보존

- [ ] Backend API 통합
  - `POST /datasets/upload` 수정
  - 자동 감지 및 변환 로직
  - R2 업로드

- [ ] Unit Tests
  - 각 task type별 변환 테스트
  - Edge case 처리

### Phase 2: UI/UX (3일)

**목표:** 사용자 친화적인 마이그레이션 경험

- [ ] Upload UI 개선
  - 포맷 자동 감지 안내 메시지
  - 변환 진행률 표시
  - 변환 결과 미리보기

- [ ] Migration Report
  - 변환된 이미지 수
  - 경로 매핑 정보
  - 경고 및 오류 로그

- [ ] Legacy Backup 다운로드
  - v0.9 포맷 다운로드 버튼
  - 기존 툴 호환성 유지

### Phase 3: Advanced Features (1주)

**목표:** 양방향 동기화 및 하이브리드 워크플로우

- [ ] V10ToV09Exporter 구현
  - 플랫폼 → 로컬 툴 export
  - label_map.json + individual labels/ 생성

- [ ] Sync API
  - `POST /datasets/{id}/sync` 엔드포인트
  - 양방향 동기화 지원
  - 충돌 해결 로직

- [ ] CLI Tool
  - 로컬 v0.9 데이터셋 → v1.0 변환 스크립트
  - 오프라인 마이그레이션 도구

### Phase 4: Documentation & Training (3일)

**목표:** 기존 사용자 온보딩

- [ ] 마이그레이션 가이드 문서
  - 시나리오별 단계별 가이드
  - 스크린샷 포함

- [ ] 비디오 튜토리얼
  - v0.9 → v1.0 마이그레이션
  - 플랫폼 레이블러 사용법

- [ ] FAQ 페이지
  - 자주 묻는 질문
  - 트러블슈팅

---

## 부록

### A. 포맷 비교 매트릭스

| Feature | v0.9 | v1.0 | 개선 사항 |
|---------|------|------|-----------|
| **파일 구조** | summary + 개별 json | 단일 annotations.json | ✅ 단순화 |
| **경로** | 절대 경로 | 상대 경로 | ✅ Cloud 호환 |
| **메타데이터** | 제한적 | 풍부 (레이블러, 리뷰어, 타임스탬프) | ✅ 협업 지원 |
| **Task 지원** | cls, det, seg | 모든 vision task | ✅ 확장성 |
| **버전 관리** | ❌ | content_hash, version | ✅ Mutable 지원 |
| **통계** | ❌ | 자동 계산 | ✅ 빠른 인덱싱 |
| **하위 호환** | N/A | legacy_v09 필드 | ✅ 정보 보존 |

### B. 변환 매핑 테이블

#### Task Type 매핑

| v0.9 | shape_type | v1.0 |
|------|------------|------|
| cls | point | image_classification |
| det | rectangle | object_detection |
| det | polygon | instance_segmentation |
| seg | N/A | semantic_segmentation |

#### 필드 매핑

| v0.9 | v1.0 | 변환 로직 |
|------|------|-----------|
| `shapes[0].label` | `annotation.class_name` | 직접 매핑 |
| `shapes[0].points` | `annotation.bbox` | 좌표 변환 |
| `imageWidth` | `width` | 직접 매핑 |
| `imageHeight` | `height` | 직접 매핑 |
| `split` | `split` | 직접 매핑 |
| `data_summary[].img_path` | `images[].file_name` | basename 추출 |

---

**Last Updated:** 2025-01-04
**Version:** 1.0 Draft
**Author:** Development Team
