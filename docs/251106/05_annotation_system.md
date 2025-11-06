# Annotation System 문서

> **작성일**: 2025-11-06
> **Version**: 1.0
> **Status**: Production

## 목차

1. [개요](#개요)
2. [지원하는 Annotation 포맷](#지원하는-annotation-포맷)
3. [Backend Annotation 처리](#backend-annotation-처리)
4. [Frontend Annotation 시각화](#frontend-annotation-시각화)
5. [Format Conversion](#format-conversion)
6. [Annotation Validation](#annotation-validation)
7. [Best Practices](#best-practices)

---

## 개요

Vision AI Training Platform은 다양한 Computer Vision 태스크를 지원하며, 각 태스크는 특정 annotation 포맷을 요구합니다.

### 지원하는 태스크 및 Annotation 타입

| Task Type | Annotation Type | 설명 |
|-----------|----------------|------|
| Image Classification | Label (클래스 이름) | 이미지 전체에 대한 단일 레이블 |
| Object Detection | Bounding Box | 객체 위치 (x, y, width, height) + 클래스 |
| Instance Segmentation | Polygon/Mask | 객체 외곽선 또는 픽셀 마스크 + 클래스 |
| Semantic Segmentation | Pixel Mask | 모든 픽셀에 대한 클래스 레이블 |
| Pose Estimation | Keypoints | 관절 좌표 (x, y, visibility) + 연결 정보 |

### 지원하는 포맷

1. **DICE** - 자체 포맷 (ImageFolder 기반)
2. **YOLO** - Ultralytics YOLO 포맷
3. **COCO** - Microsoft COCO JSON 포맷
4. **Pascal VOC** - XML 기반 포맷

---

## 지원하는 Annotation 포맷

### 1. DICE Format (Image Classification)

**디렉토리 구조**:
```
dataset/
  ├── train/
  │   ├── cat/
  │   │   ├── image1.jpg
  │   │   └── image2.jpg
  │   └── dog/
  │       ├── image3.jpg
  │       └── image4.jpg
  └── val/
      ├── cat/
      │   └── image5.jpg
      └── dog/
          └── image6.jpg
```

**Annotation 방식**:
- 폴더 이름 = 클래스 레이블
- 이미지 파일이 해당 폴더에 위치 = 암묵적 레이블링

**메타데이터**: 없음 (디렉토리 구조로 충분)

**사용 태스크**: Image Classification

---

### 2. YOLO Format (Object Detection, Segmentation, Pose)

**디렉토리 구조**:
```
dataset/
  ├── images/
  │   ├── train/
  │   │   ├── image1.jpg
  │   │   └── image2.jpg
  │   └── val/
  │       └── image3.jpg
  ├── labels/
  │   ├── train/
  │   │   ├── image1.txt
  │   │   └── image2.txt
  │   └── val/
  │       └── image3.txt
  └── data.yaml
```

#### 2.1. YOLO Detection Format

**labels/train/image1.txt**:
```
<class_id> <x_center> <y_center> <width> <height>
```

**예시**:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

**설명**:
- `class_id`: 클래스 인덱스 (0부터 시작)
- `x_center, y_center`: 바운딩 박스 중심점 (이미지 크기로 정규화, 0~1)
- `width, height`: 바운딩 박스 크기 (이미지 크기로 정규화, 0~1)

#### 2.2. YOLO Segmentation Format

**labels/train/image1.txt**:
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
```

**예시**:
```
0 0.1 0.2 0.3 0.2 0.3 0.5 0.1 0.5
```

**설명**:
- Polygon 좌표들을 나열
- 모든 좌표는 이미지 크기로 정규화 (0~1)
- 마지막 점은 자동으로 첫 점과 연결

#### 2.3. YOLO Pose Format

**labels/train/image1.txt**:
```
<class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_v> <kp2_x> <kp2_y> <kp2_v> ...
```

**예시** (17 keypoints for person):
```
0 0.5 0.5 0.3 0.6 0.45 0.2 2 0.55 0.2 2 0.5 0.3 2 ...
```

**설명**:
- `kp_x, kp_y`: Keypoint 좌표 (정규화)
- `kp_v`: Visibility (0=not labeled, 1=labeled but not visible, 2=labeled and visible)

#### 2.4. data.yaml

**data.yaml**:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

# Classes
names:
  0: cat
  1: dog
  2: bird

# Number of classes
nc: 3

# Keypoint info (pose only)
kpt_shape: [17, 3]  # [num_keypoints, 3 (x, y, visibility)]
```

---

### 3. COCO Format

**디렉토리 구조**:
```
dataset/
  ├── images/
  │   ├── train2017/
  │   │   ├── 000000000001.jpg
  │   │   └── 000000000002.jpg
  │   └── val2017/
  │       └── 000000000003.jpg
  └── annotations/
      ├── instances_train2017.json
      └── instances_val2017.json
```

#### 3.1. COCO Detection/Segmentation JSON

**instances_train2017.json**:
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "000000000001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 300],
      "area": 60000,
      "iscrowd": 0,
      "segmentation": [[x1, y1, x2, y2, x3, y3, ...]]
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "cat",
      "supercategory": "animal"
    }
  ]
}
```

**설명**:
- `bbox`: [x, y, width, height] (픽셀 좌표, 정규화 안 됨)
- `segmentation`: Polygon 좌표 리스트 (픽셀 좌표)
- `iscrowd`: 0 = instance, 1 = crowd (semantic segmentation)

#### 3.2. COCO Keypoint JSON

**person_keypoints_train2017.json**:
```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [
        411, 100, 2,
        422, 95, 2,
        0, 0, 0,
        ...
      ],
      "num_keypoints": 15,
      "bbox": [100, 50, 300, 400]
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "keypoints": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
      ],
      "skeleton": [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
      ]
    }
  ]
}
```

**설명**:
- `keypoints`: [x, y, v, x, y, v, ...] 형태의 flat array
- `skeleton`: 연결할 keypoint 인덱스 쌍 (시각화용)

---

### 4. Pascal VOC Format

**디렉토리 구조**:
```
dataset/
  ├── JPEGImages/
  │   ├── 2007_000001.jpg
  │   └── 2007_000002.jpg
  ├── Annotations/
  │   ├── 2007_000001.xml
  │   └── 2007_000002.xml
  └── ImageSets/
      └── Main/
          ├── train.txt
          └── val.txt
```

#### 4.1. Annotation XML

**Annotations/2007_000001.xml**:
```xml
<annotation>
  <folder>VOC2007</folder>
  <filename>2007_000001.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <object>
    <name>cat</name>
    <pose>Frontal</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>450</ymax>
    </bndbox>
  </object>
  <object>
    <name>dog</name>
    <bndbox>
      <xmin>350</xmin>
      <ymin>200</ymin>
      <xmax>550</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
</annotation>
```

**ImageSets/Main/train.txt**:
```
2007_000001
2007_000002
```

---

## Backend Annotation 처리

### Dataset Upload 시 Annotation 추출

**파일**: `mvp/backend/app/api/datasets.py`

```python
def analyze_dataset(dataset_dir: Path, format: str) -> dict:
    """
    데이터셋 분석: 클래스 수, 분포, annotation 통계 추출

    Returns:
        {
            "num_classes": int,
            "class_distribution": dict,
            "annotation_stats": dict,
        }
    """

    if format == "dice":
        return analyze_dice_dataset(dataset_dir)
    elif format == "yolo":
        return analyze_yolo_dataset(dataset_dir)
    elif format == "coco":
        return analyze_coco_dataset(dataset_dir)
    elif format == "voc":
        return analyze_voc_dataset(dataset_dir)
    else:
        raise ValueError(f"Unsupported format: {format}")


def analyze_yolo_dataset(dataset_dir: Path) -> dict:
    """
    YOLO 데이터셋 분석

    Returns:
        {
            "num_classes": 3,
            "class_distribution": {"cat": 150, "dog": 200, "bird": 100},
            "annotation_stats": {
                "total_images": 450,
                "total_annotations": 680,
                "avg_objects_per_image": 1.51,
            }
        }
    """
    # Read data.yaml
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError("data.yaml not found")

    import yaml
    with open(yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)

    num_classes = data_yaml["nc"]
    class_names = data_yaml["names"]

    # Count annotations
    labels_dir = dataset_dir / "labels" / "train"
    class_counts = {name: 0 for name in class_names.values()}
    total_annotations = 0

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    class_name = class_names[class_id]
                    class_counts[class_name] += 1
                    total_annotations += 1

    total_images = len(list(labels_dir.glob("*.txt")))

    return {
        "num_classes": num_classes,
        "class_distribution": class_counts,
        "annotation_stats": {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "avg_objects_per_image": total_annotations / total_images if total_images > 0 else 0,
        }
    }


def analyze_coco_dataset(dataset_dir: Path) -> dict:
    """
    COCO 데이터셋 분석
    """
    import json

    # Read annotations JSON
    ann_file = dataset_dir / "annotations" / "instances_train2017.json"
    if not ann_file.exists():
        raise FileNotFoundError("COCO annotations file not found")

    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # Extract categories
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    num_classes = len(categories)

    # Count annotations per class
    class_counts = {name: 0 for name in categories.values()}

    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        cat_name = categories[cat_id]
        class_counts[cat_name] += 1

    total_images = len(coco_data["images"])
    total_annotations = len(coco_data["annotations"])

    return {
        "num_classes": num_classes,
        "class_distribution": class_counts,
        "annotation_stats": {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "avg_objects_per_image": total_annotations / total_images if total_images > 0 else 0,
        }
    }
```

### Database 저장

**Table**: `datasets`

```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    format VARCHAR(50) NOT NULL,  -- 'dice', 'yolo', 'coco', 'voc'
    num_classes INTEGER,
    class_distribution JSONB,  -- {"cat": 150, "dog": 200}
    annotation_stats JSONB,    -- {"total_annotations": 350, "avg_objects_per_image": 1.5}
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Example Data**:
```json
{
  "id": "8ab5c6e4-0f92-4fff-8f4e-441c08d94cef",
  "format": "yolo",
  "num_classes": 3,
  "class_distribution": {
    "cat": 150,
    "dog": 200,
    "bird": 100
  },
  "annotation_stats": {
    "total_images": 450,
    "total_annotations": 680,
    "avg_objects_per_image": 1.51
  }
}
```

---

## Frontend Annotation 시각화

### Dataset Preview - Bounding Box Visualization

**컴포넌트**: `AnnotationViewer.tsx`

```typescript
import { useRef, useEffect } from 'react';

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  class_name: string;
  confidence?: number;
}

const AnnotationViewer = ({
  imageUrl,
  annotations,
  imageWidth,
  imageHeight
}: {
  imageUrl: string;
  annotations: BoundingBox[];
  imageWidth: number;
  imageHeight: number;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Load image
    const img = new Image();
    img.src = imageUrl;

    img.onload = () => {
      // Set canvas size
      canvas.width = imageWidth;
      canvas.height = imageHeight;

      // Draw image
      ctx.drawImage(img, 0, 0, imageWidth, imageHeight);

      // Draw bounding boxes
      annotations.forEach((box, idx) => {
        // Convert normalized coords to pixel coords (if needed)
        const x = box.x * imageWidth;
        const y = box.y * imageHeight;
        const w = box.width * imageWidth;
        const h = box.height * imageHeight;

        // Draw box
        ctx.strokeStyle = CLASS_COLORS[idx % CLASS_COLORS.length];
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        // Draw label background
        const label = box.class_name + (box.confidence ? ` ${(box.confidence * 100).toFixed(0)}%` : '');
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = CLASS_COLORS[idx % CLASS_COLORS.length];
        ctx.fillRect(x, y - 20, textWidth + 10, 20);

        // Draw label text
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText(label, x + 5, y - 5);
      });
    };
  }, [imageUrl, annotations, imageWidth, imageHeight]);

  return (
    <div className="annotation-viewer">
      <canvas ref={canvasRef} />
    </div>
  );
};

const CLASS_COLORS = [
  '#EF4444',  // Red
  '#10B981',  // Green
  '#3B82F6',  // Blue
  '#F59E0B',  // Yellow
  '#8B5CF6',  // Purple
  '#EC4899',  // Pink
];
```

### Keypoint Visualization

**컴포넌트**: `KeypointViewer.tsx`

```typescript
interface Keypoint {
  x: number;
  y: number;
  visibility: number;  // 0, 1, 2
  name: string;
}

interface Skeleton {
  from: number;  // keypoint index
  to: number;
}

const KeypointViewer = ({
  imageUrl,
  keypoints,
  skeleton,
  imageWidth,
  imageHeight
}: {
  imageUrl: string;
  keypoints: Keypoint[];
  skeleton: Skeleton[];
  imageWidth: number;
  imageHeight: number;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.src = imageUrl;

    img.onload = () => {
      canvas.width = imageWidth;
      canvas.height = imageHeight;

      // Draw image
      ctx.drawImage(img, 0, 0, imageWidth, imageHeight);

      // Draw skeleton connections
      skeleton.forEach((conn) => {
        const kp1 = keypoints[conn.from];
        const kp2 = keypoints[conn.to];

        if (kp1.visibility === 2 && kp2.visibility === 2) {
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(kp1.x * imageWidth, kp1.y * imageHeight);
          ctx.lineTo(kp2.x * imageWidth, kp2.y * imageHeight);
          ctx.stroke();
        }
      });

      // Draw keypoints
      keypoints.forEach((kp) => {
        const x = kp.x * imageWidth;
        const y = kp.y * imageHeight;

        if (kp.visibility === 2) {
          // Visible keypoint
          ctx.fillStyle = '#EF4444';
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
        } else if (kp.visibility === 1) {
          // Labeled but not visible
          ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    };
  }, [imageUrl, keypoints, skeleton, imageWidth, imageHeight]);

  return <canvas ref={canvasRef} />;
};
```

### Segmentation Mask Visualization

**컴포넌트**: `SegmentationViewer.tsx`

```typescript
const SegmentationViewer = ({
  imageUrl,
  maskUrl,  // URL to mask image (PNG with color-coded classes)
  alpha = 0.5
}: {
  imageUrl: string;
  maskUrl: string;
  alpha?: number;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    const mask = new Image();

    img.src = imageUrl;
    mask.src = maskUrl;

    let loadedCount = 0;

    const onLoad = () => {
      loadedCount++;
      if (loadedCount === 2) {
        canvas.width = img.width;
        canvas.height = img.height;

        // Draw original image
        ctx.drawImage(img, 0, 0);

        // Overlay mask with transparency
        ctx.globalAlpha = alpha;
        ctx.drawImage(mask, 0, 0);
        ctx.globalAlpha = 1.0;
      }
    };

    img.onload = onLoad;
    mask.onload = onLoad;
  }, [imageUrl, maskUrl, alpha]);

  return <canvas ref={canvasRef} />;
};
```

---

## Format Conversion

### DICE → YOLO (Classification to Detection)

**Use Case**: ImageFolder 포맷을 YOLO detection 포맷으로 변환 (전체 이미지를 바운딩 박스로)

**파일**: `mvp/training/platform_sdk/converters.py`

```python
def dice_to_yolo_detection(
    dice_dir: Path,
    output_dir: Path,
    image_size: Tuple[int, int] = (640, 640)
) -> None:
    """
    Convert DICE (ImageFolder) to YOLO detection format

    Creates bounding boxes covering entire images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Get class names from directories
    class_names = []
    for split in ["train", "val"]:
        split_dir = dice_dir / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir() and class_dir.name not in class_names:
                    class_names.append(class_dir.name)

    class_names.sort()
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    # Process each split
    for split in ["train", "val"]:
        split_dir = dice_dir / split
        if not split_dir.exists():
            continue

        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_id = class_to_id[class_name]

            for img_path in class_dir.glob("*.[jJ][pP][gG]"):
                # Copy image
                dest_img = images_dir / split / img_path.name
                shutil.copy(img_path, dest_img)

                # Create label (full image bounding box)
                label_path = labels_dir / split / f"{img_path.stem}.txt"
                with open(label_path, "w") as f:
                    # Full image bbox: center (0.5, 0.5), size (1.0, 1.0)
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    # Create data.yaml
    with open(output_dir / "data.yaml", "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")
```

### COCO → YOLO

**파일**: `mvp/training/platform_sdk/converters.py`

```python
def coco_to_yolo(
    coco_dir: Path,
    output_dir: Path,
    task_type: str = "detection"  # "detection", "segmentation", "pose"
) -> None:
    """
    Convert COCO format to YOLO format
    """
    import json
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

        # Load COCO annotations
        if task_type == "detection":
            ann_file = coco_dir / "annotations" / f"instances_{split}2017.json"
        elif task_type == "pose":
            ann_file = coco_dir / "annotations" / f"person_keypoints_{split}2017.json"
        else:
            ann_file = coco_dir / "annotations" / f"instances_{split}2017.json"

        if not ann_file.exists():
            continue

        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        # Build category mapping
        categories = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}
        category_names = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

        # Group annotations by image_id
        anns_by_image = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)

        # Process each image
        for img_info in coco_data["images"]:
            img_id = img_info["id"]
            img_file = img_info["file_name"]
            img_width = img_info["width"]
            img_height = img_info["height"]

            # Copy image
            src_img = coco_dir / "images" / f"{split}2017" / img_file
            dest_img = images_dir / split / img_file
            shutil.copy(src_img, dest_img)

            # Create label file
            label_path = labels_dir / split / f"{Path(img_file).stem}.txt"

            with open(label_path, "w") as f:
                if img_id not in anns_by_image:
                    continue

                for ann in anns_by_image[img_id]:
                    cat_id = categories[ann["category_id"]]

                    if task_type == "detection":
                        # Convert bbox: [x, y, w, h] -> [x_center, y_center, w, h] (normalized)
                        x, y, w, h = ann["bbox"]
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        w_norm = w / img_width
                        h_norm = h / img_height

                        f.write(f"{cat_id} {x_center} {y_center} {w_norm} {h_norm}\n")

                    elif task_type == "segmentation":
                        # Convert segmentation polygon
                        if "segmentation" in ann and len(ann["segmentation"]) > 0:
                            polygon = ann["segmentation"][0]  # Take first polygon
                            # Normalize coordinates
                            normalized = []
                            for i in range(0, len(polygon), 2):
                                x_norm = polygon[i] / img_width
                                y_norm = polygon[i + 1] / img_height
                                normalized.extend([x_norm, y_norm])

                            # Write: class_id x1 y1 x2 y2 ...
                            coords_str = " ".join([f"{c:.6f}" for c in normalized])
                            f.write(f"{cat_id} {coords_str}\n")

                    elif task_type == "pose":
                        # Convert keypoints
                        keypoints = ann["keypoints"]  # [x, y, v, x, y, v, ...]
                        bbox = ann["bbox"]

                        # Bounding box (normalized)
                        x, y, w, h = bbox
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        w_norm = w / img_width
                        h_norm = h / img_height

                        # Keypoints (normalized)
                        kp_normalized = []
                        for i in range(0, len(keypoints), 3):
                            kp_x = keypoints[i] / img_width
                            kp_y = keypoints[i + 1] / img_height
                            kp_v = keypoints[i + 2]
                            kp_normalized.extend([kp_x, kp_y, kp_v])

                        kp_str = " ".join([f"{k:.6f}" if isinstance(k, float) else str(k) for k in kp_normalized])
                        f.write(f"{cat_id} {x_center} {y_center} {w_norm} {h_norm} {kp_str}\n")

    # Create data.yaml
    class_names = [category_names[cat_id] for cat_id in sorted(category_names.keys())]

    with open(output_dir / "data.yaml", "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")

        if task_type == "pose":
            # Add keypoint info
            num_keypoints = len(coco_data["categories"][0]["keypoints"])
            f.write(f"kpt_shape: [{num_keypoints}, 3]\n")
```

---

## Annotation Validation

### Backend Validation

**파일**: `mvp/backend/app/utils/validators.py`

```python
def validate_yolo_annotation(
    label_path: Path,
    num_classes: int,
    task_type: str = "detection"
) -> Tuple[bool, str]:
    """
    Validate YOLO annotation file

    Returns:
        (is_valid, error_message)
    """
    try:
        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()

                if len(parts) == 0:
                    continue

                # Check class_id
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= num_classes:
                        return False, f"Line {line_num}: Invalid class_id {class_id} (must be 0-{num_classes-1})"
                except ValueError:
                    return False, f"Line {line_num}: class_id must be integer"

                if task_type == "detection":
                    # Detection: class_id x_center y_center width height
                    if len(parts) != 5:
                        return False, f"Line {line_num}: Expected 5 values, got {len(parts)}"

                    # Check coordinates are in range [0, 1]
                    for i, coord in enumerate(parts[1:5]):
                        try:
                            val = float(coord)
                            if val < 0 or val > 1:
                                return False, f"Line {line_num}: Coordinate {i+1} out of range [0, 1]: {val}"
                        except ValueError:
                            return False, f"Line {line_num}: Invalid coordinate: {coord}"

                elif task_type == "segmentation":
                    # Segmentation: class_id x1 y1 x2 y2 x3 y3 ...
                    if len(parts) < 7:  # At least 3 points (triangle)
                        return False, f"Line {line_num}: Segmentation needs at least 6 coordinates (3 points)"

                    if (len(parts) - 1) % 2 != 0:
                        return False, f"Line {line_num}: Segmentation coordinates must be pairs (x, y)"

                    # Validate all coordinates
                    for coord in parts[1:]:
                        try:
                            val = float(coord)
                            if val < 0 or val > 1:
                                return False, f"Line {line_num}: Coordinate out of range: {val}"
                        except ValueError:
                            return False, f"Line {line_num}: Invalid coordinate: {coord}"

        return True, ""

    except Exception as e:
        return False, f"Failed to read file: {str(e)}"


def validate_coco_json(json_path: Path) -> Tuple[bool, str]:
    """
    Validate COCO JSON structure
    """
    import json

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Check required fields
        required_fields = ["images", "annotations", "categories"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Validate images
        if not isinstance(data["images"], list):
            return False, "images must be a list"

        for img in data["images"]:
            if "id" not in img or "file_name" not in img:
                return False, "Each image must have 'id' and 'file_name'"

        # Validate categories
        if not isinstance(data["categories"], list):
            return False, "categories must be a list"

        category_ids = set()
        for cat in data["categories"]:
            if "id" not in cat or "name" not in cat:
                return False, "Each category must have 'id' and 'name'"
            category_ids.add(cat["id"])

        # Validate annotations
        image_ids = {img["id"] for img in data["images"]}

        for ann in data["annotations"]:
            if "id" not in ann or "image_id" not in ann or "category_id" not in ann:
                return False, "Each annotation must have 'id', 'image_id', and 'category_id'"

            if ann["image_id"] not in image_ids:
                return False, f"Annotation references non-existent image_id: {ann['image_id']}"

            if ann["category_id"] not in category_ids:
                return False, f"Annotation references non-existent category_id: {ann['category_id']}"

        return True, ""

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"
```

---

## Best Practices

### 1. Annotation 품질 관리

**권장사항**:
- 모든 업로드된 데이터셋에 대해 자동 validation 수행
- 잘못된 annotation 감지 시 사용자에게 경고 메시지 표시
- 클래스 분포 불균형 감지 및 알림

**예시 코드**:
```python
# Check class imbalance
class_counts = dataset_info["class_distribution"]
max_count = max(class_counts.values())
min_count = min(class_counts.values())

if max_count / min_count > 10:
    warnings.append(f"Class imbalance detected: {max_count/min_count:.1f}x difference")
```

### 2. 좌표계 정규화

**YOLO 포맷 규칙**:
- 모든 좌표는 이미지 크기로 나눈 값 (0~1 범위)
- `x_center = (x_pixel + width_pixel / 2) / image_width`
- `y_center = (y_pixel + height_pixel / 2) / image_height`

**COCO 포맷 규칙**:
- 픽셀 좌표 그대로 사용 (정규화 안 함)
- `bbox = [x_pixel, y_pixel, width_pixel, height_pixel]`

### 3. Annotation 파일 구조 검증

**데이터셋 업로드 시 체크리스트**:
- ✅ 모든 이미지에 대응하는 annotation 파일 존재
- ✅ Annotation 파일의 class_id가 유효 범위 내
- ✅ 좌표값이 valid range 내 (YOLO: 0~1, COCO: 0~image_size)
- ✅ data.yaml/JSON 파일이 올바른 형식
- ✅ 클래스 이름에 특수문자 없음

### 4. Format Conversion Best Practices

**변환 시 주의사항**:
- 원본 데이터 백업 유지
- 변환 후 샘플 데이터 수동 검증
- 변환 로그 저장 (어떤 이미지가 변환되었는지)

**로그 예시**:
```
[INFO] Converting COCO to YOLO
[INFO] Found 1000 images in train split
[INFO] Found 500 images in val split
[INFO] Converted 3500 bounding boxes
[INFO] Created data.yaml with 10 classes
[INFO] Conversion complete: /path/to/output
```

### 5. Frontend Visualization Performance

**대용량 데이터셋 처리**:
- Canvas 크기 제한 (max 1920x1080)
- 한 번에 표시하는 annotation 수 제한 (max 100개)
- Pagination 적용

**코드 예시**:
```typescript
// Limit annotations to prevent performance issues
const visibleAnnotations = annotations.slice(0, 100);

if (annotations.length > 100) {
  console.warn(`Showing first 100 of ${annotations.length} annotations`);
}
```

---

## 참고 문서

- [YOLO Format Documentation](https://docs.ultralytics.com/datasets/)
- [COCO Format Specification](https://cocodataset.org/#format-data)
- [Pascal VOC Format](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Backend API 명세서](./01_backend_api_specification.md)
- [User Flow Scenarios](./04_user_flow_scenarios.md)
