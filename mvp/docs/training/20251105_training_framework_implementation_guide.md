# Training Framework Implementation Guide

**작성일**: 2025-11-05
**작성자**: Claude Code
**목적**: 새로운 training framework 추가 시 참고할 구현 가이드라인

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Pattern](#architecture-pattern)
3. [Implementation Checklist](#implementation-checklist)
4. [Component Details](#component-details)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Overview

본 문서는 ultralytics YOLO 구현 과정에서 얻은 지식을 바탕으로 작성된 training framework 구현 가이드입니다. 새로운 프레임워크(timm, huggingface 등)를 추가할 때 이 가이드라인을 따르면 일관성 있고 효율적인 구현이 가능합니다.

### Core Principles

1. **환경 격리**: Backend와 Training Service는 HTTP로만 통신
2. **프레임워크 독립성**: 각 프레임워크는 독립된 서비스로 동작
3. **데이터 접근 통일**: 모든 데이터는 R2를 통해 접근
4. **메트릭 표준화**: MLflow를 통한 실험 추적
5. **체크포인트 중앙화**: R2 Storage에 프로젝트 단위로 저장

---

## Architecture Pattern

### 1. Service Isolation Architecture

```
┌─────────────┐
│   Backend   │  FastAPI (Port 8000)
│  (SQLite)   │
└──────┬──────┘
       │ HTTP API
       ├──────────────┬──────────────┐
       │              │              │
┌──────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│   Timm      │ │Ultralytics│ │ HuggingFace│
│  Service    │ │  Service  │ │  Service   │
│ (Port 8001) │ │(Port 8002)│ │(Port 8003) │
└──────┬──────┘ └────┬─────┘ └─────┬──────┘
       │             │              │
       └─────────────┴──────────────┘
                     │
              ┌──────▼──────┐
              │  R2 Storage │
              │  (Datasets, │
              │ Checkpoints)│
              └─────────────┘
```

**왜 이렇게 설계했나?**
- 로컬 개발 환경과 Railway 배포 환경의 일치
- 프레임워크별 의존성 격리 (별도 venv)
- 독립적인 스케일링 가능
- 프레임워크 추가/제거 용이

### 2. Directory Structure

```
mvp/
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── api/
│   │   │   ├── training.py    # Training job API
│   │   │   └── test_inference.py  # Inference API
│   │   └── utils/
│   │       └── training_manager.py  # HTTP client
│   └── venv/                  # Backend virtual env
│
├── training/                   # Training Services
│   ├── api_server.py          # FastAPI server for each framework
│   ├── train.py               # Training entrypoint
│   ├── run_quick_inference.py # Inference entrypoint
│   │
│   ├── adapters/              # Framework adapters
│   │   ├── base.py            # Base adapter interface
│   │   ├── timm_adapter.py
│   │   └── ultralytics_adapter.py
│   │
│   ├── platform_sdk/          # Unified SDK
│   │   ├── __init__.py
│   │   ├── base.py            # TrainingAdapter base class
│   │   └── storage.py         # R2 storage utilities
│   │
│   ├── converters/            # Format converters
│   │   ├── dice_to_yolo.py
│   │   └── yolo_to_dice.py
│   │
│   ├── venv-timm/             # Timm virtual env
│   └── venv-ultralytics/      # Ultralytics virtual env
│
└── scripts/
    ├── setup-timm-service.bat
    ├── start-timm-service.bat
    ├── setup-ultralytics-service.bat
    └── start-ultralytics-service.bat
```

---

## Implementation Checklist

### Phase 1: Service Setup

- [ ] Create virtual environment for the framework
- [ ] Install framework-specific dependencies
- [ ] Create setup script (`setup-{framework}-service.bat`)
- [ ] Create start script (`start-{framework}-service.bat`)
- [ ] Configure service port (8001 for timm, 8002 for ultralytics, etc.)

### Phase 2: Adapter Implementation

- [ ] Create adapter class inheriting from `TrainingAdapter`
- [ ] Implement `prepare_model()` - Load pretrained or custom model
- [ ] Implement `prepare_data()` - Dataset loading and preprocessing
- [ ] Implement `train()` - Training loop with MLflow logging
- [ ] Implement `validate()` - Validation loop
- [ ] Implement `load_checkpoint()` - Load from local or R2
- [ ] Implement `run_inference()` - Single image inference

### Phase 3: API Endpoints

- [ ] Add training endpoint to `api_server.py`
- [ ] Add inference endpoint to `api_server.py`
- [ ] Add model listing endpoint
- [ ] Update `TrainingServiceClient` in Backend

### Phase 4: Data Pipeline

- [ ] Implement format converter (if needed)
- [ ] Implement train/val split logic
- [ ] Add R2 dataset download logic
- [ ] Test with sample dataset

### Phase 5: MLflow Integration

- [ ] Configure MLflow tracking URI
- [ ] Log hyperparameters
- [ ] Log metrics per epoch
- [ ] Log model artifacts
- [ ] Test MLflow UI

### Phase 6: Checkpoint Management

- [ ] Implement checkpoint saving during training
- [ ] Implement R2 upload after training completion
- [ ] Update database with R2 checkpoint paths
- [ ] Implement checkpoint download for inference

### Phase 7: Testing

- [ ] Test training with pretrained weights
- [ ] Test training with custom dataset
- [ ] Test checkpoint-based inference
- [ ] Test pretrained weight inference
- [ ] Verify R2 upload/download
- [ ] Check MLflow metrics

---

## Component Details

### 1. Platform SDK (`platform_sdk/`)

#### 1.1 TrainingAdapter Base Class

**Purpose**: 프레임워크 공통 기능 제공

**Key Methods**:
```python
class TrainingAdapter:
    def __init__(self, job_id, project_id, config, logger=None):
        self.job_id = job_id
        self.project_id = project_id
        self.config = config
        self.logger = logger or self._setup_logger()
        self.mlflow_client = self._setup_mlflow()

    # Lifecycle hooks
    def on_epoch_begin(self, epoch): pass
    def on_epoch_end(self, epoch, metrics): pass
    def on_train_begin(self): pass
    def on_train_end(self, final_metrics, checkpoint_dir): pass

    # Metric logging
    def log_metric(self, key, value, step):
        """Log to both MLflow and Database"""
        pass

    # Storage
    def download_dataset(self, r2_path, local_dir):
        """Download dataset from R2"""
        pass
```

**구현 시 주의사항**:
- `on_train_end()`에서 R2 체크포인트 업로드 수행
- `log_metric()`은 MLflow와 DB 모두에 기록
- 에러 발생 시 반드시 로깅 (프론트엔드에서 확인 가능)

#### 1.2 Storage Utilities (`storage.py`)

**R2 Operations**:
```python
def upload_checkpoint(checkpoint_path, job_id, checkpoint_name, project_id) -> bool:
    """
    Upload checkpoint to R2.

    Path: r2://bucket/checkpoints/projects/{project_id}/jobs/{job_id}/{checkpoint_name}
    Returns: True on success, False on failure (non-blocking)
    """
    pass

def download_checkpoint(checkpoint_path, dest_path=None) -> str:
    """
    Download checkpoint from R2 or verify local path.

    - If r2:// URL: Download to temp directory
    - If local path: Verify existence and return
    - Caching: Already downloaded files are reused

    Returns: Local file path
    """
    pass

def download_dataset(r2_path, local_dir) -> str:
    """
    Download dataset from R2.

    - Supports r2://bucket/datasets/... paths
    - Extracts .zip files automatically
    - Returns: Local directory path
    """
    pass
```

**Best Practices**:
- 체크포인트 업로드 실패는 non-blocking (학습은 성공으로 처리)
- 다운로드된 체크포인트는 캐싱 (재사용)
- R2 경로는 항상 `r2://` prefix 사용

### 2. Adapter Implementation

#### 2.1 Model Preparation

```python
def prepare_model(self):
    """
    Load pretrained model or initialize from scratch.

    Steps:
    1. Check if checkpoint_path is provided
    2. If yes: Load from checkpoint (via download_checkpoint)
    3. If no: Load pretrained weights from framework
    4. Configure model for task (num_classes, etc.)
    5. Move to device (GPU/CPU)
    """

    # Example (ultralytics)
    if self.checkpoint_path:
        # Custom checkpoint
        local_path = download_checkpoint(self.checkpoint_path)
        model = YOLO(local_path)
    else:
        # Pretrained
        model = YOLO('yolo11n.pt')  # Auto-download from ultralytics

    return model
```

**Pretrained Weight 관리 방법**:

| Framework | Pretrained Weights | Storage |
|-----------|-------------------|---------|
| Ultralytics | `yolo11n.pt`, `yolo11s.pt` | Auto-downloaded to `~/.cache/ultralytics/` |
| Timm | `resnet50`, `efficientnet_b0` | Auto-downloaded to `~/.cache/torch/hub/` |
| HuggingFace | `google/vit-base-patch16-224` | Auto-downloaded to `~/.cache/huggingface/` |

**프레임워크별 캐싱 경로를 활용**하면 매번 다운로드할 필요 없음.

#### 2.2 Data Preparation

```python
def prepare_data(self):
    """
    Prepare dataset for training.

    Steps:
    1. Check dataset format (DICE, YOLO, ImageFolder, etc.)
    2. If needed, convert format (e.g., DICE → YOLO)
    3. Download from R2 if not local
    4. Split into train/val (stratified if classification)
    5. Create data loader configuration
    """

    # Example
    if self.dataset_format == 'dice':
        # Convert DICE to framework format
        converter = DiceToYoloConverter(
            dice_dir=self.dataset_path,
            output_dir=self.work_dir / 'yolo_data'
        )
        yolo_dir = converter.convert(
            split_ratio=0.8,
            strategy='stratified'
        )
        self.dataset_path = yolo_dir

    # Framework-specific loader
    return create_dataloader(self.dataset_path)
```

**Data Format 전략**:

1. **DICE Format (표준)**:
   ```
   dataset/
   ├── images/
   │   ├── img1.jpg
   │   └── img2.jpg
   └── annotations/
       ├── img1.json
       └── img2.json
   ```

2. **Framework Format**:
   - YOLO: `images/`, `labels/`, `data.yaml`
   - ImageFolder: `train/class1/`, `train/class2/`, ...
   - COCO: `images/`, `annotations.json`

3. **Converter Pattern**:
   ```python
   class DiceToFrameworkConverter:
       def __init__(self, dice_dir, output_dir):
           pass

       def convert(self, split_ratio=0.8, strategy='stratified'):
           # 1. Parse DICE annotations
           # 2. Split train/val
           # 3. Write framework format
           # 4. Return output_dir
           pass
   ```

#### 2.3 Training Loop

```python
def train(self):
    """
    Main training loop with MLflow logging.
    """
    # MLflow setup
    mlflow.set_tracking_uri(self.mlflow_uri)
    mlflow.set_experiment(f"project_{self.project_id}")

    with mlflow.start_run(run_name=f"job_{self.job_id}"):
        # Log hyperparameters
        mlflow.log_params({
            'model': self.model_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
        })

        # Training loop
        for epoch in range(self.epochs):
            self.on_epoch_begin(epoch)

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate_epoch(epoch)

            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            for key, value in metrics.items():
                self.log_metric(key, value, step=epoch)

            self.on_epoch_end(epoch, metrics)

            # Save checkpoint (if best)
            if self._is_best_epoch(metrics):
                self.save_checkpoint(epoch)

        # Training complete
        checkpoint_dir = self.work_dir / 'weights'
        self.on_train_end(final_metrics, checkpoint_dir)
```

**MLflow 로깅 전략**:

```python
# Hyperparameters (한 번만)
mlflow.log_params({
    'model': 'yolo11n',
    'task': 'detect',
    'epochs': 100,
    'batch_size': 16,
})

# Metrics (매 epoch)
mlflow.log_metric('train/loss', 0.5, step=epoch)
mlflow.log_metric('val/map50', 0.75, step=epoch)

# Artifacts (학습 종료 시)
mlflow.log_artifact('runs/detect/train/weights/best.pt')
mlflow.log_artifact('runs/detect/train/results.png')
```

#### 2.4 Metric Transmission

**Database Schema**:
```sql
CREATE TABLE training_metrics (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    step INTEGER,
    loss REAL,
    accuracy REAL,
    learning_rate REAL,
    checkpoint_path TEXT,  -- R2 URL
    extra_metrics TEXT,    -- JSON for framework-specific metrics
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES training_jobs(id)
);
```

**Metric Logging**:
```python
def log_metric(self, key, value, step):
    """
    Log metric to both MLflow and Database.

    Standard fields: loss, accuracy, learning_rate
    Extra fields: stored in extra_metrics as JSON
    """
    # MLflow
    mlflow.log_metric(key, value, step=step)

    # Database
    metric_record = {
        'job_id': self.job_id,
        'epoch': step,
        'loss': value if key == 'loss' else None,
        'accuracy': value if key == 'accuracy' else None,
        'learning_rate': value if key == 'learning_rate' else None,
        'extra_metrics': {key: value} if key not in ['loss', 'accuracy', 'lr'] else None
    }

    # POST to Backend API
    requests.post(
        f"{API_URL}/training/jobs/{self.job_id}/metrics",
        json=metric_record
    )
```

**프론트엔드 표시**:
- `loss`, `accuracy`, `learning_rate`: 별도 컬럼
- 나머지: `extra_metrics` JSON 파싱하여 동적 컬럼 생성

### 3. Checkpoint Management

#### 3.1 Checkpoint Saving Strategy

**학습 중**:
```python
def _train_epoch(self, epoch):
    # ... training code ...

    # Save checkpoint if best
    if self._is_best_epoch(metrics):
        checkpoint_path = self.work_dir / 'weights' / f'best.pt'
        self.model.save(checkpoint_path)

        # Do NOT upload to R2 here (wait until training completes)
```

**학습 완료 시**:
```python
def on_train_end(self, final_metrics, checkpoint_dir):
    """
    Called after training completes.
    Upload best checkpoint to R2 and update DB.
    """
    uploaded_checkpoints = {}

    # Upload best.pt
    best_pt = checkpoint_dir / 'best.pt'
    if best_pt.exists():
        success = upload_checkpoint(
            checkpoint_path=str(best_pt),
            job_id=self.job_id,
            checkpoint_name='best.pt',
            project_id=self.project_id
        )

        if success:
            r2_path = f'r2://vision-platform-prod/checkpoints/projects/{self.project_id}/jobs/{self.job_id}/best.pt'
            best_epoch = self._find_best_epoch()
            uploaded_checkpoints[best_epoch] = r2_path

    # Update database
    if uploaded_checkpoints:
        self._update_checkpoint_paths(uploaded_checkpoints)
```

**왜 학습 완료 시 한 번만 업로드?**
- R2 API 호출 횟수 절감 (100 epoch × 2 uploads = 200 calls vs 2 calls)
- 학습 속도 저하 방지 (epoch마다 업로드 시 병목)
- 비용 절감 (Class B operations: $4.50/million)
- 중단 시에도 로컬에 checkpoint 존재 (재시작 가능)

#### 3.2 Checkpoint Path Format

**R2 Storage Structure**:
```
r2://vision-platform-prod/
├── checkpoints/
│   └── projects/
│       └── {project_id}/
│           └── jobs/
│               └── {job_id}/
│                   ├── best.pt
│                   └── last.pt
└── datasets/
    └── {dataset_id}/
        └── dataset.zip
```

**Database Storage**:
```python
# training_metrics table
checkpoint_path = "r2://vision-platform-prod/checkpoints/projects/2/jobs/20/best.pt"
```

**Inference 시 로딩**:
```python
# 1. DB에서 checkpoint_path 조회
checkpoint_path = "r2://..."

# 2. R2에서 다운로드
local_path = download_checkpoint(checkpoint_path)
# → C:/Users/.../AppData/Local/Temp/checkpoints/best.pt

# 3. 모델 로드
model.load(local_path)
```

### 4. Inference API

#### 4.1 API Design

**Endpoint**: `POST /inference/quick`

**Request** (Multipart):
```python
{
    'file': (filename, BytesIO(image_bytes), 'image/jpeg'),  # Multipart file
    'training_job_id': '20',
    'framework': 'ultralytics',
    'model_name': 'yolo11n',
    'task_type': 'object_detection',
    'num_classes': '80',
    'checkpoint_path': 'r2://...' or None,  # Optional
    'confidence_threshold': '0.25',
    'iou_threshold': '0.45',
    # ... task-specific params
}
```

**Response** (JSON):
```json
{
    "task_type": "object_detection",
    "inference_time_ms": 45.2,
    "predicted_boxes": [
        {
            "x1": 100, "y1": 50, "x2": 200, "y2": 150,
            "confidence": 0.95,
            "label": "person"
        }
    ],
    "overlay_base64": "iVBORw0KGgoAAAANS...",  // Optional
    "predicted_mask_base64": "iVBORw0KGgoAAAANS..."  // Optional
}
```

#### 4.2 Image Transmission Methods

**1. File Path (Local only)**:
```python
# Backend → Training Service
response = requests.post(url, json={
    'image_path': '/path/to/image.jpg',
    ...
})
```
❌ 배포 환경에서 작동 안 함 (컨테이너 격리)

**2. Multipart Upload (Recommended)**:
```python
# Backend
with open(image_path, 'rb') as f:
    image_bytes = f.read()

files = {
    'file': ('image.jpg', BytesIO(image_bytes), 'image/jpeg')
}
response = requests.post(url, files=files, data=params)
```
✅ 모든 환경에서 작동
✅ 메모리 기반 이미지 지원 (PIL, OpenCV)

**3. Base64 Encoding**:
```python
# Backend
with open(image_path, 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(url, json={
    'image_base64': image_base64,
    ...
})
```
✅ JSON으로 전송 가능
❌ 33% 오버헤드
❌ 큰 이미지는 payload 크기 증가

#### 4.3 Result Transmission

**결과 이미지 전송 방법**:

**Option 1: File Path (X)**:
```json
{
    "overlay_path": "/tmp/overlay.png"
}
```
❌ 컨테이너 격리 환경에서 접근 불가

**Option 2: Base64 Encoding (O)**:
```json
{
    "overlay_base64": "iVBORw0KGgoAAAANS...",
    "predicted_mask_base64": "iVBORw0KGgoAAAANS..."
}
```
✅ JSON에 직접 포함
✅ 추가 HTTP 요청 불필요
✅ 프론트엔드에서 바로 표시 가능

**Implementation**:
```python
# Training Service
import base64

# Run inference
result = model.predict(image_path)

# Encode result images
with open(result['overlay_path'], 'rb') as f:
    overlay_base64 = base64.b64encode(f.read()).decode('utf-8')

# Cleanup temp files
os.remove(result['overlay_path'])

# Return JSON
return {
    'overlay_base64': overlay_base64,
    'predicted_boxes': result['boxes'],
    ...
}
```

**프론트엔드 표시**:
```tsx
{result.overlay_base64 && (
    <img src={`data:image/png;base64,${result.overlay_base64}`} />
)}
```

### 5. Data Format Conversion

#### 5.1 DICE to Framework Format

**Converter Interface**:
```python
class DiceToFrameworkConverter:
    def __init__(self, dice_dir: str, output_dir: str):
        self.dice_dir = Path(dice_dir)
        self.output_dir = Path(output_dir)

    def convert(
        self,
        split_ratio: float = 0.8,
        strategy: str = 'stratified',  # 'stratified', 'random', 'sequential'
        split_file: Optional[str] = None  # Predefined split
    ) -> str:
        """
        Convert DICE format to framework format.

        Returns: output_dir path
        """
        pass
```

**Implementation Steps**:

1. **Parse DICE Annotations**:
```python
def _parse_dice_annotations(self):
    annotations = []
    for json_file in (self.dice_dir / 'annotations').glob('*.json'):
        with open(json_file) as f:
            data = json.load(f)

        # Extract task-specific info
        if self.task_type == 'object_detection':
            for shape in data['shapes']:
                annotations.append({
                    'image': data['imagePath'],
                    'label': shape['label'],
                    'bbox': shape['points'],  # [[x1,y1], [x2,y2]]
                })

    return annotations
```

2. **Stratified Split** (Classification):
```python
def _stratified_split(self, annotations, split_ratio):
    """
    Split by class distribution.
    Ensures each class has same train/val ratio.
    """
    from sklearn.model_selection import train_test_split

    # Group by class
    class_groups = {}
    for ann in annotations:
        label = ann['label']
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(ann)

    train_data = []
    val_data = []

    # Split each class
    for label, items in class_groups.items():
        train, val = train_test_split(
            items,
            train_size=split_ratio,
            random_state=42
        )
        train_data.extend(train)
        val_data.extend(val)

    return train_data, val_data
```

3. **Write Framework Format**:
```python
def _write_yolo_format(self, train_data, val_data):
    """
    YOLO format:
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
    """
    # Create directories
    for split in ['train', 'val']:
        (self.output_dir / 'images' / split).mkdir(parents=True)
        (self.output_dir / 'labels' / split).mkdir(parents=True)

    # Write train data
    for ann in train_data:
        # Copy image
        src = self.dice_dir / 'images' / ann['image']
        dst = self.output_dir / 'images' / 'train' / ann['image']
        shutil.copy(src, dst)

        # Write label
        label_file = dst.with_suffix('.txt')
        with open(label_file, 'w') as f:
            # YOLO format: class_id x_center y_center width height
            f.write(f"{ann['class_id']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n")

    # Write data.yaml
    with open(self.output_dir / 'data.yaml', 'w') as f:
        yaml.dump({
            'path': str(self.output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': self.class_names
        }, f)
```

#### 5.2 Framework to DICE Format

**Use Case**: 외부 툴에서 annotation 후 다시 가져오기

```python
class FrameworkToDiceConverter:
    def convert(self, framework_dir: str, output_dir: str):
        """
        Convert framework format back to DICE.
        """
        # Parse framework annotations
        annotations = self._parse_framework_annotations(framework_dir)

        # Write DICE format
        for image_name, shapes in annotations.items():
            dice_annotation = {
                'version': '1.0',
                'imagePath': image_name,
                'shapes': shapes
            }

            json_file = Path(output_dir) / 'annotations' / f'{Path(image_name).stem}.json'
            with open(json_file, 'w') as f:
                json.dump(dice_annotation, f, indent=2)
```

### 6. Train/Val Split Strategies

#### 6.1 Strategy Types

**1. Stratified Split** (분류, 클래스 불균형 데이터):
```python
strategy = 'stratified'
# 각 클래스별로 동일한 비율로 분할
# 예: dog 80개 → train 64, val 16
#     cat 20개 → train 16, val 4
```
✅ 클래스 분포 유지
✅ 소수 클래스 보호

**2. Random Split** (균형 잡힌 데이터):
```python
strategy = 'random'
# 전체 데이터를 무작위로 분할
```
✅ 간단
⚠️ 클래스 불균형 가능

**3. Sequential Split**:
```python
strategy = 'sequential'
# 앞에서부터 순서대로 분할
```
⚠️ 시간순 데이터에만 사용
⚠️ 일반적으로 비추천

**4. Predefined Split** (외부 split 파일):
```python
split_file = 'train.txt'  # 학습 이미지 목록
# train.txt:
# image1.jpg
# image2.jpg
# ...
```
✅ 재현 가능
✅ 벤치마크 데이터셋

#### 6.2 Implementation

```python
def split_dataset(
    annotations: List[Dict],
    split_ratio: float = 0.8,
    strategy: str = 'stratified',
    split_file: Optional[str] = None
):
    """
    Split dataset into train/val.
    """
    if split_file:
        # Use predefined split
        with open(split_file) as f:
            train_images = set(line.strip() for line in f)

        train_data = [ann for ann in annotations if ann['image'] in train_images]
        val_data = [ann for ann in annotations if ann['image'] not in train_images]

    elif strategy == 'stratified':
        # Stratified split
        train_data, val_data = _stratified_split(annotations, split_ratio)

    elif strategy == 'random':
        # Random split
        random.shuffle(annotations)
        split_idx = int(len(annotations) * split_ratio)
        train_data = annotations[:split_idx]
        val_data = annotations[split_idx:]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return train_data, val_data
```

### 7. MLflow Experiment Management

#### 7.1 Experiment Hierarchy

```
MLflow
└── Experiments
    ├── project_1          # 프로젝트별 experiment
    │   ├── job_10         # Training job (run)
    │   ├── job_11
    │   └── job_12
    ├── project_2
    │   ├── job_20
    │   └── job_21
    └── ...
```

**Naming Convention**:
- Experiment: `project_{project_id}`
- Run: `job_{job_id}`

#### 7.2 MLflow Logging Pattern

```python
# Setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"project_{self.project_id}")

with mlflow.start_run(run_name=f"job_{self.job_id}"):
    # 1. Log parameters (once at start)
    mlflow.log_params({
        'model': self.model_name,
        'framework': self.framework,
        'task_type': self.task_type,
        'epochs': self.epochs,
        'batch_size': self.batch_size,
        'learning_rate': self.learning_rate,
        'optimizer': 'AdamW',
        'dataset': self.dataset_path,
    })

    # 2. Log metrics (per epoch)
    for epoch in range(epochs):
        metrics = train_epoch(epoch)

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)

    # 3. Log artifacts (at end)
    mlflow.log_artifact('weights/best.pt')
    mlflow.log_artifact('results.png')
    mlflow.log_artifact('confusion_matrix.png')

    # 4. Log model (optional)
    mlflow.pytorch.log_model(model, "model")
```

#### 7.3 Metric Naming Convention

**Standard Metrics**:
- `train/loss`
- `train/accuracy`
- `val/loss`
- `val/accuracy`
- `val/map50` (object detection)
- `val/precision`
- `val/recall`
- `learning_rate`

**Framework-specific**:
- Ultralytics: `metrics/mAP50-95`, `metrics/precision`, `metrics/recall`
- Timm: `train/top1`, `train/top5`, `val/top1`, `val/top5`

### 8. Database Schema

#### 8.1 Core Tables

**training_jobs**:
```sql
CREATE TABLE training_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,

    -- Model
    framework TEXT NOT NULL,           -- 'timm', 'ultralytics', 'huggingface'
    model_name TEXT NOT NULL,          -- 'resnet50', 'yolo11n'
    task_type TEXT NOT NULL,           -- 'image_classification', 'object_detection'

    -- Dataset
    dataset_path TEXT NOT NULL,        -- R2 path or local
    dataset_format TEXT,                -- 'dice', 'yolo', 'imagefolder'
    num_classes INTEGER,

    -- Hyperparameters
    epochs INTEGER,
    batch_size INTEGER,
    learning_rate REAL,

    -- Status
    status TEXT DEFAULT 'pending',     -- 'pending', 'running', 'completed', 'failed'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,

    -- Output
    output_dir TEXT,
    mlflow_run_id TEXT,

    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

**training_metrics**:
```sql
CREATE TABLE training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,

    -- Epoch info
    epoch INTEGER NOT NULL,
    step INTEGER,

    -- Standard metrics
    loss REAL,
    accuracy REAL,
    learning_rate REAL,

    -- Checkpoint
    checkpoint_path TEXT,              -- R2 URL

    -- Extra metrics (JSON)
    extra_metrics TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (job_id) REFERENCES training_jobs(id)
);
```

**projects**:
```sql
CREATE TABLE projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

#### 8.2 Data Relationships

```
users
  └── projects (1:N)
        ├── training_jobs (1:N)
        │     └── training_metrics (1:N)
        └── checkpoints/ (R2 Storage)
              └── jobs/
                    └── {job_id}/
                          ├── best.pt
                          └── last.pt
```

---

## Best Practices

### 1. Error Handling

```python
# Training Service
try:
    # Training code
    result = adapter.train()
    return {'status': 'success', 'result': result}
except Exception as e:
    # Log error
    logger.error(f"Training failed: {e}")
    logger.error(traceback.format_exc())

    # Update job status
    update_job_status(job_id, 'failed', error_message=str(e))

    # Return error (don't crash)
    return {'status': 'error', 'message': str(e)}, 500
```

### 2. Logging

```python
# Use structured logging
logger.info(f"[TRAIN] Starting training job {job_id}")
logger.info(f"[TRAIN] Model: {model_name}, Task: {task_type}")
logger.info(f"[TRAIN] Dataset: {dataset_path}")

# Log to stdout (프론트엔드에서 확인 가능)
print(f"[EPOCH {epoch}] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
sys.stdout.flush()
```

### 3. Resource Cleanup

```python
# Training Service
@app.post("/training/start")
async def start_training(request: TrainingRequest):
    temp_dir = None
    try:
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())

        # Download dataset
        dataset_dir = download_dataset(request.dataset_path, temp_dir)

        # Training
        result = train(dataset_dir)

        return result

    finally:
        # Cleanup
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
```

### 4. Configuration Management

```python
# Use environment variables
import os

TIMM_SERVICE_URL = os.getenv('TIMM_SERVICE_URL', 'http://localhost:8001')
ULTRALYTICS_SERVICE_URL = os.getenv('ULTRALYTICS_SERVICE_URL', 'http://localhost:8002')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

R2_ENDPOINT = os.getenv('AWS_S3_ENDPOINT_URL')
R2_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
R2_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
R2_BUCKET = os.getenv('R2_BUCKET', 'vision-platform-prod')
```

### 5. Testing Strategy

```python
# 1. Unit Tests (adapters)
def test_prepare_model():
    adapter = TimmAdapter(job_id=1, project_id=1, config={...})
    model = adapter.prepare_model()
    assert model is not None

# 2. Integration Tests (format converters)
def test_dice_to_yolo_conversion():
    converter = DiceToYoloConverter('tests/fixtures/dice', '/tmp/yolo')
    output_dir = converter.convert()
    assert (Path(output_dir) / 'data.yaml').exists()

# 3. E2E Tests (full training pipeline)
def test_training_pipeline():
    response = requests.post(
        'http://localhost:8002/training/start',
        json={...}
    )
    assert response.status_code == 200
```

---

## Troubleshooting

### Common Issues

#### 1. "Module not found" in Training Service

**문제**: Backend와 Training Service가 다른 venv 사용
```
ModuleNotFoundError: No module named 'ultralytics'
```

**해결**:
```bash
# Activate correct venv
cd mvp/training
venv-ultralytics\Scripts\activate
pip install ultralytics
```

#### 2. R2 Upload Fails

**문제**: 체크포인트 업로드 실패
```
[R2 WARNING] Failed to upload checkpoint to R2: ...
```

**원인**:
- R2 credentials 미설정
- 네트워크 문제
- 파일 크기 제한

**해결**:
- 환경 변수 확인: `AWS_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- 업로드는 non-blocking이므로 학습은 성공으로 처리됨
- 로컬에 checkpoint 존재 확인

#### 3. Dataset Format Mismatch

**문제**: 프레임워크가 데이터셋 형식을 인식 못함
```
ValueError: Dataset format not recognized
```

**해결**:
- `dataset_format` 파라미터 확인
- Converter 사용: `DiceToYoloConverter`, `DiceToImageFolderConverter`
- 변환된 데이터셋 구조 확인

#### 4. MLflow Connection Failed

**문제**: MLflow 서버 연결 실패
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**해결**:
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000
```

#### 5. Checkpoint Loading from R2 Failed

**문제**: R2 checkpoint 다운로드 실패
```
FileNotFoundError: Failed to download checkpoint from R2
```

**해결**:
- R2 URL 형식 확인: `r2://bucket/checkpoints/projects/{id}/jobs/{id}/best.pt`
- R2 credentials 확인
- 체크포인트가 실제로 업로드되었는지 R2 Console에서 확인

---

## Quick Start for New Framework

### Step-by-Step Guide

```bash
# 1. Create virtual environment
cd mvp/training
python -m venv venv-{framework}
venv-{framework}\Scripts\activate

# 2. Install dependencies
pip install {framework}
pip install mlflow boto3 requests

# 3. Create adapter
# Copy from adapters/ultralytics_adapter.py
# Modify for your framework

# 4. Add API endpoint
# Edit api_server.py
# Add /training/start and /inference/quick

# 5. Create setup script
# Copy scripts/setup-ultralytics-service.bat
# Modify for your framework

# 6. Test locally
python api_server.py --framework {framework} --port 8003

# 7. Test training
curl -X POST http://localhost:8003/training/start -d '{...}'

# 8. Test inference
curl -X POST http://localhost:8003/inference/quick -F 'file=@test.jpg' -F '...'
```

---

## Appendix

### A. Environment Variables

```bash
# Backend
DATABASE_URL=sqlite:///./backend.db

# Training Services
TIMM_SERVICE_URL=http://localhost:8001
ULTRALYTICS_SERVICE_URL=http://localhost:8002
HUGGINGFACE_SERVICE_URL=http://localhost:8003

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# R2 Storage
AWS_S3_ENDPOINT_URL=https://[account-id].r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
R2_BUCKET=vision-platform-prod

# API
API_BASE_URL=http://localhost:8000/api/v1
```

### B. Port Allocation

| Service | Port | Purpose |
|---------|------|---------|
| Backend | 8000 | Main API |
| Timm | 8001 | Training/Inference |
| Ultralytics | 8002 | Training/Inference |
| HuggingFace | 8003 | Training/Inference |
| MLflow | 5000 | Experiment tracking |
| Frontend | 3000 | Next.js dev server |

### C. File Size Limits

| Item | Limit | Notes |
|------|-------|-------|
| Dataset (R2) | 5 GB | Per dataset |
| Checkpoint (R2) | 500 MB | Per checkpoint |
| Image (Inference) | 10 MB | Per image |
| Multipart Upload | 100 MB | Per request |

### D. Related Documentation

- [Ultralytics Implementation](../trainer/IMPLEMENTATION_STATUS.md)
- [Dataset Split Strategy](../datasets/20251105_093103_dataset_split_strategy.md)
- [Checkpoint Management](./20251105_checkpoint_management_and_r2_upload_policy.md)
- [Inference API](./20251105_inference_api_training_service_integration.md)

---

**Last Updated**: 2025-11-05
**Version**: 1.0
**Maintainer**: Claude Code
