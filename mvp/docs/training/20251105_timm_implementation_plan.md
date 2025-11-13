# Timm Framework Implementation Plan

**작성일**: 2025-11-05
**목표**: Image Classification을 위한 timm 프레임워크 구현
**참고**: [Training Framework Implementation Guide](./20251105_training_framework_implementation_guide.md)

---

## Overview

timm (PyTorch Image Models)은 최신 이미지 분류 모델을 제공하는 라이브러리입니다. 이번 구현에서는 ultralytics 구현 경험을 바탕으로 효율적으로 진행합니다.

### Target Models

- **ResNet Family**: `resnet18`, `resnet34`, `resnet50`, `resnet101`
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- **Vision Transformer**: `vit_base_patch16_224`, `vit_small_patch16_224`
- **MobileNet**: `mobilenetv3_large_100`, `mobilenetv3_small_100`

### Task Types

- `image_classification` (Primary)
- `multi_label_classification` (Future)

---

## Implementation Phases

### Phase 1: Environment Setup (30분)

#### 1.1 Create Virtual Environment

```bash
cd mvp/training
python -m venv venv-timm
venv-timm\Scripts\activate
```

#### 1.2 Install Dependencies

**Requirements** (`requirements-timm.txt`):
```txt
# Core
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0

# Training
mlflow>=2.8.0
tensorboard>=2.14.0

# Data
albumentations>=1.3.0
pillow>=10.0.0

# Storage
boto3>=1.28.0

# API
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

# Utils
pyyaml>=6.0
tqdm>=4.66.0
```

```bash
pip install -r requirements-timm.txt
```

#### 1.3 Create Setup Script

**File**: `mvp/scripts/setup-timm-service.bat`

```batch
@echo off
echo Setting up Timm Training Service...

cd %~dp0..\training

:: Check if venv exists
if exist venv-timm (
    echo Virtual environment already exists.
    echo To recreate, delete venv-timm folder first.
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv-timm

:: Activate and install dependencies
echo Installing dependencies...
call venv-timm\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements-timm.txt

echo.
echo ========================================
echo Timm Training Service setup complete!
echo ========================================
echo.
echo To start the service, run:
echo   scripts\start-timm-service.bat
echo.
pause
```

#### 1.4 Create Start Script

**File**: `mvp/scripts/start-timm-service.bat`

```batch
@echo off
echo Starting Timm Training Service on port 8001...

cd %~dp0..\training

:: Check if venv exists
if not exist venv-timm (
    echo Error: Virtual environment not found.
    echo Please run setup-timm-service.bat first.
    pause
    exit /b 1
)

:: Activate venv
call venv-timm\Scripts\activate.bat

:: Start service
echo Starting FastAPI server...
python api_server.py --framework timm --port 8001

pause
```

**Deliverables**:
- ✅ `venv-timm/` directory
- ✅ `requirements-timm.txt`
- ✅ `scripts/setup-timm-service.bat`
- ✅ `scripts/start-timm-service.bat`

---

### Phase 2: Data Format Converter (1시간)

#### 2.1 DICE to ImageFolder Converter

**File**: `mvp/training/converters/dice_to_imagefolder.py`

**ImageFolder Structure**:
```
imagefolder_dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    ├── class1/
    │   └── img5.jpg
    └── class2/
        └── img6.jpg
```

**Implementation**:
```python
"""
DICE to ImageFolder Converter for timm classification
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random


class DiceToImageFolderConverter:
    """
    Convert DICE annotation format to ImageFolder format for timm.

    DICE format:
        dataset/
        ├── images/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── annotations/
            ├── img1.json
            └── img2.json

    ImageFolder format:
        output/
        ├── train/
        │   ├── class1/
        │   │   └── img1.jpg
        │   └── class2/
        │       └── img2.jpg
        └── val/
            ├── class1/
            │   └── img3.jpg
            └── class2/
                └── img4.jpg
    """

    def __init__(self, dice_dir: str, output_dir: str):
        self.dice_dir = Path(dice_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.dice_dir / 'images'
        self.annotations_dir = self.dice_dir / 'annotations'

        # Validate input
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")

    def convert(
        self,
        split_ratio: float = 0.8,
        strategy: str = 'stratified',
        split_file: Optional[str] = None,
        random_seed: int = 42
    ) -> str:
        """
        Convert DICE to ImageFolder format with train/val split.

        Args:
            split_ratio: Train/val split ratio (default: 0.8)
            strategy: Split strategy ('stratified', 'random', 'sequential')
            split_file: Optional path to file with predefined train images
            random_seed: Random seed for reproducibility

        Returns:
            Path to output directory
        """
        print(f"[DICE→ImageFolder] Starting conversion")
        print(f"[DICE→ImageFolder] Input: {self.dice_dir}")
        print(f"[DICE→ImageFolder] Output: {self.output_dir}")

        # Parse DICE annotations
        annotations = self._parse_dice_annotations()
        print(f"[DICE→ImageFolder] Found {len(annotations)} annotated images")

        # Get class distribution
        class_counts = self._get_class_distribution(annotations)
        print(f"[DICE→ImageFolder] Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  - {class_name}: {count} images")

        # Split into train/val
        if split_file:
            print(f"[DICE→ImageFolder] Using predefined split from: {split_file}")
            train_data, val_data = self._split_by_file(annotations, split_file)
        elif strategy == 'stratified':
            print(f"[DICE→ImageFolder] Using stratified split (ratio: {split_ratio})")
            train_data, val_data = self._stratified_split(annotations, split_ratio, random_seed)
        elif strategy == 'random':
            print(f"[DICE→ImageFolder] Using random split (ratio: {split_ratio})")
            train_data, val_data = self._random_split(annotations, split_ratio, random_seed)
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")

        print(f"[DICE→ImageFolder] Train: {len(train_data)} images")
        print(f"[DICE→ImageFolder] Val: {len(val_data)} images")

        # Write ImageFolder format
        self._write_imagefolder(train_data, val_data)

        print(f"[DICE→ImageFolder] Conversion complete!")
        return str(self.output_dir)

    def _parse_dice_annotations(self) -> List[Dict]:
        """Parse DICE annotations and extract class labels."""
        annotations = []

        for json_file in self.annotations_dir.glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_name = data.get('imagePath')
            if not image_name:
                print(f"[WARNING] No imagePath in {json_file.name}, skipping")
                continue

            # Extract class label from shapes
            shapes = data.get('shapes', [])
            if not shapes:
                print(f"[WARNING] No shapes in {json_file.name}, skipping")
                continue

            # For classification, use the label of the first shape
            # (assuming single-label classification)
            class_label = shapes[0].get('label')
            if not class_label:
                print(f"[WARNING] No label in {json_file.name}, skipping")
                continue

            # Verify image exists
            image_path = self.images_dir / image_name
            if not image_path.exists():
                print(f"[WARNING] Image not found: {image_name}, skipping")
                continue

            annotations.append({
                'image_name': image_name,
                'class_label': class_label,
                'image_path': image_path
            })

        return annotations

    def _get_class_distribution(self, annotations: List[Dict]) -> Dict[str, int]:
        """Get class distribution."""
        class_counts = defaultdict(int)
        for ann in annotations:
            class_counts[ann['class_label']] += 1
        return dict(class_counts)

    def _stratified_split(
        self,
        annotations: List[Dict],
        split_ratio: float,
        random_seed: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Stratified split: maintain class distribution in train/val.
        """
        random.seed(random_seed)

        # Group by class
        class_groups = defaultdict(list)
        for ann in annotations:
            class_groups[ann['class_label']].append(ann)

        train_data = []
        val_data = []

        # Split each class
        for class_label, items in class_groups.items():
            # Shuffle within class
            random.shuffle(items)

            # Split
            split_idx = int(len(items) * split_ratio)
            train_data.extend(items[:split_idx])
            val_data.extend(items[split_idx:])

        # Shuffle combined data
        random.shuffle(train_data)
        random.shuffle(val_data)

        return train_data, val_data

    def _random_split(
        self,
        annotations: List[Dict],
        split_ratio: float,
        random_seed: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Random split."""
        random.seed(random_seed)
        random.shuffle(annotations)

        split_idx = int(len(annotations) * split_ratio)
        return annotations[:split_idx], annotations[split_idx:]

    def _split_by_file(
        self,
        annotations: List[Dict],
        split_file: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split by predefined file."""
        with open(split_file, 'r') as f:
            train_images = set(line.strip() for line in f)

        train_data = [ann for ann in annotations if ann['image_name'] in train_images]
        val_data = [ann for ann in annotations if ann['image_name'] not in train_images]

        return train_data, val_data

    def _write_imagefolder(self, train_data: List[Dict], val_data: List[Dict]):
        """Write ImageFolder format."""
        # Create directory structure
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Write train data
        print(f"[DICE→ImageFolder] Writing train split...")
        for ann in train_data:
            class_dir = train_dir / ann['class_label']
            class_dir.mkdir(exist_ok=True)

            src = ann['image_path']
            dst = class_dir / ann['image_name']
            shutil.copy2(src, dst)

        # Write val data
        print(f"[DICE→ImageFolder] Writing val split...")
        for ann in val_data:
            class_dir = val_dir / ann['class_label']
            class_dir.mkdir(exist_ok=True)

            src = ann['image_path']
            dst = class_dir / ann['image_name']
            shutil.copy2(src, dst)

        # Write class names file
        class_names = sorted(set(ann['class_label'] for ann in train_data + val_data))
        class_file = self.output_dir / 'classes.txt'
        with open(class_file, 'w') as f:
            f.write('\n'.join(class_names))

        print(f"[DICE→ImageFolder] Wrote {len(class_names)} classes to {class_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert DICE to ImageFolder format')
    parser.add_argument('--dice-dir', required=True, help='DICE dataset directory')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--strategy', default='stratified', choices=['stratified', 'random', 'sequential'])
    parser.add_argument('--split-file', help='Predefined split file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    converter = DiceToImageFolderConverter(args.dice_dir, args.output_dir)
    output_dir = converter.convert(
        split_ratio=args.split_ratio,
        strategy=args.strategy,
        split_file=args.split_file,
        random_seed=args.seed
    )

    print(f"\nConversion complete! Dataset ready at: {output_dir}")
```

**Deliverables**:
- ✅ `converters/dice_to_imagefolder.py`
- ✅ Stratified split 지원
- ✅ Class distribution 유지

---

### Phase 3: TimmAdapter Implementation (2시간)

#### 3.1 Adapter Structure

**File**: `mvp/training/adapters/timm_adapter.py`

**Key Methods**:

```python
class TimmAdapter(TrainingAdapter):
    """Adapter for timm image classification models."""

    def __init__(self, job_id, project_id, config, logger=None):
        super().__init__(job_id, project_id, config, logger)

        # Timm-specific config
        self.model_name = config['model_name']  # 'resnet50'
        self.num_classes = config['num_classes']
        self.pretrained = config.get('pretrained', True)
        self.img_size = config.get('img_size', 224)

        # Training config
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config.get('weight_decay', 0.0001)

        # Augmentation
        self.use_aug = config.get('use_augmentation', True)

    def prepare_model(self):
        """Load timm model."""
        import timm

        if self.checkpoint_path:
            # Load from checkpoint
            from platform_sdk.storage import download_checkpoint
            local_path = download_checkpoint(self.checkpoint_path)

            # Create model architecture
            model = timm.create_model(
                self.model_name,
                pretrained=False,
                num_classes=self.num_classes
            )

            # Load weights
            checkpoint = torch.load(local_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            self.logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
        else:
            # Create with pretrained weights
            model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=self.num_classes
            )

            self.logger.info(f"Created model {self.model_name} (pretrained={self.pretrained})")

        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        self.model = model
        self.device = device

        return model

    def prepare_data(self):
        """Prepare dataloaders."""
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Convert DICE to ImageFolder if needed
        if self.dataset_format == 'dice':
            from converters.dice_to_imagefolder import DiceToImageFolderConverter

            converter = DiceToImageFolderConverter(
                dice_dir=self.dataset_path,
                output_dir=str(self.work_dir / 'imagefolder_data')
            )
            imagefolder_dir = converter.convert(
                split_ratio=0.8,
                strategy='stratified'
            )
            self.dataset_path = imagefolder_dir

        # Data augmentation
        if self.use_aug:
            train_transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        train_dataset = datasets.ImageFolder(
            root=Path(self.dataset_path) / 'train',
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=Path(self.dataset_path) / 'val',
            transform=val_transform
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = train_dataset.classes

        self.logger.info(f"Train dataset: {len(train_dataset)} images, {len(self.class_names)} classes")
        self.logger.info(f"Val dataset: {len(val_dataset)} images")

        return train_loader, val_loader

    def train(self):
        """Main training loop."""
        import mlflow

        # Setup
        self.prepare_model()
        train_loader, val_loader = self.prepare_data()

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs
        )

        criterion = torch.nn.CrossEntropyLoss()

        # MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(f"project_{self.project_id}")

        with mlflow.start_run(run_name=f"job_{self.job_id}"):
            # Log params
            mlflow.log_params({
                'model': self.model_name,
                'num_classes': self.num_classes,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'img_size': self.img_size,
                'pretrained': self.pretrained,
            })

            # Training loop
            best_acc = 0.0
            for epoch in range(self.epochs):
                self.on_epoch_begin(epoch)

                # Train
                train_metrics = self._train_epoch(epoch, train_loader, optimizer, criterion)

                # Validate
                val_metrics = self._validate_epoch(epoch, val_loader, criterion)

                # Scheduler step
                scheduler.step()

                # Log metrics
                metrics = {
                    **train_metrics,
                    **val_metrics,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }

                for key, value in metrics.items():
                    self.log_metric(key, value, step=epoch)

                # Save best checkpoint
                if val_metrics['val/accuracy'] > best_acc:
                    best_acc = val_metrics['val/accuracy']
                    self._save_checkpoint(epoch, optimizer, metrics)

                self.on_epoch_end(epoch, metrics)

            # Training complete
            checkpoint_dir = self.work_dir / 'checkpoints'
            final_metrics = {'best_accuracy': best_acc}
            self.on_train_end(final_metrics, checkpoint_dir)

        return {'status': 'success', 'best_accuracy': best_acc}

    def _train_epoch(self, epoch, train_loader, optimizer, criterion):
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log progress
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch}] Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {
            'train/loss': avg_loss,
            'train/accuracy': accuracy
        }

    def _validate_epoch(self, epoch, val_loader, criterion):
        """Validate one epoch."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        print(f"[Epoch {epoch}] Val Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

        return {
            'val/loss': avg_loss,
            'val/accuracy': accuracy
        }

    def _save_checkpoint(self, epoch, optimizer, metrics):
        """Save checkpoint."""
        checkpoint_dir = self.work_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / 'best.pt'

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }, checkpoint_path)

        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def run_inference(self, image_path: str, top_k: int = 5):
        """Run inference on single image."""
        from PIL import Image
        from torchvision import transforms

        # Prepare model
        if not hasattr(self, 'model'):
            self.prepare_model()

        self.model.eval()

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Top-K predictions
        top_probs, top_indices = probs.topk(top_k)

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class_id': int(idx),
                'class_name': self.class_names[idx] if hasattr(self, 'class_names') else f'class_{idx}',
                'confidence': float(prob)
            })

        return {
            'task_type': 'image_classification',
            'top_k_predictions': predictions,
            'predicted_class': predictions[0]['class_name'],
            'confidence': predictions[0]['confidence']
        }
```

**Deliverables**:
- ✅ `adapters/timm_adapter.py`
- ✅ ImageFolder format 지원
- ✅ Pretrained weights 자동 다운로드
- ✅ Data augmentation
- ✅ Top-K inference

---

### Phase 4: API Endpoints (1시간)

#### 4.1 Update api_server.py

**Add timm routes**:

```python
# In api_server.py

@app.post("/training/start")
async def start_training(request: TrainingRequest):
    """Start training job (timm framework)."""
    if request.framework != 'timm':
        return {'error': 'This service only handles timm framework'}, 400

    try:
        # Import adapter
        from adapters.timm_adapter import TimmAdapter

        # Create adapter
        adapter = TimmAdapter(
            job_id=request.training_job_id,
            project_id=request.project_id,
            config={
                'model_name': request.model_name,
                'num_classes': request.num_classes,
                'dataset_path': request.dataset_path,
                'dataset_format': request.dataset_format,
                'epochs': request.epochs,
                'batch_size': request.batch_size,
                'learning_rate': request.learning_rate,
                'checkpoint_path': request.checkpoint_path,
                'img_size': request.img_size or 224,
            }
        )

        # Train
        result = adapter.train()

        return {'status': 'success', 'result': result}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}, 500


@app.post("/inference/quick")
async def quick_inference(
    file: UploadFile = File(...),
    training_job_id: int = Form(...),
    model_name: str = Form(...),
    num_classes: int = Form(...),
    checkpoint_path: Optional[str] = Form(None),
    top_k: int = Form(5),
):
    """Quick inference with timm model."""
    import uuid
    import tempfile
    import os

    # Save uploaded image
    temp_dir = tempfile.gettempdir()
    temp_image_path = os.path.join(temp_dir, f"inference_{uuid.uuid4()}.jpg")

    try:
        # Write uploaded file
        with open(temp_image_path, "wb") as f:
            f.write(await file.read())

        # Create adapter
        from adapters.timm_adapter import TimmAdapter

        adapter = TimmAdapter(
            job_id=training_job_id,
            project_id=1,  # Dummy for inference
            config={
                'model_name': model_name,
                'num_classes': num_classes,
                'checkpoint_path': checkpoint_path,
                'img_size': 224,
            }
        )

        # Run inference
        result = adapter.run_inference(temp_image_path, top_k=top_k)

        return result

    finally:
        # Cleanup
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
```

**Deliverables**:
- ✅ Training endpoint
- ✅ Inference endpoint
- ✅ Multipart file upload

---

### Phase 5: Backend Integration (30분)

#### 5.1 Update TrainingServiceClient

**File**: `mvp/backend/app/utils/training_manager.py`

```python
# Add timm service URL
TIMM_SERVICE_URL = os.getenv('TIMM_SERVICE_URL', 'http://localhost:8001')

class TrainingServiceClient:
    def __init__(self, framework: str):
        self.framework = framework

        if framework == 'timm':
            self.base_url = TIMM_SERVICE_URL
        elif framework == 'ultralytics':
            self.base_url = ULTRALYTICS_SERVICE_URL
        else:
            raise ValueError(f"Unknown framework: {framework}")
```

**Deliverables**:
- ✅ Timm service URL 추가
- ✅ Framework routing

---

### Phase 6: Testing (1시간)

#### 6.1 Prepare Test Dataset

**Create sample ImageNet-style dataset**:
```
test_data/
├── train/
│   ├── cat/
│   │   ├── cat1.jpg
│   │   └── cat2.jpg
│   └── dog/
│       ├── dog1.jpg
│       └── dog2.jpg
└── val/
    ├── cat/
    │   └── cat3.jpg
    └── dog/
        └── dog3.jpg
```

#### 6.2 Test Training

```bash
# Start timm service
cd mvp/scripts
start-timm-service.bat

# In another terminal, test training
curl -X POST http://localhost:8001/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "training_job_id": 100,
    "project_id": 1,
    "framework": "timm",
    "model_name": "resnet18",
    "num_classes": 2,
    "dataset_path": "C:/test_data",
    "dataset_format": "imagefolder",
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.001
  }'
```

#### 6.3 Test Inference

```bash
# Pretrained inference
curl -X POST http://localhost:8001/inference/quick \
  -F "file=@test.jpg" \
  -F "training_job_id=100" \
  -F "model_name=resnet18" \
  -F "num_classes=1000" \
  -F "top_k=5"

# Fine-tuned inference
curl -X POST http://localhost:8001/inference/quick \
  -F "file=@test.jpg" \
  -F "training_job_id=100" \
  -F "model_name=resnet18" \
  -F "num_classes=2" \
  -F "checkpoint_path=r2://bucket/checkpoints/projects/1/jobs/100/best.pt" \
  -F "top_k=5"
```

**Test Checklist**:
- [ ] Training with ImageFolder format
- [ ] Training with DICE format (auto-conversion)
- [ ] Pretrained weight inference (ImageNet 1000 classes)
- [ ] Fine-tuned model inference (custom classes)
- [ ] MLflow experiment tracking
- [ ] Checkpoint upload to R2
- [ ] Checkpoint download for inference
- [ ] Frontend integration

---

## Key Differences from Ultralytics

### 1. Data Format

| Aspect | Ultralytics | Timm |
|--------|-------------|------|
| Primary format | YOLO (images/, labels/, data.yaml) | ImageFolder (train/class/, val/class/) |
| Annotation | Bounding boxes in .txt | Directory structure = labels |
| Converter | DICE → YOLO | DICE → ImageFolder |
| Split strategy | Stratified by class & annotations | Stratified by class count |

### 2. Model Loading

| Aspect | Ultralytics | Timm |
|--------|-------------|------|
| Pretrained | `YOLO('yolo11n.pt')` | `timm.create_model('resnet50', pretrained=True)` |
| Custom checkpoint | `YOLO(checkpoint_path)` | Load state_dict to model |
| Cache location | `~/.cache/ultralytics/` | `~/.cache/torch/hub/` |

### 3. Training API

| Aspect | Ultralytics | Timm |
|--------|-------------|------|
| Training call | `model.train(data=..., epochs=...)` | Manual training loop |
| Metrics | Built-in (mAP, precision, recall) | Manual calculation (loss, accuracy) |
| Callbacks | Built-in callback system | Custom hooks |

### 4. Inference

| Aspect | Ultralytics | Timm |
|--------|-------------|------|
| Input | Image path or PIL | Image path or PIL |
| Output | Boxes, masks, keypoints | Class probabilities |
| Visualization | Built-in (overlay, masks) | Manual (no visualization) |

---

## Expected Outcomes

### Supported Models

After implementation, the platform will support:

**Timm** (Port 8001):
- ResNet: `resnet18`, `resnet50`
- EfficientNet: `efficientnet_b0`
- ViT: `vit_base_patch16_224`

**Ultralytics** (Port 8002):
- YOLO: `yolo11n`, `yolo11n-seg`, `yolo11n-pose`

### Task Coverage

| Task | Framework | Status |
|------|-----------|--------|
| Image Classification | Timm | ✅ Phase 7 |
| Object Detection | Ultralytics | ✅ Implemented |
| Instance Segmentation | Ultralytics | ✅ Implemented |
| Pose Estimation | Ultralytics | ✅ Implemented |

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Setup | 30분 | venv, scripts |
| Phase 2: Converter | 1시간 | DICE→ImageFolder |
| Phase 3: Adapter | 2시간 | TimmAdapter |
| Phase 4: API | 1시간 | Endpoints |
| Phase 5: Backend | 30분 | Integration |
| Phase 6: Testing | 1시간 | Validation |
| **Total** | **6시간** | Timm framework ready |

---

## Success Criteria

- [ ] Timm service starts on port 8001
- [ ] Training with ResNet18 completes successfully
- [ ] MLflow logs metrics correctly
- [ ] Checkpoint uploads to R2
- [ ] Pretrained inference works (ImageNet 1000 classes)
- [ ] Fine-tuned inference works (custom classes)
- [ ] Frontend displays training progress
- [ ] Frontend displays inference results

---

## Next Steps After Timm

1. **HuggingFace Transformers** (Port 8003)
   - Vision-language models (CLIP, BLIP)
   - Object detection (DETR, DINO)
   - Segmentation (SegFormer, Mask2Former)

2. **Framework Comparison UI**
   - Side-by-side model comparison
   - Performance benchmarks
   - Model recommendations

3. **AutoML Features**
   - Automatic model selection
   - Hyperparameter tuning
   - Neural architecture search

---

**References**:
- [Training Framework Implementation Guide](./20251105_training_framework_implementation_guide.md)
- [Timm Documentation](https://github.com/huggingface/pytorch-image-models)
- [Ultralytics Implementation](../trainer/IMPLEMENTATION_STATUS.md)
