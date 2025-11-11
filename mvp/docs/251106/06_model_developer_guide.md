# Model Developer Guide - 플랫폼 플러그인 개발 가이드

> **작성일**: 2025-11-06
> **Version**: 1.0
> **Status**: Production
> **대상 독자**: 새로운 프레임워크/모델을 플랫폼에 통합하려는 개발자

## 목차

1. [개요](#개요)
2. [사전 준비](#사전-준비)
3. [Step 1: 프로젝트 구조 생성](#step-1-프로젝트-구조-생성)
4. [Step 2: 모델 Registry 정의](#step-2-모델-registry-정의)
5. [Step 3: Adapter 클래스 구현](#step-3-adapter-클래스-구현)
6. [Step 4: Dataset 처리 구현](#step-4-dataset-처리-구현)
7. [Step 5: Training Loop 구현](#step-5-training-loop-구현)
8. [Step 6: Metrics 및 Logging](#step-6-metrics-및-logging)
9. [Step 7: Config Schema 정의](#step-7-config-schema-정의)
10. [Step 8: API Server 통합](#step-8-api-server-통합)
11. [Step 9: 테스트](#step-9-테스트)
12. [Step 10: 배포](#step-10-배포)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)

---

## 개요

### 플랫폼 플러그인 아키텍처

Vision AI Training Platform은 **Adapter Pattern**을 사용하여 다양한 딥러닝 프레임워크를 지원합니다.

```
[Platform SDK] (공통 인터페이스)
      ↓
[Your Adapter] (프레임워크별 구현)
      ↓
[Framework] (timm, ultralytics, transformers, etc.)
```

### 개발 과정 개요

1. **모델 Registry** - 지원할 모델 목록 정의
2. **Adapter 구현** - TrainingAdapter 추상 클래스 상속
3. **Dataset 처리** - 데이터셋 로딩 및 전처리
4. **Training Loop** - 학습 로직 구현
5. **Metrics/Logging** - 플랫폼 SDK 연동
6. **Config Schema** - Advanced Config UI 정의
7. **API 통합** - Training Service에 등록
8. **테스트 및 배포**

---

## 사전 준비

### 1. 개발 환경

**필수 요구사항**:
- Python 3.11+
- Git
- 가상 환경 (venv 또는 conda)

**추천 도구**:
- VS Code with Python extension
- Docker (테스트용)

### 2. Platform SDK 이해

**핵심 클래스**:
```python
from platform_sdk import (
    # Configuration
    ModelConfig,           # 모델 설정
    DatasetConfig,         # 데이터셋 설정
    TrainingConfig,        # 학습 설정

    # Enums
    TaskType,              # 태스크 타입
    DatasetFormat,         # 데이터셋 포맷

    # Base Class
    TrainingAdapter,       # Adapter 베이스 클래스

    # Results
    MetricsResult,         # 메트릭 결과

    # Utilities
    TrainingLogger,        # 로깅 유틸리티
    get_dataset,           # 데이터셋 다운로드
)
```

### 3. 디렉토리 구조

```
mvp/training/
  ├── platform_sdk/           # Platform SDK (수정 불가)
  │   ├── __init__.py
  │   ├── base.py
  │   ├── config.py
  │   └── logger.py
  ├── adapters/               # Adapter 구현
  │   ├── __init__.py
  │   ├── timm_adapter.py
  │   ├── ultralytics_adapter.py
  │   └── your_adapter.py     # 여기에 새 Adapter 추가
  ├── api_server.py           # Training Service API
  ├── train.py                # Training 진입점
  └── requirements-xxx.txt    # 프레임워크별 의존성
```

---

## Step 1: 프로젝트 구조 생성

### 1.1. 가상 환경 생성

```bash
cd mvp/training

# 프레임워크별 venv 생성 (예: MyFramework)
python -m venv venv-myframework

# 활성화 (Windows)
venv-myframework\Scripts\activate

# 활성화 (Linux/Mac)
source venv-myframework/bin/activate
```

### 1.2. 기본 의존성 설치

**requirements-myframework.txt** 생성:
```txt
# Platform SDK dependencies
python-dotenv
requests
pillow
mlflow
boto3
pydantic

# Your framework
your-framework>=1.0.0

# Additional dependencies
torch>=2.0.0
torchvision>=0.15.0
```

```bash
pip install -r requirements-myframework.txt
```

### 1.3. Adapter 파일 생성

```bash
touch adapters/myframework_adapter.py
```

---

## Step 2: 모델 Registry 정의

### 2.1. 모델 목록 정의

**adapters/myframework_adapter.py**:
```python
"""MyFramework Adapter for Vision AI Training Platform"""

from typing import List, Dict, Any

# 지원하는 모델 목록
MYFRAMEWORK_MODELS = [
    {
        "model_name": "my_resnet50",
        "framework": "myframework",
        "task_types": ["image_classification"],
        "default_image_size": 224,
        "parameters": "25.6M",
        "pretrained": True,
        "description": "ResNet-50 implementation in MyFramework"
    },
    {
        "model_name": "my_efficientnet",
        "framework": "myframework",
        "task_types": ["image_classification"],
        "default_image_size": 224,
        "parameters": "5.3M",
        "pretrained": True,
        "description": "EfficientNet-B0 implementation"
    },
    {
        "model_name": "my_detector",
        "framework": "myframework",
        "task_types": ["object_detection"],
        "default_image_size": 640,
        "parameters": "50M",
        "pretrained": True,
        "description": "Object detection model"
    },
]


def get_models(task_type: str = None) -> List[Dict[str, Any]]:
    """
    모델 목록 반환 (플랫폼이 호출)

    Args:
        task_type: 필터링할 태스크 타입 (optional)

    Returns:
        모델 메타데이터 리스트
    """
    if task_type:
        return [
            m for m in MYFRAMEWORK_MODELS
            if task_type in m["task_types"]
        ]
    return MYFRAMEWORK_MODELS
```

### 2.2. 모델 Registry 설명

**각 필드의 의미**:
- `model_name`: 모델 식별자 (고유해야 함)
- `framework`: 프레임워크 이름 (소문자)
- `task_types`: 지원하는 태스크 타입 목록
- `default_image_size`: 기본 이미지 크기
- `parameters`: 파라미터 수 (표시용)
- `pretrained`: 사전학습 가중치 존재 여부
- `description`: 모델 설명

---

## Step 3: Adapter 클래스 구현

### 3.1. Adapter 기본 구조

```python
from platform_sdk import (
    TrainingAdapter,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    MetricsResult,
    TaskType,
)
from pathlib import Path
from typing import List, Optional
import torch


class MyFrameworkAdapter(TrainingAdapter):
    """
    MyFramework Adapter 구현
    """

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        output_dir: str,
        job_id: int,
        project_id: Optional[int] = None,
        logger: Optional[TrainingLogger] = None,
    ):
        """
        Adapter 초기화

        Args:
            model_config: 모델 설정
            dataset_config: 데이터셋 설정
            training_config: 학습 설정
            output_dir: 출력 디렉토리
            job_id: 학습 작업 ID
            project_id: 프로젝트 ID
            logger: 로거 인스턴스
        """
        super().__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            training_config=training_config,
            output_dir=output_dir,
            job_id=job_id,
            project_id=project_id,
            logger=logger,
        )

        # 프레임워크별 초기화
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None


    def prepare_model(self) -> None:
        """
        모델 준비 (추상 메서드 구현 필수)
        """
        print(f"[INFO] Preparing model: {self.model_config.model_name}")

        # 1. 모델 생성
        from myframework import create_model

        self.model = create_model(
            model_name=self.model_config.model_name,
            num_classes=self.model_config.num_classes,
            pretrained=self.model_config.pretrained,
        )

        # 2. Device로 이동
        device = self.training_config.device
        self.model.to(device)

        print(f"[INFO] Model loaded on {device}")


    def prepare_dataset(self) -> None:
        """
        데이터셋 준비 (추상 메서드 구현 필수)
        """
        print(f"[INFO] Preparing dataset: {self.dataset_config.dataset_path}")

        # Step 4에서 상세 구현
        pass


    def train(
        self,
        start_epoch: int = 0,
        checkpoint_path: Optional[str] = None,
        resume_training: bool = False
    ) -> List[MetricsResult]:
        """
        학습 실행 (추상 메서드 구현 필수)

        Args:
            start_epoch: 시작 에포크
            checkpoint_path: 체크포인트 경로
            resume_training: 재개 여부

        Returns:
            에포크별 메트릭 결과
        """
        print("[INFO] Starting training")

        # Step 5에서 상세 구현
        pass
```

### 3.2. 필수 구현 메서드

| 메서드 | 설명 | 필수 여부 |
|--------|------|----------|
| `prepare_model()` | 모델 생성 및 초기화 | ✅ 필수 |
| `prepare_dataset()` | 데이터셋 로딩 및 전처리 | ✅ 필수 |
| `train()` | 학습 루프 실행 | ✅ 필수 |
| `get_advanced_config_schema()` | Config Schema 반환 | ⚠️ 선택 (권장) |
| `get_config_presets()` | Config Preset 반환 | ⚠️ 선택 |

---

## Step 4: Dataset 처리 구현

### 4.1. Dataset Loader 구현

```python
def prepare_dataset(self) -> None:
    """데이터셋 준비"""

    dataset_path = Path(self.dataset_config.dataset_path)
    dataset_format = self.dataset_config.format

    print(f"[INFO] Loading dataset from: {dataset_path}")
    print(f"[INFO] Dataset format: {dataset_format}")

    # Task Type에 따라 분기
    if self.model_config.task_type == TaskType.image_classification:
        self._prepare_classification_dataset(dataset_path, dataset_format)
    elif self.model_config.task_type == TaskType.object_detection:
        self._prepare_detection_dataset(dataset_path, dataset_format)
    else:
        raise ValueError(f"Unsupported task type: {self.model_config.task_type}")


def _prepare_classification_dataset(
    self,
    dataset_path: Path,
    dataset_format: DatasetFormat
) -> None:
    """Image Classification 데이터셋 준비"""

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((self.model_config.image_size, self.model_config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((self.model_config.image_size, self.model_config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ImageFolder 형식 지원
    if dataset_format == DatasetFormat.dice or dataset_format == DatasetFormat.imagefolder:
        train_dataset = datasets.ImageFolder(
            root=dataset_path / "train",
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=dataset_path / "val",
            transform=val_transform
        )

        # 클래스 수 자동 감지
        if not self.model_config.num_classes:
            self.model_config.num_classes = len(train_dataset.classes)

        print(f"[INFO] Detected {self.model_config.num_classes} classes")
        print(f"[INFO] Classes: {train_dataset.classes}")

    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    # DataLoader 생성
    self.train_loader = DataLoader(
        train_dataset,
        batch_size=self.training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    self.val_loader = DataLoader(
        val_dataset,
        batch_size=self.training_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples: {len(val_dataset)}")
```

### 4.2. Custom Dataset 구현 (선택)

복잡한 데이터셋 포맷의 경우:

```python
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    """Custom Dataset for MyFramework"""

    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        # 데이터셋 로딩 로직
        samples = []
        # ... 구현
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        # 이미지 로딩
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
```

---

## Step 5: Training Loop 구현

### 5.1. Optimizer 및 Scheduler 설정

```python
def _setup_optimizer(self) -> None:
    """Optimizer 설정"""

    # Advanced Config에서 설정 가져오기
    advanced_config = self.training_config.advanced_config or {}
    optimizer_config = advanced_config.get("optimizer", {})

    optimizer_type = optimizer_config.get("type", "adam")
    lr = optimizer_config.get("learning_rate", self.training_config.learning_rate)
    weight_decay = optimizer_config.get("weight_decay", 0.0)

    # Optimizer 생성
    if optimizer_type == "adam":
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == "adamw":
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == "sgd":
        momentum = optimizer_config.get("momentum", 0.9)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    print(f"[INFO] Optimizer: {optimizer_type}, LR: {lr}")


def _setup_scheduler(self) -> None:
    """Learning Rate Scheduler 설정"""

    advanced_config = self.training_config.advanced_config or {}
    scheduler_config = advanced_config.get("scheduler", {})

    scheduler_type = scheduler_config.get("type", "none")

    if scheduler_type == "cosine":
        T_max = scheduler_config.get("T_max", self.training_config.epochs)
        eta_min = scheduler_config.get("eta_min", 1e-6)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=T_max,
            eta_min=eta_min
        )

    elif scheduler_type == "step":
        step_size = scheduler_config.get("step_size", 30)
        gamma = scheduler_config.get("gamma", 0.1)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma
        )

    elif scheduler_type == "none":
        self.scheduler = None

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    print(f"[INFO] Scheduler: {scheduler_type}")
```

### 5.2. Training Loop 구현

```python
def train(
    self,
    start_epoch: int = 0,
    checkpoint_path: Optional[str] = None,
    resume_training: bool = False
) -> List[MetricsResult]:
    """학습 실행"""

    # 1. 모델 및 데이터셋 준비
    self.prepare_model()
    self.prepare_dataset()

    # 2. Optimizer 및 Scheduler 설정
    self._setup_optimizer()
    self._setup_scheduler()

    # 3. Checkpoint 로딩 (있는 경우)
    if checkpoint_path:
        self._load_checkpoint(checkpoint_path, resume_training)

    # 4. Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # 5. Training Loop
    metrics_history = []
    best_accuracy = 0.0

    for epoch in range(start_epoch, self.training_config.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{self.training_config.epochs}")
        print(f"{'='*80}")

        # Train one epoch
        train_loss = self._train_one_epoch(epoch, criterion)

        # Validation
        val_loss, val_metrics = self._validate(epoch, criterion)

        # Metrics
        accuracy = val_metrics.get("accuracy", 0.0)

        # Learning rate (after scheduler step)
        if self.scheduler:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]['lr']

        # Create MetricsResult
        metrics_result = MetricsResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics={
                "accuracy": accuracy,
                "learning_rate": current_lr,
            }
        )

        metrics_history.append(metrics_result)

        # Log to platform (Step 6에서 상세 설명)
        if self.logger and self.logger.enabled:
            self.logger.log_metrics(
                epoch=epoch,
                loss=train_loss,
                accuracy=accuracy,
                learning_rate=current_lr,
                extra_metrics={"val_loss": val_loss}
            )

        # Save checkpoint
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy

        self._save_checkpoint(epoch, metrics_result, is_best=is_best)

        # Print summary
        print(f"\n[EPOCH {epoch+1}] Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

    return metrics_history


def _train_one_epoch(self, epoch: int, criterion) -> float:
    """한 에포크 학습"""

    self.model.train()
    total_loss = 0.0
    device = self.training_config.device

    for batch_idx, (images, labels) in enumerate(self.train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()

        # Progress
        if (batch_idx + 1) % 10 == 0:
            print(f"[Batch {batch_idx+1}/{len(self.train_loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(self.train_loader)
    return avg_loss


def _validate(self, epoch: int, criterion) -> tuple:
    """Validation"""

    self.model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    device = self.training_config.device

    with torch.no_grad():
        for images, labels in self.val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = self.model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(self.val_loader)
    accuracy = correct / total

    metrics = {
        "accuracy": accuracy,
    }

    return avg_loss, metrics
```

### 5.3. Checkpoint 저장/로딩

```python
def _save_checkpoint(
    self,
    epoch: int,
    metrics: MetricsResult,
    is_best: bool = False
) -> None:
    """체크포인트 저장"""

    checkpoint_dir = Path(self.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint 데이터
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "metrics": {
            "train_loss": metrics.train_loss,
            "val_loss": metrics.val_loss,
            "accuracy": metrics.metrics.get("accuracy", 0.0),
        },
        "model_config": {
            "model_name": self.model_config.model_name,
            "num_classes": self.model_config.num_classes,
        },
    }

    if self.scheduler:
        checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

    # 에포크별 저장
    epoch_path = checkpoint_dir / f"epoch_{epoch}.pth"
    torch.save(checkpoint, epoch_path)

    # Latest 저장
    latest_path = checkpoint_dir / "latest.pth"
    torch.save(checkpoint, latest_path)

    # Best 저장
    if is_best:
        best_path = checkpoint_dir / "best.pth"
        torch.save(checkpoint, best_path)
        print(f"[INFO] Best checkpoint saved: {best_path}")

    # R2에 업로드 (TrainingAdapter base class method 활용)
    self._upload_checkpoint_to_r2(epoch_path, epoch, metrics)


def _load_checkpoint(
    self,
    checkpoint_path: str,
    resume_training: bool
) -> None:
    """체크포인트 로딩"""

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=self.training_config.device)

    # 모델 가중치 로딩
    self.model.load_state_dict(checkpoint["model_state_dict"])

    if resume_training:
        # Optimizer, Scheduler도 복원
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"[INFO] Resumed from epoch {checkpoint['epoch']}")
    else:
        print(f"[INFO] Loaded pretrained weights only")
```

---

## Step 6: Metrics 및 Logging

### 6.1. TrainingLogger 활용

Platform SDK의 `TrainingLogger`를 사용하여 백엔드와 통신합니다.

```python
from platform_sdk import TrainingLogger

# Adapter __init__에서 초기화됨
self.logger = logger  # TrainingLogger 인스턴스
```

### 6.2. Metrics 로깅

```python
# Training Loop에서 매 에포크마다 호출
if self.logger and self.logger.enabled:
    self.logger.log_metrics(
        epoch=epoch,
        loss=train_loss,           # 필수
        accuracy=accuracy,         # 선택
        learning_rate=current_lr,  # 선택
        extra_metrics={            # 선택: 추가 메트릭
            "val_loss": val_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        }
    )
```

**자동으로 처리되는 것**:
- ✅ Backend API 호출 (`POST /api/v1/internal/training/callback`)
- ✅ Database 저장 (`training_metrics` 테이블)
- ✅ WebSocket 브로드캐스트 (Frontend 실시간 업데이트)

### 6.3. Status 업데이트

```python
# 학습 시작 시
if self.logger:
    self.logger.update_status("running")

# 학습 완료 시
if self.logger:
    self.logger.update_status("completed")

# 실패 시
if self.logger:
    self.logger.update_status("failed", error=str(exception))
```

### 6.4. 일반 로그 메시지

```python
if self.logger:
    self.logger.log_message("Starting data loading", level="INFO")
    self.logger.log_message("GPU memory usage: 8GB", level="DEBUG")
    self.logger.log_message("Warning: batch size too large", level="WARNING")
    self.logger.log_message("Training failed", level="ERROR")
```

### 6.5. Validation Results 저장

TrainingAdapter의 base class 메서드를 사용합니다:

```python
# Validation 완료 후
self._save_validation_result(
    epoch=epoch,
    task_type=self.model_config.task_type,
    primary_metric_name="accuracy",
    primary_metric_value=accuracy,
    overall_loss=val_loss,
    metrics={
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    },
    per_class_metrics=per_class_metrics,  # Optional
    confusion_matrix=confusion_matrix,    # Optional
    checkpoint_path=checkpoint_path,
)
```

**자동으로 처리되는 것**:
- ✅ Database 저장 (`validation_results` 테이블)
- ✅ PostgreSQL/SQLite 자동 선택
- ✅ Checkpoint 경로 기록

---

## Step 7: Config Schema 정의

Advanced Config UI를 위한 스키마를 정의합니다.

### 7.1. ConfigSchema 정의

```python
from platform_sdk.config import ConfigField, ConfigSchema

@staticmethod
def get_advanced_config_schema(task_type: str) -> ConfigSchema:
    """
    Advanced Config Schema 정의

    Args:
        task_type: 태스크 타입

    Returns:
        ConfigSchema 객체
    """

    fields = [
        # Optimizer
        ConfigField(
            name="optimizer.type",
            type="select",
            label="Optimizer",
            description="Optimization algorithm",
            default="adam",
            options=[
                {"value": "adam", "label": "Adam", "description": "Adaptive Moment Estimation"},
                {"value": "adamw", "label": "AdamW", "description": "Adam with weight decay"},
                {"value": "sgd", "label": "SGD", "description": "Stochastic Gradient Descent"},
            ]
        ),

        ConfigField(
            name="optimizer.learning_rate",
            type="number",
            label="Learning Rate",
            description="Initial learning rate",
            default=0.001,
            min=1e-6,
            max=1.0,
            step=1e-5,
        ),

        ConfigField(
            name="optimizer.weight_decay",
            type="number",
            label="Weight Decay",
            description="L2 regularization",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.0001,
        ),

        # Scheduler
        ConfigField(
            name="scheduler.type",
            type="select",
            label="LR Scheduler",
            description="Learning rate scheduling strategy",
            default="cosine",
            options=[
                {"value": "cosine", "label": "Cosine Annealing"},
                {"value": "step", "label": "Step LR"},
                {"value": "none", "label": "None"},
            ]
        ),

        ConfigField(
            name="scheduler.T_max",
            type="number",
            label="T_max (Cosine)",
            description="Maximum iterations for cosine annealing",
            default=50,
            min=1,
            max=1000,
            step=1,
            condition={"field": "scheduler.type", "value": "cosine"},  # Conditional display
        ),

        # Data Augmentation
        ConfigField(
            name="augmentation.enabled",
            type="boolean",
            label="Enable Augmentation",
            description="Apply data augmentation",
            default=True,
        ),

        ConfigField(
            name="augmentation.random_flip",
            type="boolean",
            label="Random Flip",
            default=True,
            condition={"field": "augmentation.enabled", "value": True},
        ),

        # Mixed Precision
        ConfigField(
            name="mixed_precision",
            type="boolean",
            label="Mixed Precision Training",
            description="Use FP16 for faster training",
            default=False,
        ),

        # Gradient Clipping
        ConfigField(
            name="gradient_clip_value",
            type="number",
            label="Gradient Clip Value",
            description="Max gradient norm (0 = disabled)",
            default=0.0,
            min=0.0,
            max=10.0,
            step=0.1,
        ),
    ]

    return ConfigSchema(fields=fields)
```

### 7.2. Config Presets 정의

```python
@staticmethod
def get_config_presets(task_type: str) -> Dict[str, Any]:
    """
    Config Preset 정의

    Returns:
        Preset 딕셔너리
    """

    return {
        "basic": {
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "type": "none",
            },
            "augmentation": {
                "enabled": False,
            },
            "mixed_precision": False,
            "gradient_clip_value": 0.0,
        },

        "standard": {
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.01,
            },
            "scheduler": {
                "type": "cosine",
                "T_max": 50,
                "eta_min": 1e-6,
            },
            "augmentation": {
                "enabled": True,
                "random_flip": True,
            },
            "mixed_precision": True,
            "gradient_clip_value": 1.0,
        },

        "aggressive": {
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.003,
                "weight_decay": 0.05,
            },
            "scheduler": {
                "type": "cosine",
                "T_max": 100,
                "eta_min": 1e-7,
            },
            "augmentation": {
                "enabled": True,
                "random_flip": True,
            },
            "mixed_precision": True,
            "gradient_clip_value": 5.0,
        },
    }
```

---

## Step 8: API Server 통합

### 8.1. Adapter 등록

**adapters/__init__.py** 수정:

```python
from .timm_adapter import TimmAdapter
from .ultralytics_adapter import UltralyticsAdapter
from .myframework_adapter import MyFrameworkAdapter  # 추가

# Adapter Registry
ADAPTER_REGISTRY = {
    "timm": TimmAdapter,
    "ultralytics": UltralyticsAdapter,
    "myframework": MyFrameworkAdapter,  # 추가
}
```

### 8.2. API Server에 등록

**api_server.py** 수정:

```python
from adapters import ADAPTER_REGISTRY
from adapters.myframework_adapter import get_models as get_myframework_models

# Model list endpoint
@app.get("/models/list")
def list_models(task_type: Optional[str] = None):
    """모델 목록 반환"""
    from adapters.myframework_adapter import get_models

    models = get_models(task_type)
    return {"models": models, "total": len(models)}


# Config schema endpoint
@app.get("/config/schema")
def get_config_schema(task_type: Optional[str] = None):
    """Config Schema 반환"""
    from adapters.myframework_adapter import MyFrameworkAdapter

    schema = MyFrameworkAdapter.get_advanced_config_schema(task_type)
    presets = MyFrameworkAdapter.get_config_presets(task_type)

    return {
        "schema": schema.to_dict(),
        "presets": presets,
    }
```

### 8.3. Training Service 실행

```python
# api_server.py에 이미 구현되어 있음
@app.post("/training/start")
def start_training(request: TrainingRequest):
    """학습 시작"""

    # Adapter 선택
    adapter_class = ADAPTER_REGISTRY.get(request.framework)

    # train.py 실행
    # ... (기존 구현 활용)
```

---

## Step 9: 테스트

### 9.1. 단위 테스트

**tests/test_myframework_adapter.py**:

```python
import pytest
from adapters.myframework_adapter import MyFrameworkAdapter, get_models
from platform_sdk import ModelConfig, DatasetConfig, TrainingConfig, TaskType, DatasetFormat


def test_get_models():
    """모델 목록 조회 테스트"""
    models = get_models()
    assert len(models) > 0
    assert all("model_name" in m for m in models)


def test_get_models_filtered():
    """태스크 타입 필터링 테스트"""
    models = get_models(task_type="image_classification")
    assert all("image_classification" in m["task_types"] for m in models)


def test_adapter_initialization():
    """Adapter 초기화 테스트"""

    model_config = ModelConfig(
        framework="myframework",
        task_type=TaskType.image_classification,
        model_name="my_resnet50",
        num_classes=10,
        image_size=224,
        pretrained=False,
    )

    dataset_config = DatasetConfig(
        dataset_path="/path/to/dataset",
        format=DatasetFormat.dice,
    )

    training_config = TrainingConfig(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        device="cpu",
    )

    adapter = MyFrameworkAdapter(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        output_dir="/tmp/output",
        job_id=1,
    )

    assert adapter is not None
    assert adapter.model_config.model_name == "my_resnet50"


def test_config_schema():
    """Config Schema 테스트"""
    schema = MyFrameworkAdapter.get_advanced_config_schema("image_classification")

    assert schema is not None
    assert len(schema.fields) > 0

    # Optimizer 필드 확인
    optimizer_fields = [f for f in schema.fields if f.name.startswith("optimizer")]
    assert len(optimizer_fields) > 0
```

### 9.2. 통합 테스트

```python
def test_training_flow(tmp_path):
    """전체 학습 플로우 테스트"""

    # 1. Dummy dataset 생성
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "train" / "class1").mkdir(parents=True)
    (dataset_dir / "val" / "class1").mkdir(parents=True)

    # Dummy images
    from PIL import Image
    import numpy as np

    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(dataset_dir / "train" / "class1" / f"img_{i}.jpg")

    # 2. Config 생성
    model_config = ModelConfig(
        framework="myframework",
        task_type=TaskType.image_classification,
        model_name="my_resnet50",
        num_classes=1,
        image_size=224,
        pretrained=False,
    )

    dataset_config = DatasetConfig(
        dataset_path=str(dataset_dir),
        format=DatasetFormat.dice,
    )

    training_config = TrainingConfig(
        epochs=2,  # 짧게 테스트
        batch_size=2,
        learning_rate=0.001,
        device="cpu",
    )

    # 3. Adapter 생성 및 학습
    adapter = MyFrameworkAdapter(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        output_dir=str(tmp_path / "output"),
        job_id=1,
    )

    metrics = adapter.train()

    # 4. 검증
    assert len(metrics) == 2  # 2 epochs
    assert all(m.epoch >= 0 for m in metrics)
    assert all(m.train_loss >= 0 for m in metrics)
```

### 9.3. 로컬 테스트 실행

```bash
# 단위 테스트
pytest tests/test_myframework_adapter.py -v

# 특정 테스트만
pytest tests/test_myframework_adapter.py::test_get_models -v

# Coverage 확인
pytest tests/ --cov=adapters.myframework_adapter --cov-report=html
```

---

## Step 10: 배포

### 10.1. Requirements 정리

**requirements-myframework.txt** 최종 확인:

```txt
# Platform SDK
python-dotenv==1.0.0
requests==2.31.0
pillow==10.1.0
mlflow==2.9.2
boto3==1.34.0
pydantic==2.5.0

# Framework
myframework==1.0.0

# Deep Learning
torch==2.1.2
torchvision==0.16.2

# Additional
numpy==1.24.3
tqdm==4.66.1
```

### 10.2. Docker 이미지 생성 (선택)

**Dockerfile.myframework**:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-myframework.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-myframework.txt

# Copy code
COPY platform_sdk/ /app/platform_sdk/
COPY adapters/myframework_adapter.py /app/adapters/
COPY train.py /app/
COPY api_server.py /app/

# Environment
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Run API server
CMD ["python", "api_server.py"]
```

### 10.3. Railway 배포

**.railwayignore** 업데이트:

```
# Add your venv
venv-myframework/
```

**환경 변수 설정** (Railway Dashboard):
```bash
# Backend에서 Training Service 호출 시 사용
MYFRAMEWORK_SERVICE_URL=http://myframework-service:5000

# R2 credentials (이미 설정되어 있음)
R2_ENDPOINT=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
```

### 10.4. 통합 확인

1. **Backend에서 모델 목록 조회**:
   ```bash
   curl http://localhost:8000/api/v1/models/list?framework=myframework
   ```

2. **Config Schema 조회**:
   ```bash
   curl http://localhost:8000/api/v1/training/config-schema?framework=myframework
   ```

3. **학습 작업 생성 및 시작**:
   ```bash
   # POST /api/v1/training/jobs
   # POST /api/v1/training/jobs/{job_id}/start
   ```

---

## Best Practices

### 1. 코드 구조

✅ **DO**:
- 한 파일에 한 Adapter만 구현
- 명확한 메서드 분리 (`_prepare_xxx`, `_train_one_epoch` 등)
- Type hints 사용
- Docstring 작성

❌ **DON'T**:
- 너무 긴 메서드 (50 lines 이하 권장)
- 하드코딩된 값 (config에서 가져오기)
- Global 변수 사용

### 2. 에러 처리

```python
try:
    self.model = create_model(...)
except Exception as e:
    if self.logger:
        self.logger.log_message(f"Model creation failed: {e}", level="ERROR")
        self.logger.update_status("failed", error=str(e))
    raise
```

### 3. 메모리 관리

```python
# GPU 메모리 정리
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 사용 후 명시적 삭제
del large_tensor
```

### 4. 성능 최적화

```python
# DataLoader num_workers 조정
self.train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # CPU 코어 수에 맞게
    pin_memory=True,  # GPU 사용 시 True
    persistent_workers=True,  # Worker 재사용
)

# Mixed Precision Training
if advanced_config.get("mixed_precision"):
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    with autocast():
        outputs = self.model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(self.optimizer)
    scaler.update()
```

### 5. Logging 권장사항

```python
# 주요 단계마다 로깅
print(f"[INFO] Starting epoch {epoch}")
print(f"[INFO] Loading checkpoint from {path}")
print(f"[WARNING] GPU memory usage high: {memory_usage}GB")
print(f"[ERROR] Failed to load dataset: {error}")

# Platform logger 적극 활용
if self.logger:
    self.logger.log_message("Custom message", level="INFO")
```

---

## Troubleshooting

### 문제 1: 모델이 플랫폼에 표시되지 않음

**원인**:
- Adapter가 `ADAPTER_REGISTRY`에 등록되지 않음
- `get_models()` 함수가 없거나 잘못 구현됨

**해결**:
```python
# adapters/__init__.py 확인
from .myframework_adapter import MyFrameworkAdapter

ADAPTER_REGISTRY = {
    "myframework": MyFrameworkAdapter,  # 이 줄이 있는지 확인
}

# api_server.py 확인
@app.get("/models/list")
def list_models(task_type: Optional[str] = None):
    from adapters.myframework_adapter import get_models
    # ...
```

### 문제 2: Dataset을 찾을 수 없음

**원인**:
- R2에서 다운로드 실패
- 경로가 잘못됨

**해결**:
```python
# Debug logging 추가
print(f"[DEBUG] Dataset path: {self.dataset_config.dataset_path}")
print(f"[DEBUG] Path exists: {Path(self.dataset_config.dataset_path).exists()}")

# R2 credentials 확인
import os
print(f"[DEBUG] R2_ENDPOINT: {os.getenv('R2_ENDPOINT')}")
```

### 문제 3: Metrics가 Frontend에 표시되지 않음

**원인**:
- `TrainingLogger` 비활성화
- Backend API 연결 실패

**해결**:
```python
# Logger 활성화 확인
if self.logger:
    print(f"[DEBUG] Logger enabled: {self.logger.enabled}")
    print(f"[DEBUG] Backend URL: {self.logger.backend_url}")

# 수동으로 metrics 로깅 테스트
if self.logger and self.logger.enabled:
    self.logger.log_metrics(
        epoch=0,
        loss=1.0,
        accuracy=0.5,
    )
```

### 문제 4: CUDA Out of Memory

**원인**:
- Batch size가 너무 큼
- 모델이 너무 큼

**해결**:
```python
# Batch size 줄이기
batch_size = self.training_config.batch_size
if torch.cuda.is_available():
    # GPU 메모리에 따라 조정
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory < 8 * 1024**3:  # 8GB 미만
        batch_size = min(batch_size, 16)

# Gradient accumulation 사용
accumulation_steps = 4
for i, (images, labels) in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### 문제 5: Advanced Config가 적용되지 않음

**원인**:
- Config Schema 정의 오류
- Config 파싱 오류

**해결**:
```python
# Advanced config 확인
advanced_config = self.training_config.advanced_config
print(f"[DEBUG] Advanced config: {advanced_config}")

# Nested 값 안전하게 가져오기
optimizer_config = advanced_config.get("optimizer", {}) if advanced_config else {}
optimizer_type = optimizer_config.get("type", "adam")
```

---

## 참고 문서

- [SDK & Adapter Pattern Guide](./02_sdk_adapter_pattern.md) - 아키텍처 상세
- [Config Schema Guide](./03_config_schema_guide.md) - Config 시스템
- [Backend API Specification](./01_backend_api_specification.md) - API 명세
- [User Flow Scenarios](./04_user_flow_scenarios.md) - 전체 플로우
- [Annotation System](./05_annotation_system.md) - 데이터셋 포맷

---

## 예제 코드 저장소

완전한 예제 구현은 다음 파일을 참고하세요:

- **TimmAdapter**: `mvp/training/adapters/timm_adapter.py`
- **UltralyticsAdapter**: `mvp/training/adapters/ultralytics_adapter.py`

이 두 Adapter는 프로덕션에서 사용 중이며, 새로운 Adapter 개발 시 참고할 수 있는 Best Practice 구현입니다.
