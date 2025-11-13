# SDK & Adapter Pattern 개발 가이드

> **작성일**: 2025-11-06
> **Version**: 1.0
> **대상**: Training Service 개발자

## 목차

1. [개요](#개요)
2. [Platform SDK 구조](#platform-sdk-구조)
3. [Adapter Pattern](#adapter-pattern)
4. [새로운 프레임워크 추가하기](#새로운-프레임워크-추가하기)
5. [Config Schema 정의](#config-schema-정의)
6. [Best Practices](#best-practices)

---

## 개요

Vision AI Training Platform은 **Adapter Pattern**을 사용하여 여러 딥러닝 프레임워크를 통합합니다. 각 프레임워크(timm, Ultralytics, HuggingFace 등)는 자체 API를 가지고 있지만, Platform SDK를 통해 **통일된 인터페이스**로 추상화됩니다.

### 핵심 개념

```
┌─────────────────────────────────────────────────────────┐
│                      Backend API                        │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP Request
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Training Service API                       │
│               (api_server.py)                           │
└────────────────┬────────────────────────────────────────┘
                 │ subprocess.Popen()
                 ▼
┌─────────────────────────────────────────────────────────┐
│                    train.py                             │
│          (Adapter 선택 및 초기화)                        │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴────────┬──────────────┬────────────┐
        ▼                 ▼              ▼            ▼
  ┌──────────┐   ┌───────────────┐  ┌─────────┐  ┌──────────┐
  │  timm    │   │ Ultralytics   │  │ HuggingF│  │ Custom   │
  │ Adapter  │   │   Adapter     │  │  Adapter│  │ Adapter  │
  └──────────┘   └───────────────┘  └─────────┘  └──────────┘
        │                 │              │            │
        ▼                 ▼              ▼            ▼
  ┌──────────┐   ┌───────────────┐  ┌─────────┐  ┌──────────┐
  │   timm   │   │   YOLO API    │  │Transform│  │ PyTorch  │
  │   API    │   │               │  │   API   │  │   API    │
  └──────────┘   └───────────────┘  └─────────┘  └──────────┘
```

### 왜 Adapter Pattern인가?

1. **프레임워크 독립성**: 각 프레임워크의 변경이 다른 부분에 영향을 주지 않음
2. **확장성**: 새로운 프레임워크 추가가 기존 코드에 영향 없음
3. **유지보수성**: 각 Adapter가 독립적으로 관리됨
4. **테스트 용이성**: Adapter별로 독립적인 테스트 가능

---

## Platform SDK 구조

### 디렉토리 구조

```
mvp/training/
├── platform_sdk/           # 공통 SDK
│   ├── __init__.py
│   ├── base.py            # 추상 클래스 및 공통 유틸
│   ├── logger.py          # TrainingLogger
│   └── storage.py         # R2 Storage 클라이언트
├── adapters/              # 프레임워크별 Adapter
│   ├── __init__.py
│   ├── timm_adapter.py
│   ├── ultralytics_adapter.py
│   └── huggingface_adapter.py (planned)
└── train.py               # 메인 학습 스크립트
```

### 핵심 클래스

#### 1. **TrainingAdapter** (추상 클래스)

모든 Adapter가 상속해야 하는 베이스 클래스입니다.

```python
# mvp/training/platform_sdk/base.py

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

class TrainingAdapter(ABC):
    """모든 프레임워크 Adapter의 베이스 클래스"""

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        output_dir: str,
        job_id: int,
        project_id: Optional[int] = None,
        logger: Optional[TrainingLogger] = None
    ):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.output_dir = output_dir
        self.job_id = job_id
        self.project_id = project_id
        self.logger = logger

        # 모델 및 데이터 객체
        self.model = None
        self.train_loader = None
        self.val_loader = None

    @abstractmethod
    def prepare_model(self) -> None:
        """모델 초기화 및 준비"""
        pass

    @abstractmethod
    def prepare_dataset(self) -> None:
        """데이터셋 로드 및 전처리"""
        pass

    @abstractmethod
    def train(
        self,
        start_epoch: int = 0,
        checkpoint_path: Optional[str] = None,
        resume_training: bool = False
    ) -> List[MetricsResult]:
        """학습 실행"""
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int, metrics: dict) -> str:
        """체크포인트 저장"""
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_path: str,
        resume_training: bool = False
    ) -> int:
        """체크포인트 로드"""
        pass
```

#### 2. **Config 클래스들**

```python
# mvp/training/platform_sdk/base.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class TaskType(Enum):
    """지원하는 태스크 타입"""
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POSE_ESTIMATION = "pose_estimation"

class DatasetFormat(Enum):
    """지원하는 데이터셋 포맷"""
    IMAGEFOLDER = "imagefolder"
    COCO = "coco"
    YOLO = "yolo"
    VOC = "voc"
    DICE = "dice"

@dataclass
class ModelConfig:
    """모델 설정"""
    framework: str              # timm, ultralytics, huggingface
    task_type: TaskType
    model_name: str             # resnet50, yolov8n, etc.
    pretrained: bool = True
    num_classes: Optional[int] = None
    image_size: int = 224

@dataclass
class DatasetConfig:
    """데이터셋 설정"""
    dataset_path: str
    format: DatasetFormat

@dataclass
class TrainingConfig:
    """학습 설정"""
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "adam"
    device: str = "cuda"
    advanced_config: Optional[Dict[str, Any]] = None
```

#### 3. **MetricsResult**

```python
@dataclass
class MetricsResult:
    """학습 메트릭 결과"""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    metrics: Dict[str, float] = None  # accuracy, precision, etc.

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
```

---

## Adapter Pattern

### TimmAdapter 예제

```python
# mvp/training/adapters/timm_adapter.py

import torch
import timm
from torch.utils.data import DataLoader
from platform_sdk import TrainingAdapter, ModelConfig, MetricsResult

class TimmAdapter(TrainingAdapter):
    """timm 프레임워크용 Adapter"""

    def prepare_model(self) -> None:
        """timm 모델 초기화"""
        print(f"[timm] Loading model: {self.model_config.model_name}")

        self.model = timm.create_model(
            self.model_config.model_name,
            pretrained=self.model_config.pretrained,
            num_classes=self.model_config.num_classes
        )

        self.model = self.model.to(self.training_config.device)

        print(f"[timm] Model loaded successfully")

    def prepare_dataset(self) -> None:
        """데이터셋 로드"""
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.Resize(self.model_config.image_size),
            transforms.CenterCrop(self.model_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ImageFolder 형식 가정
        train_dataset = datasets.ImageFolder(
            root=f"{self.dataset_config.dataset_path}/train",
            transform=transform
        )

        val_dataset = datasets.ImageFolder(
            root=f"{self.dataset_config.dataset_path}/val",
            transform=transform
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=4
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=4
        )

        print(f"[timm] Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")

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

        # 2. Optimizer 및 Scheduler 초기화
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)

        # 3. Checkpoint 로드 (옵션)
        if checkpoint_path:
            start_epoch = self.load_checkpoint(checkpoint_path, resume_training)

        # 4. 학습 루프
        all_metrics = []

        for epoch in range(start_epoch, self.training_config.epochs):
            epoch_num = epoch + 1

            # Train
            train_loss, train_acc = self._train_epoch(epoch_num, optimizer)

            # Validate
            val_loss, val_acc = self._validate_epoch(epoch_num)

            # Metrics 저장
            metrics_result = MetricsResult(
                epoch=epoch_num,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics={
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc
                }
            )
            all_metrics.append(metrics_result)

            # Checkpoint 저장
            checkpoint_path = self.save_checkpoint(epoch_num, metrics_result.metrics)

            # Scheduler step
            if scheduler:
                scheduler.step()

        return all_metrics

    def _train_epoch(self, epoch: int, optimizer) -> tuple:
        """1 epoch 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.training_config.device)
            labels = labels.to(self.training_config.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, metrics: dict) -> str:
        """체크포인트 저장"""
        checkpoint_path = f"{self.output_dir}/epoch_{epoch}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics
        }, checkpoint_path)

        return checkpoint_path
```

### UltralyticsAdapter 예제

```python
# mvp/training/adapters/ultralytics_adapter.py

from ultralytics import YOLO
from platform_sdk import TrainingAdapter, MetricsResult

class UltralyticsAdapter(TrainingAdapter):
    """Ultralytics YOLO 프레임워크용 Adapter"""

    def prepare_model(self) -> None:
        """YOLO 모델 초기화"""
        print(f"[YOLO] Loading model: {self.model_config.model_name}")

        # YOLO는 pretrained weights 자동 다운로드
        self.model = YOLO(f"{self.model_config.model_name}.pt")

        print(f"[YOLO] Model loaded successfully")

    def prepare_dataset(self) -> None:
        """데이터셋 준비 (YOLO format data.yaml)"""
        import yaml

        # YOLO는 data.yaml 파일 필요
        data_yaml = f"{self.dataset_config.dataset_path}/data.yaml"

        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        self.data_yaml = data_yaml
        self.num_classes = data_config['nc']

        print(f"[YOLO] Dataset prepared: {self.num_classes} classes")

    def train(
        self,
        start_epoch: int = 0,
        checkpoint_path: Optional[str] = None,
        resume_training: bool = False
    ) -> List[MetricsResult]:
        """YOLO 학습 실행"""

        self.prepare_model()
        self.prepare_dataset()

        # YOLO는 자체 training API 사용
        results = self.model.train(
            data=self.data_yaml,
            epochs=self.training_config.epochs,
            batch=self.training_config.batch_size,
            imgsz=self.model_config.image_size,
            device=self.training_config.device,
            project=self.output_dir,
            name="train",
            resume=resume_training and checkpoint_path is not None
        )

        # YOLO 결과를 MetricsResult로 변환
        metrics_list = []
        for epoch in range(self.training_config.epochs):
            metrics_result = MetricsResult(
                epoch=epoch + 1,
                train_loss=results.metrics[f"train/box_loss_{epoch}"],
                val_loss=results.metrics.get(f"val/box_loss_{epoch}"),
                metrics={
                    "mAP50": results.metrics.get(f"val/mAP50_{epoch}", 0.0),
                    "mAP50-95": results.metrics.get(f"val/mAP50-95_{epoch}", 0.0)
                }
            )
            metrics_list.append(metrics_result)

        return metrics_list

    def save_checkpoint(self, epoch: int, metrics: dict) -> str:
        """YOLO는 자동으로 checkpoint 저장"""
        # YOLO는 weights 폴더에 자동 저장
        return f"{self.output_dir}/train/weights/last.pt"
```

---

## 새로운 프레임워크 추가하기

### Step 1: Adapter 클래스 생성

```python
# mvp/training/adapters/my_framework_adapter.py

from platform_sdk import TrainingAdapter, MetricsResult
from typing import List, Optional

class MyFrameworkAdapter(TrainingAdapter):
    """새로운 프레임워크용 Adapter"""

    def prepare_model(self) -> None:
        # 모델 초기화 로직
        pass

    def prepare_dataset(self) -> None:
        # 데이터셋 로드 로직
        pass

    def train(
        self,
        start_epoch: int = 0,
        checkpoint_path: Optional[str] = None,
        resume_training: bool = False
    ) -> List[MetricsResult]:
        # 학습 로직
        pass

    def save_checkpoint(self, epoch: int, metrics: dict) -> str:
        # 체크포인트 저장 로직
        pass

    def load_checkpoint(
        self,
        checkpoint_path: str,
        resume_training: bool = False
    ) -> int:
        # 체크포인트 로드 로직
        pass
```

### Step 2: Adapter Registry에 등록

```python
# mvp/training/adapters/__init__.py

from .timm_adapter import TimmAdapter
from .ultralytics_adapter import UltralyticsAdapter
from .my_framework_adapter import MyFrameworkAdapter

ADAPTER_REGISTRY = {
    "timm": TimmAdapter,
    "ultralytics": UltralyticsAdapter,
    "my_framework": MyFrameworkAdapter,  # 추가!
}

__all__ = [
    "ADAPTER_REGISTRY",
    "TimmAdapter",
    "UltralyticsAdapter",
    "MyFrameworkAdapter",
]
```

### Step 3: Config Schema 정의

```python
# mvp/training/adapters/my_framework_adapter.py

class MyFrameworkAdapter(TrainingAdapter):

    @staticmethod
    def get_config_schema() -> ConfigSchema:
        """프레임워크별 Config Schema 반환"""
        return ConfigSchema(
            framework="my_framework",
            task_types=[TaskType.IMAGE_CLASSIFICATION],
            fields=[
                ConfigField(
                    name="optimizer.type",
                    type="select",
                    label="Optimizer",
                    default="adam",
                    options=[
                        {"value": "adam", "label": "Adam"},
                        {"value": "sgd", "label": "SGD"}
                    ]
                ),
                # ... 더 많은 필드
            ]
        )
```

### Step 4: API Server에 등록

```python
# mvp/training/api_server.py

# 환경변수로 프레임워크 지정
FRAMEWORK = os.environ.get("FRAMEWORK", "unknown")

# 지원 프레임워크 자동 감지
if FRAMEWORK in ADAPTER_REGISTRY:
    print(f"[INFO] Framework '{FRAMEWORK}' is supported")
else:
    print(f"[WARNING] Framework '{FRAMEWORK}' is not registered")
```

---

## Config Schema 정의

### ConfigSchema 클래스

```python
# mvp/training/platform_sdk/base.py

@dataclass
class ConfigField:
    """Config 필드 정의"""
    name: str                    # optimizer.learning_rate
    type: str                    # number, select, boolean, range
    label: str                   # "Learning Rate"
    description: str = ""
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Dict]] = None

@dataclass
class ConfigSchema:
    """프레임워크별 Config Schema"""
    framework: str
    task_types: List[TaskType]
    fields: List[ConfigField]
    presets: Optional[Dict[str, Dict]] = None
```

### 실제 사용 예제

```python
class TimmAdapter(TrainingAdapter):

    @staticmethod
    def get_config_schema() -> ConfigSchema:
        return ConfigSchema(
            framework="timm",
            task_types=[TaskType.IMAGE_CLASSIFICATION],
            fields=[
                # Optimizer
                ConfigField(
                    name="optimizer.type",
                    type="select",
                    label="Optimizer",
                    description="Optimization algorithm",
                    default="adamw",
                    options=[
                        {"value": "adam", "label": "Adam"},
                        {"value": "adamw", "label": "AdamW"},
                        {"value": "sgd", "label": "SGD"}
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
                    step=1e-5
                ),
                # Scheduler
                ConfigField(
                    name="scheduler.type",
                    type="select",
                    label="Scheduler",
                    default="cosine",
                    options=[
                        {"value": "none", "label": "None"},
                        {"value": "step", "label": "Step LR"},
                        {"value": "cosine", "label": "Cosine Annealing"}
                    ]
                ),
                # Augmentation
                ConfigField(
                    name="augmentation.enabled",
                    type="boolean",
                    label="Enable Augmentation",
                    default=True
                ),
                ConfigField(
                    name="augmentation.random_flip",
                    type="boolean",
                    label="Random Flip",
                    default=True
                ),
            ],
            presets={
                "basic": {
                    "optimizer": {"type": "adam", "learning_rate": 0.001},
                    "scheduler": {"type": "none"},
                    "augmentation": {"enabled": True, "random_flip": True}
                },
                "standard": {
                    "optimizer": {"type": "adamw", "learning_rate": 0.0003},
                    "scheduler": {"type": "cosine"},
                    "augmentation": {"enabled": True, "random_flip": True}
                }
            }
        )
```

---

## Best Practices

### 1. **에러 처리**

```python
def prepare_model(self) -> None:
    try:
        self.model = timm.create_model(...)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        if self.logger:
            self.logger.log_message(f"Model load failed: {e}", level="ERROR")
        raise
```

### 2. **로깅**

```python
def train(self, ...):
    # TrainingLogger 사용
    if self.logger:
        self.logger.update_status("running")
        self.logger.log_message("Starting training...")

    for epoch in range(...):
        # Metric 전송
        if self.logger:
            self.logger.log_metric(
                epoch=epoch,
                metrics={"loss": train_loss, "accuracy": train_acc}
            )
```

### 3. **체크포인트 관리**

```python
def save_checkpoint(self, epoch: int, metrics: dict) -> str:
    # R2 Storage에 자동 업로드
    checkpoint_path = f"{self.output_dir}/epoch_{epoch}.pth"

    torch.save(state_dict, checkpoint_path)

    # R2 업로드
    if self.project_id:
        from platform_sdk import upload_checkpoint_to_r2
        r2_path = upload_checkpoint_to_r2(
            checkpoint_path,
            project_id=self.project_id,
            job_id=self.job_id,
            epoch=epoch
        )
        print(f"[Checkpoint] Uploaded to R2: {r2_path}")

    return checkpoint_path
```

### 4. **Validation 결과 저장**

```python
def _validate_epoch(self, epoch: int):
    # ... validation 로직 ...

    # Validation 결과를 DB에 저장
    validation_metrics = ValidationMetrics(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        primary_metric_name="accuracy",
        primary_metric_value=accuracy,
        overall_loss=val_loss,
        classification=ClassificationMetrics(...)
    )

    # DB에 저장 (base.py의 _save_validation_result 사용)
    val_result_id = self._save_validation_result(
        epoch=epoch,
        validation_metrics=validation_metrics,
        checkpoint_path=checkpoint_path
    )
```

---

## 참고 문서

- [Config Schema 가이드](./03_config_schema_guide.md)
- [Backend API 명세서](./01_backend_api_specification.md)
- [기존 Adapter 설계 문서](../training/20251105_training_framework_implementation_guide.md)
