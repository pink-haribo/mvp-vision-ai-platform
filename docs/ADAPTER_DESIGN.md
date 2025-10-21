# Training Adapter Design

**작성일**: 2025-10-20
**버전**: 1.0
**목적**: 다중 프레임워크 및 태스크를 지원하는 확장 가능한 학습 Adapter 설계

---

## 목차

1. [개요](#개요)
2. [지원 프레임워크 및 모델](#지원-프레임워크-및-모델)
   - [Ultralytics (YOLO)](#1-ultralytics-yolo)
   - [HuggingFace Transformers](#2-huggingface-transformers)
3. [Adapter 아키텍처](#adapter-아키텍처)
4. [구현 예시](#구현-예시)
5. [통합 방법](#통합-방법)
6. [로드맵](#로드맵)

---

## 개요

### 배경

현재 시스템은 timm 기반 ResNet-50 분류 모델만 지원합니다. 하지만 실제 사용자 요구사항은 다양합니다:

- Object Detection (객체 탐지)
- Instance/Semantic Segmentation (분할)
- OCR (문자 인식)
- Pose Estimation (포즈 추정)
- Image Captioning (이미지 설명 생성)
- 기타 Vision-Language 태스크

이러한 태스크들은 각기 다른 라이브러리와 API를 사용합니다:
- **Ultralytics**: YOLOv8/v9 기반 detection, segmentation, pose
- **HuggingFace Transformers**: ViT, DETR, TrOCR, BLIP-2 등
- **timm**: 분류 백본 (기존)

### 설계 목표

1. **확장성**: 새로운 프레임워크/모델 추가 시 기존 코드 수정 최소화
2. **일관성**: 모든 프레임워크가 동일한 인터페이스 제공
3. **유지보수성**: 각 프레임워크별 구현 독립적으로 관리
4. **통합성**: MLflow, 체크포인트, 메트릭 로깅 자동화

### 핵심 패턴: Adapter Pattern

```
┌─────────────────────────────────────────┐
│     TrainingAdapter (Interface)         │
│  - prepare_model()                      │
│  - prepare_dataset()                    │
│  - train()                              │
│  - validate()                           │
│  - save_checkpoint()                    │
└─────────────────────────────────────────┘
              ▲        ▲        ▲
              │        │        │
     ┌────────┴───┐   │   ┌────┴───────┐
     │            │   │   │            │
┌────┴────┐  ┌───┴───┴──┐  ┌──────────┴──┐
│  Timm   │  │Ultralytics│  │Transformers │
│ Adapter │  │  Adapter  │  │   Adapter   │
└─────────┘  └───────────┘  └─────────────┘
```

---

## 지원 프레임워크 및 모델

### 1. Ultralytics (YOLO)

**공식 사이트**: https://docs.ultralytics.com/

#### A. Object Detection (객체 탐지)

| 모델 | 크기 | 용도 | mAP | 속도 |
|------|------|------|-----|------|
| `yolov8n.pt` | Nano | 모바일/엣지 | 37.3 | 가장 빠름 |
| `yolov8s.pt` | Small | 균형 | 44.9 | 빠름 |
| `yolov8m.pt` | Medium | 정확도 우선 | 50.2 | 중간 |
| `yolov8l.pt` | Large | 고정확도 | 52.9 | 느림 |
| `yolov8x.pt` | XLarge | 최고 정확도 | 53.9 | 가장 느림 |
| `yolov9c.pt` | YOLOv9 Compact | 2024 최신 | 53.0 | 빠름 |
| `yolov9e.pt` | YOLOv9 Extended | SOTA | 55.6 | 느림 |

**특화 모델**:
- `yolov8n-p2.pt`: 작은 객체 탐지 특화
- `yolov8n-p6.pt`: 큰 이미지 (1280px+) 처리

#### B. Instance Segmentation (인스턴스 분할)

| 모델 | 설명 |
|------|------|
| `yolov8n-seg.pt` | Nano segmentation |
| `yolov8s-seg.pt` | Small segmentation |
| `yolov8m-seg.pt` | Medium segmentation |
| `yolov8l-seg.pt` | Large segmentation |
| `yolov8x-seg.pt` | XLarge segmentation |

#### C. Pose Estimation (포즈 추정)

사람의 17개 키포인트 탐지 (어깨, 팔꿈치, 무릎 등)

| 모델 | 설명 |
|------|------|
| `yolov8n-pose.pt` | Nano pose estimation |
| `yolov8s-pose.pt` | Small pose |
| `yolov8m-pose.pt` | Medium pose |
| `yolov8l-pose.pt` | Large pose |

#### D. Oriented Bounding Box (OBB)

항공/위성 이미지용 회전된 박스 탐지

| 모델 | 설명 |
|------|------|
| `yolov8n-obb.pt` | Nano OBB |
| `yolov8s-obb.pt` | Small OBB |
| `yolov8m-obb.pt` | Medium OBB |

#### E. Classification (분류)

| 모델 | 설명 |
|------|------|
| `yolov8n-cls.pt` | 이미지 분류 |
| `yolov8s-cls.pt` | Small classification |
| `yolov8m-cls.pt` | Medium classification |

**데이터셋 형식**: YOLO 형식 (txt annotation) 또는 COCO 형식

---

### 2. HuggingFace Transformers

**공식 사이트**: https://huggingface.co/models

#### A. Image Classification

| 모델 | 설명 | 파라미터 | 특징 |
|------|------|----------|------|
| `google/vit-base-patch16-224` | Vision Transformer | 86M | Transformer 기반 분류 |
| `google/vit-large-patch16-224` | ViT Large | 304M | 높은 정확도 |
| `facebook/dinov2-base` | DINOv2 | 86M | Self-supervised, 범용성 |
| `facebook/dinov2-large` | DINOv2 Large | 304M | SOTA 성능 |
| `facebook/dinov2-giant` | DINOv2 Giant | 1.1B | 최고 성능 |
| `microsoft/swin-base-patch4-window7-224` | Swin Transformer | 88M | 계층적 구조 |
| `facebook/convnext-base-224` | ConvNeXt | 89M | 현대적 CNN |

#### B. Object Detection

| 모델 | 설명 | 특징 |
|------|------|------|
| `facebook/detr-resnet-50` | DETR | Transformer 기반 detection |
| `facebook/detr-resnet-101` | DETR Large | 더 높은 정확도 |
| `microsoft/conditional-detr-resnet-50` | Conditional DETR | 빠른 수렴 |
| `IDEA-Research/grounding-dino-base` | Grounding DINO | Zero-shot detection |
| `google/owlvit-base-patch32` | OWL-ViT | 텍스트 기반 detection |
| `google/owlvit-large-patch14` | OWL-ViT Large | Open-vocabulary detection |

#### C. Semantic Segmentation

| 모델 | 설명 | 특징 |
|------|------|------|
| `nvidia/segformer-b0-finetuned-ade-512-512` | SegFormer B0 | 효율적 |
| `nvidia/segformer-b5-finetuned-ade-640-640` | SegFormer B5 | 높은 정확도 |
| `facebook/mask2former-swin-base-ade` | Mask2Former | Universal segmentation |
| `facebook/mask2former-swin-large-ade` | Mask2Former Large | SOTA 성능 |

#### D. OCR (Optical Character Recognition)

| 모델 | 설명 | 용도 |
|------|------|------|
| `microsoft/trocr-base-handwritten` | TrOCR | 손글씨 인식 |
| `microsoft/trocr-base-printed` | TrOCR | 인쇄물 인식 |
| `microsoft/trocr-large-printed` | TrOCR Large | 높은 정확도 |
| `naver-clova-ix/donut-base` | Donut | 문서 이해 (OCR-free) |
| `microsoft/layoutlmv3-base` | LayoutLMv3 | 문서 레이아웃 + OCR |

#### E. Depth Estimation

| 모델 | 설명 | 특징 |
|------|------|------|
| `Intel/dpt-large` | DPT | Dense Prediction Transformer |
| `LiheYoung/depth-anything-base` | Depth Anything | 2024 최신, 범용성 |
| `LiheYoung/depth-anything-large` | Depth Anything Large | 최고 정확도 |

#### F. Image Captioning

| 모델 | 설명 | 특징 |
|------|------|------|
| `Salesforce/blip2-opt-2.7b` | BLIP-2 | Vision-Language 모델 |
| `Salesforce/blip2-flan-t5-xl` | BLIP-2 T5 | 더 나은 언어 이해 |
| `microsoft/git-base` | GIT | Generative Image-to-Text |
| `microsoft/git-large` | GIT Large | 높은 품질 캡션 |

#### G. Visual Question Answering (VQA)

| 모델 | 설명 |
|------|------|
| `Salesforce/blip2-opt-2.7b` | BLIP-2 VQA |
| `liuhaotian/llava-v1.5-7b` | LLaVA | 대화형 VQA |
| `liuhaotian/llava-v1.5-13b` | LLaVA Large | 더 나은 추론 능력 |

#### H. Zero-Shot Classification

| 모델 | 설명 |
|------|------|
| `openai/clip-vit-base-patch32` | CLIP | 텍스트-이미지 정렬 |
| `openai/clip-vit-large-patch14` | CLIP Large | 높은 정확도 |
| `google/siglip-base-patch16-224` | SigLIP | 개선된 CLIP |

#### I. Video Classification

| 모델 | 설명 |
|------|------|
| `MCG-NJU/videomae-base` | VideoMAE | 비디오 이해 |
| `facebook/timesformer-base-finetuned-k400` | TimeSformer | 시공간 Transformer |

**데이터셋 형식**: ImageFolder (분류), COCO (detection), Custom (OCR/VQA)

---

## Adapter 아키텍처

### 디렉토리 구조

```
mvp/training/
├── adapters/
│   ├── __init__.py
│   ├── base.py                  # 공통 인터페이스 및 데이터 클래스
│   ├── timm_adapter.py          # timm (ResNet, EfficientNet 등)
│   ├── ultralytics_adapter.py   # YOLOv8/v9 (detection, segmentation, pose)
│   └── transformers_adapter.py  # HuggingFace (모든 Vision-Language 태스크)
├── train.py                     # 통합 엔트리포인트
└── requirements.txt
```

### 핵심 데이터 클래스

#### TaskType (Enum)

지원하는 모든 태스크 타입 정의:

```python
class TaskType(Enum):
    # Vision
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"

    # Vision-Language
    IMAGE_CAPTIONING = "image_captioning"
    VISUAL_QA = "visual_qa"
    OCR = "ocr"
    DOCUMENT_UNDERSTANDING = "document_understanding"

    # Zero-Shot
    ZERO_SHOT_CLASSIFICATION = "zero_shot_classification"
    ZERO_SHOT_DETECTION = "zero_shot_detection"

    # Video
    VIDEO_CLASSIFICATION = "video_classification"
    VIDEO_DETECTION = "video_detection"
```

#### DatasetFormat (Enum)

```python
class DatasetFormat(Enum):
    IMAGE_FOLDER = "imagefolder"  # 분류용
    COCO = "coco"                 # Detection, Segmentation
    YOLO = "yolo"                 # YOLO 형식
    PASCAL_VOC = "voc"
    CUSTOM = "custom"
```

#### ModelConfig

```python
@dataclass
class ModelConfig:
    framework: str              # 'ultralytics', 'transformers', 'timm'
    task_type: TaskType
    model_name: str
    pretrained: bool = True
    num_classes: Optional[int] = None
    image_size: Union[int, tuple] = 224
    custom_config: Dict[str, Any] = None
```

#### DatasetConfig

```python
@dataclass
class DatasetConfig:
    dataset_path: str
    format: DatasetFormat
    train_split: str = "train"
    val_split: str = "val"
    test_split: Optional[str] = None
    augmentation: Dict[str, Any] = None
```

#### TrainingConfig

```python
@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    device: str = "cuda"
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    early_stopping: Optional[Dict] = None
```

#### MetricsResult

통일된 메트릭 형식:

```python
@dataclass
class MetricsResult:
    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    metrics: Dict[str, float] = None

    # Task-specific metrics 예시:
    # - Classification: accuracy, top5_accuracy, f1_score
    # - Detection: mAP, mAP50, mAP75, precision, recall
    # - Segmentation: mIoU, pixel_accuracy
    # - OCR: CER (Character Error Rate), WER (Word Error Rate)
```

### 공통 인터페이스

```python
class TrainingAdapter(ABC):
    """모든 학습 프레임워크의 공통 인터페이스"""

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        output_dir: str,
        job_id: int
    ):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.output_dir = output_dir
        self.job_id = job_id

    # ========== 필수 구현 메서드 ==========

    @abstractmethod
    def prepare_model(self) -> None:
        """모델 초기화"""
        pass

    @abstractmethod
    def prepare_dataset(self) -> None:
        """데이터셋 로드 및 전처리"""
        pass

    @abstractmethod
    def train_epoch(self, epoch: int) -> MetricsResult:
        """1 에포크 학습"""
        pass

    @abstractmethod
    def validate(self, epoch: int) -> MetricsResult:
        """검증 수행"""
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int, metrics: MetricsResult) -> str:
        """체크포인트 저장"""
        pass

    # ========== 공통 메서드 ==========

    def train(self) -> List[MetricsResult]:
        """
        전체 학습 프로세스

        1. 모델 준비
        2. 데이터셋 준비
        3. 각 에포크마다:
           - 학습
           - 검증
           - MLflow 로깅
           - 체크포인트 저장
        """
        import mlflow

        self.prepare_model()
        self.prepare_dataset()

        all_metrics = []

        mlflow.set_experiment("vision-ai-training")
        with mlflow.start_run(run_name=f"job_{self.job_id}"):
            # Config 로깅
            mlflow.log_params({
                "model_name": self.model_config.model_name,
                "task_type": self.model_config.task_type.value,
                "epochs": self.training_config.epochs,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
            })

            # 학습 루프
            for epoch in range(self.training_config.epochs):
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate(epoch)

                combined_metrics = MetricsResult(
                    epoch=epoch,
                    step=train_metrics.step,
                    train_loss=train_metrics.train_loss,
                    val_loss=val_metrics.train_loss,
                    metrics={
                        **train_metrics.metrics,
                        **{f"val_{k}": v for k, v in val_metrics.metrics.items()}
                    }
                )

                all_metrics.append(combined_metrics)
                self.log_metrics_to_mlflow(combined_metrics)

                if epoch % 5 == 0 or epoch == self.training_config.epochs - 1:
                    self.save_checkpoint(epoch, combined_metrics)

        return all_metrics

    def log_metrics_to_mlflow(self, metrics: MetricsResult) -> None:
        """MLflow에 메트릭 로깅"""
        import mlflow

        log_dict = {"train_loss": metrics.train_loss}
        if metrics.val_loss:
            log_dict["val_loss"] = metrics.val_loss
        if metrics.metrics:
            log_dict.update(metrics.metrics)

        mlflow.log_metrics(log_dict, step=metrics.epoch)
```

---

## 구현 예시

### 1. UltralyticsAdapter

**지원 태스크**:
- Object Detection
- Instance Segmentation
- Pose Estimation
- Classification
- OBB

```python
from ultralytics import YOLO
from .base import TrainingAdapter, MetricsResult, TaskType

class UltralyticsAdapter(TrainingAdapter):
    """Ultralytics YOLO 모델용 Adapter"""

    TASK_SUFFIX_MAP = {
        TaskType.OBJECT_DETECTION: "",
        TaskType.INSTANCE_SEGMENTATION: "-seg",
        TaskType.POSE_ESTIMATION: "-pose",
        TaskType.IMAGE_CLASSIFICATION: "-cls",
    }

    def prepare_model(self):
        suffix = self.TASK_SUFFIX_MAP.get(self.task_type, "")
        model_path = f"{self.model_config.model_name}{suffix}.pt"
        self.model = YOLO(model_path)
        print(f"Loaded YOLO model: {model_path}")

    def prepare_dataset(self):
        # YOLO는 data.yaml 파일 필요
        self.data_yaml = self._create_data_yaml()

    def _create_data_yaml(self) -> str:
        """YOLO 형식 data.yaml 생성"""
        import yaml

        data = {
            'path': self.dataset_config.dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'nc': self.model_config.num_classes,
            'names': [f'class_{i}' for i in range(self.model_config.num_classes)]
        }

        yaml_path = f"{self.output_dir}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        return yaml_path

    def train(self) -> List[MetricsResult]:
        """YOLO는 자체 학습 API 사용"""
        self.prepare_model()
        self.prepare_dataset()

        # YOLO 학습
        results = self.model.train(
            data=self.data_yaml,
            epochs=self.training_config.epochs,
            imgsz=self.model_config.image_size,
            batch=self.training_config.batch_size,
            lr0=self.training_config.learning_rate,
            device=self.training_config.device,
            project=self.output_dir,
            name=f'job_{self.job_id}',
            exist_ok=True
        )

        return self._convert_yolo_results(results)

    def _convert_yolo_results(self, results) -> List[MetricsResult]:
        """YOLO 결과를 MetricsResult로 변환"""
        metrics_list = []
        for epoch, result in enumerate(results):
            metrics = MetricsResult(
                epoch=epoch,
                step=epoch,
                train_loss=result.get('train/box_loss', 0),
                val_loss=result.get('val/box_loss', 0),
                metrics={
                    'mAP50': result.get('metrics/mAP50', 0),
                    'mAP50-95': result.get('metrics/mAP50-95', 0),
                    'precision': result.get('metrics/precision', 0),
                    'recall': result.get('metrics/recall', 0),
                }
            )
            metrics_list.append(metrics)
        return metrics_list

    def save_checkpoint(self, epoch: int, metrics: MetricsResult) -> str:
        return f"{self.output_dir}/weights/best.pt"
```

### 2. TransformersAdapter

**지원 태스크**:
- Image Classification (ViT, DINOv2, Swin)
- Object Detection (DETR, OWL-ViT)
- Semantic Segmentation (SegFormer, Mask2Former)
- OCR (TrOCR, Donut)
- Image Captioning (BLIP-2, GIT)
- VQA (BLIP-2, LLaVA)

```python
from transformers import (
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForSemanticSegmentation,
    VisionEncoderDecoderModel,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from .base import TrainingAdapter, MetricsResult, TaskType

class TransformersAdapter(TrainingAdapter):
    """HuggingFace Transformers용 Adapter"""

    MODEL_CLASS_MAP = {
        TaskType.IMAGE_CLASSIFICATION: AutoModelForImageClassification,
        TaskType.OBJECT_DETECTION: AutoModelForObjectDetection,
        TaskType.SEMANTIC_SEGMENTATION: AutoModelForSemanticSegmentation,
        TaskType.OCR: VisionEncoderDecoderModel,
    }

    def prepare_model(self):
        model_class = self.MODEL_CLASS_MAP[self.task_type]

        self.model = model_class.from_pretrained(
            self.model_config.model_name,
            num_labels=self.model_config.num_classes,
            ignore_mismatched_sizes=True
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_config.model_name
        )

    def prepare_dataset(self):
        from datasets import load_dataset

        self.dataset = load_dataset(
            'imagefolder',
            data_dir=self.dataset_config.dataset_path
        )

        self.dataset = self.dataset.map(
            self._preprocess_function,
            batched=True
        )

    def _preprocess_function(self, examples):
        """이미지 전처리"""
        images = examples['image']
        inputs = self.processor(images, return_tensors='pt')
        inputs['labels'] = examples['label']
        return inputs

    def train(self) -> List[MetricsResult]:
        """HuggingFace Trainer API 사용"""
        self.prepare_model()
        self.prepare_dataset()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.training_config.epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            learning_rate=self.training_config.learning_rate,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            load_best_model_at_end=True,
            report_to="mlflow"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset.get('validation'),
            compute_metrics=self._compute_metrics
        )

        trainer.train()
        return self._convert_trainer_results(trainer)

    def _compute_metrics(self, eval_pred):
        """메트릭 계산 (task-specific)"""
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }

    def save_checkpoint(self, epoch: int, metrics: MetricsResult) -> str:
        return f"{self.output_dir}/checkpoint-{epoch}"
```

---

## 통합 방법

### 1. 엔트리포인트 (train.py)

```python
import argparse
import json
from adapters.base import ModelConfig, DatasetConfig, TrainingConfig, TaskType, DatasetFormat
from adapters.timm_adapter import TimmAdapter
from adapters.ultralytics_adapter import UltralyticsAdapter
from adapters.transformers_adapter import TransformersAdapter

# Adapter 레지스트리
ADAPTER_REGISTRY = {
    'timm': TimmAdapter,
    'ultralytics': UltralyticsAdapter,
    'transformers': TransformersAdapter,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--job_id', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Config 로드
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Config 객체 생성
    model_config = ModelConfig(
        framework=config['framework'],
        task_type=TaskType(config['task_type']),
        model_name=config['model_name'],
        num_classes=config.get('num_classes'),
        image_size=config.get('image_size', 224)
    )

    dataset_config = DatasetConfig(
        dataset_path=config['dataset_path'],
        format=DatasetFormat(config.get('dataset_format', 'imagefolder'))
    )

    training_config = TrainingConfig(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )

    # Adapter 선택 및 학습
    adapter_class = ADAPTER_REGISTRY[config['framework']]
    adapter = adapter_class(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        output_dir=args.output_dir,
        job_id=args.job_id
    )

    print(f"Starting training with {config['framework']} adapter...")
    metrics = adapter.train()
    print(f"Training completed! Final metrics: {metrics[-1]}")

if __name__ == '__main__':
    main()
```

### 2. Backend 호출 예시

```python
# mvp/backend/app/api/training.py
@router.post("/jobs/{job_id}/start")
async def start_training(job_id: int, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).get(job_id)

    # Config JSON 생성
    config = {
        'framework': job.framework,        # 'ultralytics', 'transformers'
        'task_type': job.task_type,        # 'object_detection', 'ocr'
        'model_name': job.model_name,      # 'yolov8n', 'microsoft/trocr-base'
        'num_classes': job.num_classes,
        'dataset_path': job.dataset_path,
        'dataset_format': job.dataset_format,
        'epochs': job.epochs,
        'batch_size': job.batch_size,
        'learning_rate': job.learning_rate,
    }

    config_path = f'/tmp/config_{job_id}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)

    # 학습 프로세스 시작
    process = subprocess.Popen([
        'python', 'mvp/training/train.py',
        '--config', config_path,
        '--job_id', str(job_id),
        '--output_dir', f'outputs/job_{job_id}'
    ])

    return {"status": "started", "pid": process.pid}
```

### 3. 사용 예시

#### Object Detection (YOLOv8)

```json
{
  "framework": "ultralytics",
  "task_type": "object_detection",
  "model_name": "yolov8n",
  "num_classes": 80,
  "dataset_path": "/data/coco",
  "dataset_format": "coco",
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.001
}
```

#### OCR (TrOCR)

```json
{
  "framework": "transformers",
  "task_type": "ocr",
  "model_name": "microsoft/trocr-base-printed",
  "dataset_path": "/data/text_images",
  "dataset_format": "custom",
  "epochs": 30,
  "batch_size": 8,
  "learning_rate": 0.00005
}
```

#### Image Classification (DINOv2)

```json
{
  "framework": "transformers",
  "task_type": "image_classification",
  "model_name": "facebook/dinov2-base",
  "num_classes": 10,
  "dataset_path": "/data/imagenet10",
  "dataset_format": "imagefolder",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001
}
```

---

## 로드맵

### Phase 1: MVP (현재 - 1개월)

**목표**: 기본 Adapter 구조 확립 및 3개 핵심 태스크 지원

- [x] Adapter 설계 문서 작성
- [ ] `base.py` 구현 (공통 인터페이스)
- [ ] `TimmAdapter` 리팩토링 (기존 ResNet)
- [ ] `UltralyticsAdapter` 구현 (YOLOv8 detection)
- [ ] `TransformersAdapter` 구현 (TrOCR)
- [ ] Backend API 확장 (framework, task_type 파라미터)
- [ ] Frontend UI 업데이트 (모델/태스크 선택)

**지원 태스크**:
1. Image Classification (timm/resnet50)
2. Object Detection (ultralytics/yolov8n)
3. OCR (transformers/trocr-base-printed)

### Phase 2: 확장 (1-3개월)

**목표**: 추가 태스크 및 모델 지원

- [ ] Instance Segmentation (yolov8n-seg)
- [ ] Semantic Segmentation (segformer-b0)
- [ ] Pose Estimation (yolov8n-pose)
- [ ] Image Classification (dinov2-base, vit-base)
- [ ] Zero-Shot Detection (grounding-dino)
- [ ] 데이터셋 형식 변환 유틸리티 (COCO ↔ YOLO)

**지원 모델 수**: 10+

### Phase 3: 고급 기능 (3-6개월)

**목표**: Vision-Language 태스크 및 고급 기능

- [ ] Image Captioning (BLIP-2)
- [ ] Visual QA (LLaVA)
- [ ] Depth Estimation (Depth Anything)
- [ ] Video Classification (VideoMAE)
- [ ] 멀티모달 학습 지원
- [ ] AutoML (하이퍼파라미터 자동 튜닝)
- [ ] Distributed Training (DDP)

**지원 모델 수**: 20+

### Phase 4: 프로덕션 (6-12개월)

**목표**: 엔터프라이즈급 기능

- [ ] Docker 컨테이너화
- [ ] Kubernetes 오케스트레이션
- [ ] Cloud GPU 통합 (AWS Batch, SageMaker)
- [ ] Model Serving (TorchServe, ONNX)
- [ ] A/B Testing
- [ ] Model Monitoring & Drift Detection

---

## 참고 자료

### 공식 문서

- **Ultralytics**: https://docs.ultralytics.com/
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **timm**: https://huggingface.co/docs/timm

### 논문

- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **YOLOv9**: https://arxiv.org/abs/2402.13616
- **Vision Transformer (ViT)**: https://arxiv.org/abs/2010.11929
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **DETR**: https://arxiv.org/abs/2005.12872
- **TrOCR**: https://arxiv.org/abs/2109.10282
- **BLIP-2**: https://arxiv.org/abs/2301.12597
- **Depth Anything**: https://arxiv.org/abs/2401.10891

### 관련 프로젝트

- **MMDetection**: https://github.com/open-mmlab/mmdetection
- **Detectron2**: https://github.com/facebookresearch/detectron2
- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR

---

**문서 버전**: 1.0
**최종 수정**: 2025-10-20
**작성자**: Claude Code
**리뷰 필요**: Backend API 설계, 데이터셋 변환 로직
