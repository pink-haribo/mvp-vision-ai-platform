# HuggingFace Transformers Implementation Plan (Week 5-6)

**Document Version:** 2.0
**Created:** 2025-10-30
**Updated:** 2025-10-30
**Status:** Implementation Plan

---

## Executive Summary

Week 5-6에 HuggingFace Transformers 프레임워크를 추가하여 플랫폼의 task 다양성을 확장합니다.

**목표:**
- ✅ 4개의 다양한 task type 지원 (Classification, Detection, Segmentation, Super-Resolution)
- ✅ SOTA 최신 모델 적용 (D-FINE, EoMT - CVPR 2025)
- ✅ Docker 격리 환경 구축
- ✅ TransformersAdapter 구현

**모델 선정 (4개):**
1. **ViT** - Image Classification (기본)
2. **D-FINE** - Object Detection (SOTA, 2025-04 추가)
3. **EoMT** - Semantic Segmentation (CVPR 2025 Highlight)
4. **Swin2SR** - Super-Resolution (Image Restoration)

---

## 모델 상세 검토

### 1. ViT (Vision Transformer) - Image Classification

**기본 정보:**
- **Paper:** "An Image is Worth 16x16 Words" (ICLR 2021)
- **Model ID:** `google/vit-base-patch16-224`
- **Parameters:** 86M
- **Task:** Image Classification
- **Input Size:** 224×224
- **HuggingFace Support:** ✅ Full (AutoModelForImageClassification)

**주요 특징:**
- 이미지를 16×16 패치로 나누어 transformer에 입력
- ImageNet-21k pretrained → ImageNet-1k fine-tuned
- Attention 기반 global context 모델링

**구현 난이도:** ⭐ Low (가장 기본적인 모델)

**API 사용:**
```python
from transformers import ViTImageProcessor, ViTForImageClassification

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=num_classes
)
```

**학습 방법:**
- HuggingFace Trainer API 사용
- ImageFolder → HF Dataset 변환
- Standard cross-entropy loss

**예상 학습 시간:** ~30 min (10 epochs, sample dataset)

---

### 2. D-FINE - Object Detection

**기본 정보:**
- **Paper:** "D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement"
- **Model ID:** `ustc-community/dfine_x_coco`
- **Release:** 2024-10-17, HF 추가: 2025-04-29
- **Task:** Object Detection
- **Input Size:** 640×640
- **HuggingFace Support:** ✅ Full (DFineForObjectDetection)

**주요 특징:**
- DETR 기반의 real-time detector
- Fine-grained Distribution Refinement (FDR)
- Global Optimal Localization Self-Distillation (GO-LSD)
- **Performance:** 57.1% / 59.3% AP (SOTA in real-time detection)

**구현 난이도:** ⭐⭐ Medium (DETR 구조, bbox regression)

**API 사용:**
```python
from transformers import AutoImageProcessor, DFineForObjectDetection

processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_coco")
```

**학습 방법:**
- Dataset: COCO format
- Loss: FDR loss + GO-LSD distillation
- Metrics: mAP50, mAP50-95

**Dataset Format 요구사항:**
- COCO JSON annotations
- ImageFolder → COCO converter 필요

**예상 학습 시간:** ~2 hours (50 epochs, detection dataset)

---

### 3. EoMT (Encoder-only Mask Transformer) - Segmentation

**기본 정보:**
- **Paper:** "Your ViT is Secretly an Image Segmentation Model" (CVPR 2025 Highlight)
- **Model Range:** 0.3B ~ 7B parameters
- **Task:** Semantic Segmentation, Panoptic Segmentation
- **Input Size:** Flexible (ViT-based)
- **HuggingFace Support:** ✅ Main branch (transformers>=4.48)

**주요 특징:**
- **혁신:** Task-specific components 없이 순수 ViT로 segmentation
- **성능:** SOTA와 동등하면서 4x 빠름 (ViT-L)
- Encoder-only 구조로 단순화
- Large-scale pretraining으로 inductive bias 학습

**구현 난이도:** ⭐⭐⭐ Medium-High (segmentation output, mask processing)

**API 사용:**
```python
from transformers import EoMTModel, EoMTForSemanticSegmentation

model = EoMTForSemanticSegmentation.from_pretrained(
    "tue-mps/eomt-vit-large"
)
```

**학습 방법:**
- Dataset: Semantic segmentation masks
- Output: Per-pixel class predictions
- Metrics: mIoU, pixel accuracy

**Dataset Format 요구사항:**
- Images + segmentation masks (PNG)
- Class labels per pixel

**예상 학습 시간:** ~3 hours (100 epochs, segmentation dataset)

**도전 과제:**
- Mask 형태의 label 처리
- Large model size (7B까지)
- Post-processing (mask refinement)

---

### 4. Swin2SR - Super-Resolution

**기본 정보:**
- **Paper:** "Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution" (ECCV 2022)
- **Model ID:** `caidas/swin2sr-classicalsr-x2-64`
- **Parameters:** 11.9M
- **Task:** Super-Resolution, Image Restoration
- **Upscale Factor:** x2, x3, x4, x8
- **HuggingFace Support:** ✅ Full (Swin2SRModel)

**주요 특징:**
- SwinTransformer v2 기반
- 3가지 task 지원:
  1. Image Super-Resolution (2x/3x/4x/8x)
  2. JPEG Compression Artifact Removal
  3. Image Denoising
- Training stability 개선 (SwinV2 layers)

**구현 난이도:** ⭐⭐ Medium (새로운 task type이지만 구조 단순)

**API 사용:**
```python
from transformers import Swin2SRModel, Swin2SRImageProcessor

processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2sr-classicalsr-x2-64")
model = Swin2SRModel.from_pretrained("caidas/swin2sr-classicalsr-x2-64")
```

**학습 방법:**
- Dataset: HR-LR image pairs
- Loss: L1 loss (pixel-wise)
- Metrics: PSNR, SSIM

**Dataset Format 요구사항:**
- High-resolution images (target)
- Low-resolution images (input) - downscaled
- Paired structure: `HR/`, `LR_x2/`

**예상 학습 시간:** ~4 hours (500 epochs, SR dataset)

**새로운 Task Type 추가 필요:**
```python
# platform_sdk/base.py
class TaskType(Enum):
    ...
    SUPER_RESOLUTION = "super_resolution"  # 새로 추가
```

---

## 구현 우선순위 및 일정

### Week 5 (Day 1-7): 기반 구축 + ViT + Swin2SR

**Day 1-2: 기반 구축**
- [ ] `requirements-huggingface.txt` 작성
- [ ] `huggingface_models.py` 레지스트리 작성 (4개 모델)
- [ ] `TaskType.SUPER_RESOLUTION` 추가
- [ ] Dockerfile.huggingface 작성
- [ ] Docker 이미지 빌드 테스트

**Day 3-4: ViT (Classification)**
- [ ] `TransformersAdapter` 기본 구조 작성
- [ ] Classification task 구현
- [ ] ImageFolder → HF Dataset 변환
- [ ] HF Trainer API 통합
- [ ] ViT 학습 테스트 (sample_dataset)

**Day 5-7: Swin2SR (Super-Resolution)**
- [ ] Super-Resolution task 구현
- [ ] HR-LR dataset 로더 작성
- [ ] PSNR/SSIM metric 계산
- [ ] Swin2SR 학습 테스트
- [ ] Frontend SR task UI 추가

**Milestone:** ViT + Swin2SR 검증 완료

---

### Week 6 (Day 8-12): D-FINE + EoMT + 통합

**Day 8-9: D-FINE (Detection)**
- [ ] Detection task 구현
- [ ] COCO format dataset 지원
- [ ] Bounding box visualization
- [ ] mAP metric 계산
- [ ] D-FINE 학습 테스트

**Day 10-11: EoMT (Segmentation)**
- [ ] Segmentation task 구현
- [ ] Mask dataset 로더 작성
- [ ] mIoU metric 계산
- [ ] Mask visualization
- [ ] EoMT 학습 테스트

**Day 12: 통합 및 검증**
- [ ] Backend API 통합
- [ ] Frontend 4개 task 지원
- [ ] Docker 환경 전체 테스트
- [ ] 성능 벤치마크
- [ ] 문서 업데이트

**Milestone:** 4개 모델 전체 검증 완료

---

## 기술 아키텍처

### 1. Requirements (requirements-huggingface.txt)

```txt
# Base PyTorch (from requirements-base.txt)
# torch==2.1.0
# torchvision==0.16.0

# HuggingFace Core
transformers==4.48.0
accelerate==0.25.0
datasets==2.16.0

# Vision Processing
opencv-python==4.8.1.78
albumentations==1.3.1

# Evaluation
evaluate==0.4.1
scikit-learn==1.3.2

# Metrics
scikit-image==0.22.0  # PSNR, SSIM for SR
pycocotools==2.0.7    # COCO evaluation for detection
```

**Size Estimate:** ~2.5GB (transformers 매우 큼)

---

### 2. Model Registry (huggingface_models.py)

```python
"""HuggingFace Transformers model registry."""

from typing import Dict, Any

HUGGINGFACE_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {

    # ========== Image Classification ==========

    "google/vit-base-patch16-224": {
        "display_name": "Vision Transformer (ViT) Base",
        "description": "Transformer-based image classification - Attention-based global context",
        "framework": "huggingface",
        "task_type": "image_classification",
        "model_id": "google/vit-base-patch16-224",
        "params": "86M",
        "input_size": 224,
        "pretrained_available": True,
        "recommended_batch_size": 32,
        "recommended_lr": 3e-4,
        "tags": ["p1", "transformer", "attention", "imagenet", "2021"],
        "priority": 1,

        "architecture": {
            "type": "Vision Transformer",
            "patch_size": 16,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
        },

        "performance": {
            "imagenet_top1": "81.3%",
            "imagenet_top5": "96.5%",
            "inference_speed": "~50 images/sec (V100)",
        },

        "use_cases": [
            {
                "title": "General Image Classification",
                "description": "Fine-grained classification with global context understanding",
                "dataset": "Custom ImageNet-style dataset",
                "metrics": {
                    "before": "ResNet-50: 76.1% accuracy",
                    "after": "ViT-Base: 81.3% accuracy with attention visualization"
                }
            }
        ]
    },

    # ========== Object Detection ==========

    "ustc-community/dfine_x_coco": {
        "display_name": "D-FINE (Detection Fine-grained)",
        "description": "SOTA real-time detector - Fine-grained bbox refinement (57.1% AP)",
        "framework": "huggingface",
        "task_type": "object_detection",
        "model_id": "ustc-community/dfine_x_coco",
        "params": "67M",
        "input_size": 640,
        "pretrained_available": True,
        "recommended_batch_size": 8,
        "recommended_lr": 1e-4,
        "tags": ["p1", "detection", "detr", "real-time", "sota", "2024"],
        "priority": 1,

        "architecture": {
            "type": "DETR-based",
            "backbone": "ResNet-50",
            "features": ["FDR", "GO-LSD"],
        },

        "performance": {
            "coco_map50": "57.1%",
            "coco_map50-95": "40.8%",
            "inference_speed": "Real-time (>30 FPS)",
        },

        "use_cases": [
            {
                "title": "Precise Object Localization",
                "description": "High-precision bounding box detection for industrial inspection",
                "dataset": "Custom COCO-format dataset",
                "metrics": {
                    "before": "YOLOv8: 50.2% mAP50",
                    "after": "D-FINE: 57.1% mAP50 with fine-grained localization"
                }
            }
        ]
    },

    # ========== Semantic Segmentation ==========

    "tue-mps/eomt-vit-large": {
        "display_name": "EoMT (Encoder-only Mask Transformer)",
        "description": "CVPR 2025 Highlight - Segmentation without task-specific components",
        "framework": "huggingface",
        "task_type": "semantic_segmentation",
        "model_id": "tue-mps/eomt-vit-large",
        "params": "304M",
        "input_size": 518,
        "pretrained_available": True,
        "recommended_batch_size": 4,
        "recommended_lr": 1e-4,
        "tags": ["p1", "segmentation", "vit", "encoder-only", "cvpr2025"],
        "priority": 1,

        "architecture": {
            "type": "Encoder-only ViT",
            "backbone": "ViT-Large",
            "innovation": "No task-specific decoder",
        },

        "performance": {
            "ade20k_miou": "53.0%",
            "inference_speed": "4x faster than Mask2Former",
        },

        "use_cases": [
            {
                "title": "Fast Semantic Segmentation",
                "description": "Efficient pixel-wise classification for autonomous driving",
                "dataset": "Custom segmentation masks",
                "metrics": {
                    "before": "Mask2Former: 50.1% mIoU, 2.5s/image",
                    "after": "EoMT: 53.0% mIoU, 0.6s/image (4x faster)"
                }
            }
        ]
    },

    # ========== Super-Resolution ==========

    "caidas/swin2sr-classicalsr-x2-64": {
        "display_name": "Swin2SR (2x Super-Resolution)",
        "description": "Image restoration and super-resolution - 2x upscaling",
        "framework": "huggingface",
        "task_type": "super_resolution",
        "model_id": "caidas/swin2sr-classicalsr-x2-64",
        "params": "11.9M",
        "input_size": "variable",
        "upscale_factor": 2,
        "pretrained_available": True,
        "recommended_batch_size": 16,
        "recommended_lr": 2e-4,
        "tags": ["p1", "super-resolution", "restoration", "swin", "2022"],
        "priority": 1,

        "architecture": {
            "type": "Swin Transformer V2",
            "window_size": 8,
            "features": ["Residual Swin Transformer Blocks"],
        },

        "performance": {
            "psnr": "33.89 dB (Set5 dataset)",
            "ssim": "0.9195",
            "inference_speed": "~20 images/sec (512x512 → 1024x1024)",
        },

        "use_cases": [
            {
                "title": "Image Quality Enhancement",
                "description": "Upscale low-resolution medical images for better diagnosis",
                "dataset": "HR-LR paired images",
                "metrics": {
                    "before": "Bicubic: 30.1 dB PSNR",
                    "after": "Swin2SR: 33.9 dB PSNR with artifact removal"
                }
            }
        ]
    },
}
```

---

### 3. TransformersAdapter Structure

```python
# mvp/training/adapters/transformers_adapter.py

from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

class TransformersAdapter(TrainingAdapter):
    """HuggingFace Transformers adapter for vision tasks."""

    def __init__(self, ...):
        super().__init__(...)
        self.task_type = model_config.task_type
        self.processor = None
        self.trainer = None

    def prepare_model(self):
        """Load model based on task type."""
        if self.task_type == TaskType.IMAGE_CLASSIFICATION:
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_config.model_name,
                num_labels=self.model_config.num_classes
            )
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_config.model_name
            )

        elif self.task_type == TaskType.OBJECT_DETECTION:
            from transformers import DFineForObjectDetection
            self.model = DFineForObjectDetection.from_pretrained(
                self.model_config.model_name
            )
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_config.model_name
            )

        elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            from transformers import EoMTForSemanticSegmentation
            self.model = EoMTForSemanticSegmentation.from_pretrained(
                self.model_config.model_name
            )
            # ... processor setup

        elif self.task_type == TaskType.SUPER_RESOLUTION:
            from transformers import Swin2SRModel
            self.model = Swin2SRModel.from_pretrained(
                self.model_config.model_name
            )
            # ... processor setup

    def prepare_dataset(self):
        """Convert ImageFolder to HF Dataset."""
        from datasets import Dataset, Image

        if self.task_type == TaskType.IMAGE_CLASSIFICATION:
            # ImageFolder → HF Dataset with image processor
            self.train_dataset = self._create_classification_dataset("train")
            self.val_dataset = self._create_classification_dataset("val")

        elif self.task_type == TaskType.OBJECT_DETECTION:
            # COCO format → HF Dataset
            self.train_dataset = self._create_detection_dataset("train")
            self.val_dataset = self._create_detection_dataset("val")

        # ... other tasks

    def train_epoch(self, epoch: int) -> MetricsResult:
        """Train using HF Trainer API."""
        # Configure TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,  # Single epoch
            per_device_train_batch_size=self.training_config.batch_size,
            learning_rate=self.training_config.learning_rate,
            # ...
        )

        # Create Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            callbacks=[MLflowCallback(self.job_id)],  # Custom callback
        )

        # Train
        self.trainer.train()

        # Extract metrics
        metrics = self.trainer.state.log_history[-1]
        return self._convert_metrics(metrics, epoch)

    def validate(self, epoch: int) -> MetricsResult:
        """Evaluate using HF Trainer."""
        eval_results = self.trainer.evaluate()
        return self._convert_metrics(eval_results, epoch)

    def _convert_metrics(self, hf_metrics: dict, epoch: int) -> MetricsResult:
        """Convert HF metrics to platform MetricsResult."""
        # Task-specific metric conversion
        if self.task_type == TaskType.IMAGE_CLASSIFICATION:
            return MetricsResult(
                epoch=epoch,
                step=0,
                train_loss=hf_metrics.get('loss', 0.0),
                val_loss=hf_metrics.get('eval_loss', 0.0),
                metrics={
                    'accuracy': hf_metrics.get('eval_accuracy', 0.0),
                }
            )
        # ... other tasks
```

---

### 4. Dockerfile.huggingface

```dockerfile
# mvp/docker/Dockerfile.huggingface
FROM vision-platform-base:latest

# Set working directory
WORKDIR /workspace

# Copy HuggingFace requirements
COPY training/requirements/requirements-base.txt /tmp/
COPY training/requirements/requirements-huggingface.txt /tmp/

# Install HuggingFace dependencies
RUN pip install --no-cache-dir -r /tmp/requirements-huggingface.txt && \
    rm /tmp/requirements-huggingface.txt /tmp/requirements-base.txt

# Copy HuggingFace adapter
COPY training/adapters/__init__.py /opt/vision-platform/adapters/
COPY training/adapters/transformers_adapter.py /opt/vision-platform/adapters/

# Copy HuggingFace model registry
COPY training/model_registry/__init__.py /opt/vision-platform/model_registry/
COPY training/model_registry/huggingface_models.py /opt/vision-platform/model_registry/

# Copy validators
COPY training/validators/ /opt/vision-platform/validators/

# Verify installation
RUN python -c "import transformers; print(f'transformers: {transformers.__version__}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import transformers; print('OK')" || exit 1
```

**Image Size Estimate:** ~12GB (transformers + models 매우 큼)

---

## 주요 구현 도전 과제

### 1. HuggingFace Trainer API 통합

**Challenge:** PyTorch training loop과 완전히 다른 구조

**Solution:**
- TrainerCallback으로 MLflow 로깅 연결
- `trainer.state.log_history`에서 metrics 추출
- Custom callback으로 epoch마다 DB 저장

```python
class MLflowCallback(TrainerCallback):
    """Custom callback to log metrics to MLflow."""

    def __init__(self, job_id):
        self.job_id = job_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged."""
        # logs에서 metrics 추출 → MLflow 로깅
        mlflow.log_metrics(logs, step=state.global_step)
```

---

### 2. Dataset 변환

**Challenge:** ImageFolder → HuggingFace Dataset 변환

**Solution:**
```python
from datasets import Dataset, Image as HFImage
from PIL import Image
import os

def _create_classification_dataset(self, split: str):
    """Convert ImageFolder to HF Dataset."""
    data_dir = os.path.join(self.dataset_config.dataset_path, split)

    # Collect image paths and labels
    images = []
    labels = []
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        for img_name in os.listdir(label_dir):
            images.append(os.path.join(label_dir, img_name))
            labels.append(label_name)

    # Create HF Dataset
    dataset = Dataset.from_dict({
        'image': images,
        'label': labels
    })

    # Cast image column to HF Image type
    dataset = dataset.cast_column('image', HFImage())

    # Apply image processor
    def preprocess(examples):
        images = [img.convert("RGB") for img in examples['image']]
        inputs = self.processor(images, return_tensors="pt")
        inputs['labels'] = examples['label']
        return inputs

    dataset = dataset.map(preprocess, batched=True)
    return dataset
```

---

### 3. Super-Resolution Dataset

**Challenge:** HR-LR paired images 로딩

**Dataset Structure:**
```
dataset/
├── train/
│   ├── HR/
│   │   ├── 0001.png (1024×1024)
│   │   └── 0002.png
│   └── LR_x2/
│       ├── 0001.png (512×512)
│       └── 0002.png
└── val/
    ├── HR/
    └── LR_x2/
```

**Solution:**
```python
def _create_sr_dataset(self, split: str):
    """Create SR dataset with HR-LR pairs."""
    hr_dir = os.path.join(self.dataset_config.dataset_path, split, "HR")
    lr_dir = os.path.join(self.dataset_config.dataset_path, split, f"LR_x{self.upscale_factor}")

    # Collect paired images
    hr_images = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])
    lr_images = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])

    dataset = Dataset.from_dict({
        'lr_image': lr_images,
        'hr_image': hr_images
    })

    dataset = dataset.cast_column('lr_image', HFImage())
    dataset = dataset.cast_column('hr_image', HFImage())

    return dataset
```

---

### 4. Detection Dataset (COCO Format)

**Challenge:** COCO JSON annotations → HF Dataset

**Solution:**
```python
from pycocotools.coco import COCO

def _create_detection_dataset(self, split: str):
    """Convert COCO format to HF Dataset."""
    image_dir = os.path.join(self.dataset_config.dataset_path, split)
    ann_file = os.path.join(self.dataset_config.dataset_path, f"{split}.json")

    coco = COCO(ann_file)
    image_ids = coco.getImgIds()

    images = []
    annotations = []

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        images.append(os.path.join(image_dir, img_info['file_name']))
        annotations.append({
            'boxes': [ann['bbox'] for ann in anns],
            'labels': [ann['category_id'] for ann in anns]
        })

    dataset = Dataset.from_dict({
        'image': images,
        'annotations': annotations
    })

    return dataset
```

---

## TaskType 추가

**platform_sdk/base.py 업데이트:**
```python
class TaskType(Enum):
    """Supported task types."""
    # Vision
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    SUPER_RESOLUTION = "super_resolution"  # 새로 추가 ✅

    # Vision-Language
    IMAGE_CAPTIONING = "image_captioning"
    VISUAL_QA = "visual_qa"
    OCR = "ocr"
    DOCUMENT_UNDERSTANDING = "document_understanding"

    # Zero-Shot
    ZERO_SHOT_CLASSIFICATION = "zero_shot_classification"
    ZERO_SHOT_DETECTION = "zero_shot_detection"
```

**Metrics 정의 추가:**
```python
TASK_PRIMARY_METRICS = {
    TaskType.IMAGE_CLASSIFICATION: 'accuracy',
    TaskType.OBJECT_DETECTION: 'mAP50',
    TaskType.SEMANTIC_SEGMENTATION: 'miou',
    TaskType.SUPER_RESOLUTION: 'psnr',  # 새로 추가 ✅
}

TASK_STANDARD_METRICS = {
    # ... existing ...

    TaskType.SUPER_RESOLUTION: {  # 새로 추가 ✅
        'psnr': MetricDefinition(
            label='PSNR',
            format='float',
            higher_is_better=True,
            description='Peak Signal-to-Noise Ratio (dB)'
        ),
        'ssim': MetricDefinition(
            label='SSIM',
            format='percent',
            higher_is_better=True,
            description='Structural Similarity Index'
        ),
        'lpips': MetricDefinition(
            label='LPIPS',
            format='float',
            higher_is_better=False,
            description='Learned Perceptual Image Patch Similarity'
        ),
    },
}
```

---

## Backend 통합

### TrainingManager 업데이트

```python
# mvp/backend/app/utils/training_manager.py

IMAGE_MAP = {
    "timm": "vision-platform-timm:latest",
    "ultralytics": "vision-platform-ultralytics:latest",
    "huggingface": "vision-platform-huggingface:latest",  # 추가 ✅
}
```

### Adapter Registry 업데이트

```python
# mvp/training/adapters/__init__.py

try:
    from .transformers_adapter import TransformersAdapter
except ImportError:
    TransformersAdapter = None

# Adapter registry
ADAPTER_REGISTRY = {}
if TimmAdapter is not None:
    ADAPTER_REGISTRY['timm'] = TimmAdapter
if UltralyticsAdapter is not None:
    ADAPTER_REGISTRY['ultralytics'] = UltralyticsAdapter
if TransformersAdapter is not None:
    ADAPTER_REGISTRY['huggingface'] = TransformersAdapter  # 추가 ✅
```

---

## Frontend 지원

### Task Type UI 추가

**TrainingConfigPanel.tsx 업데이트:**
```typescript
const TASK_TYPES = [
  { value: 'image_classification', label: '이미지 분류', icon: '🖼️' },
  { value: 'object_detection', label: '객체 검출', icon: '🎯' },
  { value: 'semantic_segmentation', label: '시맨틱 분할', icon: '🗺️' },
  { value: 'super_resolution', label: '초해상도', icon: '🔍' },  // 추가 ✅
];
```

---

## 테스트 계획

### 1. Unit Tests

**ViT Classification:**
```python
def test_vit_classification():
    adapter = TransformersAdapter(
        model_config=ModelConfig(
            framework='huggingface',
            task_type=TaskType.IMAGE_CLASSIFICATION,
            model_name='google/vit-base-patch16-224',
            num_classes=2
        ),
        # ...
    )
    adapter.prepare_model()
    adapter.prepare_dataset()
    metrics = adapter.train_epoch(1)
    assert metrics.metrics['accuracy'] > 0
```

**Swin2SR Super-Resolution:**
```python
def test_swin2sr():
    adapter = TransformersAdapter(
        model_config=ModelConfig(
            framework='huggingface',
            task_type=TaskType.SUPER_RESOLUTION,
            model_name='caidas/swin2sr-classicalsr-x2-64',
        ),
        # ...
    )
    adapter.prepare_model()
    metrics = adapter.train_epoch(1)
    assert metrics.metrics['psnr'] > 0
```

### 2. Integration Tests

**Docker Environment:**
```bash
# Build image
cd mvp/docker
./build.sh

# Test ViT training
docker run --rm \
  -v $(pwd)/data/datasets/sample_dataset:/workspace/dataset:ro \
  -v $(pwd)/data/outputs:/workspace/output:rw \
  vision-platform-huggingface:latest \
  python /opt/vision-platform/train.py \
    --framework huggingface \
    --task_type image_classification \
    --model_name google/vit-base-patch16-224 \
    --dataset_path /workspace/dataset \
    --epochs 2 \
    --job_id 200
```

### 3. E2E Tests

**Frontend → Backend → Training:**
1. Frontend에서 ViT 모델 선택
2. 학습 시작 (sample_dataset)
3. 실시간 metrics 확인
4. 학습 완료 후 모델 다운로드
5. Inference 테스트

---

## 성능 벤치마크 (예상)

| Model | Task | Dataset | Epochs | Time | Metric |
|-------|------|---------|--------|------|--------|
| ViT | Classification | sample (64 imgs) | 10 | 30 min | 85% acc |
| D-FINE | Detection | COCO subset (500 imgs) | 50 | 2 hours | 50% mAP50 |
| EoMT | Segmentation | ADE20K subset | 100 | 3 hours | 45% mIoU |
| Swin2SR | Super-Res | DIV2K subset (800 imgs) | 500 | 4 hours | 33 dB PSNR |

---

## 리스크 및 완화 방안

### 1. Transformers 라이브러리 크기

**Risk:** transformers 패키지가 매우 크고 의존성 많음

**Mitigation:**
- Docker layer caching 활용
- Base image에 공통 의존성 포함
- `--no-cache-dir` 옵션으로 불필요한 파일 제거

### 2. Model Download 시간

**Risk:** Pretrained weights 다운로드에 시간 소요

**Mitigation:**
- Docker 이미지 빌드 시 모델 사전 다운로드
- HuggingFace cache 활용 (`~/.cache/huggingface/`)

### 3. Super-Resolution Dataset 부족

**Risk:** HR-LR paired dataset 준비 어려움

**Mitigation:**
- DIV2K dataset 사용 (공개 데이터셋)
- On-the-fly downsampling으로 LR 이미지 생성
- Sample dataset 제공

### 4. EoMT 모델 크기

**Risk:** 7B 모델은 메모리 부족 가능

**Mitigation:**
- 작은 모델부터 시작 (ViT-Base: 0.3B)
- Gradient checkpointing 활용
- Mixed precision training (fp16)

---

## Deliverables

**Week 5 완료 시:**
- [ ] TransformersAdapter 구현
- [ ] ViT classification 검증
- [ ] Swin2SR super-resolution 검증
- [ ] Dockerfile.huggingface
- [ ] Docker 이미지 빌드 완료

**Week 6 완료 시:**
- [ ] D-FINE detection 검증
- [ ] EoMT segmentation 검증
- [ ] 4개 모델 전체 통합 테스트
- [ ] Backend API 완전 지원
- [ ] Frontend 4개 task UI
- [ ] 문서 업데이트 (이 계획서)

---

## 다음 단계 (Week 7-8)

Week 7-8에는 Phase 3 (Custom Models)를 진행합니다:
- ConvNeXt (GitHub custom)
- YOLOv7 (GitHub custom)
- PP-YOLO (PaddlePaddle)
- ViTPose (MMPose)

이를 통해 플랫폼의 극한 확장성을 검증합니다.

---

## 참고 문서

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [D-FINE Paper](https://arxiv.org/abs/2410.13842)
- [EoMT Paper](https://arxiv.org/abs/2503.19108)
- [Swin2SR Paper](https://arxiv.org/abs/2209.11345)
- [IMPLEMENTATION_PRIORITY_ANALYSIS.md](./IMPLEMENTATION_PRIORITY_ANALYSIS.md)

---

*Document Version: 2.0*
*Created: 2025-10-30*
*Author: Claude Code*
