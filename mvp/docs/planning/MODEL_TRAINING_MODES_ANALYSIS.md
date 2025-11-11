# Model Training Modes Analysis

## Training Mode Definitions

### 1. Full Training (from scratch)
- 모델을 랜덤 초기화에서부터 완전히 새로 학습
- 대용량 데이터셋 필요 (보통 10만 장 이상)
- 학습 시간: 매우 김 (수일~수주)
- GPU 리소스: 매우 많이 필요
- 사용 시나리오:
  - 완전히 새로운 도메인
  - 대규모 데이터셋 확보
  - Pretrained 모델이 적합하지 않은 경우

### 2. Fine-tuning
- Pretrained 모델의 가중치를 시작점으로 추가 학습
- 소규모~중규모 데이터셋으로 가능 (보통 수백~수만 장)
- 학습 시간: 짧음 (수시간~수일)
- GPU 리소스: 적당
- 사용 시나리오:
  - 일반적인 사용 케이스 (가장 추천)
  - 제한된 데이터셋
  - 빠른 프로토타이핑

### 3. Inference Only
- 학습 불가, 추론만 가능
- Pretrained 가중치를 그대로 사용
- 리소스: 최소
- 사용 시나리오:
  - Zero-shot 모델 (YOLO-World 등)
  - 평가/데모 목적
  - 추가 학습이 불가능한 특수 모델

---

## Model-by-Model Analysis

### timm Models

#### ResNet-50 / ResNet-18
- **Supported Modes**: `full_training`, `fine_tuning`
- **Recommended**: `fine_tuning`
- **Pretrained**: ImageNet-1K (1.28M images, 1000 classes)
- **Full Training Requirements**:
  - Dataset: 100K+ images
  - GPU: 1x A100 or 4x V100
  - Time: 3-7 days
- **Fine-tuning Requirements**:
  - Dataset: 1K-100K images
  - GPU: 1x V100 or RTX 3090
  - Time: 2-24 hours
- **Notes**: Full training은 대규모 데이터셋에서만 권장

#### EfficientNet-B0 / EfficientNetV2-S
- **Supported Modes**: `full_training`, `fine_tuning`
- **Recommended**: `fine_tuning`
- **Pretrained**: ImageNet-1K
- **Full Training Requirements**:
  - Dataset: 100K+ images
  - GPU: 1x A100
  - Time: 5-10 days (compound scaling으로 더 오래 걸림)
- **Fine-tuning Requirements**:
  - Dataset: 500-50K images
  - GPU: 1x V100
  - Time: 2-12 hours
- **Notes**: EfficientNet은 특히 fine-tuning 효율이 좋음

---

### Ultralytics Models

#### YOLOv11n / YOLOv11m / YOLOv8n / YOLOv8s
- **Supported Modes**: `full_training`, `fine_tuning`
- **Recommended**: `fine_tuning`
- **Pretrained**: COCO (118K images, 80 classes)
- **Full Training Requirements**:
  - Dataset: 50K+ images (detection은 classification보다 적어도 됨)
  - GPU: 1x A100 or 2x V100
  - Time: 2-5 days
- **Fine-tuning Requirements**:
  - Dataset: 500-10K images
  - GPU: 1x RTX 3090 or V100
  - Time: 1-8 hours
- **Notes**: YOLO는 fine-tuning이 매우 효과적, 소규모 데이터셋도 OK

#### YOLO11n-seg / YOLO11m-seg / YOLOv8n-seg / YOLOv8s-seg
- **Supported Modes**: `full_training`, `fine_tuning`
- **Recommended**: `fine_tuning`
- **Pretrained**: COCO (segmentation masks)
- **Full Training Requirements**:
  - Dataset: 30K+ images with masks
  - GPU: 2x A100
  - Time: 3-7 days
- **Fine-tuning Requirements**:
  - Dataset: 300-5K images with masks
  - GPU: 1x V100
  - Time: 2-12 hours
- **Notes**: Segmentation은 annotation 비용이 높으므로 fine-tuning 강력 권장

#### YOLO11n-pose / YOLOv8n-pose
- **Supported Modes**: `full_training`, `fine_tuning`
- **Recommended**: `fine_tuning`
- **Pretrained**: COCO-Pose (person keypoints)
- **Full Training Requirements**:
  - Dataset: 20K+ images with keypoint annotations
  - GPU: 1x A100
  - Time: 2-5 days
- **Fine-tuning Requirements**:
  - Dataset: 500-5K images
  - GPU: 1x RTX 3090
  - Time: 1-6 hours
- **Notes**: Pose estimation은 annotation이 복잡하므로 fine-tuning 권장

#### YOLO-World-v2-s / YOLO-World-v2-m
- **Supported Modes**: `inference_only`, `fine_tuning`
- **Recommended**: `inference_only` (zero-shot), `fine_tuning` (성능 향상 시)
- **Pretrained**: Large-scale vision-language dataset (CC3M, Objects365, etc.)
- **Fine-tuning Requirements** (선택적):
  - Dataset: 1K-10K images
  - GPU: 1x V100
  - Time: 3-12 hours
  - Purpose: Zero-shot 성능 향상
- **Notes**:
  - Zero-shot detection이 주 목적
  - Fine-tuning은 선택적 (특정 도메인 성능 향상)
  - Full training 지원 안함 (vision-language 사전학습 필요)

---

### HuggingFace Models

#### Vision Transformer (ViT) - google/vit-base-patch16-224
- **Supported Modes**: `fine_tuning`, `full_training` (비권장)
- **Recommended**: `fine_tuning`
- **Pretrained**: ImageNet-21K → ImageNet-1K (두 단계 사전학습)
- **Full Training Requirements** (비권장):
  - Dataset: 1M+ images (ViT는 data-hungry)
  - GPU: 8x A100
  - Time: 2-4 weeks
  - Notes: ViT는 inductive bias 없어서 대량 데이터 필수
- **Fine-tuning Requirements**:
  - Dataset: 500-50K images
  - GPU: 1x V100
  - Time: 2-24 hours
- **Notes**: ViT는 반드시 fine-tuning 사용 (full training은 매우 비효율적)

#### D-FINE - ustc-community/dfine-x-coco
- **Supported Modes**: `fine_tuning`
- **Recommended**: `fine_tuning`
- **Pretrained**: COCO (object detection)
- **Full Training**: 지원 안함 (복잡한 사전학습 프로세스 필요)
- **Fine-tuning Requirements**:
  - Dataset: 1K-20K images (COCO format)
  - GPU: 1x A100 or 2x V100
  - Time: 6-24 hours
- **Notes**:
  - DETR 기반 모델로 fine-tuning만 지원
  - Full training은 매우 복잡 (multi-stage pretraining)

#### EoMT - tue-mps/eomt-vit-large
- **Supported Modes**: `fine_tuning`
- **Recommended**: `fine_tuning`
- **Pretrained**: Large-scale segmentation dataset
- **Full Training**: 지원 안함
- **Fine-tuning Requirements**:
  - Dataset: 1K-10K images with masks
  - GPU: 1x A100 (large model)
  - Time: 12-48 hours
- **Notes**:
  - ViT-Large 기반으로 fine-tuning만 가능
  - 대용량 모델 (304M params)이므로 full training 비현실적

#### Swin2SR (2x/4x) - caidas/swin2sr-classicalsr-x2-64, x4-64
- **Supported Modes**: `fine_tuning`
- **Recommended**: `fine_tuning`
- **Pretrained**: DIV2K, Flickr2K (HR-LR paired images)
- **Full Training**: 지원 안함 (paired data 구축 어려움)
- **Fine-tuning Requirements**:
  - Dataset: 500-5K HR-LR image pairs
  - GPU: 1x RTX 3090 or V100
  - Time: 24-72 hours (많은 iteration 필요)
- **Notes**:
  - Super-resolution은 paired data 필요
  - Full training은 대규모 paired dataset 구축이 어려움

---

## Summary Table

| Framework | Model | Full Training | Fine-tuning | Inference Only | Recommended |
|-----------|-------|---------------|-------------|----------------|-------------|
| timm | ResNet-50/18 | ✅ | ✅ | ❌ | Fine-tuning |
| timm | EfficientNet-B0 | ✅ | ✅ | ❌ | Fine-tuning |
| timm | EfficientNetV2-S | ✅ | ✅ | ❌ | Fine-tuning |
| ultralytics | YOLO11n/m | ✅ | ✅ | ❌ | Fine-tuning |
| ultralytics | YOLOv8n/s | ✅ | ✅ | ❌ | Fine-tuning |
| ultralytics | YOLO11n/m-seg | ✅ | ✅ | ❌ | Fine-tuning |
| ultralytics | YOLOv8n/s-seg | ✅ | ✅ | ❌ | Fine-tuning |
| ultralytics | YOLO11n-pose | ✅ | ✅ | ❌ | Fine-tuning |
| ultralytics | YOLOv8n-pose | ✅ | ✅ | ❌ | Fine-tuning |
| ultralytics | YOLO-World-v2-s/m | ❌ | ✅ | ✅ | Inference Only |
| huggingface | ViT-Base | ⚠️ | ✅ | ❌ | Fine-tuning |
| huggingface | D-FINE | ❌ | ✅ | ❌ | Fine-tuning |
| huggingface | EoMT | ❌ | ✅ | ❌ | Fine-tuning |
| huggingface | Swin2SR-2x/4x | ❌ | ✅ | ❌ | Fine-tuning |

**Legend:**
- ✅ Supported
- ⚠️ Supported but not recommended
- ❌ Not supported

---

## Recommended Defaults by Dataset Size

### Image Classification
- **< 1,000 images**: Fine-tuning only (small models like ResNet-18, EfficientNet-B0)
- **1,000 - 10,000 images**: Fine-tuning (any model)
- **10,000 - 100,000 images**: Fine-tuning (recommended) or Full training (optional)
- **> 100,000 images**: Fine-tuning or Full training (both viable)

### Object Detection
- **< 500 images**: Fine-tuning only (YOLO11n, YOLOv8n)
- **500 - 5,000 images**: Fine-tuning (any YOLO)
- **5,000 - 50,000 images**: Fine-tuning (recommended) or Full training (optional)
- **> 50,000 images**: Fine-tuning or Full training

### Segmentation
- **< 300 images**: Fine-tuning only (YOLO-seg small)
- **300 - 3,000 images**: Fine-tuning (any segmentation model)
- **> 3,000 images**: Fine-tuning (recommended) or Full training (optional)

### Super-Resolution
- **< 500 pairs**: Fine-tuning only
- **> 500 pairs**: Fine-tuning (full training 불가)

---

## Implementation Plan

### Phase 1: Schema Update
1. Add `training_modes` field to model registry:
   ```python
   "training_modes": {
       "supported": ["full_training", "fine_tuning"],  # or ["inference_only"]
       "recommended": "fine_tuning",
       "requirements": {
           "full_training": {
               "min_dataset_size": 100000,
               "min_gpu": "A100",
               "estimated_time": "3-7 days"
           },
           "fine_tuning": {
               "min_dataset_size": 1000,
               "min_gpu": "V100",
               "estimated_time": "2-24 hours"
           }
       }
   }
   ```

2. Update backend ModelInfo schema
3. Update frontend ModelInfo type

### Phase 2: UI Implementation
1. Add training mode badges to ModelCard
2. Show training mode details in ModelGuideDrawer
3. Add training mode filter (optional)

### Phase 3: Training Flow Integration
1. Disable/enable training options based on model's supported modes
2. Show different UI for inference-only models
3. Add warnings for full training with small datasets
4. Recommend fine-tuning based on dataset size

---

## Next Steps

1. ✅ Analyze all models
2. ⬜ Update model registries with training_modes field
3. ⬜ Update backend API schema
4. ⬜ Update frontend types
5. ⬜ Implement UI badges and filters
6. ⬜ Integrate with training flow
