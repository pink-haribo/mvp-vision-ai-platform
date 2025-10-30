# Week 1 Model Selection - Revised (2024 Latest)

**Document Version:** 2.0
**Updated:** 2025-10-30
**Status:** Implementation Ready

---

## 변경 이유

**기존 계획의 문제점**:
- ❌ YOLOv8만 포함, YOLOv11 (2024년 9월 최신) 누락
- ❌ YOLO-World (open-vocabulary detection) 같은 혁신적 모델 미포함
- ❌ timm 최신 모델 (NaFlexViT, MobileNetV4/V5) 누락
- ❌ EfficientNetV2 누락

**개선된 선택 기준**:
1. ⭐ **최신성 우선**: 2024년 출시 모델 포함
2. 🚀 **특수 기능**: Open-vocabulary, SAM 등 차별화 기능
3. 📊 **실용성**: 실제 프로덕션에서 많이 사용
4. 📏 **크기 범위**: nano ~ xlarge 다양한 크기
5. 🏗️ **아키텍처 다양성**: CNN, ViT, Hybrid 골고루

---

## 최신 모델 현황 (2024)

### Ultralytics 최신

| 모델 | 출시 | 주요 특징 |
|------|------|-----------|
| **YOLOv11** | 2024년 9월 | YOLOv8 대비 22% 적은 파라미터, 더 높은 mAP, C3k2/SPPF/C2PSA 블록 |
| **YOLO-World** | CVPR 2024 | Open-vocabulary detection, zero-shot, custom prompts |
| **YOLOv10** | 2024년 5월 | NMS-free, 효율성 개선 |
| **YOLOv9** | 2024년 초 | Programmable gradient, generalized ELAN |

### timm 최신

| 모델 | 출시 | 주요 특징 |
|------|------|-----------|
| **NaFlexViT** | 2024년 후반 | Variable aspect/resolution, FlexiViT + NaViT + NaFlex 통합 |
| **MobileNetV5** | 2024년 | Gemma 3n encoder, 최신 모바일 아키텍처 |
| **MobileNetV4** | 2024년 | Universal Inverted Bottleneck |
| **SigLIP-2** | 2024년 | NaFlex ViT encoder, 88.1% top-1 |
| **EfficientNetV2** | 2021년 | EfficientNetV1보다 훨씬 개선 (누락되어 있었음) |

---

## Week 1 추가 모델 (최종안)

### 🔥 Priority 1: 반드시 추가 (최신 + 핵심)

#### Ultralytics (12개)

##### YOLOv11 (최신, 필수!) - 5개
```python
"yolo11n": {
    "display_name": "YOLOv11 Nano",
    "description": "Latest YOLO (Sep 2024) - Ultra-lightweight, 22% fewer params than YOLOv8",
    "params": "2.6M",
    "task_type": "object_detection",
    "input_size": 640,
    "recommended_batch_size": 64,
    "recommended_lr": 0.01,
    "tags": ["latest", "2024", "ultralight", "realtime", "sota"]
}

"yolo11s": {
    "display_name": "YOLOv11 Small",
    "params": "9.4M",
    "tags": ["latest", "2024", "lightweight", "fast", "sota"]
}

"yolo11m": {
    "display_name": "YOLOv11 Medium",
    "params": "20.1M",
    "tags": ["latest", "2024", "balanced", "sota"]
}

"yolo11l": {
    "display_name": "YOLOv11 Large",
    "params": "25.3M",
    "tags": ["latest", "2024", "accurate", "sota"]
}

"yolo11x": {
    "display_name": "YOLOv11 Extra-Large",
    "params": "56.9M",
    "tags": ["latest", "2024", "heavy", "maximum-accuracy", "sota"]
}
```

##### YOLO-World (혁신적!) - 2개
```python
"yolo_world_v2_s": {
    "display_name": "YOLO-World-v2 Small",
    "description": "Open-vocabulary detection (CVPR 2024) - Detect ANY object with text prompts",
    "params": "22M",
    "task_type": "open_vocabulary_detection",  # 신규 task!
    "input_size": 640,
    "recommended_batch_size": 16,
    "recommended_lr": 0.01,
    "tags": ["cvpr2024", "open-vocab", "zero-shot", "innovative", "text-prompt"]
}

"yolo_world_v2_m": {
    "display_name": "YOLO-World-v2 Medium",
    "description": "Open-vocabulary detection - 35.4 AP @ 52 FPS on LVIS",
    "params": "42M",
    "tags": ["cvpr2024", "open-vocab", "zero-shot", "accurate"]
}
```

##### Segmentation (최신) - 3개
```python
"yolo11n_seg": {
    "display_name": "YOLOv11n-Seg",
    "description": "Latest segmentation model (Sep 2024)",
    "params": "2.9M",
    "task_type": "instance_segmentation",
    "tags": ["latest", "2024", "segmentation", "ultralight"]
}

"yolo11m_seg": {
    "display_name": "YOLOv11m-Seg",
    "params": "22.4M",
    "tags": ["latest", "2024", "segmentation", "balanced"]
}

"yolo11x_seg": {
    "display_name": "YOLOv11x-Seg",
    "params": "62.1M",
    "tags": ["latest", "2024", "segmentation", "accurate"]
}
```

##### Pose (최신) - 2개
```python
"yolo11m_pose": {
    "display_name": "YOLOv11m-Pose",
    "description": "Latest pose estimation - 17 keypoints",
    "params": "21.8M",
    "task_type": "pose_estimation",
    "tags": ["latest", "2024", "pose", "keypoints"]
}

"yolo11l_pose": {
    "display_name": "YOLOv11l-Pose",
    "params": "26.9M",
    "tags": ["latest", "2024", "pose", "accurate"]
}
```

---

#### timm (15개)

##### 최신 Mobile (2024) - 3개
```python
"mobilenetv4_conv_medium": {
    "display_name": "MobileNetV4-Medium",
    "description": "Latest mobile architecture (2024) with Universal Inverted Bottleneck",
    "params": "9.7M",
    "input_size": 224,
    "pretrained_available": True,
    "recommended_batch_size": 128,
    "recommended_lr": 0.001,
    "tags": ["latest", "2024", "mobile", "efficient", "uib"]
}

"mobilenetv5_large": {
    "display_name": "MobileNetV5-Large",
    "description": "Cutting-edge mobile model for Gemma 3n (2024)",
    "params": "12M",
    "tags": ["latest", "2024", "mobile", "gemma", "sota"]
}

"mobilenetv3_large_100": {
    "display_name": "MobileNetV3-Large",
    "description": "Popular mobile CNN (baseline comparison)",
    "params": "5.5M",
    "tags": ["mobile", "efficient", "baseline"]
}
```

##### 최신 ViT (2024) - 3개
```python
"vit_so150m_patch16_224": {
    "display_name": "ViT-SO150M/16 (SigLIP-2)",
    "description": "SigLIP-2 NaFlex ViT (2024) - 88.1% top-1 accuracy",
    "params": "150M",
    "input_size": 224,
    "recommended_batch_size": 32,
    "recommended_lr": 0.0003,
    "tags": ["latest", "2024", "vit", "siglip", "sota", "88.1%"]
}

"vit_base_patch16_224": {
    "display_name": "ViT-Base/16",
    "description": "Standard Vision Transformer (baseline)",
    "params": "86M",
    "tags": ["vit", "transformer", "baseline"]
}

"vit_large_patch16_224": {
    "display_name": "ViT-Large/16",
    "description": "Large Vision Transformer",
    "params": "307M",
    "tags": ["vit", "transformer", "heavy"]
}
```

##### EfficientNet 계열 - 4개
```python
"efficientnetv2_s": {
    "display_name": "EfficientNetV2-Small",
    "description": "Improved EfficientNet (2021) - Training efficiency++",
    "params": "21.5M",
    "input_size": 384,
    "recommended_batch_size": 64,
    "recommended_lr": 0.001,
    "tags": ["efficient", "modern", "v2", "fast-training"]
}

"efficientnetv2_m": {
    "display_name": "EfficientNetV2-Medium",
    "params": "54M",
    "input_size": 480,
    "tags": ["efficient", "modern", "v2"]
}

"efficientnet_b0": {
    "display_name": "EfficientNet-B0",
    "description": "Original EfficientNet (baseline)",
    "params": "5.3M",
    "input_size": 224,
    "tags": ["efficient", "lightweight", "baseline"]
}

"efficientnet_b4": {
    "display_name": "EfficientNet-B4",
    "params": "19M",
    "input_size": 380,
    "tags": ["efficient", "accurate"]
}
```

##### ResNet 계열 (baseline) - 3개
```python
"resnet18": {
    "display_name": "ResNet-18",
    "description": "Classic lightweight CNN (baseline)",
    "params": "11.7M",
    "input_size": 224,
    "tags": ["classic", "lightweight", "baseline", "fast"]
}

"resnet50": {
    "display_name": "ResNet-50",
    "description": "Most popular baseline CNN",
    "params": "25.6M",
    "tags": ["classic", "baseline", "popular", "standard"]
}

"resnet101": {
    "display_name": "ResNet-101",
    "description": "Deep ResNet",
    "params": "44.5M",
    "tags": ["classic", "deep", "accurate"]
}
```

##### ConvNeXt (Modern CNN) - 2개
```python
"convnext_tiny": {
    "display_name": "ConvNeXt-Tiny",
    "description": "Modern CNN with Transformer design principles (2022)",
    "params": "28M",
    "input_size": 224,
    "recommended_batch_size": 64,
    "recommended_lr": 0.001,
    "tags": ["modern", "cnn", "transformer-style", "balanced"]
}

"convnext_base": {
    "display_name": "ConvNeXt-Base",
    "params": "89M",
    "tags": ["modern", "cnn", "transformer-style", "accurate"]
}
```

---

### 📊 Priority 2: 선택적 추가 (시간 여유 시)

#### YOLOv10 (2개) - NMS-free 특징
```python
"yolov10n": {
    "display_name": "YOLOv10 Nano",
    "description": "NMS-free detection (May 2024)",
    "params": "2.3M",
    "tags": ["2024", "nms-free", "efficient"]
}

"yolov10s": {
    "display_name": "YOLOv10 Small",
    "params": "7.2M",
    "tags": ["2024", "nms-free", "fast"]
}
```

#### MaxViT (Hybrid) - 2개
```python
"maxvit_tiny_tf_224": {
    "display_name": "MaxViT-Tiny",
    "description": "Hybrid CNN + ViT with multi-axis attention",
    "params": "31M",
    "tags": ["hybrid", "cnn+vit", "multi-axis-attention"]
}

"maxvit_small_tf_224": {
    "display_name": "MaxViT-Small",
    "params": "69M",
    "tags": ["hybrid", "cnn+vit"]
}
```

---

## 최종 모델 개수 요약

| 프레임워크 | Priority 1 | Priority 2 | 총합 |
|-----------|-----------|-----------|------|
| **Ultralytics** | 12개 | 2개 | 14개 |
| **timm** | 15개 | 2개 | 17개 |
| **총합** | **27개** | **4개** | **31개** |

---

## 모델 분류 (Tag 기반)

### By Recency
- **2024 최신** (9개): YOLOv11 (5), YOLO-World (2), MobileNetV4/V5 (2), SigLIP-2 (1)
- **2023-2024** (4개): YOLOv10 (2), EfficientNetV2 (2)
- **Baseline** (5개): ResNet (3), EfficientNet-B0/B4 (2)

### By Architecture
- **CNN**: ResNet (3), EfficientNet (4), ConvNeXt (2), MobileNet (3)
- **ViT**: ViT (3), SigLIP-2 (1)
- **Hybrid**: MaxViT (2)
- **YOLO**: YOLOv11 (10), YOLO-World (2), YOLOv10 (2)

### By Size
- **Nano/Tiny** (< 5M): YOLOv11n (2.6M), MobileNetV3 (5.5M), EfficientNet-B0 (5.3M)
- **Small/Medium** (5-30M): 대부분
- **Large** (30-100M): ViT-Large, ConvNeXt-Base
- **XLarge** (> 100M): SigLIP-2 (150M)

### By Special Features
- **Open-vocabulary**: YOLO-World (2)
- **Zero-shot**: YOLO-World
- **NMS-free**: YOLOv10 (2)
- **Mobile-optimized**: MobileNetV3/V4/V5 (3)
- **SOTA 2024**: YOLOv11, SigLIP-2, MobileNetV5

---

## 구현 우선순위

### Day 1-2: Core Infrastructure (반드시)
1. 모델 레지스트리 시스템 구축
2. API 엔드포인트 (`/models/list`)
3. Frontend 모델 선택 UI

### Day 3-4: Priority 1 Models (27개)
1. **Ultralytics** (12개)
   - YOLOv11 계열 (10개): Detection (5) + Segmentation (3) + Pose (2)
   - YOLO-World (2개): 특별 처리 필요 (open-vocab task 추가)

2. **timm** (15개)
   - Mobile: MobileNetV3/V4/V5 (3개)
   - ViT: Standard + SigLIP-2 (3개)
   - EfficientNet: V1 + V2 (4개)
   - ResNet: 18/50/101 (3개)
   - ConvNeXt (2개)

### Day 5: Testing & Validation
- 각 모델로 간단한 학습 실행
- UI에서 모델 선택 테스트
- 정상 동작 확인

### Day 6-7: Priority 2 (Optional, 4개)
- YOLOv10 (2개)
- MaxViT (2개)

---

## 특별 고려사항

### 1. YOLO-World (Open-Vocabulary)

**새로운 Task Type 추가 필요**:
```python
# mvp/training/adapters/base.py
class TaskType(Enum):
    # ... existing
    OPEN_VOCABULARY_DETECTION = "open_vocabulary_detection"  # 신규!
```

**사용 예시**:
```python
# 기존 YOLO: 고정된 클래스
result = model.predict("image.jpg")  # → 미리 학습된 80개 클래스만

# YOLO-World: 동적 클래스
result = model.set_classes(["cat", "dog", "car"]).predict("image.jpg")
# → 임의의 클래스 지정 가능! (zero-shot)
```

**구현 고려사항**:
- TrainingConfig에 `custom_prompts` 필드 추가
- UltralyticsAdapter에서 YOLO-World 특수 처리
- Frontend에서 텍스트 입력 UI 추가

### 2. YOLOv11 vs YOLOv8

**마이그레이션 간단**:
```python
# YOLOv8 (기존)
model = YOLO("yolov8n.pt")

# YOLOv11 (신규)
model = YOLO("yolo11n.pt")  # 동일한 API!
```

**UltralyticsAdapter 수정 불필요** - 모델명만 변경하면 자동 지원 ✅

### 3. timm 최신 모델

**대부분 자동 지원**:
```python
model = timm.create_model("mobilenetv4_conv_medium", pretrained=True)
# TimmAdapter 수정 불필요 ✅
```

**단, 주의사항**:
- MobileNetV4/V5: 최신 timm 버전 필요 (>= 0.9.0)
- SigLIP-2: 특수한 preprocessing 필요할 수 있음

---

## 검증 체크리스트

각 모델에 대해:
- [ ] 모델 메타데이터 정의 (레지스트리)
- [ ] API로 목록 조회 가능
- [ ] Frontend에서 선택 가능
- [ ] 학습 정상 실행 (최소 3 epochs)
- [ ] Validation metrics 계산
- [ ] Checkpoint 저장/로드
- [ ] Inference 수행

**특수 모델 추가 검증**:
- [ ] YOLO-World: Custom prompts 동작
- [ ] YOLOv11: YOLOv8 대비 성능 확인
- [ ] SigLIP-2: Preprocessing 정상 동작

---

## 예상 시간

| 작업 | 시간 | 비고 |
|------|------|------|
| Infrastructure | 2일 | 레지스트리, API, UI |
| Priority 1 (27개) | 2일 | 대부분 자동 지원 |
| Testing | 1일 | 주요 모델만 테스트 |
| Priority 2 (4개) | 1일 | Optional |
| Documentation | 0.5일 | 리포트 작성 |
| **총 소요** | **6-7일** | |

---

## 다음 단계 (Week 2 준비)

Week 1 완료 후:
1. **검증 리포트 작성**
   - 각 모델 성능 기록
   - 발견된 문제점 정리
   - Docker 분리 시 고려사항

2. **Docker 분리 준비**
   - Requirements 분석 (YOLOv11, YOLO-World 버전)
   - Platform SDK 설계 검토
   - 의존성 충돌 사전 확인

---

## 결론

**변경 요약**:
- 기존: 18개 모델 (YOLOv8, 구형 timm)
- 개선: 27개 모델 (YOLOv11, YOLO-World, 최신 timm)
- 추가: Open-vocabulary detection, 2024 최신 모델들

**핵심 개선**:
1. ✅ YOLOv11 (2024 Sep) - 22% 더 효율적
2. ✅ YOLO-World (CVPR 2024) - Zero-shot detection
3. ✅ MobileNetV4/V5 (2024) - 최신 모바일
4. ✅ SigLIP-2 (2024) - 88.1% top-1 accuracy
5. ✅ EfficientNetV2 - 기존에 누락됨

**예상 효과**:
- 플랫폼의 최신성 입증
- 차별화된 기능 제공 (open-vocab)
- 다양한 use case 커버

---

*Document Version: 2.0*
*Created: 2025-10-30*
*Author: Vision AI Platform Team*
