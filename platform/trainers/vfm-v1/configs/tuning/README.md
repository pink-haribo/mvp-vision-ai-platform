# Test-Time Adaptation (TTA) for YOLO-World

## 개요

Test-Time Adaptation (TTA)은 사전학습된 모델을 **테스트 시점에 소량의 라벨된 데이터**로 빠르게 도메인에 적응시키는 기법입니다.

## 동기

| 문제 | 해결책 |
|------|--------|
| 새로운 도메인에서 성능 저하 | 소량의 prompt 이미지로 빠른 적응 |
| 전체 재학습은 비용이 큼 | Backbone 동결 + Neck/Head만 학습 |
| 과적합 위험 | 적은 epoch, learning rate decay |

## 기술적 원리

### 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOLOWorldDetector                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MultiModalYOLOBackbone                      │   │
│  │  ┌─────────────────────┐  ┌─────────────────────────┐   │   │
│  │  │   Image Backbone    │  │  CLIP Text Encoder      │   │   │
│  │  │   (CSPDarknet)      │  │  (HuggingCLIP)          │   │   │
│  │  │   ❄️ FROZEN         │  │  ❄️ FROZEN              │   │   │
│  │  └─────────────────────┘  └─────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │   Neck (YOLOWorldPAFPN)  🔥 TRAINABLE                   │   │
│  │   - 멀티스케일 피처 융합                                  │   │
│  │   - 텍스트-이미지 크로스어텐션                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │   Head (YOLOWorldHead)   🔥 TRAINABLE                   │   │
│  │   - 클래스 예측 (텍스트 임베딩 기반)                      │   │
│  │   - 바운딩박스 회귀                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 학습 전략

| 구성요소 | 전략 | 이유 |
|----------|------|------|
| Image Backbone | ❄️ 동결 (frozen_stages=4) | 일반적 시각 표현 유지 |
| Text Encoder | ❄️ 동결 (frozen_modules=['all']) | CLIP 언어 이해 유지 |
| Neck | 🔥 학습 (lr_mult=1.0) | 도메인 특화 피처 융합 |
| Head | 🔥 학습 (lr_mult=1.0) | 도메인 특화 탐지 |

### Learning Rate Schedule

```python
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0),   # 동결
            'neck': dict(lr_mult=1.0),       # 학습
            'bbox_head': dict(lr_mult=1.0)   # 학습
        }))

# Linear decay: lr_final = lr_init × 0.01
param_scheduler=dict(scheduler_type='linear', lr_factor=0.01)
```

## 파일 구조

```
configs/tuning/
├── README.md                    # 이 문서
├── tta_mvtec.py                 # TTA 학습 config
└── baseline_eval_mvtec.py       # Baseline 평가 config

tools/
├── tta_eval.py                  # TTA 실행 스크립트
└── forgetting_eval.py           # Catastrophic Forgetting 평가
```

## 사용법

### 1. TTA 실행

```bash
python tools/tta_eval.py configs/tuning/tta_mvtec.py \
    --checkpoint work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth \
    --tta-epochs 20 \
    --tta-lr 5e-5
```

**파라미터:**
- `--checkpoint`: 기존 baseline 체크포인트
- `--tta-epochs`: TTA 학습 epoch 수 (기본: 10)
- `--tta-lr`: Learning rate (기본: 1e-3)
- `--work-dir`: 결과 저장 경로 (기본: work_dirs/tta_eval)

### 2. Catastrophic Forgetting 평가

```bash
python tools/forgetting_eval.py \
    --config configs/tuning/tta_mvtec.py \
    --baseline-ckpt work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth \
    --tta-ckpt work_dirs/tta_eval/tta/epoch_20.pth
```

**파라미터:**
- `--baseline-ckpt`: TTA 전 baseline 체크포인트
- `--tta-ckpt`: TTA 후 체크포인트
- `--train-ann`: Train 어노테이션 경로
- `--val-ann`: Val 어노테이션 경로

## 예상 결과

### TTA 성능 향상 (MVTec 예시)

| 메트릭 | Baseline | After TTA | 향상 |
|--------|----------|-----------|------|
| mAP | 0.330 | 0.340 | +1.0% |
| mAP_50 | 0.500 | 0.572 | **+7.2%** |
| mAP_75 | 0.442 | 0.449 | +0.7% |

### Catastrophic Forgetting 분석

| 데이터 | Baseline mAP_50 | TTA mAP_50 | 변화 |
|--------|-----------------|------------|------|
| Train | 0.989 | 0.989 | 0.00% (유지) |
| Val | 0.500 | 0.570 | +7.00% (향상) |

**결론:** ✅ No significant forgetting | Strong generalization gain

## 클래스별 성능 변화

| 클래스 | Baseline mAP_50 | TTA mAP_50 | 향상 |
|--------|-----------------|------------|------|
| defect | ~0.18 | 0.279 | +10% |
| dust | ~0.32 | 0.351 | +3% |
| discoloration | ~0.86 | 0.901 | +4% |
| coil | ~0.69 | 0.755 | +6.5% |

## 데이터 요구사항

### Prompt 데이터 (TTA 학습용)
- 위치: `data/mvtec_v2/prompt_annotations/`
- 형식: COCO JSON 포맷
- 최소 요구: **1-2장의 라벨된 이미지**

```
prompt_annotations/
├── annotations.json    # COCO 형식 어노테이션
└── JPEGImages/         # 이미지 폴더
    ├── image1.png
    └── image2.png
```

### 텍스트 클래스 정의
- 위치: `data/texts/mvtec.json`
- 형식:
```json
[
    ["defect"],
    ["coil"],
    ["discoloration"],
    ["dust"]
]
```

## 주요 설정 파라미터

### tta_mvtec.py 핵심 설정

```python
# 학습 설정
train_cfg = dict(max_epochs=20, val_interval=5)
base_lr = 0.0001

# Backbone 동결
backbone = dict(
    frozen_stages=4,  # 이미지 백본 전체 동결
    text_model=dict(frozen_modules=['all'])  # 텍스트 인코더 동결
)

# 학습 데이터
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file='prompt_annotations/annotations.json',
            data_prefix=dict(img='prompt_annotations/')
        )
    )
)
```

## 출력 파일

### TTA 실행 후
```
work_dirs/tta_eval/
├── baseline/                    # Baseline 평가 결과
├── tta/                         # TTA 학습 결과
│   ├── epoch_5.pth             # 중간 체크포인트
│   ├── epoch_10.pth
│   ├── epoch_20.pth            # 최종 체크포인트
│   └── best_*.pth              # Best 체크포인트
└── results.json                 # 전체 결과 요약
```

### Forgetting 평가 후
```
work_dirs/forgetting_eval/
├── baseline_train/              # Baseline on Train 결과
├── baseline_val/                # Baseline on Val 결과
├── tta_train/                   # TTA on Train 결과
├── tta_val/                     # TTA on Val 결과
└── forgetting_results.json      # 분석 결과
```

## 다른 데이터셋에 적용

새로운 데이터셋에 TTA를 적용하려면:

1. **Config 복사 및 수정**
```bash
cp configs/tuning/tta_mvtec.py configs/tuning/tta_your_dataset.py
```

2. **수정할 항목**
```python
# 데이터 경로
data_root = 'data/your_dataset/'
class_text_path = 'data/texts/your_dataset.json'

# 클래스 수
num_classes = YOUR_NUM_CLASSES
metainfo = dict(classes=('class1', 'class2', ...))

# 어노테이션 경로
ann_file = 'prompt_annotations/annotations.json'
```

3. **텍스트 파일 생성**
```bash
# data/texts/your_dataset.json
[["class1"], ["class2"], ...]
```

## 트러블슈팅

### 1. CUDA Out of Memory
```python
# batch_size 줄이기
train_dataloader = dict(batch_size=1)
```

### 2. 성능 향상이 없는 경우
- Learning rate 조정: `--tta-lr 1e-4` 또는 `--tta-lr 1e-5`
- Epoch 수 증가: `--tta-epochs 50`
- Prompt 이미지 추가

### 3. Overfitting 발생
- Epoch 수 감소
- Learning rate 감소
- Early stopping 활용 (best checkpoint 사용)

## 참고 문헌

- [YOLO-World Paper](https://arxiv.org/abs/2401.17270)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Test-Time Adaptation Survey](https://arxiv.org/abs/2303.15361)

