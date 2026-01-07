# YOLO-World Adapter-based Fine-tuning

이 디렉토리는 YOLO-World 모델에 Adapter를 적용하여 parameter-efficient fine-tuning을 수행하는 설정 파일들을 포함합니다.

## 📋 개요

Adapter 기반 학습은 사전 학습된 모델의 대부분의 파라미터를 고정(freeze)하고, 작은 adapter 모듈만 학습하여 효율적으로 fine-tuning하는 방법입니다.

### 장점
- **메모리 효율성**: 전체 파라미터의 1-7%만 학습
- **빠른 학습**: 적은 파라미터로 빠른 수렴
- **과적합 방지**: 제한된 파라미터로 일반화 성능 향상
- **모듈성**: 다양한 태스크에 대해 adapter만 교체 가능

## 🏗️ 구조

### Phase 1 구현

#### Option 1: BottleneckAdapter
```
Input → Down-projection → GELU → Up-projection → Output
  ↓                                                  ↑
  └──────────────── Residual ──────────────────────┘
```

**특징:**
- 간단한 bottleneck 구조
- 파라미터 수: 최소 (~1-2%)
- 빠른 학습 속도

#### Option 2: HierarchicalAdapter
```
Input → Down → GELU → Up → DoubleConv → Attention → MLP → Output
  ↓                                                           ↑
  └────────────────────── Residual ──────────────────────────┘
```

**특징:**
- Attention 메커니즘 포함
- 파라미터 수: 중간 (~5-7%)
- 더 높은 표현력

### 전략

#### Strategy A: Neck Only
- **적용 위치**: Neck (YOLOWorldPAFPN)만
- **Adapter 위치**: top_down, bottom_up layers
- **파라미터 수**: 최소
- **권장 사용**: 빠른 실험, 제한된 데이터

#### Strategy B: Multi-stage
- **적용 위치**: Backbone + Neck + Head
- **Adapter 위치**: 
  - Backbone: stage 2, 3, 4
  - Neck: top_down, bottom_up layers
  - Head: cls, reg branches
- **파라미터 수**: 중간
- **권장 사용**: 더 높은 성능 필요시

## 📁 Config 파일

| Config | Adapter Type | Strategy | 설명 |
|--------|-------------|----------|------|
| `phase1_option1_strategy_a.py` | BottleneckAdapter | Neck only | 가장 간단하고 빠른 설정 |
| `phase1_option2_strategy_a.py` | HierarchicalAdapter | Neck only | Attention 포함, Neck만 |
| `phase1_option1_strategy_b.py` | BottleneckAdapter | Multi-stage | 전체 네트워크에 간단한 adapter |
| `phase1_option2_strategy_b.py` | HierarchicalAdapter | Multi-stage | 전체 네트워크에 복잡한 adapter |

## 🚀 사용 방법

### 1. 기본 학습 (이미 완료)

먼저 기본 YOLO-World 모델을 학습합니다:

```bash
./tools/dist_train.sh ./configs/finetune_coco/vfm_v1_l_mvtec.py 1 --amp
```

이 학습은 `work_dirs/vfm_v1_l_mvtec/epoch_100.pth`에 체크포인트를 저장합니다.

### 2. Adapter 학습

기본 학습이 완료된 후, adapter를 추가하여 fine-tuning합니다:

#### Option 1 + Strategy A (권장 시작점)
```bash
./tools/dist_train.sh ./configs/adapter/phase1_option1_strategy_a.py 1 --amp
```

#### Option 2 + Strategy A
```bash
./tools/dist_train.sh ./configs/adapter/phase1_option2_strategy_a.py 1 --amp
```

#### Option 1 + Strategy B
```bash
./tools/dist_train.sh ./configs/adapter/phase1_option1_strategy_b.py 1 --amp
```

#### Option 2 + Strategy B
```bash
./tools/dist_train.sh ./configs/adapter/phase1_option2_strategy_b.py 1 --amp
```

### 3. 테스트/검증

학습된 adapter 모델을 테스트합니다:

```bash
# Validation
python tools/test.py \
    configs/adapter/phase1_option1_strategy_a.py \
    work_dirs/phase1_option1_strategy_a/epoch_50.pth \
    --work-dir work_dirs/phase1_option1_strategy_a/test

# Inference on images
python demo/image_demo.py \
    path/to/image.jpg \
    configs/adapter/phase1_option1_strategy_a.py \
    work_dirs/phase1_option1_strategy_a/epoch_50.pth \
    --texts data/texts/mvtec.json
```

## ⚙️ Config 커스터마이징

### Adapter 설정 변경

```python
# Reduction ratio 조정 (파라미터 수 조절)
adapter_reduction_ratio = 4  # 4, 8, 16 등

# Adapter 위치 변경
adapter_positions = ['top_down', 'bottom_up', 'out']  # 'reduce', 'out' 추가 가능

# Backbone adapter stages 변경
backbone_adapter_stages = [2, 3, 4]  # 1, 2, 3, 4 중 선택
```

### 학습 설정 변경

```python
# Epoch 수 조정
max_epochs = 50  # 원하는 epoch 수

# Learning rate 조정
base_lr = 1e-4  # 1e-3, 1e-4, 1e-5 등

# Batch size 조정
train_batch_size_per_gpu = 4  # GPU 메모리에 따라 조정
```

### Resume 설정

```python
# 체크포인트에서 로드만 (optimizer 상태 제외)
load_from = 'work_dirs/vfm_v1_l_mvtec/epoch_100.pth'
resume = False

# 완전한 resume (optimizer, scheduler 포함)
load_from = 'work_dirs/phase1_option1_strategy_a/epoch_20.pth'
resume = True
```

## 📊 예상 결과

### 파라미터 수 비교

| 설정 | 전체 파라미터 | 학습 파라미터 | 비율 |
|------|--------------|--------------|------|
| Full fine-tuning | ~43M | ~43M | 100% |
| Option 1 + Strategy A | ~43M | ~0.5M | ~1.2% |
| Option 2 + Strategy A | ~43M | ~1.5M | ~3.5% |
| Option 1 + Strategy B | ~43M | ~1.5M | ~3.5% |
| Option 2 + Strategy B | ~43M | ~3.0M | ~7.0% |

### 학습 속도

- **Option 1 + Strategy A**: 가장 빠름 (~1.2x faster than full)
- **Option 2 + Strategy B**: 약간 느림 (~1.05x faster than full)

## 🔍 모니터링

### TensorBoard

```bash
tensorboard --logdir work_dirs/phase1_option1_strategy_a
```

### 학습 중 확인 사항

1. **Loss 감소**: Adapter 학습도 loss가 감소해야 함
2. **mAP 향상**: Validation mAP가 기본 학습보다 향상되는지 확인
3. **Overfitting**: Validation loss가 증가하면 early stopping 고려

## 🛠️ 문제 해결

### 1. Out of Memory

```python
# Config에서 batch size 줄이기
train_batch_size_per_gpu = 2  # 4 → 2

# 또는 gradient accumulation 사용
optim_wrapper = dict(
    accumulative_counts=2,  # 2 step마다 업데이트
    ...
)
```

### 2. Adapter가 학습되지 않음

학습 로그에서 확인:
```bash
grep "lr_mult" work_dirs/phase1_option1_strategy_a/*.log
```

Adapter 파라미터가 `lr_mult=1.0`인지 확인

### 3. 성능이 향상되지 않음

- Learning rate 조정: `1e-3`, `1e-4`, `1e-5` 시도
- Adapter reduction ratio 조정: `2`, `4`, `8` 시도
- 더 많은 epoch 학습
- Strategy B (Multi-stage) 시도

## 📚 참고 자료

- [Parameter-Efficient Transfer Learning](https://arxiv.org/abs/1902.00751)
- [Adapter-based Fine-tuning](https://arxiv.org/abs/1902.00751)
- [YOLO-World Paper](https://arxiv.org/abs/2401.17270)

## 🎯 다음 단계

1. **Phase 1 완료 후**: 4가지 설정 중 가장 좋은 성능을 보이는 것 선택
2. **Hyperparameter Tuning**: Learning rate, reduction ratio 등 최적화
3. **Ensemble**: 여러 adapter 모델 앙상블
4. **Deployment**: 최종 모델 배포

---

## 🚀 Phase 2: 주요 실험 Config (Step 1~4)

### **Step 1: Dense LoRA**
**파일:** `phase2_lora_v1.py`
- LoRA adapter만 사용 (unfreezing 없음)
- Trainable: 0.2% (가장 효율적)

### **Step 2: Hybrid Moderate** ⭐ 가장 안정적
**파일:** `phase2_hybrid_v1.py`
- Dense LoRA + Selective Unfreezing
- Trainable: 2.3% (2.49M params)

### **Step 3: Hybrid Aggressive**
**파일:** `phase2_hybrid_aggressive_v1.py`
- Dense LoRA + Aggressive Unfreezing
- Trainable: 18.8% (20.78M params)
- ⚠️ Gradient explosion 발생 (Epoch 109)

### **Step 4-1: Rep-MoNA Conservative** ⭐ 공간 문맥 인식
**파일:** `phase2_step4_1_rep_mona.py`
- **Rep-MoNA LoRA** (Neck에만 적용)
- Multi-scale spatial context [3×3, 5×5, 7×7]
- Trainable: 2.3% (2.51M params)
- ✅ Scheduler 수정됨 (negative LR 방지)

### **Step 4-2: Rep-MoNA Moderate** ⭐ 공간 문맥 + 성능
**파일:** `phase2_step4_2_rep_mona.py`
- **Rep-MoNA LoRA** (Neck에만 적용)
- Step 3 개선 버전
- Trainable: ~5% (예상)
- ✅ Scheduler 수정됨 (negative LR 방지)

### **Step 5: MoE-Enhanced RepMoNA** 🆕 최신
**파일:** `phase2_step5_moe_mona.py`, `phase2_step5_moe_mona_v2.py`
- **MoE + RepMoNA** 결합
- 동적 Expert 선택 (Top-k Soft Gating)
- Multi-scale Experts [3×3, 5×5, 7×7]
- SE Block (Channel Attention)
- V2: Spatial Attention + Load Balancing Loss

**참고 논문:**
- Conv-LoRA (ICLR 2024): Convolution Meets LoRA
- MoE-Adapters (CVPR 2024): Mixture-of-Experts for Continual Learning
- Self-Expansion MoE (CVPR 2025): Pre-trained Models with MoE Adapters

### 비교표

| Step | Config | Adapter | Trainable % | 특징 |
|------|--------|---------|-------------|------|
| **Step 1** | `phase2_lora_v1.py` | LoRA | 0.2% | 가장 효율적 |
| **Step 2** | `phase2_hybrid_v1.py` | Dense LoRA | 2.3% | **가장 안정적** ✅ |
| **Step 3** | `phase2_hybrid_aggressive_v1.py` | Dense LoRA | 18.8% | Gradient explosion ❌ |
| **Step 4-1** | `phase2_step4_1_rep_mona.py` | **Rep-MoNA** | 2.3% | 공간 문맥 + 안정성 ⭐ |
| **Step 4-2** | `phase2_step4_2_rep_mona.py` | **Rep-MoNA** | ~5% | 공간 문맥 + 성능 ⭐ |
| **Step 5** | `phase2_step5_moe_mona.py` | **MoE-RepMoNA** | ~6% | 동적 Expert 선택 🆕 |
| **Step 5-V2** | `phase2_step5_moe_mona_v2.py` | **MoE-RepMoNA-V2** | ~7% | + Spatial Attn 🆕 |

### 학습 명령어

```bash
# Step 3
python tools/train.py configs/adapter/phase2_hybrid_aggressive_v1.py \
    --work-dir work_dirs/step3_aggressive

# Step 4-1 (권장)
python tools/train.py configs/adapter/phase2_step4_1_rep_mona.py \
    --work-dir work_dirs/step4_1_rep_mona_fixed

# Step 4-2 (권장)
python tools/train.py configs/adapter/phase2_step4_2_rep_mona.py \
    --work-dir work_dirs/step4_2_rep_mona_fixed

# Step 5 (MoE-Enhanced) 🆕
python tools/train.py configs/adapter/phase2_step5_moe_mona.py \
    --work-dir work_dirs/step5_moe_mona

# Step 5 V2 (+ Spatial Attention) 🆕
python tools/train.py configs/adapter/phase2_step5_moe_mona_v2.py \
    --work-dir work_dirs/step5_moe_mona_v2
```

### Step 5 MoE-RepMoNA 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    MoE-RepMoNA Adapter                      │
├─────────────────────────────────────────────────────────────┤
│   Input (B, C, H, W)                                        │
│         │                                                   │
│         ├──────────────────────────────────────┐            │
│         │                                      │            │
│         ▼                                      │            │
│   ┌─────────────┐                              │            │
│   │   Router    │ ← Soft Gating (학습 가능)     │            │
│   │  (GAP+FC)   │                              │            │
│   └─────────────┘                              │            │
│         │                                      │            │
│    [g₁, g₂, g₃] (gate weights)                 │            │
│         │                                      │            │
│   ┌─────┴─────┬─────────┐                      │            │
│   ▼           ▼         ▼                      │            │
│ Expert₁    Expert₂   Expert₃                   │            │
│ (3×3 DW)   (5×5 DW)  (7×7 DW)                  │            │
│   │           │         │                      │            │
│   └─────┬─────┴─────────┘                      │            │
│         │ Top-k Selection                      │            │
│         ▼                                      │            │
│   Weighted Sum: Σ(gᵢ × Expertᵢ)                │            │
│         │                                      │            │
│         ▼                                      │            │
│   ┌─────────────┐                              │            │
│   │  SE Block   │ ← Channel Attention          │            │
│   └─────────────┘                              │            │
│         │                                      │            │
│         ▼                                      │            │
│   ┌─────────────┐                              │            │
│   │ Up Project  │                              │            │
│   └─────────────┘                              │            │
│         │                                      │            │
│         ▼                                      │            │
│      Output ←──────────────────────────────────┘ (Residual) │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 참고 문서
- `docs/adapter_finetuning_report.md` - 전체 실험 결과
- `docs/step4_rep_mona_implementation.md` - Rep-MoNA 구현 상세

