# YOLO-World Adapter Quick Start Guide

## 🚀 빠른 시작 (5분 안에!)

### 전제 조건
- ✅ 기본 학습 완료: `work_dirs/vfm_v1_l_mvtec/epoch_100.pth` 존재
- ✅ detgpt 환경 활성화
- ✅ YOLO-World 디렉토리에 위치

### Step 1: 환경 확인 (30초)

```bash
# Conda 환경 활성화
conda activate detgpt

# 현재 디렉토리 확인
pwd  # Should be: ~/repo/YOLO-World

# 체크포인트 확인
ls work_dirs/vfm_v1_l_mvtec/epoch_100.pth
```

### Step 2: Adapter 학습 시작 (1분)

**가장 간단한 설정으로 시작:**

```bash
./tools/dist_train.sh ./configs/adapter/phase1_option1_strategy_a.py 1 --amp
```

이 명령어는:
- ✅ BottleneckAdapter 사용 (가장 간단)
- ✅ Neck에만 Adapter 추가 (Strategy A)
- ✅ epoch_100.pth에서 resume
- ✅ 50 epochs 학습
- ✅ Adapter만 학습 (나머지는 freeze)

### Step 3: 학습 모니터링 (실시간)

**새 터미널에서:**

```bash
# TensorBoard 실행
conda activate detgpt
cd ~/repo/YOLO-World
tensorboard --logdir work_dirs/phase1_option1_strategy_a
```

브라우저에서 `http://localhost:6006` 접속

**또는 로그 확인:**

```bash
# 실시간 로그 확인
tail -f work_dirs/phase1_option1_strategy_a/*.log

# Loss 확인
grep "loss" work_dirs/phase1_option1_strategy_a/*.log | tail -20

# mAP 확인
grep "bbox_mAP" work_dirs/phase1_option1_strategy_a/*.log
```

### Step 4: 학습 완료 후 테스트 (2분)

```bash
# Validation
python tools/test.py \
    configs/adapter/phase1_option1_strategy_a.py \
    work_dirs/phase1_option1_strategy_a/epoch_50.pth

# 특정 이미지로 테스트
python demo/image_demo.py \
    data/mvtec_v2/val_annotations/image_001.jpg \
    configs/adapter/phase1_option1_strategy_a.py \
    work_dirs/phase1_option1_strategy_a/epoch_50.pth \
    --texts data/texts/mvtec.json
```

## 🎯 다른 설정 시도하기

### Option 2 (HierarchicalAdapter) 시도

```bash
./tools/dist_train.sh ./configs/adapter/phase1_option2_strategy_a.py 1 --amp
```

**차이점:**
- 더 복잡한 Adapter (Attention 포함)
- 더 많은 파라미터 (~3.5%)
- 더 높은 성능 기대

### Strategy B (Multi-stage) 시도

```bash
./tools/dist_train.sh ./configs/adapter/phase1_option1_strategy_b.py 1 --amp
```

**차이점:**
- Backbone + Neck + Head에 모두 Adapter
- 더 많은 파라미터 (~3.5%)
- 더 높은 표현력

### 모든 조합 시도

```bash
# Option 1 + Strategy A (가장 간단)
./tools/dist_train.sh ./configs/adapter/phase1_option1_strategy_a.py 1 --amp

# Option 2 + Strategy A (Attention 추가)
./tools/dist_train.sh ./configs/adapter/phase1_option2_strategy_a.py 1 --amp

# Option 1 + Strategy B (Multi-stage)
./tools/dist_train.sh ./configs/adapter/phase1_option1_strategy_b.py 1 --amp

# Option 2 + Strategy B (최대 성능)
./tools/dist_train.sh ./configs/adapter/phase1_option2_strategy_b.py 1 --amp
```

## 🔧 커스터마이징

### Learning Rate 변경

Config 파일 수정:
```python
# configs/adapter/phase1_option1_strategy_a.py
base_lr = 1e-4  # 기본값
# base_lr = 1e-3  # 더 빠른 수렴
# base_lr = 1e-5  # 더 안정적
```

### Epoch 수 변경

```python
max_epochs = 50  # 기본값
# max_epochs = 30  # 빠른 실험
# max_epochs = 100  # 충분한 학습
```

### Adapter Reduction Ratio 변경

```python
adapter_reduction_ratio = 4  # 기본값
# adapter_reduction_ratio = 2  # 더 많은 파라미터
# adapter_reduction_ratio = 8  # 더 적은 파라미터
```

## 📊 성능 비교

### 예상 결과

| 설정 | 학습 파라미터 | 학습 시간 | 예상 mAP |
|------|--------------|----------|----------|
| Full fine-tuning | 100% | 1.0x | Baseline |
| Option 1 + Strategy A | ~1.2% | 0.8x | Baseline + 1-3% |
| Option 2 + Strategy A | ~3.5% | 0.9x | Baseline + 2-5% |
| Option 1 + Strategy B | ~3.5% | 0.9x | Baseline + 2-5% |
| Option 2 + Strategy B | ~7.0% | 0.95x | Baseline + 3-7% |

### 비교 방법

```bash
# 1. 기본 모델 성능
python tools/test.py \
    configs/finetune_coco/vfm_v1_l_mvtec.py \
    work_dirs/vfm_v1_l_mvtec/epoch_100.pth

# 2. Adapter 모델 성능
python tools/test.py \
    configs/adapter/phase1_option1_strategy_a.py \
    work_dirs/phase1_option1_strategy_a/epoch_50.pth

# 3. 결과 비교
# - bbox_mAP
# - bbox_mAP_50
# - bbox_mAP_75
# - Per-class AP
```

## 🐛 문제 해결

### Out of Memory

```python
# Config에서 batch size 줄이기
train_batch_size_per_gpu = 2  # 4 → 2
```

### 학습이 시작되지 않음

```bash
# 체크포인트 경로 확인
ls work_dirs/vfm_v1_l_mvtec/epoch_100.pth

# Config 파일 확인
python -c "from mmengine.config import Config; cfg = Config.fromfile('configs/adapter/phase1_option1_strategy_a.py'); print(cfg.load_from)"
```

### Adapter가 학습되지 않음

```bash
# 학습 로그에서 확인
grep "lr_mult" work_dirs/phase1_option1_strategy_a/*.log | head -20

# Adapter 파라미터가 lr_mult=1.0인지 확인
```

### 성능이 향상되지 않음

1. **Learning rate 조정**
   ```python
   base_lr = 1e-3  # 더 높게
   # 또는
   base_lr = 1e-5  # 더 낮게
   ```

2. **더 많은 epoch**
   ```python
   max_epochs = 100
   ```

3. **다른 설정 시도**
   - Option 2 (HierarchicalAdapter)
   - Strategy B (Multi-stage)

## 📝 체크리스트

학습 시작 전:
- [ ] detgpt 환경 활성화
- [ ] epoch_100.pth 존재 확인
- [ ] GPU 메모리 확인 (nvidia-smi)
- [ ] 디스크 공간 확인 (df -h)

학습 중:
- [ ] Loss가 감소하는지 확인
- [ ] GPU 사용률 확인 (nvidia-smi)
- [ ] TensorBoard 모니터링

학습 완료 후:
- [ ] Validation 수행
- [ ] mAP 비교
- [ ] 체크포인트 저장 확인
- [ ] 최종 모델 선택

## 🎓 추가 리소스

- **상세 가이드**: `configs/adapter/README.md`
- **구현 요약**: `docs/adapter_implementation_summary.md`
- **원본 Config**: `configs/finetune_coco/vfm_v1_l_mvtec.py`

## 💡 팁

1. **처음 시도**: phase1_option1_strategy_a.py로 시작
2. **성능 중요**: phase1_option2_strategy_b.py 시도
3. **빠른 실험**: max_epochs=30으로 설정
4. **안정적 학습**: base_lr=1e-5로 시작

## 🎉 성공 사례

```bash
# 학습 시작
./tools/dist_train.sh ./configs/adapter/phase1_option1_strategy_a.py 1 --amp

# 예상 출력:
# Epoch [1/50] ... loss: 15.3
# Epoch [10/50] ... loss: 8.2, bbox_mAP: 0.15
# Epoch [20/50] ... loss: 5.1, bbox_mAP: 0.22
# Epoch [30/50] ... loss: 3.8, bbox_mAP: 0.28
# Epoch [40/50] ... loss: 2.9, bbox_mAP: 0.32
# Epoch [50/50] ... loss: 2.3, bbox_mAP: 0.35

# 성공! 🎉
```

Happy training! 🚀

