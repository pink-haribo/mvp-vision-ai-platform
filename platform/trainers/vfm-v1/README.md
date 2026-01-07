# VFM Train & Test 실행 가이드

## 빠른 시작

### Train
```bash
./tools/dist_train.sh configs/finetune_coco/vfm_v1_l_mvtec.py 4
```

### Test
```bash
./tools/dist_test.sh configs/finetune_coco/vfm_v1_l_mvtec.py work_dirs/vfm_v1_l_mvtec/latest.pth 4
```

---

## 1. 사전 준비

### 1.1 데이터셋 구조
```
data/mvtec_v2/
├── labels.txt                    # 클래스 목록
├── train/                        # 원본 (LabelMe)
├── train_annotations/            # COCO 변환
│   ├── annotations.json
│   └── *.jpg
├── val/
└── val_annotations/
    ├── annotations.json
    └── *.jpg
```

### 1.2 labels.txt 예시
```
defect
coil
discoloration
dust
```

### 1.3 텍스트 임베딩 파일
```
data/texts/mvtec.json
```

---

## 2. Train

### 기본 실행
```bash
# 단일 GPU
python tools/train.py configs/finetune_coco/vfm_v1_l_mvtec.py

# 멀티 GPU (3장)
./tools/dist_train.sh configs/finetune_coco/vfm_v1_l_mvtec.py 3

# Mixed Precision
./tools/dist_train.sh configs/finetune_coco/vfm_v1_l_mvtec.py 3 --amp
```

### 주요 파라미터 변경
```bash
# Batch size 변경
./tools/dist_train.sh configs/finetune_coco/vfm_v1_l_mvtec.py 3 \
    --cfg-options train_batch_size_per_gpu=8

# Learning rate 변경
./tools/dist_train.sh configs/finetune_coco/vfm_v1_l_mvtec.py 3 \
    --cfg-options base_lr=0.0001

# Epoch 변경
./tools/dist_train.sh configs/finetune_coco/vfm_v1_l_mvtec.py 3 \
    --cfg-options max_epochs=50

# 복합 변경
./tools/dist_train.sh configs/finetune_coco/vfm_v1_l_mvtec.py 3 \
    --cfg-options train_batch_size_per_gpu=8 base_lr=0.0001 max_epochs=50
```

### 출력 디렉토리
```
work_dirs/vfm_v1_l_mvtec/
├── vfm_v1_l_mvtec.py          # 실행 config 복사본
├── 20250122_143052.log        # 학습 로그
├── epoch_20.pth               # 주기별 체크포인트
├── epoch_40.pth
├── latest.pth                 # 최신
└── best_coco_bbox_mAP*.pth    # Best (옵션)
```

---

## 3. Test

### 기본 실행
```bash
# 특정 체크포인트
./tools/dist_test.sh \
    configs/finetune_coco/vfm_v1_l_mvtec.py \
    work_dirs/vfm_v1_l_mvtec/epoch_100.pth \
    3

# 최신 체크포인트
./tools/dist_test.sh \
    configs/finetune_coco/vfm_v1_l_mvtec.py \
    work_dirs/vfm_v1_l_mvtec/latest.pth \
    3
```

### 결과 저장
```bash
./tools/dist_test.sh \
    configs/finetune_coco/vfm_v1_l_mvtec.py \
    work_dirs/vfm_v1_l_mvtec/latest.pth \
    3 \
    --out results/mvtec_results.pkl
```

### 출력 예시
```
 Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.567
 Average Precision  (AP) @[ IoU=0.50      ] = 0.723
 Average Precision  (AP) @[ IoU=0.75      ] = 0.612
 
 Per-class AP:
 | class         | AP     |
 |---------------|--------|
 | defect        | 0.621  |
 | coil          | 0.534  |
 | discoloration | 0.498  |
 | dust          | 0.615  |
```

---

## 4. 주요 Config 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `train_batch_size_per_gpu` | 3 | GPU당 배치 크기 |
| `max_epochs` | 100 | 최대 epoch |
| `base_lr` | 2e-4 | 학습률 |
| `weight_decay` | 0.05 | Weight decay |
| `val_interval` | 10 | 검증 주기 (epoch) |
| `save_epoch_intervals` | 20 | 체크포인트 저장 주기 |

---

## 5. 문제 해결

### OOM (GPU 메모리 부족)
```bash
# batch_size 줄이기
--cfg-options train_batch_size_per_gpu=2
```

### 학습이 느림
```bash
# Mixed Precision 사용
./tools/dist_train.sh ... --amp

# num_workers 조정 (config 내)
persistent_workers = True
```

### 검증 자주 하고 싶음
```bash
--cfg-options val_interval=5
```

---

## 6. 전체 예제

```bash
# 1. 데이터 확인
ls data/mvtec_v2/labels.txt
ls data/mvtec_v2/train_annotations/annotations.json
ls data/texts/mvtec.json

# 2. Train (GPU 4장, 100 epoch)
./tools/dist_train.sh \
    configs/finetune_coco/vfm_v1_l_mvtec.py \
    4 \
    --amp

# 3. Test
./tools/dist_test.sh \
    configs/finetune_coco/vfm_v1_l_mvtec.py \
    work_dirs/vfm_v1_l_mvtec/latest.pth \
    4
```