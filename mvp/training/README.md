# Training Module

PyTorch 기반 학습 스크립트

## 구조

- **train_classification.py**: 메인 학습 스크립트 (CLI)
- **models/**: 모델 정의
- **data/**: 데이터 로더 및 전처리
- **training/**: 학습 로직 (Trainer, Metrics)
- **utils/**: 유틸리티 (Logger, Checkpoint)
- **configs/**: 기본 설정

## 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 학습 실행
python train_classification.py \
  --data_dir ../data/uploads/my_dataset \
  --output_dir ../data/outputs/experiment_1 \
  --model resnet50 \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001
```

## 출력 형식

stdout으로 진행률을 출력하여 Backend에서 파싱합니다:

```
[EPOCH] 1/50
[TRAIN] Loss: 2.456, Accuracy: 23.4%
[VAL] Loss: 2.123, Accuracy: 35.2%
[PROGRESS] 2%
```
