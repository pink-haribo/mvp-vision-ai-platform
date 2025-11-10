# Trainer 스크립트 상세 분석

## 1. train.py - 학습 스크립트

### 책임
- Dataset 다운로드 (Storage → 로컬)
- 학습 실행
- Epoch마다 validation
- Checkpoint 저장 및 업로드
- 메트릭 전송 (Backend Callback + MLflow)
- 학습 완료/실패 알림

### 실행 흐름
```python
1. 환경변수 파싱 및 검증
2. Storage 클라이언트 초기화 (MinIO/R2)
3. Dataset 다운로드
   - S3에서 dataset.zip 다운로드
   - 압축 해제 및 검증
4. MLflow 실험 시작
5. 모델 초기화 (pretrained or from scratch)
6. Checkpoint resume (옵션)
7. Training loop
   For each epoch:
     - Train step
     - Validation step
     - Metrics 계산
     - Callback: POST /validation-results
     - Callback: POST /training-metrics
     - MLflow: log_metrics()
     - Checkpoint 저장
     - Checkpoint 업로드 (S3)
8. 학습 완료
   - Final metrics 전송
   - Callback: PATCH /status (completed)
   - Best checkpoint 마킹
9. 정리 (임시 파일 삭제)
```

### 입력 (환경변수)
```bash
# Required
JOB_ID=1
TASK_TYPE=object_detection
MODEL_NAME=yolo11n
DATASET_ID=dataset-uuid
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=0.001
CALLBACK_URL=http://backend/internal/training/1
INTERNAL_AUTH_TOKEN=secret

# Optional
CHECKPOINT_PATH=s3://bucket/checkpoints/model.pt  # Resume용
IMAGE_SIZE=640
OPTIMIZER=adam
WEIGHT_DECAY=0.0005
SCHEDULER=cosine
```

### 출력
- stdout: 진행 상황 로그
- Storage: checkpoints/*.pt
- MLflow: metrics, params, artifacts
- Backend: validation results, training metrics, status updates

### 에러 처리
```python
try:
    train()
except KeyboardInterrupt:
    print("[WARN] Training interrupted by user")
    notify_status("cancelled")
except Exception as e:
    print(f"[ERROR] Training failed: {e}")
    traceback.print_exc()
    notify_status("failed", error=str(e))
finally:
    cleanup_temp_files()
```

---

## 2. predict.py - 추론 스크립트

### 책임
- Checkpoint 다운로드
- 이미지 로드
- 추론 실행
- 결과 반환 (JSON or 이미지)

### 실행 흐름
```python
1. 환경변수 파싱
2. Checkpoint 다운로드 (S3 or 로컬 경로)
3. 모델 로드
4. 이미지 다운로드 (S3 or URL or 로컬)
5. 추론 실행
6. 결과 포맷팅
   - Detection: [{"bbox": [...], "class": "cat", "confidence": 0.95}]
   - Classification: [{"class": "cat", "confidence": 0.95}]
7. 결과 저장 (옵션)
   - 시각화 이미지 저장
   - JSON 저장
8. 결과 반환 (stdout JSON or 파일 경로)
```

### 입력
```bash
# Required
CHECKPOINT_PATH=s3://bucket/checkpoints/best.pt
INPUT_PATH=s3://bucket/images/test.jpg  # or local path

# Optional
OUTPUT_PATH=s3://bucket/predictions/result.json
VISUALIZE=true  # 시각화 이미지 생성
CONFIDENCE_THRESHOLD=0.5
```

### 출력
```json
{
  "predictions": [
    {
      "bbox": [100, 200, 300, 400],
      "class": "cat",
      "confidence": 0.95
    }
  ],
  "visualization_url": "s3://bucket/vis/result.jpg"
}
```

### 사용 사례
- **API 추론**: Frontend → Backend → predict.py (subprocess)
- **배치 추론**: Backend Job → predict.py (K8s Job)
- **테스트**: 사용자가 수동 실행

---

## 3. evaluate.py - 평가 스크립트

### 책임
- Checkpoint 다운로드
- 검증셋 다운로드
- 전체 검증셋 평가
- 상세 메트릭 계산
- 평가 리포트 생성

### 실행 흐름
```python
1. 환경변수 파싱
2. Checkpoint 다운로드
3. 검증셋 다운로드
4. 모델 로드
5. 전체 검증셋 평가
   For each image:
     - 추론
     - GT와 비교
     - 메트릭 누적
6. 최종 메트릭 계산
   - mAP, Precision, Recall
   - Per-class metrics
   - Confusion matrix
7. 리포트 생성
   - Markdown report
   - 시각화 (PR curve, confusion matrix)
8. 결과 저장 (S3) 및 반환
```

### 입력
```bash
CHECKPOINT_PATH=s3://bucket/checkpoints/best.pt
DATASET_ID=dataset-uuid
SPLIT=val  # val or test
OUTPUT_DIR=s3://bucket/evaluations/job-1/
```

### 출력
```
s3://bucket/evaluations/job-1/
├── metrics.json          # {"mAP50": 0.85, ...}
├── report.md             # Markdown 리포트
├── confusion_matrix.png
└── pr_curve.png
```

### 사용 사례
- 학습 후 최종 평가
- 여러 checkpoint 비교
- 리더보드 생성

---

## 4. export.py - 모델 변환 스크립트

### 책임
- Checkpoint 다운로드
- 모델 변환 (ONNX, TensorRT, CoreML 등)
- 변환된 모델 검증
- Storage 업로드

### 실행 흐름
```python
1. 환경변수 파싱
2. Checkpoint 다운로드
3. 모델 로드
4. 포맷 변환
   - ONNX: torch.onnx.export()
   - TensorRT: onnx2trt
   - CoreML: coremltools.convert()
5. 변환 검증
   - 샘플 이미지로 추론 테스트
   - 정확도 비교
6. 변환된 모델 업로드 (S3)
```

### 입력
```bash
CHECKPOINT_PATH=s3://bucket/checkpoints/best.pt
EXPORT_FORMAT=onnx  # onnx, tensorrt, coreml, tflite
OUTPUT_PATH=s3://bucket/exports/model.onnx

# Format-specific
ONNX_OPSET=13
TENSORRT_PRECISION=fp16
```

### 출력
```
s3://bucket/exports/
├── model.onnx
├── metadata.json  # {"input_shape": [1,3,640,640], "classes": [...]}
└── benchmark.json # {"inference_time_ms": 15.2}
```

### 사용 사례
- 모바일 배포 (CoreML, TFLite)
- 서버 최적화 (TensorRT)
- 웹 배포 (ONNX.js)

---

## 스크립트 우선순위

### Phase 1 (MVP 필수)
- ✅ **train.py** - 학습 기능 필수
- ✅ **predict.py** - 추론 기능 필수 (API 서빙용)

### Phase 2 (추가 기능)
- ⏳ **evaluate.py** - 상세 평가 (있으면 좋음, train.py의 validation으로 대체 가능)
- ⏳ **export.py** - 모델 변환 (나중에 필요할 때)

### Not Needed Now
- ❌ **serve.py** - 추론 서버 (predict.py를 subprocess로 호출하면 됨)
- ❌ **validate_dataset.py** - 데이터셋 검증 (train.py 내부에 포함)
