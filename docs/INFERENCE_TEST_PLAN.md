# Inference Job Pattern - API Test Plan

## Overview

프론트엔드 테스트는 느리고 디버깅이 어렵습니다. 이 문서는 InferenceJob Pattern을 curl/bash로 직접 테스트하는 체계적인 계획입니다.

**목표**: 프론트엔드와 동일한 흐름을 API 레벨에서 직접 재현하여 빠르게 검증

## Test Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Frontend Flow (Slow)                                            │
│ UI Click → Wait → Check UI → Repeat                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Replace with
┌─────────────────────────────────────────────────────────────────┐
│ API Test Flow (Fast)                                            │
│ curl → Check response → curl → Check DB → Check S3             │
└─────────────────────────────────────────────────────────────────┘
```

## Test Prerequisites

### 1. Real Data Setup

```bash
# Training Job 확인
curl http://localhost:8000/api/v1/training/jobs/23 | jq .

# Expected output:
# - status: "completed"
# - checkpoint_path: "s3://training-checkpoints/checkpoints/23/best.pt"
# - task_type: "detection"

# 실제 테스트 이미지 준비 (COCO val2017에서 가져온 실제 이미지)
TEST_IMAGE_DIR="platform/backend/test_inference_images"
mkdir -p $TEST_IMAGE_DIR

# 실제 이미지 복사 (프론트엔드에서 사용할 이미지와 동일)
cp platform/backend/data/coco128/images/train2017/000000000030.jpg $TEST_IMAGE_DIR/
cp platform/backend/data/coco128/images/train2017/000000000034.jpg $TEST_IMAGE_DIR/
cp platform/backend/data/coco128/images/train2017/000000000036.jpg $TEST_IMAGE_DIR/

# 이미지 존재 확인
ls -lh $TEST_IMAGE_DIR/
```

**중요**: 더미 데이터 없음. 실제 프론트엔드가 사용할 이미지와 동일한 파일 사용.

### 2. Environment Variables

```bash
export API_BASE_URL="http://localhost:8000/api/v1"
export TRAINING_JOB_ID=23
export CHECKPOINT_PATH="s3://training-checkpoints/checkpoints/23/best.pt"
```

## Test Scenarios

### Scenario 1: Single Image Inference (가장 기본)

**사용자 시나리오**:
- 사용자가 학습 완료된 Job 23 페이지에서 "Test Inference" 탭 클릭
- 이미지 1개 업로드 (000000000030.jpg)
- "Run Inference" 버튼 클릭
- 결과 대기 및 확인

**API Test Steps**:

#### Step 1: Upload Image to S3

```bash
# 프론트엔드가 하는 것과 동일: multipart/form-data로 이미지 업로드
curl -X POST "$API_BASE_URL/test_inference/inference/upload-images?training_job_id=$TRAINING_JOB_ID" \
  -F "files=@$TEST_IMAGE_DIR/000000000030.jpg" \
  -H "Content-Type: multipart/form-data" \
  | jq .

# Expected Response:
# {
#   "status": "success",
#   "inference_session_id": "uuid-here",
#   "s3_prefix": "s3://training-checkpoints/inference/{uuid}/",
#   "uploaded_files": [
#     {
#       "original_filename": "000000000030.jpg",
#       "unique_filename": "uuid.jpg",
#       "s3_uri": "s3://training-checkpoints/inference/{uuid}/uuid.jpg"
#     }
#   ],
#   "total_files": 1
# }

# Save s3_prefix for next step
export S3_PREFIX=$(curl -X POST "$API_BASE_URL/test_inference/inference/upload-images?training_job_id=$TRAINING_JOB_ID" \
  -F "files=@$TEST_IMAGE_DIR/000000000030.jpg" \
  2>/dev/null | jq -r '.s3_prefix')

echo "S3_PREFIX=$S3_PREFIX"
```

**Validation**:
- HTTP 200 OK
- Response contains `inference_session_id`
- `total_files` == 1
- S3에 파일 실제 업로드 확인:
  ```bash
  # MinIO 직접 확인
  aws --endpoint-url http://localhost:9002 s3 ls training-checkpoints/inference/ --recursive | grep $(echo $S3_PREFIX | cut -d'/' -f5)
  ```

#### Step 2: Create InferenceJob

```bash
# 프론트엔드가 전송하는 payload와 동일
curl -X POST "$API_BASE_URL/test_inference/inference/jobs" \
  -H "Content-Type: application/json" \
  -d "{
    \"training_job_id\": $TRAINING_JOB_ID,
    \"checkpoint_path\": \"$CHECKPOINT_PATH\",
    \"inference_type\": \"batch\",
    \"input_data\": {
      \"image_paths_s3\": \"$S3_PREFIX\",
      \"confidence_threshold\": 0.25,
      \"iou_threshold\": 0.45,
      \"max_detections\": 100,
      \"save_visualizations\": true
    }
  }" \
  | jq .

# Save inference_job_id
export INFERENCE_JOB_ID=$(curl -X POST "$API_BASE_URL/test_inference/inference/jobs" \
  -H "Content-Type: application/json" \
  -d "{
    \"training_job_id\": $TRAINING_JOB_ID,
    \"checkpoint_path\": \"$CHECKPOINT_PATH\",
    \"inference_type\": \"batch\",
    \"input_data\": {
      \"image_paths_s3\": \"$S3_PREFIX\",
      \"confidence_threshold\": 0.25,
      \"iou_threshold\": 0.45,
      \"max_detections\": 100,
      \"save_visualizations\": true
    }
  }" 2>/dev/null | jq -r '.id')

echo "INFERENCE_JOB_ID=$INFERENCE_JOB_ID"
```

**Expected Response**:
```json
{
  "id": 1,
  "training_job_id": 23,
  "checkpoint_path": "s3://training-checkpoints/checkpoints/23/best.pt",
  "inference_type": "batch",
  "status": "pending",
  "task_type": "detection",
  "created_at": "2025-11-18T08:30:00Z",
  "input_data": {
    "image_paths_s3": "s3://training-checkpoints/inference/{uuid}/",
    "confidence_threshold": 0.25,
    ...
  }
}
```

**Validation**:
- HTTP 201 Created
- Response contains `id` (InferenceJob ID)
- `status` == "pending"
- `task_type` == "detection"

**DB Check**:
```bash
# PostgreSQL에서 직접 확인
docker exec -it platform-postgres-1 psql -U admin -d platform -c \
  "SELECT id, status, task_type, created_at FROM inference_jobs WHERE id = $INFERENCE_JOB_ID;"
```

#### Step 3: Wait for Background Task Completion

```bash
# 프론트엔드의 polling과 동일
for i in {1..60}; do
  STATUS=$(curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID" | jq -r '.status')
  echo "[Poll $i/60] Status: $STATUS"

  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi

  sleep 2
done

echo "Final Status: $STATUS"
```

**Validation**:
- Status transitions: `pending` → `running` → `completed`
- 최대 60초 (30 polls) 내 완료
- Status가 `failed`면 즉시 중단하고 로그 확인

**Backend Log Check**:
```bash
# Background task 실행 로그 확인
tail -100 platform/backend/backend.log | grep -E "INFERENCE|predict.py|callback"
```

**Subprocess Check**:
```bash
# predict.py가 실행되었는지 확인
ps aux | grep predict.py

# 완료 후에는 프로세스 종료되어야 함
```

#### Step 4: Fetch Results

```bash
# 프론트엔드가 결과를 가져오는 것과 동일
curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID/results" | jq .
```

**Expected Response**:
```json
{
  "inference_job_id": 1,
  "status": "completed",
  "total_images": 1,
  "results": [
    {
      "id": 1,
      "image_name": "000000000030.jpg",
      "image_path": "s3://training-checkpoints/inference/{uuid}/{uuid}.jpg",
      "predictions": [
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.89,
          "bbox": [123.4, 234.5, 456.7, 567.8]
        },
        ...
      ],
      "predicted_boxes": [
        {
          "label": "person",
          "confidence": 0.89,
          "x1": 123.4,
          "y1": 234.5,
          "x2": 456.7,
          "y2": 567.8
        },
        ...
      ],
      "inference_time_ms": 45.2
    }
  ]
}
```

**Validation**:
- `total_images` == 1
- `results` array has 1 element
- Each prediction has: `class_name`, `confidence`, `bbox`
- `predicted_boxes` matches `predictions`

**DB Check**:
```bash
# InferenceResult 레코드 확인
docker exec -it platform-postgres-1 psql -U admin -d platform -c \
  "SELECT id, image_name, array_length(predictions, 1) as num_predictions, inference_time_ms
   FROM inference_results
   WHERE inference_job_id = $INFERENCE_JOB_ID;"
```

#### Step 5: Verify Callback Was Called

```bash
# Backend 로그에서 callback 호출 확인
grep "INFERENCE CALLBACK" platform/backend/backend.log | tail -5

# Expected log entries:
# [INFERENCE CALLBACK] Received completion for inference job 1
# [INFERENCE CALLBACK] Status: completed, Total images: 1
# [INFERENCE CALLBACK] Created 1 InferenceResult records
# [INFERENCE CALLBACK] Updated inference job 1 - status: completed
```

---

### Scenario 2: Multiple Images Inference

**사용자 시나리오**:
- 이미지 3개 업로드 (000000000030.jpg, 000000000034.jpg, 000000000036.jpg)
- 배치 추론 실행

**API Test Steps**:

```bash
# Step 1: Upload 3 images
curl -X POST "$API_BASE_URL/test_inference/inference/upload-images?training_job_id=$TRAINING_JOB_ID" \
  -F "files=@$TEST_IMAGE_DIR/000000000030.jpg" \
  -F "files=@$TEST_IMAGE_DIR/000000000034.jpg" \
  -F "files=@$TEST_IMAGE_DIR/000000000036.jpg" \
  | jq .

# Expected: total_files == 3

export S3_PREFIX=$(curl -X POST "$API_BASE_URL/test_inference/inference/upload-images?training_job_id=$TRAINING_JOB_ID" \
  -F "files=@$TEST_IMAGE_DIR/000000000030.jpg" \
  -F "files=@$TEST_IMAGE_DIR/000000000034.jpg" \
  -F "files=@$TEST_IMAGE_DIR/000000000036.jpg" \
  2>/dev/null | jq -r '.s3_prefix')

# Step 2-5: Same as Scenario 1 but expect 3 results
# ...

# Validation:
# - total_images == 3
# - results array has 3 elements
# - DB has 3 InferenceResult records
```

---

### Scenario 3: Pretrained Weight Inference

**사용자 시나리오**:
- Checkpoint dropdown에서 "Pretrained Weight" 선택
- 이미지 1개로 추론

**API Test Steps**:

```bash
# Training Job의 pretrained weight 경로 확인
PRETRAINED_WEIGHT=$(curl -s "$API_BASE_URL/training/jobs/$TRAINING_JOB_ID" | jq -r '.model_name')
echo "Model: $PRETRAINED_WEIGHT"  # Expected: yolo11n.pt

# Checkpoint path를 모델 이름으로 설정
export CHECKPOINT_PATH="$PRETRAINED_WEIGHT"

# Step 1-5: Same as Scenario 1
# predict.py should download pretrained weight from Ultralytics
```

---

### Scenario 4: Error Handling - Invalid Checkpoint

**사용자 시나리오**:
- 존재하지 않는 checkpoint 경로 입력

**API Test Steps**:

```bash
# Create job with invalid checkpoint
curl -X POST "$API_BASE_URL/test_inference/inference/jobs" \
  -H "Content-Type: application/json" \
  -d "{
    \"training_job_id\": $TRAINING_JOB_ID,
    \"checkpoint_path\": \"s3://training-checkpoints/checkpoints/999/best.pt\",
    \"inference_type\": \"batch\",
    \"input_data\": {
      \"image_paths_s3\": \"$S3_PREFIX\"
    }
  }" \
  | jq .

# Expected: Job created (checkpoint validation happens in subprocess)

# Wait for status
for i in {1..30}; do
  STATUS=$(curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID" | jq -r '.status')
  echo "[Poll $i/30] Status: $STATUS"

  if [ "$STATUS" = "failed" ]; then
    echo "Job failed as expected"
    break
  fi

  sleep 2
done

# Check error message
curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID" | jq '.error_message'

# Expected: Error message about checkpoint not found
```

---

## Complete Test Script

모든 시나리오를 자동으로 실행하는 스크립트:

```bash
#!/bin/bash
# File: platform/backend/test_inference_e2e.sh

set -e

API_BASE_URL="http://localhost:8000/api/v1"
TRAINING_JOB_ID=23
TEST_IMAGE_DIR="platform/backend/test_inference_images"

echo "=== InferenceJob Pattern E2E Test ==="
echo ""

# Scenario 1: Single Image
echo "Scenario 1: Single Image Inference"
echo "-----------------------------------"

# Upload
echo "[1/5] Uploading image to S3..."
UPLOAD_RESPONSE=$(curl -s -X POST "$API_BASE_URL/test_inference/inference/upload-images?training_job_id=$TRAINING_JOB_ID" \
  -F "files=@$TEST_IMAGE_DIR/000000000030.jpg")

echo "$UPLOAD_RESPONSE" | jq .

S3_PREFIX=$(echo "$UPLOAD_RESPONSE" | jq -r '.s3_prefix')
TOTAL_FILES=$(echo "$UPLOAD_RESPONSE" | jq -r '.total_files')

if [ "$TOTAL_FILES" != "1" ]; then
  echo "❌ Upload failed: expected 1 file, got $TOTAL_FILES"
  exit 1
fi
echo "✓ Upload success: $S3_PREFIX"
echo ""

# Create job
echo "[2/5] Creating InferenceJob..."
CHECKPOINT_PATH="s3://training-checkpoints/checkpoints/$TRAINING_JOB_ID/best.pt"

JOB_RESPONSE=$(curl -s -X POST "$API_BASE_URL/test_inference/inference/jobs" \
  -H "Content-Type: application/json" \
  -d "{
    \"training_job_id\": $TRAINING_JOB_ID,
    \"checkpoint_path\": \"$CHECKPOINT_PATH\",
    \"inference_type\": \"batch\",
    \"input_data\": {
      \"image_paths_s3\": \"$S3_PREFIX\",
      \"confidence_threshold\": 0.25,
      \"iou_threshold\": 0.45,
      \"max_detections\": 100
    }
  }")

echo "$JOB_RESPONSE" | jq .

INFERENCE_JOB_ID=$(echo "$JOB_RESPONSE" | jq -r '.id')
JOB_STATUS=$(echo "$JOB_RESPONSE" | jq -r '.status')

if [ "$JOB_STATUS" != "pending" ]; then
  echo "❌ Job creation failed: expected status 'pending', got '$JOB_STATUS'"
  exit 1
fi
echo "✓ Job created: ID=$INFERENCE_JOB_ID"
echo ""

# Poll status
echo "[3/5] Waiting for completion..."
for i in {1..60}; do
  STATUS=$(curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID" | jq -r '.status')
  echo "  [Poll $i/60] Status: $STATUS"

  if [ "$STATUS" = "completed" ]; then
    echo "✓ Job completed"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "❌ Job failed"
    curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID" | jq '.error_message'
    exit 1
  fi

  sleep 2
done
echo ""

# Fetch results
echo "[4/5] Fetching results..."
RESULTS=$(curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID/results")
echo "$RESULTS" | jq .

TOTAL_IMAGES=$(echo "$RESULTS" | jq -r '.total_images')
NUM_RESULTS=$(echo "$RESULTS" | jq -r '.results | length')

if [ "$TOTAL_IMAGES" != "1" ] || [ "$NUM_RESULTS" != "1" ]; then
  echo "❌ Results mismatch: expected 1 image and 1 result"
  exit 1
fi
echo "✓ Results fetched: 1 image processed"
echo ""

# Verify predictions
echo "[5/5] Verifying predictions..."
NUM_PREDICTIONS=$(echo "$RESULTS" | jq -r '.results[0].predictions | length')
echo "  Number of predictions: $NUM_PREDICTIONS"

if [ "$NUM_PREDICTIONS" -gt 0 ]; then
  echo "✓ Predictions found"
  echo "$RESULTS" | jq '.results[0].predictions[0]'
else
  echo "⚠ No predictions (image might not contain objects)"
fi
echo ""

echo "==================================="
echo "✅ All tests passed!"
echo "==================================="
```

**사용법**:
```bash
chmod +x platform/backend/test_inference_e2e.sh
./platform/backend/test_inference_e2e.sh
```

---

## Validation Checklist

각 시나리오 실행 후 확인해야 할 항목:

### API Response Validation
- [ ] HTTP status code가 예상과 일치
- [ ] Response JSON schema가 올바름
- [ ] 필수 필드가 모두 존재
- [ ] 데이터 타입이 올바름

### Database Validation
- [ ] InferenceJob 레코드 생성됨
- [ ] Status transition이 올바름 (pending → running → completed)
- [ ] InferenceResult 레코드 개수가 이미지 개수와 일치
- [ ] Predictions JSON이 올바른 구조

### Storage Validation
- [ ] S3에 이미지가 업로드됨
- [ ] S3 URI가 정확함
- [ ] 파일이 실제로 존재함

### Process Validation
- [ ] Background task가 시작됨
- [ ] predict.py subprocess가 실행됨
- [ ] Callback API가 호출됨
- [ ] 프로세스가 정상 종료됨

### Log Validation
- [ ] Backend 로그에 에러 없음
- [ ] Callback 로그가 기록됨
- [ ] Timing이 합리적 (< 60초)

---

## Benefits of This Approach

1. **속도**: 프론트엔드 클릭 대신 curl 실행 (10x faster)
2. **디버깅**: 각 단계별 response/log 즉시 확인
3. **재현성**: 스크립트로 자동화, 언제든 재실행 가능
4. **신뢰성**: 실제 데이터로 테스트, 더미 없음
5. **CI/CD**: GitHub Actions에 통합 가능

---

## Next Steps

1. 이 문서 기반으로 실제 테스트 실행
2. 문제 발견 시 즉시 수정
3. 모든 시나리오 통과 후 프론트엔드 연동
4. 프론트엔드 연동 시 문제 없어야 함 (API 이미 검증됨)
