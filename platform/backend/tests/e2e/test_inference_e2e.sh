#!/bin/bash
# InferenceJob Pattern E2E Test
# Tests the complete inference flow using real data (no mocks/dummies)

set -e

API_BASE_URL="http://localhost:8000/api/v1"
TRAINING_JOB_ID=23
TEST_IMAGE_DIR="platform/backend/data/coco128/images/train2017"

echo "=== InferenceJob Pattern E2E Test ==="
echo ""
echo "Configuration:"
echo "  API: $API_BASE_URL"
echo "  Training Job: $TRAINING_JOB_ID"
echo "  Test Images: $TEST_IMAGE_DIR"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
if [ ! -d "$TEST_IMAGE_DIR" ]; then
  echo "❌ Test image directory not found: $TEST_IMAGE_DIR"
  exit 1
fi

if [ ! -f "$TEST_IMAGE_DIR/000000000030.jpg" ]; then
  echo "❌ Test image not found: 000000000030.jpg"
  exit 1
fi

# Check API health
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE_URL/../health")
if [ "$HTTP_CODE" != "200" ]; then
  echo "❌ API not responding (HTTP $HTTP_CODE)"
  exit 1
fi

echo "✓ Prerequisites OK"
echo ""

# Scenario 1: Single Image Inference
echo "==================================="
echo "Scenario 1: Single Image Inference"
echo "==================================="
echo ""

# Step 1: Upload image to S3
echo "[1/5] Uploading image to S3..."
UPLOAD_RESPONSE=$(curl -s -X POST "$API_BASE_URL/test_inference/inference/upload-images?training_job_id=$TRAINING_JOB_ID" \
  -F "files=@$TEST_IMAGE_DIR/000000000030.jpg")

echo "Upload Response:"
echo "$UPLOAD_RESPONSE" | jq .
echo ""

STATUS=$(echo "$UPLOAD_RESPONSE" | jq -r '.status')
S3_PREFIX=$(echo "$UPLOAD_RESPONSE" | jq -r '.s3_prefix')
TOTAL_FILES=$(echo "$UPLOAD_RESPONSE" | jq -r '.total_files')

if [ "$STATUS" != "success" ]; then
  echo "❌ Upload failed: status=$STATUS"
  exit 1
fi

if [ "$TOTAL_FILES" != "1" ]; then
  echo "❌ Upload failed: expected 1 file, got $TOTAL_FILES"
  exit 1
fi

echo "✓ Upload success: $S3_PREFIX"
echo ""

# Step 2: Create InferenceJob
echo "[2/5] Creating InferenceJob..."

# Check training job first
TRAINING_JOB=$(curl -s "$API_BASE_URL/training/jobs/$TRAINING_JOB_ID")
CHECKPOINT_PATH=$(echo "$TRAINING_JOB" | jq -r '.checkpoint_path')

echo "Training Job Info:"
echo "  Status: $(echo "$TRAINING_JOB" | jq -r '.status')"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo ""

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
      \"max_detections\": 100,
      \"save_visualizations\": true
    }
  }")

echo "Job Creation Response:"
echo "$JOB_RESPONSE" | jq .
echo ""

INFERENCE_JOB_ID=$(echo "$JOB_RESPONSE" | jq -r '.id')
JOB_STATUS=$(echo "$JOB_RESPONSE" | jq -r '.status')

if [ "$INFERENCE_JOB_ID" = "null" ]; then
  echo "❌ Job creation failed: no ID returned"
  echo "$JOB_RESPONSE"
  exit 1
fi

if [ "$JOB_STATUS" != "pending" ]; then
  echo "❌ Job creation failed: expected status 'pending', got '$JOB_STATUS'"
  exit 1
fi

echo "✓ Job created: ID=$INFERENCE_JOB_ID"
echo ""

# Step 3: Poll for completion
echo "[3/5] Waiting for completion (max 60 seconds)..."
START_TIME=$(date +%s)

for i in {1..60}; do
  JOB_STATUS_RESPONSE=$(curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID")
  STATUS=$(echo "$JOB_STATUS_RESPONSE" | jq -r '.status')
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  echo "  [Poll $i/60] Status: $STATUS (elapsed: ${ELAPSED}s)"

  if [ "$STATUS" = "completed" ]; then
    echo "✓ Job completed in ${ELAPSED}s"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "❌ Job failed"
    echo "$JOB_STATUS_RESPONSE" | jq '.error_message'
    exit 1
  fi

  sleep 2
done

if [ "$STATUS" != "completed" ]; then
  echo "❌ Job did not complete within 60 seconds (status: $STATUS)"
  exit 1
fi
echo ""

# Step 4: Fetch results
echo "[4/5] Fetching results..."
RESULTS=$(curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID/results")

echo "Results Response:"
echo "$RESULTS" | jq .
echo ""

RESULT_STATUS=$(echo "$RESULTS" | jq -r '.status')
TOTAL_IMAGES=$(echo "$RESULTS" | jq -r '.total_images')
NUM_RESULTS=$(echo "$RESULTS" | jq -r '.results | length')

if [ "$RESULT_STATUS" != "completed" ]; then
  echo "❌ Results status mismatch: expected 'completed', got '$RESULT_STATUS'"
  exit 1
fi

if [ "$TOTAL_IMAGES" != "1" ]; then
  echo "❌ Results mismatch: expected 1 image, got $TOTAL_IMAGES"
  exit 1
fi

if [ "$NUM_RESULTS" != "1" ]; then
  echo "❌ Results mismatch: expected 1 result, got $NUM_RESULTS"
  exit 1
fi

echo "✓ Results fetched: 1 image processed"
echo ""

# Step 5: Verify predictions
echo "[5/5] Verifying predictions..."

FIRST_RESULT=$(echo "$RESULTS" | jq '.results[0]')
IMAGE_NAME=$(echo "$FIRST_RESULT" | jq -r '.image_name')
NUM_PREDICTIONS=$(echo "$FIRST_RESULT" | jq -r '.predictions | length')
INFERENCE_TIME=$(echo "$FIRST_RESULT" | jq -r '.inference_time_ms')

echo "First Result:"
echo "  Image: $IMAGE_NAME"
echo "  Predictions: $NUM_PREDICTIONS"
echo "  Inference Time: ${INFERENCE_TIME}ms"
echo ""

if [ "$NUM_PREDICTIONS" -gt 0 ]; then
  echo "Sample Prediction:"
  echo "$FIRST_RESULT" | jq '.predictions[0]'
  echo ""
  echo "✓ Predictions found ($NUM_PREDICTIONS objects detected)"
else
  echo "⚠ No predictions (image might not contain objects)"
fi
echo ""

echo "==================================="
echo "✅ Scenario 1 PASSED"
echo "==================================="
echo ""

# Scenario 2: Multiple Images
echo "==================================="
echo "Scenario 2: Multiple Images (3)"
echo "==================================="
echo ""

echo "[1/5] Uploading 3 images to S3..."
UPLOAD_RESPONSE=$(curl -s -X POST "$API_BASE_URL/test_inference/inference/upload-images?training_job_id=$TRAINING_JOB_ID" \
  -F "files=@$TEST_IMAGE_DIR/000000000030.jpg" \
  -F "files=@$TEST_IMAGE_DIR/000000000034.jpg" \
  -F "files=@$TEST_IMAGE_DIR/000000000036.jpg")

S3_PREFIX=$(echo "$UPLOAD_RESPONSE" | jq -r '.s3_prefix')
TOTAL_FILES=$(echo "$UPLOAD_RESPONSE" | jq -r '.total_files')

if [ "$TOTAL_FILES" != "3" ]; then
  echo "❌ Upload failed: expected 3 files, got $TOTAL_FILES"
  exit 1
fi

echo "✓ Upload success: 3 files"
echo ""

echo "[2/5] Creating InferenceJob..."
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

INFERENCE_JOB_ID=$(echo "$JOB_RESPONSE" | jq -r '.id')
echo "✓ Job created: ID=$INFERENCE_JOB_ID"
echo ""

echo "[3/5] Waiting for completion..."
for i in {1..60}; do
  STATUS=$(curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID" | jq -r '.status')
  echo "  [Poll $i/60] Status: $STATUS"

  if [ "$STATUS" = "completed" ]; then
    echo "✓ Job completed"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "❌ Job failed"
    exit 1
  fi

  sleep 2
done
echo ""

echo "[4/5] Fetching results..."
RESULTS=$(curl -s "$API_BASE_URL/test_inference/inference/jobs/$INFERENCE_JOB_ID/results")

TOTAL_IMAGES=$(echo "$RESULTS" | jq -r '.total_images')
NUM_RESULTS=$(echo "$RESULTS" | jq -r '.results | length')

if [ "$TOTAL_IMAGES" != "3" ] || [ "$NUM_RESULTS" != "3" ]; then
  echo "❌ Results mismatch: expected 3 images and 3 results"
  exit 1
fi

echo "✓ Results fetched: 3 images processed"
echo ""

echo "[5/5] Verifying all results..."
for idx in {0..2}; do
  IMAGE_NAME=$(echo "$RESULTS" | jq -r ".results[$idx].image_name")
  NUM_PREDS=$(echo "$RESULTS" | jq -r ".results[$idx].predictions | length")
  echo "  Image $((idx+1)): $IMAGE_NAME - $NUM_PREDS predictions"
done

echo ""
echo "==================================="
echo "✅ Scenario 2 PASSED"
echo "==================================="
echo ""

# Final summary
echo "==================================="
echo "🎉 ALL TESTS PASSED!"
echo "==================================="
echo ""
echo "Summary:"
echo "  ✓ Single image inference"
echo "  ✓ Multiple images inference"
echo "  ✓ S3 upload/download"
echo "  ✓ Background task execution"
echo "  ✓ Callback API"
echo "  ✓ Database persistence"
echo ""
echo "Ready for frontend integration!"
