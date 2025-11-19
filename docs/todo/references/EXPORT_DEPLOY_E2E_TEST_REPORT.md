# Export & Deploy E2E Test Report

**Test Date**: 2025-11-19
**Test Result**: 8/11 Passed (73%)
**Status**: Core Export functionality verified

---

## Test Summary

Export & Deploy 기능의 E2E 테스트를 실행하여 핵심 기능이 정상 동작함을 확인했습니다.

### Test Environment

- **Backend**: FastAPI on localhost:8000
- **MinIO (Internal)**: localhost:9002 (checkpoints, exports)
- **MinIO (External)**: localhost:9000 (datasets)
- **Training Job ID**: 23 (yolo11n, object detection)

---

## Test Results

### Passed Tests (8/11)

| Test | Description | Result |
|------|-------------|--------|
| Authentication | Login and get JWT token | PASS |
| Get Training Job Info | Retrieve training job details | PASS |
| Get Export Capabilities | Query supported export formats | PASS |
| Create Export Job | Create ONNX export job | PASS |
| Poll Export Job Status | Wait for export completion (~15s) | PASS |
| Get Export Download URL | Generate presigned URL for download | PASS |
| Create Deployment | Create platform_endpoint deployment | PASS |
| List Deployments | Query deployments for training job | PASS |

### Failed Tests (3/11)

| Test | Description | Reason |
|------|-------------|--------|
| Deactivate Deployment | PATCH /deployments/{id}/deactivate | Endpoint not implemented |
| Activate Deployment | PATCH /deployments/{id}/activate | Endpoint not implemented |
| Test Inference | Run inference on deployed model | No API key (deployment pending) |

---

## Bugs Fixed During Testing

### 1. Windows cp949 Encoding Error
- **Location**: `test_export_deploy_e2e.py:print_summary()`
- **Error**: `UnicodeEncodeError: 'cp949' codec can't encode character`
- **Fix**: Removed emoji characters from print output

### 2. Task Type Mismatch
- **Location**: `platform/backend/app/api/export.py`
- **Error**: `Task type 'detection' not supported for framework 'ultralytics'`
- **Fix**: Added `TASK_TYPE_ALIASES` dictionary and `normalize_task_type()` function
```python
TASK_TYPE_ALIASES = {
    "detection": "object_detection",
    "segmentation": "instance_segmentation",
    "pose": "pose_estimation",
    "classification": "image_classification",
}
```

### 3. Invalid dynamic_axes Schema
- **Location**: `test_export_deploy_e2e.py`
- **Error**: `422: Input should be a valid dictionary` for dynamic_axes
- **Fix**: Removed `dynamic_axes: False` from default export config

### 4. ONNX Packages Not Installed
- **Location**: Ultralytics trainer venv
- **Error**: `No module named 'onnx'`
- **Fix**: `pip install onnx onnxslim onnxruntime`

### 5. Export Callback URL Mismatch
- **Location**: `platform/trainers/ultralytics/export.py:460`
- **Error**: `POST /api/v1/export/1/callback/completion HTTP/1.1" 404`
- **Fix**: Changed URL to `/export/jobs/{id}/callback/completion`

### 6. export_format Variable Not Defined
- **Location**: `platform/trainers/ultralytics/export.py:396`
- **Error**: `name 'export_format' is not defined`
- **Fix**: Extract from metadata: `export_format = metadata.get('model_info', {}).get('export_format', 'unknown')`

### 7. Download Endpoint Missing
- **Location**: `platform/backend/app/api/export.py`
- **Error**: `GET /api/v1/export/{id}/download HTTP/1.1" 404`
- **Fix**: Added `get_export_download_url` endpoint with presigned URL generation

### 8. S3 URI Parsing for Presigned URL
- **Location**: `platform/backend/app/api/export.py:get_export_download_url`
- **Error**: `500 Internal Server Error` (invalid object_key)
- **Fix**: Extract object_key from S3 URI (`s3://bucket/key` -> `key`)

### 9. Deployment URL Path
- **Location**: `test_export_deploy_e2e.py`
- **Error**: `POST /api/v1/deployments HTTP/1.1" 404`
- **Fix**: Changed to `/api/v1/export/deployments` (router prefix is `/export`)

### 10. Missing deployment_name Field
- **Location**: `test_export_deploy_e2e.py:create_deployment()`
- **Error**: `422: Field required: deployment_name`
- **Fix**: Added `deployment_name` to request body

---

## Files Modified

### Backend
- `platform/backend/app/api/export.py`
  - Added `dual_storage` import
  - Added `TASK_TYPE_ALIASES` and `normalize_task_type()`
  - Added `get_export_download_url` endpoint
  - Fixed S3 URI to object_key extraction

### Trainer
- `platform/trainers/ultralytics/export.py`
  - Fixed callback URL path
  - Fixed `export_format` variable bug

### Tests
- `platform/backend/tests/e2e/test_export_deploy_e2e.py`
  - Removed emoji characters
  - Fixed export config
  - Fixed deployment URLs
  - Added `deployment_name` field

---

## Test Command

```bash
cd platform/backend
venv/Scripts/python.exe tests/e2e/test_export_deploy_e2e.py \
  --training-job-id 23 \
  --test-image test_images/000000000025.jpg \
  --no-cleanup
```

---

## Remaining Work

### Backend Implementation Needed
1. **Activate Deployment Endpoint**: `PATCH /export/deployments/{id}/activate`
2. **Deactivate Deployment Endpoint**: `PATCH /export/deployments/{id}/deactivate`
3. **Deployment Execution**: Background task to actually deploy the model

### Frontend Alignment
- Frontend uses `/api/v1/deployments` but backend expects `/api/v1/export/deployments`
- Need to either:
  - Update frontend to use `/export/deployments`, or
  - Create separate router for deployments without `/export` prefix

---

## Conclusion

Export 기능의 핵심 E2E 플로우가 검증되었습니다:
- Export job 생성 및 실행
- ONNX 모델 변환
- MinIO 업로드 및 presigned URL 생성
- Deployment 레코드 생성

Activate/Deactivate 엔드포인트 구현 후 전체 E2E 테스트가 통과할 것으로 예상됩니다.
