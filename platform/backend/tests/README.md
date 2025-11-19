# Backend Tests

Backend 테스트 파일 모음.

## Folders

### e2e/
End-to-End 테스트 (전체 플로우 검증)
- `test_inference_e2e.py` - Inference 전체 플로우 테스트

### integration/
통합 테스트 (API 및 서비스 연동)
- `test_api_response.py` - API 응답 테스트
- `test_dual_storage.py` - Dual storage 아키텍처 테스트
- `test_endpoint.py` - 엔드포인트 테스트
- `test_organization_registration.py` - Organization 등록 테스트
- `test_upload.py` - 파일 업로드 테스트

## Usage

```bash
cd platform/backend

# E2E 테스트
venv/Scripts/python.exe tests/e2e/test_inference_e2e.py --job-id 23 --pretrained --images "test_images/*.jpg"

# Integration 테스트
venv/Scripts/python.exe tests/integration/test_upload.py
```

## Test Guide

E2E 테스트 작성 가이드는 [E2E_TEST_GUIDE.md](../../../docs/E2E_TEST_GUIDE.md) 참고.
