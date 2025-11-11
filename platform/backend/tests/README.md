# Backend Tests

이 폴더에는 Vision AI Training Platform의 백엔드 테스트 코드가 포함되어 있습니다.

## 테스트 구조

```
tests/
├── conftest.py              # pytest fixtures (DB, client, 샘플 데이터)
├── unit/                    # 단위 테스트
│   ├── test_adapter_imports.py    # Adapter 모듈 import 테스트
│   └── test_encoding.py           # UTF-8 인코딩 테스트
└── integration/             # 통합 테스트
    ├── test_yolo11n_bug.py        # yolo11n → yolov8n 버그 테스트
    └── test_models_api.py         # /models/list API 테스트
```

## 설치

```bash
cd mvp/backend

# pytest 및 의존성 설치
../../mvp/backend/venv/Scripts/pip.exe install pytest pytest-asyncio httpx pillow numpy

# 또는 requirements-test.txt가 있다면
../../mvp/backend/venv/Scripts/pip.exe install -r requirements-test.txt
```

## 테스트 실행

### 전체 테스트 실행

```bash
cd mvp/backend
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/ -v
```

### 특정 테스트 파일 실행

```bash
# yolo11n 버그 테스트만 실행
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/integration/test_yolo11n_bug.py -v

# 인코딩 테스트만 실행
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/unit/test_encoding.py -v

# 모델 API 테스트만 실행
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/integration/test_models_api.py -v
```

### 특정 테스트 케이스만 실행

```bash
# yolo11n job creation 테스트만 실행
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/integration/test_yolo11n_bug.py::TestYolo11nBugFix::test_yolo11n_job_creation -v
```

### Coverage 리포트와 함께 실행

```bash
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/ --cov=app --cov-report=html
```

## 테스트 우선순위

### P0: 버그 재발 방지 테스트 (즉시 실행)

이 테스트들은 현재 발견된 버그가 다시 발생하지 않도록 보장합니다.

```bash
# yolo11n → yolov8n 버그 테스트
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/integration/test_yolo11n_bug.py -v

# UTF-8 인코딩 테스트
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/unit/test_encoding.py -v

# Adapter import 테스트
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/unit/test_adapter_imports.py -v
```

### P1: 핵심 기능 테스트 (우선 실행)

```bash
# 모델 레지스트리 API 테스트
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/integration/test_models_api.py -v
```

## 주요 테스트 설명

### test_yolo11n_bug.py

**목적:** yolo11n 선택 시 yolov8n으로 저장되는 버그 재발 방지

**테스트 케이스:**
- `test_yolo11n_job_creation`: yolo11n으로 job 생성 시 DB에 yolo11n 저장 확인
- `test_yolo11m_job_creation`: yolo11m도 동일하게 테스트
- `test_yolo11l_job_creation`: yolo11l도 동일하게 테스트
- `test_yolo8_models_still_work`: yolov8 모델들이 여전히 작동하는지 확인
- `test_database_persistence`: DB에 직접 쿼리하여 확인

### test_encoding.py

**목적:** Windows cp949 인코딩 문제 방지

**테스트 케이스:**
- `test_subprocess_with_utf8_output`: subprocess가 UTF-8 출력을 처리할 수 있는지
- `test_yolo_output_with_special_chars`: YOLO 출력의 이모지/프로그레스바 처리
- `test_korean_path_handling`: 한글 경로 처리
- `test_mixed_encoding_in_stderr`: stderr UTF-8 처리

### test_adapter_imports.py

**목적:** Docker 컨테이너에서 adapter import 오류 방지

**테스트 케이스:**
- `test_base_adapter_import`: base.py import 성공
- `test_ultralytics_adapter_import`: ultralytics_adapter.py import 성공
- `test_timm_adapter_import`: timm_adapter.py import 성공
- `test_ultralytics_adapter_uses_base`: 상속 관계 확인

### test_models_api.py

**목적:** 모델 레지스트리 API 정확성 검증

**테스트 케이스:**
- `test_models_list_endpoint`: /models/list가 정상 작동
- `test_yolo11_models_in_list`: yolo11n, yolo11m, yolo11l이 목록에 포함
- `test_yolo11n_metadata`: yolo11n 메타데이터 정확성
- `test_model_status_field`: status 필드 존재 및 유효성
- `test_yolo8_and_yolo11_coexist`: yolov8과 yolo11이 공존

## CI/CD 통합

GitHub Actions workflow 예시:

```yaml
name: Backend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          cd mvp/backend
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      - name: Run P0 tests
        run: |
          cd mvp/backend
          python -m pytest tests/integration/test_yolo11n_bug.py -v
          python -m pytest tests/unit/test_encoding.py -v
```

## 문제 해결

### pytest not found

```bash
../../mvp/backend/venv/Scripts/pip.exe install pytest
```

### ModuleNotFoundError: No module named 'app'

```bash
# mvp/backend 디렉토리에서 실행해야 합니다
cd mvp/backend
../../mvp/backend/venv/Scripts/python.exe -m pytest tests/ -v
```

### 데이터베이스 관련 오류

conftest.py가 in-memory SQLite를 사용하므로 별도 DB 설정 불필요합니다.

## 다음 단계

### P2: 추가 테스트 작성 (향후)

- `test_training_job_lifecycle.py`: 학습 작업 생명주기 테스트
- `test_inference_pretrained.py`: Pretrained 모델 추론 테스트
- `test_dataset_analyzer.py`: 데이터셋 분석 테스트
- `test_mlflow_integration.py`: MLflow 연동 테스트
