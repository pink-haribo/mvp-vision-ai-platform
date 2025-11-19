# Utility Scripts

Backend 유틸리티 스크립트 모음.

## Folders

### setup/
초기 설정 및 데이터 생성 스크립트
- `init_db.py` - 데이터베이스 초기화
- `create_admin.py` - 관리자 계정 생성
- `create_sample_dataset.py` - 샘플 데이터셋 생성

### check/
검증 및 확인 스크립트
- `check_admin.py` - 관리자 계정 확인
- `check_minio.py` - MinIO 연결 확인
- `check_checkpoint_columns.py` - DB 컬럼 확인

### convert/
데이터 변환 스크립트
- `convert_yolo_seg_to_platform.py` - YOLO segmentation 데이터셋 변환

## Usage

```bash
cd platform/backend
venv/Scripts/python.exe scripts/setup/init_db.py
venv/Scripts/python.exe scripts/check/check_minio.py
```
