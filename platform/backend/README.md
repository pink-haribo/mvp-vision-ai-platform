# Backend Module

FastAPI 기반 백엔드 서비스

## 구조

- **api/**: HTTP 엔드포인트 및 WebSocket
- **core/**: 핵심 비즈니스 로직
  - **llm/**: LLM 기반 자연어 파싱
  - **training/**: 학습 프로세스 관리
  - **websocket/**: WebSocket 연결 관리
- **db/**: 데이터베이스 모델 및 세션
- **schemas/**: Pydantic 스키마 (입출력 검증)
- **utils/**: 유틸리티 함수

## 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
uvicorn app.main:app --reload --port 8000
```

## API 문서

서버 실행 후: http://localhost:8000/docs
