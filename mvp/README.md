# MVP - Vision AI Training Platform

간소화된 MVP 구현

## 📁 구조

```
mvp/
├── backend/         # FastAPI 백엔드
├── frontend/        # Next.js 프론트엔드 (예정)
├── training/        # PyTorch 학습 스크립트
├── shared/          # 공유 유틸리티
├── data/            # 런타임 데이터
└── scripts/         # 헬퍼 스크립트
```

## 🚀 Quick Start

### 1. 환경 설정

```bash
# MVP 설정 (루트 디렉토리에서)
make -f Makefile.mvp mvp-setup

# .env.mvp 파일 편집
# ANTHROPIC_API_KEY 추가
```

### 2. Backend 실행

```bash
# 개발 서버 시작
make -f Makefile.mvp mvp-backend

# API 문서 확인
# http://localhost:8000/docs
```

### 3. 테스트

```bash
# Backend 테스트
make -f Makefile.mvp mvp-test
```

## 📚 문서

- [Backend](backend/README.md)
- [Training](training/README.md)
- [MVP 계획](../MVP_PLAN.md)
- [폴더 구조](../MVP_STRUCTURE.md)

## 🔧 개발

### Backend

```bash
cd mvp/backend
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 실행
uvicorn app.main:app --reload --port 8000

# 테스트
pytest tests/
```

### Training

```bash
cd mvp/training
pip install -r requirements.txt

# 학습 실행
python train_classification.py --help
```

## 📝 TODO

- [ ] Backend main.py 구현
- [ ] LLM 파싱 구현
- [ ] 학습 스크립트 구현
- [ ] WebSocket 구현
- [ ] Frontend 구현

## 🎯 MVP 범위

**포함:**
- ✅ 자연어 파싱 (Claude)
- ✅ ResNet50 Classification
- ✅ 로컬 학습 실행
- ✅ 실시간 진행률 (WebSocket)

**제외:**
- ❌ 다중 모델
- ❌ 사용자 인증
- ❌ Kubernetes
- ❌ 추론 API
