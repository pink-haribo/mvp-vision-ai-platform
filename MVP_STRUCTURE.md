# MVP 폴더 구조

## 📁 전체 구조

```
mvp-vision-platform/
├── mvp/                           # MVP 전용 디렉토리
│   ├── frontend/                  # Next.js 프론트엔드
│   ├── backend/                   # FastAPI 백엔드
│   ├── training/                  # 학습 모듈
│   ├── shared/                    # 공유 유틸리티
│   ├── data/                      # 데이터 저장소
│   └── scripts/                   # 유틸리티 스크립트
│
├── docs/                          # 문서 (기존)
├── .env.mvp                       # MVP 환경 변수
├── docker-compose.mvp.yml         # MVP용 Docker Compose (선택)
└── Makefile.mvp                   # MVP용 Make 명령어
```

---

## 🎯 모듈별 상세 구조

### 1. Backend 모듈

```
mvp/backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 앱 진입점
│   ├── config.py                  # 설정 관리
│   │
│   ├── api/                       # API 엔드포인트
│   │   ├── __init__.py
│   │   ├── deps.py               # 의존성 (DB 세션 등)
│   │   ├── chat.py               # 채팅 엔드포인트
│   │   ├── training.py           # 학습 엔드포인트
│   │   └── websocket.py          # WebSocket 엔드포인트
│   │
│   ├── core/                      # 핵심 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── llm/                  # LLM 관련
│   │   │   ├── __init__.py
│   │   │   ├── parser.py         # 자연어 파싱
│   │   │   ├── prompts.py        # 프롬프트 템플릿
│   │   │   └── models.py         # LLM 응답 모델
│   │   │
│   │   ├── training/             # 학습 관리
│   │   │   ├── __init__.py
│   │   │   ├── manager.py        # 학습 프로세스 관리
│   │   │   ├── monitor.py        # 진행률 모니터링
│   │   │   └── executor.py       # Subprocess 실행
│   │   │
│   │   └── websocket/            # WebSocket 관리
│   │       ├── __init__.py
│   │       ├── manager.py        # 연결 관리
│   │       └── events.py         # 이벤트 타입
│   │
│   ├── db/                        # 데이터베이스
│   │   ├── __init__.py
│   │   ├── base.py               # Base 모델
│   │   ├── session.py            # DB 세션
│   │   └── models/               # DB 모델
│   │       ├── __init__.py
│   │       ├── chat.py           # ChatSession 모델
│   │       └── training.py       # TrainingWorkflow 모델
│   │
│   ├── schemas/                   # Pydantic 스키마
│   │   ├── __init__.py
│   │   ├── chat.py               # 채팅 스키마
│   │   ├── training.py           # 학습 스키마
│   │   └── common.py             # 공통 스키마
│   │
│   └── utils/                     # 유틸리티
│       ├── __init__.py
│       ├── logger.py             # 로깅
│       └── helpers.py            # 헬퍼 함수
│
├── tests/                         # 테스트
│   ├── __init__.py
│   ├── conftest.py               # Pytest 설정
│   ├── test_llm_parser.py
│   ├── test_training_manager.py
│   └── test_api.py
│
├── alembic/                       # DB 마이그레이션 (선택)
│   ├── versions/
│   └── env.py
│
├── .env.example                   # 환경 변수 예시
├── requirements.txt               # Python 패키지
├── requirements-dev.txt           # 개발 전용 패키지
├── pyproject.toml                # Python 프로젝트 설정
└── README.md                      # Backend 문서
```

**핵심 모듈:**
- **api/**: HTTP 엔드포인트 (RESTful API + WebSocket)
- **core/llm/**: LLM 파싱 로직
- **core/training/**: 학습 프로세스 관리
- **core/websocket/**: WebSocket 연결 관리
- **db/**: 데이터베이스 모델 및 세션
- **schemas/**: 입출력 데이터 검증

---

### 2. Frontend 모듈

```
mvp/frontend/
├── app/                           # Next.js App Router
│   ├── layout.tsx                # 루트 레이아웃
│   ├── page.tsx                  # 메인 페이지
│   ├── globals.css               # 글로벌 스타일
│   │
│   └── api/                      # API Routes (선택)
│       └── health/
│           └── route.ts
│
├── components/                    # React 컴포넌트
│   ├── chat/                     # 채팅 관련
│   │   ├── ChatPanel.tsx         # 채팅 패널
│   │   ├── MessageList.tsx       # 메시지 리스트
│   │   ├── MessageInput.tsx      # 입력창
│   │   └── Message.tsx           # 개별 메시지
│   │
│   ├── training/                 # 학습 관련
│   │   ├── TrainingPanel.tsx     # 학습 패널
│   │   ├── ProgressBar.tsx       # 진행률 바
│   │   ├── MetricsDisplay.tsx    # 메트릭 표시
│   │   └── StatusBadge.tsx       # 상태 뱃지
│   │
│   ├── ui/                       # 재사용 UI 컴포넌트
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Input.tsx
│   │   └── Spinner.tsx
│   │
│   └── layout/                   # 레이아웃 컴포넌트
│       ├── Header.tsx
│       └── Container.tsx
│
├── lib/                          # 라이브러리/유틸리티
│   ├── api/                      # API 클라이언트
│   │   ├── client.ts            # Axios/Fetch 클라이언트
│   │   ├── chat.ts              # 채팅 API
│   │   └── training.ts          # 학습 API
│   │
│   ├── websocket/               # WebSocket
│   │   ├── client.ts            # WebSocket 클라이언트
│   │   └── hooks.ts             # WebSocket Hooks
│   │
│   ├── store/                   # 상태 관리 (Zustand)
│   │   ├── chat.ts              # 채팅 상태
│   │   └── training.ts          # 학습 상태
│   │
│   └── utils/                   # 유틸리티
│       ├── format.ts            # 포맷팅
│       └── cn.ts                # classnames 유틸
│
├── types/                        # TypeScript 타입
│   ├── chat.ts
│   ├── training.ts
│   └── api.ts
│
├── hooks/                        # Custom Hooks
│   ├── useChat.ts
│   ├── useTraining.ts
│   └── useWebSocket.ts
│
├── styles/                       # 스타일
│   └── components/
│
├── public/                       # 정적 파일
│   ├── images/
│   └── fonts/
│
├── tests/                        # 테스트
│   ├── unit/
│   └── e2e/
│
├── .env.local.example           # 환경 변수 예시
├── next.config.js               # Next.js 설정
├── tailwind.config.js           # Tailwind 설정
├── tsconfig.json                # TypeScript 설정
├── package.json                 # Node 패키지
└── README.md                    # Frontend 문서
```

**핵심 모듈:**
- **components/chat/**: 채팅 UI
- **components/training/**: 학습 모니터링 UI
- **lib/api/**: Backend API 통신
- **lib/websocket/**: 실시간 WebSocket 통신
- **lib/store/**: 전역 상태 관리

---

### 3. Training 모듈

```
mvp/training/
├── __init__.py
├── train_classification.py       # 메인 학습 스크립트
│
├── models/                       # 모델 정의
│   ├── __init__.py
│   └── resnet.py                # ResNet 래퍼
│
├── data/                         # 데이터 로더
│   ├── __init__.py
│   ├── loader.py                # DataLoader 생성
│   └── transforms.py            # 데이터 전처리
│
├── training/                     # 학습 로직
│   ├── __init__.py
│   ├── trainer.py               # Trainer 클래스
│   ├── metrics.py               # 메트릭 계산
│   └── callbacks.py             # 학습 콜백
│
├── utils/                        # 유틸리티
│   ├── __init__.py
│   ├── logger.py                # 로깅 (stdout)
│   └── checkpoint.py            # 체크포인트 저장
│
├── configs/                      # 설정
│   ├── __init__.py
│   └── default.py               # 기본 설정
│
├── requirements.txt              # Training 전용 패키지
└── README.md                     # Training 문서
```

**핵심 모듈:**
- **train_classification.py**: CLI로 실행 가능한 메인 스크립트
- **training/trainer.py**: 학습 루프 구현
- **data/loader.py**: 데이터셋 로딩
- **utils/logger.py**: stdout으로 진행률 출력

---

### 4. Shared 모듈 (공유)

```
mvp/shared/
├── __init__.py
├── types.py                      # 공통 타입 정의
├── constants.py                  # 상수
└── schemas.py                    # 공통 스키마
```

**역할:** Backend와 Training 간 공유되는 타입/상수

---

### 5. Data 디렉토리 (런타임 데이터)

```
mvp/data/
├── uploads/                      # 업로드된 데이터셋
│   └── .gitkeep
├── outputs/                      # 학습 결과
│   └── .gitkeep
├── models/                       # 저장된 모델
│   └── .gitkeep
├── logs/                         # 로그 파일
│   └── .gitkeep
└── db/                          # SQLite DB
    └── mvp.db
```

---

### 6. Scripts (유틸리티)

```
mvp/scripts/
├── setup_mvp.sh                 # MVP 초기 설정
├── create_sample_dataset.py     # 샘플 데이터셋 생성
├── reset_db.py                  # DB 초기화
└── run_dev.sh                   # 개발 서버 실행
```

---

## 🔧 설정 파일

### .env.mvp
```bash
# LLM
ANTHROPIC_API_KEY=sk-ant-xxx

# Database
DATABASE_URL=sqlite:///./mvp/data/db/mvp.db

# Paths
UPLOAD_DIR=./mvp/data/uploads
OUTPUT_DIR=./mvp/data/outputs
MODEL_DIR=./mvp/data/models
LOG_DIR=./mvp/data/logs

# API
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_PORT=3000

# Training
DEFAULT_EPOCHS=50
DEFAULT_BATCH_SIZE=32
DEFAULT_LEARNING_RATE=0.001
```

### Makefile.mvp
```makefile
.PHONY: help mvp-setup mvp-dev mvp-backend mvp-frontend mvp-clean

help:
	@echo "MVP Commands:"
	@echo "  make mvp-setup     - Setup MVP environment"
	@echo "  make mvp-dev       - Run both backend and frontend"
	@echo "  make mvp-backend   - Run backend only"
	@echo "  make mvp-frontend  - Run frontend only"
	@echo "  make mvp-clean     - Clean generated files"

mvp-setup:
	@echo "Setting up MVP..."
	cd mvp/backend && pip install -r requirements.txt
	cd mvp/frontend && pnpm install
	cp .env.mvp.example .env.mvp
	python mvp/scripts/reset_db.py

mvp-backend:
	cd mvp/backend && uvicorn app.main:app --reload --port 8000

mvp-frontend:
	cd mvp/frontend && pnpm dev

mvp-dev:
	@echo "Starting MVP (backend + frontend)..."
	@make -j2 mvp-backend mvp-frontend

mvp-clean:
	rm -rf mvp/data/db/*.db
	rm -rf mvp/data/uploads/*
	rm -rf mvp/data/outputs/*
	find mvp -type d -name "__pycache__" -exec rm -rf {} +
```

---

## 📦 패키지 파일

### mvp/backend/requirements.txt
```
# Web Framework
fastapi==0.108.0
uvicorn[standard]==0.25.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
alembic==1.13.0

# LLM
langchain==0.1.0
langchain-anthropic==0.1.0
langchain-core==0.1.0

# WebSocket
python-socketio==5.10.0
websockets==12.0

# Utils
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0

# ML (lightweight for config validation)
torch==2.1.0  # CPU only
torchvision==0.16.0
timm==0.9.12
```

### mvp/training/requirements.txt
```
# Deep Learning
torch==2.1.0
torchvision==0.16.0
timm==0.9.12

# Data Processing
numpy==1.26.0
pillow==10.1.0

# Utils
tqdm==4.66.0
pyyaml==6.0.1
```

### mvp/frontend/package.json (주요 부분)
```json
{
  "dependencies": {
    "next": "14.0.4",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "zustand": "4.4.7",
    "socket.io-client": "4.6.1",
    "axios": "1.6.2",
    "tailwindcss": "3.3.6"
  },
  "devDependencies": {
    "@types/node": "20.10.4",
    "@types/react": "18.2.45",
    "typescript": "5.3.3"
  }
}
```

---

## 📋 모듈 간 의존성

```
┌─────────────────────────────────────────────┐
│              Frontend (Next.js)              │
│  - HTTP API 호출                             │
│  - WebSocket 연결                            │
└──────────────────┬──────────────────────────┘
                   │ HTTP + WebSocket
                   ↓
┌─────────────────────────────────────────────┐
│            Backend (FastAPI)                 │
│  - API 엔드포인트                            │
│  - LLM 파싱                                  │
│  - 학습 프로세스 관리                        │
│  - WebSocket 이벤트 전송                     │
└──────────────────┬──────────────────────────┘
                   │ Subprocess
                   ↓
┌─────────────────────────────────────────────┐
│       Training (Python Script)               │
│  - PyTorch 학습                              │
│  - stdout으로 진행률 출력                    │
└─────────────────────────────────────────────┘
```

---

## 🚀 다음 단계

1. **폴더 구조 생성**: `make mvp-create-structure`
2. **초기 파일 생성**: 각 모듈의 `__init__.py`, 기본 파일
3. **Backend 구현**: Day 1-4
4. **Frontend 구현**: Day 5
5. **Training 구현**: Day 6-7
6. **통합**: Day 8-10

준비되셨으면 폴더 구조를 실제로 생성하겠습니다!
