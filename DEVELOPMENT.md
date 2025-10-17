# 개발 환경 설정 가이드

## 목차
- [Prerequisites](#prerequisites)
- [로컬 개발 환경](#로컬-개발-환경)
- [개발 워크플로우](#개발-워크플로우)
- [코딩 컨벤션](#코딩-컨벤션)
- [디버깅](#디버깅)
- [테스트](#테스트)
- [트러블슈팅](#트러블슈팅)

## Prerequisites

### 필수 도구

#### 1. Node.js

```bash
# nvm 설치 (권장)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Node.js 20.x 설치
nvm install 20
nvm use 20
nvm alias default 20

# 확인
node --version  # v20.x.x
npm --version   # 10.x.x
```

#### 2. Python

```bash
# pyenv 설치 (권장)
curl https://pyenv.run | bash

# Python 3.11 설치
pyenv install 3.11.6
pyenv global 3.11.6

# 확인
python --version  # Python 3.11.6
```

#### 3. Docker Desktop

[Docker Desktop 설치](https://www.docker.com/products/docker-desktop)

```bash
# 확인
docker --version     # Docker version 24.x.x
docker-compose --version  # Docker Compose version v2.x.x
```

#### 4. Kubernetes CLI

```bash
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# 확인
kubectl version --client
```

#### 5. Helm

```bash
# macOS
brew install helm

# Linux
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 확인
helm version
```

### 권장 도구

```bash
# pnpm (빠른 Node.js 패키지 매니저)
npm install -g pnpm

# Poetry (Python 패키지 매니저)
curl -sSL https://install.python-poetry.org | python3 -

# k9s (Kubernetes 대시보드)
brew install derailed/k9s/k9s

# jq (JSON 처리)
brew install jq

# httpie (HTTP 클라이언트)
brew install httpie

# act (GitHub Actions 로컬 테스트)
brew install act
```

### IDE 설정

#### VS Code Extensions

```bash
# 필수
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension dbaeumer.vscode-eslint
code --install-extension esbenp.prettier-vscode
code --install-extension bradlc.vscode-tailwindcss

# 추천
code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
code --install-extension ms-azuretools.vscode-docker
code --install-extension eamodio.gitlens
code --install-extension GitHub.copilot
```

#### VS Code 설정 (.vscode/settings.json)

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "tailwindCSS.experimental.classRegex": [
    ["cva\\(([^)]*)\\)", "[\"'`]([^\"'`]*).*?[\"'`]"]
  ]
}
```

---

## 로컬 개발 환경

### 1. 레포지토리 클론

```bash
git clone https://github.com/your-org/vision-platform.git
cd vision-platform
```

### 2. 환경 변수 설정

#### 루트 디렉토리

```bash
cp .env.example .env
```

**.env 예시:**
```bash
# LLM API Keys
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=postgresql://admin:devpass@localhost:5432/vision_platform
MONGODB_URL=mongodb://localhost:27017/vision_platform
REDIS_URL=redis://localhost:6379

# Object Storage
S3_ENDPOINT=http://localhost:9000
S3_BUCKET=vision-platform-dev
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

# Services
API_GATEWAY_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000

# Temporal
TEMPORAL_HOST=localhost:7233

# Auth
JWT_SECRET=your-super-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

#### Frontend

```bash
cp frontend/.env.example frontend/.env.local
```

**frontend/.env.local:**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_ENVIRONMENT=development
```

#### Backend 서비스

각 서비스별로:
```bash
cp backend/intent-parser/.env.example backend/intent-parser/.env
cp backend/orchestrator/.env.example backend/orchestrator/.env
cp backend/model-registry/.env.example backend/model-registry/.env
cp backend/data-service/.env.example backend/data-service/.env
cp backend/vm-controller/.env.example backend/vm-controller/.env
```

### 3. Docker Compose로 인프라 실행

```bash
# PostgreSQL, MongoDB, Redis, MinIO 실행
docker-compose up -d

# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: vision-postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: vision_platform
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: devpass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin"]
      interval: 10s
      timeout: 5s
      retries: 5

  mongodb:
    image: mongo:7
    container_name: vision-mongodb
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/test --quiet
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7.2-alpine
    container_name: vision-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  minio:
    image: minio/minio:latest
    container_name: vision-minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"  # API
      - "9001:9001"  # Console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

volumes:
  postgres_data:
  mongo_data:
  redis_data:
  minio_data:
```

### 4. 데이터베이스 초기화

```bash
# PostgreSQL 마이그레이션
cd backend/orchestrator
poetry install
poetry run alembic upgrade head

# MongoDB 인덱스 생성 (선택)
python scripts/init_mongodb.py
```

### 5. Frontend 실행

```bash
cd frontend

# 의존성 설치
pnpm install

# 개발 서버 실행
pnpm dev

# 브라우저에서 http://localhost:3000 접속
```

**사용 가능한 스크립트:**
```bash
pnpm dev          # 개발 서버
pnpm build        # 프로덕션 빌드
pnpm start        # 프로덕션 서버
pnpm lint         # ESLint 실행
pnpm type-check   # TypeScript 타입 체크
```

### 6. Backend 서비스 실행

각 서비스를 별도 터미널에서 실행합니다.

#### Terminal 1: Intent Parser

```bash
cd backend/intent-parser
poetry install
poetry run uvicorn app.main:app --reload --port 8001

# http://localhost:8001/docs (Swagger UI)
```

#### Terminal 2: Orchestrator

```bash
cd backend/orchestrator
poetry install
poetry run uvicorn app.main:app --reload --port 8002

# http://localhost:8002/docs
```

#### Terminal 3: Model Registry

```bash
cd backend/model-registry
poetry install
poetry run uvicorn app.main:app --reload --port 8003

# http://localhost:8003/docs
```

#### Terminal 4: Data Service

```bash
cd backend/data-service
poetry install
poetry run uvicorn app.main:app --reload --port 8004

# http://localhost:8004/docs
```

#### Terminal 5: VM Controller

```bash
cd backend/vm-controller
poetry install
poetry run uvicorn app.main:app --reload --port 8005

# http://localhost:8005/docs
```

#### Terminal 6: Telemetry Service

```bash
cd backend/telemetry
poetry install
poetry run uvicorn app.main:app --reload --port 8006

# http://localhost:8006/docs
```

### 7. Temporal 로컬 실행 (선택)

```bash
# Temporal Dev Server (SQLite 기반)
temporal server start-dev

# Temporal UI
# http://localhost:8233
```

또는 Docker로 실행:
```bash
docker run -d -p 7233:7233 -p 8233:8233 temporalio/auto-setup:latest
```

### 8. Makefile 활용 (권장)

**Makefile:**
```makefile
.PHONY: help dev-up dev-down dev-frontend dev-backend test-all

help:
	@echo "Vision Platform Development Commands"
	@echo "  make dev-up        - Start all services"
	@echo "  make dev-down      - Stop all services"
	@echo "  make dev-frontend  - Start frontend only"
	@echo "  make dev-backend   - Start all backend services"
	@echo "  make test-all      - Run all tests"

dev-up:
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	sleep 10
	@echo "Infrastructure is ready!"

dev-down:
	docker-compose down

dev-frontend:
	cd frontend && pnpm dev

dev-backend:
	@echo "Starting all backend services..."
	# tmux를 사용하여 여러 서비스를 한 번에 실행
	tmux new-session -d -s vision-backend
	tmux send-keys -t vision-backend:0 'cd backend/intent-parser && poetry run uvicorn app.main:app --reload --port 8001' C-m
	tmux split-window -t vision-backend:0 -h
	tmux send-keys -t vision-backend:0.1 'cd backend/orchestrator && poetry run uvicorn app.main:app --reload --port 8002' C-m
	# ... 다른 서비스들
	tmux attach -t vision-backend

test-all:
	cd frontend && pnpm test
	cd backend/intent-parser && poetry run pytest
	cd backend/orchestrator && poetry run pytest
```

**사용:**
```bash
make dev-up          # 인프라 시작
make dev-frontend    # Frontend 실행
make dev-backend     # 모든 Backend 실행
make dev-down        # 인프라 종료
```

---

## 개발 워크플로우

### 브랜치 전략 (Git Flow)

```
main (protected)
├── develop (default branch)
│   ├── feature/chat-interface
│   ├── feature/model-adapter-timm
│   ├── feature/workflow-engine
│   └── bugfix/dataset-validation
└── release/v1.0.0
```

**브랜치 네이밍:**
```
feature/<feature-name>    # 새 기능
bugfix/<bug-name>         # 버그 수정
hotfix/<issue>            # 긴급 수정
release/<version>         # 릴리즈 준비
```

### 커밋 컨벤션 (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅 (동작 변경 없음)
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드/설정 변경
- `perf`: 성능 개선

**예시:**
```bash
feat(frontend): add chat interface component

- Implement real-time chat UI
- Add message history
- Connect to WebSocket

Closes #123

---

fix(orchestrator): handle timeout error in workflow

The workflow was not properly handling timeout errors,
causing the system to hang indefinitely.

Fixes #456

---

docs(api): update endpoint specifications

Add detailed request/response examples for /workflows endpoint
```

### Pull Request 프로세스

1. **기능 개발**
```bash
git checkout develop
git checkout -b feature/new-feature
# ... 개발 ...
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature
```

2. **PR 생성**
- GitHub에서 Pull Request 생성
- Template에 따라 설명 작성
- Reviewer 지정

3. **코드 리뷰 체크리스트**
- [ ] 코드가 컨벤션을 따르는가?
- [ ] 테스트가 포함되어 있는가?
- [ ] 테스트가 통과하는가?
- [ ] 문서가 업데이트 되었는가?
- [ ] Breaking change가 있다면 CHANGELOG 업데이트?
- [ ] 성능에 영향이 있는가?

4. **Merge**
- Squash and merge (권장)
- Rebase and merge (깔끔한 히스토리)
- Merge commit (브랜치 히스토리 유지)

---

## 코딩 컨벤션

### Python (PEP 8 + Black)

```python
# 좋은 예
def calculate_accuracy(
    predictions: List[int],
    targets: List[int]
) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        Accuracy score between 0 and 1
    """
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets)

# 나쁜 예
def calc_acc(p,t):
    return sum([1 for i in range(len(p)) if p[i]==t[i]])/len(p)
```

**Linting & Formatting:**
```bash
# Black (auto-formatting)
black .

# isort (import sorting)
isort .

# flake8 (linting)
flake8 .

# mypy (type checking)
mypy .
```

**pyproject.toml:**
```toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

### TypeScript / JavaScript (Prettier + ESLint)

```typescript
// 좋은 예
interface TrainingConfig {
  modelName: string;
  epochs: number;
  batchSize: number;
}

async function startTraining(config: TrainingConfig): Promise<WorkflowResult> {
  const response = await fetch('/api/workflows', {
    method: 'POST',
    body: JSON.stringify(config),
  });
  
  if (!response.ok) {
    throw new Error(`Training failed: ${response.statusText}`);
  }
  
  return response.json();
}

// 나쁜 예
function start(c) {
  return fetch('/api/workflows', {method: 'POST', body: JSON.stringify(c)}).then(r => r.json())
}
```

**.prettierrc:**
```json
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100
}
```

**.eslintrc.json:**
```json
{
  "extends": [
    "next/core-web-vitals",
    "plugin:@typescript-eslint/recommended",
    "prettier"
  ],
  "rules": {
    "@typescript-eslint/no-unused-vars": "error",
    "@typescript-eslint/no-explicit-any": "warn"
  }
}
```

---

## 디버깅

### Frontend 디버깅 (VS Code)

**.vscode/launch.json:**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Next.js: debug server-side",
      "type": "node-terminal",
      "request": "launch",
      "command": "pnpm dev"
    },
    {
      "name": "Next.js: debug client-side",
      "type": "chrome",
      "request": "launch",
      "url": "http://localhost:3000"
    }
  ]
}
```

### Backend 디버깅 (Python debugpy)

```python
# main.py
import debugpy

if __name__ == "__main__":
    # Debug mode
    debugpy.listen(("0.0.0.0", 5678))
    print("⏳ Waiting for debugger attach...")
    debugpy.wait_for_client()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**.vscode/launch.json:**
```json
{
  "configurations": [
    {
      "name": "Python: Attach",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      }
    }
  ]
}
```

### 로그 확인

```bash
# Docker 로그
docker-compose logs -f postgres
docker-compose logs -f redis

# Backend 서비스 로그
tail -f backend/intent-parser/logs/app.log

# Kubernetes 로그 (프로덕션)
kubectl logs -f deployment/intent-parser -n vision-platform
kubectl logs -f pod/trainer-wf_789xyz -n training

# 로그 레벨 변경
export LOG_LEVEL=DEBUG
poetry run uvicorn app.main:app --reload
```

---

## 테스트

### Frontend 테스트

```bash
cd frontend

# Unit & Integration tests (Jest + React Testing Library)
pnpm test

# Watch mode
pnpm test:watch

# Coverage
pnpm test:coverage

# E2E tests (Playwright)
pnpm test:e2e

# E2E UI mode
pnpm test:e2e:ui
```

**테스트 예시:**
```typescript
// components/Button.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

describe('Button', () => {
  it('renders correctly', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('handles click events', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    
    fireEvent.click(screen.getByText('Click me'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('shows loading state', () => {
    render(<Button loading>Click me</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
```

### Backend 테스트

```bash
cd backend/intent-parser

# Unit tests
poetry run pytest tests/unit -v

# Integration tests
poetry run pytest tests/integration -v

# E2E tests
poetry run pytest tests/e2e -v

# Coverage
poetry run pytest --cov=app --cov-report=html tests/

# 특정 테스트만
poetry run pytest tests/unit/test_parser.py::test_parse_classification -v
```

**테스트 예시:**
```python
# tests/unit/test_parser.py
import pytest
from app.services.intent_parser import IntentParser

@pytest.fixture
def parser():
    return IntentParser()

@pytest.mark.asyncio
async def test_parse_classification(parser):
    message = "ResNet50으로 고양이 3종류 분류"
    result = await parser.parse(message)
    
    assert result.task_type == "classification"
    assert result.model_name == "resnet50"
    assert result.num_classes == 3

@pytest.mark.asyncio
async def test_parse_incomplete_intent(parser):
    message = "모델 학습하고 싶어요"
    result = await parser.parse(message)
    
    assert result.type == "clarification_needed"
    assert len(result.questions) > 0
```

### 전체 테스트 실행

```bash
# 루트 디렉토리에서
make test-all

# 또는
./scripts/run-all-tests.sh
```

---

## 트러블슈팅

### 포트 충돌

```bash
# 사용 중인 포트 확인
lsof -i :3000
lsof -i :8000

# 프로세스 종료
kill -9 <PID>

# 또는 포트 변경
PORT=3001 pnpm dev
```

### Docker 용량 부족

```bash
# 디스크 사용량 확인
docker system df

# 미사용 컨테이너/이미지 정리
docker system prune -a

# 볼륨까지 정리 (주의: 데이터 삭제됨)
docker system prune -a --volumes
```

### 데이터베이스 마이그레이션 오류

```bash
# PostgreSQL 초기화
docker-compose down -v
docker-compose up -d postgres

# 마이그레이션 재실행
cd backend/orchestrator
poetry run alembic downgrade base
poetry run alembic upgrade head
```

### Python 의존성 충돌

```bash
# 가상환경 재생성
cd backend/intent-parser
rm -rf .venv
poetry install

# 특정 패키지 업데이트
poetry update langchain

# Lock 파일 재생성
poetry lock --no-update
```

### Node.js 모듈 오류

```bash
# node_modules 재설치
cd frontend
rm -rf node_modules
rm pnpm-lock.yaml
pnpm install

# 캐시 정리
pnpm store prune
```

### Kubernetes 로컬 클러스터 오류

```bash
# Docker Desktop Kubernetes 재시작
# Settings → Kubernetes → Disable → Enable

# 또는 minikube 사용
minikube start --driver=docker
minikube dashboard
```

---

## 다음 단계

- [아키텍처 이해하기](ARCHITECTURE.md)
- [API 명세 확인](API_SPECIFICATION.md)
- [디자인 시스템 적용](design/DESIGN_SYSTEM.md)
- [배포하기](infrastructure/DEPLOYMENT.md)
