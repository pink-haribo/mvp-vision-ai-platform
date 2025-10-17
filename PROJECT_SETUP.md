# 프로젝트 초기 설정 가이드

이 문서는 프로젝트의 폴더 구조를 생성하고 초기 파일들을 설정하는 가이드입니다.

## 목차
- [프로젝트 구조 개요](#프로젝트-구조-개요)
- [폴더 구조 생성](#폴더-구조-생성)
- [Frontend 설정](#frontend-설정)
- [Backend 서비스 설정](#backend-서비스-설정)
- [Infrastructure 설정](#infrastructure-설정)
- [Scripts 설정](#scripts-설정)
- [초기 파일 생성](#초기-파일-생성)

---

## 프로젝트 구조 개요

```
vision-platform/
├── .github/                    # GitHub Actions, PR templates
├── frontend/                   # Next.js 애플리케이션
├── backend/                    # Backend 마이크로서비스
│   ├── intent-parser/         # LLM 기반 의도 파싱 서비스
│   ├── orchestrator/          # Temporal 워크플로우 관리
│   ├── model-registry/        # 모델 메타데이터 관리
│   ├── data-service/          # 데이터셋 처리
│   ├── vm-controller/         # Kubernetes 관리
│   └── telemetry/             # 메트릭 수집
├── training-runner/           # 학습 실행 환경 (Docker)
├── infrastructure/            # IaC, K8s manifests
├── scripts/                   # 유틸리티 스크립트
├── docs/                      # 추가 문서
├── config/                    # 설정 파일
└── tests/                     # E2E 테스트
```

---

## 폴더 구조 생성

### 1. 루트 레벨 디렉토리

```bash
# 프로젝트 루트에서 실행
mkdir -p .github/workflows
mkdir -p .github/ISSUE_TEMPLATE
mkdir -p .github/PULL_REQUEST_TEMPLATE

mkdir -p frontend
mkdir -p backend
mkdir -p training-runner
mkdir -p infrastructure
mkdir -p scripts
mkdir -p docs
mkdir -p config
mkdir -p tests
```

### 2. Backend 서비스 디렉토리

```bash
# Backend 서비스 구조 생성
cd backend

mkdir -p intent-parser/{app,tests,alembic,scripts}
mkdir -p orchestrator/{app,tests,alembic,scripts}
mkdir -p model-registry/{app,tests,alembic,scripts}
mkdir -p data-service/{app,tests,alembic,scripts}
mkdir -p vm-controller/{app,tests,alembic,scripts}
mkdir -p telemetry/{app,tests,alembic,scripts}

cd ..
```

### 3. Infrastructure 디렉토리

```bash
mkdir -p infrastructure/{terraform,kubernetes,helm,ansible}
mkdir -p infrastructure/kubernetes/{base,overlays}
mkdir -p infrastructure/kubernetes/overlays/{development,staging,production}
```

### 4. Config 디렉토리

```bash
mkdir -p config/{prometheus,grafana,temporal,nginx}
mkdir -p config/grafana/{dashboards,datasources}
```

---

## Frontend 설정

### 1. Next.js 프로젝트 초기화

```bash
cd frontend

# pnpm 사용 (권장)
pnpm create next-app@latest . --typescript --tailwind --app --use-pnpm

# 또는 수동 설정
pnpm init
pnpm add next@latest react@latest react-dom@latest
pnpm add -D typescript @types/react @types/node
pnpm add -D tailwindcss postcss autoprefixer
pnpm add -D eslint eslint-config-next
```

### 2. Frontend 폴더 구조

```bash
cd frontend

mkdir -p app/{api,\(auth\),\(dashboard\)}
mkdir -p app/\(auth\)/{login,register}
mkdir -p app/\(dashboard\)/{projects,models,datasets,settings}

mkdir -p components/{ui,layout,features}
mkdir -p lib
mkdir -p hooks
mkdir -p types
mkdir -p styles
mkdir -p public/{images,fonts}
mkdir -p tests/{unit,integration,e2e}
```

### 3. 필수 패키지 설치

```bash
cd frontend

# UI 라이브러리
pnpm add class-variance-authority clsx tailwind-merge
pnpm add @radix-ui/react-slot
pnpm add lucide-react

# 상태 관리
pnpm add zustand

# 데이터 페칭
pnpm add @tanstack/react-query

# WebSocket
pnpm add socket.io-client

# 폼 관리
pnpm add react-hook-form @hookform/resolvers zod

# 차트
pnpm add recharts

# 날짜
pnpm add date-fns

# 테스트
pnpm add -D jest @testing-library/react @testing-library/jest-dom
pnpm add -D @playwright/test
```

### 4. Frontend 설정 파일

**tsconfig.json:**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

**tailwind.config.js:**
```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Design system colors
      },
      fontFamily: {
        sans: ['SUIT', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
```

**package.json scripts:**
```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "type-check": "tsc --noEmit",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:e2e": "playwright test",
    "test:e2e:ui": "playwright test --ui",
    "format": "prettier --write \"**/*.{ts,tsx,json,md}\""
  }
}
```

---

## Backend 서비스 설정

각 서비스는 동일한 구조를 따릅니다. Intent Parser를 예시로 설명합니다.

### 1. Poetry 프로젝트 초기화

```bash
cd backend/intent-parser

# Poetry 초기화
poetry init --name intent-parser --python "^3.11"

# 기본 패키지 설치
poetry add fastapi uvicorn[standard] sqlalchemy alembic psycopg2-binary
poetry add pydantic pydantic-settings
poetry add langchain langchain-anthropic langchain-openai
poetry add python-jose[cryptography] passlib[bcrypt]
poetry add python-multipart
poetry add redis pymongo boto3

# 개발 의존성
poetry add -D pytest pytest-asyncio pytest-cov
poetry add -D black isort flake8 mypy
poetry add -D httpx  # API 테스트용
```

### 2. Backend 서비스 폴더 구조

```bash
cd backend/intent-parser

mkdir -p app/{api,core,models,schemas,services,utils}
mkdir -p app/api/{v1,dependencies}
mkdir -p app/api/v1/endpoints
mkdir -p tests/{unit,integration,e2e}
mkdir -p alembic/versions
mkdir -p scripts
```

### 3. 기본 파일 구조

**app/main.py:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title="Intent Parser Service",
    description="Natural language to training config parser",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

**app/core/config.py:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001

    # Database
    DATABASE_URL: str

    # LLM
    ANTHROPIC_API_KEY: str
    LLM_MODEL: str = "claude-3-5-sonnet-20241022"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"

settings = Settings()
```

**pyproject.toml:**
```toml
[tool.poetry]
name = "intent-parser"
version = "0.1.0"
description = "Intent Parser Service"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.108.0"
uvicorn = {extras = ["standard"], version = "^0.25.0"}
sqlalchemy = "^2.0.0"
alembic = "^1.13.0"
langchain = "^0.1.0"
pydantic = "^2.5.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
black = "^23.12.0"
isort = "^5.13.0"
flake8 = "^6.1.0"
mypy = "^1.7.0"

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
ignore_missing_imports = true

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 4. 다른 Backend 서비스 생성

```bash
# Orchestrator
cd backend/orchestrator
poetry init --name orchestrator --python "^3.11"
# (동일한 설정 반복)

# Model Registry
cd backend/model-registry
poetry init --name model-registry --python "^3.11"
# (동일한 설정 반복)

# 나머지 서비스도 동일하게...
```

---

## Infrastructure 설정

### 1. Kubernetes Manifests

```bash
cd infrastructure/kubernetes/base

# Namespace
cat > namespace.yaml <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: vision-platform
EOF

# Example deployment
cat > deployment-template.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: intent-parser
  namespace: vision-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: intent-parser
  template:
    metadata:
      labels:
        app: intent-parser
    spec:
      containers:
      - name: intent-parser
        image: vision-platform/intent-parser:latest
        ports:
        - containerPort: 8001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
EOF
```

### 2. Terraform 구조

```bash
cd infrastructure/terraform

mkdir -p modules/{vpc,eks,rds,s3}

# Main Terraform file
cat > main.tf <<EOF
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "vision-platform-terraform-state"
    key    = "terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source = "./modules/vpc"
  # ...
}
EOF
```

---

## Scripts 설정

### 1. 유틸리티 스크립트

```bash
cd scripts

# Database seed script
cat > seed_data.py <<EOF
#!/usr/bin/env python3
"""Seed database with sample data"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# ... seed logic

if __name__ == "__main__":
    asyncio.run(seed_database())
EOF

chmod +x seed_data.py

# MongoDB indexes
cat > init_mongodb.py <<EOF
#!/usr/bin/env python3
"""Initialize MongoDB indexes"""

from pymongo import MongoClient, ASCENDING, DESCENDING

def create_indexes():
    client = MongoClient(MONGODB_URL)
    db = client.vision_platform

    # chat_sessions
    db.chat_sessions.create_index([("sessionId", ASCENDING)], unique=True)
    # ...

if __name__ == "__main__":
    create_indexes()
EOF

chmod +x init_mongodb.py
```

### 2. 개발 스크립트

```bash
# 전체 설정 스크립트
cat > scripts/setup-dev.sh <<'EOF'
#!/bin/bash
set -e

echo "🚀 Setting up Vision Platform development environment..."

# 환경 변수 확인
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and fill in your API keys!"
fi

# Infrastructure 시작
echo "Starting infrastructure..."
make infra-up

# Frontend 설정
echo "Setting up frontend..."
cd frontend
pnpm install
cd ..

# Backend 설정
echo "Setting up backend services..."
make backend-install-all

# Database 마이그레이션
echo "Running migrations..."
make db-migrate

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Run 'make frontend-dev' to start frontend"
echo "  3. Run 'make backend-<service>' to start backend services"
EOF

chmod +x scripts/setup-dev.sh
```

---

## 초기 파일 생성

### 1. GitHub Actions

```bash
mkdir -p .github/workflows

cat > .github/workflows/ci.yml <<EOF
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  frontend-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
        with:
          version: 8
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'
          cache-dependency-path: frontend/pnpm-lock.yaml

      - name: Install dependencies
        working-directory: frontend
        run: pnpm install

      - name: Lint
        working-directory: frontend
        run: pnpm lint

      - name: Type check
        working-directory: frontend
        run: pnpm type-check

      - name: Test
        working-directory: frontend
        run: pnpm test

  backend-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [intent-parser, orchestrator, model-registry]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Poetry
        run: pipx install poetry

      - name: Install dependencies
        working-directory: backend/${{ matrix.service }}
        run: poetry install

      - name: Lint
        working-directory: backend/${{ matrix.service }}
        run: |
          poetry run black --check app tests
          poetry run flake8 app tests

      - name: Test
        working-directory: backend/${{ matrix.service }}
        run: poetry run pytest -v
EOF
```

### 2. .gitignore

```bash
cat > .gitignore <<'EOF'
# Environment
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# Node.js
node_modules/
.pnpm-store/
.next/
out/
.npm
.yarn-integrity

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary
tmp/
temp/
*.tmp

# Docker
.dockerignore

# Terraform
*.tfstate
*.tfstate.*
.terraform/

# Alembic
alembic/versions/*.py
!alembic/versions/__init__.py
EOF
```

---

## 실행 순서

프로젝트 생성 후 다음 순서대로 진행하세요:

```bash
# 1. 폴더 구조 생성
./scripts/setup-project-structure.sh

# 2. Frontend 설정
cd frontend
pnpm create next-app@latest . --typescript --tailwind --app
pnpm install

# 3. Backend 서비스 설정
cd backend/intent-parser
poetry init && poetry install

# 4. Infrastructure 시작
make infra-up

# 5. Database 설정
make db-migrate
make mongo-indexes
make db-seed

# 6. 개발 서버 실행
make frontend-dev
make backend-intent-parser
```

---

## 다음 단계

- [개발 환경 설정](DEVELOPMENT.md)
- [기여 가이드](CONTRIBUTING.md)
- [아키텍처 문서](ARCHITECTURE.md)
