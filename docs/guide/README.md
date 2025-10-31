# Vision AI Training Platform - Team Guide

**Version**: 1.0
**Last Updated**: 2025-10-31
**Status**: MVP Phase

---

## 문서 개요 (Overview)

이 가이드는 Vision AI Training Platform의 전체 아키텍처와 주요 컴포넌트를 설명합니다. 팀원들이 각자의 역할에 맞는 컴포넌트를 이해하고, 통합 지점을 파악하여 효과적으로 협업할 수 있도록 돕습니다.

### 핵심 개념

- **플러그인 아키텍처**: Adapter Pattern을 통한 다중 ML 프레임워크 지원
- **의존성 격리**: Docker 기반 프레임워크별 독립 이미지
- **자연어 인터페이스**: LLM을 활용한 직관적인 모델 학습 설정
- **실시간 모니터링**: MLflow + Prometheus + Grafana 통합

---

## 역할별 추천 읽기 순서

### 프론트엔드 개발자 (Frontend Developer)
1. [Executive Summary](./01-executive-summary.md) - 플랫폼 개요
2. [High-Level Architecture](./02-architecture/high-level-architecture.md) - 전체 구조
3. **[Frontend Component](./03-components/frontend.md)** ⭐ 핵심 문서
4. [Integration Points](./04-integration.md) - Backend와의 통신
5. [Development Workflow](./05-development-workflow.md) - 개발 환경 설정

### 백엔드 개발자 (Backend Developer)
1. [Executive Summary](./01-executive-summary.md) - 플랫폼 개요
2. [Design Patterns](./02-architecture/design-patterns.md) - 핵심 패턴 이해
3. **[Backend API Component](./03-components/backend.md)** ⭐ 핵심 문서
4. [Data Flow](./02-architecture/data-flow.md) - 데이터 흐름
5. [Database & Storage](./03-components/database-storage.md) - DB 스키마
6. [Integration Points](./04-integration.md) - Training과의 연동
7. [Development Workflow](./05-development-workflow.md) - API 테스트

### 모델/학습 개발자 (ML/Training Developer)
1. [Executive Summary](./01-executive-summary.md) - 플랫폼 개요
2. [Design Patterns](./02-architecture/design-patterns.md) - Adapter Pattern 이해
3. **[Training Infrastructure](./03-components/training-infrastructure.md)** ⭐ 핵심 문서
4. [Docker Isolation](./03-components/docker-isolation.md) - 의존성 격리
5. [Data Pipeline](./03-components/data-pipeline.md) - 데이터셋 처리
6. [Development Workflow](./05-development-workflow.md) - 새 모델 추가 방법
7. [Key Files Reference](./07-appendices/key-files-reference.md) - 주요 파일 위치

### 데이터 엔지니어 (Data Engineer)
1. [Executive Summary](./01-executive-summary.md) - 플랫폼 개요
2. **[Data Pipeline](./03-components/data-pipeline.md)** ⭐ 핵심 문서
3. [Database & Storage](./03-components/database-storage.md) - 데이터 스키마
4. [Data Flow](./02-architecture/data-flow.md) - 데이터 흐름
5. [Development Workflow](./05-development-workflow.md) - 데이터셋 추가

### DevOps/인프라 담당자 (DevOps Engineer)
1. [Executive Summary](./01-executive-summary.md) - 플랫폼 개요
2. [High-Level Architecture](./02-architecture/high-level-architecture.md) - 전체 구조
3. **[Docker Isolation](./03-components/docker-isolation.md)** ⭐ 핵심 문서
4. [Deployment & Operations](./06-deployment.md) - 배포 및 모니터링
5. [Integration Points](./04-integration.md) - 컴포넌트 통신
6. [Development Workflow](./05-development-workflow.md) - Docker 빌드

### Product Manager / Team Lead
1. **[Executive Summary](./01-executive-summary.md)** ⭐ 전체 개요
2. [High-Level Architecture](./02-architecture/high-level-architecture.md) - 아키텍처 이해
3. [Design Patterns](./02-architecture/design-patterns.md) - 설계 원칙
4. [ADR](./07-appendices/adr.md) - 주요 의사결정 내역
5. [Development Workflow](./05-development-workflow.md) - 개발 프로세스

---

## 전체 목차 (Table of Contents)

### 1. [Executive Summary](./01-executive-summary.md)
플랫폼의 비전, 핵심 가치, 설계 원칙, 기술 스택 요약

- 1.1 플랫폼 비전과 핵심 가치
- 1.2 주요 설계 원칙
- 1.3 기술 스택 요약
- 1.4 현재 구현 상태 (MVP Phase)

---

### 2. Platform Architecture Overview
플랫폼의 전체 아키텍처와 핵심 설계 패턴

#### [2.1 High-Level Architecture](./02-architecture/high-level-architecture.md)
- End-to-End 플로우
- 3-Tier 아키텍처
- 시스템 컴포넌트 다이어그램

#### [2.2 Core Design Patterns](./02-architecture/design-patterns.md)
- Adapter Pattern (다중 프레임워크 통합)
- Strategy Pattern (실행 모드 선택)
- Observer Pattern (Callbacks 시스템)

#### [2.3 Data Flow](./02-architecture/data-flow.md)
- 학습 작업 생성 플로우
- 학습 실행 플로우
- 메트릭 수집 플로우

---

### 3. Component Deep Dive
각 컴포넌트의 상세 설명 및 구현 가이드

#### [3.1 Frontend](./03-components/frontend.md)
- 기술 스택 (Next.js, TypeScript, Tailwind)
- 주요 화면 구성
- State Management 전략
- Backend 통신 방법
- 주요 파일 위치

#### [3.2 Backend API](./03-components/backend.md)
- 기술 스택 (FastAPI, SQLAlchemy)
- API 구조 및 엔드포인트
- 핵심 서비스 (TrainingManager, ConversationManager, DatasetAnalyzer)
- Database Models
- 주요 파일 위치

#### [3.3 Training Infrastructure](./03-components/training-infrastructure.md)
- Adapter Pattern 아키텍처
- Adapter 구현 가이드
- Training Workflow
- Callbacks System
- Inference System
- 새 프레임워크 추가 방법

#### [3.4 Docker & Dependency Isolation](./03-components/docker-isolation.md)
- Docker Image 분리 전략
- Image 빌드 구조
- Execution Modes (Subprocess vs Docker)
- Volume Mounts & Networking
- 실제 명령어 예시

#### [3.5 Database & Storage](./03-components/database-storage.md)
- Database Schema (ERD)
- MLflow Integration
- File Storage 구조
- 주요 쿼리 패턴

#### [3.6 Data Pipeline](./03-components/data-pipeline.md)
- 지원 Dataset Formats
- Dataset Analyzer
- Data Augmentation
- Format Conversion

---

### 4. [Integration Points](./04-integration.md)
컴포넌트 간 통합 지점 및 통신 방법

- 4.1 Frontend ↔ Backend
- 4.2 Backend ↔ Training
- 4.3 Training ↔ Storage
- 4.4 Adapter ↔ Framework

---

### 5. [Development Workflow](./05-development-workflow.md)
개발 환경 설정 및 작업 프로세스

- 5.1 Local Development Setup
- 5.2 Adding New Model/Framework
- 5.3 Testing Strategy
- 5.4 Debugging Tips

---

### 6. [Deployment & Operations](./06-deployment.md)
배포 및 운영 가이드

- 6.1 Environment Configuration
- 6.2 Monitoring (Prometheus, Grafana)
- 6.3 Troubleshooting

---

### 7. Appendices
추가 참고 자료 및 용어 정의

#### [7.1 Key Files Reference](./07-appendices/key-files-reference.md)
주요 파일 위치 및 역할 종합 표

#### [7.2 Glossary](./07-appendices/glossary.md)
플랫폼 용어 정의

#### [7.3 Architecture Decision Records](./07-appendices/adr.md)
주요 설계 결정 이유 (Why Adapter? Why Docker?)

#### [7.4 Related Documents](./07-appendices/related-docs.md)
기존 문서 링크 및 설명

---

## Quick Reference (빠른 참조)

### 주요 디렉토리
```
mvp-vision-ai-platform/
├── mvp/
│   ├── backend/              # FastAPI Backend
│   │   └── app/
│   │       ├── api/          # API 엔드포인트
│   │       ├── services/     # 비즈니스 로직
│   │       └── utils/        # TrainingManager 등
│   │
│   ├── frontend/             # Next.js Frontend
│   │   ├── components/       # UI 컴포넌트
│   │   └── pages/            # 페이지
│   │
│   ├── training/             # Training Execution
│   │   ├── adapters/         # Framework Adapters
│   │   └── train.py          # 학습 엔트리포인트
│   │
│   └── docker/               # Docker 이미지 정의
│
└── docs/
    ├── architecture/         # 아키텍처 설계 문서
    ├── planning/             # 구현 계획
    └── guide/                # 팀 가이드 (본 문서)
```

### 핵심 파일
| 파일 | 설명 |
|------|------|
| mvp/backend/app/utils/training_manager.py | 학습 실행 관리 (Subprocess/Docker) |
| mvp/training/adapters/base.py | Adapter 인터페이스 및 Callbacks |
| mvp/training/adapters/timm_adapter.py | timm 프레임워크 구현 |
| mvp/training/adapters/ultralytics_adapter.py | YOLO 프레임워크 구현 |
| mvp/docker/build.sh | Docker 이미지 빌드 스크립트 |
| mvp/backend/app/db/models.py | Database 모델 정의 |

### 주요 명령어
```bash
# Backend 실행
cd mvp/backend
source venv/Scripts/activate  # Windows
uvicorn app.main:app --reload --port 8000

# Frontend 실행
cd mvp/frontend
pnpm dev

# Docker 이미지 빌드
cd mvp/docker
./build.sh

# Training 테스트 (로컬)
cd mvp/training
python train.py --framework timm --model resnet18 --task_type image_classification ...
```

---

## 문서 업데이트 정책

- **메이저 업데이트**: 아키텍처 변경 시 (버전 번호 변경)
- **마이너 업데이트**: 새 컴포넌트 추가 시 (Last Updated 날짜 변경)
- **패치**: 오타 수정, 예시 추가 등

---

## 피드백 및 기여

문서 개선 제안이나 오류 발견 시:
1. GitHub Issue 생성
2. Pull Request 제출
3. 팀 채널에 공유

---

**작성자**: Claude Code  
**리뷰 필요**: 모든 팀원  
**문서 관리자**: Project Lead
