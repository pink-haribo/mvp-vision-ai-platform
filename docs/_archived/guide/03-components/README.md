# Component Deep Dive

이 섹션에서는 플랫폼의 각 주요 컴포넌트를 상세히 설명합니다. 각 팀원은 자신의 역할에 맞는 컴포넌트를 중점적으로 읽으시기 바랍니다.

## 목차

### [3.1 Frontend](./frontend.md)
Next.js 기반 프론트엔드 컴포넌트

**주요 내용**:
- 기술 스택 (Next.js 14, TypeScript, Tailwind CSS, Zustand)
- 주요 화면 구성 (Chat, Dashboard, Training Detail)
- State Management 전략
- Backend API 통신 방법
- 주요 파일 위치

**대상 독자**: 프론트엔드 개발자 ⭐

---

### [3.2 Backend API](./backend.md)
FastAPI 기반 백엔드 서비스

**주요 내용**:
- 기술 스택 (FastAPI, SQLAlchemy, Pydantic)
- API 엔드포인트 구조
- 핵심 서비스 (TrainingManager, ConversationManager, DatasetAnalyzer)
- Database Models (ERD)
- 주요 파일 위치

**대상 독자**: 백엔드 개발자 ⭐

---

### [3.3 Training Infrastructure](./training-infrastructure.md)
Adapter Pattern 기반 학습 실행 인프라

**주요 내용**:
- Adapter Pattern 아키텍처
- 기존 Adapters (Timm, Ultralytics, HuggingFace)
- 새 프레임워크 추가 가이드
- Training Workflow (prepare → train → validate → save)
- Callbacks System (MLflow, DB 로깅)
- Inference System

**대상 독자**: 모델/학습 개발자 ⭐

---

### [3.4 Docker & Dependency Isolation](./docker-isolation.md)
Docker 기반 의존성 격리 전략

**주요 내용**:
- Docker Image 분리 전략 (Base + Framework Images)
- Image 빌드 구조 (Dockerfile.base, Dockerfile.timm 등)
- TrainingManager Execution Modes (Subprocess vs Docker)
- Volume Mounts & Networking
- 실제 Docker 명령어 예시

**대상 독자**: 모델/학습 개발자, DevOps 엔지니어 ⭐

---

### [3.5 Database & Storage](./database-storage.md)
데이터베이스 스키마 및 파일 저장소

**주요 내용**:
- Database Schema (training_jobs, training_metrics, validation_results)
- MLflow Integration (실험 추적)
- File Storage 구조 (datasets, outputs, mlflow)
- 주요 쿼리 패턴

**대상 독자**: 백엔드 개발자, 데이터 엔지니어 ⭐

---

### [3.6 Data Pipeline](./data-pipeline.md)
데이터셋 처리 및 전처리 파이프라인

**주요 내용**:
- 지원 Dataset Formats (ImageFolder, COCO, YOLO, VOC)
- Dataset Analyzer (자동 클래스 감지, 통계 분석)
- Data Augmentation 전략
- Format Conversion (향후)

**대상 독자**: 데이터 엔지니어, 모델/학습 개발자 ⭐

---

## 읽기 순서 추천

### 처음 읽는 경우
1. 자신의 역할에 맞는 컴포넌트 (⭐ 표시) 먼저 읽기
2. 해당 컴포넌트와 연결된 다른 컴포넌트 읽기
3. 통합 지점 확인 ([Integration Points](../04-integration.md))

### 기능 개발 시
1. 기능이 속한 컴포넌트 확인
2. 관련 컴포넌트 문서 참조
3. 주요 파일 위치 확인

### 디버깅 시
1. 문제가 발생한 컴포넌트 문서 참조
2. Data Flow 확인 ([Data Flow](../02-architecture/data-flow.md))
3. 로그 및 메트릭 확인 방법 참조

---

[← 돌아가기](../README.md)
