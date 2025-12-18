# Vision AI Training Platform - 아키텍처 개요

> 팀 멤버를 위한 플랫폼 구조 설명서

---

## 한 줄 요약

**"자연어로 대화하면 AI 모델이 학습되는 플랫폼"**

```
사용자: "YOLO11n으로 객체 탐지 모델 만들어줘"
  ↓
플랫폼이 알아서 설정 → 학습 실행 → 결과 제공
```

---

## 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         사용자 (웹 브라우저)                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (Next.js)                                             │
│  - 채팅 인터페이스 (자연어 입력)                                   │
│  - 학습 진행 상황 실시간 모니터링                                  │
│  - 데이터셋/프로젝트 관리 UI                                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI)                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Chat API    │  │ Training    │  │ Dataset / Project API   │  │
│  │ (LLM 연동)  │  │ API         │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│              ┌─────────────────────┐                            │
│              │  Temporal Workflow  │  ← 학습 작업 오케스트레이션  │
│              │  (상태 관리, 재시도) │                            │
│              └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Trainers (프레임워크별 독립 서비스)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Ultralytics  │  │    timm      │  │    HuggingFace       │   │
│  │ (YOLO 모델)  │  │ (분류 모델)   │  │    (Transformers)    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Infrastructure                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │
│  │ MinIO   │  │ Redis   │  │ Postgres│  │ ClearML / MLflow    │ │
│  │ (파일)  │  │ (캐시)  │  │  (DB)   │  │ (실험 추적)         │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### 1. Frontend (Next.js 14)
| 역할 | 설명 |
|------|------|
| 채팅 UI | 자연어로 모델 학습 요청 |
| 모니터링 | 실시간 학습 진행률, 메트릭 표시 |
| 관리 | 데이터셋 업로드, 프로젝트 관리 |

### 2. Backend (FastAPI)
| 역할 | 설명 |
|------|------|
| LLM 연동 | 자연어 → 학습 설정으로 변환 (Claude/GPT) |
| API 서버 | REST API + WebSocket (실시간 업데이트) |
| Temporal | 학습 작업 스케줄링, 재시도, 상태 관리 |

### 3. Trainers (학습 실행 서비스)
| Trainer | 지원 모델 |
|---------|----------|
| **ultralytics** | YOLOv11, YOLO-World, SAM2 (Detection, Segmentation, Pose) |
| **timm** | ResNet, EfficientNet, ViT (Image Classification) |
| **huggingface** | Transformers 기반 모델 (예정) |

> **중요**: 각 Trainer는 독립된 Docker 컨테이너로 실행됩니다.
> Backend와 Trainer는 **HTTP API로만** 통신합니다 (파일시스템 공유 X).

---

## 데이터 흐름

```
1. 사용자가 채팅으로 요청
   "coco 데이터셋으로 YOLO11n 학습해줘"
       │
       ▼
2. LLM이 의도 파악 & 설정 생성
   → model: yolo11n, dataset: coco, epochs: 100
       │
       ▼
3. Temporal Workflow 생성
   → Job ID 발급, 상태: PENDING
       │
       ▼
4. Trainer Pod 실행 (Kubernetes)
   → 데이터셋 다운로드 (MinIO)
   → 학습 시작
   → 메트릭 전송 (epoch, loss, mAP)
       │
       ▼
5. 실시간 업데이트
   → WebSocket으로 Frontend에 진행상황 전달
   → ClearML/MLflow에 메트릭 기록
       │
       ▼
6. 학습 완료
   → 모델 체크포인트 저장 (MinIO)
   → 결과 요약 표시
```

---

## 배포 환경 (3-Tier)

| Tier | 환경 | 학습 방식 | 스토리지 |
|------|------|----------|---------|
| **Tier 1** | 로컬 개발 | Subprocess | MinIO (로컬) |
| **Tier 2** | Kind (K8s 개발) | Kubernetes Job | MinIO (클러스터) |
| **Tier 3** | 프로덕션 | Kubernetes Job | S3 / R2 |

> **핵심 원칙**: 동일한 코드, 환경변수만 다름

---

## 기술 스택 요약

| 영역 | 기술 |
|------|------|
| Frontend | Next.js 14, React 18, TailwindCSS, Zustand |
| Backend | FastAPI, Python 3.11, PostgreSQL |
| Orchestration | Temporal Workflow Engine |
| Training | PyTorch, Ultralytics, timm |
| Storage | PostgreSQL (메타데이터), Redis (캐시), MinIO/S3 (파일) |
| Observability | ClearML, MLflow, Prometheus, Grafana |
| Infrastructure | Docker Compose, Kubernetes, Helm |

---

## 주요 설계 원칙

### 1. 완전한 격리 (Isolation)
- Backend와 Trainer는 파일시스템을 공유하지 않음
- 모든 파일은 S3 호환 스토리지(MinIO)를 통해 교환

### 2. API Contract 기반 플러그인
- 새로운 프레임워크 추가 = 새 Trainer 컨테이너 추가
- 표준 인터페이스: `train.py`, `capabilities.json`

### 3. 콜백 기반 상태 업데이트
- Trainer → Backend: HTTP 콜백 (heartbeat, metrics, done)
- JWT 토큰으로 인증

### 4. Temporal 워크플로우
- 자동 재시도 (exponential backoff)
- 타임아웃 & 하트비트 모니터링
- 실패 시 자동 정리 (cleanup)

---

## 폴더 구조

```
platform/
├── backend/          # FastAPI 백엔드
│   ├── app/
│   │   ├── api/      # REST API 엔드포인트
│   │   ├── core/     # 비즈니스 로직
│   │   ├── db/       # 데이터베이스 모델
│   │   ├── services/ # 서비스 레이어
│   │   └── workflows/# Temporal 워크플로우
│   └── ...
│
├── frontend/         # Next.js 프론트엔드
│   ├── app/          # App Router 페이지
│   ├── components/   # React 컴포넌트
│   └── lib/          # 유틸리티
│
├── trainers/         # 프레임워크별 학습 서비스
│   ├── ultralytics/  # YOLO 모델
│   ├── timm/         # 분류 모델
│   └── huggingface/  # Transformers
│
└── infrastructure/   # 인프라 설정
    ├── docker-compose.yml
    └── charts/       # Helm 차트
```

---

## 접속 URL (로컬 개발 환경)

| 서비스 | URL |
|--------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| ClearML | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3200 |
| MinIO Console | http://localhost:9001 |

---

## 관련 문서

- [CLAUDE.md](../CLAUDE.md) - 개발 가이드라인
- [platform/README.md](../platform/README.md) - 상세 설정 방법
- [Quick Start](../platform/QUICK_START.md) - 빠른 시작 가이드

---

*최종 업데이트: 2025-12-18*
