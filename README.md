# Vision AI Training Platform

> 자연어로 Vision 모델을 학습하는 AI 플랫폼

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Node](https://img.shields.io/badge/node-20.x-green.svg)](https://nodejs.org/)

## 🎯 개요

Vision AI Training Platform은 개발자가 자연어로 대화하듯 Vision 모델을 학습할 수 있는 플랫폼입니다.

**주요 기능:**
- 🗣️ 자연어 기반 모델 설정
- 🚀 다양한 모델 아키텍처 지원 (timm, HuggingFace, Ultralytics 등)
- 📊 실시간 학습 모니터링
- 🔌 원클릭 추론 API 생성
- 🎨 직관적인 UI/UX

## 🏗️ 아키텍처

```
Frontend (Next.js) ←→ API Gateway ←→ Backend Services
                                      ↓
                               Orchestrator (Temporal)
                                      ↓
                         Training Runner (Kubernetes)
```

[상세 아키텍처 →](docs/ARCHITECTURE.md)

## 🚀 Quick Start

> **처음 시작하시나요?** [GETTING_STARTED.md](GETTING_STARTED.md) - 5분 안에 Training 실행하기

### Prerequisites

```bash
- Docker Desktop 4.26+
- Kind (Kubernetes in Docker)
- kubectl 1.28+
```

**설치 (Windows):**
```powershell
# Kind 설치
winget install -e --id Kubernetes.kind

# kubectl 설치
winget install -e --id Kubernetes.kubectl
```

### 로컬 개발 환경 시작 (한 번에!)

```powershell
# 1. 레포지토리 클론
git clone https://github.com/your-org/vision-platform.git
cd vision-platform

# 2. 개발 환경 시작 (처음 실행 시 10-15분 소요)
.\dev-start.ps1

# 완료! 다음 서비스에 접근 가능:
# - MLflow:     http://localhost:30500
# - Grafana:    http://localhost:30030 (admin/admin)
# - Prometheus: http://localhost:30090
# - MinIO:      http://localhost:30901 (minioadmin/minioadmin)
```

**이후 실행 (빠른 시작):**
```powershell
# 이미지 빌드 스킵 (2-3분 소요)
.\dev-start.ps1 -SkipBuild
```

**상태 확인:**
```powershell
# 현재 환경 상태 확인
.\dev-status.ps1

# 실시간 모니터링
.\dev-status.ps1 -Watch
```

**환경 종료:**
```powershell
# 중지 (데이터 유지)
.\dev-stop.ps1

# 완전 삭제
.\dev-stop.ps1 -DeleteCluster
```

[개발 스크립트 상세 가이드 →](DEV_SCRIPTS.md)

---

### 개발 워크플로우 (Training 코드 수정 시)

**매번 Docker 이미지를 빌드하지 않고 빠르게 개발:**

**1. 로컬 개발 (가장 빠름 ⚡)**
```powershell
# Python으로 직접 실행 (MLflow, MinIO는 K8s 사용)
.\dev-train-local.ps1 -Script mvp/training/train.py

# 코드 수정 → 즉시 실행 → 결과 확인 (초 단위)
```

**2. K8s 테스트 (ConfigMap 주입)**
```powershell
# 이미지 빌드 없이 K8s에서 실행
.\dev-train-k8s.ps1 -Watch

# 코드를 ConfigMap으로 주입 → 기존 이미지 사용 (분 단위)
```

**3. 이미지 빌드 (최종 배포)**
```powershell
# 코드가 안정화되었을 때만
cd mvp/training/docker
.\build.ps1 -Target ultralytics
```

[개발 워크플로우 상세 가이드 →](DEV_WORKFLOW.md)

---

### 수동 설정 (고급 사용자)
cp .env.example .env

# 3. 의존성 설치 & 실행
make dev-up

# Frontend: http://localhost:3000
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

[상세 개발 가이드 →](docs/DEVELOPMENT.md)

## 📦 프로젝트 구조

```
vision-platform/
├── frontend/              # Next.js 애플리케이션
├── backend/
│   ├── api-gateway/      # Kong 설정
│   ├── intent-parser/    # LLM 기반 의도 파싱
│   ├── orchestrator/     # Temporal 워크플로우
│   ├── model-registry/   # 모델 관리
│   ├── data-service/     # 데이터 처리
│   └── vm-controller/    # K8s 클러스터 관리
├── training-runner/      # 학습 실행 환경
├── infrastructure/       # Terraform, K8s manifests
└── docs/                 # 문서
```

## 🛠️ 기술 스택

**Frontend:** Next.js 14, React 18, TailwindCSS, Zustand  
**Backend:** FastAPI, Python 3.11, PostgreSQL, Redis, MongoDB  
**AI/ML:** LangChain, Claude/GPT-4, PyTorch, timm, transformers  
**Orchestration:** Temporal, Celery, Kubernetes  
**Infrastructure:** Docker, Terraform, AWS/GCP

[전체 기술 스택 →](docs/ARCHITECTURE.md#tech-stack)

## 📖 문서

- [아키텍처 설계](docs/ARCHITECTURE.md)
- [API 명세](docs/API_SPECIFICATION.md)
- [개발 가이드](docs/DEVELOPMENT.md)
- [디자인 시스템](docs/design/DESIGN_SYSTEM.md)
- [배포 가이드](docs/infrastructure/DEPLOYMENT.md)

## 🎯 사용 예시

### 자연어로 모델 학습

```
User: "ResNet50으로 고양이 품종 3개 분류하는 모델 만들어줘"

AI: 알겠습니다! 다음 설정으로 진행할게요:
    - Model: ResNet50 (ImageNet pretrained)
    - Task: Image Classification
    - Classes: 3개
    - Epochs: 100 (권장)
    - Batch Size: 32
    
    데이터셋은 어디 있나요?

User: "내 Google Drive의 cat_breeds 폴더"

AI: 학습을 시작합니다! 🚀
```

### 실시간 모니터링

- 📊 Epoch 진행률, Loss/Accuracy 실시간 업데이트
- 💻 GPU/메모리 사용량 모니터링
- 📈 Loss Curve 시각화
- 🔔 학습 완료 알림

### 추론 API 생성

학습 완료 후 원클릭으로 REST API 생성:

```bash
curl -X POST https://api.vision-platform.com/inference/wf_789xyz/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@cat.jpg"

# Response
{
  "predictions": [
    {"class": "페르시안", "confidence": 0.95},
    {"class": "샴", "confidence": 0.03},
    {"class": "러시안블루", "confidence": 0.02}
  ]
}
```

## 🤝 기여하기

기여를 환영합니다! [CONTRIBUTING.md](docs/CONTRIBUTING.md)를 참고해주세요.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📊 로드맵

### Phase 1 (MVP) - Q1 2025
- [x] 기본 UI/UX
- [x] 자연어 파싱 (LLM)
- [x] timm, HuggingFace 모델 지원
- [x] 로컬 학습 실행
- [ ] 기본 텔레메트리

### Phase 2 - Q2 2025
- [ ] Kubernetes 배포
- [ ] Temporal 워크플로우
- [ ] 5+ 모델 프레임워크 지원
- [ ] 분산 학습
- [ ] Advanced 모니터링

### Phase 3 - Q3 2025
- [ ] Auto-scaling
- [ ] Multi-tenancy
- [ ] Enterprise 기능
- [ ] Cost optimization

## 📄 라이선스

MIT License - [LICENSE](LICENSE) 파일 참고

## 📧 문의

- 이슈: [GitHub Issues](https://github.com/your-org/vision-platform/issues)
- 이메일: team@vision-platform.com
- Slack: [Join our community](https://vision-platform.slack.com)

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 도움을 받았습니다:

- [PyTorch](https://pytorch.org/)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Temporal](https://temporal.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)

---

Made with ❤️ by Vision AI Team
