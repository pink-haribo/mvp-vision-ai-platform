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

### Prerequisites

```bash
- Docker Desktop 4.26+
- Node.js 20.x
- Python 3.11+
- kubectl 1.28+
```

### 로컬 실행 (5분 안에)

```bash
# 1. 레포지토리 클론
git clone https://github.com/your-org/vision-platform.git
cd vision-platform

# 2. 환경 변수 설정
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
