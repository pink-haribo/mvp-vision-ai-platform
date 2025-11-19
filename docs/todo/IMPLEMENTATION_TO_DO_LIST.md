# Implementation To-Do List

Vision AI Training Platform 구현 진행 상황 추적 문서.

**총 진행률**: 95% (209/222 tasks)
**최종 업데이트**: 2025-11-19

---

## Progress Summary

| Phase | Status | Progress | Reference |
|-------|--------|----------|-----------|
| 0. Infrastructure | 🔄 95% | 주요 완료, Backend K8s 배포 대기 | [TIER0_SETUP.md](../development/TIER0_SETUP.md) |
| 1. User & Project | 🔄 75% | Organization/Role 완료, Invitation 진행중 | - |
| 2. Dataset Management | ✅ 85% | Split/Snapshot 완료 | - |
| 3. Training Services | ✅ 88% | Phase 3.1-3.6 완료 | [Phase 3 References](#phase-3-references) |
| 4. Experiment & MLflow | 🔄 86% | 기본 통합 완료, UI 대기 | - |
| 5. Analytics | ⬜ 0% | 미시작 | - |
| 6. Deployment | ⬜ 0% | 미시작 | - |
| 7. Trainer Marketplace | ⬜ 0% | 계획 완료 | [TRAINER_MARKETPLACE_VISION.md](../planning/TRAINER_MARKETPLACE_VISION.md) |

---

## Phase 0: Infrastructure Setup (95%)

### 0.1 Kind Cluster Setup ✅
- [x] Kind config 생성
- [x] Namespace 생성 (platform, training, monitoring, temporal)
- [x] Helm charts 배포 (PostgreSQL, Redis, MinIO, Prometheus, Grafana, Loki, Temporal)

### 0.2 Platform Services 🔄 (60%)
- [x] PostgreSQL, Redis, MinIO, Monitoring Stack 배포 완료
- [ ] Backend ConfigMap/Secret 생성
- [ ] Backend Dockerfile 작성
- [ ] Backend Deployment/Service 배포
- [ ] Frontend Dockerfile 작성
- [ ] Frontend Deployment/Service 배포

### 0.3 MLflow Service ✅
- [x] MLflow K8s manifest 작성
- [x] MLflow 배포 및 UI 접근 확인 (http://localhost:30500)

### 0.4 Observability Stack ✅
- [x] kube-prometheus-stack 배포
- [x] Loki 배포
- [x] Grafana datasource 설정

### 0.5 Temporal Orchestration ✅
- [x] Temporal Server 배포
- [x] Temporal UI 접근 확인 (http://localhost:30233)
- [ ] Backend에 Temporal Worker 코드 추가

### 0.6 Backend Training Mode 🔄
- [x] Subprocess executor 구현 (`training_subprocess.py`)
- [ ] K8s executor 구현 (`k8s_executor.py`)
- [ ] TrainingManager 추상화

### 0.7 Scripts & Documentation ✅
- [x] Helm 배포 스크립트
- [x] 개발 환경 시작 스크립트
- [x] QUICK_START.md

### 0.8 Migration to Tier 2 ⬜
- [ ] Trainer Docker 이미지 빌드
- [ ] K8s Job training 테스트

**Reference**: [TIER0_SETUP.md](../development/TIER0_SETUP.md)

---

## Phase 1: User & Project (75%)

### 1.1 Organization & Role System ✅
- [x] Organization/UserRole 모델
- [x] 마이그레이션
- [x] 회원가입 시 Organization 자동 생성
- [ ] API Permission 체크 적용
- [ ] Role 기반 UI 권한 제어

### 1.2 Experiment Model & MLflow ✅
- [x] Experiment/ExperimentStar/ExperimentNote 모델
- [x] MLflowService 클래스
- [x] Experiment API endpoints
- [ ] TrainingJob-Experiment 자동 연결
- [ ] Frontend Experiment UI

### 1.3 Invitation System 🔄
- [x] Invitation 모델 및 마이그레이션
- [x] Email Service 구현
- [x] Invitation API endpoints
- [x] Password reset 기능
- [ ] Frontend Invitation 페이지
- [ ] Email 검증 페이지

### 1.4 Audit Log System ⬜
- [ ] AuditLog 모델
- [ ] AuditLogger 서비스
- [ ] API 통합

---

## Phase 2: Dataset Management (85%)

### 2.1 Dataset Split Strategy ✅
- [x] 3-Level Priority split 구현
- [x] Split ratio 설정

### 2.2 Snapshot Management ✅
- [x] Snapshot API
- [x] Dataset 버전 추적

### 2.3 Version Management & Download ⬜
- [ ] Dataset versioning
- [ ] Download API

### 2.4 Organization-level Datasets ⬜
- [ ] Organization 공유 데이터셋

### 2.5 Dataset Metrics & Statistics ⬜
- [ ] 데이터셋 통계 API

---

## Phase 3: Training Services (88%)

### 3.1 Trainer Architecture ✅
- [x] Ultralytics trainer 분리
- [x] Convention-based export design
- [x] CLI interface 표준화

**Reference**: [EXPORT_CONVENTION.md](../EXPORT_CONVENTION.md)

### 3.1.1 Checkpoint Management ✅
- [x] best.pt/last.pt 저장
- [x] checkpoint_best_path/checkpoint_last_path 필드 추가
- [x] 프론트엔드 체크포인트 선택 UI

**Reference**: [PHASE_3_1_1_CHECKPOINT_UPDATE.md](../planning/PHASE_3_1_1_CHECKPOINT_UPDATE.md)

### 3.2 Advanced Config Schema ✅
- [x] 동적 config schema 시스템
- [x] Hyperparameter validation
- [x] 트레이너별 config 분리

**Reference**: [ADVANCED_CONFIG_SCHEMA.md](../ADVANCED_CONFIG_SCHEMA.md)

### 3.3 Dual Storage Architecture ✅
- [x] Internal MinIO (9002) / External MinIO (9000) 분리
- [x] Dataset/inference 버킷 분리

### 3.4 Additional Trainers ⬜
- [ ] timm trainer
- [ ] HuggingFace trainer
- [ ] Custom trainer support

### 3.5 Evaluation & Inference CLI ✅
- [x] predict.py CLI
- [x] Pretrained weight 지원
- [x] S3 checkpoint 다운로드

**Reference**: [PHASE_3_5_INFERENCE_PLAN.md](../planning/PHASE_3_5_INFERENCE_PLAN.md)

### 3.5.1 Quick Test Inference ✅
- [x] TestInferencePanel UI
- [x] /test_inference API

### 3.5.2 Inference Job Pattern ✅
- [x] InferenceJob 모델
- [x] Async job execution
- [x] S3 결과 저장
- [x] E2E 테스트 완료

**Reference**: [INFERENCE_JOB_PATTERN.md](../INFERENCE_JOB_PATTERN.md), [E2E_TEST_GUIDE.md](../E2E_TEST_GUIDE.md)

### 3.6 Model Export & Deployment ✅ (100%)
- [x] ExportJob/Deployment 모델
- [x] Export formats (ONNX, TensorRT, CoreML, TFLite)
- [x] Deployment types (Platform Endpoint, Edge, Container, Download)
- [x] Model Capabilities System
- [x] Frontend Export UI (CreateExportModal, DeploymentList)
- [x] Platform Inference Endpoint
- [x] Runtime Wrappers (Python, C++)

**Reference**: [PHASE_3_6_EXPORT_DEPLOYMENT_PLAN.md](../planning/PHASE_3_6_EXPORT_DEPLOYMENT_PLAN.md), [MODEL_CAPABILITIES_SYSTEM.md](../MODEL_CAPABILITIES_SYSTEM.md)

---

## Phase 4: Experiment & MLflow (86%)

- [x] MLflow tracking 통합
- [x] Experiment 모델 및 API
- [x] MLflowMetricsCharts 컴포넌트
- [ ] Frontend Experiment 관리 UI
- [ ] Experiment 비교 기능

---

## Phase 5: Analytics & Monitoring (0%)

- [ ] Usage tracking
- [ ] Cost analytics
- [ ] Performance dashboards

---

## Phase 6: Deployment & Infrastructure (0%)

- [ ] Production deployment 분석
- [ ] CI/CD pipeline
- [ ] Auto-scaling

---

## Phase 7: Trainer Marketplace (0%)

### 7.1 Trainer Validation Infrastructure ⬜
- [ ] Docker image validation
- [ ] API compliance testing
- [ ] Security scanning

### 7.2 Trainer Upload API ⬜
- [ ] Upload endpoint
- [ ] Registry integration
- [ ] Versioning

### 7.3 Frontend Upload UI ⬜
- [ ] Trainer 업로드 폼
- [ ] Validation 결과 표시

### 7.4 Marketplace ⬜
- [ ] Trainer 검색/브라우징
- [ ] Rating/Review
- [ ] Usage analytics

**Reference**: [TRAINER_MARKETPLACE_VISION.md](../planning/TRAINER_MARKETPLACE_VISION.md)

---

## Testing

### E2E Test Status
- [x] Inference Test (Pretrained + Checkpoint)
- [ ] Training Test
- [ ] Export Test
- [ ] Dataset Upload Test

**Reference**: [E2E_TEST_GUIDE.md](../E2E_TEST_GUIDE.md)

---

## Phase 3 References

| Document | Description |
|----------|-------------|
| [EXPORT_CONVENTION.md](../EXPORT_CONVENTION.md) | Export convention for trainers |
| [ADVANCED_CONFIG_SCHEMA.md](../ADVANCED_CONFIG_SCHEMA.md) | Dynamic config schema system |
| [PHASE_3_5_INFERENCE_PLAN.md](../planning/PHASE_3_5_INFERENCE_PLAN.md) | Inference feature design |
| [PHASE_3_6_EXPORT_DEPLOYMENT_PLAN.md](../planning/PHASE_3_6_EXPORT_DEPLOYMENT_PLAN.md) | Export & deployment system |
| [MODEL_CAPABILITIES_SYSTEM.md](../MODEL_CAPABILITIES_SYSTEM.md) | Model capabilities design |
| [INFERENCE_JOB_PATTERN.md](../INFERENCE_JOB_PATTERN.md) | InferenceJob async pattern |
| [E2E_TEST_GUIDE.md](../E2E_TEST_GUIDE.md) | E2E testing principles |

---

## Quick Links

- **Main Checklist**: [MVP_TO_PLATFORM_CHECKLIST.md](../planning/MVP_TO_PLATFORM_CHECKLIST.md) (상세 진행 로그)
- **Migration Guide**: [MVP_TO_PLATFORM_MIGRATION.md](../planning/MVP_TO_PLATFORM_MIGRATION.md)
- **Session Logs**: [CONVERSATION_LOG.md](../CONVERSATION_LOG.md)
