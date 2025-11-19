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
| 6. Model Deployment & Serving | ⬜ 0% | Triton 기반 고도화 배포 계획 완료 | [Phase 6 Details](#phase-6-model-deployment--serving-0) |
| 7. Trainer Marketplace | ⬜ 0% | 계획 완료 | [TRAINER_MARKETPLACE_VISION.md](../planning/TRAINER_MARKETPLACE_VISION.md) |
| 8. E2E Testing | ⬜ 5% | 기본 테스트 완료, 전체 커버리지 필요 | [E2E_TEST_GUIDE.md](../E2E_TEST_GUIDE.md) |

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

### 0.9 Real-time Updates (WebSocket) 🔄 (80%)
현재 polling 방식을 WebSocket으로 전환하여 실시간 업데이트 구현.

**문제점**: 현재 프론트엔드가 3초 간격으로 polling하여 서버 부하 및 지연 발생

**목표**: CLAUDE.md 원칙 준수 - "Real-time updates MUST go through WebSocket, not polling"

**Backend**:
- [x] WebSocket 엔드포인트 구현 (`/api/v1/ws/training`)
- [x] WebSocket Manager 구현 (broadcast, job/session subscription)
- [x] Job 상태 변경 시 WebSocket broadcast
- [x] Export job 상태 변경 시 WebSocket broadcast
- [ ] Redis Pub/Sub 연동 (다중 인스턴스 지원) - 단일 인스턴스에서는 불필요

**Frontend**:
- [x] WebSocket 연결 관리 훅 (`useTrainingMonitor`)
- [x] Training job 상태 실시간 업데이트
- [x] Training metrics 실시간 스트리밍
- [x] Export job 상태 실시간 업데이트
- [~] Inference job 상태 - 단기 작업이므로 polling 유지 (2초 간격, 최대 2분)

**Polling 제거 완료**:
- [x] `ExportJobList.tsx` - 3초 폴링 제거, refreshKey 패턴 적용
- [x] `TrainingPanel` - metrics 폴링 제거, WebSocket onMetrics 콜백 적용
- [x] `MLflowMetricsCharts.tsx` - 5초 폴링 제거, refreshKey 패턴 적용
- [~] `TestInferencePanel` - 단기 작업 polling 유지 (적절한 패턴)

**구현 파일**:
- `platform/backend/app/api/websocket.py` - WebSocket router
- `platform/backend/app/services/websocket_manager.py` - Connection manager
- `platform/frontend/hooks/useTrainingMonitor.ts` - WebSocket hook

**Reference**: [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) - WebSocket Message Types 섹션

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

## Phase 6: Model Deployment & Serving (0%)

Production-grade 모델 서빙 인프라 구현. Export된 모델을 실제 추론 서비스로 배포.

### 6.1 Inference Server Infrastructure ⬜
**목표**: Triton Inference Server 기반 고성능 모델 서빙

- [ ] Inference Server 선택 및 아키텍처 설계
  - [ ] Triton vs ONNX Runtime vs TorchServe 비교 분석
  - [ ] 멀티 모델 서빙 전략
- [ ] Triton Inference Server 배포
  - [ ] K8s Deployment manifest
  - [ ] Model repository 구조 설계 (S3 연동)
  - [ ] 모델 버전 관리 (model versioning)
- [ ] 동적 배칭 (Dynamic Batching)
  - [ ] 배치 크기 최적화
  - [ ] 최대 지연 시간 설정
- [ ] GPU 메모리 관리
  - [ ] 모델별 메모리 할당
  - [ ] 다중 GPU 분배

### 6.2 Platform Endpoint Service ⬜
**목표**: 관리형 추론 API 제공

- [ ] Endpoint Manager 서비스
  - [ ] Deployment → Triton 모델 로딩 자동화
  - [ ] 모델 활성화/비활성화 API
  - [ ] 헬스체크 및 readiness probe
- [ ] API Gateway 연동
  - [ ] Kong/Envoy 설정
  - [ ] Rate limiting
  - [ ] Request routing (deployment_id → model)
- [ ] 인증/인가
  - [ ] API Key 생성 및 관리
  - [ ] Key rotation
  - [ ] Scope/Permission 설정
- [ ] 추론 API 구현
  - [ ] `POST /v1/infer/{deployment_id}`
  - [ ] 이미지 전처리 (base64, URL, multipart)
  - [ ] 결과 후처리 (task_type별 포맷)

### 6.3 Auto-scaling & Resource Management ⬜
**목표**: 트래픽에 따른 자동 스케일링

- [ ] Horizontal Pod Autoscaler (HPA)
  - [ ] CPU/Memory 기반 스케일링
  - [ ] Custom metrics (요청 수, 지연시간)
- [ ] Vertical Pod Autoscaler (VPA)
  - [ ] GPU 메모리 최적화
- [ ] Cluster Autoscaler
  - [ ] 노드 자동 추가/제거
- [ ] 리소스 쿼터 관리
  - [ ] Organization별 GPU 할당량
  - [ ] 동시 요청 수 제한

### 6.4 Monitoring & Observability ⬜
**목표**: 실시간 성능 모니터링 및 알림

- [ ] Prometheus 메트릭 수집
  - [ ] 요청 수 (requests/sec)
  - [ ] 지연 시간 (p50, p95, p99)
  - [ ] 처리량 (throughput)
  - [ ] GPU 사용률
  - [ ] 모델별 메트릭
- [ ] Grafana 대시보드
  - [ ] Deployment 상태 대시보드
  - [ ] 성능 트렌드 시각화
  - [ ] 에러율 모니터링
- [ ] 알림 설정
  - [ ] 지연시간 임계치 초과
  - [ ] 에러율 증가
  - [ ] 리소스 부족

### 6.5 Usage Tracking & Billing ⬜
**목표**: 사용량 추적 및 과금 기반 데이터

- [ ] 요청 로깅
  - [ ] 요청/응답 메타데이터 저장
  - [ ] 처리 시간 기록
- [ ] 사용량 집계
  - [ ] Organization별 일/월 사용량
  - [ ] Deployment별 통계
- [ ] 과금 데이터
  - [ ] GPU 시간 계산
  - [ ] 요청 수 기반 과금
  - [ ] 비용 예측

### 6.6 Edge & Container Deployment ⬜
**목표**: 자체 호스팅 배포 옵션

- [ ] Edge Package 생성
  - [ ] 경량 런타임 번들링
  - [ ] 플랫폼별 최적화 (ARM, x86)
  - [ ] 오프라인 추론 지원
- [ ] Container Image 빌드
  - [ ] Dockerfile 템플릿
  - [ ] Registry push (Docker Hub, GCR, ECR)
  - [ ] 이미지 크기 최적화
- [ ] Runtime Wrappers
  - [ ] Python SDK
  - [ ] C++ SDK
  - [ ] REST API 서버 포함 옵션

### 6.7 CI/CD Pipeline ⬜
**목표**: 자동화된 배포 파이프라인

- [ ] GitHub Actions 워크플로우
  - [ ] 테스트 자동화
  - [ ] 이미지 빌드
  - [ ] K8s 배포
- [ ] GitOps (ArgoCD)
  - [ ] 선언적 배포 관리
  - [ ] 롤백 자동화
- [ ] 카나리 배포
  - [ ] 트래픽 분할
  - [ ] 자동 롤백

**Reference**: [PHASE_3_6_EXPORT_DEPLOYMENT_PLAN.md](../planning/PHASE_3_6_EXPORT_DEPLOYMENT_PLAN.md)

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

## Phase 8: Comprehensive E2E Testing (0%)

E2E 테스트는 프론트엔드가 보내는 모든 요청 조합을 검증해야 함.
핵심 원칙: "API가 동작하는가?"가 아니라 "프론트엔드의 모든 UI 조합이 동작하는가?"

### 8.1 Export Feature Tests ⬜

**8.1.1 ONNX Export Options**
- [ ] Basic export (opset_version only)
- [ ] With dynamic_axes enabled
- [ ] With validation_config
- [ ] Different opset versions (13, 14, 15, 16, 17, 18)
- [ ] With embed_preprocessing

**8.1.2 TensorRT Export Options**
- [ ] Basic export
- [ ] With FP16 precision
- [ ] With INT8 quantization
- [ ] Different max_batch_size values

**8.1.3 CoreML Export Options**
- [ ] Basic export
- [ ] Different minimum_deployment_target (iOS13-17)

**8.1.4 Other Formats**
- [ ] TFLite export
- [ ] TorchScript export
- [ ] OpenVINO export

**8.1.5 Export Download & Deploy Flow**
- [ ] Presigned URL generation
- [ ] Deployment creation (all types)
- [ ] Deployment activate/deactivate

### 8.2 Training Feature Tests ⬜

**8.2.1 Training Job Creation**
- [ ] Basic training config
- [ ] Custom hyperparameters (lr, epochs, batch_size)
- [ ] Different model selections
- [ ] Different task types (detection, segmentation, pose)

**8.2.2 Training Monitoring**
- [ ] Real-time metrics polling/WebSocket
- [ ] Progress tracking
- [ ] Checkpoint saving verification

**8.2.3 Training Completion**
- [ ] Best checkpoint saved
- [ ] Last checkpoint saved
- [ ] MLflow metrics logged

### 8.3 Inference Feature Tests ⬜

**8.3.1 Pretrained Model Inference**
- [x] YOLO pretrained weights
- [ ] Different image formats (jpg, png, webp)
- [ ] Batch inference

**8.3.2 Checkpoint Inference**
- [x] Custom trained checkpoint
- [ ] Best vs Last checkpoint selection

**8.3.3 Inference Results**
- [ ] Result visualization
- [ ] S3 result storage
- [ ] Result download

### 8.4 Dataset Management Tests ⬜

**8.4.1 Dataset Upload**
- [ ] Zip file upload
- [ ] Auto-format detection (YOLO, COCO, ImageFolder)
- [ ] Split ratio configuration

**8.4.2 Dataset Operations**
- [ ] Snapshot creation
- [ ] Dataset listing
- [ ] Dataset deletion

### 8.5 Deployment Feature Tests ⬜

**8.5.1 Platform Endpoint**
- [ ] Endpoint creation
- [ ] API key generation
- [ ] Inference via endpoint

**8.5.2 Other Deployment Types**
- [ ] Edge package creation
- [ ] Container image creation
- [ ] Direct download

### 8.6 API Schema Consistency Tests ⬜

**핵심: Frontend 요청 ↔ Backend 스키마 일치 검증**

- [ ] Export capabilities response (`supported_formats` vs `formats`)
- [ ] Export job request (all fields match schema)
- [ ] Deployment request (all fields match schema)
- [ ] Training job request (all fields match schema)
- [ ] Inference request (all fields match schema)

### 8.7 Error Handling Tests ⬜

- [ ] Invalid training_job_id handling
- [ ] Missing required fields handling
- [ ] Authentication errors
- [ ] File not found errors
- [ ] Network timeout handling

### 8.8 Test Infrastructure ⬜

- [ ] Test fixtures (sample datasets, checkpoints)
- [ ] CI/CD integration
- [ ] Test coverage reporting
- [ ] Automated regression testing

**References**:
- [E2E_TEST_GUIDE.md](../E2E_TEST_GUIDE.md)
- [EXPORT_DEPLOY_E2E_TEST_REPORT.md](./reference/EXPORT_DEPLOY_E2E_TEST_REPORT.md)

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
