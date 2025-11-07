# Outdated Documentation Files

**Last Updated**: 2025-11-07
**Current Implementation**: MVP Phase with K8s + Railway + SQLite/PostgreSQL
**Status**: ✅ Outdated files moved to `docs/_archived/` (2025-11-07)

이 문서는 현재 구현 상황과 맞지 않아 참고용으로만 사용해야 하는 outdated 문서들을 정리합니다.

## 📦 Archive Status

**23개의 outdated 문서들이 `docs/_archived/` 폴더로 이동되었습니다.**

👉 **See**: `docs/_archived/README.md` for complete archive details

---

## 🔴 Critical Outdated (전체 아키텍처 - MVP에서 사용 안함)

**✅ All files below have been moved to `docs/_archived/`**

### Architecture → `_archived/architecture/`
- **~~docs/architecture/ARCHITECTURE.md~~** → **[_archived/architecture/ARCHITECTURE.md](./_archived/architecture/ARCHITECTURE.md)** (2025-10-17)
  - 전체 시스템 아키텍처 (Temporal, MongoDB, PostgreSQL, Redis, MinIO, Kubernetes)
  - MVP에서는 SQLite/PostgreSQL + K8s만 사용
  - 참고용으로만 활용, 실제 구현과 다름

- **~~docs/architecture/CLOUD_GPU_ARCHITECTURE.md~~** → **[_archived/architecture/CLOUD_GPU_ARCHITECTURE.md](./_archived/architecture/CLOUD_GPU_ARCHITECTURE.md)** (2025-10-24)
  - 클라우드 GPU 인프라 설계
  - MVP는 로컬/Railway CPU만 사용
  - 향후 확장 시 참고

- **~~docs/architecture/CONVERSATION_STATE_ARCHITECTURE.md~~** → **[_archived/architecture/CONVERSATION_STATE_ARCHITECTURE.md](./_archived/architecture/CONVERSATION_STATE_ARCHITECTURE.md)** (2025-10-21)
  - LLM 대화 상태 관리 아키텍처
  - MVP는 단순 stateless 대화
  - 실제로는 간단한 LLM 호출만 구현됨

- **~~docs/architecture/DOCKER_IMAGE_SEPARATION.md~~** → **[_archived/architecture/DOCKER_IMAGE_SEPARATION.md](./_archived/architecture/DOCKER_IMAGE_SEPARATION.md)** (2025-10-29)
  - Docker 이미지 분리 전략
  - MVP는 Framework별 Training Service로 분리 완료
  - 현재 구현: timm-service, ultralytics-service, huggingface-service

### API → `_archived/api/`
- **~~docs/api/API_SPECIFICATION.md~~** → **[_archived/api/API_SPECIFICATION.md](./_archived/api/API_SPECIFICATION.md)** (2025-10-17)
  - 전체 API 명세 (모든 서비스 포함)
  - MVP는 Backend + Training Service만
  - 실제 API는 훨씬 단순함
  - 대신 **docs/251106/01_backend_api_specification.md** 참고

### Development → `_archived/development/`
- **~~docs/development/DEVELOPMENT.md~~** → **[_archived/development/DEVELOPMENT.md](./_archived/development/DEVELOPMENT.md)** (2025-10-17)
  - 전체 개발 환경 설정 (PostgreSQL, MongoDB, Redis, Temporal 등)
  - MVP는 SQLite + Kind 클러스터만
  - 대신 **GETTING_STARTED.md** (root) 또는 **DEV_WORKFLOW.md** (root) 참고

- **~~docs/development/PROJECT_SETUP.md~~** → **[_archived/development/PROJECT_SETUP.md](./_archived/development/PROJECT_SETUP.md)** (2025-10-17)
  - 초기 프로젝트 설정
  - MVP 구조와 다름
  - 대신 **docs/planning/MVP_STRUCTURE.md** 참고

---

## 🟡 Partially Outdated (일부 정보만 유효)

**✅ All files below have been moved to `docs/_archived/`**

### Design → `_archived/design/`
- **~~docs/design/DESIGN_SYSTEM.md~~** → **[_archived/design/DESIGN_SYSTEM.md](./_archived/design/DESIGN_SYSTEM.md)** (2025-10-17)
  - 전체 디자인 시스템 (Font, Color, Component)
  - MVP는 단순한 UI만 구현
  - 일부 색상/폰트 정보는 유효

- **~~docs/design/UI_COMPONENTS.md~~** → **[_archived/design/UI_COMPONENTS.md](./_archived/design/UI_COMPONENTS.md)** (2025-10-27)
  - 상세 UI 컴포넌트 명세
  - MVP는 필수 컴포넌트만 구현
  - 참고용

### Planning → `_archived/planning/`
- **~~docs/planning/WEEK1_MODEL_SELECTION_REVISED.md~~** → **[_archived/planning/WEEK1_MODEL_SELECTION_REVISED.md](./_archived/planning/WEEK1_MODEL_SELECTION_REVISED.md)** (2025-10-30)
- **~~docs/planning/WEEK1_P0_FINAL.md~~** → **[_archived/planning/WEEK1_P0_FINAL.md](./_archived/planning/WEEK1_P0_FINAL.md)** (2025-10-30)
- **~~docs/planning/WEEK1_PHASED_IMPLEMENTATION.md~~** → **[_archived/planning/WEEK1_PHASED_IMPLEMENTATION.md](./_archived/planning/WEEK1_PHASED_IMPLEMENTATION.md)** (2025-10-30)
  - Week 1 계획 문서들
  - 이미 지난 계획
  - 역사적 기록용

- **~~docs/planning/IMPLEMENTATION_PRIORITY_ANALYSIS.md~~** → **[_archived/planning/IMPLEMENTATION_PRIORITY_ANALYSIS.md](./_archived/planning/IMPLEMENTATION_PRIORITY_ANALYSIS.md)** (2025-10-30)
- **~~docs/planning/MODEL_PLUGIN_VALIDATION_PLAN.md~~** → **[_archived/planning/MODEL_PLUGIN_VALIDATION_PLAN.md](./_archived/planning/MODEL_PLUGIN_VALIDATION_PLAN.md)** (2025-10-30)
  - 구현 우선순위 분석
  - 일부는 이미 구현됨, 일부는 미구현
  - 현재 상태는 **docs/trainer/IMPLEMENTATION_STATUS.md** 참고

- **~~docs/planning/DOCKER_IMPLEMENTATION_PLAN.md~~** → **[_archived/planning/DOCKER_IMPLEMENTATION_PLAN.md](./_archived/planning/DOCKER_IMPLEMENTATION_PLAN.md)** (2025-10-30)
  - Docker 구현 계획
  - 일부 구현됨 (K8s Job)
  - 실제 구현은 다소 다름

### Guide → `_archived/guide/`
- **~~docs/guide/~~** (전체 폴더, 2025-10-31) → **[_archived/guide/](./_archived/guide/)**
  - 01-executive-summary.md
  - 02-architecture/README.md
  - 03-components/README.md
  - 07-appendices/README.md
  - ADD_NEW_MODEL.md
  - README.md
  - 전체 플랫폼 가이드
  - MVP 단계에서는 과도하게 상세함
  - 대신 **docs/251106/** 폴더의 문서들 참고

### Analysis → `_archived/analysis/`
- **~~docs/analysis/BUG_FIX_SQLALCHEMY_JSON.md~~** → **[_archived/analysis/BUG_FIX_SQLALCHEMY_JSON.md](./_archived/analysis/BUG_FIX_SQLALCHEMY_JSON.md)** (2025-10-22)
- **~~docs/analysis/DEBUG_INFRASTRUCTURE_ISSUE.md~~** → **[_archived/analysis/DEBUG_INFRASTRUCTURE_ISSUE.md](./_archived/analysis/DEBUG_INFRASTRUCTURE_ISSUE.md)** (2025-10-22)
  - 특정 버그 수정 기록
  - 역사적 기록용, 이미 해결됨

---

## ✅ Up-to-date (현재 유효한 문서)

### K8s (최신)
- **docs/k8s/20251107_development_workflow_setup.md** (2025-11-07)
- **docs/k8s/20251107_kind_vs_minikube_production_continuity.md** (2025-11-07)
- **docs/k8s/20251106_kubernetes_job_migration_plan.md** (2025-11-07)
- **docs/k8s/K8S_TRAINING_FAQ.md** (2025-11-07)

### 251106 (최신 구현 명세)
- **docs/251106/01_backend_api_specification.md**
- **docs/251106/02_sdk_adapter_pattern.md**
- **docs/251106/03_config_schema_guide.md**
- **docs/251106/04_user_flow_scenarios.md**
- **docs/251106/05_annotation_system.md**
- **docs/251106/06_model_developer_guide.md**

### Training (최신)
- **docs/training/20251105_checkpoint_management_and_r2_upload_policy.md**
- **docs/training/20251105_inference_api_training_service_integration.md**
- **docs/training/20251105_r2_pretrained_weights_management.md**
- **docs/training/20251105_timm_implementation_plan.md**
- **docs/training/20251105_training_framework_implementation_guide.md**

### Datasets (최신)
- **docs/datasets/20251105_093103_dataset_split_strategy.md**
- **docs/datasets/CURRENT_STATUS.md** (2025-11-04)
- **docs/datasets/DATASET_MANAGEMENT_DESIGN.md** (2025-11-04)
- **docs/datasets/IMPLEMENTATION_PLAN.md** (2025-11-04)

### Production (최신)
- **docs/production/RAILWAY_SETUP_GUIDE.md** (2025-11-02)
- **docs/production/FRAMEWORK_ISOLATION_DEPLOYMENT.md** (2025-11-03)
- **docs/production/DYNAMIC_MODEL_REGISTRATION.md** (2025-11-03)
- **docs/production/CLOUDFLARE_R2_SETUP.md** (2025-11-03)
- **docs/production/RAILWAY_MLFLOW_SETUP.md** (2025-11-03)
- **docs/production/PLATFORM_SDK_LOGGER_INTEGRATION.md** (2025-11-03)
- **docs/production/TRAINING_VISIBILITY_ARCHITECTURE.md** (2025-11-03)

### Scenarios (최신)
- **docs/scenarios/** (전체, 2025-11-03)

### LLM (최신)
- **docs/llm/** (전체, 2025-11-02)

### Planning (일부 최신)
- **docs/planning/MVP_PLAN.md** (2025-10-17) - 여전히 유효
- **docs/planning/MVP_STRUCTURE.md** (2025-10-17) - 여전히 유효
- **docs/planning/MVP_DESIGN_GUIDE.md** (2025-10-17) - 여전히 유효
- **docs/planning/VALIDATION_SYSTEM_IMPLEMENTATION.md** (2025-10-29) - 구현 완료
- **docs/planning/TEST_INFERENCE_IMPLEMENTATION_PLAN.md** (2025-10-29) - 구현 완료

---

## 📌 Usage Guidelines

### 새로운 개발자를 위한 추천 순서

1. **시작**:
   - `GETTING_STARTED.md` (root)
   - `DEV_WORKFLOW.md` (root)
   - `QUICK_DEV_GUIDE.md` (root)

2. **MVP 이해**:
   - `docs/planning/MVP_PLAN.md`
   - `docs/planning/MVP_STRUCTURE.md`
   - `docs/planning/MVP_DESIGN_GUIDE.md`

3. **현재 구현**:
   - `docs/251106/` (전체)
   - `docs/k8s/` (전체)
   - `docs/training/` (전체)

4. **Production 배포**:
   - `docs/production/RAILWAY_SETUP_GUIDE.md`
   - `docs/production/FRAMEWORK_ISOLATION_DEPLOYMENT.md`

### Outdated 문서 활용법
- ❌ 그대로 따라하지 마세요 (구현과 다름)
- ✅ 설계 의도, 배경 이해에 활용
- ✅ 향후 확장 시 참고 자료로 활용
- ✅ 전체 아키텍처 이해 (최종 목표)

---

## 🔄 Next Steps

향후 이 outdated 문서들을 정리할 계획:
1. 명확히 outdated된 문서는 `docs/_archived/` 폴더로 이동
2. 일부 유효한 문서는 현재 구현에 맞게 업데이트
3. 새로운 통합 문서 작성 (MVP 중심)
