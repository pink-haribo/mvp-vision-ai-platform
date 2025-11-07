# Archived Documentation

**Status**: Outdated / Superseded
**Archive Date**: 2025-11-07

이 폴더에는 현재 구현과 맞지 않아 보관된 문서들이 있습니다.

## ⚠️ 주의사항

**이 문서들을 그대로 따라하지 마세요!**

- 현재 MVP 구현과 다른 내용입니다
- 전체 아키텍처(Temporal, MongoDB, Redis 등)를 기준으로 작성되었습니다
- MVP는 단순화된 스택(K8s + Railway + SQLite/PostgreSQL)을 사용합니다

## 📚 용도

**참고용으로만 사용하세요:**
- 설계 의도 및 배경 이해
- 전체 시스템의 최종 목표 파악
- 향후 확장 시 참고 자료

## 🔍 최신 문서는 어디에?

현재 구현에 맞는 문서는 다음 위치에 있습니다:

- **현재 API**: `docs/251106/01_backend_api_specification.md`
- **K8s 설정**: `docs/k8s/20251107_development_workflow_setup.md`
- **배포 가이드**: `docs/production/RAILWAY_SETUP_GUIDE.md`
- **개발 시작**: Root 폴더의 `GETTING_STARTED.md`, `DEV_WORKFLOW.md`

👉 **전체 가이드**: `docs/OUTDATED_FILES.md` 참고

---

## 📁 Archive Structure

### `/architecture` - 전체 시스템 아키텍처 (MVP에서 미구현)
- **ARCHITECTURE.md** (2025-10-17) - Temporal, MongoDB, Redis 포함 전체 아키텍처
- **CLOUD_GPU_ARCHITECTURE.md** (2025-10-24) - 클라우드 GPU 인프라 (MVP는 로컬/CPU만)
- **CONVERSATION_STATE_ARCHITECTURE.md** (2025-10-21) - LLM 대화 상태 관리 (MVP는 stateless)
- **DOCKER_IMAGE_SEPARATION.md** (2025-10-29) - Docker 이미지 분리 전략

### `/api` - 전체 API 명세 (MVP는 단순화됨)
- **API_SPECIFICATION.md** (2025-10-17) - 모든 서비스 포함 전체 API 명세

### `/development` - 전체 개발 환경 (MVP는 단순화됨)
- **DEVELOPMENT.md** (2025-10-17) - PostgreSQL, MongoDB, Redis, Temporal 등 설정
- **PROJECT_SETUP.md** (2025-10-17) - 초기 프로젝트 설정

### `/design` - 전체 디자인 시스템 (MVP는 단순화됨)
- **DESIGN_SYSTEM.md** (2025-10-17) - 전체 디자인 토큰 및 가이드라인
- **UI_COMPONENTS.md** (2025-10-27) - 상세 컴포넌트 라이브러리

### `/planning` - 구 계획 문서들
- **WEEK1_MODEL_SELECTION_REVISED.md** (2025-10-30)
- **WEEK1_P0_FINAL.md** (2025-10-30)
- **WEEK1_PHASED_IMPLEMENTATION.md** (2025-10-30)
- **IMPLEMENTATION_PRIORITY_ANALYSIS.md** (2025-10-30)
- **MODEL_PLUGIN_VALIDATION_PLAN.md** (2025-10-30)
- **DOCKER_IMPLEMENTATION_PLAN.md** (2025-10-30)

### `/analysis` - 버그 수정 기록 (이미 해결됨)
- **BUG_FIX_SQLALCHEMY_JSON.md** (2025-10-22)
- **DEBUG_INFRASTRUCTURE_ISSUE.md** (2025-10-22)

### `/guide` - 전체 플랫폼 가이드 (MVP에 과도함)
- **README.md**
- **ADD_NEW_MODEL.md**
- **01-executive-summary.md**
- **02-architecture/README.md**
- **03-components/README.md**
- **07-appendices/README.md**

---

## 🗓️ Archive History

**2025-11-07**: Initial archive creation
- Moved 25+ outdated documents
- Created clear separation between MVP and full architecture docs
- Preserved for historical reference and future expansion

---

**For current documentation, see**: `docs/README.md` or `docs/OUTDATED_FILES.md`
