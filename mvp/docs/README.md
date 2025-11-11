# MVP Documentation

**Status**: ✅ MVP 완료 (유지 모드)
**Purpose**: MVP 구현 과정 및 아키텍처 문서 보관

---

## 디렉토리 구조

- **guides/** - 개발 가이드 (GETTING_STARTED, DEV_WORKFLOW 등)
- **architecture/** - MVP 아키텍처 설계
- **datasets/** - 데이터셋 관리 설계
- **llm/** - LLM 통합 구현
- **k8s/** - Kubernetes 마이그레이션
- **planning/** - MVP 계획 및 구조
- **production/** - MVP 프로덕션 배포
- **trainer/**, **training/** - Training 시스템 설계
- **issues/** - 구현 중 발생한 이슈 및 해결
- **251106/** - 특정 시점 문서 스냅샷
- **analysis/** - MVP 분석 문서
- **features/** - 기능 설계
- **scenarios/** - 사용 시나리오
- **_archived/** - 구 버전 아카이브 문서

---

## 주요 문서

### 계획
- [MVP_PLAN.md](planning/MVP_PLAN.md) - MVP 2주 구현 계획
- [MVP_STRUCTURE.md](planning/MVP_STRUCTURE.md) - MVP 폴더 구조
- [MVP_DESIGN_GUIDE.md](planning/MVP_DESIGN_GUIDE.md) - MVP 설계 가이드

### 아키텍처
- [ADAPTER_DESIGN.md](architecture/ADAPTER_DESIGN.md) - Adapter 패턴 설계
- [DATABASE_SCHEMA.md](architecture/DATABASE_SCHEMA.md) - 데이터베이스 스키마

### 데이터셋
- [DATASET_MANAGEMENT_DESIGN.md](datasets/DATASET_MANAGEMENT_DESIGN.md) - 데이터셋 관리 설계
- [DATASET_FORMAT_SUMMARY.md](datasets/DATASET_FORMAT_SUMMARY.md) - 데이터셋 포맷 정리

### LLM
- [INTENT_MAPPING.md](llm/INTENT_MAPPING.md) - 자연어 → 설정 매핑
- [LLM_CONTROL_STRATEGY.md](llm/LLM_CONTROL_STRATEGY.md) - LLM 제어 전략

### 개발 가이드
- [GETTING_STARTED.md](guides/GETTING_STARTED.md) - 시작하기
- [DEV_WORKFLOW.md](guides/DEV_WORKFLOW.md) - 개발 워크플로우

---

## 기타 문서

- [METRIC_COLLECTION_IMPLEMENTATION.md](METRIC_COLLECTION_IMPLEMENTATION.md) - 메트릭 수집 구현
- [OUTDATED_FILES.md](OUTDATED_FILES.md) - 구 버전 파일 목록
- [UX_FLOW.md](UX_FLOW.md) - 사용자 경험 플로우

---

## Platform 문서

MVP 이후 Platform 개발 문서는 [platform/docs/](../../platform/docs/)를 참고하세요.
