# Docs Cleanup Plan

**Date**: 2025-01-11
**Branch**: `repo-cleanup`
**Goal**: MVP와 Platform 문서 분리 및 정리

---

## 현재 상황

### 문서 위치
1. **docs/** (루트) - MVP + Platform 문서 혼재
2. **mvp/docs/** - MVP 개발 가이드만 존재
3. **platform/docs/** - Platform 설계 문서만 존재

### 문제점
- MVP 관련 문서가 루트 docs/에 산재
- Platform 개발 시 MVP 문서가 방해됨
- 문서 역할이 불명확 (어떤게 현재 유효한 문서인지?)

---

## 분류 기준

### MVP 문서 (mvp/docs로 이동)
- MVP 구현 과정에서 작성된 모든 문서
- MVP 아키텍처, 구현 계획, 이슈 등
- **특징**: 과거 시제, MVP 버전 특정, 구현 완료됨

### Platform 문서 (platform/docs 유지)
- Platform 아키텍처 설계 문서
- Platform 개발 가이드
- **특징**: 현재 진행형, 프로덕션 목표

### 공용 문서 (docs/ 루트 유지)
- 프로젝트 전체 히스토리 (CONVERSATION_LOG.md)
- 문서 인덱스 (README.md)
- 아카이브 (_archived/)

---

## 문서 분류

### A. MVP 문서 → mvp/docs/

#### 1. 날짜별 문서
```
docs/251106/ → mvp/docs/251106/
```
- 6개 파일 (backend API, SDK, config, user flow, annotation, model developer guide)
- **이유**: MVP 구현 시점의 문서

#### 2. MVP 분석 문서
```
docs/analysis/ → mvp/docs/analysis/
```
- ANALYSIS.md, BREAKTHROUGH.md
- **이유**: MVP 구현 과정의 분석

#### 3. MVP 아키텍처
```
docs/architecture/ → mvp/docs/architecture/
```
- ADAPTER_DESIGN.md, DATABASE_SCHEMA.md
- **이유**: MVP 아키텍처 (platform/docs/architecture와 다름)

#### 4. 데이터셋 설계
```
docs/datasets/ → mvp/docs/datasets/
```
- 12개 파일 (dataset management, format, storage, UI plan 등)
- **이유**: MVP 데이터셋 구현 설계

#### 5. 기능 설계
```
docs/features/ → mvp/docs/features/
```
- DATASET_SOURCES_DESIGN.md
- **이유**: MVP 기능 설계

#### 6. 이슈 트래킹
```
docs/issues/ → mvp/docs/issues/
```
- yolo_validation_metrics.md
- **이유**: MVP 구현 중 이슈

#### 7. Kubernetes 마이그레이션
```
docs/k8s/ → mvp/docs/k8s/
```
- 5개 파일 (job migration, workflow, config schema 등)
- **이유**: MVP K8s 마이그레이션 과정

#### 8. LLM 구현
```
docs/llm/ → mvp/docs/llm/
```
- 9개 파일 (Gemini, intent mapping, MCP, phase progress 등)
- **이유**: MVP LLM 구현 과정

#### 9. 계획 문서
```
docs/planning/ → mvp/docs/planning/
```
- MVP_PLAN.md, MVP_STRUCTURE.md, MVP_DESIGN_GUIDE.md 등
- **이유**: MVP 계획 문서

#### 10. 프로덕션 배포 (MVP)
```
docs/production/ → mvp/docs/production/
```
- MVP 프로덕션 배포 관련
- **이유**: MVP 프로덕션 경험

#### 11. 시나리오
```
docs/scenarios/ → mvp/docs/scenarios/
```
- MVP 사용 시나리오
- **이유**: MVP 기능 설명

#### 12. Trainer 설계
```
docs/trainer/ → mvp/docs/trainer/
```
- MVP trainer 구현 설계
- **이유**: MVP trainer 아키텍처

#### 13. Training 설계
```
docs/training/ → mvp/docs/training/
```
- MVP training 구현 설계
- **이유**: MVP training 아키텍처

#### 14. 개별 MVP 문서
```
docs/METRIC_COLLECTION_IMPLEMENTATION.md → mvp/docs/
docs/OUTDATED_FILES.md → mvp/docs/
```

### B. Platform 문서 (이미 platform/docs에 존재)

#### platform/docs/architecture/
- BACKEND_DESIGN.md
- DATASET_SPLIT_STRATEGY.md
- DATASET_STORAGE_STRATEGY.md
- ERROR_HANDLING_DESIGN.md
- INTEGRATION_FAILURE_HANDLING.md
- OPERATIONS_RUNBOOK.md
- TRAINER_DESIGN.md
- 등 17개 파일

#### platform/docs/development/
- 3_TIER_DEVELOPMENT.md
- TEMPORAL_INTEGRATION.md

#### platform/docs/migration/
- MVP_TO_PLATFORM.md

### C. 공용 문서 (docs/ 루트 유지)

```
docs/
├── _archived/               # 이미 아카이브된 문서들
├── reviews/                 # 설계 리뷰 (이미 이동 완료)
├── CONVERSATION_LOG.md      # 프로젝트 전체 대화 로그
└── README.md                # 문서 인덱스 (업데이트 필요)
```

---

## 정리 후 구조

```
프로젝트/
├── docs/                            # 공용 문서
│   ├── _archived/                   # 아카이브
│   ├── reviews/                     # 설계 리뷰
│   ├── CONVERSATION_LOG.md          # 대화 로그
│   └── README.md                    # 문서 인덱스 (전체)
│
├── mvp/
│   └── docs/                        # ⭐ MVP 전체 문서
│       ├── 251106/                  # ⭐ MOVED
│       ├── analysis/                # ⭐ MOVED
│       ├── architecture/            # ⭐ MOVED (MVP 아키텍처)
│       ├── datasets/                # ⭐ MOVED
│       ├── features/                # ⭐ MOVED
│       ├── guides/                  # 이미 존재 (개발 가이드)
│       ├── issues/                  # ⭐ MOVED
│       ├── k8s/                     # ⭐ MOVED
│       ├── llm/                     # ⭐ MOVED
│       ├── planning/                # ⭐ MOVED
│       ├── production/              # ⭐ MOVED
│       ├── scenarios/               # ⭐ MOVED
│       ├── trainer/                 # ⭐ MOVED
│       ├── training/                # ⭐ MOVED
│       ├── METRIC_COLLECTION_IMPLEMENTATION.md  # ⭐ MOVED
│       ├── OUTDATED_FILES.md        # ⭐ MOVED
│       └── README.md                # ⭐ NEW (MVP 문서 인덱스)
│
└── platform/
    └── docs/                        # Platform 문서
        ├── architecture/            # Platform 아키텍처
        ├── development/             # Platform 개발 가이드
        ├── migration/               # MVP → Platform 마이그레이션
        └── README.md                # Platform 문서 인덱스
```

---

## 실행 계획

### Step 1: MVP 문서 이동

```bash
# MVP 디렉토리 준비
mkdir -p mvp/docs

# 문서 이동
mv docs/251106 mvp/docs/
mv docs/analysis mvp/docs/
mv docs/architecture mvp/docs/
mv docs/datasets mvp/docs/
mv docs/features mvp/docs/
mv docs/issues mvp/docs/
mv docs/k8s mvp/docs/
mv docs/llm mvp/docs/
mv docs/planning mvp/docs/
mv docs/production mvp/docs/
mv docs/scenarios mvp/docs/
mv docs/trainer mvp/docs/
mv docs/training mvp/docs/

# 개별 파일 이동
mv docs/METRIC_COLLECTION_IMPLEMENTATION.md mvp/docs/
mv docs/OUTDATED_FILES.md mvp/docs/
```

### Step 2: MVP 문서 README 생성

```bash
# mvp/docs/README.md 생성
cat > mvp/docs/README.md << 'EOF'
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
- [DATASET_MANAGEMENT_DESIGN.md](datasets/DATASET_MANAGEMENT_DESIGN.md)
- [DATASET_FORMAT_SUMMARY.md](datasets/DATASET_FORMAT_SUMMARY.md)

### LLM
- [INTENT_MAPPING.md](llm/INTENT_MAPPING.md) - 자연어 → 설정 매핑
- [LLM_CONTROL_STRATEGY.md](llm/LLM_CONTROL_STRATEGY.md)

---

## Platform 문서

MVP 이후 Platform 개발 문서는 [platform/docs/](../../platform/docs/)를 참고하세요.
EOF
```

### Step 3: docs/README.md 업데이트

```bash
cat > docs/README.md << 'EOF'
# Documentation Index

Vision AI Training Platform 전체 문서 인덱스입니다.

---

## 문서 위치

### 📁 MVP 문서
**위치**: [mvp/docs/](../mvp/docs/)
**상태**: ✅ MVP 완료 (유지 모드)
**내용**: MVP 구현 과정, 아키텍처, 계획, 이슈 등

### 📁 Platform 문서
**위치**: [platform/docs/](../platform/docs/)
**상태**: ⏳ Platform 개발 진행 중
**내용**: Platform 아키텍처, 개발 가이드, 마이그레이션

### 📁 공용 문서
**위치**: `docs/` (현재 디렉토리)
- [CONVERSATION_LOG.md](CONVERSATION_LOG.md) - 프로젝트 대화 로그
- [reviews/](reviews/) - 설계 리뷰 문서
- [_archived/](_archived/) - 아카이브된 문서

---

## Quick Links

### MVP
- [MVP 계획](../mvp/docs/planning/MVP_PLAN.md)
- [MVP 구조](../mvp/docs/planning/MVP_STRUCTURE.md)
- [MVP 아키텍처](../mvp/docs/architecture/)

### Platform
- [Platform 아키텍처](../platform/docs/architecture/)
- [3-Tier 개발](../platform/docs/development/3_TIER_DEVELOPMENT.md)
- [에러 핸들링](../platform/docs/architecture/ERROR_HANDLING_DESIGN.md)
- [운영 가이드](../platform/docs/architecture/OPERATIONS_RUNBOOK.md)

### 리뷰
- [최종 설계 리뷰](reviews/FINAL_DESIGN_REVIEW_2025-01-11.md)

---

**Last Updated**: 2025-01-11
EOF
```

### Step 4: 검증 및 커밋

```bash
# 구조 확인
ls -la docs/
ls -la mvp/docs/
ls -la platform/docs/

# Git 상태 확인
git status

# 커밋
git add -A
git commit -m "docs: separate MVP and Platform documentation

Move MVP-related documentation to mvp/docs/:
- 251106/, analysis/, architecture/ (MVP)
- datasets/, features/, issues/
- k8s/, llm/, planning/
- production/, scenarios/, trainer/, training/
- METRIC_COLLECTION_IMPLEMENTATION.md, OUTDATED_FILES.md

Keep in docs/ root:
- CONVERSATION_LOG.md (project history)
- reviews/ (design reviews)
- _archived/ (already archived)
- README.md (updated documentation index)

Platform documentation remains in platform/docs/.

This separation clarifies:
- MVP docs (completed, maintenance mode)
- Platform docs (active development)
"
```

---

## 예상 효과

### Before
```
docs/ - MVP + Platform 문서 혼재 (17개 디렉토리)
mvp/docs/ - 개발 가이드만 (1개 디렉토리)
platform/docs/ - Platform 문서만 (3개 디렉토리)
```

### After
```
docs/ - 공용 문서만 (3개)
  ├── _archived/
  ├── reviews/
  ├── CONVERSATION_LOG.md
  └── README.md

mvp/docs/ - MVP 전체 문서 (15개 디렉토리)
  ├── guides/ (이미 존재)
  ├── 251106/, analysis/, architecture/
  ├── datasets/, features/, issues/
  ├── k8s/, llm/, planning/
  ├── production/, scenarios/, trainer/, training/
  └── README.md (NEW)

platform/docs/ - Platform 문서 (3개 디렉토리)
  ├── architecture/
  ├── development/
  ├── migration/
  └── README.md
```

### 장점
1. **명확한 역할 분리**: MVP(과거) vs Platform(현재)
2. **Platform 개발 집중**: 루트 docs/에 방해되는 MVP 문서 제거
3. **문서 찾기 쉬움**: MVP 문서는 mvp/docs, Platform 문서는 platform/docs
4. **히스토리 보존**: MVP 구현 과정 문서 보관
5. **독립성**: MVP 문서를 수정해도 Platform에 영향 없음

---

## 주의사항

1. **링크 깨짐**: 문서 간 상호 참조 링크 확인 필요
2. **README 업데이트**: docs/README.md, mvp/docs/README.md 신규 작성
3. **DOCUMENTATION_MAP.md**: 루트의 DOCUMENTATION_MAP.md 업데이트 필요

---

**End of Plan**
