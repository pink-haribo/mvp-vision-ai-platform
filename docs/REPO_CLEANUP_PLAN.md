# Repository Cleanup Plan

**Date**: 2025-01-11
**Branch**: `repo-cleanup`
**Goal**: MVP 완료 후 Platform 개발 단계 진입에 맞춘 프로젝트 구조 재정리

---

## 현재 상황

### 프로젝트 단계
- ✅ **MVP 완료**: `mvp/` 폴더에 MVP 구현 완료
- ⏳ **Platform 개발 진입**: `platform/` 폴더에 프로덕션 아키텍처 설계 중

### 문제점
1. **MVP와 Platform 파일 혼재**: MVP 관련 파일들이 루트에 산재
2. **루트 디렉토리 과다**: 40+ 파일 (MVP 스크립트, 설정, 문서 혼재)
3. **불필요한 실행 데이터**: `mlruns/`, `runs/`, `yolov8n.pt` (~13MB)
4. **테스트/임시 파일**: `test_*.json` (11개), 분석 문서
5. **중복 설정 폴더**: `config/`, `docker/` 용도 불명확

---

## 정리 원칙

### 핵심 원칙
```
MVP 파일 → mvp/ 폴더
Platform 파일 → platform/ 폴더 또는 루트
공용 파일 → 루트 (문서, 빌드 설정)
```

### 디렉토리 역할 명확화
- `mvp/` - MVP 구현 완료 (유지 모드, 수정 최소화)
- `platform/` - Platform 개발 진행 중 (활발한 개발)
- `docs/` - 프로젝트 전체 문서 (MVP + Platform)
- `루트` - 프로젝트 메타 파일만 (README, CLAUDE.md, Makefile 등)

---

## 정리 계획

### Phase 1: 삭제 (즉시 실행)

#### 1.1 실행 데이터 디렉토리 삭제
```bash
rm -rf mlruns/      # MLflow 로컬 실행 결과 (재생성 가능)
rm -rf runs/        # YOLO 실행 결과 (재생성 가능)
rm -f yolov8n.pt    # 모델 파일 6.5MB (다운로드 가능)
```

#### 1.2 테스트/임시 파일 삭제
```bash
rm -f test_*.json                                # API 테스트용 (11개)
rm -f nul                                        # 빈 파일
rm -f mvp-vision-ai-platform-analysis-251110.md  # 임시 분석 문서
```

### Phase 2: MVP 파일 이동 (mvp/ 폴더로)

#### 2.1 MVP 빌드 설정
```bash
mv Makefile.mvp mvp/Makefile
```
**이유**: MVP 전용 Makefile은 mvp/ 폴더에서 관리

#### 2.2 MVP Docker Compose
```bash
mkdir -p mvp/infrastructure
mv docker-compose.dev.yml mvp/infrastructure/
```
**이유**: MVP 개발 환경 설정

#### 2.3 MVP 개발 스크립트
```bash
mkdir -p mvp/scripts/dev
mv dev-start.ps1 mvp/scripts/dev/
mv dev-status.ps1 mvp/scripts/dev/
mv dev-stop.ps1 mvp/scripts/dev/
mv dev-train-k8s.ps1 mvp/scripts/dev/
mv dev-train-local.ps1 mvp/scripts/dev/
mv analyze_class_dist.py mvp/scripts/dev/
```
**이유**: MVP 개발 시 사용한 스크립트들

#### 2.4 MVP 유틸리티 스크립트
```bash
# scripts/ 전체를 mvp/scripts/utils/로 이동
mkdir -p mvp/scripts/utils
mv scripts/convert_to_dice_format.py mvp/scripts/utils/
mv scripts/convert_to_dice_format_v2.py mvp/scripts/utils/
mv scripts/create_dataset_zips.py mvp/scripts/utils/
mv scripts/test_r2_connection.py mvp/scripts/utils/
mv scripts/upload_pretrained_models.py mvp/scripts/utils/
mv scripts/upload_sample_datasets.py mvp/scripts/utils/
mv scripts/init-postgres.sql mvp/scripts/utils/

# scripts/ 폴더 삭제 (비어있으면)
rmdir scripts 2>/dev/null || true
```
**이유**: MVP 단계에서 사용한 유틸리티 스크립트들

#### 2.5 MVP 설정 폴더
```bash
# config/ 폴더는 MVP 인프라 설정
mv config mvp/infrastructure/
mv docker mvp/infrastructure/
```
**이유**: MVP 단계의 모니터링 스택 설정 (Grafana, Loki, Prometheus, Temporal)

#### 2.6 MVP 개발 문서
```bash
mkdir -p mvp/docs/guides
mv GETTING_STARTED.md mvp/docs/guides/
mv QUICK_DEV_GUIDE.md mvp/docs/guides/
mv DEV_WORKFLOW.md mvp/docs/guides/
mv DEV_SCRIPTS.md mvp/docs/guides/
```
**이유**: MVP 개발 가이드 문서들

### Phase 3: Platform 파일 정리

#### 3.1 Platform 인프라 설정 (향후 생성 예정)
```bash
mkdir -p platform/infrastructure/compose
mkdir -p platform/infrastructure/k8s
```
**참고**: Platform용 docker-compose.yml은 향후 생성 예정

#### 3.2 설계 리뷰 문서
```bash
mkdir -p docs/reviews
mv FINAL_DESIGN_REVIEW_2025-01-11.md docs/reviews/
```

### Phase 4: 루트 정리 (최소화)

#### 4.1 유지할 파일 (12개)
```
프로젝트 메타 문서 (4개):
✅ README.md              - 프로젝트 전체 소개
✅ CLAUDE.md              - Claude Code 가이드
✅ CONTRIBUTING.md        - 기여 가이드
✅ DOCUMENTATION_MAP.md   - 문서 맵

빌드 설정 (5개):
✅ Makefile              - 전체 프로젝트 Make 명령어
✅ .dockerignore         - Docker 빌드 제외
✅ .gitignore            - Git 추적 제외
✅ .railwayignore        - Railway 배포 제외

환경 설정 (1개):
✅ .env.r2.example       - R2 환경 변수 예시

인프라 설정 (1개):
✅ docker-compose.yml    - 전체 프로젝트 인프라 (향후 Platform용으로 업데이트)

개발 도구 (2개):
✅ .claude/              - Claude Code 설정
✅ .github/              - GitHub Actions
```

#### 4.2 삭제할 파일 (이미 Phase 1, 2에서 이동/삭제)
```
❌ Makefile.mvp          → mvp/Makefile
❌ docker-compose.dev.yml → mvp/infrastructure/
❌ dev-*.ps1 (5개)       → mvp/scripts/dev/
❌ analyze_class_dist.py → mvp/scripts/dev/
❌ scripts/ (전체)       → mvp/scripts/utils/
❌ config/               → mvp/infrastructure/
❌ docker/               → mvp/infrastructure/
❌ GETTING_STARTED.md    → mvp/docs/guides/
❌ QUICK_DEV_GUIDE.md    → mvp/docs/guides/
❌ DEV_WORKFLOW.md       → mvp/docs/guides/
❌ DEV_SCRIPTS.md        → mvp/docs/guides/
❌ test_*.json (11개)    → 삭제
❌ mlruns/, runs/        → 삭제
❌ yolov8n.pt            → 삭제
```

### Phase 5: .gitignore 업데이트

```bash
cat >> .gitignore << 'EOF'

# ===== Runtime Data (Local Execution) =====
mlruns/
runs/

# Model weights (download if needed)
*.pt
*.pth
*.onnx
*.tflite

# ===== Test Files =====
test_*.json
test_*.py
**/test_output/

# ===== Temporary Files =====
*-analysis-*.md
nul

EOF
```

---

## 정리 후 구조

```
mvp-vision-ai-platform/
├── .claude/                          # Claude Code 설정
├── .github/                          # GitHub Actions
│
├── docs/                             # 프로젝트 전체 문서 ⭐ 공용
│   ├── architecture/                 # Platform 아키텍처 설계
│   ├── datasets/                     # 데이터셋 문서
│   ├── api/                          # API 문서
│   └── reviews/                      # 설계 리뷰 ⭐ NEW
│       └── FINAL_DESIGN_REVIEW_2025-01-11.md
│
├── mvp/                              # ⭐ MVP 완료 (유지 모드)
│   ├── backend/
│   ├── frontend/
│   ├── training/
│   ├── k8s/
│   ├── docs/                         # ⭐ MVP 전용 문서
│   │   └── guides/                   # ⭐ NEW
│   │       ├── GETTING_STARTED.md    # ⭐ MOVED
│   │       ├── QUICK_DEV_GUIDE.md    # ⭐ MOVED
│   │       ├── DEV_WORKFLOW.md       # ⭐ MOVED
│   │       └── DEV_SCRIPTS.md        # ⭐ MOVED
│   ├── infrastructure/               # ⭐ NEW
│   │   ├── docker-compose.dev.yml    # ⭐ MOVED
│   │   ├── config/                   # ⭐ MOVED (Grafana, Loki, Prometheus, Temporal)
│   │   └── docker/                   # ⭐ MOVED (MLflow)
│   ├── scripts/                      # ⭐ MVP 스크립트
│   │   ├── dev/                      # ⭐ NEW
│   │   │   ├── dev-start.ps1         # ⭐ MOVED
│   │   │   ├── dev-status.ps1        # ⭐ MOVED
│   │   │   ├── dev-stop.ps1          # ⭐ MOVED
│   │   │   ├── dev-train-k8s.ps1     # ⭐ MOVED
│   │   │   ├── dev-train-local.ps1   # ⭐ MOVED
│   │   │   └── analyze_class_dist.py # ⭐ MOVED
│   │   └── utils/                    # ⭐ NEW
│   │       ├── convert_to_dice_format.py      # ⭐ MOVED
│   │       ├── convert_to_dice_format_v2.py   # ⭐ MOVED
│   │       ├── create_dataset_zips.py         # ⭐ MOVED
│   │       ├── test_r2_connection.py          # ⭐ MOVED
│   │       ├── upload_pretrained_models.py    # ⭐ MOVED
│   │       ├── upload_sample_datasets.py      # ⭐ MOVED
│   │       └── init-postgres.sql/             # ⭐ MOVED
│   ├── Makefile                      # ⭐ NEW (from Makefile.mvp)
│   └── README.md                     # MVP README
│
├── platform/                         # ⭐ Platform 개발 진행 중
│   ├── backend/
│   ├── docs/
│   │   ├── architecture/
│   │   └── development/
│   ├── frontend/
│   └── infrastructure/               # Platform 인프라 (향후 생성)
│       ├── compose/                  # docker-compose.yml (향후)
│       └── k8s/                      # K8s manifests (향후)
│
├── .dockerignore                     # Docker 빌드 제외
├── .env.r2.example                   # 환경 변수 예시
├── .gitignore                        # Git 추적 제외 ⭐ 업데이트 예정
├── .railwayignore                    # Railway 배포 제외
├── CLAUDE.md                         # Claude Code 가이드
├── CONTRIBUTING.md                   # 기여 가이드
├── docker-compose.yml                # 전체 프로젝트 인프라 (향후 Platform용)
├── DOCUMENTATION_MAP.md              # 문서 맵
├── Makefile                          # 전체 프로젝트 Make 명령어
└── README.md                         # 프로젝트 소개

⭐ = NEW or MOVED or 업데이트 예정
```

---

## 실행 순서

### Step 1: 백업 (선택)
```bash
# 현재 브랜치: repo-cleanup
git add -A
git commit -m "chore: snapshot before major cleanup"
```

### Step 2: 삭제 작업 (Phase 1)
```bash
rm -rf mlruns/ runs/
rm -f yolov8n.pt
rm -f test_*.json nul
rm -f mvp-vision-ai-platform-analysis-251110.md
```

### Step 3: MVP 디렉토리 준비 (Phase 2)
```bash
mkdir -p mvp/docs/guides
mkdir -p mvp/infrastructure
mkdir -p mvp/scripts/dev
mkdir -p mvp/scripts/utils
```

### Step 4: MVP 파일 이동 (Phase 2)
```bash
# 빌드 설정
mv Makefile.mvp mvp/Makefile

# 인프라
mv docker-compose.dev.yml mvp/infrastructure/
mv config mvp/infrastructure/
mv docker mvp/infrastructure/

# 개발 스크립트
mv dev-*.ps1 mvp/scripts/dev/
mv analyze_class_dist.py mvp/scripts/dev/

# 유틸리티 스크립트
mv scripts/* mvp/scripts/utils/
rmdir scripts

# 개발 문서
mv GETTING_STARTED.md mvp/docs/guides/
mv QUICK_DEV_GUIDE.md mvp/docs/guides/
mv DEV_WORKFLOW.md mvp/docs/guides/
mv DEV_SCRIPTS.md mvp/docs/guides/
```

### Step 5: Platform 디렉토리 준비 (Phase 3)
```bash
mkdir -p docs/reviews
mv FINAL_DESIGN_REVIEW_2025-01-11.md docs/reviews/
```

### Step 6: .gitignore 업데이트 (Phase 5)
```bash
cat >> .gitignore << 'EOF'

# ===== Runtime Data (Local Execution) =====
mlruns/
runs/

# Model weights (download if needed)
*.pt
*.pth
*.onnx
*.tflite

# ===== Test Files =====
test_*.json
test_*.py
**/test_output/

# ===== Temporary Files =====
*-analysis-*.md
nul

EOF
```

### Step 7: 검증
```bash
# 루트 파일 확인 (12개 정도만 남아야 함)
ls -1 | wc -l

# MVP 구조 확인
ls -la mvp/

# Platform 구조 확인
ls -la platform/

# Git 상태 확인
git status
```

### Step 8: 커밋
```bash
git add -A
git commit -m "chore(repo): reorganize for Platform development phase

Separate MVP (completed) and Platform (in development) files:

**Deleted:**
- Runtime data: mlruns/, runs/, yolov8n.pt (~13MB)
- Test files: test_*.json (11 files)
- Temporary: mvp-vision-ai-platform-analysis-251110.md

**Moved to mvp/:**
- Build: Makefile.mvp → mvp/Makefile
- Infrastructure: docker-compose.dev.yml, config/, docker/
- Dev scripts: dev-*.ps1 (5), analyze_class_dist.py
- Utils: scripts/* → mvp/scripts/utils/
- Docs: dev guides → mvp/docs/guides/

**Organized:**
- Design reviews → docs/reviews/
- Root: Only 12 essential files (README, Makefile, config)

This restructure reflects project transition from MVP (completed)
to Platform (active development) phase.
"
```

---

## 검증 체크리스트

### 삭제 확인
- [ ] `mlruns/` 디렉토리 삭제됨
- [ ] `runs/` 디렉토리 삭제됨
- [ ] `yolov8n.pt` 파일 삭제됨 (~6.5MB)
- [ ] `test_*.json` 파일 11개 삭제됨
- [ ] `nul`, `mvp-vision-ai-platform-analysis-251110.md` 삭제됨

### MVP 파일 이동 확인
- [ ] `Makefile.mvp` → `mvp/Makefile`
- [ ] `docker-compose.dev.yml` → `mvp/infrastructure/`
- [ ] `config/` → `mvp/infrastructure/config/`
- [ ] `docker/` → `mvp/infrastructure/docker/`
- [ ] `dev-*.ps1` (5개) → `mvp/scripts/dev/`
- [ ] `analyze_class_dist.py` → `mvp/scripts/dev/`
- [ ] `scripts/*` → `mvp/scripts/utils/`
- [ ] 개발 가이드 문서 (4개) → `mvp/docs/guides/`

### Platform 파일 정리 확인
- [ ] `FINAL_DESIGN_REVIEW_2025-01-11.md` → `docs/reviews/`

### 루트 정리 확인
- [ ] 루트에 12개 파일만 남음
- [ ] MVP 관련 파일 없음
- [ ] 테스트/임시 파일 없음
- [ ] 실행 데이터 없음

### 기능 검증
- [ ] `mvp/Makefile` 사용 가능 (mvp 폴더 내에서)
- [ ] `mvp/infrastructure/docker-compose.dev.yml` 사용 가능
- [ ] `mvp/scripts/dev/dev-start.ps1` 실행 가능
- [ ] Git 크기 감소 확인 (~13MB)

---

## 예상 효과

### Before
```
루트: 40+ 파일 (MVP + Platform + 테스트 + 실행데이터 혼재)
MVP 파일 분산: 루트, scripts/, config/, docker/
```

### After
```
루트: 12개 파일 (프로젝트 메타 파일만)
mvp/: MVP 관련 파일 모두 통합 (빌드, 인프라, 스크립트, 문서)
platform/: Platform 개발 파일만
```

### 장점
1. **명확한 단계 분리**: MVP(완료) vs Platform(진행중)
2. **루트 깔끔**: 프로젝트 메타 파일만
3. **MVP 독립성**: mvp/ 폴더에서 완전히 독립적으로 실행 가능
4. **Platform 집중**: 루트에서 Platform 개발에 집중 가능
5. **Git 크기 감소**: ~13MB (실행 데이터 제거)
6. **신규 개발자 친화**: MVP와 Platform 구분 명확

---

## 주의사항

### 1. 스크립트 경로 변경
**영향**: `mvp/scripts/dev/dev-start.ps1` 등
**조치**: 스크립트 내부 상대 경로 확인 필요 (루트 → mvp/)

### 2. 문서 링크
**영향**: DOCUMENTATION_MAP.md, README.md
**조치**: MVP 문서 경로 업데이트

### 3. CI/CD
**영향**: GitHub Actions (.github/workflows/)
**조치**: MVP 관련 workflow 경로 업데이트

### 4. docker-compose.yml
**영향**: 루트의 `docker-compose.yml`
**조치**: 향후 Platform용으로 업데이트 필요 (현재는 MVP용)

### 5. Makefile
**영향**: 루트 `Makefile`
**조치**: MVP 명령어 제거 또는 `mvp/Makefile` 위임

---

## Next Steps

1. ✅ 이 계획 리뷰 및 승인
2. ⏳ Phase 1 실행 (삭제)
3. ⏳ Phase 2 실행 (MVP 파일 이동)
4. ⏳ Phase 3 실행 (Platform 파일 정리)
5. ⏳ Phase 4-5 실행 (.gitignore, 검증)
6. ⏳ 스크립트 경로 수정
7. ⏳ 문서 링크 업데이트
8. ⏳ 커밋 및 PR

---

**End of Plan**
