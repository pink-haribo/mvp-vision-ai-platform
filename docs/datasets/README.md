# 데이터셋 관리 문서

이 디렉토리는 Vision AI Training Platform의 데이터셋 관리 시스템 설계 및 구현 문서를 포함합니다.

## 📚 핵심 문서

### 🚀 [DATASET_FORMAT_SUMMARY.md](./DATASET_FORMAT_SUMMARY.md) ⭐ **먼저 읽으세요!**
**데이터셋 포맷 전략 요약 (Executive Summary)**

플랫폼 데이터셋 포맷 전체 전략의 요약본입니다. 5분 안에 전체 그림을 파악할 수 있습니다.

**핵심 내용**:
- v0.9 (Legacy) vs v1.0 (Platform) 비교
- 3-Tier 호환성 모델 (Native / Auto-Migration / Dual-Format)
- 기존 사용자 마이그레이션 전략
- 구현 계획 (3주)

**읽어야 할 사람**: 모든 개발자, 기획자, PM

---

### 📖 [PLATFORM_DATASET_FORMAT.md](./PLATFORM_DATASET_FORMAT.md)
**플랫폼 데이터셋 포맷 완전 사양서**

v1.0 Platform Format의 완전한 스펙 문서입니다. 구현 시 참조하세요.

**주요 내용**:
- 기존 v0.9 포맷 분석 및 한계
- 새로운 v1.0 포맷 설계 (annotations.json)
- Task별 Annotation 스키마 (Classification, Detection, Segmentation, Pose, Super-Resolution)
- 하위 호환성 전략 (V09ToV10Migrator)
- 마이그레이션 가이드 (시나리오별)
- 구현 계획 (Phase 1~4)

**읽어야 할 사람**:
- Backend 개발자 (포맷 변환 구현)
- Frontend 개발자 (UI/UX 설계)

---

### 💡 [MIGRATION_EXAMPLES.md](./MIGRATION_EXAMPLES.md)
**v0.9 → v1.0 마이그레이션 실제 예시**

Before/After 비교와 함께 실제 변환 예시를 제공합니다.

**포함된 예시**:
- Classification Dataset 변환
- Object Detection Dataset 변환
- Instance Segmentation Dataset 변환
- Semantic Segmentation Dataset 변환
- Bbox/Segmentation 좌표 변환 로직

**읽어야 할 사람**:
- 마이그레이션 로직 구현 개발자
- 테스트 케이스 작성자

---

### 🔍 [DESIGN_DECISIONS.md](./DESIGN_DECISIONS.md) ⚡ **핵심 쟁점**
**플랫폼 포맷 설계의 핵심 결정 사항**

v0.9 vs v1.0의 두 가지 핵심 쟁점에 대한 상세 비교 및 결정 근거입니다.

**핵심 쟁점**:
1. **단일 vs 분산 Annotation** (annotations.json vs labels/*.json)
   - 성능 비교: 로딩 10-30배, Cloud 30배 빠름
   - Cloud 비용: 99.8% 절약 ($48.6 → $0.081)
   - 확장성: 100K+ images 지원 가능

2. **상대 경로 vs 절대 경로** (images/img.jpg vs E:/datasets/img.jpg)
   - 이식성: 어디서든 작동 vs 드라이브 고정
   - Cloud 호환: S3/R2/GCS vs 불가능
   - 공유: ZIP 압축만 vs 경로 깨짐

**최종 결정**:
- ✅ 통합형 Annotation (성능, 비용, 확장성 우위)
- ✅ 상대 경로 (이식성, Cloud 호환, 공유 우위)
- ✅ 100% 하위 호환 보장 (auto-migration + legacy backup)

**읽어야 할 사람**:
- 아키텍처 결정에 관심 있는 개발자
- "왜 이렇게 설계했는가" 궁금한 사람
- 기존 사용자 호환성이 걱정되는 사람

---

### 🌐 [MULTIMODAL_EXTENSION.md](./MULTIMODAL_EXTENSION.md) 🆕 **멀티모달 확장**
**DICE Format v1.1: Vision+Text 멀티모달 지원**

v1.0을 확장하여 텍스트 데이터를 포함하는 멀티모달 태스크를 지원합니다.

**지원 태스크 (8종)**:
- Image Captioning (이미지 → 텍스트 설명)
- Visual Question Answering (VQA)
- Visual Grounding (텍스트 → Bbox)
- OCR (이미지 → 텍스트+위치)
- Dense Captioning (Region별 설명)
- Image-Text Matching
- Text-to-Image Retrieval
- Visual Dialogue

**핵심 변경**:
- `format_version: "1.1"`
- `modalities: ["image", "text"]`
- `text_config` 필드 추가
- 텍스트 annotation 스키마 (captions, qa_pairs, referring_expressions, text_regions 등)

**Framework 호환**: HuggingFace datasets, CLIP, BLIP, LLaVA, GLIP

**읽어야 할 사람**:
- 멀티모달 태스크 구현 개발자
- Vision-Language 모델 학습 담당자
- "VQA/Captioning 지원되나요?" 궁금한 사람

---

### 📦 [UNLABELED_DATASETS.md](./UNLABELED_DATASETS.md) ❓ **FAQ**
**레이블 없는 데이터셋 처리 및 meta.json vs annotations.json**

실무에서 자주 발생하는 질문에 대한 답변입니다.

**핵심 질문**:
1. **레이블이 없는 파일은 어떻게 관리되나?**
   - `annotation: null` 상태로 저장
   - `status: "unlabeled"` 명시적 표시
   - 증분 레이블링 지원 (Day 1: 20장, Day 2: 30장...)

2. **meta.json과 annotations.json의 차이는?**
   - meta.json: 경량 메타데이터 (~1KB), 빠른 변경 감지
   - annotations.json: 실제 레이블 데이터 (~800KB-200MB)

**주요 내용**:
- Unlabeled 데이터셋 워크플로우
- 부분 레이블링 상태에서 학습 가능 여부
- 증분 레이블링 프로세스
- API 엔드포인트 예시

**읽어야 할 사람**:
- 레이블링 워크플로우 구현 개발자
- UI/UX 설계자
- "이미지만 업로드하고 나중에 레이블링" 시나리오 궁금한 사람

---

### 🗺️ [DATASET_MANAGEMENT_PLAN.md](./DATASET_MANAGEMENT_PLAN.md)
**데이터셋 관리 시스템 종합 계획 및 현황**

R2 기반 데이터셋 관리 시스템의 전체 비전과 현재 구현 상태입니다.

**주요 내용**:
- R2 전환 배경 및 목표
- Dataset as First-Class Entity
- Mutable 데이터셋 설계
- Public/Private/Organization 권한
- 현재 구현 상태
- Phase별 로드맵

**읽어야 할 사람**:
- 데이터셋 시스템 아키텍트
- 프로젝트 전체 현황 파악 필요자

---

### 🎨 [DATASET_UI_PLAN.md](./DATASET_UI_PLAN.md)
**데이터셋 업로드/관리 UI 설계**

데이터셋 워크스페이스 UI 설계 및 구현 계획입니다.

**주요 내용**:
- 2-Column Layout (목록 + 상세)
- DatasetWorkspace 컴포넌트 구조
- Upload with Drag & Drop
- Phase별 구현 계획

**읽어야 할 사람**: Frontend 개발자

---

## 📄 예시 파일

### v1.1 Multimodal Format 예시 🆕
- [example-v1.1-vqa.json](./example-v1.1-vqa.json) - Visual Question Answering
- [example-v1.1-captioning.json](./example-v1.1-captioning.json) - Image Captioning
- [example-v1.1-ocr.json](./example-v1.1-ocr.json) - OCR (Optical Character Recognition)

### v1.0 Platform Format 예시
- [example-v1.0-classification.json](./example-v1.0-classification.json) - Classification 예시
- [example-v1.0-detection.json](./example-v1.0-detection.json) - Object Detection 예시

### v0.9 Legacy Format 예시
- [label-example-classification.json](./label-example-classification.json) - Classification
- [label-example-detection.json](./label-example-detection.json) - Detection (Bbox)
- [label-example-segmentation.json](./label-example-segmentation.json) - Segmentation (Polygon)
- [label-map-example.json](./label-map-example.json) - Summary 파일

---

## 🗂️ 관련 문서

### 원래 설계 문서
- [`docs/features/DATASET_SOURCES_DESIGN.md`](../features/DATASET_SOURCES_DESIGN.md)
  - 데이터셋 소스 유형 (로컬, 클라우드, 공개 데이터셋, DB, HTTP, Git/DVC)
  - 자동 분석 설계
  - 형식 자동 감지 로직
  - UI/UX 설계 (Before/After)
  - Phase별 구현 계획 (원본)

### 데이터베이스 스키마
- [`docs/architecture/DATABASE_SCHEMA.md`](../architecture/DATABASE_SCHEMA.md)
  - Dataset 테이블 스키마
  - DatasetPermission 테이블 스키마
  - 관계 및 인덱스

### API 명세
- [`docs/api/API_SPECIFICATION.md`](../api/API_SPECIFICATION.md)
  - `/api/v1/datasets/*` 엔드포인트 명세
  - Request/Response 예시

---

## 🚀 빠른 시작

### 현재 상태 파악
```bash
# 브랜치 확인
git branch
# feature/dataset-entity

# 최근 커밋 확인
git log --oneline -10 | grep dataset
```

### 구현된 기능 테스트

#### 1. 플랫폼 샘플 데이터셋 조회
```bash
curl http://localhost:8000/api/v1/datasets/available | jq
```

#### 2. 로컬 폴더 분석
```bash
curl -X POST http://localhost:8000/api/v1/datasets/analyze \
  -H "Content-Type: application/json" \
  -d '{"path": "C:\\datasets\\imagenet-10", "format_hint": null}'
```

#### 3. Admin 패널 확인
- http://localhost:3000 로그인
- Admin 계정으로 로그인 (admin@example.com)
- Sidebar → "데이터셋 관리" 버튼 클릭

---

## 📋 TODO: 다음 단계

### 긴급 (Priority 1)
- [ ] DatasetSourceSelector 컴포넌트 구현
- [ ] PlatformDatasetTab 구현
- [ ] LocalDatasetTab 구현
- [ ] TrainingConfigPanel에 통합

### 단기 (Priority 2-3)
- [ ] Dataset 상세 조회/수정/삭제 API
- [ ] AdminDatasetsPanel 상세 모달
- [ ] DatasetUploadModal 구현

### 중기 (Priority 4-6)
- [ ] HuggingFace Datasets 통합
- [ ] S3/GCS 통합
- [ ] 데이터셋 버전 관리

자세한 내용은 [DATASET_MANAGEMENT_PLAN.md](./DATASET_MANAGEMENT_PLAN.md)를 참조하세요.

---

## 🏗️ 주요 파일 위치

### Backend
```
mvp/backend/app/
├── db/
│   └── models.py                    # Dataset, DatasetPermission 모델
├── api/
│   └── datasets.py                  # Dataset API 엔드포인트
└── utils/
    └── dataset_analyzer.py          # 자동 분석 로직

mvp/training/
└── adapters/
    └── dataset_handler.py           # R2 lazy download
```

### Frontend
```
mvp/frontend/
├── types/
│   └── dataset.ts                   # TypeScript 타입
├── components/
│   ├── AdminDatasetsPanel.tsx       # Admin 테이블 UI
│   ├── datasets/
│   │   ├── DatasetCard.tsx
│   │   ├── DatasetList.tsx
│   │   └── DatasetPanel.tsx
│   └── TrainingConfigPanel.tsx      # (개선 필요)
└── app/
    └── datasets/
        └── page.tsx                 # 테스트 페이지
```

---

## 📞 질문 및 피드백

데이터셋 관리 관련 질문이나 피드백은:
1. GitHub Issues에 `label:dataset` 태그로 등록
2. DATASET_MANAGEMENT_PLAN.md 문서 업데이트
3. 팀 미팅에서 논의

---

**Last Updated**: 2025-01-04 (Format Design Complete)
**Maintainer**: Development Team

---

## 📅 최근 업데이트

- **2025-01-04 (v1.1)**: 멀티모달(Vision+Text) 확장 설계 완료 🆕
  - MULTIMODAL_EXTENSION.md 추가 (v1.1 스펙, 8가지 멀티모달 태스크)
  - Image Captioning, VQA, Visual Grounding, OCR, Dense Captioning 등 지원
  - example-v1.1-vqa.json, example-v1.1-captioning.json, example-v1.1-ocr.json 추가
  - HuggingFace datasets, CLIP, BLIP, LLaVA, GLIP 호환성 설계
  - v1.0 완전 하위 호환 (format_version 필드만 변경)
  - 구현 계획 수립 (9주, 2개월)

- **2025-01-04 (v1.0)**: Platform Dataset Format v1.0 설계 완료
  - DATASET_FORMAT_SUMMARY.md 추가 (Executive Summary)
  - PLATFORM_DATASET_FORMAT.md 추가 (완전 스펙, 500줄)
  - MIGRATION_EXAMPLES.md 추가 (Before/After 예시)
  - DESIGN_DECISIONS.md 추가 (핵심 쟁점 비교)
  - UNLABELED_DATASETS.md 추가 (FAQ: 레이블 없는 데이터셋 처리)
  - v0.9 하위 호환성 전략 수립 (100% 보장)
  - 성능 분석: 10-30배 빠른 로딩, 99.8% 비용 절감
  - meta.json vs annotations.json 역할 구분 명확화

- **2025-11-03**: Dataset Management Plan 수립
  - R2 기반 아키텍처 설계
  - Mutable 데이터셋 설계
