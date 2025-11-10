# K8s Refactoring 계획

## 개요

`mvp/training/` 디렉토리를 삭제하고 완전히 새로운 Trainer 아키텍처로 리팩토링합니다.

**목표:**
- Backend와 Trainer의 완전한 분리 (파일 시스템 격리)
- 프레임워크별 독립적인 Trainer 서비스 (trainer-ultralytics, trainer-timm 등)
- 환경별 실행 방식 통일 (subprocess, Kind K8s, Railway K8s)
- 명확한 API Contract (Plugin 개발자 가이드)
- 의존성 격리 및 간결한 코드

## 문서 구성

### 1. 아키텍처 및 설계

**[ARCHITECTURE_DECISIONS.md](./ARCHITECTURE_DECISIONS.md)** ⭐ NEW
K8s Job 패턴 제안 분석 및 MVP 적용 결정

- ✅ 수용: API Contract 구조화, JWT 인증, 상태머신, Trace ID
- ⚠️ 부분 수용: 로그 수집, Heartbeat 간격, Presigned URL
- 🔄 보류: Temporal/Argo, Redis Streams, 별도 클러스터, Manifest
- ❌ 거절: OpenAPI SDK 자동 생성, mTLS (MVP)

**[PLUGIN_GUIDE.md](./PLUGIN_GUIDE.md)** ⭐ NEW
Trainer Plugin 개발 가이드 (필수 읽기!)

- API Contract 명세 (환경변수, Callback API, Storage, MLflow)
- 구현 옵션 (utils.py 재사용 vs 직접 구현 vs 자체 라이브러리)
- 예제 코드 및 테스트 방법

**[trainer_architecture.md](./trainer_architecture.md)**
전체 디렉토리 구조 및 파일 설계

- trainer-ultralytics/ 상세 (train.py, predict.py, utils.py)
- Backend 변경사항 (training_manager_k8s.py)

---

### 2. 환경 및 실행

**[trainer_env_comparison.md](./trainer_env_comparison.md)**
실행 환경별 차이점 분석

- Local (Subprocess) vs Kind K8s vs Production (Railway)
- 공통 사항: 환경변수 기반, Callback API, Storage 격리

**[trainer_scripts_analysis.md](./trainer_scripts_analysis.md)**
필요한 스크립트 상세 분석

- train.py (필수), predict.py (필수)
- evaluate.py (보류), export.py (보류)

---

### 3. 구현

**[implementation_plan_v2.md](./implementation_plan_v2.md)** ⭐ NEW
Phase별 구현 계획 (12-18시간, 8 Phase)

- Phase 1: Backend API Contract (heartbeat, event, done)
- Phase 2: JWT 인증
- Phase 3: 상태머신 + Trace ID
- Phase 4-5: trainer-ultralytics 구현
- Phase 6-7: 통합 테스트
- Phase 8: 문서화

**[trainer_common_functionality.md](./trainer_common_functionality.md)**
공통 기능 분석 및 중복 방지 전략

- 옵션 A: utils.py 복사 (권장)
- 옵션 B/C: Shared Package / Inline 구현

## 핵심 결정사항

### 1. ⭐ 완전한 파일 시스템 격리 (Local/Production 동일)

**Backend는 절대 Trainer의 파일 시스템에 접근하지 않음:**

```python
# ❌ 금지 (로컬에서도!)
dataset_path = "/data/datasets/abc123"  # 로컬 경로 전달
checkpoint = open("../trainer-ultralytics/checkpoints/best.pt")  # 파일 직접 읽기

# ✅ 올바른 방식
env["DATASET_ID"] = "abc123"  # ID만 전달
# Trainer가 Storage에서 다운로드: s3.download_file(...)
```

**모든 데이터는 Storage를 통해서만:**
- Dataset: Backend → MinIO 업로드 → Trainer 다운로드
- Checkpoint: Trainer → MinIO 업로드 → Backend는 URL만 저장

---

### 2. ⭐ API Contract (Plugin 인터페이스)

**Trainer가 지켜야 할 계약:**

#### 입력: 환경변수
```bash
JOB_ID, TRACE_ID, BACKEND_BASE_URL, CALLBACK_TOKEN
DATASET_ID, MODEL_NAME, EPOCHS, BATCH_SIZE
STORAGE_TYPE, R2_ENDPOINT, R2_ACCESS_KEY_ID
```

#### 출력: HTTP Callback API
```http
POST /v1/jobs/{id}/heartbeat  # Epoch마다
POST /v1/jobs/{id}/event      # 중요 이벤트
POST /v1/jobs/{id}/done       # 최종 완료
```

#### 출력: Storage (S3 API)
```python
s3.download_file(bucket, f"datasets/{dataset_id}.zip", ...)
s3.upload_file(local, bucket, f"checkpoints/job-{id}/best.pt")
```

**내부 구현은 자유** (utils.py 복사 or 직접 구현 or 자체 라이브러리)

---

### 3. 보안: JWT 콜백 인증

```python
# Backend: Job 생성 시 토큰 발급 (6-24시간)
token = jwt.encode({"job_id": job_id, "exp": ...}, SECRET_KEY)

# Trainer: 모든 Callback에 포함
headers = {"Authorization": f"Bearer {token}"}

# Backend: 토큰 검증 + job_id 바인딩
if payload["job_id"] != expected_job_id:
    raise HTTPException(401, "Token job_id mismatch")
```

---

### 4. 상태머신

```
PENDING → QUEUED → RUNNING → {SUCCEEDED | FAILED | CANCELLED}
```

- **PENDING**: DB 생성됨
- **QUEUED**: K8s Job 제출됨 (Pod 대기)
- **RUNNING**: 학습 실행 중
- **SUCCEEDED**: 정상 완료
- **FAILED**: 에러 발생
- **CANCELLED**: 사용자 취소

---

### 5. Trace ID (분산 추적)

```python
# Job 생성 시
trace_id = str(uuid.uuid4())

# 모든 로그/이벤트에 포함
print(f"[TRACE:{trace_id}] Training started")

# Loki 쿼리
{job_id="123"} |= "TRACE:abc-def"
```

---

### 6. Monorepo 유지
- `mvp/trainer-ultralytics/`, `mvp/trainer-timm/` 형태로 구성
- 각 trainer는 완전히 독립적
- Git repository는 하나로 유지

---

### 7. Shared SDK 없음
- `mvp/shared/` 디렉토리 없음
- 각 trainer에 `utils.py` 복사 (~150줄)
- 완전한 독립성 > 코드 중복

---

### 8. 환경변수 기반 설정
- argparse 사용 안 함
- 모든 설정을 환경변수로 전달
- Backend가 환경변수 주입 책임

---

### 9. Storage 추상화
- `STORAGE_TYPE=minio|r2`로 자동 분기
- boto3 S3-compatible 통일
- Local: MinIO, Production: R2

## 다음 단계

1. ✅ 문서 작성 완료 (8개 파일)
2. 계획 리뷰 및 수정 ← **현재 단계**
3. `mvp/training/` 삭제 준비
4. Phase 1 시작: Backend API Contract 구현

## 타임라인 (최신)

**v2 구현 계획 (12-18시간):**

- Phase 1: Backend API Contract (heartbeat, event, done) - 2-3h
- Phase 2: JWT 인증 - 1-2h
- Phase 3: 상태머신 + Trace ID - 1-2h
- Phase 4: trainer-ultralytics utils.py - 1-2h
- Phase 5: trainer-ultralytics train.py - 3-4h
- Phase 6: Backend 연동 + K8s Job - 2-3h
- Phase 7: 통합 테스트 - 1-2h
- Phase 8: 문서화 - 1h
- **Total: 12-18시간**
