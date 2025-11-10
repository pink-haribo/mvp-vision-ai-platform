# 아키텍처 결정사항 (Architecture Decisions)

## 제안된 K8s Job 패턴 분석

제안 핵심: "컨트롤 플레인(명령·상태)"과 "데이터/로그 플레인(I/O·관측)" 분리

---

## ✅ MVP에서 수용 (즉시 적용)

### 1. API Contract 구조화

**기존 (단순):**
```
PATCH /internal/training/{id}/status
POST /internal/training/{id}/validation-results
```

**개선 (구조화):**
```
POST /v1/jobs/{id}/heartbeat     # 5-10초 간격, 진행률
POST /v1/jobs/{id}/event          # 중요 이벤트 (epoch end, checkpoint saved)
POST /v1/jobs/{id}/done           # 최종 완료 (MLflow run_id, artifacts)
```

**적용 방식:**
- `heartbeat`: 기존 `training-metrics` 엔드포인트를 확장
- `event`: 새로운 이벤트 스트림 엔드포인트
- `done`: 기존 `PATCH /status` 확장

**장점:**
- Frontend 실시간 업데이트 용이
- 디버깅 편함 (이벤트 타임라인)
- 재시도/재개 시 상태 복원 쉬움

---

### 2. 콜백 인증 (JWT)

**현재:** `X-Internal-Auth: {고정 토큰}`

**개선:**
```python
# Backend: Job 생성 시 단기 토큰 발급
import jwt
token = jwt.encode({
    "job_id": job_id,
    "exp": datetime.utcnow() + timedelta(hours=6),
    "scope": "training-callback"
}, SECRET_KEY, algorithm="HS256")

# Job에 환경변수로 전달
env["CALLBACK_TOKEN"] = token

# Trainer: 콜백 시 사용
headers = {"Authorization": f"Bearer {token}"}
```

**Backend 검증:**
```python
def verify_callback_token(token: str, expected_job_id: int):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if payload["job_id"] != expected_job_id:
            raise HTTPException(401, "Token job_id mismatch")
        return True
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
```

**장점:**
- Job별 고유 토큰 (재사용 방지)
- 시간 제한 (6시간 후 자동 만료)
- job_id 바인딩 (다른 job 접근 불가)

---

### 3. 상태머신 명확화

**상태 정의:**
```python
class JobStatus(str, Enum):
    PENDING = "pending"      # DB에 생성됨
    QUEUED = "queued"        # K8s Job 제출됨 (Pod 대기 중)
    RUNNING = "running"      # 학습 실행 중
    SUCCEEDED = "succeeded"  # 정상 완료
    FAILED = "failed"        # 에러로 실패
    CANCELLED = "cancelled"  # 사용자 취소
```

**전이 규칙:**
```
PENDING → QUEUED → RUNNING → {SUCCEEDED | FAILED | CANCELLED}
         ↓         ↓
      FAILED    CANCELLED
```

**DB 스키마:**
```python
class TrainingJob(Base):
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    queued_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 재시도 관리
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
```

---

### 4. Trace ID 추가

**목적:** 분산 추적, 로그 상관관계

```python
import uuid

# Job 생성 시
trace_id = str(uuid.uuid4())
job.trace_id = trace_id

# Trainer 환경변수
env["TRACE_ID"] = trace_id

# 모든 로그에 포함
print(f"[TRACE:{trace_id}] Training started")

# Callback에 포함
requests.post(callback_url, json={"trace_id": trace_id, ...})

# Loki 쿼리
{job_id="123"} |= "TRACE:abc-def"
```

---

### 5. K8s Job 템플릿 구체화

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-{{job_id}}
  labels:
    app: trainer
    job-id: "{{job_id}}"
    trace-id: "{{trace_id}}"
spec:
  ttlSecondsAfterFinished: 3600  # 1시간 후 자동 삭제
  backoffLimit: 0                 # 재시도 없음 (Backend에서 관리)
  activeDeadlineSeconds: 86400    # 24시간 타임아웃
  template:
    metadata:
      labels:
        job-id: "{{job_id}}"
        trace-id: "{{trace_id}}"
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: ghcr.io/yourorg/trainer-ultralytics:v1.0.0
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        env:
          # Job 식별
          - name: JOB_ID
            value: "{{job_id}}"
          - name: TRACE_ID
            value: "{{trace_id}}"

          # Callback
          - name: BACKEND_BASE_URL
            value: "https://api.example.com"
          - name: CALLBACK_TOKEN
            valueFrom:
              secretKeyRef:
                name: callback-token-{{job_id}}
                key: token

          # Storage
          - name: STORAGE_TYPE
            value: "r2"
          - name: R2_ENDPOINT
            value: "https://xxx.r2.cloudflarestorage.com"
          - name: R2_ACCESS_KEY_ID
            valueFrom:
              secretKeyRef:
                name: r2-credentials
                key: access_key_id
          - name: R2_SECRET_ACCESS_KEY
            valueFrom:
              secretKeyRef:
                name: r2-credentials
                key: secret_access_key

          # Training config
          - name: DATASET_ID
            value: "{{dataset_id}}"
          - name: MODEL_NAME
            value: "{{model_name}}"
          - name: EPOCHS
            value: "{{epochs}}"
```

---

## ⚠️ MVP에서 부분 수용 (간소화 적용)

### 1. 로그/메트릭 수집

**제안:** 중앙 Fluent Bit/OTel Collector → Loki/OTel Backend

**MVP 적용:**
- **Local (subprocess):** 파일 기반 로그 → Promtail → Loki
- **Production (K8s):** stdout → Loki (Grafana Cloud)
- OTel은 나중에 (Phase 2)

**이유:** MVP는 단순성 우선, 중앙 인프라 없음

---

### 2. Heartbeat 간격

**제안:** 5-10초 간격

**MVP 적용:** Epoch 단위 (10초~수분)

**이유:**
- YOLO는 epoch이 명확한 단위
- 너무 잦은 heartbeat는 오버헤드
- Epoch end callback으로 충분

---

### 3. Presigned URL

**제안:** R2 presigned GET/PUT (제로트러스트)

**MVP 적용:** R2 RO/WO 키 직접 전달

**이유:**
- MVP는 내부 네트워크만
- Presigned URL 생성 로직 추가 복잡도
- 나중에 외부 제공 시 추가

---

## 🔄 Phase 2로 보류 (MVP 이후)

### 1. Temporal/Argo Workflows

**제안:** Workflow 엔진으로 재시도/타임아웃 관리

**보류 이유:**
- MVP는 직접 K8s API 호출로 충분
- Temporal 학습 곡선
- 인프라 추가 복잡도

**적용 시점:** Job 수가 많아지고 복잡한 워크플로우 필요 시

---

### 2. Redis Streams (이벤트 중계)

**제안:** Backend → Redis Streams → WebSocket

**보류 이유:**
- MVP는 Backend가 직접 WebSocket emit
- Redis 인프라 추가

**적용 시점:** 동시 사용자 증가, 스케일 필요 시

---

### 3. 별도 클러스터

**제안:** 학습 전용 클러스터 분리

**보류 이유:**
- MVP는 단일 클러스터 (Railway)
- 네트워크 분리 복잡도

**적용 시점:** GPU 클러스터 최적화 필요 시

---

### 4. Manifest 기반 데이터셋

**제안:** `_manifest.json` (버전드 키 + 메타데이터)

**보류 이유:**
- MVP는 단순 zip
- Manifest 생성/검증 로직 필요

**적용 시점:** 대용량 데이터셋, 증분 업데이트 필요 시

---

### 5. mTLS

**제안:** 상호 인증 (클러스터 간)

**보류 이유:**
- MVP는 JWT로 충분
- 인증서 관리 복잡도

**적용 시점:** 엔터프라이즈 배포, 규제 요구사항 있을 시

---

## ❌ 거절 (현재 방향과 불일치)

### 1. OpenAPI SDK 자동 생성

**이유:**
- Trainer는 Python만 (openapi-python-client 불필요)
- Backend API는 내부용 (외부 SDK 미제공)

**대안:** API Contract 문서 + 예제 코드

---

## 최종 MVP 아키텍처 (조정 후)

### 컨트롤 플레인
```
Backend API:
  POST /v1/jobs                      # Job 생성
  GET /v1/jobs/{id}                  # 상태 조회
  POST /v1/jobs/{id}/cancel          # 취소

Callback API (Trainer → Backend):
  POST /v1/jobs/{id}/heartbeat       # Epoch end마다
  POST /v1/jobs/{id}/event           # 중요 이벤트
  POST /v1/jobs/{id}/done            # 최종 완료
```

### 데이터 플레인
```
입력: R2 RO 키 → s3://bucket/datasets/{id}.zip
출력: R2 WO 키 → s3://bucket/checkpoints/job-{id}/best.pt
```

### 보안
```
JWT 토큰 (6시간):
  - job_id 바인딩
  - 재사용 방지
  - Backend 검증
```

### 상태머신
```
PENDING → QUEUED → RUNNING → {SUCCEEDED|FAILED|CANCELLED}
```

### 로그
```
Local: stdout → 파일 → Promtail → Loki
Prod: stdout → Loki (Grafana Cloud)
```

---

## 구현 순서 업데이트

1. **Phase 1:** API Contract 구체화 (heartbeat, event, done)
2. **Phase 2:** JWT 콜백 인증
3. **Phase 3:** 상태머신 + Trace ID
4. **Phase 4:** K8s Job 템플릿
5. **Phase 5:** train.py 구현 (새 API 사용)
6. **Phase 6:** Backend 콜백 엔드포인트
7. **Phase 7:** 통합 테스트

**Phase 2 (향후):**
- Temporal/Argo
- Redis Streams
- Presigned URL
- Manifest 데이터셋
- mTLS
