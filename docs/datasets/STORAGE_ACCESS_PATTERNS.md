# Dataset Storage Access Patterns

## Overview

이 문서는 Vision AI Platform의 데이터셋 저장 및 접근 패턴에 대한 설계 결정사항을 정리합니다.

**핵심 질문:**
1. 레이블러에서 이미지를 어떻게 로드하는가?
2. Training Service가 데이터셋에 어떻게 접근하는가?
3. 각 단계에서 압축/해제는 언제 일어나는가?

---

## 1. Presigned URL 방식 분석

### 1.1 성능 비교

| 방식 | 첫 이미지 로드 | 이후 이미지 | 100장 전환 시간 | 서버 부하 |
|------|---------------|------------|----------------|----------|
| **Presigned URL** (직접 R2) | 200-500ms | 100-300ms | ~15초 | 없음 |
| **Backend 중계** | 500-1000ms | 300-800ms | ~40초 | 높음 |
| **Backend + 캐시** | 500-1000ms | 50-100ms | ~8초 | 중간 (디스크 사용) |

**결론:** Presigned URL이 가장 효율적 (서버 부하 없음, CDN 활용)

### 1.2 Presigned URL로 가능한 작업

```typescript
// ✅ 가능한 모든 레이블링 작업
const img = new Image()
img.src = presignedUrl  // R2에서 직접 로드

img.onload = () => {
  // 1. Canvas에 렌더링
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

  // 2. 확대/축소 (Segmentation 필수)
  ctx.scale(zoomLevel, zoomLevel)
  ctx.translate(panX, panY)

  // 3. Bounding Box 그리기 (Object Detection)
  ctx.strokeRect(x, y, width, height)

  // 4. Polygon 그리기 (Instance Segmentation)
  ctx.beginPath()
  points.forEach((p, i) => {
    i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)
  })
  ctx.closePath()
  ctx.stroke()

  // 5. 픽셀 데이터 접근 (Semantic Segmentation)
  const imageData = ctx.getImageData(0, 0, width, height)
  // 픽셀별 클래스 할당
}
```

**핵심:** 이미지가 브라우저 메모리에 로드되면 모든 Canvas API 사용 가능

### 1.3 보안 고려사항

```python
# Backend에서 Presigned URL 생성
from datetime import timedelta

def generate_presigned_url(dataset_id: str, image_path: str, user_id: str):
    # 1. 권한 검증
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not has_permission(user_id, dataset):
        raise PermissionError()

    # 2. Presigned URL 생성 (유효기간 1시간)
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': settings.S3_BUCKET,
            'Key': f'datasets/{dataset_id}/{image_path}'
        },
        ExpiresIn=3600  # 1 hour
    )

    return url
```

**보안:**
- ✅ Backend에서 권한 검증 후 URL 생성
- ✅ 유효기간 제한 (1시간)
- ✅ 직접 R2 접근 불가 (Presigned URL 필요)

---

## 2. Training Service 데이터셋 접근 방식

### 2.1 접근 방식 비교

#### Option A: Training Service → R2 직접 접근 (권장)

```
User → Backend → Training Service
              ↓           ↓
             R2 ← ← ← ← ← R2 (직접 다운로드)
```

**장점:**
- ✅ Backend 부하 없음 (메타데이터 전달만)
- ✅ 확장성 좋음 (Training Service 여러 개 가능)
- ✅ Backend 디스크 사용 없음
- ✅ 간단한 아키텍처

**단점:**
- ⚠️ Training Service가 R2 credentials 필요
- ⚠️ 네트워크 구성 필요 (VPC 내 R2 접근)

**구현:**
```python
# Backend
@router.post("/training/start")
async def start_training(request: TrainingRequest):
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()

    training_request = {
        "dataset": {
            "storage_type": "r2",
            "bucket": settings.S3_BUCKET,
            "path": dataset.storage_path,  # "datasets/{id}.zip"
        },
        "model": request.model_name,
        "hyperparameters": request.hyperparameters
    }

    # Training Service에 R2 경로만 전달
    response = requests.post(f"{TRAINING_SERVICE_URL}/train", json=training_request)

# Training Service
@app.post("/train")
async def train(request: TrainingRequest):
    # R2에서 직접 다운로드
    s3_client = boto3.client('s3', ...)
    s3_client.download_file(
        request.dataset.bucket,
        request.dataset.path,
        f"/tmp/{dataset_id}.zip"
    )

    # 압축 해제 & 학습
    extract_zip(f"/tmp/{dataset_id}.zip")
    train_model(...)
```

#### Option B: Backend 중계 방식

```
User → Backend → Training Service
        ↓ (다운로드)   ↑ (업로드)
        R2
```

**장점:**
- ✅ Training Service에 R2 credentials 불필요
- ✅ 중앙 집중식 권한 관리
- ✅ 데이터셋 전처리 가능 (Backend에서)

**단점:**
- ❌ Backend 디스크 사용 (100GB+ 가능)
- ❌ Backend가 병목 (동시 학습 시)
- ❌ 네트워크 대역폭 2배 사용 (R2→Backend→Training)
- ❌ 구현 복잡도 증가

**구현:**
```python
# Backend
@router.post("/training/start")
async def start_training(request: TrainingRequest):
    # 1. R2에서 다운로드 (Backend 로컬)
    dataset_zip = download_from_r2(request.dataset_id)
    # → /tmp/backend/{dataset_id}.zip (100GB+)

    # 2. Training Service에 업로드
    with open(dataset_zip, 'rb') as f:
        files = {'dataset': f}
        response = requests.post(
            f"{TRAINING_SERVICE_URL}/train",
            files=files,  # 100GB+ 업로드!
            data=training_request
        )
```

#### Option C: Hybrid (공유 볼륨)

```
Backend → Training Service
  ↓            ↓
Shared Volume ← R2 (한 번만 다운로드)
```

**장점:**
- ✅ 중복 다운로드 방지
- ✅ 여러 Training에서 재사용

**단점:**
- ⚠️ Kubernetes 환경에서 PVC 필요
- ⚠️ 볼륨 관리 복잡도
- ⚠️ Stateful 서비스 (확장성 저하)

### 2.2 권장 방식: Option A (직접 접근)

**이유:**
1. **단순성**: 추가 인프라 불필요
2. **확장성**: Training Service를 여러 개 실행 가능
3. **성능**: 불필요한 중간 단계 제거
4. **비용**: Backend 스토리지 비용 절감

**보안 구현:**
```yaml
# Railway/Kubernetes Secrets
training-service:
  env:
    AWS_S3_ENDPOINT_URL: https://4b324fd59e236f471c6ff612658615a0.r2.cloudflarestorage.com
    AWS_ACCESS_KEY_ID: <R2_READ_ONLY_TOKEN>  # 읽기 전용!
    AWS_SECRET_ACCESS_KEY: <SECRET>

    # Backend 권한 검증
    BACKEND_INTERNAL_URL: https://backend.railway.app/internal
    BACKEND_INTERNAL_AUTH_TOKEN: <TOKEN>
```

**플로우:**
```
1. User → Backend: "학습 시작"
2. Backend: 권한 검증 + Training Job 생성
3. Backend → Training Service: Job 정보 + Dataset 경로
4. Training Service → Backend: "권한 확인"
5. Backend → Training Service: "OK"
6. Training Service → R2: Dataset 다운로드 (읽기 전용)
7. Training Service: 학습 실행
```

---

## 3. 전체 데이터 플로우

### 3.1 시나리오 1: 완성된 데이터셋 업로드

```
┌─────────────────────────────────────────┐
│ Step 1: 사용자가 로컬에서 준비          │
└─────────────────────────────────────────┘
Local:
  my-dataset/
  ├── images/
  │   ├── 000001.jpg
  │   └── 000002.jpg
  ├── annotations.json  (모든 레이블 포함)
  └── meta.json

  → zip 압축: my-dataset.zip

┌─────────────────────────────────────────┐
│ Step 2: 프론트엔드 업로드               │
└─────────────────────────────────────────┘
Frontend:
  POST /api/v1/datasets/upload
  FormData: file = my-dataset.zip

Backend:
  1. DICE Format 검증
  2. Dataset ID 생성: platform-{name}-{uuid}
  3. R2 업로드: datasets/{dataset_id}.zip
  4. DB 등록: 메타데이터만

R2 결과:
  datasets/platform-my-dataset-abc123/
  └── my-dataset.zip  (73KB)

DB 결과:
  Dataset {
    id: "platform-my-dataset-abc123",
    storage_type: "r2",
    storage_path: "datasets/platform-my-dataset-abc123.zip",
    num_images: 100,
    format: "dice"
  }

┌─────────────────────────────────────────┐
│ Step 3: 학습 시작                       │
└─────────────────────────────────────────┘
User → Backend:
  POST /api/v1/training/start
  { "dataset_id": "platform-my-dataset-abc123" }

Backend → Training Service:
  POST {TRAINING_SERVICE_URL}/train
  {
    "dataset": {
      "bucket": "vision-platform-prod",
      "path": "datasets/platform-my-dataset-abc123.zip"
    }
  }

Training Service:
  1. R2에서 다운로드
     → /tmp/platform-my-dataset-abc123.zip

  2. 압축 해제
     → /tmp/platform-my-dataset-abc123/images/

  3. 학습 실행
     → Model checkpoint 생성

  4. Checkpoint R2 업로드
     → models/{job_id}/checkpoint.pth

  5. 임시 파일 삭제
```

### 3.2 시나리오 2: 레이블링 툴 사용 (미래 구현)

```
┌─────────────────────────────────────────┐
│ Step 1: 원본 이미지 업로드               │
└─────────────────────────────────────────┘
Frontend:
  POST /api/v1/datasets/create
  { "name": "My Dataset", "task_type": "object_detection" }

  → Dataset ID 생성

  POST /api/v1/datasets/{dataset_id}/images (100번 반복)
  FormData: file = image001.jpg

Backend:
  각 이미지를 개별 업로드
  → R2: datasets/{dataset_id}/images/000001.jpg

R2 결과:
  datasets/platform-my-dataset-abc123/
  ├── images/
  │   ├── 000001.jpg
  │   ├── 000002.jpg
  │   └── ...100개
  ├── annotations.json (빈 레이블)
  └── meta.json

┌─────────────────────────────────────────┐
│ Step 2: 레이블링 작업                   │
└─────────────────────────────────────────┘
Frontend (Labeling Tool):
  1. GET /api/v1/datasets/{dataset_id}/images
     → Backend: Presigned URLs 생성 (유효기간 1시간)
     → [url1, url2, ...]

  2. 브라우저에서 R2 이미지 직접 로드
     <img src={presignedUrl} />

  3. Canvas에서 레이블링
     - Bbox 그리기
     - Polygon 그리기
     - 클래스 선택

  4. 레이블 저장
     PATCH /api/v1/datasets/{dataset_id}/annotations/{image_id}
     { "annotation": { "bbox": [...], "class_id": 5 } }

Backend:
  1. R2에서 annotations.json 다운로드
  2. 해당 이미지 annotation 업데이트
  3. R2에 재업로드
     → annotations.json (1개 이미지 레이블 추가)

┌─────────────────────────────────────────┐
│ Step 3: 레이블 수정 (5장)               │
└─────────────────────────────────────────┘
Frontend:
  PATCH /api/v1/datasets/{dataset_id}/annotations/000005
  { "annotation": { "bbox": [수정된값] } }

Backend:
  1. annotations.json 다운로드
  2. 000005 레이블만 업데이트
  3. annotations.json 재업로드

비용: annotations.json 다운로드/업로드만 (몇 KB)

┌─────────────────────────────────────────┐
│ Step 4: 학습 준비 (Snapshot 생성)       │
└─────────────────────────────────────────┘
User → Backend:
  POST /api/v1/training/start
  { "dataset_id": "platform-my-dataset-abc123" }

Backend:
  1. 첫 학습인가? → Snapshot 생성 필요
  2. Snapshot 생성 Job 시작
     - R2에서 images/ + annotations.json 다운로드
     - 로컬에서 zip 압축
     - R2 업로드: datasets/{dataset_id}/snapshots/v1.zip
  3. 이후 학습은 snapshot 사용

R2 결과:
  datasets/platform-my-dataset-abc123/
  ├── images/ (원본, 개별 파일)
  ├── annotations.json (실시간 업데이트)
  └── snapshots/
      └── v1.zip (학습용, 한 번만 생성)

Training Service:
  1. Snapshot 다운로드
     → /tmp/platform-my-dataset-abc123.zip
  2. 압축 해제 & 학습
  3. 완료

┌─────────────────────────────────────────┐
│ Step 5: 레이블 재수정 후 재학습         │
└─────────────────────────────────────────┘
User: 레이블 5개 더 수정
  → annotations.json만 업데이트

User: 재학습 시작
Backend:
  1. annotations.json이 변경됨 감지
  2. 새 Snapshot 생성
     → snapshots/v2.zip
  3. Training Service에 v2.zip 경로 전달

Training Service:
  v2.zip 다운로드 & 학습
```

---

## 4. 압축/해제 타이밍 요약

| 작업 | 압축 여부 | 수행 위치 | 타이밍 |
|------|----------|----------|--------|
| **완성 데이터셋 업로드** | Zip | 사용자 로컬 | 업로드 전 |
| **개별 이미지 업로드** | 없음 | - | 즉시 R2 |
| **레이블링** | 없음 | - | annotations.json만 업데이트 |
| **레이블 조회** | 없음 | - | Presigned URL 직접 접근 |
| **학습용 Snapshot 생성** | Zip | Backend | 첫 학습 시작 전 |
| **학습 실행** | Unzip | Training Service | 학습 시작 직후 |
| **Checkpoint 저장** | 없음 | Training Service | Epoch마다 |

---

## 5. R2 스토리지 구조

### 5.1 완성 데이터셋 (현재 구현)

```
vision-platform-prod/
├── datasets/
│   ├── platform-imagenet-10-{uuid}/
│   │   └── dataset.zip          (73 KB)
│   ├── platform-coco8-{uuid}/
│   │   └── dataset.zip          (203 KB)
│   └── ...
└── models/
    └── {job_id}/
        ├── checkpoint-epoch-10.pth
        └── final-model.pth
```

### 5.2 레이블링 툴 지원 (미래 구현)

```
vision-platform-prod/
├── datasets/
│   └── platform-my-dataset-{uuid}/
│       ├── images/              (원본, 개별 파일)
│       │   ├── 000001.jpg
│       │   ├── 000002.jpg
│       │   └── ... (100개)
│       ├── annotations.json     (실시간 업데이트)
│       ├── meta.json
│       └── snapshots/           (학습용, 한 번만 생성)
│           ├── v1.zip           (첫 학습)
│           ├── v2.zip           (레이블 수정 후)
│           └── latest -> v2.zip
└── models/
    └── ...
```

---

## 6. 비용 분석

### 6.1 완성 데이터셋 방식

**업로드:**
- Zip 파일 1회 업로드: 100MB × $0.001/GB = $0.0001
- 총: **$0.0001**

**학습:**
- Zip 다운로드 (Training Service): 100MB × $0.001/GB = $0.0001
- 총: **$0.0001**

**재학습:**
- 동일한 zip 재다운로드: $0.0001
- 총: **$0.0001**

### 6.2 레이블링 툴 방식

**업로드 (100장):**
- 개별 이미지 100회: 100MB × $0.001/GB = $0.0001
- annotations.json: 무시 가능
- 총: **$0.0001**

**레이블링 (100장):**
- Presigned URL (읽기): 100회 × 무료 = $0
- annotations.json 업데이트 100회: ~10KB × 100 = 1MB → 무시 가능
- 총: **~$0**

**Snapshot 생성:**
- 다운로드 (images + annotations): 100MB × $0.001/GB = $0.0001
- 업로드 (zip): 100MB × $0.001/GB = $0.0001
- 총: **$0.0002**

**학습:**
- Snapshot 다운로드: $0.0001
- 총: **$0.0001**

**레이블 수정 후 재학습:**
- annotations.json 업데이트: 무시 가능
- 새 Snapshot 생성: $0.0002
- 새 Snapshot 다운로드: $0.0001
- 총: **$0.0003**

**결론:** 레이블링 툴 방식도 매우 저렴 (100장 기준 $0.001 미만)

---

## 7. 구현 우선순위

### Phase 1: 현재 (완성 데이터셋만) ✅
- [x] Zip 업로드 API
- [x] R2 저장
- [x] DB 메타데이터
- [x] Training Service R2 직접 접근

### Phase 2: 레이블링 툴 지원 (미래)
- [ ] 개별 이미지 업로드 API
- [ ] Presigned URL 생성 API
- [ ] Annotation 부분 업데이트 API
- [ ] Snapshot 생성 시스템
- [ ] 레이블러 UI 구현

### Phase 3: 고급 기능
- [ ] 데이터셋 버전 관리
- [ ] Collaborative 레이블링 (여러 사용자)
- [ ] Auto-labeling (AI 보조)
- [ ] 데이터셋 통계/시각화

---

## 8. 주요 결정사항

### 8.1 Presigned URL 사용 ✅
- **결정**: 레이블러에서 Presigned URL로 이미지 직접 로드
- **이유**: 성능, 서버 부하 감소, CDN 활용
- **단점**: 없음

### 8.2 Training Service R2 직접 접근 ✅
- **결정**: Training Service가 R2에서 직접 다운로드
- **이유**: 단순성, 확장성, Backend 부하 감소
- **트레이드오프**: Training Service에 R2 credentials 필요 (읽기 전용)

### 8.3 Snapshot 시스템 (미래)
- **결정**: 학습 시작 시 Snapshot 생성
- **이유**: 개별 파일 + 실시간 업데이트 지원하면서도 학습 효율성 유지
- **트레이드오프**: 첫 학습 시작 시간 증가 (zip 생성 시간)

### 8.4 Zip 저장 방식
- **결정**: R2에 zip 파일 그대로 저장
- **이유**: 저장 공간 절약, 빠른 전송
- **트레이드오프**: 개별 파일 접근 불가 (Snapshot 시스템으로 해결)

---

## 9. 참고 자료

- **DICE Format 사양**: `docs/datasets/PLATFORM_DATASET_FORMAT.md`
- **Backend API**: `mvp/backend/app/api/datasets.py`
- **R2 Storage 유틸**: `mvp/backend/app/utils/r2_storage.py`
- **환경 변수**: `.env.r2`
