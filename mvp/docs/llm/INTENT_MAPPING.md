# Intent Mapping Reference

## 개요

사용자의 자연어 발화를 구체적인 기능으로 매핑하는 참조 가이드입니다.

**목적**:
- LLM 프롬프트 엔지니어링 가이드
- 새로운 인텐트 추가 시 참조
- 테스트 케이스 작성 기준

---

## 인텐트 분류 체계

```
ROOT
├── TRAINING (학습 관련)
│   ├── CREATE (생성)
│   ├── CONTROL (제어)
│   ├── MONITOR (모니터링)
│   └── MANAGE (관리)
├── INFERENCE (추론 관련)
│   ├── QUICK (빠른 추론)
│   ├── BATCH (배치 추론)
│   └── TEST (테스트 실행)
├── DATASET (데이터셋 관련)
│   ├── ANALYZE (분석)
│   ├── VALIDATE (검증)
│   └── LIST (목록)
├── MODEL (모델 관련)
│   ├── SEARCH (검색)
│   ├── COMPARE (비교)
│   ├── INFO (정보)
│   └── RECOMMEND (추천)
├── PROJECT (프로젝트 관련)
│   ├── CREATE (생성)
│   ├── MANAGE (관리)
│   └── COLLABORATE (협업)
└── RESULTS (결과 관련)
    ├── VIEW (조회)
    ├── ANALYZE (분석)
    └── EXPORT (내보내기)
```

---

## 1. TRAINING Intents

### 1.1 TRAINING.CREATE - 학습 생성

**목적**: 새로운 학습 작업 생성

#### **발화 패턴**

| 패턴 | 예시 | 추출 정보 |
|------|------|----------|
| `{모델}로 {작업} 학습` | "ResNet50으로 분류 학습해줘" | model: resnet50, task: classification |
| `{작업} 모델 만들기` | "고양이 검출 모델 만들어줘" | task: detection, subject: cat |
| `{데이터셋}으로 학습` | "C:/datasets/cats 로 학습하고 싶어요" | dataset_path: C:/datasets/cats |
| `{모델} + {데이터셋}` | "YOLO로 C:/data/defect 학습" | model: yolo, dataset_path: C:/data/defect |
| 상세 설정 포함 | "EfficientNet, lr 0.0005, 150 epoch" | model: efficientnet, lr: 0.0005, epochs: 150 |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 기본값 | 예시 |
|--------|------|------|--------|------|
| `model_name` | ✅ | str | - | "resnet50", "yolov8m" |
| `task_type` | ✅ | str | - | "classification", "detection" |
| `dataset_path` | ✅ | str | - | "C:/datasets/cats" |
| `framework` | ❌ | str | 자동 추론 | "timm", "ultralytics" |
| `epochs` | ❌ | int | 100 | 50, 200 |
| `batch_size` | ❌ | int | 32 | 16, 64 |
| `learning_rate` | ❌ | float | 0.001 | 0.0001, 0.01 |
| `num_classes` | ❌ | int | 자동 감지 | 3, 10 |
| `project_id` | ❌ | int | null | 5 |
| `experiment_name` | ❌ | str | null | "baseline_v1" |

#### **응답 Action**

**Case 1: 정보 충분**
```python
Action: CONFIRM_TRAINING
Data: {
    "config": <완전한 학습 설정>,
    "summary": <설정 요약>,
    "estimated_time": <예상 소요 시간>
}
Message: "다음 설정으로 학습을 시작하시겠어요?"
```

**Case 2: 정보 부족**
```python
Action: ASK_CLARIFICATION
Data: {
    "missing_fields": ["dataset_path", "num_classes"],
    "questions": [
        "데이터셋 경로를 알려주세요.",
        "몇 개의 클래스를 분류하시나요?"
    ]
}
```

**Case 3: 데이터셋 분석 필요**
```python
Action: ANALYZE_DATASET
Data: {
    "dataset_path": <경로>
}
Message: "데이터셋을 먼저 분석하겠습니다..."
```

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `POST /training/jobs` | `create_training_job(...)` |
| `POST /training/jobs/{id}/start` | `start_training(job_id=...)` |

#### **예제 대화**

```
사용자: "ResNet50으로 고양이 3종류 분류 학습해줘"

AI 분석:
- Intent: TRAINING.CREATE
- 추출:
  - model_name: "resnet50"
  - task_type: "classification"
  - num_classes: 3
- 누락:
  - dataset_path ❌

AI 응답:
Action: ASK_CLARIFICATION
"좋습니다! ResNet50으로 3개 클래스 분류 모델을 만들어드릴게요.
 데이터셋은 어디에 있나요?"

사용자: "C:/datasets/cats 폴더에 있어"

AI 분석:
- Intent: PROVIDE_INFO
- 추출:
  - dataset_path: "C:/datasets/cats"
- 컨텍스트에서 병합:
  - model_name: "resnet50" (이전)
  - task_type: "classification" (이전)
  - num_classes: 3 (이전)
  - dataset_path: "C:/datasets/cats" (신규)

AI 응답:
Action: ANALYZE_DATASET
[데이터셋 분석 API 호출]

결과:
- 포맷: ImageFolder
- 클래스: [persian, siamese, bengal]
- 총 이미지: 1,200장

Action: CONFIRM_TRAINING
"데이터셋 분석 완료!
 - 클래스: persian, siamese, bengal
 - 이미지: 각 약 400장

 설정 요약:
 - 모델: ResNet50
 - Epochs: 100 (기본값)
 - Learning Rate: 0.001
 - Batch Size: 32

 이대로 학습을 시작하시겠어요?"

사용자: "네"

AI 분석:
- Intent: CONFIRM

AI 응답:
Action: START_TRAINING
[학습 생성 및 시작 API 호출]
"학습을 시작했습니다! (Job ID: 12345)"
```

---

### 1.2 TRAINING.CONTROL - 학습 제어

**목적**: 학습 작업 시작/중지/재개

#### **발화 패턴**

| 동작 | 패턴 | 예시 |
|------|------|------|
| **시작** | `학습 시작` | "학습 시작해줘" |
| | `{Job ID} 시작` | "12345번 작업 시작" |
| | `begin/start training` | "start training" |
| **중지** | `학습 중지/멈춤` | "학습 중지해줘" |
| | `{Job ID} 중지` | "12345번 중지" |
| | `stop/cancel` | "cancel training" |
| **재개** | `학습 재개/계속` | "학습 재개해줘" |
| | `{Job ID} 재개` | "12340번 재개" |
| | `resume from checkpoint` | "resume from epoch 50" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `action` | ✅ | str | "start", "stop", "resume" |
| `job_id` | ❌ | int | 명시적 지정. 없으면 컨텍스트에서 추론 |
| `checkpoint_path` | ❌ | str | 재개 시 체크포인트 (선택) |
| `save_checkpoint` | ❌ | bool | 중지 시 체크포인트 저장 여부 (기본: true) |

#### **컨텍스트 추론**

```python
# 컨텍스트 우선순위
1. 명시적 Job ID: "12345번 중지" → job_id = 12345
2. 현재 활성 작업: context.current_training_job → job_id = context.current_training_job
3. 가장 최근 작업: context.last_created_job → job_id = context.last_created_job
4. 없으면 목록 제시: "어떤 작업을 중지하시겠어요?"
```

#### **API/Tool 매핑**

| 동작 | Gemini Track | MCP Track |
|------|--------------|-----------|
| 시작 | `POST /training/jobs/{id}/start` | `start_training(job_id, ...)` |
| 중지 | `POST /training/jobs/{id}/cancel` | `stop_training(job_id, ...)` |
| 재개 | `POST /training/jobs/{id}/restart` | `start_training(job_id, resume_from_checkpoint=...)` |

#### **예제 대화**

```
사용자: "학습 중지해줘"

AI 분석:
- Intent: TRAINING.CONTROL
- Action: stop
- job_id: 컨텍스트에서 추론

컨텍스트 확인:
- current_training_job: 12345 (현재 실행 중인 작업)

AI 응답:
Action: STOP_TRAINING
[API 호출: POST /training/jobs/12345/cancel]
"학습 작업 #12345를 중지했습니다.
 체크포인트가 저장되었습니다: outputs/job_12345/checkpoints/epoch_35.pt"

컨텍스트 업데이트:
- current_training_job: null
- last_stopped_job: 12345
```

---

### 1.3 TRAINING.MONITOR - 학습 모니터링

**목적**: 학습 진행 상황 조회

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `현재 상태/진행 상황` | "현재 학습 진행 상황 알려줘" |
| `{Job ID} 상태` | "12345번 작업 어떻게 돼?" |
| `loss/accuracy` | "loss 그래프 보여줘" |
| `남은 시간` | "학습 언제 끝나?" |
| `실행 중인 작업` | "지금 학습 중인 거 있어?" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `query_type` | ✅ | str | "status", "metrics", "logs", "eta" |
| `job_id` | ❌ | int | 없으면 컨텍스트 추론 |
| `metric_name` | ❌ | str | "loss", "accuracy" 등 |

#### **API/Tool 매핑**

| 쿼리 타입 | Gemini Track | MCP Track |
|----------|--------------|-----------|
| 상태 | `GET /training/jobs/{id}/status` | `get_training_status(job_id)` |
| 메트릭 | `GET /training/jobs/{id}/metrics` | Resource: `training://jobs/{id}/metrics` |
| 로그 | `GET /training/jobs/{id}/logs` | Resource: `training://jobs/{id}/logs` |

#### **예제 대화**

```
사용자: "현재 학습 진행 상황 알려줘"

AI 분석:
- Intent: TRAINING.MONITOR
- Query Type: status

AI 응답:
Action: SHOW_TRAINING_STATUS
[API: GET /training/jobs/12345/status]

"진행 중인 학습 작업:
 - Job ID: 12345
 - 모델: ResNet-50
 - 진행률: Epoch 35/100 (35%)
 - 현재 Accuracy: 0.823
 - 현재 Loss: 0.234
 - 예상 남은 시간: 45분
 - 상태: Training

 📊 최근 10 epoch의 loss 변화:
 Epoch 26-35: 0.456 → 0.234 (-48%)

 최근 loss가 거의 변하지 않고 있습니다 (0.234 → 0.231).
 Early stopping을 고려하시겠어요?"
```

---

### 1.4 TRAINING.MANAGE - 학습 관리

**목적**: 학습 작업 목록 조회, 비교, 삭제

#### **발화 패턴**

| 동작 | 패턴 | 예시 |
|------|------|------|
| **목록** | `학습 목록/리스트` | "학습 작업 목록 보여줘" |
| | `실행 중인 작업` | "지금 실행 중인 거 있어?" |
| | `완료된 작업` | "완료된 학습들 보여줘" |
| **비교** | `{Job IDs} 비교` | "12340번이랑 12345번 비교해줘" |
| | `최고 성능` | "가장 accuracy 높은 모델 찾아줘" |
| **삭제** | `{Job ID} 삭제` | "12340번 삭제해줘" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `action` | ✅ | str | "list", "compare", "delete" |
| `filter_status` | ❌ | str | "running", "completed", "failed" |
| `filter_project` | ❌ | int | 프로젝트 ID |
| `job_ids` | ❌ | list[int] | 비교할 작업 ID들 |
| `sort_by` | ❌ | str | "accuracy", "created_at" |

#### **API/Tool 매핑**

| 동작 | Gemini Track | MCP Track |
|------|--------------|-----------|
| 목록 | `GET /training/jobs?status=...` | `list_training_jobs(status=...)` |
| 비교 | Frontend 로직 | `compare_experiments(job_ids=...)` |
| 삭제 | `DELETE /training/jobs/{id}` | `delete_training_job(job_id)` |

#### **예제 대화**

```
사용자: "완료된 학습들 중에서 mAP가 가장 높은 거 찾아줘"

AI 분석:
- Intent: TRAINING.MANAGE
- Action: list + filter + sort
- Filter: status = "completed"
- Sort: mAP (descending)

AI 응답:
Action: LIST_EXPERIMENTS
[API: GET /training/jobs?status=completed&sort_by=primary_metric&order=desc]

"완료된 학습 작업 중 mAP 상위 3개:

1. Job #12340 - YOLOv8m
   - mAP50: 0.876
   - mAP50-95: 0.654
   - 학습 시간: 2시간 15분
   - 완료: 2025-10-28

2. Job #12335 - YOLOv8s
   - mAP50: 0.843
   - mAP50-95: 0.621
   - 학습 시간: 1시간 40분
   - 완료: 2025-10-27

3. Job #12330 - YOLO11m
   - mAP50: 0.821
   - mAP50-95: 0.598
   - 학습 시간: 2시간 30분
   - 완료: 2025-10-26

가장 성능이 좋은 모델은 #12340 (YOLOv8m)입니다.
이 모델로 추론을 실행하시겠어요?"
```

---

## 2. INFERENCE Intents

### 2.1 INFERENCE.QUICK - 빠른 추론

**목적**: 단일 이미지 빠른 추론

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `이미지 {동작}` | "이 이미지 분류해줘" |
| `{이미지} 추론` | "test.jpg 추론 돌려줘" |
| `검출/분류/분석` | "이 사진에서 객체 검출해줘" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `image_path` | ✅ | str | 이미지 경로 또는 세션 업로드 ID |
| `job_id` | ❌ | int | 사용할 모델. 없으면 컨텍스트에서 추론 |
| `confidence_threshold` | ❌ | float | 신뢰도 임계값 (기본: 0.5) |

#### **컨텍스트 추론**

```python
# 모델 선택 우선순위
1. 명시적 Job ID: "12340번 모델로 추론" → job_id = 12340
2. 최근 완료된 작업: context.last_completed_job → job_id = ...
3. 가장 성능 좋은 모델: "가장 accuracy 높은 모델" → job_id = best
4. 없으면 선택 요청: "어떤 모델을 사용하시겠어요?"
```

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `POST /inference/quick` | `run_quick_inference(job_id, image=...)` |
| `POST /inference/upload-image` | (이미지 업로드) |

#### **예제 대화**

```
사용자: [이미지 업로드] "이 이미지 분류해줘"

AI 분석:
- Intent: INFERENCE.QUICK
- image_path: session_upload_123
- job_id: 컨텍스트 추론 필요

컨텍스트 확인:
- last_completed_job: 12340 (ResNet50, Accuracy: 0.95)

AI 응답:
Action: RUN_QUICK_INFERENCE
[API: POST /inference/quick]
Request: {
    "job_id": 12340,
    "image_session_id": "session_upload_123"
}

결과:
{
    "predictions": [
        {"class": "persian", "confidence": 0.92},
        {"class": "siamese", "confidence": 0.06},
        {"class": "bengal", "confidence": 0.02}
    ]
}

"분류 결과:
 1. Persian 고양이 (92% 확신)
 2. Siamese 고양이 (6%)
 3. Bengal 고양이 (2%)

 모델: ResNet50 (#12340)
 추론 시간: 0.05초"
```

---

### 2.2 INFERENCE.BATCH - 배치 추론

**목적**: 여러 이미지 일괄 추론

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `폴더 추론` | "C:/test_images 폴더 추론해줘" |
| `여러 이미지` | "이 이미지들 전부 분류해줘" |
| `배치 처리` | "test 폴더 전체에 대해 추론 돌려줘" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `image_source` | ✅ | str | 폴더 경로 또는 이미지 목록 |
| `job_id` | ❌ | int | 사용할 모델 |
| `save_results` | ❌ | bool | 결과 저장 여부 (기본: true) |

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `POST /inference/jobs` | `run_batch_inference(job_id, images=...)` |

#### **예제 대화**

```
사용자: "C:/test_images 폴더의 모든 이미지에 대해 추론 돌려줘"

AI 분석:
- Intent: INFERENCE.BATCH
- image_source: "C:/test_images"

AI 응답:
Action: RUN_BATCH_INFERENCE
[API: POST /inference/jobs]
Request: {
    "training_job_id": 12340,
    "image_dir": "C:/test_images"
}

"배치 추론을 시작했습니다. (Inference Job ID: inf_789)
 - 이미지 수: 45장
 - 모델: ResNet50 (#12340)
 - 예상 소요 시간: 약 2분

 완료되면 알려드리겠습니다!"

[2분 후]

"배치 추론이 완료되었습니다!
 - 총 45장 처리
 - 평균 신뢰도: 0.87
 - 소요 시간: 1분 52초

 결과 요약:
 - Persian: 18장
 - Siamese: 15장
 - Bengal: 12장

 결과를 내보내시겠어요?"
```

---

### 2.3 INFERENCE.TEST - 테스트 실행

**목적**: 레이블된 데이터셋에 대해 추론 + 메트릭 계산

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `테스트 실행` | "테스트 데이터셋으로 평가해줘" |
| `정확도 측정` | "validation 데이터로 accuracy 확인해줘" |
| `성능 평가` | "이 모델 성능 테스트해줘" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `test_dataset_path` | ✅ | str | 테스트 데이터셋 경로 |
| `job_id` | ❌ | int | 평가할 모델 |

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `POST /test_inference/test/runs` | `run_test(job_id, test_dataset=...)` |

---

## 3. DATASET Intents

### 3.1 DATASET.ANALYZE - 데이터셋 분석

**목적**: 데이터셋 구조, 품질 분석

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `데이터셋 분석` | "데이터셋 분석해줘" |
| `{경로} 확인/분석` | "C:/datasets/my_data 구조 확인해줘" |
| `클래스 분포` | "클래스별 이미지 수 알려줘" |
| `품질 체크` | "데이터 품질 괜찮아?" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `dataset_path` | ✅ | str | 데이터셋 경로 |
| `deep_analysis` | ❌ | bool | 상세 분석 여부 (기본: false) |

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `POST /datasets/analyze` | `analyze_dataset(dataset_path)` |

#### **예제 대화**

```
사용자: "C:/datasets/defect 데이터셋 분석해줘"

AI 분석:
- Intent: DATASET.ANALYZE
- dataset_path: "C:/datasets/defect"

AI 응답:
Action: ANALYZE_DATASET
[API: POST /datasets/analyze]

"데이터셋 분석 중..."

결과:
{
    "format": "ImageFolder",
    "num_classes": 3,
    "classes": ["scratch", "dent", "normal"],
    "total_images": 1200,
    "class_distribution": {
        "scratch": 400,
        "dent": 350,
        "normal": 450
    },
    "imbalance_ratio": 1.29,
    "quality_score": 85,
    "recommendations": [
        "클래스 불균형이 약간 있습니다. Class weights 적용을 권장합니다.",
        "이미지 해상도가 일정하지 않습니다. Resize 설정을 확인하세요."
    ]
}

"📊 데이터셋 분석 결과:

**기본 정보:**
- 포맷: ImageFolder (분류 작업)
- 클래스: 3개 (scratch, dent, normal)
- 총 이미지: 1,200장

**클래스 분포:**
- scratch: 400장 (33%)
- dent: 350장 (29%)
- normal: 450장 (38%)

**품질 평가:**
- 품질 점수: 85/100 (양호)
- 불균형 비율: 1.29 (낮음)
- 손상된 파일: 없음 ✅

**권장 사항:**
⚠️ 클래스 불균형이 약간 있습니다. Class weights 적용을 권장합니다.
⚠️ 이미지 해상도가 일정하지 않습니다. Resize 설정을 확인하세요.

**추천 모델:**
1. EfficientNet-B0 (정확도와 속도 균형)
2. ResNet-50 (높은 정확도)

이 데이터셋으로 바로 학습을 시작하시겠어요?"
```

---

## 4. MODEL Intents

### 4.1 MODEL.SEARCH - 모델 검색

**목적**: 조건에 맞는 모델 찾기

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `{작업} 모델` | "객체 검출 모델 뭐 있어?" |
| `{프레임워크} 모델` | "YOLO 모델들 보여줘" |
| `{특성} 모델` | "빠른 모델 추천해줘" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `task_type` | ❌ | str | "classification", "detection" 등 |
| `framework` | ❌ | str | "timm", "ultralytics", "transformers" |
| `tags` | ❌ | list[str] | ["fast", "accurate", "lightweight"] |

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `GET /models/list?task_type=...` | `search_models(task_type=...)` |

---

### 4.2 MODEL.INFO - 모델 정보

**목적**: 특정 모델의 상세 정보

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `{모델} 정보` | "ResNet50 정보 알려줘" |
| `{모델} 장단점` | "EfficientNet 장단점 뭐야?" |
| `{모델} 벤치마크` | "YOLO 성능 어때?" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `model_name` | ✅ | str | 모델 이름 |
| `framework` | ❌ | str | 프레임워크 (자동 추론) |

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `GET /models/{framework}/{name}/guide` | `get_model_guide(framework, model_name)` |

---

### 4.3 MODEL.COMPARE - 모델 비교

**목적**: 여러 모델 비교

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `{모델들} 비교` | "ResNet50이랑 EfficientNet 비교해줘" |
| `뭐가 더 좋아?` | "YOLO랑 Faster R-CNN 중에 뭐가 나아?" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `models` | ✅ | list[str] | 비교할 모델 이름들 |

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `GET /models/compare?models=...` | `compare_models(model_specs=[...])` |

---

### 4.4 MODEL.RECOMMEND - 모델 추천

**목적**: 상황에 맞는 모델 추천

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `추천해줘` | "어떤 모델이 좋을까?" |
| `{작업} 추천` | "객체 검출에 뭐 쓰면 돼?" |
| `{조건} 모델` | "빠르고 정확한 모델 추천" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `task_type` | ✅ | str | 작업 유형 |
| `dataset_size` | ❌ | int | 데이터셋 크기 (컨텍스트에서 추론 가능) |
| `priority` | ❌ | str | "speed", "accuracy", "balanced" |

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| LLM 추론 + Model Registry | `recommend_model(task_type, dataset_size, priority)` 또는 MCP Prompt |

---

## 5. PROJECT Intents

### 5.1 PROJECT.CREATE - 프로젝트 생성

**목적**: 새 프로젝트 생성

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `프로젝트 생성` | "새 프로젝트 만들어줘" |
| `{이름} 프로젝트` | "불량 검사 프로젝트 만들고 싶어" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `project_name` | ✅ | str | 프로젝트 이름 |
| `description` | ❌ | str | 설명 |

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `POST /projects` | `create_project(name, description)` |

---

## 6. RESULTS Intents

### 6.1 RESULTS.VIEW - 결과 조회

**목적**: 학습/검증/추론 결과 확인

#### **발화 패턴**

| 패턴 | 예시 |
|------|------|
| `결과 보여줘` | "validation 결과 보여줘" |
| `confusion matrix` | "confusion matrix 확인해줘" |
| `{Job ID} 결과` | "12345번 결과 어때?" |

#### **추출 엔티티**

| 엔티티 | 필수 | 타입 | 설명 |
|--------|------|------|------|
| `result_type` | ✅ | str | "validation", "test", "inference" |
| `job_id` | ❌ | int | 작업 ID (컨텍스트 추론) |
| `visualization` | ❌ | str | "confusion_matrix", "pr_curve" 등 |

#### **API/Tool 매핑**

| Gemini Track | MCP Track |
|--------------|-----------|
| `GET /validation/jobs/{id}/results` | `get_validation_results(job_id)` |
| `GET /test_inference/test/runs/{id}` | `get_test_results(test_run_id)` |

---

## 7. 복합 Intent 처리

### 7.1 Sequential Intents (순차 실행)

**예시**: "데이터셋 분석하고, 모델 추천받고, 바로 학습 시작해줘"

**분해**:
1. DATASET.ANALYZE
2. MODEL.RECOMMEND
3. TRAINING.CREATE + TRAINING.CONTROL (시작)

**처리**:
```python
# Gemini Track
1. ANALYZE_DATASET → dataset_info 저장
2. RECOMMEND_MODELS(dataset_info) → recommended_model 저장
3. CONFIRM_TRAINING(recommended_model, dataset_info)
4. 사용자 확인 후 START_TRAINING

# MCP Track
LLM이 자동으로 도구 체이닝:
1. analyze_dataset(path) → analysis_result
2. recommend_model(
     task_type=analysis_result.task_type,
     dataset_size=analysis_result.total_images
   ) → recommendation
3. create_training_job(
     model_name=recommendation.model,
     dataset_path=path,
     ...
   ) → job
4. start_training(job_id=job.job_id)
```

### 7.2 Conditional Intents (조건부)

**예시**: "학습 중이면 중지하고, 아니면 시작해줘"

**처리**:
```python
# 1. 현재 상태 확인
status = get_training_status(context.current_training_job)

# 2. 조건 평가
if status.status == "running":
    action = stop_training(job_id)
else:
    action = start_training(job_id)
```

### 7.3 Comparative Intents (비교)

**예시**: "지난 3개 학습 중에서 가장 좋은 거 찾아서 추론 돌려줘"

**처리**:
```python
# 1. 학습 목록 조회
jobs = list_training_jobs(limit=3, sort_by="accuracy", order="desc")

# 2. 최고 성능 선택
best_job = jobs[0]

# 3. 추론 실행
inference_result = run_quick_inference(job_id=best_job.id, image=...)
```

---

## 8. 컨텍스트 관리 전략

### 8.1 엔티티 추적

```python
class ConversationContext:
    # 현재 활성 엔티티
    current_training_job: int | None = None
    current_inference_job: int | None = None
    current_project: int | None = None
    current_dataset: str | None = None

    # 최근 작업
    last_created_job: int | None = None
    last_completed_job: int | None = None
    last_stopped_job: int | None = None

    # 임시 데이터
    partial_config: dict = {}
    pending_questions: list = []

    # 사용자 선호도
    preferred_models: dict[str, str] = {}  # task_type -> model_name
    typical_epochs: int = 100
    typical_batch_size: int = 32
```

### 8.2 컨텍스트 업데이트 규칙

**학습 생성 시**:
```python
context.last_created_job = job.id
context.current_training_job = job.id
context.current_project = job.project_id
context.current_dataset = job.dataset_path
```

**학습 시작 시**:
```python
context.current_training_job = job.id
```

**학습 완료 시**:
```python
context.current_training_job = None
context.last_completed_job = job.id
```

**학습 중지 시**:
```python
context.current_training_job = None
context.last_stopped_job = job.id
```

### 8.3 모호성 해결 전략

**Case 1: Job ID 누락**
```
사용자: "학습 중지해줘"

해결:
1. current_training_job 확인 → 있으면 사용
2. 없으면 실행 중인 작업 조회 → 1개면 사용, 여러 개면 선택 요청
3. 없으면 "현재 실행 중인 학습이 없습니다"
```

**Case 2: 모델 미지정**
```
사용자: "추론 돌려줘"

해결:
1. last_completed_job 확인 → 있으면 사용
2. 없으면 가장 최근 성공한 작업 조회
3. 여러 개면 선택 요청: "어떤 모델을 사용하시겠어요?"
```

**Case 3: 데이터셋 미지정**
```
사용자: "학습 시작해줘"

해결:
1. partial_config.dataset_path 확인
2. context.current_dataset 확인
3. 없으면 질문: "데이터셋 경로를 알려주세요"
```

---

## 9. 에러 및 예외 처리

### 9.1 일반적인 에러

| 에러 상황 | 사용자 메시지 | 대안 제시 |
|-----------|--------------|----------|
| 데이터셋 경로 없음 | "데이터셋을 찾을 수 없습니다" | "경로를 다시 확인해주세요" |
| 권한 없음 | "이 작업에 대한 권한이 없습니다" | "프로젝트 소유자에게 문의하세요" |
| 작업 실행 중 | "이미 학습이 실행 중입니다" | "중지 후 다시 시작하시겠어요?" |
| 리소스 부족 | "GPU 메모리가 부족합니다" | "Batch size를 줄여보세요 (현재: 32 → 권장: 16)" |

### 9.2 Fallback 전략

**LLM 실패 시**:
```python
# 1. Regex 기반 Fallback 파서 시도
fallback_result = regex_parser.parse(user_message)

# 2. 실패 시 명확한 에러 메시지
if not fallback_result:
    return "죄송합니다. 요청을 이해하지 못했습니다. 다시 설명해주시겠어요?"
```

---

## 10. 테스트 케이스

### 10.1 Intent Recognition Tests

```python
test_cases = [
    # TRAINING.CREATE
    {
        "input": "ResNet50으로 고양이 분류 학습해줘",
        "expected_intent": "TRAINING.CREATE",
        "expected_entities": {
            "model_name": "resnet50",
            "task_type": "classification"
        }
    },

    # TRAINING.CONTROL
    {
        "input": "학습 중지",
        "expected_intent": "TRAINING.CONTROL",
        "expected_entities": {
            "action": "stop"
        }
    },

    # DATASET.ANALYZE
    {
        "input": "C:/datasets/cats 분석해줘",
        "expected_intent": "DATASET.ANALYZE",
        "expected_entities": {
            "dataset_path": "C:/datasets/cats"
        }
    },

    # MODEL.RECOMMEND
    {
        "input": "객체 검출에 뭐 쓰면 좋아?",
        "expected_intent": "MODEL.RECOMMEND",
        "expected_entities": {
            "task_type": "object_detection"
        }
    },

    # 복합 Intent
    {
        "input": "데이터셋 분석하고 모델 추천해줘",
        "expected_intent": ["DATASET.ANALYZE", "MODEL.RECOMMEND"],
        "expected_flow": "sequential"
    }
]
```

---

## 11. 프롬프트 엔지니어링 가이드

### 11.1 System Prompt 구조

```
You are an AI assistant for a computer vision training platform.

**Your capabilities:**
- Create and manage training jobs
- Run inference on images
- Analyze datasets
- Search and recommend models
- Manage projects and experiments

**Available intents:**
{인텐트 목록 및 설명}

**Available tools:**
{도구 목록 및 사용법}

**Context management:**
- Track current_training_job, current_project, etc.
- Remember user preferences
- Provide proactive suggestions

**Response guidelines:**
- Always respond in Korean
- Be concise and helpful
- Ask clarifying questions when needed
- Provide context and reasoning
- Suggest next steps
```

### 11.2 Few-Shot Examples

```
User: "ResNet50으로 학습해줘"
Assistant: {
    "intent": "TRAINING.CREATE",
    "action": "ASK_CLARIFICATION",
    "questions": ["데이터셋 경로를 알려주세요", "몇 개의 클래스를 분류하시나요?"],
    "message": "ResNet50으로 학습을 설정하겠습니다. 몇 가지 확인할게요..."
}

User: "C:/datasets/cats 이고 3개 클래스야"
Assistant: {
    "intent": "PROVIDE_INFO",
    "action": "ANALYZE_DATASET",
    "dataset_path": "C:/datasets/cats",
    "message": "데이터셋을 분석하겠습니다..."
}
```

---

## 부록: Quick Reference

### 주요 Intent → API 매핑 요약

| Intent | API Endpoint | MCP Tool |
|--------|-------------|----------|
| TRAINING.CREATE | `POST /training/jobs` | `create_training_job` |
| TRAINING.START | `POST /training/jobs/{id}/start` | `start_training` |
| TRAINING.STOP | `POST /training/jobs/{id}/cancel` | `stop_training` |
| TRAINING.STATUS | `GET /training/jobs/{id}/status` | `get_training_status` |
| INFERENCE.QUICK | `POST /inference/quick` | `run_quick_inference` |
| INFERENCE.BATCH | `POST /inference/jobs` | `run_batch_inference` |
| DATASET.ANALYZE | `POST /datasets/analyze` | `analyze_dataset` |
| MODEL.SEARCH | `GET /models/list` | `search_models` |
| MODEL.INFO | `GET /models/{fw}/{name}/guide` | `get_model_guide` |
| MODEL.RECOMMEND | LLM Inference | `recommend_model` |

