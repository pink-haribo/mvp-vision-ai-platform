# 사용자 플로우 시나리오 상세 문서

> **작성일**: 2025-11-06
> **Version**: 1.0
> **Status**: Production

## 목차

1. [개요](#개요)
2. [Scenario 1: 모델 선택](#scenario-1-모델-선택)
3. [Scenario 2: 데이터셋 선택/업로드](#scenario-2-데이터셋-선택업로드)
4. [Scenario 3: Config 설정](#scenario-3-config-설정)
5. [Scenario 4: 학습 시작](#scenario-4-학습-시작)
6. [Scenario 5: 메트릭 수집 및 표시](#scenario-5-메트릭-수집-및-표시)
7. [Scenario 6: 추론](#scenario-6-추론)

---

## 개요

이 문서는 Vision AI Training Platform의 주요 사용자 플로우를 단계별로 상세히 설명합니다.

각 시나리오는 다음 관점에서 분석됩니다:
- **Frontend**: React 컴포넌트, 사용자 상호작용, API 호출
- **Backend**: FastAPI 엔드포인트, 비즈니스 로직, DB 쿼리
- **Training Service**: 학습 실행, 메트릭 로깅, 체크포인트 저장
- **Storage**: R2 오브젝트 스토리지 접근 (데이터셋, 체크포인트, 아티팩트)
- **Database**: PostgreSQL 테이블 조회/수정

---

## Scenario 1: 모델 선택

### User Story
사용자가 학습할 모델을 선택하는 과정

### Flow Diagram
```
[Frontend: ModelPanel]
    ↓ (1) User clicks task type filter
    ↓ (2) GET /api/v1/models/list?task_type=image_classification
[Backend: models.py]
    ↓ (3) GET http://timm-service:5000/models/list
    ↓ (4) GET http://ultralytics-service:5001/models/list
[Training Services]
    ↓ (5) Return model metadata
[Backend]
    ↓ (6) Aggregate and return
[Frontend]
    ↓ (7) Display model cards
```

### 1단계: Frontend 초기 렌더링

**컴포넌트**: `frontend/components/training/ModelPanel.tsx`

```typescript
const ModelPanel = () => {
  const [taskType, setTaskType] = useState<string>("image_classification");
  const [models, setModels] = useState<Model[]>([]);

  // Fetch models on mount or task type change
  useEffect(() => {
    fetchModels();
  }, [taskType]);

  const fetchModels = async () => {
    // API call to Backend
    const response = await fetch(
      `/api/v1/models/list?task_type=${taskType}`
    );
    const data = await response.json();
    setModels(data.models);
  };
};
```

**API Call**:
```http
GET /api/v1/models/list?task_type=image_classification
Authorization: Bearer <token>
```

### 2단계: Backend - Model List API

**파일**: `mvp/backend/app/api/models.py`

```python
@router.get("/list", response_model=schemas.ModelListResponse)
async def list_models(
    task_type: Optional[str] = None,
    framework: Optional[str] = None,
):
    """
    사용 가능한 모델 목록 조회
    - Training Services에서 실시간으로 가져옴 (No DB query)
    """

    # Step 1: Get timm models
    timm_models = []
    try:
        timm_client = TrainingServiceClient(framework="timm")
        response = timm_client.get_models(task_type=task_type)
        timm_models = response.get("models", [])
    except Exception as e:
        logger.error(f"Failed to fetch timm models: {e}")

    # Step 2: Get ultralytics models
    ultralytics_models = []
    try:
        ultralytics_client = TrainingServiceClient(framework="ultralytics")
        response = ultralytics_client.get_models(task_type=task_type)
        ultralytics_models = response.get("models", [])
    except Exception as e:
        logger.error(f"Failed to fetch ultralytics models: {e}")

    # Step 3: Aggregate results
    all_models = timm_models + ultralytics_models

    # Step 4: Filter by framework if specified
    if framework:
        all_models = [m for m in all_models if m["framework"] == framework]

    return {"models": all_models, "total": len(all_models)}
```

**DB Access**: 없음 (Training Services에서 실시간 조회)

**TrainingServiceClient 내부**:
```python
def get_models(self, task_type: Optional[str] = None) -> Dict:
    """
    GET http://timm-service:5000/models/list?task_type=...
    """
    url = f"{self.base_url}/models/list"
    params = {"task_type": task_type} if task_type else {}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()
```

### 3단계: Training Service - Model Registry

**timm-service**: `mvp/training/api_server.py` (timm)

```python
@app.get("/models/list")
def list_models(task_type: Optional[str] = None):
    """
    timm 모델 목록 반환
    - TIMM_MODELS 레지스트리에서 조회 (No DB)
    """
    models = TIMM_MODELS  # Static registry in memory

    # Filter by task type
    if task_type:
        models = [m for m in models if task_type in m["task_types"]]

    return {"models": models, "total": len(models)}
```

**TIMM_MODELS 예시**:
```python
TIMM_MODELS = [
    {
        "model_name": "resnet50",
        "framework": "timm",
        "task_types": ["image_classification"],
        "default_image_size": 224,
        "parameters": "25.6M",
        "pretrained": True,
        "description": "ResNet-50 with ImageNet pretrained weights"
    },
    {
        "model_name": "efficientnet_b0",
        "framework": "timm",
        "task_types": ["image_classification"],
        "default_image_size": 224,
        "parameters": "5.3M",
        "pretrained": True,
        "description": "EfficientNet-B0"
    }
]
```

**ultralytics-service**: Similar structure with YOLO models

### 4단계: Frontend - Display Model Cards

**컴포넌트**: `ModelCard.tsx`

```typescript
{models.map((model) => (
  <ModelCard
    key={model.model_name}
    model={model}
    selected={selectedModel === model.model_name}
    onClick={() => handleSelectModel(model)}
  />
))}
```

**State Update**:
```typescript
const handleSelectModel = (model: Model) => {
  setSelectedModel(model.model_name);
  setSelectedFramework(model.framework);

  // Update parent form state
  onModelChange({
    framework: model.framework,
    model_name: model.model_name,
    task_type: model.task_types[0],
    image_size: model.default_image_size,
    pretrained: model.pretrained,
  });
};
```

### Summary

| Layer | Action | API/DB/Storage |
|-------|--------|----------------|
| Frontend | User selects task type filter | - |
| Frontend | Call Backend API | `GET /api/v1/models/list` |
| Backend | Query Training Services | `GET http://timm-service:5000/models/list` |
| Training Service | Return model registry | In-memory static data |
| Backend | Aggregate and return | - |
| Frontend | Display model cards | - |

**No Database Access**: 모델 목록은 Training Services의 메모리 레지스트리에서 제공

---

## Scenario 2: 데이터셋 선택/업로드

### User Story
사용자가 새 데이터셋을 업로드하거나 기존 데이터셋을 선택

### Flow Diagram
```
[Frontend: DatasetPanel]
    ↓ (1) User uploads files
    ↓ (2) POST /api/v1/datasets (multipart/form-data)
[Backend: datasets.py]
    ↓ (3) Save to temp directory
    ↓ (4) Upload to R2 Storage
    ↓ (5) INSERT INTO datasets table
[Database: PostgreSQL]
    ↓ (6) Return dataset metadata
[Frontend]
    ↓ (7) Display in dataset list
```

### 1단계: Frontend - File Upload

**컴포넌트**: `DatasetUploadPanel.tsx`

```typescript
const handleUpload = async (files: File[]) => {
  const formData = new FormData();

  // Add files
  files.forEach((file) => {
    formData.append("files", file);
  });

  // Add metadata
  formData.append("name", datasetName);
  formData.append("description", description);
  formData.append("format", format); // "dice", "yolo", "coco", etc.
  formData.append("visibility", "private");
  formData.append("project_id", String(projectId));

  // API call
  const response = await fetch("/api/v1/datasets", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
    },
    body: formData,
  });

  const dataset = await response.json();
  onUploadComplete(dataset);
};
```

**API Call**:
```http
POST /api/v1/datasets
Content-Type: multipart/form-data
Authorization: Bearer <token>

--boundary
Content-Disposition: form-data; name="name"
My Dataset
--boundary
Content-Disposition: form-data; name="format"
dice
--boundary
Content-Disposition: form-data; name="files"; filename="image1.jpg"
Content-Type: image/jpeg

<binary data>
--boundary--
```

### 2단계: Backend - Dataset Upload API

**파일**: `mvp/backend/app/api/datasets.py`

```python
@router.post("", response_model=schemas.DatasetResponse)
async def create_dataset(
    name: str = Form(...),
    description: str = Form(""),
    format: str = Form(...),  # dice, yolo, coco
    visibility: str = Form("private"),
    project_id: int = Form(...),
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    데이터셋 생성 및 R2 업로드
    """

    # Step 1: Generate dataset ID
    dataset_id = str(uuid.uuid4())

    # Step 2: Save files to temp directory
    temp_dir = Path(f"/tmp/datasets/{dataset_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []
    total_size = 0

    for file in files:
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            total_size += len(content)
        file_paths.append(file_path)

    # Step 3: Upload to R2 Storage
    r2_prefix = f"datasets/{dataset_id}/"

    for file_path in file_paths:
        relative_path = file_path.relative_to(temp_dir)
        r2_key = f"{r2_prefix}{relative_path}"

        s3_client.upload_file(
            str(file_path),
            bucket_name=settings.R2_BUCKET,
            object_name=r2_key
        )

    # Step 4: Analyze dataset (detect num_classes, class_distribution)
    dataset_info = analyze_dataset(temp_dir, format)

    # Step 5: Create database record
    db_dataset = models.Dataset(
        id=dataset_id,
        name=name,
        description=description,
        format=format,
        visibility=visibility,
        project_id=project_id,
        user_id=current_user.id,
        file_count=len(files),
        total_size_bytes=total_size,
        num_classes=dataset_info.get("num_classes"),
        class_distribution=dataset_info.get("class_distribution"),
        storage_path=r2_prefix,
    )

    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)

    # Step 6: Cleanup temp directory
    shutil.rmtree(temp_dir)

    return db_dataset
```

### 3단계: R2 Storage Upload

**Storage Operations**:
```python
# R2 Storage Structure
datasets/
  └── 8ab5c6e4-0f92-4fff-8f4e-441c08d94cef/  # dataset_id
      ├── train/
      │   ├── class1/
      │   │   ├── image1.jpg
      │   │   └── image2.jpg
      │   └── class2/
      │       └── image3.jpg
      └── val/
          └── class1/
              └── image4.jpg
```

**boto3 Upload**:
```python
import boto3

s3_client = boto3.client(
    's3',
    endpoint_url=settings.R2_ENDPOINT,
    aws_access_key_id=settings.R2_ACCESS_KEY_ID,
    aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
)

s3_client.upload_file(
    '/tmp/datasets/8ab5c6e4.../train/class1/image1.jpg',
    bucket_name='vision-platform',
    object_name='datasets/8ab5c6e4.../train/class1/image1.jpg'
)
```

### 4단계: Database - Insert Dataset Record

**Table**: `datasets`

```sql
INSERT INTO datasets (
    id,
    name,
    description,
    format,
    visibility,
    project_id,
    user_id,
    file_count,
    total_size_bytes,
    num_classes,
    class_distribution,
    storage_path,
    created_at,
    updated_at
) VALUES (
    '8ab5c6e4-0f92-4fff-8f4e-441c08d94cef',
    'My Dataset',
    'Description here',
    'dice',
    'private',
    2,
    1,
    25,
    1048576,
    10,
    '{"cat": 10, "dog": 15}'::jsonb,
    'datasets/8ab5c6e4-0f92-4fff-8f4e-441c08d94cef/',
    NOW(),
    NOW()
);
```

**Query Result**:
```python
# SQLAlchemy ORM returns Dataset object
dataset = Dataset(
    id='8ab5c6e4-0f92-4fff-8f4e-441c08d94cef',
    name='My Dataset',
    format='dice',
    num_classes=10,
    file_count=25,
    # ...
)
```

### 5단계: Frontend - Display in Dataset List

**컴포넌트**: `DatasetList.tsx`

```typescript
// Fetch datasets for current project
useEffect(() => {
  const fetchDatasets = async () => {
    const response = await fetch(
      `/api/v1/datasets?project_id=${projectId}`
    );
    const data = await response.json();
    setDatasets(data.datasets);
  };

  fetchDatasets();
}, [projectId]);

// Render dataset cards
{datasets.map((dataset) => (
  <DatasetCard
    key={dataset.id}
    dataset={dataset}
    selected={selectedDatasetId === dataset.id}
    onClick={() => setSelectedDatasetId(dataset.id)}
  />
))}
```

### Summary

| Layer | Action | API/DB/Storage |
|-------|--------|----------------|
| Frontend | User uploads files | - |
| Frontend | Call Backend API | `POST /api/v1/datasets` (multipart) |
| Backend | Save to temp directory | `/tmp/datasets/{uuid}/` |
| Backend | Upload to R2 | boto3.upload_file() |
| Backend | Analyze dataset | In-memory processing |
| Backend | Insert DB record | `INSERT INTO datasets` |
| Database | Store metadata | PostgreSQL: `datasets` table |
| Backend | Return dataset object | JSON response |
| Frontend | Display in list | React state update |

**Storage Flow**: Local temp → R2 Storage → Delete temp
**Database**: PostgreSQL `datasets` table

---

## Scenario 3: Config 설정

### User Story
사용자가 학습 설정을 구성 (기본 파라미터 + Advanced Config)

### Flow Diagram
```
[Frontend: ConfigPanel]
    ↓ (1) User selects framework
    ↓ (2) GET /api/v1/training/config-schema?framework=timm
[Backend: training.py]
    ↓ (3) GET http://timm-service:5000/config/schema
[Training Service]
    ↓ (4) Return ConfigSchema
[Backend]
    ↓ (5) Return schema + presets
[Frontend]
    ↓ (6) Dynamically render UI components
    ↓ (7) User modifies values
    ↓ (8) Validate and build final config
```

### 1단계: Frontend - Request Config Schema

**컴포넌트**: `AdvancedConfigPanel.tsx`

```typescript
const AdvancedConfigPanel = ({ framework, taskType }) => {
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [config, setConfig] = useState<any>({});

  useEffect(() => {
    fetchSchema();
  }, [framework, taskType]);

  const fetchSchema = async () => {
    const response = await fetch(
      `/api/v1/training/config-schema?framework=${framework}&task_type=${taskType}`
    );
    const data = await response.json();
    setSchema(data.schema);

    // Initialize config with defaults
    const defaultConfig = buildDefaultConfig(data.schema);
    setConfig(defaultConfig);
  };
};
```

**API Call**:
```http
GET /api/v1/training/config-schema?framework=timm&task_type=image_classification
Authorization: Bearer <token>
```

### 2단계: Backend - Config Schema API

**파일**: `mvp/backend/app/api/training.py`

```python
@router.get("/config-schema", response_model=schemas.ConfigSchemaResponse)
async def get_config_schema(
    framework: str,
    task_type: Optional[str] = None,
):
    """
    고급 설정 스키마 조회
    - Training Service에서 실시간으로 가져옴
    """

    # Step 1: Get schema from Training Service
    try:
        client = TrainingServiceClient(framework=framework)
        response = client.get_config_schema(task_type=task_type)

        return {
            "framework": framework,
            "task_type": task_type,
            "schema": response["schema"],
            "presets": response.get("presets", {}),
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Training Service unavailable: {str(e)}"
        )
```

**TrainingServiceClient**:
```python
def get_config_schema(self, task_type: Optional[str] = None) -> Dict:
    """
    GET http://timm-service:5000/config/schema?task_type=...
    """
    url = f"{self.base_url}/config/schema"
    params = {"task_type": task_type} if task_type else {}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()
```

### 3단계: Training Service - Return Schema

**timm-service**: `mvp/training/api_server.py`

```python
@app.get("/config/schema")
def get_config_schema(task_type: Optional[str] = None):
    """
    timm Advanced Config Schema 반환
    """
    from adapters.timm_adapter import TimmAdapter

    # Get schema from adapter
    schema = TimmAdapter.get_advanced_config_schema(task_type)
    presets = TimmAdapter.get_config_presets(task_type)

    return {
        "schema": schema.to_dict(),
        "presets": presets,
    }
```

**TimmAdapter Schema Definition**:
```python
@staticmethod
def get_advanced_config_schema(task_type: str) -> ConfigSchema:
    """Define Advanced Config schema for timm"""

    fields = [
        ConfigField(
            name="optimizer.type",
            type="select",
            label="Optimizer",
            description="Optimization algorithm",
            default="adamw",
            options=[
                {"value": "adam", "label": "Adam"},
                {"value": "adamw", "label": "AdamW"},
                {"value": "sgd", "label": "SGD"},
            ]
        ),
        ConfigField(
            name="optimizer.learning_rate",
            type="number",
            label="Learning Rate",
            default=0.001,
            min=1e-6,
            max=1.0,
            step=1e-5,
        ),
        ConfigField(
            name="scheduler.type",
            type="select",
            label="LR Scheduler",
            default="cosine",
            options=[
                {"value": "cosine", "label": "Cosine Annealing"},
                {"value": "step", "label": "Step LR"},
                {"value": "none", "label": "None"},
            ]
        ),
        # ... more fields
    ]

    return ConfigSchema(fields=fields)
```

**Response Example**:
```json
{
  "schema": {
    "fields": [
      {
        "name": "optimizer.type",
        "type": "select",
        "label": "Optimizer",
        "description": "Optimization algorithm",
        "default": "adamw",
        "options": [
          {"value": "adam", "label": "Adam"},
          {"value": "adamw", "label": "AdamW"}
        ]
      },
      {
        "name": "optimizer.learning_rate",
        "type": "number",
        "label": "Learning Rate",
        "default": 0.001,
        "min": 1e-6,
        "max": 1.0
      }
    ]
  },
  "presets": {
    "basic": {
      "optimizer": {"type": "adam", "learning_rate": 0.001},
      "scheduler": {"type": "none"}
    },
    "standard": {
      "optimizer": {"type": "adamw", "learning_rate": 0.001},
      "scheduler": {"type": "cosine"}
    }
  }
}
```

### 4단계: Frontend - Dynamic UI Rendering

**컴포넌트**: `DynamicConfigField.tsx`

```typescript
const DynamicConfigField = ({ field, value, onChange }) => {
  switch (field.type) {
    case 'select':
      return (
        <Select
          label={field.label}
          value={value}
          onChange={(val) => onChange(field.name, val)}
          options={field.options}
        />
      );

    case 'number':
      return (
        <NumberInput
          label={field.label}
          value={value}
          onChange={(val) => onChange(field.name, val)}
          min={field.min}
          max={field.max}
          step={field.step}
        />
      );

    case 'boolean':
      return (
        <Switch
          label={field.label}
          checked={value}
          onChange={(val) => onChange(field.name, val)}
        />
      );

    default:
      return null;
  }
};
```

**Rendering Loop**:
```typescript
{schema.fields.map((field) => (
  <DynamicConfigField
    key={field.name}
    field={field}
    value={getNestedValue(config, field.name)}
    onChange={handleConfigChange}
  />
))}
```

**State Management**:
```typescript
const handleConfigChange = (fieldName: string, value: any) => {
  // fieldName: "optimizer.learning_rate"
  // value: 0.001

  setConfig((prev) => {
    const updated = { ...prev };

    // Handle nested fields
    const parts = fieldName.split('.');
    let current = updated;

    for (let i = 0; i < parts.length - 1; i++) {
      if (!current[parts[i]]) {
        current[parts[i]] = {};
      }
      current = current[parts[i]];
    }

    current[parts[parts.length - 1]] = value;

    return updated;
  });
};
```

**Final Config Object**:
```typescript
{
  optimizer: {
    type: "adamw",
    learning_rate: 0.001,
    weight_decay: 0.01,
    betas: [0.9, 0.999]
  },
  scheduler: {
    type: "cosine",
    T_max: 50,
    eta_min: 1e-6
  },
  augmentation: {
    enabled: true,
    random_flip: true,
    random_flip_prob: 0.5
  },
  mixed_precision: true,
  gradient_clip_value: 1.0
}
```

### Summary

| Layer | Action | API/DB/Storage |
|-------|--------|----------------|
| Frontend | User selects framework | - |
| Frontend | Request schema | `GET /api/v1/training/config-schema` |
| Backend | Forward to Training Service | `GET http://timm-service:5000/config/schema` |
| Training Service | Return ConfigSchema | In-memory schema definition |
| Backend | Return schema + presets | JSON response |
| Frontend | Dynamically render UI | React components |
| Frontend | User modifies values | Local state management |
| Frontend | Build final config object | Nested object construction |

**No Database Access**: Schema는 Training Services에서 실시간 제공
**Dynamic UI**: Schema에 기반하여 컴포넌트 동적 생성

---

## Scenario 4: 학습 시작

### User Story
사용자가 모든 설정을 완료하고 학습을 시작

### Flow Diagram
```
[Frontend: StartTrainingButton]
    ↓ (1) User clicks "Start Training"
    ↓ (2) POST /api/v1/training/jobs (create job)
[Backend: training.py]
    ↓ (3) INSERT INTO training_jobs
[Database: PostgreSQL]
    ↓ (4) Return job object
[Frontend]
    ↓ (5) POST /api/v1/training/jobs/{job_id}/start
[Backend]
    ↓ (6) POST http://timm-service:5000/training/start
[Training Service]
    ↓ (7) subprocess.Popen(train.py)
[train.py]
    ↓ (8) Download dataset from R2
    ↓ (9) Start training loop
```

### 1단계: Frontend - Create Training Job

**컴포넌트**: `StartTrainingButton.tsx`

```typescript
const handleStartTraining = async () => {
  // Step 1: Create training job
  const jobPayload = {
    session_id: sessionId,
    project_id: projectId,
    experiment_name: experimentName,
    tags: tags,
    notes: notes,
    config: {
      framework: framework,
      model_name: modelName,
      task_type: taskType,
      dataset_id: datasetId,
      dataset_format: datasetFormat,
      num_classes: numClasses,
      epochs: epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      advanced_config: advancedConfig, // From ConfigPanel
    }
  };

  const createResponse = await fetch("/api/v1/training/jobs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify(jobPayload),
  });

  const job = await createResponse.json();
  const jobId = job.id;

  // Step 2: Start training
  const startResponse = await fetch(
    `/api/v1/training/jobs/${jobId}/start`,
    {
      method: "POST",
      headers: { "Authorization": `Bearer ${token}` },
    }
  );

  // Step 3: Navigate to monitoring page
  router.push(`/training/${jobId}`);
};
```

### 2단계: Backend - Create Training Job

**파일**: `mvp/backend/app/api/training.py`

```python
@router.post("/jobs", response_model=schemas.TrainingJobResponse)
async def create_training_job(
    request: schemas.TrainingJobCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    학습 작업 생성 (아직 시작하지 않음)
    """

    # Step 1: Validate dataset exists
    dataset = db.query(models.Dataset).filter(
        models.Dataset.id == request.config.dataset_id
    ).first()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Step 2: Generate output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/app/data/outputs/job_{timestamp}"

    # Step 3: Create database record
    db_job = models.TrainingJob(
        session_id=request.session_id,
        project_id=request.project_id,
        user_id=current_user.id,
        experiment_name=request.experiment_name,
        tags=request.tags,
        notes=request.notes,
        framework=request.config.framework,
        model_name=request.config.model_name,
        task_type=request.config.task_type,
        dataset_id=request.config.dataset_id,
        dataset_path=request.config.dataset_id,  # UUID for R2 download
        dataset_format=request.config.dataset_format,
        num_classes=request.config.num_classes,
        output_dir=output_dir,
        epochs=request.config.epochs,
        batch_size=request.config.batch_size,
        learning_rate=request.config.learning_rate,
        status="pending",
        advanced_config=request.config.advanced_config,  # Store as JSON
        primary_metric="accuracy",  # Default, can be overridden
        primary_metric_mode="max",
    )

    db.add(db_job)
    db.commit()
    db.refresh(db_job)

    return db_job
```

**Database Insert**:
```sql
INSERT INTO training_jobs (
    session_id, project_id, user_id,
    experiment_name, tags, notes,
    framework, model_name, task_type,
    dataset_id, dataset_path, dataset_format,
    num_classes, output_dir,
    epochs, batch_size, learning_rate,
    status, advanced_config,
    primary_metric, primary_metric_mode,
    created_at
) VALUES (
    1, 2, 1,
    'ResNet-50 Experiment', ARRAY['baseline'], 'Initial training',
    'timm', 'resnet50', 'image_classification',
    '8ab5c6e4-0f92-4fff-8f4e-441c08d94cef',
    '8ab5c6e4-0f92-4fff-8f4e-441c08d94cef',
    'dice',
    10, '/app/data/outputs/job_20251106_120000',
    50, 32, 0.001,
    'pending', '{"optimizer": {"type": "adamw"}}'::jsonb,
    'accuracy', 'max',
    NOW()
) RETURNING id;
```

**Response**:
```json
{
  "id": 5,
  "session_id": 1,
  "project_id": 2,
  "framework": "timm",
  "model_name": "resnet50",
  "status": "pending",
  "created_at": "2025-11-06T12:00:00Z"
}
```

### 3단계: Backend - Start Training Job

**파일**: `mvp/backend/app/api/training.py`

```python
@router.post("/jobs/{job_id}/start", response_model=schemas.TrainingJobResponse)
async def start_training_job(
    job_id: int,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    db: Session = Depends(get_db),
):
    """
    학습 작업 시작
    """

    # Step 1: Get job from database
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == job_id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Step 2: Validate status
    if job.status not in ["pending", "stopped"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start job with status: {job.status}"
        )

    # Step 3: Initialize training manager
    training_manager = TrainingManager(db)

    # Step 4: Start training via Training Service
    success = training_manager.start_training(
        job_id=job_id,
        checkpoint_path=checkpoint_path,
        resume=resume
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to start training"
        )

    # Step 5: Update status
    job.status = "running"
    job.started_at = datetime.now()
    db.commit()
    db.refresh(job)

    return job
```

**TrainingManager.start_training()** (mvp/backend/app/utils/training_manager.py):
```python
def start_training(
    self,
    job_id: int,
    checkpoint_path: Optional[str] = None,
    resume: bool = False
) -> bool:
    """
    Start training via Training Service API
    """

    # Step 1: Get job from DB
    job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    # Step 2: Prepare job config
    job_config = {
        "job_id": job.id,
        "framework": job.framework,
        "model_name": job.model_name,
        "task_type": job.task_type,
        "dataset_path": job.dataset_path,  # UUID for R2
        "dataset_format": job.dataset_format,
        "num_classes": job.num_classes,
        "output_dir": job.output_dir,
        "epochs": job.epochs,
        "batch_size": job.batch_size,
        "learning_rate": job.learning_rate,
        "advanced_config": job.advanced_config,  # JSON → dict
        "project_id": job.project_id,
        "checkpoint_path": checkpoint_path,
        "resume": resume,
    }

    # Step 3: Initialize Training Service client
    client = TrainingServiceClient(framework=job.framework)

    # Step 4: Call Training Service API
    try:
        response = client.start_training(job_config)
        return True
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return False
```

**TrainingServiceClient.start_training()**:
```python
def start_training(self, job_config: Dict[str, Any]) -> bool:
    """
    POST http://timm-service:5000/training/start
    """
    url = f"{self.base_url}/training/start"

    response = requests.post(
        url,
        json=job_config,
        timeout=30
    )

    response.raise_for_status()
    return True
```

### 4단계: Training Service - Start Subprocess

**timm-service**: `mvp/training/api_server.py`

```python
@app.post("/training/start")
def start_training(request: TrainingRequest):
    """
    Start training subprocess
    """
    job_id = request.job_id

    # Get python executable path
    venv_dir = os.path.join(os.path.dirname(__file__), "venv-timm")
    if os.path.exists(venv_dir):
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")  # Windows
    else:
        python_exe = sys.executable

    # Get train.py path
    train_script = os.path.join(os.path.dirname(__file__), "train.py")

    # Build command
    cmd = [
        python_exe,
        train_script,
        "--framework", request.framework,
        "--task_type", request.task_type,
        "--model_name", request.model_name,
        "--dataset_path", request.dataset_path,  # UUID
        "--dataset_format", request.dataset_format,
        "--num_classes", str(request.num_classes),
        "--output_dir", request.output_dir,
        "--epochs", str(request.epochs),
        "--batch_size", str(request.batch_size),
        "--learning_rate", str(request.learning_rate),
        "--job_id", str(request.job_id),
    ]

    # Add advanced_config as JSON string
    if request.advanced_config:
        cmd.extend([
            "--advanced_config",
            json.dumps(request.advanced_config)
        ])

    # Add checkpoint args
    if request.checkpoint_path:
        cmd.extend(["--checkpoint_path", request.checkpoint_path])
    if request.resume:
        cmd.append("--resume")

    # Start subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Store process
    job_processes[job_id] = process

    # Start log monitoring thread
    threading.Thread(
        target=monitor_training_logs,
        args=(job_id, process),
        daemon=True
    ).start()

    return {"status": "started", "job_id": job_id}
```

### 5단계: train.py - Dataset Download and Training

**파일**: `mvp/training/train.py`

```python
def main():
    args = parse_args()

    # Step 1: Download dataset from R2 if needed
    if '/' not in args.dataset_path:
        # UUID detected, download from R2
        local_dataset_path = get_dataset(
            dataset_id=args.dataset_path,
            download_fn=None  # R2 only
        )
    else:
        # Local path
        local_dataset_path = args.dataset_path

    # Step 2: Create configuration objects
    model_config = ModelConfig(
        framework=args.framework,
        task_type=TaskType(args.task_type),
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )

    dataset_config = DatasetConfig(
        dataset_path=local_dataset_path,
        format=DatasetFormat(args.dataset_format),
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        advanced_config=args.advanced_config,  # From API
    )

    # Step 3: Initialize Training Logger
    logger = TrainingLogger(job_id=args.job_id)
    logger.update_status("running")

    # Step 4: Create adapter
    adapter = TimmAdapter(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        output_dir=args.output_dir,
        job_id=args.job_id,
        project_id=args.project_id,
        logger=logger
    )

    # Step 5: Train
    metrics = adapter.train(
        start_epoch=0,
        checkpoint_path=args.checkpoint_path,
        resume_training=args.resume
    )

    # Step 6: Update status
    logger.update_status("completed")
```

**get_dataset() - R2 Download** (platform_sdk/utils.py):
```python
def get_dataset(dataset_id: str, download_fn=None) -> str:
    """
    Download dataset from R2 to local cache

    Returns:
        Local path to dataset directory
    """
    # Check cache first
    cache_dir = Path.home() / ".cache" / "vision-platform" / "datasets" / dataset_id

    if cache_dir.exists():
        return str(cache_dir)

    # Download from R2
    import boto3

    s3_client = boto3.client(
        's3',
        endpoint_url=os.getenv('R2_ENDPOINT'),
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    )

    # List objects in dataset prefix
    prefix = f"datasets/{dataset_id}/"
    response = s3_client.list_objects_v2(
        Bucket=os.getenv('R2_BUCKET'),
        Prefix=prefix
    )

    # Download each file
    cache_dir.mkdir(parents=True, exist_ok=True)

    for obj in response.get('Contents', []):
        key = obj['Key']
        relative_path = key[len(prefix):]
        local_path = cache_dir / relative_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3_client.download_file(
            Bucket=os.getenv('R2_BUCKET'),
            Key=key,
            Filename=str(local_path)
        )

    return str(cache_dir)
```

### Summary

| Layer | Action | API/DB/Storage |
|-------|--------|----------------|
| Frontend | User clicks Start Training | - |
| Frontend | Create job | `POST /api/v1/training/jobs` |
| Backend | Validate dataset | `SELECT * FROM datasets WHERE id = ?` |
| Backend | Insert job record | `INSERT INTO training_jobs` |
| Database | Store job metadata | PostgreSQL: `training_jobs` table |
| Frontend | Start training | `POST /api/v1/training/jobs/{id}/start` |
| Backend | Update status | `UPDATE training_jobs SET status = 'running'` |
| Backend | Call Training Service | `POST http://timm-service:5000/training/start` |
| Training Service | Start subprocess | `subprocess.Popen(train.py)` |
| train.py | Download dataset | boto3.download_file() from R2 |
| train.py | Start training loop | TimmAdapter.train() |

**Database**: `training_jobs` table for job metadata
**Storage**: R2 download for dataset files

---

## Scenario 5: 메트릭 수집 및 표시

### User Story
학습 중 실시간 메트릭 수집 및 프론트엔드 표시

### Flow Diagram
```
[train.py: TimmAdapter]
    ↓ (1) Epoch complete, compute metrics
    ↓ (2) logger.log_metrics(epoch, metrics)
[TrainingLogger]
    ↓ (3) POST http://backend:8000/api/v1/internal/training/callback
[Backend: internal_api.py]
    ↓ (4) INSERT INTO training_metrics
[Database]
    ↓ (5) Emit to WebSocket clients
[Frontend: WebSocket]
    ↓ (6) Update chart state
```

### 1단계: Training Loop - Compute Metrics

**파일**: `mvp/training/adapters/timm_adapter.py`

```python
class TimmAdapter(TrainingAdapter):
    def train(self, ...):
        # Training loop
        for epoch in range(start_epoch, self.training_config.epochs):
            # Train one epoch
            train_loss = self._train_one_epoch(epoch)

            # Validation
            val_loss, val_metrics = self._validate(epoch)

            # Build MetricsResult
            metrics_result = MetricsResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics={
                    "accuracy": val_metrics["accuracy"],
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                }
            )

            # Log to backend
            if self.logger.enabled:
                self.logger.log_metrics(
                    epoch=epoch,
                    loss=train_loss,
                    accuracy=val_metrics["accuracy"],
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    extra_metrics={
                        "val_loss": val_loss,
                        "precision": val_metrics.get("precision", 0),
                        "recall": val_metrics.get("recall", 0),
                    }
                )

            # Save checkpoint
            self._save_checkpoint(epoch, metrics_result)
```

### 2단계: TrainingLogger - Send to Backend

**파일**: `mvp/training/platform_sdk/logger.py`

```python
class TrainingLogger:
    def log_metrics(
        self,
        epoch: int,
        loss: float,
        accuracy: float = None,
        learning_rate: float = None,
        extra_metrics: dict = None
    ):
        """
        Send metrics to Backend via Internal API
        """
        if not self.enabled:
            return

        # Prepare payload
        payload = {
            "job_id": self.job_id,
            "event": "epoch_complete",
            "data": {
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": learning_rate,
                "extra_metrics": extra_metrics or {},
            }
        }

        # Send to Backend
        try:
            response = requests.post(
                f"{self.backend_url}/api/v1/internal/training/callback",
                json=payload,
                headers={
                    "X-Internal-Auth": os.getenv("INTERNAL_SECRET"),
                },
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            print(f"[WARNING] Failed to log metrics: {e}")
```

### 3단계: Backend - Internal Callback API

**파일**: `mvp/backend/app/api/internal_api.py`

```python
@router.post("/training/callback")
async def training_callback(
    request: schemas.TrainingCallback,
    x_internal_auth: str = Header(None),
    db: Session = Depends(get_db),
):
    """
    Training Service에서 Backend로 콜백
    """

    # Step 1: Validate internal auth
    if x_internal_auth != settings.INTERNAL_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Step 2: Handle event
    if request.event == "epoch_complete":
        # Insert metrics to database
        metric = models.TrainingMetric(
            job_id=request.job_id,
            epoch=request.data["epoch"],
            loss=request.data["loss"],
            accuracy=request.data.get("accuracy"),
            learning_rate=request.data.get("learning_rate"),
            extra_metrics=request.data.get("extra_metrics", {}),
        )

        db.add(metric)
        db.commit()

        # Emit to WebSocket clients
        await websocket_manager.broadcast_to_job(
            job_id=request.job_id,
            message={
                "type": "training_progress",
                "data": {
                    "epoch": request.data["epoch"],
                    "loss": request.data["loss"],
                    "accuracy": request.data.get("accuracy"),
                    "learning_rate": request.data.get("learning_rate"),
                }
            }
        )

    return {"status": "received", "job_id": request.job_id}
```

**Database Insert**:
```sql
INSERT INTO training_metrics (
    job_id,
    epoch,
    loss,
    accuracy,
    learning_rate,
    extra_metrics,
    created_at
) VALUES (
    5,
    10,
    0.45,
    0.85,
    0.0008,
    '{"val_loss": 0.50, "precision": 0.84}'::jsonb,
    NOW()
);
```

### 4단계: WebSocket - Broadcast to Clients

**WebSocketManager** (mvp/backend/app/utils/websocket.py):
```python
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: int):
        await websocket.accept()

        if job_id not in self.active_connections:
            self.active_connections[job_id] = []

        self.active_connections[job_id].append(websocket)

    async def broadcast_to_job(self, job_id: int, message: dict):
        """
        Broadcast message to all clients watching this job
        """
        if job_id not in self.active_connections:
            return

        # Remove disconnected clients
        active = []

        for websocket in self.active_connections[job_id]:
            try:
                await websocket.send_json(message)
                active.append(websocket)
            except Exception:
                # Client disconnected
                pass

        self.active_connections[job_id] = active
```

**WebSocket Endpoint** (mvp/backend/app/api/websocket.py):
```python
@router.websocket("/training/{job_id}")
async def training_websocket(
    websocket: WebSocket,
    job_id: int
):
    """
    WebSocket for real-time training updates
    """
    await websocket_manager.connect(websocket, job_id)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket, job_id)
```

### 5단계: Frontend - WebSocket Client

**Hook**: `useTrainingWebSocket.ts`

```typescript
export const useTrainingWebSocket = (jobId: number) => {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Create WebSocket connection
    const ws = new WebSocket(
      `ws://localhost:8000/ws/training/${jobId}`
    );

    ws.onopen = () => {
      setIsConnected(true);
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === "training_progress") {
        // Update metrics state
        setMetrics((prev) => [
          ...prev,
          {
            epoch: message.data.epoch,
            loss: message.data.loss,
            accuracy: message.data.accuracy,
            learning_rate: message.data.learning_rate,
          }
        ]);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log("WebSocket disconnected");
    };

    // Cleanup
    return () => {
      ws.close();
    };
  }, [jobId]);

  return { metrics, isConnected };
};
```

### 6단계: Frontend - Chart Rendering

**컴포넌트**: `TrainingMetricsChart.tsx`

```typescript
const TrainingMetricsChart = ({ jobId }) => {
  const { metrics, isConnected } = useTrainingWebSocket(jobId);

  // Prepare chart data
  const chartData = {
    labels: metrics.map((m) => m.epoch),
    datasets: [
      {
        label: "Loss",
        data: metrics.map((m) => m.loss),
        borderColor: "rgb(239, 68, 68)",
        yAxisID: "y",
      },
      {
        label: "Accuracy",
        data: metrics.map((m) => m.accuracy),
        borderColor: "rgb(16, 185, 129)",
        yAxisID: "y1",
      },
    ],
  };

  return (
    <div>
      <div className="status">
        {isConnected ? "🟢 Connected" : "🔴 Disconnected"}
      </div>

      <Line data={chartData} options={chartOptions} />

      <div className="current-metrics">
        {metrics.length > 0 && (
          <>
            <div>Epoch: {metrics[metrics.length - 1].epoch}</div>
            <div>Loss: {metrics[metrics.length - 1].loss.toFixed(4)}</div>
            <div>Accuracy: {(metrics[metrics.length - 1].accuracy * 100).toFixed(2)}%</div>
          </>
        )}
      </div>
    </div>
  );
};
```

### Summary

| Layer | Action | API/DB/Storage |
|-------|--------|----------------|
| train.py | Epoch complete, compute metrics | - |
| TrainingLogger | Send to Backend | `POST /api/v1/internal/training/callback` |
| Backend | Insert metrics | `INSERT INTO training_metrics` |
| Database | Store metrics | PostgreSQL: `training_metrics` table |
| Backend | Broadcast via WebSocket | WebSocketManager.broadcast_to_job() |
| Frontend | Receive WebSocket message | `ws.onmessage` |
| Frontend | Update state | React setState |
| Frontend | Re-render chart | react-chartjs-2 |

**Real-time Flow**: train.py → Backend API → Database → WebSocket → Frontend
**No Polling**: WebSocket provides push-based updates

---

## Scenario 6: 추론

### User Story
학습된 체크포인트로 추론 실행

### Flow Diagram
```
[Frontend: InferencePanel]
    ↓ (1) User uploads image + selects job
    ↓ (2) POST /api/v1/test-inference (multipart)
[Backend: inference.py]
    ↓ (3) SELECT best_checkpoint_path FROM training_jobs
    ↓ (4) POST http://timm-service:5000/inference/predict
[Training Service]
    ↓ (5) Download checkpoint from R2
    ↓ (6) Load model + run inference
    ↓ (7) Return predictions
[Backend]
    ↓ (8) Return to Frontend
[Frontend]
    ↓ (9) Display predictions
```

### 1단계: Frontend - Upload Image for Inference

**컴포넌트**: `InferencePanel.tsx`

```typescript
const handleInference = async (file: File) => {
  const formData = new FormData();

  // Add image file
  formData.append("file", file);

  // Add training job ID
  formData.append("training_job_id", String(selectedJobId));

  // Optional parameters
  formData.append("confidence_threshold", "0.25");
  formData.append("top_k", "5");

  // API call
  const response = await fetch("/api/v1/test-inference", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
    },
    body: formData,
  });

  const result = await response.json();
  setPredictions(result.predictions);
};
```

**API Call**:
```http
POST /api/v1/test-inference
Content-Type: multipart/form-data
Authorization: Bearer <token>

--boundary
Content-Disposition: form-data; name="training_job_id"
5
--boundary
Content-Disposition: form-data; name="file"; filename="test.jpg"
Content-Type: image/jpeg

<binary image data>
--boundary--
```

### 2단계: Backend - Inference API

**파일**: `mvp/backend/app/api/inference.py`

```python
@router.post("/test-inference")
async def test_inference(
    training_job_id: int = Form(...),
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
    top_k: int = Form(5),
    db: Session = Depends(get_db),
):
    """
    체크포인트 기반 추론
    """

    # Step 1: Get training job from database
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == training_job_id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Step 2: Find best checkpoint
    checkpoint_path = job.best_checkpoint_path

    if not checkpoint_path:
        # Fallback: get latest checkpoint
        checkpoint_path = os.path.join(job.output_dir, "checkpoints", "latest.pth")

    # Step 3: Save uploaded image to temp file
    temp_file = f"/tmp/inference_{uuid.uuid4()}.jpg"
    with open(temp_file, "wb") as f:
        content = await file.read()
        f.write(content)

    # Step 4: Initialize Training Service client
    client = TrainingServiceClient(framework=job.framework)

    # Step 5: Call inference API
    try:
        result = client.run_inference(
            checkpoint_path=checkpoint_path,
            image_path=temp_file,
            model_name=job.model_name,
            task_type=job.task_type,
            num_classes=job.num_classes,
            confidence_threshold=confidence_threshold,
            top_k=top_k,
        )

        # Cleanup temp file
        os.remove(temp_file)

        return {
            "predictions": result["predictions"],
            "task_type": job.task_type,
            "model_name": job.model_name,
            "inference_time_ms": result.get("inference_time_ms", 0),
        }

    except Exception as e:
        # Cleanup temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
```

**Database Query**:
```sql
SELECT
    id, framework, model_name, task_type,
    num_classes, best_checkpoint_path, output_dir
FROM training_jobs
WHERE id = 5;
```

**Result**:
```python
job = TrainingJob(
    id=5,
    framework="timm",
    model_name="resnet50",
    task_type="image_classification",
    num_classes=10,
    best_checkpoint_path="r2://vision-platform/checkpoints/project_2/job_5/epoch_45.pth",
    output_dir="/app/data/outputs/job_20251106_120000"
)
```

### 3단계: Training Service - Run Inference

**timm-service**: `mvp/training/api_server.py`

```python
@app.post("/inference/predict")
def predict(
    checkpoint_path: str,
    image_path: str,
    model_name: str,
    task_type: str,
    num_classes: int,
    confidence_threshold: float = 0.25,
    top_k: int = 5,
):
    """
    Run inference with checkpoint
    """
    import time
    from adapters.timm_adapter import TimmAdapter

    # Step 1: Download checkpoint from R2 if needed
    if checkpoint_path.startswith("r2://"):
        local_checkpoint = download_checkpoint_from_r2(checkpoint_path)
    else:
        local_checkpoint = checkpoint_path

    # Step 2: Load model
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    # Step 3: Load checkpoint weights
    checkpoint = torch.load(local_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Step 4: Preprocess image
    from PIL import Image
    import torchvision.transforms as T

    image = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Step 5: Run inference
    start_time = time.time()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    inference_time = (time.time() - start_time) * 1000  # ms

    # Step 6: Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    # Step 7: Load class names
    class_names = load_class_names(checkpoint_path)  # From checkpoint metadata

    # Step 8: Build predictions
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        if prob.item() >= confidence_threshold:
            predictions.append({
                "label": class_names[idx.item()],
                "label_id": idx.item(),
                "confidence": prob.item(),
            })

    return {
        "predictions": predictions,
        "inference_time_ms": inference_time,
    }
```

**R2 Checkpoint Download**:
```python
def download_checkpoint_from_r2(r2_path: str) -> str:
    """
    Download checkpoint from R2

    Args:
        r2_path: r2://bucket/checkpoints/project_2/job_5/epoch_45.pth

    Returns:
        Local path to downloaded checkpoint
    """
    # Parse R2 path
    parts = r2_path.replace("r2://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]

    # Local cache path
    cache_dir = Path.home() / ".cache" / "vision-platform" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / key.replace("/", "_")

    # Download if not cached
    if not local_path.exists():
        s3_client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=str(local_path)
        )

    return str(local_path)
```

### 4단계: Frontend - Display Predictions

**컴포넌트**: `PredictionResults.tsx`

```typescript
const PredictionResults = ({ predictions, imageUrl }) => {
  return (
    <div className="prediction-results">
      <div className="image-preview">
        <img src={imageUrl} alt="Input" />
      </div>

      <div className="predictions">
        <h3>Predictions</h3>

        {predictions.map((pred, idx) => (
          <div key={idx} className="prediction-item">
            <div className="rank">#{idx + 1}</div>
            <div className="label">{pred.label}</div>
            <div className="confidence">
              {(pred.confidence * 100).toFixed(2)}%
            </div>
            <div className="confidence-bar">
              <div
                className="bar-fill"
                style={{ width: `${pred.confidence * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

**Example Output**:
```
Predictions:
#1 cat      95.32%  [████████████████████]
#2 dog       4.12%  [█                   ]
#3 bird      0.45%  [                    ]
#4 horse     0.08%  [                    ]
#5 fish      0.03%  [                    ]
```

### Summary

| Layer | Action | API/DB/Storage |
|-------|--------|----------------|
| Frontend | User uploads image | - |
| Frontend | Call inference API | `POST /api/v1/test-inference` (multipart) |
| Backend | Get training job | `SELECT * FROM training_jobs WHERE id = ?` |
| Database | Return job metadata | PostgreSQL: `training_jobs` table |
| Backend | Save temp file | `/tmp/inference_<uuid>.jpg` |
| Backend | Call Training Service | `POST http://timm-service:5000/inference/predict` |
| Training Service | Download checkpoint | boto3.download_file() from R2 |
| Training Service | Load model weights | torch.load() |
| Training Service | Run inference | model(input_tensor) |
| Training Service | Return predictions | JSON response |
| Backend | Cleanup temp file | os.remove() |
| Backend | Return to Frontend | JSON response |
| Frontend | Display predictions | React component |

**Storage**: R2 for checkpoint download
**Database**: `training_jobs` table for metadata

---

## 참고 문서

- [Backend API 명세서](./01_backend_api_specification.md)
- [SDK & Adapter Pattern](./02_sdk_adapter_pattern.md)
- [Config Schema 가이드](./03_config_schema_guide.md)
- [기존 API 명세](../api/API_SPECIFICATION.md)
- [아키텍처 문서](../architecture/ARCHITECTURE.md)
