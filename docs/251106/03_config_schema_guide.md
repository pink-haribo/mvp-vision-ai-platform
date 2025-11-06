# Config Schema 시스템 가이드

> **작성일**: 2025-11-06
> **Version**: 1.0
> **대상**: Training Service 개발자, Backend 개발자, Frontend 개발자

## 목차

1. [개요](#개요)
2. [전체 흐름](#전체-흐름)
3. [Training Service: Schema 정의](#training-service-schema-정의)
4. [Backend: Schema 라우팅](#backend-schema-라우팅)
5. [Frontend: Dynamic UI 생성](#frontend-dynamic-ui-생성)
6. [Advanced Config 처리](#advanced-config-처리)
7. [실전 예제](#실전-예제)

---

## 개요

**Config Schema 시스템**은 다양한 프레임워크의 고급 설정을 **동적으로 UI에 표시**하고, 사용자 입력을 **검증 및 처리**하는 시스템입니다.

### 핵심 아이디어

1. **Training Service가 Schema 정의**: 각 프레임워크(timm, ultralytics)가 자신의 설정 스키마를 직접 정의
2. **Backend가 Schema 중계**: Frontend 요청을 적절한 Training Service로 라우팅
3. **Frontend가 Dynamic UI 생성**: Schema 기반으로 폼 자동 생성

### 왜 Dynamic Schema인가?

- ✅ **프레임워크 독립성**: 각 프레임워크가 독립적으로 설정 추가/변경
- ✅ **중앙 집중식 관리**: Training Service에서 모든 설정 관리
- ✅ **자동 UI 생성**: Frontend 코드 수정 없이 UI 자동 업데이트
- ✅ **타입 안전성**: Schema 기반 검증

---

## 전체 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                         사용자                                    │
│         "timm으로 학습하고 싶은데 어떤 설정이 있지?"              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Frontend                                   │
│  GET /api/v1/training/config-schema?framework=timm                │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Backend                                    │
│  1. framework=timm 확인                                           │
│  2. TIMM_SERVICE_URL 조회                                        │
│  3. GET {TIMM_SERVICE_URL}/config-schema 요청                    │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                   Training Service (timm)                         │
│  1. TimmAdapter.get_config_schema() 호출                         │
│  2. Schema JSON 생성                                             │
│  3. Response 반환                                                │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Backend                                    │
│  Response 그대로 Frontend로 전달                                  │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Frontend                                   │
│  1. Schema 파싱                                                   │
│  2. Dynamic Form 생성                                            │
│  3. 사용자에게 표시                                               │
└───────────────────────────────────────────────────────────────────┘
```

---

## Training Service: Schema 정의

### Step 1: ConfigField 정의

```python
# mvp/training/platform_sdk/base.py

@dataclass
class ConfigField:
    """Config 필드 정의"""

    # 필수 필드
    name: str                    # 필드 경로 (예: "optimizer.learning_rate")
    type: str                    # 필드 타입 (number, select, boolean, range, text)
    label: str                   # UI에 표시될 라벨

    # 선택 필드
    description: str = ""        # 필드 설명 (tooltip)
    default: Any = None          # 기본값

    # Number 타입 전용
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    # Select 타입 전용
    options: Optional[List[Dict]] = None  # [{"value": "adam", "label": "Adam"}]

    # Boolean 타입 전용
    # (추가 필드 없음)

    # 조건부 표시 (선택)
    depends_on: Optional[str] = None      # 다른 필드에 의존
    visible_when: Optional[Any] = None    # 표시 조건
```

### Step 2: 프레임워크별 Schema 구현

```python
# mvp/training/adapters/timm_adapter.py

class TimmAdapter(TrainingAdapter):

    @staticmethod
    def get_config_schema(task_type: Optional[str] = None) -> ConfigSchema:
        """timm 프레임워크의 Config Schema 반환"""

        fields = []

        # ========== Optimizer ==========
        fields.append(ConfigField(
            name="optimizer.type",
            type="select",
            label="Optimizer",
            description="Optimization algorithm to use",
            default="adamw",
            options=[
                {
                    "value": "adam",
                    "label": "Adam",
                    "description": "Adaptive Moment Estimation - general purpose optimizer"
                },
                {
                    "value": "adamw",
                    "label": "AdamW",
                    "description": "Adam with decoupled weight decay - recommended"
                },
                {
                    "value": "sgd",
                    "label": "SGD",
                    "description": "Stochastic Gradient Descent with momentum"
                }
            ]
        ))

        fields.append(ConfigField(
            name="optimizer.learning_rate",
            type="number",
            label="Learning Rate",
            description="Initial learning rate for training",
            default=0.001,
            min=1e-6,
            max=1.0,
            step=1e-5
        ))

        fields.append(ConfigField(
            name="optimizer.weight_decay",
            type="number",
            label="Weight Decay",
            description="L2 regularization coefficient",
            default=0.01,
            min=0.0,
            max=1.0,
            step=0.001
        ))

        # ========== Scheduler ==========
        fields.append(ConfigField(
            name="scheduler.type",
            type="select",
            label="Learning Rate Scheduler",
            description="Learning rate scheduling strategy",
            default="cosine",
            options=[
                {"value": "none", "label": "None", "description": "Constant learning rate"},
                {"value": "step", "label": "Step LR", "description": "Reduce LR at fixed intervals"},
                {"value": "cosine", "label": "Cosine Annealing", "description": "Cosine decay"},
                {"value": "exponential", "label": "Exponential", "description": "Exponential decay"}
            ]
        ))

        # Scheduler: Cosine 전용 필드 (조건부 표시)
        fields.append(ConfigField(
            name="scheduler.T_max",
            type="number",
            label="T_max (Cosine)",
            description="Maximum number of iterations for cosine annealing",
            default=50,
            min=1,
            max=1000,
            step=1,
            depends_on="scheduler.type",
            visible_when="cosine"
        ))

        # ========== Augmentation ==========
        fields.append(ConfigField(
            name="augmentation.enabled",
            type="boolean",
            label="Enable Data Augmentation",
            description="Apply data augmentation during training",
            default=True
        ))

        fields.append(ConfigField(
            name="augmentation.random_flip",
            type="boolean",
            label="Random Horizontal Flip",
            description="Randomly flip images horizontally",
            default=True,
            depends_on="augmentation.enabled",
            visible_when=True
        ))

        fields.append(ConfigField(
            name="augmentation.color_jitter",
            type="boolean",
            label="Color Jitter",
            description="Random brightness, contrast, saturation, hue",
            default=False,
            depends_on="augmentation.enabled",
            visible_when=True
        ))

        # ========== Advanced ==========
        fields.append(ConfigField(
            name="mixed_precision",
            type="boolean",
            label="Mixed Precision Training",
            description="Use FP16 for faster training (requires GPU)",
            default=False
        ))

        fields.append(ConfigField(
            name="gradient_clip_value",
            type="number",
            label="Gradient Clipping",
            description="Max gradient norm (0 = disabled)",
            default=0.0,
            min=0.0,
            max=10.0,
            step=0.1
        ))

        # Presets 정의
        presets = {
            "basic": {
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0
                },
                "scheduler": {"type": "none"},
                "augmentation": {
                    "enabled": True,
                    "random_flip": True,
                    "color_jitter": False
                },
                "mixed_precision": False,
                "gradient_clip_value": 0.0
            },
            "standard": {
                "optimizer": {
                    "type": "adamw",
                    "learning_rate": 0.0003,
                    "weight_decay": 0.01
                },
                "scheduler": {
                    "type": "cosine",
                    "T_max": 50,
                    "eta_min": 1e-6
                },
                "augmentation": {
                    "enabled": True,
                    "random_flip": True,
                    "color_jitter": True
                },
                "mixed_precision": True,
                "gradient_clip_value": 1.0
            }
        }

        return ConfigSchema(
            framework="timm",
            task_types=[TaskType.IMAGE_CLASSIFICATION],
            fields=fields,
            presets=presets
        )
```

### Step 3: API Endpoint 추가

```python
# mvp/training/api_server.py

from adapters import ADAPTER_REGISTRY

@app.get("/config-schema")
async def get_config_schema(task_type: str = None):
    """
    Config Schema 반환 (프레임워크별)

    Query Parameters:
        task_type (str, optional): Task type for filtering
    """
    # FRAMEWORK 환경변수에서 프레임워크 확인
    framework = os.environ.get("FRAMEWORK", "unknown")

    if framework not in ADAPTER_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Framework '{framework}' not supported"
        )

    # Adapter 클래스의 get_config_schema() 호출
    adapter_class = ADAPTER_REGISTRY[framework]

    try:
        schema = adapter_class.get_config_schema(task_type=task_type)

        # ConfigSchema를 JSON으로 변환
        return {
            "framework": schema.framework,
            "task_types": [tt.value for tt in schema.task_types],
            "schema": {
                "fields": [
                    {
                        "name": field.name,
                        "type": field.type,
                        "label": field.label,
                        "description": field.description,
                        "default": field.default,
                        "min": field.min,
                        "max": field.max,
                        "step": field.step,
                        "options": field.options,
                        "depends_on": field.depends_on,
                        "visible_when": field.visible_when
                    }
                    for field in schema.fields
                ]
            },
            "presets": schema.presets
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate config schema: {str(e)}"
        )
```

---

## Backend: Schema 라우팅

### Backend API Endpoint

```python
# mvp/backend/app/api/training.py

@router.get("/config-schema")
async def get_config_schema(framework: str, task_type: str = None):
    """
    Training Service로부터 Config Schema를 가져와서 Frontend로 전달

    Args:
        framework (str): 프레임워크 이름 (timm, ultralytics)
        task_type (str, optional): Task type for filtering

    Returns:
        Config Schema JSON
    """
    try:
        logger.info(f"[config-schema] Requested framework={framework}, task_type={task_type}")

        # TrainingServiceClient를 사용하여 Training Service와 통신
        from app.utils.training_client import TrainingServiceClient
        import requests

        # Initialize Training Service client
        client = TrainingServiceClient(framework=framework)

        logger.info(f"[config-schema] Fetching schema from Training Service: {client.base_url}")

        # Call Training Service /config-schema endpoint
        response = requests.get(
            f"{client.base_url}/config-schema",
            params={"task_type": task_type or ""},
            timeout=30
        )

        # Handle errors
        if response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Framework '{framework}' not supported or config schema not available"
            )
        elif response.status_code == 503:
            raise HTTPException(
                status_code=503,
                detail=f"Training Service for framework '{framework}' is not available"
            )
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Training Service error: {response.text}"
            )

        # Parse response
        schema_data = response.json()

        logger.info(f"[config-schema] Schema retrieved with {len(schema_data.get('schema', {}).get('fields', []))} fields")

        return schema_data

    except HTTPException:
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[config-schema] Connection error to Training Service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Training Service for framework '{framework}' is not reachable"
        )
    except Exception as e:
        logger.error(f"[config-schema] Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get configuration schema: {str(e)}"
        )
```

---

## Frontend: Dynamic UI 생성

### Step 1: Schema 가져오기

```typescript
// mvp/frontend/lib/api/training.ts

export interface ConfigField {
  name: string;
  type: 'number' | 'select' | 'boolean' | 'range' | 'text';
  label: string;
  description?: string;
  default?: any;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{
    value: string;
    label: string;
    description?: string;
  }>;
  depends_on?: string;
  visible_when?: any;
}

export interface ConfigSchema {
  framework: string;
  task_types: string[];
  schema: {
    fields: ConfigField[];
  };
  presets: Record<string, any>;
}

export async function getConfigSchema(
  framework: string,
  taskType?: string
): Promise<ConfigSchema> {
  const params = new URLSearchParams({ framework });
  if (taskType) params.append('task_type', taskType);

  const response = await fetch(
    `${API_BASE_URL}/training/config-schema?${params}`,
    {
      headers: {
        Authorization: `Bearer ${getAccessToken()}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch config schema: ${response.statusText}`);
  }

  return response.json();
}
```

### Step 2: Dynamic Form 컴포넌트

```typescript
// mvp/frontend/components/training/DynamicConfigForm.tsx

import { useState, useEffect } from 'react';
import { ConfigField, ConfigSchema } from '@/lib/api/training';

interface DynamicConfigFormProps {
  framework: string;
  taskType?: string;
  value: any;
  onChange: (config: any) => void;
}

export default function DynamicConfigForm({
  framework,
  taskType,
  value,
  onChange,
}: DynamicConfigFormProps) {
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [loading, setLoading] = useState(true);
  const [config, setConfig] = useState(value || {});

  // Schema 로드
  useEffect(() => {
    async function loadSchema() {
      try {
        const schemaData = await getConfigSchema(framework, taskType);
        setSchema(schemaData);

        // Initialize config with defaults
        const defaultConfig: any = {};
        schemaData.schema.fields.forEach((field) => {
          setNestedValue(defaultConfig, field.name, field.default);
        });
        setConfig({ ...defaultConfig, ...value });
      } catch (error) {
        console.error('Failed to load config schema:', error);
      } finally {
        setLoading(false);
      }
    }
    loadSchema();
  }, [framework, taskType]);

  // Nested value 설정 함수 (예: "optimizer.learning_rate")
  function setNestedValue(obj: any, path: string, value: any) {
    const parts = path.split('.');
    let current = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      if (!current[parts[i]]) current[parts[i]] = {};
      current = current[parts[i]];
    }
    current[parts[parts.length - 1]] = value;
  }

  function getNestedValue(obj: any, path: string) {
    const parts = path.split('.');
    let current = obj;
    for (const part of parts) {
      if (current === undefined || current === null) return undefined;
      current = current[part];
    }
    return current;
  }

  // 필드 값 변경 핸들러
  function handleFieldChange(fieldName: string, newValue: any) {
    const newConfig = { ...config };
    setNestedValue(newConfig, fieldName, newValue);
    setConfig(newConfig);
    onChange(newConfig);
  }

  // 필드 렌더링
  function renderField(field: ConfigField) {
    // 조건부 표시 체크
    if (field.depends_on) {
      const dependValue = getNestedValue(config, field.depends_on);
      if (dependValue !== field.visible_when) {
        return null; // 숨김
      }
    }

    const currentValue = getNestedValue(config, field.name);

    switch (field.type) {
      case 'number':
        return (
          <div key={field.name} className="mb-4">
            <label className="block text-sm font-medium mb-1">
              {field.label}
            </label>
            {field.description && (
              <p className="text-xs text-gray-500 mb-2">{field.description}</p>
            )}
            <input
              type="number"
              value={currentValue || field.default || 0}
              onChange={(e) =>
                handleFieldChange(field.name, parseFloat(e.target.value))
              }
              min={field.min}
              max={field.max}
              step={field.step}
              className="w-full px-3 py-2 border rounded-md"
            />
          </div>
        );

      case 'select':
        return (
          <div key={field.name} className="mb-4">
            <label className="block text-sm font-medium mb-1">
              {field.label}
            </label>
            {field.description && (
              <p className="text-xs text-gray-500 mb-2">{field.description}</p>
            )}
            <select
              value={currentValue || field.default}
              onChange={(e) => handleFieldChange(field.name, e.target.value)}
              className="w-full px-3 py-2 border rounded-md"
            >
              {field.options?.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            {/* Option description */}
            {field.options?.map((option) => {
              if (option.value === currentValue && option.description) {
                return (
                  <p key={option.value} className="text-xs text-gray-500 mt-1">
                    {option.description}
                  </p>
                );
              }
              return null;
            })}
          </div>
        );

      case 'boolean':
        return (
          <div key={field.name} className="mb-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={currentValue || field.default || false}
                onChange={(e) =>
                  handleFieldChange(field.name, e.target.checked)
                }
                className="mr-2"
              />
              <span className="text-sm font-medium">{field.label}</span>
            </label>
            {field.description && (
              <p className="text-xs text-gray-500 ml-6 mt-1">
                {field.description}
              </p>
            )}
          </div>
        );

      default:
        return null;
    }
  }

  // Preset 적용
  function applyPreset(presetName: string) {
    if (!schema) return;
    const presetConfig = schema.presets[presetName];
    if (presetConfig) {
      setConfig(presetConfig);
      onChange(presetConfig);
    }
  }

  if (loading) {
    return <div>Loading config schema...</div>;
  }

  if (!schema) {
    return <div>Failed to load config schema</div>;
  }

  return (
    <div className="space-y-4">
      {/* Preset 선택 */}
      {schema.presets && Object.keys(schema.presets).length > 0 && (
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">
            Config Presets
          </label>
          <div className="flex gap-2">
            {Object.keys(schema.presets).map((presetName) => (
              <button
                key={presetName}
                onClick={() => applyPreset(presetName)}
                className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
              >
                {presetName}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Dynamic Fields */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {schema.schema.fields.map((field) => renderField(field))}
      </div>
    </div>
  );
}
```

---

## Advanced Config 처리

### 학습 시작 시 Advanced Config 전달

```typescript
// Frontend에서 학습 시작

const advancedConfig = {
  optimizer: {
    type: 'adamw',
    learning_rate: 0.0003,
    weight_decay: 0.01
  },
  scheduler: {
    type: 'cosine',
    T_max: 50
  },
  augmentation: {
    enabled: true,
    random_flip: true
  }
};

const jobRequest = {
  config: {
    framework: 'timm',
    model_name: 'resnet50',
    // ... 기본 설정
    advanced_config: advancedConfig  // ← Advanced Config 추가
  }
};

await createTrainingJob(jobRequest);
```

### Backend에서 Advanced Config 저장

```python
# mvp/backend/app/api/training.py

@router.post("/jobs")
async def create_training_job(job_request: TrainingJobCreate, db: Session = Depends(get_db)):
    # ... 기본 필드 처리 ...

    # Advanced Config 저장 (JSON 컬럼)
    job = models.TrainingJob(
        # ... 기본 필드 ...
        advanced_config=job_request.config.advanced_config.model_dump() if job_request.config.advanced_config else None
    )

    db.add(job)
    db.commit()
```

### Training Service에서 Advanced Config 적용

```python
# mvp/training/train.py

def main():
    args = parse_args()

    # Advanced config 로드 (DB 또는 Command line)
    advanced_config = None

    if args.advanced_config:
        # API mode: command line에서 전달됨
        advanced_config = json.loads(args.advanced_config)

    # Adapter에 전달
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        advanced_config=advanced_config  # ← 전달
    )

    adapter = adapter_class(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        # ...
    )

    adapter.train()
```

### Adapter에서 Advanced Config 사용

```python
# mvp/training/adapters/timm_adapter.py

class TimmAdapter(TrainingAdapter):

    def _create_optimizer(self):
        """Advanced Config 기반 Optimizer 생성"""

        advanced = self.training_config.advanced_config

        if advanced and 'optimizer' in advanced:
            opt_config = advanced['optimizer']
            opt_type = opt_config.get('type', 'adam')
            lr = opt_config.get('learning_rate', 0.001)
            weight_decay = opt_config.get('weight_decay', 0.0)

            if opt_type == 'adam':
                return torch.optim.Adam(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            elif opt_type == 'adamw':
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=opt_config.get('betas', (0.9, 0.999))
                )
            # ... 기타 Optimizer

        # 기본값
        return torch.optim.Adam(self.model.parameters(), lr=self.training_config.learning_rate)
```

---

## 실전 예제

### 시나리오: YOLO Config Schema

```python
# mvp/training/adapters/ultralytics_adapter.py

class UltralyticsAdapter(TrainingAdapter):

    @staticmethod
    def get_config_schema(task_type: Optional[str] = None) -> ConfigSchema:
        fields = [
            # Image size
            ConfigField(
                name="imgsz",
                type="select",
                label="Image Size",
                default=640,
                options=[
                    {"value": 320, "label": "320x320"},
                    {"value": 640, "label": "640x640"},
                    {"value": 1280, "label": "1280x1280"}
                ]
            ),

            # Confidence threshold
            ConfigField(
                name="conf",
                type="number",
                label="Confidence Threshold",
                description="Minimum confidence for detections",
                default=0.25,
                min=0.0,
                max=1.0,
                step=0.05
            ),

            # IOU threshold
            ConfigField(
                name="iou",
                type="number",
                label="IOU Threshold",
                description="NMS IOU threshold",
                default=0.45,
                min=0.0,
                max=1.0,
                step=0.05
            ),

            # Augmentation
            ConfigField(
                name="hsv_h",
                type="number",
                label="HSV-Hue Augmentation",
                default=0.015,
                min=0.0,
                max=1.0,
                step=0.001
            ),
        ]

        presets = {
            "speed": {
                "imgsz": 320,
                "conf": 0.25,
                "augmentation": "minimal"
            },
            "accuracy": {
                "imgsz": 1280,
                "conf": 0.1,
                "augmentation": "aggressive"
            }
        }

        return ConfigSchema(
            framework="ultralytics",
            task_types=[TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION],
            fields=fields,
            presets=presets
        )
```

---

## 참고 문서

- [SDK & Adapter Pattern](./02_sdk_adapter_pattern.md)
- [Backend API 명세서](./01_backend_api_specification.md)
- [User Flow Scenarios](./04_user_flow_scenarios.md)
