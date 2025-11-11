# Trainer Implementation Plan

## Overview

Transform the basic training functionality into a production-ready, comprehensive training system with advanced configuration, validation, testing, export, and control features.

**Timeline:** 4-6 weeks (phased approach)
**Priority:** High - Core platform functionality
**Complexity:** High - Multiple frameworks, formats, and integrations

---

## Current State Analysis

### What We Have
- Basic training job creation and execution
- Simple parameter configuration (epochs, batch_size, learning_rate)
- ResNet-50 classification with ImageFolder format
- Basic status tracking (pending, running, completed, failed)
- MLflow integration for experiment tracking

### What's Missing
- Advanced training configuration (optimizer, scheduler, augmentation)
- Validation during training with detailed metrics
- Test/inference functionality with visualization
- Model export to deployment formats
- Training control (pause, resume, distributed training)
- Image-by-image result analysis
- Advanced metrics (mAP, confusion matrix, etc.)
- Framework-specific optimizations

---

## Plugin Architecture & Model Registry

### Design Philosophy

**Goal:** Enable model developers to integrate their models with minimal boilerplate, focusing only on model logic and training optimization.

**Key Principles:**
1. **Convention over Configuration** - Sensible defaults, minimal required code
2. **Declarative Configuration** - Models declare their needs, platform handles the rest
3. **Auto-Discovery** - Drop a model file, platform automatically registers it
4. **Dynamic UI Generation** - Configuration UI generated from model schema
5. **Zero Platform Lock-in** - Models remain framework-native, no proprietary APIs

---

### Model Registry System

#### Architecture Overview

```
Platform Core
├── Model Registry (discovers and manages models)
├── Schema Validator (validates model implementations)
├── Config Generator (generates frontend UI schemas)
└── Runtime Executor (runs training/inference)

Model Plugins (developer-provided)
├── models/
│   ├── my_custom_resnet/
│   │   ├── __init__.py
│   │   ├── model.py (implements BaseModel)
│   │   └── config_schema.json (optional, for complex configs)
│   ├── yolov8_custom/
│   │   └── model.py
│   └── efficient_transformer/
│       └── model.py
```

---

### Standard Model Interface

Every model must implement a minimal interface:

```python
# training/models/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ConfigField(BaseModel):
    """Describes a single configuration field"""
    name: str
    type: str  # int, float, str, bool, select, multiselect
    default: Any
    description: str
    required: bool = False

    # For select/multiselect
    options: Optional[List[str]] = None

    # For numeric types
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    # UI hints
    group: Optional[str] = None  # optimizer, scheduler, augmentation, etc.
    advanced: bool = False  # show in advanced settings only

class ConfigSchema(BaseModel):
    """Model's configuration requirements"""
    fields: List[ConfigField]
    presets: Optional[Dict[str, Dict[str, Any]]] = None  # easy, medium, advanced

class ModelMetadata(BaseModel):
    """Model metadata for registry"""
    name: str
    version: str
    author: str
    framework: str  # pytorch, timm, ultralytics, custom
    task_type: str  # classification, detection, segmentation, etc.
    description: str
    tags: List[str] = []

    # Requirements
    min_gpu_memory_gb: Optional[float] = None
    recommended_batch_size: Optional[int] = None
    supported_image_sizes: List[int] = []

class BaseModel(ABC):
    """Base class that all models must implement"""

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> ModelMetadata:
        """Return model metadata for registry"""
        pass

    @classmethod
    @abstractmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Return configuration schema for this model"""
        pass

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize model with configuration"""
        pass

    @abstractmethod
    def train(self, train_loader, val_loader=None, callbacks=None):
        """Main training loop"""
        pass

    @abstractmethod
    def validate(self, val_loader):
        """Validation loop, returns metrics dict"""
        pass

    @abstractmethod
    def infer(self, image_or_batch):
        """Inference on single image or batch"""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        """Save model checkpoint"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        pass

    @abstractmethod
    def export(self, path: str, format: str, **kwargs):
        """Export model to deployment format"""
        pass
```

---

### Example: Custom Model Implementation

Minimal example - developer writes only ~100 lines:

```python
# training/models/my_custom_resnet/model.py

from training.models.base import BaseModel, ModelMetadata, ConfigSchema, ConfigField
import torch
import torch.nn as nn
from torchvision.models import resnet50

class MyCustomResNet(BaseModel):
    """Custom ResNet with my special sauce"""

    @classmethod
    def get_metadata(cls):
        return ModelMetadata(
            name="my-custom-resnet50",
            version="1.0.0",
            author="John Doe",
            framework="pytorch",
            task_type="classification",
            description="ResNet50 with custom attention mechanism",
            tags=["resnet", "attention", "classification"],
            min_gpu_memory_gb=4.0,
            recommended_batch_size=32,
            supported_image_sizes=[224, 256, 384]
        )

    @classmethod
    def get_config_schema(cls):
        """Declare what configurations this model needs"""
        return ConfigSchema(
            fields=[
                # Optimizer settings
                ConfigField(
                    name="optimizer_type",
                    type="select",
                    default="adam",
                    options=["adam", "sgd", "adamw"],
                    description="Optimizer to use",
                    group="optimizer"
                ),
                ConfigField(
                    name="learning_rate",
                    type="float",
                    default=1e-3,
                    min=1e-6,
                    max=1.0,
                    description="Initial learning rate",
                    group="optimizer"
                ),
                ConfigField(
                    name="weight_decay",
                    type="float",
                    default=1e-4,
                    min=0.0,
                    max=1e-2,
                    description="Weight decay (L2 regularization)",
                    group="optimizer",
                    advanced=True
                ),

                # Scheduler settings
                ConfigField(
                    name="scheduler_type",
                    type="select",
                    default="cosine",
                    options=["cosine", "step", "exponential", "plateau"],
                    description="Learning rate scheduler",
                    group="scheduler"
                ),
                ConfigField(
                    name="warmup_epochs",
                    type="int",
                    default=5,
                    min=0,
                    max=50,
                    description="Number of warmup epochs",
                    group="scheduler"
                ),

                # Augmentation settings
                ConfigField(
                    name="use_mixup",
                    type="bool",
                    default=False,
                    description="Enable mixup augmentation",
                    group="augmentation",
                    advanced=True
                ),
                ConfigField(
                    name="mixup_alpha",
                    type="float",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    description="Mixup alpha parameter",
                    group="augmentation",
                    advanced=True
                ),

                # Model-specific settings
                ConfigField(
                    name="attention_layers",
                    type="multiselect",
                    default=["layer3", "layer4"],
                    options=["layer1", "layer2", "layer3", "layer4"],
                    description="Which layers to add attention",
                    group="model"
                ),
                ConfigField(
                    name="dropout_rate",
                    type="float",
                    default=0.1,
                    min=0.0,
                    max=0.9,
                    description="Dropout rate before classifier",
                    group="model"
                ),
            ],
            presets={
                "easy": {
                    "optimizer_type": "adam",
                    "learning_rate": 1e-3,
                    "scheduler_type": "cosine",
                    "use_mixup": False
                },
                "medium": {
                    "optimizer_type": "adamw",
                    "learning_rate": 5e-4,
                    "weight_decay": 1e-4,
                    "scheduler_type": "cosine",
                    "warmup_epochs": 5,
                    "use_mixup": True,
                    "mixup_alpha": 0.2
                },
                "advanced": {
                    "optimizer_type": "adamw",
                    "learning_rate": 3e-4,
                    "weight_decay": 5e-4,
                    "scheduler_type": "cosine",
                    "warmup_epochs": 10,
                    "use_mixup": True,
                    "mixup_alpha": 0.4,
                    "dropout_rate": 0.2
                }
            }
        )

    def __init__(self, config):
        self.config = config
        self.model = resnet50(pretrained=True)

        # Add custom attention layers based on config
        self._add_attention_layers()

        # Setup optimizer
        self.optimizer = self._build_optimizer()

        # Setup scheduler
        self.scheduler = self._build_scheduler()

    def train(self, train_loader, val_loader=None, callbacks=None):
        """Training loop - developer implements their training logic"""
        for epoch in range(self.config['epochs']):
            # Training code here
            for batch in train_loader:
                loss = self._train_step(batch)

            # Callbacks for platform integration
            if callbacks and callbacks.on_epoch_end:
                callbacks.on_epoch_end(epoch, {'loss': loss})

    def validate(self, val_loader):
        """Validation - returns metrics dict"""
        # Validation code here
        return {
            'accuracy': 0.95,
            'loss': 0.23,
            'top5_accuracy': 0.99
        }

    def infer(self, image_or_batch):
        """Inference"""
        with torch.no_grad():
            output = self.model(image_or_batch)
        return output

    # ... other methods
```

---

### Auto-Discovery & Registration

```python
# training/registry.py

import importlib
import os
from pathlib import Path
from typing import Dict, Type

class ModelRegistry:
    """Automatic model discovery and registration"""

    def __init__(self):
        self._models: Dict[str, Type[BaseModel]] = {}

    def discover_models(self, models_dir: str = "training/models"):
        """
        Automatically discover and register all models

        Convention: Any file named model.py in subdirectories
        that contains a class inheriting from BaseModel
        """
        models_path = Path(models_dir)

        for model_dir in models_path.iterdir():
            if not model_dir.is_dir():
                continue

            model_file = model_dir / "model.py"
            if not model_file.exists():
                continue

            # Dynamically import the module
            module_path = f"training.models.{model_dir.name}.model"
            try:
                module = importlib.import_module(module_path)

                # Find BaseModel subclass
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, BaseModel) and
                        attr is not BaseModel):

                        # Register the model
                        metadata = attr.get_metadata()
                        self._models[metadata.name] = attr

                        print(f"✓ Registered model: {metadata.name} v{metadata.version}")

            except Exception as e:
                print(f"✗ Failed to load model from {model_dir.name}: {e}")

    def get_model(self, name: str) -> Type[BaseModel]:
        """Get model class by name"""
        if name not in self._models:
            raise ValueError(f"Model {name} not found in registry")
        return self._models[name]

    def list_models(self, framework: str = None, task_type: str = None):
        """List all registered models with optional filtering"""
        models = []
        for name, model_cls in self._models.items():
            metadata = model_cls.get_metadata()

            # Apply filters
            if framework and metadata.framework != framework:
                continue
            if task_type and metadata.task_type != task_type:
                continue

            models.append(metadata.dict())

        return models

    def get_config_schema(self, model_name: str) -> ConfigSchema:
        """Get configuration schema for a model"""
        model_cls = self.get_model(model_name)
        return model_cls.get_config_schema()

# Global registry instance
registry = ModelRegistry()
```

---

### API Integration

```python
# app/api/models.py

from fastapi import APIRouter, HTTPException
from training.registry import registry

router = APIRouter()

@router.get("/models")
def list_models(framework: str = None, task_type: str = None):
    """
    List all available models

    Returns model metadata for display in UI
    """
    return registry.list_models(framework=framework, task_type=task_type)

@router.get("/models/{model_name}/config-schema")
def get_model_config_schema(model_name: str):
    """
    Get configuration schema for a model

    Frontend uses this to dynamically generate configuration UI
    """
    try:
        schema = registry.get_config_schema(model_name)
        return schema.dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/models/{model_name}/metadata")
def get_model_metadata(model_name: str):
    """Get detailed model metadata"""
    model_cls = registry.get_model(model_name)
    return model_cls.get_metadata().dict()

@router.post("/models/{model_name}/validate-config")
def validate_config(model_name: str, config: dict):
    """Validate configuration before training"""
    schema = registry.get_config_schema(model_name)

    # Validate all required fields
    errors = []
    for field in schema.fields:
        if field.required and field.name not in config:
            errors.append(f"Missing required field: {field.name}")

        # Type validation
        if field.name in config:
            value = config[field.name]
            # Validate based on field.type, min, max, options, etc.

    if errors:
        raise HTTPException(status_code=400, detail=errors)

    return {"valid": True}
```

---

### Dynamic UI Generation

Frontend automatically generates UI from config schema:

```typescript
// frontend/components/training/DynamicConfigPanel.tsx

interface ConfigField {
  name: string;
  type: 'int' | 'float' | 'str' | 'bool' | 'select' | 'multiselect';
  default: any;
  description: string;
  required?: boolean;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  group?: string;
  advanced?: boolean;
}

interface ConfigSchema {
  fields: ConfigField[];
  presets?: Record<string, Record<string, any>>;
}

export function DynamicConfigPanel({ modelName }: { modelName: string }) {
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [config, setConfig] = useState<Record<string, any>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    // Fetch schema from API
    fetch(`/api/v1/models/${modelName}/config-schema`)
      .then(res => res.json())
      .then(setSchema);
  }, [modelName]);

  if (!schema) return <Loading />;

  // Group fields by group
  const groupedFields = schema.fields.reduce((acc, field) => {
    const group = field.group || 'general';
    if (!acc[group]) acc[group] = [];
    acc[group].push(field);
    return acc;
  }, {} as Record<string, ConfigField[]>);

  const renderField = (field: ConfigField) => {
    // Skip advanced fields if not shown
    if (field.advanced && !showAdvanced) return null;

    switch (field.type) {
      case 'int':
      case 'float':
        return (
          <NumberInput
            label={field.name}
            description={field.description}
            value={config[field.name] ?? field.default}
            onChange={(v) => setConfig({ ...config, [field.name]: v })}
            min={field.min}
            max={field.max}
            step={field.step}
            required={field.required}
          />
        );

      case 'bool':
        return (
          <Checkbox
            label={field.name}
            description={field.description}
            checked={config[field.name] ?? field.default}
            onChange={(v) => setConfig({ ...config, [field.name]: v })}
          />
        );

      case 'select':
        return (
          <Select
            label={field.name}
            description={field.description}
            value={config[field.name] ?? field.default}
            onChange={(v) => setConfig({ ...config, [field.name]: v })}
            options={field.options || []}
            required={field.required}
          />
        );

      case 'multiselect':
        return (
          <MultiSelect
            label={field.name}
            description={field.description}
            value={config[field.name] ?? field.default}
            onChange={(v) => setConfig({ ...config, [field.name]: v })}
            options={field.options || []}
          />
        );

      default:
        return (
          <TextInput
            label={field.name}
            description={field.description}
            value={config[field.name] ?? field.default}
            onChange={(v) => setConfig({ ...config, [field.name]: v })}
            required={field.required}
          />
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Preset selector */}
      {schema.presets && (
        <PresetSelector
          presets={schema.presets}
          onSelect={(preset) => setConfig(preset)}
        />
      )}

      {/* Advanced toggle */}
      <button onClick={() => setShowAdvanced(!showAdvanced)}>
        {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
      </button>

      {/* Grouped configuration fields */}
      {Object.entries(groupedFields).map(([group, fields]) => (
        <Accordion key={group} title={group}>
          <div className="space-y-4">
            {fields.map(field => (
              <div key={field.name}>{renderField(field)}</div>
            ))}
          </div>
        </Accordion>
      ))}
    </div>
  );
}
```

---

### Developer Experience

#### To add a new model:

1. **Create directory**: `training/models/my_model/`
2. **Write model.py**: Implement BaseModel (~100-200 lines)
3. **Drop it in**: Platform auto-discovers on startup
4. **Done!** Model appears in UI with auto-generated config form

#### Example file structure:

```
training/models/
├── base.py (provided by platform)
├── my_custom_resnet/
│   └── model.py (developer writes this)
├── yolov8_optimized/
│   ├── model.py
│   └── utils.py (optional helper)
└── efficient_vit/
    ├── model.py
    └── requirements.txt (optional extra deps)
```

#### What developers DON'T need to worry about:

- ✅ Database integration
- ✅ API endpoints
- ✅ Frontend UI generation
- ✅ Authentication/authorization
- ✅ MLflow tracking
- ✅ Checkpoint management
- ✅ Distributed training setup
- ✅ Export infrastructure
- ✅ Monitoring/logging

#### What developers DO focus on:

- ✅ Model architecture
- ✅ Training logic
- ✅ Optimization strategies
- ✅ Configuration options
- ✅ Task-specific metrics

---

### Backward Compatibility

Existing hardcoded models (ResNet, YOLO, etc.) will be migrated to plugin format:

```python
# training/models/builtin/resnet50/model.py

class ResNet50(BaseModel):
    """Built-in ResNet50 implementation"""
    # ... implementation using the new interface
```

This ensures:
1. Consistency across all models
2. Same features for all (export, distributed, etc.)
3. Easy testing of new features with known models

---

### Validation & Safety

```python
# training/validator.py

class ModelValidator:
    """Validates model implementations before registration"""

    @staticmethod
    def validate_model_class(model_cls: Type[BaseModel]):
        """Check if model class implements required methods"""
        required_methods = [
            'get_metadata', 'get_config_schema',
            '__init__', 'train', 'validate', 'infer',
            'save_checkpoint', 'load_checkpoint', 'export'
        ]

        for method in required_methods:
            if not hasattr(model_cls, method):
                raise ValidationError(f"Model must implement {method}()")

        # Validate metadata
        metadata = model_cls.get_metadata()
        if not metadata.name:
            raise ValidationError("Model must have a name")

        # Validate config schema
        schema = model_cls.get_config_schema()
        for field in schema.fields:
            if field.type not in ['int', 'float', 'str', 'bool', 'select', 'multiselect']:
                raise ValidationError(f"Invalid field type: {field.type}")

        return True
```

---

### Framework-Specific Adapters

For developers who want even less boilerplate, provide framework-specific base classes:

```python
# training/models/adapters/timm_adapter.py

from training.models.base import BaseModel
import timm

class TimmModelAdapter(BaseModel):
    """Adapter for timm models - even less code needed"""

    def __init__(self, config):
        super().__init__(config)
        # Auto-setup optimizer, scheduler based on config
        self.model = timm.create_model(
            self.config['model_name'],
            pretrained=True,
            num_classes=self.config['num_classes']
        )
        self.optimizer = self._auto_build_optimizer()
        self.scheduler = self._auto_build_scheduler()

    def train(self, train_loader, val_loader=None, callbacks=None):
        # Provided training loop with callbacks
        return self._default_train_loop(train_loader, val_loader, callbacks)

    # ... other default implementations

# Developer just needs to specify config schema!
class MyTimmModel(TimmModelAdapter):
    @classmethod
    def get_metadata(cls):
        return ModelMetadata(
            name="my-efficient-net",
            framework="timm",
            task_type="classification",
            # ...
        )

    @classmethod
    def get_config_schema(cls):
        # Just declare what configs you want
        return ConfigSchema(fields=[...])
```

---

### Summary: Developer Journey

```
1. Copy example model.py
2. Change model architecture (10 lines)
3. Declare configuration needs (20 lines)
4. Implement training logic (50 lines)
5. Test locally
6. Drop in models/ directory
7. Platform auto-registers
8. UI automatically generated
9. Users can select and configure your model
10. Done! 🎉
```

**Total developer effort: ~1-2 hours for a new model**
**Platform handles: Everything else**

This design ensures the platform remains open, extensible, and developer-friendly while maintaining consistency and quality across all models.

---

## Implementation Phases

### Phase 1: Advanced Training Configuration (Week 1-2)
**Goal:** Provide comprehensive training parameters for all frameworks and tasks

#### 1.1 Backend Schema Design
**Files:** `app/schemas/training.py`, `app/schemas/configs.py`

Create hierarchical configuration schemas:

```python
# Base configuration
class TrainingConfig(BaseModel):
    # Existing basic params
    epochs: int
    batch_size: int
    learning_rate: float

    # New advanced params
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    augmentation: AugmentationConfig
    preprocessing: PreprocessConfig
    validation: ValidationConfig
    early_stopping: EarlyStoppingConfig | None

# Optimizer configuration
class OptimizerConfig(BaseModel):
    type: str  # adam, sgd, adamw, rmsprop
    weight_decay: float = 0.0
    momentum: float = 0.9  # for SGD
    betas: tuple[float, float] = (0.9, 0.999)  # for Adam
    amsgrad: bool = False

# Scheduler configuration
class SchedulerConfig(BaseModel):
    type: str  # step, cosine, plateau, exponential, onecycle
    step_size: int | None = None
    gamma: float = 0.1
    warmup_epochs: int = 0
    min_lr: float = 1e-6

# Augmentation configuration
class AugmentationConfig(BaseModel):
    enabled: bool = True
    # Spatial augmentations
    random_flip: bool = True
    random_rotation: int = 15  # degrees
    random_crop: bool = True
    resize_scale: tuple[float, float] = (0.8, 1.0)

    # Color augmentations
    color_jitter: ColorJitterConfig | None
    random_grayscale: float = 0.0
    gaussian_blur: float = 0.0

    # Advanced
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    auto_augment: str | None = None  # randaugment, autoaugment

# Preprocessing configuration
class PreprocessConfig(BaseModel):
    normalize: bool = True
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]
    resize_method: str = "bilinear"  # bilinear, bicubic, nearest

# Validation configuration
class ValidationConfig(BaseModel):
    enabled: bool = True
    val_split: float = 0.2  # if no separate val set
    val_interval: int = 1  # validate every N epochs
    save_best_only: bool = True
    metric: str = "accuracy"  # or loss, mAP, etc.
```

#### 1.2 Database Schema Updates
**Files:** `app/db/models.py`, migrations

Add JSON field for storing advanced configs:
```python
class TrainingJob(Base):
    # ... existing fields ...

    # New fields
    optimizer_config = Column(JSON, nullable=True)
    scheduler_config = Column(JSON, nullable=True)
    augmentation_config = Column(JSON, nullable=True)
    preprocessing_config = Column(JSON, nullable=True)
    validation_config = Column(JSON, nullable=True)

    # Training state
    current_epoch = Column(Integer, default=0)
    best_metric = Column(Float, nullable=True)
    checkpoint_path = Column(String(500), nullable=True)
```

#### 1.3 Adapter Pattern Extension
**Files:** `training/adapters/*.py`

Extend adapters to support advanced configs:
```python
class TrainingAdapter(ABC):
    @abstractmethod
    def build_optimizer(self, config: OptimizerConfig, model_params):
        """Build optimizer from config"""

    @abstractmethod
    def build_scheduler(self, config: SchedulerConfig, optimizer):
        """Build learning rate scheduler"""

    @abstractmethod
    def build_augmentation_pipeline(self, config: AugmentationConfig):
        """Build augmentation pipeline"""
```

#### 1.4 Frontend Configuration UI
**Files:** `frontend/components/training/*.tsx`

Create modular configuration panels:
- `OptimizerPanel.tsx` - Optimizer type and parameters
- `SchedulerPanel.tsx` - LR scheduler configuration
- `AugmentationPanel.tsx` - Data augmentation settings
- `PreprocessingPanel.tsx` - Preprocessing options
- `ValidationPanel.tsx` - Validation settings

Use accordion/tabs for organization:
```
Training Configuration
├── Basic Settings (epochs, batch size, LR)
├── Optimizer Settings (expand/collapse)
├── Scheduler Settings (expand/collapse)
├── Augmentation Settings (expand/collapse)
├── Preprocessing Settings (expand/collapse)
└── Validation Settings (expand/collapse)
```

**Deliverables:**
- [ ] Complete schema definitions
- [ ] Database migration script
- [ ] Extended adapter implementations
- [ ] Frontend configuration UI components
- [ ] API endpoints for config validation
- [ ] Unit tests for config validation

---

### Phase 2: Validation System (Week 2-3)
**Goal:** Real-time validation with detailed metrics and visualizations

#### 2.1 Validation Metrics System
**Files:** `training/metrics/*.py`

Implement comprehensive metrics:

```python
# Classification metrics
class ClassificationMetrics:
    - accuracy (top-1, top-5)
    - precision, recall, f1-score (per-class and macro/micro)
    - confusion matrix
    - ROC curve, AUC
    - class-wise accuracy

# Detection metrics
class DetectionMetrics:
    - mAP (mAP50, mAP75, mAP50-95)
    - mAP per class
    - Precision-Recall curves
    - IoU distribution
    - Detection confidence histogram

# Segmentation metrics
class SegmentationMetrics:
    - IoU (mean IoU, per-class IoU)
    - Dice coefficient
    - Pixel accuracy
    - Boundary F1 score
```

#### 2.2 Validation Results Storage
**Files:** `app/db/models.py`

New models for validation results:
```python
class ValidationResult(Base):
    __tablename__ = "validation_results"

    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))
    epoch = Column(Integer)

    # Aggregate metrics
    metrics = Column(JSON)  # {"accuracy": 0.95, "loss": 0.23, ...}

    # Detailed results
    per_class_metrics = Column(JSON)
    confusion_matrix = Column(JSON)

    # Visualization data
    visualization_data = Column(JSON)  # For charts

    created_at = Column(DateTime, default=datetime.utcnow)

class ValidationImageResult(Base):
    __tablename__ = "validation_image_results"

    id = Column(Integer, primary_key=True)
    validation_result_id = Column(Integer, ForeignKey("validation_results.id"))

    image_path = Column(String(500))
    image_name = Column(String(200))

    # Ground truth
    true_label = Column(String(100))  # for classification
    true_boxes = Column(JSON)  # for detection
    true_mask_path = Column(String(500))  # for segmentation

    # Predictions
    predicted_label = Column(String(100))
    predicted_boxes = Column(JSON)
    predicted_mask_path = Column(String(500))
    confidence = Column(Float)

    # Metrics for this image
    is_correct = Column(Boolean)
    iou = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
```

#### 2.3 Validation Execution
**Files:** `training/validators/*.py`

Implement validators for each task type:
```python
class Validator(ABC):
    @abstractmethod
    def validate_epoch(self, model, dataloader, epoch: int) -> ValidationResult:
        """Run validation and return results"""

    @abstractmethod
    def compute_metrics(self, predictions, ground_truths) -> dict:
        """Compute metrics from predictions"""

    @abstractmethod
    def visualize_results(self, results: ValidationResult) -> dict:
        """Generate visualization data"""
```

#### 2.4 Frontend Validation Dashboard
**Files:** `frontend/components/validation/*.tsx`

Create validation result viewers:
- `ValidationMetricsPanel.tsx` - Metrics overview (cards, charts)
- `ValidationImageGrid.tsx` - Image-by-image results table
- `ConfusionMatrixView.tsx` - Interactive confusion matrix
- `MetricsChart.tsx` - Training/validation curves
- `ValidationImageDetail.tsx` - Detailed view of single image result

**Features:**
- Filter by correct/incorrect predictions
- Sort by confidence, loss, etc.
- Side-by-side ground truth and prediction
- Zoomed view for detection boxes and segmentation masks

**Deliverables:**
- [ ] Metrics calculation implementations
- [ ] Database models and migrations
- [ ] Validator implementations per framework
- [ ] API endpoints for validation results
- [ ] Frontend validation dashboard
- [ ] WebSocket updates for real-time metrics

---

### Phase 3: Test/Inference System (Week 3-4)
**Goal:** Post-training testing and inference capabilities

#### 3.1 Test Dataset Management
**Files:** `app/api/datasets.py`, `app/db/models.py`

Add test dataset support:
```python
class Dataset(Base):
    # ... existing fields ...
    dataset_type = Column(String(20))  # train, val, test

class TestRun(Base):
    __tablename__ = "test_runs"

    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))
    checkpoint_path = Column(String(500))

    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    dataset_path = Column(String(500), nullable=True)  # for ad-hoc test

    status = Column(String(20))  # pending, running, completed
    metrics = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
```

#### 3.2 Inference Engine
**Files:** `training/inference/*.py`

Implement fast inference:
```python
class InferenceEngine:
    def __init__(self, model_path: str, config: dict):
        self.model = self.load_model(model_path)
        self.preprocess = self.build_preprocessing(config)

    def predict_single(self, image_path: str) -> dict:
        """Single image inference"""

    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        """Batch inference"""

    def predict_dataset(self, dataset_path: str) -> TestResult:
        """Full dataset inference with metrics"""
```

#### 3.3 Single Image Inference API
**Files:** `app/api/inference.py`

Create inference endpoints:
```python
@router.post("/inference/single")
async def predict_single_image(
    training_job_id: int,
    image: UploadFile,
    checkpoint: str = "best"
):
    """
    Single image inference with visualization
    Returns: prediction, confidence, visualization
    """

@router.post("/inference/batch")
async def predict_batch(
    training_job_id: int,
    images: list[UploadFile]
):
    """Batch inference"""

@router.post("/test-runs")
async def create_test_run(
    training_job_id: int,
    dataset_id: int | None = None,
    dataset_path: str | None = None
):
    """Run test on dataset"""
```

#### 3.4 Frontend Test/Inference UI
**Files:** `frontend/components/inference/*.tsx`

Create inference interfaces:
- `SingleImageInference.tsx` - Upload and test single image
- `BatchInference.tsx` - Upload multiple images
- `TestRunCreator.tsx` - Select test dataset and run
- `TestResultsViewer.tsx` - View test run results (similar to validation)
- `InferenceComparison.tsx` - Compare multiple checkpoints

**Features:**
- Drag-and-drop image upload
- Real-time inference with loading states
- Visualization overlays (boxes, masks, heatmaps)
- Download results as CSV/JSON
- Checkpoint selection (best, latest, specific epoch)

**Deliverables:**
- [ ] Test dataset management
- [ ] Inference engine implementations
- [ ] API endpoints for inference and test runs
- [ ] Frontend inference UI components
- [ ] Result visualization components
- [ ] Performance benchmarking

---

### Phase 4: Model Export System (Week 4-5)
**Goal:** Export trained models to deployment formats

#### 4.1 Export Formats
**Files:** `training/export/*.py`

Support multiple export formats:

```python
class ModelExporter(ABC):
    @abstractmethod
    def export(self, model_path: str, output_path: str, format: str):
        """Export model to specified format"""

# Supported formats per framework
EXPORT_FORMATS = {
    "pytorch": [
        "torchscript",  # .pt
        "onnx",         # .onnx
        "tensorrt",     # .engine (for NVIDIA GPUs)
        "coreml",       # .mlmodel (for iOS)
    ],
    "timm": [
        "onnx",
        "torchscript",
    ],
    "ultralytics": [
        "onnx",
        "tensorrt",
        "coreml",
        "tflite",       # TensorFlow Lite
        "edgetpu",      # Google Coral
    ]
}
```

#### 4.2 Export Configuration
**Files:** `app/schemas/export.py`

```python
class ExportConfig(BaseModel):
    format: str

    # ONNX options
    opset_version: int = 13
    dynamic_axes: bool = True

    # TensorRT options
    fp16: bool = True
    int8: bool = False
    max_batch_size: int = 1

    # Optimization options
    quantization: str | None = None  # dynamic, static, qat
    pruning: float | None = None  # pruning ratio

    # Metadata
    include_preprocessing: bool = True
    include_postprocessing: bool = True

class ExportJob(Base):
    __tablename__ = "export_jobs"

    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))
    checkpoint_path = Column(String(500))

    format = Column(String(50))
    config = Column(JSON)

    status = Column(String(20))
    output_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)  # bytes

    # Benchmarking
    inference_time_ms = Column(Float, nullable=True)
    throughput_fps = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
```

#### 4.3 Export API
**Files:** `app/api/export.py`

```python
@router.post("/export")
async def export_model(
    training_job_id: int,
    export_config: ExportConfig,
    checkpoint: str = "best"
):
    """Export trained model to deployment format"""

@router.get("/export/{export_job_id}")
async def get_export_status(export_job_id: int):
    """Get export job status"""

@router.get("/export/{export_job_id}/download")
async def download_exported_model(export_job_id: int):
    """Download exported model file"""

@router.post("/export/{export_job_id}/benchmark")
async def benchmark_exported_model(export_job_id: int):
    """Benchmark exported model performance"""
```

#### 4.4 Frontend Export UI
**Files:** `frontend/components/export/*.tsx`

- `ExportConfigPanel.tsx` - Configure export settings
- `ExportJobsList.tsx` - View export history
- `ExportBenchmark.tsx` - View benchmark results
- `DeploymentGuide.tsx` - Show deployment instructions per format

**Features:**
- Format-specific options (show only relevant configs)
- Size and performance comparison table
- Download exported models
- Deployment code snippets

**Deliverables:**
- [ ] Export implementations per format
- [ ] Database models for export jobs
- [ ] API endpoints for export operations
- [ ] Frontend export UI
- [ ] Benchmarking utilities
- [ ] Deployment documentation

---

### Phase 5: Training Control System (Week 5-6)
**Goal:** Advanced training control (pause, resume, distributed)

#### 5.1 Training State Management
**Files:** `app/db/models.py`, `training/state/*.py`

```python
class TrainingJob(Base):
    # ... existing fields ...

    # Control fields
    is_paused = Column(Boolean, default=False)
    pause_requested = Column(Boolean, default=False)
    stop_requested = Column(Boolean, default=False)

    # Resume support
    can_resume = Column(Boolean, default=False)
    resume_checkpoint = Column(String(500), nullable=True)
    resume_epoch = Column(Integer, nullable=True)

    # Distributed training
    world_size = Column(Integer, default=1)  # number of GPUs
    rank = Column(Integer, default=0)
    distributed_backend = Column(String(20), nullable=True)  # nccl, gloo

class TrainingCheckpoint(Base):
    __tablename__ = "training_checkpoints"

    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))

    epoch = Column(Integer)
    checkpoint_path = Column(String(500))
    checkpoint_type = Column(String(20))  # auto, best, manual

    metrics = Column(JSON)
    file_size = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)
```

#### 5.2 Training Control API
**Files:** `app/api/training_control.py`

```python
@router.post("/training/{job_id}/pause")
async def pause_training(job_id: int):
    """Request training pause"""

@router.post("/training/{job_id}/resume")
async def resume_training(job_id: int):
    """Resume paused training"""

@router.post("/training/{job_id}/stop")
async def stop_training(job_id: int):
    """Stop training gracefully"""

@router.get("/training/{job_id}/checkpoints")
async def list_checkpoints(job_id: int):
    """List all saved checkpoints"""

@router.post("/training/{job_id}/checkpoint")
async def save_manual_checkpoint(job_id: int):
    """Manually save checkpoint"""
```

#### 5.3 Distributed Training Support
**Files:** `training/distributed/*.py`

```python
class DistributedTrainer:
    def __init__(self, config: TrainingConfig, world_size: int):
        self.world_size = world_size
        self.backend = "nccl" if torch.cuda.is_available() else "gloo"

    def setup(self, rank: int):
        """Setup distributed process group"""

    def cleanup(self):
        """Cleanup distributed resources"""

    def train_distributed(self):
        """Main distributed training loop"""
```

Configuration for distributed training:
```python
class DistributedConfig(BaseModel):
    enabled: bool = False
    world_size: int = 1  # number of GPUs
    backend: str = "nccl"
    find_unused_parameters: bool = False

    # Data parallelism
    strategy: str = "ddp"  # ddp, fsdp, deepspeed
```

#### 5.4 Frontend Training Control UI
**Files:** `frontend/components/training/*.tsx`

- `TrainingControlPanel.tsx` - Pause/resume/stop buttons
- `CheckpointManager.tsx` - View and manage checkpoints
- `DistributedConfigPanel.tsx` - Configure distributed training
- `TrainingProgress.tsx` - Enhanced progress with control

**Features:**
- Real-time control buttons (pause/resume/stop)
- Checkpoint timeline visualization
- GPU utilization monitoring (if distributed)
- Confirm dialogs for destructive actions

**Deliverables:**
- [ ] Training state management
- [ ] Pause/resume/stop implementations
- [ ] Checkpoint management system
- [ ] Distributed training support
- [ ] Frontend control UI
- [ ] WebSocket for real-time control

---

## Additional Considerations

### Performance Optimization
- [ ] Implement data loading optimizations (prefetch, num_workers)
- [ ] Add mixed precision training (AMP) support
- [ ] Profile training performance and identify bottlenecks
- [ ] Optimize validation to avoid slowing down training

### Error Handling
- [ ] Implement comprehensive error handling in training loops
- [ ] Add automatic retry for transient failures
- [ ] Graceful degradation for missing features
- [ ] Clear error messages for user issues

### Monitoring & Logging
- [ ] Enhanced logging with structured logs
- [ ] Training progress WebSocket updates
- [ ] Resource usage monitoring (CPU, GPU, memory)
- [ ] Alert system for training failures

### Documentation
- [ ] User guide for advanced training features
- [ ] API documentation with examples
- [ ] Framework-specific best practices
- [ ] Troubleshooting guide

### Testing
- [ ] Unit tests for all new components
- [ ] Integration tests for training workflows
- [ ] End-to-end tests for complete training runs
- [ ] Performance benchmarks

---

## Dependencies & Requirements

### Backend
```python
# requirements.txt additions
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
ultralytics>=8.0.0
onnx>=1.14.0
onnxruntime>=1.15.0
tensorrt>=8.6.0  # optional
coremltools>=6.3  # optional
albumentations>=1.3.0  # for augmentation
```

### Frontend
```typescript
// package.json additions
"recharts": "^2.5.0",  // for charts
"react-dropzone": "^14.2.0",  // for file upload
"react-image-crop": "^10.1.0",  // for image cropping
```

### Infrastructure
- GPU support required for training
- Sufficient storage for checkpoints and results
- Redis for distributed training coordination (optional)

---

## Risk Assessment

### High Risk
- **Distributed training complexity** - May require significant debugging
  - Mitigation: Start with single GPU, add distributed incrementally

- **Framework compatibility** - Different frameworks have different APIs
  - Mitigation: Use adapter pattern, test thoroughly

### Medium Risk
- **Export format support** - Some formats may not work for all models
  - Mitigation: Document limitations, provide fallbacks

- **Performance bottlenecks** - Validation may slow down training
  - Mitigation: Profile and optimize, make validation async

### Low Risk
- **UI complexity** - Many configuration options may overwhelm users
  - Mitigation: Provide presets, collapsible sections, tooltips

---

## Success Metrics

### Functionality
- [ ] All 5 phases fully implemented and tested
- [ ] Support for 3+ frameworks (timm, Ultralytics, custom PyTorch)
- [ ] Support for 3+ task types (classification, detection, segmentation)

### Performance
- [ ] Training speed within 10% of baseline (without validation overhead)
- [ ] Inference latency < 100ms for typical models
- [ ] Export success rate > 95%

### Usability
- [ ] Complete training workflow < 5 clicks
- [ ] All features documented with examples
- [ ] < 5% error rate in user testing

---

## Cloud GPU Support

### Overview

The platform is designed from the ground up to support both **local** and **cloud GPU execution** transparently. The plugin architecture ensures that model developers write code once, and it runs anywhere.

**Detailed Documentation:** See [Cloud GPU Architecture](../architecture/CLOUD_GPU_ARCHITECTURE.md) for complete implementation details.

### Key Features

#### 1. Execution Strategy Pattern

```python
# Local or Cloud - same interface
class ExecutionStrategy(ABC):
    async def submit_job(job_id, model_plugin, config)
    async def get_status(execution_id)
    async def cancel_job(execution_id)
    async def get_metrics(execution_id)
```

**Local Executor:**
- Runs training as subprocess on local machine
- Direct file system access
- Zero network overhead
- Free (uses existing hardware)

**Cloud Executor (AWS/GCP/Azure):**
- Launches GPU container on cloud
- Uploads dataset to S3/GCS
- Downloads checkpoints and metrics via object storage
- Pay-per-use pricing
- Unlimited scalability

#### 2. Automatic Selection

Platform automatically chooses best execution location:

```python
dispatcher = JobDispatcher()
executor = await dispatcher.select_executor(config)

# Selection logic:
# 1. If local GPU available → Use local (free!)
# 2. If user specifies cloud → Use cloud
# 3. If local busy → Queue or use cloud
# 4. Compare costs → Choose cheapest option
```

#### 3. Transparent Communication

**Real-time updates work the same locally or in cloud:**

```
Local:
  Training Process → File System → Backend API → WebSocket → Frontend

Cloud:
  Training Container → S3 → Backend Monitor → WebSocket → Frontend
```

Frontend receives identical WebSocket messages regardless of execution location!

#### 4. Plugin Compatibility

**Model plugins don't change!**

```python
class MyModel(BaseModel):
    def train(self, train_loader, val_loader, callbacks):
        for epoch in range(epochs):
            loss = self.train_step(batch)

            # Callbacks work the same locally or in cloud
            callbacks.on_epoch_end(epoch, {'loss': loss})
```

Callbacks automatically:
- **Local**: Update database + WebSocket directly
- **Cloud**: Upload metrics to S3 → Monitor polls → Update database + WebSocket

### Architecture Flow

#### Cloud Training Flow

```
1. User submits training job via Frontend

2. Backend API:
   - Creates TrainingJob record
   - JobDispatcher selects CloudExecutor
   - CloudExecutor packages model code → uploads to S3
   - CloudExecutor uploads dataset → S3 (if needed)
   - CloudExecutor launches ECS/GKE container with GPU

3. GPU Container starts:
   - Downloads model package from S3
   - Downloads dataset from S3
   - Initializes model from plugin registry
   - Starts training

4. During Training:
   - CloudTrainingAgent reports metrics → S3
   - CloudTrainingAgent uploads checkpoints → S3
   - CloudTrainingAgent streams logs → CloudWatch/Stackdriver

5. Backend Monitor (every 10s):
   - Polls S3 for new metrics
   - Updates TrainingJob in database
   - Sends WebSocket updates to Frontend

6. Frontend:
   - Receives WebSocket updates
   - Updates training dashboard in real-time
   - Same UI as local training!

7. Training Completes:
   - Final checkpoint → S3
   - Container auto-terminates (no idle billing)
   - Backend updates job status
   - Frontend shows completion
```

### Cost Optimization

✅ **Dataset Caching**: Upload once, reuse for multiple jobs
✅ **Spot Instances**: 90% cost reduction for non-critical jobs
✅ **Auto-shutdown**: Container terminates immediately after training
✅ **Right-sizing**: Auto-select GPU instance based on model requirements
✅ **Compression**: Compress datasets and checkpoints for faster transfer

Example costs (AWS):
- g4dn.xlarge (1 GPU, 16GB): $0.526/hour
- p3.2xlarge (1 V100, 61GB): $3.06/hour
- Training 100 epochs (~2 hours): $1-6

### Implementation Timeline

**Phase 0 (Foundation) - Week 1:**
- [x] Design plugin architecture
- [ ] Implement ExecutionStrategy interface
- [ ] Implement LocalExecutor
- [ ] Implement JobDispatcher

**Phase 1 (Basic Cloud) - Week 2-3:**
- [ ] AWSExecutor with ECS
- [ ] CloudTrainingAgent for metric reporting
- [ ] S3 integration for datasets/checkpoints
- [ ] Monitoring service for cloud jobs
- [ ] Docker container for training

**Phase 2 (Optimization) - Week 4:**
- [ ] Spot instance support
- [ ] Dataset caching system
- [ ] Cost analytics dashboard
- [ ] Auto-scaling based on load

**Phase 3 (Multi-cloud) - Future:**
- [ ] GCP support (GKE + GCS)
- [ ] Azure support (AKS + Blob Storage)
- [ ] Hybrid scheduling strategies
- [ ] Cross-cloud cost comparison

### Developer Experience

**For model developers: ZERO changes needed!**

Write model once → runs on local GPU, AWS, GCP, Azure automatically.

```python
# This code works everywhere
class MyModel(BaseModel):
    def train(self, train_loader, val_loader, callbacks):
        # Your training logic
        pass
```

Platform handles:
- ✅ Packaging and deployment
- ✅ Dataset transfer
- ✅ Checkpoint management
- ✅ Metric reporting
- ✅ Cloud provisioning
- ✅ Cost tracking
- ✅ Failure recovery

---

## Future Enhancements (Post-MVP)

- **AutoML features**: Hyperparameter tuning, NAS
- **Dataset versioning**: Track dataset changes, data lineage
- **Model versioning**: Full model lifecycle management
- **A/B testing**: Compare multiple models in production
- **Edge deployment**: Deploy to mobile, embedded devices
- **Model monitoring**: Track model performance in production
- **Federated learning**: Distributed training across edge devices
- **Transfer learning UI**: Visual interface for fine-tuning
- **Multi-GPU orchestration**: Automatic workload distribution
- **Training analytics**: Detailed cost and performance analysis

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Advanced Config | 1-2 weeks | Config schemas, UI panels |
| Phase 2: Validation | 1 week | Metrics, results storage, dashboard |
| Phase 3: Test/Inference | 1 week | Inference engine, test runs, UI |
| Phase 4: Export | 1 week | Export formats, benchmarking |
| Phase 5: Control | 1 week | Pause/resume, distributed training |
| **Total** | **4-6 weeks** | **Production-ready training system** |

---

## Getting Started

1. Create new branch: `feat/advanced-trainer`
2. Start with Phase 1 (Advanced Configuration)
3. Implement and test each phase sequentially
4. Create feature branches for each phase if needed
5. Merge to main after thorough testing

---

*Document Version: 1.0*
*Last Updated: 2025-10-24*
*Author: Development Team*
