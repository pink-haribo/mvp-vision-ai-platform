# Test/Inference System Implementation Plan

## Overview

Implement a comprehensive, framework-agnostic test and inference system that allows users to:
- Run inference on single images with interactive visualization
- Run batch inference on multiple images
- Test trained models on test datasets with detailed metrics
- Compare results across different checkpoints
- Export inference results for downstream use

**Timeline:** 1-2 weeks
**Priority:** High - Essential for model deployment workflow
**Complexity:** Medium - Builds on existing validation system patterns

---

## Current State Analysis

### What We Have (from Validation System)

✅ **Database Infrastructure:**
- `validation_results`: Stores epoch-level metrics
- `validation_image_results`: Stores per-image predictions
- Task-agnostic schema with JSON flexibility

✅ **Adapter Pattern:**
- `TrainingAdapter` base class with `validate()` method
- `TimmAdapter`: Classification validation
- `UltralyticsAdapter`: Detection/segmentation/pose validation
- Standardized metrics computation via `ValidationMetricsCalculator`

✅ **Frontend Components:**
- Task-specific metric viewers (ClassificationMetricsView, DetectionMetricsView)
- Interactive image viewers with bbox/mask visualization
- Slide panels, navigation controls, filter options

✅ **API Design:**
- RESTful endpoints for querying results
- Filtering by class, correctness, confidence
- Image serving via FileResponse

### What's Missing for Test/Inference

❌ **Checkpoint Loading Infrastructure:**
- No standardized `load_checkpoint()` in base adapter
- No checkpoint metadata management
- No checkpoint selection UI

❌ **Inference Execution:**
- No standalone inference engine
- No preprocessing pipeline for ad-hoc images
- No postprocessing/visualization pipeline

❌ **Test Dataset Management:**
- No test dataset registration
- No test run tracking
- No test vs validation differentiation

❌ **Inference Result Storage:**
- No inference-specific database models
- No inference history tracking
- No result comparison features

---

## Design Principles

### 1. Framework Independence

**Goal:** Any framework can plug in inference capability with minimal code.

**Strategy:** Extend `TrainingAdapter` with standardized inference methods:
```python
class TrainingAdapter(ABC):
    # Existing methods...

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str, inference_mode: bool = True):
        """Load checkpoint for inference or training resume."""
        pass

    @abstractmethod
    def infer_single(self, image_path: str) -> InferenceResult:
        """Run inference on a single image."""
        pass

    @abstractmethod
    def infer_batch(self, image_paths: List[str]) -> List[InferenceResult]:
        """Run batch inference on multiple images."""
        pass

    def infer_dataset(self, dataset_path: str, split: str = "test") -> TestRunResult:
        """Run inference on entire dataset and compute metrics."""
        # Default implementation using infer_batch()
        pass
```

### 2. Reuse Validation Infrastructure

**Goal:** Avoid code duplication, leverage existing validation code.

**Strategy:**
- Reuse `ValidationMetricsCalculator` for test metrics
- Reuse visualization components for inference results
- Reuse database schemas with minor extensions

### 3. Unified Result Format

**Goal:** Consistent result structure across all task types.

**Strategy:** Use `InferenceResult` dataclass similar to `ValidationMetrics`:
```python
@dataclass
class InferenceResult:
    """Single image inference result."""
    image_path: str
    image_name: str
    task_type: TaskType

    # Classification
    predicted_label: Optional[str] = None
    predicted_label_id: Optional[int] = None
    confidence: Optional[float] = None
    top5_predictions: Optional[List[Dict]] = None

    # Detection
    predicted_boxes: Optional[List[Dict]] = None

    # Segmentation
    predicted_mask: Optional[np.ndarray] = None
    predicted_mask_path: Optional[str] = None

    # Pose
    predicted_keypoints: Optional[List[Dict]] = None

    # Common
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
```

### 4. Test vs Inference: Separation of Concerns

**Key Insight:** All major frameworks (TensorFlow, PyTorch, YOLO, HuggingFace) clearly separate `evaluate()` and `predict()`.

#### Industry Standard Pattern

```python
# TensorFlow/Keras
model.evaluate(test_dataset)  # With labels → metrics
model.predict(images)          # Without labels → predictions

# Ultralytics YOLO
model.val(data='coco.yaml')    # With labels → mAP, metrics
model.predict('image.jpg')     # Without labels → boxes

# HuggingFace
trainer.evaluate(dataset)      # With labels → metrics
pipeline(images)               # Without labels → predictions
```

#### Our Design: Hybrid Approach

**Core Inference Engine (Shared):**
```python
class TrainingAdapter(ABC):
    @abstractmethod
    def infer_single(image_path) -> InferenceResult:
        """Pure inference logic - label-agnostic."""
        pass
```

**Use Case Runners (Separated):**
```python
class TestRunner:
    """Test with LABELED data → compute metrics."""
    def run_test(dataset_path):
        # 1. Load labeled dataset
        # 2. Run inference (using adapter.infer_single)
        # 3. Compute metrics (compare with labels)
        # 4. Save to test_runs table
        pass

class InferenceRunner:
    """Inference with UNLABELED data → visualize."""
    def run_inference(image_paths):
        # 1. Run inference (using adapter.infer_single)
        # 2. Generate visualizations
        # 3. Save to inference_jobs table
        pass
```

#### Comparison Table

| Aspect | Test | Inference |
|--------|------|-----------|
| **Purpose** | Model evaluation | Production prediction |
| **Input** | Labeled dataset | Unlabeled images |
| **Output** | Metrics (quantitative) | Predictions (qualitative) |
| **Ground Truth** | Required | Not needed |
| **Use Case** | Post-training evaluation | Real-time service, batch processing |
| **Database** | `test_runs` + metrics | `inference_jobs` + visualizations |
| **UI Focus** | Confusion matrix, error analysis | Drag-and-drop, real-time feedback |

#### Connection to Export System

```
Training (PyTorch) → Test (validate accuracy) → Export (ONNX/TensorRT) → Inference (production)
                                                                           ↓
                                                                      실시간 서비스
                                                                      엣지 디바이스
```

**Key Points:**
- Test uses **original PyTorch model** for accuracy validation
- Exported model is used for **Inference only** (production deployment)
- Compare accuracy before/after export (optional test with exported model)

---

## Database Schema

### New Tables

#### 1. `test_runs`

Stores test run metadata and aggregate metrics.

```sql
CREATE TABLE test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_job_id INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    dataset_split TEXT DEFAULT 'test',

    -- Status
    status TEXT NOT NULL, -- pending, running, completed, failed

    -- Task info
    task_type TEXT NOT NULL,
    primary_metric_name TEXT,
    primary_metric_value REAL,

    -- Metrics (task-agnostic JSON)
    overall_loss REAL,
    metrics JSON,  -- {"accuracy": 0.95, "mAP50": 0.88, ...}
    per_class_metrics JSON,
    confusion_matrix JSON,

    -- Metadata
    class_names JSON,
    total_images INTEGER,
    inference_time_ms REAL,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    FOREIGN KEY (training_job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
);
```

#### 2. `test_image_results`

Stores per-image test results (similar to `validation_image_results`).

```sql
CREATE TABLE test_image_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_run_id INTEGER NOT NULL,

    -- Image info
    image_path TEXT,
    image_name TEXT NOT NULL,
    image_index INTEGER,

    -- Classification
    true_label TEXT,
    true_label_id INTEGER,
    predicted_label TEXT,
    predicted_label_id INTEGER,
    confidence REAL,
    top5_predictions JSON,

    -- Detection
    true_boxes JSON,
    predicted_boxes JSON,

    -- Segmentation
    true_mask_path TEXT,
    predicted_mask_path TEXT,

    -- Pose
    true_keypoints JSON,
    predicted_keypoints JSON,

    -- Metrics
    is_correct BOOLEAN,
    iou REAL,
    oks REAL,

    -- Performance
    inference_time_ms REAL,

    -- Extra data
    extra_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (test_run_id) REFERENCES test_runs(id) ON DELETE CASCADE
);

CREATE INDEX idx_test_image_results_test_run_id ON test_image_results(test_run_id);
CREATE INDEX idx_test_image_results_true_label_id ON test_image_results(true_label_id);
CREATE INDEX idx_test_image_results_predicted_label_id ON test_image_results(predicted_label_id);
CREATE INDEX idx_test_image_results_is_correct ON test_image_results(is_correct);
```

#### 3. `inference_jobs`

Stores inference job metadata (for ad-hoc inference without ground truth).

```sql
CREATE TABLE inference_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_job_id INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL,

    -- Input
    inference_type TEXT NOT NULL, -- single, batch, dataset
    input_data JSON,  -- {"image_paths": [...]} or {"dataset_path": "..."}

    -- Status
    status TEXT NOT NULL, -- pending, running, completed, failed
    error_message TEXT,

    -- Task info
    task_type TEXT NOT NULL,

    -- Performance
    total_images INTEGER DEFAULT 0,
    total_inference_time_ms REAL,
    avg_inference_time_ms REAL,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    FOREIGN KEY (training_job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
);
```

#### 4. `inference_results`

Stores per-image inference results (no ground truth).

```sql
CREATE TABLE inference_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_job_id INTEGER NOT NULL,

    -- Image info
    image_path TEXT NOT NULL,
    image_name TEXT NOT NULL,
    image_index INTEGER,

    -- Classification
    predicted_label TEXT,
    predicted_label_id INTEGER,
    confidence REAL,
    top5_predictions JSON,

    -- Detection
    predicted_boxes JSON,

    -- Segmentation
    predicted_mask_path TEXT,

    -- Pose
    predicted_keypoints JSON,

    -- Performance
    inference_time_ms REAL,
    preprocessing_time_ms REAL,
    postprocessing_time_ms REAL,

    -- Visualization
    visualization_path TEXT,  -- Path to annotated image

    -- Extra data
    extra_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (inference_job_id) REFERENCES inference_jobs(id) ON DELETE CASCADE
);

CREATE INDEX idx_inference_results_job_id ON inference_results(inference_job_id);
```

---

## Implementation Phases

### Phase 1: Adapter Extension (Days 1-2)

#### 1.1 Extend Base Adapter

**File:** `mvp/training/adapters/base.py`

Add inference methods to `TrainingAdapter`:

```python
class TrainingAdapter(ABC):
    # ... existing methods ...

    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_path: str,
        inference_mode: bool = True,
        device: str = None
    ) -> None:
        """
        Load checkpoint for inference or training resume.

        Args:
            checkpoint_path: Path to checkpoint file
            inference_mode: If True, load for inference (eval mode, no optimizer)
                           If False, load for training (restore optimizer, scheduler)
            device: Device to load model on (cuda/cpu), auto-detect if None
        """
        pass

    @abstractmethod
    def preprocess_image(self, image_path: str) -> Any:
        """
        Preprocess single image for inference.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor/array ready for model
        """
        pass

    @abstractmethod
    def infer_single(self, image_path: str) -> InferenceResult:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file

        Returns:
            InferenceResult with predictions
        """
        pass

    def infer_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> List[InferenceResult]:
        """
        Run batch inference on multiple images.

        Default implementation uses infer_single() in batches.
        Subclasses can override for optimized batching.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for inference

        Returns:
            List of InferenceResult
        """
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            for img_path in batch:
                result = self.infer_single(img_path)
                results.append(result)
        return results

    def test_dataset(
        self,
        dataset_path: str,
        split: str = "test",
        save_results: bool = True
    ) -> TestRunResult:
        """
        Run inference on entire dataset and compute metrics.

        This is similar to validate() but runs on test set after training.

        Args:
            dataset_path: Path to dataset
            split: Dataset split to test on (test, val, train)
            save_results: If True, save to test_runs table

        Returns:
            TestRunResult with metrics and per-image results
        """
        # Load test dataset
        test_loader = self._load_dataset(dataset_path, split)

        # Run inference on all images
        all_predictions = []
        all_labels = []
        image_results = []

        for batch in test_loader:
            # ... run inference ...
            pass

        # Compute metrics using ValidationMetricsCalculator
        from mvp.training.validators.metrics import ValidationMetricsCalculator

        test_metrics = ValidationMetricsCalculator.compute_metrics(
            task_type=self.task_type,
            predictions=all_predictions,
            labels=all_labels,
            class_names=self.class_names
        )

        # Save results if requested
        if save_results:
            test_run_id = self._save_test_run(test_metrics, image_results)

        return TestRunResult(
            metrics=test_metrics,
            image_results=image_results,
            test_run_id=test_run_id if save_results else None
        )
```

#### 1.2 Implement in TimmAdapter

**File:** `mvp/training/adapters/timm_adapter.py`

```python
class TimmAdapter(TrainingAdapter):
    # ... existing methods ...

    def load_checkpoint(
        self,
        checkpoint_path: str,
        inference_mode: bool = True,
        device: str = None
    ) -> None:
        """Load checkpoint for timm classification model."""
        import torch

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        # Set to eval mode for inference
        if inference_mode:
            self.model.eval()

        print(f"[TimmAdapter] Loaded checkpoint from {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'accuracy' in checkpoint:
            print(f"  Accuracy: {checkpoint['accuracy']:.4f}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image for inference."""
        from PIL import Image
        import torch

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply validation transforms
        if self.val_transforms is None:
            self.val_transforms = self.build_val_transforms()

        tensor = self.val_transforms(image)

        # Add batch dimension
        return tensor.unsqueeze(0)

    def infer_single(self, image_path: str) -> InferenceResult:
        """Run inference on single image."""
        import torch
        import time
        from pathlib import Path

        # Timing
        start_time = time.time()

        # Preprocess
        input_tensor = self.preprocess_image(image_path)
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        preprocess_time = (time.time() - start_time) * 1000

        # Inference
        infer_start = time.time()
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)

        inference_time = (time.time() - infer_start) * 1000

        # Postprocessing
        post_start = time.time()

        # Top-1 prediction
        confidence, pred_id = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred_id = pred_id.item()
        predicted_label = self.class_names[pred_id] if self.class_names else str(pred_id)

        # Top-5 predictions
        top5_probs, top5_ids = torch.topk(probs, min(5, len(self.class_names)), dim=1)
        top5_predictions = [
            {
                'label_id': int(top5_ids[0, i].item()),
                'label': self.class_names[int(top5_ids[0, i].item())],
                'confidence': float(top5_probs[0, i].item())
            }
            for i in range(top5_probs.size(1))
        ]

        postprocess_time = (time.time() - post_start) * 1000

        return InferenceResult(
            image_path=image_path,
            image_name=Path(image_path).name,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            predicted_label=predicted_label,
            predicted_label_id=pred_id,
            confidence=confidence,
            top5_predictions=top5_predictions,
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocess_time,
            postprocessing_time_ms=postprocess_time
        )
```

#### 1.3 Implement in UltralyticsAdapter

**File:** `mvp/training/adapters/ultralytics_adapter.py`

```python
class UltralyticsAdapter(TrainingAdapter):
    # ... existing methods ...

    def load_checkpoint(
        self,
        checkpoint_path: str,
        inference_mode: bool = True,
        device: str = None
    ) -> None:
        """Load checkpoint for YOLO model."""
        from ultralytics import YOLO

        # YOLO handles checkpoint loading internally
        self.model = YOLO(checkpoint_path)

        if device:
            self.model.to(device)

        print(f"[UltralyticsAdapter] Loaded YOLO checkpoint from {checkpoint_path}")

    def preprocess_image(self, image_path: str):
        """
        Preprocess not needed for YOLO - handles internally.
        Return image_path as-is.
        """
        return image_path

    def infer_single(self, image_path: str) -> InferenceResult:
        """Run inference on single image with YOLO."""
        import time
        from pathlib import Path

        # YOLO inference
        start_time = time.time()
        results = self.model(image_path, verbose=False)
        inference_time = (time.time() - start_time) * 1000

        result = results[0]

        # Extract predictions based on task
        if self.task_type == TaskType.OBJECT_DETECTION:
            return self._extract_detection_result(
                result, image_path, inference_time
            )
        elif self.task_type == TaskType.INSTANCE_SEGMENTATION:
            return self._extract_segmentation_result(
                result, image_path, inference_time
            )
        elif self.task_type == TaskType.POSE_ESTIMATION:
            return self._extract_pose_result(
                result, image_path, inference_time
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _extract_detection_result(
        self,
        result,
        image_path: str,
        inference_time: float
    ) -> InferenceResult:
        """Extract detection predictions from YOLO result."""
        from pathlib import Path

        # Extract boxes
        boxes = result.boxes
        predicted_boxes = []

        for i in range(len(boxes)):
            box = boxes[i]
            predicted_boxes.append({
                'class_id': int(box.cls.item()),
                'bbox': box.xywh[0].cpu().tolist(),  # [x_center, y_center, w, h]
                'confidence': float(box.conf.item()),
                'format': 'yolo'
            })

        return InferenceResult(
            image_path=image_path,
            image_name=Path(image_path).name,
            task_type=TaskType.OBJECT_DETECTION,
            predicted_boxes=predicted_boxes,
            inference_time_ms=inference_time,
            preprocessing_time_ms=0.0,  # YOLO handles internally
            postprocessing_time_ms=0.0
        )
```

**Deliverables:**
- [ ] Extended `TrainingAdapter` base class with inference methods
- [ ] `load_checkpoint()` implemented in TimmAdapter
- [ ] `load_checkpoint()` implemented in UltralyticsAdapter
- [ ] `infer_single()` implemented in TimmAdapter
- [ ] `infer_single()` implemented in UltralyticsAdapter
- [ ] `infer_batch()` with default implementation
- [ ] Unit tests for inference methods

---

### Phase 2: Backend API (Days 3-4)

#### 2.1 Pydantic Schemas

**File:** `mvp/backend/app/schemas/inference.py`

```python
"""Inference API schemas."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime


# ========== Inference Request/Response ==========

class SingleInferenceRequest(BaseModel):
    """Request for single image inference."""
    training_job_id: int
    checkpoint: str = "best"  # "best", "last", or epoch number

class BatchInferenceRequest(BaseModel):
    """Request for batch inference."""
    training_job_id: int
    checkpoint: str = "best"
    batch_size: int = 32

class InferenceResultResponse(BaseModel):
    """Single image inference result."""
    id: int
    inference_job_id: int
    image_path: str
    image_name: str
    image_index: Optional[int] = None

    # Classification
    predicted_label: Optional[str] = None
    predicted_label_id: Optional[int] = None
    confidence: Optional[float] = None
    top5_predictions: Optional[List[Dict[str, Any]]] = None

    # Detection
    predicted_boxes: Optional[List[Dict[str, Any]]] = None

    # Segmentation
    predicted_mask_path: Optional[str] = None

    # Pose
    predicted_keypoints: Optional[List[Dict[str, Any]]] = None

    # Performance
    inference_time_ms: float
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0

    # Visualization
    visualization_path: Optional[str] = None

    created_at: datetime

    class Config:
        from_attributes = True

class InferenceJobResponse(BaseModel):
    """Inference job metadata."""
    id: int
    training_job_id: int
    checkpoint_path: str
    inference_type: str
    task_type: str
    status: str
    total_images: int
    total_inference_time_ms: Optional[float] = None
    avg_inference_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class InferenceJobListResponse(BaseModel):
    """List of inference jobs."""
    total_count: int
    jobs: List[InferenceJobResponse]


# ========== Test Run Request/Response ==========

class TestRunRequest(BaseModel):
    """Request to create a test run."""
    training_job_id: int
    checkpoint: str = "best"
    dataset_path: str
    dataset_split: str = "test"

class TestRunResponse(BaseModel):
    """Test run result."""
    id: int
    training_job_id: int
    checkpoint_path: str
    dataset_path: str
    dataset_split: str
    status: str
    task_type: str
    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[float] = None
    overall_loss: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    per_class_metrics: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    class_names: Optional[List[str]] = None
    total_images: int
    inference_time_ms: Optional[float] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class TestRunListResponse(BaseModel):
    """List of test runs."""
    training_job_id: int
    total_count: int
    runs: List[TestRunResponse]

class TestImageResultResponse(BaseModel):
    """Per-image test result."""
    id: int
    test_run_id: int
    image_path: Optional[str] = None
    image_name: str
    image_index: Optional[int] = None

    # Classification
    true_label: Optional[str] = None
    true_label_id: Optional[int] = None
    predicted_label: Optional[str] = None
    predicted_label_id: Optional[int] = None
    confidence: Optional[float] = None
    top5_predictions: Optional[List[Dict[str, Any]]] = None

    # Detection
    true_boxes: Optional[List[Dict[str, Any]]] = None
    predicted_boxes: Optional[List[Dict[str, Any]]] = None

    # Segmentation
    true_mask_path: Optional[str] = None
    predicted_mask_path: Optional[str] = None

    # Pose
    true_keypoints: Optional[List[Dict[str, Any]]] = None
    predicted_keypoints: Optional[List[Dict[str, Any]]] = None

    # Metrics
    is_correct: bool
    iou: Optional[float] = None
    oks: Optional[float] = None
    inference_time_ms: Optional[float] = None

    extra_data: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class TestImageResultListResponse(BaseModel):
    """List of test image results."""
    test_run_id: int
    total_count: int
    correct_count: int
    incorrect_count: int
    class_names: Optional[List[str]] = None
    images: List[TestImageResultResponse]
```

#### 2.2 Database Models

**File:** `mvp/backend/app/db/models.py`

Add new models:

```python
class TestRun(Base):
    __tablename__ = "test_runs"

    id = Column(Integer, primary_key=True, index=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)
    dataset_path = Column(String(500), nullable=False)
    dataset_split = Column(String(20), default="test")

    # Status
    status = Column(String(20), nullable=False)  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)

    # Task info
    task_type = Column(String(50), nullable=False)
    primary_metric_name = Column(String(50), nullable=True)
    primary_metric_value = Column(Float, nullable=True)

    # Metrics
    overall_loss = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)
    per_class_metrics = Column(JSON, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)

    # Metadata
    class_names = Column(JSON, nullable=True)
    total_images = Column(Integer, default=0)
    inference_time_ms = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="test_runs")
    image_results = relationship("TestImageResult", back_populates="test_run", cascade="all, delete-orphan")


class TestImageResult(Base):
    __tablename__ = "test_image_results"

    id = Column(Integer, primary_key=True, index=True)
    test_run_id = Column(Integer, ForeignKey("test_runs.id", ondelete="CASCADE"), nullable=False)

    # Image info
    image_path = Column(String(500), nullable=True)
    image_name = Column(String(200), nullable=False)
    image_index = Column(Integer, nullable=True)

    # Classification
    true_label = Column(String(100), nullable=True)
    true_label_id = Column(Integer, nullable=True)
    predicted_label = Column(String(100), nullable=True)
    predicted_label_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    top5_predictions = Column(JSON, nullable=True)

    # Detection
    true_boxes = Column(JSON, nullable=True)
    predicted_boxes = Column(JSON, nullable=True)

    # Segmentation
    true_mask_path = Column(String(500), nullable=True)
    predicted_mask_path = Column(String(500), nullable=True)

    # Pose
    true_keypoints = Column(JSON, nullable=True)
    predicted_keypoints = Column(JSON, nullable=True)

    # Metrics
    is_correct = Column(Boolean, nullable=False)
    iou = Column(Float, nullable=True)
    oks = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=True)

    # Extra
    extra_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    test_run = relationship("TestRun", back_populates="image_results")


class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id = Column(Integer, primary_key=True, index=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)

    # Input
    inference_type = Column(String(20), nullable=False)  # single, batch, dataset
    input_data = Column(JSON, nullable=True)

    # Status
    status = Column(String(20), nullable=False)
    error_message = Column(Text, nullable=True)

    # Task info
    task_type = Column(String(50), nullable=False)

    # Performance
    total_images = Column(Integer, default=0)
    total_inference_time_ms = Column(Float, nullable=True)
    avg_inference_time_ms = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="inference_jobs")
    results = relationship("InferenceResult", back_populates="inference_job", cascade="all, delete-orphan")


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)
    inference_job_id = Column(Integer, ForeignKey("inference_jobs.id", ondelete="CASCADE"), nullable=False)

    # Image info
    image_path = Column(String(500), nullable=False)
    image_name = Column(String(200), nullable=False)
    image_index = Column(Integer, nullable=True)

    # Classification
    predicted_label = Column(String(100), nullable=True)
    predicted_label_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    top5_predictions = Column(JSON, nullable=True)

    # Detection
    predicted_boxes = Column(JSON, nullable=True)

    # Segmentation
    predicted_mask_path = Column(String(500), nullable=True)

    # Pose
    predicted_keypoints = Column(JSON, nullable=True)

    # Performance
    inference_time_ms = Column(Float, nullable=False)
    preprocessing_time_ms = Column(Float, default=0.0)
    postprocessing_time_ms = Column(Float, default=0.0)

    # Visualization
    visualization_path = Column(String(500), nullable=True)

    # Extra
    extra_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    inference_job = relationship("InferenceJob", back_populates="results")
```

Add relationships to `TrainingJob`:

```python
class TrainingJob(Base):
    # ... existing fields ...

    # Relationships
    test_runs = relationship("TestRun", back_populates="training_job", cascade="all, delete-orphan")
    inference_jobs = relationship("InferenceJob", back_populates="training_job", cascade="all, delete-orphan")
```

#### 2.3 API Endpoints

**File:** `mvp/backend/app/api/inference.py`

```python
"""Inference API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from pathlib import Path

from app.db.database import get_db
from app.db import models
from app.schemas import inference as inference_schemas


router = APIRouter(prefix="/inference", tags=["inference"])


# ========== Single Image Inference ==========

@router.post("/single", response_model=inference_schemas.InferenceResultResponse)
async def infer_single_image(
    training_job_id: int,
    image: UploadFile = File(...),
    checkpoint: str = "best",
    db: Session = Depends(get_db)
):
    """
    Run inference on a single uploaded image.

    Args:
        training_job_id: Training job ID
        image: Uploaded image file
        checkpoint: Checkpoint to use ("best", "last", or epoch number)
        db: Database session

    Returns:
        InferenceResultResponse with prediction
    """
    # Verify job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == training_job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {training_job_id} not found")

    # Resolve checkpoint path
    checkpoint_path = _resolve_checkpoint_path(job, checkpoint)
    if not Path(checkpoint_path).exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

    # Save uploaded image temporarily
    import tempfile
    import shutil

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        # Create inference job
        inference_job = models.InferenceJob(
            training_job_id=training_job_id,
            checkpoint_path=checkpoint_path,
            inference_type="single",
            status="running",
            task_type=job.task_type,
            input_data={"image_name": image.filename}
        )
        db.add(inference_job)
        db.commit()
        db.refresh(inference_job)

        # Run inference
        from app.utils.inference_runner import InferenceRunner

        runner = InferenceRunner(job, checkpoint_path)
        result = runner.infer_single(tmp_path)

        # Save result to database
        inference_result = models.InferenceResult(
            inference_job_id=inference_job.id,
            image_path=tmp_path,
            image_name=image.filename,
            predicted_label=result.predicted_label,
            predicted_label_id=result.predicted_label_id,
            confidence=result.confidence,
            top5_predictions=result.top5_predictions,
            predicted_boxes=result.predicted_boxes,
            predicted_mask_path=result.predicted_mask_path,
            predicted_keypoints=result.predicted_keypoints,
            inference_time_ms=result.inference_time_ms,
            preprocessing_time_ms=result.preprocessing_time_ms,
            postprocessing_time_ms=result.postprocessing_time_ms
        )
        db.add(inference_result)

        # Update job status
        inference_job.status = "completed"
        inference_job.total_images = 1
        inference_job.total_inference_time_ms = result.inference_time_ms
        inference_job.avg_inference_time_ms = result.inference_time_ms

        db.commit()
        db.refresh(inference_result)

        return inference_result

    except Exception as e:
        # Update job with error
        inference_job.status = "failed"
        inference_job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)


# ========== Batch Inference ==========

@router.post("/batch", response_model=inference_schemas.InferenceJobResponse)
async def infer_batch_images(
    training_job_id: int,
    images: List[UploadFile] = File(...),
    checkpoint: str = "best",
    batch_size: int = 32,
    db: Session = Depends(get_db)
):
    """
    Run inference on multiple uploaded images.

    Args:
        training_job_id: Training job ID
        images: List of uploaded image files
        checkpoint: Checkpoint to use
        batch_size: Batch size for inference
        db: Database session

    Returns:
        InferenceJobResponse with job ID (async processing)
    """
    # Verify job exists
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == training_job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {training_job_id} not found")

    # Resolve checkpoint path
    checkpoint_path = _resolve_checkpoint_path(job, checkpoint)
    if not Path(checkpoint_path).exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

    # Create inference job
    inference_job = models.InferenceJob(
        training_job_id=training_job_id,
        checkpoint_path=checkpoint_path,
        inference_type="batch",
        status="pending",
        task_type=job.task_type,
        input_data={"num_images": len(images), "batch_size": batch_size}
    )
    db.add(inference_job)
    db.commit()
    db.refresh(inference_job)

    # TODO: Queue for async processing
    # For MVP, run synchronously

    return inference_job


# ========== Test Run ==========

@router.post("/test-runs", response_model=inference_schemas.TestRunResponse)
async def create_test_run(
    request: inference_schemas.TestRunRequest,
    db: Session = Depends(get_db)
):
    """
    Create a test run on a dataset.

    Args:
        request: TestRunRequest with job_id, checkpoint, dataset_path
        db: Database session

    Returns:
        TestRunResponse with test run ID (async processing)
    """
    # Verify job exists
    job = db.query(models.TrainingJob).filter(
        models.TrainingJob.id == request.training_job_id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {request.training_job_id} not found")

    # Resolve checkpoint path
    checkpoint_path = _resolve_checkpoint_path(job, request.checkpoint)
    if not Path(checkpoint_path).exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

    # Verify dataset exists
    if not Path(request.dataset_path).exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

    # Create test run
    test_run = models.TestRun(
        training_job_id=request.training_job_id,
        checkpoint_path=checkpoint_path,
        dataset_path=request.dataset_path,
        dataset_split=request.dataset_split,
        status="pending",
        task_type=job.task_type
    )
    db.add(test_run)
    db.commit()
    db.refresh(test_run)

    # TODO: Queue for async processing
    # For MVP, run synchronously in subprocess

    return test_run


@router.get("/test-runs/{test_run_id}", response_model=inference_schemas.TestRunResponse)
async def get_test_run(
    test_run_id: int,
    db: Session = Depends(get_db)
):
    """Get test run by ID."""
    test_run = db.query(models.TestRun).filter(models.TestRun.id == test_run_id).first()
    if not test_run:
        raise HTTPException(status_code=404, detail=f"Test run {test_run_id} not found")
    return test_run


@router.get("/jobs/{training_job_id}/test-runs", response_model=inference_schemas.TestRunListResponse)
async def list_test_runs(
    training_job_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all test runs for a training job."""
    query = db.query(models.TestRun).filter(
        models.TestRun.training_job_id == training_job_id
    ).order_by(models.TestRun.created_at.desc())

    total_count = query.count()
    runs = query.offset(skip).limit(limit).all()

    return inference_schemas.TestRunListResponse(
        training_job_id=training_job_id,
        total_count=total_count,
        runs=runs
    )


# ========== Helper Functions ==========

def _resolve_checkpoint_path(job: models.TrainingJob, checkpoint: str) -> str:
    """
    Resolve checkpoint identifier to actual file path.

    Args:
        job: TrainingJob instance
        checkpoint: "best", "last", or epoch number

    Returns:
        Path to checkpoint file
    """
    from pathlib import Path

    output_dir = Path(job.output_dir)

    if checkpoint == "best":
        # Find best checkpoint based on metrics
        # For now, use simple naming convention
        checkpoint_path = output_dir / "checkpoints" / "best.pth"
    elif checkpoint == "last":
        checkpoint_path = output_dir / "checkpoints" / "last.pth"
    else:
        # Assume epoch number
        try:
            epoch = int(checkpoint)
            checkpoint_path = output_dir / "checkpoints" / f"epoch_{epoch}.pth"
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid checkpoint: {checkpoint}")

    return str(checkpoint_path)
```

**Deliverables:**
- [ ] Pydantic schemas for inference API
- [ ] Database models (TestRun, TestImageResult, InferenceJob, InferenceResult)
- [ ] Database migration script
- [ ] API endpoints for inference and test runs
- [ ] InferenceRunner utility class
- [ ] Unit tests for API endpoints

---

### Phase 3: Frontend UI (Days 5-7)

#### 3.1 Single Image Inference UI

**File:** `mvp/frontend/components/inference/SingleImageInference.tsx`

Features:
- Drag-and-drop or browse to upload image
- Checkpoint selector (best, last, specific epoch)
- Real-time inference with loading state
- Task-specific visualization:
  - Classification: Top-5 predictions with confidence bars
  - Detection: Image with bbox overlays
  - Segmentation: Image with mask overlay
  - Pose: Image with keypoint skeleton
- Performance metrics (inference time, FPS)
- Download annotated image button

#### 3.2 Test Run UI

**File:** `mvp/frontend/components/inference/TestRunCreator.tsx`

Features:
- Training job selector
- Checkpoint selector
- Dataset path input with validation
- Split selector (test/val/train)
- Create test run button
- Progress indicator

**File:** `mvp/frontend/components/inference/TestRunViewer.tsx`

Features:
- Reuse ValidationMetricsView components (ClassificationMetricsView, DetectionMetricsView)
- Display test metrics (accuracy, mAP, etc.)
- Per-class metrics table
- Confusion matrix (classification)
- Per-image results table with filtering
- Compare with validation results (side-by-side)

#### 3.3 Inference History UI

**File:** `mvp/frontend/components/inference/InferenceHistory.tsx`

Features:
- List of all inference jobs and test runs
- Status indicators (pending, running, completed, failed)
- Filter by status, checkpoint, date range
- View results button
- Delete button

**Deliverables:**
- [ ] SingleImageInference component
- [ ] TestRunCreator component
- [ ] TestRunViewer component (reuses validation viewers)
- [ ] InferenceHistory component
- [ ] API integration with React Query
- [ ] E2E tests

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Adapter Extension | 2 days | Inference methods in adapters |
| Phase 2: Backend API | 2 days | Database, schemas, API endpoints |
| Phase 3: Frontend UI | 3 days | Inference UI components |
| **Total** | **7 days (1-2 weeks)** | **Complete test/inference system** |

---

## Success Metrics

### Functionality
- [ ] Single image inference works for all task types
- [ ] Batch inference handles 100+ images
- [ ] Test runs compute correct metrics
- [ ] Results match validation system format

### Performance
- [ ] Single image inference < 100ms (excluding model time)
- [ ] Batch inference throughput > 10 images/sec
- [ ] UI responds within 500ms for single inference

### Usability
- [ ] Drag-and-drop upload works smoothly
- [ ] Checkpoint selection is intuitive
- [ ] Results are visualized clearly
- [ ] Test run creation < 3 clicks

---

## Future Enhancements (Post-MVP)

- **Model comparison**: Side-by-side comparison of multiple checkpoints
- **Batch download**: Download all inference results as ZIP
- **Inference API**: RESTful API for programmatic inference
- **TorchScript export**: Export optimized model for deployment
- **Inference caching**: Cache recent inference results
- **Grad-CAM visualization**: Show model attention for classification
- **Confidence filtering**: Filter predictions by confidence threshold
- **Export formats**: Export results to CSV, JSON, COCO format

---

*Document Version: 1.0*
*Last Updated: 2025-10-29*
*Author: Development Team*
