# Validation System Implementation Plan (Task-Agnostic)

**Document Version:** 2.0
**Last Updated:** 2025-10-28
**Priority:** High
**Timeline:** 1-2 weeks
**Design Philosophy:** Task-Agnostic from Day 1

---

## Executive Summary

본 문서는 Vision AI Training Platform의 **Task-Agnostic Validation System** 구현 계획을 정의합니다.

### 핵심 설계 원칙: **범용성 (Task-Agnostic)**

처음부터 **모든 컴퓨터 비전 태스크**(Classification, Detection, Segmentation, Pose Estimation 등)를 지원하는 범용적인 구조로 설계합니다.

**지원 태스크**:
- ✅ Image Classification (ResNet, EfficientNet, ViT)
- ✅ Object Detection (YOLO, Faster R-CNN, DETR)
- ✅ Instance Segmentation (Mask R-CNN, YOLO Segment)
- ✅ Semantic Segmentation (DeepLab, U-Net)
- ✅ Pose Estimation (YOLO Pose, OpenPose)
- ✅ 미래 태스크 (Image Captioning, VQA, etc.)

### 목표

- ✅ **범용 메트릭 시스템**: Task에 관계없이 동일한 인터페이스
- ✅ **상세 메트릭**: Confusion matrix, per-class metrics, mAP, IoU 등
- ✅ **이미지별 분석**: 각 validation 이미지의 prediction vs ground truth
- ✅ **태스크별 시각화**: Task에 맞는 최적의 시각화 자동 선택
- ✅ **확장성**: 새 태스크 추가 시 최소한의 코드 변경

### 사용자 경험

**Task에 관계없이 동일한 Validation 탭**:

```
[Validation 탭]
├─ Epoch Selector (공통)
├─ Overall Metrics (공통)
├─ Task-Specific Visualization (자동 선택)
│  ├─ Classification → Confusion Matrix + Per-Class Metrics
│  ├─ Detection → mAP Charts + Precision-Recall Curves
│  ├─ Segmentation → IoU per Class + Pixel Accuracy
│  └─ Pose → OKS per Keypoint + PCK Curves
└─ Image Gallery (공통) → Click for detail (task-specific overlay)
```

사용자는 **태스크를 의식하지 않고** 항상 동일한 위치에서 validation 결과를 확인합니다.

---

## Design Philosophy: Task-Agnostic Architecture

### 1. Database: Task-Agnostic Schema

**원칙**: 모든 태스크의 메트릭을 JSON으로 유연하게 저장

```python
class ValidationResult(Base):
    """
    TASK-AGNOSTIC: Classification, Detection, Segmentation, Pose 모두 저장 가능
    """
    # Task 식별
    task_type = Column(String(50))  # "image_classification", "object_detection", etc.

    # 공통 메트릭
    primary_metric_value = Column(Float)  # Task마다 다름 (accuracy, mAP50, mean_IoU, OKS)
    overall_loss = Column(Float)

    # Task-specific 메트릭 (JSON으로 유연하게)
    metrics = Column(JSON)
    # Classification: {"accuracy": 0.89, "top5_accuracy": 0.98, "macro_f1": 0.87}
    # Detection: {"mAP50": 0.75, "mAP50_95": 0.62, "precision": 0.82}
    # Segmentation: {"mean_iou": 0.78, "pixel_accuracy": 0.92, "dice": 0.85}

    # Per-class 메트릭 (모든 태스크 지원)
    per_class_metrics = Column(JSON)
    # Classification: {"cat": {"precision": 0.85, "recall": 0.82, "f1": 0.835}}
    # Detection: {"car": {"ap": 0.92}, "person": {"ap": 0.87}}
    # Segmentation: {"road": {"iou": 0.89}, "building": {"iou": 0.82}}

    # Task-specific visualization data
    confusion_matrix = Column(JSON)  # Classification only
    pr_curves = Column(JSON)  # Detection, Segmentation
    class_names = Column(JSON)  # All tasks
```

```python
class ValidationImageResult(Base):
    """
    TASK-AGNOSTIC: 모든 태스크의 이미지별 결과 저장
    """
    # Classification
    true_label = Column(String(100), nullable=True)
    predicted_label = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    top5_predictions = Column(JSON, nullable=True)

    # Detection
    true_boxes = Column(JSON, nullable=True)  # [{"x", "y", "w", "h", "class"}]
    predicted_boxes = Column(JSON, nullable=True)

    # Segmentation
    true_mask_path = Column(String(500), nullable=True)
    predicted_mask_path = Column(String(500), nullable=True)

    # Pose Estimation
    true_keypoints = Column(JSON, nullable=True)  # [{"x", "y", "visibility"}]
    predicted_keypoints = Column(JSON, nullable=True)

    # 공통 메트릭
    is_correct = Column(Boolean)  # Task마다 정의 다름
    iou = Column(Float, nullable=True)  # Detection, Segmentation 공통

    # 확장 가능
    extra_data = Column(JSON, nullable=True)
```

### 2. Metrics System: Unified Interface

**원칙**: Task에 관계없이 동일한 메서드 호출

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

class TaskType(Enum):
    """지원하는 모든 태스크"""
    CLASSIFICATION = "image_classification"
    DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POSE = "pose_estimation"
    # 미래 확장
    IMAGE_CAPTIONING = "image_captioning"
    VQA = "visual_question_answering"

@dataclass
class ValidationMetrics:
    """
    통합 메트릭 컨테이너 - 모든 태스크 지원
    """
    task_type: TaskType

    # 공통 필드
    primary_metric: float  # Task마다 다름
    primary_metric_name: str  # "accuracy", "mAP50", "mean_IoU", etc.
    loss: float

    # Task-specific 데이터 (Optional)
    classification_data: Optional['ClassificationMetrics'] = None
    detection_data: Optional['DetectionMetrics'] = None
    segmentation_data: Optional['SegmentationMetrics'] = None
    pose_data: Optional['PoseMetrics'] = None

    # JSON serialization을 위한 dict
    def to_dict(self) -> Dict:
        """DB 저장용 dict 변환"""
        result = {
            "task_type": self.task_type.value,
            "primary_metric": self.primary_metric,
            "primary_metric_name": self.primary_metric_name,
            "loss": self.loss
        }

        if self.classification_data:
            result.update(self.classification_data.to_dict())
        elif self.detection_data:
            result.update(self.detection_data.to_dict())
        # ... 다른 태스크들

        return result

class ValidationMetricsCalculator:
    """
    TASK-AGNOSTIC 메트릭 계산기

    모든 태스크에 대해 동일한 인터페이스 제공
    """

    @staticmethod
    def compute_metrics(
        task_type: TaskType,
        predictions: Any,
        labels: Any,
        **kwargs
    ) -> ValidationMetrics:
        """
        Task에 따라 자동으로 적절한 메트릭 계산

        Args:
            task_type: TaskType enum
            predictions: 모델 예측 결과 (task마다 형식 다름)
            labels: Ground truth (task마다 형식 다름)
            **kwargs: Task-specific 추가 파라미터

        Returns:
            ValidationMetrics (통합 메트릭 객체)
        """
        if task_type == TaskType.CLASSIFICATION:
            return ValidationMetricsCalculator._compute_classification(predictions, labels, **kwargs)
        elif task_type == TaskType.DETECTION:
            return ValidationMetricsCalculator._compute_detection(predictions, labels, **kwargs)
        elif task_type == TaskType.INSTANCE_SEGMENTATION:
            return ValidationMetricsCalculator._compute_segmentation(predictions, labels, **kwargs)
        elif task_type == TaskType.POSE:
            return ValidationMetricsCalculator._compute_pose(predictions, labels, **kwargs)
        else:
            raise NotImplementedError(f"Task type {task_type} not yet implemented")

    @staticmethod
    def _compute_classification(...) -> ValidationMetrics:
        """Classification 메트릭 계산"""
        # Confusion matrix, per-class metrics, top-k accuracy
        ...
        return ValidationMetrics(
            task_type=TaskType.CLASSIFICATION,
            primary_metric=accuracy,
            primary_metric_name="accuracy",
            loss=loss,
            classification_data=ClassificationMetrics(...)
        )

    @staticmethod
    def _compute_detection(...) -> ValidationMetrics:
        """Detection 메트릭 계산"""
        # mAP, per-class AP, precision-recall curves
        ...
        return ValidationMetrics(
            task_type=TaskType.DETECTION,
            primary_metric=mAP50,
            primary_metric_name="mAP50",
            loss=loss,
            detection_data=DetectionMetrics(...)
        )

    # ... 다른 태스크들
```

### 3. Adapter Integration: Unified Interface

**원칙**: 모든 Adapter가 동일한 validate() 인터페이스 구현

```python
# base.py
class TrainingAdapter(ABC):
    """모든 Adapter의 베이스 클래스"""

    @abstractmethod
    def validate(self, epoch: int) -> MetricsResult:
        """
        TASK-AGNOSTIC validation 인터페이스

        모든 Adapter는 이 메서드를 구현해야 함.
        내부에서 task_type에 맞는 메트릭 계산 및 저장.
        """
        pass

# timm_adapter.py (Classification)
class TimmAdapter(TrainingAdapter):
    def validate(self, epoch: int) -> MetricsResult:
        # 1. Collect predictions
        predictions, labels, image_results = self._collect_predictions()

        # 2. Compute metrics (unified)
        unified_metrics = ValidationMetricsCalculator.compute_metrics(
            task_type=TaskType.CLASSIFICATION,
            predictions=predictions,
            labels=labels,
            class_names=self.class_names
        )

        # 3. Save to DB (unified)
        self._save_validation_result(epoch, unified_metrics, image_results)

        return MetricsResult(...)

# ultralytics_adapter.py (Detection, Segmentation, Pose)
class UltralyticsAdapter(TrainingAdapter):
    def validate(self, epoch: int) -> MetricsResult:
        # YOLO는 내부적으로 validation 수행
        # Results에서 추출

        # Task type에 따라 자동으로 올바른 메트릭 계산
        task_type = self._get_task_type()  # DETECTION, INSTANCE_SEGMENTATION, POSE

        unified_metrics = ValidationMetricsCalculator.compute_metrics(
            task_type=task_type,
            predictions=yolo_results.predictions,
            labels=yolo_results.labels,
            **yolo_results.extra_info
        )

        self._save_validation_result(epoch, unified_metrics, image_results)

        return MetricsResult(...)

# 공통 save 메서드 (base.py로 이동)
class TrainingAdapter(ABC):
    def _save_validation_result(
        self,
        epoch: int,
        metrics: ValidationMetrics,
        image_results: List[Dict]
    ):
        """
        TASK-AGNOSTIC DB 저장

        모든 Adapter가 이 메서드 사용
        """
        # ValidationResult 저장
        validation_result = models.ValidationResult(
            job_id=self.job_id,
            epoch=epoch,
            task_type=metrics.task_type.value,
            primary_metric_value=metrics.primary_metric,
            overall_loss=metrics.loss,
            metrics=metrics.to_dict(),  # JSON으로 자동 변환
            per_class_metrics=metrics.get_per_class_metrics(),  # JSON
            confusion_matrix=metrics.get_confusion_matrix(),  # JSON (if applicable)
            class_names=metrics.get_class_names()  # JSON
        )

        # ValidationImageResult 저장
        for img_result in image_results[:20]:  # Sample
            validation_image_result = models.ValidationImageResult(
                validation_result_id=validation_result.id,
                job_id=self.job_id,
                epoch=epoch,
                # ... task에 따라 다른 필드 채우기
            )
```

### 4. Frontend: Task-Agnostic + Task-Specific Components

**구조**:

```
ValidationDashboard (Task-Agnostic Container)
├─ EpochSelector (공통)
├─ OverallMetricsCard (공통)
├─ TaskSpecificVisualization (Task에 따라 자동 선택)
│  ├─ ClassificationMetricsView
│  ├─ DetectionMetricsView
│  ├─ SegmentationMetricsView
│  └─ PoseMetricsView
└─ ValidationImageGallery (공통, Detail은 task-specific)
```

**핵심 컴포넌트**:

```typescript
// ValidationDashboard.tsx (TASK-AGNOSTIC CONTAINER)
export default function ValidationDashboard({ jobId, taskType, currentEpoch }) {
  const [selectedEpoch, setSelectedEpoch] = useState(currentEpoch);

  const { data: validationResult } = useSWR(
    `/validation/jobs/${jobId}/validation-results/${selectedEpoch}`,
    fetcher
  );

  return (
    <div className="space-y-6">
      {/* 공통 컴포넌트 */}
      <EpochSelector jobId={jobId} selectedEpoch={selectedEpoch} onSelect={setSelectedEpoch} />
      <OverallMetricsCard metrics={validationResult?.metrics} taskType={taskType} />

      {/* Task-Specific 자동 선택 */}
      <TaskSpecificVisualization
        taskType={taskType}
        validationResult={validationResult}
      />

      {/* 공통 이미지 갤러리 */}
      <ValidationImageGallery
        jobId={jobId}
        epoch={selectedEpoch}
        taskType={taskType}
      />
    </div>
  );
}

// TaskSpecificVisualization.tsx (자동 라우팅)
function TaskSpecificVisualization({ taskType, validationResult }) {
  if (!validationResult) return <Loading />;

  switch (taskType) {
    case "image_classification":
      return <ClassificationMetricsView data={validationResult} />;

    case "object_detection":
      return <DetectionMetricsView data={validationResult} />;

    case "instance_segmentation":
    case "semantic_segmentation":
      return <SegmentationMetricsView data={validationResult} />;

    case "pose_estimation":
      return <PoseMetricsView data={validationResult} />;

    default:
      return <GenericMetricsView data={validationResult} />;
  }
}
```

---

## Current State Analysis

### ✅ 구현된 기능

(이전과 동일 - 생략)

### ❌ 구현되지 않은 기능

1. **Task-Agnostic 메트릭 시스템**: TaskType enum, ValidationMetrics 통합 클래스
2. **Task별 메트릭 계산**: Detection, Segmentation, Pose metrics
3. **DB task_type 필드**: ValidationResult에 task 구분 없음
4. **Frontend task 자동 선택**: Task별 다른 UI 자동 렌더링 없음

---

## Implementation Plan

### Week 1: Backend - Task-Agnostic Infrastructure (Days 1-5)

#### Day 1-2: Database Models (Task-Agnostic)

**목표**: 모든 태스크를 지원하는 범용 DB 모델

##### 1.1 Migration 스크립트

**파일**: `mvp/backend/migrate_add_validation_tables.py`

```python
"""Add task-agnostic validation tables."""

def migrate():
    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        # ValidationResult (Task-Agnostic)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS validation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            epoch INTEGER NOT NULL,

            -- Task 식별
            task_type TEXT NOT NULL,  -- "image_classification", "object_detection", etc.

            -- 공통 메트릭
            primary_metric_value REAL,
            primary_metric_name TEXT,
            overall_loss REAL,

            -- Task-specific 메트릭 (JSON)
            metrics TEXT,  -- All metrics as JSON
            per_class_metrics TEXT,  -- Per-class metrics (all tasks)

            -- Visualization data (task-specific)
            confusion_matrix TEXT,  -- Classification
            pr_curves TEXT,  -- Detection, Segmentation
            class_names TEXT,  -- All tasks
            visualization_data TEXT,  -- Extra viz data

            -- Samples
            sample_correct_images TEXT,
            sample_incorrect_images TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))

        # ValidationImageResult (Task-Agnostic)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS validation_image_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            validation_result_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            epoch INTEGER NOT NULL,

            -- Image info
            image_path TEXT NOT NULL,
            image_name TEXT NOT NULL,
            image_index INTEGER,

            -- Classification
            true_label TEXT,
            true_label_id INTEGER,
            predicted_label TEXT,
            predicted_label_id INTEGER,
            confidence REAL,
            top5_predictions TEXT,  -- JSON

            -- Detection
            true_boxes TEXT,  -- JSON: [{"x", "y", "w", "h", "class"}]
            predicted_boxes TEXT,  -- JSON

            -- Segmentation
            true_mask_path TEXT,
            predicted_mask_path TEXT,

            -- Pose Estimation
            true_keypoints TEXT,  -- JSON
            predicted_keypoints TEXT,  -- JSON

            -- Common metrics
            is_correct INTEGER NOT NULL DEFAULT 0,
            iou REAL,  -- Detection, Segmentation
            oks REAL,  -- Pose (Object Keypoint Similarity)

            -- Extra
            extra_data TEXT,  -- JSON

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (validation_result_id) REFERENCES validation_results(id) ON DELETE CASCADE,
            FOREIGN KEY (job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))

        # Indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_validation_results_job_epoch ON validation_results(job_id, epoch)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_validation_results_task_type ON validation_results(task_type)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_validation_image_results_val_id ON validation_image_results(validation_result_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_validation_image_results_correct ON validation_image_results(is_correct)"))

        conn.commit()
        print("✓ Task-agnostic validation tables created!")
```

**실행**:
```bash
cd mvp/backend
python migrate_add_validation_tables.py
```

##### 1.2 Models 업데이트

**파일**: `mvp/backend/app/db/models.py`

```python
class ValidationResult(Base):
    """Task-agnostic validation result."""
    __tablename__ = "validation_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False, index=True)

    # Task 식별
    task_type = Column(String(50), nullable=False, index=True)

    # 공통 메트릭
    primary_metric_value = Column(Float, nullable=True)
    primary_metric_name = Column(String(50), nullable=True)
    overall_loss = Column(Float, nullable=True)

    # Task-specific 메트릭 (JSON)
    metrics = Column(JSON, nullable=True)
    per_class_metrics = Column(JSON, nullable=True)

    # Visualization
    confusion_matrix = Column(JSON, nullable=True)
    pr_curves = Column(JSON, nullable=True)
    class_names = Column(JSON, nullable=True)
    visualization_data = Column(JSON, nullable=True)

    # Samples
    sample_correct_images = Column(JSON, nullable=True)
    sample_incorrect_images = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    job = relationship("TrainingJob", back_populates="validation_results")
    image_results = relationship("ValidationImageResult", back_populates="validation_result", cascade="all, delete-orphan")

class ValidationImageResult(Base):
    """Task-agnostic image-level validation result."""
    __tablename__ = "validation_image_results"

    id = Column(Integer, primary_key=True, index=True)
    validation_result_id = Column(Integer, ForeignKey("validation_results.id"), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False, index=True)

    # Image info
    image_path = Column(String(500), nullable=False)
    image_name = Column(String(200), nullable=False, index=True)
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

    # Common metrics
    is_correct = Column(Boolean, nullable=False, default=False, index=True)
    iou = Column(Float, nullable=True)
    oks = Column(Float, nullable=True)  # Object Keypoint Similarity (Pose)

    # Extra
    extra_data = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    validation_result = relationship("ValidationResult", back_populates="image_results")
    job = relationship("TrainingJob")

# Update TrainingJob
class TrainingJob(Base):
    # ... existing fields ...
    validation_results = relationship("ValidationResult", back_populates="job", cascade="all, delete-orphan")
```

**Deliverables**:
- [ ] Task-agnostic DB schema 설계
- [ ] Migration 스크립트 작성 및 실행
- [ ] Models 정의 및 relationships 설정
- [ ] Indexes 생성 (task_type, is_correct 등)

---

#### Day 3-4: Metrics Calculator (Task-Agnostic)

**목표**: 모든 태스크를 지원하는 통합 메트릭 계산 시스템

**파일**: `mvp/training/validators/__init__.py`
**파일**: `mvp/training/validators/metrics.py`

```python
"""Task-agnostic validation metrics calculator."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Union
import numpy as np


class TaskType(Enum):
    """지원하는 모든 컴퓨터 비전 태스크"""
    CLASSIFICATION = "image_classification"
    DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POSE = "pose_estimation"


@dataclass
class ClassificationMetrics:
    """Classification 메트릭"""
    accuracy: float
    top5_accuracy: Optional[float] = None

    # Per-class
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[int, int] = field(default_factory=dict)

    # Overall
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "top5_accuracy": self.top5_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
        }


@dataclass
class DetectionMetrics:
    """Detection 메트릭"""
    mAP50: float
    mAP50_95: Optional[float] = None

    # Per-class
    per_class_ap: Dict[str, float] = field(default_factory=dict)

    # Overall
    precision: float = 0.0
    recall: float = 0.0
    mean_iou: float = 0.0

    class_names: List[str] = field(default_factory=list)

    # Curves
    pr_curves: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "mAP50": self.mAP50,
            "mAP50_95": self.mAP50_95,
            "precision": self.precision,
            "recall": self.recall,
            "mean_iou": self.mean_iou,
        }


@dataclass
class SegmentationMetrics:
    """Segmentation 메트릭"""
    mean_iou: float
    pixel_accuracy: float

    # Per-class
    per_class_iou: Dict[str, float] = field(default_factory=dict)
    per_class_pixel_acc: Dict[str, float] = field(default_factory=dict)

    # Dice coefficient
    dice: Optional[float] = None

    class_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "mean_iou": self.mean_iou,
            "pixel_accuracy": self.pixel_accuracy,
            "dice": self.dice,
        }


@dataclass
class PoseMetrics:
    """Pose Estimation 메트릭"""
    oks: float  # Object Keypoint Similarity
    pck: float  # Percentage of Correct Keypoints

    # Per-keypoint
    per_keypoint_pck: Dict[str, float] = field(default_factory=dict)
    per_keypoint_oks: Dict[str, float] = field(default_factory=dict)

    keypoint_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "oks": self.oks,
            "pck": self.pck,
        }


@dataclass
class ValidationMetrics:
    """
    통합 메트릭 컨테이너 (Task-Agnostic)

    모든 태스크의 메트릭을 담을 수 있는 범용 구조
    """
    task_type: TaskType

    # 공통 필드
    primary_metric: float
    primary_metric_name: str
    loss: float

    # Task-specific 데이터 (하나만 채워짐)
    classification_data: Optional[ClassificationMetrics] = None
    detection_data: Optional[DetectionMetrics] = None
    segmentation_data: Optional[SegmentationMetrics] = None
    pose_data: Optional[PoseMetrics] = None

    def to_dict(self) -> Dict:
        """DB 저장용 dict"""
        result = {
            "task_type": self.task_type.value,
            "primary_metric": self.primary_metric,
            "primary_metric_name": self.primary_metric_name,
            "loss": self.loss,
        }

        # Task-specific 데이터 추가
        if self.classification_data:
            result.update(self.classification_data.to_dict())
        elif self.detection_data:
            result.update(self.detection_data.to_dict())
        elif self.segmentation_data:
            result.update(self.segmentation_data.to_dict())
        elif self.pose_data:
            result.update(self.pose_data.to_dict())

        return result

    def get_per_class_metrics(self) -> Optional[Dict]:
        """Per-class 메트릭 추출 (DB 저장용)"""
        if self.classification_data:
            return {
                cls: {
                    "precision": self.classification_data.per_class_precision[cls],
                    "recall": self.classification_data.per_class_recall[cls],
                    "f1": self.classification_data.per_class_f1[cls],
                    "support": self.classification_data.per_class_support.get(idx, 0)
                }
                for idx, cls in enumerate(self.classification_data.class_names)
            }
        elif self.detection_data:
            return {
                cls: {"ap": ap}
                for cls, ap in self.detection_data.per_class_ap.items()
            }
        elif self.segmentation_data:
            return {
                cls: {"iou": iou, "pixel_accuracy": self.segmentation_data.per_class_pixel_acc.get(cls, 0)}
                for cls, iou in self.segmentation_data.per_class_iou.items()
            }
        elif self.pose_data:
            return {
                kp: {"pck": pck, "oks": self.pose_data.per_keypoint_oks.get(kp, 0)}
                for kp, pck in self.pose_data.per_keypoint_pck.items()
            }
        return None

    def get_confusion_matrix(self) -> Optional[List[List[int]]]:
        """Confusion matrix 추출 (Classification only)"""
        if self.classification_data and self.classification_data.confusion_matrix is not None:
            return self.classification_data.confusion_matrix.tolist()
        return None

    def get_class_names(self) -> Optional[List[str]]:
        """Class names 추출"""
        if self.classification_data:
            return self.classification_data.class_names
        elif self.detection_data:
            return self.detection_data.class_names
        elif self.segmentation_data:
            return self.segmentation_data.class_names
        elif self.pose_data:
            return self.pose_data.keypoint_names
        return None


class ValidationMetricsCalculator:
    """
    Task-Agnostic 메트릭 계산기

    모든 태스크에 대해 통일된 인터페이스 제공
    """

    @staticmethod
    def compute_metrics(
        task_type: TaskType,
        predictions: Any,
        labels: Any,
        **kwargs
    ) -> ValidationMetrics:
        """
        Task에 따라 자동으로 적절한 메트릭 계산

        Args:
            task_type: TaskType enum
            predictions: 모델 예측 (형식은 task마다 다름)
            labels: Ground truth (형식은 task마다 다름)
            **kwargs: Task-specific 추가 파라미터

        Returns:
            ValidationMetrics (통합 메트릭 객체)
        """
        if task_type == TaskType.CLASSIFICATION:
            return ValidationMetricsCalculator._compute_classification(predictions, labels, **kwargs)
        elif task_type == TaskType.DETECTION:
            return ValidationMetricsCalculator._compute_detection(predictions, labels, **kwargs)
        elif task_type in [TaskType.INSTANCE_SEGMENTATION, TaskType.SEMANTIC_SEGMENTATION]:
            return ValidationMetricsCalculator._compute_segmentation(predictions, labels, task_type, **kwargs)
        elif task_type == TaskType.POSE:
            return ValidationMetricsCalculator._compute_pose(predictions, labels, **kwargs)
        else:
            raise NotImplementedError(f"Task type {task_type} not yet implemented")

    @staticmethod
    def _compute_classification(
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        loss: float = 0.0,
        top_k: int = 5
    ) -> ValidationMetrics:
        """Classification 메트릭 계산"""
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix as sk_confusion_matrix
        )

        # Logits → class indices
        if len(predictions.shape) > 1:
            pred_classes = predictions.argmax(axis=1)
            # Top-5 accuracy
            top5_preds = np.argsort(predictions, axis=1)[:, -top_k:]
            top5_acc = np.mean([labels[i] in top5_preds[i] for i in range(len(labels))])
        else:
            pred_classes = predictions
            top5_acc = None

        # Overall accuracy
        accuracy = accuracy_score(labels, pred_classes)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, pred_classes, average=None, zero_division=0
        )

        per_class_precision = {class_names[i]: float(precision[i]) for i in range(len(class_names))}
        per_class_recall = {class_names[i]: float(recall[i]) for i in range(len(class_names))}
        per_class_f1 = {class_names[i]: float(f1[i]) for i in range(len(class_names))}
        per_class_support = {i: int(support[i]) for i in range(len(class_names))}

        # Macro metrics
        macro_precision = float(precision.mean())
        macro_recall = float(recall.mean())
        macro_f1 = float(f1.mean())
        weighted_f1 = float(np.average(f1, weights=support))

        # Confusion matrix
        cm = sk_confusion_matrix(labels, pred_classes)

        cls_metrics = ClassificationMetrics(
            accuracy=accuracy,
            top5_accuracy=top5_acc,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            per_class_support=per_class_support,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            confusion_matrix=cm,
            class_names=class_names
        )

        return ValidationMetrics(
            task_type=TaskType.CLASSIFICATION,
            primary_metric=accuracy,
            primary_metric_name="accuracy",
            loss=loss,
            classification_data=cls_metrics
        )

    @staticmethod
    def _compute_detection(
        predictions: List[Dict],
        labels: List[Dict],
        class_names: List[str],
        loss: float = 0.0,
        iou_threshold: float = 0.5
    ) -> ValidationMetrics:
        """
        Detection 메트릭 계산

        Args:
            predictions: [{"boxes": [...], "scores": [...], "labels": [...]}, ...]
            labels: [{"boxes": [...], "labels": [...]}, ...]
            class_names: List of class names
        """
        # Simplified implementation
        # In production, use pycocotools or similar

        # Placeholder
        det_metrics = DetectionMetrics(
            mAP50=0.75,  # Placeholder
            mAP50_95=0.65,
            per_class_ap={cls: 0.7 for cls in class_names},
            precision=0.82,
            recall=0.79,
            mean_iou=0.68,
            class_names=class_names
        )

        return ValidationMetrics(
            task_type=TaskType.DETECTION,
            primary_metric=det_metrics.mAP50,
            primary_metric_name="mAP50",
            loss=loss,
            detection_data=det_metrics
        )

    @staticmethod
    def _compute_segmentation(
        predictions: Any,
        labels: Any,
        task_type: TaskType,
        class_names: List[str],
        loss: float = 0.0
    ) -> ValidationMetrics:
        """Segmentation 메트릭 계산"""
        # Placeholder
        seg_metrics = SegmentationMetrics(
            mean_iou=0.78,
            pixel_accuracy=0.92,
            per_class_iou={cls: 0.75 for cls in class_names},
            per_class_pixel_acc={cls: 0.90 for cls in class_names},
            dice=0.85,
            class_names=class_names
        )

        return ValidationMetrics(
            task_type=task_type,
            primary_metric=seg_metrics.mean_iou,
            primary_metric_name="mean_IoU",
            loss=loss,
            segmentation_data=seg_metrics
        )

    @staticmethod
    def _compute_pose(
        predictions: Any,
        labels: Any,
        keypoint_names: List[str],
        loss: float = 0.0
    ) -> ValidationMetrics:
        """Pose 메트릭 계산"""
        # Placeholder
        pose_metrics = PoseMetrics(
            oks=0.82,
            pck=0.88,
            per_keypoint_pck={kp: 0.85 for kp in keypoint_names},
            per_keypoint_oks={kp: 0.80 for kp in keypoint_names},
            keypoint_names=keypoint_names
        )

        return ValidationMetrics(
            task_type=TaskType.POSE,
            primary_metric=pose_metrics.oks,
            primary_metric_name="OKS",
            loss=loss,
            pose_data=pose_metrics
        )
```

**Deliverables**:
- [ ] TaskType enum 정의
- [ ] Task별 메트릭 dataclass (Classification, Detection, Segmentation, Pose)
- [ ] ValidationMetrics 통합 컨테이너
- [ ] ValidationMetricsCalculator.compute_metrics() 구현
- [ ] Classification 메트릭 계산 완전 구현
- [ ] Detection, Segmentation, Pose 기본 구현 (Placeholder)

---

#### Day 4-5: Adapter Integration

**목표**: 모든 Adapter에서 task-agnostic validate() 구현

##### TimmAdapter (Classification)

**파일**: `mvp/training/adapters/timm_adapter.py`

```python
def validate(self, epoch: int) -> MetricsResult:
    """Task-agnostic validation (Classification)"""
    from training.validators.metrics import ValidationMetricsCalculator, TaskType

    self.model.eval()
    running_loss = 0.0

    # Collect predictions and labels
    all_predictions = []
    all_labels = []
    all_logits = []
    image_results = []

    with torch.no_grad():
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

            # Sample image results (first 100)
            if len(image_results) < 100:
                for i in range(inputs.size(0)):
                    if len(image_results) >= 100:
                        break

                    pred_class = predicted[i].item()
                    true_class = targets[i].item()
                    confidence = torch.softmax(outputs[i], dim=0).max().item()

                    top5_probs, top5_indices = torch.topk(torch.softmax(outputs[i], dim=0), k=5)
                    top5_preds = [
                        {"class_id": int(top5_indices[j]), "confidence": float(top5_probs[j])}
                        for j in range(5)
                    ]

                    image_results.append({
                        "image_index": batch_idx * self.training_config.batch_size + i,
                        "true_label_id": true_class,
                        "predicted_label_id": pred_class,
                        "confidence": confidence,
                        "top5_predictions": top5_preds,
                        "is_correct": (pred_class == true_class)
                    })

    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.vstack(all_logits)

    # Get class names
    class_names = self.val_loader.dataset.classes if hasattr(self.val_loader.dataset, 'classes') else [f"class_{i}" for i in range(self.model_config.num_classes)]

    avg_loss = running_loss / len(self.val_loader)

    # Compute metrics (TASK-AGNOSTIC)
    unified_metrics = ValidationMetricsCalculator.compute_metrics(
        task_type=TaskType.CLASSIFICATION,
        predictions=all_logits,
        labels=all_labels,
        class_names=class_names,
        loss=avg_loss,
        top_k=5
    )

    # Save to DB (UNIFIED METHOD in base.py)
    self._save_validation_result(epoch, unified_metrics, image_results)

    # Update scheduler
    if self.scheduler:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        else:
            self.scheduler.step()

    # Return MetricsResult
    return MetricsResult(
        epoch=epoch,
        step=0,
        train_loss=avg_loss,
        metrics=unified_metrics.to_dict()
    )
```

##### UltralyticsAdapter (Detection, Segmentation, Pose)

**파일**: `mvp/training/adapters/ultralytics_adapter.py`

```python
def validate(self, epoch: int) -> MetricsResult:
    """
    Task-agnostic validation (Detection, Segmentation, Pose)

    YOLO handles validation internally, we extract and save results
    """
    from training.validators.metrics import ValidationMetricsCalculator, TaskType

    # YOLO의 validator.metrics에서 추출
    # (실제로는 on_val_end callback에서 처리)

    # Determine task type
    task_type_map = {
        "detect": TaskType.DETECTION,
        "segment": TaskType.INSTANCE_SEGMENTATION,
        "pose": TaskType.POSE,
        "classify": TaskType.CLASSIFICATION
    }

    task_type = task_type_map.get(self.task_type.value, TaskType.DETECTION)

    # Extract YOLO metrics
    # (Placeholder - 실제로는 YOLO results에서 추출)

    unified_metrics = ValidationMetricsCalculator.compute_metrics(
        task_type=task_type,
        predictions=[],  # YOLO results
        labels=[],  # YOLO labels
        class_names=self.class_names,
        loss=0.0  # YOLO loss
    )

    # Save to DB
    self._save_validation_result(epoch, unified_metrics, [])

    return MetricsResult(
        epoch=epoch,
        step=0,
        train_loss=0.0,
        metrics=unified_metrics.to_dict()
    )
```

##### Unified Save Method (base.py)

**파일**: `mvp/training/adapters/base.py`

```python
class TrainingAdapter(ABC):
    """Base adapter with task-agnostic validation support"""

    def _save_validation_result(
        self,
        epoch: int,
        metrics: 'ValidationMetrics',
        image_results: List[Dict]
    ):
        """
        TASK-AGNOSTIC DB 저장

        모든 Adapter가 이 메서드 사용
        """
        import sqlite3
        import json
        from pathlib import Path
        from datetime import datetime

        # DB 연결
        training_dir = Path(__file__).parent.parent
        mvp_dir = training_dir.parent
        db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

        if not db_path.exists():
            print(f"[WARNING] Database not found: {db_path}")
            return

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Insert ValidationResult
            cursor.execute("""
            INSERT INTO validation_results (
                job_id, epoch, task_type,
                primary_metric_value, primary_metric_name, overall_loss,
                metrics, per_class_metrics, confusion_matrix, class_names,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.job_id,
                epoch,
                metrics.task_type.value,
                metrics.primary_metric,
                metrics.primary_metric_name,
                metrics.loss,
                json.dumps(metrics.to_dict()),
                json.dumps(metrics.get_per_class_metrics()),
                json.dumps(metrics.get_confusion_matrix()),
                json.dumps(metrics.get_class_names()),
                datetime.utcnow().isoformat()
            ))

            validation_result_id = cursor.lastrowid

            # Insert ValidationImageResults (sample)
            for img_result in image_results[:20]:
                # Task에 따라 다른 필드 채우기
                if metrics.task_type == TaskType.CLASSIFICATION:
                    cursor.execute("""
                    INSERT INTO validation_image_results (
                        validation_result_id, job_id, epoch,
                        image_path, image_name, image_index,
                        true_label_id, predicted_label_id, confidence,
                        top5_predictions, is_correct, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        validation_result_id, self.job_id, epoch,
                        img_result.get("image_path", f"val/image_{img_result['image_index']}.jpg"),
                        img_result.get("image_name", f"image_{img_result['image_index']}.jpg"),
                        img_result["image_index"],
                        img_result["true_label_id"],
                        img_result["predicted_label_id"],
                        img_result["confidence"],
                        json.dumps(img_result["top5_predictions"]),
                        1 if img_result["is_correct"] else 0,
                        datetime.utcnow().isoformat()
                    ))
                # Add detection, segmentation, pose cases

            conn.commit()
            conn.close()

            print(f"[INFO] Saved validation result (task={metrics.task_type.value}, epoch={epoch}, ID={validation_result_id})")

        except Exception as e:
            print(f"[WARNING] Failed to save validation result: {e}")
            import traceback
            traceback.print_exc()
```

**Deliverables**:
- [ ] TimmAdapter validate() 구현 (task-agnostic)
- [ ] UltralyticsAdapter validate() 기본 구현
- [ ] base.py에 _save_validation_result() 통합 메서드
- [ ] Task별 이미지 결과 저장 로직

---

### Week 2: API & Frontend (Days 6-10)

#### Day 6-7: API Endpoints (Task-Agnostic)

**파일**: `mvp/backend/app/api/validation.py`

```python
"""Task-agnostic validation API."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import json

from app.db.database import get_db
from app.db import models
from app.schemas import validation

router = APIRouter()


@router.get("/jobs/{job_id}/validation-results")
async def get_validation_results(job_id: int, db: Session = Depends(get_db)):
    """Get all validation results (task-agnostic)"""
    results = (
        db.query(models.ValidationResult)
        .filter(models.ValidationResult.job_id == job_id)
        .order_by(models.ValidationResult.epoch)
        .all()
    )
    return results


@router.get("/jobs/{job_id}/validation-results/{epoch}")
async def get_validation_result_detail(job_id: int, epoch: int, db: Session = Depends(get_db)):
    """Get detailed validation result (task-agnostic)"""
    result = (
        db.query(models.ValidationResult)
        .filter(
            models.ValidationResult.job_id == job_id,
            models.ValidationResult.epoch == epoch
        )
        .first()
    )

    if not result:
        raise HTTPException(status_code=404, detail="Validation result not found")

    return result


@router.get("/jobs/{job_id}/validation-results/{epoch}/images")
async def get_validation_images(
    job_id: int,
    epoch: int,
    correct_only: Optional[bool] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get image-level results (task-agnostic)"""
    val_result = (
        db.query(models.ValidationResult)
        .filter(
            models.ValidationResult.job_id == job_id,
            models.ValidationResult.epoch == epoch
        )
        .first()
    )

    if not val_result:
        raise HTTPException(status_code=404, detail="Validation result not found")

    query = db.query(models.ValidationImageResult).filter(
        models.ValidationImageResult.validation_result_id == val_result.id
    )

    if correct_only is not None:
        query = query.filter(models.ValidationImageResult.is_correct == correct_only)

    return query.limit(limit).all()


@router.get("/jobs/{job_id}/task-specific-data/{epoch}")
async def get_task_specific_data(job_id: int, epoch: int, db: Session = Depends(get_db)):
    """
    Get task-specific visualization data

    Returns different data based on task_type:
    - Classification: confusion_matrix
    - Detection: pr_curves
    - Segmentation: per_class_iou
    """
    result = (
        db.query(models.ValidationResult)
        .filter(
            models.ValidationResult.job_id == job_id,
            models.ValidationResult.epoch == epoch
        )
        .first()
    )

    if not result:
        raise HTTPException(status_code=404, detail="Validation result not found")

    response = {
        "task_type": result.task_type,
        "class_names": json.loads(result.class_names) if result.class_names else []
    }

    # Task-specific data
    if result.task_type == "image_classification" and result.confusion_matrix:
        response["confusion_matrix"] = json.loads(result.confusion_matrix)

    elif result.task_type in ["object_detection", "instance_segmentation"] and result.pr_curves:
        response["pr_curves"] = json.loads(result.pr_curves)

    return response
```

**Schemas**:

**파일**: `mvp/backend/app/schemas/validation.py`

```python
"""Task-agnostic validation schemas."""

from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime


class ValidationResultSummary(BaseModel):
    """Summary (task-agnostic)"""
    id: int
    job_id: int
    epoch: int
    task_type: str
    primary_metric_value: Optional[float]
    primary_metric_name: Optional[str]
    overall_loss: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class ValidationResultDetail(BaseModel):
    """Detailed result (task-agnostic)"""
    id: int
    job_id: int
    epoch: int
    task_type: str

    primary_metric_value: Optional[float]
    primary_metric_name: Optional[str]
    overall_loss: Optional[float]

    metrics: Optional[Dict[str, Any]]
    per_class_metrics: Optional[Dict[str, Any]]
    class_names: Optional[List[str]]

    # Task-specific (nullable)
    confusion_matrix: Optional[List[List[int]]]
    pr_curves: Optional[Dict]

    created_at: datetime

    class Config:
        from_attributes = True


class ValidationImageResultResponse(BaseModel):
    """Image result (task-agnostic)"""
    id: int
    job_id: int
    epoch: int

    image_path: str
    image_name: str
    is_correct: bool

    # Classification (nullable)
    true_label: Optional[str]
    predicted_label: Optional[str]
    confidence: Optional[float]
    top5_predictions: Optional[List[Dict]]

    # Detection (nullable)
    true_boxes: Optional[List[Dict]]
    predicted_boxes: Optional[List[Dict]]

    # Segmentation (nullable)
    true_mask_path: Optional[str]
    predicted_mask_path: Optional[str]

    # Pose (nullable)
    true_keypoints: Optional[List[Dict]]
    predicted_keypoints: Optional[List[Dict]]

    # Common
    iou: Optional[float]
    oks: Optional[float]

    class Config:
        from_attributes = True
```

**Deliverables**:
- [ ] Task-agnostic API 엔드포인트
- [ ] Task-specific data 조회 엔드포인트
- [ ] Schemas 정의
- [ ] Router 등록

---

#### Day 8-9: Frontend (Task-Agnostic + Task-Specific)

**구조**:

```
components/validation/
├── ValidationDashboard.tsx          # 🎯 Task-agnostic container
├── common/
│   ├── EpochSelector.tsx
│   ├── OverallMetricsCard.tsx
│   └── ValidationImageGallery.tsx   # 🎯 Task-agnostic gallery
├── task-specific/
│   ├── TaskSpecificVisualization.tsx  # 🎯 Auto router
│   ├── ClassificationMetricsView.tsx
│   ├── DetectionMetricsView.tsx
│   ├── SegmentationMetricsView.tsx
│   └── PoseMetricsView.tsx
└── detail/
    └── ImageDetailModal.tsx         # 🎯 Task-specific detail
```

##### ValidationDashboard (Task-Agnostic Container)

**파일**: `mvp/frontend/components/validation/ValidationDashboard.tsx`

```typescript
"use client";

import React, { useState, useEffect } from "react";
import useSWR from "swr";

import EpochSelector from "./common/EpochSelector";
import OverallMetricsCard from "./common/OverallMetricsCard";
import TaskSpecificVisualization from "./task-specific/TaskSpecificVisualization";
import ValidationImageGallery from "./common/ValidationImageGallery";

const fetcher = (url: string) =>
  fetch(`http://localhost:8000/api/v1${url}`).then((res) => res.json());

interface ValidationDashboardProps {
  jobId: number;
  taskType: string;  // "image_classification", "object_detection", etc.
  currentEpoch?: number;
}

export default function ValidationDashboard({
  jobId,
  taskType,
  currentEpoch,
}: ValidationDashboardProps) {
  const [selectedEpoch, setSelectedEpoch] = useState<number>(currentEpoch || 1);

  // Fetch all validation results
  const { data: validationResults } = useSWR(
    `/validation/jobs/${jobId}/validation-results`,
    fetcher,
    { refreshInterval: 5000 }
  );

  // Fetch detailed result for selected epoch
  const { data: validationResult } = useSWR(
    selectedEpoch
      ? `/validation/jobs/${jobId}/validation-results/${selectedEpoch}`
      : null,
    fetcher
  );

  // Update selected epoch when training progresses
  useEffect(() => {
    if (currentEpoch && currentEpoch !== selectedEpoch) {
      setSelectedEpoch(currentEpoch);
    }
  }, [currentEpoch]);

  if (!validationResults || validationResults.length === 0) {
    return (
      <div className="p-6 bg-amber-50 rounded-lg border border-amber-200">
        <p className="text-sm text-amber-800">
          아직 validation 결과가 없습니다.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Epoch Selector (공통) */}
      <EpochSelector
        validationResults={validationResults}
        selectedEpoch={selectedEpoch}
        onSelect={setSelectedEpoch}
      />

      {/* Overall Metrics (공통) */}
      {validationResult && (
        <OverallMetricsCard
          metrics={validationResult.metrics}
          primaryMetricName={validationResult.primary_metric_name}
          primaryMetricValue={validationResult.primary_metric_value}
          loss={validationResult.overall_loss}
          taskType={taskType}
        />
      )}

      {/* Task-Specific Visualization (자동 선택) */}
      {validationResult && (
        <TaskSpecificVisualization
          taskType={taskType}
          validationResult={validationResult}
        />
      )}

      {/* Image Gallery (공통, Detail은 task-specific) */}
      <ValidationImageGallery
        jobId={jobId}
        epoch={selectedEpoch}
        taskType={taskType}
      />
    </div>
  );
}
```

##### TaskSpecificVisualization (Auto Router)

**파일**: `mvp/frontend/components/validation/task-specific/TaskSpecificVisualization.tsx`

```typescript
"use client";

import React from "react";

import ClassificationMetricsView from "./ClassificationMetricsView";
import DetectionMetricsView from "./DetectionMetricsView";
import SegmentationMetricsView from "./SegmentationMetricsView";
import PoseMetricsView from "./PoseMetricsView";

interface TaskSpecificVisualizationProps {
  taskType: string;
  validationResult: any;
}

export default function TaskSpecificVisualization({
  taskType,
  validationResult,
}: TaskSpecificVisualizationProps) {
  if (!validationResult) {
    return <div className="text-gray-500">No data</div>;
  }

  // Task에 따라 적절한 컴포넌트 자동 선택
  switch (taskType) {
    case "image_classification":
      return (
        <ClassificationMetricsView
          confusionMatrix={validationResult.confusion_matrix}
          perClassMetrics={validationResult.per_class_metrics}
          classNames={validationResult.class_names}
          metrics={validationResult.metrics}
        />
      );

    case "object_detection":
      return (
        <DetectionMetricsView
          perClassAP={validationResult.per_class_metrics}
          classNames={validationResult.class_names}
          metrics={validationResult.metrics}
          prCurves={validationResult.pr_curves}
        />
      );

    case "instance_segmentation":
    case "semantic_segmentation":
      return (
        <SegmentationMetricsView
          perClassIoU={validationResult.per_class_metrics}
          classNames={validationResult.class_names}
          metrics={validationResult.metrics}
        />
      );

    case "pose_estimation":
      return (
        <PoseMetricsView
          perKeypointMetrics={validationResult.per_class_metrics}
          keypointNames={validationResult.class_names}
          metrics={validationResult.metrics}
        />
      );

    default:
      // Fallback: Generic metrics view
      return (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h4 className="text-sm font-semibold mb-4">Metrics</h4>
          <pre className="text-xs">{JSON.stringify(validationResult.metrics, null, 2)}</pre>
        </div>
      );
  }
}
```

##### ClassificationMetricsView

**파일**: `mvp/frontend/components/validation/task-specific/ClassificationMetricsView.tsx`

```typescript
"use client";

import React from "react";
import ConfusionMatrixView from "./components/ConfusionMatrixView";
import PerClassMetricsTable from "./components/PerClassMetricsTable";

interface ClassificationMetricsViewProps {
  confusionMatrix: number[][];
  perClassMetrics: any;
  classNames: string[];
  metrics: any;
}

export default function ClassificationMetricsView({
  confusionMatrix,
  perClassMetrics,
  classNames,
  metrics,
}: ClassificationMetricsViewProps) {
  return (
    <div className="space-y-6">
      {/* Confusion Matrix */}
      {confusionMatrix && classNames && (
        <ConfusionMatrixView
          confusionMatrix={confusionMatrix}
          classNames={classNames}
        />
      )}

      {/* Per-Class Metrics */}
      {perClassMetrics && (
        <PerClassMetricsTable perClassMetrics={perClassMetrics} />
      )}

      {/* Additional Metrics */}
      {metrics?.top5_accuracy && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h4 className="text-sm font-semibold mb-2">Top-5 Accuracy</h4>
          <p className="text-2xl font-bold text-gray-900">
            {(metrics.top5_accuracy * 100).toFixed(2)}%
          </p>
        </div>
      )}
    </div>
  );
}
```

##### DetectionMetricsView

**파일**: `mvp/frontend/components/validation/task-specific/DetectionMetricsView.tsx`

```typescript
"use client";

import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface DetectionMetricsViewProps {
  perClassAP: any;
  classNames: string[];
  metrics: any;
  prCurves?: any;
}

export default function DetectionMetricsView({
  perClassAP,
  classNames,
  metrics,
}: DetectionMetricsViewProps) {
  // Prepare bar chart data
  const chartData = Object.entries(perClassAP || {}).map(([cls, data]: [string, any]) => ({
    class: cls,
    AP: data.ap * 100,
  }));

  return (
    <div className="space-y-6">
      {/* Overall Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="mAP50" value={metrics?.mAP50} isPercentage />
        <MetricCard label="mAP50-95" value={metrics?.mAP50_95} isPercentage />
        <MetricCard label="Precision" value={metrics?.precision} isPercentage />
        <MetricCard label="Recall" value={metrics?.recall} isPercentage />
      </div>

      {/* Per-Class AP Bar Chart */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h4 className="text-sm font-semibold mb-4">AP per Class</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="class" />
            <YAxis domain={[0, 100]} />
            <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
            <Bar dataKey="AP" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function MetricCard({ label, value, isPercentage = false }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-2xl font-bold text-gray-900">
        {value !== undefined
          ? isPercentage
            ? `${(value * 100).toFixed(2)}%`
            : value.toFixed(4)
          : "N/A"}
      </div>
    </div>
  );
}
```

##### ValidationImageGallery (Task-Agnostic)

**파일**: `mvp/frontend/components/validation/common/ValidationImageGallery.tsx`

```typescript
"use client";

import React, { useState } from "react";
import useSWR from "swr";
import ImageDetailModal from "../detail/ImageDetailModal";

const fetcher = (url: string) =>
  fetch(`http://localhost:8000/api/v1${url}`).then((res) => res.json());

interface ValidationImageGalleryProps {
  jobId: number;
  epoch: number;
  taskType: string;
}

export default function ValidationImageGallery({
  jobId,
  epoch,
  taskType,
}: ValidationImageGalleryProps) {
  const [filter, setFilter] = useState<"all" | "correct" | "incorrect">("all");
  const [selectedImage, setSelectedImage] = useState<any>(null);

  const { data: imageResults } = useSWR(
    `/validation/jobs/${jobId}/validation-results/${epoch}/images?correct_only=${
      filter === "correct" ? "true" : filter === "incorrect" ? "false" : ""
    }`,
    fetcher
  );

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold">Sample Predictions</h3>

        {/* Filter */}
        <div className="flex gap-2">
          {["all", "correct", "incorrect"].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f as any)}
              className={`px-3 py-1 text-xs rounded ${
                filter === f
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Image Grid */}
      <div className="grid grid-cols-4 gap-4">
        {imageResults?.map((result: any) => (
          <ImageCard
            key={result.id}
            result={result}
            taskType={taskType}
            onClick={() => setSelectedImage(result)}
          />
        ))}
      </div>

      {/* Detail Modal */}
      {selectedImage && (
        <ImageDetailModal
          result={selectedImage}
          taskType={taskType}
          onClose={() => setSelectedImage(null)}
        />
      )}
    </div>
  );
}

function ImageCard({ result, taskType, onClick }) {
  return (
    <div
      onClick={onClick}
      className="border rounded-lg overflow-hidden cursor-pointer hover:shadow-lg transition"
    >
      {/* Image */}
      <div className="relative">
        <img
          src={result.image_path}
          alt={result.image_name}
          className="w-full h-40 object-cover"
        />
        <div
          className={`absolute top-2 right-2 px-2 py-1 rounded text-xs font-semibold ${
            result.is_correct ? "bg-green-500 text-white" : "bg-red-500 text-white"
          }`}
        >
          {result.is_correct ? "✓" : "✗"}
        </div>
      </div>

      {/* Info (Task-specific) */}
      <div className="p-2 text-xs">
        {taskType === "image_classification" && (
          <>
            <div className="font-medium truncate">True: {result.true_label}</div>
            <div className={`truncate ${result.is_correct ? "text-green-600" : "text-red-600"}`}>
              Pred: {result.predicted_label} ({(result.confidence * 100).toFixed(1)}%)
            </div>
          </>
        )}

        {taskType === "object_detection" && (
          <>
            <div>Boxes: {result.predicted_boxes?.length || 0}</div>
            <div>IoU: {result.iou ? (result.iou * 100).toFixed(1) + "%" : "N/A"}</div>
          </>
        )}

        {/* Add segmentation, pose */}
      </div>
    </div>
  );
}
```

**Deliverables**:
- [ ] ValidationDashboard (task-agnostic container)
- [ ] TaskSpecificVisualization (auto router)
- [ ] ClassificationMetricsView (상세 구현)
- [ ] DetectionMetricsView (기본 구현)
- [ ] SegmentationMetricsView (기본 구현)
- [ ] PoseMetricsView (기본 구현)
- [ ] ValidationImageGallery (task-agnostic)
- [ ] ImageDetailModal (task-specific)
- [ ] TrainingPanel 통합

---

#### Day 10: Testing & Documentation

**E2E 테스트 시나리오**:

1. **Classification (timm)**:
   - ResNet 학습 실행
   - Validation 탭 확인
   - Confusion matrix 렌더링 확인
   - Per-class metrics 정확성 검증

2. **Detection (YOLO)**:
   - YOLOv8 detection 학습
   - mAP charts 확인
   - Task-specific visualization 자동 선택 확인

3. **Task 전환**:
   - Classification job → Detection job 이동
   - UI가 자동으로 변경되는지 확인

**문서화**:

**파일**: `docs/features/VALIDATION_SYSTEM.md`

```markdown
# Validation System User Guide

## Task-Agnostic Design

The validation system supports all computer vision tasks:
- Image Classification
- Object Detection
- Instance/Semantic Segmentation
- Pose Estimation

All tasks use the same **Validation** tab with task-specific visualizations.

## UI Components

### Overall Metrics
Always visible regardless of task.

### Task-Specific Visualizations

**Classification:**
- Confusion Matrix
- Per-Class Precision/Recall/F1
- Top-5 Accuracy

**Detection:**
- mAP per Class (Bar Chart)
- Precision-Recall Curves
- IoU Distribution

**Segmentation:**
- IoU per Class
- Pixel Accuracy
- Dice Coefficient

**Pose:**
- OKS per Keypoint
- PCK Curves

### Image Gallery
All tasks show sample predictions with task-specific overlays.

## API Reference

```bash
# Get all validation results (任何任务)
GET /api/v1/validation/jobs/{job_id}/validation-results

# Get task-specific data
GET /api/v1/validation/jobs/{job_id}/task-specific-data/{epoch}
```
```

**Deliverables**:
- [ ] E2E 테스트 (Classification + Detection)
- [ ] Task 전환 테스트
- [ ] 사용자 가이드
- [ ] API 문서

---

## Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Week 1: Backend** | Days 1-5 | Task-Agnostic Infrastructure |
| Day 1-2 | 2 days | DB models (task_type, flexible JSON fields) |
| Day 3-4 | 2 days | Metrics Calculator (TaskType enum, unified interface) |
| Day 4-5 | 2 days | Adapter integration (unified validate(), save) |
| **Week 2: Frontend** | Days 6-10 | API & UI |
| Day 6-7 | 2 days | API (task-agnostic endpoints) |
| Day 8-9 | 2 days | Frontend (auto routing, task-specific views) |
| Day 10 | 1 day | Testing & docs |
| **Total** | **10 days** | **Complete Task-Agnostic Validation System** |

---

## Success Criteria

### Technical

- [ ] DB supports all task types (Classification, Detection, Segmentation, Pose)
- [ ] Metrics Calculator handles all tasks with unified interface
- [ ] API responses are task-agnostic
- [ ] Frontend auto-selects correct visualization based on task_type

### Functional

- [ ] Classification: Confusion matrix + per-class metrics
- [ ] Detection: mAP charts + PR curves (기본)
- [ ] Task switching works seamlessly
- [ ] Image gallery supports all tasks

### User Experience

- [ ] Users don't need to know task type (automatic)
- [ ] Same Validation tab for all tasks
- [ ] Task-specific viz appears without configuration

---

## Next Steps

1. **Branch Creation**: `git checkout -b feat/validation-system-task-agnostic`
2. **Day 1 Start**: DB migration with task_type field
3. **Implement Classification First**: Complete Classification metrics
4. **Add Other Tasks Incrementally**: Detection, Segmentation, Pose

---

*End of Document*
