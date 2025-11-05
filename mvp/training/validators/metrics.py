"""
Task-agnostic validation metrics calculator.

This module provides a unified interface for computing validation metrics
across all computer vision tasks (classification, detection, segmentation, pose).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix as sklearn_confusion_matrix,
    top_k_accuracy_score,
)
import sys
from pathlib import Path

# Add parent directory to path to import TaskType from platform_sdk
sys.path.insert(0, str(Path(__file__).parent.parent))
from platform_sdk import TaskType


@dataclass
class ClassificationMetrics:
    """Metrics for image classification tasks."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    top5_accuracy: Optional[float] = None

    # Per-class metrics
    per_class_precision: Optional[List[float]] = None
    per_class_recall: Optional[List[float]] = None
    per_class_f1: Optional[List[float]] = None
    per_class_support: Optional[List[int]] = None

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "top5_accuracy": float(self.top5_accuracy) if self.top5_accuracy else None,
        }

    def per_class_to_dict(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Convert per-class metrics to dictionary."""
        if not self.per_class_precision or not self.class_names:
            return None

        result = {}
        for i, class_name in enumerate(self.class_names):
            result[class_name] = {
                "precision": float(self.per_class_precision[i]),
                "recall": float(self.per_class_recall[i]),
                "f1_score": float(self.per_class_f1[i]),
                "support": int(self.per_class_support[i]) if self.per_class_support else 0,
            }
        return result

    def confusion_matrix_to_list(self) -> Optional[List[List[int]]]:
        """Convert confusion matrix to nested list for JSON."""
        if self.confusion_matrix is None:
            return None
        return self.confusion_matrix.tolist()


@dataclass
class DetectionMetrics:
    """Metrics for object detection tasks."""

    map_50: float  # mAP at IoU=0.5
    map_50_95: float  # mAP at IoU=0.5:0.95
    precision: float
    recall: float

    # Per-class AP
    per_class_ap: Optional[Dict[str, float]] = None

    # PR curves for visualization
    pr_curves: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "mAP@0.5": float(self.map_50),
            "mAP@0.5:0.95": float(self.map_50_95),
            "precision": float(self.precision),
            "recall": float(self.recall),
        }


@dataclass
class SegmentationMetrics:
    """Metrics for segmentation tasks (instance or semantic)."""

    mean_iou: float  # Mean Intersection over Union
    pixel_accuracy: float
    mean_precision: float
    mean_recall: float

    # Per-class IoU
    per_class_iou: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "mean_iou": float(self.mean_iou),
            "pixel_accuracy": float(self.pixel_accuracy),
            "mean_precision": float(self.mean_precision),
            "mean_recall": float(self.mean_recall),
        }


@dataclass
class PoseMetrics:
    """Metrics for pose estimation tasks."""

    oks: float  # Object Keypoint Similarity
    pck: float  # Percentage of Correct Keypoints
    mean_precision: float
    mean_recall: float

    # Per-keypoint metrics
    per_keypoint_pck: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "OKS": float(self.oks),
            "PCK": float(self.pck),
            "mean_precision": float(self.mean_precision),
            "mean_recall": float(self.mean_recall),
        }


@dataclass
class ValidationMetrics:
    """Container for task-agnostic validation metrics."""

    task_type: TaskType
    primary_metric_name: str
    primary_metric_value: float
    overall_loss: Optional[float] = None

    # Task-specific metrics (only one will be populated)
    classification: Optional[ClassificationMetrics] = None
    detection: Optional[DetectionMetrics] = None
    segmentation: Optional[SegmentationMetrics] = None
    pose: Optional[PoseMetrics] = None

    def get_task_metrics(self) -> Union[
        ClassificationMetrics,
        DetectionMetrics,
        SegmentationMetrics,
        PoseMetrics,
    ]:
        """Get the populated task-specific metrics."""
        if self.classification:
            return self.classification
        elif self.detection:
            return self.detection
        elif self.segmentation:
            return self.segmentation
        elif self.pose:
            return self.pose
        else:
            raise ValueError("No task-specific metrics found")

    def to_dict(self) -> Dict[str, Any]:
        """Convert overall metrics to dictionary."""
        task_metrics = self.get_task_metrics()
        return {
            "task_type": self.task_type.value,
            "primary_metric": {
                "name": self.primary_metric_name,
                "value": float(self.primary_metric_value),
            },
            "overall_loss": float(self.overall_loss) if self.overall_loss else None,
            "metrics": task_metrics.to_dict(),
        }


class ValidationMetricsCalculator:
    """
    Task-agnostic validation metrics calculator.

    This class provides a unified interface for computing validation metrics
    across all supported computer vision tasks.
    """

    @staticmethod
    def compute_metrics(
        task_type: TaskType,
        predictions: Any,
        labels: Any,
        class_names: Optional[List[str]] = None,
        loss: Optional[float] = None,
        **kwargs,
    ) -> ValidationMetrics:
        """
        Compute validation metrics for any task type.

        Args:
            task_type: Type of computer vision task
            predictions: Model predictions (format depends on task)
            labels: Ground truth labels (format depends on task)
            class_names: List of class/category names
            loss: Validation loss value
            **kwargs: Additional task-specific parameters

        Returns:
            ValidationMetrics object with task-specific metrics

        Examples:
            # Classification
            metrics = ValidationMetricsCalculator.compute_metrics(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                predictions=pred_labels,
                labels=true_labels,
                class_names=["cat", "dog", "bird"]
            )

            # Detection
            metrics = ValidationMetricsCalculator.compute_metrics(
                task_type=TaskType.OBJECT_DETECTION,
                predictions=pred_boxes,
                labels=true_boxes,
                class_names=["person", "car"]
            )
        """
        # Use string comparison for cross-module enum compatibility
        task_type_value = task_type.value if hasattr(task_type, 'value') else str(task_type)

        if task_type_value == "image_classification":
            return ValidationMetricsCalculator._compute_classification(
                predictions, labels, class_names, loss, **kwargs
            )
        elif task_type_value == "object_detection":
            return ValidationMetricsCalculator._compute_detection(
                predictions, labels, class_names, loss, **kwargs
            )
        elif task_type_value == "instance_segmentation":
            # YOLO-style instance segmentation: pre-computed detection metrics with Box/Mask separation
            # If predictions is None and map_50 is provided, use detection metrics
            if predictions is None and 'map_50' in kwargs:
                val_metrics = ValidationMetricsCalculator._compute_detection(
                    predictions, labels, class_names, loss, **kwargs
                )
                # Override task_type to instance_segmentation for proper UI display
                val_metrics.task_type = task_type
                return val_metrics
            # Otherwise use pixel-wise segmentation metrics
            return ValidationMetricsCalculator._compute_segmentation(
                predictions, labels, class_names, loss, is_instance=True, **kwargs
            )
        elif task_type_value == "semantic_segmentation":
            return ValidationMetricsCalculator._compute_segmentation(
                predictions, labels, class_names, loss, is_instance=False, **kwargs
            )
        elif task_type_value == "pose_estimation":
            return ValidationMetricsCalculator._compute_pose(
                predictions, labels, class_names, loss, **kwargs
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type} (value: {task_type_value})")

    @staticmethod
    def _compute_classification(
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]],
        loss: Optional[float],
        probabilities: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ValidationMetrics:
        """
        Compute classification metrics.

        Args:
            predictions: Predicted class indices (N,)
            labels: True class indices (N,)
            class_names: List of class names
            loss: Validation loss
            probabilities: Class probabilities (N, num_classes) for top-k accuracy
        """
        # Convert to numpy if needed
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # Overall metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )

        # Top-5 accuracy (if probabilities provided)
        top5_accuracy = None
        if probabilities is not None:
            # Determine actual number of classes from labels and probabilities
            num_label_classes = len(np.unique(labels))
            num_prob_classes = probabilities.shape[1]

            print(f"[DEBUG] Validation metrics:")
            print(f"  - Labels: {labels.shape}, unique classes: {num_label_classes}, range: [{labels.min()}, {labels.max()}]")
            print(f"  - Probabilities: {probabilities.shape}, num_classes: {num_prob_classes}")

            if num_prob_classes != num_label_classes:
                print(f"[WARNING] Model output classes ({num_prob_classes}) != dataset classes ({num_label_classes})")
                print(f"[WARNING] Using only first {num_label_classes} classes for metrics")
                # Slice probabilities to match actual number of classes
                probabilities = probabilities[:, :num_label_classes]

            if probabilities.shape[1] >= 5:
                top5_accuracy = top_k_accuracy_score(
                    labels, probabilities, k=min(5, probabilities.shape[1])
                )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = (
            precision_recall_fscore_support(
                labels, predictions, average=None, zero_division=0
            )
        )

        # Confusion matrix
        cm = sklearn_confusion_matrix(labels, predictions)

        # Create ClassificationMetrics
        clf_metrics = ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            top5_accuracy=top5_accuracy,
            per_class_precision=per_class_precision.tolist(),
            per_class_recall=per_class_recall.tolist(),
            per_class_f1=per_class_f1.tolist(),
            per_class_support=per_class_support.tolist(),
            confusion_matrix=cm,
            class_names=class_names,
        )

        # Wrap in ValidationMetrics
        return ValidationMetrics(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            primary_metric_name="accuracy",
            primary_metric_value=accuracy,
            overall_loss=loss,
            classification=clf_metrics,
        )

    @staticmethod
    def _compute_detection(
        predictions: Any,
        labels: Any,
        class_names: Optional[List[str]],
        loss: Optional[float],
        **kwargs,
    ) -> ValidationMetrics:
        """
        Compute object detection metrics.

        For YOLO models, metrics are pre-computed by the framework.
        Pass them via kwargs: map_50, map_50_95, precision, recall

        Args:
            predictions: Predicted bounding boxes (not used if metrics provided)
            labels: Ground truth bounding boxes (not used if metrics provided)
            class_names: List of class names
            loss: Validation loss
            **kwargs: Pre-computed metrics from YOLO (map_50, map_50_95, precision, recall)
        """
        # Extract pre-computed metrics from kwargs
        map_50 = kwargs.get('map_50', 0.0)
        map_50_95 = kwargs.get('map_50_95', 0.0)
        precision = kwargs.get('precision', 0.0)
        recall = kwargs.get('recall', 0.0)

        detection_metrics = DetectionMetrics(
            map_50=map_50,
            map_50_95=map_50_95,
            precision=precision,
            recall=recall,
        )

        return ValidationMetrics(
            task_type=TaskType.OBJECT_DETECTION,
            primary_metric_name="mAP@0.5",
            primary_metric_value=map_50,
            overall_loss=loss,
            detection=detection_metrics,
        )

    @staticmethod
    def _compute_segmentation(
        predictions: Any,
        labels: Any,
        class_names: Optional[List[str]],
        loss: Optional[float],
        is_instance: bool = False,
        **kwargs,
    ) -> ValidationMetrics:
        """
        Compute segmentation metrics.

        TODO: Implement full segmentation metrics calculation.
        For now, returns stub metrics.

        Args:
            predictions: Predicted masks
            labels: Ground truth masks
            class_names: List of class names
            loss: Validation loss
            is_instance: True for instance segmentation, False for semantic
        """
        # Stub implementation for Week 1

        seg_metrics = SegmentationMetrics(
            mean_iou=0.0,
            pixel_accuracy=0.0,
            mean_precision=0.0,
            mean_recall=0.0,
        )

        task_type = (
            TaskType.INSTANCE_SEGMENTATION
            if is_instance
            else TaskType.SEMANTIC_SEGMENTATION
        )

        return ValidationMetrics(
            task_type=task_type,
            primary_metric_name="mean_iou",
            primary_metric_value=0.0,
            overall_loss=loss,
            segmentation=seg_metrics,
        )

    @staticmethod
    def _compute_pose(
        predictions: Any,
        labels: Any,
        class_names: Optional[List[str]],
        loss: Optional[float],
        **kwargs,
    ) -> ValidationMetrics:
        """
        Compute pose estimation metrics.

        TODO: Implement full pose metrics calculation.
        For now, returns stub metrics.

        Args:
            predictions: Predicted keypoints
            labels: Ground truth keypoints
            class_names: Keypoint names
            loss: Validation loss
        """
        # Stub implementation for Week 1

        pose_metrics = PoseMetrics(
            oks=0.0,
            pck=0.0,
            mean_precision=0.0,
            mean_recall=0.0,
        )

        return ValidationMetrics(
            task_type=TaskType.POSE_ESTIMATION,
            primary_metric_name="OKS",
            primary_metric_value=0.0,
            overall_loss=loss,
            pose=pose_metrics,
        )
