"""
Test and Inference runner utilities.

Provides classes for running test and inference on trained models
using the framework-independent adapter pattern.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

# Add training module to path
import os
backend_dir = Path(__file__).parent.parent.parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))

from app.db import models
from training.adapters.base import TrainingAdapter, InferenceResult, TaskType
from training.adapters.timm_adapter import TimmAdapter
from training.adapters.ultralytics_adapter import UltralyticsAdapter
from training.validators.metrics_calculator import ValidationMetricsCalculator


class TestRunner:
    """
    Run tests on labeled datasets with metric calculation.

    Uses the adapter pattern to support any framework.
    """

    def __init__(self, db: Session):
        """
        Initialize test runner.

        Args:
            db: Database session
        """
        self.db = db

    def run_test(self, test_run_id: int) -> bool:
        """
        Run test on labeled dataset.

        Loads checkpoint, runs inference on all images, calculates metrics.

        Args:
            test_run_id: TestRun ID to execute

        Returns:
            True if test completed successfully
        """
        # Get test run from database
        test_run = self.db.query(models.TestRun).filter(
            models.TestRun.id == test_run_id
        ).first()

        if not test_run:
            print(f"[ERROR] Test run {test_run_id} not found")
            return False

        if test_run.status != "pending":
            print(f"[ERROR] Test run {test_run_id} is not in pending status")
            return False

        # Get training job
        job = self.db.query(models.TrainingJob).filter(
            models.TrainingJob.id == test_run.training_job_id
        ).first()

        if not job:
            print(f"[ERROR] Training job {test_run.training_job_id} not found")
            test_run.status = "failed"
            test_run.error_message = f"Training job {test_run.training_job_id} not found"
            self.db.commit()
            return False

        try:
            # Update status to running
            test_run.status = "running"
            test_run.started_at = datetime.utcnow()
            self.db.commit()

            print(f"[TEST] Starting test run {test_run_id} for job {job.id}")
            print(f"[TEST] Framework: {job.framework}, Task: {job.task_type}, Model: {job.model_name}")
            print(f"[TEST] Checkpoint: {test_run.checkpoint_path}")
            print(f"[TEST] Dataset: {test_run.dataset_path}")

            # Create adapter
            adapter = self._create_adapter(job, test_run.dataset_path)
            if not adapter:
                raise ValueError(f"Unsupported framework: {job.framework}")

            # Load checkpoint for inference
            print(f"[TEST] Loading checkpoint...")
            adapter.load_checkpoint(
                checkpoint_path=test_run.checkpoint_path,
                inference_mode=True
            )

            # Get image paths from dataset
            print(f"[TEST] Loading dataset...")
            image_paths = self._get_image_paths(test_run.dataset_path, test_run.dataset_split)
            test_run.total_images = len(image_paths)
            self.db.commit()

            print(f"[TEST] Found {len(image_paths)} images in {test_run.dataset_split} split")

            # Run inference on all images
            print(f"[TEST] Running inference...")
            start_time = time.time()
            inference_results = adapter.infer_batch(image_paths)
            total_inference_time = (time.time() - start_time) * 1000  # ms

            test_run.inference_time_ms = total_inference_time
            self.db.commit()

            print(f"[TEST] Inference completed in {total_inference_time:.2f}ms")

            # Store image-level results
            print(f"[TEST] Storing image results...")
            self._store_test_image_results(test_run, inference_results)

            # Calculate metrics
            print(f"[TEST] Calculating metrics...")
            metrics = self._calculate_metrics(test_run, inference_results)

            # Update test run with metrics
            test_run.overall_loss = metrics.get("loss")
            test_run.primary_metric_name = metrics.get("primary_metric_name")
            test_run.primary_metric_value = metrics.get("primary_metric_value")
            test_run.metrics = metrics.get("metrics")
            test_run.per_class_metrics = metrics.get("per_class_metrics")
            test_run.confusion_matrix = metrics.get("confusion_matrix")
            test_run.class_names = metrics.get("class_names")

            # Mark as completed
            test_run.status = "completed"
            test_run.completed_at = datetime.utcnow()
            self.db.commit()

            print(f"[TEST] Test run {test_run_id} completed successfully")
            print(f"[TEST] {test_run.primary_metric_name}: {test_run.primary_metric_value:.4f}")

            return True

        except Exception as e:
            print(f"[ERROR] Test run {test_run_id} failed: {e}")
            import traceback
            traceback.print_exc()

            test_run.status = "failed"
            test_run.error_message = str(e)
            test_run.completed_at = datetime.utcnow()
            self.db.commit()

            return False

    def _create_adapter(self, job: models.TrainingJob, dataset_path: str) -> Optional[TrainingAdapter]:
        """Create adapter instance based on framework."""
        if job.framework == "timm":
            return TimmAdapter(
                model_name=job.model_name,
                task_type=TaskType(job.task_type),
                num_classes=job.num_classes,
                dataset_path=dataset_path,
                output_dir=job.output_dir,
                job_id=job.id
            )
        elif job.framework == "ultralytics":
            return UltralyticsAdapter(
                model_name=job.model_name,
                task_type=TaskType(job.task_type),
                num_classes=job.num_classes,
                dataset_path=dataset_path,
                output_dir=job.output_dir,
                job_id=job.id
            )
        else:
            return None

    def _get_image_paths(self, dataset_path: str, split: str) -> List[str]:
        """
        Get image paths from dataset.

        Currently supports ImageFolder format.
        Future: Support YOLO, COCO formats.
        """
        dataset_dir = Path(dataset_path)
        split_dir = dataset_dir / split

        if not split_dir.exists():
            # Try without split directory (flat structure)
            split_dir = dataset_dir

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend([str(p) for p in split_dir.rglob(f"*{ext}")])

        return sorted(image_paths)

    def _store_test_image_results(self, test_run: models.TestRun, results: List[InferenceResult]):
        """Store image-level test results in database."""
        for idx, result in enumerate(results):
            # Get ground truth from dataset structure (ImageFolder format)
            true_label, true_label_id = self._extract_ground_truth(result)

            # Determine if prediction is correct
            is_correct = False
            if result.task_type == TaskType.IMAGE_CLASSIFICATION:
                is_correct = (result.predicted_label_id == true_label_id)
            # TODO: Add correctness logic for detection/segmentation/pose

            # Create image result record
            image_result = models.TestImageResult(
                test_run_id=test_run.id,
                training_job_id=test_run.training_job_id,
                image_path=result.image_path,
                image_name=result.image_name,
                image_index=idx,
                # Ground truth
                true_label=true_label,
                true_label_id=true_label_id,
                # Predictions
                predicted_label=result.predicted_label,
                predicted_label_id=result.predicted_label_id,
                confidence=result.confidence,
                top5_predictions=result.top5_predictions,
                predicted_boxes=result.predicted_boxes,
                predicted_mask_path=result.predicted_mask_path,
                predicted_keypoints=result.predicted_keypoints,
                # Metrics
                is_correct=is_correct,
                iou=result.extra_data.get("iou") if result.extra_data else None,
                oks=result.extra_data.get("oks") if result.extra_data else None,
                # Performance
                inference_time_ms=result.inference_time_ms,
                preprocessing_time_ms=result.preprocessing_time_ms,
                postprocessing_time_ms=result.postprocessing_time_ms,
                extra_data=result.extra_data
            )

            self.db.add(image_result)

        self.db.commit()

    def _extract_ground_truth(self, result: InferenceResult) -> tuple:
        """
        Extract ground truth label from image path (ImageFolder format).

        ImageFolder structure: dataset/split/class_name/image.jpg
        """
        image_path = Path(result.image_path)
        class_name = image_path.parent.name

        # TODO: Map class_name to class_id using dataset metadata
        # For now, return class name and None for ID
        return class_name, None

    def _calculate_metrics(self, test_run: models.TestRun, results: List[InferenceResult]) -> Dict[str, Any]:
        """
        Calculate test metrics using ValidationMetricsCalculator.

        Reuses validation system's metric calculation logic.
        """
        # Query image results from database
        image_results = self.db.query(models.TestImageResult).filter(
            models.TestImageResult.test_run_id == test_run.id
        ).all()

        # Use ValidationMetricsCalculator
        calculator = ValidationMetricsCalculator(
            task_type=TaskType(test_run.task_type)
        )

        # Convert TestImageResult to format expected by calculator
        # For classification
        if test_run.task_type == "image_classification":
            true_labels = [r.true_label_id for r in image_results if r.true_label_id is not None]
            pred_labels = [r.predicted_label_id for r in image_results if r.predicted_label_id is not None]

            if not true_labels or not pred_labels:
                print("[WARNING] No valid labels found for metric calculation")
                return {
                    "primary_metric_name": "accuracy",
                    "primary_metric_value": 0.0,
                    "metrics": {},
                    "per_class_metrics": {},
                    "confusion_matrix": []
                }

            metrics = calculator.calculate_classification_metrics(
                true_labels=true_labels,
                pred_labels=pred_labels,
                pred_probs=None  # TODO: Extract confidences if needed
            )

            return {
                "primary_metric_name": "accuracy",
                "primary_metric_value": metrics.get("accuracy", 0.0),
                "metrics": metrics,
                "per_class_metrics": metrics.get("per_class", {}),
                "confusion_matrix": metrics.get("confusion_matrix", []),
                "class_names": metrics.get("class_names", [])
            }

        # TODO: Add detection/segmentation/pose metrics

        return {}


class InferenceRunner:
    """
    Run inference on unlabeled data (production use case).

    Uses the adapter pattern to support any framework.
    """

    def __init__(self, db: Session):
        """
        Initialize inference runner.

        Args:
            db: Database session
        """
        self.db = db

    def run_inference(self, inference_job_id: int) -> bool:
        """
        Run inference on unlabeled data.

        Loads checkpoint, runs predictions, stores results.

        Args:
            inference_job_id: InferenceJob ID to execute

        Returns:
            True if inference completed successfully
        """
        # Get inference job from database
        inference_job = self.db.query(models.InferenceJob).filter(
            models.InferenceJob.id == inference_job_id
        ).first()

        if not inference_job:
            print(f"[ERROR] Inference job {inference_job_id} not found")
            return False

        if inference_job.status != "pending":
            print(f"[ERROR] Inference job {inference_job_id} is not in pending status")
            return False

        # Get training job
        job = self.db.query(models.TrainingJob).filter(
            models.TrainingJob.id == inference_job.training_job_id
        ).first()

        if not job:
            print(f"[ERROR] Training job {inference_job.training_job_id} not found")
            inference_job.status = "failed"
            inference_job.error_message = f"Training job {inference_job.training_job_id} not found"
            self.db.commit()
            return False

        try:
            # Update status to running
            inference_job.status = "running"
            inference_job.started_at = datetime.utcnow()
            self.db.commit()

            print(f"[INFERENCE] Starting inference job {inference_job_id} for job {job.id}")
            print(f"[INFERENCE] Framework: {job.framework}, Task: {job.task_type}, Model: {job.model_name}")
            print(f"[INFERENCE] Checkpoint: {inference_job.checkpoint_path}")

            # Create adapter (use dummy dataset path for inference)
            adapter = self._create_adapter(job, job.dataset_path)
            if not adapter:
                raise ValueError(f"Unsupported framework: {job.framework}")

            # Load checkpoint for inference
            print(f"[INFERENCE] Loading checkpoint...")
            adapter.load_checkpoint(
                checkpoint_path=inference_job.checkpoint_path,
                inference_mode=True
            )

            # Get image paths from input_data
            input_data = json.loads(inference_job.input_data) if isinstance(inference_job.input_data, str) else inference_job.input_data
            image_paths = self._get_image_paths_from_input(inference_job.inference_type, input_data)

            inference_job.total_images = len(image_paths)
            self.db.commit()

            print(f"[INFERENCE] Processing {len(image_paths)} images")

            # Run inference
            print(f"[INFERENCE] Running inference...")
            start_time = time.time()
            inference_results = adapter.infer_batch(image_paths)
            total_inference_time = (time.time() - start_time) * 1000  # ms

            inference_job.total_inference_time_ms = total_inference_time
            inference_job.avg_inference_time_ms = total_inference_time / len(image_paths) if image_paths else 0
            self.db.commit()

            print(f"[INFERENCE] Inference completed in {total_inference_time:.2f}ms")
            print(f"[INFERENCE] Average time per image: {inference_job.avg_inference_time_ms:.2f}ms")

            # Store inference results
            print(f"[INFERENCE] Storing results...")
            self._store_inference_results(inference_job, inference_results)

            # Mark as completed
            inference_job.status = "completed"
            inference_job.completed_at = datetime.utcnow()
            self.db.commit()

            print(f"[INFERENCE] Inference job {inference_job_id} completed successfully")

            return True

        except Exception as e:
            print(f"[ERROR] Inference job {inference_job_id} failed: {e}")
            import traceback
            traceback.print_exc()

            inference_job.status = "failed"
            inference_job.error_message = str(e)
            inference_job.completed_at = datetime.utcnow()
            self.db.commit()

            return False

    def _create_adapter(self, job: models.TrainingJob, dataset_path: str) -> Optional[TrainingAdapter]:
        """Create adapter instance based on framework."""
        if job.framework == "timm":
            return TimmAdapter(
                model_name=job.model_name,
                task_type=TaskType(job.task_type),
                num_classes=job.num_classes,
                dataset_path=dataset_path,
                output_dir=job.output_dir,
                job_id=job.id
            )
        elif job.framework == "ultralytics":
            return UltralyticsAdapter(
                model_name=job.model_name,
                task_type=TaskType(job.task_type),
                num_classes=job.num_classes,
                dataset_path=dataset_path,
                output_dir=job.output_dir,
                job_id=job.id
            )
        else:
            return None

    def _get_image_paths_from_input(self, inference_type: str, input_data: Dict[str, Any]) -> List[str]:
        """
        Get image paths from input_data based on inference type.

        Args:
            inference_type: 'single', 'batch', or 'dataset'
            input_data: Input data dictionary

        Returns:
            List of image paths
        """
        if inference_type == "single":
            return [input_data.get("image_path")]
        elif inference_type == "batch":
            return input_data.get("image_paths", [])
        elif inference_type == "dataset":
            dataset_path = input_data.get("dataset_path")
            # Get all images from dataset directory
            dataset_dir = Path(dataset_path)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            image_paths = []
            for ext in image_extensions:
                image_paths.extend([str(p) for p in dataset_dir.rglob(f"*{ext}")])
            return sorted(image_paths)
        else:
            return []

    def _store_inference_results(self, inference_job: models.InferenceJob, results: List[InferenceResult]):
        """Store inference results in database (predictions only, no ground truth)."""
        for idx, result in enumerate(results):
            inference_result = models.InferenceResult(
                inference_job_id=inference_job.id,
                training_job_id=inference_job.training_job_id,
                image_path=result.image_path,
                image_name=result.image_name,
                image_index=idx,
                # Predictions only (no ground truth)
                predicted_label=result.predicted_label,
                predicted_label_id=result.predicted_label_id,
                confidence=result.confidence,
                top5_predictions=result.top5_predictions,
                predicted_boxes=result.predicted_boxes,
                predicted_mask_path=result.predicted_mask_path,
                predicted_keypoints=result.predicted_keypoints,
                # Performance
                inference_time_ms=result.inference_time_ms,
                preprocessing_time_ms=result.preprocessing_time_ms,
                postprocessing_time_ms=result.postprocessing_time_ms,
                extra_data=result.extra_data
            )

            self.db.add(inference_result)

        self.db.commit()
