"""HuggingFace Transformers adapter for vision tasks."""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from tqdm import tqdm
from PIL import Image

from platform_sdk import (
    TrainingAdapter,
    MetricsResult,
    TaskType,
    DatasetFormat,
    ConfigSchema,
    ConfigField,
    InferenceResult,
)


class TransformersAdapter(TrainingAdapter):
    """
    Adapter for HuggingFace Transformers.

    Supported tasks:
    - Image Classification (ViT, DeiT, etc.)
    - Object Detection (D-FINE, DETR, etc.)
    - Semantic Segmentation (EoMT, SegFormer, etc.)
    - Super-Resolution (Swin2SR, etc.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.processor = None  # HF Image Processor
        self.trainer = None    # HF Trainer
        self.train_dataset = None  # HF Dataset
        self.val_dataset = None    # HF Dataset
        self.current_epoch = 0

        # Task-specific attributes
        self.upscale_factor = getattr(self.model_config, 'upscale_factor', 2)  # For SR

    @classmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Return configuration schema for HuggingFace models."""
        # For now, return basic schema
        # TODO: Implement task-specific schemas
        fields = [
            ConfigField(
                name="use_pretrained",
                type="bool",
                default=True,
                description="Use pretrained weights",
                required=False
            ),
            ConfigField(
                name="freeze_backbone",
                type="bool",
                default=False,
                description="Freeze backbone weights (only train classifier)",
                required=False
            ),
        ]

        return ConfigSchema(fields=fields, presets=None)

    def prepare_model(self) -> None:
        """
        Initialize model based on task type.

        Supports:
        - IMAGE_CLASSIFICATION: AutoModelForImageClassification
        - OBJECT_DETECTION: DFineForObjectDetection
        - SEMANTIC_SEGMENTATION: EoMTForSemanticSegmentation
        - SUPER_RESOLUTION: Swin2SRModel
        """
        from transformers import AutoImageProcessor

        task_type = self.model_config.task_type
        model_name = self.model_config.model_name

        print(f"\nLoading HuggingFace model: {model_name}")
        print(f"Task type: {task_type.value}")

        if task_type == TaskType.IMAGE_CLASSIFICATION:
            self._prepare_classification_model()
        elif task_type == TaskType.OBJECT_DETECTION:
            self._prepare_detection_model()
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            self._prepare_segmentation_model()
        elif task_type == TaskType.SUPER_RESOLUTION:
            self._prepare_super_resolution_model()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        # Move model to device
        device = self.training_config.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, using CPU")
            device = "cpu"

        self.model = self.model.to(device)
        print(f"Model loaded on {device}")

    def _prepare_classification_model(self):
        """Prepare model for image classification."""
        from transformers import AutoModelForImageClassification, AutoImageProcessor

        model_name = self.model_config.model_name
        num_labels = self.model_config.num_classes

        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Load model
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # Allow different num_labels
        )

        print(f"  - Processor: {model_name}")
        print(f"  - Model: {type(self.model).__name__}")
        print(f"  - Number of classes: {num_labels}")

    def _prepare_detection_model(self):
        """Prepare model for object detection."""
        raise NotImplementedError("Object detection not yet implemented in Phase 2")

    def _prepare_segmentation_model(self):
        """Prepare model for semantic segmentation."""
        raise NotImplementedError("Semantic segmentation not yet implemented in Phase 2")

    def _prepare_super_resolution_model(self):
        """Prepare model for super-resolution."""
        raise NotImplementedError("Super-resolution not yet implemented in Phase 2")

    def prepare_dataset(self) -> None:
        """
        Load and preprocess dataset.

        Converts dataset to HuggingFace Dataset format.
        """
        task_type = self.model_config.task_type

        print(f"\n{'='*80}")
        print("DATA PREPARATION")
        print(f"{'='*80}")
        print(f"[CONFIG] Dataset Path: {self.dataset_config.dataset_path}")
        print(f"[CONFIG] Format: {self.dataset_config.format.value}")
        print(f"[CONFIG] Task: {task_type.value}")

        if task_type == TaskType.IMAGE_CLASSIFICATION:
            self._prepare_classification_dataset()
        elif task_type == TaskType.OBJECT_DETECTION:
            self._prepare_detection_dataset()
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            self._prepare_segmentation_dataset()
        elif task_type == TaskType.SUPER_RESOLUTION:
            self._prepare_super_resolution_dataset()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        print(f"[INFO] Train samples: {len(self.train_dataset)}")
        print(f"[INFO] Val samples: {len(self.val_dataset)}")

    def _prepare_classification_dataset(self):
        """
        Convert ImageFolder to HuggingFace Dataset for classification.

        ImageFolder structure:
        dataset/
        ├── train/
        │   ├── class1/
        │   │   ├── img1.jpg
        │   │   └── img2.jpg
        │   └── class2/
        └── val/
        """
        from datasets import Dataset, Features, Image as HFImage, ClassLabel
        import os
        from pathlib import Path

        dataset_path = self.dataset_config.dataset_path

        # Collect train images
        train_dir = os.path.join(dataset_path, "train")
        val_dir = os.path.join(dataset_path, "val")

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise ValueError(f"Train or val directory not found in {dataset_path}")

        # Get class names
        class_names = sorted([d for d in os.listdir(train_dir)
                             if os.path.isdir(os.path.join(train_dir, d))])
        self.class_names = class_names
        print(f"[INFO] Detected {len(class_names)} classes: {class_names}")

        # Create train dataset
        self.train_dataset = self._load_imagefolder_split(train_dir, class_names)
        self.val_dataset = self._load_imagefolder_split(val_dir, class_names)

        # Apply preprocessing
        self.train_dataset = self.train_dataset.map(
            self._preprocess_classification,
            batched=True,
            remove_columns=["image"]  # Remove raw image, keep processed
        )

        self.val_dataset = self.val_dataset.map(
            self._preprocess_classification,
            batched=True,
            remove_columns=["image"]
        )

        print(f"[INFO] Preprocessing complete")

    def _load_imagefolder_split(self, split_dir: str, class_names: List[str]):
        """Load images from ImageFolder structure."""
        from datasets import Dataset, Features, Image as HFImage, ClassLabel

        images = []
        labels = []

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"[WARNING] Class directory not found: {class_dir}")
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    images.append(img_path)
                    labels.append(class_idx)

        # Create HF Dataset
        dataset = Dataset.from_dict({
            "image": images,
            "label": labels
        })

        # Cast image column to HF Image type
        dataset = dataset.cast_column("image", HFImage())

        return dataset

    def _preprocess_classification(self, examples):
        """
        Preprocess images using HuggingFace processor.

        Args:
            examples: Batch of examples with 'image' and 'label'

        Returns:
            Preprocessed examples with 'pixel_values' and 'labels'
        """
        # Convert PIL images to RGB
        images = [img.convert("RGB") for img in examples["image"]]

        # Process images
        inputs = self.processor(images, return_tensors="pt")

        # Add labels
        inputs["labels"] = examples["label"]

        return inputs

    def _prepare_detection_dataset(self):
        """Convert COCO format to HuggingFace Dataset for detection."""
        raise NotImplementedError("Detection dataset not yet implemented")

    def _prepare_segmentation_dataset(self):
        """Load segmentation masks and convert to HuggingFace Dataset."""
        raise NotImplementedError("Segmentation dataset not yet implemented")

    def _prepare_super_resolution_dataset(self):
        """Load HR-LR paired images for super-resolution."""
        raise NotImplementedError("Super-resolution dataset not yet implemented")

    def train_epoch(self, epoch: int) -> MetricsResult:
        """
        Train one epoch using HuggingFace Trainer.

        Args:
            epoch: Current epoch number

        Returns:
            MetricsResult with training metrics
        """
        from transformers import Trainer, TrainingArguments, TrainerCallback
        import mlflow

        print(f"\n[Epoch {epoch}/{self.training_config.epochs}]")
        self.current_epoch = epoch

        # Define training arguments for single epoch
        output_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,  # Train only 1 epoch per call
            per_device_train_batch_size=self.training_config.batch_size,
            learning_rate=self.training_config.learning_rate,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            save_strategy="no",  # Don't auto-save, we'll handle it
            evaluation_strategy="no",  # We'll call evaluate() separately
            report_to="none",  # Don't use wandb/tensorboard
            disable_tqdm=False,
        )

        # Create custom callback for MLflow logging
        callback = MLflowCallback(self.job_id, epoch)

        # Create Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            callbacks=[callback],
        )

        # Train
        print(f"[TRAIN] Starting training...")
        train_result = self.trainer.train()

        # Extract metrics
        metrics = train_result.metrics
        train_loss = metrics.get("train_loss", 0.0)

        print(f"[TRAIN] Loss: {train_loss:.4f}")

        # Log to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)

        return MetricsResult(
            epoch=epoch,
            step=0,
            train_loss=train_loss,
            val_loss=None,  # Will be filled by validate()
            metrics={}
        )

    def validate(self, epoch: int) -> MetricsResult:
        """
        Run validation using HuggingFace Trainer.

        Args:
            epoch: Current epoch number

        Returns:
            MetricsResult with validation metrics
        """
        import mlflow

        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train_epoch() first.")

        print(f"[VAL] Running validation...")

        # Evaluate
        eval_results = self.trainer.evaluate(eval_dataset=self.val_dataset)

        # Extract metrics
        val_loss = eval_results.get("eval_loss", 0.0)

        # For classification, HF Trainer doesn't compute accuracy by default
        # We need to compute it manually
        if self.model_config.task_type == TaskType.IMAGE_CLASSIFICATION:
            accuracy = self._compute_classification_accuracy()
            print(f"[VAL] Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Log to MLflow
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

            return MetricsResult(
                epoch=epoch,
                step=0,
                train_loss=0.0,  # Already logged in train_epoch
                val_loss=val_loss,
                metrics={
                    "accuracy": accuracy,
                }
            )
        else:
            print(f"[VAL] Loss: {val_loss:.4f}")
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            return MetricsResult(
                epoch=epoch,
                step=0,
                train_loss=0.0,
                val_loss=val_loss,
                metrics={}
            )

    def _compute_classification_accuracy(self) -> float:
        """Compute classification accuracy on validation set."""
        self.model.eval()
        correct = 0
        total = 0

        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in self.val_dataset:
                # Get pixel values and labels
                pixel_values = torch.tensor(batch["pixel_values"]).unsqueeze(0).to(device)
                labels = batch["labels"]

                # Forward pass
                outputs = self.model(pixel_values)
                predictions = outputs.logits.argmax(dim=-1).cpu().item()

                # Count correct
                if predictions == labels:
                    correct += 1
                total += 1

        accuracy = (correct / total) * 100.0
        return accuracy

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = os.path.join(
            self.output_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )

        # Save model using HF save_pretrained
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)

        print(f"[CHECKPOINT] Saved to {checkpoint_dir}")

        # Save best model
        if is_best:
            best_dir = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            self.model.save_pretrained(best_dir)
            self.processor.save_pretrained(best_dir)
            print(f"[CHECKPOINT] Best model saved to {best_dir}")

        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        from transformers import AutoModelForImageClassification, AutoImageProcessor

        print(f"[CHECKPOINT] Loading from {checkpoint_path}")

        self.model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint_path)

        device = self.training_config.device
        self.model = self.model.to(device)

        print(f"[CHECKPOINT] Loaded successfully")

    def inference(self, image_path: str) -> InferenceResult:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image

        Returns:
            InferenceResult with predictions
        """
        import time

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        start_time = time.time()
        inputs = self.processor(image, return_tensors="pt")
        preprocessing_time = (time.time() - start_time) * 1000

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        inference_time = (time.time() - start_time) * 1000

        # Post-process
        start_time = time.time()
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = probs.argmax(dim=-1).item()
        confidence = probs[0, predicted_class].item()

        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probs[0], k=min(5, probs.shape[-1]))
        top5_predictions = [
            {
                "class_id": idx.item(),
                "class_name": self.class_names[idx.item()] if hasattr(self, 'class_names') else str(idx.item()),
                "confidence": prob.item()
            }
            for prob, idx in zip(top5_probs, top5_indices)
        ]
        postprocessing_time = (time.time() - start_time) * 1000

        return InferenceResult(
            image_path=image_path,
            image_name=os.path.basename(image_path),
            task_type=TaskType.IMAGE_CLASSIFICATION,
            predicted_label=self.class_names[predicted_class] if hasattr(self, 'class_names') else str(predicted_class),
            predicted_label_id=predicted_class,
            confidence=confidence,
            top5_predictions=top5_predictions,
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocessing_time,
            postprocessing_time_ms=postprocessing_time,
        )


class MLflowCallback:
    """Custom callback for MLflow logging."""

    def __init__(self, job_id: int, epoch: int):
        self.job_id = job_id
        self.epoch = epoch

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged."""
        if logs:
            import mlflow
            # Log metrics to MLflow
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"hf_{key}", value, step=state.global_step)
