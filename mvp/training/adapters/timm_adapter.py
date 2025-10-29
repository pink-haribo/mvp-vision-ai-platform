"""Timm adapter for image classification."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from .base import TrainingAdapter, MetricsResult, TaskType, DatasetFormat, ConfigSchema, ConfigField


class TimmAdapter(TrainingAdapter):
    """
    Adapter for timm (PyTorch Image Models).

    Supported tasks:
    - Image Classification (ResNet, EfficientNet, ViT, etc.)
    """

    @classmethod
    def get_config_schema(cls) -> ConfigSchema:
        """Return configuration schema for timm models."""
        from training.config_schemas import get_timm_schema
        return get_timm_schema()

    @classmethod
    def _get_config_schema_inline(cls) -> ConfigSchema:
        """Return configuration schema for timm models (image classification)."""
        fields = [
            # ========== Optimizer Settings ==========
            ConfigField(
                name="optimizer_type",
                type="select",
                default="adam",
                options=["adam", "adamw", "sgd", "rmsprop"],
                description="Optimizer algorithm",
                group="optimizer",
                required=False
            ),
            ConfigField(
                name="weight_decay",
                type="float",
                default=0.0001,
                min=0.0,
                max=0.1,
                step=0.0001,
                description="L2 regularization (weight decay)",
                group="optimizer",
                advanced=True
            ),
            ConfigField(
                name="momentum",
                type="float",
                default=0.9,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Momentum for SGD optimizer",
                group="optimizer",
                advanced=True
            ),

            # ========== Scheduler Settings ==========
            ConfigField(
                name="scheduler_type",
                type="select",
                default="cosine",
                options=["none", "step", "cosine", "plateau", "exponential"],
                description="Learning rate scheduler",
                group="scheduler",
                required=False
            ),
            ConfigField(
                name="warmup_epochs",
                type="int",
                default=5,
                min=0,
                max=50,
                step=1,
                description="Number of warmup epochs",
                group="scheduler",
                advanced=False
            ),
            ConfigField(
                name="step_size",
                type="int",
                default=30,
                min=1,
                max=100,
                step=1,
                description="Step size for StepLR scheduler",
                group="scheduler",
                advanced=True
            ),
            ConfigField(
                name="gamma",
                type="float",
                default=0.1,
                min=0.01,
                max=1.0,
                step=0.01,
                description="Multiplicative factor of learning rate decay",
                group="scheduler",
                advanced=True
            ),
            ConfigField(
                name="eta_min",
                type="float",
                default=0.000001,
                min=0.0,
                max=0.01,
                step=0.000001,
                description="Minimum learning rate for CosineAnnealingLR",
                group="scheduler",
                advanced=True
            ),

            # ========== Augmentation Settings ==========
            ConfigField(
                name="aug_enabled",
                type="bool",
                default=True,
                description="Enable data augmentation",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="random_flip",
                type="bool",
                default=True,
                description="Random horizontal flip",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="random_flip_prob",
                type="float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Probability of random flip",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="random_rotation",
                type="bool",
                default=False,
                description="Random rotation",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="rotation_degrees",
                type="int",
                default=15,
                min=0,
                max=180,
                step=5,
                description="Maximum rotation degrees",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="random_crop",
                type="bool",
                default=True,
                description="Random crop with resize",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="color_jitter",
                type="bool",
                default=False,
                description="Random color jitter",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="brightness",
                type="float",
                default=0.2,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Brightness variation",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="contrast",
                type="float",
                default=0.2,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Contrast variation",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="saturation",
                type="float",
                default=0.2,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Saturation variation",
                group="augmentation",
                advanced=True
            ),
            ConfigField(
                name="hue",
                type="float",
                default=0.1,
                min=0.0,
                max=0.5,
                step=0.05,
                description="Hue variation",
                group="augmentation",
                advanced=True
            ),
            ConfigField(
                name="random_erasing",
                type="bool",
                default=False,
                description="Random erasing augmentation",
                group="augmentation",
                advanced=True
            ),
            ConfigField(
                name="mixup",
                type="bool",
                default=False,
                description="Mixup augmentation (image blending)",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="mixup_alpha",
                type="float",
                default=0.2,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Mixup alpha parameter",
                group="augmentation",
                advanced=False
            ),
            ConfigField(
                name="cutmix",
                type="bool",
                default=False,
                description="CutMix augmentation (region blending)",
                group="augmentation",
                required=False
            ),
            ConfigField(
                name="cutmix_alpha",
                type="float",
                default=1.0,
                min=0.0,
                max=2.0,
                step=0.1,
                description="CutMix alpha parameter",
                group="augmentation",
                advanced=False
            ),

            # ========== Validation Settings ==========
            ConfigField(
                name="val_interval",
                type="int",
                default=1,
                min=1,
                max=10,
                step=1,
                description="Validate every N epochs",
                group="validation",
                required=False
            ),
        ]

        presets = {
            "easy": {
                "optimizer_type": "adam",
                "scheduler_type": "cosine",
                "aug_enabled": True,
                "random_flip": True,
                "mixup": False,
                "cutmix": False,
            },
            "medium": {
                "optimizer_type": "adamw",
                "weight_decay": 0.0001,
                "scheduler_type": "cosine",
                "warmup_epochs": 5,
                "aug_enabled": True,
                "random_flip": True,
                "color_jitter": True,
                "mixup": True,
                "mixup_alpha": 0.2,
            },
            "advanced": {
                "optimizer_type": "adamw",
                "weight_decay": 0.0005,
                "scheduler_type": "cosine",
                "warmup_epochs": 10,
                "eta_min": 0.000001,
                "aug_enabled": True,
                "random_flip": True,
                "random_rotation": True,
                "color_jitter": True,
                "random_erasing": True,
                "mixup": True,
                "mixup_alpha": 0.4,
                "cutmix": True,
                "cutmix_alpha": 1.0,
            }
        }

        return ConfigSchema(fields=fields, presets=presets)

    def prepare_model(self):
        """Initialize timm model."""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm not installed. Install with: pip install timm"
            )

        print(f"Loading timm model: {self.model_config.model_name}")
        self.model = timm.create_model(
            self.model_config.model_name,
            pretrained=self.model_config.pretrained,
            num_classes=self.model_config.num_classes
        )

        # Move to device
        device = torch.device(self.training_config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.device = device

        print(f"Model loaded on {self.device}")

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Build optimizer from advanced config (or use basic config)
        print("\n" + "="*80)
        print("OPTIMIZER CONFIGURATION")
        print("="*80)
        self.optimizer = self.build_optimizer(self.model.parameters())
        print(f"[CONFIG] Optimizer Type: {self.optimizer.__class__.__name__}")

        # Print optimizer parameters
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            print(f"[CONFIG] Parameter Group {group_idx}:")
            print(f"         - Learning Rate: {param_group['lr']}")
            if 'weight_decay' in param_group:
                print(f"         - Weight Decay: {param_group['weight_decay']}")
            if 'momentum' in param_group:
                print(f"         - Momentum: {param_group['momentum']}")
            if 'betas' in param_group:
                print(f"         - Betas: {param_group['betas']}")

        # Build scheduler from advanced config (optional)
        print("\n" + "="*80)
        print("SCHEDULER CONFIGURATION")
        print("="*80)
        self.scheduler = self.build_scheduler(self.optimizer)
        if self.scheduler:
            print(f"[CONFIG] Scheduler Type: {self.scheduler.__class__.__name__}")
            # Print scheduler-specific parameters
            if hasattr(self.scheduler, 'step_size'):
                print(f"         - Step Size: {self.scheduler.step_size}")
            if hasattr(self.scheduler, 'gamma'):
                print(f"         - Gamma: {self.scheduler.gamma}")
            if hasattr(self.scheduler, 'T_max'):
                print(f"         - T_max: {self.scheduler.T_max}")
            if hasattr(self.scheduler, 'eta_min'):
                print(f"         - Min LR: {self.scheduler.eta_min}")
        else:
            print("[CONFIG] No scheduler configured (learning rate will be constant)")

        self.best_val_acc = 0.0

    def prepare_dataset(self):
        """Prepare dataset for training with advanced config transforms."""
        import os
        from torchvision import datasets

        print("\n" + "="*80)
        print("DATA AUGMENTATION CONFIGURATION")
        print("="*80)
        print(f"[CONFIG] Dataset Path: {self.dataset_config.dataset_path}")

        # Build transforms using advanced config (or defaults)
        train_transform = self.build_train_transforms()
        val_transform = self.build_val_transforms()

        # Print augmentation details
        if self.training_config.advanced_config and 'augmentation' in self.training_config.advanced_config:
            aug_config = self.training_config.advanced_config['augmentation']
            print(f"[CONFIG] Augmentation Enabled: {aug_config.get('enabled', False)}")
            if aug_config.get('enabled'):
                print(f"[CONFIG] Active Augmentations:")
                if aug_config.get('random_flip'):
                    print(f"         - Random Horizontal Flip (p={aug_config.get('random_flip_prob', 0.5)})")
                if aug_config.get('random_rotation'):
                    print(f"         - Random Rotation (degrees={aug_config.get('rotation_degrees', 15)})")
                if aug_config.get('random_crop'):
                    print(f"         - Random Crop")
                if aug_config.get('color_jitter'):
                    print(f"         - Color Jitter (brightness={aug_config.get('brightness', 0.2)}, contrast={aug_config.get('contrast', 0.2)})")
                if aug_config.get('random_erasing'):
                    print(f"         - Random Erasing (p={aug_config.get('random_erasing_prob', 0.5)})")
                if aug_config.get('mixup'):
                    print(f"         - Mixup (alpha={aug_config.get('mixup_alpha', 0.2)})")
                if aug_config.get('cutmix'):
                    print(f"         - CutMix (alpha={aug_config.get('cutmix_alpha', 1.0)})")
        else:
            print(f"[CONFIG] Using default augmentation transforms")

        print(f"\n[CONFIG] Train Transform Pipeline:")
        for idx, transform in enumerate(train_transform.transforms):
            print(f"         {idx+1}. {transform.__class__.__name__}")

        print(f"\n[CONFIG] Validation Transform Pipeline:")
        for idx, transform in enumerate(val_transform.transforms):
            print(f"         {idx+1}. {transform.__class__.__name__}")

        # Create datasets
        train_dir = os.path.join(self.dataset_config.dataset_path, "train")
        val_dir = os.path.join(self.dataset_config.dataset_path, "val")

        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")
        if not os.path.exists(val_dir):
            raise ValueError(f"Validation directory not found: {val_dir}")

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

        num_classes = len(train_dataset.classes)
        print(f"Detected classes: {num_classes}")

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

    def train_epoch(self, epoch: int) -> MetricsResult:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Print current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"\n[EPOCH {epoch + 1}] Learning Rate: {current_lr:.6f}")

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item()

            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return MetricsResult(
            epoch=epoch,
            step=epoch * len(self.train_loader),
            train_loss=avg_loss,
            metrics={
                'train_accuracy': accuracy,
            }
        )

    def validate(self, epoch: int) -> MetricsResult:
        """Run validation with comprehensive metrics calculation."""
        import numpy as np
        from validators import ValidationMetricsCalculator, TaskType as ValidatorTaskType

        self.model.eval()
        running_loss = 0.0

        # Collect all predictions and labels for comprehensive metrics
        all_predictions = []
        all_labels = []
        all_probabilities = []

        # Collect per-image results for database storage
        image_results = []
        image_index = 0

        # Get image paths if available from dataset
        image_paths = None
        dataset = self.val_loader.dataset
        if hasattr(dataset, 'samples'):
            # ImageFolder or similar - samples is list of (path, label)
            image_paths = [path for path, _ in dataset.samples]
            print(f"[Validation] Found {len(image_paths)} image paths from dataset.samples")
        elif hasattr(dataset, 'imgs'):
            # Some datasets use imgs instead of samples
            image_paths = [path for path, _ in dataset.imgs]
            print(f"[Validation] Found {len(image_paths)} image paths from dataset.imgs")
        elif hasattr(dataset, 'dataset'):
            # Subset wrapper - check underlying dataset
            underlying = dataset.dataset
            if hasattr(underlying, 'samples'):
                image_paths = [underlying.samples[i][0] for i in dataset.indices]
                print(f"[Validation] Found {len(image_paths)} image paths from Subset.dataset.samples")
            elif hasattr(underlying, 'imgs'):
                image_paths = [underlying.imgs[i][0] for i in dataset.indices]
                print(f"[Validation] Found {len(image_paths)} image paths from Subset.dataset.imgs")

        if not image_paths:
            print(f"[Validation] WARNING: Could not extract image paths from dataset, will use placeholders")

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                # Collect for metrics calculation
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())

                # Collect per-image results
                batch_probs = probabilities.cpu().numpy()
                batch_preds = predicted.cpu().numpy()
                batch_labels = targets.cpu().numpy()

                for i in range(len(targets)):
                    # Get actual image path and name if available
                    image_path = None
                    image_name = f'image_{image_index}'

                    if image_paths and image_index < len(image_paths):
                        from pathlib import Path
                        full_path = image_paths[image_index]
                        image_path = full_path
                        image_name = Path(full_path).name

                    # Get top-5 predictions with confidence scores
                    top5_indices = batch_probs[i].argsort()[-5:][::-1]  # Top 5 class indices
                    top5_predictions = [
                        {
                            'label_id': int(idx),
                            'confidence': float(batch_probs[i][idx])
                        }
                        for idx in top5_indices
                    ]

                    image_results.append({
                        'image_index': image_index,
                        'image_name': image_name,
                        'image_path': image_path,
                        'true_label_id': int(batch_labels[i]),
                        'predicted_label_id': int(batch_preds[i]),
                        'confidence': float(batch_probs[i][batch_preds[i]]),
                        'top5_predictions': top5_predictions,
                        'is_correct': batch_preds[i] == batch_labels[i]
                    })
                    image_index += 1

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.vstack(all_probabilities)

        avg_loss = running_loss / len(self.val_loader)

        # Get class names from dataset if available
        class_names = None
        if hasattr(self.val_loader.dataset, 'classes'):
            class_names = self.val_loader.dataset.classes
        elif hasattr(self.val_loader.dataset, 'dataset') and hasattr(self.val_loader.dataset.dataset, 'classes'):
            class_names = self.val_loader.dataset.dataset.classes

        # Compute comprehensive validation metrics using ValidationMetricsCalculator
        validation_metrics = ValidationMetricsCalculator.compute_metrics(
            task_type=ValidatorTaskType.CLASSIFICATION,
            predictions=all_predictions,
            labels=all_labels,
            class_names=class_names,
            loss=avg_loss,
            probabilities=all_probabilities
        )

        # Save validation result to database
        validation_result_id = self._save_validation_result(epoch, validation_metrics)

        # Save per-image results to database
        if validation_result_id:
            self._save_validation_image_results(validation_result_id, epoch, image_results, class_names)

        # Extract metrics for return
        clf_metrics = validation_metrics.classification
        accuracy = clf_metrics.accuracy * 100.0  # Convert to percentage for consistency

        # Update scheduler if it exists
        if self.scheduler:
            old_lr = self.optimizer.param_groups[0]['lr']
            # ReduceLROnPlateau needs a metric value
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                # Other schedulers are stepped per epoch
                self.scheduler.step()

            # Check if learning rate changed
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"[SCHEDULER] Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")

        # Track best accuracy
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy

        # Build metrics dict for return
        metrics_dict = {
            'val_accuracy': accuracy,
            'val_precision': clf_metrics.precision * 100.0,
            'val_recall': clf_metrics.recall * 100.0,
            'val_f1_score': clf_metrics.f1_score * 100.0,
            'best_val_accuracy': self.best_val_acc,
        }

        if clf_metrics.top5_accuracy is not None:
            metrics_dict['val_top5_accuracy'] = clf_metrics.top5_accuracy * 100.0

        return MetricsResult(
            epoch=epoch,
            step=epoch * len(self.val_loader),
            train_loss=avg_loss,  # Using train_loss field for val_loss
            metrics=metrics_dict
        )

    def save_checkpoint(self, epoch: int, metrics: MetricsResult) -> str:
        """Save model checkpoint."""
        os.makedirs(self.output_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            self.output_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': {
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                **metrics.metrics
            }
        }

        # Add scheduler state if it exists
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        # Also save best model
        if metrics.metrics.get('val_accuracy', 0) == self.best_val_acc:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(self.model.state_dict(), best_path)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True) -> int:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            resume_training: If True, restore optimizer and scheduler states for resuming training

        Returns:
            Epoch number from the checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\n{'='*80}")
        print(f"LOADING CHECKPOINT")
        print(f"{'='*80}")
        print(f"[CHECKPOINT] Path: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[CHECKPOINT] Restored model state from epoch {checkpoint['epoch']}")

        if resume_training:
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"[CHECKPOINT] Restored optimizer state")

            # Load scheduler state if it exists
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"[CHECKPOINT] Restored scheduler state")

            # Update best_val_acc if available
            if 'metrics' in checkpoint and 'best_val_accuracy' in checkpoint['metrics']:
                self.best_val_acc = checkpoint['metrics']['best_val_accuracy']
                print(f"[CHECKPOINT] Restored best validation accuracy: {self.best_val_acc:.2f}%")
        else:
            print("[CHECKPOINT] Only model weights loaded (optimizer and scheduler states not restored)")

        print(f"{'='*80}\n")
        return checkpoint['epoch']
