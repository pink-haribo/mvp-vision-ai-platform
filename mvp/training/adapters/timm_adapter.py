"""Timm adapter for image classification."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional, Dict, Any

from platform_sdk import TrainingAdapter, MetricsResult, TaskType, DatasetFormat, ConfigSchema, ConfigField


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

        import torch.nn as nn  # Import at function level to avoid scope issues

        # If num_classes not provided, detect from dataset first
        if self.model_config.num_classes is None:
            print("[INFO] num_classes not provided, loading dataset to auto-detect...")
            self.prepare_dataset()  # This will load dataset and populate self.train_loader

            # Get num_classes from loaded dataset
            if hasattr(self, 'train_loader') and self.train_loader:
                dataset = self.train_loader.dataset
                if hasattr(dataset, 'classes'):
                    self.model_config.num_classes = len(dataset.classes)
                elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
                    # For Subset, access underlying dataset
                    self.model_config.num_classes = len(dataset.dataset.classes)
                else:
                    raise ValueError("Cannot determine num_classes from dataset")

                print(f"[INFO] Auto-detected num_classes: {self.model_config.num_classes}")
            else:
                raise ValueError("Failed to load dataset for num_classes detection")

        print("\n" + "="*80)
        print("MODEL INITIALIZATION")
        print("="*80)
        print(f"Loading timm model: {self.model_config.model_name}")
        print(f"[CONFIG] Requested num_classes: {self.model_config.num_classes}")
        print(f"[CONFIG] Pretrained: {self.model_config.pretrained}")

        # First try: Load with num_classes parameter
        self.model = timm.create_model(
            self.model_config.model_name,
            pretrained=self.model_config.pretrained,
            num_classes=self.model_config.num_classes
        )

        # Verify model output size
        def get_actual_num_classes(model):
            """Helper to get actual output classes from model"""
            if hasattr(model, 'num_classes'):
                return model.num_classes
            elif hasattr(model, 'get_classifier'):
                classifier = model.get_classifier()
                if hasattr(classifier, 'out_features'):
                    return classifier.out_features
                elif hasattr(classifier, 'weight'):
                    return classifier.weight.shape[0]
            # Try to get from fc/head layer
            for name in ['fc', 'head', 'classifier']:
                if hasattr(model, name):
                    layer = getattr(model, name)
                    if hasattr(layer, 'out_features'):
                        return layer.out_features
                    elif hasattr(layer, 'weight'):
                        return layer.weight.shape[0]
            return None

        actual_num_classes = get_actual_num_classes(self.model)
        print(f"[VERIFY] Model actual output classes: {actual_num_classes}")

        if actual_num_classes != self.model_config.num_classes:
            print(f"[WARNING] Model output classes ({actual_num_classes}) != requested ({self.model_config.num_classes})")
            print(f"[FIX] Force reset classifier to {self.model_config.num_classes} classes...")

            # Method 1: Try reset_classifier (timm standard method)
            if hasattr(self.model, 'reset_classifier'):
                self.model.reset_classifier(self.model_config.num_classes)
                print(f"[FIX] Called reset_classifier({self.model_config.num_classes})")
            else:
                print(f"[WARNING] Model does not have reset_classifier method")

            # Method 2: Directly replace classifier layer
            if hasattr(self.model, 'get_classifier'):
                old_classifier = self.model.get_classifier()
                if hasattr(old_classifier, 'in_features'):
                    in_features = old_classifier.in_features
                    new_classifier = nn.Linear(in_features, self.model_config.num_classes)
                    # Try to set new classifier
                    for name in ['fc', 'head', 'classifier']:
                        if hasattr(self.model, name):
                            setattr(self.model, name, new_classifier)
                            print(f"[FIX] Replaced model.{name} with Linear({in_features}, {self.model_config.num_classes})")
                            break

            # Verify again
            new_num_classes = get_actual_num_classes(self.model)
            print(f"[VERIFY] After fix: {new_num_classes} classes")

            if new_num_classes != self.model_config.num_classes:
                print(f"[ERROR] Failed to set num_classes!")
                print(f"[ERROR] Expected {self.model_config.num_classes}, got {new_num_classes}")
                raise ValueError(f"Cannot set model to {self.model_config.num_classes} classes")
            else:
                print(f"[SUCCESS] Model successfully configured with {self.model_config.num_classes} classes")
        else:
            print(f"[SUCCESS] Model correctly initialized with {self.model_config.num_classes} classes")

        print("="*80 + "\n")

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
        # Skip if already loaded (avoids double loading when called from prepare_model)
        if hasattr(self, 'train_loader') and self.train_loader is not None:
            print("[INFO] Dataset already loaded, skipping...")
            return

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

        # Create datasets based on format
        dataset_format = self.dataset_config.format.value.lower()

        if dataset_format == "dice":
            # DICE format: Use text-file-based split (no file copy)
            print(f"[prepare_dataset] Dataset format: DICE (text-file split)")

            from converters.dice_split_generator import generate_dice_split
            from converters.text_file_image_dataset import create_text_file_datasets

            # Check if splits already exist
            splits_dir = os.path.join(self.dataset_config.dataset_path, "splits")
            train_txt = os.path.join(splits_dir, "train.txt")

            if not os.path.exists(train_txt):
                print(f"[prepare_dataset] Generating split files...")
                generate_dice_split(
                    dice_root=self.dataset_config.dataset_path,
                    split_ratio=0.8,
                    split_strategy="stratified",
                    seed=42
                )
            else:
                print(f"[prepare_dataset] Using existing split files: {splits_dir}")

            # Load datasets using text file splits
            train_dataset, val_dataset = create_text_file_datasets(
                dataset_root=self.dataset_config.dataset_path,
                train_transform=train_transform,
                val_transform=val_transform,
                splits_dir="splits"
            )

        else:
            # ImageFolder format: Use directory structure
            print(f"[prepare_dataset] Dataset format: ImageFolder")

            train_dir = os.path.join(self.dataset_config.dataset_path, "train")
            val_dir = os.path.join(self.dataset_config.dataset_path, "val")

            # Check for training directory
            if not os.path.exists(train_dir):
                # Maybe all images are directly in dataset_path (no train subfolder)
                if os.path.isdir(self.dataset_config.dataset_path):
                    train_dir = self.dataset_config.dataset_path
                    print(f"[prepare_dataset] No 'train' folder found, using dataset root: {train_dir}")
                else:
                    raise ValueError(f"Training directory not found: {train_dir}")

            # Check if validation directory exists
            if not os.path.exists(val_dir):
                print(f"[prepare_dataset] Val folder not found, auto-splitting train data...")
                print(f"[prepare_dataset] Splitting dataset with ratio 0.8 (train) / 0.2 (val)")

                # Load full dataset from train directory
                full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

                # Calculate split sizes
                total_size = len(full_dataset)
                train_size = int(0.8 * total_size)
                val_size = total_size - train_size

                print(f"[prepare_dataset] Total images: {total_size}")
                print(f"[prepare_dataset] Train: {train_size}, Val: {val_size}")

                # Split dataset with fixed seed for reproducibility
                import torch
                generator = torch.Generator().manual_seed(42)
                train_dataset, val_subset = torch.utils.data.random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=generator
                )

                # Create validation dataset with val_transform
                # We need to create a new dataset with the same subset but different transform
                val_dataset = datasets.ImageFolder(train_dir, transform=val_transform)
                val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)

                print(f"[prepare_dataset] Auto-split completed successfully")
            else:
                # Both train and val folders exist
                print(f"[prepare_dataset] Using existing train/val split")
                train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
                val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

        # Get number of classes from train_dataset
        if hasattr(train_dataset, 'classes'):
            num_classes = len(train_dataset.classes)
        elif hasattr(train_dataset, 'dataset'):
            # For Subset, access the underlying dataset
            num_classes = len(train_dataset.dataset.classes)
        else:
            raise ValueError("Cannot determine number of classes from dataset")

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

            # Debug: Check output shape on first batch
            if batch_idx == 0 and epoch == 0:
                print(f"\n[DEBUG] Training - First batch:")
                print(f"  - Input shape: {inputs.shape}")
                print(f"  - Output shape: {outputs.shape}")
                print(f"  - Target shape: {targets.shape}, range: [{targets.min().item()}, {targets.max().item()}]")
                print(f"  - Model output classes: {outputs.shape[1]}")

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
            task_type=ValidatorTaskType.IMAGE_CLASSIFICATION,
            predictions=all_predictions,
            labels=all_labels,
            class_names=class_names,
            loss=avg_loss,
            probabilities=all_probabilities
        )

        # Get checkpoint path for this epoch (use last.pt as it's the most recent)
        # Match YOLO structure: {output_dir}/job_{job_id}/weights/
        checkpoint_dir = os.path.join(self.output_dir, f"job_{self.job_id}", "weights")
        checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
        if not os.path.exists(checkpoint_path):
            # Fallback to best.pt
            best_path = os.path.join(checkpoint_dir, "best.pt")
            if os.path.exists(best_path):
                checkpoint_path = best_path
            else:
                checkpoint_path = None

        # Save validation result to database
        validation_result_id = self._save_validation_result(epoch, validation_metrics, checkpoint_path)

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
        """
        Save model checkpoint.

        Saves best.pt and last.pt like ultralytics for consistency.
        - last.pt: Most recent checkpoint (updated every epoch)
        - best.pt: Best checkpoint based on validation accuracy

        Storage structure matches YOLO: {output_dir}/job_{job_id}/weights/best.pt
        """
        # Create checkpoint directory matching YOLO structure
        checkpoint_dir = os.path.join(self.output_dir, f"job_{self.job_id}", "weights")
        os.makedirs(checkpoint_dir, exist_ok=True)

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

        # Always save last.pt (most recent checkpoint)
        last_path = os.path.join(checkpoint_dir, "last.pt")
        torch.save(checkpoint, last_path)
        print(f"[CHECKPOINT] Saved last.pt (epoch {epoch})")

        # Save best.pt if this is the best model so far
        current_accuracy = metrics.metrics.get('val_accuracy', 0)
        if current_accuracy == self.best_val_acc:
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"[CHECKPOINT] Saved best.pt (accuracy: {current_accuracy:.2f}%)")

        # Return path to best checkpoint for compatibility
        best_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(best_path):
            return best_path
        else:
            return last_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        inference_mode: bool = True,
        device: Optional[str] = None
    ) -> None:
        """
        Load checkpoint for inference or training resume.

        Args:
            checkpoint_path: Path to checkpoint file
            inference_mode: If True, load for inference (eval mode, no optimizer)
                           If False, load for training resume (restore full state)
            device: Device to load model on ('cuda', 'cpu'), auto-detect if None
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Determine device
        if device is None:
            device = self.device if hasattr(self, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n{'='*80}")
        print(f"LOADING CHECKPOINT")
        print(f"{'='*80}")
        print(f"[CHECKPOINT] Path: {checkpoint_path}")
        print(f"[CHECKPOINT] Mode: {'Inference' if inference_mode else 'Training Resume'}")
        print(f"[CHECKPOINT] Device: {device}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load model state
        if 'model_state_dict' in checkpoint:
            # Load with strict=False to handle classifier name mismatches
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            print(f"[CHECKPOINT] Restored model state from epoch {checkpoint.get('epoch', 'unknown')}")
            if missing_keys:
                print(f"[CHECKPOINT] Missing keys (expected for classifier): {missing_keys}")
            if unexpected_keys:
                print(f"[CHECKPOINT] Unexpected keys (ignored): {unexpected_keys}")
        else:
            # Handle checkpoints that are just state_dict (e.g., best_model.pt)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            print(f"[CHECKPOINT] Restored model state (simple format)")
            if missing_keys:
                print(f"[CHECKPOINT] Missing keys (expected for classifier): {missing_keys}")
            if unexpected_keys:
                print(f"[CHECKPOINT] Unexpected keys (ignored): {unexpected_keys}")

        # Set model to appropriate mode
        if inference_mode:
            self.model.eval()
            print(f"[CHECKPOINT] Model set to eval mode")
        else:
            # Training resume - restore optimizer and scheduler
            if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"[CHECKPOINT] Restored optimizer state")

            if hasattr(self, 'scheduler') and self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"[CHECKPOINT] Restored scheduler state")

            # Update best_val_acc if available
            if 'metrics' in checkpoint and 'best_val_accuracy' in checkpoint['metrics']:
                self.best_val_acc = checkpoint['metrics']['best_val_accuracy']
                print(f"[CHECKPOINT] Restored best validation accuracy: {self.best_val_acc:.2f}%")

            print(f"[CHECKPOINT] Training state restored")

        print(f"{'='*80}\n")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess single image for inference.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor with shape (1, 3, H, W)
        """
        from PIL import Image

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply validation transforms
        if not hasattr(self, 'val_transforms') or self.val_transforms is None:
            self.val_transforms = self.build_val_transforms()

        tensor = self.val_transforms(image)

        # Add batch dimension
        return tensor.unsqueeze(0)

    def infer_single(self, image_path: str) -> 'InferenceResult':
        """
        Run inference on single image.

        Args:
            image_path: Path to image file

        Returns:
            InferenceResult with classification predictions
        """
        import time
        from pathlib import Path
        from platform_sdk import InferenceResult, TaskType

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

        # Get class name
        if hasattr(self, 'class_names') and self.class_names:
            predicted_label = self.class_names[pred_id]
        else:
            predicted_label = str(pred_id)

        # Top-5 predictions
        num_classes = len(self.class_names) if hasattr(self, 'class_names') and self.class_names else output.size(1)
        top_k = min(5, num_classes)
        top5_probs, top5_ids = torch.topk(probs, top_k, dim=1)

        top5_predictions = []
        for i in range(top_k):
            label_id = int(top5_ids[0, i].item())
            if hasattr(self, 'class_names') and self.class_names:
                label = self.class_names[label_id]
            else:
                label = str(label_id)

            top5_predictions.append({
                'label_id': label_id,
                'label': label,
                'confidence': float(top5_probs[0, i].item())
            })

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
