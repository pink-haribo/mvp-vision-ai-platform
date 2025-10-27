"""Timm adapter for image classification."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from .base import TrainingAdapter, MetricsResult, TaskType, DatasetFormat


class TimmAdapter(TrainingAdapter):
    """
    Adapter for timm (PyTorch Image Models).

    Supported tasks:
    - Image Classification (ResNet, EfficientNet, ViT, etc.)
    """

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
        """Run validation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total

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

        return MetricsResult(
            epoch=epoch,
            step=epoch * len(self.val_loader),
            train_loss=avg_loss,  # Using train_loss field for val_loss
            metrics={
                'val_accuracy': accuracy,
                'best_val_accuracy': self.best_val_acc,
            }
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
