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

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()

        if self.training_config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate
            )
        elif self.training_config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config.optimizer}")

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        self.best_val_acc = 0.0

    def prepare_dataset(self):
        """Prepare dataset for training."""
        from data.dataset import create_dataloaders

        print(f"Loading dataset from: {self.dataset_config.dataset_path}")

        self.train_loader, self.val_loader, num_classes = create_dataloaders(
            dataset_path=self.dataset_config.dataset_path,
            batch_size=self.training_config.batch_size,
            num_workers=4
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Detected classes: {num_classes}")

    def train_epoch(self, epoch: int) -> MetricsResult:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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

        # Update scheduler
        self.scheduler.step(avg_loss)

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

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': {
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                **metrics.metrics
            }
        }, checkpoint_path)

        # Also save best model
        if metrics.metrics.get('val_accuracy', 0) == self.best_val_acc:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(self.model.state_dict(), best_path)

        return checkpoint_path
