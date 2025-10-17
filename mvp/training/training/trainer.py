"""Training loop implementation."""

import json
import os
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer for image classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
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
            pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                running_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss / (batch_idx + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def train(self, num_epochs: int, output_dir: str) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            output_dir: Directory to save checkpoints

        Returns:
            Dictionary with final training results
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"[TRAIN_START] Starting training for {num_epochs} epochs")
        print(f"[DEVICE] Using device: {self.device}")
        print(f"[MODEL] Total parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update learning rate scheduler
            self.scheduler.step(val_metrics["loss"])

            epoch_time = time.time() - epoch_start_time

            # Print metrics in JSON format for easy parsing
            metrics_json = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time,
            }

            # This line is parsed by backend
            print(f"[METRICS] {json.dumps(metrics_json)}")

            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                checkpoint_path = os.path.join(output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_metrics["accuracy"],
                    'val_loss': val_metrics["loss"],
                }, checkpoint_path)
                print(f"[CHECKPOINT] Saved best model to {checkpoint_path}")

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]

        print(f"[TRAIN_END] Training completed")
        print(f"[RESULT] Best validation accuracy: {self.best_val_acc:.2f}%")

        return {
            "best_val_accuracy": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "final_checkpoint": os.path.join(output_dir, "best_model.pth"),
        }
