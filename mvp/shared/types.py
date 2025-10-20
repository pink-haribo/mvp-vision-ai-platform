"""Shared type definitions."""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration."""
    model: str
    task: str
    num_classes: int
    dataset_path: str
    output_dir: str
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "num_classes": self.num_classes,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }


@dataclass
class TrainingMetrics:
    """Training metrics."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
        }
