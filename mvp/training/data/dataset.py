"""Dataset loader for image classification."""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or validation.

    Args:
        train: Whether this is for training (includes augmentation)

    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_dataloaders(
    dataset_path: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and validation dataloaders.

    Expected directory structure:
    dataset_path/
        train/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
        val/
            class1/
                img1.jpg
            class2/
                img1.jpg

    Args:
        dataset_path: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")

    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory not found: {val_dir}")

    # Create datasets
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=get_transforms(train=True)
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=get_transforms(train=False)
    )

    # Get number of classes
    num_classes = len(train_dataset.classes)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


def get_dataset_info(dataset_path: str) -> dict:
    """
    Get dataset information.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Dictionary with dataset information
    """
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")

    info = {
        "train_exists": os.path.exists(train_dir),
        "val_exists": os.path.exists(val_dir),
        "train_classes": [],
        "val_classes": [],
        "train_samples": 0,
        "val_samples": 0,
    }

    if info["train_exists"]:
        train_dataset = datasets.ImageFolder(train_dir)
        info["train_classes"] = train_dataset.classes
        info["train_samples"] = len(train_dataset)

    if info["val_exists"]:
        val_dataset = datasets.ImageFolder(val_dir)
        info["val_classes"] = val_dataset.classes
        info["val_samples"] = len(val_dataset)

    return info
