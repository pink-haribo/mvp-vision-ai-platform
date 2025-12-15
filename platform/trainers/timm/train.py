#!/usr/bin/env python3
"""
Timm (PyTorch Image Models) Trainer

Simple CLI script for training image classification models using timm.
All observability (MLflow, Loki, Prometheus) is handled by Backend.

Usage:
    python train.py \\
        --job-id 123 \\
        --model-name resnet50 \\
        --dataset-s3-uri s3://bucket/datasets/abc-123/ \\
        --callback-url http://localhost:8000/api/v1/training \\
        --config '{"epochs": 50, "batch_size": 32, "learning_rate": 0.001}'

Environment Variables (alternative to CLI args):
    JOB_ID, MODEL_NAME, DATASET_S3_URI, CALLBACK_URL, CONFIG

Exit Codes:
    0 = Success
    1 = Training failure
    2 = Callback failure
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from trainer_sdk import ErrorType, TrainerSDK

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Timm Image Classification Trainer')

    parser.add_argument('--job-id', type=str, help='Training job ID')
    parser.add_argument('--model-name', type=str, help='Timm model name (e.g., resnet50)')
    parser.add_argument('--dataset-s3-uri', type=str, help='S3 URI to dataset')
    parser.add_argument('--callback-url', type=str, help='Backend API base URL')
    parser.add_argument('--config', type=str, help='Training config JSON string')
    parser.add_argument('--config-file', type=str, help='Path to training config JSON file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from args or environment"""
    # Priority: CLI args > env vars
    job_id = args.job_id or os.getenv('JOB_ID')
    model_name = args.model_name or os.getenv('MODEL_NAME')
    dataset_s3_uri = args.dataset_s3_uri or os.getenv('DATASET_S3_URI')
    callback_url = args.callback_url or os.getenv('CALLBACK_URL')

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    elif args.config:
        config = json.loads(args.config)
    elif os.getenv('CONFIG'):
        config = json.loads(os.getenv('CONFIG'))
    else:
        config = {}

    # Validate required fields
    if not all([job_id, model_name, dataset_s3_uri, callback_url]):
        raise ValueError("Missing required arguments: job_id, model_name, dataset_s3_uri, callback_url")

    # Set environment variables for SDK
    os.environ['JOB_ID'] = str(job_id)
    os.environ['CALLBACK_URL'] = callback_url

    return {
        'job_id': job_id,
        'model_name': model_name,
        'dataset_s3_uri': dataset_s3_uri,
        'callback_url': callback_url,
        'config': config
    }


def build_transforms(config: Dict[str, Any], is_train: bool = True):
    """Build data transforms for training or validation"""
    img_size = config.get('imgsz', config.get('image_size', 224))

    if is_train:
        transform_list = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        # Optional augmentations
        if config.get('color_jitter', False):
            transform_list.append(transforms.ColorJitter(
                brightness=config.get('brightness', 0.2),
                contrast=config.get('contrast', 0.2),
                saturation=config.get('saturation', 0.2),
                hue=config.get('hue', 0.1)
            ))

        if config.get('random_rotation', False):
            transform_list.append(transforms.RandomRotation(
                degrees=config.get('rotation_degrees', 15)
            ))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if config.get('random_erasing', False):
            transform_list.append(transforms.RandomErasing(p=0.5))

        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create timm model with specified number of classes"""
    logger.info(f"Creating model: {model_name} with {num_classes} classes")

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer based on config"""
    lr = config.get('learning_rate', config.get('lr', 0.001))
    weight_decay = config.get('weight_decay', 0.0001)
    optimizer_type = config.get('optimizer', 'adamw').lower()

    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], steps_per_epoch: int):
    """Create learning rate scheduler based on config"""
    scheduler_type = config.get('scheduler', 'cosine').lower()
    epochs = config.get('epochs', 100)
    warmup_epochs = config.get('warmup_epochs', 5)

    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs * steps_per_epoch,
            eta_min=config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('gamma', 0.1),
            patience=config.get('patience', 10)
        )
    else:
        return None


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    sdk: TrainerSDK
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            current_loss = running_loss / (len(pbar) if len(pbar) > 0 else 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save last checkpoint
    last_path = save_dir / 'last.pt'
    torch.save(checkpoint, last_path)
    logger.info(f"Saved checkpoint: {last_path}")

    # Save best checkpoint
    if is_best:
        best_path = save_dir / 'best.pt'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint: {best_path}")

    return last_path if not is_best else save_dir / 'best.pt'


def main():
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        # Load configuration
        cfg = load_config(args)
        job_id = cfg['job_id']
        model_name = cfg['model_name']
        dataset_s3_uri = cfg['dataset_s3_uri']
        config = cfg['config']

        logger.info(f"Starting training job {job_id}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Dataset: {dataset_s3_uri}")
        logger.info(f"Config: {json.dumps(config, indent=2)}")

        # Initialize SDK
        sdk = TrainerSDK()

        # Report training started
        sdk.report_started(
            model_name=model_name,
            config=config,
            dataset_uri=dataset_s3_uri
        )

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Download dataset
        logger.info("Downloading dataset...")
        dataset_path = sdk.download_dataset(dataset_s3_uri)
        logger.info(f"Dataset downloaded to: {dataset_path}")

        # Create output directory
        output_dir = Path(f"./outputs/job_{job_id}")
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Build transforms
        train_transform = build_transforms(config, is_train=True)
        val_transform = build_transforms(config, is_train=False)

        # Load datasets
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"

        if not train_dir.exists():
            train_dir = dataset_path
            logger.warning(f"No 'train' folder found, using dataset root")

        train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
        num_classes = len(train_dataset.classes)
        logger.info(f"Detected {num_classes} classes: {train_dataset.classes}")

        if val_dir.exists():
            val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
        else:
            # Auto-split
            logger.info("No 'val' folder found, auto-splitting 80/20...")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        # Create dataloaders
        batch_size = config.get('batch_size', config.get('batch', 32))
        num_workers = config.get('workers', 4)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Create model
        pretrained = config.get('pretrained', True)
        model = create_model(model_name, num_classes, pretrained=pretrained)
        model = model.to(device)

        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config, len(train_loader))

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training parameters
        epochs = config.get('epochs', 100)
        best_val_acc = 0.0

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, epoch, sdk
            )

            # Validate
            val_metrics = validate(model, val_loader, criterion, device, epoch)

            # Update scheduler if ReduceLROnPlateau
            if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics['learning_rate'] = optimizer.param_groups[0]['lr']

            # Check if best
            is_best = val_metrics['val_accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['val_accuracy']

            # Save checkpoint
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, metrics, weights_dir, is_best=is_best
            )

            # Report progress to backend
            sdk.report_progress(
                epoch=epoch,
                total_epochs=epochs,
                metrics=metrics,
                checkpoint_path=str(checkpoint_path) if is_best else None
            )

            logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
            )

        # Upload final checkpoint
        best_checkpoint_path = weights_dir / "best.pt"
        if best_checkpoint_path.exists():
            logger.info("Uploading best checkpoint...")
            checkpoint_s3_uri = sdk.upload_checkpoint(best_checkpoint_path)
            logger.info(f"Checkpoint uploaded to: {checkpoint_s3_uri}")

        # Report completion
        final_metrics = {
            'best_val_accuracy': best_val_acc,
            'final_train_loss': train_metrics['train_loss'],
            'final_val_loss': val_metrics['val_loss'],
        }

        sdk.report_completed(
            metrics=final_metrics,
            checkpoint_path=str(best_checkpoint_path),
            output_dir=str(output_dir)
        )

        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())

        try:
            sdk = TrainerSDK()
            sdk.report_failed(
                error_type=ErrorType.TRAINING_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
        except Exception as callback_error:
            logger.error(f"Failed to report error: {callback_error}")
            return 2

        return 1


if __name__ == "__main__":
    sys.exit(main())
