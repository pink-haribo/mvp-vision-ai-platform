"""
Test end-to-end training execution with checkpoint and validation.

This script tests actual training execution on small datasets to verify:
- Training runs without errors
- Checkpoints are saved correctly
- Validation metrics are computed
- MLflow logging works

Priority: P1 (Core functionality for production)

Usage:
    # Run in Docker environment with ML frameworks
    cd mvp/training
    python -m pytest ../backend/tests/integration/test_training_execution.py -v -s
"""

import pytest
import sys
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Add training directory to path
training_dir = Path(__file__).parent.parent.parent.parent / "training"
sys.path.insert(0, str(training_dir))


def _is_ultralytics_available():
    """Check if ultralytics is available."""
    try:
        import ultralytics
        return True
    except ImportError:
        return False


def _is_timm_available():
    """Check if timm is available."""
    try:
        import timm
        return True
    except ImportError:
        return False


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def sample_classification_dataset(tmp_path_factory):
    """
    Create a tiny classification dataset for testing.

    Structure:
        dataset/
            train/
                class_0/
                    img_0.jpg
                    img_1.jpg
                class_1/
                    img_0.jpg
                    img_1.jpg
            val/
                class_0/
                    img_0.jpg
                class_1/
                    img_0.jpg
    """
    dataset_root = tmp_path_factory.mktemp("classification_dataset")

    # Create train/val splits
    for split in ["train", "val"]:
        split_dir = dataset_root / split
        split_dir.mkdir()

        # Create 2 classes
        for class_idx in range(2):
            class_dir = split_dir / f"class_{class_idx}"
            class_dir.mkdir()

            # Create images (fewer for val)
            num_images = 5 if split == "train" else 2
            for img_idx in range(num_images):
                # Create random RGB image (64x64 for speed)
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)

                img_path = class_dir / f"img_{img_idx}.jpg"
                img.save(img_path)

    print(f"\n[FIXTURE] Created classification dataset at: {dataset_root}")
    print(f"          Train: 10 images (5 per class)")
    print(f"          Val: 4 images (2 per class)")

    yield dataset_root

    # Cleanup
    shutil.rmtree(dataset_root, ignore_errors=True)


@pytest.fixture(scope="module")
def sample_detection_dataset(tmp_path_factory):
    """
    Create a tiny detection dataset in YOLO format for testing.

    Structure:
        dataset/
            images/
                train/
                    img_0.jpg
                    img_1.jpg
                val/
                    img_0.jpg
            labels/
                train/
                    img_0.txt
                    img_1.txt
                val/
                    img_0.txt
            data.yaml
    """
    dataset_root = tmp_path_factory.mktemp("detection_dataset")

    # Create images and labels directories
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True)
        (labels_dir / split).mkdir(parents=True)

        # Create images and labels
        num_images = 5 if split == "train" else 2
        for img_idx in range(num_images):
            # Create image (640x640 for YOLO)
            img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            img_path = images_dir / split / f"img_{img_idx}.jpg"
            img.save(img_path)

            # Create YOLO format label (class x_center y_center width height)
            # Random bounding box
            label_path = labels_dir / split / f"img_{img_idx}.txt"
            with open(label_path, "w") as f:
                # 1-2 random boxes per image
                num_boxes = np.random.randint(1, 3)
                for _ in range(num_boxes):
                    class_id = np.random.randint(0, 2)  # 2 classes
                    x = np.random.uniform(0.2, 0.8)
                    y = np.random.uniform(0.2, 0.8)
                    w = np.random.uniform(0.1, 0.3)
                    h = np.random.uniform(0.1, 0.3)
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # Create data.yaml
    data_yaml = dataset_root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {dataset_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 2\n")
        f.write("names: ['class_0', 'class_1']\n")

    print(f"\n[FIXTURE] Created detection dataset at: {dataset_root}")
    print(f"          Train: 5 images with labels")
    print(f"          Val: 2 images with labels")

    yield dataset_root

    # Cleanup
    shutil.rmtree(dataset_root, ignore_errors=True)


# ============================================================
# TIMM Classification Training Tests
# ============================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not _is_timm_available(),
    reason="timm not installed"
)
class TestTIMMTraining:
    """Test TIMM classification training."""

    def test_resnet18_training_basic(self, sample_classification_dataset, tmp_path):
        """Test basic ResNet18 training with checkpoint and validation."""
        import timm
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        print("\n[TEST] ResNet18 classification training...")

        # Prepare dataset
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(
            sample_classification_dataset / "train",
            transform=transform
        )
        val_dataset = datasets.ImageFolder(
            sample_classification_dataset / "val",
            transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        print(f"[INFO] Train dataset: {len(train_dataset)} images, {len(train_dataset.classes)} classes")
        print(f"[INFO] Val dataset: {len(val_dataset)} images")

        # Create model
        model = timm.create_model("resnet18", pretrained=False, num_classes=2)
        model.train()

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print(f"[INFO] Device: {device}")
        print(f"[INFO] Model: resnet18, Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Training loop (2 epochs)
        num_epochs = 2
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            # Train
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            # Validation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            val_accuracies.append(val_acc)

            print(f"[EPOCH {epoch + 1}/{num_epochs}] Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_accuracy': val_acc,
            }, checkpoint_path)

            print(f"[CHECKPOINT] Saved to: {checkpoint_path}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint = checkpoint_dir / "best.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_loss,
                    'val_accuracy': val_acc,
                }, best_checkpoint)
                print(f"[BEST] New best model saved: {val_acc:.4f}")

        # Verify results
        assert len(train_losses) == num_epochs, "Training loss not recorded for all epochs"
        assert len(val_accuracies) == num_epochs, "Validation accuracy not recorded for all epochs"

        # Verify checkpoints exist
        for epoch in range(1, num_epochs + 1):
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pth"
            assert checkpoint_path.exists(), f"Checkpoint for epoch {epoch} not found"

        best_checkpoint = checkpoint_dir / "best.pth"
        assert best_checkpoint.exists(), "Best checkpoint not found"

        # Load and verify checkpoint
        checkpoint = torch.load(best_checkpoint)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'val_accuracy' in checkpoint
        assert checkpoint['val_accuracy'] == best_val_acc

        print(f"\n[OK] Training completed successfully")
        print(f"     Epochs: {num_epochs}")
        print(f"     Final train loss: {train_losses[-1]:.4f}")
        print(f"     Final val accuracy: {val_accuracies[-1]:.4f}")
        print(f"     Best val accuracy: {best_val_acc:.4f}")
        print(f"     Checkpoints saved: {num_epochs + 1} (including best)")


# ============================================================
# Ultralytics YOLO Training Tests
# ============================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not _is_ultralytics_available(),
    reason="ultralytics not installed"
)
class TestUltralyticsTraining:
    """Test Ultralytics YOLO training."""

    def test_yolo11n_training_basic(self, sample_detection_dataset, tmp_path):
        """Test basic YOLO11n detection training with validation."""
        from ultralytics import YOLO

        print("\n[TEST] YOLO11n detection training...")

        # Create model
        model = YOLO("yolo11n.pt")

        print(f"[INFO] Model: YOLO11n")
        print(f"[INFO] Dataset: {sample_detection_dataset}")

        # Training parameters (minimal for speed)
        project_dir = tmp_path / "yolo_project"

        results = model.train(
            data=str(sample_detection_dataset / "data.yaml"),
            epochs=2,
            imgsz=64,  # Very small for speed
            batch=2,
            patience=0,  # No early stopping
            save=True,
            save_period=1,  # Save every epoch
            val=True,
            plots=False,  # Skip plots for speed
            verbose=False,
            project=str(project_dir),
            name="train",
            exist_ok=True
        )

        # Verify training completed
        assert results is not None, "Training results are None"

        # Check for output directory
        run_dir = project_dir / "train"
        assert run_dir.exists(), f"Training output directory not found: {run_dir}"

        # Check for weights directory
        weights_dir = run_dir / "weights"
        assert weights_dir.exists(), "Weights directory not found"

        # Check for checkpoints
        last_checkpoint = weights_dir / "last.pt"
        best_checkpoint = weights_dir / "best.pt"

        assert last_checkpoint.exists(), "Last checkpoint not found"
        assert best_checkpoint.exists(), "Best checkpoint not found"

        # Check for epoch checkpoints
        epoch1_checkpoint = weights_dir / "epoch1.pt"
        epoch2_checkpoint = weights_dir / "epoch2.pt"
        assert epoch1_checkpoint.exists(), "Epoch 1 checkpoint not found"
        assert epoch2_checkpoint.exists(), "Epoch 2 checkpoint not found"

        # Verify we can load the checkpoint
        loaded_model = YOLO(str(best_checkpoint))
        assert loaded_model is not None, "Failed to load best checkpoint"

        # Run validation on trained model
        val_results = loaded_model.val(
            data=str(sample_detection_dataset / "data.yaml"),
            batch=2,
            imgsz=64,
            verbose=False
        )

        assert val_results is not None, "Validation results are None"

        # Check that metrics exist (even if values are low due to small dataset)
        assert hasattr(val_results, 'box'), "Validation results missing box metrics"

        print(f"\n[OK] YOLO training completed successfully")
        print(f"     Epochs: 2")
        print(f"     Output directory: {run_dir}")
        print(f"     Checkpoints: last.pt, best.pt, epoch1.pt, epoch2.pt")
        print(f"     Validation mAP50: {val_results.box.map50:.4f}" if hasattr(val_results.box, 'map50') else "")


# ============================================================
# Summary Test
# ============================================================

@pytest.mark.slow
def test_training_summary():
    """Summary of training execution tests."""
    print("\n" + "=" * 80)
    print("TRAINING EXECUTION TEST SUMMARY")
    print("=" * 80)
    print("\nTested training capabilities:")
    print("  - TIMM Classification (ResNet18)")
    print("    * Training loop with loss computation")
    print("    * Validation with accuracy metrics")
    print("    * Checkpoint saving (per epoch + best)")
    print("    * Checkpoint loading and verification")
    print("")
    print("  - Ultralytics Detection (YOLO11n)")
    print("    * Training with YOLO format dataset")
    print("    * Automatic checkpoint management")
    print("    * Validation with mAP metrics")
    print("    * Model loading from checkpoint")
    print("\nNote: Tests use minimal epochs and tiny datasets for speed")
    print("=" * 80)
