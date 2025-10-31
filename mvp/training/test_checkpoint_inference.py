#!/usr/bin/env python3
"""
Standalone checkpoint inference test.

Tests that trained checkpoints can be loaded and used for inference:
- TIMM classification checkpoint
- YOLO detection checkpoint
- YOLO segmentation checkpoint

Usage:
    python test_checkpoint_inference.py
"""

import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def create_classification_dataset(root_dir):
    """Create a tiny classification dataset."""
    print(f"[SETUP] Creating classification dataset...")

    for split in ["train", "val"]:
        split_dir = Path(root_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for class_idx in range(2):
            class_dir = split_dir / f"class_{class_idx}"
            class_dir.mkdir(exist_ok=True)

            num_images = 5 if split == "train" else 2
            for img_idx in range(num_images):
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = class_dir / f"img_{img_idx}.jpg"
                img.save(img_path)


def create_detection_dataset(root_dir):
    """Create a tiny YOLO detection dataset."""
    print(f"[SETUP] Creating detection dataset...")

    root = Path(root_dir)
    images_dir = root / "images"
    labels_dir = root / "labels"

    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

        num_images = 5 if split == "train" else 2
        for img_idx in range(num_images):
            img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = images_dir / split / f"img_{img_idx}.jpg"
            img.save(img_path)

            label_path = labels_dir / split / f"img_{img_idx}.txt"
            with open(label_path, "w") as f:
                num_boxes = np.random.randint(1, 3)
                for _ in range(num_boxes):
                    class_id = np.random.randint(0, 2)
                    x = np.random.uniform(0.2, 0.8)
                    y = np.random.uniform(0.2, 0.8)
                    w = np.random.uniform(0.1, 0.3)
                    h = np.random.uniform(0.1, 0.3)
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    data_yaml = root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 2\n")
        f.write("names: ['class_0', 'class_1']\n")


def test_classification_checkpoint_inference():
    """Test TIMM classification checkpoint inference."""
    import timm
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    print("\n" + "=" * 80)
    print("TEST 1: TIMM CLASSIFICATION CHECKPOINT INFERENCE")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "dataset"
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Create dataset
        create_classification_dataset(dataset_dir)

        # Prepare dataset
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(
            dataset_dir / "train",
            transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        # Train for 1 epoch
        print(f"\n[TRAIN] Training ResNet18 for 1 epoch...")
        model = timm.create_model("resnet18", pretrained=False, num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")
        model.to(device)

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        checkpoint_path = checkpoint_dir / "resnet18_epoch1.pth"
        torch.save({
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        print(f"[SAVE] Checkpoint saved: {checkpoint_path.name}")

        # Test: Load checkpoint and run inference
        print(f"\n[INFERENCE] Loading checkpoint for inference...")

        # Create new model
        inference_model = timm.create_model("resnet18", pretrained=False, num_classes=2)
        inference_model.to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        inference_model.load_state_dict(checkpoint['model_state_dict'])
        inference_model.eval()

        print(f"  ✓ Checkpoint loaded successfully")

        # Run inference on a test image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        test_tensor = transform(test_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = inference_model(test_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        print(f"  ✓ Inference successful")
        print(f"    Predicted class: {predicted_class}")
        print(f"    Confidence: {confidence:.4f}")

        # Verify output shape
        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
        assert 0 <= predicted_class < 2, f"Invalid predicted class: {predicted_class}"
        assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"

        print(f"\n✅ Classification checkpoint inference test passed!")


def test_detection_checkpoint_inference():
    """Test YOLO detection checkpoint inference."""
    from ultralytics import YOLO

    print("\n" + "=" * 80)
    print("TEST 2: YOLO DETECTION CHECKPOINT INFERENCE")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "dataset"
        output_dir = Path(tmpdir) / "output"

        # Create dataset
        create_detection_dataset(dataset_dir)

        # Train for 1 epoch
        print(f"\n[TRAIN] Training YOLO11n for 1 epoch...")
        model = YOLO("yolo11n.pt")

        project_dir = output_dir / "yolo_project"
        results = model.train(
            data=str(dataset_dir / "data.yaml"),
            epochs=1,
            imgsz=64,
            batch=2,
            patience=0,
            save=True,
            val=False,  # Skip validation for speed
            plots=False,
            verbose=False,
            project=str(project_dir),
            name="train",
            exist_ok=True
        )

        checkpoint_path = project_dir / "train" / "weights" / "last.pt"
        print(f"[SAVE] Checkpoint saved: {checkpoint_path}")

        # Test: Load checkpoint and run inference
        print(f"\n[INFERENCE] Loading checkpoint for inference...")

        # Load checkpoint
        inference_model = YOLO(str(checkpoint_path))
        print(f"  ✓ Checkpoint loaded successfully")

        # Create test image
        test_image_path = Path(tmpdir) / "test_image.jpg"
        test_image = Image.fromarray(
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        )
        test_image.save(test_image_path)

        # Run inference
        results = inference_model(str(test_image_path), verbose=False)

        print(f"  ✓ Inference successful")

        # Verify results
        assert results is not None, "Inference results are None"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        result = results[0]
        assert hasattr(result, "boxes"), "Result missing 'boxes' attribute"

        num_detections = len(result.boxes)
        print(f"    Detected {num_detections} objects")

        if num_detections > 0:
            boxes = result.boxes
            assert hasattr(boxes, "xyxy"), "Boxes missing 'xyxy' attribute"
            assert hasattr(boxes, "conf"), "Boxes missing 'conf' attribute"
            assert hasattr(boxes, "cls"), "Boxes missing 'cls' attribute"

            print(f"    First detection confidence: {boxes.conf[0].item():.4f}")

        print(f"\n✅ Detection checkpoint inference test passed!")


def test_segmentation_checkpoint_inference():
    """Test YOLO segmentation checkpoint inference."""
    from ultralytics import YOLO

    print("\n" + "=" * 80)
    print("TEST 3: YOLO SEGMENTATION CHECKPOINT INFERENCE")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # We'll use a pretrained model instead of training (faster)
        print(f"\n[LOAD] Loading pretrained YOLO11n-seg model...")
        model = YOLO("yolo11n-seg.pt")

        # Save as checkpoint (simulating trained model)
        checkpoint_path = Path(tmpdir) / "yolo11n-seg_checkpoint.pt"
        model.save(checkpoint_path)
        print(f"[SAVE] Model saved to: {checkpoint_path.name}")

        # Test: Load checkpoint and run inference
        print(f"\n[INFERENCE] Loading checkpoint for inference...")

        # Load checkpoint
        inference_model = YOLO(str(checkpoint_path))
        print(f"  ✓ Checkpoint loaded successfully")

        # Create test image
        test_image_path = Path(tmpdir) / "test_image.jpg"
        test_image = Image.fromarray(
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        )
        test_image.save(test_image_path)

        # Run inference
        results = inference_model(str(test_image_path), verbose=False)

        print(f"  ✓ Inference successful")

        # Verify results
        assert results is not None, "Inference results are None"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        result = results[0]
        assert hasattr(result, "boxes"), "Result missing 'boxes' attribute"
        assert hasattr(result, "masks"), "Result missing 'masks' attribute"

        num_detections = len(result.boxes)
        print(f"    Detected {num_detections} objects")

        if num_detections > 0 and result.masks is not None:
            masks = result.masks
            assert hasattr(masks, "data"), "Masks missing 'data' attribute"
            print(f"    Generated {len(masks)} segmentation masks")
            print(f"    Mask shape: {masks.data[0].shape}")
        else:
            print(f"    No segmentation masks (valid for empty scene)")

        print(f"\n✅ Segmentation checkpoint inference test passed!")


def main():
    """Run all checkpoint inference tests."""
    print("\n" + "=" * 80)
    print("CHECKPOINT INFERENCE TESTS")
    print("=" * 80)
    print("\nTesting that trained checkpoints can be loaded and used for inference.")
    print("This verifies:")
    print("  - Checkpoint saving/loading works correctly")
    print("  - Models can run inference after loading from checkpoint")
    print("  - Output format is correct for each task type")

    try:
        # Test 1: Classification
        test_classification_checkpoint_inference()

        # Test 2: Detection
        test_detection_checkpoint_inference()

        # Test 3: Segmentation
        test_segmentation_checkpoint_inference()

        # Summary
        print("\n" + "=" * 80)
        print("✅ ALL CHECKPOINT INFERENCE TESTS PASSED")
        print("=" * 80)
        print("\nSuccessfully tested:")
        print("  ✓ TIMM classification checkpoint inference")
        print("  ✓ YOLO detection checkpoint inference")
        print("  ✓ YOLO segmentation checkpoint inference")
        print("\nAll checkpoints can be loaded and used for inference!")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
