#!/usr/bin/env python3
"""
Standalone segmentation training test.

Tests YOLO11n-seg instance segmentation training to verify:
- Training runs without errors
- Checkpoints are saved
- Validation metrics work (mAP, masks)

Usage:
    python test_segmentation_training.py
"""

import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def create_segmentation_dataset(root_dir):
    """Create a tiny YOLO segmentation dataset."""
    print(f"\n[SETUP] Creating segmentation dataset at {root_dir}...")

    root = Path(root_dir)
    images_dir = root / "images"
    labels_dir = root / "labels"

    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

        num_images = 5 if split == "train" else 2
        for img_idx in range(num_images):
            # Create image
            img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = images_dir / split / f"img_{img_idx}.jpg"
            img.save(img_path)

            # Create YOLO segmentation label (class x1 y1 x2 y2 x3 y3 ...)
            label_path = labels_dir / split / f"img_{img_idx}.txt"
            with open(label_path, "w") as f:
                num_objects = np.random.randint(1, 3)
                for _ in range(num_objects):
                    class_id = np.random.randint(0, 2)

                    # Generate polygon points (normalized coordinates)
                    num_points = 6  # Simple hexagon
                    center_x = np.random.uniform(0.3, 0.7)
                    center_y = np.random.uniform(0.3, 0.7)
                    radius = np.random.uniform(0.1, 0.2)

                    points = []
                    for i in range(num_points):
                        angle = 2 * np.pi * i / num_points
                        x = center_x + radius * np.cos(angle)
                        y = center_y + radius * np.sin(angle)
                        # Clamp to [0, 1]
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        points.append(f"{x:.6f}")
                        points.append(f"{y:.6f}")

                    # Format: class_id x1 y1 x2 y2 x3 y3 ...
                    f.write(f"{class_id} " + " ".join(points) + "\n")

    # Create data.yaml
    data_yaml = root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 2\n")
        f.write("names: ['class_0', 'class_1']\n")

    print(f"[OK] Segmentation dataset created: Train=5 images, Val=2 images")


def test_yolo_segmentation_training():
    """Test YOLO11n-seg segmentation training."""
    from ultralytics import YOLO

    print("\n" + "=" * 80)
    print("TESTING YOLO11N-SEG SEGMENTATION TRAINING")
    print("=" * 80)

    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "dataset"
        output_dir = Path(tmpdir) / "output"

        # Create dataset
        create_segmentation_dataset(dataset_dir)

        # Train YOLO segmentation model
        print(f"\n[TRAINING] Starting YOLO11n-seg for 2 epochs...")
        model = YOLO("yolo11n-seg.pt")

        project_dir = output_dir / "yolo_project"

        results = model.train(
            data=str(dataset_dir / "data.yaml"),
            epochs=2,
            imgsz=64,
            batch=2,
            patience=0,
            save=True,
            save_period=1,
            val=True,
            plots=False,
            verbose=False,
            project=str(project_dir),
            name="train",
            exist_ok=True
        )

        print(f"[TRAINING] Completed")

        # Verify outputs
        run_dir = project_dir / "train"
        weights_dir = run_dir / "weights"

        print(f"\n[VERIFY] Checking outputs...")
        assert run_dir.exists(), f"Missing output directory: {run_dir}"
        assert weights_dir.exists(), "Missing weights directory"
        print(f"  ✓ Output directory: {run_dir}")

        # Check for checkpoints
        checkpoints = ["last.pt", "best.pt", "epoch1.pt"]
        for cp_name in checkpoints:
            cp = weights_dir / cp_name
            assert cp.exists(), f"Missing checkpoint: {cp_name}"
            print(f"  ✓ {cp_name}")

        print(f"  (Note: epoch2.pt = last.pt for final epoch)")

        # Load and validate
        loaded_model = YOLO(str(weights_dir / "best.pt"))
        print(f"\n[OK] Best model loaded successfully")

        # Run validation
        val_results = loaded_model.val(
            data=str(dataset_dir / "data.yaml"),
            batch=2,
            imgsz=64,
            verbose=False
        )

        assert val_results is not None, "Validation results are None"

        # Check metrics (boxes and masks)
        assert hasattr(val_results, 'box'), "Validation results missing box metrics"
        assert hasattr(val_results, 'seg'), "Validation results missing seg metrics"

        print(f"[OK] Validation completed")

        # Box metrics
        if hasattr(val_results.box, 'map50'):
            print(f"     Box mAP50: {val_results.box.map50:.4f}")

        # Segmentation metrics
        if hasattr(val_results.seg, 'map50'):
            print(f"     Seg mAP50: {val_results.seg.map50:.4f}")

        # Parse results.csv to check segmentation metrics
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            import csv
            print(f"\n[METRICS] Checking results.csv...")

            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [name.strip() for name in reader.fieldnames]

                last_row = None
                for row in reader:
                    last_row = row

                if last_row:
                    epoch = last_row.get('epoch', '').strip()
                    box_map50 = last_row.get('metrics/mAP50(B)', '').strip()
                    seg_map50 = last_row.get('metrics/mAP50(M)', '').strip()

                    print(f"  Final epoch: {epoch}")
                    print(f"  Box mAP50: {box_map50}")
                    print(f"  Seg mAP50: {seg_map50}")

                    # Verify segmentation metrics exist
                    assert 'metrics/mAP50(M)' in reader.fieldnames, \
                        "Segmentation mAP50(M) not found in results"
                    print(f"  ✓ Segmentation metrics present in results.csv")

        print("\n" + "=" * 80)
        print("✅ YOLO11N-SEG SEGMENTATION TRAINING TEST PASSED")
        print("=" * 80)
        print(f"\n✓ Training completed successfully")
        print(f"✓ Checkpoints saved: last.pt, best.pt, epoch1.pt")
        print(f"✓ Validation metrics computed (box + seg)")
        print(f"✓ Segmentation-specific metrics verified")

    return 0


if __name__ == "__main__":
    try:
        exit_code = test_yolo_segmentation_training()
        print(f"\n✅ All segmentation training tests passed!")
        exit(exit_code)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
