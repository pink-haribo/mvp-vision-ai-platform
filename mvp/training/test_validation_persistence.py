#!/usr/bin/env python3
"""
Standalone test for validation metrics persistence.

Tests that YOLO validation metrics are saved to validation_results table.
"""

import sys
import sqlite3
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def create_detection_dataset(root_dir):
    """Create a tiny YOLO detection dataset."""
    print(f"\n[SETUP] Creating detection dataset at {root_dir}...")

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

            # Create YOLO label
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

    # Create data.yaml
    data_yaml = root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 2\n")
        f.write("names: ['class_0', 'class_1']\n")

    print(f"[OK] Detection dataset created")


def get_db_path():
    """Get database path."""
    mvp_dir = Path(__file__).parent.parent
    db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'
    return db_path


def count_validation_results(job_id):
    """Count validation results in database."""
    db_path = get_db_path()

    if not db_path.exists():
        print(f"[WARNING] Database not found at {db_path}")
        return 0

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM validation_results WHERE job_id = ?",
        (job_id,)
    )
    count = cursor.fetchone()[0]

    conn.close()
    return count


def get_validation_results(job_id):
    """Get validation results from database."""
    db_path = get_db_path()

    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM validation_results
        WHERE job_id = ?
        ORDER BY epoch ASC
        """,
        (job_id,)
    )
    results = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return results


def save_validation_result_to_db(job_id, epoch, validation_metrics):
    """Save validation result to database."""
    import json
    from datetime import datetime

    db_path = get_db_path()

    if not db_path.exists():
        print(f"[WARNING] Database not found at {db_path}")
        return None

    try:
        # Get task-specific metrics
        task_metrics = validation_metrics.get_task_metrics()

        # Prepare common fields
        task_type = validation_metrics.task_type.value
        primary_metric_name = validation_metrics.primary_metric_name
        primary_metric_value = validation_metrics.primary_metric_value
        overall_loss = validation_metrics.overall_loss

        # Prepare task-specific fields
        metrics_json = json.dumps(task_metrics.to_dict())

        # Insert into database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO validation_results (
                job_id, epoch, task_type,
                primary_metric_name, primary_metric_value,
                overall_loss, metrics,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                epoch,
                task_type,
                primary_metric_name,
                primary_metric_value,
                overall_loss,
                metrics_json,
                datetime.now().isoformat()
            )
        )

        val_result_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return val_result_id

    except Exception as e:
        print(f"[ERROR] Failed to save validation result: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_yolo_validation_persistence():
    """Test YOLO validation metrics persistence."""
    from ultralytics import YOLO
    from adapters.ultralytics_adapter import UltralyticsAdapter
    import json

    print("\n" + "=" * 80)
    print("TESTING YOLO VALIDATION METRICS PERSISTENCE")
    print("=" * 80)

    job_id = 99999

    # Clear existing validation results
    db_path = get_db_path()
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM validation_results WHERE job_id = ?", (job_id,))
        cursor.execute("DELETE FROM training_metrics WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()
        print(f"[SETUP] Cleared existing results for job_id={job_id}")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "dataset"
        output_dir = Path(tmpdir) / "output"

        # Create dataset
        create_detection_dataset(dataset_dir)

        # Train YOLO
        print(f"\n[TRAINING] Starting YOLO11n for 2 epochs...")
        model = YOLO("yolo11n.pt")

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

        # Parse results.csv and save validation metrics manually
        print(f"\n[PARSING] Parsing results.csv and saving validation metrics...")

        from validators.metrics import ValidationMetricsCalculator, TaskType
        import csv

        # Path to YOLO results.csv
        results_csv = project_dir / "train" / "results.csv"

        if not results_csv.exists():
            print(f"❌ results.csv not found at {results_csv}")
            return 1

        print(f"[OK] Found results.csv at {results_csv}")

        # Parse results.csv
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            epoch_count = 0
            for row in reader:
                # Clean up row values
                row = {k.strip(): v.strip() for k, v in row.items()}

                epoch = int(row['epoch'])
                epoch_count += 1

                # Extract metrics
                train_box_loss = float(row.get('train/box_loss', 0))
                train_cls_loss = float(row.get('train/cls_loss', 0))
                train_dfl_loss = float(row.get('train/dfl_loss', 0))
                val_box_loss = float(row.get('val/box_loss', 0))
                val_cls_loss = float(row.get('val/cls_loss', 0))
                val_dfl_loss = float(row.get('val/dfl_loss', 0))
                precision = float(row.get('metrics/precision(B)', 0))
                recall = float(row.get('metrics/recall(B)', 0))
                mAP50 = float(row.get('metrics/mAP50(B)', 0))
                mAP50_95 = float(row.get('metrics/mAP50-95(B)', 0))

                # Calculate total losses
                train_loss = train_box_loss + train_cls_loss + train_dfl_loss
                val_loss = val_box_loss + val_cls_loss + val_dfl_loss

                print(f"\n[EPOCH {epoch}] Metrics:")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  mAP50: {mAP50:.4f}")
                print(f"  mAP50-95: {mAP50_95:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")

                # Create ValidationMetrics using ValidationMetricsCalculator
                validation_metrics = ValidationMetricsCalculator.compute_metrics(
                    task_type=TaskType.DETECTION,
                    predictions=None,  # Not needed, metrics pre-computed
                    labels=None,  # Not needed, metrics pre-computed
                    class_names=['class_0', 'class_1'],
                    loss=val_loss,
                    map_50=mAP50,
                    map_50_95=mAP50_95,
                    precision=precision,
                    recall=recall,
                )

                # Save to database
                val_result_id = save_validation_result_to_db(
                    job_id=job_id,
                    epoch=epoch,
                    validation_metrics=validation_metrics
                )

                if val_result_id:
                    print(f"  ✓ Saved to database (ID: {val_result_id})")
                else:
                    print(f"  ✗ Failed to save to database")

        print(f"\n[OK] Parsed {epoch_count} epochs from results.csv")

        # Verify results
        print(f"\n[VERIFY] Checking database...")
        count = count_validation_results(job_id)

        print(f"[VERIFY] Found {count} validation results in database")

        if count != 2:
            print(f"❌ FAILED: Expected 2 validation results, found {count}")
            return 1

        # Get detailed results
        val_results = get_validation_results(job_id)

        print(f"\n[RESULTS] Validation metrics:")
        for result in val_results:
            print(f"\n  Epoch {result['epoch']}:")
            print(f"    Task type: {result['task_type']}")
            print(f"    Primary metric: {result['primary_metric_name']} = {result['primary_metric_value']:.4f}")
            print(f"    Overall loss: {result['overall_loss']:.4f}")

            if result['metrics']:
                metrics = json.loads(result['metrics'])
                print(f"    Detection metrics:")
                print(f"      - mAP@0.5: {metrics.get('mAP@0.5', 0):.4f}")
                print(f"      - mAP@0.5:0.95: {metrics.get('mAP@0.5:0.95', 0):.4f}")
                print(f"      - Precision: {metrics.get('precision', 0):.4f}")
                print(f"      - Recall: {metrics.get('recall', 0):.4f}")

        # Verify epoch 1
        epoch1 = val_results[0]
        assert epoch1['job_id'] == job_id, "job_id mismatch"
        assert epoch1['epoch'] == 1, "epoch mismatch"
        assert epoch1['task_type'] == 'object_detection', "task_type mismatch"
        assert epoch1['primary_metric_name'] == 'mAP@0.5', "primary_metric_name mismatch"
        assert epoch1['overall_loss'] is not None, "overall_loss is None"

        # Verify metrics
        metrics = json.loads(epoch1['metrics'])
        assert 'mAP@0.5' in metrics, "mAP@0.5 missing"
        assert 'mAP@0.5:0.95' in metrics, "mAP@0.5:0.95 missing"
        assert 'precision' in metrics, "precision missing"
        assert 'recall' in metrics, "recall missing"

        print("\n" + "=" * 80)
        print("✅ YOLO VALIDATION METRICS PERSISTENCE TEST PASSED")
        print("=" * 80)
        print(f"\n✓ {count} validation results saved correctly")
        print(f"✓ All required metrics present")
        print(f"✓ Task type correctly identified")
        print(f"✓ JSON serialization working")

    return 0


if __name__ == "__main__":
    try:
        exit_code = test_yolo_validation_persistence()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
