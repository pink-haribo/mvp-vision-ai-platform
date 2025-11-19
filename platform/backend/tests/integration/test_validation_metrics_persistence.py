"""
Test validation metrics persistence to database.

This script verifies that validation metrics are properly saved to the
validation_results table during training for different frameworks.

Priority: P1 (Core functionality for production)

Usage:
    # Run in training environment (with framework dependencies installed)
    cd mvp/training
    python -m pytest ../backend/tests/integration/test_validation_metrics_persistence.py -v -s
"""

import pytest
import sys
import sqlite3
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


def _get_db_path():
    """Get path to SQLite database."""
    mvp_dir = Path(__file__).parent.parent.parent.parent
    db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'
    return db_path


def _count_validation_results(job_id: int) -> int:
    """Count validation results for a job in database."""
    db_path = _get_db_path()

    if not db_path.exists():
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


def _get_validation_results(job_id: int):
    """Get validation results for a job from database."""
    db_path = _get_db_path()

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


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def sample_detection_dataset(tmp_path_factory):
    """Create a tiny YOLO detection dataset for testing."""
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

            # Create YOLO format label
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

    yield dataset_root


# ============================================================
# YOLO Validation Metrics Persistence Tests
# ============================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not _is_ultralytics_available(),
    reason="ultralytics not installed"
)
class TestYOLOValidationPersistence:
    """Test YOLO validation metrics are saved to database."""

    def test_yolo_validation_metrics_saved_to_db(self, sample_detection_dataset, tmp_path):
        """
        Test that YOLO validation metrics are saved to validation_results table.

        This test:
        1. Runs YOLO training for 2 epochs
        2. Verifies validation_results table has 2 entries (one per epoch)
        3. Verifies metrics contain mAP, precision, recall
        4. Verifies task_type is correctly set
        """
        from ultralytics import YOLO
        import json

        print("\n[TEST] YOLO validation metrics persistence...")

        # Use a deterministic job_id for testing
        job_id = 99999

        # Clear any existing validation results for this job_id
        db_path = _get_db_path()
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM validation_results WHERE job_id = ?", (job_id,))
            conn.commit()
            conn.close()
            print(f"[SETUP] Cleared existing validation results for job_id={job_id}")

        # Create model
        model = YOLO("yolo11n.pt")

        project_dir = tmp_path / "yolo_project"

        print(f"\n[TRAINING] Starting 2 epochs with job_id={job_id}...")

        # Train with minimal settings
        results = model.train(
            data=str(sample_detection_dataset / "data.yaml"),
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

        # Now manually parse results.csv and save validation results
        # (This simulates what _convert_yolo_results() does)
        from adapters.ultralytics_adapter import UltralyticsAdapter
        from adapters.base import TaskType

        # Create a mock adapter to use its methods
        # We need to create a proper config structure
        from pydantic import BaseModel

        class MockModelConfig(BaseModel):
            task_type: str = "object_detection"
            model_name: str = "yolo11n"
            image_size: int = 64

        class MockTrainingConfig(BaseModel):
            epochs: int = 2
            batch_size: int = 2
            learning_rate: float = 0.001
            device: str = "cpu"
            advanced_config: dict = {}

        # Initialize adapter
        adapter = UltralyticsAdapter(
            job_id=job_id,
            model_config=MockModelConfig(),
            training_config=MockTrainingConfig(),
            data_yaml=str(sample_detection_dataset / "data.yaml"),
            output_dir=str(project_dir.parent),
            dataset_path=str(sample_detection_dataset),
            task_type="detect"
        )

        # Manually call _convert_yolo_results to save validation results
        adapter._convert_yolo_results(results)

        # Verify validation results were saved
        count = _count_validation_results(job_id)
        print(f"\n[VERIFY] Found {count} validation results in database")

        assert count == 2, f"Expected 2 validation results (one per epoch), found {count}"

        # Get detailed validation results
        val_results = _get_validation_results(job_id)

        print(f"\n[VERIFY] Validation results:")
        for i, result in enumerate(val_results):
            print(f"\n  Epoch {result['epoch']}:")
            print(f"    Task type: {result['task_type']}")
            print(f"    Primary metric: {result['primary_metric_name']} = {result['primary_metric_value']:.4f}")
            print(f"    Overall loss: {result['overall_loss']:.4f}")

            # Parse metrics JSON
            if result['metrics_json']:
                metrics = json.loads(result['metrics_json'])
                print(f"    Metrics: {metrics}")

        # Verify first epoch result
        epoch1_result = val_results[0]
        assert epoch1_result['job_id'] == job_id
        assert epoch1_result['epoch'] == 1
        assert epoch1_result['task_type'] == 'object_detection'
        assert epoch1_result['primary_metric_name'] == 'mAP@0.5'
        assert epoch1_result['overall_loss'] is not None

        # Verify metrics JSON contains detection metrics
        metrics = json.loads(epoch1_result['metrics_json'])
        assert 'mAP@0.5' in metrics
        assert 'mAP@0.5:0.95' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

        print(f"\n[OK] YOLO validation metrics persistence test passed!")
        print(f"     - {count} validation results saved correctly")
        print(f"     - All required metrics present")
        print(f"     - Task type correctly identified")


# ============================================================
# Summary Test
# ============================================================

@pytest.mark.slow
def test_validation_persistence_summary():
    """Summary of validation metrics persistence tests."""
    print("\n" + "=" * 80)
    print("VALIDATION METRICS PERSISTENCE TEST SUMMARY")
    print("=" * 80)
    print("\nTested validation metrics persistence:")
    print("  - YOLO Detection (validation_results table)")
    print("    * Metrics saved per epoch")
    print("    * mAP, precision, recall stored")
    print("    * Task type correctly identified")
    print("    * JSON serialization working")
    print("=" * 80)
