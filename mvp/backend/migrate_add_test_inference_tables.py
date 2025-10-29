"""
Add task-agnostic test and inference tables.

This migration creates:
1. test_runs - Stores test results on labeled datasets (post-training)
2. test_image_results - Stores image-level test results with ground truth
3. inference_jobs - Stores inference jobs on unlabeled images (production)
4. inference_results - Stores per-image inference results (predictions only)

All tables support all computer vision tasks:
- Image Classification
- Object Detection
- Instance/Semantic Segmentation
- Pose Estimation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.core.config import settings


def migrate():
    """Run migration to add test and inference tables."""
    print("\n" + "="*80)
    print("TEST/INFERENCE TABLES MIGRATION - Task-Agnostic Design")
    print("="*80)

    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        print("\n[1/6] Creating test_runs table...")

        # Create test_runs table (Task-Agnostic)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS test_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_job_id INTEGER NOT NULL,
            checkpoint_path TEXT NOT NULL,
            dataset_path TEXT NOT NULL,
            dataset_split TEXT DEFAULT 'test',

            -- Status
            status TEXT NOT NULL,
            error_message TEXT,

            -- Task identification
            task_type TEXT NOT NULL,
            primary_metric_name TEXT,
            primary_metric_value REAL,

            -- Metrics (task-agnostic JSON)
            overall_loss REAL,
            metrics TEXT,
            per_class_metrics TEXT,
            confusion_matrix TEXT,

            -- Metadata
            class_names TEXT,
            total_images INTEGER DEFAULT 0,
            inference_time_ms REAL,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,

            FOREIGN KEY (training_job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))
        print("   [OK] test_runs table created")

        print("\n[2/6] Creating test_image_results table...")

        # Create test_image_results table (Task-Agnostic)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS test_image_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_run_id INTEGER NOT NULL,
            training_job_id INTEGER NOT NULL,

            -- Image info
            image_path TEXT NOT NULL,
            image_name TEXT NOT NULL,
            image_index INTEGER,

            -- Classification
            true_label TEXT,
            true_label_id INTEGER,
            predicted_label TEXT,
            predicted_label_id INTEGER,
            confidence REAL,
            top5_predictions TEXT,

            -- Detection
            true_boxes TEXT,
            predicted_boxes TEXT,

            -- Segmentation
            true_mask_path TEXT,
            predicted_mask_path TEXT,

            -- Pose Estimation
            true_keypoints TEXT,
            predicted_keypoints TEXT,

            -- Common metrics
            is_correct INTEGER NOT NULL DEFAULT 0,
            iou REAL,
            oks REAL,

            -- Performance
            inference_time_ms REAL,
            preprocessing_time_ms REAL,
            postprocessing_time_ms REAL,

            -- Extra
            extra_data TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (test_run_id) REFERENCES test_runs(id) ON DELETE CASCADE,
            FOREIGN KEY (training_job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))
        print("   [OK] test_image_results table created")

        print("\n[3/6] Creating inference_jobs table...")

        # Create inference_jobs table (Production inference, no labels)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS inference_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_job_id INTEGER NOT NULL,
            checkpoint_path TEXT NOT NULL,

            -- Input
            inference_type TEXT NOT NULL,
            input_data TEXT,

            -- Status
            status TEXT NOT NULL,
            error_message TEXT,

            -- Task identification
            task_type TEXT NOT NULL,

            -- Performance metrics
            total_images INTEGER DEFAULT 0,
            total_inference_time_ms REAL,
            avg_inference_time_ms REAL,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,

            FOREIGN KEY (training_job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))
        print("   [OK] inference_jobs table created")

        print("\n[4/6] Creating inference_results table...")

        # Create inference_results table (Predictions only, no ground truth)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS inference_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inference_job_id INTEGER NOT NULL,
            training_job_id INTEGER NOT NULL,

            -- Image info
            image_path TEXT NOT NULL,
            image_name TEXT NOT NULL,
            image_index INTEGER,

            -- Classification
            predicted_label TEXT,
            predicted_label_id INTEGER,
            confidence REAL,
            top5_predictions TEXT,

            -- Detection
            predicted_boxes TEXT,

            -- Segmentation
            predicted_mask_path TEXT,

            -- Pose Estimation
            predicted_keypoints TEXT,

            -- Performance
            inference_time_ms REAL,
            preprocessing_time_ms REAL,
            postprocessing_time_ms REAL,

            -- Extra
            extra_data TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (inference_job_id) REFERENCES inference_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY (training_job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))
        print("   [OK] inference_results table created")

        print("\n[5/6] Creating indexes...")

        # Test tables indexes
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_test_runs_training_job
        ON test_runs(training_job_id)
        """))
        print("   [OK] idx_test_runs_training_job")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_test_runs_status
        ON test_runs(status)
        """))
        print("   [OK] idx_test_runs_status")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_test_image_results_test_run
        ON test_image_results(test_run_id)
        """))
        print("   [OK] idx_test_image_results_test_run")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_test_image_results_training_job
        ON test_image_results(training_job_id)
        """))
        print("   [OK] idx_test_image_results_training_job")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_test_image_results_correct
        ON test_image_results(is_correct)
        """))
        print("   [OK] idx_test_image_results_correct")

        # Inference tables indexes
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_inference_jobs_training_job
        ON inference_jobs(training_job_id)
        """))
        print("   [OK] idx_inference_jobs_training_job")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_inference_jobs_status
        ON inference_jobs(status)
        """))
        print("   [OK] idx_inference_jobs_status")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_inference_results_inference_job
        ON inference_results(inference_job_id)
        """))
        print("   [OK] idx_inference_results_inference_job")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_inference_results_training_job
        ON inference_results(training_job_id)
        """))
        print("   [OK] idx_inference_results_training_job")

        print("\n[6/6] Committing changes...")
        conn.commit()
        print("   [OK] Changes committed")

    print("\n" + "="*80)
    print("[SUCCESS] MIGRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nTask-agnostic test and inference tables are now ready.")
    print("\nTest Tables (labeled data):")
    print("  - test_runs - Store test results with metrics")
    print("  - test_image_results - Store per-image predictions with ground truth")
    print("\nInference Tables (unlabeled data):")
    print("  - inference_jobs - Store inference job metadata")
    print("  - inference_results - Store predictions only (no ground truth)")
    print("\nSupported tasks:")
    print("  - Image Classification")
    print("  - Object Detection")
    print("  - Instance/Semantic Segmentation")
    print("  - Pose Estimation")
    print("\n")


if __name__ == "__main__":
    migrate()
