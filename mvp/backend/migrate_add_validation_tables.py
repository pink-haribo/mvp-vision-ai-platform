"""
Add task-agnostic validation tables.

This migration creates:
1. validation_results - Stores validation metrics per epoch (task-agnostic)
2. validation_image_results - Stores image-level validation results (task-agnostic)

Both tables support all computer vision tasks:
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
    """Run migration to add validation tables."""
    print("\n" + "="*80)
    print("VALIDATION TABLES MIGRATION - Task-Agnostic Design")
    print("="*80)

    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        print("\n[1/4] Creating validation_results table...")

        # Create validation_results table (Task-Agnostic)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS validation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            epoch INTEGER NOT NULL,

            -- Task identification
            task_type TEXT NOT NULL,

            -- Common metrics
            primary_metric_value REAL,
            primary_metric_name TEXT,
            overall_loss REAL,

            -- Task-specific metrics (JSON)
            metrics TEXT,
            per_class_metrics TEXT,

            -- Visualization data (task-specific)
            confusion_matrix TEXT,
            pr_curves TEXT,
            class_names TEXT,
            visualization_data TEXT,

            -- Sample images
            sample_correct_images TEXT,
            sample_incorrect_images TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))
        print("   [OK] validation_results table created")

        print("\n[2/4] Creating validation_image_results table...")

        # Create validation_image_results table (Task-Agnostic)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS validation_image_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            validation_result_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            epoch INTEGER NOT NULL,

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

            -- Extra
            extra_data TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (validation_result_id) REFERENCES validation_results(id) ON DELETE CASCADE,
            FOREIGN KEY (job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))
        print("   [OK] validation_image_results table created")

        print("\n[3/4] Creating indexes...")

        # Create indexes
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_validation_results_job_epoch
        ON validation_results(job_id, epoch)
        """))
        print("   [OK] idx_validation_results_job_epoch")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_validation_results_task_type
        ON validation_results(task_type)
        """))
        print("   [OK] idx_validation_results_task_type")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_validation_image_results_val_id
        ON validation_image_results(validation_result_id)
        """))
        print("   [OK] idx_validation_image_results_val_id")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_validation_image_results_job_epoch
        ON validation_image_results(job_id, epoch)
        """))
        print("   [OK] idx_validation_image_results_job_epoch")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_validation_image_results_correct
        ON validation_image_results(is_correct)
        """))
        print("   [OK] idx_validation_image_results_correct")

        print("\n[4/4] Committing changes...")
        conn.commit()
        print("   [OK] Changes committed")

    print("\n" + "="*80)
    print("[SUCCESS] MIGRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nTask-agnostic validation tables are now ready.")
    print("Supported tasks:")
    print("  - Image Classification")
    print("  - Object Detection")
    print("  - Instance/Semantic Segmentation")
    print("  - Pose Estimation")
    print("\n")


if __name__ == "__main__":
    migrate()
