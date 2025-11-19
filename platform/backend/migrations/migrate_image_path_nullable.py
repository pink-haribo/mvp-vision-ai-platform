"""
Migration: Make image_path nullable in validation_image_results table.

This allows storing validation results even when actual image paths are not available.
Image paths can be populated later if needed for visualization.
"""

import sqlite3
from pathlib import Path


def migrate():
    """Make image_path column nullable."""
    # Get database path
    backend_dir = Path(__file__).parent
    mvp_dir = backend_dir.parent
    db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    print(f"Migrating database at {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
        print("Creating backup table...")
        cursor.execute("""
            CREATE TABLE validation_image_results_backup AS
            SELECT * FROM validation_image_results
        """)

        print("Dropping old table...")
        cursor.execute("DROP TABLE validation_image_results")

        print("Creating new table with nullable image_path...")
        cursor.execute("""
            CREATE TABLE validation_image_results (
                id INTEGER PRIMARY KEY,
                validation_result_id INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                image_path TEXT,
                image_name TEXT NOT NULL,
                image_index INTEGER,
                true_label TEXT,
                true_label_id INTEGER,
                predicted_label TEXT,
                predicted_label_id INTEGER,
                confidence REAL,
                top5_predictions TEXT,
                true_boxes TEXT,
                predicted_boxes TEXT,
                true_mask_path TEXT,
                predicted_mask_path TEXT,
                true_keypoints TEXT,
                predicted_keypoints TEXT,
                is_correct INTEGER NOT NULL DEFAULT 0,
                iou REAL,
                oks REAL,
                extra_data TEXT,
                created_at TEXT,
                FOREIGN KEY (validation_result_id) REFERENCES validation_results (id),
                FOREIGN KEY (job_id) REFERENCES training_jobs (id)
            )
        """)

        print("Creating indexes...")
        cursor.execute("CREATE INDEX idx_validation_image_results_validation_result_id ON validation_image_results (validation_result_id)")
        cursor.execute("CREATE INDEX idx_validation_image_results_job_id ON validation_image_results (job_id)")
        cursor.execute("CREATE INDEX idx_validation_image_results_epoch ON validation_image_results (epoch)")
        cursor.execute("CREATE INDEX idx_validation_image_results_image_name ON validation_image_results (image_name)")
        cursor.execute("CREATE INDEX idx_validation_image_results_is_correct ON validation_image_results (is_correct)")

        print("Restoring data from backup...")
        cursor.execute("""
            INSERT INTO validation_image_results
            SELECT * FROM validation_image_results_backup
        """)

        row_count = cursor.rowcount
        print(f"Restored {row_count} rows")

        print("Dropping backup table...")
        cursor.execute("DROP TABLE validation_image_results_backup")

        conn.commit()
        print("Migration completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        print("Rolling back changes...")
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
