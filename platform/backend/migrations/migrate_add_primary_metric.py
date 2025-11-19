"""
Migration script to add primary_metric fields to training_jobs table.

This adds:
- primary_metric: Name of the metric to optimize (e.g., 'accuracy', 'mAP50')
- primary_metric_mode: Whether to maximize or minimize ('max' or 'min')
"""

import sqlite3
import sys
from pathlib import Path

# Get database path
backend_dir = Path(__file__).parent
mvp_dir = backend_dir.parent
db_path = mvp_dir / 'data' / 'db' / 'vision_platform.db'

if not db_path.exists():
    print(f"Error: Database not found at {db_path}")
    sys.exit(1)

print(f"Migrating database: {db_path}")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

try:
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(training_jobs)")
    columns = {col[1] for col in cursor.fetchall()}

    if 'primary_metric' in columns:
        print("primary_metric column already exists, skipping migration")
        sys.exit(0)

    print("Adding primary_metric and primary_metric_mode columns...")

    # Add primary_metric column
    cursor.execute("""
        ALTER TABLE training_jobs
        ADD COLUMN primary_metric VARCHAR(100) DEFAULT 'loss'
    """)

    # Add primary_metric_mode column
    cursor.execute("""
        ALTER TABLE training_jobs
        ADD COLUMN primary_metric_mode VARCHAR(10) DEFAULT 'min'
    """)

    # Update existing jobs with framework-appropriate defaults
    print("Updating existing jobs with default primary metrics...")

    # Classification tasks (timm) -> accuracy (max)
    cursor.execute("""
        UPDATE training_jobs
        SET primary_metric = 'accuracy', primary_metric_mode = 'max'
        WHERE task_type = 'image_classification' AND framework = 'timm'
    """)

    # Detection tasks (ultralytics) -> mAP50 (max)
    cursor.execute("""
        UPDATE training_jobs
        SET primary_metric = 'mAP50', primary_metric_mode = 'max'
        WHERE task_type = 'object_detection' AND framework = 'ultralytics'
    """)

    conn.commit()
    print("Migration completed successfully!")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM training_jobs WHERE primary_metric IS NOT NULL")
    count = cursor.fetchone()[0]
    print(f"Updated {count} training jobs with primary metrics")

except Exception as e:
    conn.rollback()
    print(f"Migration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    conn.close()
