"""Migration: Make num_classes nullable for non-classification tasks."""

import sqlite3
import os
from pathlib import Path

# Database path (go up one level from mvp/backend to mvp/, then into data/db)
DB_PATH = Path(__file__).parent.parent / "data" / "db" / "vision_platform.db"

def migrate():
    """Make num_classes column nullable."""

    if not DB_PATH.exists():
        print(f"[ERROR] Database not found at: {DB_PATH}")
        return False

    print(f"[INFO] Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        print("[INFO] Starting migration to make num_classes nullable...")

        # SQLite doesn't support ALTER COLUMN directly, need to recreate table
        # Step 1: Create new table with correct schema
        cursor.execute("""
            CREATE TABLE training_jobs_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                framework VARCHAR(50) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                task_type VARCHAR(50) NOT NULL,
                num_classes INTEGER,  -- Now nullable
                dataset_path VARCHAR(500) NOT NULL,
                dataset_format VARCHAR(50) NOT NULL,
                output_dir VARCHAR(500) NOT NULL,
                epochs INTEGER NOT NULL DEFAULT 50,
                batch_size INTEGER NOT NULL DEFAULT 32,
                learning_rate REAL NOT NULL DEFAULT 0.001,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                error_message TEXT,
                process_id INTEGER,
                final_accuracy REAL,
                best_checkpoint_path VARCHAR(500),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                started_at DATETIME,
                completed_at DATETIME,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        print("[OK] Created new table with nullable num_classes")

        # Step 2: Copy data from old table (set num_classes to NULL for non-classification)
        cursor.execute("""
            INSERT INTO training_jobs_new
            SELECT
                id, session_id, framework, model_name, task_type,
                CASE
                    WHEN task_type = 'image_classification' THEN num_classes
                    ELSE NULL
                END as num_classes,
                dataset_path, dataset_format, output_dir, epochs, batch_size, learning_rate,
                status, error_message, process_id, final_accuracy, best_checkpoint_path,
                created_at, started_at, completed_at
            FROM training_jobs
        """)
        rows_copied = cursor.rowcount
        print(f"[OK] Copied {rows_copied} rows to new table")

        # Step 3: Drop old table
        cursor.execute("DROP TABLE training_jobs")
        print("[OK] Dropped old table")

        # Step 4: Rename new table
        cursor.execute("ALTER TABLE training_jobs_new RENAME TO training_jobs")
        print("[OK] Renamed new table to training_jobs")

        # Commit changes
        conn.commit()
        print("[SUCCESS] Migration completed successfully!")
        return True

    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    success = migrate()
    exit(0 if success else 1)
