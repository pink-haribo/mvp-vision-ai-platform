"""Migration: Add projects table and experiment metadata to training_jobs."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "db" / "vision_platform.db"


def migrate():
    """Add projects table and experiment-related columns to training_jobs."""

    if not DB_PATH.exists():
        print(f"[ERROR] Database not found at: {DB_PATH}")
        return False

    print(f"[INFO] Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        print("[INFO] Starting migration to add projects and experiment metadata...")

        # Step 1: Create projects table
        print("[STEP 1] Creating projects table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(200) NOT NULL UNIQUE,
                description TEXT,
                task_type VARCHAR(50),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("[OK] Projects table created")

        # Step 2: Create default "Uncategorized" project
        print("[STEP 2] Creating default 'Uncategorized' project...")
        cursor.execute("""
            INSERT OR IGNORE INTO projects (name, description, task_type)
            VALUES (?, ?, ?)
        """, (
            "Uncategorized",
            "Default project for experiments without a specific project",
            None
        ))
        default_project_id = cursor.lastrowid
        if default_project_id == 0:
            # Already exists, fetch it
            cursor.execute("SELECT id FROM projects WHERE name = 'Uncategorized'")
            default_project_id = cursor.fetchone()[0]
        print(f"[OK] Default project ID: {default_project_id}")

        # Step 3: Check if migration already done
        cursor.execute("PRAGMA table_info(training_jobs)")
        columns = [column[1] for column in cursor.fetchall()]

        if "project_id" in columns:
            print("[INFO] Migration already applied, skipping...")
            conn.close()
            return True

        # Step 4: Create new training_jobs table with additional columns
        print("[STEP 3] Creating new training_jobs table with experiment metadata...")
        cursor.execute("""
            CREATE TABLE training_jobs_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                project_id INTEGER,

                -- Experiment metadata (NEW)
                experiment_name VARCHAR(200),
                tags TEXT,  -- JSON array stored as text
                notes TEXT,
                mlflow_run_id VARCHAR(100),

                -- Training configuration
                framework VARCHAR(50) NOT NULL DEFAULT 'timm',
                model_name VARCHAR(100) NOT NULL,
                task_type VARCHAR(50) NOT NULL,
                num_classes INTEGER,
                dataset_path VARCHAR(500) NOT NULL,
                dataset_format VARCHAR(50) NOT NULL DEFAULT 'imagefolder',
                output_dir VARCHAR(500) NOT NULL,

                -- Hyperparameters
                epochs INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                learning_rate REAL NOT NULL,

                -- Status and results
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                error_message TEXT,
                process_id INTEGER,
                final_accuracy REAL,
                best_checkpoint_path VARCHAR(500),

                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                started_at DATETIME,
                completed_at DATETIME,

                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        print("[OK] New training_jobs table created")

        # Step 5: Copy data from old table to new table
        print("[STEP 4] Copying data from old table...")
        cursor.execute("""
            INSERT INTO training_jobs_new (
                id, session_id, project_id,
                experiment_name, tags, notes, mlflow_run_id,
                framework, model_name, task_type, num_classes,
                dataset_path, dataset_format, output_dir,
                epochs, batch_size, learning_rate,
                status, error_message, process_id,
                final_accuracy, best_checkpoint_path,
                created_at, started_at, completed_at
            )
            SELECT
                id, session_id, NULL as project_id,
                NULL as experiment_name, NULL as tags, NULL as notes, NULL as mlflow_run_id,
                framework, model_name, task_type, num_classes,
                dataset_path, dataset_format, output_dir,
                epochs, batch_size, learning_rate,
                status, error_message, process_id,
                final_accuracy, best_checkpoint_path,
                created_at, started_at, completed_at
            FROM training_jobs
        """)
        rows_copied = cursor.rowcount
        print(f"[OK] Copied {rows_copied} rows to new table")

        # Step 6: Update existing training jobs to link to default project (optional)
        print("[STEP 5] Linking existing experiments to default project...")
        cursor.execute("""
            UPDATE training_jobs_new
            SET project_id = ?
            WHERE project_id IS NULL
        """, (default_project_id,))
        rows_updated = cursor.rowcount
        print(f"[OK] Linked {rows_updated} experiments to default project")

        # Step 7: Drop old table
        print("[STEP 6] Dropping old training_jobs table...")
        cursor.execute("DROP TABLE training_jobs")
        print("[OK] Old table dropped")

        # Step 8: Rename new table
        print("[STEP 7] Renaming new table to training_jobs...")
        cursor.execute("ALTER TABLE training_jobs_new RENAME TO training_jobs")
        print("[OK] Table renamed")

        # Step 9: Create index on project_id for faster queries
        print("[STEP 8] Creating index on project_id...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_jobs_project_id
            ON training_jobs(project_id)
        """)
        print("[OK] Index created")

        # Commit changes
        conn.commit()
        print("\n" + "="*60)
        print("[SUCCESS] Migration completed successfully!")
        print(f"  - Created projects table")
        print(f"  - Created default project (ID: {default_project_id})")
        print(f"  - Added experiment metadata columns to training_jobs")
        print(f"  - Migrated {rows_copied} existing experiments")
        print(f"  - Linked {rows_updated} experiments to default project")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    success = migrate()
    exit(0 if success else 1)
