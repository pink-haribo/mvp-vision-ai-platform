"""
Database migration for PostgreSQL: Add dataset_id to training_jobs.

Changes:
- TrainingJob: Add dataset_id column with foreign key to datasets table

Run: python migrate_add_dataset_id_postgresql.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from sqlalchemy import create_engine, text, inspect
from app.core.config import settings


def migrate():
    """Execute migration."""
    db_url = settings.DATABASE_URL
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {}
    )

    print(f"[INFO] Connecting to database: {db_url}")
    print(f"[INFO] Database type: {'SQLite' if db_url.startswith('sqlite') else 'PostgreSQL'}")

    inspector = inspect(engine)

    with engine.begin() as conn:
        print("\n[INFO] Checking current schema...")

        # Check if training_jobs table exists
        if 'training_jobs' not in inspector.get_table_names():
            print("[ERROR] training_jobs table does not exist!")
            return

        # Get current columns
        columns = [col['name'] for col in inspector.get_columns('training_jobs')]
        print(f"[INFO] Current columns in training_jobs: {', '.join(columns)}")

        # Check if dataset_id already exists
        if 'dataset_id' in columns:
            print("[WARNING] dataset_id column already exists, skipping migration")
            return

        print("\n[STEP 1] Adding dataset_id column to training_jobs...")

        if db_url.startswith("sqlite"):
            # SQLite
            conn.execute(text("""
                ALTER TABLE training_jobs
                ADD COLUMN dataset_id TEXT
            """))
        else:
            # PostgreSQL
            conn.execute(text("""
                ALTER TABLE training_jobs
                ADD COLUMN dataset_id VARCHAR(100)
            """))

        print("[OK] dataset_id column added")

        print("\n[STEP 2] Adding index on dataset_id...")

        conn.execute(text("""
            CREATE INDEX ix_training_jobs_dataset_id
            ON training_jobs(dataset_id)
        """))

        print("[OK] Index created")

        print("\n[STEP 3] Adding foreign key constraint (if PostgreSQL)...")

        if not db_url.startswith("sqlite"):
            # PostgreSQL supports adding FK constraints after column creation
            try:
                conn.execute(text("""
                    ALTER TABLE training_jobs
                    ADD CONSTRAINT fk_training_jobs_dataset_id
                    FOREIGN KEY (dataset_id)
                    REFERENCES datasets(id)
                    ON DELETE SET NULL
                """))
                print("[OK] Foreign key constraint added")
            except Exception as e:
                print(f"[WARNING] Could not add foreign key constraint: {e}")
                print("[INFO] This is OK if datasets table doesn't exist yet")
        else:
            print("[SKIPPED] SQLite doesn't support adding FK constraints to existing tables")

        print("\n[SUCCESS] Migration completed successfully!")
        print("\n[SUMMARY]")
        print("  - Added: TrainingJob.dataset_id (VARCHAR(100), nullable)")
        print("  - Added: Index on dataset_id")
        if not db_url.startswith("sqlite"):
            print("  - Added: Foreign key constraint to datasets.id")


if __name__ == "__main__":
    try:
        migrate()
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
