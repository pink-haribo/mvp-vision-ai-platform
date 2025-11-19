"""
Database migration script for Dataset versioning and snapshot support.

Changes:
- Dataset: Remove task_type, add versioning and snapshot fields
- TrainingJob: Add dataset_snapshot_id for immutable training references

Run: python migrate_dataset_versioning.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from sqlalchemy import create_engine, text
from app.core.config import settings

def migrate():
    """Execute migration."""
    db_url = settings.DATABASE_URL
    engine = create_engine(db_url, connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {})

    print(f"[INFO] Connecting to database: {db_url}")

    with engine.begin() as conn:
        print("\n[INFO] Checking current schema...")

        # Check if migration already applied
        result = conn.execute(text("""
            SELECT COUNT(*) as cnt FROM pragma_table_info('datasets')
            WHERE name = 'labeled'
        """))
        if result.fetchone()[0] > 0:
            print("[WARNING] Migration already applied (labeled field exists)")
            return

        print("\n[STEP 1] Creating new datasets table with updated schema...")

        # Create new table with updated schema
        conn.execute(text("""
            CREATE TABLE datasets_new (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,

                owner_id INTEGER,

                visibility TEXT NOT NULL DEFAULT 'private',
                tags TEXT,

                storage_path TEXT NOT NULL,
                storage_type TEXT NOT NULL DEFAULT 'r2',

                format TEXT NOT NULL,
                labeled INTEGER NOT NULL DEFAULT 0,
                annotation_path TEXT,
                num_classes INTEGER,
                num_images INTEGER NOT NULL DEFAULT 0,
                class_names TEXT,

                is_snapshot INTEGER NOT NULL DEFAULT 0,
                parent_dataset_id TEXT,
                snapshot_created_at TEXT,
                version_tag TEXT,

                status TEXT NOT NULL DEFAULT 'active',
                integrity_status TEXT NOT NULL DEFAULT 'valid',

                version INTEGER NOT NULL DEFAULT 1,
                content_hash TEXT,
                last_modified_at TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE SET NULL,
                FOREIGN KEY (parent_dataset_id) REFERENCES datasets_new(id) ON DELETE CASCADE
            )
        """))

        print("[OK] New table created")

        print("\n[STEP 2] Migrating existing data...")

        # Copy data from old table to new table (excluding task_type)
        conn.execute(text("""
            INSERT INTO datasets_new (
                id, name, description, owner_id, visibility, tags,
                storage_path, storage_type, format, num_classes, num_images, class_names,
                version, content_hash, last_modified_at, created_at, updated_at,
                labeled, is_snapshot, status, integrity_status
            )
            SELECT
                id, name, description, owner_id, visibility, tags,
                storage_path, storage_type, format, num_classes, num_images, class_names,
                version, content_hash, last_modified_at, created_at, updated_at,
                0 as labeled,
                0 as is_snapshot,
                'active' as status,
                'valid' as integrity_status
            FROM datasets
        """))

        print("[OK] Data migrated")

        print("\n[STEP 3] Replacing old table...")

        # Drop old table and rename new table
        conn.execute(text("DROP TABLE datasets"))
        conn.execute(text("ALTER TABLE datasets_new RENAME TO datasets"))

        print("[OK] Table replaced")

        print("\n[STEP 4] Recreating indexes...")

        # Recreate indexes
        conn.execute(text("CREATE INDEX ix_datasets_id ON datasets(id)"))
        conn.execute(text("CREATE INDEX ix_datasets_owner_id ON datasets(owner_id)"))
        conn.execute(text("CREATE INDEX ix_datasets_visibility ON datasets(visibility)"))

        print("[OK] Indexes created")

        print("\n[STEP 5] Updating TrainingJob table...")

        # Check if dataset_snapshot_id already exists
        result = conn.execute(text("""
            SELECT COUNT(*) as cnt FROM pragma_table_info('training_jobs')
            WHERE name = 'dataset_snapshot_id'
        """))

        if result.fetchone()[0] == 0:
            # Add dataset_snapshot_id column
            conn.execute(text("""
                ALTER TABLE training_jobs
                ADD COLUMN dataset_snapshot_id TEXT
            """))

            # Add foreign key index
            conn.execute(text("""
                CREATE INDEX ix_training_jobs_dataset_snapshot_id
                ON training_jobs(dataset_snapshot_id)
            """))

            print("[OK] TrainingJob.dataset_snapshot_id added")
        else:
            print("[WARNING] TrainingJob.dataset_snapshot_id already exists")

        print("\n[STEP 6] Recreating dataset_permissions foreign key...")

        # Recreate dataset_permissions table with new FK
        conn.execute(text("""
            CREATE TABLE dataset_permissions_new (
                id INTEGER PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                granted_by INTEGER,
                granted_at TEXT NOT NULL,

                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (granted_by) REFERENCES users(id) ON DELETE SET NULL
            )
        """))

        # Copy data
        conn.execute(text("""
            INSERT INTO dataset_permissions_new
            SELECT * FROM dataset_permissions
        """))

        # Replace table
        conn.execute(text("DROP TABLE dataset_permissions"))
        conn.execute(text("ALTER TABLE dataset_permissions_new RENAME TO dataset_permissions"))

        # Recreate indexes
        conn.execute(text("CREATE INDEX ix_dataset_permissions_id ON dataset_permissions(id)"))
        conn.execute(text("CREATE INDEX ix_dataset_permissions_dataset_id ON dataset_permissions(dataset_id)"))
        conn.execute(text("CREATE INDEX ix_dataset_permissions_user_id ON dataset_permissions(user_id)"))

        print("[OK] dataset_permissions updated")

        print("\n[SUCCESS] Migration completed successfully!")
        print("\n[SUMMARY]")
        print("  - Removed: Dataset.task_type")
        print("  - Added: Dataset.labeled, annotation_path, is_snapshot, parent_dataset_id,")
        print("           snapshot_created_at, version_tag, status, integrity_status")
        print("  - Added: TrainingJob.dataset_snapshot_id")
        print("\n[NOTE] task_type moved to TrainingJob (already exists)")


if __name__ == "__main__":
    try:
        migrate()
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
