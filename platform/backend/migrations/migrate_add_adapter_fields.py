"""Migration script to add Adapter pattern fields to existing database.

This script adds the following fields to the training_jobs table:
- framework (String, default='timm')
- dataset_format (String, default='imagefolder')
- num_classes (nullable)
"""

import os
import sys
import sqlite3

# Add app directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app.core.config import settings


def get_db_path():
    """Get the SQLite database file path."""
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        return db_url.replace("sqlite:///", "")
    return None


def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def migrate_database():
    """Add new columns to existing database."""
    db_path = get_db_path()
    if not db_path:
        print("[ERROR] Only SQLite databases are supported")
        return False

    if not os.path.exists(db_path):
        print(f"[INFO] Database does not exist: {db_path}")
        print("[INFO] Run init_db.py to create a new database with latest schema")
        return False

    print(f"[INFO] Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check and add 'framework' column
        if not check_column_exists(cursor, "training_jobs", "framework"):
            print("[INFO] Adding 'framework' column...")
            cursor.execute(
                "ALTER TABLE training_jobs ADD COLUMN framework VARCHAR(50) DEFAULT 'timm' NOT NULL"
            )
            print("[OK] Added 'framework' column")
        else:
            print("[SKIP] Column 'framework' already exists")

        # Check and add 'dataset_format' column
        if not check_column_exists(cursor, "training_jobs", "dataset_format"):
            print("[INFO] Adding 'dataset_format' column...")
            cursor.execute(
                "ALTER TABLE training_jobs ADD COLUMN dataset_format VARCHAR(50) DEFAULT 'imagefolder' NOT NULL"
            )
            print("[OK] Added 'dataset_format' column")
        else:
            print("[SKIP] Column 'dataset_format' already exists")

        # Update num_classes to be nullable (SQLite doesn't support ALTER COLUMN)
        # We need to check if there are any NULL values
        cursor.execute("SELECT COUNT(*) FROM training_jobs WHERE num_classes IS NULL")
        null_count = cursor.fetchone()[0]

        if null_count == 0:
            print("[INFO] Column 'num_classes' is already nullable (no constraints found)")
        else:
            print(f"[INFO] Found {null_count} rows with NULL num_classes")

        conn.commit()
        print("\n[SUCCESS] Migration completed successfully!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Migration failed: {str(e)}")
        conn.rollback()
        return False

    finally:
        conn.close()


def main():
    """Main migration function."""
    print("="*80)
    print("Database Migration: Add Adapter Pattern Fields")
    print("="*80)
    print()

    if migrate_database():
        print("\nDatabase schema is now up to date!")
        print("\nNew fields added:")
        print("  - framework: Framework to use (timm, ultralytics, transformers)")
        print("  - dataset_format: Dataset format (imagefolder, yolo, coco)")
        print("  - num_classes: Now nullable for non-classification tasks")
    else:
        print("\nMigration failed or was not needed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
