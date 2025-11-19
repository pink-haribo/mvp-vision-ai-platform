"""Migration script to add advanced configuration field to training_jobs table.

This script adds the following field to the training_jobs table:
- advanced_config (JSON, nullable)
  Stores advanced training configurations including optimizer, scheduler,
  augmentation, preprocessing, and validation settings.
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
    """Add advanced_config column to training_jobs table."""
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
        # Check and add 'advanced_config' column
        if not check_column_exists(cursor, "training_jobs", "advanced_config"):
            print("[INFO] Adding 'advanced_config' column...")
            cursor.execute(
                "ALTER TABLE training_jobs ADD COLUMN advanced_config JSON"
            )
            print("[OK] Added 'advanced_config' column")
        else:
            print("[SKIP] Column 'advanced_config' already exists")

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
    print("Database Migration: Add Advanced Configuration Field")
    print("="*80)
    print()

    if migrate_database():
        print("\nDatabase schema is now up to date!")
        print("\nNew field added:")
        print("  - advanced_config: JSON field for advanced training configurations")
        print("    (optimizer, scheduler, augmentation, preprocessing, validation)")
    else:
        print("\nMigration failed or was not needed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
