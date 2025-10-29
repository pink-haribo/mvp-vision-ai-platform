"""
Migration: Add checkpoint_path field to validation_results table

This migration adds a checkpoint_path column to the validation_results table
to store the path to the checkpoint file used for each validation run.
"""

import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings


def migrate():
    """Add checkpoint_path column to validation_results table."""

    # Get database path
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    print(f"[MIGRATION] Database path: {db_path}")

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(validation_results)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'checkpoint_path' in columns:
            print("[MIGRATION] Column 'checkpoint_path' already exists. Skipping.")
            return

        # Add checkpoint_path column
        print("[MIGRATION] Adding checkpoint_path column to validation_results...")
        cursor.execute("""
            ALTER TABLE validation_results
            ADD COLUMN checkpoint_path VARCHAR(500)
        """)

        conn.commit()
        print("[MIGRATION] Successfully added checkpoint_path column")

    except Exception as e:
        print(f"[MIGRATION] Error during migration: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    print("="*80)
    print("MIGRATION: Add checkpoint_path to validation_results")
    print("="*80)
    migrate()
    print("="*80)
    print("Migration completed successfully!")
    print("="*80)
