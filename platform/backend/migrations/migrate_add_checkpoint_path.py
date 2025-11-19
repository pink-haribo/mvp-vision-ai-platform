"""Add checkpoint_path column to training_metrics table."""

import os
import sys
import sqlite3
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings


def migrate():
    """Add checkpoint_path column to training_metrics table."""
    # Get database path from settings
    db_url = settings.DATABASE_URL

    # Extract SQLite database path
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        # Handle relative paths
        if not os.path.isabs(db_path):
            # Database path is relative, make it absolute from project root
            project_root = os.path.dirname(os.path.dirname(backend_dir))
            db_path = os.path.join(project_root, db_path)
    else:
        print(f"[ERROR] Unsupported database URL: {db_url}")
        sys.exit(1)

    print(f"[INFO] Database path: {db_path}")

    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found: {db_path}")
        sys.exit(1)

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(training_metrics)")
        columns = [row[1] for row in cursor.fetchall()]

        if "checkpoint_path" in columns:
            print("[INFO] Column 'checkpoint_path' already exists in training_metrics table")
            return

        # Add checkpoint_path column
        print("[INFO] Adding 'checkpoint_path' column to training_metrics table...")
        cursor.execute("""
            ALTER TABLE training_metrics
            ADD COLUMN checkpoint_path VARCHAR(500)
        """)

        conn.commit()
        print("[SUCCESS] Migration completed successfully!")

        # Verify the column was added
        cursor.execute("PRAGMA table_info(training_metrics)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"[INFO] Current columns in training_metrics: {', '.join(columns)}")

    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    print("="*80)
    print("Migration: Add checkpoint_path to training_metrics")
    print("="*80)
    migrate()
