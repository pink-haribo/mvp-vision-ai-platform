"""
Migration script to add split_config column to datasets table.

This script adds train/val split configuration support to datasets.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file first
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text

# Get DATABASE_URL from environment
database_url = os.getenv("DATABASE_URL", "sqlite:///../data/db/platform_vision.db")
print(f"Using database: {database_url}")

# Determine if using PostgreSQL
is_postgresql = database_url.startswith("postgresql")

# Create engine
engine = create_engine(database_url)

def run_migration():
    """Run the migration."""
    print("=" * 60)
    print("MIGRATION: Add Dataset Split Configuration")
    print("=" * 60)

    with engine.connect() as conn:
        print("\n[1/2] Checking if split_config column already exists...")

        try:
            if is_postgresql:
                # PostgreSQL: Use information_schema
                result = conn.execute(text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name='datasets' AND column_name='split_config'
                """))
                exists = result.fetchone() is not None
            else:
                # SQLite: Use PRAGMA
                result = conn.execute(text("PRAGMA table_info(datasets)"))
                columns = [row[1] for row in result.fetchall()]
                exists = 'split_config' in columns

            if exists:
                print("[OK] split_config column already exists")
                print("\n[SUCCESS] Migration completed (no changes needed)")
                return

        except Exception as e:
            print(f"[ERROR] Error checking column existence: {e}")
            return

        print("\n[2/2] Adding split_config column to datasets table...")
        try:
            if is_postgresql:
                # PostgreSQL: JSON type
                conn.execute(text("""
                    ALTER TABLE datasets
                    ADD COLUMN split_config JSON
                """))
            else:
                # SQLite: JSON type (stored as TEXT)
                conn.execute(text("""
                    ALTER TABLE datasets
                    ADD COLUMN split_config JSON
                """))

            conn.commit()
            print("[OK] split_config column added")
            print("    - Column type: JSON")
            print("    - Nullable: True")
            print("    - Purpose: Store train/val split configuration")

        except Exception as e:
            print(f"[ERROR] Error adding column: {e}")
            conn.rollback()
            return

        print("\n[SUCCESS] Migration completed successfully!")
        print("\nNext steps:")
        print("  1. Restart backend server to apply model changes")
        print("  2. Use POST /datasets/{id}/split to configure splits")
        print("  3. Split info will be cached in this column from annotations.json")


if __name__ == "__main__":
    run_migration()
