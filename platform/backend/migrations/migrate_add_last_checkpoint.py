"""Migration script to add last_checkpoint_path column to training_jobs table.

Run this script to update the PostgreSQL database schema:
    python migrate_add_last_checkpoint.py
"""

import psycopg2

# Database connection details
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "platform",
    "user": "admin",
    "password": "devpass"
}

def migrate():
    """Add last_checkpoint_path column to training_jobs table."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'training_jobs' AND column_name = 'last_checkpoint_path'"
        )

        if cursor.fetchone():
            print("OK: Column 'last_checkpoint_path' already exists, skipping migration")
            return

        # Add last_checkpoint_path column
        cursor.execute("""
            ALTER TABLE training_jobs
            ADD COLUMN last_checkpoint_path VARCHAR(500)
        """)

        conn.commit()
        print("OK: Successfully added 'last_checkpoint_path' column to training_jobs table")

    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    migrate()
