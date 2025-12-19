"""Migration script to add custom_docker_image column to training_jobs table.

This column allows specifying a custom Docker image for new/custom training frameworks
that don't have a pre-built trainer image (e.g., native torch, openmm).

Run this script to update the PostgreSQL database schema:
    python migrate_add_custom_docker_image.py
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
    """Add custom_docker_image column to training_jobs table."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'training_jobs' AND column_name = 'custom_docker_image'"
        )

        if cursor.fetchone():
            print("OK: Column 'custom_docker_image' already exists, skipping migration")
            return

        # Add custom_docker_image column
        cursor.execute("""
            ALTER TABLE training_jobs
            ADD COLUMN custom_docker_image VARCHAR(500)
        """)

        conn.commit()
        print("OK: Successfully added 'custom_docker_image' column to training_jobs table")

    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    migrate()
