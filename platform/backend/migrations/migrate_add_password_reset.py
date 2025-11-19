"""Add password reset fields to users table.

This migration adds the following fields to the users table:
- password_reset_token: String(255), nullable, unique, indexed
- password_reset_expires: DateTime, nullable

These fields are used for the forgot password / reset password flow.
"""

from pathlib import Path
from dotenv import load_dotenv

# Load .env file before anything else
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"[STARTUP] Loaded .env from: {env_path}")

from sqlalchemy import create_engine, text
from app.core.config import settings

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)


def run_migration():
    """Run the migration."""
    print("[MIGRATION] Adding password reset fields to users table...")

    with engine.connect() as conn:
        # Check if password_reset_token column already exists
        try:
            result = conn.execute(text("PRAGMA table_info(users)"))
            columns = [row[1] for row in result]

            if 'password_reset_token' in columns:
                print("[MIGRATION] password_reset_token already exists, skipping")
                return

            # Add password_reset_token column
            print("[MIGRATION] Adding password_reset_token column...")
            conn.execute(text("""
                ALTER TABLE users ADD COLUMN password_reset_token VARCHAR(255)
            """))

            # Create unique index on password_reset_token
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_password_reset_token
                ON users(password_reset_token)
            """))

            # Add password_reset_expires column
            print("[MIGRATION] Adding password_reset_expires column...")
            conn.execute(text("""
                ALTER TABLE users ADD COLUMN password_reset_expires DATETIME
            """))

            conn.commit()
            print("[MIGRATION] Password reset fields added successfully!")

        except Exception as e:
            print(f"[MIGRATION ERROR] Failed to add password reset fields: {e}")
            raise


if __name__ == "__main__":
    print("=" * 60)
    print("Migration: Add password reset fields to users")
    print("=" * 60)
    run_migration()
    print("[MIGRATION] Complete!")
