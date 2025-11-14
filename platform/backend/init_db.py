"""Initialize database and create necessary directories."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"[INFO] Loaded .env from: {env_path}")
else:
    print(f"[WARNING] No .env file found at {env_path}")

# Add app directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app.db.database import init_db, get_db
from app.core.config import settings
from app.db import models  # Import models so Base knows about tables
from app.core.security import get_password_hash


def create_directories():
    """Create necessary directories for data storage."""
    directories = [
        settings.UPLOAD_DIR,
        settings.OUTPUT_DIR,
        settings.MODEL_DIR,
        settings.LOG_DIR,
    ]

    # Add database directory only for SQLite
    if settings.DATABASE_URL.startswith("sqlite:///"):
        db_dir = os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", ""))
        if db_dir:  # Only add if not empty string
            directories.append(db_dir)

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created directory: {directory}")


def create_admin_user():
    """Create default admin user."""
    db = next(get_db())

    try:
        # Check if admin already exists
        admin = db.query(models.User).filter(models.User.email == "admin@example.com").first()

        if admin:
            print("[INFO] Admin user already exists")
            return

        # Create admin user
        admin_user = models.User(
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),
            full_name="Admin User",
            company="Platform",
            division="Engineering",
            department="Development",
            badge_color="indigo",
            system_role="admin",  # Admin role
            is_active=True
        )

        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)

        print(f"[OK] Admin user created:")
        print(f"     Email: admin@example.com")
        print(f"     Password: admin123")
        print(f"     User ID: {admin_user.id}")

    finally:
        db.close()


def main():
    """Initialize database and directories."""
    print("Initializing MVP database...")

    # Create directories
    print("\nCreating data directories...")
    create_directories()

    # Initialize database
    print("\nCreating database tables...")
    init_db()
    print("[OK] Database tables created")

    # Create admin user
    print("\nCreating admin user...")
    create_admin_user()

    print("\n[SUCCESS] Database initialization complete!")
    print(f"\nDatabase location: {settings.DATABASE_URL}")
    print("\n===========================================")
    print("Login Credentials:")
    print("  Email: admin@example.com")
    print("  Password: admin123")
    print("===========================================")


if __name__ == "__main__":
    main()
