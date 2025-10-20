"""Initialize database and create necessary directories."""

import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app.db.database import init_db
from app.core.config import settings
from app.db import models  # Import models so Base knows about tables


def create_directories():
    """Create necessary directories for data storage."""
    directories = [
        settings.UPLOAD_DIR,
        settings.OUTPUT_DIR,
        settings.MODEL_DIR,
        settings.LOG_DIR,
        os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "")),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created directory: {directory}")


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

    print("\n[SUCCESS] Database initialization complete!")
    print(f"\nDatabase location: {settings.DATABASE_URL}")


if __name__ == "__main__":
    main()
