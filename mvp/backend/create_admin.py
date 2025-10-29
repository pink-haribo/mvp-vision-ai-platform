"""Create admin user account."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.orm import Session
from app.db.database import SessionLocal, engine
from app.db import models
from passlib.context import CryptContext
from datetime import datetime

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_admin_user():
    """Create admin user with credentials: admin@example.com / admin123"""

    db: Session = SessionLocal()

    try:
        # Check if admin user already exists
        existing = db.query(models.User).filter(
            models.User.email == "admin@example.com"
        ).first()

        if existing:
            print("Admin user already exists!")
            print(f"  Email: {existing.email}")
            print(f"  Role: {existing.system_role}")
            return

        # Create admin user
        hashed_password = pwd_context.hash("admin123")

        admin_user = models.User(
            email="admin@example.com",
            hashed_password=hashed_password,
            full_name="Admin User",
            system_role="admin",
            is_active=True,
            badge_color="violet",  # Special color for admin
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)

        print("[OK] Admin user created successfully!")
        print(f"  Email: {admin_user.email}")
        print(f"  Password: admin123")
        print(f"  Role: {admin_user.system_role}")
        print(f"  User ID: {admin_user.id}")

        # Create default "Uncategorized" project for admin
        uncategorized = models.Project(
            name="Uncategorized",
            description="Default project for uncategorized experiments",
            task_type=None,
            user_id=admin_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        db.add(uncategorized)
        db.commit()
        db.refresh(uncategorized)

        print(f"\n[OK] Default project created!")
        print(f"  Project: {uncategorized.name}")
        print(f"  Project ID: {uncategorized.id}")

    except Exception as e:
        print(f"[ERROR] Error creating admin user: {e}")
        db.rollback()
        raise

    finally:
        db.close()

if __name__ == "__main__":
    print("Creating admin user...\n")
    create_admin_user()
    print("\n[SUCCESS] Setup complete! You can now login with:")
    print("   Email: admin@example.com")
    print("   Password: admin123")
