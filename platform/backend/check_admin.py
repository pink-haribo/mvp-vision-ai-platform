"""Check admin user exists."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.db.database import SessionLocal
from app.db import models

db = SessionLocal()

try:
    admin = db.query(models.User).filter(
        models.User.email == "admin@example.com"
    ).first()

    if admin:
        print("Admin user found:")
        print(f"  ID: {admin.id}")
        print(f"  Email: {admin.email}")
        print(f"  Name: {admin.full_name}")
        print(f"  Role: {admin.system_role}")
        print(f"  Active: {admin.is_active}")
        print(f"  Badge Color: {admin.badge_color}")

        # Check projects
        projects = db.query(models.Project).filter(
            models.Project.user_id == admin.id
        ).all()

        print(f"\nProjects owned by admin: {len(projects)}")
        for p in projects:
            print(f"  - {p.name} (ID: {p.id})")
    else:
        print("Admin user NOT found")

finally:
    db.close()
