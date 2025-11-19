"""
Migration script to add User authentication system.

This script:
1. Creates users table
2. Creates project_members table
3. Adds user_id to projects, sessions, training_jobs
4. Creates a default admin user
5. Assigns all existing data to the admin user
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from app.core.config import settings
from app.core.security import get_password_hash

# Create engine
engine = create_engine(settings.DATABASE_URL)
Session = sessionmaker(bind=engine)

def run_migration():
    """Run the migration."""
    print("=" * 60)
    print("MIGRATION: Add User Authentication System")
    print("=" * 60)

    with engine.connect() as conn:
        print("\n[1/6] Creating users table...")
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    hashed_password VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255),
                    company VARCHAR(100),
                    company_custom VARCHAR(255),
                    division VARCHAR(100),
                    division_custom VARCHAR(255),
                    department VARCHAR(255),
                    phone_number VARCHAR(50),
                    bio TEXT,
                    system_role VARCHAR(50) NOT NULL DEFAULT 'guest',
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            print("[OK] Users table created")
        except Exception as e:
            print(f"[ERROR] Error creating users table: {e}")
            return

        print("\n[2/6] Creating project_members table...")
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS project_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    role VARCHAR(20) NOT NULL DEFAULT 'member',
                    invited_by INTEGER,
                    joined_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (invited_by) REFERENCES users(id) ON DELETE SET NULL
                )
            """))
            conn.commit()
            print("[OK] Project members table created")
        except Exception as e:
            print(f"[ERROR] Error creating project_members table: {e}")
            return

        print("\n[3/6] Creating default admin user...")
        try:
            # Check if admin user already exists
            result = conn.execute(text("SELECT id FROM users WHERE email = 'admin@example.com'"))
            existing_user = result.fetchone()

            if existing_user:
                admin_id = existing_user[0]
                print(f"[OK] Admin user already exists (ID: {admin_id})")
            else:
                # Create default admin user
                hashed_password = get_password_hash("admin123")  # Default password
                conn.execute(text("""
                    INSERT INTO users (email, hashed_password, full_name, system_role, is_active, created_at, updated_at)
                    VALUES (:email, :password, :full_name, 'admin', 1, :now, :now)
                """), {
                    "email": "admin@example.com",
                    "password": hashed_password,
                    "full_name": "System Administrator",
                    "now": datetime.utcnow()
                })
                conn.commit()

                # Get the newly created admin ID
                result = conn.execute(text("SELECT id FROM users WHERE email = 'admin@example.com'"))
                admin_id = result.fetchone()[0]
                print(f"[OK] Admin user created (ID: {admin_id})")
                print(f"  Email: admin@example.com")
                print(f"  Password: admin123")
                print(f"  [WARNING]  CHANGE THIS PASSWORD IN PRODUCTION!")
        except Exception as e:
            print(f"[ERROR] Error creating admin user: {e}")
            return

        print(f"\n[4/6] Adding user_id column to projects table...")
        try:
            # Check if column already exists
            result = conn.execute(text("PRAGMA table_info(projects)"))
            columns = [row[1] for row in result.fetchall()]

            if 'user_id' not in columns:
                # Add user_id column
                conn.execute(text("ALTER TABLE projects ADD COLUMN user_id INTEGER"))
                conn.commit()

                # Assign all existing projects to admin user
                conn.execute(text("""
                    UPDATE projects
                    SET user_id = :admin_id
                    WHERE user_id IS NULL
                """), {"admin_id": admin_id})
                conn.commit()
                print(f"[OK] user_id column added and existing projects assigned to admin")
            else:
                print("[OK] user_id column already exists")
        except Exception as e:
            print(f"[ERROR] Error adding user_id to projects: {e}")
            return

        print(f"\n[5/6] Adding user_id column to sessions table...")
        try:
            # Check if column already exists
            result = conn.execute(text("PRAGMA table_info(sessions)"))
            columns = [row[1] for row in result.fetchall()]

            if 'user_id' not in columns:
                # Add user_id column
                conn.execute(text("ALTER TABLE sessions ADD COLUMN user_id INTEGER"))
                conn.commit()

                # Assign all existing sessions to admin user
                conn.execute(text("""
                    UPDATE sessions
                    SET user_id = :admin_id
                    WHERE user_id IS NULL
                """), {"admin_id": admin_id})
                conn.commit()
                print(f"[OK] user_id column added and existing sessions assigned to admin")
            else:
                print("[OK] user_id column already exists")
        except Exception as e:
            print(f"[ERROR] Error adding user_id to sessions: {e}")
            return

        print(f"\n[6/6] Adding created_by column to training_jobs table...")
        try:
            # Check if column already exists
            result = conn.execute(text("PRAGMA table_info(training_jobs)"))
            columns = [row[1] for row in result.fetchall()]

            if 'created_by' not in columns:
                # Add created_by column
                conn.execute(text("ALTER TABLE training_jobs ADD COLUMN created_by INTEGER"))
                conn.commit()

                # Assign all existing training jobs to admin user
                conn.execute(text("""
                    UPDATE training_jobs
                    SET created_by = :admin_id
                    WHERE created_by IS NULL
                """), {"admin_id": admin_id})
                conn.commit()
                print(f"[OK] created_by column added and existing jobs assigned to admin")
            else:
                print("[OK] created_by column already exists")
        except Exception as e:
            print(f"[ERROR] Error adding created_by to training_jobs: {e}")
            return

    print("\n" + "=" * 60)
    print("[OK] MIGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nDefault admin credentials:")
    print("  Email: admin@example.com")
    print("  Password: admin123")
    print("\n[WARNING]  IMPORTANT: Change the admin password after first login!")
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
