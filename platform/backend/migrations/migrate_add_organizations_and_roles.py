"""
Migration script to add Organizations and 5-tier Role System.

This script:
1. Creates organizations table
2. Adds organization_id to users table
3. Adds organization_id to projects table
4. Adds avatar_name to users table
5. Converts system_role values to match UserRole enum
6. Creates a default organization and assigns existing users
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
from datetime import datetime

# Get DATABASE_URL from environment
database_url = os.getenv("DATABASE_URL", "sqlite:///../data/db/platform_vision.db")
print(f"Using database: {database_url}")

# Create engine
engine = create_engine(database_url)

def run_migration():
    """Run the migration."""
    print("=" * 60)
    print("MIGRATION: Add Organizations and 5-tier Role System")
    print("=" * 60)

    with engine.connect() as conn:
        print("\n[1/6] Creating organizations table...")
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS organizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(255) NOT NULL,
                    company VARCHAR(255) NOT NULL,
                    division VARCHAR(255),
                    max_users INTEGER,
                    max_storage_gb INTEGER,
                    max_gpu_hours_per_month INTEGER,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            print("[OK] Organizations table created")
        except Exception as e:
            print(f"[ERROR] Error creating organizations table: {e}")
            return

        print("\n[2/6] Creating default organization...")
        try:
            # Check if default organization already exists
            result = conn.execute(text("SELECT id FROM organizations WHERE company = 'Default'"))
            existing_org = result.fetchone()

            if existing_org:
                org_id = existing_org[0]
                print(f"[OK] Default organization already exists (ID: {org_id})")
            else:
                # Create default organization
                conn.execute(text("""
                    INSERT INTO organizations (name, company, division, created_at, updated_at)
                    VALUES (:name, :company, :division, :now, :now)
                """), {
                    "name": "Default Organization",
                    "company": "Default",
                    "division": "Engineering",
                    "now": datetime.utcnow()
                })
                conn.commit()

                # Get the newly created org ID
                result = conn.execute(text("SELECT id FROM organizations WHERE company = 'Default'"))
                org_id = result.fetchone()[0]
                print(f"[OK] Default organization created (ID: {org_id})")
        except Exception as e:
            print(f"[ERROR] Error creating default organization: {e}")
            return

        print(f"\n[3/6] Adding organization_id and avatar_name columns to users table...")
        try:
            # Check if columns already exist
            result = conn.execute(text("PRAGMA table_info(users)"))
            columns = [row[1] for row in result.fetchall()]

            if 'organization_id' not in columns:
                # Add organization_id column
                conn.execute(text("ALTER TABLE users ADD COLUMN organization_id INTEGER"))
                conn.commit()

                # Assign all existing users to default organization
                conn.execute(text("""
                    UPDATE users
                    SET organization_id = :org_id
                    WHERE organization_id IS NULL
                """), {"org_id": org_id})
                conn.commit()
                print(f"[OK] organization_id column added and existing users assigned to default org")
            else:
                print("[OK] organization_id column already exists")

            if 'avatar_name' not in columns:
                # Add avatar_name column
                conn.execute(text("ALTER TABLE users ADD COLUMN avatar_name VARCHAR(100)"))
                conn.commit()
                print(f"[OK] avatar_name column added")
            else:
                print("[OK] avatar_name column already exists")
        except Exception as e:
            print(f"[ERROR] Error adding columns to users: {e}")
            return

        print(f"\n[4/6] Adding organization_id column to projects table...")
        try:
            # Check if column already exists
            result = conn.execute(text("PRAGMA table_info(projects)"))
            columns = [row[1] for row in result.fetchall()]

            if 'organization_id' not in columns:
                # Add organization_id column
                conn.execute(text("ALTER TABLE projects ADD COLUMN organization_id INTEGER"))
                conn.commit()

                # Assign all existing projects to default organization
                conn.execute(text("""
                    UPDATE projects
                    SET organization_id = :org_id
                    WHERE organization_id IS NULL
                """), {"org_id": org_id})
                conn.commit()
                print(f"[OK] organization_id column added and existing projects assigned to default org")
            else:
                print("[OK] organization_id column already exists")
        except Exception as e:
            print(f"[ERROR] Error adding organization_id to projects: {e}")
            return

        print(f"\n[5/6] Converting system_role values to match UserRole enum...")
        try:
            # Map old values to new enum MEMBER NAMES (uppercase)
            # SQLAlchemy Enum stores the member name, not the value
            role_mappings = {
                'admin': 'ADMIN',
                'manager': 'MANAGER',
                'engineer': 'ENGINEER_I',  # Default engineer to engineer_i
                'engineer_ii': 'ENGINEER_II',
                'engineer_i': 'ENGINEER_I',
                'guest': 'GUEST',
                'user': 'ENGINEER_I',  # Map generic 'user' to engineer_i
                # Also map lowercase enum values to uppercase member names
                'ADMIN': 'ADMIN',
                'MANAGER': 'MANAGER',
                'ENGINEER_II': 'ENGINEER_II',
                'ENGINEER_I': 'ENGINEER_I',
                'GUEST': 'GUEST',
            }

            # Get all unique roles currently in use
            result = conn.execute(text("SELECT DISTINCT system_role FROM users"))
            existing_roles = [row[0] for row in result.fetchall() if row[0]]

            print(f"  Found existing roles: {existing_roles}")

            # Update each role to match enum member name
            for old_role in existing_roles:
                if old_role in role_mappings:
                    new_role = role_mappings[old_role]
                    if old_role != new_role:
                        conn.execute(text("""
                            UPDATE users
                            SET system_role = :new_role
                            WHERE system_role = :old_role
                        """), {"new_role": new_role, "old_role": old_role})
                        print(f"  Converted '{old_role}' -> '{new_role}'")
                    else:
                        print(f"  '{old_role}' already correct")
                else:
                    # Default unknown roles to GUEST (uppercase)
                    conn.execute(text("""
                        UPDATE users
                        SET system_role = 'GUEST'
                        WHERE system_role = :old_role
                    """), {"old_role": old_role})
                    print(f"  [WARNING] Unknown role '{old_role}' converted to 'GUEST'")

            conn.commit()
            print("[OK] System roles converted to match UserRole enum")
        except Exception as e:
            print(f"[ERROR] Error converting system_role values: {e}")
            return

        print(f"\n[6/6] Verifying migration...")
        try:
            # Count organizations
            result = conn.execute(text("SELECT COUNT(*) FROM organizations"))
            org_count = result.fetchone()[0]
            print(f"  Organizations: {org_count}")

            # Count users with organizations
            result = conn.execute(text("SELECT COUNT(*) FROM users WHERE organization_id IS NOT NULL"))
            users_with_org = result.fetchone()[0]
            print(f"  Users with organization: {users_with_org}")

            # Count projects with organizations
            result = conn.execute(text("SELECT COUNT(*) FROM projects WHERE organization_id IS NOT NULL"))
            projects_with_org = result.fetchone()[0]
            print(f"  Projects with organization: {projects_with_org}")

            # Show role distribution
            result = conn.execute(text("""
                SELECT system_role, COUNT(*) as count
                FROM users
                GROUP BY system_role
                ORDER BY count DESC
            """))
            print("\n  Role distribution:")
            for row in result.fetchall():
                print(f"    {row[0]}: {row[1]} users")

            print("\n[OK] Migration verification complete")
        except Exception as e:
            print(f"[WARNING] Error during verification: {e}")

    print("\n" + "=" * 60)
    print("[OK] MIGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nChanges applied:")
    print("  1. Organizations table created")
    print("  2. Default organization created")
    print("  3. organization_id added to users and projects")
    print("  4. avatar_name added to users")
    print("  5. system_role values converted to 5-tier system:")
    print("     - admin (all permissions)")
    print("     - manager (can grant roles below manager)")
    print("     - engineer_ii (advanced features)")
    print("     - engineer_i (basic features)")
    print("     - guest (limited: 1 project, 1 dataset)")
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
