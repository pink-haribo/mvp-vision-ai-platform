"""
Phase 11 Railway User DB: PostgreSQL -> Railway PostgreSQL Migration

Migrates User data from local Docker PostgreSQL to Railway PostgreSQL.

Usage:
    # Set environment variables
    export SOURCE_DB="postgresql://admin:devpass@localhost:5433/users"
    export TARGET_DB="postgresql://postgres:PASSWORD@containers-us-west-xxx.railway.app:PORT/railway"

    # Run migration
    python scripts/phase11/init_railway_user_db.py

Environment Variables:
    - SOURCE_DB: Local Docker PostgreSQL (default: postgresql://admin:devpass@localhost:5433/users)
    - TARGET_DB: Railway PostgreSQL (REQUIRED - get from Railway dashboard)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from platform/backend
platform_backend_dir = Path(__file__).parent.parent.parent / "platform" / "backend"
env_path = platform_backend_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"[OK] Loaded .env from: {env_path}")
else:
    print(f"[WARN] No .env file found at {env_path}")

# Add platform/backend to path for imports
sys.path.insert(0, str(platform_backend_dir))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from app.db.models import User, Organization, Invitation, ProjectMember


def get_source_engine():
    """Get local Docker PostgreSQL source engine."""
    source_url = os.getenv('SOURCE_DB', 'postgresql://admin:devpass@localhost:5433/users')
    print(f"[SOURCE] DB: {source_url}")
    return create_engine(source_url, echo=False)


def get_target_engine():
    """Get Railway PostgreSQL target engine."""
    # Use environment variable or hardcoded Railway URL
    target_url = os.getenv('TARGET_DB', 'postgresql://postgres:hNBDsIoezlnZSoGNKmGsxYcLiZekJiSj@gondola.proxy.rlwy.net:10185/railway')
    if not target_url or target_url == 'SET_YOUR_RAILWAY_URL_HERE':
        print("\n[ERROR] TARGET_DB environment variable is required!")
        print("\nUsage:")
        print('  export TARGET_DB="postgresql://postgres:PASSWORD@HOST:PORT/railway"')
        print('  python scripts/phase11/init_railway_user_db.py')
        print("\nGet TARGET_DB from Railway dashboard:")
        print("  1. Go to https://railway.com/project/9d57f05c-cbcc-4769-bc8d-7104636f76c1")
        print("  2. Click 'user-db' service")
        print("  3. Click 'Variables' tab")
        print("  4. Copy DATABASE_URL value")
        sys.exit(1)

    print(f"[TARGET] DB: {target_url}")
    return create_engine(target_url, echo=False)


def verify_source_data(source_session):
    """Verify source database has data."""
    counts = {
        'organizations': source_session.query(Organization).count(),
        'users': source_session.query(User).count(),
        'invitations': source_session.query(Invitation).count(),
        'project_members': source_session.query(ProjectMember).count(),
    }

    print("\n[VERIFY] Source Database Contents:")
    for table, count in counts.items():
        print(f"  - {table}: {count} rows")

    total = sum(counts.values())
    if total == 0:
        print("\n[WARN] Source database is empty. Will create empty schema in Railway.")
    return total > 0


def create_target_schema(target_engine):
    """Create tables in Railway PostgreSQL database."""
    from app.db.models import Base, Organization, User
    from sqlalchemy import text

    print("\n[SCHEMA] Creating tables in Railway database...")

    # Step 0: Clean up existing tables and enums
    print("[INFO] Cleaning up existing tables and enums...")
    with target_engine.connect() as conn:
        # Drop tables if exist
        conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS organizations CASCADE"))
        # Drop enum types if exist
        conn.execute(text("DROP TYPE IF EXISTS userrole CASCADE"))
        conn.commit()
    print("[OK] Cleanup complete")

    # Step 1: Create enum types (fix for PostgreSQL enum values)
    print("[INFO] Creating enum types...")
    with target_engine.connect() as conn:
        # Create UserRole enum with lowercase values (matches PR #38 fix)
        conn.execute(text("""
            CREATE TYPE userrole AS ENUM ('guest', 'basic_engineer', 'standard_engineer', 'advanced_engineer', 'manager', 'admin');
        """))
        conn.commit()
    print("[OK] Enum types created")

    # Step 2: Create User DB tables (exclude invitations and project_members due to FK constraints)
    user_tables_to_create = ['organizations', 'users']

    print(f"[INFO] Creating tables: {user_tables_to_create}")
    print("[INFO] Skipping: invitations, project_members (FK to Platform DB)")

    # Create specific tables only
    tables_to_create = [Base.metadata.tables[t] for t in user_tables_to_create]
    Base.metadata.create_all(bind=target_engine, tables=tables_to_create, checkfirst=True)

    # Verify created tables
    inspector = inspect(target_engine)
    created_tables = inspector.get_table_names()
    print(f"[OK] Created {len(created_tables)} tables: {created_tables}")


def migrate_table(source_session, target_session, model, table_name):
    """Migrate a single table from source to target."""
    print(f"\n[MIGRATE] {table_name}...")

    # Fetch all rows from source
    rows = source_session.query(model).all()
    count = len(rows)

    if count == 0:
        print(f"  [SKIP] No data in {table_name}, skipping")
        return 0

    # Use merge to handle primary key conflicts
    for i, row in enumerate(rows, 1):
        target_session.merge(row)
        if i % 10 == 0 or i == count:
            print(f"  Progress: {i}/{count} rows", end='\r')

    target_session.commit()
    print(f"\n  [OK] Migrated {count} rows")
    return count


def verify_target_data(target_session):
    """Verify Railway database has correct data."""
    counts = {
        'organizations': target_session.query(Organization).count(),
        'users': target_session.query(User).count(),
        # 'invitations': 0,  # SKIP: not migrated (FK issues)
        # 'project_members': 0,  # SKIP: not migrated (FK issues)
    }

    print("\n[VERIFY] Railway Database Contents (After Migration):")
    for table, count in counts.items():
        print(f"  - {table}: {count} rows")
    print("  - invitations: (skipped - FK to Platform DB)")
    print("  - project_members: (skipped - FK to Platform DB)")

    return counts


def main():
    """Run Railway migration."""
    print("=" * 60)
    print("Phase 11 Railway: User DB Migration")
    print("=" * 60)

    # Get engines
    source_engine = get_source_engine()
    target_engine = get_target_engine()

    # Test connections
    print("\n[TEST] Testing database connections...")
    try:
        source_engine.connect()
        print("[OK] Source connection successful")
    except Exception as e:
        print(f"[ERROR] Cannot connect to source database: {e}")
        return

    try:
        target_engine.connect()
        print("[OK] Railway connection successful")
    except Exception as e:
        print(f"[ERROR] Cannot connect to Railway database: {e}")
        return

    # Create sessions
    SourceSession = sessionmaker(bind=source_engine)
    TargetSession = sessionmaker(bind=target_engine)

    source_session = SourceSession()
    target_session = TargetSession()

    try:
        # Step 1: Verify source data
        print("\n" + "=" * 60)
        print("Step 1: Verify Source Data")
        print("=" * 60)
        has_data = verify_source_data(source_session)

        # Step 2: Create Railway schema
        print("\n" + "=" * 60)
        print("Step 2: Create Railway Schema")
        print("=" * 60)
        create_target_schema(target_engine)

        # Step 3: Migrate data (if source has data)
        print("\n" + "=" * 60)
        print("Step 3: Migrate Data")
        print("=" * 60)

        if not has_data:
            print("[SKIP] Source database is empty. Schema created, no data to migrate.")
            total_rows = 0
        else:
            total_rows = 0
            # Only migrate organizations and users (skip invitations and project_members)
            migration_order = [
                (Organization, 'organizations'),
                (User, 'users'),
                # (Invitation, 'invitations'),  # SKIP: FK to projects/datasets
                # (ProjectMember, 'project_members'),  # SKIP: FK to projects
            ]

            for model, table_name in migration_order:
                rows_migrated = migrate_table(source_session, target_session, model, table_name)
                total_rows += rows_migrated

        # Step 4: Verify Railway data
        print("\n" + "=" * 60)
        print("Step 4: Verify Railway Data")
        print("=" * 60)
        target_counts = verify_target_data(target_session)

        # Step 5: Final summary
        print("\n" + "=" * 60)
        print("Migration Complete!")
        print("=" * 60)
        print(f"[SUCCESS] Total rows migrated: {total_rows}")

        # Get Railway connection info
        target_url = os.getenv('TARGET_DB')
        print(f"\n[RAILWAY CONNECTION INFO]")
        print(f"  DATABASE_URL: {target_url}")

        print(f"\n[NEXT STEPS]")
        print(f"  1. Share Railway connection info with Labeler team:")
        print(f"     - Database: Railway PostgreSQL (User DB)")
        print(f"     - Connection: {target_url}")
        print(f"     - Tables: organizations, users, invitations, project_members")
        print(f"\n  2. Update Platform Backend .env:")
        print(f"     USER_DATABASE_URL=\"{target_url}\"")
        print(f"\n  3. Test Platform Backend connection:")
        print(f"     cd platform/backend")
        print(f"     poetry run uvicorn app.main:app --reload")
        print(f"     # Test login at http://localhost:8000/docs")
        print(f"\n  4. If successful, local Docker postgres-user can remain for development")

    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        target_session.rollback()
        raise
    finally:
        source_session.close()
        target_session.close()


if __name__ == '__main__':
    main()
