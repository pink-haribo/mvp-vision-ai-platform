"""Database initialization script for Vision AI Training Platform.

This script creates all required database tables and optionally creates an admin user.

Usage:
    python init_db.py                 # Initialize both databases
    python init_db.py --platform-only # Initialize platform DB only
    python init_db.py --user-only     # Initialize user DB only
    python init_db.py --reset         # Drop and recreate all tables + create admin user (CAUTION!)

Environment Variables:
    DATABASE_URL - PostgreSQL connection string for Platform DB
    USER_DATABASE_URL - PostgreSQL connection string for User DB
"""

import sys
import argparse
from datetime import datetime
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

from app.core.config import settings
from app.db.database import Base
from app.db.models import (
    Organization, Invitation, User, ProjectMember, UserRole,
    DatasetSnapshot, Project, Experiment, ExperimentStar, ExperimentNote,
    Session, Message, TrainingJob, TrainingMetric, TrainingLog,
    ValidationResult, ValidationImageResult, TestRun, TestImageResult,
    InferenceJob, InferenceResult, ExportJob, DeploymentTarget, DeploymentHistory
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_table_names(engine):
    """Get list of existing tables in database."""
    inspector = inspect(engine)
    return inspector.get_table_names()


def drop_enum_types(engine):
    """Drop PostgreSQL enum types to allow recreation with new values."""
    enum_types = ['userrole', 'invitationtype', 'invitationstatus',
                  'exportformat', 'exportjobstatus', 'deploymenttype',
                  'deploymentstatus', 'deploymenteventtype']

    with engine.connect() as conn:
        for enum_type in enum_types:
            try:
                conn.execute(text(f"DROP TYPE IF EXISTS {enum_type} CASCADE"))
                print(f"  Dropped enum type: {enum_type}")
            except Exception as e:
                print(f"  Warning: Could not drop {enum_type}: {e}")
        conn.commit()


def create_admin_user(engine):
    """Create default admin user with credentials: admin@example.com / admin123"""
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # Check if admin user already exists
        existing = db.query(User).filter(User.email == "admin@example.com").first()

        if existing:
            print("ℹ️  Admin user already exists")
            print(f"   Email: {existing.email}")
            print(f"   Role: {existing.system_role}")
            return

        # Create admin user
        hashed_password = pwd_context.hash("admin123")

        admin_user = User(
            email="admin@example.com",
            hashed_password=hashed_password,
            full_name="Admin User",
            system_role=UserRole.ADMIN,
            is_active=True,
            badge_color="violet",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)

        print("✅ Admin user created successfully!")
        print(f"   Email: admin@example.com")
        print(f"   Password: admin123")
        print(f"   Role: {admin_user.system_role.value}")
        print(f"   User ID: {admin_user.id}")

        # Create default "Uncategorized" project for admin
        uncategorized = Project(
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

        print()
        print("✅ Default project created!")
        print(f"   Project: {uncategorized.name}")
        print(f"   Project ID: {uncategorized.id}")

    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        db.rollback()
        raise

    finally:
        db.close()


def create_platform_db(engine, reset=False):
    """Create Platform DB tables.

    Args:
        engine: SQLAlchemy engine for Platform DB
        reset: If True, drop all existing tables first (CAUTION!)
    """
    print("=" * 60)
    print("Platform Database Initialization")
    print("=" * 60)
    print(f"Database URL: {settings.DATABASE_URL}")
    print()

    # Get existing tables
    existing_tables = get_table_names(engine)
    print(f"Existing tables: {len(existing_tables)}")
    if existing_tables:
        print(f"  {', '.join(existing_tables)}")
    print()

    if reset:
        print("⚠️  WARNING: Dropping all existing tables and enum types...")
        confirmation = input("Are you sure? Type 'yes' to confirm: ")
        if confirmation.lower() != 'yes':
            print("Aborted.")
            return False

        # Drop tables
        Base.metadata.drop_all(bind=engine)
        print("✅ All tables dropped.")

        # Drop enum types (PostgreSQL specific)
        print("Dropping enum types...")
        drop_enum_types(engine)
        print("✅ All enum types dropped.")
        print()

    # Create all tables
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

    # Verify creation
    new_tables = get_table_names(engine)
    print()
    print(f"✅ Database initialized successfully!")
    print(f"Total tables: {len(new_tables)}")
    print()
    print("Tables created:")
    for table in sorted(new_tables):
        print(f"  - {table}")
    print()

    return True


def create_user_db(engine, reset=False):
    """Create User DB tables.

    Note: Currently uses same schema as Platform DB for shared User model.
    In production, you may want to separate User-specific tables.

    Args:
        engine: SQLAlchemy engine for User DB
        reset: If True, drop all existing tables first (CAUTION!)
    """
    print("=" * 60)
    print("User Database Initialization")
    print("=" * 60)
    print(f"Database URL: {settings.USER_DATABASE_URL}")
    print()

    # Get existing tables
    existing_tables = get_table_names(engine)
    print(f"Existing tables: {len(existing_tables)}")
    if existing_tables:
        print(f"  {', '.join(existing_tables)}")
    print()

    if reset:
        print("⚠️  WARNING: Dropping all existing tables...")
        confirmation = input("Are you sure? Type 'yes' to confirm: ")
        if confirmation.lower() != 'yes':
            print("Aborted.")
            return False
        Base.metadata.drop_all(bind=engine)
        print("✅ All tables dropped.")
        print()

    # Create tables
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

    # Verify creation
    new_tables = get_table_names(engine)
    print()
    print(f"✅ User database initialized successfully!")
    print(f"Total tables: {len(new_tables)}")
    print()
    print("Tables created:")
    for table in sorted(new_tables):
        print(f"  - {table}")
    print()

    return True


def test_connection(engine, db_name):
    """Test database connection."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        print(f"✅ {db_name} connection successful")
        return True
    except Exception as e:
        print(f"❌ {db_name} connection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Initialize Vision AI Platform databases")
    parser.add_argument('--platform-only', action='store_true', help='Initialize Platform DB only')
    parser.add_argument('--user-only', action='store_true', help='Initialize User DB only')
    parser.add_argument('--reset', action='store_true', help='Drop and recreate all tables + create admin (CAUTION!)')
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("Vision AI Training Platform - Database Initialization")
    print("=" * 60)
    print()

    # Determine which databases to initialize
    init_platform = not args.user_only
    init_user = not args.platform_only

    if args.reset:
        print("⚠️  RESET MODE: All existing data will be deleted!")
        print("   After reset, admin user will be created automatically.")
        print()

    success = True
    platform_engine = None

    # Initialize Platform DB
    if init_platform:
        try:
            platform_engine = create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                echo=False
            )

            if not test_connection(platform_engine, "Platform DB"):
                success = False
            else:
                print()
                db_created = create_platform_db(platform_engine, reset=args.reset)

                # Create admin user after reset
                if db_created and args.reset:
                    print("Creating admin user...")
                    print()
                    create_admin_user(platform_engine)
                    print()

        except Exception as e:
            print(f"❌ Platform DB initialization failed: {e}")
            success = False

    # Initialize User DB (if separate from Platform DB)
    if init_user and settings.USER_DATABASE_URL:
        # Only initialize if User DB is different from Platform DB
        if settings.USER_DATABASE_URL != settings.DATABASE_URL:
            try:
                user_engine = create_engine(
                    settings.USER_DATABASE_URL,
                    pool_pre_ping=True,
                    echo=False
                )

                if not test_connection(user_engine, "User DB"):
                    success = False
                else:
                    print()
                    create_user_db(user_engine, reset=args.reset)
                    user_engine.dispose()
            except Exception as e:
                print(f"❌ User DB initialization failed: {e}")
                success = False
        else:
            print("ℹ️  User DB uses same database as Platform DB (skipping separate initialization)")
            print()

    # Cleanup
    if platform_engine:
        platform_engine.dispose()

    # Summary
    print("=" * 60)
    if success:
        print("✅ Database initialization completed successfully!")
        print()
        if args.reset:
            print("===========================================")
            print("Login Credentials:")
            print("  Email: admin@example.com")
            print("  Password: admin123")
            print("===========================================")
            print()
        print("Next steps:")
        print("1. Start the backend server:")
        print("   poetry run uvicorn app.main:app --reload --port 8000")
        print()
        print("2. Start the Temporal worker:")
        print("   poetry run python -m app.workflows.worker")
        print()
        print("3. Access the API documentation:")
        print("   http://localhost:8000/docs")
    else:
        print("❌ Database initialization completed with errors.")
        print("Please check the error messages above and try again.")
        sys.exit(1)
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
