"""
Migration script to add Invitation system.

This script:
1. Creates invitations table
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
    print("MIGRATION: Add Invitation System")
    print("=" * 60)

    with engine.connect() as conn:
        print("\n[1/2] Creating invitations table...")
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS invitations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token VARCHAR(255) NOT NULL UNIQUE,

                    invitation_type VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',

                    organization_id INTEGER,
                    project_id INTEGER,
                    dataset_id VARCHAR(100),

                    inviter_id INTEGER NOT NULL,
                    invitee_email VARCHAR(255) NOT NULL,
                    invitee_id INTEGER,

                    invitee_role VARCHAR(20) NOT NULL DEFAULT 'guest',

                    message TEXT,

                    expires_at DATETIME NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    accepted_at DATETIME,

                    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
                    FOREIGN KEY (inviter_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (invitee_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """))
            conn.commit()
            print("[OK] Invitations table created")
        except Exception as e:
            print(f"[ERROR] Error creating invitations table: {e}")
            return

        print("\n[2/2] Creating indexes for performance...")
        try:
            # Create indexes for foreign keys and frequent queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_invitations_token
                ON invitations(token)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_invitations_invitation_type
                ON invitations(invitation_type)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_invitations_status
                ON invitations(status)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_invitations_organization_id
                ON invitations(organization_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_invitations_project_id
                ON invitations(project_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_invitations_dataset_id
                ON invitations(dataset_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_invitations_inviter_id
                ON invitations(inviter_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_invitations_invitee_email
                ON invitations(invitee_email)
            """))
            conn.commit()
            print("[OK] Indexes created")
        except Exception as e:
            print(f"[WARNING] Error creating indexes: {e}")

        print("\n[3/2] Verifying migration...")
        try:
            # Count invitations
            result = conn.execute(text("SELECT COUNT(*) FROM invitations"))
            inv_count = result.fetchone()[0]
            print(f"  Invitations: {inv_count}")

            # Check table structure
            result = conn.execute(text("PRAGMA table_info(invitations)"))
            columns = [row[1] for row in result.fetchall()]
            print(f"  Columns: {len(columns)}")

            # Check required columns exist
            required_columns = ['token', 'invitation_type', 'status', 'inviter_id', 'invitee_email', 'expires_at']
            missing = [col for col in required_columns if col not in columns]
            if missing:
                print(f"  [WARNING] Missing columns: {missing}")
            else:
                print(f"  [OK] All required columns present")

            print("\n[OK] Migration verification complete")
        except Exception as e:
            print(f"[WARNING] Error during verification: {e}")

    print("\n" + "=" * 60)
    print("[OK] MIGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nChanges applied:")
    print("  1. invitations table created")
    print("  2. Indexes created for performance")
    print("\nInvitation types supported:")
    print("  - ORGANIZATION: Invite user to organization")
    print("  - PROJECT: Invite user to project")
    print("  - DATASET: Invite user to dataset")
    print("\nNext steps:")
    print("  - Implement Email Service for sending invitations")
    print("  - Create Invitation API endpoints")
    print("  - Update Auth API for signup-with-invitation")
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
