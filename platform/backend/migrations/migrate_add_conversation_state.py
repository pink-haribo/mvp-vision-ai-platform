"""
Add conversation state management to sessions table

This migration adds:
- state: conversation state (initial, gathering_config, selecting_project, etc.)
- temp_data: JSON field for temporary conversation data

Run: python migrate_add_conversation_state.py
"""

from app.db.database import engine
from sqlalchemy import text

def upgrade():
    """Add conversation state columns"""
    with engine.connect() as conn:
        # Check each column individually
        result = conn.execute(text(
            "SELECT name FROM pragma_table_info('sessions') WHERE name = 'state'"
        ))
        has_state = result.fetchone() is not None

        result = conn.execute(text(
            "SELECT name FROM pragma_table_info('sessions') WHERE name = 'temp_data'"
        ))
        has_temp_data = result.fetchone() is not None

        # Add state column if missing
        if not has_state:
            conn.execute(text(
                "ALTER TABLE sessions ADD COLUMN state VARCHAR(50) DEFAULT 'initial' NOT NULL"
            ))
            print("[OK] Added 'state' column")
        else:
            print("[SKIP] 'state' column already exists")

        # Add temp_data column if missing
        if not has_temp_data:
            conn.execute(text(
                "ALTER TABLE sessions ADD COLUMN temp_data TEXT DEFAULT '{}' NOT NULL"
            ))
            print("[OK] Added 'temp_data' column")
        else:
            print("[SKIP] 'temp_data' column already exists")

        # Create index on state
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_sessions_state ON sessions(state)"
        ))
        print("[OK] Created/verified index on 'state'")

        # Update existing sessions to have initial state and empty temp_data
        if not has_state or not has_temp_data:
            conn.execute(text(
                "UPDATE sessions SET state = COALESCE(state, 'initial'), temp_data = COALESCE(temp_data, '{}') WHERE state IS NULL OR temp_data IS NULL"
            ))
            print("[OK] Updated existing sessions")

        conn.commit()
        print("[OK] Migration completed successfully!")

def downgrade():
    """Remove conversation state columns"""
    with engine.connect() as conn:
        # SQLite doesn't support DROP COLUMN easily
        # We'll create a new table without these columns and copy data

        print("[WARNING] Downgrade not fully supported in SQLite")
        print("To rollback, restore from backup or recreate database")

if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: Add Conversation State")
    print("=" * 60)

    try:
        upgrade()
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        raise
