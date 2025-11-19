"""
Migrate existing sessions to Phase 1+2 architecture

This script migrates existing sessions by:
1. Setting state to 'initial' for all sessions
2. Clearing temp_data
3. Preserving all messages and training jobs

Run: python migrate_existing_sessions.py
"""

from app.db.database import engine
from sqlalchemy import text

def migrate():
    """Migrate existing sessions to new architecture"""
    with engine.connect() as conn:
        # Check if columns exist
        result = conn.execute(text(
            "SELECT name FROM pragma_table_info('sessions') WHERE name = 'state'"
        ))
        has_state = result.fetchone() is not None

        result = conn.execute(text(
            "SELECT name FROM pragma_table_info('sessions') WHERE name = 'temp_data'"
        ))
        has_temp_data = result.fetchone() is not None

        if not has_state or not has_temp_data:
            print("[ERROR] State/temp_data columns not found!")
            print("[INFO] Please run migrate_add_conversation_state.py first")
            return False

        # Get count of sessions
        result = conn.execute(text("SELECT COUNT(*) FROM sessions"))
        session_count = result.fetchone()[0]

        print(f"[INFO] Found {session_count} sessions to migrate")

        if session_count == 0:
            print("[OK] No sessions to migrate")
            return True

        # Reset all sessions to initial state
        conn.execute(text(
            """
            UPDATE sessions
            SET state = 'initial',
                temp_data = '{}'
            WHERE state IS NULL OR state != 'initial'
            """
        ))

        affected = conn.execute(text(
            "SELECT changes()"
        )).fetchone()[0]

        print(f"[OK] Migrated {affected} sessions to initial state")
        print("[OK] All existing messages and training jobs preserved")

        conn.commit()
        return True


def verify_migration():
    """Verify migration was successful"""
    with engine.connect() as conn:
        # Check state distribution
        result = conn.execute(text(
            "SELECT state, COUNT(*) FROM sessions GROUP BY state"
        ))

        print("\n[VERIFICATION] Session state distribution:")
        for row in result:
            print(f"  - {row[0]}: {row[1]} sessions")

        # Check temp_data
        result = conn.execute(text(
            "SELECT COUNT(*) FROM sessions WHERE temp_data = '{}'"
        ))
        empty_count = result.fetchone()[0]

        print(f"\n[VERIFICATION] {empty_count} sessions have empty temp_data")

        # Check messages preserved
        result = conn.execute(text(
            "SELECT COUNT(*) FROM messages"
        ))
        message_count = result.fetchone()[0]

        print(f"[VERIFICATION] {message_count} messages preserved")

        # Check training jobs preserved
        result = conn.execute(text(
            "SELECT COUNT(*) FROM training_jobs"
        ))
        job_count = result.fetchone()[0]

        print(f"[VERIFICATION] {job_count} training jobs preserved")


if __name__ == "__main__":
    print("=" * 60)
    print("Session Migration: Phase 1+2 Architecture")
    print("=" * 60)

    try:
        success = migrate()

        if success:
            verify_migration()
            print("\n[OK] Migration completed successfully!")
        else:
            print("\n[ERROR] Migration failed!")

    except Exception as e:
        print(f"[ERROR] Migration error: {e}")
        raise
