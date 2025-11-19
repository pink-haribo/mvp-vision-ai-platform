"""Add badge_color column to users table."""

import random
from sqlalchemy import create_engine, text
from app.core.config import settings

# Badge colors
BADGE_COLORS = [
    'red', 'orange', 'amber', 'yellow',
    'lime', 'green', 'emerald', 'teal',
    'cyan', 'sky', 'blue', 'indigo',
    'violet', 'purple', 'fuchsia', 'pink'
]

def migrate():
    """Add badge_color column and assign random colors to existing users."""
    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text(
            "SELECT COUNT(*) FROM pragma_table_info('users') WHERE name='badge_color'"
        ))
        exists = result.scalar() > 0

        if exists:
            print("[OK] badge_color column already exists")
        else:
            # Add column
            print("Adding badge_color column...")
            conn.execute(text("ALTER TABLE users ADD COLUMN badge_color VARCHAR(20)"))
            conn.commit()
            print("[OK] Column added")

        # Assign random colors to users with NULL badge_color
        print("Assigning random colors to existing users...")
        result = conn.execute(text("SELECT id FROM users WHERE badge_color IS NULL"))
        user_ids = [row[0] for row in result]

        if user_ids:
            for user_id in user_ids:
                color = random.choice(BADGE_COLORS)
                conn.execute(
                    text("UPDATE users SET badge_color = :color WHERE id = :user_id"),
                    {"color": color, "user_id": user_id}
                )
            conn.commit()
            print(f"[OK] Assigned colors to {len(user_ids)} users")
        else:
            print("[OK] All users already have badge colors")

        print("\n[SUCCESS] Migration completed successfully!")

if __name__ == "__main__":
    migrate()
