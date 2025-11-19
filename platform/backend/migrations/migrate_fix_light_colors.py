"""Fix light badge colors (amber, yellow, lime) to darker alternatives."""

import random
from sqlalchemy import create_engine, text
from app.core.config import settings

# Valid dark badge colors
DARK_COLORS = [
    'red', 'orange', 'green', 'emerald',
    'teal', 'cyan', 'sky', 'blue',
    'indigo', 'violet', 'purple', 'fuchsia', 'pink'
]

# Light colors to replace
LIGHT_COLORS = ['amber', 'yellow', 'lime']

def migrate():
    """Replace light badge colors with dark alternatives."""
    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        # Find users with light colors
        result = conn.execute(
            text("SELECT id, email, badge_color FROM users WHERE badge_color IN ('amber', 'yellow', 'lime')")
        )
        users = list(result)

        if not users:
            print("[OK] No users with light colors found")
            return

        print(f"Found {len(users)} users with light colors")

        # Update each user
        for user_id, email, old_color in users:
            new_color = random.choice(DARK_COLORS)
            conn.execute(
                text("UPDATE users SET badge_color = :new_color WHERE id = :user_id"),
                {"new_color": new_color, "user_id": user_id}
            )
            print(f"  {email}: {old_color} -> {new_color}")

        conn.commit()
        print(f"\n[SUCCESS] Updated {len(users)} users")

if __name__ == "__main__":
    migrate()
