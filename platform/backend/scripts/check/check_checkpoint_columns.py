"""Check if checkpoint columns exist in training_jobs table."""

import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="platform",
    user="admin",
    password="devpass"
)

cursor = conn.cursor()
cursor.execute(
    "SELECT column_name FROM information_schema.columns "
    "WHERE table_name = 'training_jobs' "
    "AND column_name IN ('best_checkpoint_path', 'last_checkpoint_path')"
)
columns = [row[0] for row in cursor.fetchall()]
print(f"Checkpoint columns in training_jobs: {columns}")

if 'last_checkpoint_path' in columns:
    print("OK: last_checkpoint_path column already exists")
else:
    print("WARN: last_checkpoint_path column does NOT exist - migration needed")

cursor.close()
conn.close()
