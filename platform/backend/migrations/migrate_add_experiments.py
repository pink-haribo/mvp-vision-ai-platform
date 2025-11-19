"""
Migration script to add Experiment tracking system.

This script:
1. Creates experiments table
2. Creates experiment_stars table
3. Creates experiment_notes table
4. Adds experiment_id to training_jobs table
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
    print("MIGRATION: Add Experiment Tracking System")
    print("=" * 60)

    with engine.connect() as conn:
        print("\n[1/4] Creating experiments table...")
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    tags JSON,

                    mlflow_experiment_id VARCHAR(100) UNIQUE,
                    mlflow_experiment_name VARCHAR(255),

                    num_runs INTEGER NOT NULL DEFAULT 0,
                    num_completed_runs INTEGER NOT NULL DEFAULT 0,
                    best_metrics JSON,

                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """))
            conn.commit()
            print("[OK] Experiments table created")
        except Exception as e:
            print(f"[ERROR] Error creating experiments table: {e}")
            return

        print("\n[2/4] Creating experiment_stars table...")
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS experiment_stars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    starred_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(experiment_id, user_id)
                )
            """))
            conn.commit()
            print("[OK] Experiment_stars table created")
        except Exception as e:
            print(f"[ERROR] Error creating experiment_stars table: {e}")
            return

        print("\n[3/4] Creating experiment_notes table...")
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS experiment_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """))
            conn.commit()
            print("[OK] Experiment_notes table created")
        except Exception as e:
            print(f"[ERROR] Error creating experiment_notes table: {e}")
            return

        print("\n[4/4] Adding experiment_id column to training_jobs table...")
        try:
            # Check if column already exists
            result = conn.execute(text("PRAGMA table_info(training_jobs)"))
            columns = [row[1] for row in result.fetchall()]

            if 'experiment_id' not in columns:
                # Add experiment_id column
                conn.execute(text("ALTER TABLE training_jobs ADD COLUMN experiment_id INTEGER"))
                conn.commit()
                print("[OK] experiment_id column added to training_jobs")
            else:
                print("[OK] experiment_id column already exists")
        except Exception as e:
            print(f"[ERROR] Error adding experiment_id to training_jobs: {e}")
            return

        print("\n[5/4] Creating indexes for performance...")
        try:
            # Create indexes for foreign keys and frequent queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_experiments_project_id
                ON experiments(project_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_experiments_mlflow_experiment_id
                ON experiments(mlflow_experiment_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_experiment_stars_experiment_id
                ON experiment_stars(experiment_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_experiment_stars_user_id
                ON experiment_stars(user_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_experiment_notes_experiment_id
                ON experiment_notes(experiment_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_jobs_experiment_id
                ON training_jobs(experiment_id)
            """))
            conn.commit()
            print("[OK] Indexes created")
        except Exception as e:
            print(f"[WARNING] Error creating indexes: {e}")

        print("\n[6/4] Verifying migration...")
        try:
            # Count experiments
            result = conn.execute(text("SELECT COUNT(*) FROM experiments"))
            exp_count = result.fetchone()[0]
            print(f"  Experiments: {exp_count}")

            # Count experiment_stars
            result = conn.execute(text("SELECT COUNT(*) FROM experiment_stars"))
            stars_count = result.fetchone()[0]
            print(f"  Experiment stars: {stars_count}")

            # Count experiment_notes
            result = conn.execute(text("SELECT COUNT(*) FROM experiment_notes"))
            notes_count = result.fetchone()[0]
            print(f"  Experiment notes: {notes_count}")

            # Check training_jobs experiment_id column
            result = conn.execute(text("PRAGMA table_info(training_jobs)"))
            columns = [row[1] for row in result.fetchall()]
            has_exp_id = 'experiment_id' in columns
            print(f"  Training_jobs has experiment_id: {has_exp_id}")

            print("\n[OK] Migration verification complete")
        except Exception as e:
            print(f"[WARNING] Error during verification: {e}")

    print("\n" + "=" * 60)
    print("[OK] MIGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nChanges applied:")
    print("  1. experiments table created")
    print("  2. experiment_stars table created (user favorites)")
    print("  3. experiment_notes table created (markdown notes)")
    print("  4. experiment_id added to training_jobs")
    print("  5. Indexes created for performance")
    print("\nNext steps:")
    print("  - Implement MLflowService for experiment tracking")
    print("  - Create Experiment API endpoints")
    print("  - Update frontend to support experiment management")
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
