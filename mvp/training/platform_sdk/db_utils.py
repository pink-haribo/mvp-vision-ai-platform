"""
Database connection utilities for Training Service.

Provides centralized database connection management for both local (SQLite)
and production (PostgreSQL) environments.
"""

import os
from pathlib import Path
from typing import Tuple, Any


def get_db_connection() -> Tuple[Any, Any, str]:
    """
    Get database connection for training code.

    This function:
    1. Checks DATABASE_URL environment variable
    2. If PostgreSQL URL -> connects to PostgreSQL
    3. If SQLite URL or not set -> raises error (must be explicit)

    NO FALLBACK. If DATABASE_URL is wrong or missing, fail loudly.

    Returns:
        Tuple[connection, cursor, placeholder]:
            - connection: Database connection object
            - cursor: Database cursor object
            - placeholder: SQL placeholder style ('%s' for PostgreSQL, '?' for SQLite)

    Raises:
        ValueError: If DATABASE_URL is not set or invalid
        FileNotFoundError: If SQLite database file doesn't exist
        ImportError: If required database driver is not installed
    """
    database_url = os.getenv('DATABASE_URL')

    if not database_url:
        raise ValueError(
            "DATABASE_URL environment variable not set. "
            "Training code requires explicit database configuration. "
            "Set DATABASE_URL to either:\n"
            "  - PostgreSQL URL (Railway): postgresql://user:pass@host:port/db\n"
            "  - SQLite URL (local): sqlite:///path/to/database.db"
        )

    # PostgreSQL (Railway production)
    if database_url.startswith('postgresql://') or database_url.startswith('postgres://'):
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            )

        try:
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()
            print(f"[DB] Connected to PostgreSQL")
            return conn, cursor, '%s'
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to PostgreSQL database: {e}\n"
                f"Check DATABASE_URL format and database availability."
            )

    # SQLite (local development)
    elif database_url.startswith('sqlite:///'):
        import sqlite3

        # Extract path from SQLite URL
        db_path = database_url.replace('sqlite:///', '')
        db_path = Path(db_path)

        if not db_path.exists():
            raise FileNotFoundError(
                f"SQLite database not found at: {db_path}\n"
                f"Run 'python mvp/backend/init_db.py' to create the database."
            )

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            print(f"[DB] Connected to SQLite: {db_path}")
            return conn, cursor, '?'
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to SQLite database: {e}\n"
                f"Check database file permissions and integrity."
            )

    # Unknown database type
    else:
        raise ValueError(
            f"Unsupported DATABASE_URL format: {database_url}\n"
            f"Supported formats:\n"
            f"  - postgresql://user:pass@host:port/db (PostgreSQL)\n"
            f"  - sqlite:///path/to/database.db (SQLite)"
        )


def close_db_connection(conn: Any, cursor: Any) -> None:
    """
    Close database connection and cursor.

    Args:
        conn: Database connection object
        cursor: Database cursor object
    """
    try:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    except Exception as e:
        print(f"[WARNING] Error closing database connection: {e}")
