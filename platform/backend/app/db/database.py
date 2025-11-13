"""Database connection and session management."""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# Detect database type from URL
is_sqlite = settings.DATABASE_URL.startswith("sqlite")

# Create engine with database-specific configuration
if is_sqlite:
    # SQLite-specific configuration
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={
            "check_same_thread": False,  # Needed for SQLite
            "timeout": 30.0,  # 30 second timeout for locks
        },
    )

    # Enable WAL mode for better concurrent access
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.close()
else:
    # PostgreSQL/other databases - use connection pooling
    engine = create_engine(
        settings.DATABASE_URL,
        pool_size=5,  # Number of connections to maintain
        max_overflow=10,  # Maximum overflow connections
        pool_pre_ping=True,  # Verify connections before using
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database - create all tables."""
    Base.metadata.create_all(bind=engine)
