# app/database/session.py
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging
from typing import Generator, List, Dict, Any, Optional

from ..config import get_settings

logger = logging.getLogger(__name__)

# Get database URL from settings
settings = get_settings()
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# Create engine with appropriate configuration
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    # SQLite specific configuration
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL query logging
    )
    
    # Enable foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
        
else:
    # PostgreSQL/MySQL configuration
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,   # Recycle connections after 1 hour
        echo=False
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import Base from models to ensure all models are loaded
from .models import Base

def init_db():
    """Initialize database - create all tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    
    Usage in FastAPI:
    ```python
    @app.get("/items/")
    def read_items(db: Session = Depends(get_db)):
        return db.query(Item).all()
    ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_context():
    """
    Context manager for database session
    
    Usage:
    ```python
    with get_db_context() as db:
        user = db.query(User).first()
    ```
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

class DatabaseManager:
    """
    Database manager for handling transactions and batch operations
    """
    
    def __init__(self):
        self.session: Optional[Session] = None
        
    def __enter__(self):
        self.session = SessionLocal()
        return self.session
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
        else:
            try:
                self.session.commit()
            except Exception:
                self.session.rollback()
                raise
            
    def bulk_insert(self, objects: List[Base]):
        """Bulk insert objects"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
            
        try:
            self.session.bulk_save_objects(objects)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Bulk insert failed: {e}")
            raise
            
    def bulk_update(self, mappings: List[Dict]):
        """Bulk update objects"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
            
        try:
            self.session.bulk_update_mappings(mappings)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Bulk update failed: {e}")
            raise

# Database utilities

def check_database_connection() -> bool:
    """Check if database is accessible"""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False

def get_database_info() -> Dict[str, Any]:
    """Get database information"""
    try:
        with engine.connect() as conn:
            # Get database version
            if SQLALCHEMY_DATABASE_URL.startswith("postgresql"):
                result = conn.execute("SELECT version()")
                version = result.scalar()
            elif SQLALCHEMY_DATABASE_URL.startswith("mysql"):
                result = conn.execute("SELECT VERSION()")
                version = result.scalar()
            elif SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
                result = conn.execute("SELECT sqlite_version()")
                version = f"SQLite {result.scalar()}"
            else:
                version = "Unknown"
                
            # Get table count
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            return {
                "connected": True,
                "database_type": engine.dialect.name,
                "version": version,
                "table_count": len(tables),
                "tables": tables,
                "pool_size": getattr(engine.pool, 'size', None),
                "checked_out_connections": getattr(engine.pool, 'checked_out', 0)
            }
            
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {
            "connected": False,
            "error": str(e)
        }

def reset_database():
    """Reset database - drop all tables and recreate"""
    try:
        logger.warning("Resetting database - all data will be lost!")
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database reset completed")
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise

def backup_database(backup_path: str):
    """
    Create database backup (SQLite only)
    
    For PostgreSQL/MySQL, use native backup tools
    """
    if not SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
        raise NotImplementedError("Backup only implemented for SQLite. Use native tools for PostgreSQL/MySQL")
        
    import shutil
    import os
    
    # Extract database file path from URL
    db_path = SQLALCHEMY_DATABASE_URL.replace("sqlite:///", "")
    
    if os.path.exists(db_path):
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
    else:
        raise FileNotFoundError(f"Database file not found: {db_path}")

def vacuum_database():
    """Optimize database (clean up and reclaim space)"""
    try:
        with engine.connect() as conn:
            if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
                conn.execute("VACUUM")
            elif SQLALCHEMY_DATABASE_URL.startswith("postgresql"):
                conn.execute("VACUUM ANALYZE")
            elif SQLALCHEMY_DATABASE_URL.startswith("mysql"):
                # For MySQL, optimize each table
                inspector = inspect(engine)
                for table in inspector.get_table_names():
                    conn.execute(f"OPTIMIZE TABLE {table}")
                    
        logger.info("Database optimization completed")
    except Exception as e:
        logger.error(f"Database optimization failed: {e}")
        raise

# Import utilities
from sqlalchemy import inspect
from typing import Dict, Any, List, Optional

# Create database on module import if it doesn't exist
try:
    # This will create tables if they don't exist
    init_db()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    # Don't raise here to allow the app to start even if DB is down
    # The app can handle this gracefully and retry later