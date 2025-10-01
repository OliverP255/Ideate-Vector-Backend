"""
Database connection and initialization.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.services.config import get_settings

logger = logging.getLogger(__name__)

# Global database objects
engine = None
SessionLocal = None
Base = declarative_base()


async def init_db():
    """Initialize database connection and create tables."""
    global engine, SessionLocal
    
    try:
        settings = get_settings()
        
        # Create database URL
        database_url = (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        
        # Create engine
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=settings.debug
        )
        
        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.warning(f"Database initialization failed (running in development mode): {e}")
        # In development mode without Docker, we'll continue without database
        # This allows the API to start and health checks to work
        engine = None
        SessionLocal = None


def get_db():
    """Get database session."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
