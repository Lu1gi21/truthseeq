"""
Database configuration and connection management for TruthSeeQ.

This module provides:
- Async database connection setup
- Session management
- Database initialization utilities
- Connection pooling configuration
"""

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """
    Base class for all database models.
    
    Provides common functionality and metadata configuration for all tables.
    """
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )


class DatabaseManager:
    """
    Database connection and session management.
    
    Handles database engine creation, connection pooling, and session lifecycle
    for the TruthSeeQ application.
    """
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    
    def create_engine(self, database_url: str, echo: bool = False) -> AsyncEngine:
        """
        Create and configure the database engine.
        
        Args:
            database_url: PostgreSQL connection URL
            echo: Whether to log SQL statements
            
        Returns:
            Configured AsyncEngine instance
            
        Raises:
            ValueError: If database_url is invalid
        """
        if not database_url:
            raise ValueError("Database URL cannot be empty")
        
        # Configure engine with connection pooling
        engine_kwargs = {
            "url": database_url,
            "echo": echo,
            "future": True,  # Use SQLAlchemy 2.0 style
            "pool_pre_ping": True,  # Verify connections before use
            "pool_recycle": 300,  # Recycle connections after 5 minutes
            "pool_size": 10,  # Number of connections to maintain
            "max_overflow": 20,  # Additional connections allowed
        }
        
        # Use NullPool for testing to avoid connection issues
        if "sqlite" in database_url or "test" in database_url:
            engine_kwargs["poolclass"] = NullPool
        
        self._engine = create_async_engine(**engine_kwargs)
        
        # Create session factory
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
        
        logger.info(f"Database engine created for: {database_url.split('@')[-1]}")
        return self._engine
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        Yields:
            AsyncSession: Database session instance
            
        Raises:
            RuntimeError: If engine is not initialized
        """
        if not self._session_factory:
            raise RuntimeError("Database engine not initialized. Call create_engine first.")
        
        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """
        Perform a database health check.
        
        Returns:
            True if database is accessible, False otherwise
        """
        if not self._engine:
            return False
        
        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close the database engine and all connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database engine closed")
    
    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Get the current database engine."""
        return self._engine


# Global database manager instance
db_manager = DatabaseManager()


async def init_database() -> None:
    """
    Initialize the database connection.
    
    Creates the database engine and session factory using configuration
    from settings.
    
    Raises:
        ValueError: If database configuration is invalid
        ConnectionError: If database connection fails
    """
    try:
        database_url = settings.get_database_url()
        echo_sql = settings.DEBUG and settings.DATABASE_ECHO
        
        engine = db_manager.create_engine(database_url, echo=echo_sql)
        
        # Test the connection
        health_ok = await db_manager.health_check()
        if not health_ok:
            raise ConnectionError("Failed to connect to database")
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting database sessions.
    
    Yields:
        AsyncSession: Database session for request handling
        
    Example:
        ```python
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            # Use db session here
            pass
        ```
    """
    async for session in db_manager.get_session():
        yield session


async def create_tables() -> None:
    """
    Create all database tables.
    
    This should typically be done through Alembic migrations in production,
    but can be useful for development and testing.
    
    Raises:
        RuntimeError: If database engine is not initialized
    """
    if not db_manager.engine:
        raise RuntimeError("Database engine not initialized")
    
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created")


async def drop_tables() -> None:
    """
    Drop all database tables.
    
    ⚠️ WARNING: This will delete all data! Use with caution.
    
    Raises:
        RuntimeError: If database engine is not initialized
    """
    if not db_manager.engine:
        raise RuntimeError("Database engine not initialized")
    
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.warning("All database tables dropped")


# Export commonly used items
__all__ = [
    "Base",
    "DatabaseManager", 
    "db_manager",
    "init_database",
    "get_db",
    "create_tables",
    "drop_tables",
]
