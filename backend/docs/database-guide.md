# Database Implementation Guide

This guide covers SQLAlchemy ORM, Alembic migrations, and PostgreSQL setup for TruthSeeQ.

## 📋 Installation & Setup

```bash
pip install sqlalchemy==2.0.41
pip install alembic==1.16.1
pip install asyncpg==0.30.0          # Async PostgreSQL driver
pip install psycopg2-binary==2.9.10  # Sync PostgreSQL driver
```

## 🗄️ Database Configuration

### Database Settings (`app/config.py`)

```python
from pydantic_settings import BaseSettings
from sqlalchemy.engine import URL

class Settings(BaseSettings):
    # Database URLs
    DATABASE_URL: str  # Async URL for SQLAlchemy
    DATABASE_URL_SYNC: str  # Sync URL for Alembic
    
    # Connection pool settings
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    
    # Database configuration
    DB_ECHO: bool = False  # Set to True for SQL logging
    
    @property
    def database_url_object(self) -> URL:
        """Get SQLAlchemy URL object for async connections."""
        return URL.create(
            "postgresql+asyncpg",
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.DB_NAME,
        )

settings = Settings()
```

### Database Connection (`app/database/database.py`)

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import create_engine
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Async engine for application use
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True,  # Validate connections before use
)

# Sync engine for Alembic migrations
sync_engine = create_engine(
    settings.DATABASE_URL_SYNC,
    echo=settings.DB_ECHO,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True,
)

# Session factories
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

SessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)

# Base model class
class Base(DeclarativeBase):
    """Base class for all database models."""
    pass

# Dependency for FastAPI
async def get_db() -> AsyncSession:
    """Get async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Test database connection
async def test_connection():
    """Test database connection."""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: sync_conn.execute("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
```

## 📊 Database Models

### Base Models (`app/database/models.py`)

```python
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, 
    JSON, Enum, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid
import enum

from app.database.database import Base

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

class UserSession(Base, TimestampMixin):
    """User session tracking for rate limiting and analytics."""
    
    __tablename__ = "user_sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    session_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    content_items = relationship("ContentItem", back_populates="session")
    fact_check_results = relationship("FactCheckResult", back_populates="session")
    feed_posts = relationship("FeedPost", back_populates="session")

class ContentStatus(enum.Enum):
    """Status of content processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ContentItem(Base, TimestampMixin):
    """Stored content items for fact-checking."""
    
    __tablename__ = "content_items"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    url: Mapped[Optional[str]] = mapped_column(Text)
    title: Mapped[Optional[str]] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_domain: Mapped[Optional[str]] = mapped_column(String(255))
    scraped_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[ContentStatus] = mapped_column(
        Enum(ContentStatus),
        default=ContentStatus.PENDING
    )
    
    # Foreign keys
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_sessions.session_id"))
    
    # Relationships
    session = relationship("UserSession", back_populates="content_items")
    metadata_items = relationship("ContentMetadata", back_populates="content_item")
    fact_check_results = relationship("FactCheckResult", back_populates="content_item")

class ContentMetadata(Base, TimestampMixin):
    """Metadata associated with content items."""
    
    __tablename__ = "content_metadata"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    content_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("content_items.id", ondelete="CASCADE")
    )
    metadata_type: Mapped[str] = mapped_column(String(100), nullable=False)
    metadata_value: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    
    # Relationships
    content_item = relationship("ContentItem", back_populates="metadata_items")

class FactCheckVerdict(enum.Enum):
    """Possible fact-check verdicts."""
    TRUE = "true"
    FALSE = "false"
    INCONCLUSIVE = "inconclusive"
    SATIRE = "satire"

class FactCheckResult(Base, TimestampMixin):
    """Results of fact-checking analysis."""
    
    __tablename__ = "fact_check_results"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    content_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("content_items.id", ondelete="CASCADE")
    )
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_sessions.session_id"))
    
    # Analysis results
    verdict: Mapped[FactCheckVerdict] = mapped_column(Enum(FactCheckVerdict))
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    claims_analyzed: Mapped[List[str]] = mapped_column(JSONB)
    ai_model_used: Mapped[str] = mapped_column(String(100))
    
    # Processing metadata
    processing_time_seconds: Mapped[float] = mapped_column(Float)
    processing_steps: Mapped[List[str]] = mapped_column(JSONB)
    errors: Mapped[Optional[List[str]]] = mapped_column(JSONB)
    
    # Relationships
    content_item = relationship("ContentItem", back_populates="fact_check_results")
    session = relationship("UserSession", back_populates="fact_check_results")
    sources = relationship("FactCheckSource", back_populates="fact_check_result")

class FactCheckSource(Base, TimestampMixin):
    """Sources used in fact-checking analysis."""
    
    __tablename__ = "fact_check_sources"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    fact_check_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fact_check_results.id", ondelete="CASCADE")
    )
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50))  # e.g., "news", "fact_check", "academic"
    relevance_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    fact_check_result = relationship("FactCheckResult", back_populates="sources")

class FeedPost(Base, TimestampMixin):
    """Social feed posts."""
    
    __tablename__ = "feed_posts"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    content_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("content_items.id", ondelete="CASCADE")
    )
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_sessions.session_id"))
    
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text)
    verdict: Mapped[FactCheckVerdict] = mapped_column(Enum(FactCheckVerdict))
    confidence: Mapped[float] = mapped_column(Float)
    
    # Engagement metrics
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    session = relationship("UserSession", back_populates="feed_posts")
    likes = relationship("FeedLike", back_populates="post")

class FeedLike(Base, TimestampMixin):
    """Likes on feed posts."""
    
    __tablename__ = "feed_likes"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    post_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("feed_posts.id", ondelete="CASCADE")
    )
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_sessions.session_id"))
    
    # Relationships
    post = relationship("FeedPost", back_populates="likes")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('post_id', 'session_id', name='unique_post_like'),
        Index('idx_feed_likes_post_session', 'post_id', 'session_id'),
    )

# Indexes for performance
Index('idx_content_items_session', ContentItem.session_id)
Index('idx_content_items_status', ContentItem.status)
Index('idx_content_items_created', ContentItem.created_at)
Index('idx_fact_check_results_session', FactCheckResult.session_id)
Index('idx_fact_check_results_content', FactCheckResult.content_id)
Index('idx_fact_check_results_verdict', FactCheckResult.verdict)
Index('idx_feed_posts_created', FeedPost.created_at)
Index('idx_user_sessions_last_activity', UserSession.last_activity)
```

## 🔄 Alembic Configuration

### Alembic Environment (`alembic/env.py`)

```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.database.models import Base
from app.core.config import settings

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the database URL from settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL_SYNC)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Alembic Configuration (`alembic.ini`)

```ini
# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = alembic

# template used to generate migration file names; The default value is %%(rev)s_%%(slug)s
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
prepend_sys_path = .

# timezone to use when rendering the date within the migration file
# as well as the filename.
timezone =

# max length of characters to apply to the
# "slug" field
truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
sourceless = false

# version number format to use for new migration files
version_num_format = %04d

# version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses
# os.pathsep. If this key is omitted entirely, it falls back to the legacy
# behavior of splitting on spaces and/or commas.
version_path_separator = :

# the output encoding used when revision files
# are written from script.py.mako
output_encoding = utf-8

[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.

# format using "black" - use the console_scripts runner, against the "black" entrypoint
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 79 REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

## 🛠️ Database Operations

### Repository Pattern (`app/database/repositories.py`)

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

from app.database.models import (
    UserSession, ContentItem, FactCheckResult, 
    FeedPost, FeedLike, ContentStatus, FactCheckVerdict
)

class BaseRepository:
    """Base repository class with common operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db

class ContentRepository(BaseRepository):
    """Repository for content-related operations."""
    
    async def create_content_item(
        self,
        session_id: str,
        content: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        source_domain: Optional[str] = None
    ) -> ContentItem:
        """Create a new content item."""
        content_item = ContentItem(
            session_id=session_id,
            content=content,
            url=url,
            title=title,
            source_domain=source_domain,
            status=ContentStatus.PENDING
        )
        
        self.db.add(content_item)
        await self.db.flush()
        return content_item
    
    async def get_content_by_id(self, content_id: uuid.UUID) -> Optional[ContentItem]:
        """Get content item by ID."""
        stmt = select(ContentItem).where(ContentItem.id == content_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update_content_status(
        self, 
        content_id: uuid.UUID, 
        status: ContentStatus
    ) -> bool:
        """Update content status."""
        stmt = (
            update(ContentItem)
            .where(ContentItem.id == content_id)
            .values(status=status, updated_at=func.now())
        )
        result = await self.db.execute(stmt)
        return result.rowcount > 0
    
    async def get_session_content_history(
        self,
        session_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[ContentItem]:
        """Get content history for a session."""
        stmt = (
            select(ContentItem)
            .where(ContentItem.session_id == session_id)
            .order_by(ContentItem.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        return result.scalars().all()

class FactCheckRepository(BaseRepository):
    """Repository for fact-checking operations."""
    
    async def create_fact_check_result(
        self,
        content_id: uuid.UUID,
        session_id: str,
        verdict: FactCheckVerdict,
        confidence_score: float,
        reasoning: str,
        claims_analyzed: List[str],
        ai_model_used: str,
        processing_time: float,
        processing_steps: List[str],
        errors: Optional[List[str]] = None
    ) -> FactCheckResult:
        """Create a new fact-check result."""
        fact_check = FactCheckResult(
            content_id=content_id,
            session_id=session_id,
            verdict=verdict,
            confidence_score=confidence_score,
            reasoning=reasoning,
            claims_analyzed=claims_analyzed,
            ai_model_used=ai_model_used,
            processing_time_seconds=processing_time,
            processing_steps=processing_steps,
            errors=errors or []
        )
        
        self.db.add(fact_check)
        await self.db.flush()
        return fact_check
    
    async def get_fact_check_by_content_id(
        self, 
        content_id: uuid.UUID
    ) -> Optional[FactCheckResult]:
        """Get fact-check result by content ID."""
        stmt = (
            select(FactCheckResult)
            .where(FactCheckResult.content_id == content_id)
            .options(selectinload(FactCheckResult.sources))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_verdict_statistics(
        self, 
        days: int = 7
    ) -> Dict[str, int]:
        """Get verdict statistics for the last N days."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        stmt = (
            select(FactCheckResult.verdict, func.count())
            .where(FactCheckResult.created_at >= since_date)
            .group_by(FactCheckResult.verdict)
        )
        
        result = await self.db.execute(stmt)
        return {verdict.value: count for verdict, count in result.all()}

class SessionRepository(BaseRepository):
    """Repository for session management."""
    
    async def get_or_create_session(
        self,
        session_id: str,
        ip_address: str,
        user_agent: Optional[str] = None
    ) -> UserSession:
        """Get existing session or create a new one."""
        # Try to get existing session
        stmt = select(UserSession).where(UserSession.session_id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if session:
            # Update last activity
            session.last_activity = func.now()
            return session
        
        # Create new session
        session = UserSession(
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.add(session)
        await self.db.flush()
        return session
    
    async def increment_request_count(self, session_id: str) -> bool:
        """Increment request count for a session."""
        stmt = (
            update(UserSession)
            .where(UserSession.session_id == session_id)
            .values(
                request_count=UserSession.request_count + 1,
                last_activity=func.now()
            )
        )
        result = await self.db.execute(stmt)
        return result.rowcount > 0
    
    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """Clean up sessions older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        stmt = delete(UserSession).where(
            UserSession.last_activity < cutoff_date
        )
        result = await self.db.execute(stmt)
        return result.rowcount
```

## 🔧 Migration Commands

### Common Migration Operations

```bash
# Create a new migration
alembic revision --autogenerate -m "Add user sessions table"

# Apply all pending migrations
alembic upgrade head

# Rollback to previous migration
alembic downgrade -1

# Show current revision
alembic current

# Show migration history
alembic history --verbose

# Reset to a specific revision
alembic downgrade <revision_id>

# Merge multiple migration heads
alembic merge -m "Merge migration heads" <revision1> <revision2>

# Generate SQL without executing
alembic upgrade head --sql

# Mark migration as applied without running
alembic stamp head
```

### Database Initialization Script

```python
# scripts/init_db.py
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.database.models import Base
from app.core.config import settings

async def init_database():
    """Initialize database with all tables."""
    engine = create_async_engine(settings.DATABASE_URL)
    
    async with engine.begin() as conn:
        # Drop all tables (use with caution!)
        # await conn.run_sync(Base.metadata.drop_all)
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    print("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_database())
```

## 🧪 Testing Database Operations

### Database Testing (`tests/test_database/test_repositories.py`)

```python
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database.models import Base
from app.database.repositories import ContentRepository, FactCheckRepository

# Test database URL (use separate test database)
TEST_DATABASE_URL = "postgresql+asyncpg://test_user:test_pass@localhost/truthseeq_test"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def db_session(test_engine):
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.mark.asyncio
async def test_create_content_item(db_session):
    """Test creating a content item."""
    repo = ContentRepository(db_session)
    
    content_item = await repo.create_content_item(
        session_id="test_session",
        content="Test content for fact-checking",
        url="https://example.com/article",
        title="Test Article"
    )
    
    assert content_item.id is not None
    assert content_item.content == "Test content for fact-checking"
    assert content_item.session_id == "test_session"

@pytest.mark.asyncio
async def test_create_fact_check_result(db_session):
    """Test creating a fact-check result."""
    content_repo = ContentRepository(db_session)
    fact_repo = FactCheckRepository(db_session)
    
    # Create content item first
    content_item = await content_repo.create_content_item(
        session_id="test_session",
        content="Test content"
    )
    
    # Create fact-check result
    fact_check = await fact_repo.create_fact_check_result(
        content_id=content_item.id,
        session_id="test_session",
        verdict=FactCheckVerdict.TRUE,
        confidence_score=0.85,
        reasoning="Analysis shows this claim is accurate",
        claims_analyzed=["Test claim"],
        ai_model_used="gpt-4-turbo-preview",
        processing_time=5.2,
        processing_steps=["extraction", "analysis"]
    )
    
    assert fact_check.id is not None
    assert fact_check.verdict == FactCheckVerdict.TRUE
    assert fact_check.confidence_score == 0.85
```

## 📊 Performance Optimization

### Database Indexing Strategy

```sql
-- Key indexes for performance
CREATE INDEX CONCURRENTLY idx_content_items_session_created 
ON content_items(session_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_fact_check_results_content_verdict 
ON fact_check_results(content_id, verdict);

CREATE INDEX CONCURRENTLY idx_user_sessions_last_activity_ip 
ON user_sessions(last_activity DESC, ip_address);

CREATE INDEX CONCURRENTLY idx_feed_posts_created_verdict 
ON feed_posts(created_at DESC, verdict);

-- Partial indexes for active sessions
CREATE INDEX CONCURRENTLY idx_active_sessions 
ON user_sessions(last_activity) 
WHERE last_activity > NOW() - INTERVAL '1 day';
```

### Connection Pool Configuration

```python
# Optimal connection pool settings
ENGINE_CONFIG = {
    "pool_size": 20,          # Base number of connections
    "max_overflow": 30,       # Additional connections
    "pool_timeout": 30,       # Timeout for getting connection
    "pool_recycle": 3600,     # Recycle connections every hour
    "pool_pre_ping": True,    # Validate connections
    "echo": False,            # Set to True for SQL debugging
}
```

## 🔗 Additional Resources

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [AsyncPG Documentation](https://magicstack.github.io/asyncpg/)
- [FastAPI Database Tutorial](https://fastapi.tiangolo.com/tutorial/sql-databases/) 