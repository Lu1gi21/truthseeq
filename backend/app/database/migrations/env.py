"""
Alembic migration environment for TruthSeeQ database.

This module configures Alembic to work with the TruthSeeQ database models
and connection settings.
"""

import asyncio
import logging
from logging.config import fileConfig
from typing import Any

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import our models and configuration
from app.config import settings
from app.database.database import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the database URL in the Alembic config
# Use synchronous URL for Alembic migrations
database_url = settings.database.get_sync_database_url()
config.set_main_option("sqlalchemy.url", database_url)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Run migrations with a database connection.
    
    Args:
        connection: SQLAlchemy database connection
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
        # Include object naming for better constraint names
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def include_object(object: Any, name: str, type_: str, reflected: bool, compare_to: Any) -> bool:
    """
    Determine whether to include an object in the migration.
    
    This function can be used to filter out certain objects
    from being included in migrations.
    
    Args:
        object: The schema object
        name: The name of the object
        type_: The type of object (table, column, etc.)
        reflected: Whether the object was reflected from the database
        compare_to: The object being compared to
        
    Returns:
        True if the object should be included, False otherwise
    """
    # Skip certain system tables or views if needed
    if type_ == "table" and name in ["spatial_ref_sys"]:
        return False
    
    return True


async def run_async_migrations() -> None:
    """
    Run migrations in online mode using async engine.
    
    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Get the database URL - use async URL for async engine
    database_url = settings.database.get_database_url()
    
    # Create configuration for the engine
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = database_url
    
    # Create async engine
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    try:
        asyncio.run(run_async_migrations())
    except Exception as e:
        logging.error(f"Migration failed: {e}")
        raise


# Determine whether to run in offline or online mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online() 