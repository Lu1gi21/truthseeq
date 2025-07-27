"""
Configuration management for TruthSeeQ backend.

This module provides centralized configuration management using Pydantic settings
with environment variable support and validation.

Settings include:
- Database configuration
- API settings  
- Security and authentication
- Rate limiting configuration
- AI service settings
- Logging and monitoring
"""

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field, PostgresDsn, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL connection parameters
    POSTGRES_SERVER: str = Field(default="localhost", description="PostgreSQL server host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL server port")
    POSTGRES_USER: str = Field(default="truthseeq", description="PostgreSQL username")
    POSTGRES_PASSWORD: str = Field(default="dev_password", description="PostgreSQL password")
    POSTGRES_DB: str = Field(default="truthseeq", description="PostgreSQL database name")
    
    # SQLAlchemy configuration
    DATABASE_ECHO: bool = Field(default=False, description="Echo SQL statements to logs")
    DATABASE_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, description="Maximum overflow connections")
    DATABASE_POOL_RECYCLE: int = Field(default=300, description="Connection recycle time in seconds")
    
    # Test database configuration
    TEST_POSTGRES_DB: str = Field(default="truthseeq_test", description="Test database name")
    
    def get_database_url(self, test: bool = False) -> str:
        """
        Generate PostgreSQL database URL.
        
        Args:
            test: Whether to use test database
            
        Returns:
            PostgreSQL connection URL for SQLAlchemy
        """
        db_name = self.TEST_POSTGRES_DB if test else self.POSTGRES_DB
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{db_name}"
    
    def get_sync_database_url(self, test: bool = False) -> str:
        """
        Generate synchronous PostgreSQL database URL for Alembic.
        
        Args:
            test: Whether to use test database
            
        Returns:
            Synchronous PostgreSQL connection URL
        """
        db_name = self.TEST_POSTGRES_DB if test else self.POSTGRES_DB
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{db_name}"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and rate limiting."""
    
    REDIS_HOST: str = Field(default="localhost", description="Redis server host")
    REDIS_PORT: int = Field(default=6379, description="Redis server port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    REDIS_DECODE_RESPONSES: bool = Field(default=True, description="Decode Redis responses")
    
    # Connection pool settings
    REDIS_MAX_CONNECTIONS: int = Field(default=100, description="Maximum Redis connections")
    REDIS_RETRY_ON_TIMEOUT: bool = Field(default=True, description="Retry on Redis timeout")
    
    def get_redis_url(self) -> str:
        """
        Generate Redis connection URL.
        
        Returns:
            Redis connection URL
        """
        auth_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""
    
    # Default rate limits (requests per minute)
    RATE_LIMIT_DEFAULT: int = Field(default=60, description="Default requests per minute")
    RATE_LIMIT_FACT_CHECK: int = Field(default=10, description="Fact-check requests per minute")
    RATE_LIMIT_FEED: int = Field(default=100, description="Feed requests per minute")
    RATE_LIMIT_CONTENT: int = Field(default=20, description="Content requests per minute")
    
    # Rate limiting windows
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Rate limiting window in seconds")
    RATE_LIMIT_CLEANUP_INTERVAL: int = Field(default=300, description="Cleanup interval in seconds")
    
    # Burst allowances
    RATE_LIMIT_BURST_MULTIPLIER: float = Field(default=1.5, description="Burst multiplier for rate limits")


class SecuritySettings(BaseSettings):
    """Security and authentication configuration."""
    
    # API Keys and secrets
    SECRET_KEY: str = Field(default="dev_secret_key_change_in_production", description="Secret key for signing JWT tokens")
    API_KEY_HEADER: str = Field(default="X-API-Key", description="API key header name")
    
    # JWT configuration
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    JWT_EXPIRE_MINUTES: int = Field(default=30, description="JWT token expiration in minutes")
    
    # CORS settings
    ALLOWED_HOSTS: List[str] = Field(default=["*"], description="Allowed hosts for CORS")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed origins for CORS")
    CORS_METHODS: List[str] = Field(default=["*"], description="Allowed methods for CORS")
    CORS_HEADERS: List[str] = Field(default=["*"], description="Allowed headers for CORS")
    
    # Security headers
    ENABLE_SECURITY_HEADERS: bool = Field(default=True, description="Enable security headers")


class AISettings(BaseSettings):
    """AI and LangChain configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # OpenAI configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4.1-mini-2025-04-14", description="Default OpenAI model")
    OPENAI_TEMPERATURE: float = Field(default=0.0, description="OpenAI temperature setting")
    OPENAI_MAX_TOKENS: int = Field(default=1000, description="Maximum tokens for OpenAI")
    
    # Anthropic configuration
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    ANTHROPIC_MODEL: str = Field(default="claude-3-sonnet-20240229", description="Default Anthropic model")
    
    # LangGraph configuration
    LANGGRAPH_CHECKPOINT_BACKEND: str = Field(default="memory", description="LangGraph checkpoint backend")
    LANGGRAPH_MAX_ITERATIONS: int = Field(default=50, description="Maximum LangGraph iterations")
    
    # Content analysis settings
    CONTENT_ANALYSIS_BATCH_SIZE: int = Field(default=10, description="Batch size for content analysis")
    FACT_CHECK_CONFIDENCE_THRESHOLD: float = Field(default=0.7, description="Minimum confidence threshold")


class ScrapingSettings(BaseSettings):
    """Web scraping configuration."""
    
    # Scraping limits
    SCRAPER_MAX_CONCURRENT: int = Field(default=5, description="Maximum concurrent scraping jobs")
    SCRAPER_REQUEST_DELAY: float = Field(default=1.0, description="Delay between requests in seconds")
    SCRAPER_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")
    
    # User agent settings
    SCRAPER_USER_AGENT: str = Field(
        default="TruthSeeQ/1.0 (+https://truthseeq.com/bot)",
        description="User agent for scraping requests"
    )
    
    # Content extraction
    CONTENT_MIN_LENGTH: int = Field(default=100, description="Minimum content length for processing")
    CONTENT_MAX_LENGTH: int = Field(default=50000, description="Maximum content length for processing")


class BraveSearchSettings(BaseSettings):
    """Brave Search API configuration."""
    
    BRAVE_API_KEY: Optional[str] = Field(default=None, description="Brave Search API key")
    API_KEY: Optional[str] = Field(default=None, description="Brave Search API key (alias)")
    BRAVE_SEARCH_API_KEY: Optional[str] = Field(default=None, description="Brave Search API key (legacy)")
    BRAVE_SEARCH_BASE_URL: str = Field(
        default="https://api.search.brave.com/res/v1/web/search",
        description="Brave Search API base URL"
    )
    BRAVE_SEARCH_TIMEOUT: int = Field(default=10, description="Brave Search API timeout in seconds")
    BRAVE_SEARCH_MAX_RESULTS: int = Field(default=20, description="Maximum search results to return")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Logging configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="structured", description="Log format (structured or text)")
    LOG_FILE: Optional[str] = Field(default=None, description="Log file path")
    
    # Metrics configuration
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(default=8001, description="Metrics server port")
    
    # Health check configuration
    HEALTH_CHECK_INTERVAL: int = Field(default=60, description="Health check interval in seconds")


class Settings(BaseSettings):
    """
    Main application settings.
    
    Combines all configuration sections and provides application-wide settings.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Application information
    APP_NAME: str = Field(default="TruthSeeQ", description="Application name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    APP_DESCRIPTION: str = Field(
        default="AI-powered misinformation detection platform",
        description="Application description"
    )
    
    # Environment
    ENVIRONMENT: str = Field(default="development", description="Environment name")
    DEBUG: bool = Field(default=False, description="Debug mode")
    TESTING: bool = Field(default=False, description="Testing mode")
    
    # API configuration
    API_V1_STR: str = Field(default="/api/v1", description="API version prefix")
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")
    
    # Documentation
    DOCS_URL: str = Field(default="/docs", description="Documentation URL")
    REDOC_URL: str = Field(default="/redoc", description="ReDoc URL")
    OPENAPI_URL: str = Field(default="/openapi.json", description="OpenAPI schema URL")
    
    # Subsystem settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    ai: AISettings = Field(default_factory=AISettings)
    scraping: ScrapingSettings = Field(default_factory=ScrapingSettings)
    brave_search: BraveSearchSettings = Field(default_factory=BraveSearchSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed = ["development", "staging", "production", "testing"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    @validator("DEBUG")
    def validate_debug(cls, v, values):
        """Validate debug setting based on environment."""
        if values.get("ENVIRONMENT") == "production" and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v
    
    def get_database_url(self, test: bool = False) -> str:
        """
        Get database connection URL.
        
        Args:
            test: Whether to use test database
            
        Returns:
            Database connection URL
        """
        return self.database.get_database_url(test=test or self.TESTING)
    
    def get_redis_url(self) -> str:
        """
        Get Redis connection URL.
        
        Returns:
            Redis connection URL
        """
        return self.redis.get_redis_url()
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == "production"
    
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.TESTING or self.ENVIRONMENT == "testing"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Configured Settings instance
        
    Note:
        This function is cached to avoid re-reading environment variables
        on every call. In production, settings should be static.
    """
    return Settings()


# Global settings instance
settings = get_settings()


# Configure logging based on settings
def configure_logging():
    """Configure application logging based on settings."""
    log_level = getattr(logging, settings.monitoring.LOG_LEVEL.upper())
    
    if settings.monitoring.LOG_FORMAT == "structured":
        # Configure structured logging (could integrate with structlog)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        logging.basicConfig(level=log_level)
    
    # Set specific logger levels
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.database.DATABASE_ECHO else logging.WARNING
    )
    logging.getLogger("uvicorn").setLevel(logging.INFO)


# Initialize logging
if not settings.TESTING:
    configure_logging()


# Export commonly used items
__all__ = [
    "Settings",
    "DatabaseSettings",
    "RedisSettings", 
    "RateLimitSettings",
    "SecuritySettings",
    "AISettings",
    "ScrapingSettings",
    "MonitoringSettings",
    "settings",
    "get_settings",
    "configure_logging",
]
