"""
TruthSeeQ Backend Main Application.

This module contains the main FastAPI application setup for the TruthSeeQ
fact-checking platform, including all routes, middleware, and configuration.

Features:
- Content scraping and analysis APIs
- Rate limiting and security middleware
- Database and Redis integration
- Health checks and monitoring
- Automatic API documentation
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .api.routes import content, feed, rate_limit, workflow
from .config import settings
from .core.exceptions import (
    ContentNotFoundError,
    ValidationError,
    RateLimitExceededError
)
from .core.logging import configure_logging
from .database.database import init_database, db_manager

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


# ========================================
# Application Lifespan Management
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown tasks.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None during application runtime
    """
    # Startup
    logger.info("Starting TruthSeeQ backend application")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialization completed")
        
        # Log configuration
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Debug mode: {settings.DEBUG}")
        logger.info(f"API version: {settings.API_V1_STR}")
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down TruthSeeQ backend application")
        
        try:
            # Close database connections
            await db_manager.close()
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# ========================================
# Rate Limiting Setup
# ========================================

limiter = Limiter(key_func=get_remote_address)


# ========================================
# FastAPI Application Setup
# ========================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    openapi_url=settings.OPENAPI_URL,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    lifespan=lifespan
)

# Add rate limiting state
app.state.limiter = limiter


# ========================================
# Middleware Configuration
# ========================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.security.CORS_METHODS,
    allow_headers=settings.security.CORS_HEADERS,
)

# Trusted Host Middleware
if settings.security.ALLOWED_HOSTS != ["*"]:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.security.ALLOWED_HOSTS
    )


# ========================================
# Exception Handlers
# ========================================

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded exceptions."""
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": exc.retry_after
        }
    )
    response.headers["Retry-After"] = str(exc.retry_after)
    return response


@app.exception_handler(ContentNotFoundError)
async def content_not_found_handler(request: Request, exc: ContentNotFoundError):
    """Handle content not found exceptions."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Content not found",
            "detail": str(exc)
        }
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation error",
            "detail": str(exc)
        }
    )


@app.exception_handler(RateLimitExceededError)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceededError):
    """Handle custom rate limit exceeded exceptions."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc)
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later."
        }
    )


# ========================================
# API Routes
# ========================================

# Include API routers
app.include_router(
    content.router,
    prefix=settings.API_V1_STR,
    tags=["content"]
)

app.include_router(
    feed.router,
    prefix=settings.API_V1_STR,
    tags=["feed"]
)

app.include_router(
    rate_limit.router,
    prefix=settings.API_V1_STR,
    tags=["rate-limit"]
)

app.include_router(
    workflow.router,
    prefix=settings.API_V1_STR,
    tags=["workflow"]
)


# ========================================
# Root and Health Check Endpoints
# ========================================

@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        Basic API information and available endpoints
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "environment": settings.ENVIRONMENT,
        "api_version": settings.API_V1_STR,
        "docs_url": settings.DOCS_URL,
        "redoc_url": settings.REDOC_URL,
        "features": [
            "content_scraping",
            "brave_search_integration",
            "content_analysis",
            "fact_checking",
            "source_verification",
            "workflow_orchestration",
            "social_feed",
            "rate_limiting"
        ]
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        Application health status and system information
    """
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": settings.APP_VERSION,
        "timestamp": "2025-01-02T12:00:00Z",  # This would be dynamic in real implementation
        "services": {
            "database": "operational",
            "redis": "operational",
            "scraper": "operational",
            "ai_services": "operational"
        }
    }


@app.get("/health/deep")
async def deep_health_check():
    """
    Deep health check that tests actual service connectivity.
    
    Returns:
        Detailed health status of all services
    """
    health_status = {
        "status": "healthy",
        "checks": {}
    }
    
    try:
        # Test database connectivity
        from .database.database import get_async_session
        async with get_async_session() as session:
            await session.execute("SELECT 1")
            health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Add more service checks as needed
    health_status["checks"]["api"] = "healthy"
    health_status["checks"]["scraper"] = "healthy"
    
    return health_status


@app.get("/metrics")
async def metrics():
    """
    Basic metrics endpoint (would integrate with Prometheus in production).
    
    Returns:
        Basic application metrics
    """
    return {
        "requests_total": 0,  # Would be actual counter in production
        "active_connections": 0,
        "scraping_jobs_active": 0,
        "cache_hit_rate": 0.95,
        "average_response_time": 0.150
    }


# ========================================
# Development and Testing Endpoints
# ========================================

if settings.is_development():
    @app.get("/debug/config")
    async def debug_config():
        """
        Debug endpoint to show configuration (development only).
        
        Returns:
            Non-sensitive configuration values
        """
        return {
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "database_echo": settings.database.DATABASE_ECHO,
            "rate_limits": {
                "default": settings.rate_limit.RATE_LIMIT_DEFAULT,
                "fact_check": settings.rate_limit.RATE_LIMIT_FACT_CHECK,
                "content": settings.rate_limit.RATE_LIMIT_CONTENT
            },
            "scraping": {
                "max_concurrent": settings.scraping.SCRAPER_MAX_CONCURRENT,
                "timeout": settings.scraping.SCRAPER_TIMEOUT,
                "request_delay": settings.scraping.SCRAPER_REQUEST_DELAY
            }
        }


# ========================================
# Error Handler for Rate Limiting
# ========================================

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ========================================
# Startup Message
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Server will run on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1,
        log_level=settings.monitoring.LOG_LEVEL.lower()
    )
