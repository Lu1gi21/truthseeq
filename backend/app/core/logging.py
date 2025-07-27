"""
Logging configuration for TruthSeeQ backend.

This module provides centralized logging configuration for the application,
including structured logging, log levels, and output formatting.
"""

import logging
import logging.config
import sys
from typing import Dict, Any

from ..config import settings


def configure_logging() -> None:
    """
    Configure application logging based on settings.
    
    Sets up logging with appropriate levels, formats, and handlers
    based on the current environment and configuration.
    """
    # Get log level from settings
    log_level = getattr(logging, settings.monitoring.LOG_LEVEL.upper(), logging.INFO)
    
    # Configure basic logging
    if settings.monitoring.LOG_FORMAT == "structured":
        # Structured logging format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
    else:
        # Simple text format
        log_format = "%(levelname)s - %(name)s - %(message)s"
        date_format = None
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.database.DATABASE_ECHO else logging.WARNING
    )
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Configure application loggers
    app_logger = logging.getLogger("app")
    app_logger.setLevel(log_level)
    
    # Add file handler if log file is specified
    if settings.monitoring.LOG_FILE:
        try:
            file_handler = logging.FileHandler(settings.monitoring.LOG_FILE)
            file_handler.setFormatter(logging.Formatter(log_format, date_format))
            app_logger.addHandler(file_handler)
        except Exception as e:
            logging.error(f"Failed to configure file logging: {e}")
    
    logging.info(f"Logging configured - Level: {settings.monitoring.LOG_LEVEL}, Format: {settings.monitoring.LOG_FORMAT}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_request(request_id: str, method: str, path: str, status_code: int, duration: float) -> None:
    """
    Log HTTP request details.
    
    Args:
        request_id: Unique request identifier
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration: Request duration in seconds
    """
    logger = get_logger("app.requests")
    
    if status_code >= 400:
        logger.warning(f"Request {request_id}: {method} {path} - {status_code} ({duration:.3f}s)")
    else:
        logger.info(f"Request {request_id}: {method} {path} - {status_code} ({duration:.3f}s)")


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log application errors with context.
    
    Args:
        error: Exception that occurred
        context: Additional context information
    """
    logger = get_logger("app.errors")
    
    error_msg = f"Error: {type(error).__name__}: {str(error)}"
    if context:
        error_msg += f" - Context: {context}"
    
    logger.error(error_msg, exc_info=True)


def log_performance(operation: str, duration: float, details: Dict[str, Any] = None) -> None:
    """
    Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        details: Additional performance details
    """
    logger = get_logger("app.performance")
    
    perf_msg = f"Performance: {operation} - {duration:.3f}s"
    if details:
        perf_msg += f" - Details: {details}"
    
    if duration > 1.0:  # Log slow operations as warnings
        logger.warning(perf_msg)
    else:
        logger.info(perf_msg)
