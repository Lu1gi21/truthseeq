"""
Custom exceptions for TruthSeeQ backend.

This module defines custom exception classes used throughout the application
for better error handling and user feedback.
"""

from typing import Optional, Any, Dict


class TruthSeeQError(Exception):
    """Base exception class for TruthSeeQ application."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize TruthSeeQ error.
        
        Args:
            message: Error message
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ContentNotFoundError(TruthSeeQError):
    """Raised when requested content is not found."""
    pass


class ValidationError(TruthSeeQError):
    """Raised when input validation fails."""
    pass


class RateLimitExceededError(TruthSeeQError):
    """Raised when rate limit is exceeded."""
    pass


class WorkflowExecutionError(TruthSeeQError):
    """Raised when workflow execution fails."""
    pass


class ScrapingError(TruthSeeQError):
    """Raised when content scraping fails."""
    pass


class DatabaseError(TruthSeeQError):
    """Raised when database operations fail."""
    pass


class AIAnalysisError(TruthSeeQError):
    """Raised when AI analysis fails."""
    pass


class ConfigurationError(TruthSeeQError):
    """Raised when configuration is invalid or missing."""
    pass


class AuthenticationError(TruthSeeQError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(TruthSeeQError):
    """Raised when authorization fails."""
    pass


class WorkflowError(TruthSeeQError):
    """Raised when workflow execution fails."""
    pass


# Export all exceptions
__all__ = [
    "TruthSeeQError",
    "ContentNotFoundError",
    "ValidationError",
    "RateLimitExceededError",
    "DatabaseError",
    "ScrapingError",
    "AIProcessingError",
    "ConfigurationError",
    "AuthenticationError",
    "AuthorizationError",
    "WorkflowError",
]
