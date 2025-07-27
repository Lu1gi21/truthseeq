"""
API dependencies for TruthSeeQ backend.

This module provides dependency injection functions for FastAPI routes,
including session management, rate limiting, and authentication.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.database import get_db
from ..database.models import UserSession
from ..core.rate_limiting import RateLimiter

logger = logging.getLogger(__name__)


async def get_current_session(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[UserSession]:
    """
    Get current user session from request.
    
    This is a simplified implementation that creates anonymous sessions
    based on IP address and user agent. In a real application, you would
    implement proper authentication.
    
    Args:
        request: FastAPI request object
        db: Database session
        
    Returns:
        UserSession instance or None if session creation fails
    """
    try:
        # Get client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # For now, return None to indicate anonymous access
        # In a real implementation, you would:
        # 1. Check for existing session
        # 2. Create new session if needed
        # 3. Return the session
        
        logger.debug(f"Anonymous access from {client_ip}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting current session: {e}")
        return None


async def get_rate_limiter(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> RateLimiter:
    """
    Get rate limiter instance for the current request.
    
    Args:
        request: FastAPI request object
        db: Database session
        
    Returns:
        RateLimiter instance
    """
    try:
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Create rate limiter with client identifier
        # Use only IP address for rate limiting to avoid database field length issues
        rate_limiter = RateLimiter(
            db=db,
            client_id=client_ip
        )
        
        return rate_limiter
        
    except Exception as e:
        logger.error(f"Error creating rate limiter: {e}")
        # Return a basic rate limiter that doesn't enforce limits
        return RateLimiter(db=db, client_id="fallback")


async def require_authentication(
    current_session: Optional[UserSession] = Depends(get_current_session)
) -> UserSession:
    """
    Require authentication for protected endpoints.
    
    Args:
        current_session: Current user session
        
    Returns:
        UserSession instance
        
    Raises:
        HTTPException: If no valid session is found
    """
    if not current_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return current_session


async def get_optional_session(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[UserSession]:
    """
    Get optional session - doesn't require authentication.
    
    Args:
        request: FastAPI request object
        db: Database session
        
    Returns:
        UserSession instance or None
    """
    return await get_current_session(request, db)
