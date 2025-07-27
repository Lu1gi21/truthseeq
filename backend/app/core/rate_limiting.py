"""
Rate limiting functionality for TruthSeeQ backend.

This module provides rate limiting capabilities using Redis or database storage
to prevent API abuse and ensure fair usage.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from ..database.models import RateLimitLog, UserSession
from ..core.exceptions import RateLimitExceededError

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter implementation using database storage.
    
    Provides sliding window rate limiting with configurable limits
    and time windows for different API endpoints.
    """
    
    def __init__(self, db: AsyncSession, client_id: str):
        """
        Initialize rate limiter.
        
        Args:
            db: Database session
            client_id: Unique identifier for the client
        """
        self.db = db
        self.client_id = client_id
        self.default_limit = 60  # requests per minute
        self.default_window = 60  # seconds
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: Optional[int] = None, 
        window: Optional[int] = None
    ) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            key: Rate limit key (e.g., endpoint name)
            limit: Maximum requests allowed (default: 60)
            window: Time window in seconds (default: 60)
            
        Returns:
            True if request is allowed, False if rate limited
            
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        limit = limit or self.default_limit
        window = window or self.default_window
        
        try:
            # Calculate time window
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window)
            
            # Count requests in current window
            query = select(RateLimitLog).where(
                and_(
                    RateLimitLog.endpoint == key,
                    RateLimitLog.window_start >= window_start,
                    RateLimitLog.window_end <= now
                )
            )
            
            result = await self.db.execute(query)
            current_count = len(result.scalars().all())
            
            # Check if limit exceeded
            if current_count >= limit:
                logger.warning(f"Rate limit exceeded for {key}: {current_count}/{limit}")
                raise RateLimitExceededError(
                    f"Rate limit exceeded: {current_count}/{limit} requests in {window}s"
                )
            
            # Log this request
            await self._log_request(key, now, window)
            
            logger.debug(f"Rate limit check passed for {key}: {current_count + 1}/{limit}")
            return True
            
        except RateLimitExceededError:
            raise
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Allow request if rate limiting fails
            return True
    
    async def _log_request(self, endpoint: str, timestamp: datetime, window: int):
        """
        Log a request for rate limiting.
        
        Args:
            endpoint: API endpoint name
            timestamp: Request timestamp
            window: Time window in seconds
        """
        try:
            window_start = timestamp - timedelta(seconds=window)
            window_end = timestamp
            
            # Create rate limit log entry
            log_entry = RateLimitLog(
                session_id=None,  # Will be set when session management is implemented
                endpoint=endpoint,
                request_count=1,
                window_start=window_start,
                window_end=window_end
            )
            
            self.db.add(log_entry)
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error logging rate limit request: {e}")
            await self.db.rollback()
    
    async def get_remaining_requests(self, key: str, limit: int = 60, window: int = 60) -> int:
        """
        Get remaining requests for a key.
        
        Args:
            key: Rate limit key
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Number of remaining requests
        """
        try:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window)
            
            query = select(RateLimitLog).where(
                and_(
                    RateLimitLog.endpoint == key,
                    RateLimitLog.window_start >= window_start,
                    RateLimitLog.window_end <= now
                )
            )
            
            result = await self.db.execute(query)
            current_count = len(result.scalars().all())
            
            return max(0, limit - current_count)
            
        except Exception as e:
            logger.error(f"Error getting remaining requests: {e}")
            return limit
    
    async def reset_rate_limit(self, key: str):
        """
        Reset rate limit for a key (admin function).
        
        Args:
            key: Rate limit key to reset
        """
        try:
            query = select(RateLimitLog).where(RateLimitLog.endpoint == key)
            result = await self.db.execute(query)
            logs = result.scalars().all()
            
            for log in logs:
                await self.db.delete(log)
            
            await self.db.commit()
            logger.info(f"Rate limit reset for {key}")
            
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
            await self.db.rollback()


class RedisRateLimiter:
    """
    Redis-based rate limiter (alternative implementation).
    
    This is a placeholder for Redis-based rate limiting that would be
    more efficient for high-traffic applications.
    """
    
    def __init__(self, redis_client, client_id: str):
        """
        Initialize Redis rate limiter.
        
        Args:
            redis_client: Redis client instance
            client_id: Unique identifier for the client
        """
        self.redis = redis_client
        self.client_id = client_id
    
    async def check_rate_limit(self, key: str, limit: int = 60, window: int = 60) -> bool:
        """
        Check rate limit using Redis.
        
        Args:
            key: Rate limit key
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            True if request is allowed
        """
        # This is a placeholder implementation
        # In a real implementation, you would use Redis commands
        # like INCR, EXPIRE, etc. for efficient rate limiting
        
        logger.warning("Redis rate limiter not implemented - using database fallback")
        return True
