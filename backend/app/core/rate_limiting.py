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
    Database-backed rate limiter with sliding window implementation.
    
    Tracks API requests per session/endpoint using a sliding window
    approach for more accurate rate limiting than fixed windows.
    """
    
    def __init__(self, db: AsyncSession, client_id: str):
        """
        Initialize rate limiter.
        
        Args:
            db: Database session
            client_id: Client identifier (IP, user ID, etc.)
        """
        self.db = db
        self.client_id = client_id
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: Optional[int] = None, 
        window: Optional[int] = None
    ) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            key: Rate limit key (endpoint name)
            limit: Maximum requests allowed (default: 60)
            window: Time window in seconds (default: 60)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        try:
            # Use defaults if not specified
            limit = limit or 60
            window = window or 60
            
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window)
            
            # Get current request count in window
            query = select(RateLimitLog).where(
                and_(
                    RateLimitLog.endpoint == key,
                    RateLimitLog.window_start >= window_start,
                    RateLimitLog.window_end <= now
                )
            )
            
            result = await self.db.execute(query)
            current_count = len(result.scalars().all())
            
            # Check if request would exceed limit
            if current_count >= limit:
                logger.warning(f"Rate limit exceeded for {key}: {current_count}/{limit}")
                return False
            
            # Log the request
            await self._log_request(key, now, window)
            return True
            
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
            
            # Create or get user session for anonymous users
            session_id = await self._get_or_create_session()
            
            # Create rate limit log entry
            log_entry = RateLimitLog(
                session_id=session_id,
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
    
    async def _get_or_create_session(self) -> str:
        """
        Get or create a user session for anonymous users.
        
        Returns:
            Session ID for the current client
        """
        try:
            # Try to find existing session for this client
            query = select(UserSession).where(
                UserSession.ip_address == self.client_id
            ).order_by(UserSession.last_activity.desc())
            
            result = await self.db.execute(query)
            existing_session = result.scalar_one_or_none()
            
            if existing_session:
                # Update last activity
                existing_session.last_activity = datetime.utcnow()
                existing_session.request_count += 1
                await self.db.commit()
                return existing_session.session_id
            else:
                # Create new session
                from uuid import uuid4
                new_session = UserSession(
                    session_id=uuid4(),
                    ip_address=self.client_id,
                    user_agent="anonymous",  # TODO: Get from request headers
                    request_count=1
                )
                
                self.db.add(new_session)
                await self.db.commit()
                return new_session.session_id
                
        except Exception as e:
            logger.error(f"Error creating/getting session: {e}")
            # Fallback: create a temporary session ID
            # This should rarely happen, but provides a safety net
            from uuid import uuid4
            return str(uuid4())
    
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
            True if request is allowed, False if rate limited
        """
        # TODO: Implement Redis-based rate limiting
        return True
