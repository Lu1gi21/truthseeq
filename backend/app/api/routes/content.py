"""
Content-related API routes for TruthSeeQ platform.

This module provides RESTful API endpoints for content management, including:
- Content scraping and batch processing
- Search and content discovery via Brave Search
- Content quality validation and analysis
- Content deduplication and similarity analysis
- Background job status monitoring

Routes:
    POST /content/scrape - Scrape URLs and store content
    POST /content/search - Search for content using Brave Search
    GET /content/{content_id} - Get content by ID
    POST /content/{content_id}/validate - Validate content quality
    POST /content/deduplicate - Perform deduplication analysis
    GET /jobs/{job_id} - Get job status
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

from ...core.exceptions import (
    ContentNotFoundError,
    ValidationError,
    RateLimitExceededError
)
from ...database.database import get_async_session
from ...database.models import ContentItem, ContentMetadata, UserSession
from ...schemas.content import (
    ScrapingRequest,
    ScrapingResponse,
    SearchRequest,
    SearchResponse,
    ContentItemResponse,
    ContentValidationRequest,
    ContentValidationResponse,
    DeduplicationRequest,
    DeduplicationResponse,
    JobStatus
)
from ...services.scraper_service import ScraperService
from ..dependencies import get_current_session, get_rate_limiter

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/content", tags=["content"])


# ========================================
# Dependency Injection
# ========================================

async def get_scraper_service(db: AsyncSession = Depends(get_async_session)) -> ScraperService:
    """
    Get ScraperService instance with database session.
    
    Args:
        db: Database session
        
    Returns:
        Initialized ScraperService
    """
    return ScraperService(db)


# ========================================
# Content Scraping Endpoints
# ========================================

@router.post("/scrape", response_model=ScrapingResponse)
async def scrape_content(
    request: ScrapingRequest,
    background_tasks: BackgroundTasks,
    scraper_service: ScraperService = Depends(get_scraper_service),
    current_session: Optional[UserSession] = Depends(get_current_session),
    rate_limiter = Depends(get_rate_limiter)
):
    """
    Scrape content from provided URLs.
    
    Initiates scraping of multiple URLs using the advanced scraper with
    content quality analysis, deduplication, and database storage.
    
    Args:
        request: Scraping request with URLs and options
        background_tasks: FastAPI background tasks
        scraper_service: Injected scraper service
        current_session: Current user session (optional)
        rate_limiter: Rate limiting dependency
        
    Returns:
        ScrapingResponse with job ID and initial results
        
    Raises:
        HTTPException: If rate limit exceeded or validation fails
    """
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(
            key=f"scrape:{current_session.session_id if current_session else 'anonymous'}",
            limit=10,  # 10 scraping requests per minute
            window=60
        )
        
        logger.info(f"Starting content scraping for {len(request.urls)} URLs")
        
        # Perform scraping
        result = await scraper_service.scrape_urls(
            request, 
            session_id=current_session.session_id if current_session else None
        )
        
        logger.info(f"Scraping completed: job_id={result.job_id}, successful={result.successful}")
        
        return result
        
    except RateLimitExceededError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for scraping requests")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during scraping")


@router.post("/search", response_model=SearchResponse)
async def search_content(
    request: SearchRequest,
    scraper_service: ScraperService = Depends(get_scraper_service),
    current_session: Optional[UserSession] = Depends(get_current_session),
    rate_limiter = Depends(get_rate_limiter)
):
    """
    Search for content using Brave Search API.
    
    Performs web or news search using Brave Search with support for
    various filters and localization options.
    
    Args:
        request: Search request with query and options
        scraper_service: Injected scraper service
        current_session: Current user session (optional)
        rate_limiter: Rate limiting dependency
        
    Returns:
        SearchResponse with search results
        
    Raises:
        HTTPException: If rate limit exceeded or search fails
    """
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(
            key=f"search:{current_session.session_id if current_session else 'anonymous'}",
            limit=30,  # 30 search requests per minute
            window=60
        )
        
        logger.info(f"Searching content: query='{request.query}', type={request.search_type}")
        
        # Perform search
        result = await scraper_service.search_content(request)
        
        logger.info(f"Search completed: {result.total_results} results in {result.search_time:.2f}s")
        
        return result
        
    except RateLimitExceededError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for search requests")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during search")


# ========================================
# Content Retrieval Endpoints
# ========================================

@router.get("/{content_id}", response_model=ContentItemResponse)
async def get_content(
    content_id: int,
    db: AsyncSession = Depends(get_async_session),
    current_session: Optional[UserSession] = Depends(get_current_session)
):
    """
    Get content item by ID.
    
    Retrieves a specific content item with its metadata from the database.
    
    Args:
        content_id: Content item ID
        db: Database session
        current_session: Current user session (optional)
        
    Returns:
        ContentItemResponse with content and metadata
        
    Raises:
        HTTPException: If content not found
    """
    try:
        # Query content with metadata
        stmt = (
            select(ContentItem)
            .options(selectinload(ContentItem.metadata))
            .where(ContentItem.id == content_id)
        )
        result = await db.execute(stmt)
        content_item = result.scalar_one_or_none()
        
        if not content_item:
            raise HTTPException(status_code=404, detail=f"Content {content_id} not found")
        
        logger.info(f"Retrieved content {content_id} for session {current_session.session_id if current_session else 'anonymous'}")
        
        return ContentItemResponse.from_orm(content_item)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve content {content_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=List[ContentItemResponse])
async def list_content(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    domain: Optional[str] = Query(None, description="Filter by source domain"),
    status: Optional[str] = Query(None, description="Filter by content status"),
    db: AsyncSession = Depends(get_async_session),
    current_session: Optional[UserSession] = Depends(get_current_session)
):
    """
    List content items with pagination and filtering.
    
    Retrieves content items from the database with support for pagination
    and filtering by domain and status.
    
    Args:
        skip: Number of items to skip
        limit: Number of items to return
        domain: Optional domain filter
        status: Optional status filter
        db: Database session
        current_session: Current user session (optional)
        
    Returns:
        List of ContentItemResponse objects
    """
    try:
        # Build query with filters
        stmt = select(ContentItem).options(selectinload(ContentItem.metadata))
        
        # Apply filters
        if domain:
            stmt = stmt.where(ContentItem.source_domain == domain)
        if status:
            stmt = stmt.where(ContentItem.status == status)
        
        # Apply pagination
        stmt = stmt.offset(skip).limit(limit)
        
        # Execute query
        result = await db.execute(stmt)
        content_items = result.scalars().all()
        
        logger.info(f"Listed {len(content_items)} content items for session {current_session.session_id if current_session else 'anonymous'}")
        
        return [ContentItemResponse.from_orm(item) for item in content_items]
        
    except Exception as e:
        logger.error(f"Failed to list content: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ========================================
# Content Validation Endpoints
# ========================================

@router.post("/{content_id}/validate", response_model=ContentValidationResponse)
async def validate_content(
    content_id: int,
    scraper_service: ScraperService = Depends(get_scraper_service),
    current_session: Optional[UserSession] = Depends(get_current_session),
    rate_limiter = Depends(get_rate_limiter)
):
    """
    Validate content quality and authenticity.
    
    Performs comprehensive content validation including quality analysis,
    readability assessment, and source credibility evaluation.
    
    Args:
        content_id: Content item ID to validate
        scraper_service: Injected scraper service
        current_session: Current user session (optional)
        rate_limiter: Rate limiting dependency
        
    Returns:
        ContentValidationResponse with validation results
        
    Raises:
        HTTPException: If content not found or validation fails
    """
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(
            key=f"validate:{current_session.session_id if current_session else 'anonymous'}",
            limit=20,  # 20 validation requests per minute
            window=60
        )
        
        logger.info(f"Validating content {content_id}")
        
        # Perform validation
        result = await scraper_service.validate_content(content_id)
        
        logger.info(f"Content {content_id} validation completed: valid={result.is_valid}, confidence={result.confidence:.2f}")
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RateLimitExceededError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for validation requests")
    except Exception as e:
        logger.error(f"Content validation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during validation")


# ========================================
# Content Deduplication Endpoints
# ========================================

@router.post("/deduplicate", response_model=DeduplicationResponse)
async def deduplicate_content(
    request: DeduplicationRequest,
    scraper_service: ScraperService = Depends(get_scraper_service),
    current_session: Optional[UserSession] = Depends(get_current_session),
    rate_limiter = Depends(get_rate_limiter)
):
    """
    Perform content deduplication analysis.
    
    Analyzes content similarity and identifies duplicates using text
    comparison, URL similarity, and semantic analysis.
    
    Args:
        request: Deduplication request with content IDs and options
        scraper_service: Injected scraper service
        current_session: Current user session (optional)
        rate_limiter: Rate limiting dependency
        
    Returns:
        DeduplicationResponse with similarity analysis and duplicate groups
        
    Raises:
        HTTPException: If rate limit exceeded or analysis fails
    """
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(
            key=f"deduplicate:{current_session.session_id if current_session else 'anonymous'}",
            limit=5,  # 5 deduplication requests per minute
            window=60
        )
        
        logger.info(f"Starting deduplication analysis for {len(request.content_ids)} content items")
        
        # Perform deduplication
        result = await scraper_service.deduplicate_content(
            request.content_ids,
            request.similarity_threshold
        )
        
        logger.info(f"Deduplication completed: {result.duplicates_found} duplicates found, {result.duplicates_removed} would be removed")
        
        return result
        
    except RateLimitExceededError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for deduplication requests")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during deduplication")


# ========================================
# Job Status Endpoints
# ========================================

@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: UUID,
    scraper_service: ScraperService = Depends(get_scraper_service),
    current_session: Optional[UserSession] = Depends(get_current_session)
):
    """
    Get background job status.
    
    Retrieves the current status of a background job (scraping, validation, etc.)
    including progress, completion status, and any error messages.
    
    Args:
        job_id: Job identifier
        scraper_service: Injected scraper service
        current_session: Current user session (optional)
        
    Returns:
        JobStatus with current job information
        
    Raises:
        HTTPException: If job not found
    """
    try:
        job_status = scraper_service.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        logger.info(f"Retrieved job status {job_id}: {job_status.status} ({job_status.progress:.1%})")
        
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve job status {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ========================================
# Utility Endpoints
# ========================================

@router.post("/cleanup")
async def cleanup_old_jobs(
    max_age_hours: int = Query(24, ge=1, le=168, description="Maximum age in hours"),
    scraper_service: ScraperService = Depends(get_scraper_service),
    current_session: Optional[UserSession] = Depends(get_current_session)
):
    """
    Clean up old job records.
    
    Removes old job status records to prevent memory buildup.
    Requires an active session.
    
    Args:
        max_age_hours: Maximum age in hours for keeping job records
        scraper_service: Injected scraper service
        current_session: Current user session (required)
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If unauthorized or cleanup fails
    """
    try:
        if not current_session:
            raise HTTPException(status_code=401, detail="Authentication required for cleanup")
        
        await scraper_service.cleanup_old_jobs(max_age_hours)
        
        logger.info(f"Job cleanup completed for jobs older than {max_age_hours} hours")
        
        return {"message": f"Cleaned up jobs older than {max_age_hours} hours"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during cleanup")


# ========================================
# Health Check Endpoint
# ========================================

@router.get("/health")
async def health_check():
    """
    Health check endpoint for content service.
    
    Returns:
        Service health status
    """
    return {
        "service": "content",
        "status": "healthy",
        "features": [
            "scraping",
            "search",
            "validation", 
            "deduplication",
            "job_tracking"
        ]
    }
