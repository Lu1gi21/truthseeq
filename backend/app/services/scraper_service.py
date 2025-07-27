"""
Advanced scraper service integrating with TruthSeeQ platform.

This module provides a comprehensive scraping service that integrates the existing
advanced_scraper.py with the TruthSeeQ platform, adding features like:

- Brave Search API integration for content discovery
- Database integration for content storage and retrieval
- Content deduplication and quality scoring
- Background task management for async scraping
- Rate limiting and error handling
- Content preprocessing for AI analysis

Classes:
    BraveSearchClient: Client for Brave Search API integration
    ScraperService: Main service class for content scraping and management
    ContentQualityAnalyzer: Analyzes and scores content quality
    ContentDeduplicator: Handles content deduplication
"""

import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse
from uuid import UUID, uuid4
import re
import json

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload

# Import the existing advanced scraper
from ..advanced_scraper import AdvancedScraper, ScrapingResult as AdvancedScrapingResult

from ..config import settings
from ..database.models import (
    ContentItem, ContentMetadata, UserSession, 
    ContentStatus, FactCheckResult
)
from ..schemas.content import (
    ScrapingRequest, ScrapingResponse, ScrapingResult,
    SearchRequest, SearchResponse, SearchResult,
    ContentQualityMetrics, ContentValidationResponse,
    ContentSimilarity, DeduplicationResponse,
    JobStatus
)

logger = logging.getLogger(__name__)


class BraveSearchClient:
    """
    Client for Brave Search API integration.
    
    Provides methods for web search, news search, and content discovery
    using the Brave Search API with proper error handling and rate limiting.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Brave Search client.
        
        Args:
            api_key: Brave Search API key (defaults to environment variable)
        """
        # Try multiple possible API key sources with better debugging
        self.api_key = api_key or self._get_api_key_from_config()
        
        if not self.api_key:
            logger.warning("Brave Search API key not provided - search functionality will be limited")
            logger.info("To enable Brave Search, set one of these environment variables:")
            logger.info("  - BRAVE_API_KEY")
            logger.info("  - BRAVE_SEARCH_API_KEY") 
            logger.info("  - API_KEY")
            logger.info("Register at https://api.search.brave.com/register to get an API key")
        else:
            logger.info("Brave Search API key configured successfully")
        
        self.base_url = "https://api.search.brave.com/res/v1"
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key or ""
            }
        )
    
    def _get_api_key_from_config(self) -> Optional[str]:
        """
        Get API key from configuration with detailed logging.
        
        Returns:
            API key if found, None otherwise
        """
        # Check all possible sources
        sources = [
            ("settings.brave_search.BRAVE_API_KEY", settings.brave_search.BRAVE_API_KEY),
            ("settings.brave_search.API_KEY", settings.brave_search.API_KEY),
            ("settings.brave_search.BRAVE_SEARCH_API_KEY", settings.brave_search.BRAVE_SEARCH_API_KEY),
            ("os.getenv('BRAVE_API_KEY')", os.getenv("BRAVE_API_KEY")),
            ("os.getenv('BRAVE_SEARCH_API_KEY')", os.getenv("BRAVE_SEARCH_API_KEY")),
            ("os.getenv('API_KEY')", os.getenv("API_KEY"))
        ]
        
        for source_name, value in sources:
            if value:
                logger.debug(f"Found API key from {source_name}")
                return value
        
        logger.debug("No API key found in any configuration source")
        return None
    
    async def search_web(
        self, 
        query: str, 
        count: int = 10,
        country: Optional[str] = None,
        language: Optional[str] = None,
        freshness: Optional[str] = None,
        safe_search: str = "moderate"
    ) -> SearchResponse:
        """
        Perform web search using Brave Search API.
        
        Args:
            query: Search query
            count: Number of results (1-20)
            country: Country code for localized results
            language: Language code for results
            freshness: Freshness filter (pd, pw, pm, py)
            safe_search: Safe search setting
            
        Returns:
            SearchResponse with results
            
        Raises:
            httpx.HTTPError: If API request fails
        """
        if not self.api_key:
            logger.warning("No Brave API key - returning empty search results")
            return SearchResponse(
                query=query,
                total_results=0,
                results=[],
                search_time=0.0,
                suggestions=[]
            )
        
        start_time = time.time()
        
        params = {
            "q": query,
            "count": min(count, 20),  # API limit
            "safesearch": safe_search,
            "result_filter": "web"
        }
        
        if country:
            params["country"] = country
        if language:
            params["search_lang"] = language
        if freshness:
            params["freshness"] = freshness
        
        try:
            response = await self.client.get(f"{self.base_url}/web/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            search_time = time.time() - start_time
            
            # Parse results
            results = []
            web_results = data.get("web", {}).get("results", [])
            
            for result in web_results:
                try:
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        description=result.get("description", ""),
                        source_domain=urlparse(result.get("url", "")).netloc,
                        relevance_score=None  # Brave doesn't provide relevance scores
                    )
                    results.append(search_result)
                except Exception as e:
                    logger.warning(f"Failed to parse search result: {e}")
                    continue
            
            return SearchResponse(
                query=query,
                total_results=len(results),
                results=results,
                search_time=search_time,
                suggestions=data.get("query", {}).get("altered", "").split() if data.get("query", {}).get("altered") else []
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Brave Search API HTTP error: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 401:
                logger.error("Authentication failed - check your Brave Search API key")
                logger.error("Make sure your API key is valid and properly configured")
            elif e.response.status_code == 403:
                logger.error("Access forbidden - check your API key permissions")
            elif e.response.status_code == 429:
                logger.error("Rate limit exceeded - too many requests")
            return SearchResponse(
                query=query,
                total_results=0,
                results=[],
                search_time=time.time() - start_time,
                suggestions=[]
            )
        except httpx.RequestError as e:
            logger.error(f"Brave Search API request error: {e}")
            return SearchResponse(
                query=query,
                total_results=0,
                results=[],
                search_time=time.time() - start_time,
                suggestions=[]
            )
        except Exception as e:
            logger.error(f"Brave Search API unexpected error: {e}")
            return SearchResponse(
                query=query,
                total_results=0,
                results=[],
                search_time=time.time() - start_time,
                suggestions=[]
            )
    
    async def search_news(
        self,
        query: str,
        count: int = 10,
        country: Optional[str] = None,
        freshness: Optional[str] = None
    ) -> SearchResponse:
        """
        Perform news search using Brave Search API.
        
        Args:
            query: Search query
            count: Number of results
            country: Country code for localized results
            freshness: Freshness filter
            
        Returns:
            SearchResponse with news results
        """
        if not self.api_key:
            return SearchResponse(
                query=query,
                total_results=0,
                results=[],
                search_time=0.0,
                suggestions=[]
            )
        
        start_time = time.time()
        
        params = {
            "q": query,
            "count": min(count, 20),
            "result_filter": "news"
        }
        
        if country:
            params["country"] = country
        if freshness:
            params["freshness"] = freshness
        
        try:
            response = await self.client.get(f"{self.base_url}/web/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            search_time = time.time() - start_time
            
            results = []
            news_results = data.get("news", {}).get("results", [])
            
            for result in news_results:
                try:
                    # Parse publication date if available
                    published_date = None
                    if result.get("age"):
                        # Simple age parsing - could be enhanced
                        pass  # Brave provides relative times, would need more sophisticated parsing
                    
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        description=result.get("description", ""),
                        published_date=published_date,
                        source_domain=urlparse(result.get("url", "")).netloc
                    )
                    results.append(search_result)
                except Exception as e:
                    logger.warning(f"Failed to parse news result: {e}")
                    continue
            
            return SearchResponse(
                query=query,
                total_results=len(results),
                results=results,
                search_time=search_time,
                suggestions=[]
            )
            
        except httpx.HTTPError as e:
            logger.error(f"Brave Search API error for news: {e}")
            return SearchResponse(
                query=query,
                total_results=0,
                results=[],
                search_time=time.time() - start_time,
                suggestions=[]
            )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class ContentQualityAnalyzer:
    """
    Analyzes and scores content quality for TruthSeeQ platform.
    
    Provides comprehensive content quality assessment including:
    - Length and readability analysis
    - Structure and formatting evaluation
    - Metadata completeness scoring
    - Source credibility assessment
    """
    
    @staticmethod
    def analyze_content_quality(
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        source_domain: Optional[str] = None
    ) -> ContentQualityMetrics:
        """
        Analyze content quality and return detailed metrics.
        
        Args:
            content: Main content text
            title: Content title
            metadata: Content metadata
            source_domain: Source domain for credibility assessment
            
        Returns:
            ContentQualityMetrics with detailed scores
        """
        # Length score (based on content length)
        content_length = len(content.strip())
        if content_length < 100:
            length_score = 0.2
        elif content_length < 500:
            length_score = 0.5
        elif content_length < 2000:
            length_score = 0.8
        else:
            length_score = 1.0
        
        # Readability score (simplified - counts sentences, paragraphs)
        sentences = len(re.split(r'[.!?]+', content))
        paragraphs = len(content.split('\n\n'))
        avg_sentence_length = content_length / max(sentences, 1)
        
        if avg_sentence_length < 15:
            readability_score = 1.0
        elif avg_sentence_length < 25:
            readability_score = 0.8
        elif avg_sentence_length < 40:
            readability_score = 0.6
        else:
            readability_score = 0.4
        
        # Structure score (based on presence of headings, lists, etc.)
        structure_indicators = [
            bool(re.search(r'^#+\s', content, re.MULTILINE)),  # Markdown headings
            bool(re.search(r'^\*\s|\d+\.\s', content, re.MULTILINE)),  # Lists
            bool(title and len(title.strip()) > 0),  # Has title
            paragraphs > 1  # Multiple paragraphs
        ]
        structure_score = sum(structure_indicators) / len(structure_indicators)
        
        # Metadata completeness score
        metadata = metadata or {}
        expected_metadata = ['author', 'publish_date', 'description', 'tags']
        metadata_completeness = sum(
            1 for key in expected_metadata if key in metadata and metadata[key]
        ) / len(expected_metadata)
        
        # Source credibility (basic domain-based assessment)
        source_credibility = ContentQualityAnalyzer._assess_source_credibility(source_domain)
        
        # Overall score (weighted average)
        overall_score = (
            length_score * 0.25 +
            readability_score * 0.25 +
            structure_score * 0.20 +
            metadata_completeness * 0.15 +
            source_credibility * 0.15
        )
        
        return ContentQualityMetrics(
            length_score=length_score,
            readability_score=readability_score,
            structure_score=structure_score,
            metadata_completeness=metadata_completeness,
            source_credibility=source_credibility,
            overall_score=overall_score
        )
    
    @staticmethod
    def _assess_source_credibility(domain: Optional[str]) -> float:
        """
        Assess source credibility based on domain.
        
        Args:
            domain: Source domain
            
        Returns:
            Credibility score (0-1)
        """
        if not domain:
            return 0.5
        
        domain = domain.lower()
        
        # High credibility domains
        high_credibility = {
            'reuters.com', 'bbc.com', 'apnews.com', 'npr.org',
            'wikipedia.org', 'nature.com', 'science.org',
            'ncbi.nlm.nih.gov', 'who.int', 'cdc.gov',
            'gov.uk', 'europa.eu'
        }
        
        # Medium credibility domains
        medium_credibility = {
            'cnn.com', 'nytimes.com', 'washingtonpost.com',
            'theguardian.com', 'wsj.com', 'forbes.com',
            'bloomberg.com', 'economist.com'
        }
        
        # Check for high credibility
        if any(trusted in domain for trusted in high_credibility):
            return 0.9
        
        # Check for medium credibility
        if any(medium in domain for medium in medium_credibility):
            return 0.7
        
        # Check for common low-credibility indicators
        low_credibility_indicators = [
            'blog', 'wordpress', 'tumblr', 'medium.com',
            'facebook.com', 'twitter.com', 'reddit.com'
        ]
        
        if any(indicator in domain for indicator in low_credibility_indicators):
            return 0.3
        
        # Default score for unknown domains
        return 0.5


class ContentDeduplicator:
    """
    Handles content deduplication using multiple similarity methods.
    
    Provides methods for:
    - Text-based similarity comparison
    - Semantic similarity analysis
    - URL-based duplicate detection
    - Batch deduplication processing
    """
    
    @staticmethod
    def calculate_text_similarity(content1: str, content2: str) -> float:
        """
        Calculate text similarity using basic string comparison.
        
        Args:
            content1: First content text
            content2: Second content text
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize content
        def normalize(text: str) -> str:
            return re.sub(r'\s+', ' ', text.lower().strip())
        
        norm1 = normalize(content1)
        norm2 = normalize(content2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Simple character-based similarity
        if norm1 == norm2:
            return 1.0
        
        # Calculate overlap ratio
        shorter, longer = (norm1, norm2) if len(norm1) < len(norm2) else (norm2, norm1)
        
        # Find longest common substring ratio
        max_overlap = 0
        for i in range(len(shorter)):
            for j in range(i + 1, len(shorter) + 1):
                substring = shorter[i:j]
                if substring in longer:
                    max_overlap = max(max_overlap, len(substring))
        
        return max_overlap / max(len(shorter), 1)
    
    @staticmethod
    def calculate_url_similarity(url1: str, url2: str) -> float:
        """
        Calculate URL similarity based on domain and path.
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            Similarity score (0-1)
        """
        try:
            parsed1 = urlparse(url1)
            parsed2 = urlparse(url2)
            
            # Exact match
            if url1 == url2:
                return 1.0
            
            # Different domains
            if parsed1.netloc != parsed2.netloc:
                return 0.0
            
            # Same domain, compare paths
            path1 = parsed1.path.strip('/')
            path2 = parsed2.path.strip('/')
            
            if path1 == path2:
                return 0.9  # Same path, might differ in query params
            
            # Calculate path similarity
            path_parts1 = path1.split('/')
            path_parts2 = path2.split('/')
            
            common_parts = 0
            for p1, p2 in zip(path_parts1, path_parts2):
                if p1 == p2:
                    common_parts += 1
                else:
                    break
            
            max_parts = max(len(path_parts1), len(path_parts2))
            if max_parts == 0:
                return 0.8
            
            return 0.3 + (common_parts / max_parts) * 0.5
            
        except Exception:
            return 0.0
    
    @staticmethod
    def generate_content_hash(content: str, title: Optional[str] = None) -> str:
        """
        Generate a hash for content deduplication.
        
        Args:
            content: Content text
            title: Optional title
            
        Returns:
            Content hash string
        """
        # Normalize content for consistent hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        if title:
            normalized = title.lower().strip() + " " + normalized
        
        return hashlib.sha256(normalized.encode()).hexdigest()


class ScraperService:
    """
    Main scraper service integrating with TruthSeeQ platform.
    
    Provides comprehensive scraping functionality including:
    - Integration with advanced_scraper.py
    - Brave Search API integration
    - Database storage and retrieval
    - Content quality analysis and deduplication
    - Background task management
    - Rate limiting and error handling
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize scraper service.
        
        Args:
            db_session: Database session for content storage
        """
        self.db = db_session
        self.scraper = AdvancedScraper(
            max_retries=settings.scraping.SCRAPER_MAX_CONCURRENT // 2,
            timeout=settings.scraping.SCRAPER_TIMEOUT,
            delay_between_requests=settings.scraping.SCRAPER_REQUEST_DELAY,
        )
        self.brave_client = BraveSearchClient()
        self.quality_analyzer = ContentQualityAnalyzer()
        self.deduplicator = ContentDeduplicator()
        
        # Job tracking
        self.active_jobs: Dict[UUID, JobStatus] = {}
        
        logger.info("ScraperService initialized with advanced scraper integration")
    
    async def search_content(self, request: SearchRequest) -> SearchResponse:
        """
        Search for content using Brave Search API.
        
        Args:
            request: Search request parameters
            
        Returns:
            SearchResponse with search results
        """
        logger.info(f"Searching for content: {request.query}")
        
        if request.search_type == "news":
            return await self.brave_client.search_news(
                query=request.query,
                count=request.max_results,
                country=request.country,
                freshness=request.freshness
            )
        else:
            return await self.brave_client.search_web(
                query=request.query,
                count=request.max_results,
                country=request.country,
                language=request.language,
                freshness=request.freshness,
                safe_search=request.safe_search
            )
    
    async def scrape_urls(self, request: ScrapingRequest, session_id: Optional[UUID] = None) -> ScrapingResponse:
        """
        Scrape multiple URLs and store results in database.
        
        Args:
            request: Scraping request with URLs and parameters
            session_id: Optional user session ID
            
        Returns:
            ScrapingResponse with results
        """
        job_id = uuid4()
        start_time = time.time()
        
        logger.info(f"Starting scraping job {job_id} for {len(request.urls)} URLs")
        
        # Create job status
        job_status = JobStatus(
            job_id=job_id,
            job_type="scraping",
            status="running",
            progress=0.0,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow()
        )
        self.active_jobs[job_id] = job_status
        
        try:
            # Convert URLs to strings and check for existing content
            url_strings = [str(url) for url in request.urls]
            
            # Check for existing content if not forcing rescrape
            existing_content = {}
            if not request.force_rescrape:
                existing_content = await self._get_existing_content(url_strings)
                logger.info(f"Found {len(existing_content)} existing content items")
            
            # Filter URLs that need scraping
            urls_to_scrape = [
                url for url in url_strings 
                if request.force_rescrape or url not in existing_content
            ]
            
            logger.info(f"Scraping {len(urls_to_scrape)} new URLs")
            
            # Perform scraping using advanced scraper
            scraping_results = []
            if urls_to_scrape:
                advanced_results = self.scraper.scrape_batch(
                    urls_to_scrape, 
                    max_workers=settings.scraping.SCRAPER_MAX_CONCURRENT
                )
                
                # Process and store results
                for i, result in enumerate(advanced_results):
                    progress = (i + 1) / len(advanced_results)
                    job_status.progress = progress * 0.8  # Reserve 20% for post-processing
                    
                    scraping_result = await self._process_scraping_result(
                        result, request.include_metadata, session_id
                    )
                    scraping_results.append(scraping_result)
            
            # Add existing content to results
            for url, content_item in existing_content.items():
                scraping_result = ScrapingResult(
                    url=url,
                    success=True,
                    content=content_item.content,
                    title=content_item.title,
                    metadata={},  # Could populate from database if needed
                    method_used="existing",
                    response_time=0.0,
                    quality_score=None  # Could calculate if needed
                )
                scraping_results.append(scraping_result)
            
            # Final statistics
            successful = sum(1 for r in scraping_results if r.success)
            failed = len(scraping_results) - successful
            processing_time = time.time() - start_time
            
            # Update job status
            job_status.status = "completed"
            job_status.progress = 1.0
            job_status.completed_at = datetime.utcnow()
            job_status.result_summary = {
                "total_urls": len(request.urls),
                "successful": successful,
                "failed": failed,
                "processing_time": processing_time
            }
            
            logger.info(f"Scraping job {job_id} completed: {successful} successful, {failed} failed")
            
            return ScrapingResponse(
                job_id=job_id,
                total_urls=len(request.urls),
                successful=successful,
                failed=failed,
                results=scraping_results,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Scraping job {job_id} failed: {e}")
            job_status.status = "failed"
            job_status.error_message = str(e)
            job_status.completed_at = datetime.utcnow()
            raise
    
    async def _get_existing_content(self, urls: List[str]) -> Dict[str, ContentItem]:
        """
        Get existing content items from database.
        
        Args:
            urls: List of URLs to check
            
        Returns:
            Dictionary mapping URLs to ContentItem objects
        """
        stmt = select(ContentItem).where(ContentItem.url.in_(urls))
        result = await self.db.execute(stmt)
        content_items = result.scalars().all()
        
        return {item.url: item for item in content_items}
    
    async def _process_scraping_result(
        self, 
        result: AdvancedScrapingResult, 
        include_metadata: bool,
        session_id: Optional[UUID]
    ) -> ScrapingResult:
        """
        Process a scraping result from advanced scraper.
        
        Args:
            result: Advanced scraper result
            include_metadata: Whether to extract metadata
            session_id: Optional user session ID
            
        Returns:
            Processed ScrapingResult
        """
        if not result.success:
            return ScrapingResult(
                url=result.url,
                success=False,
                error_message=result.error_message,
                method_used=result.method_used,
                response_time=result.response_time
            )
        
        try:
            # Analyze content quality
            quality_metrics = self.quality_analyzer.analyze_content_quality(
                content=result.content,
                title=result.title,
                metadata=result.metadata,
                source_domain=urlparse(result.url).netloc
            )
            
            # Store content in database
            content_item = ContentItem(
                url=result.url,
                title=result.title,
                content=result.content,
                source_domain=urlparse(result.url).netloc,
                status=ContentStatus.COMPLETED
            )
            
            self.db.add(content_item)
            await self.db.flush()  # Get the ID
            
            # Store metadata if requested
            if include_metadata and result.metadata:
                for key, value in result.metadata.items():
                    metadata_item = ContentMetadata(
                        content_id=content_item.id,
                        metadata_type=key,
                        metadata_value=str(value)
                    )
                    self.db.add(metadata_item)
            
            await self.db.commit()
            
            return ScrapingResult(
                url=result.url,
                success=True,
                content=result.content,
                title=result.title,
                metadata=result.metadata,
                method_used=result.method_used,
                response_time=result.response_time,
                quality_score=quality_metrics.overall_score
            )
            
        except Exception as e:
            logger.error(f"Failed to process scraping result for {result.url}: {e}")
            await self.db.rollback()
            
            return ScrapingResult(
                url=result.url,
                success=False,
                error_message=f"Processing error: {str(e)}",
                method_used=result.method_used,
                response_time=result.response_time
            )
    
    async def validate_content(self, content_id: int) -> ContentValidationResponse:
        """
        Validate content quality and authenticity.
        
        Args:
            content_id: ID of content to validate
            
        Returns:
            ContentValidationResponse with validation results
        """
        # Get content from database
        stmt = select(ContentItem).where(ContentItem.id == content_id)
        result = await self.db.execute(stmt)
        content_item = result.scalar_one_or_none()
        
        if not content_item:
            raise ValueError(f"Content item {content_id} not found")
        
        # Analyze quality
        quality_metrics = self.quality_analyzer.analyze_content_quality(
            content=content_item.content,
            title=content_item.title,
            source_domain=content_item.source_domain
        )
        
        # Determine validation result
        is_valid = quality_metrics.overall_score >= 0.6
        confidence = quality_metrics.overall_score
        
        # Generate recommendations
        recommendations = []
        if quality_metrics.length_score < 0.5:
            recommendations.append("Content is too short - consider expanding with more details")
        if quality_metrics.readability_score < 0.5:
            recommendations.append("Content readability could be improved - use shorter sentences")
        if quality_metrics.structure_score < 0.5:
            recommendations.append("Content structure could be improved - add headings and paragraphs")
        if quality_metrics.source_credibility < 0.5:
            recommendations.append("Source credibility is questionable - verify with additional sources")
        
        return ContentValidationResponse(
            content_id=content_id,
            validation_type="quality",
            is_valid=is_valid,
            confidence=confidence,
            quality_metrics=quality_metrics,
            issues=[],
            recommendations=recommendations
        )
    
    async def deduplicate_content(self, content_ids: List[int], similarity_threshold: float = 0.8) -> DeduplicationResponse:
        """
        Perform content deduplication analysis.
        
        Args:
            content_ids: List of content IDs to analyze
            similarity_threshold: Threshold for considering content duplicate
            
        Returns:
            DeduplicationResponse with deduplication results
        """
        # Get content items
        stmt = select(ContentItem).where(ContentItem.id.in_(content_ids))
        result = await self.db.execute(stmt)
        content_items = {item.id: item for item in result.scalars().all()}
        
        similarities = []
        duplicate_groups = []
        duplicates_removed = 0
        
        # Compare all pairs
        processed_ids = set()
        for i, id1 in enumerate(content_ids):
            if id1 in processed_ids or id1 not in content_items:
                continue
                
            current_group = [id1]
            
            for id2 in content_ids[i+1:]:
                if id2 in processed_ids or id2 not in content_items:
                    continue
                
                # Calculate similarities
                text_sim = self.deduplicator.calculate_text_similarity(
                    content_items[id1].content,
                    content_items[id2].content
                )
                
                url_sim = self.deduplicator.calculate_url_similarity(
                    content_items[id1].url,
                    content_items[id2].url
                )
                
                # Combined similarity (weighted)
                combined_sim = (text_sim * 0.7) + (url_sim * 0.3)
                
                similarity = ContentSimilarity(
                    content_id_1=id1,
                    content_id_2=id2,
                    similarity_score=combined_sim,
                    similarity_type="combined",
                    details={
                        "text_similarity": text_sim,
                        "url_similarity": url_sim
                    }
                )
                similarities.append(similarity)
                
                # Check if duplicate
                if combined_sim >= similarity_threshold:
                    current_group.append(id2)
                    processed_ids.add(id2)
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                duplicates_removed += len(current_group) - 1  # Keep one from each group
            
            processed_ids.add(id1)
        
        return DeduplicationResponse(
            total_content=len(content_ids),
            duplicates_found=sum(len(group) - 1 for group in duplicate_groups),
            duplicates_removed=duplicates_removed,
            duplicate_groups=duplicate_groups,
            similarities=similarities
        )
    
    def get_job_status(self, job_id: UUID) -> Optional[JobStatus]:
        """
        Get status of a background job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobStatus if found, None otherwise
        """
        return self.active_jobs.get(job_id)
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Clean up old job status records.
        
        Args:
            max_age_hours: Maximum age in hours for keeping job records
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = [
            job_id for job_id, job in self.active_jobs.items()
            if job.created_at < cutoff_time
        ]
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} old job records")
    
    async def close(self):
        """Clean up resources."""
        await self.brave_client.close()
        logger.info("ScraperService closed")
