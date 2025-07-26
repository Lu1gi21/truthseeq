"""
Web search tools for LangGraph workflows.

This module provides tools for web searching and content discovery
that can be used within LangGraph workflows for fact-checking and
source verification.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from app.services.scraper_service import ScraperService
from app.schemas.content import SearchRequest, SearchResult

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    
    query: str = Field(
        ..., 
        description="Search query to execute",
        example="climate change latest research 2024"
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=50
    )
    search_type: str = Field(
        default="web",
        description="Type of search (web, news, academic)",
        example="web"
    )
    freshness: Optional[str] = Field(
        default=None,
        description="Content freshness filter (pd=past day, pw=past week, pm=past month)",
        example="pw"
    )


class WebSearchTool(BaseTool):
    """
    Tool for performing web searches to find relevant sources for fact-checking.
    
    This tool integrates with search APIs to find authoritative sources
    that can be used to verify claims and statements in content.
    """
    
    name: str = "web_search"
    description: str = """
    Search the web for relevant information to verify claims or find authoritative sources.
    Use this tool when you need to find recent, reliable information about a topic
    or when verifying specific claims in content.
    """
    args_schema: type[BaseModel] = WebSearchInput
    
    def __init__(self, scraper_service: Optional[ScraperService] = None):
        """Initialize the web search tool."""
        super().__init__()
        self.scraper_service = scraper_service or ScraperService()
    
    def _run(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "web",
        freshness: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute web search and return results.
        
        Args:
            query: Search query to execute
            max_results: Maximum number of results to return
            search_type: Type of search (web, news, academic)
            freshness: Content freshness filter
            run_manager: LangChain callback manager
            
        Returns:
            List of search results with metadata
        """
        try:
            # Create search request
            search_request = SearchRequest(
                query=query,
                max_results=max_results,
                search_type=search_type,
                freshness=freshness
            )
            
            # Execute search using scraper service
            search_results = self.scraper_service.search_content(search_request)
            
            # Format results for LangGraph
            formatted_results = []
            for result in search_results.results:
                formatted_results.append({
                    "title": result.title,
                    "url": str(result.url),
                    "description": result.description,
                    "source_domain": result.source_domain,
                    "relevance_score": result.relevance_score,
                    "published_date": result.published_date.isoformat() if result.published_date else None
                })
            
            logger.info(f"Web search completed for query '{query}' with {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in web search tool: {str(e)}")
            return []


class SourceCredibilityChecker(BaseTool):
    """
    Tool for checking the credibility of web sources.
    
    This tool analyzes domain reputation, source type, and other factors
    to assess the credibility of sources found during fact-checking.
    """
    
    name: str = "source_credibility_checker"
    description: str = """
    Check the credibility and reputation of a web source or domain.
    Use this tool to assess whether a source is reliable and authoritative
    before using it for fact-checking.
    """
    
    def __init__(self):
        """Initialize the source credibility checker."""
        super().__init__()
        # Known credible domains (could be expanded with a database)
        self.credible_domains = {
            "reuters.com", "ap.org", "bbc.com", "npr.org", "nytimes.com",
            "washingtonpost.com", "wsj.com", "nature.com", "science.org",
            "nih.gov", "who.int", "cdc.gov", "un.org", "europa.eu"
        }
        
        # Known unreliable domains
        self.unreliable_domains = {
            "infowars.com", "naturalnews.com", "beforeitsnews.com"
        }
    
    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Check the credibility of a source URL.
        
        Args:
            url: URL to check for credibility
            run_manager: LangChain callback manager
            
        Returns:
            Dictionary with credibility assessment
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Check against known credible/unreliable domains
            if domain in self.credible_domains:
                credibility_score = 0.9
                reputation = "high"
                reason = "Known credible news organization or academic institution"
            elif domain in self.unreliable_domains:
                credibility_score = 0.1
                reputation = "low"
                reason = "Known unreliable source with history of misinformation"
            else:
                # Basic heuristics for unknown domains
                credibility_score = self._assess_unknown_domain(domain, parsed_url)
                reputation = "medium" if credibility_score > 0.5 else "low"
                reason = "Domain assessment based on URL structure and patterns"
            
            return {
                "url": url,
                "domain": domain,
                "credibility_score": credibility_score,
                "reputation": reputation,
                "reason": reason,
                "is_known_source": domain in (self.credible_domains | self.unreliable_domains)
            }
            
        except Exception as e:
            logger.error(f"Error in source credibility checker: {str(e)}")
            return {
                "url": url,
                "credibility_score": 0.0,
                "reputation": "unknown",
                "reason": f"Error assessing credibility: {str(e)}",
                "is_known_source": False
            }
    
    def _assess_unknown_domain(self, domain: str, parsed_url) -> float:
        """
        Assess credibility of unknown domains using basic heuristics.
        
        Args:
            domain: Domain name to assess
            parsed_url: Parsed URL object
            
        Returns:
            Credibility score between 0.0 and 1.0
        """
        score = 0.5  # Base score for unknown domains
        
        # Check for HTTPS (security indicator)
        if parsed_url.scheme == "https":
            score += 0.1
        
        # Check for educational/academic domains
        if any(edu in domain for edu in [".edu", ".ac.", "university", "college"]):
            score += 0.2
        
        # Check for government domains
        if any(gov in domain for gov in [".gov", ".mil", "government"]):
            score += 0.2
        
        # Check for news-related domains
        if any(news in domain for news in ["news", "times", "post", "tribune", "herald"]):
            score += 0.1
        
        # Penalize suspicious patterns
        if any(suspicious in domain for suspicious in ["clickbait", "fake", "hoax", "conspiracy"]):
            score -= 0.3
        
        return max(0.0, min(1.0, score))


class ContentExtractor(BaseTool):
    """
    Tool for extracting content from web pages.
    
    This tool uses the scraper service to extract and clean content
    from web pages for further analysis in fact-checking workflows.
    """
    
    name: str = "content_extractor"
    description: str = """
    Extract and clean content from a web page URL.
    Use this tool to get the main content from a webpage for analysis
    or to verify information from a source.
    """
    
    def __init__(self, scraper_service: Optional[ScraperService] = None):
        """Initialize the content extractor."""
        super().__init__()
        self.scraper_service = scraper_service or ScraperService()
    
    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Extract content from a web page.
        
        Args:
            url: URL to extract content from
            run_manager: LangChain callback manager
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            # Use scraper service to extract content
            scraping_request = SearchRequest(urls=[url])
            scraping_result = self.scraper_service.scrape_content(scraping_request)
            
            if not scraping_result.results:
                return {
                    "url": url,
                    "success": False,
                    "error": "No content extracted"
                }
            
            result = scraping_result.results[0]
            
            return {
                "url": url,
                "success": result.success,
                "title": result.title,
                "content": result.content,
                "metadata": result.metadata,
                "quality_score": result.quality_score,
                "method_used": result.method_used
            }
            
        except Exception as e:
            logger.error(f"Error in content extractor: {str(e)}")
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }
