"""
Custom Tools for TruthSeeQ Workflows

This module provides LangChain tools for web search, content scraping, and analysis
that integrate with the advanced scraper and external APIs.
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse, quote_plus

import aiohttp
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import local modules
from ..config import settings
from ..advanced_scraper import AdvancedScraper, ScrapingResult

logger = logging.getLogger(__name__)


class WebSearchResult(BaseModel):
    """Model for web search results."""
    title: str
    url: str
    snippet: str
    relevance_score: float = Field(description="Relevance score from 0 to 1")


class BraveSearchTool:
    """
    Tool for performing web searches using Brave Search API.
    
    This tool provides access to Brave Search for finding relevant sources
    and information for fact-checking and verification.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Brave Search tool.
        
        Args:
            api_key: Brave Search API key (optional, will use config if not provided)
        """
        self.api_key = api_key or settings.brave_search.API_KEY
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            })
    
    def search(
        self, 
        query: str, 
        count: int = 10, 
        search_lang: str = "en_US",
        country: str = "US"
    ) -> List[WebSearchResult]:
        """
        Perform a web search using Brave Search.
        
        Args:
            query: Search query string
            count: Number of results to return (max 20)
            search_lang: Search language
            country: Country for search results
            
        Returns:
            List of search results
        """
        if not self.api_key:
            logger.warning("Brave Search API key not configured, using fallback search")
            return self._fallback_search(query, count)
        
        try:
            params = {
                "q": query,
                "count": min(count, 20),  # Brave API limit
                "search_lang": search_lang,
                "country": country,
                "safesearch": "moderate"
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for result in data.get("web", {}).get("results", []):
                results.append(WebSearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("description", ""),
                    relevance_score=0.8  # Default relevance score
                ))
            
            logger.info(f"Brave Search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Brave Search failed: {e}")
            return self._fallback_search(query, count)
    
    def _fallback_search(self, query: str, count: int) -> List[WebSearchResult]:
        """
        Fallback search using DuckDuckGo Instant Answer API.
        
        Args:
            query: Search query
            count: Number of results
            
        Returns:
            List of search results
        """
        try:
            # Use DuckDuckGo Instant Answer API as fallback
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Add abstract if available
            if data.get("Abstract"):
                results.append(WebSearchResult(
                    title=data.get("AbstractSource", "DuckDuckGo"),
                    url=data.get("AbstractURL", ""),
                    snippet=data.get("Abstract", ""),
                    relevance_score=0.9
                ))
            
            # Add related topics
            for topic in data.get("RelatedTopics", [])[:count-1]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(WebSearchResult(
                        title="Related Topic",
                        url=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", ""),
                        relevance_score=0.7
                    ))
            
            logger.info(f"Fallback search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []


class ContentScrapingTool:
    """
    Tool for scraping content from URLs using the advanced scraper.
    
    This tool provides intelligent content extraction with anti-detection
    capabilities and LLM-optimized output.
    """
    
    def __init__(self):
        """Initialize content scraping tool."""
        self.scraper = AdvancedScraper(
            max_retries=2,
            timeout=15,
            delay_between_requests=0.5,
            use_cookies=True,
            use_method_cache=True
        )
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a single URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing scraped content and metadata
        """
        try:
            result = self.scraper.scrape(url)
            
            return {
                "url": url,
                "success": result.success,
                "content": result.content,
                "title": result.title,
                "metadata": result.metadata or {},
                "method_used": result.method_used,
                "response_time": result.response_time,
                "error_message": result.error_message
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                "url": url,
                "success": False,
                "content": "",
                "title": None,
                "metadata": {},
                "method_used": "failed",
                "response_time": 0.0,
                "error_message": str(e)
            }
    
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape content from multiple URLs in parallel.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraping results
        """
        try:
            results = self.scraper.scrape_batch(urls, max_workers=8)
            
            return [
                {
                    "url": result.url,
                    "success": result.success,
                    "content": result.content,
                    "title": result.title,
                    "metadata": result.metadata or {},
                    "method_used": result.method_used,
                    "response_time": result.response_time,
                    "error_message": result.error_message
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to scrape URLs: {e}")
            return [
                {
                    "url": url,
                    "success": False,
                    "content": "",
                    "title": None,
                    "metadata": {},
                    "method_used": "failed",
                    "response_time": 0.0,
                    "error_message": str(e)
                }
                for url in urls
            ]


class FactCheckingDatabaseTool:
    """
    Tool for querying fact-checking databases and sources.
    
    This tool provides access to various fact-checking databases
    and known reliable sources for verification.
    """
    
    def __init__(self):
        """Initialize fact-checking database tool."""
        self.fact_checking_sources = {
            "snopes": "https://www.snopes.com",
            "factcheck": "https://www.factcheck.org",
            "politifact": "https://www.politifact.com",
            "reuters_fact_check": "https://www.reuters.com/fact-check",
            "ap_fact_check": "https://apnews.com/hub/fact-checking",
            "bbc_reality_check": "https://www.bbc.com/news/reality_check"
        }
        
        self.reliable_domains = {
            "government": [
                ".gov", ".mil", ".state.gov", ".whitehouse.gov"
            ],
            "academic": [
                ".edu", ".ac.uk", ".ac.za", ".ac.jp"
            ],
            "news": [
                "reuters.com", "ap.org", "bbc.com", "npr.org",
                "pbs.org", "cbc.ca", "abc.net.au"
            ],
            "fact_checking": [
                "snopes.com", "factcheck.org", "politifact.com",
                "reuters.com", "apnews.com"
            ]
        }
    
    def check_domain_reliability(self, domain: str) -> Dict[str, Any]:
        """
        Check the reliability of a domain based on known patterns.
        
        Args:
            domain: Domain to check
            
        Returns:
            Dictionary with reliability assessment
        """
        domain_lower = domain.lower()
        
        # Check for government domains
        is_government = any(gov_domain in domain_lower for gov_domain in self.reliable_domains["government"])
        
        # Check for academic domains
        is_academic = any(acad_domain in domain_lower for acad_domain in self.reliable_domains["academic"])
        
        # Check for reliable news domains
        is_reliable_news = any(news_domain in domain_lower for news_domain in self.reliable_domains["news"])
        
        # Check for fact-checking domains
        is_fact_checking = any(fc_domain in domain_lower for fc_domain in self.reliable_domains["fact_checking"])
        
        # Calculate reliability score
        reliability_score = 0.0
        trust_indicators = []
        
        if is_government:
            reliability_score += 0.9
            trust_indicators.append("government_domain")
        
        if is_academic:
            reliability_score += 0.8
            trust_indicators.append("academic_domain")
        
        if is_reliable_news:
            reliability_score += 0.7
            trust_indicators.append("reliable_news_source")
        
        if is_fact_checking:
            reliability_score += 0.8
            trust_indicators.append("fact_checking_source")
        
        # Check for red flags
        red_flags = []
        suspicious_patterns = [
            "fake", "conspiracy", "truth", "real", "exposed", "secret",
            "they_dont_want_you_to_know", "shocking", "amazing"
        ]
        
        for pattern in suspicious_patterns:
            if pattern in domain_lower:
                red_flags.append(f"suspicious_pattern: {pattern}")
                reliability_score -= 0.3
        
        return {
            "domain": domain,
            "reliability_score": max(0.0, min(1.0, reliability_score)),
            "is_government": is_government,
            "is_academic": is_academic,
            "is_reliable_news": is_reliable_news,
            "is_fact_checking": is_fact_checking,
            "trust_indicators": trust_indicators,
            "red_flags": red_flags,
            "assessment": self._get_reliability_assessment(reliability_score)
        }
    
    def _get_reliability_assessment(self, score: float) -> str:
        """
        Get human-readable reliability assessment.
        
        Args:
            score: Reliability score
            
        Returns:
            Assessment string
        """
        if score >= 0.8:
            return "highly_reliable"
        elif score >= 0.6:
            return "reliable"
        elif score >= 0.4:
            return "moderately_reliable"
        elif score >= 0.2:
            return "low_reliability"
        else:
            return "unreliable"
    
    def search_fact_checking_databases(self, query: str) -> List[Dict[str, Any]]:
        """
        Search fact-checking databases for relevant fact checks.
        
        Args:
            query: Search query
            
        Returns:
            List of fact-check results
        """
        # This would integrate with fact-checking database APIs
        # For now, return a placeholder structure
        return [
            {
                "source": "snopes",
                "title": "Sample fact check",
                "url": "https://www.snopes.com/fact-check/",
                "verdict": "false",
                "relevance_score": 0.7
            }
        ]


# LangChain tool definitions
@tool
def web_search(query: str, count: int = 10) -> List[Dict[str, Any]]:
    """
    Search the web for information using Brave Search.
    
    Args:
        query: Search query string
        count: Number of results to return (max 20)
        
    Returns:
        List of search results with title, url, snippet, and relevance score
    """
    search_tool = BraveSearchTool()
    results = search_tool.search(query, count)
    
    return [
        {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "relevance_score": result.relevance_score
        }
        for result in results
    ]


@tool
def scrape_content(url: str) -> Dict[str, Any]:
    """
    Scrape content from a URL using advanced anti-detection methods.
    
    Args:
        url: URL to scrape
        
    Returns:
        Dictionary with scraped content, title, metadata, and success status
    """
    scraping_tool = ContentScrapingTool()
    return scraping_tool.scrape_url(url)


@tool
def scrape_multiple_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape content from multiple URLs in parallel.
    
    Args:
        urls: List of URLs to scrape
        
    Returns:
        List of scraping results for each URL
    """
    scraping_tool = ContentScrapingTool()
    return scraping_tool.scrape_urls(urls)


@tool
def check_domain_reliability(domain: str) -> Dict[str, Any]:
    """
    Check the reliability and credibility of a domain.
    
    Args:
        domain: Domain to check (e.g., "example.com")
        
    Returns:
        Dictionary with reliability assessment, trust indicators, and red flags
    """
    db_tool = FactCheckingDatabaseTool()
    return db_tool.check_domain_reliability(domain)


@tool
def search_fact_checking_databases(query: str) -> List[Dict[str, Any]]:
    """
    Search fact-checking databases for relevant fact checks.
    
    Args:
        query: Search query for fact checks
        
    Returns:
        List of fact-check results from various databases
    """
    db_tool = FactCheckingDatabaseTool()
    return db_tool.search_fact_checking_databases(query)


# Tool registry for easy access
WORKFLOW_TOOLS = {
    "web_search": web_search,
    "scrape_content": scrape_content,
    "scrape_multiple_urls": scrape_multiple_urls,
    "check_domain_reliability": check_domain_reliability,
    "search_fact_checking_databases": search_fact_checking_databases
}


def get_workflow_tools() -> List:
    """
    Get all available workflow tools.
    
    Returns:
        List of LangChain tools
    """
    return list(WORKFLOW_TOOLS.values())


def get_tool_by_name(name: str):
    """
    Get a specific tool by name.
    
    Args:
        name: Tool name
        
    Returns:
        Tool function or None if not found
    """
    return WORKFLOW_TOOLS.get(name) 