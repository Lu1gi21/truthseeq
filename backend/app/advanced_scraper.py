"""
Advanced web scraper with anti-detection capabilities and LLM-optimized output.

This module provides a robust web scraping solution that can handle 403 errors,
rate limiting, and other anti-bot measures while returning clean, structured
content suitable for LLM processing.
"""

import os
import time
import random
import requests
import threading
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path

# Web scraping and parsing
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import undetected_chromedriver as uc

# Content processing
import re
import markdown
from readability import Document
import trafilatura
from trafilatura.settings import use_config

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Result of a web scraping operation."""
    url: str
    content: str
    title: Optional[str] = None
    metadata: Optional[Dict] = None
    success: bool = True
    error_message: Optional[str] = None
    method_used: str = "unknown"
    response_time: float = 0.0

class UserAgentRotator:
    """Rotates user agents to avoid detection."""
    
    USER_AGENTS = [
        # Modern Chrome
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        
        # Firefox
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        
        # Safari
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        
        # Edge
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
    ]
    
    def __init__(self):
        self.current_index = 0
        self.lock = threading.Lock()
    
    def get_random_agent(self) -> str:
        """Get a random user agent."""
        return random.choice(self.USER_AGENTS)
    
    def get_next_agent(self) -> str:
        """Get the next user agent in rotation."""
        with self.lock:
            agent = self.USER_AGENTS[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.USER_AGENTS)
            return agent

class CookieManager:
    """Manages cookies for persistent sessions across scraping attempts."""
    
    def __init__(self, cookie_file: str = "scraping_cookies.json"):
        self.cookie_file = cookie_file
        self.cookies = self._load_cookies()
    
    def _load_cookies(self) -> Dict[str, Dict]:
        """Load cookies from file."""
        try:
            if os.path.exists(self.cookie_file):
                with open(self.cookie_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cookies: {e}")
        return {}
    
    def _save_cookies(self):
        """Save cookies to file."""
        try:
            with open(self.cookie_file, 'w') as f:
                json.dump(self.cookies, f)
        except Exception as e:
            logger.warning(f"Failed to save cookies: {e}")
    
    def get_cookies_for_domain(self, domain: str) -> List[Dict]:
        """Get cookies for a specific domain."""
        return self.cookies.get(domain, [])
    
    def set_cookies_for_domain(self, domain: str, cookies: List[Dict]):
        """Set cookies for a specific domain."""
        self.cookies[domain] = cookies
        self._save_cookies()
    
    def add_cookie(self, domain: str, name: str, value: str, **kwargs):
        """Add a single cookie for a domain."""
        if domain not in self.cookies:
            self.cookies[domain] = []
        
        cookie = {"name": name, "value": value, **kwargs}
        self.cookies[domain].append(cookie)
        self._save_cookies()

class ProxyManager:
    """Manages proxy rotation for avoiding IP-based blocking."""
    
    def __init__(self, proxy_list: Optional[List[str]] = None):
        self.proxies = proxy_list or []
        self.current_index = 0
        self.lock = threading.Lock()
        self.failed_proxies = set()
    
    def add_proxy(self, proxy: str):
        """Add a proxy to the rotation."""
        if proxy not in self.proxies:
            self.proxies.append(proxy)
    
    def get_next_proxy(self) -> Optional[Dict[str, str]]:
        """Get the next proxy in rotation."""
        if not self.proxies:
            return None
            
        with self.lock:
            # Skip failed proxies
            while self.current_index < len(self.proxies):
                proxy = self.proxies[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.proxies)
                
                if proxy not in self.failed_proxies:
                    return {"http": proxy, "https": proxy}
            
            # If all proxies failed, reset and try again
            self.failed_proxies.clear()
            self.current_index = 0
            return self.get_next_proxy()
    
    def mark_proxy_failed(self, proxy: str):
        """Mark a proxy as failed."""
        self.failed_proxies.add(proxy)

class MethodCacheManager:
    """Manages caching of successful scraping methods per domain to avoid repeated failures."""
    
    def __init__(self, cache_file: str = "scraping_method_cache.json"):
        self.cache_file = cache_file
        self.method_cache = self._load_cache()
        self.lock = threading.Lock()
        
        # Known problematic domains that typically require specific methods
        self.known_domains = {
            "crunchbase.com": "selenium",
            "pitchbook.com": "selenium", 
            "linkedin.com": "selenium",
            "zoominfo.com": "selenium",
            "rocketreach.co": "selenium",
            "radaris.com": "selenium",
            "wikipedia.org": "requests",
            "investopedia.com": "requests",
            "medium.com": "requests",
            "github.com": "requests",
            "stackoverflow.com": "requests",
            "reddit.com": "selenium",
            "twitter.com": "selenium",
            "facebook.com": "selenium",
            "instagram.com": "selenium"
        }
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load method cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load method cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save method cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.method_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save method cache: {e}")
    
    def get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc.lower()
    
    def get_optimal_method_order(self, url: str) -> List[str]:
        """
        Get the optimal order of methods to try for a domain.
        
        Returns:
            List of method names in order of preference: ["requests", "selenium", "undetected"]
        """
        domain = self.get_domain(url)
        
        with self.lock:
            # Check if we have a cached successful method for this domain
            if domain in self.method_cache:
                successful_method = self.method_cache[domain].get("successful_method")
                if successful_method:
                    # Return the successful method first, then others
                    if successful_method == "requests":
                        return ["requests", "selenium", "undetected"]
                    elif successful_method == "selenium":
                        return ["selenium", "undetected", "requests"]
                    elif successful_method == "undetected":
                        return ["undetected", "selenium", "requests"]
            
            # Check known domains
            if domain in self.known_domains:
                known_method = self.known_domains[domain]
                if known_method == "requests":
                    return ["requests", "selenium", "undetected"]
                else:
                    return ["selenium", "undetected", "requests"]
            
            # Default order for unknown domains
            return ["requests", "selenium", "undetected"]
    
    def record_success(self, url: str, method: str):
        """Record a successful scraping method for a domain."""
        domain = self.get_domain(url)
        
        with self.lock:
            if domain not in self.method_cache:
                self.method_cache[domain] = {}
            
            self.method_cache[domain]["successful_method"] = method
            self.method_cache[domain]["last_success"] = time.time()
            self.method_cache[domain]["success_count"] = self.method_cache[domain].get("success_count", 0) + 1
            
            self._save_cache()
    
    def record_failure(self, url: str, method: str):
        """Record a failed scraping method for a domain."""
        domain = self.get_domain(url)
        
        with self.lock:
            if domain not in self.method_cache:
                self.method_cache[domain] = {}
            
            if "failed_methods" not in self.method_cache[domain]:
                self.method_cache[domain]["failed_methods"] = {}
            
            self.method_cache[domain]["failed_methods"][method] = {
                "last_failure": time.time(),
                "failure_count": self.method_cache[domain]["failed_methods"].get(method, {}).get("failure_count", 0) + 1
            }
            
            self._save_cache()
    
    def should_skip_method(self, url: str, method: str) -> bool:
        """
        Determine if a method should be skipped based on previous failures.
        
        Args:
            url: URL to scrape
            method: Method name to check
            
        Returns:
            True if method should be skipped, False otherwise
        """
        domain = self.get_domain(url)
        
        with self.lock:
            if domain not in self.method_cache:
                return False
            
            # If we have a successful method, skip others
            successful_method = self.method_cache[domain].get("successful_method")
            if successful_method and successful_method != method:
                return True
            
            # Check if this method has failed too many times recently
            failed_methods = self.method_cache[domain].get("failed_methods", {})
            if method in failed_methods:
                failure_info = failed_methods[method]
                failure_count = failure_info.get("failure_count", 0)
                last_failure = failure_info.get("last_failure", 0)
                
                # Skip if failed more than 3 times in the last hour
                if failure_count >= 3 and (time.time() - last_failure) < 3600:
                    return True
            
            return False

class ContentCleaner:
    """Cleans and structures scraped content for LLM consumption."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common noise patterns
        noise_patterns = [
            r'cookie\s+policy',
            r'privacy\s+policy',
            r'terms\s+of\s+service',
            r'subscribe\s+to\s+newsletter',
            r'follow\s+us\s+on',
            r'share\s+this',
            r'loading\.\.\.',
            r'please\s+wait',
            r'javascript\s+is\s+required',
            r'enable\s+javascript',
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove email patterns (optional)
        # text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone patterns (optional)
        # text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Clean up punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_main_content(html: str, url: str) -> Tuple[str, Optional[str]]:
        """
        Extract main content from HTML using multiple methods.
        
        Args:
            html: HTML content
            url: Source URL
            
        Returns:
            Tuple of (content, title)
        """
        # Method 1: Try trafilatura (best for article extraction)
        try:
            config = use_config()
            config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
            content = trafilatura.extract(html, config=config)
            if content and len(content.strip()) > 200:
                title = trafilatura.extract_metadata(html).get('title')
                return content, title
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed: {e}")
        
        # Method 2: Try readability-lxml
        try:
            doc = Document(html)
            content = doc.summary()
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    return text, doc.title()
        except Exception as e:
            logger.debug(f"Readability extraction failed: {e}")
        
        # Method 3: Manual extraction with BeautifulSoup
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()
            
            # Try to find main content areas
            main_selectors = [
                'main', 'article', '[role="main"]', '.main-content', '.content',
                '#content', '.post-content', '.entry-content', '.article-content'
            ]
            
            main_content = None
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                # Fallback to body
                main_content = soup.body or soup
            
            # Extract title
            title = None
            title_elem = soup.find('title') or soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Extract text
            text = main_content.get_text(separator=' ', strip=True)
            return text, title
            
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {e}")
        
        return "", None
    
    @staticmethod
    def structure_for_llm(content: str, title: Optional[str] = None, url: str = "") -> str:
        """
        Structure content for optimal LLM consumption.
        
        Args:
            content: Clean content
            title: Page title
            url: Source URL
            
        Returns:
            Structured content for LLM
        """
        if not content:
            return f"Error: No content could be extracted from {url}"
        
        # Clean the content
        content = ContentCleaner.clean_text(content)
        
        # Structure the output
        structured_content = []
        
        if title:
            structured_content.append(f"# {title}")
            structured_content.append("")
        
        # Add source URL
        structured_content.append(f"**Source:** {url}")
        structured_content.append("")
        
        # Add content
        structured_content.append("## Content")
        structured_content.append("")
        structured_content.append(content)
        
        return "\n".join(structured_content)
    
    @staticmethod
    def extract_metadata_from_html(html: str) -> dict:
        """
        Extract metadata from HTML, including meta tags, OpenGraph, Twitter cards, canonical, etc.
        Args:
            html: HTML content
        Returns:
            Dictionary of metadata fields
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {}
        # Standard meta tags
        for meta in soup.find_all('meta'):
            if meta.get('name'):
                name = meta['name'].lower()
                metadata[name] = meta.get('content', '')
            if meta.get('property'):
                prop = meta['property'].lower()
                metadata[prop] = meta.get('content', '')
        # Canonical link
        canonical = soup.find('link', rel='canonical')
        if canonical and canonical.get('href'):
            metadata['canonical'] = canonical['href']
        # Title (if not already set)
        if 'title' not in metadata:
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text(strip=True)
        # Twitter card tags
        for tag in ['twitter:card', 'twitter:title', 'twitter:description', 'twitter:image']:
            if tag not in metadata:
                meta_tag = soup.find('meta', attrs={'name': tag})
                if meta_tag and meta_tag.get('content'):
                    metadata[tag] = meta_tag['content']
        # OpenGraph tags
        for tag in ['og:title', 'og:description', 'og:image', 'og:type', 'og:url']:
            if tag not in metadata:
                meta_tag = soup.find('meta', attrs={'property': tag})
                if meta_tag and meta_tag.get('content'):
                    metadata[tag] = meta_tag['content']
        return metadata

class AdvancedScraper:
    """Advanced web scraper with anti-detection capabilities."""
    
    def __init__(self, 
                 use_proxies: bool = False,
                 proxy_list: Optional[List[str]] = None,
                 max_retries: int = 2,  # Reduced from 3 to 2
                 timeout: int = 15,  # Reduced from 30 to 15
                 delay_between_requests: float = 0.5,  # Reduced from 1.0 to 0.5
                 use_cookies: bool = True,
                 cookie_file: str = "scraping_cookies.json",
                 use_method_cache: bool = True,
                 method_cache_file: str = "scraping_method_cache.json"):
        """
        Initialize the advanced scraper with optimized timeouts for faster failure.
        
        Args:
            use_proxies: Whether to use proxy rotation
            proxy_list: List of proxy URLs
            max_retries: Maximum number of retry attempts (reduced for speed)
            timeout: Base request timeout in seconds (reduced for faster failure)
            delay_between_requests: Delay between requests in seconds (reduced for speed)
            use_cookies: Whether to use cookie management
            cookie_file: File to store cookies
            use_method_cache: Whether to use method caching for domain optimization
            method_cache_file: File to store method cache
        """
        self.user_agent_rotator = UserAgentRotator()
        self.proxy_manager = ProxyManager(proxy_list) if use_proxies else None
        self.cookie_manager = CookieManager(cookie_file) if use_cookies else None
        self.method_cache_manager = MethodCacheManager(method_cache_file) if use_method_cache else None
        self.max_retries = max_retries
        self.timeout = timeout
        self.delay_between_requests = delay_between_requests
        
        # Method-specific timeouts - DRAMATICALLY REDUCED for faster failure
        self.requests_timeout = min(timeout, 8)  # Reduced from 15 to 8s for requests
        self.selenium_timeout = min(timeout * 1.2, 12)  # Reduced from 25 to 12s for Selenium
        self.undetected_timeout = min(timeout * 1.5, 18)  # Reduced from 35 to 18s for undetected
        
        # Circuit breaker: domains that consistently fail
        self.failed_domains = set()
        self.domain_failure_count = {}
        self.max_domain_failures = 3  # After 3 failures, skip domain entirely
        
        # Initialize session with persistent cookies
        self.session = requests.Session()
        if self.cookie_manager:
            # Load existing cookies into session
            for domain, cookies in self.cookie_manager.cookies.items():
                for cookie in cookies:
                    self.session.cookies.set(
                        cookie.get('name', ''),
                        cookie.get('value', ''),
                        domain=cookie.get('domain', domain)
                    )
    
    def configure_proxy_rotation(self, proxy_list: List[str]):
        """Configure proxy rotation with a list of proxy URLs."""
        if not self.proxy_manager:
            self.proxy_manager = ProxyManager(proxy_list)
        else:
            for proxy in proxy_list:
                self.proxy_manager.add_proxy(proxy)
    
    def add_common_cookies(self):
        """Add common cookies that might help bypass detection."""
        if not self.cookie_manager:
            return
            
        # Common cookies for financial/investment sites
        common_cookies = {
            "crunchbase.com": [
                {"name": "cb_analytics", "value": "1", "domain": ".crunchbase.com"},
                {"name": "cb_user_consent", "value": "1", "domain": ".crunchbase.com"},
            ],
            "pitchbook.com": [
                {"name": "pb_analytics", "value": "1", "domain": ".pitchbook.com"},
                {"name": "pb_user_consent", "value": "1", "domain": ".pitchbook.com"},
            ]
        }
        
        for domain, cookies in common_cookies.items():
            for cookie in cookies:
                self.cookie_manager.add_cookie(domain, **cookie)

    def _get_request_headers(self) -> Dict[str, str]:
        """Get headers for HTTP requests."""
        headers = {
            'User-Agent': self.user_agent_rotator.get_random_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        return headers
    
    def _get_chrome_options(self, headless: bool = True) -> Options:
        """Get Chrome options optimized for scraping with enhanced anti-detection."""
        options = Options()
        
        if headless:
            options.add_argument("--headless")
        
        # Enhanced anti-detection options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        # Additional anti-detection measures
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-ipc-flooding-protection")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-extension")
        options.add_argument("--disable-component-extensions-with-background-pages")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-translate")
        options.add_argument("--hide-scrollbars")
        options.add_argument("--mute-audio")
        options.add_argument("--no-first-run")
        options.add_argument("--safebrowsing-disable-auto-update")
        options.add_argument("--disable-client-side-phishing-detection")
        options.add_argument("--disable-hang-monitor")
        options.add_argument("--disable-prompt-on-repost")
        options.add_argument("--disable-domain-reliability")
        options.add_argument("--disable-component-update")
        options.add_argument("--disable-features=AudioServiceOutOfProcess")
        
        # Performance options
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")
        options.add_argument("--disable-javascript")
        
        # Window and viewport - randomize to avoid fingerprinting
        import random
        width = random.randint(1200, 1920)
        height = random.randint(800, 1080)
        options.add_argument(f"--window-size={width},{height}")
        options.add_argument("--start-maximized")
        
        # Random user agent
        options.add_argument(f"--user-agent={self.user_agent_rotator.get_random_agent()}")
        
        # Additional experimental options for better stealth
        options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0,
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.media_stream": 2,
        })
        
        return options
    
    def _simulate_human_behavior(self, driver):
        """Simulate human-like behavior to avoid detection (optimized for speed)."""
        import random
        import time
        
        # Reduced random mouse movements (1-3 instead of 2-5)
        for _ in range(random.randint(1, 3)):
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            try:
                from selenium.webdriver.common.action_chains import ActionChains
                actions = ActionChains(driver)
                actions.move_by_offset(x, y).perform()
                time.sleep(random.uniform(0.05, 0.2))  # Reduced from 0.1-0.3
            except:
                pass
        
        # Quick scrolling
        scroll_amount = random.randint(100, 300)  # Reduced from 100-500
        driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        time.sleep(random.uniform(0.2, 0.8))  # Reduced from 0.5-1.5
        
        # Reduced wait time
        time.sleep(random.uniform(0.5, 1.5))  # Reduced from 1-3

    def _requests_scrape(self, url: str) -> ScrapingResult:
        """Attempt to scrape using requests library."""
        start_time = time.time()
        
        try:
            headers = self._get_request_headers()
            proxies = self.proxy_manager.get_next_proxy() if self.proxy_manager else None
            
            response = self.session.get(
                url, 
                headers=headers, 
                proxies=proxies,
                timeout=self.requests_timeout,
                allow_redirects=True
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                html_content = response.text
                content, title = ContentCleaner.extract_main_content(html_content, url)
                metadata = ContentCleaner.extract_metadata_from_html(html_content)
                if content and len(content.strip()) > 100:
                    structured_content = ContentCleaner.structure_for_llm(content, title, url)
                    return ScrapingResult(
                        url=url,
                        content=structured_content,
                        title=title,
                        metadata=metadata,
                        success=True,
                        method_used="requests",
                        response_time=response_time
                    )
            
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=f"HTTP {response.status_code}",
                method_used="requests",
                response_time=response_time
            )
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=str(e),
                method_used="requests",
                response_time=response_time
            )
    
    def _selenium_scrape(self, url: str, use_undetected: bool = False) -> ScrapingResult:
        """Attempt to scrape using Selenium with enhanced anti-detection and better error handling."""
        start_time = time.time()
        driver = None
        
        # Use appropriate timeout based on method
        timeout = self.undetected_timeout if use_undetected else self.selenium_timeout
        
        try:
            if use_undetected:
                # Use undetected-chromedriver for better anti-detection
                driver = uc.Chrome(
                    options=self._get_chrome_options(headless=True),
                    version_main=None,  # Auto-detect version
                    driver_executable_path=None,  # Let it auto-detect
                    browser_executable_path=None,  # Let it auto-detect
                    suppress_welcome=True,  # Suppress welcome page
                    use_subprocess=True  # Use subprocess for better isolation
                )
            else:
                # Use regular Selenium with improved options
                options = self._get_chrome_options(headless=True)
                # Add additional options for better stability
                options.add_argument("--disable-gpu-sandbox")
                options.add_argument("--disable-software-rasterizer")
                options.add_argument("--disable-background-networking")
                options.add_argument("--disable-default-apps")
                options.add_argument("--disable-extensions")
                options.add_argument("--disable-sync")
                options.add_argument("--disable-translate")
                options.add_argument("--hide-scrollbars")
                options.add_argument("--metrics-recording-only")
                options.add_argument("--mute-audio")
                options.add_argument("--no-first-run")
                options.add_argument("--safebrowsing-disable-auto-update")
                options.add_argument("--ignore-certificate-errors")
                options.add_argument("--ignore-ssl-errors")
                options.add_argument("--ignore-certificate-errors-spki-list")
                options.add_argument("--allow-running-insecure-content")
                options.add_argument("--disable-web-security")
                options.add_argument("--allow-cross-origin-auth-prompt")
                
                driver = webdriver.Chrome(options=options)
            
            # Set page load timeout
            driver.set_page_load_timeout(timeout)
            driver.implicitly_wait(5)  # Add implicit wait
            
            # Add cookies if available for this domain
            if self.cookie_manager:
                domain = urlparse(url).netloc
                cookies = self.cookie_manager.get_cookies_for_domain(domain)
                if cookies:
                    # Navigate to domain first to set cookies
                    try:
                        driver.get(f"https://{domain}")
                        for cookie in cookies:
                            try:
                                driver.add_cookie(cookie)
                            except Exception as cookie_error:
                                logger.debug(f"Failed to add cookie: {cookie_error}")
                    except Exception as domain_error:
                        logger.debug(f"Failed to navigate to domain for cookies: {domain_error}")
            
            # Navigate to URL
            driver.get(url)
            
            # Simulate human behavior
            self._simulate_human_behavior(driver)
            
            # Wait for page to load with shorter timeout
            try:
                WebDriverWait(driver, min(5, timeout // 3)).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.debug(f"Timeout waiting for body element on {url}")
            
            # Additional wait for dynamic content (further reduced)
            time.sleep(random.uniform(0.2, 0.5))
            
            # Get page source
            html_content = driver.page_source
            title = driver.title
            
            response_time = time.time() - start_time
            
            if html_content and len(html_content.strip()) > 100:
                content, extracted_title = ContentCleaner.extract_main_content(html_content, url)
                metadata = ContentCleaner.extract_metadata_from_html(html_content)
                if content and len(content.strip()) > 100:
                    structured_content = ContentCleaner.structure_for_llm(content, extracted_title or title, url)
                    return ScrapingResult(
                        url=url,
                        content=structured_content,
                        title=extracted_title or title,
                        metadata=metadata,
                        success=True,
                        method_used="undetected_selenium" if use_undetected else "selenium",
                        response_time=response_time
                    )
            
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message="No content extracted",
                method_used="undetected_selenium" if use_undetected else "selenium",
                response_time=response_time
            )
            
        except TimeoutException:
            response_time = time.time() - start_time
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=f"Timeout after {timeout}s",
                method_used="undetected_selenium" if use_undetected else "selenium",
                response_time=response_time
            )
        except WebDriverException as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            # Handle common WebDriver connection issues
            if "connection refused" in error_msg.lower() or "no connection" in error_msg.lower():
                error_msg = "WebDriver connection failed - browser may not be available"
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=error_msg,
                method_used="undetected_selenium" if use_undetected else "selenium",
                response_time=response_time
            )
        except Exception as e:
            response_time = time.time() - start_time
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=str(e),
                method_used="undetected_selenium" if use_undetected else "selenium",
                response_time=response_time
            )
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def _is_domain_blacklisted(self, url: str) -> bool:
        """
        Check if a domain should be skipped due to consistent failures.
        
        Args:
            url: URL to check
            
        Returns:
            True if domain should be skipped
        """
        domain = urlparse(url).netloc
        return domain in self.failed_domains
    
    def _record_domain_failure(self, url: str):
        """
        Record a failure for a domain and blacklist if too many failures.
        
        Args:
            url: URL that failed
        """
        domain = urlparse(url).netloc
        self.domain_failure_count[domain] = self.domain_failure_count.get(domain, 0) + 1
        
        if self.domain_failure_count[domain] >= self.max_domain_failures:
            self.failed_domains.add(domain)
            logger.warning(f"Domain {domain} blacklisted due to {self.domain_failure_count[domain]} consecutive failures")

    def _should_skip_url(self, url: str) -> bool:
        """
        Pre-filter URLs to skip known problematic domains and patterns.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL should be skipped
        """
        domain = urlparse(url).netloc.lower()
        
        # Known problematic domains that consistently fail or block scrapers
        problematic_domains = {
            'claim.midle.io',  # From the logs - consistently failed
            'linkedin.com',    # Heavy anti-bot protection
            'facebook.com',    # Heavy anti-bot protection
            'twitter.com',     # Heavy anti-bot protection
            'instagram.com',   # Heavy anti-bot protection
            'tiktok.com',      # Heavy anti-bot protection
            'snapchat.com',    # Heavy anti-bot protection
            'reddit.com',      # Rate limiting and anti-bot
            'youtube.com',     # Heavy anti-bot protection
            'pinterest.com',   # Anti-bot protection
            'medium.com',      # Rate limiting
            'substack.com',    # Rate limiting
            'newsletter.com',  # Rate limiting
            'mailchimp.com',   # Rate limiting
            'sendgrid.com',    # Rate limiting
            'hubspot.com',     # Rate limiting
            'salesforce.com',  # Rate limiting
            'zendesk.com',     # Rate limiting
            'intercom.com',    # Rate limiting
            'drift.com',       # Rate limiting
            'calendly.com',    # Rate limiting
            'typeform.com',    # Rate limiting
            'survey-monkey.com', # Rate limiting
            'qualtrics.com',   # Rate limiting
            'google.com',      # Heavy anti-bot protection
            'googleusercontent.com', # Heavy anti-bot protection
            'amazon.com',      # Heavy anti-bot protection
            'apple.com',       # Heavy anti-bot protection
            'microsoft.com',   # Heavy anti-bot protection
            'github.com',      # Rate limiting
            'gitlab.com',      # Rate limiting
            'bitbucket.org',   # Rate limiting
            'stackoverflow.com', # Rate limiting
            'quora.com',       # Rate limiting
            'wikipedia.org',   # Usually works but can be slow
            'wikimedia.org',   # Usually works but can be slow
        }
        
        # Check for problematic domains
        if domain in problematic_domains:
            logger.info(f"  - Skipping known problematic domain: {domain}")
            return True
        
        # Check for subdomains of problematic domains
        for problematic_domain in problematic_domains:
            if domain.endswith('.' + problematic_domain):
                logger.info(f"  - Skipping subdomain of problematic domain: {domain}")
                return True
        
        # Check for URL patterns that are likely to fail
        problematic_patterns = [
            '/login',
            '/signin',
            '/auth',
            '/register',
            '/signup',
            '/checkout',
            '/cart',
            '/payment',
            '/billing',
            '/account',
            '/profile',
            '/settings',
            '/admin',
            '/dashboard',
            '/api/',
            '/graphql',
            '/rest/',
            '/oauth',
            '/callback',
            '/webhook',
            '/hook',
            '/bot',
            '/crawler',
            '/scraper',
            '/spider',
            '/robot',
            '/automation',
        ]
        
        url_lower = url.lower()
        for pattern in problematic_patterns:
            if pattern in url_lower:
                logger.info(f"  - Skipping URL with problematic pattern: {pattern}")
                return True
        
        return False

    def scrape(self, url: str) -> ScrapingResult:
        """
        Scrape a URL using intelligent method selection with circuit breakers.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapingResult with content and metadata
        """
        logger.info(f"Scraping: {url}")
        
        # Pre-filter URL
        if self._should_skip_url(url):
            return ScrapingResult(
                url=url,
                content=f"Error: URL {url} is blacklisted or has a problematic pattern.",
                success=False,
                error_message="URL blacklisted or problematic",
                method_used="pre_filtered"
            )
        
        # Check if domain is blacklisted
        if self._is_domain_blacklisted(url):
            logger.info(f"  - Skipping blacklisted domain: {urlparse(url).netloc}")
            return ScrapingResult(
                url=url,
                content=f"Error: Domain {urlparse(url).netloc} is blacklisted due to consistent failures",
                success=False,
                error_message="Domain blacklisted",
                method_used="blacklisted"
            )
        
        # Get optimal method order for this domain
        if self.method_cache_manager:
            method_order = self.method_cache_manager.get_optimal_method_order(url)
        else:
            method_order = ["requests", "selenium", "undetected"]
        
        # Try methods in optimal order with faster failure
        for method in method_order:
            # Check if we should skip this method based on cache
            if self.method_cache_manager and self.method_cache_manager.should_skip_method(url, method):
                logger.info(f"  - Skipping {method} method (cached as ineffective)")
                continue
            
            # Map method names to actual scraping functions
            if method == "requests":
                logger.info(f"  - Attempting requests method (timeout: {self.requests_timeout}s)")
                result = self._requests_scrape(url)
                method_used = "requests"
            elif method == "selenium":
                logger.info(f"  - Attempting Selenium method (timeout: {self.selenium_timeout}s)")
                result = self._selenium_scrape(url, use_undetected=False)
                method_used = "selenium"
            elif method == "undetected":
                logger.info(f"  - Attempting undetected-chromedriver method (timeout: {self.undetected_timeout}s)")
                result = self._selenium_scrape(url, use_undetected=True)
                method_used = "undetected_selenium"
            else:
                continue
            
            # Record success/failure in cache and domain tracking
            if result.success:
                if self.method_cache_manager:
                    self.method_cache_manager.record_success(url, method_used)
                logger.info(f"Successfully scraped {url} using {method_used} in {result.response_time:.2f}s")
                return result
            else:
                if self.method_cache_manager:
                    self.method_cache_manager.record_failure(url, method_used)
                logger.info(f"Failed to scrape {url} using {method_used}: {result.error_message}")
            
            # Add minimal delay between methods (reduced)
            if method != method_order[-1]:  # Don't delay after last method
                time.sleep(self.delay_between_requests * 0.3)  # Reduced from 0.5 to 0.3
        
        # All methods failed - record domain failure
        self._record_domain_failure(url)
        logger.warning(f"All scraping methods failed for {url}")
        return ScrapingResult(
            url=url,
            content=f"Error: Failed to scrape {url}. All methods attempted.",
            success=False,
            error_message="All methods failed",
            method_used="all_failed"
        )
    
    def scrape_batch(self, urls: List[str], max_workers: int = 8) -> List[ScrapingResult]:
        """
        Scrape multiple URLs in parallel with improved efficiency and failure handling.
        
        Args:
            urls: List of URLs to scrape
            max_workers: Maximum number of parallel workers (reduced default to prevent overwhelming Ollama)
            
        Returns:
            List of ScrapingResult objects
        """
        if not urls:
            return []
        
        # Pre-filter URLs to remove problematic ones
        filtered_urls = []
        skipped_urls = []
        for url in urls:
            if self._should_skip_url(url):
                skipped_urls.append(url)
            else:
                filtered_urls.append(url)
        
        if skipped_urls:
            logger.info(f"Pre-filtered {len(skipped_urls)} problematic URLs: {skipped_urls}")
        
        if not filtered_urls:
            logger.warning("No URLs remaining after pre-filtering")
            return [ScrapingResult(
                url=url,
                content=f"Error: URL {url} was pre-filtered as problematic",
                success=False,
                error_message="Pre-filtered",
                method_used="pre_filtered"
            ) for url in urls]
        
        # Optimize worker count based on URL count - reduced to prevent overwhelming Ollama
        optimal_workers = min(max_workers, len(filtered_urls), 8)
        
        logger.info(f"Scraping {len(filtered_urls)} URLs with {optimal_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all scraping tasks
            future_to_url = {executor.submit(self.scrape, url): url for url in filtered_urls}
            
            # Collect results as they complete with better error handling
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log performance metrics
                    if result.success:
                        logger.info(f"✅ Scraped {url} in {result.response_time:.2f}s using {result.method_used}")
                    else:
                        logger.warning(f"❌ Failed {url}: {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    results.append(ScrapingResult(
                        url=url,
                        content=f"Error: {str(e)}",
                        success=False,
                        error_message=str(e),
                        method_used="exception"
                    ))
        
        # Add pre-filtered results
        for url in skipped_urls:
            results.append(ScrapingResult(
                url=url,
                content=f"Error: URL {url} was pre-filtered as problematic",
                success=False,
                error_message="Pre-filtered",
                method_used="pre_filtered"
            ))
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        logger.info(f"Batch scraping complete: {successful} successful, {failed} failed out of {len(urls)} total URLs")
        
        return results

# Convenience function for easy use
def scrape_url(url: str, **kwargs) -> str:
    """
    Convenience function to scrape a single URL and return content.
    
    Args:
        url: URL to scrape
        **kwargs: Additional arguments for AdvancedScraper
        
    Returns:
        Scraped content as string
    """
    scraper = AdvancedScraper(**kwargs)
    result = scraper.scrape(url)
    return result.content if result.success else f"Error: {result.error_message}"

def scrape_urls(urls: List[str], **kwargs) -> List[str]:
    """
    Convenience function to scrape multiple URLs and return content list.
    
    Args:
        urls: List of URLs to scrape
        **kwargs: Additional arguments for AdvancedScraper
        
    Returns:
        List of scraped content strings
    """
    scraper = AdvancedScraper(**kwargs)
    results = scraper.scrape_batch(urls)
    return [result.content if result.success else f"Error: {result.error_message}" 
            for result in results] 