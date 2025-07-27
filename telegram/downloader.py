"""
Media downloader service for social media platforms
"""
import os
import re
import yt_dlp
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaDownloader:
    """Handles downloading media from various social media platforms"""
    
    def __init__(self):
        """Initialize the downloader with configuration"""
        self.config = Config()
        self.download_path = Path(self.config.DOWNLOAD_PATH)
        self.download_path.mkdir(exist_ok=True)
        
    def is_supported_platform(self, url: str) -> bool:
        """
        Check if the URL is from a supported platform
        
        Args:
            url: The URL to check
            
        Returns:
            bool: True if platform is supported
        """
        return any(platform in url.lower() for platform in self.config.SUPPORTED_PLATFORMS)
    
    def extract_platform(self, url: str) -> str:
        """
        Extract platform name from URL
        
        Args:
            url: The URL to extract platform from
            
        Returns:
            str: Platform name
        """
        for platform in self.config.SUPPORTED_PLATFORMS:
            if platform in url.lower():
                return platform.split('.')[0]
        return "unknown"
    
    def get_download_options(self, platform: str) -> Dict[str, Any]:
        """
        Get platform-specific download options
        
        Args:
            platform: Platform name
            
        Returns:
            Dict: Download options for yt-dlp
        """
        options = self.config.YTDLP_OPTIONS.copy()
        
        # Platform-specific options
        if platform == "instagram":
            options.update({
                'format': 'best',
                'extract_flat': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'cookiesfrombrowser': ('chrome',),  # Try to use Chrome cookies
                'extractor_args': {
                    'instagram': {
                        'login': None,  # Will try to use browser cookies
                    }
                }
            })
        elif platform == "tiktok":
            options.update({
                'format': 'best[height<=720]',
                'extract_flat': False,
            })
        elif platform == "youtube":
            options.update({
                'format': 'best[filesize<50M]/best[height<=720]',
                'extract_flat': False,
            })
        elif platform == "twitter":
            options.update({
                'format': 'best',
                'extract_flat': False,
            })
        
        # Set output template with platform prefix
        options['outtmpl'] = f'{platform}_%(title)s.%(ext)s'
        
        return options
    
    async def download_media(self, url: str, instagram_creds: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Download media from the given URL
        
        Args:
            url: The URL to download from
            instagram_creds: Optional Instagram credentials (username, password)
            
        Returns:
            Optional[str]: Path to downloaded file or None if failed
        """
        try:
            # Validate URL
            if not self.is_supported_platform(url):
                logger.warning(f"Unsupported platform for URL: {url}")
                return None
            
            # Extract platform
            platform = self.extract_platform(url)
            logger.info(f"Downloading from {platform}: {url}")
            
            # Get platform-specific options
            options = self.get_download_options(platform)
            options['outtmpl'] = str(self.download_path / options['outtmpl'])
            
            # For Instagram, try multiple approaches
            if platform == "instagram":
                return await self._download_instagram(url, options, instagram_creds)
            
            # Download using yt-dlp for other platforms
            with yt_dlp.YoutubeDL(options) as ydl:
                # Get info first
                info = ydl.extract_info(url, download=False)
                
                # Check file size
                if 'filesize' in info and info['filesize']:
                    file_size_mb = info['filesize'] / (1024 * 1024)
                    if file_size_mb > self.config.MAX_FILE_SIZE:
                        logger.warning(f"File too large: {file_size_mb:.2f}MB")
                        return None
                
                # Download the file
                ydl.download([url])
                
                # Find the downloaded file
                downloaded_file = self._find_downloaded_file(info.get('title', ''), platform)
                
                if downloaded_file and downloaded_file.exists():
                    logger.info(f"Successfully downloaded: {downloaded_file}")
                    return str(downloaded_file)
                else:
                    logger.error("Download completed but file not found")
                    return None
                    
        except Exception as e:
            logger.error(f"Download failed for {url}: {str(e)}")
            return None
    
    async def _download_instagram(self, url: str, options: Dict[str, Any], instagram_creds: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Special handling for Instagram downloads with multiple fallback methods
        
        Args:
            url: Instagram URL
            options: Download options
            instagram_creds: Optional Instagram credentials (username, password)
            
        Returns:
            Optional[str]: Path to downloaded file or None if failed
        """
        # Method 0: Try with user credentials (if provided)
        if instagram_creds:
            try:
                logger.info("Trying Instagram download with user credentials...")
                creds_options = options.copy()
                creds_options.update({
                    'username': instagram_creds['username'],
                    'password': instagram_creds['password'],
                    'cookiesfrombrowser': None,  # Don't use browser cookies when using credentials
                    'extractor_args': {
                        'instagram': {
                            'login': True,
                        }
                    }
                })
                
                with yt_dlp.YoutubeDL(creds_options) as ydl:
                    info = ydl.extract_info(url, download=False)
                    ydl.download([url])
                    downloaded_file = self._find_downloaded_file(info.get('title', ''), 'instagram')
                    if downloaded_file and downloaded_file.exists():
                        return str(downloaded_file)
            except Exception as e:
                logger.warning(f"Instagram download with credentials failed: {str(e)}")
        
        # Method 1: Try with browser cookies
        try:
            logger.info("Trying Instagram download with browser cookies...")
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                ydl.download([url])
                downloaded_file = self._find_downloaded_file(info.get('title', ''), 'instagram')
                if downloaded_file and downloaded_file.exists():
                    return str(downloaded_file)
        except Exception as e:
            logger.warning(f"Instagram download with cookies failed: {str(e)}")
        
        # Method 2: Try with different user agent and no cookies
        try:
            logger.info("Trying Instagram download with different user agent...")
            options_copy = options.copy()
            options_copy.update({
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'cookiesfrombrowser': None,
                'extractor_args': {
                    'instagram': {
                        'login': None,
                    }
                }
            })
            
            with yt_dlp.YoutubeDL(options_copy) as ydl:
                info = ydl.extract_info(url, download=False)
                ydl.download([url])
                downloaded_file = self._find_downloaded_file(info.get('title', ''), 'instagram')
                if downloaded_file and downloaded_file.exists():
                    return str(downloaded_file)
        except Exception as e:
            logger.warning(f"Instagram download with different user agent failed: {str(e)}")
        
        # Method 3: Try with minimal options
        try:
            logger.info("Trying Instagram download with minimal options...")
            minimal_options = {
                'format': 'best',
                'outtmpl': options['outtmpl'],
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(minimal_options) as ydl:
                info = ydl.extract_info(url, download=False)
                ydl.download([url])
                downloaded_file = self._find_downloaded_file(info.get('title', ''), 'instagram')
                if downloaded_file and downloaded_file.exists():
                    return str(downloaded_file)
        except Exception as e:
            logger.warning(f"Instagram download with minimal options failed: {str(e)}")
        
        logger.error("All Instagram download methods failed")
        return None
    
    def _find_downloaded_file(self, title: str, platform: str) -> Optional[Path]:
        """
        Find the downloaded file in the download directory
        
        Args:
            title: Video title
            platform: Platform name
            
        Returns:
            Optional[Path]: Path to downloaded file
        """
        try:
            # Clean title for filename matching
            clean_title = re.sub(r'[^\w\s-]', '', title).strip()
            clean_title = re.sub(r'[-\s]+', '-', clean_title)
            
            # Look for files with platform prefix
            for file_path in self.download_path.glob(f"{platform}_*"):
                if file_path.is_file():
                    return file_path
            
            # Fallback: return the most recent file
            files = list(self.download_path.glob("*"))
            if files:
                return max(files, key=lambda x: x.stat().st_mtime)
                
        except Exception as e:
            logger.error(f"Error finding downloaded file: {str(e)}")
        
        return None
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Clean up old downloaded files
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        try:
            for file_path in self.download_path.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}") 