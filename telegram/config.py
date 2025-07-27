"""
Configuration settings for the Telegram bot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Bot configuration settings"""
    
    # Telegram Bot Token
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8040479024:AAF60yF-pQYpvlOcnxHebi_QyQ-wY4MyBYg")
    
    # Download settings
    DOWNLOAD_PATH = os.getenv("DOWNLOAD_PATH", "./downloads")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50"))  # MB
    
    # Supported platforms
    SUPPORTED_PLATFORMS = [
        "youtube.com",
        "youtu.be",
        "instagram.com",
        "tiktok.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "fb.com",
        "reddit.com",
        "pinterest.com",
        "snapchat.com"
    ]
    
    # yt-dlp options
    YTDLP_OPTIONS = {
        'format': 'best[filesize<50M]/best',
        'outtmpl': '%(title)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    # Bot messages
    MESSAGES = {
        'welcome': "🤖 Welcome! Send me any social media link and I'll download the content for you.",
        'processing': "⏳ Processing your link...",
        'success': "✅ Download completed!",
        'error': "❌ Sorry, I couldn't download that content.",
        'unsupported': "❌ This platform is not supported yet.",
        'file_too_large': "❌ File is too large to send via Telegram.",
        'invalid_link': "❌ Please send a valid social media link."
    } 