"""
Telegram bot for downloading social media content
"""
import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

from config import Config
from downloader import MediaDownloader

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SocialMediaBot:
    """Telegram bot for downloading social media content"""
    
    def __init__(self):
        """Initialize the bot with configuration and downloader"""
        self.config = Config()
        self.downloader = MediaDownloader()
        self.app = None
        self.instagram_credentials = None
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            self.config.MESSAGES['welcome'],
            parse_mode=ParseMode.HTML
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
 🤖 <b>Social Media Downloader Bot</b>

 <b>Supported Platforms:</b>
 • YouTube (youtube.com, youtu.be)
 • Instagram (instagram.com) - Enhanced with login support
 • TikTok (tiktok.com)
 • Twitter/X (twitter.com, x.com)
 • Facebook (facebook.com, fb.com)
 • Reddit (reddit.com)
 • Pinterest (pinterest.com)
 • Snapchat (snapchat.com)

 <b>How to use:</b>
 Simply send me any social media link and I'll download the content for you!

 <b>Commands:</b>
 /start - Start the bot
 /help - Show this help message
 /cleanup - Clean up old downloaded files
 /login username password - Login with Instagram credentials

 <b>Instagram Login:</b>
 If Instagram downloads fail, you can try with your credentials:
 <code>/login your_username your_password</code>
 Your credentials are only used temporarily and won't be stored.

 <b>Note:</b>
 • Maximum file size: 50MB
 • Files are automatically cleaned up after 24 hours
 • Instagram credentials are cleared after each download for security
        """
        await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)
    
    async def cleanup_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /cleanup command"""
        try:
            self.downloader.cleanup_old_files()
            await update.message.reply_text("🧹 Cleanup completed!")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            await update.message.reply_text("❌ Cleanup failed. Please try again later.")
    
    async def login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /login command for Instagram credentials"""
        try:
            # Check if we have arguments
            if not context.args or len(context.args) < 2:
                await update.message.reply_text(
                    "❌ Please provide your Instagram username and password.\n"
                    "Format: <code>/login username password</code>",
                    parse_mode=ParseMode.HTML
                )
                return
            
            username = context.args[0]
            password = context.args[1]
            
            # Store credentials temporarily for this user
            user_id = update.effective_user.id
            self.instagram_credentials = {
                'username': username,
                'password': password,
                'user_id': user_id
            }
            
            await update.message.reply_text(
                "✅ Instagram credentials saved temporarily.\n"
                "Now send the Instagram post link you want to download.",
                parse_mode=ParseMode.HTML
            )
            
        except Exception as e:
            logger.error(f"Login command error: {str(e)}")
            await update.message.reply_text("❌ Failed to save credentials. Please try again.")
    
    def extract_urls(self, text: str) -> list:
        """
        Extract URLs from text message
        
        Args:
            text: Text message to extract URLs from
            
        Returns:
            list: List of extracted URLs
        """
        # URL regex pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages"""
        message = update.message
        text = message.text
        
        # Extract URLs from message
        urls = self.extract_urls(text)
        
        if not urls:
            await message.reply_text(self.config.MESSAGES['invalid_link'])
            return
        
        # Process each URL
        for url in urls:
            await self.process_url(message, url)
    
    async def process_url(self, message, url: str):
        """Process a single URL for downloading"""
        try:
            # Check if platform is supported
            if not self.downloader.is_supported_platform(url):
                await message.reply_text(self.config.MESSAGES['unsupported'])
                return
            
            # Send processing message
            processing_msg = await message.reply_text(self.config.MESSAGES['processing'])
            
            # Check if we have Instagram credentials for this user
            user_id = message.from_user.id
            instagram_creds = None
            if (self.instagram_credentials and 
                self.instagram_credentials['user_id'] == user_id and 
                "instagram" in url.lower()):
                instagram_creds = {
                    'username': self.instagram_credentials['username'],
                    'password': self.instagram_credentials['password']
                }
            
            # Download the media
            file_path = await self.downloader.download_media(url, instagram_creds)
            
            if file_path and Path(file_path).exists():
                # Send the downloaded file
                await self.send_file(message, file_path)
                await processing_msg.delete()
                
                # Clear credentials after successful download
                if instagram_creds:
                    self.instagram_credentials = None
                    await message.reply_text("🔒 Instagram credentials cleared for security.")
            else:
                await processing_msg.edit_text(self.config.MESSAGES['error'])
                
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            
            # Provide specific error message for Instagram
            if "instagram" in url.lower():
                await message.reply_text(
                    "❌ Instagram download failed. Instagram has strict anti-scraping measures.\n\n"
                    "🔐 <b>Try with your Instagram credentials:</b>\n"
                    "Send your Instagram username and password in this format:\n"
                    "<code>/login username password</code>\n\n"
                    "⚠️ <b>Note:</b> Your credentials will only be used for this download and won't be stored.\n\n"
                    "Or you can try:\n"
                    "• Using a different Instagram post\n"
                    "• Using a different social media platform",
                    parse_mode=ParseMode.HTML
                )
            else:
                await message.reply_text(self.config.MESSAGES['error'])
    
    async def send_file(self, message, file_path: str):
        """Send downloaded file to user"""
        try:
            file_path = Path(file_path)
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.MAX_FILE_SIZE:
                await message.reply_text(self.config.MESSAGES['file_too_large'])
                return
            
            # Determine file type and send accordingly
            if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                await message.reply_video(
                    video=open(file_path, 'rb'),
                    caption=f"📹 Downloaded from {self.downloader.extract_platform(str(file_path))}"
                )
            elif file_path.suffix.lower() in ['.mp3', '.wav', '.m4a']:
                await message.reply_audio(
                    audio=open(file_path, 'rb'),
                    caption=f"🎵 Downloaded from {self.downloader.extract_platform(str(file_path))}"
                )
            elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                await message.reply_photo(
                    photo=open(file_path, 'rb'),
                    caption=f"🖼️ Downloaded from {self.downloader.extract_platform(str(file_path))}"
                )
            else:
                await message.reply_document(
                    document=open(file_path, 'rb'),
                    caption=f"📄 Downloaded from {self.downloader.extract_platform(str(file_path))}"
                )
            
            await message.reply_text(self.config.MESSAGES['success'])
            
        except Exception as e:
            logger.error(f"Error sending file {file_path}: {str(e)}")
            await message.reply_text(self.config.MESSAGES['error'])
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Exception while handling an update: {context.error}")
    
    def run(self):
        """Start the bot"""
        # Create application
        self.app = Application.builder().token(self.config.BOT_TOKEN).build()
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("cleanup", self.cleanup_command))
        self.app.add_handler(CommandHandler("login", self.login_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Add error handler
        self.app.add_error_handler(self.error_handler)
        
        # Start the bot
        logger.info("Starting bot...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function to run the bot"""
    bot = SocialMediaBot()
    bot.run()

if __name__ == "__main__":
    main() 