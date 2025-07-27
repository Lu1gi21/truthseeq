#!/usr/bin/env python3
"""
Test script for Brave Search API configuration.

This script helps debug Brave Search API key issues by:
1. Checking all possible API key sources
2. Testing the API connection
3. Providing detailed error messages
"""

import os
import sys
import logging
from typing import Optional

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.config import settings
from app.services.scraper_service import BraveSearchClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_api_key_sources() -> Optional[str]:
    """
    Check all possible API key sources and return the first found.
    
    Returns:
        API key if found, None otherwise
    """
    sources = [
        ("settings.brave_search.BRAVE_API_KEY", settings.brave_search.BRAVE_API_KEY),
        ("settings.brave_search.API_KEY", settings.brave_search.API_KEY),
        ("settings.brave_search.BRAVE_SEARCH_API_KEY", settings.brave_search.BRAVE_SEARCH_API_KEY),
        ("os.getenv('BRAVE_API_KEY')", os.getenv("BRAVE_API_KEY")),
        ("os.getenv('BRAVE_SEARCH_API_KEY')", os.getenv("BRAVE_SEARCH_API_KEY")),
        ("os.getenv('API_KEY')", os.getenv("API_KEY"))
    ]
    
    print("🔍 Checking API key sources:")
    for source_name, value in sources:
        if value:
            print(f"  ✅ {source_name}: {'*' * min(len(value), 8)}...")
            return value
        else:
            print(f"  ❌ {source_name}: Not found")
    
    return None


async def test_brave_api():
    """
    Test Brave Search API connection.
    """
    print("\n🚀 Testing Brave Search API...")
    
    # Check API key
    api_key = check_api_key_sources()
    
    if not api_key:
        print("\n❌ No API key found!")
        print("\n📋 To fix this issue:")
        print("1. Register at https://api.search.brave.com/register")
        print("2. Get your API key from the dashboard")
        print("3. Set one of these environment variables:")
        print("   - BRAVE_API_KEY=your_api_key_here")
        print("   - BRAVE_SEARCH_API_KEY=your_api_key_here")
        print("   - API_KEY=your_api_key_here")
        print("\n💡 Note: Even free plans require a credit card for identity verification")
        return False
    
    print(f"\n✅ API key found: {'*' * min(len(api_key), 8)}...")
    
    # Test API connection
    try:
        client = BraveSearchClient(api_key)
        
        print("\n🔍 Testing search functionality...")
        result = await client.search_web("test query", count=1)
        
        if result.total_results > 0:
            print("✅ API connection successful!")
            print(f"   Found {result.total_results} results")
            print(f"   Search time: {result.search_time:.2f}s")
            return True
        else:
            print("⚠️  API connected but no results returned")
            print("   This might be normal for a test query")
            return True
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    finally:
        await client.close()


def main():
    """
    Main test function.
    """
    print("🧪 Brave Search API Configuration Test")
    print("=" * 50)
    
    # Check environment
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug mode: {settings.DEBUG}")
    
    # Test API
    import asyncio
    success = asyncio.run(test_brave_api())
    
    if success:
        print("\n🎉 All tests passed! Your Brave Search API is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main() 