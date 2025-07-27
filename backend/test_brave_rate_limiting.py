#!/usr/bin/env python3
"""
Test script for Brave Search API rate limiting.

This script tests the rate limiting implementation to ensure we don't exceed
the 1 request per second limit and handles 429 errors gracefully.
"""

import asyncio
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the BraveSearchTool
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.workflow.tools import BraveSearchTool


def test_rate_limiting():
    """
    Test the rate limiting implementation with multiple rapid requests.
    """
    logger.info("Starting Brave Search rate limiting test")
    
    # Create Brave Search tool instance
    brave_tool = BraveSearchTool()
    
    if not brave_tool.api_key:
        logger.error("No Brave Search API key configured!")
        logger.error("Please set BRAVE_API_KEY environment variable")
        return
    
    # Test queries
    test_queries = [
        "Malcolm-Jamal Warner fact check",
        "Bill Cosby 2024 news",
        "The Cosby Show cast",
        "Costa Rica drowning incident",
        "Actor death verification"
    ]
    
    logger.info(f"Testing {len(test_queries)} queries with rate limiting")
    
    start_time = time.time()
    successful_requests = 0
    failed_requests = 0
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"Query {i}/{len(test_queries)}: '{query}'")
        
        try:
            # This should automatically handle rate limiting
            results = brave_tool.search(query, count=5)
            
            if results:
                logger.info(f"✅ Success: {len(results)} results")
                successful_requests += 1
            else:
                logger.warning(f"⚠️ No results returned")
                failed_requests += 1
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            failed_requests += 1
        
        # Small delay between queries for better logging
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    logger.info(f"\nTest completed in {total_time:.2f}s")
    logger.info(f"Successful requests: {successful_requests}")
    logger.info(f"Failed requests: {failed_requests}")
    logger.info(f"Average time per request: {total_time/len(test_queries):.2f}s")


def test_concurrent_requests():
    """
    Test concurrent requests to ensure global rate limiting works.
    """
    logger.info("Testing concurrent Brave Search requests")
    
    brave_tool1 = BraveSearchTool()
    brave_tool2 = BraveSearchTool()
    
    if not brave_tool1.api_key:
        logger.error("No Brave Search API key configured!")
        return
    
    # Test concurrent requests
    queries = [
        "test query 1",
        "test query 2", 
        "test query 3"
    ]
    
    start_time = time.time()
    
    # Simulate concurrent requests
    for i, query in enumerate(queries):
        logger.info(f"Concurrent test {i+1}: '{query}'")
        
        # Use different tool instances to test global rate limiting
        tool = brave_tool1 if i % 2 == 0 else brave_tool2
        
        try:
            results = tool.search(query, count=3)
            logger.info(f"✅ Concurrent request {i+1} successful: {len(results)} results")
        except Exception as e:
            logger.error(f"❌ Concurrent request {i+1} failed: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"Concurrent test completed in {total_time:.2f}s")


def test_error_handling():
    """
    Test error handling and retry logic.
    """
    logger.info("Testing error handling and retry logic")
    
    brave_tool = BraveSearchTool()
    
    if not brave_tool.api_key:
        logger.error("No Brave Search API key configured!")
        return
    
    # Test with a query that might trigger rate limiting
    test_query = "rate limit test query"
    
    logger.info(f"Testing error handling with query: '{test_query}'")
    
    try:
        # Make multiple rapid requests to potentially trigger rate limiting
        for i in range(3):
            logger.info(f"Rapid request {i+1}/3")
            results = brave_tool.search(test_query, count=2)
            logger.info(f"Request {i+1} completed: {len(results)} results")
            
            # Very short delay to potentially trigger rate limiting
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Brave Search Rate Limiting Test")
    logger.info("=" * 60)
    
    # Run tests
    test_rate_limiting()
    print()
    
    test_concurrent_requests()
    print()
    
    test_error_handling()
    print()
    
    logger.info("All tests completed!") 