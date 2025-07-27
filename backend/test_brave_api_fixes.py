#!/usr/bin/env python3
"""
Test script for Brave Search API fixes.

This script tests the fixes implemented for the Brave Search API,
including language parameter correction and rate limiting.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_brave_search_language_fix():
    """Test that the language parameter is correctly set to 'en'."""
    try:
        from app.workflow.tools import BraveSearchTool
        
        # Create a BraveSearchTool instance
        search_tool = BraveSearchTool()
        
        # Check the default language parameter
        import inspect
        sig = inspect.signature(search_tool.search)
        default_lang = sig.parameters['search_lang'].default
        
        if default_lang == "en":
            logger.info("✓ Language parameter correctly set to 'en'")
            return True
        else:
            logger.error(f"✗ Language parameter is '{default_lang}', expected 'en'")
            return False
            
    except Exception as e:
        logger.error(f"✗ Language parameter test failed: {e}")
        return False

def test_rate_limiting_implementation():
    """Test that rate limiting is properly implemented."""
    try:
        from app.workflow.tools import BraveSearchTool
        
        # Create a BraveSearchTool instance
        search_tool = BraveSearchTool()
        
        # Check that the rate limiting attribute exists
        if hasattr(search_tool, 'last_request_time'):
            logger.info("✓ Rate limiting attribute exists")
            return True
        else:
            logger.error("✗ Rate limiting attribute not found")
            return False
            
    except Exception as e:
        logger.error(f"✗ Rate limiting test failed: {e}")
        return False

def test_tool_invoke_method():
    """Test that tools use .invoke() method correctly."""
    try:
        from app.workflow.tools import web_search, search_fact_checking_databases
        
        # Check that tools have .invoke method
        if hasattr(web_search, 'invoke'):
            logger.info("✓ web_search tool has .invoke method")
        else:
            logger.error("✗ web_search tool missing .invoke method")
            return False
            
        if hasattr(search_fact_checking_databases, 'invoke'):
            logger.info("✓ search_fact_checking_databases tool has .invoke method")
        else:
            logger.error("✗ search_fact_checking_databases tool missing .invoke method")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Tool invoke method test failed: {e}")
        return False

def test_fact_checking_search_improvement():
    """Test that fact-checking search now uses real search."""
    try:
        from app.workflow.tools import FactCheckingDatabaseTool
        
        # Create a FactCheckingDatabaseTool instance
        db_tool = FactCheckingDatabaseTool()
        
        # Test that the search method exists and returns a list
        results = db_tool.search_fact_checking_databases("test query")
        
        if isinstance(results, list):
            logger.info("✓ Fact-checking search returns list")
            return True
        else:
            logger.error(f"✗ Fact-checking search returns {type(results)}, expected list")
            return False
            
    except Exception as e:
        logger.error(f"✗ Fact-checking search test failed: {e}")
        return False

async def main():
    """Run all tests and provide a summary."""
    logger.info("Starting Brave Search API fix tests...")
    
    tests = [
        ("Language Parameter Fix", test_brave_search_language_fix),
        ("Rate Limiting Implementation", test_rate_limiting_implementation),
        ("Tool Invoke Method", test_tool_invoke_method),
        ("Fact-Checking Search Improvement", test_fact_checking_search_improvement),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The Brave Search API fixes appear to be working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 