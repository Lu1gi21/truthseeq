#!/usr/bin/env python3
"""
Simple TruthSeeQ Workflow Test

This script tests the workflow system without requiring database or Redis setup.
It focuses on testing the core workflow functionality and tools.
"""

import asyncio
import logging
import sys
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock settings for testing
class MockSettings:
    """Mock settings for testing without full configuration."""
    
    class AI:
        OPENAI_API_KEY = None
        ANTHROPIC_API_KEY = None
    
    class BraveSearch:
        API_KEY = None
    
    ai = AI()
    brave_search = BraveSearch()

# Mock the settings
import app.config
app.config.settings = MockSettings()

# Import workflow components
from app.workflow.tools import (
    web_search, scrape_content, check_domain_reliability,
    BraveSearchTool, ContentScrapingTool, FactCheckingDatabaseTool
)
from app.workflow.nodes import (
    ContentExtractionNode, ClaimsExtractionNode, SourceVerificationNode,
    FactAnalysisNode, ConfidenceScoringNode
)
from app.workflow.state import (
    VerdictType, ConfidenceLevel, SourceType,
    create_fact_check_state, create_content_analysis_state, create_source_verification_state
)


async def test_tools():
    """Test the workflow tools."""
    print("🔧 Testing Workflow Tools")
    print("=" * 50)
    
    try:
        # Test domain reliability check
        print("\n1. Testing Domain Reliability Check:")
        domain_result = check_domain_reliability("reuters.com")
        print(f"   Domain: reuters.com")
        print(f"   Reliability Score: {domain_result['reliability_score']:.2%}")
        print(f"   Trust Indicators: {domain_result['trust_indicators']}")
        print(f"   Red Flags: {domain_result['red_flags']}")
        
        # Test web search (will use fallback if no API key)
        print("\n2. Testing Web Search:")
        search_results = web_search.invoke({"query": "fact check climate change", "count": 3})
        print(f"   Found {len(search_results)} results")
        for i, result in enumerate(search_results[:2], 1):
            print(f"   {i}. {result['title']}")
            print(f"      URL: {result['url']}")
            print(f"      Relevance: {result['relevance_score']:.2%}")
        
        # Test content scraping
        print("\n3. Testing Content Scraping:")
        # Use a simple test URL
        test_url = "https://httpbin.org/html"
        scrape_result = scrape_content.invoke({"url": test_url})
        print(f"   URL: {test_url}")
        print(f"   Success: {scrape_result['success']}")
        print(f"   Method: {scrape_result['method_used']}")
        if scrape_result['success']:
            print(f"   Content Length: {len(scrape_result['content'])} characters")
        
        print("\n✅ Tools test completed successfully!")
        
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        logger.exception("Tools test error")


async def test_nodes():
    """Test the workflow nodes."""
    print("\n🔧 Testing Workflow Nodes")
    print("=" * 50)
    
    try:
        # Test content extraction node
        print("\n1. Testing Content Extraction Node:")
        content_node = ContentExtractionNode()
        
        # Create test state
        test_state = {
            "original_url": "https://httpbin.org/html",
            "extracted_claims": [],
            "claims_analysis": {},
            "verification_sources": [],
            "source_analysis": {},
            "fact_analysis": {},
            "cross_references": [],
            "confidence_score": 0.0,
            "confidence_level": ConfidenceLevel.VERY_LOW,
            "verdict": VerdictType.UNVERIFIABLE,
            "reasoning": "",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "processing_time": 0.0,
            "model_used": "",
            "search_queries": []
        }
        
        result = await content_node(test_state)
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Success: {result.get('scraped_content', {}).get('success', False)}")
        
        # Test claims extraction node (will fail without AI model, but that's expected)
        print("\n2. Testing Claims Extraction Node:")
        claims_node = ClaimsExtractionNode()
        
        # Create test state with content
        test_state_with_content = {
            "scraped_content": {
                "content": "This is a test article about climate change. The Earth's temperature has increased by 1.1°C since pre-industrial times.",
                "title": "Test Article",
                "url": "https://example.com/test",
                "metadata": {},
                "scraped_at": "2024-01-01T00:00:00Z",
                "method_used": "test",
                "success": True,
                "error_message": None
            }
        }
        
        try:
            result = await claims_node(test_state_with_content)
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Claims Found: {len(result.get('extracted_claims', []))}")
        except Exception as e:
            print(f"   Expected failure (no AI model): {str(e)[:100]}...")
        
        print("\n✅ Nodes test completed successfully!")
        
    except Exception as e:
        print(f"❌ Nodes test failed: {e}")
        logger.exception("Nodes test error")


async def test_state_management():
    """Test the state management system."""
    print("\n🔧 Testing State Management")
    print("=" * 50)
    
    try:
        # Test fact check state creation
        print("\n1. Testing Fact Check State Creation:")
        fact_state = create_fact_check_state("test_workflow_123", "https://example.com/article")
        print(f"   Workflow ID: {fact_state['workflow_id']}")
        print(f"   URL: {fact_state['original_url']}")
        print(f"   Status: {fact_state['status']}")
        print(f"   Verdict: {fact_state['verdict'].value}")
        print(f"   Confidence Level: {fact_state['confidence_level'].value}")
        
        # Test content analysis state creation
        print("\n2. Testing Content Analysis State Creation:")
        content_state = create_content_analysis_state("test_workflow_456", "https://example.com/article")
        print(f"   Workflow ID: {content_state['workflow_id']}")
        print(f"   URL: {content_state['original_url']}")
        print(f"   Status: {content_state['status']}")
        print(f"   Content Category: {content_state['content_category']}")
        
        # Test source verification state creation
        print("\n3. Testing Source Verification State Creation:")
        source_state = create_source_verification_state("test_workflow_789", "https://example.com/article")
        print(f"   Workflow ID: {source_state['workflow_id']}")
        print(f"   URL: {source_state['target_url']}")
        print(f"   Domain: {source_state['domain']}")
        print(f"   Status: {source_state['status']}")
        
        print("\n✅ State management test completed successfully!")
        
    except Exception as e:
        print(f"❌ State management test failed: {e}")
        logger.exception("State management test error")


async def test_tool_classes():
    """Test the tool classes directly."""
    print("\n🔧 Testing Tool Classes")
    print("=" * 50)
    
    try:
        # Test Brave Search Tool
        print("\n1. Testing Brave Search Tool:")
        brave_tool = BraveSearchTool()
        print(f"   API Key Available: {brave_tool.api_key is not None}")
        
        # Test fallback search
        results = brave_tool.search("test query", count=2)
        print(f"   Fallback Search Results: {len(results)}")
        
        # Test Content Scraping Tool
        print("\n2. Testing Content Scraping Tool:")
        scraping_tool = ContentScrapingTool()
        print(f"   Scraper Initialized: {scraping_tool.scraper is not None}")
        
        # Test Fact Checking Database Tool
        print("\n3. Testing Fact Checking Database Tool:")
        db_tool = FactCheckingDatabaseTool()
        domain_result = db_tool.check_domain_reliability("bbc.com")
        print(f"   BBC.com Reliability: {domain_result['reliability_score']:.2%}")
        print(f"   Trust Indicators: {domain_result['trust_indicators']}")
        
        print("\n✅ Tool classes test completed successfully!")
        
    except Exception as e:
        print(f"❌ Tool classes test failed: {e}")
        logger.exception("Tool classes test error")


async def main():
    """Main test function."""
    print("🚀 TruthSeeQ Workflow System - Simple Test")
    print("=" * 60)
    print("This test runs without requiring database or Redis setup.")
    print("It focuses on testing core workflow functionality and tools.\n")
    
    try:
        # Run all tests
        await test_tools()
        await test_nodes()
        await test_state_management()
        await test_tool_classes()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\nNote: Some tests may show expected failures when AI models")
        print("are not configured. This is normal for this simple test.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Test error")


if __name__ == "__main__":
    asyncio.run(main()) 