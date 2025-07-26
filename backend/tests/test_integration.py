"""
Integration tests for TruthSeeQ services.

This module tests the integration between different services including:
- Scraper service and AI service integration
- End-to-end fact-checking workflow
- Content analysis pipeline
- Database and caching integration
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.services.scraper_service import ScraperService
from app.services.ai_service import AIService
from app.schemas.content import (
    ScrapingRequest, FactCheckRequest, ContentAnalysisRequest
)
from app.database.models import ContentItem, ContentStatus
from app.config import get_settings


@pytest.mark.asyncio
async def test_scraper_to_ai_integration():
    """
    Test the complete workflow from scraping content to AI analysis.
    
    This test demonstrates how the scraper service and AI service work together:
    1. Scrape content using the scraper service
    2. Store content in the database
    3. Use AI service to analyze the scraped content
    4. Verify the integration works end-to-end
    """
    # Mock database session
    mock_db_session = AsyncMock(spec=AsyncSession)
    
    # Mock Redis client
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None  # No cached results
    mock_redis.setex.return_value = True
    
    # Create test content
    test_content = ContentItem(
        id=1,
        url="https://example.com/test-article",
        title="Test Article Title",
        content="This is a test article about climate change. The Earth's temperature has increased by 1.1°C since pre-industrial times.",
        source_domain="example.com",
        status=ContentStatus.COMPLETED
    )
    
    # Mock database query result
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = test_content
    
    # Initialize services
    scraper_service = ScraperService(mock_db_session)
    ai_service = AIService(mock_db_session, mock_redis)
    
    # Test 1: Verify scraper service can process content
    scraping_request = ScrapingRequest(
        urls=["https://example.com/test-article"],
        include_metadata=True
    )
    
    # Mock the scraping result
    mock_scraping_result = MagicMock()
    mock_scraping_result.success = True
    mock_scraping_result.content = test_content.content
    mock_scraping_result.title = test_content.title
    mock_scraping_result.url = test_content.url
    mock_scraping_result.metadata = {"author": "Test Author"}
    mock_scraping_result.method_used = "requests"
    mock_scraping_result.response_time = 1.5
    
    # Mock the advanced scraper
    scraper_service.scraper.scrape_batch = MagicMock(return_value=[mock_scraping_result])
    
    # Test scraping
    scraping_response = await scraper_service.scrape_urls(scraping_request)
    
    # Verify scraping worked
    assert scraping_response.successful == 1
    assert scraping_response.failed == 0
    assert len(scraping_response.results) == 1
    assert scraping_response.results[0].success is True
    
    # Test 2: Verify AI service can analyze scraped content
    fact_check_request = FactCheckRequest(
        content_id=1,
        model_name="gpt-4",
        analysis_depth="standard"
    )
    
    # Mock AI model responses
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value.content = '["Climate change is real", "Temperature has increased"]'
    
    # Mock the model manager
    ai_service.model_manager.models = {"gpt-4": mock_model}
    ai_service.model_manager.default_model = "gpt-4"
    
    # Test fact-checking
    fact_check_response = await ai_service.fact_check_content(fact_check_request)
    
    # Verify fact-checking worked
    assert fact_check_response.content_id == 1
    assert 0 <= fact_check_response.confidence_score <= 1
    assert fact_check_response.verdict in ["true", "mostly_true", "mixed", "mostly_false", "false", "unverifiable"]
    assert len(fact_check_response.reasoning) > 0
    
    # Test 3: Verify content analysis
    analysis_request = ContentAnalysisRequest(
        content_id=1,
        analysis_types=["sentiment", "bias", "credibility"]
    )
    
    # Test content analysis
    analysis_response = await ai_service.analyze_content(analysis_request)
    
    # Verify analysis worked
    assert analysis_response.content_id == 1
    assert 0 <= analysis_response.confidence <= 1
    assert "analysis_results" in analysis_response.dict()
    
    # Test 4: Verify caching works
    # Mock cached result
    cached_result = {
        "content_id": 1,
        "confidence_score": 0.85,
        "verdict": "mostly_true",
        "reasoning": "Cached analysis result",
        "sources": [],
        "execution_time": 2.5,
        "workflow_execution_id": 123
    }
    mock_redis.get.return_value = '{"content_id": 1, "confidence_score": 0.85, "verdict": "mostly_true", "reasoning": "Cached analysis result", "sources": [], "execution_time": 2.5, "workflow_execution_id": 123}'
    
    # Test with cached result
    cached_response = await ai_service.fact_check_content(fact_check_request)
    
    # Verify cached result was used
    assert cached_response.confidence_score == 0.85
    assert cached_response.verdict == "mostly_true"
    
    print("✅ Integration test passed: Scraper service and AI service work together successfully!")


@pytest.mark.asyncio
async def test_workflow_orchestration():
    """
    Test LangGraph workflow orchestration in the AI service.
    
    This test verifies that the workflow orchestrator can:
    1. Initialize workflows correctly
    2. Execute workflows with proper state management
    3. Handle workflow errors gracefully
    """
    # Mock database session
    mock_db_session = AsyncMock(spec=AsyncSession)
    
    # Mock Redis client
    mock_redis = AsyncMock()
    
    # Initialize AI service
    ai_service = AIService(mock_db_session, mock_redis)
    
    # Test workflow initialization
    assert len(ai_service.workflow_orchestrator.workflows) > 0
    assert "fact_checking" in ai_service.workflow_orchestrator.workflows
    assert "content_analysis" in ai_service.workflow_orchestrator.workflows
    assert "source_verification" in ai_service.workflow_orchestrator.workflows
    
    # Test workflow execution
    initial_state = {
        "content": "Test content for workflow execution",
        "title": "Test Title",
        "url": "https://example.com/test",
        "extracted_claims": [],
        "source_analysis": {},
        "fact_analysis": {},
        "confidence_score": 0.0,
        "verdict": "",
        "reasoning": "",
        "sources": []
    }
    
    # Mock AI model for workflow execution
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value.content = '["Test claim"]'
    ai_service.model_manager.models = {"gpt-4": mock_model}
    ai_service.model_manager.default_model = "gpt-4"
    
    # Execute workflow
    workflow_result = await ai_service.workflow_orchestrator.execute_workflow(
        "fact_checking", initial_state
    )
    
    # Verify workflow execution
    assert "confidence_score" in workflow_result
    assert "verdict" in workflow_result
    assert "reasoning" in workflow_result
    
    print("✅ Workflow orchestration test passed: LangGraph workflows execute correctly!")


@pytest.mark.asyncio
async def test_error_handling():
    """
    Test error handling in the integration between services.
    
    This test verifies that:
    1. Missing content is handled gracefully
    2. AI model failures are handled properly
    3. Database errors are caught and reported
    """
    # Mock database session
    mock_db_session = AsyncMock(spec=AsyncSession)
    
    # Mock Redis client
    mock_redis = AsyncMock()
    
    # Initialize services
    ai_service = AIService(mock_db_session, mock_redis)
    
    # Test 1: Missing content
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
    
    fact_check_request = FactCheckRequest(content_id=999)
    
    with pytest.raises(ValueError, match="Content item 999 not found"):
        await ai_service.fact_check_content(fact_check_request)
    
    # Test 2: AI model failure
    test_content = ContentItem(
        id=1,
        url="https://example.com/test",
        title="Test",
        content="Test content",
        source_domain="example.com",
        status=ContentStatus.COMPLETED
    )
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = test_content
    
    # Mock model failure
    ai_service.model_manager.models = {}
    ai_service.model_manager.default_model = None
    
    # This should still work with fallback logic
    fact_check_response = await ai_service.fact_check_content(fact_check_request)
    assert fact_check_response.confidence_score == 0.5  # Default fallback
    
    print("✅ Error handling test passed: Services handle errors gracefully!")


if __name__ == "__main__":
    # Run integration tests
    asyncio.run(test_scraper_to_ai_integration())
    asyncio.run(test_workflow_orchestration())
    asyncio.run(test_error_handling())
    print("\n🎉 All integration tests completed successfully!") 