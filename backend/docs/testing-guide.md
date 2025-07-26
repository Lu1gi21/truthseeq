# Testing Implementation Guide

This guide covers comprehensive testing strategies for the TruthSeeQ backend using pytest, coverage analysis, and testing best practices.

## 📋 Installation & Setup

```bash
pip install pytest==8.3.4
pip install pytest-asyncio==0.25.0
pip install pytest-cov==6.0.0
pip install pytest-mock==3.14.0
pip install factory-boy==3.3.1
pip install faker==33.1.0
pip install httpx==0.28.1  # For testing FastAPI endpoints
```

## 🏗️ Testing Structure

### Test Directory Organization

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── test_api/
│   ├── __init__.py
│   ├── test_content.py        # Content API endpoint tests
│   ├── test_feed.py           # Feed API endpoint tests
│   └── test_rate_limit.py     # Rate limiting tests
├── test_services/
│   ├── __init__.py
│   ├── test_ai_service.py     # AI service tests
│   ├── test_fact_checker.py   # Fact-checking service tests
│   └── test_scraper_service.py # Web scraping service tests
├── test_langgraph/
│   ├── __init__.py
│   ├── test_fact_checking.py  # LangGraph workflow tests
│   └── test_nodes.py          # Individual node tests
├── test_database/
│   ├── __init__.py
│   ├── test_models.py         # Database model tests
│   └── test_repositories.py   # Repository tests
├── test_core/
│   ├── __init__.py
│   ├── test_config.py         # Configuration tests
│   └── test_rate_limiting.py  # Rate limiting logic tests
└── fixtures/
    ├── __init__.py
    ├── sample_content.py       # Sample content for testing
    └── mock_responses.py       # Mock API responses
```

## ⚙️ Configuration & Fixtures

### Main Configuration (`tests/conftest.py`)

```python
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from unittest.mock import AsyncMock, MagicMock

from app.main import app
from app.database.database import get_db, Base
from app.core.config import settings
from app.services.ai_service import AIService
from app.services.fact_checker import FactCheckerService

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://test_user:test_pass@localhost/truthseeq_test"

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop all tables after tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def db_session(test_engine) -> AsyncSession:
    """Create a fresh database session for each test."""
    async_session = sessionmaker(
        test_engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        # Start a transaction
        transaction = await session.begin()
        
        yield session
        
        # Rollback the transaction to keep tests isolated
        await transaction.rollback()

@pytest.fixture
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with database dependency override."""
    
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    # Clean up
    app.dependency_overrides.clear()

@pytest.fixture
def mock_ai_service() -> AIService:
    """Create a mock AI service for testing."""
    mock_service = MagicMock(spec=AIService)
    
    # Mock the verify_content method
    async def mock_verify_content(content, content_type, session_id):
        return {
            "verdict": "inconclusive",
            "confidence_score": 0.5,
            "reasoning": "Mock analysis result",
            "sources_checked": ["https://example.com"],
            "claims": ["Mock claim"],
            "extracted_content": content,
            "ai_model_used": "mock-model",
            "processing_time": 1.0,
            "processing_steps": ["mock_step"],
            "errors": []
        }
    
    mock_service.verify_content = AsyncMock(side_effect=mock_verify_content)
    return mock_service

@pytest.fixture
def sample_content():
    """Sample content for testing."""
    return {
        "text_content": "The Earth is round and orbits the sun. This is a scientific fact.",
        "url_content": "https://example.com/test-article",
        "false_claim": "Vaccines cause autism and contain microchips.",
        "satire_content": "Local Man Discovers Internet After 20 Years, Immediately Regrets It"
    }

@pytest.fixture
def sample_session():
    """Sample session data for testing."""
    return {
        "id": "test_session_123",
        "ip_address": "127.0.0.1",
        "user_agent": "TestAgent/1.0",
        "created_at": 1635724800.0
    }
```

## 🧪 API Testing

### Content API Tests (`tests/test_api/test_content.py`)

```python
import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock

class TestContentAPI:
    """Test content verification API endpoints."""
    
    @pytest.mark.asyncio
    async def test_verify_text_content(self, client: AsyncClient):
        """Test text content verification."""
        request_data = {
            "content_type": "text",
            "content": "The Earth is round.",
            "priority": 3
        }
        
        with patch("app.services.fact_checker.FactCheckerService") as mock_service:
            # Mock the fact checker response
            mock_service.return_value.verify_content = AsyncMock(return_value={
                "verification_id": "test-id-123",
                "session_id": "test_session",
                "content_type": "text",
                "original_content": "The Earth is round.",
                "extracted_content": "The Earth is round.",
                "result": {
                    "verdict": "true",
                    "confidence_score": 0.95,
                    "reasoning": "Scientific consensus confirms this.",
                    "sources_checked": ["https://nasa.gov"],
                    "ai_model_used": "gpt-4-turbo-preview"
                },
                "processing_time_seconds": 2.5,
                "created_at": "2024-01-01T12:00:00Z"
            })
            
            response = await client.post("/api/v1/content/verify", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["verification_id"] == "test-id-123"
            assert data["result"]["verdict"] == "true"
            assert data["result"]["confidence_score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_verify_url_content(self, client: AsyncClient):
        """Test URL content verification."""
        request_data = {
            "content_type": "url",
            "content": "https://example.com/article",
            "priority": 2
        }
        
        with patch("app.services.fact_checker.FactCheckerService") as mock_service:
            mock_service.return_value.verify_content = AsyncMock(return_value={
                "verification_id": "test-url-123",
                "session_id": "test_session",
                "content_type": "url",
                "original_content": "https://example.com/article",
                "extracted_content": "Article content extracted from URL...",
                "result": {
                    "verdict": "inconclusive",
                    "confidence_score": 0.3,
                    "reasoning": "Insufficient reliable sources found.",
                    "sources_checked": [],
                    "ai_model_used": "gpt-4-turbo-preview"
                },
                "processing_time_seconds": 8.1,
                "created_at": "2024-01-01T12:00:00Z"
            })
            
            response = await client.post("/api/v1/content/verify", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["content_type"] == "url"
            assert data["result"]["verdict"] == "inconclusive"
    
    @pytest.mark.asyncio
    async def test_invalid_content_type(self, client: AsyncClient):
        """Test validation error for invalid content type."""
        request_data = {
            "content_type": "invalid",
            "content": "Some content",
            "priority": 3
        }
        
        response = await client.post("/api/v1/content/verify", json=request_data)
        
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data
    
    @pytest.mark.asyncio
    async def test_content_too_short(self, client: AsyncClient):
        """Test validation error for content that's too short."""
        request_data = {
            "content_type": "text",
            "content": "Hi",  # Too short
            "priority": 3
        }
        
        response = await client.post("/api/v1/content/verify", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_batch_verification(self, client: AsyncClient):
        """Test batch content verification."""
        request_data = {
            "items": [
                {"content_type": "text", "content": "Claim 1", "priority": 3},
                {"content_type": "text", "content": "Claim 2", "priority": 2},
            ]
        }
        
        with patch("app.services.fact_checker.FactCheckerService") as mock_service:
            mock_service.return_value.verify_content = AsyncMock(
                side_effect=[
                    {"verification_id": "batch-1", "result": {"verdict": "true"}},
                    {"verification_id": "batch-2", "result": {"verdict": "false"}},
                ]
            )
            
            response = await client.post("/api/v1/content/batch-verify", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["verification_id"] == "batch-1"
            assert data[1]["verification_id"] == "batch-2"
    
    @pytest.mark.asyncio
    async def test_batch_size_limit(self, client: AsyncClient):
        """Test batch size limit enforcement."""
        request_data = {
            "items": [
                {"content_type": "text", "content": f"Claim {i}", "priority": 3}
                for i in range(11)  # 11 items, exceeds limit of 10
            ]
        }
        
        response = await client.post("/api/v1/content/batch-verify", json=request_data)
        
        assert response.status_code == 400
        error_data = response.json()
        assert "Batch size cannot exceed 10 items" in error_data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_verification_result(self, client: AsyncClient):
        """Test getting verification result by ID."""
        verification_id = "test-verification-123"
        
        with patch("app.services.fact_checker.FactCheckerService") as mock_service:
            mock_service.return_value.get_verification_result = AsyncMock(return_value={
                "verification_id": verification_id,
                "result": {"verdict": "true", "confidence_score": 0.9}
            })
            
            response = await client.get(f"/api/v1/content/{verification_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["verification_id"] == verification_id
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_verification(self, client: AsyncClient):
        """Test 404 for non-existent verification."""
        verification_id = "nonexistent-id"
        
        with patch("app.services.fact_checker.FactCheckerService") as mock_service:
            mock_service.return_value.get_verification_result = AsyncMock(return_value=None)
            
            response = await client.get(f"/api/v1/content/{verification_id}")
            
            assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_verification_history(self, client: AsyncClient):
        """Test getting verification history."""
        with patch("app.services.fact_checker.FactCheckerService") as mock_service:
            mock_service.return_value.get_session_history = AsyncMock(return_value=[
                {"id": "item-1", "content": "Test 1", "created_at": "2024-01-01T12:00:00Z"},
                {"id": "item-2", "content": "Test 2", "created_at": "2024-01-01T11:00:00Z"},
            ])
            
            response = await client.get("/api/v1/content/history?limit=10&offset=0")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["id"] == "item-1"
```

## 🤖 AI Service Testing

### AI Service Tests (`tests/test_services/test_ai_service.py`)

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.ai_service import AIService

class TestAIService:
    """Test AI service functionality."""
    
    @pytest.fixture
    def ai_service(self):
        """Create AI service instance for testing."""
        return AIService()
    
    @pytest.mark.asyncio
    async def test_verify_text_content(self, ai_service):
        """Test text content verification."""
        with patch("app.services.ai_service.create_fact_checking_workflow") as mock_workflow:
            # Mock the workflow execution
            mock_final_state = MagicMock()
            mock_final_state.verdict = "true"
            mock_final_state.confidence_score = 0.85
            mock_final_state.reasoning = "Scientific consensus supports this claim."
            mock_final_state.sources_checked = ["https://nasa.gov"]
            mock_final_state.claims = ["The Earth is round"]
            mock_final_state.extracted_content = "The Earth is round."
            mock_final_state.ai_model_used = "gpt-4-turbo-preview"
            mock_final_state.processing_steps = ["content_extraction", "fact_analysis"]
            mock_final_state.errors = []
            
            mock_workflow.return_value.ainvoke = AsyncMock(return_value=mock_final_state)
            
            result = await ai_service.verify_content(
                content="The Earth is round.",
                content_type="text",
                session_id="test_session"
            )
            
            assert result["verdict"] == "true"
            assert result["confidence_score"] == 0.85
            assert "Scientific consensus" in result["reasoning"]
            assert len(result["sources_checked"]) > 0
    
    @pytest.mark.asyncio
    async def test_verify_url_content(self, ai_service):
        """Test URL content verification."""
        with patch("app.services.ai_service.create_fact_checking_workflow") as mock_workflow:
            mock_final_state = MagicMock()
            mock_final_state.verdict = "false"
            mock_final_state.confidence_score = 0.9
            mock_final_state.reasoning = "Multiple reliable sources contradict this claim."
            mock_final_state.sources_checked = ["https://factcheck.org", "https://snopes.com"]
            mock_final_state.claims = ["False claim about vaccines"]
            mock_final_state.extracted_content = "Extracted article content..."
            mock_final_state.ai_model_used = "gpt-4-turbo-preview"
            mock_final_state.processing_steps = ["content_extraction", "claim_identification", "fact_analysis"]
            mock_final_state.errors = []
            
            mock_workflow.return_value.ainvoke = AsyncMock(return_value=mock_final_state)
            
            result = await ai_service.verify_content(
                content="https://example.com/false-claim-article",
                content_type="url",
                session_id="test_session"
            )
            
            assert result["verdict"] == "false"
            assert result["confidence_score"] == 0.9
            assert len(result["sources_checked"]) == 2
    
    @pytest.mark.asyncio
    async def test_ai_service_error_handling(self, ai_service):
        """Test error handling in AI service."""
        with patch("app.services.ai_service.create_fact_checking_workflow") as mock_workflow:
            # Simulate workflow failure
            mock_workflow.return_value.ainvoke = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )
            
            result = await ai_service.verify_content(
                content="Test content",
                content_type="text",
                session_id="test_session"
            )
            
            # Should return inconclusive result with error information
            assert result["verdict"] == "inconclusive"
            assert result["confidence_score"] == 0.0
            assert "technical error" in result["reasoning"]
            assert len(result["errors"]) > 0
            assert "API rate limit exceeded" in result["errors"][0]
    
    @pytest.mark.asyncio
    async def test_content_caching(self, ai_service):
        """Test that identical content is cached."""
        with patch("app.services.ai_cache.redis_client") as mock_redis:
            # First call - cache miss
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            
            with patch("app.services.ai_service.create_fact_checking_workflow") as mock_workflow:
                mock_final_state = MagicMock()
                mock_final_state.verdict = "true"
                mock_final_state.confidence_score = 0.8
                mock_workflow.return_value.ainvoke = AsyncMock(return_value=mock_final_state)
                
                result1 = await ai_service.verify_content(
                    content="Test content",
                    content_type="text",
                    session_id="session1"
                )
                
                # Second call - should hit cache
                import json
                mock_redis.get.return_value = json.dumps(result1, default=str)
                
                result2 = await ai_service.verify_content(
                    content="Test content",
                    content_type="text",
                    session_id="session2"
                )
                
                # Results should be identical
                assert result1["verdict"] == result2["verdict"]
                assert result1["confidence_score"] == result2["confidence_score"]
                
                # Verify cache was called
                assert mock_redis.get.call_count >= 1
```

## 🕸️ LangGraph Testing

### Workflow Tests (`tests/test_langgraph/test_fact_checking.py`)

```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langgraph.workflows.fact_checking import FactCheckingState, create_fact_checking_workflow

class TestFactCheckingWorkflow:
    """Test LangGraph fact-checking workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self):
        """Test complete workflow execution with mocked nodes."""
        with patch("langgraph.nodes.content_extraction.content_extraction_node") as mock_extraction, \
             patch("langgraph.nodes.claim_identification.claim_identification_node") as mock_claims, \
             patch("langgraph.nodes.source_verification.source_verification_node") as mock_sources, \
             patch("langgraph.nodes.fact_analysis.fact_analysis_node") as mock_analysis, \
             patch("langgraph.nodes.confidence_scoring.confidence_scoring_node") as mock_confidence:
            
            # Mock each node's response
            mock_extraction.return_value = {
                "extracted_content": "Clean extracted content",
                "processing_steps": ["content_extraction"]
            }
            
            mock_claims.return_value = {
                "claims": ["Claim 1", "Claim 2"],
                "processing_steps": ["content_extraction", "claim_identification"]
            }
            
            mock_sources.return_value = {
                "sources_checked": ["https://reliable-source.com"],
                "processing_steps": ["content_extraction", "claim_identification", "source_verification"]
            }
            
            mock_analysis.return_value = {
                "verdict": "true",
                "reasoning": "Evidence supports the claims",
                "processing_steps": ["content_extraction", "claim_identification", "source_verification", "fact_analysis"]
            }
            
            mock_confidence.return_value = {
                "confidence_score": 0.85,
                "processing_steps": ["content_extraction", "claim_identification", "source_verification", "fact_analysis", "confidence_scoring"]
            }
            
            # Create workflow and initial state
            workflow = create_fact_checking_workflow()
            initial_state = FactCheckingState(
                content="Test content for verification",
                content_type="text",
                session_id="test_session"
            )
            
            # Execute workflow
            final_state = await workflow.ainvoke(initial_state)
            
            # Verify final state
            assert final_state.extracted_content == "Clean extracted content"
            assert len(final_state.claims) == 2
            assert final_state.verdict == "true"
            assert final_state.confidence_score == 0.85
            assert len(final_state.processing_steps) == 5
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test workflow behavior when nodes fail."""
        with patch("langgraph.nodes.content_extraction.content_extraction_node") as mock_extraction:
            # Simulate node failure
            mock_extraction.side_effect = Exception("Content extraction failed")
            
            workflow = create_fact_checking_workflow()
            initial_state = FactCheckingState(
                content="Test content",
                content_type="url",
                session_id="test_session"
            )
            
            # Workflow should handle the error gracefully
            final_state = await workflow.ainvoke(initial_state)
            
            # Check error was recorded
            assert len(final_state.errors) > 0
            assert "Content extraction failed" in final_state.errors[0]
    
    @pytest.mark.asyncio
    async def test_url_content_extraction(self):
        """Test URL content extraction node."""
        from langgraph.nodes.content_extraction import extract_url_content
        
        with patch("requests.get") as mock_get:
            # Mock successful HTTP response
            mock_response = MagicMock()
            mock_response.content = b'<html><body><article>Test article content</article></body></html>'
            mock_get.return_value = mock_response
            
            with patch("langchain_community.document_loaders.WebBaseLoader") as mock_loader:
                mock_doc = MagicMock()
                mock_doc.page_content = "Test article content"
                mock_loader.return_value.load.return_value = [mock_doc]
                
                content = await extract_url_content("https://example.com/article")
                
                assert "Test article content" in content
                assert len(content) > 0
    
    @pytest.mark.asyncio
    async def test_claim_identification(self):
        """Test claim identification node."""
        from langgraph.nodes.claim_identification import claim_identification_node
        
        with patch("langchain_openai.ChatOpenAI") as mock_llm:
            # Mock LLM response
            mock_response = MagicMock()
            mock_response.content = '["Claim 1: The sky is blue", "Claim 2: Water is wet"]'
            
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = mock_response
            
            mock_llm.return_value.__or__ = lambda self, other: mock_chain
            
            state = FactCheckingState(
                extracted_content="The sky is blue and water is wet.",
                content_type="text",
                session_id="test"
            )
            
            result = await claim_identification_node(state)
            
            assert len(result["claims"]) == 2
            assert "sky is blue" in result["claims"][0].lower()
    
    @pytest.mark.asyncio  
    async def test_confidence_scoring(self):
        """Test confidence scoring node."""
        from langgraph.nodes.confidence_scoring import confidence_scoring_node
        
        with patch("langchain_openai.ChatOpenAI") as mock_llm:
            # Mock LLM response with confidence score
            mock_response = MagicMock()
            mock_response.content = "0.85"
            
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = mock_response
            
            mock_llm.return_value.__or__ = lambda self, other: mock_chain
            
            state = FactCheckingState(
                verdict="true",
                reasoning="Strong evidence supports this claim",
                claims=["Test claim"],
                sources_checked=["https://reliable.com"],
                content_type="text",
                session_id="test"
            )
            
            result = await confidence_scoring_node(state)
            
            assert result["confidence_score"] == 0.85
            assert "confidence_scoring" in result["processing_steps"]
```

## 📊 Database Testing

### Repository Tests (`tests/test_database/test_repositories.py`)

```python
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.repositories import ContentRepository, FactCheckRepository, SessionRepository
from app.database.models import ContentStatus, FactCheckVerdict

class TestContentRepository:
    """Test content repository operations."""
    
    @pytest.mark.asyncio
    async def test_create_content_item(self, db_session: AsyncSession):
        """Test creating a content item."""
        repo = ContentRepository(db_session)
        
        content_item = await repo.create_content_item(
            session_id="test_session",
            content="Test content for fact-checking",
            url="https://example.com/article",
            title="Test Article",
            source_domain="example.com"
        )
        
        assert content_item.id is not None
        assert content_item.content == "Test content for fact-checking"
        assert content_item.session_id == "test_session"
        assert content_item.status == ContentStatus.PENDING
        assert content_item.source_domain == "example.com"
    
    @pytest.mark.asyncio
    async def test_update_content_status(self, db_session: AsyncSession):
        """Test updating content status."""
        repo = ContentRepository(db_session)
        
        # Create content item
        content_item = await repo.create_content_item(
            session_id="test_session",
            content="Test content"
        )
        
        # Update status
        success = await repo.update_content_status(
            content_item.id,
            ContentStatus.COMPLETED
        )
        
        assert success is True
        
        # Verify status was updated
        updated_item = await repo.get_content_by_id(content_item.id)
        assert updated_item.status == ContentStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_get_session_content_history(self, db_session: AsyncSession):
        """Test getting content history for a session."""
        repo = ContentRepository(db_session)
        session_id = "test_session"
        
        # Create multiple content items
        await repo.create_content_item(session_id, "Content 1")
        await repo.create_content_item(session_id, "Content 2")
        await repo.create_content_item(session_id, "Content 3")
        
        # Get history
        history = await repo.get_session_content_history(
            session_id,
            limit=2,
            offset=0
        )
        
        assert len(history) == 2
        # Should be ordered by created_at DESC
        assert history[0].content == "Content 3"
        assert history[1].content == "Content 2"

class TestFactCheckRepository:
    """Test fact-check repository operations."""
    
    @pytest.mark.asyncio
    async def test_create_fact_check_result(self, db_session: AsyncSession):
        """Test creating a fact-check result."""
        content_repo = ContentRepository(db_session)
        fact_repo = FactCheckRepository(db_session)
        
        # Create content item first
        content_item = await content_repo.create_content_item(
            session_id="test_session",
            content="Test content"
        )
        
        # Create fact-check result
        fact_check = await fact_repo.create_fact_check_result(
            content_id=content_item.id,
            session_id="test_session",
            verdict=FactCheckVerdict.TRUE,
            confidence_score=0.85,
            reasoning="Analysis shows this claim is accurate",
            claims_analyzed=["Test claim"],
            ai_model_used="gpt-4-turbo-preview",
            processing_time=5.2,
            processing_steps=["extraction", "analysis"]
        )
        
        assert fact_check.id is not None
        assert fact_check.verdict == FactCheckVerdict.TRUE
        assert fact_check.confidence_score == 0.85
        assert len(fact_check.claims_analyzed) == 1
        assert fact_check.processing_time_seconds == 5.2
    
    @pytest.mark.asyncio
    async def test_get_verdict_statistics(self, db_session: AsyncSession):
        """Test getting verdict statistics."""
        content_repo = ContentRepository(db_session)
        fact_repo = FactCheckRepository(db_session)
        
        # Create multiple fact-check results
        for verdict in [FactCheckVerdict.TRUE, FactCheckVerdict.FALSE, FactCheckVerdict.TRUE]:
            content_item = await content_repo.create_content_item(
                session_id="test_session",
                content=f"Content for {verdict.value}"
            )
            
            await fact_repo.create_fact_check_result(
                content_id=content_item.id,
                session_id="test_session",
                verdict=verdict,
                confidence_score=0.8,
                reasoning="Test reasoning",
                claims_analyzed=["Test claim"],
                ai_model_used="test-model",
                processing_time=1.0,
                processing_steps=["test"]
            )
        
        # Get statistics
        stats = await fact_repo.get_verdict_statistics(days=7)
        
        assert stats["true"] == 2
        assert stats["false"] == 1
        assert "inconclusive" not in stats  # No inconclusive results created
```

## 📈 Coverage & Performance Testing

### Coverage Configuration (`.coveragerc`)

```ini
[run]
source = app
omit = 
    app/tests/*
    app/migrations/*
    */venv/*
    */site-packages/*
    */distutils/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod

precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

### Performance Testing (`tests/test_performance.py`)

```python
import pytest
import asyncio
import time
from httpx import AsyncClient

class TestPerformance:
    """Test API performance and load handling."""
    
    @pytest.mark.asyncio
    async def test_content_verification_performance(self, client: AsyncClient):
        """Test content verification response time."""
        request_data = {
            "content_type": "text",
            "content": "Performance test content",
            "priority": 3
        }
        
        start_time = time.time()
        response = await client.post("/api/v1/content/verify", json=request_data)
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 5.0  # Should respond within 5 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client: AsyncClient):
        """Test handling of concurrent requests."""
        request_data = {
            "content_type": "text",
            "content": "Concurrent test content",
            "priority": 3
        }
        
        # Send 10 concurrent requests
        tasks = []
        for i in range(10):
            task = client.post("/api/v1/content/verify", json=request_data)
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Check that all requests completed
        assert len(responses) == 10
        
        # Check response times
        total_time = end_time - start_time
        assert total_time < 30.0  # All 10 requests should complete within 30 seconds
        
        # Check that most requests succeeded
        successful_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
        assert len(successful_responses) >= 8  # At least 80% success rate
```

## 🚀 Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=app --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_api/test_content.py

# Run specific test method
pytest tests/test_api/test_content.py::TestContentAPI::test_verify_text_content

# Run tests with specific markers
pytest -m "not slow"

# Run tests in parallel (with pytest-xdist)
pytest -n auto

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x

# Run only failed tests from last run
pytest --lf
```

### Test Configuration (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --disable-warnings
    --tb=short
    -ra
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    api: marks tests as API tests
    database: marks tests as database tests
asyncio_mode = auto
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

## 📊 Best Practices

### 1. Test Organization
- Group related tests in classes
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Keep tests independent and isolated

### 2. Mocking Strategy
- Mock external dependencies (APIs, databases)
- Use dependency injection for easier testing
- Mock at the service layer, not implementation details
- Verify mock calls when behavior matters

### 3. Fixtures & Data
- Use factories for creating test data
- Keep test data minimal but realistic
- Use parametrized tests for multiple scenarios
- Clean up test data after each test

### 4. Async Testing
- Use `pytest-asyncio` for async test support
- Handle event loops properly
- Test async error conditions
- Mock async dependencies correctly

### 5. Performance Testing
- Set reasonable performance thresholds
- Test under realistic load conditions
- Monitor resource usage during tests
- Use profiling tools for optimization

## 🔗 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [SQLAlchemy Testing Guide](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites) 