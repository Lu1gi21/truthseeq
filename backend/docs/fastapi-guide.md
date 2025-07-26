# FastAPI Implementation Guide

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

## 🚀 Key Features

- **Fast**: Very high performance, on par with NodeJS and Go
- **Fast to code**: Increase development speed by 200-300%
- **Fewer bugs**: Reduce about 40% of human errors
- **Intuitive**: Great editor support with completion
- **Easy**: Designed to be easy to use and learn
- **Standards-based**: Based on OpenAPI and JSON Schema

## 📋 Installation & Setup

```bash
pip install fastapi[standard]==0.116.1
pip install uvicorn[standard]==0.35.0
```

## 🏗️ Basic Application Structure

### Main Application (`app/main.py`)

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time

from app.core.config import settings
from app.core.logging import setup_logging
from app.api import api_router
from app.core.exceptions import custom_exception_handler

# Setup logging
setup_logging()

# Create FastAPI instance
app = FastAPI(
    title="TruthSeeQ API",
    description="AI-powered misinformation detection platform",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Add timing middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handlers
app.add_exception_handler(Exception, custom_exception_handler)

# Include routers
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
```

## 🛠️ Configuration Management (`app/config.py`)

```python
from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "TruthSeeQ"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str
    DATABASE_URL_SYNC: str
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

## 🔗 API Routes Structure

### Router Setup (`app/api/__init__.py`)

```python
from fastapi import APIRouter
from app.api.routes import content, feed, rate_limit

api_router = APIRouter()

api_router.include_router(
    content.router, 
    prefix="/content", 
    tags=["content"]
)
api_router.include_router(
    feed.router, 
    prefix="/feed", 
    tags=["feed"]
)
api_router.include_router(
    rate_limit.router, 
    prefix="/rate-limit", 
    tags=["rate-limiting"]
)
```

### Content Routes (`app/api/routes/content.py`)

```python
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional
import uuid

from app.schemas.content import (
    ContentVerificationRequest,
    ContentVerificationResponse,
    ContentItem,
    BatchVerificationRequest
)
from app.services.fact_checker import FactCheckerService
from app.services.scraper_service import ScraperService
from app.api.dependencies import get_current_session, rate_limit
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post(
    "/verify",
    response_model=ContentVerificationResponse,
    summary="Verify content for misinformation",
    description="Submit content (URL or text) for AI-powered fact-checking analysis"
)
async def verify_content(
    request: ContentVerificationRequest,
    background_tasks: BackgroundTasks,
    session: dict = Depends(get_current_session),
    fact_checker: FactCheckerService = Depends(),
    _: None = Depends(rate_limit)
):
    """
    Verify content for misinformation using AI analysis.
    
    - **content_type**: Either 'url' or 'text'
    - **content**: The URL or text to analyze
    - **priority**: Processing priority (1-5, default 3)
    """
    try:
        logger.info(f"Content verification request", extra={
            "session_id": session["id"],
            "content_type": request.content_type,
            "priority": request.priority
        })
        
        # Process the verification request
        result = await fact_checker.verify_content(
            content_type=request.content_type,
            content=request.content,
            session_id=session["id"],
            priority=request.priority
        )
        
        # Schedule background tasks
        background_tasks.add_task(
            fact_checker.update_analytics,
            session["id"],
            result.verification_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Content verification failed", extra={
            "session_id": session["id"],
            "error": str(e)
        })
        raise HTTPException(
            status_code=500,
            detail="Content verification failed"
        )

@router.get(
    "/{verification_id}",
    response_model=ContentVerificationResponse,
    summary="Get verification results"
)
async def get_verification_result(
    verification_id: uuid.UUID,
    session: dict = Depends(get_current_session),
    fact_checker: FactCheckerService = Depends()
):
    """Get the results of a content verification by ID."""
    result = await fact_checker.get_verification_result(
        verification_id, 
        session["id"]
    )
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail="Verification result not found"
        )
    
    return result

@router.get(
    "/history",
    response_model=List[ContentItem],
    summary="Get verification history"
)
async def get_verification_history(
    limit: int = 20,
    offset: int = 0,
    session: dict = Depends(get_current_session),
    fact_checker: FactCheckerService = Depends()
):
    """Get the verification history for the current session."""
    return await fact_checker.get_session_history(
        session["id"],
        limit=limit,
        offset=offset
    )

@router.post(
    "/batch-verify",
    response_model=List[ContentVerificationResponse],
    summary="Batch verify multiple content items"
)
async def batch_verify_content(
    request: BatchVerificationRequest,
    background_tasks: BackgroundTasks,
    session: dict = Depends(get_current_session),
    fact_checker: FactCheckerService = Depends(),
    _: None = Depends(rate_limit)
):
    """
    Verify multiple content items in batch.
    Limited to 10 items per request.
    """
    if len(request.items) > 10:
        raise HTTPException(
            status_code=400,
            detail="Batch size cannot exceed 10 items"
        )
    
    results = []
    for item in request.items:
        try:
            result = await fact_checker.verify_content(
                content_type=item.content_type,
                content=item.content,
                session_id=session["id"],
                priority=item.priority or 3
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Batch item verification failed", extra={
                "session_id": session["id"],
                "error": str(e)
            })
            # Continue with other items
            continue
    
    return results
```

## 📝 Pydantic Schemas

### Content Schemas (`app/schemas/content.py`)

```python
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Optional, Literal
from datetime import datetime
import uuid

class ContentVerificationRequest(BaseModel):
    """Request model for content verification."""
    
    content_type: Literal["url", "text"] = Field(
        ..., 
        description="Type of content to verify"
    )
    content: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="The URL or text content to verify"
    )
    priority: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Processing priority (1=highest, 5=lowest)"
    )
    
    @validator('content')
    def validate_content(cls, v, values):
        content_type = values.get('content_type')
        if content_type == 'url':
            # Basic URL validation
            if not v.startswith(('http://', 'https://')):
                raise ValueError('URL must start with http:// or https://')
        elif content_type == 'text':
            if len(v.strip()) < 10:
                raise ValueError('Text content must be at least 10 characters')
        return v

class VerificationResult(BaseModel):
    """AI verification result details."""
    
    verdict: Literal["true", "false", "inconclusive", "satire"] = Field(
        ...,
        description="Verification verdict"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)"
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation of the verdict"
    )
    sources_checked: List[str] = Field(
        default_factory=list,
        description="List of sources consulted"
    )
    ai_model_used: str = Field(
        ...,
        description="AI model used for analysis"
    )

class ContentVerificationResponse(BaseModel):
    """Response model for content verification."""
    
    verification_id: uuid.UUID = Field(
        ...,
        description="Unique verification identifier"
    )
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    content_type: str = Field(
        ...,
        description="Type of content verified"
    )
    original_content: str = Field(
        ...,
        description="Original content submitted"
    )
    extracted_content: Optional[str] = Field(
        None,
        description="Extracted content (for URLs)"
    )
    result: VerificationResult = Field(
        ...,
        description="Verification results"
    )
    processing_time_seconds: float = Field(
        ...,
        description="Time taken to process"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp of verification"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

class BatchVerificationItem(BaseModel):
    """Single item in batch verification request."""
    
    content_type: Literal["url", "text"]
    content: str = Field(..., min_length=1, max_length=10000)
    priority: Optional[int] = Field(default=3, ge=1, le=5)

class BatchVerificationRequest(BaseModel):
    """Request model for batch content verification."""
    
    items: List[BatchVerificationItem] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of content items to verify"
    )
```

## 🔒 Dependencies

### Authentication & Session Management (`app/api/dependencies.py`)

```python
from fastapi import Depends, HTTPException, Request
from typing import Dict, Any
import hashlib
import time

from app.core.rate_limiting import RateLimiter
from app.core.logging import get_logger

logger = get_logger(__name__)

async def get_current_session(request: Request) -> Dict[str, Any]:
    """
    Get or create a session for the current request.
    Uses IP address and User-Agent for session identification.
    """
    # Create session identifier
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    session_id = hashlib.md5(
        f"{client_ip}:{user_agent}".encode()
    ).hexdigest()
    
    return {
        "id": session_id,
        "ip_address": client_ip,
        "user_agent": user_agent,
        "created_at": time.time()
    }

# Rate limiting dependency
rate_limiter = RateLimiter()

async def rate_limit(request: Request) -> None:
    """Apply rate limiting based on IP address."""
    client_ip = request.client.host
    
    if not await rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )
```

## 🚀 Development & Testing

### Running the Application

```bash
# Development with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Testing with FastAPI

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_verify_content():
    response = client.post(
        "/api/v1/content/verify",
        json={
            "content_type": "text",
            "content": "This is a test statement to verify.",
            "priority": 3
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "verification_id" in data
    assert "result" in data
```

## 📖 Best Practices

### 1. Request/Response Models
- Always use Pydantic models for request/response validation
- Include comprehensive field validation
- Add detailed descriptions for API documentation

### 2. Error Handling
- Use custom exception handlers
- Return consistent error responses
- Log errors with contextual information

### 3. Async Operations
- Use async/await for I/O operations
- Implement background tasks for non-blocking operations
- Use dependency injection for service classes

### 4. Security
- Implement rate limiting
- Validate all input data
- Use CORS middleware appropriately
- Don't expose sensitive information in responses

### 5. Documentation
- Use FastAPI's automatic OpenAPI generation
- Add comprehensive docstrings
- Include request/response examples

## 🔗 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Starlette Documentation](https://www.starlette.io/) 