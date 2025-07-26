# TruthSeeQ Scraper Service Guide

## Overview

The TruthSeeQ Scraper Service is a comprehensive content scraping and analysis system that integrates with the existing `advanced_scraper.py` and adds enterprise-grade features for content discovery, quality assessment, and deduplication.

## Phase 2 Implementation Features

### 🔍 **Brave Search Integration**
- **Web Search**: Search the web using Brave Search API with filtering options
- **News Search**: Specialized news content discovery with freshness filters
- **Localization**: Country and language-specific search results
- **Rate Limiting**: Built-in API rate limiting and error handling

### 🤖 **Advanced Scraping Engine**
- **Multi-Method Scraping**: Integration with `advanced_scraper.py` for robust content extraction
- **Background Processing**: Async scraping with job tracking and progress monitoring
- **Content Quality Analysis**: Automatic assessment of content readability, structure, and credibility
- **Deduplication**: Intelligent duplicate detection using text and URL similarity

### 📊 **Content Management**
- **Database Integration**: Automatic storage of scraped content with metadata
- **Quality Scoring**: Comprehensive content quality metrics
- **Validation System**: Content authenticity and quality validation
- **Batch Processing**: Efficient handling of multiple URLs simultaneously

## Setup and Configuration

### 1. Environment Variables

Copy `backend/.env.example` to `backend/.env` and configure:

```bash
# Brave Search API Configuration
BRAVE_API_KEY=your-brave-search-api-key-here

# Database Configuration
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_USER=truthseeq
POSTGRES_PASSWORD=your-database-password
POSTGRES_DB=truthseeq

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Scraping Configuration
SCRAPER_MAX_CONCURRENT=5
SCRAPER_REQUEST_DELAY=1.0
SCRAPER_TIMEOUT=30
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Database Setup

```bash
# Run database migrations
alembic upgrade head
```

### 4. Start the Service

```bash
# Development
uvicorn app.main:app --reload

# Production
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## API Endpoints

### Content Search

#### Search Web Content
```http
POST /api/v1/content/search
Content-Type: application/json

{
  "query": "climate change latest research",
  "max_results": 10,
  "search_type": "web",
  "country": "US",
  "language": "en",
  "freshness": "pd"
}
```

#### Search News Content
```http
POST /api/v1/content/search
Content-Type: application/json

{
  "query": "AI breakthrough",
  "max_results": 20,
  "search_type": "news",
  "freshness": "pw"
}
```

### Content Scraping

#### Scrape URLs
```http
POST /api/v1/content/scrape
Content-Type: application/json

{
  "urls": [
    "https://example.com/article1",
    "https://example.com/article2"
  ],
  "priority": 5,
  "force_rescrape": false,
  "include_metadata": true
}
```

Response:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "total_urls": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "url": "https://example.com/article1",
      "success": true,
      "content": "Extracted content...",
      "title": "Article Title",
      "method_used": "requests",
      "response_time": 1.23,
      "quality_score": 0.85
    }
  ],
  "processing_time": 2.45
}
```

### Content Management

#### Get Content by ID
```http
GET /api/v1/content/123
```

#### List Content with Filtering
```http
GET /api/v1/content/?skip=0&limit=50&domain=example.com&status=completed
```

### Content Validation

#### Validate Content Quality
```http
POST /api/v1/content/123/validate
```

Response:
```json
{
  "content_id": 123,
  "validation_type": "quality",
  "is_valid": true,
  "confidence": 0.85,
  "quality_metrics": {
    "length_score": 0.8,
    "readability_score": 0.9,
    "structure_score": 0.7,
    "metadata_completeness": 0.6,
    "source_credibility": 0.9,
    "overall_score": 0.8
  },
  "recommendations": [
    "Content structure could be improved - add headings and paragraphs"
  ]
}
```

### Content Deduplication

#### Analyze Content Similarity
```http
POST /api/v1/content/deduplicate
Content-Type: application/json

{
  "content_ids": [1, 2, 3, 4, 5],
  "similarity_threshold": 0.8,
  "deduplication_strategy": "keep_first"
}
```

Response:
```json
{
  "total_content": 5,
  "duplicates_found": 2,
  "duplicates_removed": 2,
  "duplicate_groups": [[1, 3], [2, 4]],
  "similarities": [
    {
      "content_id_1": 1,
      "content_id_2": 3,
      "similarity_score": 0.95,
      "similarity_type": "combined",
      "details": {
        "text_similarity": 0.98,
        "url_similarity": 0.9
      }
    }
  ]
}
```

### Job Status Monitoring

#### Check Job Status
```http
GET /api/v1/content/jobs/123e4567-e89b-12d3-a456-426614174000
```

Response:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "job_type": "scraping",
  "status": "completed",
  "progress": 1.0,
  "created_at": "2025-01-02T12:00:00Z",
  "started_at": "2025-01-02T12:00:01Z",
  "completed_at": "2025-01-02T12:00:05Z",
  "result_summary": {
    "total_urls": 10,
    "successful": 8,
    "failed": 2,
    "processing_time": 4.23
  }
}
```

## Python SDK Usage

### Basic Scraping

```python
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.scraper_service import ScraperService
from app.schemas.content import ScrapingRequest

async def scrape_content():
    # Initialize service
    async with get_async_session() as db:
        scraper = ScraperService(db)
        
        # Create scraping request
        request = ScrapingRequest(
            urls=["https://example.com/article"],
            include_metadata=True
        )
        
        # Perform scraping
        result = await scraper.scrape_urls(request)
        
        print(f"Scraped {result.successful} URLs successfully")
        
        # Clean up
        await scraper.close()

# Run async function
asyncio.run(scrape_content())
```

### Content Search

```python
import asyncio
from app.services.scraper_service import ScraperService
from app.schemas.content import SearchRequest

async def search_content():
    async with get_async_session() as db:
        scraper = ScraperService(db)
        
        # Search for content
        request = SearchRequest(
            query="artificial intelligence",
            max_results=10,
            search_type="news"
        )
        
        result = await scraper.search_content(request)
        
        print(f"Found {result.total_results} results:")
        for item in result.results:
            print(f"- {item.title} ({item.source_domain})")
        
        await scraper.close()

asyncio.run(search_content())
```

### Content Quality Analysis

```python
async def analyze_quality():
    async with get_async_session() as db:
        scraper = ScraperService(db)
        
        # Validate specific content
        validation = await scraper.validate_content(content_id=123)
        
        print(f"Content valid: {validation.is_valid}")
        print(f"Overall score: {validation.quality_metrics.overall_score:.2f}")
        
        if validation.recommendations:
            print("Recommendations:")
            for rec in validation.recommendations:
                print(f"- {rec}")
```

## Quality Scoring System

The scraper service includes a comprehensive quality scoring system:

### Quality Metrics

1. **Length Score (25%)**: Based on content length adequacy
   - < 100 chars: 0.2
   - 100-500 chars: 0.5  
   - 500-2000 chars: 0.8
   - > 2000 chars: 1.0

2. **Readability Score (25%)**: Based on sentence structure
   - Average sentence length and paragraph structure
   - Shorter sentences score higher

3. **Structure Score (20%)**: Based on content organization
   - Presence of headings, lists, paragraphs
   - Title availability

4. **Metadata Completeness (15%)**: Based on available metadata
   - Author, publish date, description, tags
   - More complete metadata scores higher

5. **Source Credibility (15%)**: Based on domain reputation
   - High credibility: Reuters, BBC, Nature, WHO, etc. (0.9)
   - Medium credibility: CNN, NYT, WSJ, etc. (0.7)
   - Low credibility: Blogs, social media (0.3)
   - Unknown domains: (0.5)

## Deduplication System

The deduplication system uses multiple similarity algorithms:

### Similarity Methods

1. **Text Similarity**: Longest common substring analysis
2. **URL Similarity**: Domain and path comparison
3. **Combined Score**: Weighted combination (70% text + 30% URL)

### Deduplication Strategies

- **keep_first**: Keep the first item in each duplicate group
- **keep_best**: Keep the item with highest quality score
- **merge**: Combine metadata from duplicates (future feature)

## Rate Limiting

The service implements comprehensive rate limiting:

- **Search**: 30 requests per minute
- **Scraping**: 10 requests per minute  
- **Validation**: 20 requests per minute
- **Deduplication**: 5 requests per minute

Rate limits are applied per session/IP address.

## Error Handling

The service provides robust error handling:

### Common Error Codes

- **400**: Validation error (invalid request format)
- **404**: Content not found
- **429**: Rate limit exceeded
- **500**: Internal server error

### Error Response Format

```json
{
  "error": "Rate limit exceeded",
  "detail": "Rate limit exceeded for scraping requests",
  "retry_after": 60
}
```

## Integration with Next Phases

The Phase 2 implementation is designed for seamless integration with future phases:

### Phase 3: AI Integration
- Content preprocessing for LangGraph workflows
- Quality metrics for AI model input selection
- Structured content format for fact-checking

### Phase 4: Social Features  
- Content scoring for feed algorithms
- Deduplication for preventing spam
- Source credibility for user trust indicators

### Phase 5: Background Tasks
- Job tracking system ready for Celery integration
- Async processing architecture in place
- Scalable batch processing design

## Monitoring and Observability

### Health Checks

```http
GET /health          # Basic health check
GET /health/deep     # Detailed service connectivity check
GET /metrics         # Basic metrics (Prometheus-compatible)
```

### Logging

The service provides structured logging with:
- Request/response tracking
- Performance metrics
- Error context
- Job progress monitoring

### Performance Monitoring

Key metrics to monitor:
- Scraping success rate
- Average response times
- Quality score distributions
- Deduplication effectiveness
- Rate limit hit rates

## Best Practices

### For Developers

1. **Always use rate limiting**: Respect API limits and implement proper backoff
2. **Handle errors gracefully**: Check response status and handle failures
3. **Use job tracking**: For batch operations, monitor job progress
4. **Quality validation**: Validate content before further processing
5. **Cleanup resources**: Always close scraper service connections

### For Production

1. **Environment Configuration**: Use production-grade settings
2. **Database Optimization**: Ensure proper indexing and connection pooling
3. **Monitoring Setup**: Implement comprehensive logging and metrics
4. **Security**: Configure proper CORS, rate limiting, and authentication
5. **Scaling**: Consider horizontal scaling for high-volume usage

## Troubleshooting

### Common Issues

#### Brave Search API Errors
```
Error: No Brave API key - returning empty search results
```
**Solution**: Set `BRAVE_API_KEY` in environment variables

#### Database Connection Issues
```
Error: Database connection failed
```
**Solution**: Check PostgreSQL configuration and connection string

#### Scraping Failures
```
Error: All scraping methods failed for URL
```
**Solution**: Check if URL is accessible, not rate-limited, or in blocklist

#### Rate Limiting
```
Error: Rate limit exceeded
```
**Solution**: Implement backoff strategy and respect rate limits

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

This provides detailed request/response logging and error context.

## Support and Contributing

For issues, feature requests, or contributions:

1. Check the logs for detailed error information
2. Verify environment configuration
3. Test with simple examples first
4. Refer to the API documentation at `/docs`

The Phase 2 implementation provides a solid foundation for the TruthSeeQ platform's content processing pipeline, ready for integration with AI workflows and social features in future phases. 