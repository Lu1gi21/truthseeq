# TruthSeeQ Workflow API Guide

## Overview

The TruthSeeQ Workflow API provides a comprehensive REST interface for executing and managing LangGraph-based workflows for fact-checking, content analysis, and source verification. This API addresses the previous design issues by providing direct access to the powerful workflow orchestrator capabilities.

## 🚀 Key Improvements

### 1. **Direct Workflow Access**
- **Before**: Workflows were only accessible internally through the orchestrator
- **After**: Direct API endpoints for each workflow type with proper REST interface

### 2. **Comprehensive Workflow Management**
- **Before**: No way to track workflow status or get execution history
- **After**: Full workflow lifecycle management with status tracking, history, and metrics

### 3. **Proper API Structure**
- **Before**: Workflow capabilities were hidden behind content scraping endpoints
- **After**: Dedicated workflow endpoints that clearly separate concerns

### 4. **Better User Experience**
- **Before**: Users had to go through content scraping to get workflow results
- **After**: Direct workflow execution with immediate access to results

## 📋 API Endpoints

### Workflow Execution Endpoints

#### 1. Fact-Checking Workflow
```http
POST /api/v1/workflow/fact-check
Content-Type: application/json

{
  "url": "https://example.com/article",
  "model_name": "gpt-4",
  "force_refresh": false
}
```

**Response:**
```json
{
  "workflow_id": "fact_check_12345",
  "url": "https://example.com/article",
  "verdict": "mostly_true",
  "confidence_score": 0.85,
  "reasoning": "Analysis based on 5 reliable sources...",
  "supporting_evidence": [...],
  "contradicting_evidence": [...],
  "processing_time": 12.5,
  "model_used": "gpt-4"
}
```

#### 2. Content Analysis Workflow
```http
POST /api/v1/workflow/content-analysis
Content-Type: application/json

{
  "url": "https://example.com/article",
  "model_name": "gpt-4",
  "force_refresh": false
}
```

**Response:**
```json
{
  "workflow_id": "content_analysis_12345",
  "url": "https://example.com/article",
  "content_quality_score": 0.78,
  "readability_score": 0.65,
  "sentiment_analysis": {...},
  "bias_analysis": {...},
  "content_category": "news",
  "topics": ["politics", "economy"],
  "processing_time": 8.2,
  "model_used": "gpt-4"
}
```

#### 3. Source Verification Workflow
```http
POST /api/v1/workflow/source-verification?url=https://example.com&model_name=gpt-4
```

**Response:**
```json
{
  "workflow_id": "source_verification_12345",
  "url": "https://example.com",
  "reputation_score": 0.82,
  "verification_score": 0.75,
  "verification_status": "verified",
  "trust_indicators": ["government_domain", "ssl_valid"],
  "red_flags": [],
  "processing_time": 3.1,
  "model_used": "gpt-4"
}
```

#### 4. Comprehensive Analysis
```http
POST /api/v1/workflow/comprehensive?url=https://example.com&model_name=gpt-4
```

**Response:**
```json
{
  "workflow_id": "comprehensive_12345",
  "url": "https://example.com",
  "processing_time": 25.3,
  "fact_checking": {...},
  "content_analysis": {...},
  "source_verification": {...}
}
```

### Workflow Management Endpoints

#### 1. Get Workflow Status
```http
GET /api/v1/workflow/{workflow_id}/status
```

**Response:**
```json
{
  "workflow_id": "fact_check_12345",
  "workflow_type": "fact_checking",
  "status": "completed",
  "created_at": "2025-01-02T12:00:00Z",
  "execution_time": 12.5,
  "content_id": 123
}
```

#### 2. Get Workflow History
```http
GET /api/v1/workflow/history?skip=0&limit=10
```

**Response:**
```json
{
  "items": [
    {
      "workflow_id": "fact_check_12345",
      "workflow_type": "fact_checking",
      "status": "completed",
      "created_at": "2025-01-02T12:00:00Z",
      "execution_time": 12.5
    }
  ],
  "total": 150,
  "skip": 0,
  "limit": 10
}
```

#### 3. Get Workflow Metrics
```http
GET /api/v1/workflow/metrics
```

**Response:**
```json
{
  "total_workflows": 150,
  "completed_workflows": 142,
  "failed_workflows": 8,
  "average_execution_time": 15.2,
  "workflow_types": {
    "fact_checking": {
      "completed": {"count": 50, "average_time": 18.5},
      "failed": {"count": 3, "average_time": 5.2}
    },
    "content_analysis": {
      "completed": {"count": 45, "average_time": 12.1},
      "failed": {"count": 2, "average_time": 3.8}
    }
  },
  "active_workflows": 3
}
```

#### 4. Workflow Health Check
```http
GET /api/v1/workflow/health
```

**Response:**
```json
{
  "service": "workflow",
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "active_workflows": 3,
  "total_workflows": 150,
  "average_execution_time": 15.2
}
```

## 🔧 Rate Limiting

The workflow API implements intelligent rate limiting based on workflow complexity:

- **Fact-Checking**: 5 requests per minute (most resource-intensive)
- **Content Analysis**: 10 requests per minute
- **Source Verification**: 15 requests per minute (lightweight)
- **Comprehensive Analysis**: 3 requests per minute (runs all workflows)

## 📊 Error Handling

### Rate Limit Exceeded
```json
{
  "error": "Rate limit exceeded",
  "detail": "Rate limit exceeded for fact-checking requests"
}
```

### Workflow Execution Error
```json
{
  "error": "Internal server error during fact-checking",
  "detail": "Workflow execution failed"
}
```

### Workflow Not Found
```json
{
  "error": "Workflow not found",
  "detail": "Workflow fact_check_12345 not found"
}
```

## 🚀 Usage Examples

### Python Client Example

```python
import asyncio
import aiohttp

async def fact_check_example():
    async with aiohttp.ClientSession() as session:
        # Execute fact-checking workflow
        payload = {
            "url": "https://example.com/article",
            "model_name": "gpt-4"
        }
        
        async with session.post(
            "http://localhost:8000/api/v1/workflow/fact-check",
            json=payload
        ) as response:
            result = await response.json()
            
            print(f"Verdict: {result['verdict']}")
            print(f"Confidence: {result['confidence_score']:.2%}")
            print(f"Processing Time: {result['processing_time']:.2f}s")

# Run the example
asyncio.run(fact_check_example())
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

async function factCheckExample() {
    try {
        const response = await axios.post('http://localhost:8000/api/v1/workflow/fact-check', {
            url: 'https://example.com/article',
            model_name: 'gpt-4'
        });
        
        const result = response.data;
        console.log(`Verdict: ${result.verdict}`);
        console.log(`Confidence: ${(result.confidence_score * 100).toFixed(1)}%`);
        console.log(`Processing Time: ${result.processing_time.toFixed(2)}s`);
        
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

factCheckExample();
```

### cURL Examples

#### Fact-Checking
```bash
curl -X POST "http://localhost:8000/api/v1/workflow/fact-check" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "model_name": "gpt-4"
  }'
```

#### Get Workflow Status
```bash
curl "http://localhost:8000/api/v1/workflow/fact_check_12345/status"
```

#### Get Workflow History
```bash
curl "http://localhost:8000/api/v1/workflow/history?limit=5"
```

## 🔄 Workflow Lifecycle

1. **Initiation**: Client sends workflow execution request
2. **Validation**: API validates request parameters and applies rate limiting
3. **Execution**: Workflow orchestrator executes the appropriate LangGraph workflow
4. **Processing**: Workflow runs through its nodes (content extraction, analysis, etc.)
5. **Completion**: Results are cached and returned to client
6. **Tracking**: Workflow execution is logged for history and metrics

## 📈 Performance Considerations

### Caching Strategy
- **Redis Caching**: Workflow results are cached with configurable TTL
- **Cache Keys**: Based on URL hash and model name for efficient retrieval
- **Cache Invalidation**: Force refresh option bypasses cache

### Parallel Processing
- **Comprehensive Analysis**: Runs all workflows in parallel for efficiency
- **Background Tasks**: Long-running workflows can be executed asynchronously
- **Resource Management**: Configurable limits on concurrent workflows

### Monitoring
- **Execution Metrics**: Track success rates, average times, and error rates
- **Resource Usage**: Monitor database and Redis connections
- **Health Checks**: Regular health monitoring for all services

## 🔒 Security Features

- **Rate Limiting**: Prevents API abuse with configurable limits
- **Input Validation**: Comprehensive validation of all request parameters
- **Error Handling**: Secure error responses that don't leak sensitive information
- **Session Management**: Optional session-based tracking for authenticated users

## 🛠️ Configuration

### Environment Variables
```bash
# Workflow Configuration
WORKFLOW_CACHE_TTL=3600
WORKFLOW_MAX_CONCURRENT=10
WORKFLOW_DEFAULT_MODEL=gpt-4

# Rate Limiting
RATE_LIMIT_FACT_CHECK=5
RATE_LIMIT_CONTENT_ANALYSIS=10
RATE_LIMIT_SOURCE_VERIFICATION=15
RATE_LIMIT_COMPREHENSIVE=3
```

### Database Schema
The workflow API uses the existing database schema with additional tables for:
- `AIWorkflowExecution`: Tracks workflow execution status
- `FactCheckResult`: Stores fact-checking results
- `AIAnalysisResult`: Stores content analysis results

## 🎯 Best Practices

1. **Use Appropriate Workflows**: Choose the right workflow for your use case
2. **Handle Rate Limits**: Implement exponential backoff for rate limit errors
3. **Monitor Status**: Use status endpoints for long-running workflows
4. **Cache Results**: Leverage caching to avoid redundant analysis
5. **Error Handling**: Implement proper error handling for all API calls

## 🔮 Future Enhancements

- **WebSocket Support**: Real-time workflow status updates
- **Batch Processing**: Execute multiple workflows in a single request
- **Custom Workflows**: User-defined workflow configurations
- **Advanced Analytics**: Detailed performance and accuracy metrics
- **Integration APIs**: Third-party service integrations

This new API design provides a much more logical and user-friendly interface for accessing TruthSeeQ's powerful workflow capabilities, addressing the previous design issues and providing a comprehensive solution for fact-checking and content analysis. 