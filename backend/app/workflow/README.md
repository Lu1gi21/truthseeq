# TruthSeeQ Workflow System

A comprehensive LangGraph-based workflow system for fact-checking, content analysis, and source verification using advanced AI models and web scraping capabilities.

## 🚀 Features

- **Fact-Checking Workflow**: Extract claims, verify sources, and generate confidence scores
- **Content Analysis Workflow**: Sentiment analysis, bias detection, and content quality assessment
- **Source Verification Workflow**: Domain analysis, reputation checking, and trust indicators
- **Advanced Web Scraping**: Anti-detection capabilities with multiple fallback methods
- **Brave Search Integration**: Web search for verification sources
- **Caching System**: Redis-based result caching for performance
- **Parallel Processing**: Execute multiple workflows simultaneously
- **Error Handling**: Robust error handling and recovery mechanisms

## 📋 Prerequisites

- Python 3.11+
- Redis server
- PostgreSQL database
- AI model API keys (OpenAI, Anthropic, etc.)

## 🛠️ Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# AI Models
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Brave Search (optional)
BRAVE_SEARCH_API_KEY=your_brave_search_key

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/truthseeq

# Redis
REDIS_URL=redis://localhost:6379
```

3. Initialize database:
```bash
alembic upgrade head
```

## 🎯 Quick Start

### Basic Usage

```python
import asyncio
from app.workflow.workflows import fact_check_url, analyze_content, verify_source

async def main():
    url = "https://example.com/article"
    
    # Fact-check a URL
    fact_result = await fact_check_url(url)
    print(f"Verdict: {fact_result['verdict']}")
    print(f"Confidence: {fact_result['confidence_score']:.2%}")
    
    # Analyze content
    content_result = await analyze_content(url)
    print(f"Quality: {content_result['content_quality_score']:.2%}")
    
    # Verify source
    source_result = await verify_source(url)
    print(f"Reputation: {source_result['reputation_score']:.2%}")

asyncio.run(main())
```

### Using Workflow Manager

```python
from app.workflow.workflows import WorkflowManager

async def comprehensive_analysis():
    manager = WorkflowManager("gpt-4")
    
    # Execute all workflows in parallel
    result = await manager.execute_comprehensive_analysis("https://example.com/article")
    
    print(f"Fact-checking: {result['fact_checking']['verdict']}")
    print(f"Content quality: {result['content_analysis']['content_quality_score']:.2%}")
    print(f"Source reputation: {result['source_verification']['reputation_score']:.2%}")
```

### Using Individual Workflows

```python
from app.workflow.workflows import FactCheckingWorkflow

async def custom_fact_check():
    workflow = FactCheckingWorkflow("gpt-4")
    result = await workflow.execute("https://example.com/article")
    
    # Access detailed results
    claims = result['extracted_claims']
    sources = result['verification_sources']
    evidence = result['supporting_evidence']
```

## 🔧 Workflow Components

### 1. Fact-Checking Workflow

**Purpose**: Verify factual accuracy of content claims

**Steps**:
1. **Content Extraction**: Scrape and preprocess content
2. **Claims Extraction**: Identify factual claims using AI
3. **Source Verification**: Find and verify supporting sources
4. **Fact Analysis**: Analyze factual accuracy of claims
5. **Confidence Scoring**: Generate final verdict and confidence score

**Output**:
```python
{
    "verdict": "mostly_true",
    "confidence_score": 0.85,
    "reasoning": "Analysis based on 5 reliable sources...",
    "extracted_claims": [...],
    "verification_sources": [...],
    "supporting_evidence": [...],
    "contradicting_evidence": [...]
}
```

### 2. Content Analysis Workflow

**Purpose**: Analyze content quality, bias, and characteristics

**Steps**:
1. **Content Extraction**: Scrape and preprocess content
2. **Sentiment Analysis**: Analyze emotional tone and sentiment
3. **Bias Detection**: Identify potential biases and political leanings
4. **Quality Assessment**: Evaluate content structure and readability
5. **Credibility Analysis**: Assess source reliability

**Output**:
```python
{
    "content_quality_score": 0.78,
    "readability_score": 0.65,
    "sentiment_score": 0.2,
    "emotional_tone": "neutral",
    "bias_level": "low",
    "content_category": "news",
    "topics": ["politics", "economy"],
    "key_insights": [...],
    "recommendations": [...]
}
```

### 3. Source Verification Workflow

**Purpose**: Verify source credibility and reliability

**Steps**:
1. **Domain Analysis**: Analyze domain characteristics and age
2. **Reputation Check**: Check domain reputation and trust indicators
3. **Fact-Checking Lookup**: Search known fact-checking databases
4. **Cross-Reference**: Compare with other reliable sources
5. **Verification Result**: Generate final verification assessment

**Output**:
```python
{
    "reputation_score": 0.82,
    "verification_score": 0.75,
    "verification_status": "verified",
    "trust_indicators": ["government_domain", "ssl_valid"],
    "red_flags": [],
    "verification_summary": "Source appears reliable...",
    "recommendations": [...]
}
```

## 🛠️ Custom Tools

### Web Search Tool

```python
from app.workflow.tools import web_search

# Search for verification sources
results = web_search("fact check claim about climate change", count=10)
for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Relevance: {result['relevance_score']:.2%}")
```

### Content Scraping Tool

```python
from app.workflow.tools import scrape_content

# Scrape content from URL
result = scrape_content("https://example.com/article")
if result['success']:
    print(f"Title: {result['title']}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Method: {result['method_used']}")
```

### Domain Reliability Tool

```python
from app.workflow.tools import check_domain_reliability

# Check domain reliability
result = check_domain_reliability("example.com")
print(f"Reliability: {result['reliability_score']:.2%}")
print(f"Trust indicators: {result['trust_indicators']}")
print(f"Red flags: {result['red_flags']}")
```

## 📊 State Management

The workflow system uses TypedDict for type-safe state management:

```python
from app.workflow.state import (
    FactCheckState, ContentAnalysisState, SourceVerificationState,
    create_fact_check_state, create_content_analysis_state, create_source_verification_state
)

# Create initial states
fact_state = create_fact_check_state("workflow_123", "https://example.com")
content_state = create_content_analysis_state("workflow_456", "https://example.com")
source_state = create_source_verification_state("workflow_789", "https://example.com")
```

## 🔄 Workflow Orchestration

### Using the Orchestrator

```python
from app.workflow.orchestrator import WorkflowOrchestrator
from app.schemas.content import FactCheckRequest

async def orchestrated_analysis():
    # Initialize orchestrator (requires database and Redis connections)
    orchestrator = WorkflowOrchestrator(db_session, redis_client)
    
    # Create request
    request = FactCheckRequest(url="https://example.com/article")
    
    # Execute with caching and persistence
    result = await orchestrator.execute_fact_checking(request)
    
    # Get workflow status
    status = await orchestrator.get_workflow_status(result.workflow_id)
    
    # Get metrics
    metrics = await orchestrator.get_workflow_metrics()
```

### Caching and Persistence

The orchestrator provides:
- **Result Caching**: Redis-based caching with configurable TTL
- **Database Persistence**: Store workflow results and execution history
- **Status Tracking**: Monitor workflow execution status
- **Metrics Collection**: Performance and usage metrics

## 🎮 Demo Script

Run the demo script to see the workflow system in action:

```bash
# Run all demos
python demo_workflow.py "https://example.com/article"

# Run specific demo
python demo_workflow.py "https://example.com/article" --fact-check
python demo_workflow.py "https://example.com/article" --content
python demo_workflow.py "https://example.com/article" --source
python demo_workflow.py "https://example.com/article" --comprehensive

# Use specific model
python demo_workflow.py "https://example.com/article" --all "claude-3-sonnet"
```

## 🔧 Configuration

### Model Configuration

```python
# Use different models for different workflows
fact_workflow = FactCheckingWorkflow("gpt-4")
content_workflow = ContentAnalysisWorkflow("claude-3-sonnet")
source_workflow = SourceVerificationWorkflow("gpt-3.5-turbo")
```

### Cache Configuration

```python
# Configure cache TTL
orchestrator = WorkflowOrchestrator(db_session, redis_client)
orchestrator.cache_ttl = 7200  # 2 hours
```

### Scraping Configuration

```python
# Configure advanced scraper settings
from app.advanced_scraper import AdvancedScraper

scraper = AdvancedScraper(
    max_retries=3,
    timeout=30,
    delay_between_requests=1.0,
    use_proxies=True,
    use_cookies=True
)
```

## 📈 Performance Optimization

### Parallel Execution

```python
# Execute multiple workflows in parallel
import asyncio

async def parallel_analysis(urls):
    tasks = [
        fact_check_url(url) for url in urls
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Caching Strategy

```python
# Use caching for repeated requests
result = await orchestrator.execute_fact_checking(request)  # Cached
result = await orchestrator.execute_fact_checking(request, force_refresh=True)  # Fresh
```

### Error Handling

```python
try:
    result = await fact_check_url(url)
    if result['status'] == 'completed':
        print(f"Success: {result['verdict']}")
    else:
        print(f"Failed: {result['error_message']}")
except Exception as e:
    print(f"Error: {e}")
```

## 🔍 Monitoring and Debugging

### Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.workflow")

# Log workflow events
logger.info(f"Starting fact-checking workflow for {url}")
logger.warning(f"Content extraction failed: {error}")
logger.error(f"Workflow failed: {exception}")
```

### Metrics

```python
# Get workflow metrics
metrics = await orchestrator.get_workflow_metrics()
print(f"Total workflows: {metrics['total_workflows']}")
print(f"Average execution time: {metrics['average_execution_time']:.2f}s")
print(f"Success rate: {metrics['completed_workflows'] / metrics['total_workflows']:.2%}")
```

### Health Check

```python
# Check system health
health = await orchestrator.health_check()
print(f"Status: {health['status']}")
print(f"Database: {health['database']}")
print(f"Redis: {health['redis']}")
```

## 🚀 Advanced Usage

### Custom Nodes

```python
from app.workflow.nodes import BaseNode

class CustomAnalysisNode(BaseNode):
    async def __call__(self, state):
        # Custom analysis logic
        return {"custom_result": "analysis_complete"}
```

### Custom Tools

```python
from langchain_core.tools import tool

@tool
def custom_verification_tool(query: str) -> str:
    """Custom verification tool."""
    # Custom verification logic
    return "verification_result"
```

### Workflow Composition

```python
# Compose workflows with custom logic
async def custom_workflow(url):
    # Step 1: Basic fact-checking
    fact_result = await fact_check_url(url)
    
    # Step 2: Additional analysis if needed
    if fact_result['confidence_score'] < 0.5:
        content_result = await analyze_content(url)
        source_result = await verify_source(url)
        
        # Combine results
        return {
            "fact_checking": fact_result,
            "content_analysis": content_result,
            "source_verification": source_result
        }
    
    return {"fact_checking": fact_result}
```

## 📚 API Reference

### Workflow Classes

- `FactCheckingWorkflow`: Main fact-checking workflow
- `ContentAnalysisWorkflow`: Content analysis workflow
- `SourceVerificationWorkflow`: Source verification workflow
- `WorkflowManager`: Manager for all workflows
- `WorkflowOrchestrator`: Full-featured orchestrator with persistence

### State Classes

- `FactCheckState`: State for fact-checking workflow
- `ContentAnalysisState`: State for content analysis workflow
- `SourceVerificationState`: State for source verification workflow

### Tool Functions

- `web_search()`: Search the web using Brave Search
- `scrape_content()`: Scrape content from URL
- `scrape_multiple_urls()`: Scrape multiple URLs in parallel
- `check_domain_reliability()`: Check domain reliability
- `search_fact_checking_databases()`: Search fact-checking databases

### Convenience Functions

- `fact_check_url()`: Quick fact-checking
- `analyze_content()`: Quick content analysis
- `verify_source()`: Quick source verification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the demo scripts

---

**Note**: This workflow system is designed to be modular and extensible. You can easily add new workflows, nodes, and tools to suit your specific needs. 