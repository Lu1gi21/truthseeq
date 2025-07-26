# LangChain & LangGraph Implementation Guide

LangChain is a framework for developing applications powered by language models, while LangGraph is a library for building stateful, multi-actor applications with LLMs.

## 🎯 Overview for TruthSeeQ

In TruthSeeQ, we use LangGraph to create sophisticated fact-checking workflows that can:
- Extract content from URLs
- Verify claims against reliable sources
- Generate confidence scores
- Provide detailed explanations
- Handle complex multi-step reasoning

## 📋 Installation & Setup

```bash
pip install langgraph==0.3.28
pip install langchain==0.3.18
pip install langchain-community==0.3.17
pip install langchain-core==0.3.28
pip install langchain-openai==0.2.15
pip install langchain-anthropic==0.2.10
```

## 🏗️ Core Architecture

### LangGraph Workflow Structure

```python
# langgraph/workflows/fact_checking.py
from typing import Dict, Any, List
from langgraph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

class FactCheckingState(BaseModel):
    """State for the fact-checking workflow."""
    
    # Input
    content: str
    content_type: str  # 'url' or 'text'
    session_id: str
    
    # Processing state
    extracted_content: str = ""
    claims: List[str] = []
    sources_checked: List[str] = []
    
    # Results
    verdict: str = ""
    confidence_score: float = 0.0
    reasoning: str = ""
    ai_model_used: str = ""
    
    # Metadata
    processing_steps: List[str] = []
    errors: List[str] = []

def create_fact_checking_workflow() -> StateGraph:
    """Create the main fact-checking workflow."""
    
    workflow = StateGraph(FactCheckingState)
    
    # Add nodes
    workflow.add_node("content_extraction", content_extraction_node)
    workflow.add_node("claim_identification", claim_identification_node)
    workflow.add_node("source_verification", source_verification_node)
    workflow.add_node("fact_analysis", fact_analysis_node)
    workflow.add_node("confidence_scoring", confidence_scoring_node)
    workflow.add_node("result_generation", result_generation_node)
    
    # Define the flow
    workflow.set_entry_point("content_extraction")
    workflow.add_edge("content_extraction", "claim_identification")
    workflow.add_edge("claim_identification", "source_verification")
    workflow.add_edge("source_verification", "fact_analysis")
    workflow.add_edge("fact_analysis", "confidence_scoring")
    workflow.add_edge("confidence_scoring", "result_generation")
    workflow.add_edge("result_generation", END)
    
    return workflow.compile()
```

## 🔧 Node Implementations

### Content Extraction Node

```python
# langgraph/nodes/content_extraction.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
from typing import Dict, Any

async def content_extraction_node(state: FactCheckingState) -> Dict[str, Any]:
    """
    Extract and clean content from URLs or prepare text content.
    """
    try:
        if state.content_type == "url":
            # Extract content from URL
            extracted_content = await extract_url_content(state.content)
        else:
            # Use text content as-is
            extracted_content = state.content
        
        # Clean and preprocess content
        cleaned_content = clean_content(extracted_content)
        
        return {
            "extracted_content": cleaned_content,
            "processing_steps": state.processing_steps + ["content_extraction"]
        }
        
    except Exception as e:
        return {
            "errors": state.errors + [f"Content extraction failed: {str(e)}"],
            "processing_steps": state.processing_steps + ["content_extraction_failed"]
        }

async def extract_url_content(url: str) -> str:
    """Extract main content from a URL."""
    try:
        # Load the webpage
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content found at URL")
        
        # Extract main content using BeautifulSoup for better parsing
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get main content areas
        main_content = ""
        for selector in ['article', 'main', '.content', '#content']:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text()
                break
        
        if not main_content:
            main_content = soup.get_text()
        
        return main_content
        
    except Exception as e:
        raise Exception(f"Failed to extract content from URL: {str(e)}")

def clean_content(content: str) -> str:
    """Clean and preprocess extracted content."""
    # Remove extra whitespace
    content = ' '.join(content.split())
    
    # Truncate if too long (keep first 5000 characters)
    if len(content) > 5000:
        content = content[:5000] + "..."
    
    return content
```

### Claim Identification Node

```python
# langgraph/nodes/claim_identification.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List

async def claim_identification_node(state: FactCheckingState) -> Dict[str, Any]:
    """
    Identify factual claims that can be verified.
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert fact-checker. Your task is to identify specific, 
            verifiable factual claims from the given content.
            
            Focus on:
            - Specific statements about facts, events, numbers, dates
            - Claims that can be verified against reliable sources
            - Statements presented as factual rather than opinions
            
            Ignore:
            - Personal opinions
            - Predictions about the future
            - Subjective statements
            
            Return a JSON list of claims in this format:
            ["claim 1", "claim 2", "claim 3"]
            
            Limit to the 5 most important verifiable claims."""),
            ("human", "Content: {content}")
        ])
        
        chain = prompt | llm
        result = await chain.ainvoke({"content": state.extracted_content})
        
        # Parse the claims from the LLM response
        claims = parse_claims_from_response(result.content)
        
        return {
            "claims": claims,
            "processing_steps": state.processing_steps + ["claim_identification"],
            "ai_model_used": "gpt-4-turbo-preview"
        }
        
    except Exception as e:
        return {
            "errors": state.errors + [f"Claim identification failed: {str(e)}"],
            "processing_steps": state.processing_steps + ["claim_identification_failed"]
        }

def parse_claims_from_response(response: str) -> List[str]:
    """Parse claims from LLM response."""
    try:
        import json
        # Try to parse as JSON
        claims = json.loads(response)
        if isinstance(claims, list):
            return claims[:5]  # Limit to 5 claims
    except:
        # Fallback: split by lines and clean
        lines = response.strip().split('\n')
        claims = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove bullet points, numbers, etc.
                clean_line = line.lstrip('- •123456789. ')
                if clean_line:
                    claims.append(clean_line)
        return claims[:5]
    
    return []
```

### Source Verification Node

```python
# langgraph/nodes/source_verification.py
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List

async def source_verification_node(state: FactCheckingState) -> Dict[str, Any]:
    """
    Search for reliable sources to verify claims.
    """
    try:
        search_tool = DuckDuckGoSearchRun()
        sources_checked = []
        
        # Search for each claim
        for claim in state.claims[:3]:  # Limit to top 3 claims
            search_results = await search_for_claim(claim, search_tool)
            sources_checked.extend(search_results)
        
        # Remove duplicates
        sources_checked = list(set(sources_checked))
        
        return {
            "sources_checked": sources_checked,
            "processing_steps": state.processing_steps + ["source_verification"]
        }
        
    except Exception as e:
        return {
            "errors": state.errors + [f"Source verification failed: {str(e)}"],
            "processing_steps": state.processing_steps + ["source_verification_failed"]
        }

async def search_for_claim(claim: str, search_tool) -> List[str]:
    """Search for sources related to a specific claim."""
    try:
        # Create search query
        search_query = f'"{claim}" site:reuters.com OR site:bbc.com OR site:apnews.com OR site:factcheck.org'
        
        # Perform search
        search_results = search_tool.run(search_query)
        
        # Extract URLs from search results
        sources = extract_urls_from_search(search_results)
        
        return sources[:3]  # Limit to top 3 sources per claim
        
    except Exception as e:
        return []

def extract_urls_from_search(search_results: str) -> List[str]:
    """Extract URLs from search results."""
    import re
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', search_results)
    return urls
```

### Fact Analysis Node

```python
# langgraph/nodes/fact_analysis.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

async def fact_analysis_node(state: FactCheckingState) -> Dict[str, Any]:
    """
    Analyze claims against sources and determine verdict.
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert fact-checker. Analyze the given claims against 
            the provided sources and determine the overall verdict.
            
            Possible verdicts:
            - "true": Claims are factually accurate and well-supported
            - "false": Claims are factually incorrect or misleading
            - "inconclusive": Insufficient evidence to make a determination
            - "satire": Content appears to be satirical or humorous
            
            Provide:
            1. Overall verdict (one of the above)
            2. Detailed reasoning explaining your analysis
            3. Key evidence that supports your verdict
            
            Be conservative - if you're not confident, use "inconclusive"."""),
            ("human", """
            Claims to verify:
            {claims}
            
            Sources checked:
            {sources}
            
            Original content:
            {content}
            """)
        ])
        
        chain = prompt | llm
        result = await chain.ainvoke({
            "claims": "\n".join(f"- {claim}" for claim in state.claims),
            "sources": "\n".join(f"- {source}" for source in state.sources_checked),
            "content": state.extracted_content[:1000]  # Truncate for context
        })
        
        # Parse verdict and reasoning
        verdict, reasoning = parse_analysis_result(result.content)
        
        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "processing_steps": state.processing_steps + ["fact_analysis"]
        }
        
    except Exception as e:
        return {
            "errors": state.errors + [f"Fact analysis failed: {str(e)}"],
            "processing_steps": state.processing_steps + ["fact_analysis_failed"]
        }

def parse_analysis_result(response: str) -> tuple[str, str]:
    """Parse verdict and reasoning from analysis response."""
    lines = response.strip().split('\n')
    verdict = "inconclusive"  # Default
    reasoning = response  # Default to full response
    
    # Look for verdict keywords
    response_lower = response.lower()
    if "verdict:" in response_lower:
        for line in lines:
            if "verdict:" in line.lower():
                verdict_part = line.split(":", 1)[1].strip().lower()
                if any(v in verdict_part for v in ["true", "false", "inconclusive", "satire"]):
                    for v in ["true", "false", "inconclusive", "satire"]:
                        if v in verdict_part:
                            verdict = v
                            break
                break
    else:
        # Fallback: look for verdict keywords in the response
        if "false" in response_lower or "incorrect" in response_lower:
            verdict = "false"
        elif "true" in response_lower or "accurate" in response_lower:
            verdict = "true"
        elif "satire" in response_lower or "satirical" in response_lower:
            verdict = "satire"
    
    return verdict, reasoning
```

### Confidence Scoring Node

```python
# langgraph/nodes/confidence_scoring.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

async def confidence_scoring_node(state: FactCheckingState) -> Dict[str, Any]:
    """
    Generate confidence score for the verdict.
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at assessing confidence in fact-checking results.
            
            Assign a confidence score from 0.0 to 1.0 based on:
            - Quality and reliability of sources found
            - Clarity and specificity of claims
            - Strength of evidence
            - Consistency across sources
            
            Guidelines:
            - 0.9-1.0: Very high confidence, strong evidence from multiple reliable sources
            - 0.7-0.8: High confidence, good evidence from reliable sources
            - 0.5-0.6: Moderate confidence, some evidence but limited sources
            - 0.3-0.4: Low confidence, weak or conflicting evidence
            - 0.0-0.2: Very low confidence, insufficient or unreliable evidence
            
            Return only a number between 0.0 and 1.0."""),
            ("human", """
            Verdict: {verdict}
            Claims analyzed: {num_claims}
            Sources found: {num_sources}
            Reasoning: {reasoning}
            
            What is your confidence score?""")
        ])
        
        chain = prompt | llm
        result = await chain.ainvoke({
            "verdict": state.verdict,
            "num_claims": len(state.claims),
            "num_sources": len(state.sources_checked),
            "reasoning": state.reasoning[:500]  # Truncate
        })
        
        # Parse confidence score
        confidence_score = parse_confidence_score(result.content)
        
        return {
            "confidence_score": confidence_score,
            "processing_steps": state.processing_steps + ["confidence_scoring"]
        }
        
    except Exception as e:
        return {
            "confidence_score": 0.5,  # Default moderate confidence
            "errors": state.errors + [f"Confidence scoring failed: {str(e)}"],
            "processing_steps": state.processing_steps + ["confidence_scoring_failed"]
        }

def parse_confidence_score(response: str) -> float:
    """Parse confidence score from response."""
    import re
    
    # Look for a number between 0 and 1
    numbers = re.findall(r'0?\.\d+|1\.0|0\.0', response)
    
    if numbers:
        try:
            score = float(numbers[0])
            return max(0.0, min(1.0, score))  # Clamp to 0-1 range
        except:
            pass
    
    # Fallback based on verdict
    verdict_scores = {
        "true": 0.7,
        "false": 0.7,
        "satire": 0.8,
        "inconclusive": 0.3
    }
    
    return verdict_scores.get(response.lower(), 0.5)
```

## 🔧 Service Integration

### AI Service (`app/services/ai_service.py`)

```python
from langgraph.workflows.fact_checking import create_fact_checking_workflow, FactCheckingState
from app.core.logging import get_logger
from typing import Dict, Any
import asyncio
import time

logger = get_logger(__name__)

class AIService:
    """Service for orchestrating AI workflows using LangGraph."""
    
    def __init__(self):
        self.fact_checking_workflow = create_fact_checking_workflow()
    
    async def verify_content(
        self,
        content: str,
        content_type: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Run fact-checking workflow on content.
        """
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state = FactCheckingState(
                content=content,
                content_type=content_type,
                session_id=session_id
            )
            
            # Run the workflow
            logger.info(f"Starting fact-checking workflow", extra={
                "session_id": session_id,
                "content_type": content_type
            })
            
            final_state = await self.fact_checking_workflow.ainvoke(initial_state)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Fact-checking workflow completed", extra={
                "session_id": session_id,
                "verdict": final_state.verdict,
                "confidence": final_state.confidence_score,
                "processing_time": processing_time
            })
            
            return {
                "verdict": final_state.verdict,
                "confidence_score": final_state.confidence_score,
                "reasoning": final_state.reasoning,
                "sources_checked": final_state.sources_checked,
                "claims": final_state.claims,
                "extracted_content": final_state.extracted_content,
                "ai_model_used": final_state.ai_model_used,
                "processing_time": processing_time,
                "processing_steps": final_state.processing_steps,
                "errors": final_state.errors
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Fact-checking workflow failed", extra={
                "session_id": session_id,
                "error": str(e),
                "processing_time": processing_time
            })
            
            return {
                "verdict": "inconclusive",
                "confidence_score": 0.0,
                "reasoning": f"Analysis failed due to technical error: {str(e)}",
                "sources_checked": [],
                "claims": [],
                "extracted_content": content if content_type == "text" else "",
                "ai_model_used": "unknown",
                "processing_time": processing_time,
                "processing_steps": ["error"],
                "errors": [str(e)]
            }
```

## 🧪 Testing LangGraph Workflows

### Workflow Testing (`tests/test_langgraph/test_fact_checking.py`)

```python
import pytest
from langgraph.workflows.fact_checking import (
    create_fact_checking_workflow,
    FactCheckingState
)

@pytest.mark.asyncio
async def test_fact_checking_workflow():
    """Test the complete fact-checking workflow."""
    workflow = create_fact_checking_workflow()
    
    initial_state = FactCheckingState(
        content="The Earth is flat and vaccines cause autism.",
        content_type="text",
        session_id="test_session"
    )
    
    final_state = await workflow.ainvoke(initial_state)
    
    # Assert basic structure
    assert final_state.verdict in ["true", "false", "inconclusive", "satire"]
    assert 0.0 <= final_state.confidence_score <= 1.0
    assert len(final_state.reasoning) > 0
    assert "content_extraction" in final_state.processing_steps

@pytest.mark.asyncio
async def test_url_content_extraction():
    """Test URL content extraction."""
    from langgraph.nodes.content_extraction import extract_url_content
    
    # Test with a reliable news URL
    content = await extract_url_content("https://www.bbc.com/news")
    
    assert len(content) > 100  # Should extract substantial content
    assert isinstance(content, str)

@pytest.mark.asyncio
async def test_claim_identification():
    """Test claim identification node."""
    from langgraph.nodes.claim_identification import claim_identification_node
    
    state = FactCheckingState(
        extracted_content="The President announced that unemployment is at 3.5%. The GDP grew by 2.1% last quarter.",
        content_type="text",
        session_id="test"
    )
    
    result = await claim_identification_node(state)
    
    assert len(result["claims"]) > 0
    assert any("unemployment" in claim.lower() for claim in result["claims"])
```

## 📊 Performance Optimization

### Caching and Rate Limiting

```python
# app/services/ai_cache.py
from functools import wraps
import hashlib
import json
import redis
from app.core.config import settings

redis_client = redis.from_url(settings.REDIS_URL)

def cache_ai_result(expiry_seconds: int = 3600):
    """Cache AI analysis results to avoid redundant processing."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = create_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache the result
            redis_client.setex(
                cache_key,
                expiry_seconds,
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator

def create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a unique cache key from function parameters."""
    key_data = {
        "function": func_name,
        "args": args,
        "kwargs": kwargs
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return f"ai_cache:{hashlib.md5(key_string.encode()).hexdigest()}"
```

## 📖 Best Practices

### 1. Error Handling
- Always handle LLM API failures gracefully
- Implement fallback responses
- Log errors with context for debugging

### 2. State Management
- Keep state objects lean and serializable
- Use Pydantic models for type safety
- Store intermediate results for debugging

### 3. Performance
- Cache expensive operations (URL extraction, source searches)
- Implement timeouts for external calls
- Use background tasks for non-critical operations

### 4. Quality Control
- Validate LLM outputs before using them
- Implement confidence thresholds
- Provide fallback analysis methods

## 🔗 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Anthropic API Documentation](https://docs.anthropic.com/) 