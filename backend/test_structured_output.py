"""
Test script for structured output implementation.

This script tests the structured output models to ensure they work correctly
and resolve the JSON parsing issues that were occurring in the workflow nodes.
"""

import asyncio
import json
from typing import Dict, Any

# Import the structured output models
from app.workflow.structured_output import (
    SentimentAnalysisOutput, ClaimsExtractionOutput, BiasAnalysis,
    QualityAssessmentOutput, FactAnalysisResult, ConfidenceAnalysis,
    SummaryOutput, SourceVerificationOutput
)

# Import LangChain components
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Import settings
from app.config import settings


async def test_sentiment_analysis():
    """Test sentiment analysis with structured output."""
    print("Testing Sentiment Analysis with Structured Output...")
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=SentimentAnalysisOutput)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a sentiment analysis expert. Analyze the emotional tone and sentiment of the given content.

Analyze:
- Overall sentiment (positive, negative, neutral, mixed)
- Emotional tone (joyful, angry, sad, fearful, surprised, disgusted, neutral)
- Sentiment intensity and confidence
- Key emotional triggers or themes

{format_instructions}"""),
        ("human", "Content to analyze:\n\n{content}\n\nAnalyze the sentiment and emotional tone:")
    ])
    
    # Create model (you'll need to set your API key)
    model = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        api_key=settings.ai.OPENAI_API_KEY
    )
    
    # Create chain
    chain = prompt | model | parser
    
    # Test content
    test_content = """
    The new technology breakthrough has brought incredible benefits to society. 
    People are excited about the possibilities and the positive impact it will have on our lives.
    This is truly a remarkable achievement that will change the world for the better.
    """
    
    try:
        # Get structured response
        result = await chain.ainvoke({
            "content": test_content,
            "format_instructions": parser.get_format_instructions()
        })
        
        print("✅ Sentiment Analysis Test PASSED")
        print(f"Result: {result.model_dump()}")
        return True
        
    except Exception as e:
        print(f"❌ Sentiment Analysis Test FAILED: {e}")
        return False


async def test_claims_extraction():
    """Test claims extraction with structured output."""
    print("\nTesting Claims Extraction with Structured Output...")
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=ClaimsExtractionOutput)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fact-checking assistant. Extract factual claims from the given content that can be verified through external sources.

Focus on:
- Specific statements about facts, events, or data
- Claims about people, places, or organizations
- Statistical or numerical claims
- Claims about cause and effect relationships
- Claims about dates, times, or sequences of events

Avoid:
- Opinions or subjective statements
- General background information
- Claims that are too vague to verify

{format_instructions}"""),
        ("human", "Content to analyze:\n\n{content}\n\nExtract factual claims:")
    ])
    
    # Create model
    model = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        api_key=settings.ai.OPENAI_API_KEY
    )
    
    # Create chain
    chain = prompt | model | parser
    
    # Test content
    test_content = """
    The population of New York City is 8.8 million people. 
    The city was founded in 1624 and has been the largest city in the United States since 1790.
    The average temperature in July is 77 degrees Fahrenheit.
    """
    
    try:
        # Get structured response
        result = await chain.ainvoke({
            "content": test_content,
            "format_instructions": parser.get_format_instructions()
        })
        
        print("✅ Claims Extraction Test PASSED")
        print(f"Result: {result.model_dump()}")
        return True
        
    except Exception as e:
        print(f"❌ Claims Extraction Test FAILED: {e}")
        return False


async def test_bias_detection():
    """Test bias detection with structured output."""
    print("\nTesting Bias Detection with Structured Output...")
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=BiasAnalysis)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a bias detection expert. Analyze the given content for potential biases and political leanings.

Analyze:
- Overall bias level (none, low, moderate, high)
- Types of bias (political, cultural, gender, racial, economic, etc.)
- Political leaning (left, center-left, center, center-right, right, none)
- Subjective language patterns
- Loaded terms or phrases
- One-sided arguments or perspectives

{format_instructions}"""),
        ("human", "Content to analyze:\n\n{content}\n\nDetect bias and political leanings:")
    ])
    
    # Create model
    model = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        api_key=settings.ai.OPENAI_API_KEY
    )
    
    # Create chain
    chain = prompt | model | parser
    
    # Test content
    test_content = """
    The government's new policy is clearly designed to benefit only the wealthy elite.
    This is just another example of how the system is rigged against ordinary people.
    Only a complete fool would support such a disastrous policy.
    """
    
    try:
        # Get structured response
        result = await chain.ainvoke({
            "content": test_content,
            "format_instructions": parser.get_format_instructions()
        })
        
        print("✅ Bias Detection Test PASSED")
        print(f"Result: {result.model_dump()}")
        return True
        
    except Exception as e:
        print(f"❌ Bias Detection Test FAILED: {e}")
        return False


async def test_quality_assessment():
    """Test quality assessment with structured output."""
    print("\nTesting Quality Assessment with Structured Output...")
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=QualityAssessmentOutput)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a content quality assessment expert. Analyze the quality, structure, and readability of the given content.

Analyze:
- Content structure and organization
- Readability level and complexity
- Writing quality and clarity
- Information density and completeness
- Logical flow and coherence
- Use of evidence and citations

{format_instructions}"""),
        ("human", "Content to analyze:\n\n{content}\n\nAssess content quality and structure:")
    ])
    
    # Create model
    model = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        api_key=settings.ai.OPENAI_API_KEY
    )
    
    # Create chain
    chain = prompt | model | parser
    
    # Test content
    test_content = """
    Climate change is a significant global challenge that requires immediate action.
    Scientific evidence shows that human activities are the primary driver of recent climate change.
    The Intergovernmental Panel on Climate Change (IPCC) has documented these findings extensively.
    """
    
    try:
        # Get structured response
        result = await chain.ainvoke({
            "content": test_content,
            "format_instructions": parser.get_format_instructions()
        })
        
        print("✅ Quality Assessment Test PASSED")
        print(f"Result: {result.model_dump()}")
        return True
        
    except Exception as e:
        print(f"❌ Quality Assessment Test FAILED: {e}")
        return False


async def test_fact_analysis():
    """Test fact analysis with structured output."""
    print("\nTesting Fact Analysis with Structured Output...")
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=FactAnalysisResult)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fact-checking expert. Analyze the factual accuracy of claims based on the provided verification sources.

For each claim, determine:
1. Whether the claim is supported by reliable sources
2. Whether there is contradicting evidence
3. The overall factual accuracy (true, false, inconclusive, satire)
4. Confidence level in your assessment
5. Key supporting or contradicting evidence

{format_instructions}"""),
        ("human", """Claims to analyze:
{claims}

Verification sources:
{sources}

Please analyze the factual accuracy of these claims.""")
    ])
    
    # Create model
    model = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        api_key=settings.ai.OPENAI_API_KEY
    )
    
    # Create chain
    chain = prompt | model | parser
    
    # Test data
    test_claims = "- The Earth is round\n- The population of New York City is 8.8 million"
    test_sources = """
Source: https://www.nasa.gov
Credibility: 0.95
Content: NASA has confirmed that the Earth is indeed round, as evidenced by satellite imagery and scientific measurements...

Source: https://www.census.gov
Credibility: 0.90
Content: According to the latest census data, New York City has a population of approximately 8.8 million people...
"""
    
    try:
        # Get structured response
        result = await chain.ainvoke({
            "claims": test_claims,
            "sources": test_sources,
            "format_instructions": parser.get_format_instructions()
        })
        
        print("✅ Fact Analysis Test PASSED")
        print(f"Result: {result.model_dump()}")
        return True
        
    except Exception as e:
        print(f"❌ Fact Analysis Test FAILED: {e}")
        return False


async def run_all_tests():
    """Run all structured output tests."""
    print("🚀 Starting Structured Output Tests...")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(await test_sentiment_analysis())
    results.append(await test_claims_extraction())
    results.append(await test_bias_detection())
    results.append(await test_quality_assessment())
    results.append(await test_fact_analysis())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Structured output implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_all_tests()) 