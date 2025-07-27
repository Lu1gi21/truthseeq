"""
Workflow Nodes for TruthSeeQ

This module defines individual LangGraph nodes for each step in the workflows.
Each node represents a specific task in the fact-checking, content analysis, or
source verification processes.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser

# Import local modules
from .state import (
    FactCheckState, ContentAnalysisState, SourceVerificationState,
    VerdictType, ConfidenceLevel, SourceType, Claim, Source, ScrapedContent
)
from .tools import (
    web_search, scrape_content, scrape_multiple_urls,
    check_domain_reliability, search_fact_checking_databases
)
from .structured_output import (
    SentimentAnalysisOutput, ClaimsExtractionOutput, BiasAnalysis,
    QualityAssessmentOutput, FactAnalysisResult, ConfidenceAnalysis,
    SummaryOutput, SourceVerificationOutput
)

# Ensure environment variables are loaded before importing settings
import os
from pathlib import Path

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

from ..config import settings

logger = logging.getLogger(__name__)


class BaseNode:
    """Base class for all workflow nodes."""
    
    def __init__(self, ai_model_name: Optional[str] = None):
        """
        Initialize base node.
        
        Args:
            ai_model_name: AI model to use for analysis
        """
        self.ai_model_name = ai_model_name or "gpt-4.1-mini-2025-04-14"
        self._model = None  # Lazy initialization
    
    @property
    def model(self):
        """Get AI model instance with lazy initialization."""
        if self._model is None:
            self._model = self._get_model()
        return self._model
    
    def _get_model(self):
        """Get AI model instance."""
        try:
            if "gpt" in self.ai_model_name:
                return ChatOpenAI(
                    model=self.ai_model_name,
                    temperature=0.1,
                    api_key=settings.ai.OPENAI_API_KEY
                )
            elif "claude" in self.ai_model_name:
                return ChatAnthropic(
                    model=self.ai_model_name,
                    temperature=0.1,
                    api_key=settings.ai.ANTHROPIC_API_KEY
                )
            else:
                # Default to GPT-4.1-mini-2025-04-14
                return ChatOpenAI(
                    model="gpt-4.1-mini-2025-04-14",
                    temperature=0.1,
                    api_key=settings.ai.OPENAI_API_KEY
                )
        except Exception as e:
            logger.error(f"Failed to initialize model {self.ai_model_name}: {e}")
            return None


class ContentExtractionNode(BaseNode):
    """
    Node for extracting and preprocessing content from URLs.
    
    This node handles:
    - Content scraping using advanced scraper
    - Content cleaning and preprocessing
    - Metadata extraction
    - Content validation
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from the original URL.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with scraped content
        """
        start_time = time.time()
        
        try:
            # Get URL from state - handle different state structures
            url = None
            if "original_url" in state:
                url = state["original_url"]
            elif "target_url" in state:
                url = state["target_url"]
            else:
                raise ValueError("No URL found in state")
            
            logger.info(f"Extracting content from: {url}")
            
            # Scrape content
            scraped_data = scrape_content.invoke({"url": url})
            
            if not scraped_data["success"]:
                logger.warning(f"Failed to scrape {url}: {scraped_data['error_message']}")
                return {
                    "scraped_content": ScrapedContent(
                        url=url,
                        title=None,
                        content="",
                        metadata={},
                        scraped_at=datetime.utcnow(),
                        method_used="failed",
                        success=False,
                        error_message=scraped_data["error_message"]
                    ),
                    "status": "content_extraction_failed"
                }
            
            # Create scraped content structure
            scraped_content = ScrapedContent(
                url=url,
                title=scraped_data.get("title"),
                content=scraped_data.get("content", ""),
                metadata=scraped_data.get("metadata", {}),
                scraped_at=datetime.utcnow(),
                method_used=scraped_data.get("method_used", "unknown"),
                success=True,
                error_message=None
            )
            
            # Return consistent state update
            return {
                "scraped_content": scraped_content,
                "status": "content_extracted",
                "processing_time": time.time() - start_time
            }
                
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {
                "scraped_content": ScrapedContent(
                    url=url if 'url' in locals() else "",
                    title=None,
                    content="",
                    metadata={},
                    scraped_at=datetime.utcnow(),
                    method_used="failed",
                    success=False,
                    error_message=str(e)
                ),
                "status": "content_extraction_failed",
                "error_message": str(e)
            }


class ClaimsExtractionNode(BaseNode):
    """
    Node for extracting factual claims from content.
    
    This node uses AI to identify and extract factual claims that can be
    verified through external sources.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract factual claims from content using structured output.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with extracted claims
        """
        if not self.model:
            logger.error("No AI model available for claims extraction")
            return {"extracted_claims": [], "status": "model_unavailable"}
        
        try:
            content = state.get("scraped_content", {}).get("content", "")
            if not content:
                logger.warning("No content available for claims extraction")
                return {"extracted_claims": [], "status": "no_content"}
            
            # Create structured output parser
            parser = PydanticOutputParser(pydantic_object=ClaimsExtractionOutput)
            
            # Create prompt for claims extraction with structured output
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
            
            # Create chain with structured output
            chain = prompt | self.model | parser
            
            # Get structured response
            extraction_result = await chain.ainvoke({
                "content": content[:4000],  # Limit content length
                "format_instructions": parser.get_format_instructions()
            })
            
            # Convert to Claim objects
            claims = []
            for claim_data in extraction_result.claims:
                claim = Claim(
                    id=claim_data.id,
                    text=claim_data.text,
                    confidence=claim_data.confidence,
                    category=claim_data.category,
                    entities=claim_data.entities,
                    sentiment=claim_data.sentiment
                )
                claims.append(claim)
            
            logger.info(f"Extracted {len(claims)} claims from content")
            return {
                "extracted_claims": claims,
                "status": "claims_extracted"
            }
                
        except Exception as e:
            logger.error(f"Claims extraction failed: {e}")
            return {"extracted_claims": [], "status": "extraction_failed", "error_message": str(e)}
                
        except Exception as e:
            logger.error(f"Claims extraction failed: {e}")
            return {"extracted_claims": [], "status": "claims_extraction_failed", "error_message": str(e)}


class SourceVerificationNode(BaseNode):
    """
    Node for verifying sources and finding supporting evidence.
    
    This node:
    - Analyzes domain reliability
    - Searches for supporting sources
    - Scrapes verification content
    - Assesses source credibility
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify sources and find supporting evidence.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with verification sources
        """
        try:
            claims = state.get("extracted_claims", [])
            original_url = state.get("original_url", "")
            
            if not claims:
                logger.info("No claims to verify")
                return {"verification_sources": [], "status": "no_claims"}
            
            # Check domain reliability of original source
            domain = urlparse(original_url).netloc
            domain_reliability = check_domain_reliability.invoke({"domain": domain})
            
            verification_sources = []
            search_queries = []
            
            # Generate search queries for each claim
            for claim in claims[:5]:  # Limit to top 5 claims
                query = self._generate_search_query(claim)
                search_queries.append(query)
                
                # Search for supporting sources
                search_results = web_search.invoke({"query": query, "count": 5})
                
                # Scrape top results
                urls_to_scrape = [result["url"] for result in search_results[:3]]
                scraped_results = scrape_multiple_urls.invoke({"urls": urls_to_scrape})
                
                # Process scraped results
                for i, scraped_result in enumerate(scraped_results):
                    if scraped_result["success"] and scraped_result["content"]:
                        source = Source(
                            url=scraped_result["url"],
                            title=scraped_result.get("title"),
                            content=scraped_result["content"],
                            source_type=self._classify_source_type(scraped_result["url"]),
                            credibility_score=self._calculate_credibility_score(scraped_result),
                            relevance_score=search_results[i]["relevance_score"] if i < len(search_results) else 0.5,
                            verification_status="scraped",
                            scraped_at=datetime.utcnow()
                        )
                        verification_sources.append(source)
            
            # Analyze overall source credibility
            source_analysis = {
                "original_domain_reliability": domain_reliability,
                "total_sources_found": len(verification_sources),
                "average_credibility": sum(s["credibility_score"] for s in verification_sources) / len(verification_sources) if verification_sources else 0.0,
                "reliable_sources_count": sum(1 for s in verification_sources if s["credibility_score"] > 0.7)
            }
            
            logger.info(f"Found {len(verification_sources)} verification sources")
            return {
                "verification_sources": verification_sources,
                "source_analysis": source_analysis,
                "search_queries": search_queries,
                "status": "sources_verified"
            }
            
        except Exception as e:
            logger.error(f"Source verification failed: {e}")
            return {
                "verification_sources": [],
                "source_analysis": {},
                "status": "source_verification_failed",
                "error_message": str(e)
            }
    
    def _generate_search_query(self, claim: Claim) -> str:
        """
        Generate search query for a claim.
        
        Args:
            claim: Claim to generate query for
            
        Returns:
            Search query string
        """
        # Extract key entities and create search query
        entities = claim.get("entities", [])
        claim_text = claim.get("text", "")
        
        if entities:
            # Use entities for more targeted search
            query = f'"{claim_text}" {" ".join(entities[:2])} fact check'
        else:
            # Use claim text directly
            query = f'"{claim_text}" fact check verification'
        
        return query
    
    def _classify_source_type(self, url: str) -> SourceType:
        """
        Classify the type of source based on URL.
        
        Args:
            url: Source URL
            
        Returns:
            SourceType classification
        """
        domain = urlparse(url).netloc.lower()
        
        if any(gov in domain for gov in [".gov", ".mil"]):
            return SourceType.GOVERNMENT_DOCUMENT
        elif any(edu in domain for edu in [".edu", ".ac."]):
            return SourceType.ACADEMIC_PAPER
        elif any(news in domain for news in ["reuters.com", "ap.org", "bbc.com", "npr.org"]):
            return SourceType.NEWS_ARTICLE
        elif any(social in domain for social in ["twitter.com", "facebook.com", "instagram.com"]):
            return SourceType.SOCIAL_MEDIA
        else:
            return SourceType.UNKNOWN
    
    def _calculate_credibility_score(self, scraped_result: Dict[str, Any]) -> float:
        """
        Calculate credibility score for a scraped source.
        
        Args:
            scraped_result: Scraped content result
            
        Returns:
            Credibility score (0-1)
        """
        url = scraped_result.get("url", "")
        domain = urlparse(url).netloc.lower()
        
        # Base score from domain reliability
        domain_reliability = check_domain_reliability.invoke({"domain": domain})
        base_score = domain_reliability.get("reliability_score", 0.5)
        
        # Adjust based on content quality
        content = scraped_result.get("content", "")
        content_length = len(content)
        
        if content_length > 1000:
            content_quality = 0.8
        elif content_length > 500:
            content_quality = 0.6
        else:
            content_quality = 0.3
        
        # Combine scores
        credibility_score = (base_score * 0.7) + (content_quality * 0.3)
        return min(max(credibility_score, 0.0), 1.0)


class FactAnalysisNode(BaseNode):
    """
    Node for analyzing factual accuracy of claims.
    
    This node uses AI to compare claims against verification sources
    and determine factual accuracy.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze factual accuracy of claims using structured output.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with fact analysis results
        """
        if not self.model:
            logger.error("No AI model available for fact analysis")
            return {"fact_analysis": {}, "status": "model_unavailable"}
        
        try:
            claims = state.get("extracted_claims", [])
            verification_sources = state.get("verification_sources", [])
            
            if not claims:
                logger.info("No claims to analyze")
                return {"fact_analysis": {}, "status": "no_claims"}
            
            if not verification_sources:
                logger.warning("No verification sources available")
                return {"fact_analysis": {}, "status": "no_sources"}
            
            # Prepare context for AI analysis
            claims_text = "\n".join([f"- {claim['text']}" for claim in claims])
            
            sources_text = ""
            for source in verification_sources[:5]:  # Limit to top 5 sources
                sources_text += f"\nSource: {source['url']}\nCredibility: {source['credibility_score']:.2f}\nContent: {source['content'][:500]}...\n"
            
            # Create structured output parser
            parser = PydanticOutputParser(pydantic_object=FactAnalysisResult)
            
            # Create prompt for fact analysis with structured output
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a fact-checking expert. Analyze the factual accuracy of claims based on the provided verification sources.

For each claim, determine:
1. Whether the claim is supported by reliable sources
2. Whether there is contradicting evidence
3. The overall factual accuracy (true, false, inconclusive, satire)
4. Confidence level in your assessment
5. Key supporting or contradicting evidence

{format_instructions}"""),
                ("human", f"""Claims to analyze:
{claims_text}

Verification sources:
{sources_text}

Please analyze the factual accuracy of these claims.""")
            ])
            
            # Create chain with structured output
            chain = prompt | self.model | parser
            
            # Get structured response
            fact_result = await chain.ainvoke({
                "format_instructions": parser.get_format_instructions()
            })
            
            # Convert to dict for state update
            analysis_data = fact_result.model_dump()
            
            # Update claims with analysis results
            for claim in claims:
                claim.update({
                    "accuracy": analysis_data.get("verdict"),
                    "confidence": analysis_data.get("confidence_score", 0.5),
                    "supporting_evidence": analysis_data.get("key_evidence", []),
                    "contradicting_evidence": []
                })
            
            fact_analysis = {
                "verdict": analysis_data.get("verdict"),
                "confidence_score": analysis_data.get("confidence_score"),
                "reasoning": analysis_data.get("reasoning"),
                "key_evidence": analysis_data.get("key_evidence", []),
                "sources_checked": analysis_data.get("sources_checked", []),
                "ai_model_used": analysis_data.get("ai_model_used"),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("Fact analysis completed")
            return {
                "extracted_claims": claims,
                "fact_analysis": fact_analysis,
                "status": "facts_analyzed"
            }
                
        except Exception as e:
            logger.error(f"Fact analysis failed: {e}")
            return {"fact_analysis": {}, "status": "fact_analysis_failed", "error_message": str(e)}


class ConfidenceScoringNode(BaseNode):
    """
    Node for calculating confidence scores and generating verdicts.
    
    This node combines all analysis results to produce final confidence
    scores and verdicts.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate confidence scores and generate final verdict.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with confidence scores and verdict
        """
        try:
            claims = state.get("extracted_claims", [])
            source_analysis = state.get("source_analysis", {})
            fact_analysis = state.get("fact_analysis", {})
            
            if not claims:
                return {
                    "confidence_score": 0.0,
                    "confidence_level": ConfidenceLevel.VERY_LOW,
                    "verdict": VerdictType.UNVERIFIABLE,
                    "reasoning": "No claims to analyze",
                    "status": "no_claims"
                }
            
            # Calculate confidence score based on multiple factors
            confidence_score = self._calculate_confidence_score(
                claims, source_analysis, fact_analysis
            )
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(confidence_score)
            
            # Generate verdict
            verdict = self._generate_verdict(claims, confidence_score)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(claims, source_analysis, fact_analysis, verdict)
            
            # Collect supporting and contradicting evidence
            supporting_evidence = []
            contradicting_evidence = []
            
            for claim in claims:
                supporting_evidence.extend(claim.get("supporting_evidence", []))
                contradicting_evidence.extend(claim.get("contradicting_evidence", []))
            
            logger.info(f"Generated verdict: {verdict} with confidence: {confidence_score:.2f}")
            
            return {
                "confidence_score": confidence_score,
                "confidence_level": confidence_level,
                "verdict": verdict,
                "reasoning": reasoning,
                "supporting_evidence": supporting_evidence,
                "contradicting_evidence": contradicting_evidence,
                "status": "verdict_generated"
            }
            
        except Exception as e:
            logger.error(f"Confidence scoring failed: {e}")
            return {
                "confidence_score": 0.0,
                "confidence_level": ConfidenceLevel.VERY_LOW,
                "verdict": VerdictType.UNVERIFIABLE,
                "reasoning": f"Error in analysis: {str(e)}",
                "status": "confidence_scoring_failed",
                "error_message": str(e)
            }
    
    def _calculate_confidence_score(
        self, 
        claims: List[Dict], 
        source_analysis: Dict, 
        fact_analysis: Dict
    ) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            claims: List of analyzed claims
            source_analysis: Source analysis results
            fact_analysis: Fact analysis results
            
        Returns:
            Confidence score (0-1)
        """
        # Factor weights
        weights = {
            "source_credibility": 0.25,
            "claim_confidence": 0.30,
            "evidence_strength": 0.25,
            "analysis_consistency": 0.20
        }
        
        # Source credibility factor
        source_credibility = source_analysis.get("average_credibility", 0.5)
        reliable_sources_ratio = (
            source_analysis.get("reliable_sources_count", 0) / 
            max(source_analysis.get("total_sources_found", 1), 1)
        )
        source_factor = (source_credibility + reliable_sources_ratio) / 2
        
        # Claim confidence factor
        if claims:
            claim_confidences = [claim.get("confidence", 0.5) for claim in claims]
            claim_factor = sum(claim_confidences) / len(claim_confidences)
        else:
            claim_factor = 0.5
        
        # Evidence strength factor
        total_evidence = 0
        for claim in claims:
            total_evidence += len(claim.get("supporting_evidence", []))
            total_evidence += len(claim.get("contradicting_evidence", []))
        
        evidence_factor = min(total_evidence / max(len(claims), 1) / 5, 1.0)
        
        # Analysis consistency factor
        overall_assessment = fact_analysis.get("overall_assessment", {})
        consistency_factor = overall_assessment.get("average_accuracy", 0.5)
        
        # Calculate weighted score
        confidence_score = (
            source_factor * weights["source_credibility"] +
            claim_factor * weights["claim_confidence"] +
            evidence_factor * weights["evidence_strength"] +
            consistency_factor * weights["analysis_consistency"]
        )
        
        return min(max(confidence_score, 0.0), 1.0)
    
    def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """
        Get confidence level from score.
        
        Args:
            confidence_score: Confidence score (0-1)
            
        Returns:
            ConfidenceLevel enum value
        """
        if confidence_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            return ConfidenceLevel.MODERATE
        elif confidence_score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_verdict(self, claims: List[Dict], confidence_score: float) -> VerdictType:
        """
        Generate final verdict based on claims and confidence.
        
        Args:
            claims: List of analyzed claims
            confidence_score: Overall confidence score
            
        Returns:
            VerdictType enum value
        """
        if not claims:
            return VerdictType.UNVERIFIABLE
        
        # Count verdicts by type
        verdict_counts = {}
        for claim in claims:
            accuracy = claim.get("accuracy")
            if accuracy:
                verdict_counts[accuracy] = verdict_counts.get(accuracy, 0) + 1
        
        if not verdict_counts:
            return VerdictType.UNVERIFIABLE
        
        # Get most common verdict
        most_common_verdict = max(verdict_counts.items(), key=lambda x: x[1])[0]
        
        # Map to VerdictType
        verdict_mapping = {
            "true": VerdictType.TRUE,
            "mostly_true": VerdictType.MOSTLY_TRUE,
            "partially_true": VerdictType.PARTIALLY_TRUE,
            "false": VerdictType.FALSE,
            "mostly_false": VerdictType.MOSTLY_FALSE,
            "misleading": VerdictType.MISLEADING,
            "unverifiable": VerdictType.UNVERIFIABLE
        }
        
        return verdict_mapping.get(most_common_verdict, VerdictType.UNVERIFIABLE)
    
    def _generate_reasoning(
        self, 
        claims: List[Dict], 
        source_analysis: Dict, 
        fact_analysis: Dict, 
        verdict: VerdictType
    ) -> str:
        """
        Generate human-readable reasoning for the verdict.
        
        Args:
            claims: List of analyzed claims
            source_analysis: Source analysis results
            fact_analysis: Fact analysis results
            verdict: Final verdict
            
        Returns:
            Reasoning string
        """
        reasoning_parts = []
        
        # Overall assessment
        reasoning_parts.append(f"Analysis of {len(claims)} claims resulted in a verdict of: {verdict.value}")
        
        # Source information
        total_sources = source_analysis.get("total_sources_found", 0)
        reliable_sources = source_analysis.get("reliable_sources_count", 0)
        if total_sources > 0:
            reasoning_parts.append(f"Analysis based on {total_sources} sources ({reliable_sources} reliable)")
        
        # Key findings
        overall_assessment = fact_analysis.get("overall_assessment", {})
        key_findings = overall_assessment.get("key_findings", [])
        if key_findings:
            reasoning_parts.append("Key findings:")
            reasoning_parts.extend([f"- {finding}" for finding in key_findings[:3]])
        
        # Claim summary
        if claims:
            true_claims = sum(1 for c in claims if c.get("accuracy") in ["true", "mostly_true"])
            false_claims = sum(1 for c in claims if c.get("accuracy") in ["false", "mostly_false"])
            reasoning_parts.append(f"Claims breakdown: {true_claims} verified, {false_claims} contradicted")
        
        return "\n".join(reasoning_parts)


class SentimentAnalysisNode(BaseNode):
    """
    Node for analyzing sentiment and emotional tone of content.
    
    This node uses AI to analyze the emotional tone, sentiment polarity,
    and emotional characteristics of the content.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment and emotional tone using structured output.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with sentiment analysis results
        """
        if not self.model:
            logger.error("No AI model available for sentiment analysis")
            return {"sentiment_analysis": {}, "status": "model_unavailable"}
        
        try:
            content = state.get("scraped_content", {}).get("content", "")
            if not content:
                logger.warning("No content available for sentiment analysis")
                return {"sentiment_analysis": {}, "status": "no_content"}
            
            # Create structured output parser
            parser = PydanticOutputParser(pydantic_object=SentimentAnalysisOutput)
            
            # Create prompt for sentiment analysis with structured output
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
            
            # Create chain with structured output
            chain = prompt | self.model | parser
            
            # Get structured response
            analysis_result = await chain.ainvoke({
                "content": content[:3000],  # Limit content length
                "format_instructions": parser.get_format_instructions()
            })
            
            # Convert to dict for state update
            analysis_data = analysis_result.model_dump()
            
            return {
                "sentiment_analysis": analysis_data,
                "emotional_tone": analysis_data.get("emotional_tone", "neutral"),
                "sentiment_score": analysis_data.get("sentiment_score", 0.0),
                "status": "sentiment_analyzed"
            }
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            # Return default values on any failure
            return {
                "sentiment_analysis": {
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "emotional_tone": "neutral",
                    "confidence": 0.0,
                    "key_emotions": [],
                    "reasoning": "Analysis failed due to error"
                },
                "emotional_tone": "neutral",
                "sentiment_score": 0.0,
                "status": "sentiment_analysis_failed",
                "error_message": str(e)
            }


class BiasDetectionNode(BaseNode):
    """
    Node for detecting bias and political leanings in content.
    
    This node uses AI to identify potential biases, political leanings,
    and subjective language patterns in the content.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect bias and political leanings using structured output.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with bias analysis results
        """
        if not self.model:
            logger.error("No AI model available for bias detection")
            return {"bias_analysis": {}, "status": "model_unavailable"}
        
        try:
            content = state.get("scraped_content", {}).get("content", "")
            if not content:
                logger.warning("No content available for bias detection")
                return {"bias_analysis": {}, "status": "no_content"}
            
            # Create structured output parser
            parser = PydanticOutputParser(pydantic_object=BiasAnalysis)
            
            # Create prompt for bias detection with structured output
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
            
            # Create chain with structured output
            chain = prompt | self.model | parser
            
            # Get structured response
            bias_result = await chain.ainvoke({
                "content": content[:3000],  # Limit content length
                "format_instructions": parser.get_format_instructions()
            })
            
            # Convert to dict for state update
            analysis_data = bias_result.model_dump()
            
            return {
                "bias_analysis": analysis_data,
                "bias_level": analysis_data.get("bias_level", "none"),
                "bias_types": analysis_data.get("bias_types", []),
                "political_leaning": analysis_data.get("political_leaning", None),
                "status": "bias_detected"
            }
                
        except Exception as e:
            logger.error(f"Bias detection failed: {e}")
            return {
                "bias_analysis": {
                    "bias_level": "none",
                    "bias_types": [],
                    "political_leaning": None,
                    "bias_score": 0.0,
                    "confidence": 0.0,
                    "reasoning": "Analysis failed due to error"
                },
                "bias_level": "none",
                "bias_types": [],
                "political_leaning": None,
                "status": "bias_detection_failed",
                "error_message": str(e)
            }


class QualityAssessmentNode(BaseNode):
    """
    Node for assessing content quality and structure.
    
    This node analyzes content quality, readability, structure,
    and provides quality metrics.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess content quality and structure using structured output.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with quality assessment results
        """
        if not self.model:
            logger.error("No AI model available for quality assessment")
            return {"structure_analysis": {}, "status": "model_unavailable"}
        
        try:
            content = state.get("scraped_content", {}).get("content", "")
            if not content:
                logger.warning("No content available for quality assessment")
                return {"structure_analysis": {}, "status": "no_content"}
            
            # Create structured output parser
            parser = PydanticOutputParser(pydantic_object=QualityAssessmentOutput)
            
            # Create prompt for quality assessment with structured output
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
            
            # Create chain with structured output
            chain = prompt | self.model | parser
            
            # Get structured response
            quality_result = await chain.ainvoke({
                "content": content[:3000],  # Limit content length
                "format_instructions": parser.get_format_instructions()
            })
            
            # Convert to dict for state update
            analysis_data = quality_result.model_dump()
            
            return {
                "structure_analysis": analysis_data,
                "content_quality_score": analysis_data.get("metrics", {}).get("content_quality_score", 0.0),
                "readability_score": analysis_data.get("metrics", {}).get("readability_score", 0.0),
                "status": "quality_assessed"
            }
                
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                "structure_analysis": {
                    "metrics": {
                        "content_quality_score": 0.0,
                        "readability_score": 0.0,
                        "structure_quality": 0.0,
                        "factual_accuracy": 0.0,
                        "source_credibility": 0.0
                    },
                    "content_category": "unknown",
                    "content_subcategory": "unknown",
                    "topics": [],
                    "entities": [],
                    "recommendations": [],
                    "confidence": 0.0
                },
                "content_quality_score": 0.0,
                "readability_score": 0.0,
                "status": "quality_assessment_failed",
                "error_message": str(e)
            }


class CredibilityAnalysisNode(BaseNode):
    """
    Node for analyzing source credibility and domain reputation.
    
    This node assesses the credibility of the content source,
    domain reputation, and trust indicators.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze source credibility and domain reputation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with credibility analysis results
        """
        try:
            original_url = state.get("original_url", "")
            if not original_url:
                logger.warning("No URL available for credibility analysis")
                return {"domain_reputation": {}, "status": "no_url"}
            
            domain = urlparse(original_url).netloc
            
            # Check domain reliability
            domain_reliability = check_domain_reliability.invoke({"domain": domain})
            
            # Basic SSL check (would need proper implementation)
            ssl_valid = True  # Placeholder
            
            # Basic domain age check (would need proper implementation)
            domain_age = None  # Placeholder
            
            credibility_score = domain_reliability.get("reliability_score", 0.5)
            
            return {
                "domain_reputation": domain_reliability,
                "source_credibility_score": credibility_score,
                "ssl_valid": ssl_valid,
                "domain_age": domain_age,
                "status": "credibility_analyzed"
            }
                
        except Exception as e:
            logger.error(f"Credibility analysis failed: {e}")
            return {
                "domain_reputation": {},
                "source_credibility_score": 0.0,
                "ssl_valid": False,
                "domain_age": None,
                "status": "credibility_analysis_failed",
                "error_message": str(e)
            }


class SummaryGenerationNode(BaseNode):
    """
    Node for generating comprehensive analysis summary.
    
    This node combines all analysis results to generate
    a comprehensive summary with key insights and recommendations.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis summary.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with analysis summary
        """
        if not self.model:
            logger.error("No AI model available for summary generation")
            return {"analysis_summary": "", "status": "model_unavailable"}
        
        try:
            # Collect all analysis results
            sentiment_analysis = state.get("sentiment_analysis", {})
            bias_analysis = state.get("bias_analysis", {})
            structure_analysis = state.get("structure_analysis", {})
            domain_reputation = state.get("domain_reputation", {})
            
            # Create context for summary generation
            context = f"""
Sentiment Analysis: {sentiment_analysis.get('sentiment', 'neutral')} (score: {sentiment_analysis.get('sentiment_score', 0.0)})
Bias Level: {bias_analysis.get('bias_level', 'none')}
Content Quality: {state.get('content_quality_score', 0.0)}
Source Credibility: {state.get('source_credibility_score', 0.0)}
            """.strip()
            
            # Create prompt for summary generation
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a content analysis expert. Generate a comprehensive summary of the content analysis results.

Provide:
- Overall assessment of the content
- Key insights about quality, bias, and credibility
- Recommendations for content consumers
- Summary of main findings

Keep the summary concise but informative (2-3 paragraphs)."""),
                ("human", "Analysis Results:\n\n{context}\n\nGenerate a comprehensive summary:")
            ])
            
            # Get AI response
            response = await self.model.ainvoke(
                prompt.format_messages(context=context)
            )
            
            return {
                "analysis_summary": response.content,
                "key_insights": [
                    f"Sentiment: {sentiment_analysis.get('sentiment', 'neutral')}",
                    f"Bias Level: {bias_analysis.get('bias_level', 'none')}",
                    f"Quality Score: {state.get('content_quality_score', 0.0):.2f}",
                    f"Credibility: {state.get('source_credibility_score', 0.0):.2f}"
                ],
                "recommendations": [
                    "Consider multiple sources for verification",
                    "Evaluate bias and perspective",
                    "Check source credibility"
                ],
                "status": "summary_generated"
            }
                
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                "analysis_summary": "Analysis summary generation failed.",
                "key_insights": [],
                "recommendations": [],
                "status": "summary_generation_failed",
                "error_message": str(e)
            }


class DomainAnalysisNode(BaseNode):
    """
    Node for analyzing domain characteristics and reputation.
    
    This node analyzes the domain of the target URL to assess
    its credibility, age, and reputation.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze domain characteristics.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with domain analysis results
        """
        try:
            target_url = state.get("target_url", "")
            if not target_url:
                logger.warning("No URL available for domain analysis")
                return {"domain_analysis": {}, "status": "no_url"}
            
            domain = urlparse(target_url).netloc
            
            # Check domain reliability
            domain_reliability = check_domain_reliability.invoke({"domain": domain})
            
            # Basic domain analysis (would need proper implementation)
            domain_age = None  # Placeholder for domain age lookup
            ssl_valid = True  # Placeholder for SSL validation
            
            return {
                "domain_analysis": {
                    "domain": domain,
                    "reliability": domain_reliability,
                    "age": domain_age,
                    "ssl_valid": ssl_valid
                },
                "domain_age": domain_age,
                "ssl_valid": ssl_valid,
                "status": "domain_analyzed"
            }
                
        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            return {
                "domain_analysis": {},
                "domain_age": None,
                "ssl_valid": False,
                "status": "domain_analysis_failed",
                "error_message": str(e)
            }


class ReputationCheckNode(BaseNode):
    """
    Node for checking domain reputation and trust indicators.
    
    This node checks various reputation databases and trust
    indicators to assess the domain's credibility.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check domain reputation and trust indicators.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with reputation analysis results
        """
        try:
            domain_analysis = state.get("domain_analysis", {})
            domain = domain_analysis.get("domain", "")
            
            if not domain:
                logger.warning("No domain available for reputation check")
                return {"reputation_analysis": {}, "status": "no_domain"}
            
            # Basic reputation check (would need proper implementation)
            reputation_score = 0.7  # Placeholder
            trust_indicators = []  # Placeholder
            red_flags = []  # Placeholder
            
            return {
                "reputation_analysis": {
                    "reputation_score": reputation_score,
                    "trust_indicators": trust_indicators,
                    "red_flags": red_flags
                },
                "reputation_score": reputation_score,
                "trust_indicators": trust_indicators,
                "red_flags": red_flags,
                "status": "reputation_checked"
            }
                
        except Exception as e:
            logger.error(f"Reputation check failed: {e}")
            return {
                "reputation_analysis": {},
                "reputation_score": 0.0,
                "trust_indicators": [],
                "red_flags": [],
                "status": "reputation_check_failed",
                "error_message": str(e)
            }


class FactCheckingLookupNode(BaseNode):
    """
    Node for looking up domain in fact-checking databases.
    
    This node searches various fact-checking databases to see
    if the domain has been previously fact-checked.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Look up domain in fact-checking databases.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with fact-checking lookup results
        """
        try:
            target_url = state.get("target_url", "")
            if not target_url:
                logger.warning("No URL available for fact-checking lookup")
                return {"fact_checking_results": {}, "status": "no_url"}
            
            # Search fact-checking databases
            fact_check_results = search_fact_checking_databases.invoke({"query": target_url})
            
            # Handle different return types from the tool
            if isinstance(fact_check_results, list):
                # Tool returned a list directly
                known_fact_checks = fact_check_results
                fact_check_results = {"results": fact_check_results}
            elif isinstance(fact_check_results, dict):
                # Tool returned a dict with results
                known_fact_checks = fact_check_results.get("results", [])
            else:
                # Unexpected return type
                logger.warning(f"Unexpected return type from search_fact_checking_databases: {type(fact_check_results)}")
                known_fact_checks = []
                fact_check_results = {"results": []}
            
            return {
                "fact_checking_results": fact_check_results,
                "known_fact_checks": known_fact_checks,
                "status": "fact_checking_lookup_completed"
            }
                
        except Exception as e:
            logger.error(f"Fact-checking lookup failed: {e}")
            return {
                "fact_checking_results": {},
                "known_fact_checks": [],
                "status": "fact_checking_lookup_failed",
                "error_message": str(e)
            }


class CrossReferenceNode(BaseNode):
    """
    Node for cross-referencing with other sources.
    
    This node searches for other sources that discuss the same
    content or topic to verify consistency.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-reference with other sources.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with cross-reference results
        """
        try:
            target_url = state.get("target_url", "")
            if not target_url:
                logger.warning("No URL available for cross-referencing")
                return {"cross_reference_sources": [], "status": "no_url"}
            
            # Search for related sources
            search_query = f"site:{target_url} fact check verification"
            search_results = web_search.invoke({"query": search_query, "count": 5})
            
            cross_reference_sources = []
            if isinstance(search_results, list):
                for result in search_results:
                    if isinstance(result, dict) and result.get("url") != target_url:
                        cross_reference_sources.append({
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", ""),
                            "relevance_score": 0.7  # Placeholder
                        })
            
            return {
                "cross_reference_sources": cross_reference_sources,
                "status": "cross_reference_completed"
            }
                
        except Exception as e:
            logger.error(f"Cross-reference failed: {e}")
            return {
                "cross_reference_sources": [],
                "status": "cross_reference_failed",
                "error_message": str(e)
            }


class VerificationResultNode(BaseNode):
    """
    Node for generating final verification result.
    
    This node combines all verification analysis to generate
    a comprehensive verification result with confidence score.
    """
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final verification result.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with verification result
        """
        try:
            # Collect all verification results
            domain_analysis = state.get("domain_analysis", {})
            reputation_analysis = state.get("reputation_analysis", {})
            fact_checking_results = state.get("fact_checking_results", {})
            cross_reference_sources = state.get("cross_reference_sources", [])
            
            # Calculate verification score
            domain_score = domain_analysis.get("reliability", {}).get("reliability_score", 0.5)
            reputation_score = reputation_analysis.get("reputation_score", 0.5)
            
            # Weight the scores
            verification_score = (domain_score * 0.4) + (reputation_score * 0.4) + (0.2 if cross_reference_sources else 0.0)
            verification_score = min(max(verification_score, 0.0), 1.0)
            
            # Determine verification status
            if verification_score >= 0.8:
                verification_status = "verified"
            elif verification_score >= 0.6:
                verification_status = "mostly_verified"
            elif verification_score >= 0.4:
                verification_status = "partially_verified"
            else:
                verification_status = "unverified"
            
            # Determine confidence level
            if verification_score >= 0.8:
                confidence_level = "high"
            elif verification_score >= 0.6:
                confidence_level = "moderate"
            else:
                confidence_level = "low"
            
            return {
                "verification_score": verification_score,
                "verification_status": verification_status,
                "confidence_level": confidence_level,
                "verification_summary": f"Domain verification completed with {verification_score:.2f} score",
                "recommendations": [
                    "Consider multiple sources for verification",
                    "Check source credibility",
                    "Evaluate domain reputation"
                ],
                "status": "verification_completed"
            }
                
        except Exception as e:
            logger.error(f"Verification result generation failed: {e}")
            return {
                "verification_score": 0.0,
                "verification_status": "unknown",
                "confidence_level": "very_low",
                "verification_summary": "Verification failed",
                "recommendations": [],
                "status": "verification_failed",
                "error_message": str(e)
            }


# Node registry for easy access
WORKFLOW_NODES = {
    "content_extraction": ContentExtractionNode,
    "claims_extraction": ClaimsExtractionNode,
    "source_verification": SourceVerificationNode,
    "fact_analysis": FactAnalysisNode,
    "confidence_scoring": ConfidenceScoringNode,
    # Content Analysis nodes
    "sentiment_analysis": SentimentAnalysisNode,
    "bias_detection": BiasDetectionNode,
    "quality_assessment": QualityAssessmentNode,
    "credibility_analysis": CredibilityAnalysisNode,
    "summary_generation": SummaryGenerationNode,
    # Source Verification nodes
    "domain_analysis": DomainAnalysisNode,
    "reputation_check": ReputationCheckNode,
    "fact_checking_lookup": FactCheckingLookupNode,
    "cross_reference": CrossReferenceNode,
    "verification_result": VerificationResultNode,
}


def get_node_by_name(name: str, ai_model_name: Optional[str] = None):
    """
    Get a workflow node by name.
    
    Args:
        name: Node name
        ai_model_name: Optional AI model name
        
    Returns:
        Node instance or None if not found
    """
    node_class = WORKFLOW_NODES.get(name)
    if node_class:
        return node_class(ai_model_name)
    return None


def get_available_nodes() -> List[str]:
    """
    Get list of available node names.
    
    Returns:
        List of node names
    """
    return list(WORKFLOW_NODES.keys()) 