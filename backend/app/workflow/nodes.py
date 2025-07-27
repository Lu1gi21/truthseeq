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

# Import local modules
from .state import (
    FactCheckState, ContentAnalysisState, SourceVerificationState,
    VerdictType, ConfidenceLevel, SourceType, Claim, Source, ScrapedContent
)
from .tools import (
    web_search, scrape_content, scrape_multiple_urls,
    check_domain_reliability, search_fact_checking_databases
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
        self.ai_model_name = ai_model_name or "gpt-4"
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
                # Default to GPT-4
                return ChatOpenAI(
                    model="gpt-4",
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
            scraped_data = scrape_content(url)
            
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
        Extract factual claims from content.
        
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
            
            # Create prompt for claims extraction with better JSON formatting
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

IMPORTANT: Return ONLY a valid JSON array. Do not include any additional text, explanations, or markdown formatting.

Example response format:
[
  {
    "id": "claim_1",
    "text": "The Jewish people number 15.7 million worldwide",
    "confidence": 0.8,
    "category": "statistic",
    "entities": ["Jewish people", "worldwide"],
    "sentiment": "neutral"
  }
]"""),
                ("human", "Content to analyze:\n\n{content}\n\nExtract factual claims and return as JSON array:")
            ])
            
            # Get AI response
            response = await self.model.ainvoke(
                prompt.format_messages(content=content[:4000])  # Limit content length
            )
            
            # Parse claims from response with better error handling
            try:
                # Clean the response content
                response_text = response.content.strip()
                
                # Try to extract JSON if it's wrapped in markdown
                if response_text.startswith("```json"):
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif response_text.startswith("```"):
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                # Parse JSON
                claims_data = json.loads(response_text)
                
                if not isinstance(claims_data, list):
                    logger.warning("Claims data is not a list, converting to list")
                    claims_data = [claims_data] if isinstance(claims_data, dict) else []
                
                claims = []
                
                for claim_data in claims_data:
                    if isinstance(claim_data, dict):
                        claim = Claim(
                            id=claim_data.get("id", str(uuid.uuid4())),
                            text=claim_data.get("text", ""),
                            confidence=float(claim_data.get("confidence", 0.5)),
                            category=claim_data.get("category", "other"),
                            entities=claim_data.get("entities", []),
                            sentiment=claim_data.get("sentiment", "neutral")
                        )
                        claims.append(claim)
                
                logger.info(f"Extracted {len(claims)} claims from content")
                return {
                    "extracted_claims": claims,
                    "status": "claims_extracted"
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse claims JSON: {e}")
                logger.error(f"Response content: {response.content[:200]}...")
                
                # Try to extract claims manually from the response
                try:
                    # Simple fallback: look for quoted text that might be claims
                    import re
                    potential_claims = re.findall(r'"([^"]{20,})"', response.content)
                    
                    claims = []
                    for i, claim_text in enumerate(potential_claims[:5]):  # Limit to 5 claims
                        claim = Claim(
                            id=f"fallback_claim_{i}",
                            text=claim_text,
                            confidence=0.3,  # Lower confidence for fallback
                            category="other",
                            entities=[],
                            sentiment="neutral"
                        )
                        claims.append(claim)
                    
                    if claims:
                        logger.info(f"Extracted {len(claims)} claims using fallback method")
                        return {
                            "extracted_claims": claims,
                            "status": "claims_extracted_fallback"
                        }
                except Exception as fallback_error:
                    logger.error(f"Fallback claims extraction also failed: {fallback_error}")
                
                return {"extracted_claims": [], "status": "parsing_failed"}
                
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
            domain_reliability = check_domain_reliability(domain)
            
            verification_sources = []
            search_queries = []
            
            # Generate search queries for each claim
            for claim in claims[:5]:  # Limit to top 5 claims
                query = self._generate_search_query(claim)
                search_queries.append(query)
                
                # Search for supporting sources
                search_results = web_search(query, count=5)
                
                # Scrape top results
                urls_to_scrape = [result["url"] for result in search_results[:3]]
                scraped_results = scrape_multiple_urls(urls_to_scrape)
                
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
        domain_reliability = check_domain_reliability(domain)
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
        Analyze factual accuracy of claims.
        
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
            
            # Create prompt for fact analysis
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a fact-checking expert. Analyze the factual accuracy of claims based on the provided verification sources.

For each claim, determine:
1. Whether the claim is supported by reliable sources
2. Whether there is contradicting evidence
3. The overall factual accuracy (true, mostly true, partially true, false, misleading, or unverifiable)
4. Confidence level in your assessment
5. Key supporting or contradicting evidence

Return your analysis as a JSON object with the following structure:
{
  "claims_analysis": [
    {
      "claim_id": "id",
      "accuracy": "true|mostly_true|partially_true|false|misleading|unverifiable",
      "confidence": 0.8,
      "supporting_evidence": ["evidence1", "evidence2"],
      "contradicting_evidence": ["evidence1", "evidence2"],
      "reasoning": "detailed explanation"
    }
  ],
  "overall_assessment": {
    "average_accuracy": 0.7,
    "most_common_verdict": "mostly_true",
    "confidence_level": "high|moderate|low",
    "key_findings": ["finding1", "finding2"]
  }
}"""),
                ("human", f"""Claims to analyze:
{claims_text}

Verification sources:
{sources_text}

Please analyze the factual accuracy of these claims.""")
            ])
            
            # Get AI response
            response = await self.model.ainvoke(prompt.format_messages())
            
            # Parse analysis results
            try:
                analysis_data = json.loads(response.content)
                
                # Update claims with analysis results
                claims_analysis = analysis_data.get("claims_analysis", [])
                for claim_analysis in claims_analysis:
                    claim_id = claim_analysis.get("claim_id")
                    for claim in claims:
                        if claim.get("id") == claim_id:
                            claim.update({
                                "accuracy": claim_analysis.get("accuracy"),
                                "confidence": claim_analysis.get("confidence", 0.5),
                                "supporting_evidence": claim_analysis.get("supporting_evidence", []),
                                "contradicting_evidence": claim_analysis.get("contradicting_evidence", [])
                            })
                            break
                
                fact_analysis = {
                    "claims_analysis": claims_analysis,
                    "overall_assessment": analysis_data.get("overall_assessment", {}),
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info("Fact analysis completed")
                return {
                    "extracted_claims": claims,
                    "fact_analysis": fact_analysis,
                    "status": "facts_analyzed"
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse fact analysis JSON: {e}")
                return {"fact_analysis": {}, "status": "parsing_failed"}
                
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


# Node registry for easy access
WORKFLOW_NODES = {
    "content_extraction": ContentExtractionNode,
    "claims_extraction": ClaimsExtractionNode,
    "source_verification": SourceVerificationNode,
    "fact_analysis": FactAnalysisNode,
    "confidence_scoring": ConfidenceScoringNode,
    # Content Analysis nodes
    "sentiment_analysis": ContentExtractionNode,  # Placeholder
    "bias_detection": ContentExtractionNode,  # Placeholder
    "quality_assessment": ContentExtractionNode,  # Placeholder
    "credibility_analysis": ContentExtractionNode,  # Placeholder
    "summary_generation": ContentExtractionNode,  # Placeholder
    # Source Verification nodes
    "domain_analysis": ContentExtractionNode,  # Placeholder
    "reputation_check": ContentExtractionNode,  # Placeholder
    "fact_checking_lookup": ContentExtractionNode,  # Placeholder
    "cross_reference": ContentExtractionNode,  # Placeholder
    "verification_result": ContentExtractionNode,  # Placeholder
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