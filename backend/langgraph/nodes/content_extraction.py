"""
Content extraction node for LangGraph workflows.

This module contains the content extraction node that processes
scraped content and prepares it for fact-checking analysis.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

from ..tools.content_analysis import ClaimExtractionTool, ContentQualityTool
from ..tools.web_search import ContentExtractor

logger = logging.getLogger(__name__)


class ContentExtractionNode:
    """
    Node for extracting and preparing content for fact-checking analysis.
    
    This node processes scraped content, extracts claims, assesses quality,
    and prepares the content for further analysis in the fact-checking workflow.
    """
    
    def __init__(self):
        """Initialize the content extraction node."""
        self.claim_extractor = ClaimExtractionTool()
        self.quality_assessor = ContentQualityTool()
        self.content_extractor = ContentExtractor()
    
    def extract_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and prepare content for analysis.
        
        Args:
            state: Current workflow state containing content information
            
        Returns:
            Updated state with extracted content and claims
        """
        try:
            # Get content from state
            content_id = state.get("content_id")
            content_text = state.get("content_text", "")
            content_url = state.get("content_url", "")
            
            if not content_text and content_url:
                # Extract content from URL if not already provided
                extraction_result = self.content_extractor._run(content_url)
                if extraction_result.get("success"):
                    content_text = extraction_result.get("content", "")
                    state["content_text"] = content_text
                    state["content_title"] = extraction_result.get("title", "")
                    state["content_metadata"] = extraction_result.get("metadata", {})
            
            if not content_text:
                logger.warning(f"No content text available for content_id: {content_id}")
                return {
                    **state,
                    "extraction_status": "failed",
                    "error": "No content text available for extraction"
                }
            
            # Extract claims from content
            claims_result = self.claim_extractor._run(content_text)
            extracted_claims = claims_result.get("claims", [])
            
            # Assess content quality
            quality_result = self.quality_assessor._run(content_text)
            
            # Prepare extracted content for analysis
            processed_content = {
                "text": content_text,
                "word_count": quality_result.get("word_count", 0),
                "sentence_count": quality_result.get("sentence_count", 0),
                "quality_score": quality_result.get("overall_quality_score", 0),
                "quality_level": quality_result.get("quality_level", "unknown"),
                "readability_score": quality_result.get("readability_score", 0)
            }
            
            # Extract key claims for fact-checking
            key_claims = []
            for claim_data in extracted_claims:
                claim_text = claim_data.get("claim", "")
                confidence = claim_data.get("confidence", 0.5)
                
                if claim_text and confidence > 0.3:  # Only include claims with reasonable confidence
                    key_claims.append({
                        "claim": claim_text,
                        "confidence": confidence,
                        "sentence": claim_data.get("sentence", "")
                    })
            
            # Update state with extraction results
            updated_state = {
                **state,
                "extraction_status": "completed",
                "processed_content": processed_content,
                "extracted_claims": key_claims,
                "total_claims": len(key_claims),
                "content_quality": quality_result,
                "extraction_metadata": {
                    "claim_density": claims_result.get("claim_density", 0),
                    "extraction_method": "automated",
                    "processing_time": "completed"
                }
            }
            
            logger.info(f"Content extraction completed for content_id: {content_id}")
            logger.info(f"Extracted {len(key_claims)} claims from content")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in content extraction: {str(e)}")
            return {
                **state,
                "extraction_status": "failed",
                "error": f"Content extraction failed: {str(e)}"
            }
    
    def validate_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted content for fact-checking suitability.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with validation results
        """
        try:
            processed_content = state.get("processed_content", {})
            extracted_claims = state.get("extracted_claims", [])
            
            # Validation criteria
            min_word_count = 50
            min_claims = 1
            min_quality_score = 30
            
            validation_results = {
                "is_suitable": True,
                "validation_issues": [],
                "recommendations": []
            }
            
            # Check content length
            word_count = processed_content.get("word_count", 0)
            if word_count < min_word_count:
                validation_results["is_suitable"] = False
                validation_results["validation_issues"].append(
                    f"Content too short ({word_count} words, minimum {min_word_count})"
                )
                validation_results["recommendations"].append(
                    "Consider scraping more content or using a different source"
                )
            
            # Check for claims
            if len(extracted_claims) < min_claims:
                validation_results["is_suitable"] = False
                validation_results["validation_issues"].append(
                    f"Insufficient claims extracted ({len(extracted_claims)}, minimum {min_claims})"
                )
                validation_results["recommendations"].append(
                    "Content may not contain verifiable factual claims"
                )
            
            # Check content quality
            quality_score = processed_content.get("quality_score", 0)
            if quality_score < min_quality_score:
                validation_results["validation_issues"].append(
                    f"Low content quality score ({quality_score:.1f}, minimum {min_quality_score})"
                )
                validation_results["recommendations"].append(
                    "Content quality may affect fact-checking accuracy"
                )
            
            # Check for bias indicators
            content_quality = state.get("content_quality", {})
            bias_score = content_quality.get("bias_score", 0)
            if bias_score > 0.1:
                validation_results["validation_issues"].append(
                    f"High bias detected (score: {bias_score:.2f})"
                )
                validation_results["recommendations"].append(
                    "Content shows signs of bias - fact-checking may be more challenging"
                )
            
            return {
                **state,
                "content_validation": validation_results,
                "validation_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in content validation: {str(e)}")
            return {
                **state,
                "validation_status": "failed",
                "error": f"Content validation failed: {str(e)}"
            }
    
    def prepare_for_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare extracted content for fact-checking analysis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state ready for fact-checking analysis
        """
        try:
            extracted_claims = state.get("extracted_claims", [])
            processed_content = state.get("processed_content", {})
            
            # Prioritize claims by confidence
            prioritized_claims = sorted(
                extracted_claims,
                key=lambda x: x.get("confidence", 0),
                reverse=True
            )
            
            # Select top claims for analysis (limit to prevent overwhelming)
            max_claims = 5
            selected_claims = prioritized_claims[:max_claims]
            
            # Prepare analysis-ready content
            analysis_content = {
                "main_text": processed_content.get("text", ""),
                "title": state.get("content_title", ""),
                "source_url": state.get("content_url", ""),
                "claims_to_verify": selected_claims,
                "content_metadata": {
                    "word_count": processed_content.get("word_count", 0),
                    "quality_score": processed_content.get("quality_score", 0),
                    "readability_score": processed_content.get("readability_score", 0),
                    "total_claims_found": len(extracted_claims)
                }
            }
            
            # Generate search queries for fact-checking
            search_queries = []
            for claim_data in selected_claims:
                claim_text = claim_data.get("claim", "")
                if claim_text:
                    # Create search query from claim
                    search_query = self._create_search_query(claim_text)
                    search_queries.append({
                        "query": search_query,
                        "original_claim": claim_text,
                        "confidence": claim_data.get("confidence", 0.5)
                    })
            
            return {
                **state,
                "analysis_content": analysis_content,
                "search_queries": search_queries,
                "preparation_status": "completed",
                "next_step": "source_verification"
            }
            
        except Exception as e:
            logger.error(f"Error in preparation for analysis: {str(e)}")
            return {
                **state,
                "preparation_status": "failed",
                "error": f"Preparation for analysis failed: {str(e)}"
            }
    
    def _create_search_query(self, claim_text: str) -> str:
        """
        Create a search query from a claim for fact-checking.
        
        Args:
            claim_text: The claim to convert to a search query
            
        Returns:
            Optimized search query for fact-checking
        """
        # Simple query optimization - in production, use more sophisticated NLP
        words = claim_text.split()
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these", "those"
        }
        
        # Filter out stop words and short words
        filtered_words = [
            word.lower() for word in words 
            if word.lower() not in stop_words and len(word) > 2
        ]
        
        # Take the most relevant words (up to 6 words)
        relevant_words = filtered_words[:6]
        
        return " ".join(relevant_words)
